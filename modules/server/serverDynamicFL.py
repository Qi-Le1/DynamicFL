from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import time
import math
import torch
import random
import torch.nn.functional as F
import models
from sko.GA import GA
from itertools import compress
from config import cfg
from collections import defaultdict

from _typing import (
    DatasetType,
    OptimizerType,
    DataLoaderType,
    ModelType,
    MetricType,
    LoggerType,
    ClientType,
    ServerType
)

from models.api import (
    create_model,
    make_batchnorm
)

from utils.api import (
    to_device,  
    collate
)
from optimizer.api import create_optimizer
from .serverBase import ServerBase, ClientSampler
from .serverCombinationSearch import ServerCombinationSearch

from data import (
    fetch_dataset, 
    split_dataset, 
    make_data_loader, 
    separate_dataset, 
    make_batchnorm_dataset, 
    make_batchnorm_stats
)


class ServerDynamicFL(ServerBase):

    def __init__(
        self, 
        model: ModelType,
        clients: dict[int, ClientType],
        dataset: DatasetType,
        communicatoin_info
    ) -> None:
        ServerBase.__init__(self, dataset=dataset)

        self.server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        server_optimizer = create_optimizer(model, 'server')
        self.server_optimizer_state_dict = server_optimizer.state_dict()

        self.clients = clients
        self.global_labels_distribution = super().get_global_labels_distribution(dataset)

        self.dynamic_uploaded_clients = defaultdict(list)
        self.dynamic_iterates = defaultdict(list)
        
        self.communicaton_info = communicatoin_info
        self.high_freq_clients = None
        self.low_freq_clients = None
        self.server_communication_cost_budget = communicatoin_info.server_communication_cost_budget
        self.server_high_freq_communication_cost_budget = communicatoin_info.server_high_freq_communication_cost_budget
        

    def distribute_dynamic_part(
        self,
        local_gradient_update: int,
    ):
        if local_gradient_update not in self.dynamic_uploaded_clients:
            return

        for cur_client_id in self.dynamic_uploaded_clients[local_gradient_update]:
            self.clients[cur_client_id].model_state_dict = copy.deepcopy(self.server_model_state_dict)

        del self.dynamic_uploaded_clients[local_gradient_update]
        return

    def upload_dynamic_part(
        self,
        target_gradent_update: int,
        cur_client_id: int,
    ) -> None:
        '''
        handle dynamic logic, do union operation 
        '''
        self.dynamic_uploaded_clients[target_gradent_update].append(cur_client_id)
        self.dynamic_iterates[target_gradent_update].append(copy.deepcopy(self.clients[cur_client_id].model_state_dict))
        return

    def update_dynamic_part(
        self,
        local_gradent_update: int,
    ):
        with torch.no_grad():
            if local_gradent_update not in self.dynamic_uploaded_clients:
                return
            new_model_parameters_list = self.dynamic_iterates[local_gradent_update]
            if len(new_model_parameters_list) > 0:
                model = super().create_model(track_running_stats=False, on_cpu=True)
                model.load_state_dict(self.server_model_state_dict)
                server_optimizer = create_optimizer(model, 'server')
                server_optimizer.load_state_dict(self.server_optimizer_state_dict)
                server_optimizer.zero_grad()
                
                weight = []
                for client_id in self.dynamic_uploaded_clients[local_gradent_update]:
                    valid_client = self.clients[client_id]
                    cur_data_size = len(valid_client.data_split['train'])
                    if cfg['server_aggregation'] == 'WA':
                        weight.append(cur_data_size)
                    else:
                        raise ValueError('server_aggregation must be WA')
                new_weight = [i / sum(weight) for i in weight]
                weight = torch.tensor(new_weight)
                
                for k, v in model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(new_model_parameters_list)):
                            tmp_v += weight[m] * new_model_parameters_list[m][k]
                        v.grad = (v.data - tmp_v).detach()
                server_optimizer.step()
                self.server_optimizer_state_dict = server_optimizer.state_dict()
                self.server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

            del self.dynamic_iterates[local_gradent_update]
        return

    def update_server_model(self, clients: dict[int, ClientType]) -> None:
        with torch.no_grad():
            valid_clients = [clients[i] for i in range(len(clients)) if clients[i].active]
            if valid_clients:
                model = super().create_model(track_running_stats=False, on_cpu=True)
                model.load_state_dict(self.server_model_state_dict)
                server_optimizer = create_optimizer(model, 'server')
                server_optimizer.load_state_dict(self.server_optimizer_state_dict)
                server_optimizer.zero_grad()

                weight = []
                if cfg['server_aggregation'] == 'WA':
                    for valid_client in valid_clients:
                        cur_data_size = len(valid_client.data_split['train'])
                        weight.append(cur_data_size)
                    new_weight = [i / sum(weight) for i in weight]
                    weight = torch.tensor(new_weight)
                else:
                    raise ValueError('server_aggregation must be WA')

                for k, v in model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(valid_clients)):
                            tmp_v += weight[m] * valid_clients[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                
                server_optimizer.step()
                self.server_optimizer_state_dict = server_optimizer.state_dict()
                self.server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            for i in range(len(clients)):
                clients[i].active = False
        
        self.dynamic_uploaded_clients = defaultdict(list)
        self.dynamic_iterates = defaultdict(list)
        return


    def cal_genetic_dist_func(self, *args):
  
        clients_indices_indicator = args[0]
        # print(f'clients_indices_indicator: {clients_indices_indicator}')
        comb_prob = np.array([0 for _ in range(len(self.dataset.classes_counts))])
        total_size = 0
        for i in range(len(clients_indices_indicator)):
            if int(clients_indices_indicator[i]) == 1:
                client_id = self.selected_client_ids[i]
                total_size += len(self.clients[client_id].data_split['train'])

        for i in range(len(clients_indices_indicator)):
            if int(clients_indices_indicator[i]) == 1:
                client_id = self.selected_client_ids[i]
                sub_prob = super().cal_prob_distribution(self.dataset, self.clients[client_id].data_split['train'], client_id)

                ratio = len(self.clients[client_id].data_split['train'])/total_size
                sub_prob = np.array([prob*ratio for prob in sub_prob])
                comb_prob = comb_prob + sub_prob

        res = None
        if self.genetic_metric_indicator == 'KL':
            res = super().cal_KL_divergence(comb_prob, self.global_labels_distribution)
        elif self.genetic_metric_indicator == 'QL':
            res = super().cal_QL(comb_prob, self.global_labels_distribution)

        # if self.update_min_criteria(KL_divergence, self.min_KL, clients_indices_indicator, self.min_clients_list):
        #     # self.min_clients_num = sum(clients_indices_indicator)
        #     self.min_clients_list = copy.deepcopy(clients_indices_indicator)
        #     self.min_KL = KL_divergence
        return res

    def genetic(self, num_clients, dataset, selected_client_ids, distance_type):
        self.dataset = dataset
        self.selected_client_ids = selected_client_ids
        lb = [0 for _ in range(len(selected_client_ids))]
        ub = [1 for _ in range(len(selected_client_ids))]
        precision = [1 for _ in range(len(selected_client_ids))]
        # constraint_ueq = [
        #     lambda x: 1 - sum(x),
        #     lambda x: sum(x) - 3
        #     # lambda x: self.cal_KL_func(x) - threshold
        # ]
        constraint_eq = [
            lambda x: num_clients - sum(x)
        ]
        self.genetic_metric_indicator = distance_type

        ga = GA(func=self.cal_genetic_dist_func, n_dim=len(selected_client_ids), size_pop=50, max_iter=200, prob_mut=0.001, 
                lb=lb, ub=ub, constraint_eq=constraint_eq, precision=precision)
        best_x, self.min_KL = ga.run()
        # best_x = [int(item) for item in best_x]

        # best_x = selected_client_ids * best_x
        res = []
        for i in range(len(best_x)):
            item = best_x[i]
            if int(item) == 1:
                res.append(selected_client_ids[i])
        return self.min_KL[0], res

    def distribute_local_gradient_update_list(self, selected_client_ids: list[int], dataset, logger):
        '''
        distribute local gradient update list to certain selected clients 
        according to the client ratio
        '''
        if self.global_labels_distribution == None:
            self.global_labels_distribution = super().get_global_labels_distribution(dataset)
        temp = copy.deepcopy(selected_client_ids)
        temp_2 = copy.deepcopy(selected_client_ids)
        # a = self.communicationMetaData['local_gradient_update_list_to_client_ratio']
        # local_gradient_update_list_to_client_ratio = self.communicationMetaData['local_gradient_update_list_to_client_ratio']
        # local_gradient_update_list_to_client_ratio = sorted(
        #     self.communicationMetaData['local_gradient_update_list_to_client_ratio'].items(),
        #     key=lambda x:-len(x[0])
        # )
        # for item in local_gradient_update_list_to_client_ratio:
        #     print(f'item: {item}')
        permutation_lists = super().get_selected_client_ids_permutation_lists(selected_client_ids)
        self.genetic()

        # min_dist, min_dist_combination = super().dp_find_high_freq_group_clients(
        #     temp, 
        #     permutation_lists,
        #     dataset,  
        #     logger,
        #     num_clients=len(selected_client_ids)
        # )
        num_to_select = 0
        if cfg['server_ratio'] == '1-0':
                # high_freq_ratio = float(cfg['client_ratio'].split('-')[0])

            min_dist_combination = []
            for id in selected_client_ids:
                if self.clients[id].client_communication_cost_budget >= self.server_high_freq_communication_cost:
                    min_dist_combination.append(id)

            a = copy.deepcopy(min_dist_combination)
            num_to_select = len(min_dist_combination) - 1
        elif cfg['client_ratio'] == '1-0':
            high_freq_ratio = float(cfg['server_ratio'].split('-')[0])
            num_to_select = int(high_freq_ratio * len(selected_client_ids))
            # min_dist_combination = selected_client_ids

        if num_to_select <= 0:
            num_to_select = 0
        min_dist, min_dist_combination = super().dp_dynacomm_contrast_find_high_freq_group_clients(
            temp, 
            permutation_lists,
            dataset,  
            logger,
            num_clients=num_to_select
        )

        if cfg['select_way'] == 'gene' and cfg['client_ratio'] == '1-0':
            dp_combination = copy.deepcopy(min_dist_combination)
            print(f'dp_min_dist: {min_dist}, {min_dist_combination}')
            min_dist, min_dist_combination = self.genetic(
                num_clients=num_to_select, 
                dataset=dataset,
                selected_client_ids=copy.deepcopy(selected_client_ids),
                distance_type='KL'
            )
            # print(f'genetic min distri: {min_dist_combination}')
            diff_client_count = set(dp_combination).difference(set(min_dist_combination))
            print(f'genetic_min_dist: {min_dist}, {min_dist_combination}')
            print(f'dp_genetic_diff_count: {len(diff_client_count)}')
        temp_dp_dist = copy.deepcopy(min_dist_combination)
        '''
        for random high freq client select
        # TODO: temporary solution, need to be changed    
        '''
        # num_to_select = len(min_dist_combination)
        if cfg['select_way'] == 'rand':
            a = 5
            if cfg['server_ratio'] == '1-0':
                # high_freq_ratio = float(cfg['client_ratio'].split('-')[0])

                min_dist_combination = []
                for id in selected_client_ids:
                    if self.clients[id].client_communication_cost_budget >= self.server_high_freq_communication_cost:
                        min_dist_combination.append(id)
                # a = 5
                temp_dp_dist = copy.deepcopy(min_dist_combination)
                min_dist_combination = random.sample(min_dist_combination, num_to_select)
            elif cfg['client_ratio'] == '1-0':
                # high_freq_ratio = float(cfg['server_ratio'].split('-')[0])
                # num_to_select = int(high_freq_ratio * len(selected_client_ids))
                min_dist_combination = random.sample(selected_client_ids, num_to_select)
        
        print(f'min_dist: {min_dist}')
        print(f'min_dist_combination: {min_dist_combination}')
        # cur_client_communication_cost_budget = self.clients[selected_client_ids[i-1]].client_communication_cost_budget
        # if cur_client_communication_cost_budget < self.server_high_freq_communication_cost:
        # print(min_dist_combination)
        high_freq_local_gradient_update_list = self.local_gradient_update_list_to_server_ratio[0][0]
        low_freq_local_gradient_update_list = self.local_gradient_update_list_to_server_ratio[1][0]

        # fedsgd
        if cfg['server_ratio'] == '1-0' and cfg['client_ratio'] == '1-0':
            min_dist_combination = copy.deepcopy(selected_client_ids)
            print(f'fedsgd, {min_dist_combination}')
            
        self.high_freq_clients = copy.deepcopy(min_dist_combination)
        for client_id in self.high_freq_clients:
            self.clients[client_id].local_gradient_update_list = copy.deepcopy(list(high_freq_local_gradient_update_list))
        
        self.low_freq_clients = list(set(temp_2) - set(min_dist_combination))
        for client_id in self.low_freq_clients:
            if cfg['only_high_freq'] == True:
                self.clients[client_id].local_gradient_update_list = []
            else:
                self.clients[client_id].local_gradient_update_list = copy.deepcopy(list(low_freq_local_gradient_update_list))
        
        cur_dynamicfl_cost = super().cal_communication_cost(
            model_size=cfg['normalized_model_size'],
            high_freq_client_num=len(self.high_freq_clients), 
            low_freq_client_num=len(self.low_freq_clients), 
            high_freq_communication_times=self.server_high_freq_communication_times, 
            low_freq_communication_times=self.server_low_freq_communication_times,
        )

        # all fedsgd cost
        maximum_cost = super().cal_communication_cost(
            model_size=cfg['normalized_model_size'],
            high_freq_client_num=len(selected_client_ids), 
            low_freq_client_num=0, 
            high_freq_communication_times=cfg['max_local_gradient_update'], 
            low_freq_communication_times=self.server_low_freq_communication_times,
        )

        fedavg_cost = len(selected_client_ids) * 2 * cfg['normalized_model_size']

        cur_dynamicfl_cost_ratio = cur_dynamicfl_cost / maximum_cost
        fedavg_cost_ratio = fedavg_cost / maximum_cost

        num_clients = len(selected_client_ids)
        logger.append(
            {
                f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_maximum_cost_{num_clients}": maximum_cost,
                f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_dynamicfl_cost_{num_clients}": cur_dynamicfl_cost,
                f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_fedavg_cost_{num_clients}": fedavg_cost,
                f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_dynamicfl_cost_ratio_{num_clients}": cur_dynamicfl_cost_ratio,
                f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_fedavg_cost_ratio_{num_clients}": fedavg_cost_ratio,
            }, 
            'train', 
        )

        
        return

    def train(
        self,
        dataset: DatasetType,  
        optimizer: OptimizerType, 
        metric: MetricType, 
        logger: LoggerType, 
        global_epoch: int
    ):
        logger.safe(True)
        selected_client_ids, num_active_clients = super().select_clients(clients=self.clients)
        super().distribute_server_model_to_clients(
            server_model_state_dict=self.server_model_state_dict,
            clients=self.clients
        )

        self.distribute_local_gradient_update_list(
            selected_client_ids=selected_client_ids,
            dataset=dataset,
            logger=logger
        )
        data_loader_list = []
        client_sampler_list = []
        for client_id in selected_client_ids:
            client_sampler = ClientSampler(
                batch_size=cfg['client']['batch_size']['train'], 
                data_split=copy.deepcopy(self.clients[client_id].data_split['train']),
                client_id=client_id,
                max_local_gradient_update=cfg['max_local_gradient_update'],
                high_freq_clients=self.high_freq_clients,
            )
            client_sampler_list.append(client_sampler)
            self.clients[client_id].batch_size = client_sampler.batch_size
            data_loader_list.append(make_data_loader(
                dataset={'train': dataset}, 
                tag='client',
                batch_sampler={'train': client_sampler}
            )['train']) 

        start_time = time.time()
        lr = optimizer.param_groups[0]['lr']
        for local_gradient_update in range(1, cfg['max_local_gradient_update'] + 1):
            # print(f'local_gradient_update: {local_gradient_update}')
            # update the server model parameter using self.dynamic_iterates[target_gradent_update]
            self.update_dynamic_part(local_gradent_update=local_gradient_update)
            # Distribute the new server parameter update by dynamicFL to the clients
            # that have uploaded their local parameters to local_gradient_update
            
            # TODO: reselect clients

            self.distribute_dynamic_part(local_gradient_update=local_gradient_update)
                
            for i in range(num_active_clients):
                m = selected_client_ids[i]
                if not self.is_local_gradient_update_valid(
                    local_gradient_update=local_gradient_update,
                    local_gradient_update_list=self.clients[m].local_gradient_update_list
                ):
                    continue

                if 
                grad_updates_num = self.cal_gradient_updates_num(
                    local_gradient_update=local_gradient_update,
                    local_gradient_update_list=self.clients[m].local_gradient_update_list
                )
                
                self.clients[m].train(
                    client_sampler=client_sampler_list[i],
                    data_loader=data_loader_list[i], 
                    lr=lr, 
                    metric=metric, 
                    logger=logger,
                    grad_updates_num=grad_updates_num
                )

                # upload the new local model parameter to the self.dynamic_iterates
                self.upload_dynamic_part(
                    target_gradent_update=local_gradient_update+grad_updates_num,
                    cur_client_id=m
                )
                
            super().add_dynamicFL_log(
                local_gradient_update=local_gradient_update,
                start_time=start_time,
                global_epoch=global_epoch,
                lr=lr,
                metric=metric,
                logger=logger,
            )
        
        logger.safe(False)
        logger.reset()
        self.update_server_model(clients=self.clients)
        return

    def cal_gradient_updates_num(
        self,
        local_gradient_update: int,
        local_gradient_update_list: list[int]
    ) -> int:
        '''
        Calculate the local gradient update interval until next update, which
        means current client will update gradient for gradient update num times.

        If local_gradient_update is the last ele in the local_gradient_update_list,
        then we update its gradient to max_local_gradient_update
        '''
        index = local_gradient_update_list.index(local_gradient_update)
        if index == len(local_gradient_update_list) - 1:
            return 0
        return local_gradient_update_list[index+1] - local_gradient_update

    def is_local_gradient_update_valid(
        self,
        local_gradient_update: int,
        local_gradient_update_list: list[int]
    ) -> bool:
        '''
        Check if local gradient update is in local gradient update list
        '''
        return local_gradient_update in local_gradient_update_list

    def evaluate_trained_model(
        self,
        dataset,
        batchnorm_dataset,
        logger,
        metric,
        global_epoch
    ):  
        return super().evaluate_trained_model(
            dataset=dataset,
            batchnorm_dataset=batchnorm_dataset,
            logger=logger,
            metric=metric,
            global_epoch=global_epoch,
            server_model_state_dict=self.server_model_state_dict
        )

        