from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import time
import math
import torch
import random
import itertools
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

from data import (
    fetch_dataset, 
    split_dataset, 
    make_data_loader, 
    separate_dataset, 
    make_batchnorm_dataset, 
    make_batchnorm_stats
)

class ServerCombinationSearch(ServerBase):

    def __init__(
        self, 
        model: ModelType,
        clients: dict[int, ClientType],
        dataset: DatasetType,
        communicationMetaData: dict=None
    ) -> None:

        super().__init__(dataset=dataset)
        self.server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        server_optimizer = create_optimizer(model, 'server')
        self.server_optimizer_state_dict = server_optimizer.state_dict()
        self.dynamic_uploaded_clients = defaultdict(list)
        self.dynamic_iterates = defaultdict(list)
        self.clients = clients
        self.communicationMetaData = communicationMetaData
        self.client_prob_distribution = {}
        self.genetic_metric_indicator = None
        self.high_freq_clients = None
        self.server_communication_cost_budget = communicationMetaData['server_communication_cost_budget']
        self.server_high_freq_communication_cost_budget = communicationMetaData['server_high_freq_communication_cost_budget']
        self.local_gradient_update_list_to_server_ratio = communicationMetaData['local_gradient_update_list_to_server_ratio']
        self.server_high_freq_communication_times, self.server_low_freq_communication_times = super().get_high_and_low_freq_communication_time(self.local_gradient_update_list_to_server_ratio)
        self.server_high_freq_communication_cost = super().cal_communication_cost(
            model_size=cfg['normalized_model_size'],
            high_freq_client_num=1, 
            low_freq_client_num=0, 
            high_freq_communication_times=self.server_high_freq_communication_times, 
            low_freq_communication_times=0,
        )

        self.server_low_freq_communication_cost = super().cal_communication_cost(
            model_size=cfg['normalized_model_size'],
            high_freq_client_num=0, 
            low_freq_client_num=1, 
            high_freq_communication_times=0, 
            low_freq_communication_times=self.server_low_freq_communication_times,
        ) 

    def cal_genetic_dist_func(self, *args):
        clients_indices_indicator = args[0]
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

        return res

    def brute_force(
        self, 
        global_labels_distribution, 
        num_clients, 
        selected_client_ids, 
        logger, 
        dataset,
        metric_indicator
    ):
        res = 100
        result_comb = None
        for comb in itertools.combinations(selected_client_ids, num_clients):
            comb_prob = np.array([0 for _ in range(len(dataset.classes_counts))])
            total_size = 0
            for client_id in comb:
                total_size += len(self.clients[client_id].data_split['train'])
            
            for client_id in comb:
                sub_prob = super().cal_prob_distribution(dataset, self.clients[client_id].data_split['train'], client_id)
                
                ratio = len(self.clients[client_id].data_split['train'])/total_size
                sub_prob = np.array([prob*ratio for prob in sub_prob])
                comb_prob = comb_prob + sub_prob

            cur_dist = None
            if metric_indicator == 'KL':
                cur_dist = super().cal_KL_divergence(comb_prob, global_labels_distribution)
            elif metric_indicator == 'QL':
                cur_dist = super().cal_QL(comb_prob, global_labels_distribution)

            if cur_dist < res:
                result_comb = copy.deepcopy(comb)
                res = cur_dist
        return res, result_comb

    def genetic(self, num_clients, selected_client_ids, distance_type):

        self.selected_client_ids = selected_client_ids
        lb = [0 for _ in range(len(selected_client_ids))]
        ub = [1 for _ in range(len(selected_client_ids))]
        precision = [1 for _ in range(len(selected_client_ids))]
        constraint_eq = [
            lambda x: num_clients - sum(x)
        ]
        self.genetic_metric_indicator = distance_type

        ga = GA(func=self.cal_genetic_dist_func, n_dim=len(selected_client_ids), size_pop=50, max_iter=200, prob_mut=0.001, 
                lb=lb, ub=ub, constraint_eq=constraint_eq, precision=precision)
        best_x, self.min_KL = ga.run()

        return self.min_KL[0], best_x

    
    def find_high_freq_group_clients(self, selected_client_ids, permutation_lists, dataset, logger, num_clients):
        self.global_labels_distribution = super().get_global_labels_distribution(dataset)
        self.dataset = dataset
        self.selected_client_ids = selected_client_ids

        temp = copy.deepcopy(selected_client_ids)
        temp_2 = copy.deepcopy(selected_client_ids)
        if cfg['cal_communication_cost'] == False:
            if num_clients <= 10:
                start = time.time()
                best_distance, best_combination = self.brute_force(
                    global_labels_distribution=self.global_labels_distribution, 
                    num_clients=num_clients, 
                    selected_client_ids=selected_client_ids, 
                    logger=logger, 
                    dataset=dataset,
                    metric_indicator='KL'
                )
                end = time.time()
                print(f'brute force KL time: {end-start}', flush=True)
                print(f'brute force KL_{num_clients}: {best_distance}', flush=True)
                print(f'brute force KL_comb_{num_clients}: {best_combination}', flush=True)
                logger.append(
                    {
                        f'brute_force_KL_{num_clients}': best_distance,
                        f'brute_force_KL_{num_clients}_time': end-start
                    }, 
                    'train', 
                )

            start = time.time()
            best_genetic_KL_list = []
            best_genetic_KL_combination_list = []
            for i in range(len(permutation_lists)):
                best_distance, best_combination = self.genetic(
                    num_clients=num_clients, 
                    selected_client_ids=permutation_lists[i],
                    distance_type='KL'
                )
                best_genetic_KL_list.append(best_distance)
                best_genetic_KL_combination_list.append(best_combination)
            end = time.time()
            print(f'genetic KL time: {end-start}', flush=True)
            print(f'genetic KL_{num_clients}: {min(best_genetic_KL_list)}', flush=True)
            print(f'genetic KL_comb_{num_clients}: {best_genetic_KL_combination_list[best_genetic_KL_list.index(min(best_genetic_KL_list))]}', flush=True)
            logger.append(
                {
                    f'genetic_KL_{num_clients}': min(best_genetic_KL_list),
                    f'genetic_KL_{num_clients}_time': end-start
                }, 
                'train', 
            )

            start = time.time()
            best_dp_KL_dist = []
            best_dp_KL_combination_list = []
            for i in range(len(permutation_lists)):
                best_distance, best_combination, _, _ = super().dp_combination_search(
                    dataset=dataset,
                    num_clients=num_clients, 
                    selected_client_ids=permutation_lists[i],
                    distance_type='KL',
                )
                # print(f'\n best_dp_KL_{num_clients}: {best_distance}, best_combination_{num_clients}: {best_combination}', flush=True)
                best_dp_KL_dist.append(best_distance)
                best_dp_KL_combination_list.append(best_combination)
            end = time.time()
            print(f'dp KL time: {end-start}', flush=True)
            print(f'dp KL_{num_clients}: {min(best_dp_KL_dist)}', flush=True)
            print(f'dp KL_comb_{num_clients}: {best_dp_KL_combination_list[best_dp_KL_dist.index(min(best_dp_KL_dist))]}', flush=True)
            logger.append(
                {
                    f'best_dp_KL_{num_clients}': min(best_dp_KL_dist),
                    f'best_dp_KL_{num_clients}_time': end-start
                }, 
                'train', 
            )
        elif cfg['cal_communication_cost'] == True:
            permutation_lists = super().get_selected_client_ids_permutation_lists(selected_client_ids)

            min_dist, min_dist_combination = super().dp_find_high_freq_group_clients(
                temp, 
                permutation_lists,
                dataset,
                logger,
                num_clients=len(selected_client_ids)
            )

            high_freq_local_gradient_update_list = self.local_gradient_update_list_to_server_ratio[0][0]
            low_freq_local_gradient_update_list = self.local_gradient_update_list_to_server_ratio[1][0]

            # fedsgd
            if cfg['server_ratio'] == '1-0' and cfg['client_ratio'] == '1-0':
                min_dist_combination = copy.deepcopy(selected_client_ids)
                print(f'fedsgd, {min_dist_combination}')
                
            self.high_freq_clients = copy.deepcopy(min_dist_combination)
            for client_id in self.high_freq_clients:
                self.clients[client_id].local_gradient_update_list = copy.deepcopy(list(high_freq_local_gradient_update_list))
            
            low_freq_clients = list(set(temp_2) - set(min_dist_combination))
            for client_id in low_freq_clients:
                if cfg['only_high_freq'] == True:
                    self.clients[client_id].local_gradient_update_list = []
                else:
                    self.clients[client_id].local_gradient_update_list = copy.deepcopy(list(low_freq_local_gradient_update_list))
            
            # if cfg['only_high_freq'] == True:
            #     low_freq_clients = []
            # cur_dynamicfl_cost = super().cal_communication_cost(
            #     model_size=cfg['normalized_model_size'],
            #     high_freq_client_num=len(self.high_freq_clients), 
            #     low_freq_client_num=len(low_freq_clients), 
            #     high_freq_communication_times=self.server_high_freq_communication_times, 
            #     low_freq_communication_times=self.server_low_freq_communication_times,
            # )

            # all fedsgd cost
            maximum_cost = super().cal_communication_cost(
                model_size=cfg['normalized_model_size'],
                high_freq_client_num=len(selected_client_ids), 
                low_freq_client_num=0, 
                high_freq_communication_times=cfg['max_local_gradient_update'], 
                low_freq_communication_times=self.server_low_freq_communication_times,
            )

            fedavg_cost = len(selected_client_ids) * 2 * cfg['normalized_model_size']

            
            fedavg_cost_ratio = fedavg_cost / maximum_cost
            # ratio = all_high_freq_dynamicfl_cost / maximum_cost
            logger.append(
                    {
                        f"fedavg_cost_ratio": fedavg_cost_ratio,
                    
                    }, 
                    'train', 
                )
            num_clients = len(selected_client_ids)

            cfg['upload_freq_level'] = {
                6: 1,
                5: 4,
                4: 16,
                3: 32,
                2.5: 64,
                2: 128,
                1: 256
            }

            high_freq = float(cfg['number_of_freq_levels'].split('-')[0])
            low_freq = float(cfg['number_of_freq_levels'].split('-')[1])

            low_freq_traverse_list = []
            if high_freq == 6:
                low_freq_traverse_list = [5 ,4, 3, 2.5, 2, 1]
            elif high_freq == 5:
                low_freq_traverse_list = [4, 3, 2.5, 2, 1]
            elif high_freq == 4:
                low_freq_traverse_list = [3, 2.5, 2, 1]
            elif high_freq == 3:
                low_freq_traverse_list = [2.5, 2, 1]

            for key in low_freq_traverse_list:
                temp_times = self.server_low_freq_communication_times / (cfg['upload_freq_level'][key] / cfg['upload_freq_level'][low_freq])
                cur_dynamicfl_cost = super().cal_communication_cost(
                    model_size=cfg['normalized_model_size'],
                    high_freq_client_num=len(self.high_freq_clients), 
                    low_freq_client_num=len(low_freq_clients), 
                    high_freq_communication_times=self.server_high_freq_communication_times, 
                    low_freq_communication_times=temp_times,
                )

                cur_dynamicfl_cost_ratio = cur_dynamicfl_cost / maximum_cost
                logger.append(
                    {
                        f"special_{cfg['server_ratio']}_{cfg['client_ratio']}_{high_freq}-{key}": cur_dynamicfl_cost_ratio,
                    
                    }, 
                    'train', 
                )

                print(f"{cfg['server_ratio']}_{cfg['client_ratio']}_{high_freq}-{key}: {cur_dynamicfl_cost_ratio}", flush=True)
            
            only_high_freq_dynamicfl_cost = cur_dynamicfl_cost = super().cal_communication_cost(
                    model_size=cfg['normalized_model_size'],
                    high_freq_client_num=len(self.high_freq_clients), 
                    low_freq_client_num=0, 
                    high_freq_communication_times=self.server_high_freq_communication_times, 
                    low_freq_communication_times=temp_times,
                )
            
            ratio = only_high_freq_dynamicfl_cost / maximum_cost
            logger.append(
                    {
                        f"special_{cfg['server_ratio']}_{cfg['client_ratio']}_{high_freq}-{low_freq}_1": ratio,
                    
                    }, 
                    'train', 
                )
            print(f"{cfg['server_ratio']}_{cfg['client_ratio']}_{high_freq}-{key}_1: {ratio}", flush=True)
            all_high_freq_dynamicfl_cost = cur_dynamicfl_cost = super().cal_communication_cost(
                    model_size=cfg['normalized_model_size'],
                    high_freq_client_num=len(selected_client_ids), 
                    low_freq_client_num=0, 
                    high_freq_communication_times=self.server_high_freq_communication_times, 
                    low_freq_communication_times=temp_times,
                )
            ratio = all_high_freq_dynamicfl_cost / maximum_cost
            logger.append(
                    {
                        f"special_1-0_1-0_{high_freq}-{low_freq}": ratio,
                    
                    }, 
                    'train', 
                )

            print(f"1-0_1-0_{high_freq}-{low_freq}: {ratio}", flush=True)
            logger.append(
                {
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_maximum_cost_{num_clients}": maximum_cost,
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_dynamicfl_cost_{num_clients}": cur_dynamicfl_cost,
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_fedavg_cost_{num_clients}": fedavg_cost,
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_dynamicfl_cost_ratio_{num_clients}": cur_dynamicfl_cost_ratio,
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_fedavg_cost_ratio_{num_clients}": fedavg_cost_ratio,
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_dynamicfl_high_freq_number_{num_clients}": len(min_dist_combination),

                }, 
                'train', 
            )
            print(f'best_dp_KL_{cfg["server_ratio"]}_{cfg["client_ratio"]}_ratio_communication_maximum_cost_{num_clients}: {maximum_cost}', flush=True)
            print(f'best_dp_KL_{cfg["server_ratio"]}_{cfg["client_ratio"]}_ratio_communication_cur_dynamicfl_cost_{num_clients}: {cur_dynamicfl_cost}', flush=True)
            print(f'best_dp_KL_{cfg["server_ratio"]}_{cfg["client_ratio"]}_ratio_communication_cur_dynamicfl_cost_ratio_{num_clients}: {cur_dynamicfl_cost_ratio}', flush=True)
            print(f'best_dp_KL_{cfg["server_ratio"]}_{cfg["client_ratio"]}_ratio_communication_cur_fedavg_cost_ratio_{num_clients}: {fedavg_cost_ratio}', flush=True)
            print(f'best_dp_KL_{cfg["server_ratio"]}_{cfg["client_ratio"]}_ratio_communication_cur_dynamicfl_high_freq_number_{num_clients}: {len(min_dist_combination)}', flush=True)
        return

    def distribute_local_gradient_update_list(self, selected_client_ids: list[int], dataset, logger):
        '''
        distribute local gradient update list to certain selected clients 
        according to the client ratio
        '''
        temp = copy.deepcopy(selected_client_ids)
        # a = self.communicationMetaData['local_gradient_update_list_to_client_ratio']
        
        permutation_lists = super().get_selected_client_ids_permutation_lists(selected_client_ids)

        if cfg['cal_communication_cost'] == True:
            self.find_high_freq_group_clients(
                temp, 
                permutation_lists,
                dataset,
                logger,
                num_clients=len(selected_client_ids)
            )

        
        elif cfg['cal_communication_cost'] == False:
            for size in range(1, len(selected_client_ids)+1):
                self.find_high_freq_group_clients(
                    temp, 
                    permutation_lists,
                    dataset,
                    logger,
                    num_clients=size
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
        self.distribute_local_gradient_update_list(
            selected_client_ids=selected_client_ids,
            dataset=dataset,
            logger=logger,
        )
        logger.safe(False)
        logger.reset()
        return