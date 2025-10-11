from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import math
import time
import torch
import random
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
from collections import defaultdict
from scipy.special import rel_entr

from _typing import (
    ModelType,
    ClientType,
    DatasetType
)

from utils.api import (
    to_device,  
    collate
)

from models.api import (
    create_model,
    make_batchnorm
)

from modules.api import (
    ClientDataSampler
)

from data import (
    fetch_dataset, 
    split_dataset, 
    make_data_loader, 
    separate_dataset, 
    make_batchnorm_dataset, 
    make_batchnorm_stats
)

from optimizer.api import create_optimizer


        
class ServerBase:

    def __init__(
        self,
        dataset,
        clients
    ) -> None:
        self.fix_order_picking = -1
        self.client_prob_distribution = {}
        self.group_clients_prob_distribution = np.array([0 for _ in range(len(dataset.classes_counts))])
        self.data_loader_list = []
        # self.generate_data_loader_list(dataset, clients)
        self.high_freq_clients = {}
        return
    
    def create_model(self, track_running_stats=False, on_cpu=False):
        return create_model(track_running_stats=track_running_stats, on_cpu=on_cpu)

    def create_test_model(
        self,
        model_state_dict,
        batchnorm_dataset
    ) -> object:
        model = create_model()
        model.load_state_dict(model_state_dict)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'server')
        return test_model

    def distribute_server_model_to_clients(
        self,
        server_model_state_dict,
        client_ids
    ) -> None:
        for client_id in client_ids:
            self.clients[client_id].model_state_dict = copy.deepcopy(server_model_state_dict)
        return
    
    # def generate_data_loader_list(self, dataset, clients):
    #     client_ids = np.arange(cfg['num_clients'])
    #     for client_id in client_ids:
    #         # client_sampler = ClientDataSampler(
    #         #     batch_size=cfg['client']['batch_size']['train'], 
    #         #     data_split=copy.deepcopy(clients[client_id].data_split['train']),
    #         #     client_id=client_id,
    #         #     max_local_gradient_update=cfg['max_local_gradient_update'],
    #         # )

    #         # print(f"client_id: {client_id}, {len(clients[client_id].data_split['train'])}, len(client_sampler): {len(client_sampler)}")
    #         self.data_loader_list.append(DataLoaderWrapper(make_data_loader(
    #             dataset={'train': dataset}, 
    #             tag='client',
    #             # batch_sampler={'train': client_sampler}
    #         )['train'])) 
    #     return

    def activate_clients(self, client_ids):
        # Check if client_ids is not a string and is iterable
        if hasattr(client_ids, '__iter__'):
            for client_id in client_ids:
                self.clients[client_id].active = True
        else:
            self.clients[client_ids].active = True
        return
    
    def deactivate_clients(self, client_ids):
        # Check if client_ids is not a string and is iterable
        if hasattr(client_ids, '__iter__'):
            for client_id in client_ids:
                self.clients[client_id].active = False
        else:
            self.clients[client_ids].active = False
        return
    
    def add_log(
        self,
        i,
        num_active_clients,
        start_time,
        global_epoch,
        lr,
        selected_client_ids,
        metric,
        logger
    ) -> None:
        if i % int((num_active_clients * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            global_epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
            exp_finished_time = global_epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['server']['num_epochs'] - global_epoch) * _time * num_active_clients))
            exp_progress = 100. * i / num_active_clients
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                            'Train Epoch (C): {}({:.0f}%)'.format(global_epoch, exp_progress),
                            'Learning rate: {:.6f}'.format(lr),
                            'ID: {}({}/{})'.format(selected_client_ids[i], i + 1, num_active_clients),
                            'Global Epoch Finished Time: {}'.format(global_epoch_finished_time),
                            'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']), flush=True)

    def add_prefix(self, metric):
        res = []
        for value in metric:
            res.append('{}_{}'.format('high_freq', value))
        
        for value in metric:
            res.append('{}_{}'.format('low_freq', value))
        return res
    
    def add_dynamicFL_log(
        self,
        local_gradient_update,
        start_time,
        global_epoch,
        lr,
        # selected_client_ids,
        metric,
        logger
    ) -> None:
        # if local_gradient_update % int((cfg['max_local_gradient_update'] * cfg['log_interval']) + 1) == 0:
        # _time = (time.time() - start_time) / (local_gradient_update + 1)
        _time = (time.time() - start_time)
        # global_epoch_finished_time = datetime.timedelta(seconds=_time * (cfg['max_local_gradient_update'] - local_gradient_update - 1))
        global_epoch_finished_time = datetime.timedelta(seconds=_time)
        # exp_finished_time = global_epoch_finished_time + datetime.timedelta(
        #     seconds=round((cfg['server']['num_epochs'] - global_epoch) * _time * cfg['max_local_gradient_update']))
        exp_finished_time = global_epoch_finished_time + datetime.timedelta(
            seconds=round((cfg['server']['num_epochs'] - global_epoch) * _time))
        # exp_progress = 100. * local_gradient_update / cfg['max_local_gradient_update']
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                        'Train Epoch (C): {}'.format(global_epoch),
                        'Learning rate: {:.6f}'.format(lr),
                        # 'ID: {}({}/{})'.format(selected_client_ids[i], i + 1, cfg['max_local_gradient_update']),
                        'Global Epoch Finished Time: {}'.format(global_epoch_finished_time),
                        'Experiment Finished Time: {}'.format(exp_finished_time)]}
        logger.append(info, 'train', mean=False)

        metric_names = self.add_prefix(metric.metric_name['train'])
        print(logger.write('train', metric_names), flush=True)

        # metric_names = self.add_prefix('low_freq', metric.metric_name['train'])
        # print(logger.write('train', metric_names), flush=True)
        return

   
    def update_server_model(self, clients: dict[int, ClientType]) -> None:
        with torch.no_grad():
            valid_clients = [clients[i] for i in range(len(clients)) if clients[i].active]
            if valid_clients:
                model = self.create_model(track_running_stats=False, on_cpu=True)
                model.load_state_dict(self.server_model_state_dict)
                server_optimizer = create_optimizer(model, 'server')
                server_optimizer.load_state_dict(self.server_optimizer_state_dict)
                server_optimizer.zero_grad()

                weight = []
                for valid_client in valid_clients:
                    if cfg['server_aggregation'] == 'WA':
                        weight.append(len(valid_client.data_split['train']))
                    elif cfg['server_aggregation'] == 'MA':
                        weight.append(1)
                new_weight = [i / sum(weight) for i in weight]
                weight = torch.tensor(new_weight)
                
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
        return
    
    def combine_test_dataset(
        self,
        num_active_clients: int,
        clients: dict[int, ClientType],
        selected_client_ids: list[int],
        dataset: DatasetType
    ) -> DatasetType:  
        '''
        combine the datapoint index for selected clients
        and return the dataset
        '''
        combined_datapoint_idx = []
        for i in range(num_active_clients):
            m = selected_client_ids[i]
            combined_datapoint_idx += clients[m].data_split['test']

        # dataset: DatasetType
        dataset = separate_dataset(dataset, combined_datapoint_idx)
        return dataset

    def evaluate_trained_model(
        self,
        dataset,
        batchnorm_dataset,
        logger,
        metric,
        global_epoch,
        server_model_state_dict
    ):  
        data_loader = make_data_loader(
            dataset={'test': dataset}, 
            tag='server'
        )['test']

        model = self.create_test_model(
            model_state_dict=server_model_state_dict,
            batchnorm_dataset=batchnorm_dataset
        )

        logger.safe(True)
        with torch.no_grad():
            model.train(False)
            for i, input in enumerate(data_loader):

                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(input)
                evaluation = metric.evaluate(
                    metric.metric_name['test'], 
                    input, 
                    output
                )
                logger.append(
                    evaluation, 
                    'test', 
                    input_size
                )
                
            info = {
                'info': [
                    'Model: {}'.format(cfg['model_tag']), 
                    'Test Epoch: {}({:.0f}%)'.format(global_epoch, 100.)
                ]
            }
            logger.append(info, 'test', mean=False)
            print(logger.write('test', metric.metric_name['test']), flush=True)
        logger.safe(False)
        return
    
    def cal_dp_dist_func(self, dataset, client_ids, metric_indicator):
        comb_prob = np.array([0 for _ in range(len(dataset.classes_counts))])
        total_size = 0

        for client_id in client_ids:  
            total_size += len(self.clients[client_id].data_split['train'])

        for client_id in client_ids:
            sub_prob = self.cal_prob_distribution(dataset, self.clients[client_id].data_split['train'], client_id)
            ratio = len(self.clients[client_id].data_split['train'])/total_size
            sub_prob = np.array([prob*ratio for prob in sub_prob])
            comb_prob = comb_prob + sub_prob

        res = None
        if metric_indicator == 'KL':
            res = self.cal_KL_divergence(comb_prob, self.global_labels_distribution)
        elif metric_indicator == 'QL':
            res = self.cal_QL(comb_prob, self.global_labels_distribution)
        # print('kl_res:', res, client_ids, comb_prob)
        return res
    
    def dp_combination_search(self, dataset, num_clients, selected_client_ids, distance_type):

        each_item = {
            'distance': float('inf'),
            'client_ids': [],
        }
        dp_res = [[copy.deepcopy(each_item) for _ in range(num_clients+1)] for _ in range(len(selected_client_ids)+1)]
        for i in range(1, len(selected_client_ids)+1):
            for j in range(1, num_clients+1):
                if i < j:
                    dp_res[i][j] = copy.deepcopy(dp_res[i-1][j])
                else:
                    dp_res[i][j] = copy.deepcopy(dp_res[i-1][j])
                    temp = copy.deepcopy(dp_res[i-1][j-1])
                    temp['client_ids'] += [selected_client_ids[i-1]]
                    temp['distance'] = self.cal_dp_dist_func(
                        dataset=dataset,
                        client_ids=temp['client_ids'],
                        metric_indicator=distance_type
                    )
                    # add communication cost
                    # cur_total_communication_cost = self.cal_clients_communication_cost(
                    #     model_size=cfg['normalized_model_size'], 
                    #     client_ids=temp['client_ids'],
                    # )

                    # cur_client_communication_cost_budget = self.clients[selected_client_ids[i-1]].client_communication_cost_budget
                    # print(f'cur_client_communication_cost_budget: {cur_client_communication_cost_budget}')
                    # print(f'cur_total_communication_cost: {cur_total_communication_cost}')

                    # check client's communication cost
                    # if cur_client_communication_cost_budget < self.server_high_freq_communication_cost:
                    #     continue
                    # # check server's communication cost
                    # if self.server_high_freq_communication_cost_budget < cur_total_communication_cost:
                    #     continue
                    # print(f'cur_communication_cost: {cur_communication_cost}')
                    # temp['distance'] += cur_communication_cost
                    if temp['distance'] < dp_res[i][j]['distance'] and len(temp['client_ids']) == j:
                        dp_res[i][j] = copy.deepcopy(temp)

        # if num_clients == 20:
        #     for i in range(len(dp_res)):
        #         print('\n')
        #         for j in range(len(dp_res[0])):
        #             print(f'{dp_res[i][j]["distance"]:.2f}', end=' ')
        #         print('\n')
        #         for j in range(len(dp_res[0])):
        #             print(f'{len(dp_res[i][j]["client_ids"])}', end=' ')     
                
            # a = 5
        min_distance = dp_res[-1][num_clients]['distance']
        min_client_ids = dp_res[-1][num_clients]['client_ids']
        # print(f'num_clients: {num_clients}, min_distance: {min_distance}, min_client_ids: {min_client_ids}')
        # if count_smaller_than:
        min_distance_for_all = float('inf')
        min_client_ids_for_all = []
        # for i in range(len(dp_res)):
        for j in range(len(dp_res[0])):
            if dp_res[-1][j]['distance'] < min_distance_for_all:
                min_distance_for_all = dp_res[-1][j]['distance']
                min_client_ids_for_all = dp_res[-1][j]['client_ids']
        return min_distance, min_client_ids, min_distance_for_all, min_client_ids_for_all
    
    def dp(self, dataset, num_clients, selected_client_ids, distance_type):

        each_item = {
            'distance': float('inf'),
            'client_ids': [],
        }
        dp_res = [[copy.deepcopy(each_item) for _ in range(num_clients+1)] for _ in range(len(selected_client_ids)+1)]
        for i in range(1, len(selected_client_ids)+1):
            for j in range(1, num_clients+1):
                if i < j:
                    dp_res[i][j] = copy.deepcopy(dp_res[i-1][j])
                else:
                    dp_res[i][j] = copy.deepcopy(dp_res[i-1][j])
                    temp = copy.deepcopy(dp_res[i-1][j-1])
                    temp['client_ids'] += [selected_client_ids[i-1]]
                    temp['distance'] = self.cal_dp_dist_func(
                        dataset=dataset,
                        client_ids=temp['client_ids'],
                        metric_indicator=distance_type
                    )
                    # add communication cost
                    cur_total_communication_cost = self.cal_clients_communication_cost(
                        model_size=cfg['normalized_model_size'], 
                        client_ids=temp['client_ids'],
                    )

                    cur_client_communication_cost_budget = self.clients[selected_client_ids[i-1]].client_communication_cost_budget
                    # print(f'cur_client_communication_cost_budget: {cur_client_communication_cost_budget}')
                    # print(f'cur_total_communication_cost: {cur_total_communication_cost}')

                    # check client's communication cost
                    if cur_client_communication_cost_budget < self.server_high_freq_communication_cost:
                        continue
                    # check server's communication cost
                    if self.server_high_freq_communication_cost_budget < cur_total_communication_cost:
                        continue
                    # print(f'cur_communication_cost: {cur_communication_cost}')
                    # temp['distance'] += cur_communication_cost
                    if temp['distance'] < dp_res[i][j]['distance'] and len(temp['client_ids']) == j:
                        dp_res[i][j] = copy.deepcopy(temp)

        # if num_clients == 20:
        #     for i in range(len(dp_res)):
        #         print('\n')
        #         for j in range(len(dp_res[0])):
        #             print(f'{dp_res[i][j]["distance"]:.2f}', end=' ')
        #         print('\n')
        #         for j in range(len(dp_res[0])):
        #             print(f'{len(dp_res[i][j]["client_ids"])}', end=' ')     
                
            # a = 5
        min_distance = dp_res[-1][num_clients]['distance']
        min_client_ids = dp_res[-1][num_clients]['client_ids']
        # print(dp_res[-1])
        # print(f'num_clients: {num_clients}, min_distance: {min_distance}, min_client_ids: {min_client_ids}')
        # if count_smaller_than:
        min_distance_for_all = float('inf')
        min_client_ids_for_all = []
        # for i in range(len(dp_res)):
        for j in range(len(dp_res[0])):
            if dp_res[-1][j]['distance'] < min_distance_for_all:
                min_distance_for_all = dp_res[-1][j]['distance']
                min_client_ids_for_all = dp_res[-1][j]['client_ids']
        #     print(j, f'{dp_res[-1][j]["distance"]}')
        # print('\n')
        return min_distance, min_client_ids, min_distance_for_all, min_client_ids_for_all

    def dp_dynacomm_contrast_find_high_freq_group_clients(
            self,
            temp, 
            permutation_lists,
            dataset,
            logger,
            num_clients,
            # min_best_dp_KL_dist,
            # local_gradient_update_list_to_client_ratio
        ):

            # max_communication_cost = self.cal_communication_cost(
            #     model_size=model_size, 
            #     high_freq_client_num=len(temp), 
            #     low_freq_client_num=0, 
            #     high_freq_communication_times=high_freq_communication_times, 
            #     low_freq_communication_times=0
            # )
            # communication_cost_lambda = min_best_dp_KL_dist / max_communication_cost
            # communication_cost_lambda_multiplication_list = [1, 3, 5]
            # high_freq_ratio_list = [0.3, 0.5, 0.7, 0.9]
            if self.global_labels_distribution == None:
                self.global_labels_distribution = self.get_global_labels_distribution(dataset)
            # for i in range(len(high_freq_ratio_list)):
            start = time.time()
            best_dp_KL_dist = []
            best_dp_KL_combination_list = []
            for j in range(len(permutation_lists)):
                best_distance, best_combination, _, _ = self.dp(
                    dataset=dataset,
                    num_clients=num_clients, 
                    selected_client_ids=permutation_lists[j],
                    distance_type='KL',
                )
                # print(f'\n best_dp_KL_{num_clients}: {best_distance}, best_combination_{num_clients}: {best_combination}', flush=True)
                best_dp_KL_dist.append(best_distance)
                best_dp_KL_combination_list.append(best_combination)
            end = time.time()
            min_dist_pos = best_dp_KL_dist.index(min(best_dp_KL_dist))
            min_dist_combination = best_dp_KL_combination_list[min_dist_pos]
            print(f"dp KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ with cost time: {end-start}", flush=True)
            print(f"dp KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cost_{num_clients}: {min(best_dp_KL_dist)}", flush=True)
            print(f"dp KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_comb_communication_cost_{num_clients}: {min_dist_combination}", flush=True)
            logger.append(
                {
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cost_{num_clients}": min(best_dp_KL_dist),
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cost_{num_clients}_time": end-start,
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_combination_size_{num_clients}": len(min_dist_combination)
                }, 
                'train', 
            )

            
            return min(best_dp_KL_dist), min_dist_combination
    
    def dp_find_high_freq_group_clients(
            self,
            temp, 
            permutation_lists,
            dataset,
            logger,
            num_clients,
            # min_best_dp_KL_dist,
            # local_gradient_update_list_to_client_ratio
        ):

            # max_communication_cost = self.cal_communication_cost(
            #     model_size=model_size, 
            #     high_freq_client_num=len(temp), 
            #     low_freq_client_num=0, 
            #     high_freq_communication_times=high_freq_communication_times, 
            #     low_freq_communication_times=0
            # )
            # communication_cost_lambda = min_best_dp_KL_dist / max_communication_cost
            # communication_cost_lambda_multiplication_list = [1, 3, 5]
            # high_freq_ratio_list = [0.3, 0.5, 0.7, 0.9]
            if self.global_labels_distribution == None:
                self.global_labels_distribution = self.get_global_labels_distribution(dataset)
            # for i in range(len(high_freq_ratio_list)):
            start = time.time()
            best_dp_KL_dist = []
            best_dp_KL_combination_list = []
            for j in range(len(permutation_lists)):
                _, _, best_distance, best_combination = self.dp(
                    dataset=dataset,
                    num_clients=num_clients, 
                    selected_client_ids=permutation_lists[j],
                    distance_type='KL',
                )
                # print(f'\n best_dp_KL_{num_clients}: {best_distance}, best_combination_{num_clients}: {best_combination}', flush=True)
                best_dp_KL_dist.append(best_distance)
                best_dp_KL_combination_list.append(best_combination)
            end = time.time()
            min_dist_pos = best_dp_KL_dist.index(min(best_dp_KL_dist))
            min_dist_combination = best_dp_KL_combination_list[min_dist_pos]
            print(f"dp KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ with cost time: {end-start}", flush=True)
            print(f"dp KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cost_{num_clients}: {min(best_dp_KL_dist)}", flush=True)
            print(f"dp KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_comb_communication_cost_{num_clients}: {min_dist_combination}", flush=True)
            logger.append(
                {
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cost_{num_clients}": min(best_dp_KL_dist),
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cost_{num_clients}_time": end-start,
                    f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_combination_size_{num_clients}": len(min_dist_combination)
                }, 
                'train', 
            )

            
            return min_dist_pos, min_dist_combination
    
    
    def cal_communication_cost(
        self, 
        model_size, 
        high_freq_client_num, 
        low_freq_client_num, 
        high_freq_communication_times, 
        low_freq_communication_times
    ):
        return model_size * (high_freq_client_num * high_freq_communication_times * 2 + low_freq_client_num * low_freq_communication_times * 2)

    def get_high_and_low_freq_communication_time(self, local_gradient_update_list_to_client_ratio):
        high_freq_communication_times = 0
        low_freq_communication_times = float('inf')

        for item in local_gradient_update_list_to_client_ratio:
            high_freq_communication_times = max(high_freq_communication_times, len(item[0]))
            low_freq_communication_times = min(low_freq_communication_times, len(item[0]))
        return high_freq_communication_times-1, low_freq_communication_times-1







    

    