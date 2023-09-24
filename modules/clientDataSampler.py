from __future__ import annotations

import copy
import numpy as np
import time
import torch
import torch.nn.functional as F
from config import cfg


class ClientDataSampler(torch.utils.data.Sampler):
    def __init__(self, 
        batch_size, 
        data_split, 
        client_id=None, 
        max_local_gradient_update=250, 
        selected_client_ids=None,
        high_freq_clients=None,
        group_clients_prob_distribution=None,
        cur_client_prob_distribution=None,
        dataset=None
    ):
        self.batch_size = batch_size
        self.data_split = data_split
        self.max_local_gradient_update = max_local_gradient_update
        self.client_id = client_id
        self.selected_client_ids = selected_client_ids
        self.high_freq_clients = high_freq_clients
        self.group_clients_prob_distribution = group_clients_prob_distribution
        self.cur_client_prob_distribution = cur_client_prob_distribution
        # self.dataset = dataset
        self.reset()
        self.start = 0
        self.end = len(self.idx)


    def extend_data_split(
        self, 
        local_gradient_update, 
        batch_size,
        client_id=None
    ):
        # sample without replacement
        if cfg['algo_mode'] == 'dynamicsgd':
            total_data_size = local_gradient_update * batch_size
            new_data_split = []
            while len(new_data_split) <= total_data_size:
                chosen_eles = np.random.choice(self.data_split[client_id], size=len(self.data_split[client_id]), replace=False)
                new_data_split.extend(chosen_eles)
        else:
            total_data_size = local_gradient_update * batch_size
            new_data_split = []
            while len(new_data_split) <= total_data_size:
                chosen_eles = np.random.choice(self.data_split, size=len(self.data_split), replace=False)
                new_data_split.extend(chosen_eles)
        return new_data_split

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start < self.end:
            res = self.start
            self.start += 1
            return self.idx[res]
        else:
            self.start = 0
            if cfg['algo_mode'] == 'dynamicfl':
                self.reset()
            else:
                raise StopIteration

    def __len__(self):
        return len(self.idx)
    
    def reset(self):
        # self.data_split_ = copy.deepcopy(self.data_split)
        self.idx = []
        if cfg['algo_mode'] == 'dynamicsgd':
            print('zhelijinlaile')
            for client_id in self.selected_client_ids:
                self.data_split[client_id] = self.extend_data_split(
                    local_gradient_update=self.max_local_gradient_update, 
                    batch_size=self.batch_size,
                    client_id=client_id
            ) 
            start = 0
            while self.max_local_gradient_update > 0:
                batch_idx = []
                for client_id in self.selected_client_ids:
                    # data_split_i = self.data_split_[client_id]
                    # batch_size_i = min(self.batch_size, len(data_split_i))
                    # chosen_eles = random.choices(data_split_i, k=batch_size_i)
                    # chosen_eles = np.random.choice(data_split_i, size=batch_size_i, replace=True)
                    chosen_eles = copy.deepcopy(self.data_split[client_id][start: start+self.batch_size])  
                    # batch_idx.extend(chosen_eles)
                    self.idx.append(chosen_eles)
                # self.idx.append(batch_idx)
                self.max_local_gradient_update -= 1
                start += self.batch_size
            a = 5
        elif cfg['algo_mode'] == 'dynamicfl' or cfg['algo_mode'] == 'dynamicavg':
            start = 0
            self.data_split = self.extend_data_split(
                local_gradient_update=self.max_local_gradient_update, 
                batch_size=self.batch_size
            )
            while self.max_local_gradient_update > 0:
                batch_idx = []
                chosen_eles = copy.deepcopy(self.data_split[start: start+self.batch_size])   
                batch_idx.extend(chosen_eles)
                self.idx.append(batch_idx)
                self.max_local_gradient_update -= 1
                start += self.batch_size
        elif cfg['algo_mode'] == 'fedavg' or cfg['algo_mode'] == 'scaffold' \
            or cfg['algo_mode'] == 'fedprox' or cfg['algo_mode'] == 'feddyn' \
            or cfg['algo_mode'] == 'fedensemble' or cfg['algo_mode'] == 'fedgen' or cfg['algo_mode'] == 'fednova':

            self.batch_size = min(self.batch_size, len(self.data_split))
            cur_local_gradient_update = int(cfg['local_epoch'] * len(self.data_split) / self.batch_size)
            start = 0
            self.data_split = self.extend_data_split(
                local_gradient_update=cur_local_gradient_update, 
                batch_size=self.batch_size
            )
            while cur_local_gradient_update > 0:
                batch_idx = []
                chosen_eles = copy.deepcopy(self.data_split[start: start+self.batch_size])      
                batch_idx.extend(chosen_eles)
                self.idx.append(batch_idx)
                cur_local_gradient_update -= 1
                start += self.batch_size
        else:
            raise ValueError('wrong algo mode')
        return



    def reweight_local_data_prob(self):
        start = time.time()
        client_prob = np.array(self.cur_client_prob_distribution) 
        # a = client_prob != 0
        group_clients_correspoding_prob = np.zeros(len(self.group_clients_prob_distribution))
        # group_clients_correspoding_prob = np.array([0 for _ in range(len(self.group_clients_prob_distribution))])
        # ceshi = self.group_clients_prob_distribution
        for i in range(len(client_prob)):
            # print(client_prob[i])
            if client_prob[i] > 0:
                # print('yes', self.group_clients_prob_distribution[i])
                group_clients_correspoding_prob[i] = copy.deepcopy(self.group_clients_prob_distribution[i])
                # print('zz', group_clients_correspoding_prob[i])
        # print(f'1111: {group_clients_correspoding_prob}')
        group_clients_correspoding_prob = group_clients_correspoding_prob / sum(group_clients_correspoding_prob)
        # print(f'group_clients_correspoding_prob: {group_clients_correspoding_prob}')
        # class-balanced sampling
        for i in range(len(group_clients_correspoding_prob)):
            if group_clients_correspoding_prob[i] > 0:
                group_clients_correspoding_prob[i] = 1/group_clients_correspoding_prob[i]
        normalized_group_clients_correspoding_prob = group_clients_correspoding_prob / sum(group_clients_correspoding_prob)
        # print('normalized_group_clients_correspoding_prob', normalized_group_clients_correspoding_prob)
        # for each sample
        for i in range(len(normalized_group_clients_correspoding_prob)):
            if normalized_group_clients_correspoding_prob[i]:
                normalized_group_clients_correspoding_prob[i] /= (len(self.data_split) * self.cur_client_prob_distribution[i])
        # print('dier normalized_group_clients_correspoding_prob', normalized_group_clients_correspoding_prob)
        # print('dier', normalized_group_clients_correspoding_prob)
        target_list = np.array([self.dataset[index]['target'].item() for index in self.data_split])
        reweight_local_data_prob = np.array([normalized_group_clients_correspoding_prob[target] for target in target_list])
        # reweight_local_data_prob = reweight_local_data_prob / sum(reweight_local_data_prob)
        # b = sum(reweight_local_data_prob)
        end = time.time()
        # print('haoshi:', end-start)
        return reweight_local_data_prob