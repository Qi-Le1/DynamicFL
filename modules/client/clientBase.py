from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import math
import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg

# from torchstat import stat

from utils.api import (
    to_device,  
    collate
)

from models.api import (
    create_model,
    make_batchnorm
)

from _typing import (
    ClientType,
    ModelType,
    DatasetType,
    MetricType,
    LoggerType
)

from models.api import make_batchnorm

from optimizer.api import create_optimizer

from data import make_data_loader


class ClientBase:

    def __init__(self) -> None:
        # self.init_loss_fn()
        pass
    
    def reform_model_output(self, output, loss):
        '''
        Reform the structure of output to adapt the original code
        with FedGen / FedEnsemble
        '''
        res = {
            'target': output,
            'loss': loss
        }
        return res


    def update_optimizer_state_dict(
        self,
        client_model_state_dict,
        client_optimizer_state_dict,
        client_optimizer_lr,
        server_model_state_dict
    ):
        '''
        Mitigate the gap between the optimizer state dict for 
        client model and the new server model
        '''
        client_model = create_model(track_running_stats=False, on_cpu=True)
        client_model.load_state_dict(client_model_state_dict, strict=False)
        client_optimizer_state_dict['param_groups'][0]['lr'] = client_optimizer_lr
        client_optimizer = create_optimizer(client_model, 'client')
        client_optimizer.load_state_dict(client_optimizer_state_dict)
        with torch.no_grad():  
            for k, v in client_model.named_parameters():
                parameter_type = k.split('.')[-1]
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    tmp_v = v.data.new_zeros(v.size())
                    tmp_v += server_model_state_dict[k]
                    v.grad = (v.data - tmp_v).detach()
            # clip
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), 1)
            client_optimizer.step()
        return copy.deepcopy(client_optimizer.state_dict())


# class ClientSampler(torch.utils.data.Sampler):
#     def __init__(self, batch_size, active_rate, data_split, max_local_gradient_update, selected_client_ids):
#         self.batch_size = batch_size
#         self.active_rate = active_rate
#         self.data_split = data_split
#         self.max_local_gradient_update = max_local_gradient_update
#         self.selected_client_ids = selected_client_ids
#         self.num_active_clients = len(self.selected_client_ids)
#         self.reset()

#     def reset(self):
#         # self.data_split_ = copy.deepcopy(self.data_split)
#         # self.client_idx = torch.arange(len(self.data_split))
#         # self.idx = []

#         # while len(self.client_idx) > 0:
#         #     num_active_clients = min(self.num_active_clients, len(self.client_idx))
#         #     active_client_idx = self.client_idx[torch.randperm(len(self.client_idx))][:num_active_clients]
#         #     batch_idx = []
#         #     for i in range(len(active_client_idx)):
#         #         data_split_i = self.data_split_[active_client_idx[i].item()]
#         #         batch_size_i = min(self.batch_size, len(data_split_i))
#         #         batch_idx.extend(data_split_i[:batch_size_i])
#         #         self.data_split_[active_client_idx[i].item()] = self.data_split_[active_client_idx[i].item()][
#         #                                                         batch_size_i:]
#         #     self.client_idx = torch.tensor([i for i in range(len(self.data_split_)) if len(self.data_split_[i]) > 0])
#         #     self.idx.append(batch_idx)
#         self.data_split_ = copy.deepcopy(self.data_split)
#         # self.duplicate_data_split = copy.deepcopy(self.data_split)
#         # self.client_idx = torch.arange(len(self.data_split))

#         # dont limit the local updates
#         if cfg['local_upper_bound'] == 99999:
#             self.idx = []
#             while self.max_local_gradient_update > 0:
#                 batch_idx = []
#                 for client_id in self.selected_client_ids:
#                     data_split_i = self.data_split_[client_id]
#                     batch_size_i = min(self.batch_size, len(data_split_i))
#                     # chosen_eles = random.choices(data_split_i, k=batch_size_i)
#                     chosen_eles = np.random.choice(data_split_i, size=batch_size_i, replace=True)
#                     # chosen_eles = data_split_i[:batch_size_i]
#                     # self.data_split_[client_id] = self.data_split_[client_id][batch_size_i:]
#                     # if len(self.data_split_[client_id]) == 0:
#                     #     self.data_split_[client_id] = copy.deepcopy(self.data_split[client_id])
#                     batch_idx.extend(chosen_eles)
                    
#                 # print(len(batch_idx))
#                 self.idx.append(batch_idx)
#                 self.max_local_gradient_update -= 1
#         else:
#             # limit the local updates
#             max_local_gradient_update_for_each_client = self.cal_max_local_gradient_update_for_each_client(
#                 selected_client_ids=self.selected_client_ids,
#                 data_split=self.data_split
#             )
#             # print(f'max_local_gradient_update_for_each_client: {max_local_gradient_update_for_each_client}')
#             # data_split_length = []
#             # for key, val in self.data_split.items():
#             #     data_split_length.append(len(val))
#             self.idx = []
#             while self.max_local_gradient_update > 0:
#                 batch_idx = []
#                 for client_id in self.selected_client_ids:
#                     if client_id not in max_local_gradient_update_for_each_client:
#                         data_split_i = self.data_split_[client_id]
#                         batch_size_i = min(self.batch_size, len(data_split_i))
#                         # chosen_eles = random.choices(data_split_i, k=batch_size_i)
#                         chosen_eles = np.random.choice(self.data_split, size=batch_size_i, replace=True)
#                         batch_idx.extend(chosen_eles)
#                     else:
#                         if max_local_gradient_update_for_each_client[client_id] == 0:
#                             continue
#                         data_split_i = self.data_split_[client_id]
#                         batch_size_i = min(
#                             self.batch_size, 
#                             len(data_split_i),
#                             max_local_gradient_update_for_each_client[client_id]
#                         )
#                         chosen_eles = random.choices(data_split_i, k=batch_size_i)
#                         batch_idx.extend(chosen_eles)
#                         max_local_gradient_update_for_each_client[client_id] -= batch_size_i
#                 self.idx.append(batch_idx)
#                 self.max_local_gradient_update -= 1
#         # print(f'self.idx: {self.idx}')
#         return

#     def cal_max_local_gradient_update_for_each_client(
#         self,
#         selected_client_ids,
#         data_split
#     ) -> dict:  
#         '''
#         calculate max_local_gradient_update for each client
#         based on the cfg['local_upper_bound']
#         '''
#         data_split_ = copy.deepcopy(data_split)
#         threshold = math.ceil(cfg['max_local_gradient_update']*cfg['client']['batch_size']['train'] / cfg['local_upper_bound'])
#         max_local_gradient_update_for_each_client = {}
#         for client_id in selected_client_ids:
#             if len(data_split_[client_id]) >= threshold:
#                 pass
#             else:
#                 max_local_gradient_update_for_each_client[client_id] = len(data_split_[client_id]) * cfg['local_upper_bound']

#         return max_local_gradient_update_for_each_client

#     def __iter__(self):
#         yield from self.idx

#     def __len__(self):
#         return len(self.idx)