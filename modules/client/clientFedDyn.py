from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
import models
import collections
from itertools import compress
from config import cfg

# from torchstat import stat

from utils.api import (
    to_device,  
    collate
)

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
    create_model
)

from optimizer.api import create_optimizer

from data import make_data_loader

from .clientBase import ClientBase


class ClientFedDyn(ClientBase):

    def __init__(
        self, 
        client_id: int, 
        model: ModelType, 
        data_split: list[int],
    ) -> None:

        super().__init__()
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self.prev_model_state_dict = copy.deepcopy(self.model_state_dict)
        optimizer = create_optimizer(model, 'client')
        self.optimizer_state_dict = optimizer.state_dict()
        self.active = False

        # initiate self.delta_l_k
        self.delta_l_k = copy.deepcopy(list(model.parameters()))
        for param_index, param in enumerate(model.parameters()):
            self.delta_l_k[param_index] = copy.deepcopy(param.data.new_zeros(param.size()))

    @classmethod
    def create_clients(
        cls,
        model: ModelType, 
        data_split: dict[str, dict[int, list[int]]],
    ) -> dict[int, object]:
        '''
        Create clients which organized in dict type
        
        Parameters
        ----------
        model: ModelType
        data_split: dict[str, dict[int, list[int]]]

        Returns
        -------
        dict[int, object]
        '''
        client_id = torch.arange(cfg['num_clients'])
        clients = [None for _ in range(cfg['num_clients'])]
        for m in range(len(clients)):
            clients[m] = ClientFedDyn(
                client_id=client_id[m], 
                model=model, 
                data_split={
                    'train': data_split['train'][m], 
                    'test': data_split['test'][m]
                },
            )
        return clients


    def train(
        self, 
        # dataset: DatasetType, 
        data_loader,
        lr: int, 
        metric: MetricType, 
        logger: LoggerType,
        feddyn_alpha
    ) -> None:

        model = create_model(track_running_stats=False, on_cpu=False)
        model.load_state_dict(self.model_state_dict, strict=False)

        global_weight_collector = list(model.parameters())
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = create_optimizer(model, 'client')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)
        feddyn_alpha = to_device(torch.tensor(feddyn_alpha), cfg['device'])
        # if cfg['merge_gap'] == True:
        #     new_optimizer_state_dict = super().update_optimizer_state_dict(
        #         client_model_state_dict=self.prev_model_state_dict,
        #         client_optimizer_state_dict=copy.deepcopy(self.optimizer_state_dict),
        #         client_optimizer_lr=lr,
        #         server_model_state_dict=copy.deepcopy(self.model_state_dict)
        #     )
        #     # new_optimizer_state_dict = to_device(new_optimizer_state_dict, cfg['device'])
        #     optimizer.load_state_dict(new_optimizer_state_dict)
        
        # Regenerate data loader if number of data_loader batches < grad_interval
        # number of data_loader batches = num of data points / batch_size
        # data_loader = make_data_loader(
        #     dataset={'train': dataset}, 
        #     tag='client'
        # )['train'] 
        # print('-----')
        # ceshi = 0
        # for local_epoch in range(1, cfg['local_epoch']+1):
            
            # print(f'clientFedAvg: {cur_grad_updates_num}')
        
        for i, input in enumerate(data_loader):

            input = collate(input)
            # print(f"input[id]: {input['id']}\n")
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            optimizer.zero_grad()
            output = model(input)
            
            local_weight_collector = list(model.parameters())
            # calculate 2 proximal terms
            delta_l_k_model_inner_product = 0.0
            prox_square = 0.0
            for param_index, param in enumerate(model.parameters()):
                delta_l_k_model_inner_product += torch.sum(self.delta_l_k[param_index] * \
                    local_weight_collector[param_index])
                # delta_l_k_model_inner_product += torch.sum(
                #     self.delta_l_k[param_index])
                prox_square += torch.sum(feddyn_alpha / 2 * torch.norm(local_weight_collector[param_index] - \
                    global_weight_collector[param_index])**2)

            # fed_prox_reg = 0.0
            # for param_index, param in enumerate(model.parameters()):
            #     fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
            output['loss'] = output['loss'] - delta_l_k_model_inner_product + prox_square
            # optimizer.zero_grad()
            output['loss'].backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['max_clip_norm'])
            optimizer.step()

            evaluation = metric.evaluate(
                metric.metric_name['train'], 
                input, 
                output
            )
            logger.append(
                evaluation, 
                'train', 
                n=input_size
            )
        
        # a = model.parameters()
        # b = model.state_dict()
        local_weight_collector = None
        local_weight_collector = list(model.parameters())
        for param_index, param in enumerate(model.parameters()):
            temp = copy.deepcopy(feddyn_alpha * (local_weight_collector[param_index].detach() - \
                global_weight_collector[param_index].detach()))
            self.delta_l_k[param_index] -= temp

        self.optimizer_state_dict = optimizer.state_dict()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        # if cfg['merge_gap'] == True:
        #     self.prev_model_state_dict = copy.deepcopy(self.model_state_dict)
        # print('wowow')
        return
# from __future__ import annotations

# import copy
# import datetime
# import numpy as np
# import sys
# import time
# import torch
# import torch.nn.functional as F
# import models
# import collections
# from itertools import compress
# from config import cfg

# # from torchstat import stat

# from utils.api import (
#     to_device,  
#     collate
# )

# from _typing import (
#     DatasetType,
#     OptimizerType,
#     DataLoaderType,
#     ModelType,
#     MetricType,
#     LoggerType,
#     ClientType,
#     ServerType
# )

# from models.api import (
#     create_model
# )

# from optimizer.api import create_optimizer

# from data import make_data_loader

# from .clientBase import ClientBase


# class ClientFedDyn(ClientBase):

#     def __init__(
#         self, 
#         client_id: int, 
#         model: ModelType, 
#         data_split: list[int],
#     ) -> None:

#         super().__init__()
#         self.client_id = client_id
#         self.data_split = data_split
#         self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
#         self.prev_model_state_dict = copy.deepcopy(self.model_state_dict)
#         optimizer = create_optimizer(model, 'client')
#         self.optimizer_state_dict = optimizer.state_dict()
#         self.active = False

#         # initiate self.delta_l_k
#         self.delta_l_k = copy.deepcopy(list(model.parameters()))
#         for param_index, param in enumerate(model.parameters()):
#             self.delta_l_k[param_index] = copy.deepcopy(param.data.new_zeros(param.size()))

#     @classmethod
#     def create_clients(
#         cls,
#         model: ModelType, 
#         data_split: dict[str, dict[int, list[int]]],
#     ) -> dict[int, object]:
#         '''
#         Create clients which organized in dict type
        
#         Parameters
#         ----------
#         model: ModelType
#         data_split: dict[str, dict[int, list[int]]]

#         Returns
#         -------
#         dict[int, object]
#         '''
#         client_id = torch.arange(cfg['num_clients'])
#         clients = [None for _ in range(cfg['num_clients'])]
#         for m in range(len(clients)):
#             clients[m] = ClientFedDyn(
#                 client_id=client_id[m], 
#                 model=model, 
#                 data_split={
#                     'train': data_split['train'][m], 
#                     'test': data_split['test'][m]
#                 },
#             )
#         return clients


#     def train(
#         self, 
#         # dataset: DatasetType, 
#         data_loader,
#         lr: int, 
#         metric: MetricType, 
#         logger: LoggerType,
#         feddyn_alpha
#     ) -> None:
#         '''
#         data_loader = make_data_loader({'train': dataset}, 'client')['train']
#         model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
#         model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
#         model.load_state_dict(self.model_state_dict, strict=False)
#         self.optimizer_state_dict['param_groups'][0]['lr'] = lr
#         optimizer = create_optimizer(model, 'local')
#         optimizer.load_state_dict(self.optimizer_state_dict)
#         model.train(True)
#         model.train(True)
#         for epoch in range(1, cfg['local']['num_epochs'] + 1):
#             for i, input in enumerate(data_loader):
#                 input = collate(input)
#                 input_size = input['data'].size(0)
#                 input = to_device(input, cfg['device'])
#                 optimizer.zero_grad()
#                 output = model(input)
#                 output['loss'].backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
#                 optimizer.step()
#                 evaluation = metric.evaluate(metric.metric_name['train'], input, output)
#                 logger.append(evaluation, 'train', n=input_size)
#         self.optimizer_state_dict = optimizer.state_dict()
#         self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
#         '''
#         model = create_model(track_running_stats=False, on_cpu=False)
#         model.load_state_dict(self.model_state_dict, strict=False)

#         global_weight_collector = copy.deepcopy(list(model.parameters()))
#         self.optimizer_state_dict['param_groups'][0]['lr'] = lr
#         optimizer = create_optimizer(model, 'client')
#         optimizer.load_state_dict(self.optimizer_state_dict)
#         model.train(True)

#         # if cfg['merge_gap'] == True:
#         #     new_optimizer_state_dict = super().update_optimizer_state_dict(
#         #         client_model_state_dict=self.prev_model_state_dict,
#         #         client_optimizer_state_dict=copy.deepcopy(self.optimizer_state_dict),
#         #         client_optimizer_lr=lr,
#         #         server_model_state_dict=copy.deepcopy(self.model_state_dict)
#         #     )
#         #     # new_optimizer_state_dict = to_device(new_optimizer_state_dict, cfg['device'])
#         #     optimizer.load_state_dict(new_optimizer_state_dict)
        
#         # Regenerate data loader if number of data_loader batches < grad_interval
#         # number of data_loader batches = num of data points / batch_size
#         # data_loader = make_data_loader(
#         #     dataset={'train': dataset}, 
#         #     tag='client'
#         # )['train'] 
#         # print('-----')
#         # ceshi = 0
#         # for local_epoch in range(1, cfg['local_epoch']+1):
            
#             # print(f'clientFedAvg: {cur_grad_updates_num}')
#         feddyn_alpha = to_device(torch.tensor(feddyn_alpha), cfg['device'])
#         for i, input in enumerate(data_loader):

#             input = collate(input)
#             # print(f"input[id]: {input['id']}\n")
#             input_size = input['data'].size(0)
#             input = to_device(input, cfg['device'])
#             optimizer.zero_grad()
#             output = model(input)
            
#             local_weight_collector = copy.deepcopy(list(model.parameters()))
#             # calculate 2 proximal terms
#             delta_l_k_model_inner_product = 0.0
#             l2_prox = 0.0
#             for param_index, param in enumerate(model.parameters()):
#                 delta_l_k_model_inner_product += torch.sum(copy.deepcopy(self.delta_l_k[param_index]) * \
#                     copy.deepcopy(local_weight_collector[param_index]))
#                 l2_prox += torch.sum(feddyn_alpha / 2 * torch.norm((local_weight_collector[param_index] - \
#                     global_weight_collector[param_index])**2))

#             # fed_prox_reg = 0.0
#             # for param_index, param in enumerate(model.parameters()):
#             #     fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
#             output['loss'] = output['loss'] - delta_l_k_model_inner_product
#             # optimizer.zero_grad()
#             output['loss'].backward()

#             # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['max_clip_norm'])
#             optimizer.step()

#             evaluation = metric.evaluate(
#                 metric.metric_name['train'], 
#                 input, 
#                 output
#             )
#             logger.append(
#                 evaluation, 
#                 'train', 
#                 n=input_size
#             )
        
#         local_weight_collector = copy.deepcopy(list(model.parameters()))
#         for param_index, param in enumerate(model.parameters()):
#             self.delta_l_k[param_index] -= feddyn_alpha * (local_weight_collector[param_index] - \
#                 global_weight_collector[param_index])

#         self.optimizer_state_dict = optimizer.state_dict()
#         self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
#         # feddyn_alpha = to_device(torch.tensor(feddyn_alpha), 'cpu')
#         # if cfg['merge_gap'] == True:
#         #     self.prev_model_state_dict = copy.deepcopy(self.model_state_dict)
#         # print('wowow')
#         return
