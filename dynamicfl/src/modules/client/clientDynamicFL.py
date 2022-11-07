from __future__ import annotations


import copy
import datetime
from turtle import update
import numpy as np
import sys
import time
import math
import torch
import random
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
import collections

# from torchstat import stat

from utils.api import (
    to_device,  
    collate
)

from typing import Any

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

from optimizer.api import create_optimizer

from data import make_data_loader

from .clientBase import ClientBase


class ClientDynamicFL(ClientBase):

    def __init__(
        self, 
        client_id: int, 
        model: ModelType, 
        data_split: list[int],
        local_gradient_update_list: list[int],
    ) -> None:

        super().__init__()
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer = create_optimizer(model, 'client')
        self.optimizer_state_dict = optimizer.state_dict()
        self.local_gradient_update_list=local_gradient_update_list
        self.active = False

    @classmethod
    def create_communication_meta_data(cls, client_ids: list[int]=None) -> list[int]:
        communicationMetaData = {}
        if cfg['select_client_mode'] == 'fix':
            client_ratio_to_update_thresholds = Communication.cal_fix_update_thresholds(
                max_local_gradient_update=cfg['max_local_gradient_update'], 
                client_ratio_to_number_of_uploads=cfg['client_ratio_to_number_of_uploads']
            )
            local_gradient_update_list = Communication.distribute_fix_local_gradient_update_list(
                client_ids=client_ids, 
                max_local_gradient_update=cfg['max_local_gradient_update'], 
                client_ratio_to_update_thresholds=client_ratio_to_update_thresholds
            )
            communicationMetaData['local_gradient_update_list'] = local_gradient_update_list
        elif cfg['select_client_mode'] == 'nonpre':
            client_ratio_to_update_thresholds = Communication.cal_fix_update_thresholds(
                max_local_gradient_update=cfg['max_local_gradient_update'], 
                client_ratio_to_number_of_uploads=cfg['client_ratio_to_number_of_uploads']
            )
            local_gradient_update_dict = Communication.calculate_local_gradient_update_list(
                max_local_gradient_update=cfg['max_local_gradient_update'], 
                client_ratio_to_update_thresholds=client_ratio_to_update_thresholds
            )
            communicationMetaData['local_gradient_update_dict'] = local_gradient_update_dict
        else:
            raise ValueError('select_client_mode must in fix or dynamic')
        # print(f'communicationMetaData: {communicationMetaData}')
        return communicationMetaData

    @classmethod
    def create_clients(
        cls,
        model: ModelType, 
        data_split: dict[str, dict[int, list[int]]],
    ) -> dict[int, ClientType]:
        client_ids = torch.arange(cfg['num_clients'])
        clients = [None for _ in range(cfg['num_clients'])]
        communicationMetaData = cls.create_communication_meta_data(client_ids=client_ids)
        for m in range(len(clients)):
            clients[m] = ClientDynamicFL(
                client_id=client_ids[m], 
                model=model, 
                data_split={
                    'train': data_split['train'][m], 
                    'test': data_split['test'][m]
                },
                local_gradient_update_list=communicationMetaData['local_gradient_update_list'][m]
            )
        return clients

    def total_params_num(self, model: ModelType):
        # a = model.parameters()
        total_num = 0
        for k, v in model.named_parameters():
            parameter_type = k.split('.')[-1]
            if 'weight' in parameter_type or 'bias' in parameter_type:
                total_num += v.numel()
        # total_num = sum(p.numel() for p in model.parameters())
        print(f'zhetotal_num: {total_num}')
        return total_num

    def train(
        self, 
        dataset: DatasetType, 
        lr: int, 
        metric: MetricType, 
        logger: LoggerType,
        grad_updates_num: int
    ) -> None:

        model = create_model()
        model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = create_optimizer(model, 'client')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)

        cur_grad_updates_num = 0
        while cur_grad_updates_num < grad_updates_num:
            # Regenerate data loader if number of data_loader batches < grad_interval
            # number of data_loader batches = num of data points / batch_size
            data_loader = make_data_loader(
                dataset={'train': dataset}, 
                tag='client'
            )['train'] 

            for i, input in enumerate(data_loader):
                cur_grad_updates_num += 1                      
                if cur_grad_updates_num == grad_updates_num + 1:
                    break

                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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
        self.optimizer_state_dict = optimizer.state_dict()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        return


class Communication:
    '''
    Class to handle communication issue in DynamicFL
    '''
    # TODO: dynamic/fix client raio
    @classmethod
    def cal_fix_update_thresholds(
        cls,
        max_local_gradient_update: int,
        client_ratio_to_number_of_uploads: dict[float, list[int]]
    ) -> dict[float, list[int]]:
        '''
        Calculate update_threshold
        based on max_local_gradient_update and number_of_uploads
        
        Parameters
        ----------
        client_id : list[int]
        max_local_gradient_update : int
        ratio_to_number_of_uploads : dict[float, list[int]]
            The key is the ratio of distributing client to each number_of_uploads
            The value is the number_of_uploads

        Returns
        -------
        list[int]

        Notes
        -----
        update_threshold = int(max_local_gradient_update/number_of_uploads)
        '''
        client_ratio_to_update_thresholds = collections.defaultdict(list)
        for ratio, number_of_uploads in client_ratio_to_number_of_uploads.items():
            for number in number_of_uploads:
                client_ratio_to_update_thresholds[ratio].append(int(max_local_gradient_update/number))

        return client_ratio_to_update_thresholds

    @classmethod
    def cal_local_gradient_update_list(
        cls,
        update_threshold: int
    ) -> int:
        '''
        calculate which local_gradient_update we need to go to inner loop
        of dynamicFL.

        Example:
        If the cur_local_gradient_update == 5, then local_gradient_update_list 
        is [1, 6, 11]. 
        '''
        cur_local_gradient_update = 1
        local_gradient_update_list = []
        while cur_local_gradient_update <= cfg['max_local_gradient_update']:
            local_gradient_update_list.append(cur_local_gradient_update)
            cur_local_gradient_update += update_threshold
            
        return local_gradient_update_list

    @classmethod
    def distribute_fix_local_gradient_update_list(
        cls,
        client_ids: list[int],
        max_local_gradient_update: int,
        client_ratio_to_update_thresholds: dict[float, list[int]]
    ) -> list[list[int]]:
        '''
        Distribute list[local_gradient_update] for each client based on
        update_threshold
        local_gradient_update indicates that the client needs to enter
        dynamicFL algo
        
        Parameters
        ----------
        client_id : list[int]
        max_local_gradient_update : int
        ratio_to_update_thresholds : dict[float, list[int]]
            The key is the ratio of distributing client to each update_thresholds
            The value is the update_thresholds

        Returns
        -------
        clients_update_threshold : list[list[int]]. index are the same as the index of client_ids, elements represents
            the list of local_gradient_update

        Notes
        -----
        Minimum number of uploads is 1(client must upload once in
        each local gradient update cycle)
        Maximum number of uploads is max_local_gradient_update
        '''
        temp_client_ids = copy.deepcopy(client_ids)
        temp_client_ids = temp_client_ids.tolist()
        clients_update_threshold = list(range(len(client_ids)))
        for ratio, update_threshold in client_ratio_to_update_thresholds.items():
            for threshold in update_threshold:
                threshold = min(
                    max_local_gradient_update, 
                    max(1, threshold)
                )
                if len(temp_client_ids) == 0:
                    raise ValueError('The length of selected_client_ids is 0')

                selected_client_ids = random.sample(
                    temp_client_ids, 
                    min(math.ceil(ratio * len(client_ids)), len(temp_client_ids))
                )
                
                # torch.arange(cfg['num_nodes'])[torch.randperm(cfg['num_nodes'])[:num_active_nodes]].tolist()
                local_gradient_update_list = cls.cal_local_gradient_update_list(
                    update_threshold=threshold
                )

                for index in selected_client_ids:
                    clients_update_threshold[index] = copy.deepcopy(local_gradient_update_list)
                # clients_update_threshold[selected_client_ids] = cls.cal_local_gradient_update_list(
                #     update_threshold=update_threshold
                # )
                temp_client_ids = list(set(temp_client_ids) - set(selected_client_ids))

        return clients_update_threshold
    
    @classmethod
    def calculate_local_gradient_update_list(
        cls,
        max_local_gradient_update: int,
        client_ratio_to_update_thresholds: dict[float, list[int]]
    ) -> list[list[int]]:
        '''
        calculate local gradient update list based on update_threshold
        local_gradient_update indicates that the client needs to enter
        dynamicFL algo
        
        Parameters
        ----------
        max_local_gradient_update : int
        ratio_to_update_thresholds : dict[float, list[int]]
            The key is the ratio of distributing client to each update_thresholds
            The value is the update_thresholds

        Returns
        -------
        local_gradient_update_dict : dict[int, list[int]]

        Notes
        -----
        Minimum number of uploads is 1(client must upload once in
        each local gradient update cycle)
        Maximum number of uploads is max_local_gradient_update
        '''
        local_gradient_update_dict = collections.defaultdict(list)
        for ratio, update_threshold in client_ratio_to_update_thresholds.items():
            for threshold in update_threshold:
                threshold = min(
                    max_local_gradient_update, 
                    max(1, threshold)
                )
                if len(temp_client_ids) == 0:
                    raise ValueError('The length of selected_client_ids is 0')
                
                local_gradient_update_list = cls.cal_local_gradient_update_list(
                    update_threshold=threshold
                )

                local_gradient_update_dict[ratio] = local_gradient_update_list

        return local_gradient_update_dict
    
    # @classmethod
    # def distribute_dynamic_update_thresholds(
    #     cls,
    #     client: dict[int, ClientType],
    #     client_id: list[int],
    #     ratio_to_update_thresholds: dict[float, int]
    # ) -> None:
    #     '''
    #     Distribute update_thresholds to clients based on ratio.
        
    #     Parameters
    #     ----------
    #     client : dict[int, ClientType]
    #     client_id : list[int]
    #     ratio_to_update_thresholds : dict[float, int]
    #         The key is the ratio of distributing client to each update_thresholds
    #         The value is the update_thresholds

    #     Returns
    #     -------
    #     None
    #     '''
    #     temp_client_id = copy.deepcopy(client_id)
    #     client_to_update_threshold = [_ for _ in range(len(client_id))]
    #     for ratio, update_threshold in ratio_to_update_thresholds.items():
    #         selected_client_ids = random.choice(
    #             temp_client_id, 
    #             ratio * len(client_id)
    #         )
    #         client_to_update_threshold[selected_client_ids] = update_threshold
    #         temp_client_id = list(set(temp_client_id) - set(selected_client_ids))

    #     for selected_client_id, update_threshold in client_to_update_threshold.items():
    #         client[selected_client_id].update_threshold = update_threshold
    #     return

    @classmethod
    def cal_communication_budget(
        cls,
        model_size: int,
        number_of_uploads: list[int]
    ) -> list[int]:
        '''
        Calculate communication budget
        based on model size and number_of_uploads
        
        Parameters
        ----------
        model_size : int,
        number_of_uploads : list[int]

        Returns
        -------
        list[int]

        Notes
        -----
        communication budget = number_of_uploads * 2 * model_size
        '''
        communication_budget = [i * 2 * model_size for i in number_of_uploads]
        return communication_budget