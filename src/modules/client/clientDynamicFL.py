from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg

# from torchstat import stat

from utils import (
    to_device,  
    collate
)

from _typing import (
    ModelType,
    DatasetType,
    MetricType,
    LoggerType
)

from models import make_batchnorm

from optimizer.api import create_optimizer

from data import make_data_loader

from .clientBase import ClientBase


class ClientDynamicFL(ClientBase):

    def __init__(
        self, 
        client_id: int, 
        model: ModelType, 
        data_split: list[int],
        update_threshold: int
    ) -> None:

        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self.update_threshold=update_threshold
        self.total_params_num(model)
        optimizer = create_optimizer(model, 'local')
        self.optimizer_state_dict = optimizer.state_dict()
        self.active = False
        self.buffer = None

    @classmethod
    def create_clients(
        cls,
        model: ModelType, 
        data_split: dict[str, dict[int, list[int]]],
    ) -> dict[int, ClientType]:
        client_id = torch.arange(cfg['num_clients'])
        client = [None for _ in range(cfg['num_clients'])]
        client_to_update_threshold = make_communication(client_id=client_id)
        for m in range(len(client)):
            client[m] = Client(
                client_id=client_id[m], 
                model=model, 
                data_split={
                    'train': data_split['train'][m], 
                    'test': data_split['test'][m]
                },
                update_threshold=client_to_update_threshold[m]
            )
        return client

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
        grad_interval: int
    ) -> None:

        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=False))
        model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = create_optimizer(model, 'local')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)

        updated_batch_num = 0
        while updated_batch_num < grad_interval:
            # Regenerate data loader if number of data_loader batches < grad_interval
            # number of data_loader batches = num of data points / batch_size
            data_loader = make_data_loader(
                dataset={'train': dataset}, 
                tag='client'
            )['train'] 

            for i, input in enumerate(data_loader):
                updated_batch_num += 1                      
                if updated_batch_num == grad_interval + 1:
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
                    metric_names=metric.metric_name['train'], 
                    intput=input, 
                    output=output
                )
                logger.append(
                    result=evaluation, 
                    tag='train', 
                    n=input_size
                )
                 
        self.optimizer_state_dict = optimizer.state_dict()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        # for epoch in range(1, cfg['local']['num_epochs'] + 1):
        #     for i, input in enumerate(data_loader):
        #         input = collate(input)
        #         input_size = input['data'].size(0)
        #         input = to_device(input, cfg['device'])
        #         optimizer.zero_grad()
        #         output = model(input)
        #         output['loss'].backward()
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        #         optimizer.step()
        #         evaluation = metric.evaluate(
        #             metric_names=metric.metric_name['train'], 
        #             intput=input, 
        #             output=output
        #         )
        #         logger.append(
        #             result=evaluation, 
        #             tag='train', 
        #             n=input_size
        #         )
        # self.optimizer_state_dict = optimizer.state_dict()
        # self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        return

class Communication:

    # Every iteration:
    #     number_of_uploads: 2-3-4
    #     ratio: 0.1-0.4-0.5
    # TODO: dynamic/fix client raio
    @classmethod
    def distribute_fix_update_thresholds(
        cls,
        client_id: list[int],
        max_local_gradient_update: int,
        ratio_to_update_thresholds: dict[float, int]
    ) -> list[int]:
        '''
        Distribute update_thresholds to clients based on ratio.
        
        Parameters
        ----------
        client_id : list[int]
        max_local_gradient_update : int
        ratio_to_update_thresholds : dict[float, int]
            The key is the ratio of distributing client to each update_thresholds
            The value is the update_thresholds

        Returns
        -------
        list[int]

        Notes
        -----
        Minimum number of uploads is 1(client must upload once in
        each local gradient update cycle)
        Maximum number of uploads is max_local_gradient_update
        '''
        temp_client_id = copy.deepcopy(client_id)
        client_to_update_threshold = [_ for _ in range(len(client_id))]
        for ratio, update_threshold in ratio_to_update_thresholds.items():
            update_threshold = min(
                max_local_gradient_update, 
                max(1, update_threshold)
            )
            selected_client_ids = random.choice(
                temp_client_id, 
                ratio * len(client_id)
            )
            client_to_update_threshold[selected_client_ids] = update_threshold
            temp_client_id = list(set(temp_client_id) - set(selected_client_ids))

        return client_to_update_threshold
    
    @classmethod
    def distribute_dynamic_update_thresholds(
        cls,
        client: dict[int, ClientType],
        client_id: list[int],
        ratio_to_update_thresholds: dict[float, int]
    ) -> None:
        '''
        Distribute update_thresholds to clients based on ratio.
        
        Parameters
        ----------
        client : dict[int, ClientType]
        client_id : list[int]
        ratio_to_update_thresholds : dict[float, int]
            The key is the ratio of distributing client to each update_thresholds
            The value is the update_thresholds

        Returns
        -------
        None
        '''

        temp_client_id = copy.deepcopy(client_id)
        client_to_update_threshold = [_ for _ in range(len(client_id))]
        for ratio, update_threshold in ratio_to_update_thresholds.items():
            selected_client_ids = random.choice(
                temp_client_id, 
                ratio * len(client_id)
            )
            client_to_update_threshold[selected_client_ids] = update_threshold
            temp_client_id = list(set(temp_client_id) - set(selected_client_ids))

        for selected_client_id, update_threshold in client_to_update_threshold.items():
            client[selected_client_id].update_threshold = update_threshold
        return

    @classmethod
    def cal_fix_update_thresholds(
        cls,
        client_id: list[int],
        max_local_gradient_update: int,
        ratio_to_number_of_uploads: dict[float, int]
    ) -> dict[float, int]:
        '''
        Calculate update_threshold
        based on max_local_gradient_update and
        predefined_ratio
        
        Parameters
        ----------
        client_id : list[int]
        max_local_gradient_update : int
        ratio_to_number_of_uploads : dict[float, int]
            The key is the ratio of distributing client to each number_of_uploads
            The value is the number_of_uploads

        Returns
        -------
        list[int]

        Notes
        -----
        update_threshold = int(max_local_gradient_update/number_of_uploads)
        '''
        ratio_to_update_thresholds = {}
        for ratio, number_of_uploads in ratio_to_number_of_uploads.items():
            ratio_to_update_thresholds[ratio] = int(max_local_gradient_update/number_of_uploads)

        return ratio_to_update_thresholds

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