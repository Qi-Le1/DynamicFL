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
from collections import defaultdict

from _typing import (
    ModelType,
    ClientType,
    DatasetType
)

from models.api import (
    create_model,
    make_batchnorm
)

from ...data import (
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
        dataset
    ) -> None:

        self.train_batchnorm_dataset = make_batchnorm_dataset(dataset['train'])
        return
    
    def create_model(self, track_running_stats=False):
        return create_model(track_running_stats=track_running_stats)

    def create_test_model(
        self,
        model_state_dict
    ) -> object:

        model = create_model()
        model.load_state_dict(model_state_dict)
        test_model = make_batchnorm_stats(self.train_batchnorm_dataset, model, 'global')

        return test_model

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
                seconds=round((cfg['global']['num_epochs'] - global_epoch) * _time * num_active_clients))
            exp_progress = 100. * i / num_active_clients
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                            'Train Epoch (C): {}({:.0f}%)'.format(global_epoch, exp_progress),
                            'Learning rate: {:.6f}'.format(lr),
                            'ID: {}({}/{})'.format(selected_client_ids[i], i + 1, num_active_clients),
                            'Global Epoch Finished Time: {}'.format(global_epoch_finished_time),
                            'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))

    def add_dynamicFL_log(
        local_gradient_update,
        start_time,
        global_epoch,
        lr,
        # selected_client_ids,
        metric,
        logger
    ) -> None:
        if local_gradient_update % int((cfg['max_local_gradient_update'] * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (local_gradient_update + 1)
            global_epoch_finished_time = datetime.timedelta(seconds=_time * (cfg['max_local_gradient_update'] - local_gradient_update - 1))
            exp_finished_time = global_epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - global_epoch) * _time * cfg['max_local_gradient_update']))
            exp_progress = 100. * local_gradient_update / cfg['max_local_gradient_update']
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                            'Train Epoch (C): {}({:.0f}%)'.format(global_epoch, exp_progress),
                            'Learning rate: {:.6f}'.format(lr),
                            # 'ID: {}({}/{})'.format(selected_client_ids[i], i + 1, cfg['max_local_gradient_update']),
                            'Global Epoch Finished Time: {}'.format(global_epoch_finished_time),
                            'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))

        return




    def select_clients(
        self, clients: dict[int, ClientType]
    ) -> tuple[list[int], int]:

        num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
        selected_client_ids = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
        for i in range(num_active_clients):
            clients[selected_client_ids[i]].active = True
        
        return selected_client_ids, num_active_clients
    
    def combine_train_dataset(
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
            combined_datapoint_idx += clients[m].data_split['train']

        # dataset: DatasetType
        dataset = separate_dataset(dataset, combined_datapoint_idx)
        return dataset
    
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
