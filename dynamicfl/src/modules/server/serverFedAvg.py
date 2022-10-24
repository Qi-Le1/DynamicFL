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
    DatasetType,
    OptimizerType,
    DataLoaderType,
    ModelType,
    MetricType,
    LoggerType,
    ClientType,
    ServerType
)

from optimizer.api import create_optimizer
from .serverBase import ServerBase

from data import (
    fetch_dataset, 
    split_dataset, 
    make_data_loader, 
    separate_dataset, 
    make_batchnorm_dataset, 
    make_batchnorm_stats
)


class ServerFedAvg(ServerBase):

    def __init__(
        self, 
        model: ModelType,
        clients: dict[int, ClientType],
        dataset: DatasetType
    ) -> None:

        super().__init__(dataset=dataset)
        self.server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        server_optimizer = create_optimizer(model, 'server')
        self.server_optimizer_state_dict = server_optimizer.state_dict()
        self.clients = clients

    def update_server_model(self, clients: dict[int, ClientType]) -> None:
        with torch.no_grad():
            valid_clients = [clients[i] for i in range(len(clients)) if clients[i].active]
            if valid_clients:
                model = super().create_model(track_running_stats=False, on_cpu=True)
                model.load_state_dict(self.server_model_state_dict)
                server_optimizer = create_optimizer(model, 'server')
                server_optimizer.load_state_dict(self.server_optimizer_state_dict)
                server_optimizer.zero_grad()
                weight = torch.ones(len(valid_clients))
                weight = weight / weight.sum()
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
    
    def train(
        self,
        dataset: DatasetType,  
        optimizer: OptimizerType, 
        metric: MetricType, 
        logger: LoggerType, 
        global_epoch: int
    ):
        print(f'global_epoch is: {global_epoch}')
        logger.safe(True)
        selected_client_ids, num_active_clients = super().select_clients(clients=self.clients)
        super().distribute_server_model_to_clients(
            server_model_state_dict=self.server_model_state_dict,
            clients=self.clients
        )
        start_time = time.time()
        lr = optimizer.param_groups[0]['lr']

        for i in range(num_active_clients):
            m = selected_client_ids[i]
            dataset_m = separate_dataset(dataset, self.clients[m].data_split['train'])
            if dataset_m is None:
                self.clients[m].active = False
            else:
                self.clients[m].active = True
                self.clients[m].train(
                    dataset=dataset_m, 
                    lr=lr, 
                    metric=metric, 
                    logger=logger,
                    grad_updates_num=cfg['max_local_gradient_update']
                )

            super().add_log(
                i=i,
                num_active_clients=num_active_clients,
                start_time=start_time,
                global_epoch=global_epoch,
                lr=lr,
                selected_client_ids=selected_client_ids,
                metric=metric,
                logger=logger,
            )
        logger.safe(False)
        self.update_server_model(clients=self.clients) 
        return
    
    def evaluate_trained_model(
        self,
        dataset,
        logger,
        metric,
        global_epoch
    ):  
        return super().evaluate_trained_model(
            dataset=dataset,
            logger=logger,
            metric=metric,
            global_epoch=global_epoch,
            server_model_state_dict=self.server_model_state_dict
        )