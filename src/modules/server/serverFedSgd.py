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
from models.api import (
    create_model,
    make_batchnorm
)
from optimizer.api import create_optimizer
from ..client.api import ClientFedSgd
from .serverBase import ServerBase

from ...data import separate_dataset


class ServerFedSgd(ServerBase):

    def __init__(
        self, 
        model: ModelType,
        clients: dict[int, ClientType]
    ) -> None:

        self.global_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        global_optimizer = create_optimizer(model, 'global')
        self.global_optimizer_state_dict = global_optimizer.state_dict()
        self.clients = clients

    def distribute_global_model_to_clients(
        self
    ) -> None:

        model = super().create_model(track_running_stats=False)
        model.load_state_dict(self.global_model_state_dict)
        global_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        for m in range(len(self.clients)):
            if self.clients[m].active:
                self.clients[m].model_state_dict = copy.deepcopy(global_model_state_dict)
        return

    def update_global_model(
        self, 
        client: dict[int, ClientType],
        epoch: int
    ) -> None:

        with torch.no_grad():
            # 修改
            valid_client = [client[i] for i in range(len(client)) if client[i].active]
            valid_client_idx = [i for i in range(len(client)) if client[i].active]
            
            if len(valid_client) > 0:
                model = super().create_model(track_running_stats=False)
                model.load_state_dict(self.global_model_state_dict)
                global_optimizer = create_optimizer(model, 'global')
                global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                global_optimizer.zero_grad()
                weight = torch.ones(len(valid_client))
                weight = weight / weight.sum()
                for k, v in model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(valid_client)):
                            tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                global_optimizer.step()
                self.global_optimizer_state_dict = global_optimizer.state_dict()
                self.global_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            for i in range(len(client)):
                client[i].active = False
        return
    
    # combinedDataset = combine_dataset(
#             num_active_clients=num_active_clients,
#             clients=clients,
#             client_ids=client_ids,
#         )
        
#         client = Client(
#             client_id=0, 
#             model=create_model(), 
#             data_split={
#                 'train': None, 
#                 'test': None
#             },
#             update_threshold=cfg['max_local_gradient_update'][m]
#         )

#         client[0].train(
#             dataset=combinedDataset, 
#             lr=lr, 
#             metric=metric, 
#             logger=logger,
#             grad_interval=grad_interval
#         )

  

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
        dataset = super().combine_train_dataset(
            num_active_clients=num_active_clients,
            clients=self.clients,
            selected_client_ids=selected_client_ids,
            dataset=dataset
        )
        self.distribute_global_model_to_clients()
        start_time = time.time()
        lr = optimizer.param_groups[0]['lr']

        # use 1 model to simulate FedSgd
        # TODO: dont have local optimizer
        client = ClientFedSgd(
            client_id=0, 
            model=create_model(), 
            data_split={
                'train': None, 
                'test': None
            },
        )

        client.train(
            dataset=dataset, 
            lr=lr, 
            metric=metric, 
            logger=logger,
            grad_updates_num=cfg['max_local_gradient_update']
        )

        super().add_log(
            i=0,
            num_active_clients=num_active_clients,
            start_time=start_time,
            global_epoch=global_epoch,
            lr=lr,
            selected_client_ids=selected_client_ids,
            metric=metric,
            logger=logger,
        )

        logger.safe(False)  
        self.update_global_model(
            clients=self.clients, 
            global_epoch=global_epoch
        ) 
        return
    

