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

from utils.api import (
    to_device,  
    collate
)
from optimizer.api import create_optimizer
from .serverBase import ServerBase

from ...data import (
    fetch_dataset, 
    split_dataset, 
    make_data_loader, 
    separate_dataset, 
    make_batchnorm_dataset, 
    make_batchnorm_stats
)

class ServerDynamicFL(ServerBase):

    def __init__(
        self, model: ModelType
    ) -> None:

        self.server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        server_optimizer = create_optimizer(model, 'global')
        self.server_optimizer_state_dict = server_optimizer.state_dict()

        # dict[int, list], record the participated clients
        # at certain local gradient update
        self.dynamic_uploaded_clients = defaultdict(list)
        # dict[int, list], record the model state dict
        # of the participated clients at certain
        # local gradient update
        self.dynamic_iterates = defaultdict(list)
    
    def distribute_dynamic_part(
        self,
        local_gradient_update: int,
        clients: dict[int, ClientType]
    ):
        if local_gradient_update not in self.dynamic_uploaded_clients:
            return

        for cur_client_id in self.dynamic_uploaded_clients[local_gradient_update]:
            clients[cur_client_id].model_state_dict = copy.deepcopy(self.server_model_state_dict)
        return

    def upload_dynamic_part(
        self,
        target_gradent_update: int,
        cur_client_id: int,
        client: ModelType
    ) -> None:
        '''
        handle dynamic logic, do union operation 
        '''
        self.dynamic_uploaded_clients[target_gradent_update].append(cur_client_id)
        self.dynamic_iterates[target_gradent_update].append(copy.deepcopy(client.model_state_dict))
        return

    def update_dynamic_part(
        self,
        local_gradient_update: int,
    ):
        with torch.no_grad():

            if local_gradient_update not in self.dynamic_uploaded_clients:
                return
            
            new_model_parameters_list = self.dynamic_iterates[local_gradient_update]
            
            if len(new_model_parameters_list) > 0:
                model = super().create_model()
                model.load_state_dict(self.server_model_state_dict)
                server_optimizer = create_optimizer(model, 'global')
                server_optimizer.load_state_dict(self.server_optimizer_state_dict)
                server_optimizer.zero_grad()
                weight = torch.ones(len(new_model_parameters_list))
                weight = weight / weight.sum()
                for k, v in model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(new_model_parameters_list)):
                            tmp_v += weight[m] * new_model_parameters_list[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                server_optimizer.step()
                self.server_optimizer_state_dict = server_optimizer.state_dict()
                self.server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            
            # delete the data store in 2 dicts
            del self.dynamic_uploaded_clients[local_gradient_update]
            del self.dynamic_iterates[local_gradient_update]
        return

    def distribute_server_model_to_clients(
        self, 
    ) -> None:

        model = super().create_model(track_running_stats=False)
        model.load_state_dict(self.server_model_state_dict)
        server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        for m in range(len(self.clients)):
            if self.clients[m].active:
                self.clients[m].model_state_dict = copy.deepcopy(server_model_state_dict)
        return

    def update_server_model(self, client: dict[int, ClientType]) -> None:
        with torch.no_grad():
            valid_client = [client[i] for i in range(len(client)) if client[i].active]
            if valid_client:
                model = super().create_model(track_running_stats=False)
                model.load_state_dict(self.server_model_state_dict)
                server_optimizer = create_optimizer(model, 'global')
                server_optimizer.load_state_dict(self.server_optimizer_state_dict)
                server_optimizer.zero_grad()
                weight = torch.ones(len(valid_client))
                weight = weight / weight.sum()
                for k, v in model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(valid_client)):
                            tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                server_optimizer.step()
                self.server_optimizer_state_dict = server_optimizer.state_dict()
                self.server_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

            for i in range(len(client)):
                client[i].active = False
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
        self.distribute_server_model_to_clients()
        start_time = time.time()
        lr = optimizer.param_groups[0]['lr']

        for local_gradient_update in range(1, cfg['max_local_gradient_update'] + 1):
            self.distribute_dynamic_part(
                local_gradient_update=local_gradient_update,
                clients=self.clients
            )

            for i in range(num_active_clients):

                if not self.is_local_gradient_update_valid(
                    local_gradient_update=local_gradient_update,
                    local_gradient_update_list=self.clients[m].local_gradient_update_list
                ):
                    continue

                m = selected_client_ids[i]
                dataset_m = separate_dataset(dataset, self.clients[m].data_split['train'])

                gradient_updates_num = self.cal_gradient_updates_num(
                    local_gradient_update=local_gradient_update,
                    local_gradient_update_list=self.clients[m].local_gradient_update_list
                )

                self.clients[m].train(
                    dataset=dataset_m, 
                    lr=lr, 
                    metric=metric, 
                    logger=logger,
                    gradient_updates_num=gradient_updates_num
                )

                self.upload_dynamic_part(
                    target_gradent_update=local_gradient_update+gradient_updates_num,
                    cur_client_id=m,
                    client=self.clients[m]
                )
            
            self.update_dynamic_part(target_gradent_update=local_gradient_update+1)
                
            super().add_dynamicFL_log(
                local_gradient_update=local_gradient_update,
                start_time=start_time,
                global_epoch=global_epoch,
                lr=lr,
                metric=metric,
                logger=logger,
            )

        logger.safe(False)
        self.update_server_model(
            clients=self.clients, 
            global_epoch=global_epoch
        )
        return

    def cal_gradient_updates_num(
        self,
        local_gradient_update: int,
        local_gradient_update_list: list[int]
    ) -> int:

        index = local_gradient_update_list.index(local_gradient_update)
        # already is last element
        if index == len(local_gradient_update_list) - 1:
            return cfg['max_local_gradient_update'] + 1 - local_gradient_update
        else:
            return local_gradient_update_list[index+1] - local_gradient_update

    def is_local_gradient_update_valid(
        self,
        local_gradient_update: int,
        local_gradient_update_list: list[int]
    ) -> bool:

        return local_gradient_update in local_gradient_update_list