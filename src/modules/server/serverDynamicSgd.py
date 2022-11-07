from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import time
import torch
import random
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
from ..client.api import ClientDynamicSgd
from .serverBase import ServerBase

from data import (
    separate_dataset,
    make_data_loader
)

class ServerDynamicSgd(ServerBase):

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
            # valid_clients = [clients[i] for i in range(len(clients)) if clients[i].active]
            valid_clients = [clients[0]]
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
    
    def combine_train_data_split(
        self,
        num_active_clients: int,
        clients: dict[int, ClientType],
        selected_client_ids: list[int],
    ) -> list[int]:  
        '''
        combine the datapoint index for selected clients
        and return the dataset
        '''
        combined_datapoint_idx = []
        for i in range(num_active_clients):
            m = selected_client_ids[i]
            combined_datapoint_idx += copy.deepcopy(clients[0].data_split['train'][m])

        random.shuffle(combined_datapoint_idx)
        # dataset: DatasetType
        # combined_set = set(combined_datapoint_idx)
        # print(f'combined_set_size: {len(combined_set)}')
        return combined_datapoint_idx

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
         
        start_time = time.time()
        lr = optimizer.param_groups[0]['lr']

        # combine the data for selected clients into 1 dataset
        # data_split_index_list = self.combine_train_data_split(
        #     num_active_clients=num_active_clients,
        #     clients=self.clients,
        #     selected_client_ids=selected_client_ids,
        # )  
        # dataset_0 = separate_dataset(dataset, data_split_index_list)

        client_sampler = ClientSampler(
            batch_size=cfg['client']['batch_size']['train'], 
            active_rate=cfg['active_rate'], 
            data_split=self.clients[0].data_split['train'],
            max_local_gradient_update=cfg['max_local_gradient_update'],
            selected_client_ids=selected_client_ids
        )
        # if dataset_0 is None:
        #     self.clients[0].active = False
        # else:
        self.clients[0].active = True
        self.clients[0].train(
            dataset=dataset, 
            client_sampler=client_sampler,
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
        
class ClientSampler(torch.utils.data.Sampler):
    def __init__(self, batch_size, active_rate, data_split, max_local_gradient_update, selected_client_ids):
        self.batch_size = batch_size
        self.active_rate = active_rate
        self.data_split = data_split
        self.max_local_gradient_update = max_local_gradient_update
        self.selected_client_ids = selected_client_ids
        self.num_active_clients = len(self.selected_client_ids)
        self.reset()

    def reset(self):
        # self.data_split_ = copy.deepcopy(self.data_split)
        # self.client_idx = torch.arange(len(self.data_split))
        # self.idx = []

        # while len(self.client_idx) > 0:
        #     num_active_clients = min(self.num_active_clients, len(self.client_idx))
        #     active_client_idx = self.client_idx[torch.randperm(len(self.client_idx))][:num_active_clients]
        #     batch_idx = []
        #     for i in range(len(active_client_idx)):
        #         data_split_i = self.data_split_[active_client_idx[i].item()]
        #         batch_size_i = min(self.batch_size, len(data_split_i))
        #         batch_idx.extend(data_split_i[:batch_size_i])
        #         self.data_split_[active_client_idx[i].item()] = self.data_split_[active_client_idx[i].item()][
        #                                                         batch_size_i:]
        #     self.client_idx = torch.tensor([i for i in range(len(self.data_split_)) if len(self.data_split_[i]) > 0])
        #     self.idx.append(batch_idx)
        self.data_split_ = copy.deepcopy(self.data_split)
        self.duplicate_data_split = copy.deepcopy(self.data_split)
        # self.client_idx = torch.arange(len(self.data_split))
        self.idx = []

        while self.max_local_gradient_update > 0:
            # num_active_clients = min(self.num_active_clients, len(self.client_idx))
            # active_client_idx = self.client_idx[torch.randperm(len(self.client_idx))][:num_active_clients]
            batch_idx = []
            for client_id in range(len(self.selected_client_ids)):
                data_split_i = self.data_split_[client_id]
                batch_size_i = min(self.batch_size, len(data_split_i))
                batch_idx.extend(data_split_i[:batch_size_i])
                self.data_split_[client_id] = self.data_split_[client_id][batch_size_i:]
                if len(self.data_split_[client_id]) == 0:
                    recover_data_split = copy.deepcopy(self.duplicate_data_split[client_id][:])
                    random.shuffle(recover_data_split)
                    self.data_split_[client_id] = recover_data_split
            # self.client_idx = torch.tensor([i for i in range(len(self.data_split_)) if len(self.data_split_[i]) > 0])
            self.idx.append(batch_idx)
            self.max_local_gradient_update -= 1
        return

    def __iter__(self):
        yield from self.idx

    def __len__(self):
        return len(self.idx)