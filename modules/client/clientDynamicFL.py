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
    ) -> None:

        super().__init__()
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer = create_optimizer(model, 'client')
        self.optimizer_state_dict = optimizer.state_dict()
        self.active = False

        self.freq_interval = None
        self.communication_cost_budget = None
        
    @classmethod
    def create_clients(
        cls,
        model: ModelType, 
        data_split: dict[str, dict[int, list[int]]],
    ) -> dict[int, ClientType]:
        client_ids = torch.arange(cfg['num_clients'])
        clients = [None for _ in range(cfg['num_clients'])]
        for m in range(len(clients)):
            clients[m] = ClientDynamicFL(
                client_id=client_ids[m], 
                model=model, 
                data_split={
                    'train': data_split['train'][m], 
                    'test': data_split['test'][m]
                },
            )
        return clients

    def train(
        self, 
        data_loader,
        lr: int, 
        metric: MetricType, 
        logger: LoggerType,
        grad_updates_num: int
    ) -> None:

        if grad_updates_num == 0:
            raise ValueError('grad_updates_num must > 0')

        model = create_model()
        model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = create_optimizer(model, 'client')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)

        cur_grad_updates_num = 1
        while cur_grad_updates_num <= grad_updates_num:
            for i, input in enumerate(data_loader): 
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()

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
                cur_grad_updates_num += 1                      
                if cur_grad_updates_num == grad_updates_num + 1:
                    break

        self.optimizer_state_dict = optimizer.state_dict()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        return


