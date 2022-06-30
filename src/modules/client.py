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

from optimizer.api import make_optimizer

from data import make_data_loader


class Client:

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
        optimizer = make_optimizer(model, 'local')
        self.optimizer_state_dict = optimizer.state_dict()
        self.active = False
        self.buffer = None

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

        data_loader = make_data_loader(
            dataset={'train': dataset}, 
            tag='client'
        )['train']
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=False))
        model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = make_optimizer(model, 'local')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)
        
        for i, input in enumerate(data_loader):
            if i == grad_interval:
                break
        # for l in range(1, grad_interval + 1):
            # pick batch
            
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
