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
    ClientType
)

from models import make_batchnorm
from optimizer.api import make_optimizer

class Server:

    def __init__(
        self, model: ModelType
    ) -> None:

        
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer = make_optimizer(model, 'local')
        self.optimizer_state_dict = optimizer.state_dict()
        global_optimizer = make_optimizer(model, 'global')
        self.global_optimizer_state_dict = global_optimizer.state_dict()

        self.client_state_dicts = {}
        self.client_params_diff = defaultdict(dict)
        self.client_params_diff_count = defaultdict(dict)


    def distribute(
        self, client: dict[int, ClientType]
    ) -> None:

        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=False))
        model.load_state_dict(self.model_state_dict)
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
        return

    def update(
        self, 
        client: dict[int, ClientType],
        epoch: int
    ) -> None:

        with torch.no_grad():
            # 修改
            valid_client = [client[i] for i in range(len(client)) if client[i].active]
            valid_client_idx = [i for i in range(len(client)) if client[i].active]
            
            if len(valid_client) > 0:
                model = eval('models.{}()'.format(cfg['model_name']))
                model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=False))
                model.load_state_dict(self.model_state_dict)
                global_optimizer = make_optimizer(model, 'global')
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
                self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                self.cal_diff(model, valid_client_idx, valid_client, epoch)
                self.store_client(valid_client_idx, valid_client)
            for i in range(len(client)):
                client[i].active = False
        return
    
    def store_client(
        self, 
        valid_client_idx: list[int],
        valid_client: list[ClientType],
    ) -> None:
        
        for m in range(len(valid_client_idx)):
            self.client_state_dicts[valid_client_idx[m]] = copy.deepcopy(valid_client[m].model_state_dict)
        return
    
    def cal_diff(
        self,
        model: ModelType,
        valid_client_idx: list[int],
        valid_client: list[ClientType],
        epoch: int
    ) -> None:

        # round 2
        # round 1
        for k, v in model.named_parameters():
            parameter_type = k.split('.')[-1]
            if 'weight' in parameter_type or 'bias' in parameter_type:
                for m in range(len(valid_client_idx)):
                    if valid_client_idx[m] in self.client_state_dicts:
                        diff = valid_client[m].model_state_dict[k] - self.client_state_dicts[valid_client_idx[m]][k]
                        diff_count = torch.count_nonzero(torch.gt(torch.abs(diff), cfg['diff_val'])).item()
                        # .float().mean()
                        # 0 => diff 4901450
                        # 1e-3(0.00001 => 0.001) => 10k
                        # print(f'diff_count: {diff_count}')
                        
                        self.client_params_diff[epoch][valid_client_idx[m]] = diff
                        self.client_params_diff_count[epoch][valid_client_idx[m]] = self.client_params_diff_count[epoch].get(valid_client_idx[m], 0) + diff_count

            # print('diff')
        if epoch in self.client_params_diff:
            # print('ggggg', self.client_params_diff[epoch])
            print(f'self.client_params_diff_count[epoch]: {self.client_params_diff_count[epoch]}')
        return