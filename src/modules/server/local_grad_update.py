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


class ServerLocalGradUpdate:

    def __init__(
        self, model: ModelType
    ) -> None:

        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer = make_optimizer(model, 'local')
        self.optimizer_state_dict = optimizer.state_dict()
        global_optimizer = make_optimizer(model, 'global')
        self.global_optimizer_state_dict = global_optimizer.state_dict()

        # dict[int, list]
        self.uploaded_clients = defaultdict(list)
        # dict[int, list]
        self.iterates = defaultdict(list)
    
    def upload(
        self,
        target_grad_u: int,
        cur_client_id: int,
        client: ModelType
    ) -> None:

        self.uploaded_clients[target_grad_u].append(cur_client_id)
        self.iterates[target_grad_u].append(copy.deepcopy(client.model_state_dict))

        return


    def distribute(
        self, 
        local_grad_u: int,
        client: dict[int, ClientType]
    ) -> None:

        if local_grad_u not in self.uploaded_clients:
            return

        for cur_client_id in self.uploaded_clients[local_grad_u]:
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=False))
            model.load_state_dict(self.model_state_dict)
            model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            client[cur_client_id].model_state_dict = copy.deepcopy(model_state_dict)
        return

    def update(
        self, 
        local_grad_u: int,
        # client: dict[int, ClientType],
        # epoch: int
    ) -> None:
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
        with torch.no_grad():

            if local_grad_u not in self.uploaded_clients:
                return
            
            new_model_parameters_list = self.iterates[local_grad_u]
            # valid_client = [client[i] for i in range(len(client)) if client[i].active]
            # valid_client_idx = [i for i in range(len(client)) if client[i].active]
            
            if len(new_model_parameters_list) > 0:
                model = eval('models.{}()'.format(cfg['model_name']))
                model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=False))
                model.load_state_dict(self.model_state_dict)
                global_optimizer = make_optimizer(model, 'global')
                global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                global_optimizer.zero_grad()
                weight = torch.ones(len(new_model_parameters_list))
                weight = weight / weight.sum()
                for k, v in model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        # for m in range(len(valid_client)):
                        #     tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                        for m in self.iterates[local_grad_u]:
                            tmp_v += weight[m] * new_model_parameters_list[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                global_optimizer.step()
                self.global_optimizer_state_dict = global_optimizer.state_dict()
                self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            
            # delete the data store in 2 dicts
            del self.uploaded_clients[local_grad_u]
            del self.iterates[local_grad_u]

        return