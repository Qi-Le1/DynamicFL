from __future__ import annotations

import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn as nn
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

from utils.api import (
    to_device,  
    collate
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


class ServerFedEnsemble(ServerBase):

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
        self.selected_client_ids = None

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
        
        logger.safe(True)
        selected_client_ids, num_active_clients = super().select_clients(clients=self.clients) 
        self.selected_client_ids = selected_client_ids       
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
            elif dataset_m is not None:
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
    
    # def evaluate_ensemble(
    #     self, 
    # ):
        # test_acc=0
        # loss=0
        # # TODO: 什么是testloaderfull, test dataset的总和？
        # for x, y in self.testloaderfull:
        #     target_logit_output=0
        #     for user in users:
        #         user.model.eval()
        #         user_result = user.model(x, logit=True)
        #         target_logit_output += user_result['logit']
        #     # dim=1, 按行算
        #     target_logp = F.log_softmax(target_logit_output, dim=1)
        #     test_acc += torch.sum( torch.argmax(target_logp, dim=1) == y ) #(torch.sum().item()
        #     loss+=self.loss(target_logp, y)
        # loss = loss.detach().numpy()
        # test_acc = test_acc.detach().numpy() / y.shape[0]
        # self.metrics['glob_acc'].append(test_acc)
        # self.metrics['glob_loss'].append(loss)
        # print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(test_acc, loss))
        return 

    def evaluate_trained_model(
        self,
        dataset,
        logger,
        metric,
        global_epoch,
        **kwargs
    ):  
        data_loader = make_data_loader(
            dataset={'test': dataset}, 
            tag='server'
        )['test']
        
        test_models = []
        for i in range(len(self.selected_client_ids)):
            m = self.selected_client_ids[i]
            # if self.clients[m].active == True:
            test_models.append(super().create_test_model(
                model_state_dict=self.clients[m].model_state_dict
            ))

        if len(test_models) == 0:
            raise ValueError('Ensemble test models are empty')

        logger.safe(True)
        with torch.no_grad():
            test_acc = 0
            loss = 0
            for i, input in enumerate(data_loader):

                a = input
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                
                # evaluate ensemble
                target_logit_output = 0
                for test_model in test_models:
                    test_model.train(False)
                    temp_input = copy.deepcopy(input)
                    output = test_model(temp_input)
                    # print(f"output[target]: {output['target']}")
                    target_logit_output += output['target']
                    # print(f'target_logit_output: {target_logit_output}')
                # 对每一行的所有元素进行softmax运算，并使得每一行所有元素和为1
                target_logp = F.log_softmax(target_logit_output, dim=1)
                temp_input = copy.deepcopy(input)
                test_acc += torch.sum( torch.argmax(target_logp, dim=1) == temp_input['target'] ) #(torch.sum().item()
                nll_loss = nn.NLLLoss()
                loss += nll_loss(target_logp, temp_input['target'])

            loss = loss.detach().numpy()
            test_acc = test_acc.detach().numpy() / temp_input['target'].shape[0]

            # loss = self.nll_loss(output, input['target'])
            # output = self.reform_model_output(
            #     output=output,
            #     loss=loss
            # )

            evaluation = {}
            evaluation['Loss'] = loss
            evaluation['Accuracy'] = test_acc
            logger.append(
                evaluation, 
                'test', 
                input_size
            )
                
            info = {
                'info': [
                    'Model: {}'.format(cfg['model_tag']), 
                    'Test Epoch: {}({:.0f}%)'.format(global_epoch, 100.)
                ]
            }
            logger.append(
                info, 
                'test', 
                mean=False
            )
            print(logger.write('test', metric.metric_name['test']))
        logger.safe(False)
        return
