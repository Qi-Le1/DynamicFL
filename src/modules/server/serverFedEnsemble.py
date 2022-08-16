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

from ...data import separate_dataset


class ServerFedAvg(ServerBase):

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
    
    def evaluate_ensemble(
        self, 
        clients,
        selected_client_ids,
        combined_test_dataset
    ):

        test_acc=0
        loss=0
        # TODO: 什么是testloaderfull, test dataset的总和？
        for x, y in self.testloaderfull:
            target_logit_output=0
            for user in users:
                user.model.eval()
                user_result = user.model(x, logit=True)
                target_logit_output += user_result['logit']
            # dim=1, 按行算
            target_logp = F.log_softmax(target_logit_output, dim=1)
            test_acc += torch.sum( torch.argmax(target_logp, dim=1) == y ) #(torch.sum().item()
            loss+=self.loss(target_logp, y)
        loss = loss.detach().numpy()
        test_acc = test_acc.detach().numpy() / y.shape[0]
        self.metrics['glob_acc'].append(test_acc)
        self.metrics['glob_loss'].append(loss)
        print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(test_acc, loss))

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
        self.distribute_global_model_to_clients()
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

        self.update_global_model(
            clients=self.clients, 
            global_epoch=global_epoch
        ) 

        # combine dataset for ensemble evaluation
        combined_test_dataset = super().combine_test_dataset(
            num_active_clients=num_active_clients,
            clients=self.clients,
            selected_client_ids=selected_client_ids,
            dataset=dataset
        )
        self.evaluate_ensemble(
            clients=self.clients,
            selected_client_ids=selected_client_ids,
            combined_test_dataset=combined_test_dataset
        )
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
        return
    



    def test(
        self,
        dataset,
        batchnorm_dataset,
        logger,
        metric,
        global_epoch
    ):  
        data_loader = make_data_loader(dataset, 'global')

        model = super().create_model()
        model.load_state_dict(self.model_state_dict)
        batchnorm_dataset = make_batchnorm_dataset(dataset['train'])
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')

        logger.safe(True)
        with torch.no_grad():
            model.train(False)
            for i, input in enumerate(data_loader):

                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(input)['output']
                loss = self.nll_loss(output, input['target'])
                output = self.reform_model_output(
                    output=output,
                    loss=loss
                )

                evaluation = metric.evaluate(
                    metric.metric_name['test'], 
                    input, 
                    output
                )
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
            logger.append(info, 'test', mean=False)
            print(logger.write('test', metric.metric_name['test']))
        logger.safe(False)
        return
