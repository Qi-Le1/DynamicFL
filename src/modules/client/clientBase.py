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

# from torchstat import stat

from utils import (
    to_device,  
    collate
)

from _typing import (
    ClientType,
    ModelType,
    DatasetType,
    MetricType,
    LoggerType
)

from models import make_batchnorm

from optimizer.api import create_optimizer

from data import make_data_loader


class ClientBase:

    def __init__(self) -> None:
        self.init_loss_fn()
    
    def reform_model_output(self, output, loss):
        '''
        Reform the structure of output to adapt the original code
        with FedGen / FedEnsemble
        '''
        res = {
            'target': output,
            'loss': loss
        }
        return res


    def init_loss_fn(self):
        # negative log likelihood loss
        self.nll_loss=nn.NLLLoss()
        # gen用了
        self.mse_loss = nn.MSELoss()
        # distill和gen用了
        self.kldiv_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

