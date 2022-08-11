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
    ClientType,
    ModelType,
    DatasetType,
    MetricType,
    LoggerType
)

from models import make_batchnorm

from optimizer.api import make_optimizer

from data import make_data_loader


class ClientBase:

    def __init__(self) -> None:
        return

    