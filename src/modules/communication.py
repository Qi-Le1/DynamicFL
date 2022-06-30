from __future__ import annotations

import copy
import datetime
from turtle import update
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
import models
from numpy import number, random
from itertools import compress
from config import cfg
from collections import defaultdict

from models import make_batchnorm
from optimizer.api import make_optimizer

from typing import Union

from src._typing import (
    ModelType,
    ClientType,
    Local_Gradient_Update
)


class Communication:

    @classmethod
    def cal_number_of_uploads(
        cls,
        client_id: list[int],
        max_local_gradient_update: int,
        predefined_ratio: dict[float, Local_Gradient_Update]
    ) -> list[int]:
        '''
        Calculate number of uploads for 
        all clients.
        
        Parameters
        ----------
        client_id : list[int]
        max_local_gradient_update : int
        predefined_ratio : dict[float, Local_Gradient_Update]
            The key of this dict is the ratio of each local_gradient_update
            The value of thie dict is the local_gradient_update list input by
            user

        Returns
        -------
        list[int]

        Notes
        -----
        Minimum number of uploads is 1(client must upload once in
        a max local gradient update cycle)
        Maximum number of uploads is max_local_gradient_update
        '''
        number_of_uploads = [_ for _ in range(len(client_id))]
        for ratio, local_gradient_update in predefined_ratio.items():
            local_gradient_update = min(
                max_local_gradient_update, 
                max(1, local_gradient_update)
            )
            selected_index = random.choice(
                range(len(client_id)), 
                ratio * len(client_id)
            )
            number_of_uploads[selected_index] = local_gradient_update

        return number_of_uploads

    @classmethod
    def cal_communication_budget(
        cls,
        model_size: int,
        number_of_uploads: list[int]
    ) -> list[int]:
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
        communication_budget = [i * 2 * model_size for i in number_of_uploads]
        return communication_budget

    @classmethod
    def cal_update_thresholds(
        cls,
        max_local_gradient_update: int,
        number_of_uploads: list[int]
    ) -> list[int]:
        '''
        Calculate update_threshold
        based on max_local_gradient_update and
        number_of_uploads
        
        Parameters
        ----------
        model_size : int,
        number_of_uploads : list[int]

        Returns
        -------
        list[int]

        Notes
        -----
        update_threshold = int(max_local_gradient_update/number_of_uploads)
        '''
        update_threshold = [int(max_local_gradient_update/i) for i in number_of_uploads]
        return update_threshold