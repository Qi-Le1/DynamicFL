from __future__ import annotations

import math
import copy
import random
import collections
import numpy as np
from config import cfg

class Communication:
    def __init__(self, client_ids):
        self.high_freq_communication_times = self.cal_communication_time(cfg['high_freq_interval'])
        self.low_freq_communication_times = self.cal_communication_time(cfg['low_freq_interval'])

        self.client_maximum_communication_cost = self.cal_client_communication_cost(
            model_size=cfg['normalized_model_size'],
            communication_times=cfg['max_local_gradient_update'], 
        )

        self.client_high_freq_communication_cost = self.cal_client_communication_cost(
            model_size=cfg['normalized_model_size'],
            communication_times=self.high_freq_communication_times
        )
        # same at this time
        self.client_high_freq_communication_cost_budget = self.client_high_freq_communication_cost

        self.client_low_freq_communication_cost = self.cal_client_communication_cost(
            model_size=cfg['normalized_model_size'],
            communication_times=self.low_freq_communication_times
        )
        # same at this time
        self.client_low_freq_communication_cost_budget = self.client_low_freq_communication_cost

        self.client_minimum_communication_cost = self.cal_client_communication_cost(
            model_size=cfg['normalized_model_size'],
            communication_times=1
        )

        if cfg['server_ratio'] == '1-0':
            self.server_communication_cost_budget = float('inf')
            self.server_high_freq_communication_cost_budget = float('inf')
        else:
            self.server_communication_cost_budget = self.cal_server_communication_cost(
                model_size=cfg['normalized_model_size'],
                high_freq_client_num=cfg['num_active_clients'] * cfg['high_freq_ratio'],
                low_freq_client_num=cfg['num_active_clients'] * cfg['low_freq_ratio'],
                high_freq_communication_times=self.high_freq_communication_times,
                low_freq_communication_times=self.low_freq_communication_times
            )

            self.server_high_freq_communication_cost_budget = self.cal_server_communication_cost(
                model_size=cfg['normalized_model_size'],
                high_freq_client_num=cfg['num_active_clients'] * cfg['high_freq_ratio'],
                low_freq_client_num=0,
                high_freq_communication_times=self.high_freq_communication_times,
                low_freq_communication_times=0
            )

        self.client_to_freq_interval = collections.defaultdict(int)
        self.client_to_communication_cost = collections.defaultdict(int)
        self.client_to_communication_cost_budget = collections.defaultdict(int)
        self.initialize_clients_communication_info(client_ids=client_ids)

    def cal_communication_time(self, freq):
        return math.ceil(cfg['max_local_gradient_update'] / freq)

    def cal_client_communication_cost(
        self, 
        model_size, 
        communication_times, 
    ):
        return model_size * (communication_times * 2)
    
    def cal_server_communication_cost(
        self, 
        model_size, 
        high_freq_client_num, 
        low_freq_client_num, 
        high_freq_communication_times, 
        low_freq_communication_times
    ):  
        return model_size * (high_freq_client_num * high_freq_communication_times * 2 + low_freq_client_num * low_freq_communication_times * 2)
    
    def cal_active_clients_communication_cost(
        self, 
        clients,
        client_ids
    ):
        return sum([clients[client_id].communication_cost for client_id in client_ids])
    
    def initialize_clients_communication_info(
        self,
        client_ids: list[int]
    ):
        temp_client_ids = copy.deepcopy(client_ids)
        selected_client_ids = np.random.choice(
            temp_client_ids, 
            size=int(cfg['high_freq_ratio'] * len(temp_client_ids)), 
            replace=False
        )
        for client_id in selected_client_ids:
            self.client_to_freq_interval[client_id] = cfg['high_freq_interval']
            self.client_to_communication_cost[client_id] = self.client_high_freq_communication_cost
            self.client_to_communication_cost_budget[client_id] = self.client_high_freq_communication_cost_budget

        temp_client_ids = list(set(temp_client_ids) - set(selected_client_ids))
        for client_id in temp_client_ids:
            self.client_to_freq_interval[client_id] = cfg['low_freq_interval']
            self.client_to_communication_cost[client_id] = self.client_low_freq_communication_cost
            self.client_to_communication_cost_budget[client_id] = self.client_low_freq_communication_cost_budget
        return

    @classmethod
    def cal_communication_budget(
        cls,
        model_size: int,
        number_of_freq_levels: list[int]
    ) -> list[int]:

        communication_budget = [i * 2 * model_size for i in number_of_freq_levels]
        return communication_budget