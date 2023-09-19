import copy
import math
import torch
import random
import numpy as np
from config import cfg
from sko.GA import GA
from scipy.special import rel_entr

class ClientSelector():
    def __init__(
        self,
        client_ids,
        clients,
        dataset,
        data_split,
        communication_info
    ):  
        self.client_ids = client_ids
        self.clients = clients
        self.dataset = dataset
        self.data_split = data_split

        self.max_local_gradient_update = cfg['max_local_gradient_update']
        self.clients_label_distribution = {}
        self.global_label_distribution = self.get_global_labels_distribution(dataset)

        self.communication_info = communication_info

        self.high_freq_client_ids = [key for key, _ in communication_info.client_to_freq_interval.items()]
        self.low_freq_client_ids = [key for key, _ in communication_info.client_to_freq_interval.items()]
        self.total_client_ids = copy.deepcopy(self.client_ids)

        self.high_freq_clients_list = []
        self.low_freq_clients_list = []

        self.high_freq_clients_list_index = 0
        self.low_freq_clients_list_index = 0

        self.preprocess_for_client_selection()
        return
    
    @property
    def previous_high_freq_clients(self):
        return self.high_freq_clients_list[self.high_freq_clients_list_index - 1]
    
    @property
    def previous_low_freq_clients(self):
        return self.low_freq_clients_list[self.low_freq_clients_list_index - 1]

    @property
    def current_high_freq_clients(self):
        res = self.high_freq_clients_list[self.high_freq_clients_list_index]
        self.high_freq_clients_list_index += 1
        return res
    
    @property
    def current_low_freq_clients(self):
        res = self.low_freq_clients_list[self.low_freq_clients_list_index]
        self.low_freq_clients_list_index += 1
        return res
    
    def cal_KL_divergence(self, comb_prob, global_labels_distribution):
        return sum(rel_entr(comb_prob, global_labels_distribution))
    
    def cal_QL(self, list_1, list_2):
        return sum([(item[0]-item[1])**2 for item in zip(list_1, list_2)]) 

    def cal_prob_distribution(self, dataset, data_split, client_id):
        if client_id in self.clients_label_distribution:
            return self.clients_label_distribution[client_id]
        target_list = np.array([dataset[index]['target'].item() for index in data_split])
        sub_prob = []
        for i in range(len(dataset.classes_counts)):
            sub_prob.append(sum(target_list == i)/len(target_list))
        self.client_prob_distribution[client_id] = sub_prob
        return np.array(sub_prob)
    
    def get_global_labels_distribution(self, dataset):
        total_dp = sum(dataset.classes_counts.values())
        return [val / total_dp for val in dataset.classes_counts.values()]

    def get_selected_client_ids_permutation_lists(self, selected_client_ids):
        permutation_lists = []
        for i in range(cfg['dp_ensemble_times']):
            temp = copy.deepcopy(selected_client_ids)
            np.random.shuffle(temp)
            permutation_lists.append(copy.deepcopy(temp))
        return permutation_lists
    
    def cal_genetic_dist_func(self, *args):
        clients_indices_indicator = args[0]
        comb_prob = np.array([0 for _ in range(len(self.dataset.classes_counts))])
        total_size = 0
        for i in range(len(clients_indices_indicator)):
            if int(clients_indices_indicator[i]) == 1:
                client_id = self.temp_genetic_client_ids[i]
                total_size += len(self.clients[client_id].data_split['train'])

        for i in range(len(clients_indices_indicator)):
            if int(clients_indices_indicator[i]) == 1:
                client_id = self.temp_genetic_client_ids[i]
                sub_prob = self.cal_prob_distribution(self.dataset, self.data_split['train'][client_id], client_id)
                ratio = len(self.data_split['train'][client_id])/total_size
                sub_prob = np.array([prob*ratio for prob in sub_prob])
                comb_prob = comb_prob + sub_prob

        res = self.cal_KL_divergence(comb_prob, self.global_label_distribution)
        return res

    def genetic(self, client_ids, num_clients):
        np.random.shuffle(client_ids)
        self.temp_genetic_client_ids = client_ids
        lb = [0 for _ in range(len(client_ids))]
        ub = [1 for _ in range(len(client_ids))]
        precision = [1 for _ in range(len(client_ids))]
        # constraint_ueq = [
        #     lambda x: 1 - sum(x),
        #     lambda x: sum(x) - 3
        #     # lambda x: self.cal_KL_func(x) - threshold
        # ]
        constraint_eq = [
            lambda x: num_clients - sum(x)
        ]
        ga = GA(func=self.cal_genetic_dist_func, n_dim=len(client_ids), size_pop=50, max_iter=200, prob_mut=0.001, 
                lb=lb, ub=ub, constraint_eq=constraint_eq, precision=precision)
        best_x, self.min_KL = ga.run()

        res = []
        for i in range(len(best_x)):
            item = best_x[i]
            if int(item) == 1:
                res.append(client_ids[i])
        return self.min_KL[0], res
    
    # TODO: activate clients
    # def select_clients(self, clients):
    #     if cfg['algo_mode'] != 'dynamicsgd':
    #         for i in range(num_active_clients):
    #             clients[selected_client_ids[i]].active = True
    #     return selected_client_ids, num_active_clients
    
    def select_way(self, client_ids, num_clients):
        temp_client_ids = copy.deepcopy(client_ids)
        if cfg['select_way'] == 'dyna':
            pass
        elif cfg['select_way'] == 'gene':
            return self.genetic(temp_client_ids, num_clients)
        elif cfg['select_way'] == 'rand':
            return np.random.choice(temp_client_ids, size=num_clients, replace=False)
        else:
            raise NotImplementedError
        return 

    def assign_freq_interval(self, client_ids, freq_interval):
        if hasattr(client_ids, '__iter__'):
            for client_id in client_ids:
                self.clients[client_id].freq_interval = freq_interval
        else:
            self.clients[client_ids].freq_interval = freq_interval
        return
    
    def select_clients(self, local_gradient_update, prev_selected_client_ids):
        # if local_gradient_update == 0:
        #     if cfg['only_high_freq'] == True:
        #         selected_client_ids = self.current_high_freq_clients
        #     else:
        #         selected_client_ids = self.current_high_freq_clients + self.current_low_freq_clients
        # elif local_gradient_update > 1:
        if local_gradient_update > 0 and cfg['resample_clients'] == False:
            return prev_selected_client_ids
        
        if local_gradient_update % cfg['high_freq_interval'] == 0:
            selected_client_ids = list(set(selected_client_ids).remove(set(self.previous_high_freq_clients)) + \
                set(selected_client_ids).add(set(self.current_high_freq_clients)))
            
        if cfg['only_high_freq'] == True:
            return selected_client_ids
        
        if local_gradient_update % cfg['low_freq_interval'] == 0:
            selected_client_ids = list(set(selected_client_ids).remove(set(self.previous_low_freq_clients)) + \
                set(selected_client_ids).add(set(self.current_low_freq_clients)))
        return selected_client_ids

    
    def preprocess_for_client_selection(self):
        # dynamic scenario
        if cfg['client_ratio'] == '1-0':
            temp_high_freq_clients = None
            temp_low_freq_clients = None
            for local_gradient_update in range(1, cfg['max_local_gradient_update'] + 1):
                if temp_high_freq_clients and local_gradient_update % cfg['high_freq_interval'] == 0:
                    self.total_client_ids.add(set(temp_high_freq_clients))
                
                if temp_low_freq_clients and local_gradient_update % cfg['low_freq_interval'] == 0:
                    self.total_client_ids.add(set(temp_low_freq_clients))

                if local_gradient_update == 1 or local_gradient_update % cfg['high_freq_interval'] == 0:
                    temp_high_freq_clients = self.select_way(self.total_client_ids, math.ceil(cfg['num_active_clients'] * cfg['high_freq_ratio']))
                    self.high_freq_clients_list.append(copy.deepcopy(temp_high_freq_clients))
                    self.total_client_ids.remove(set(temp_high_freq_clients))
                
                if cfg['only_high_freq'] == True:
                    continue

                if local_gradient_update == 1 or local_gradient_update % cfg['low_freq_interval'] == 0:
                    temp_low_freq_clients = self.select_way(self.total_client_ids, math.ceil(cfg['num_active_clients'] * cfg['low_freq_ratio']))
                    self.low_freq_clients_list.append(copy.deepcopy(temp_low_freq_clients))
                    self.total_client_ids.remove(set(temp_low_freq_clients))
        # fix scenario
        elif cfg['server_ratio'] == '1-0':
            temp_high_freq_clients = None
            temp_low_freq_clients = None

            for local_gradient_update in range(1, cfg['max_local_gradient_update'] + 1):
                if temp_high_freq_clients and local_gradient_update % cfg['high_freq_interval'] == 0:
                    self.total_client_ids.add(set(temp_high_freq_clients))
                
                if temp_low_freq_clients and local_gradient_update % cfg['low_freq_interval'] == 0:
                    self.total_client_ids.add(set(temp_low_freq_clients))

                if local_gradient_update == 1 or local_gradient_update % cfg['high_freq_interval'] == 0:
                    temp_high_freq_clients = self.genetic()
                    self.high_freq_clients_list.append(copy.deepcopy(temp_high_freq_clients))
                    self.total_client_ids.remove(set(temp_high_freq_clients))
                
                if cfg['only_high_freq'] == True:
                    continue

                if local_gradient_update == 1 or local_gradient_update % cfg['low_freq_interval'] == 0:
                    temp_low_freq_clients = self.genetic()
                    self.low_freq_clients_list.append(copy.deepcopy(temp_low_freq_clients))
                    self.total_client_ids.remove(set(temp_low_freq_clients))
