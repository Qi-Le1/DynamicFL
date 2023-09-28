import copy
import math
import time
import torch
import random
import numpy as np
from config import cfg
from sko.GA import GA
from collections import Counter
from scipy.special import rel_entr


class ClientSelector:
    def __init__(self, client_ids, clients, dataset, data_split, logger, communication_info):
        self.client_ids = client_ids
        self.clients = clients
        self.dataset = dataset
        self.data_split = data_split
        self.logger = logger
        self.max_local_gradient_update = cfg['max_local_gradient_update']
        self.clients_label_distribution = {}
        self.global_label_distribution = self.get_global_labels_distribution(dataset)
        self.communication_info = communication_info
        self.high_freq_client_ids = [key for key, _ in communication_info.client_to_freq_interval.items()]
        self.low_freq_client_ids = [key for key, _ in communication_info.client_to_freq_interval.items()]
        self.total_client_ids = set(copy.deepcopy(self.client_ids))
        self.high_freq_clients_list = []
        self.high_freq_clients_KL_list = []
        self.low_freq_clients_list = []
        self.low_freq_clients_KL_list = []
        self.high_freq_clients_list_index = 0
        self.low_freq_clients_list_index = 0
        self.preprocess_for_client_selection()

    @property
    def previous_high_freq_clients(self):
        if self.high_freq_clients_list_index == 0:
            return []
        return self.high_freq_clients_list[self.high_freq_clients_list_index - 1]

    @property
    def previous_low_freq_clients(self):
        if self.low_freq_clients_list_index == 0:
            return []
        return self.low_freq_clients_list[self.low_freq_clients_list_index - 1]

    @property
    def current_high_freq_clients(self):
        res = self.high_freq_clients_list[self.high_freq_clients_list_index]
        return res

    @property
    def current_low_freq_clients(self):
        res = self.low_freq_clients_list[self.low_freq_clients_list_index]
        return res

    def cal_KL_divergence(self, comb_prob, global_labels_distribution):
        return sum(rel_entr(comb_prob, global_labels_distribution))

    def cal_QL(self, list_1, list_2):
        return sum([(item[0] - item[1]) ** 2 for item in zip(list_1, list_2)])

    def cal_prob_distribution(self, dataset, data_split, client_id):
        if client_id in self.clients_label_distribution:
            return self.clients_label_distribution[client_id]
        target_list = np.array([dataset[index]['target'].item() for index in data_split])
        sub_prob = [sum(target_list == i) / len(target_list) for i in range(len(dataset.classes_counts))]
        # if cfg['select_way'] == 'dyna':
        #     # add gaussian noise to sub_prob based on the mean
        #     mean_sub_prob = np.mean(sub_prob) * 0.01
        #     # Define sigma (e.g., as a fraction of the mean)
        #     sigma = mean_sub_prob * 0.1  # Adjust this factor to your needs
        #     # Generate Gaussian noise
        #     noise = np.random.normal(loc=mean_sub_prob, scale=sigma, size=len(sub_prob))
        #     # Add the noise to sub_prob
        #     noisy_sub_prob = sub_prob + noise
        #     # Optionally, clip values to be between 0 and 1 if they fall outside this range
        #     noisy_sub_prob = np.clip(noisy_sub_prob, 0, 1)
        self.clients_label_distribution[client_id] = sub_prob
        return np.array(sub_prob)

    def cal_joint_prob_distribution(self, client_ids):
        total_size = sum(len(self.clients[client_id].data_split['train']) for client_id in client_ids)
        # comb_prob = np.array([0 for _ in range(len(self.dataset.classes_counts))])
        comb_prob = np.zeros(len(self.dataset.classes_counts), dtype=np.float64)
        for client_id in client_ids:
            sub_prob = self.cal_prob_distribution(self.dataset, self.data_split['train'][client_id], client_id)
            ratio = len(self.data_split['train'][client_id]) / total_size
            sub_prob = np.array([prob * ratio for prob in sub_prob])
            comb_prob += sub_prob
        return comb_prob

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

    def genetic(self, client_ids, num_clients):
        temp_genetic_client_ids = list(client_ids)
        np.random.shuffle(temp_genetic_client_ids)
        lb = [0 for _ in range(len(client_ids))]
        ub = [1 for _ in range(len(client_ids))]
        precision = [1 for _ in range(len(client_ids))]
        constraint_eq = [lambda x: num_clients - sum(x)]

        def cal_genetic_dist_func(*args):
            clients_indices_indicator = args[0]
            client_ids = []
            for i in range(len(clients_indices_indicator)):
                if int(clients_indices_indicator[i]) == 1:
                    client_ids.append(temp_genetic_client_ids[i])
            comb_prob = self.cal_joint_prob_distribution(client_ids)
            res = self.cal_KL_divergence(comb_prob, self.global_label_distribution)
            return res

        ga = GA(
            func=cal_genetic_dist_func,
            n_dim=len(client_ids),
            size_pop=50,
            max_iter=400,
            prob_mut=0.001,
            lb=lb,
            ub=ub,
            constraint_eq=constraint_eq,
            precision=precision
        )
        best_x, self.min_KL = ga.run()
        res = [temp_genetic_client_ids[i] for i in range(len(best_x)) if int(best_x[i]) == 1]
        return self.min_KL[0], res

    def dyna(self, client_ids, num_clients):

        def dp(client_ids, num_clients):
            each_item = {
                'distance': float('inf'),
                'dp_client_ids': [],
            }
            dp_res = [[copy.deepcopy(each_item) for _ in range(num_clients+1)] for _ in range(len(client_ids)+1)]
            for i in range(1, len(client_ids)+1):
                client_id = client_ids[i-1]
                for j in range(1, num_clients+1):
                    if i < j:
                        dp_res[i][j] = copy.deepcopy(dp_res[i-1][j])
                    else:
                        dp_res[i][j] = copy.deepcopy(dp_res[i-1][j])
                        temp = copy.deepcopy(dp_res[i-1][j-1])
                        temp['dp_client_ids'].extend([client_id])

                        comb_prob = self.cal_joint_prob_distribution(temp['dp_client_ids'])
                        temp['distance'] = self.cal_KL_divergence(comb_prob, self.global_label_distribution)
                        # add communication cost
                        if cfg['server_ratio'] == '1-0':
                            cur_total_communication_cost = self.communication_info.cal_clients_communication_cost(
                                clients=self.clients, 
                                client_ids=temp['dp_client_ids'],
                            )
                            cur_client_communication_cost_budget = self.clients[client_id].communication_cost_budget

                            # check client's communication cost
                            if cur_client_communication_cost_budget < self.communication_info.client_high_freq_communication_cost:
                                continue
                            # check server's communication cost
                            if self.server_high_freq_communication_cost_budget < cur_total_communication_cost:
                                continue

                        if temp['distance'] < dp_res[i][j]['distance'] and len(temp['dp_client_ids']) == j:
                            dp_res[i][j] = copy.deepcopy(temp)
                      
            min_distance = dp_res[-1][num_clients]['distance']
            min_client_ids = dp_res[-1][num_clients]['dp_client_ids']

            min_distance_for_all = float('inf')
            min_client_ids_for_all = []
            for j in range(len(dp_res[0])):
                if dp_res[-1][j]['distance'] < min_distance_for_all:
                    min_distance_for_all = dp_res[-1][j]['distance']
                    min_client_ids_for_all = dp_res[-1][j]['dp_client_ids']
            return min_distance, min_client_ids, min_distance_for_all, min_client_ids_for_all
        
        permutation_list = client_ids  # help me permute cfg['dp_ensemble_times'] times

        permutation_list = []
        # for _ in range(1):
        for _ in range(cfg['dp_ensemble_times']):
            # Use np.random.choice to generate a permutation of client_ids
            permutation = np.random.choice(list(client_ids), size=len(client_ids), replace=False)
            # Append this permutation to the permutation_list
            permutation_list.append(list(permutation))

        minimum_distance = float('inf')
        minimum_dist_selected_client_ids = None
        for client_list in permutation_list:
            min_dist, selected_client_ids, _, _ = dp(client_list, num_clients)
            if min_dist < minimum_distance:
                minimum_distance = min_dist
                minimum_dist_selected_client_ids = selected_client_ids
        return minimum_distance, minimum_dist_selected_client_ids

    def select_way(self, client_ids, num_clients):
        temp_client_ids = copy.deepcopy(client_ids)
        selected_client_ids = None
        if cfg['select_way'] == 'dyna':
            min_KL, selected_client_ids = self.dyna(temp_client_ids, num_clients)
        elif cfg['select_way'] == 'gene':
            min_KL, selected_client_ids = self.genetic(temp_client_ids, num_clients)
        elif cfg['select_way'] == 'rand':
            selected_client_ids = np.random.choice(list(temp_client_ids), size=num_clients, replace=False)

            comb_prob = self.cal_joint_prob_distribution(selected_client_ids)
            min_KL = self.cal_KL_divergence(comb_prob, self.global_label_distribution)
        else:
            raise NotImplementedError
        return min_KL, selected_client_ids

    def assign_freq_interval(self, client_ids, freq_interval, local_gradient_update, max_local_gradient_update):
        if local_gradient_update + freq_interval > max_local_gradient_update:
            freq_interval = max_local_gradient_update - local_gradient_update + 1
        if hasattr(client_ids, '__iter__'):
            for client_id in client_ids:
                self.clients[client_id].freq_interval = freq_interval
        else:
            self.clients[client_ids].freq_interval = freq_interval
        return

    def select_clients(self, local_gradient_update, prev_selected_client_ids, clients):
        new_selected_client_ids = []

        if local_gradient_update > 0 and not cfg['resample_clients']:
            return prev_selected_client_ids, new_selected_client_ids

        selected_client_ids = []

        if local_gradient_update % cfg['high_freq_interval'] == 0:
            selected_client_ids.extend(
                list((set(prev_selected_client_ids) - set(self.previous_high_freq_clients)) | set(self.current_high_freq_clients))
            )
            if cfg['client_ratio'] == '1-0':
                for client in self.current_high_freq_clients:
                    self.clients[client].freq_interval = cfg['high_freq_interval']
            new_selected_client_ids.extend(self.current_high_freq_clients)
            self.high_freq_clients_list_index += 1

        if cfg['only_high_freq']:
            return selected_client_ids, new_selected_client_ids

        if local_gradient_update % cfg['low_freq_interval'] == 0:
            selected_client_ids.extend(
                list((set(prev_selected_client_ids) - set(self.previous_low_freq_clients)) | set(self.current_low_freq_clients))
            )
            if cfg['client_ratio'] == '1-0':
                for client in self.current_low_freq_clients:
                    self.clients[client].freq_interval = cfg['low_freq_interval']
            new_selected_client_ids.extend(self.current_low_freq_clients)
            self.low_freq_clients_list_index += 1

        return selected_client_ids, new_selected_client_ids


    def compute_selection_probability(self, freq_clients_list):
        freq_flatten = [client for sublist in freq_clients_list for client in sublist]

        # Count the occurrences of each client ID
        freq_counts = Counter(freq_flatten)

        # Get the total number of selections
        total_high_freq_selections = len(freq_flatten)

        # Calculate selection probability for each client
        freq_probabilities = {client: 0 for client in range(cfg['num_clients'])}

        # Update selection probability only for selected clients
        for client, count in freq_counts.items():
            freq_probabilities[client] = count / total_high_freq_selections

        return freq_probabilities
    
    def preprocess_for_client_selection(self):
        # dynamic scenario
        start_time = time.time()

        if cfg['client_ratio'] == '1-0':
            temp_high_freq_clients = None
            temp_low_freq_clients = None

            for local_gradient_update in range(cfg['max_local_gradient_update']):
                if temp_high_freq_clients is not None and local_gradient_update % cfg['high_freq_interval'] == 0:
                    self.total_client_ids |= set(temp_high_freq_clients)

                if temp_low_freq_clients is not None and local_gradient_update % cfg['low_freq_interval'] == 0:
                    self.total_client_ids |= set(temp_low_freq_clients)

                if local_gradient_update % cfg['high_freq_interval'] == 0:
                    temp_KL, temp_high_freq_clients = self.select_way(
                        self.total_client_ids, math.ceil(cfg['num_active_clients'] * cfg['high_freq_ratio'])
                    )
                    self.high_freq_clients_list.append(copy.deepcopy(temp_high_freq_clients))
                    self.high_freq_clients_KL_list.append(temp_KL)
                    self.total_client_ids -= set(temp_high_freq_clients)

                if cfg['only_high_freq']:
                    continue

                if local_gradient_update % cfg['low_freq_interval'] == 0:
                    temp_KL, temp_low_freq_clients = self.select_way(
                        self.total_client_ids, math.ceil(cfg['num_active_clients'] * cfg['low_freq_ratio'])
                    )
                    self.low_freq_clients_list.append(copy.deepcopy(temp_low_freq_clients))
                    self.low_freq_clients_KL_list.append(temp_KL)
                    self.total_client_ids -= set(temp_low_freq_clients)

                if not cfg['resample_clients']:
                    break
            
            
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

                if cfg['only_high_freq']:
                    continue

                if local_gradient_update == 1 or local_gradient_update % cfg['low_freq_interval'] == 0:
                    temp_low_freq_clients = self.genetic()
                    self.low_freq_clients_list.append(copy.deepcopy(temp_low_freq_clients))
                    self.total_client_ids.remove(set(temp_low_freq_clients))

        end_time = time.time()
        print('select way time: ', end_time - start_time, flush=True)
        high_freq_clients_distribution = self.compute_selection_probability(self.high_freq_clients_list)
        # print('high_freq_clients_list: ', high_freq_clients_distribution, flush=True)
        # print('high_freq_clients_KL_list: ', self.high_freq_clients_KL_list, flush=True)

        # Calculating mean and standard deviation
        mean_value = np.mean(self.high_freq_clients_KL_list)
        std_value = np.std(self.high_freq_clients_KL_list)

        print(f'Mean: {mean_value}, Standard Deviation: {std_value}')

        low_freq_clients_distribution = self.compute_selection_probability(self.low_freq_clients_list)
        # print('low_freq_clients_list: ', low_freq_clients_distribution, flush=True)
        # print('low_freq_clients_KL_list: ', self.low_freq_clients_KL_list, flush=True)

        # print('clients_label_distribution', self.clients_label_distribution, flush=True)
        self.logger.append(
            {
                f'high_freq_clients_distribution': high_freq_clients_distribution,
                f'low_freq_clients_distribution': low_freq_clients_distribution,
            },
            'train',
            len(low_freq_clients_distribution)
            )
        return

