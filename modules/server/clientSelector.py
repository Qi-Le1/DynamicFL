import copy
from config import cfg

class ClientSelector():
    def __init__(
        self,
    ):
        self.max_local_gradient_update = cfg['max_local_gradient_update']
        self.total_clients_pool = set([i for i in range(cfg['num_clients'])])
        self.high_freq_clients_list = []
        self.low_freq_clients_list = []
        self.select_clients()
        return
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start < self.end:
            res = self.start
            self.start += 1
            return self.idx[res]
        else:
            self.start = 0
            raise StopIteration

    def __len__(self):
        return len(self.idx)
    
    def select_clients(self):
        if 'dynamic':
            temp_high_freq_clients = None
            temp_low_freq_clients = None

            for local_gradient_update in range(1, cfg['max_local_gradient_update'] + 1):
                if temp_high_freq_clients and local_gradient_update % cfg['high_freq_interval'] == 0:
                    self.total_clients_pool.add(set(temp_high_freq_clients))
                
                if temp_low_freq_clients and local_gradient_update % cfg['low_freq_interval'] == 0:
                    self.total_clients_pool.add(set(temp_low_freq_clients))

                if local_gradient_update == 1 or local_gradient_update % cfg['high_freq_interval'] == 0:
                    temp_high_freq_clients = self.genetic()
                    self.high_freq_clients_list.append(copy.deepcopy(temp_high_freq_clients))
                    self.total_clients_pool.remove(set(temp_high_freq_clients))
                
                if local_gradient_update == 1 or local_gradient_update % cfg['low_freq_interval'] == 0:
                    temp_low_freq_clients = self.genetic()
                    self.low_freq_clients_list.append(copy.deepcopy(temp_low_freq_clients))
                    self.total_clients_pool.remove(set(temp_low_freq_clients))
                    
        elif 'fix':   
