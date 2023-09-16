import argparse
import datetime
import models
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from config import cfg, process_args
# from data import fetch_dataset, make_data_loader, separate_dataset_su, make_batchnorm_stats, make_batchnorm_dataset_su
# from metrics import Metric
# from utils import save, to_device, process_control, process_dataset, create_optimizer, create_scheduler, resume, collate
# from logger import Logger

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# cudnn.benchmark = True
# parser = argparse.ArgumentParser(description='cfg')
# for k in cfg:
#     exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
# parser.add_argument('--control_name', default=None, type=str)
# args = vars(parser.parse_args())
# process_args(args)


# if __name__ == "__main__":

class ClientSampler(torch.utils.data.Sampler):
    def __init__(self, 
    ):
        self.idx = [i for i in range(100)]
        self.start = 0
        self.end = len(self.idx)

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


def iterate(clientsampler):
    for i in range(10):
        print(next(clientsampler))

a = ClientSampler()
iterate(a)
iterate(a)
iterate(a)
iterate(a)

from torch.utils.data import DataLoader, Dataset, Sampler

class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data = [i for i in range(100, 200)]
dataset = SimpleDataset(data)

dataloader = DataLoader(dataset, batch_size=1, sampler=ClientSampler())

for i, input in enumerate(dataloader):
    print(i, input, '\n')
    if i == 10:
        break

for i, input in enumerate(dataloader):
    print(i, input, '\n')
    if i == 10:
        break