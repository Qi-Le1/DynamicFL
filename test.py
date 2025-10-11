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

# class ClientSampler(torch.utils.data.Sampler):
#     def __init__(self, 
#     ):
#         self.idx = [i for i in range(100)]
#         self.start = 0
#         self.end = len(self.idx)

#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         if self.start < self.end:
#             res = self.start
#             self.start += 1
#             return self.idx[res]
#         else:
#             self.start = 0
#             raise StopIteration

#     def __len__(self):
#         return len(self.idx)


# def iterate(clientsampler):
#     for i in range(10):
#         print(next(clientsampler))

# a = ClientSampler()
# iterate(a)
# iterate(a)
# iterate(a)
# iterate(a)

# from torch.utils.data import DataLoader, Dataset, Sampler

# class SimpleDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# data = [i for i in range(100, 200)]
# dataset = SimpleDataset(data)

# dataloader = DataLoader(dataset, batch_size=1, sampler=ClientSampler())

# for i, input in enumerate(dataloader):
#     print(i, input, '\n')
#     if i == 10:
#         break

# for i, input in enumerate(dataloader):
#     print(i, input, '\n')
#     if i == 10:
#         break




# import torch
# from torch.utils.data import Dataset, DataLoader

# # Define a custom Dataset class
# class SyntheticDataset(Dataset):
#     def __init__(self, num_samples, input_dim):
#         self.num_samples = num_samples
#         self.input_dim = input_dim
#         self.data = torch.randn(num_samples, input_dim)
#         self.labels = torch.randint(0, 2, (num_samples,))

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         label = self.labels[idx]
#         return sample, label, idx

# # Create an instance of the SyntheticDataset with 100 samples of dimension 10
# synthetic_dataset = SyntheticDataset(num_samples=8, input_dim=1)

# # Create a DataLoader to handle batching of data
# data_loader = DataLoader(synthetic_dataset, batch_size=5, shuffle=False)

# # Iterate over the data loader in a for-loop
# a = iter(data_loader)
# b = next(a)
# print(b)
# c = next(a)
# print(c)

# a = iter(data_loader)
# d = next(a)
# print(d)

# import numpy as np

# # Given parameters
# mu = 0.5
# sigma = 0.05

# # Generate 10 random samples from the normal distribution
# random_samples = np.random.normal(loc=mu, scale=sigma, size=10)
# print(random_samples)


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define a simple custom dataset
class MyDataset(Dataset):
    def __init__(self, labels):
        # self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.labels[idx]

# Generate some synthetic data
np.random.seed(42)  # Optional: seeding for reproducibility
# data = np.random.randn(100, 1)  # 100 samples, each with 3 features
labels = np.random.randint(0, 2, size=10)  # Binary labels

# Convert data and labels to PyTorch tensors
# data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Create an instance of MyDataset
my_dataset = MyDataset(labels_tensor)

# Create a DataLoader
dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True)

class DataLoaderWrapper:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iterator = iter(data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
            return batch
        except StopIteration:
            # Reset the DataLoader iterator
            self.iterator = iter(self.data_loader)
            batch = next(self.iterator)
            return batch

a = DataLoaderWrapper(dataloader)
# Loop through the DataLoader (one epoch)
# for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
#     print(f"Batch {batch_idx} | Data: {batch_data.size()} | Labels: {batch_labels.size()}")

b = iter(a)
for i in range(20):
    print(next(b))
    if i == 9:
        print('zzzz')
# print('-----\n')
# for i in range(10):
#     print(next(b))
