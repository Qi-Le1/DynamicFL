import os
from config import cfg
# from datasets import ML100K, ML1M, ML10M, ML20M
# from datasets.datasets_utils import download_url, extract_file
# from utils import makedir_exist_ok, check_exists
# 
# file_list = [ML100K.file, ML1M.file, ML10M.file, ML20M.file]

# dataset_list = ['FMNIST', 'CIFAR10', 'CIFAR100']
dataset_list = ['CIFAR10']
from utils.api import process_command
from data import fetch_dataset

def main():
    process_command()
    for i in range(len(dataset_list)):
        data_name = dataset_list[i]

        fetch_dataset(data_name)
        

if __name__ == "__main__":
    main()