import os
import itertools
import json
import copy

# env_dist = os.environ
# for env in env_dist:
#     print(env, env_dist[env])

# import sys
# print(sys.executable)
# print(sys.version)
# print('---')
# print(sys)

# print(sys.path)
# CUR_FILE_PATH = os.path.abspath(__file__)
# UPPER_LEVEL_PATH = os.path.dirname(CUR_FILE_PATH)
# UPPER_UPPER_LEVEL_PATH = os.path.dirname(UPPER_LEVEL_PATH)
# # TOP_LEVEL_PATH = os.path.dirname(UPPER_UPPER_LEVEL_PATH)
# print(f'Colda Test Upper Level Path Init: {UPPER_LEVEL_PATH}')
# print(f'Colda Test Upper Upper Level Path Init: {UPPER_UPPER_LEVEL_PATH}')
# # sys.path.append(UPPER_LEVEL_PATH)
# sys.path.append(UPPER_UPPER_LEVEL_PATH)
import numpy as np
import pandas as pd
from utils.api import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse




os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(description='analyze_data')
parser.add_argument('--type', default='dp', type=str)
args = vars(parser.parse_args())

save_format = 'png'
result_path = './output/result'
vis_path = './output/vis/{}'.format(save_format)

num_experiments = 1
exp = [str(x) for x in list(range(num_experiments))]


def make_controls(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    # controls = [exp] + data_names + model_names + [control_names]
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(file):
    controls = []
    if file == 'dp' or file == 'new_dp':
        control_name = [[['CIFAR10', 'CIFAR100'], ['cnn'], ['0.1', '0.3'], ['100'], ['non-iid-l-1', 'non-iid-l-2','non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5']]]
        CIFAR10_controls_3 = make_controls(control_name)
        controls.extend(CIFAR10_controls_3)

        control_name = [[['FEMNIST'], ['cnn'], ['0.1', '0.3'], ['100'], ['non-iid-d-0.01','non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5']]]
        CIFAR10_controls_4 = make_controls(control_name)
        controls.extend(CIFAR10_controls_4)
    elif file == 'cnn':

        # control_name = [[['CIFAR10', 'CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                 ['scaffold', 'fedensemble', 'feddyn', 'fedgen'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
        # CIFAR10_controls_3 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_3)
        
        # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                 ['scaffold', 'fedensemble', 'feddyn', 'fedgen'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
        # CIFAR10_controls_4 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_4)
        # control_name = [[['CIFAR10', 'CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                 ['fednova'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
        # CIFAR10_controls_3 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_3)
        
        # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                 ['fednova'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
        # CIFAR10_controls_4 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_4)
        # control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                 ['fedprox', 'fedavg', 'fedensemble', 'scaffold'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
        # CIFAR10_controls_3 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_3)

        # control_name = [[['CIFAR10', 'CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                 ['scaffold'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
        # CIFAR10_controls_3 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_3)

        # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                 ['fedavg', 'fedprox'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
        # CIFAR10_controls_3 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_3)

        # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                      ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
        #                      ['5-1', '4-1', '6-1']]]
            
        # CIFAR10_controls_1 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_1)

        # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                     ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
        #                     ['5-1', '4-1', '6-1']]]
        # CIFAR10_controls_2 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_2)
        # control_name = [[['CIFAR10'], ['cnn'], ['0.1', '0.3', '0.5'], ['100'], ['non-iid-l-1', 'non-iid-l-2','non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                      ['dynamicfl'], ['250'], ['10'], ['nonpre']]]
        # CIFAR10_controls_3 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_3)

        # check baseline
        control_name = [[['CIFAR10', 'CIFAR100'], ['cnn', 'resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                        ['fedprox', 'fedavg', 'fedensemble', 'fednova', 'fedgen', 'scaffold', 'dynamicsgd'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
        CIFAR10_controls_3 = make_controls(control_name)
        controls.extend(CIFAR10_controls_3)

        # check baseline
        control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                        ['fedprox', 'fedavg', 'fedensemble', 'fednova', 'fedgen', 'scaffold', 'dynamicsgd'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
        CIFAR10_controls_4 = make_controls(control_name)
        controls.extend(CIFAR10_controls_4)
    elif file == 'resnet18':
        control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2','non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['nonpre'], ['0.1-0.9', '0.2-0.8', '0.3-0.7', '0.4-0.6','0.5-0.5', '0.6-0.4', '0.7-0.3', '0.8-0.2', '0.9-0.1'],
                             ['7-1', '6-1', '5-1'], ['WA'], ['1']]]
        CIFAR10_controls_3 = make_controls(control_name)
        controls.extend(CIFAR10_controls_3)
    
    elif file == 'cnn_maoge':

        control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                             ['5-1', '4-1', '6-1']]]
            
        CIFAR10_controls_1 = make_controls(control_name)
        controls.extend(CIFAR10_controls_1)

        control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                            ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                            ['5-1', '4-1', '6-1']]]
        CIFAR10_controls_2 = make_controls(control_name)
        controls.extend(CIFAR10_controls_2)
    elif file == 'high_freq':
        control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                             ['5-1', '4-1', '6-1'], ['1']]]
            
        CIFAR10_controls_1 = make_controls(control_name)
        controls.extend(CIFAR10_controls_1)

        control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                            ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                            ['5-1', '4-1', '6-1'], ['1']]]
        CIFAR10_controls_2 = make_controls(control_name)
        controls.extend(CIFAR10_controls_2)
    elif file == 'freq_ablation':
        control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                        ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                        ['6-1', '5-1', '4-1', '3-1'], ['1']]]
        CIFAR10_controls_7 = make_controls(control_name)
        controls.extend(CIFAR10_controls_7)

        control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                        ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                        ['6-5', '6-4', '6-3', '6-2.5', '6-2', '6-1', '5-4', '5-3', '5-2.5', '5-2', '5-1', '4-3', '4-2.5', '4-2', '4-1', '3-2.5', '3-2', '3-1']]]
        CIFAR10_controls_8 = make_controls(control_name)
        controls.extend(CIFAR10_controls_8)
    elif file == 'cnn_all':

        # control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                      ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
        #                      ['5-1', '4-1', '6-1']]]
            
        # CIFAR10_controls_1 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_1)

        # control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                     ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
        #                     ['5-1', '4-1', '6-1']]]
        # CIFAR10_controls_2 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_2)

        # control_name = [[['CIFAR10', 'CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                 ['fedprox', 'fedavg', 'fedensemble', 'scaffold', 'fedgen', 'scaffold', 'dynamicsgd'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
        # CIFAR10_controls_3 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_3)

        # control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                 ['fedprox', 'fedavg', 'fedensemble', 'scaffold', 'fedgen', 'scaffold', 'dynamicsgd'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
        # CIFAR10_controls_3 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_3)
        pass
    elif file == 'communication_cost':
        # check_communicatoin_cost
        control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                        ['dynamicfl'], ['5'],['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                        ['6-1', '5-1', '4-1'], ['0'], ['1']]]
        CIFAR10_controls_5 = make_controls(control_name)
        controls.extend(CIFAR10_controls_5)

        # # check_communicatoin_cost
        control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                        ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                        ['6-1', '5-1', '4-1'], ['0'], ['1']]]
        CIFAR10_controls_6 = make_controls(control_name)
        controls.extend(CIFAR10_controls_6)

        # check_communicatoin_cost
        control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                        ['dynamicfl'], ['5'],['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                        ['6-1', '5-1', '4-1'], ['0'], ['1']]]
        CIFAR10_controls_7 = make_controls(control_name)
        controls.extend(CIFAR10_controls_7)

        # # check_communicatoin_cost
        control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                        ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                        ['6-1', '5-1', '4-1'], ['0'], ['1']]]
        CIFAR10_controls_8 = make_controls(control_name)
        controls.extend(CIFAR10_controls_8)

        # check_communicatoin_cost
        control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                        ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'],  
                        ['6-1', '5-1', '4-1'], ['0'], ['1']]]
        CIFAR10_controls_9 = make_controls(control_name)
        controls.extend(CIFAR10_controls_9)

        # check_communicatoin_cost
        control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-d-0.01', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                        ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                        ['6-1', '5-1', '4-1'], ['0'], ['1']]]
        CIFAR10_controls_10 = make_controls(control_name)
        controls.extend(CIFAR10_controls_10)

        # check_communicatoin_cost
        control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2',  'non-iid-d-0.1', 'non-iid-d-0.3'], 
                        ['dynamicfl'], ['5'],['1-0'], ['0.3-0.7'], 
                        ['6-5', '6-4', '6-3', '6-2.5', '6-2', '6-1', '5-4', '5-3', '5-2.5', '5-2', '5-1', '4-3', '4-2.5', '4-2', '4-1', '3-2.5', '3-2', '3-1'], ['0'], ['1']]]
        CIFAR10_controls_11 = make_controls(control_name)
        controls.extend(CIFAR10_controls_11)
        # --- test demo
        # control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1'], 
        #                      ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4'], ['1-0'], 
        #                      ['5-1', '4-1']]]
            
        # CIFAR10_controls_1 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_1)

        # control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                     ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
        #                     ['5-1', '4-1', '6-1']]]
        # CIFAR10_controls_2 = make_controls(control_name)
        # controls.extend(CIFAR10_controls_2)

    elif file == 'resnet18_maoge':

        control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                    ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                    ['5-1', '4-1', '6-1']]]
            
        CIFAR10_controls_1 = make_controls(control_name)
        controls.extend(CIFAR10_controls_1)

        control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                    ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                    ['5-1', '4-1', '6-1']]]
        CIFAR10_controls_2 = make_controls(control_name)
        controls.extend(CIFAR10_controls_2)
    
    elif file == 'resnet18_maoge_chongfu':

        control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                    ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                    ['5-1', '4-1', '6-1']]]
            
        CIFAR10_controls_1 = make_controls(control_name)
        controls.extend(CIFAR10_controls_1)

        control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                    ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                    ['5-1', '4-1', '6-1']]]
        CIFAR10_controls_2 = make_controls(control_name)
        controls.extend(CIFAR10_controls_2)
    
    elif file == 'resnet18_all':

        control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                             ['5-1', '4-1', '6-1']]]
            
        CIFAR10_controls_1 = make_controls(control_name)
        controls.extend(CIFAR10_controls_1)

        control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                            ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                            ['5-1', '4-1', '6-1']]]
        CIFAR10_controls_2 = make_controls(control_name)
        controls.extend(CIFAR10_controls_2)

        control_name = [[['CIFAR10', 'CIFAR100', 'FEMNIST'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                        ['fedprox', 'fedavg', 'fedensemble', 'scaffold'], ['5'], ['1-0'], ['1-0'], ['6-1']]]
        CIFAR10_controls_3 = make_controls(control_name)
        controls.extend(CIFAR10_controls_3)
        a = 5
    # if file == 'fs':
    #     control_name = [[['fs']]]
    #     data_names = [['CIFAR10']]
    #     model_names = [['wresnet28x2']]
    #     cifar10_controls = make_controls(data_names, model_names, control_name)
    #     data_names = [['SVHN']]
    #     model_names = [['wresnet28x2']]
    #     svhn_controls = make_controls(data_names, model_names, control_name)
    #     data_names = [['CIFAR100']]
    #     model_names = [['wresnet28x8']]
    #     cifar100_controls = make_controls(data_names, model_names, control_name)
    #     controls = cifar10_controls + svhn_controls + cifar100_controls
    # elif file == 'ps':
    #     control_name = [[['250', '4000']]]
    #     data_names = [['CIFAR10']]
    #     model_names = [['wresnet28x2']]
    #     cifar10_controls = make_controls(data_names, model_names, control_name)
    #     control_name = [[['250', '1000']]]
    #     data_names = [['SVHN']]
    #     model_names = [['wresnet28x2']]
    #     svhn_controls = make_controls(data_names, model_names, control_name)
    #     control_name = [[['2500', '10000']]]
    #     data_names = [['CIFAR100']]
    #     model_names = [['wresnet28x8']]
    #     cifar100_controls = make_controls(data_names, model_names, control_name)
    #     controls = cifar10_controls + svhn_controls + cifar100_controls
    # elif file == 'cd':
    #     control_name = [[['250', '4000'], ['fix-mix'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0.5'], ['1']]]
    #     data_names = [['CIFAR10']]
    #     model_names = [['wresnet28x2']]
    #     cifar10_controls = make_controls(data_names, model_names, control_name)
    #     control_name = [[['250', '1000'], ['fix-mix'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0.5'], ['1']]]
    #     data_names = [['SVHN']]
    #     model_names = [['wresnet28x2']]
    #     svhn_controls = make_controls(data_names, model_names, control_name)
    #     control_name = [[['2500', '10000'], ['fix-mix'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0.5'],
    #                      ['1']]]
    #     data_names = [['CIFAR100']]
    #     model_names = [['wresnet28x8']]
    #     cifar100_controls = make_controls(data_names, model_names, control_name)
    #     controls = cifar10_controls + svhn_controls + cifar100_controls
    # elif file == 'ub':
    #     control_name = [
    #         [['250', '4000'], ['fix-mix'], ['100'], ['0.1'], ['non-iid-d-0.1', 'non-iid-d-0.3'], ['5'], ['0.5'], ['1']]]
    #     data_names = [['CIFAR10']]
    #     model_names = [['wresnet28x2']]
    #     cifar10_controls = make_controls(data_names, model_names, control_name)
    #     control_name = [
    #         [['250', '1000'], ['fix-mix'], ['100'], ['0.1'], ['non-iid-d-0.1', 'non-iid-d-0.3'], ['5'], ['0.5'], ['1']]]
    #     data_names = [['SVHN']]
    #     model_names = [['wresnet28x2']]
    #     svhn_controls = make_controls(data_names, model_names, control_name)
    #     control_name = [[['2500', '10000'], ['fix-mix'], ['100'], ['0.1'], ['non-iid-d-0.1', 'non-iid-d-0.3'], ['5'],
    #                      ['0.5'], ['1']]]
    #     data_names = [['CIFAR100']]
    #     model_names = [['wresnet28x8']]
    #     cifar100_controls = make_controls(data_names, model_names, control_name)
    #     controls = cifar10_controls + svhn_controls + cifar100_controls
    # elif file == 'loss':
    #     control_name = [[['4000'], ['fix'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0.5'], ['1']]]
    #     data_names = [['CIFAR10']]
    #     model_names = [['wresnet28x2']]
    #     cifar10_controls = make_controls(data_names, model_names, control_name)
    #     controls = cifar10_controls
    # elif file == 'local-epoch':
    #     control_name = [[['4000'], ['fix-mix'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['1'], ['0.5'], ['1']]]
    #     data_names = [['CIFAR10']]
    #     model_names = [['wresnet28x2']]
    #     cifar10_controls = make_controls(data_names, model_names, control_name)
    #     controls = cifar10_controls
    # elif file == 'gm':
    #     control_name = [[['4000'], ['fix-mix'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0'], ['1']]]
    #     data_names = [['CIFAR10']]
    #     model_names = [['wresnet28x2']]
    #     cifar10_controls = make_controls(data_names, model_names, control_name)
    #     controls = cifar10_controls
    # elif file == 'sbn':
    #     control_name = [[['250', '4000'], ['fix-mix'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0.5'], ['0']]]
    #     data_names = [['CIFAR10']]
    #     model_names = [['wresnet28x2']]
    #     cifar10_controls = make_controls(data_names, model_names, control_name)
    #     controls = cifar10_controls
    # elif file == 'alternate':
    #     control_name = [[['4000'], ['fix-batch'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0.5'],
    #                      ['1']]]
    #     data_names = [['CIFAR10']]
    #     model_names = [['wresnet28x2']]
    #     cifar10_controls_1 = make_controls(data_names, model_names, control_name)
    #     control_name = [[['4000'], ['fix', 'fix-batch'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0.5'],
    #                      ['1'], ['0']]]
    #     data_names = [['CIFAR10']]
    #     model_names = [['wresnet28x2']]
    #     cifar10_controls_2 = make_controls(data_names, model_names, control_name)
    #     controls = cifar10_controls_1 + cifar10_controls_2
    # elif file == 'fl':
    #     control_name = [
    #         [['fs'], ['sup'], ['100'], ['0.1'], ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'], ['5'],
    #          ['0.5'], ['1']]]
    #     data_names = [['CIFAR10']]
    #     model_names = [['wresnet28x2']]
    #     cifar10_controls = make_controls(data_names, model_names, control_name)
    #     control_name = [
    #         [['fs'], ['sup'], ['100'], ['0.1'], ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'], ['5'],
    #          ['0.5'], ['1']]]
    #     data_names = [['SVHN']]
    #     model_names = [['wresnet28x2']]
    #     svhn_controls = make_controls(data_names, model_names, control_name)
    #     control_name = [
    #         [['fs'], ['sup'], ['100'], ['0.1'], ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'], ['5'],
    #          ['0.5'], ['1']]]
    #     data_names = [['CIFAR100']]
    #     model_names = [['wresnet28x8']]
    #     cifar100_controls = make_controls(data_names, model_names, control_name)
    #     controls = cifar10_controls + svhn_controls + cifar100_controls
    # elif file == 'fsgd':
    #     control_name = [[['4000'], ['fix-fsgd'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['0'], ['0'], ['1']]]
    #     data_names = [['CIFAR10']]
    #     model_names = [['wresnet28x2']]
    #     cifar10_controls = make_controls(data_names, model_names, control_name)
    #     controls = cifar10_controls
    # elif file == 'frgd':
    #     control_name = [
    #         [['250', '4000'], ['fix-frgd'], ['100'], ['0.1'], ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'],
    #          ['5'], ['0.5'], ['1'], ['0']]]
    #     data_names = [['CIFAR10']]
    #     model_names = [['wresnet28x2']]
    #     cifar10_controls = make_controls(data_names, model_names, control_name)
    #     control_name = [
    #         [['250', '1000'], ['fix-frgd'], ['100'], ['0.1'], ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'],
    #          ['5'], ['0.5'], ['1'], ['0']]]
    #     data_names = [['SVHN']]
    #     model_names = [['wresnet28x2']]
    #     svhn_controls = make_controls(data_names, model_names, control_name)
    #     control_name = [[['2500', '10000'], ['fix-frgd'], ['100'], ['0.1'],
    #                      ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'], ['5'], ['0.5'], ['1'], ['0']]]
    #     data_names = [['CIFAR100']]
    #     model_names = [['wresnet28x8']]
    #     cifar100_controls = make_controls(data_names, model_names, control_name)
    #     controls = cifar10_controls + svhn_controls + cifar100_controls
    # elif file == 'fmatch':
    #     control_name = [[['250', '4000'], ['fix-fmatch'], ['100'], ['0.1'],
    #                      ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'], ['5'], ['0.5'], ['1'], ['0']]]
    #     data_names = [['CIFAR10']]
    #     model_names = [['wresnet28x2']]
    #     cifar10_controls = make_controls(data_names, model_names, control_name)
    #     control_name = [[['250', '1000'], ['fix-fmatch'], ['100'], ['0.1'],
    #                      ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'], ['5'], ['0.5'], ['1'], ['0']]]
    #     data_names = [['SVHN']]
    #     model_names = [['wresnet28x2']]
    #     svhn_controls = make_controls(data_names, model_names, control_name)
    #     control_name = [[['2500', '10000'], ['fix-fmatch'], ['100'], ['0.1'],
    #                      ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'], ['5'], ['0.5'], ['1'], ['0']]]
    #     data_names = [['CIFAR100']]
    #     model_names = [['wresnet28x8']]
    #     cifar100_controls = make_controls(data_names, model_names, control_name)
    #     controls = cifar10_controls + svhn_controls + cifar100_controls
    # else:
        # raise ValueError('Not valid file')
    return controls


def main():
    # files = ['fs', 'ps', 'cd', 'ub', 'loss', 'local-epoch', 'gm', 'sbn', 'alternate', 'fl', 'fsgd', 'frgd', 'fmatch']
    global result_path, vis_path, num_experiments, exp

    print(f"type: {args['type']}")    
    result_path = './output/result/{}'.format(args['type'])
    vis_path = './output/vis/{}'.format(args['type'])
    files = [args['type']]

    if args['type'] == 'dp' or args['type'] == 'high_freq' or args['type'] == 'new_dp':
        num_experiments = 1
    elif args['type'] == 'cnn' or args['type'] == 'resnet18' or args['type'] == 'cnn_all':
        num_experiments = 1
    elif args['type'] == 'cnn_maoge' or args['type'] == 'resnet18_maoge' or args['type'] == 'resnet18_all' or args['type'] == 'resnet18_maoge_chongfu':
        num_experiments = 1
    elif args['type'] == 'freq_ablation':
        num_experiments = 1
    elif args['type'] == 'communication_cost':
        num_experiments = 1
    else:
        raise ValueError('Not valid type')
    exp = [str(x) for x in list(range(num_experiments))]

    controls = []
    for file in files:
        controls += make_control_list(file)
    processed_result_exp, processed_result_history = process_result(controls)
    with open('{}/processed_result_exp.json'.format(result_path), 'w') as fp:
        json.dump(processed_result_exp, fp, indent=2)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    extracted_processed_result_exp = {}
    extracted_processed_result_history = {}
    # if processed_result_exp:
    extract_processed_result(extracted_processed_result_exp, processed_result_exp, [])
    # if processed_result_history:
    extract_processed_result(extracted_processed_result_history, processed_result_history, [])
    # print(f'extracted_processed_result_history: {extracted_processed_result_history}')
    if extracted_processed_result_exp:
        df_exp = make_df_exp(extracted_processed_result_exp)
    if extracted_processed_result_history:
        df_history = make_df_history(extracted_processed_result_history)
    df_exp = {}
    make_vis(df_exp, df_history)
    return


def process_result(controls):
    processed_result_exp, processed_result_history = {}, {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result_exp, processed_result_history)
    if processed_result_exp:
        summarize_result(processed_result_exp)
    if processed_result_history:
        summarize_result(processed_result_history)
    return processed_result_exp, processed_result_history


def extract_result(control, model_tag, processed_result_exp, processed_result_history):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            if 'test' not in base_result['logger']:
                for k in base_result['logger']['train'].history:
                    metric_name = k.split('/')[1]
                    if metric_name not in processed_result_history:
                        processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                    # processed_result_exp[metric_name]['exp'][exp_idx] = base_result['logger']['test'].mean[k]
                    processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['train'].history[k]
            else:
                for k in base_result['logger']['test'].mean:
                    metric_name = k.split('/')[1]
                    if metric_name not in processed_result_exp:
                        processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                        processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                    processed_result_exp[metric_name]['exp'][exp_idx] = base_result['logger']['test'].mean[k]
                    processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['train'].history[k]
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result_exp:
            processed_result_exp[control[1]] = {}
            processed_result_history[control[1]] = {}
        extract_result([control[0]] + control[2:], model_tag, processed_result_exp[control[1]],
                       processed_result_history[control[1]])
    return

# for standard error
def cal_se(std, sample_nums):
    return std / np.sqrt(sample_nums)

def summarize_result(processed_result):
    # print(f'processed_result: {processed_result}')
    if 'exp' in processed_result:
        pivot = 'exp'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0).item()
        processed_result['se'] = cal_se(np.std(processed_result[pivot], axis=0).item(), len(processed_result[pivot]))
        processed_result['max'] = np.max(processed_result[pivot], axis=0).item()
        processed_result['min'] = np.min(processed_result[pivot], axis=0).item()
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0).item()
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0).item()
        processed_result[pivot] = processed_result[pivot].tolist()
    elif 'history' in processed_result:
        pivot = 'history'
        filter_length = []
        for i in range(len(processed_result[pivot])):
            x = processed_result[pivot][i]
            if len(x) < 800:
                # print(f'len(x): {len(x)}')
                pass
            # print()
            # filter_length.append(x)
            if len(x) > 800:
                filter_length.append(x[:800])
            else:
                filter_length.append(x)
            # elif len(processed_result[pivot][i]) == 801:
            #     filter_length.append(x[:800])
            # else:
            #     filter_length.append(x + [x[-1]] * (800 - len(x)))
        # print(processed_result[pivot])

        temp_length = []
        for item in filter_length:
            temp_length.append(len(item))
        processed_result[pivot] = filter_length
        a = copy.deepcopy(processed_result[pivot])
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
        processed_result['se'] = cal_se(np.std(processed_result[pivot], axis=0), len(processed_result[pivot]))
        processed_result['max'] = np.max(processed_result[pivot], axis=0)
        processed_result['min'] = np.min(processed_result[pivot], axis=0)
        b = np.max(processed_result[pivot], axis=1)
        processed_result['mean_of_max'] = np.mean(np.max(processed_result[pivot], axis=1))
        processed_result['se_of_max'] = cal_se(np.std(np.max(processed_result[pivot], axis=1)), len(processed_result[pivot]))
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0)
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0)
        processed_result[pivot] = processed_result[pivot].tolist()
    else:
        for k, v in processed_result.items():
            # print(f'key {k}')
            # print(f'value length {len(v)}')
            summarize_result(v)
        return
    return


def extract_processed_result(extracted_processed_result, processed_result, control):
    if 'exp' in processed_result or 'history' in processed_result:
        exp_name = '_'.join(control[:-1])
        metric_name = control[-1]
        if exp_name not in extracted_processed_result:
            extracted_processed_result[exp_name] = defaultdict()
        extracted_processed_result[exp_name]['{}_mean'.format(metric_name)] = processed_result['mean']
        extracted_processed_result[exp_name]['{}_se'.format(metric_name)] = processed_result['se']

        extracted_processed_result[exp_name]['{}_mean_of_max'.format(metric_name)] = processed_result['mean_of_max']
        extracted_processed_result[exp_name]['{}_se_of_max'.format(metric_name)] = processed_result['se_of_max']
    else:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def write_xlsx(path, df, startrow=0):
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return

def write_max_xlsx(path, df, startrow=0):
    writer = pd.ExcelWriter(path, engine='xlsxwriter')

    dataset_name = ['CIFAR10_', 'CIFAR100_', 'FEMNIST_']
    resource_ratios = ['0.3-0.7', '0.6-0.4', '0.9-0.1']
    diff_freqs = ['4-1', '5-1', '6-1']    

    server_ratio_client_ratio = []
    # for client resource constraint
    for ratio in resource_ratios:
        for freq in diff_freqs:
            server_ratio_client_ratio.append(f'1-0_{ratio}_{freq}')
    
    # for server resource constraint
    for ratio in resource_ratios:
        for freq in diff_freqs:
            server_ratio_client_ratio.append(f'{ratio}_1-0_{freq}')

    def write_one_dataset(cur_dataset, df, writer, server_ratio_client_ratio, startrow=0):
        # global result_path
        dynamicfl_name = []
        for df_name in df:
            if cur_dataset in df_name:
                # print(f'write {df_name}')
                if 'dynamicfl' in df_name:
                    dynamicfl_name.append(df_name)
                    continue
                
                sheet_name = cur_dataset            
                df[df_name] = pd.concat(df[df_name])
                df[df_name].to_excel(writer, sheet_name=sheet_name, startrow=startrow + 1)
                writer.sheets[sheet_name].write_string(startrow, 0, df_name)
                startrow = startrow + len(df[df_name].index) + 3
        
        # TODO: fix this
        if args['type'] == 'freq_ablation':
            return
        if args['type'] == 'communication_cost':
            return
        dynamicfl_name_sort = sorted(dynamicfl_name, key=lambda v: server_ratio_client_ratio.index(v[-15:]))

        while len(dynamicfl_name_sort) > 0:
            for df_name in df:
                if cur_dataset in df_name and 'dynamicfl' in df_name and dynamicfl_name_sort and dynamicfl_name_sort[0] in df_name:
                    sheet_name = cur_dataset            
                    df[df_name] = pd.concat(df[df_name])
                    df[df_name].to_excel(writer, sheet_name=sheet_name, startrow=startrow + 1)
                    writer.sheets[sheet_name].write_string(startrow, 0, df_name)
                    startrow = startrow + len(df[df_name].index) + 3
                    dynamicfl_name_sort.pop(0)

        return 
    
    for i in range(len(dataset_name)):
        cur_dataset = dataset_name[i]
        write_one_dataset(cur_dataset, df, writer, server_ratio_client_ratio)
        
    writer.save()
    return

def make_df_exp(extracted_processed_result_exp):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_exp:
        control = exp_name.split('_')
        if len(control) == 3:
            data_name, model_name, num_supervised = control
            index_name = ['1']
            df_name = '_'.join([data_name, model_name, num_supervised])
        elif len(control) == 10:
            data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, \
            local_epoch, gm, sbn = control
            index_name = ['_'.join([local_epoch, gm])]
            df_name = '_'.join(
                [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, sbn])
        elif len(control) == 11:
            data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, \
            local_epoch, gm, sbn, ft = control
            index_name = ['_'.join([local_epoch, gm])]
            df_name = '_'.join(
                [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, sbn,
                 ft])
        else:
            raise ValueError('Not valid control')
        df[df_name].append(pd.DataFrame(data=extracted_processed_result_exp[exp_name], index=index_name))
    write_xlsx('{}/result_exp.xlsx'.format(result_path), df)
    return df


def make_df_history(extracted_processed_result_history):
    df = defaultdict(list)
    df_of_max_non_iid_l_1 = defaultdict(list)
    df_of_max_non_iid_l_2 = defaultdict(list)
    df_of_max_non_iid_d_01 = defaultdict(list)
    df_of_max_non_iid_d_03 = defaultdict(list)


    for exp_name in extracted_processed_result_history:
        # print(f'exp_name: {exp_name}')
        control = exp_name.split('_')
        # print(f'len_control: {len(control)}')
        # dp
        if len(control) == 7:
            data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch = control
            df_name = '_'.join(
                [data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch])
            for k in extracted_processed_result_history[exp_name]:
                index_name = ['_'.join([active_rate, data_split_mode, k])]
                # print(k)
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        # cnn & resnet18
        elif len(control) == 10:
            data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
            epoch, server_ratio, client_ratio, freq = control
            df_name = '_'.join(
                [data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
                    epoch, server_ratio, client_ratio, freq])
            
            max_mean_plus_se = []
            for k in extracted_processed_result_history[exp_name]:
                index_name = ['_'.join([active_rate, data_split_mode, k])]
                # print(f'k is: {k}')
                # print('\n')
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))

                if 'Accuracy' in k and 'mean_of_max' in k:
                    max_mean_plus_se.append(str(extracted_processed_result_history[exp_name][k].reshape(1, -1)[0][0]))
                # first is mean_of_max, second is std_of_max
                if 'Accuracy' in k and 'se_of_max' in k:
                    max_mean_plus_se.append('plus/minus')
                    # a = extracted_processed_result_history[exp_name][k].reshape(1, -1)
                    # b = extracted_processed_result_history[exp_name][k].reshape(1, -1)[0]
                    max_mean_plus_se.append(str(extracted_processed_result_history[exp_name][k].reshape(1, -1)[0][0]))
                    res = ' '.join(max_mean_plus_se)
                    if 'non-iid-l-1' in exp_name:
                        df_of_max_non_iid_l_1[df_name].append(
                        pd.DataFrame(data=[res], index=index_name))
                    elif 'non-iid-l-2' in exp_name:
                        df_of_max_non_iid_l_2[df_name].append(
                        pd.DataFrame(data=[res], index=index_name))
                    elif 'non-iid-d-0.1' in exp_name:
                        df_of_max_non_iid_d_01[df_name].append(
                        pd.DataFrame(data=[res], index=index_name))
                    elif 'non-iid-d-0.3' in exp_name:
                        df_of_max_non_iid_d_03[df_name].append(
                        pd.DataFrame(data=[res], index=index_name))
        elif len(control) == 11:
            data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
            epoch, server_ratio, client_ratio, freq, _ = control
            df_name = '_'.join(
                [data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
                    epoch, server_ratio, client_ratio, freq, _])
            
            max_mean_plus_se = []
            for k in extracted_processed_result_history[exp_name]:
                index_name = ['_'.join([active_rate, data_split_mode, k])]
                # print(f'k is: {k}')
                # print('\n')
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        elif len(control) == 12:
            data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
            epoch, server_ratio, client_ratio, freq, _, _ = control
            df_name = '_'.join(
                [data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
                    epoch, server_ratio, client_ratio, freq, _, _])
            
            max_mean_plus_se = []
            for k in extracted_processed_result_history[exp_name]:
                index_name = ['_'.join([data_name, active_rate, data_split_mode, freq, server_ratio, client_ratio, k])]
                # print(f'k is: {k}')
                # print('\n')
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
                # if 'Accuracy' in k and 'mean_of_max' in k:
                #     max_mean_plus_se.append(str(extracted_processed_result_history[exp_name][k].reshape(1, -1)[0][0]))
                # # first is mean_of_max, second is std_of_max
                # if 'Accuracy' in k and 'se_of_max' in k:
                #     max_mean_plus_se.append('plus/minus')
                #     # a = extracted_processed_result_history[exp_name][k].reshape(1, -1)
                #     # b = extracted_processed_result_history[exp_name][k].reshape(1, -1)[0]
                #     max_mean_plus_se.append(str(extracted_processed_result_history[exp_name][k].reshape(1, -1)[0][0]))
                #     res = ' '.join(max_mean_plus_se)
                #     if 'non-iid-l-1' in exp_name:
                #         df_of_max_non_iid_l_1[df_name].append(
                #         pd.DataFrame(data=[res], index=index_name))
                #     elif 'non-iid-l-2' in exp_name:
                #         df_of_max_non_iid_l_2[df_name].append(
                #         pd.DataFrame(data=[res], index=index_name))
                #     elif 'non-iid-d-0.1' in exp_name:
                #         df_of_max_non_iid_d_01[df_name].append(
                #         pd.DataFrame(data=[res], index=index_name))
                #     elif 'non-iid-d-0.3' in exp_name:
                #         df_of_max_non_iid_d_03[df_name].append(
                #         pd.DataFrame(data=[res], index=index_name))
        # if len(control) == 3:
        #     data_name, model_name, num_supervised = control
        #     index_name = ['1']
        #     for k in extracted_processed_result_history[exp_name]:
        #         df_name = '_'.join([data_name, model_name, num_supervised, k])
        #         df[df_name].append(
        #             pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        # elif len(control) == 10:
        #     data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, \
        #     local_epoch, gm, sbn = control
        #     index_name = ['_'.join([local_epoch, gm])]
        #     for k in extracted_processed_result_history[exp_name]:
        #         df_name = '_'.join(
        #             [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode,
        #              sbn, k])
        #         df[df_name].append(
        #             pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        # elif len(control) == 11:
        #     data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, \
        #     local_epoch, gm, sbn, ft = control
        #     index_name = ['_'.join([local_epoch, gm])]
        #     for k in extracted_processed_result_history[exp_name]:
        #         df_name = '_'.join(
        #             [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode,
        #              sbn, ft, k])
        #         df[df_name].append(
        #             pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        else:
            raise ValueError('Not valid control')
    write_xlsx('{}/result_history.xlsx'.format(result_path), df)
    write_max_xlsx('{}/result_history_l_1_only_max.xlsx'.format(result_path), df_of_max_non_iid_l_1)
    write_max_xlsx('{}/result_history_l_2_only_max.xlsx'.format(result_path), df_of_max_non_iid_l_2)
    write_max_xlsx('{}/result_history_d_01_only_max.xlsx'.format(result_path), df_of_max_non_iid_d_01)
    write_max_xlsx('{}/result_history_d_03_only_max.xlsx'.format(result_path), df_of_max_non_iid_d_03)
    return df


def transfer_freq_label(label):
    '''
    transfer frequency number label to frequency character label
    '''
    label_correspondence = {'6': 'a', '5': 'b', '4': 'c', '3': 'd', '2.5': 'e', '2': 'f', '1': 'g'}
    label_list = label.split('-')
    res = []
    for item in label_list:
        res.append(label_correspondence[item])
    return "-".join(res)

def transfer_data_heterogeneity(heterogeneity):

    heterogeneity_correspondence = {
        'non-iid-l-1': 'Non-IID-Class-1', 
        'non-iid-l-2': 'Non-IID-Classes-2',
        'non-iid-d-0.01': 'Non-IID-Dirichlet-0.01',
        'non-iid-d-0.1': 'Non-IID-Dirichlet-0.1',
        'non-iid-d-0.3': 'Non-IID-Dirichlet-0.3',
    }

    return heterogeneity_correspondence[heterogeneity]

def change_decimal_to_percentage(decimal):
    return '{:.2%}'.format(float(decimal))

def cut_decimal(decimal):
    decimal = float(decimal)
    return format(decimal, '.4f')

def make_vis(df_exp, df_history):
    # global result_path
    data_split_mode_dict = {'iid': 'IID', 'non-iid-l-2': 'Non-IID, $K=2$',
                            'non-iid-d-0.1': 'Non-IID, $\operatorname{Dir}(0.1)$',
                            'non-iid-d-0.3': 'Non-IID, $\operatorname{Dir}(0.3)$', 'fix-fsgd': 'DynamicSgd + FixMatch',
                            'fix-batch': 'FedAvg + FixMatch', 'fs': 'Fully Supervised', 'ps': 'Partially Supervised'}
    

    color = {'5_0.5': 'red', '1_0.5': 'orange', '5_0': 'dodgerblue', '5_0.9': 'blue', '5_0.5_nomixup': 'green',
             '5_0_nomixup': 'green', 'iid': 'red', 'non-iid-l-2': 'orange', 'non-iid-d-0.1': 'dodgerblue',
             'non-iid-d-0.3': 'green', 'fix-fsgd': 'red', 'fix-batch': 'blue',
             'fs': 'black', 'ps': 'orange',

             'Genetic': 'red',
             'DynaComm': 'orange',
             'Brute-force': 'green',

             'fedavg': 'purple',
             'fedprox': 'dodgerblue',
             'fedensemble': 'green',
             'scaffold': 'brown',
             'dynamicfl': 'red',
             'fedgen': 'pink',
             'feddyn': 'black',
             'fednova': 'orange',

             'dynamicfl_0.3-0.7': 'red',
             'dynamicfl_0.6-0.4': 'green',
             'dynamicfl_0.9-0.1': 'dodgerblue',

             'freq_6-0': 'red',
             'freq_6-5': 'green',
             'freq_6-4': 'dodgerblue',
             'freq_6-3': 'brown',
             'freq_6-2.5': 'orange',
             'freq_6-2': 'black',
             'freq_6-1': 'purple',
             
             'freq_5-0': 'red',
             'freq_5-4': 'green',
             'freq_5-3': 'dodgerblue',
             'freq_5-2.5': 'brown',
             'freq_5-2': 'orange',
             'freq_5-1': 'black',

             'freq_4-0': 'red',
             'freq_4-3': 'green',
             'freq_4-2.5': 'dodgerblue',
             'freq_4-2': 'brown',
             'freq_4-1': 'orange',

             'freq_3-0': 'red',
             'freq_3-2.5': 'green',
             'freq_3-2': 'dodgerblue',
             'freq_3-1': 'brown',
             }
    linestyle = {'5_0.5': '-', '1_0.5': '--', '5_0': ':', '5_0.5_nomixup': '-.', '5_0_nomixup': '-.',
                 '5_0.9': (0, (1, 5)), 'iid': '-', 'non-iid-l-2': '--', 'non-iid-d-0.1': '-.', 'non-iid-d-0.3': ':',
                 'fix-fsgd': '--', 'fix-batch': ':', 'fs': '-', 'ps': '-.',


                'Genetic': (0, (1, 1)),
                'DynaComm': '--',
                'Brute-force': (0, (3, 1, 1, 1, 1, 1)),
                
                'fedavg': (0, (1, 1)),
                'fedprox': '--',
                'fedensemble': (0, (3, 1, 1, 1, 1, 1)),
                'scaffold': (5, (10, 3)),
                'dynamicfl': (0, (5, 1)),
                'fedgen': (0, (3, 1, 1, 1)),
                'feddyn': (0, (3, 1, 1, 1)),
                'fednova': (0, (3, 1, 1, 1)),

                'dynamicfl_0.3-0.7': (0, (5, 1)),
                'dynamicfl_0.6-0.4': (0, (3, 1, 1, 1)),
                'dynamicfl_0.9-0.1': (0, (3, 1, 1, 1)),

                'freq_6-0': (0, (5, 1)),
                'freq_6-5': (0, (1, 1)),
                'freq_6-4': '--',
                'freq_6-3': (0, (3, 1, 1, 1, 1, 1)),
                'freq_6-2.5': (5, (10, 3)),
                'freq_6-2': (0, (3, 1, 1, 1)),
                'freq_6-1': ':',
                
                'freq_5-0': (0, (5, 1)),
                'freq_5-4': (0, (1, 1)),
                'freq_5-3': '--',
                'freq_5-2.5': (0, (3, 1, 1, 1, 1, 1)),
                'freq_5-2': (5, (10, 3)),
                'freq_5-1': (0, (3, 1, 1, 1)),

                'freq_4-0': (0, (5, 1)),
                'freq_4-3': (0, (1, 1)),
                'freq_4-2.5': '--',
                'freq_4-2': (0, (3, 1, 1, 1, 1, 1)),
                'freq_4-1': (5, (10, 3)),

                'freq_3-0': (0, (5, 1)),
                'freq_3-2.5': (0, (1, 1)),
                'freq_3-2': '--',
                'freq_3-1': (0, (3, 1, 1, 1, 1, 1)),
                }
        
    resource_ratios = ['0.3-0.7', '0.6-0.4', '0.9-0.1']
    diff_freqs = ['6-1', '5-1', '4-1']
    color_for_dynamicfl = ['red', 'green', 'dodgerblue']
    linestyle_for_dynamicfl = [(0, (5, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1))]
    
    
    for ratio in resource_ratios:
        index = 0
        for freq in diff_freqs:
            new_key = f'dynamicfl_sameResourcediffFreq_{ratio}_{freq}'
            color[new_key] = color_for_dynamicfl[index]
            linestyle[new_key] = linestyle_for_dynamicfl[index]
            index += 1

    for freq in diff_freqs:
        index = 0
        for ratio in resource_ratios:
            new_key = f'dynamicfl_sameFreqdiffResource_{ratio}_{freq}'
            color[new_key] = color_for_dynamicfl[index]
            linestyle[new_key] = linestyle_for_dynamicfl[index]
            index += 1


    loc_dict = {'Accuracy': 'lower right', 'Loss': 'upper right'}
    fontsize = {'legend': 14, 'label': 14, 'ticks': 14, 'group_x_ticks': 8}
    metric_name = 'Accuracy'
    fig = {}
    reorder_fig = []

    dynamicfl_cost_mean_list = []
    dynamicfl_cost_se_list = []
    dynamicfl_cost_ratio_mean_list = []
    dynamicfl_cost_ratio_se_list = []

    dynamicfl_cost_name = []
    dynamicfl_cost_ratio_name = []
    for df_name in df_history:
        df_name_list = df_name.split('_')
        if len(df_name_list) == 7:
            data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, epoch = df_name_list
            # _, _, _, _, data_split_mode, _, \
            # _, _, _ = df_name_list
            # df_name_std = '_'.join([data_name, model_name, num_supervised, metric_name, 'std'])
            # fig_name = '_'.join([data_name, model_name, num_supervised, metric_name])
            # fig[fig_name] = plt.figure(fig_name)
            
            # data_split_mode_dict = {
            #     'non-iid-d-0.1': []
            #     'non-iid-d-0.3': []
            #     'non-iid-l-1': []
            #     'non-iid-l-2': []
            # }

            # index_list = []
            Brute_force_KL_mean_list = []
            Brute_force_KL_se_list = []
            Brute_force_KL_time_list = []
            Brute_force_KL_time_se_list = []

            Brute_force_QL_mean_list = []
            Brute_force_QL_se_list = []
            Brute_force_QL_time_list = []
            Brute_force_QL_time_se_list = []

            Genetic_KL_mean_list = []
            Genetic_KL_se_list = []
            Genetic_KL_time_list = []
            Genetic_KL_time_se_list = []

            Genetic_QL_mean_list = []
            Genetic_QL_se_list = []
            Genetic_QL_time_list = []
            Genetic_QL_time_se_list = []

            dp_KL_mean_list = []
            dp_KL_se_list = []
            dp_KL_time_list = []
            dp_KL_time_se_list = []

            for index, row in df_history[df_name].iterrows():
                a = index
                b = row
                if 'of_max' in index:
                    continue
                if 'time' in index:
                    if 'mean' in index:
                        if 'KL' in index and 'brute_force' in index:
                            row = row + 1
                            row = np.log(row)
                            Brute_force_KL_time_list.append(np.mean(row))
                            Brute_force_KL_time_se_list.append(cal_se(np.std(row), len(row)))
                        elif 'KL' in index and 'genetic' in index:
                            row = row + 1
                            row = np.log(row)
                            Genetic_KL_time_list.append(np.mean(row))
                            Genetic_KL_time_se_list.append(cal_se(np.std(row), len(row)))
                        elif 'KL' in index and 'dp' in index:
                            # avoid negative log
                            row = row + 1
                            row = np.log(row)
                            dp_KL_time_list.append(np.mean(row))
                            dp_KL_time_se_list.append(cal_se(np.std(row), len(row)))
                    # elif 'se' in index:
                    #     if 'KL' in index and 'brute_force' in index:
                    #         Brute_force_KL_time_se_list.append(row)
                    #     elif 'KL' in index and 'genetic' in index:
                    #         Genetic_KL_time_se_list.append(row)
                    #     elif 'KL' in index and 'dp' in index:
                    #         dp_KL_time_se_list.append(row)
                elif 'mean' in index:
                    if 'KL' in index and 'brute_force' in index:
                        Brute_force_KL_mean_list.append(np.mean(row))
                        Brute_force_KL_se_list.append(cal_se(np.std(row), len(row)))
                    elif 'KL' in index and 'genetic' in index:
                        Genetic_KL_mean_list.append(np.mean(row))
                        Genetic_KL_se_list.append(cal_se(np.std(row), len(row)))
                    elif 'KL' in index and 'dp' in index:
                        dp_KL_mean_list.append(np.mean(row))
                        dp_KL_se_list.append(cal_se(np.std(row), len(row)))
                # elif 'se' in index:
                #     if 'KL' in index and 'brute_force' in index:
                #         Brute_force_KL_se_list.append(row)
                #     elif 'KL' in index and 'genetic' in index:
                #         Genetic_KL_se_list.append(row)
                #     elif 'KL' in index and 'dp' in index:
                #         dp_KL_se_list.append(row)

                a = 6
            def plot_group_clients_fig(fig_name, mean_list, se_list, key, ylabel):
                
                # for KL, get odd indices
                # fig_name = '_'.join([data_name, data_split_mode, 'KL Divergence'])
                fig[fig_name] = plt.figure(fig_name)
                y = np.array(mean_list)
                yerr = np.array(se_list)
                x = np.arange(len(y))
                x_axis = [i for i in range(1,len(y)+1)]

                fig_name_list = fig_name.split('_')
                active_rate = float(fig_name_list[3])
                if active_rate != 0.1 and 'Brute-force' in key:
                    gap = int(active_rate * 100) // 10
                    gap_list = [i for i in range(1,len(y)+1,gap)]

                    x_axis = copy.deepcopy(gap_list)
                    gap_list = [i-1 for i in gap_list]
                    x = copy.deepcopy(gap_list)
                    y = y[gap_list]
                    yerr = yerr[gap_list]
                elif len(x_axis) > 10:
                    gap = len(x_axis) // 10
                    gap_list = [i for i in range(1,len(y)+1,gap)]
                    if gap_list[-1] != len(y):
                        gap_list.append(len(y))

                    x_axis = copy.deepcopy(gap_list)
                    gap_list = [i-1 for i in gap_list]
                    x = copy.deepcopy(gap_list)
                    y = y[gap_list]
                    yerr = yerr[gap_list]
                
                # key = f'active_rate_{active_rate}_Genetic'
                plt.plot(x, y, color=color[key], linestyle=linestyle[key])
                # plt.fill_between(x, (y - yerr), (y + yerr), color=color[key], alpha=.1)
                label = f'{key}'
                # print('plot_group_clients_fig', x, y, label)
                plt.errorbar(x, y, yerr=yerr, color=color[key], linestyle=linestyle[key],
                    label=label)
                plt.xlabel('Client subset size', fontsize=fontsize['label'])
                plt.ylabel(ylabel, fontsize=fontsize['label'])
                plt.xticks(x, x_axis, fontsize=fontsize['group_x_ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
                plt.legend(loc=loc_dict['Loss'], fontsize=fontsize['legend'])
                return
            
            if len(Brute_force_KL_mean_list) > 0:
                plot_group_clients_fig(
                    '_'.join([data_name, data_split_mode, 'active-rate', active_rate, 'KL_Divergence']), 
                    Brute_force_KL_mean_list, 
                    Brute_force_KL_se_list, 
                    f'Brute-force', 
                    'KL Divergence'
                )

            # plot_group_clients_fig(
            #     '_'.join([data_name, data_split_mode, 'Quadratic Loss']), 
            #     Brute-force_QL_mean_list, 
            #     Brute-force_QL_se_list, 
            #     f'active_rate_{active_rate}_Brute-force', 
            #     'Quadratic Loss'
            # )
            if len(Genetic_KL_mean_list) > 0:
                plot_group_clients_fig(
                    '_'.join([data_name, data_split_mode, 'active-rate', active_rate, 'KL_Divergence']), 
                    Genetic_KL_mean_list, 
                    Genetic_KL_se_list, 
                    f'Genetic', 
                    'KL Divergence'
                )

            # plot_group_clients_fig(
            #     '_'.join([data_name, data_split_mode, 'Quadratic Loss']), 
            #     Genetic_QL_mean_list, 
            #     Genetic_QL_se_list, 
            #     f'active_rate_{active_rate}_Genetic', 
            #     'Quadratic Loss'
            # )
            if len(dp_KL_mean_list) > 0:
                plot_group_clients_fig(
                    '_'.join([data_name, data_split_mode, 'active-rate', active_rate, 'KL_Divergence']), 
                    dp_KL_mean_list, 
                    dp_KL_se_list, 
                    f'DynaComm', 
                    'KL Divergence'
                )

            
            # figures for time cost
            if len(Brute_force_KL_time_list) > 0:
                plot_group_clients_fig(
                    '_'.join([data_name, data_split_mode, 'active-rate', active_rate, 'KL_Divergence', 'time_cost']), 
                    Brute_force_KL_time_list, 
                    Brute_force_KL_time_se_list, 
                    f'Brute-force', 
                    'Seconds (log scale)'
                )

            if len(Genetic_KL_time_list) > 0:
                plot_group_clients_fig(
                    '_'.join([data_name, data_split_mode, 'active-rate', active_rate, 'KL_Divergence', 'time_cost']), 
                    Genetic_KL_time_list, 
                    Genetic_KL_time_se_list, 
                    f'Genetic', 
                    'Seconds (log scale)'
                )

            if len(dp_KL_time_list) > 0:
                plot_group_clients_fig(
                    '_'.join([data_name, data_split_mode, 'active-rate', active_rate, 'KL_Divergence', 'time_cost']), 
                    dp_KL_time_list, 
                    dp_KL_time_se_list, 
                    f'DynaComm', 
                    'Seconds (log scale)'
                )

            # plot_group_clients_fig(
            #     '_'.join([data_name, data_split_mode, 'Quadratic Loss', 'time_cost']), 
            #     dp_QL_time_list, 
            #     dp_QL_time_se_list, 
            #     f'active_rate_{active_rate}_dp', 
            #     'Second'
            # )


        elif len(df_name_list) == 10 or len(df_name_list) == 11:
            if len(df_name_list) == 10:
                data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
                epoch, server_ratio, client_ratio, freq = df_name_list
            elif len(df_name_list) == 11:
                data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
                epoch, server_ratio, client_ratio, freq, _ = df_name_list
            
            # print(f'plotting {df_name_list}')
            # if stat == 'std':
            #     continue
            # df_name_std = '_'.join([data_name, model_name, num_supervised, metric_name, 'std'])

            def draw_figure(plt, x, y, yerr, label, key_for_dict, x_label='Communication Rounds'):
                plt.plot(x, y, color=color[key_for_dict], linestyle=linestyle[key_for_dict], label=label)
                # plt.fill_between(x, (y - yerr), (y + yerr), color=color[algo_mode], alpha=.1)

                plt.errorbar(x, y, yerr=yerr, color=color[key_for_dict], linestyle=linestyle[key_for_dict])
                
                plt.xlabel(x_label, fontsize=fontsize['label'])
                plt.ylabel(metric_name, fontsize=fontsize['label'])
                plt.xticks(fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
                plt.legend(loc=loc_dict['Accuracy'], fontsize=fontsize['legend'])
                return
            
            dominate_agent = None
            dominate_ratio = None
            if server_ratio == '1-0':
                dominate_agent = 'client'
                dominate_ratio = client_ratio
            elif client_ratio == '1-0':
                dominate_agent = 'server'
                dominate_ratio = server_ratio

            if algo_mode == 'dynamicfl':
                if args['type'] == 'freq_ablation':
                    # print(f'freq: {freq}')
                    freq_list = freq.split('-')
                    high_freq = freq_list[0]
                    fig_name = '_'.join([data_name, data_split_mode, dominate_agent, dominate_ratio, high_freq])

                    fig[fig_name] = plt.figure(fig_name)
                    temp = df_history[df_name].iterrows()
                    for ((index, row), (index_se, row_se)) in zip(temp, temp):
                        if 'Accuracy_mean' not in index:
                            continue
                        if 'of_max' in index:
                            continue
                        
                        # only high freq
                        if len(df_name_list) == 11:
                            label = f'{high_freq}-0'
                            key_for_dict = f'freq_{high_freq}-0'
                        # high freq + low freq
                        else:
                            label = f'{freq}'
                            key_for_dict = f'freq_{freq}'

                        y = row.to_numpy()
                        yerr = row_se.to_numpy()

                        x = np.arange(len(y))
                        draw_figure(plt, x, y, yerr, label, key_for_dict)
                # elif len(df_name_list) == 11:
                #     pass
                else:
                    fig_name_sameResourcediffFreq = '_'.join([data_name, data_split_mode, dominate_agent, dominate_ratio, 'sameResourcediffFreq'])
                    fig_name_sameFreqdiffResource = '_'.join([data_name, data_split_mode, dominate_agent, freq, 'sameFreqdiffResource'])

                    fig[fig_name_sameResourcediffFreq] = plt.figure(fig_name_sameResourcediffFreq)
                    temp = df_history[df_name].iterrows()
                    for ((index, row), (index_se, row_se)) in zip(temp, temp):
                        if 'Accuracy_mean' not in index:
                            continue
                        if 'of_max' in index:
                            continue
                        label = f'{freq}'
                        key_for_dict = '_'.join(['dynamicfl_sameResourcediffFreq', dominate_ratio, freq])
                        # print(f'label, {label}')

                        y = row.to_numpy()
                        yerr = row_se.to_numpy()

                        x = np.arange(len(y))
                        draw_figure(plt, x, y, yerr, label, key_for_dict)

                    fig[fig_name_sameFreqdiffResource] = plt.figure(fig_name_sameFreqdiffResource)
                    temp = df_history[df_name].iterrows()
                    for ((index, row), (index_se, row_se)) in zip(temp, temp):
                        if 'Accuracy_mean' not in index:
                            continue
                        if 'of_max' in index:
                            continue
                        label = f'{dominate_ratio}'
                        key_for_dict = '_'.join(['dynamicfl_sameFreqdiffResource', dominate_ratio, freq])
                        # print(f'label, {label}')

                        y = row.to_numpy()
                        yerr = row_se.to_numpy()

                        x = np.arange(len(y))
                        draw_figure(plt, x, y, yerr, label, key_for_dict)
            else:
                fig_name = '_'.join([data_name, data_split_mode])
                fig[fig_name] = plt.figure(fig_name)
                temp = df_history[df_name].iterrows()
                for ((index, row), (index_se, row_se)) in zip(temp, temp):
                    if 'Accuracy_mean' not in index:
                        continue
                    
                    if 'of_max' in index:
                        continue
                    # print(f'index, {index}')
                    # print(f'index_std, {index_std}')
                    label = f'{algo_mode}'
                    # print(f'label, {label}')

                    y = row.to_numpy()
                    yerr = row_se.to_numpy()
                    # print(f'y, {y[:100]}')
                    # print(f'yerr, {yerr[:100]}')
                    x = np.arange(len(y))
                    draw_figure(plt, x, y, yerr, label, key_for_dict=algo_mode)
                    # plt.plot(x, y, color=color[algo_mode], linestyle=linestyle[algo_mode], label=label)
                    # # plt.fill_between(x, (y - yerr), (y + yerr), color=color[algo_mode], alpha=.1)

                    # plt.errorbar(x, y, yerr=yerr, color=color[algo_mode], linestyle=linestyle[algo_mode])
                    
                    # plt.xlabel('Communication Rounds', fontsize=fontsize['label'])
                    # plt.ylabel(metric_name, fontsize=fontsize['label'])
                    # plt.xticks(fontsize=fontsize['ticks'])
                    # plt.yticks(fontsize=fontsize['ticks'])
                    # plt.legend(loc=loc_dict['Accuracy'], fontsize=fontsize['legend'])
        
        # for communication cost
        elif len(df_name_list) == 12:
            data_name, model_name, active_rate, num_clients, data_split_mode, algo_mode, \
                epoch, server_ratio, client_ratio, freq, _, _ = df_name_list
            # _, _, _, _, data_split_mode, _, \
            # _, _, _ = df_name_list
            # df_name_std = '_'.join([data_name, model_name, num_supervised, metric_name, 'std'])
            # fig_name = '_'.join([data_name, model_name, num_supervised, metric_name])
            # fig[fig_name] = plt.figure(fig_name)
            
            # data_split_mode_dict = {
            #     'non-iid-d-0.1': []
            #     'non-iid-d-0.3': []
            #     'non-iid-l-1': []
            #     'non-iid-l-2': []
            # }

            # index_list = []

            # f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_dynamicfl_cost_{num_clients}": cur_dynamicfl_cost,
            # f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_fedavg_cost_{num_clients}": fedavg_cost,
            # f"best_dp_KL_{cfg['server_ratio']}_{cfg['client_ratio']}_ratio_communication_cur_dynamicfl_cost_ratio_{num_clients}": cur_dynamicfl_cost_ratio,
            
            
            for index, row in df_history[df_name].iterrows():
                a = index
                b = row
                if 'of_max' in index:
                    continue
                if 'dynamicfl_cost_ratio' in index:
                    if 'mean' in index:
                        if 'KL' in index:
                            # print(f'cost_ratio_index, {index}\n')
                            dynamicfl_cost_ratio_name.append(index)
                            dynamicfl_cost_ratio_mean_list.append(np.mean(row))
                            dynamicfl_cost_ratio_se_list.append(cal_se(np.std(row), len(row)))
                elif 'dynamicfl_cost' in index:
                    if 'mean' in index:
                        if 'KL' in index:
                            # print(f'cost_index, {index}\n')
                            dynamicfl_cost_name.append(index)
                            dynamicfl_cost_mean_list.append(np.mean(row))
                            dynamicfl_cost_se_list.append(cal_se(np.std(row), len(row)))
                
                

            
            # for ((index, row), (index_std, row_std)) in zip(temp, temp):
            #     a = index
            #     b = row
            #     c = index_std
            #     d = row_std
            

            # for ((index, row), (_, row_std)) in zip(df_history[df_name].iterrows(), df_history[df_name_std].iterrows()):
            
                # plt.errorbar(x, y, yerr=yerr, color=color[key], linestyle=linestyle[key],
                #     label=label)
                # plt.xlabel('Client subset size', fontsize=fontsize['label'])
                # plt.ylabel(ylabel, fontsize=fontsize['label'])
                # plt.xticks(x, x_axis, fontsize=fontsize['group_x_ticks'])
                # plt.yticks(fontsize=fontsize['ticks'])
                # plt.legend(loc=loc_dict['Loss'], fontsize=fontsize['legend'])
            # for ((index, row), (_, row_std)) in zip(df_history[df_name].iterrows(), df_history[df_name_std].iterrows()):
            #     y = row.to_numpy()
            #     yerr = row_std.to_numpy()
            #     x = np.arange(len(y))
            #     plt.plot(x, y, color='r', linestyle='-')
            #     plt.fill_between(x, (y - yerr), (y + yerr), color='r', alpha=.1)
            #     plt.xlabel('Communication Rounds', fontsize=fontsize['label'])
            #     plt.ylabel(metric_name, fontsize=fontsize['label'])
            #     plt.xticks(fontsize=fontsize['ticks'])
            #     plt.yticks(fontsize=fontsize['ticks'])

        # if len(df_name_list) == 5:
        #     data_name, model_name, num_supervised, metric_name, stat = df_name.split('_')
        #     if stat == 'std':
        #         continue
        #     df_name_std = '_'.join([data_name, model_name, num_supervised, metric_name, 'std'])
        #     fig_name = '_'.join([data_name, model_name, num_supervised, metric_name])
        #     fig[fig_name] = plt.figure(fig_name)
        #     for ((index, row), (_, row_std)) in zip(df_history[df_name].iterrows(), df_history[df_name_std].iterrows()):
        #         y = row.to_numpy()
        #         yerr = row_std.to_numpy()
        #         x = np.arange(len(y))
        #         plt.plot(x, y, color='r', linestyle='-')
        #         plt.fill_between(x, (y - yerr), (y + yerr), color='r', alpha=.1)
        #         plt.xlabel('Communication Rounds', fontsize=fontsize['label'])
        #         plt.ylabel(metric_name, fontsize=fontsize['label'])
        #         plt.xticks(fontsize=fontsize['ticks'])
        #         plt.yticks(fontsize=fontsize['ticks'])
        # elif len(df_name_list) == 10:
        #     data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, sbn, \
        #     metric_name, stat = df_name.split('_')
        #     if stat == 'std':
        #         continue
        #     df_name_std = '_'.join(
        #         [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, sbn,
        #          metric_name, 'std'])
        #     for ((index, row), (_, row_std)) in zip(df_history[df_name].iterrows(), df_history[df_name_std].iterrows()):
        #         y = row.to_numpy()
        #         yerr = row_std.to_numpy()
        #         x = np.arange(len(y))
        #         if index == '5_0.5' and loss_mode == 'fix-mix':
        #             fig_name = '_'.join(
        #                 [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, sbn,
        #                  metric_name])
        #             reorder_fig.append(fig_name)
        #             label_name = '{}'.format(data_split_mode_dict[data_split_mode])
        #             style = data_split_mode
        #             fig[fig_name] = plt.figure(fig_name)
        #             plt.plot(x, y, color=color[style], linestyle=linestyle[style], label=label_name)
        #             plt.fill_between(x, (y - yerr), (y + yerr), color=color[style], alpha=.1)
        #             plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
        #             plt.xlabel('Communication Rounds', fontsize=fontsize['label'])
        #             plt.ylabel(metric_name, fontsize=fontsize['label'])
        #             plt.xticks(fontsize=fontsize['ticks'])
        #             plt.yticks(fontsize=fontsize['ticks'])
        #         if data_split_mode in ['iid', 'non-iid-l-2'] and loss_mode not in ['fix-batch', 'fix-fsgd', 'fix-frgd']:
        #             fig_name = '_'.join(
        #                 [data_name, model_name, num_supervised, num_clients, active_rate, data_split_mode, sbn,
        #                  metric_name])
        #             reorder_fig.append(fig_name)
        #             fig[fig_name] = plt.figure(fig_name)
        #             local_epoch, gm = index.split('_')
        #             if loss_mode == 'fix':
        #                 label_name = '$E={}$, $\\beta_g={}$, No mixup'.format(local_epoch, gm)
        #                 style = '{}_nomixup'.format(index)
        #             else:
        #                 label_name = '$E={}$, $\\beta_g={}$'.format(local_epoch, gm)
        #                 style = index
        #             plt.plot(x, y, color=color[style], linestyle=linestyle[style], label=label_name)
        #             plt.fill_between(x, (y - yerr), (y + yerr), color=color[style], alpha=.1)
        #             plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
        #             plt.xlabel('Communication Rounds', fontsize=fontsize['label'])
        #             plt.ylabel(metric_name, fontsize=fontsize['label'])
        #             plt.xticks(fontsize=fontsize['ticks'])
        #             plt.yticks(fontsize=fontsize['ticks'])
        #         if data_split_mode in ['iid', 'non-iid-l-2'] and loss_mode == 'fix-fsgd':
        #             fix_batch_df_name = '_'.join(
        #                 [data_name, model_name, num_supervised, 'fix-batch', num_clients, active_rate, data_split_mode,
        #                  sbn, '0', metric_name, 'mean'])
        #             fix_batch_df_name_std = '_'.join(
        #                 [data_name, model_name, num_supervised, 'fix-batch', num_clients, active_rate, data_split_mode,
        #                  sbn, '0', metric_name, 'std'])
        #             fix_batch_y = list(df_history[fix_batch_df_name].iterrows())[0][1]
        #             fix_batch_y_yerr = list(df_history[fix_batch_df_name_std].iterrows())[0][1]
        #             fs_df_name = '_'.join([data_name, model_name, 'fs'])
        #             fs_df_name_std = '_'.join([data_name, model_name, 'fs'])
        #             fs_y = list(df_exp[fs_df_name].iterrows())[0][1]['{}_mean'.format(metric_name)]
        #             fs_y_yerr = list(df_exp[fs_df_name_std].iterrows())[0][1]['{}_std'.format(metric_name)]
        #             ps_df_name = '_'.join([data_name, model_name, num_supervised])
        #             ps_df_name_std = '_'.join([data_name, model_name, num_supervised])
        #             ps_y = list(df_exp[ps_df_name].iterrows())[0][1]['{}_mean'.format(metric_name)]
        #             ps_y_yerr = list(df_exp[ps_df_name_std].iterrows())[0][1]['{}_std'.format(metric_name)]
        #             fig_name = '_'.join(
        #                 [data_name, model_name, num_supervised, num_clients, active_rate, data_split_mode, sbn,
        #                  metric_name, 'fsgd'])
        #             reorder_fig.append(fig_name)
        #             fig[fig_name] = plt.figure(fig_name)
        #             label_name = '{}'.format(data_split_mode_dict['fix-fsgd'])
        #             style = 'fix-fsgd'
        #             plt.plot(x, y, color=color[style], linestyle=linestyle[style], label=label_name)
        #             plt.fill_between(x, (y - yerr), (y + yerr), color=color[style], alpha=.1)
        #             label_name = '{}'.format(data_split_mode_dict['fix-batch'])
        #             style = 'fix-batch'
        #             plt.plot(x, fix_batch_y, color=color[style], linestyle=linestyle[style], label=label_name)
        #             plt.fill_between(x, (fix_batch_y - fix_batch_y_yerr), (fix_batch_y + fix_batch_y_yerr),
        #                              color=color[style], alpha=.1)
        #             label_name = '{}'.format(data_split_mode_dict['fs'])
        #             style = 'fs'
        #             plt.plot(x, np.repeat(fs_y, len(x)), color=color[style], linestyle=linestyle[style],
        #                      label=label_name)
        #             plt.fill_between(x, np.repeat(fs_y - fs_y_yerr, len(x)), np.repeat(fs_y + fs_y_yerr, len(x)),
        #                              color=color[style], alpha=.1)
        #             label_name = '{}'.format(data_split_mode_dict['ps'])
        #             style = 'ps'
        #             plt.plot(x, np.repeat(ps_y, len(x)), color=color[style], linestyle=linestyle[style],
        #                      label=label_name)
        #             plt.fill_between(x, np.repeat(ps_y - ps_y_yerr, len(x)), np.repeat(ps_y + ps_y_yerr, len(x)),
        #                              color=color[style], alpha=.1)
        #             plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
        #             plt.xlabel('Communication Rounds', fontsize=fontsize['label'])
        #             plt.ylabel(metric_name, fontsize=fontsize['label'])
        #             plt.xticks(fontsize=fontsize['ticks'])
        #             plt.yticks(fontsize=fontsize['ticks'])
    # for fig_name in reorder_fig:
    #     fig_name_list = fig_name.split('_')
    #     data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, sbn, metric_name = fig_name_list[:8]
    #     plt.figure(fig_name)
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     if len(fig_name_list) == 9:
    #         if len(handles) == 4:
    #             handles = [handles[2], handles[3], handles[0], handles[1]]
    #             labels = [labels[2], labels[3], labels[0], labels[1]]
    #             plt.legend(handles, labels, loc=loc_dict[metric_name], fontsize=fontsize['legend'])
    #     else:
    #         if len(handles) == 4:
    #             handles = [handles[0], handles[3], handles[2], handles[1]]
    #             labels = [labels[0], labels[3], labels[2], labels[1]]
    #             plt.legend(handles, labels, loc=loc_dict[metric_name], fontsize=fontsize['legend'])
    #         if len(handles) == 5:
    #             handles = [handles[0], handles[4], handles[2], handles[3], handles[1]]
    #             labels = [labels[0], labels[4], labels[2], labels[3], labels[1]]
    #             plt.legend(handles, labels, loc=loc_dict[metric_name], fontsize=fontsize['legend'])
    def write_xlsx(path, df, startrow=0):
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        for df_name in df:
            df[df_name] = pd.concat(df[df_name])
            df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
            writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
            startrow = startrow + len(df[df_name].index) + 3
        writer.save()
        return
    
    
    
    def process_communication_cost_excel(name_item, mean_item, se_item, excel_list):
        dynamicfl_cost_name_list = name_item.split('_')
        data_name, active_rate, data_split_mode, freq, server_ratio, client_ratio = dynamicfl_cost_name_list[:6]

        dominate_agent = None
        dominate_ratio = None
        scenario = None
        if server_ratio == '1-0':
            dominate_agent = 'client'
            dominate_ratio = client_ratio
            high_freq_percentage = change_decimal_to_percentage(client_ratio.split('-')[0])
            scenario = f'Fix-{high_freq_percentage}'

        elif client_ratio == '1-0':
            dominate_agent = 'server'
            dominate_ratio = server_ratio
            # scenario = 'Dynamic'
            high_freq_percentage = change_decimal_to_percentage(server_ratio.split('-')[0])
            scenario = f'Dynamic-{high_freq_percentage}'

        print('cur_list:', dynamicfl_cost_name_list)
        transferred_freq_label = transfer_freq_label(freq)
        transferred_data_split_mode = transfer_data_heterogeneity(data_split_mode)
        cur_res = np.array([data_name, transferred_data_split_mode, transferred_freq_label, scenario, f'{cut_decimal(mean_item)} \u00B1 {cut_decimal(se_item)}'])
        excel_list["1"].append(pd.DataFrame(data=cur_res.reshape(1, -1), index=[name_item]))
        return
    
    if args['type'] == 'communication_cost':
        dynamicfl_cost_name_excel = defaultdict(list)
        dynamicfl_cost_ratio_name_excel = defaultdict(list)
        for i in range(len(dynamicfl_cost_name)):
            print(f'{dynamicfl_cost_name[i]}: {dynamicfl_cost_mean_list[i]} \u00B1 {dynamicfl_cost_se_list[i]}\n')
            process_communication_cost_excel(dynamicfl_cost_name[i], dynamicfl_cost_mean_list[i], dynamicfl_cost_se_list[i], dynamicfl_cost_name_excel)
        write_xlsx('{}/dynamicfl_communication_cost_mean.xlsx'.format(result_path), dynamicfl_cost_name_excel)

        for i in range(len(dynamicfl_cost_ratio_name)):
            print(f'{dynamicfl_cost_ratio_name[i]}: {dynamicfl_cost_ratio_mean_list[i]} \u00B1 {dynamicfl_cost_ratio_se_list[i]}\n')
            process_communication_cost_excel(dynamicfl_cost_ratio_name[i], dynamicfl_cost_ratio_mean_list[i], dynamicfl_cost_ratio_se_list[i], dynamicfl_cost_ratio_name_excel)
        write_xlsx('{}/dynamicfl_communication_cost_ratio_mean.xlsx'.format(result_path), dynamicfl_cost_ratio_name_excel)
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, save_format)
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()


# dp_QL_mean_list = []
# dp_QL_se_list = []
# dp_QL_time_list = []
# dp_QL_time_se_list = []

# dp_KL_cc_03_mean_list = []
# dp_KL_cc_03_se_list = []
# dp_KL_cc_03_time_list = []
# dp_KL_cc_03_time_se_list = []

# dp_KL_cc_05_mean_list = []
# dp_KL_cc_05_se_list = []
# dp_KL_cc_05_time_list = []
# dp_KL_cc_05_time_se_list = []

# dp_KL_cc_07_mean_list = []
# dp_KL_cc_07_se_list = []
# dp_KL_cc_07_time_list = []
# dp_KL_cc_07_time_se_list = []

# dp_KL_cc_09_mean_list = []
# dp_KL_cc_09_se_list = []
# dp_KL_cc_09_time_list = []
# dp_KL_cc_09_time_se_list = []


# if 'time' in index:
#     if 'mean' in index:
#         if 'KL' in index and 'Brute-force' in index:
#             row = row + 1
#             Brute-force_KL_time_list.append(np.mean(np.log(row)))
#             Brute-force_KL_time_se_list.append(np.std(np.log(row)))
#         elif 'KL' in index and 'Genetic' in index:
#             row = row + 1
#             Genetic_KL_time_list.append(np.mean(np.log(row)))
#             Genetic_KL_time_se_list.append(np.std(np.log(row)))
#         elif 'KL' in index and 'dp' in index:
#             # avoid negative log
#             row = row + 1
#             # if f'1_communication_cost' in index:
#             #     dp_KL_cc_1_time_list.append(np.mean(np.log(row)))
#             #     dp_KL_cc_1_time_se_list.append(np.std(np.log(row)))
#             # elif f'3_communication_cost' in index:
#             #     dp_KL_cc_3_time_list.append(np.mean(np.log(row)))
#             #     dp_KL_cc_3_time_se_list.append(np.std(np.log(row)))
#             # elif f'5_communication_cost' in index:
#             #     dp_KL_cc_5_time_list.append(np.mean(np.log(row)))
#             #     dp_KL_cc_5_time_se_list.append(np.std(np.log(row)))
#             # else:
#             dp_KL_time_list.append(np.mean(np.log(row)))
#             dp_KL_time_se_list.append(np.std(np.log(row)))
#         # elif 'QL' in index and 'Brute-force' in index:
#         #     Brute-force_QL_time_list.append(np.mean(row))
#         #     Brute-force_QL_time_se_list.append(np.std(row))
#         # elif 'QL' in index and 'Genetic' in index:
#         #     Genetic_QL_time_list.append(np.mean(row))
#         #     Genetic_QL_time_se_list.append(np.std(row))
#         # elif 'QL' in index and 'dp' in index:
#         #     dp_QL_time_list.append(np.mean(row))
#         #     dp_QL_time_se_list.append(np.std(row))
# elif 'mean' in index:
# if 'KL' in index and 'Brute-force' in index:
#     Brute-force_KL_mean_list.append(np.mean(row))
#     Brute-force_KL_se_list.append(np.std(row))
# elif 'KL' in index and 'Genetic' in index:
#     Genetic_KL_mean_list.append(np.mean(row))
#     Genetic_KL_se_list.append(np.std(row))
# elif 'KL' in index and 'dp' in index:
#     # if f'0.3_ratio_communication_cost' in index:
#     #     dp_KL_cc_03_mean_list.append(np.mean(row))
#     #     dp_KL_cc_03_se_list.append(np.std(row))
#     # elif f'0.5_ratio_communication_cost' in index:
#     #     dp_KL_cc_05_mean_list.append(np.mean(row))
#     #     dp_KL_cc_05_se_list.append(np.std(row))
#     # elif f'0.7_ratio_communication_cost' in index:
#     #     dp_KL_cc_07_mean_list.append(np.mean(row))
#     #     dp_KL_cc_07_se_list.append(np.std(row))
#     # elif f'0.9_ratio_communication_cost' in index:
#     #     dp_KL_cc_09_mean_list.append(np.mean(row))
#     #     dp_KL_cc_09_se_list.append(np.std(row))
#     # else:
#     dp_KL_mean_list.append(np.mean(row))
#     dp_KL_se_list.append(np.std(row))
# # elif 'QL' in index and 'Brute-force' in index:
# #     Brute-force_QL_mean_list.append(np.mean(row))
# #     Brute-force_QL_se_list.append(np.std(row))
# # elif 'QL' in index and 'Genetic' in index:
# #     Genetic_QL_mean_list.append(np.mean(row))
# #     Genetic_QL_se_list.append(np.std(row))
# # elif 'QL' in index and 'dp' in index:
# #     dp_QL_mean_list.append(np.mean(row))
# #     dp_QL_se_list.append(np.std(row))


# # elif 'Quadratic' in index and 'Brute-force' in index:
# # mean_list.append(np.mean(row))
# # elif 'mean' in index:
# #     # index_list.append(index)
# #     if 'KL' in index and 'Genetic' in index:
# #         Genetic_KL_mean_list.append(np.mean(row))
# #     elif 'Quadratic_Loss' in index and 'Genetic' in index:
# #         Genetic_Quadratic_Loss_mean_list.append(np.mean(row))
# #     elif 'KL' in index and 'dp' in index:
# #         dp_KL_mean_list.append(np.mean(row))
# #     elif 'Quadratic_Loss' in index and 'dp' in index:
# #         dp_Quadratic_Loss_mean_list.append(np.mean(row))
# #     elif 'KL' in index and 'Brute-force' in index:
# #         Brute-force_KL_mean_list.append(np.mean(row))
# #     # elif 'Quadratic' in index and 'Brute-force' in index:
# #     # mean_list.append(np.mean(row))
# # elif 'std' in index:
# #     if 'KL' in index and 'Genetic' in index:
# #         Genetic_KL_se_list.append(np.mean(row))
# #     elif 'Quadratic_Loss' in index and 'Genetic' in index:
# #         Genetic_Quadratic_Loss_se_list.append(np.mean(row))
# #     elif 'KL' in index and 'dp' in index:
# #         dp_KL_se_list.append(np.mean(row))
# #     elif 'Quadratic_Loss' in index and 'dp' in index:
# #         dp_Quadratic_Loss_se_list.append(np.mean(row))
# #     elif 'KL' in index and 'Brute-force' in index:
# #         Brute-force_KL_se_list.append(np.mean(row))
# # else:
# #     raise ValueError('wrong index')