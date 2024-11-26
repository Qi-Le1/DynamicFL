from numpy import number
from config import cfg
from typing import List
import collections

def process_ratio(client_ratio: str) -> List[float]:
    client_ratio = client_ratio.split('-')
    client_ratio = list(map(float, client_ratio))
    if sum(client_ratio) != 1:
        raise ValueError(
            'sum of ratio must be 1'
        )
    return client_ratio

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def process_number_of_freq_levels(
    number_of_freq_levels: str
) -> List[int]:
    number_of_freq_levels = number_of_freq_levels.split('-')
    res = []
    for item in number_of_freq_levels:
        if is_float(item):
            res.append(float(item))
        else:
            res.append(int(item))
    return res


def process_algo_parameters():
    '''
    process the algo parameter in the command

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''
    cfg['algo_mode'] = cfg['control']['algo_mode']
    if 'max_local_gradient_update' in cfg['control']:
        cfg['max_local_gradient_update'] = int(cfg['control']['max_local_gradient_update'])
    else:
        cfg['max_local_gradient_update'] = 'no_input'
        
    if 'local_epoch' in cfg['control']:
        cfg['local_epoch'] = int(cfg['control']['local_epoch'])

    if cfg['control']['algo_mode'] == 'dynamicsgd':
        pass
    elif cfg['control']['algo_mode'] == 'fedavg':
        pass
    elif cfg['control']['algo_mode'] == 'dynamicfl':

        if 'server_ratio' in cfg['control']:
            cfg['server_ratio'] = cfg['control']['server_ratio']
        else:
            cfg['server_ratio'] = '1-0'

        if 'client_ratio' in cfg['control']:
            cfg['client_ratio'] = cfg['control']['client_ratio']
        else:
            cfg['client_ratio'] = '0.5-0.5'

        if 'number_of_freq_levels' in cfg['control']:
            cfg['number_of_freq_levels'] = cfg['control']['number_of_freq_levels']
        else:
            cfg['number_of_freq_levels'] = '6-1'

        number_of_freq_levels = process_number_of_freq_levels(
            number_of_freq_levels=cfg['number_of_freq_levels']
        )

        client_ratio = process_ratio(
            client_ratio=cfg['client_ratio'],
        )

        server_ratio = process_ratio(
            client_ratio=cfg['server_ratio'],
        )
        
        if len(client_ratio) != len(number_of_freq_levels):
            raise ValueError(
                'length of client ratio is not equal to length of number_of_freq_levels'
            )
    
        if len(server_ratio) != len(number_of_freq_levels):
            raise ValueError(
                'length of server ratio is not equal to length of number_of_freq_levels'
            )
        
        client_ratio_to_number_of_freq_levels = collections.defaultdict(list)
        for i in range(len(client_ratio)):
            client_ratio_to_number_of_freq_levels[client_ratio[i]].append(number_of_freq_levels[i])
        cfg['client_ratio_to_number_of_freq_levels'] = client_ratio_to_number_of_freq_levels

        server_ratio_to_number_of_freq_levels = collections.defaultdict(list)
        for i in range(len(server_ratio)):
            server_ratio_to_number_of_freq_levels[server_ratio[i]].append(number_of_freq_levels[i])
        cfg['server_ratio_to_number_of_freq_levels'] = server_ratio_to_number_of_freq_levels
    return


def process_command():
    process_algo_parameters()
    cfg['save_interval'] = 50
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['algo_mode'] = cfg['control']['algo_mode']
    cfg['merge_gap'] = False

    if 'data_prep_norm_test' in cfg['control']:
        cfg['data_prep_norm_test'] = cfg['control']['data_prep_norm_test']
    else:
        cfg['data_prep_norm_test'] = 'bn'
    
    if 'norm' in cfg['control']:
        cfg['norm'] = cfg['control']['norm']
    else:
        cfg['norm'] = 'ln'

    if 'server_aggregation' in cfg['control']:
        cfg['server_aggregation'] = cfg['control']['server_aggregation']
    else:
        cfg['server_aggregation'] = 'WA'
    
    if 'no_training' in cfg['control']:
        cfg['no_training'] = cfg['control']['no_training']

    if 'data_prep_norm' in cfg['control']:
        cfg['data_prep_norm'] = cfg['control']['data_prep_norm']
    else:
        cfg['data_prep_norm'] = 'bn'

    cfg['dp_ensemble_times'] = 10
    data_shape = {'CIFAR10': [3, 32, 32], 'CIFAR100': [3, 32, 32], 'SVHN': [3, 32, 32], 'MNIST': [1, 28, 28], 'FEMNIST': [1, 28, 28]}
    cfg['data_shape'] = data_shape[cfg['data_name']]
    cfg['conv'] = {'hidden_size': [32, 64]}
    cfg['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}
    cfg['change_batch_size'] = False
    if 'change_batch_size' in cfg['control']:
        cfg['change_batch_size'] = True if int(cfg['control']['change_batch_size']) == 1 else False
    cfg['change_lr'] = False
    if 'change_lr' in cfg['control']:
        cfg['change_lr'] = True if int(cfg['control']['change_lr']) == 1 else False
    if 'reweight_sample' in cfg['control']:
        cfg['reweight_sample'] = True if int(cfg['control']['reweight_sample']) == 1 else False
    
    cfg['cal_communication_cost'] = False
    if 'cal_communication_cost' in cfg['control']:
        cfg['cal_communication_cost'] = True if int(cfg['control']['cal_communication_cost']) == 1 else False
    
    cfg['only_high_freq'] = False
    if 'only_high_freq' in cfg['control']:
        cfg['only_high_freq'] = True if int(cfg['control']['only_high_freq']) == 1 else False
    cfg['batch_size_threshold'] = 5
    cfg['threshold'] = 0.95
    cfg['alpha'] = 0.75
    cfg['feddyn_alpha'] = 0.1
    cfg['max_clip_norm'] = 10
    if 'num_clients' in cfg['control']:
        cfg['num_clients'] = int(cfg['control']['num_clients'])
        cfg['active_rate'] = float(cfg['control']['active_rate'])
        cfg['data_split_mode'] = cfg['control']['data_split_mode']
        
        cfg['gm'] = 0
        cfg['server'] = {}
        cfg['server']['shuffle'] = {'train': True, 'test': False}
        cfg['server']['batch_size'] = {'train': 250, 'test': 500}
        cfg['client'] = {}
        cfg['client']['shuffle'] = {'train': True, 'test': False}
        cfg['client']['batch_size'] = {'train': 10, 'test': 500}
        if 'train_batch_size' in cfg['control']:
            cfg['client']['batch_size']['train'] = int(cfg['control']['train_batch_size'])


        cfg['normalized_model_size'] = 1
        cfg['client']['optimizer_name'] = 'SGD'
        cfg['client']['lr'] = 3e-2
        if cfg['model_name'] == 'cnn':
            cfg['client']['lr'] = 1e-2
        cfg['client']['momentum'] = 0.9
        cfg['client']['weight_decay'] = 5e-4
        cfg['client']['nesterov'] = True
        cfg['client']['num_epochs'] = cfg['local_epoch']
        cfg['server']['num_epochs'] = 800
        cfg['server']['optimizer_name'] = 'SGD'
        cfg['server']['lr'] = 1
        cfg['server']['momentum'] = cfg['gm']
        cfg['server']['weight_decay'] = 0
        cfg['server']['nesterov'] = False
        cfg['server']['scheduler_name'] = 'CosineAnnealingLR'


        
    else:
        raise ValueError('no num_clients')

    
    cfg['upload_freq_level'] = {
        6: 1,
        5: 4,
        4: 16,
        3: 32,
        2.5: 64,
        2: 128,
        1: 256
    }

    print(f'cfg: {cfg}', flush=True)
    return
