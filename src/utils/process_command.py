from numpy import number
from config import cfg
from typing import List

def process_client_ratio(ratio: str) -> List[float]:
    ratio = ratio.split('-')
    ratio = list(map(float, ratio))
    if sum(ratio) != 1:
        raise ValueError(
            'sum of ratio must be 1'
        )
    return ratio

def process_number_of_uploads(
    number_of_uploads: str,
    max_local_gradient_update: int
) -> List[int]:
    number_of_uploads = number_of_uploads.split('-')
    number_of_uploads = list(map(int, number_of_uploads))
    number_of_uploads = [min(i, max_local_gradient_update) for i in number_of_uploads]
    return number_of_uploads


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
    cfg['client_ratio'] = cfg['control']['client_ratio']
    cfg['number_of_uploads'] = cfg['control']['number_of_uploads']
    cfg['max_local_gradient_update'] = int(cfg['control']['max_local_gradient_update'])

    if cfg['control']['algo_mode'] == 'fedsgd':
        pass
    elif cfg['control']['algo_mode'] == 'fedavg':
        cfg['max_local_gradient_update'] = 1
    elif cfg['control']['algo_mode'] == 'dynamicfl':
        client_ratio = process_client_ratio(
            client_ratio=cfg['control']['client_ratio']
        )
        number_of_uploads = process_number_of_uploads(
            number_of_uploads=cfg['control']['number_of_uploads'],
            max_local_gradient_update=cfg['max_local_gradient_update']
        )
        if len(cfg['client_ratio']) != len(cfg['number_of_uploads']):
            raise ValueError(
                'length of ratio is not equal to length of number_of_uploads'
            )
        
        client_ratio_to_number_of_uploads = {}
        for i in range(len(client_ratio)):
            client_ratio_to_number_of_uploads[client_ratio[i]] = number_of_uploads[i]
        cfg['client_ratio_to_number_of_uploads'] = client_ratio_to_number_of_uploads
    return


def process_command():
    process_algo_parameters()
    cfg['select_client_mode'] = cfg['control']['select_client_mode']
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['algo_mode'] = cfg['control']['algo_mode']
    data_shape = {'CIFAR10': [3, 32, 32], 'CIFAR100': [3, 32, 32], 'SVHN': [3, 32, 32]}
    cfg['data_shape'] = data_shape[cfg['data_name']]
    cfg['conv'] = {'hidden_size': [32, 64]}
    cfg['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
    # cfg['resnet9'] = {'hidden_size': [64, 128]}
    cfg['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}
    cfg['threshold'] = 0.95
    cfg['alpha'] = 0.75
    if 'num_clients' in cfg['control']:
        cfg['num_clients'] = int(cfg['control']['num_clients'])
        # cfg['active_rate'] = float(cfg['control']['active_rate'])
        cfg['active_rate'] = 0.1
        cfg['data_split_mode'] = cfg['control']['data_split_mode']
        # cfg['diff_val'] = float(cfg['control']['diff_val'])
        cfg['local_epoch'] = 5
        cfg['gm'] = 0
        cfg['server'] = {}
        cfg['server']['shuffle'] = {'train': True, 'test': False}
        cfg['server']['batch_size'] = {'train': 250, 'test': 500}
        cfg['client'] = {}
        cfg['client']['shuffle'] = {'train': True, 'test': False}
        if cfg['num_clients'] > 10:
            cfg['client']['batch_size'] = {'train': 10, 'test': 500}
        elif cfg['num_clients'] > 1:
            cfg['client']['batch_size'] = {'train': 100, 'test': 500}
        else:
            cfg['client']['batch_size'] = {'train': 250, 'test': 500}

        cfg['client']['optimizer_name'] = 'SGD'
        cfg['client']['lr'] = 3e-2
        cfg['client']['momentum'] = 0.9
        cfg['client']['weight_decay'] = 5e-4
        cfg['client']['nesterov'] = True
        cfg['client']['num_epochs'] = cfg['local_epoch']

        cfg['server']['batch_size'] = {'train': 250, 'test': 500}
        cfg['server']['shuffle'] = {'train': True, 'test': False}
        if cfg['num_clients'] > 10:
            cfg['server']['num_epochs'] = 800
        else:
            cfg['server']['num_epochs'] = 800
        cfg['server']['optimizer_name'] = 'SGD'
        cfg['server']['lr'] = 1
        cfg['server']['momentum'] = cfg['gm']
        cfg['server']['weight_decay'] = 0
        cfg['server']['nesterov'] = False
        cfg['server']['scheduler_name'] = 'CosineAnnealingLR'
    else:
        raise ValueError('no num_clients')
        # model_name = cfg['model_name']
        # cfg[model_name]['shuffle'] = {'train': True, 'test': False}
        # cfg[model_name]['optimizer_name'] = 'SGD'
        # cfg[model_name]['lr'] = 3e-2
        # cfg[model_name]['momentum'] = 0.9
        # cfg[model_name]['weight_decay'] = 5e-4
        # cfg[model_name]['nesterov'] = True
        # cfg[model_name]['scheduler_name'] = 'CosineAnnealingLR'
        # cfg[model_name]['num_epochs'] = 400
        # cfg[model_name]['batch_size'] = {'train': 250, 'test': 500}
    return
