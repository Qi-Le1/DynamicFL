from __future__ import annotations

import argparse
import datetime

import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from config import (
    cfg, 
    process_args
)

from data import (
    fetch_dataset, 
    split_dataset, 
    make_data_loader, 
    separate_dataset, 
    make_batchnorm_dataset, 
    make_batchnorm_stats
)

from metrics import Metric

from models.api import (
    CNN,
    create_model,
    make_batchnorm
)

from modules.client.api import (
    ClientDynamicFL,
    ClientFedAvg,
    ClientFedEnsemble,
    ClientFedGen,
    ClientFedProxy,
    ClientFedSgd
)


from modules.server.api import (
    ServerDynamicFL,
    ServerFedAvg,
    ServerFedEnsemble,
    ServerFedGen,
    ServerFedProxy,
    ServerFedSgd
)

from .utils.api import (
    create_model,
    save, 
    to_device, 
    process_command, 
    process_dataset,  
    resume, 
    collate
)

from _typing import (
    DatasetType,
    OptimizerType,
    DataLoaderType,
    ModelType,
    MetricType,
    LoggerType,
    ClientType,
    ServerType
)

from logger import Logger, make_logger

from optimizer.api import (
    create_optimizer,
    create_scheduler
)

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_command()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return

def make_communication(client_ids: list[int]) -> list[int]:
    if cfg['select_client_mode'] == 'fix':
        ratio_to_update_thresholds = Communication.cal_fix_update_thresholds(
            client_ids=client_ids,
            max_local_gradient_update=cfg['max_local_gradient_update'],
            ratio_to_number_of_uploads=cfg['ratio_to_number_of_uploads']
        )
        cfg['ratio_to_update_thresholds'] = ratio_to_update_thresholds
        client_to_update_threshold = Communication.distribute_fix_update_thresholds(
            client_ids=client_ids,
            max_local_gradient_update=cfg['max_local_gradient_update'],
            ratio_to_update_thresholds=ratio_to_update_thresholds
        )
    elif cfg['select_client_mode'] == 'dynamic':
        client_to_update_threshold = {}
        for id in client_ids:
            client_to_update_threshold[id] = None
    else:
        raise ValueError('select_client_mode must in fix or dynamic')
    return client_to_update_threshold

def create_clients(
    model: ModelType, 
    data_split: dict[str, dict[int, list[int]]],
) -> dict[int, ClientType]:
    '''
    Create corresponding server to cfg['algo_mode']
    
    Parameters
    ----------
    model: ModelType

    Returns
    -------
    ServerType
    '''
    if cfg['algo_mode'] == 'dynamicfl':
        return ClientDynamicFL.create_clients(
            model=model,
            data_split=data_split
        )
    elif cfg['algo_mode'] == 'fedavg':
        return ClientFedAvg.create_clients(
            model=model,
            data_split=data_split
        )
    elif cfg['algo_mode'] == 'fedensemble':
        return ClientFedEnsemble.create_clients(
            model=model,
            data_split=data_split
        )
    elif cfg['algo_mode'] == 'fedgen':
        return ClientFedGen.create_clients(
            model=model,
            data_split=data_split
        )
    elif cfg['algo_mode'] == 'fedproxy':
        return ClientFedProxy.create_clients(
            model=model,
            data_split=data_split
        )
    elif cfg['algo_mode'] == 'fedsgd':
        return ClientFedSgd.create_clients(
            model=model,
            data_split=data_split
        )
    else:
        raise ValueError('wrong algo model')
    
def create_server(model: ModelType) -> ServerType:
    '''
    Create corresponding server to cfg['algo_mode']
    
    Parameters
    ----------
    model: ModelType

    Returns
    -------
    ServerType
    '''
    if cfg['algo_mode'] == 'dynamicfl':
        return ServerDynamicFL(model)
    elif cfg['algo_mode'] == 'fedavg':
        return ServerFedAvg(model)
    elif cfg['algo_mode'] == 'fedensemble':
        return ServerFedEnsemble(model)
    elif cfg['algo_mode'] == 'fedgen':
        return ServerFedGen(model)
    elif cfg['algo_mode'] == 'fedproxy':
        return ServerFedProxy(model)
    elif cfg['algo_mode'] == 'fedsgd':
        return ServerFedSgd(model)
    else:
        raise ValueError('wrong algo model')



def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    data_loader = make_data_loader(dataset, 'global')
    model = create_model()
    optimizer = create_optimizer(model, 'local')
    scheduler = create_scheduler(optimizer, 'global')
    batchnorm_dataset = make_batchnorm_dataset(dataset['train'])
    data_split = split_dataset(dataset, cfg['num_clients'], cfg['data_split_mode'])
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})

    result = resume(cfg['model_tag'], resume_mode=cfg['resume_mode'])
    if result is None:
        last_global_epoch = 1
        clients = create_clients(model, data_split)
        server = create_server(model, clients)
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_global_epoch = result['global_epoch']
        data_split = result['data_split']
        server = result['server']
        clients = result['clients']
        optimizer.load_state_dict(result['optimizer_state_dict'])
        scheduler.load_state_dict(result['scheduler_state_dict'])
        logger = result['logger']

    for global_epoch in range(last_global_epoch, cfg['global']['num_epochs'] + 1):
        server.train_clients(
            dataset=dataset, 
            optimizer=optimizer, 
            metric=metric, 
            logger=logger, 
            global_epoch=global_epoch
        )
        server.update_global_model(
            clients=clients, 
            global_epoch=global_epoch
        )
        scheduler.step()
        model.load_state_dict(server.model_state_dict)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        test(
            data_loader=data_loader['test'], 
            model=test_model, 
            metric=metric, 
            logge=logger, 
            global_epoch=global_epoch
        )

        result = {
            'cfg': cfg, 
            'global_epoch': global_epoch + 1, 
            'server': server, 
            'clients': clients,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
            'data_split': data_split, 
            'logger': logger,
        }
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))

        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def train_clients(
    dataset: DatasetType, 
    server: ServerType, 
    optimizer: OptimizerType, 
    metric: MetricType, 
    logger: LoggerType, 
    global_epoch: int
) -> None:

    server.train_clients(
        dataset=dataset, 
        optimizer=optimizer, 
        metric=metric, 
        logger=logger, 
        global_epoch=global_epoch
    )
    return


def test(
    data_loader: DataLoaderType, 
    model: ModelType, 
    metric: MetricType, 
    logger: LoggerType, 
    epoch: int
) -> None:

    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return

def cal_grad_interval(
    local_grad_u: int,
    update_threshold: int
) -> int:
    '''
    Notes
    -----
    grad_interval = cfg['max_local_gradient_update'] - local_grad_u + 1.
    Ex. 160 - 151 = 9, but we actually need to train 9 + 1 = 10 times
    '''
    grad_interval = update_threshold
    if local_grad_u + update_threshold < cfg['max_local_gradient_update'] \
        and local_grad_u + 2 * update_threshold > cfg['max_local_gradient_update']:
        grad_interval = cfg['max_local_gradient_update'] - local_grad_u + 1
    return grad_interval

def is_local_gradient_update_valid(
    local_grad_u: int,
    update_threshold: int
) -> bool:
    '''
    Notes
    -----
    If local_grad_u == 1, we need to start training in all situations.
    If local_grad_u % update_threshold == 0, second situation that we
    need to enter training
    '''
    if local_grad_u == 1:
        return True
    elif local_grad_u % update_threshold == 0:
        return True
    
    return False

def update_update_threshold(
    clients: dict[int, ClientType],
    client_ids: list[int]
): 
    if cfg['select_client_mode'] == 'fix':
        pass
    elif cfg['select_client_mode'] == 'dynamic':
        Communication.distribute_dynamic_update_thresholds(
            clients=clients,
            client_ids=client_ids,
            ratio_to_update_thresholds=cfg['ratio_to_update_thresholds']
        )
    else:
        raise ValueError('select_client_mode wrong')

def combine_dataset(
    num_active_clients: int,
    clients: dict[int, ClientType],
    client_ids: list[int],
    dataset: DatasetType
) -> DatasetType:  
    '''
    combine the datapoint index for selected clients
    and return the dataset
    '''
    combined_datapoint_idx = []
    for i in range(num_active_clients):
        m = client_ids[i]
        combined_datapoint_idx = combined_datapoint_idx + clients[m].data_split['train']

    # dataset: DatasetType
    dataset = separate_dataset(dataset, combined_datapoint_idx)
    return dataset


# def train_clients(
#     dataset: DatasetType, 
#     server: ServerType, 
#     server_local_grad_update: ServerType,
#     optimizer: OptimizerType, 
#     metric: MetricType, 
#     logger: LoggerType, 
#     global_epoch: int
# ) -> None:

#     server.train_clients(
#         dataset=dataset, 
#         optimizer=optimizer, 
#         metric=metric, 
#         logger=logger, 
#         global_epoch=global_epoch
#     )

#     logger.safe(True)
#     num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
#     client_ids = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
#     for i in range(num_active_clients):
#         clients[client_ids[i]].active = True
#     server.distribute(clients)
#     num_active_clients = len(client_ids)
#     start_time = time.time()
#     lr = optimizer.param_groups[0]['lr']

#     for i in range(num_active_clients):
#         m = client_ids[i]
#         dataset_m = separate_dataset(dataset, clients[m].data_split['train'])
#         if dataset_m is not None:
#             clients[m].active = True
#         else:
#             clients[m].active = False

#     if cfg['algo_mode'] == 'fedsgd':
#         # combine datasets of the active clients,
#         # so that we can only train 1 model for fedsgd.
#         # Decrease computation time
#         combinedDataset = combine_dataset(
#             num_active_clients=num_active_clients,
#             clients=clients,
#             client_ids=client_ids,
#         )
        
#         client = Client(
#             client_id=0, 
#             model=create_model(), 
#             data_split={
#                 'train': None, 
#                 'test': None
#             },
#             update_threshold=cfg['max_local_gradient_update'][m]
#         )

#         client[0].train(
#             dataset=combinedDataset, 
#             lr=lr, 
#             metric=metric, 
#             logger=logger,
#             grad_interval=grad_interval
#         )
#     elif cfg['algo_mode'] == 'fedavg':
#         pass
#     elif cfg['algo_mode'] == 'fedprox':
#         pass
#     elif cfg['algo_mode'] == 'fedensemble':
#         pass
#     elif cfg['algo_mode'] == 'fedgen':
#         pass
#     elif cfg['algo_mode'] == 'dynamicfl':
#         # handle fix and dynamic selecting cliend mode
#         update_update_threshold(
#             client_id=client_id,
#             clients=clients
#         )

#         for local_grad_u in range(1, cfg['max_local_gradient_update'] + 1):
#             server_local_grad_update.distribute(
#                 local_grad_u=local_grad_u, 
#                 clients=clients
#             )

#             for i in range(num_active_clients):
#                 m = client_ids[i]
#                 dataset_m = separate_dataset(dataset, clients[m].data_split['train'])

#                 if is_local_gradient_update_valid(
#                     local_grad_u=local_grad_u,
#                     update_threshold=clients[m].update_threshold
#                 ):
#                     # 判断是否进入, 因为一次训练Interval个local gradient
#                     grad_interval = cal_grad_interval(
#                         local_grad_u=local_grad_u,
#                         update_threshold=clients[m].update_threshold
#                     )

#                     clients[m].train(
#                         dataset=dataset_m, 
#                         lr=lr, 
#                         metric=metric, 
#                         logger=logger,
#                         grad_interval=grad_interval
#                     )

#                     server_local_grad_update.upload(
#                         # subtract 1
#                         target_grad_u=local_grad_u+grad_interval-1,
#                         cur_client_id=m,
#                         clients=clients[m]
#                     )

#             server_local_grad_update.update(local_grad_u=local_grad_u)
            
#     return





if __name__ == "__main__":
    main()
