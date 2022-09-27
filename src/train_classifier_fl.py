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
    # CNN,
    create_model,
    make_batchnorm
)

from modules.client.api import (
    ClientDynamicFL,
    ClientFedAvg,
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

from utils.api import (
    save, 
    to_device, 
    process_command, 
    process_dataset,  
    resume, 
    collate
)

from models.api import create_model

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
        print(f"Experiment: {cfg['model_tag']}")
        runExperiment()
    return

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
        return ClientFedAvg.create_clients(
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
    
def create_server(
    model: ModelType,
    clients: dict[int, ClientType],
    dataset: DatasetType
) -> ServerType:
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
        return ServerDynamicFL(model, clients)
    elif cfg['algo_mode'] == 'fedavg':
        return ServerFedAvg(model, clients, dataset)
    elif cfg['algo_mode'] == 'fedensemble':
        return ServerFedEnsemble(model, clients)
    elif cfg['algo_mode'] == 'fedgen':
        return ServerFedGen(model, clients)
    elif cfg['algo_mode'] == 'fedproxy':
        return ServerFedProxy(model, clients)
    elif cfg['algo_mode'] == 'fedsgd':
        return ServerFedSgd(model, clients)
    else:
        raise ValueError('wrong algo model')



def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    # data_loader = make_data_loader(dataset, 'global')
    model = create_model()
    optimizer = create_optimizer(model, 'client')
    scheduler = create_scheduler(optimizer, 'server')
    # batchnorm_dataset = make_batchnorm_dataset(dataset['train'])
    data_split = split_dataset(dataset, cfg['num_clients'], cfg['data_split_mode'])
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})

    result = resume(cfg['model_tag'], resume_mode=cfg['resume_mode'])
    if result is None:
        last_global_epoch = 1
        clients = create_clients(
            model=model, 
            data_split=data_split
        )
        server = create_server(
            model=model, 
            clients=clients, 
            dataset=dataset['train']
        )
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_global_epoch = result['global_epoch']
        data_split = result['data_split']
        server = result['server']
        clients = result['clients']
        optimizer.load_state_dict(result['optimizer_state_dict'])
        scheduler.load_state_dict(result['scheduler_state_dict'])
        logger = result['logger']

    for global_epoch in range(last_global_epoch, cfg['server']['num_epochs'] + 1):
        
        server.train(
            dataset=dataset['train'], 
            optimizer=optimizer, 
            metric=metric, 
            logger=logger, 
            global_epoch=global_epoch
        )
        scheduler.step()

        server.evaluate_trained_model(
            dataset=dataset['test'],
            logger=logger,
            metric=metric,
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

if __name__ == "__main__":
    main()
