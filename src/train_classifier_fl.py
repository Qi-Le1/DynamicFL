from __future__ import annotations

import argparse
import datetime

import models
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

from modules.api import (
    Client,
    Communication
)

from modules.server.api import (
    ServerIteration,
    ServerLocalGradUpdate
)

from utils import (
    save, 
    to_device, 
    process_control, 
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
    make_optimizer,
    make_scheduler
)

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return

def make_model():
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
    return model

def make_communication(client_id: list[int]) -> list[int]:
    if cfg['select_client_mode'] == 'fix':
        ratio_to_update_thresholds = Communication.cal_fix_update_thresholds(
            client_id=client_id,
            max_local_gradient_update=cfg['max_local_gradient_update'],
            ratio_to_number_of_uploads=cfg['ratio_to_number_of_uploads']
        )
        cfg['ratio_to_update_thresholds'] = ratio_to_update_thresholds
        client_to_update_threshold = Communication.distribute_fix_update_thresholds(
            client_id=client_id,
            max_local_gradient_update=cfg['max_local_gradient_update'],
            ratio_to_update_thresholds=ratio_to_update_thresholds
        )
    elif cfg['select_client_mode'] == 'dynamic':
        client_to_update_threshold = {}
        for id in client_id:
            client_to_update_threshold[id] = None
    else:
        raise ValueError('select_client_mode must in fix or dynamic')
    return client_to_update_threshold

def make_server_iteration(model: ModelType) -> ServerType:
    server = ServerIteration(model)
    return server

def make_server_local_grad_update(model: ModelType) -> ServerType:
    server = ServerLocalGradUpdate(model)
    return server

def make_client(
    model: ModelType, 
    data_split: dict[str, dict[int, list[int]]],
) -> dict[int, ClientType]:
    client_id = torch.arange(cfg['num_clients'])
    client = [None for _ in range(cfg['num_clients'])]
    client_to_update_threshold = make_communication(client_id=client_id)
    for m in range(len(client)):
        client[m] = Client(
            client_id=client_id[m], 
            model=model, 
            data_split={
                'train': data_split['train'][m], 
                'test': data_split['test'][m]
            },
            update_threshold=client_to_update_threshold[m]
        )
    return client


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    data_loader = make_data_loader(dataset, 'global')
    model = make_model()
    optimizer = make_optimizer(model, 'local')
    scheduler = make_scheduler(optimizer, 'global')
    batchnorm_dataset = make_batchnorm_dataset(dataset['train'])
    data_split = split_dataset(dataset, cfg['num_clients'], cfg['data_split_mode'])
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    result = resume(cfg['model_tag'], resume_mode=cfg['resume_mode'])
    if result is None:
        last_epoch = 1
        server_iteration = make_server_iteration(model)
        server_local_grad_update = make_server_local_grad_update(model)
        client = make_client(model, data_split)
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = result['epoch']
        data_split = result['data_split']
        server_iteration = result['server_iteration']
        client = result['client']
        optimizer.load_state_dict(result['optimizer_state_dict'])
        scheduler.load_state_dict(result['scheduler_state_dict'])
        logger = result['logger']
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        train_client(
            dataset=dataset['train'], 
            server_iteration=server_iteration, 
            client=client, 
            optimizer=optimizer, 
            metric=metric, 
            logger=logger, 
            epoch=epoch
        )
        server_iteration.update(
            client=client, 
            epoch=epoch
        )
        scheduler.step()
        model.load_state_dict(server_iteration.model_state_dict)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        test(
            data_loader=data_loader['test'], 
            model=test_model, 
            metric=metric, 
            logge=logger, 
            epoch=epoch
        )
        result = {
            'cfg': cfg, 
            'epoch': epoch + 1, 
            'server_iteration': server_iteration, 
            'client': client,
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
    client: dict[int, ClientType],
    client_id: list[int]
): 
    if cfg['select_client_mode'] == 'fix':
        pass
    elif cfg['select_client_mode'] == 'dynamic':
        Communication.distribute_dynamic_update_thresholds(
            client=client,
            client_id=client_id,
            ratio_to_update_thresholds=cfg['ratio_to_update_thresholds']
        )
    else:
        raise ValueError('select_client_mode wrong')

def train_client(
    dataset: DatasetType, 
    server_iteration: ServerType, 
    server_local_grad_update: ServerType,
    client: dict[int, ClientType], 
    optimizer: OptimizerType, 
    metric: MetricType, 
    logger: LoggerType, 
    epoch: int
) -> None:

    logger.safe(True)
    num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
    for i in range(num_active_clients):
        client[client_id[i]].active = True
    server_iteration.distribute(client)
    num_active_clients = len(client_id)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']

    for i in range(num_active_clients):
        m = client_id[i]
        dataset_m = separate_dataset(dataset, client[m].data_split['train'])
        if dataset_m is not None:
            client[m].active = True
        else:
            client[m].active = False

    update_update_threshold(
        client_id=client_id,
        client=client
    )

    for local_grad_u in range(1, cfg['max_local_gradient_update'] + 1):
        server_local_grad_update.distribute(
            local_grad_u=local_grad_u, 
            client=client
        )

        for i in range(num_active_clients):
            m = client_id[i]
            dataset_m = separate_dataset(dataset, client[m].data_split['train'])

            if is_local_gradient_update_valid(
                local_grad_u=local_grad_u,
                update_threshold=client[m].update_threshold
            ):
                # 判断是否进入, 因为一次训练Interval个local gradient
                grad_interval = cal_grad_interval(
                    local_grad_u=local_grad_u,
                    update_threshold=client[m].update_threshold
                )

                client[m].train(
                    dataset=dataset_m, 
                    lr=lr, 
                    metric=metric, 
                    logger=logger,
                    grad_interval=grad_interval
                )

                server_local_grad_update.upload(
                    # subtract 1
                    target_grad_u=local_grad_u+grad_interval-1,
                    cur_client_id=m,
                    client=client[m]
                )

        server_local_grad_update.update(local_grad_u=local_grad_u)
            
            
    for i in range(num_active_clients):            
        if i % int((num_active_clients * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * _time * num_active_clients))
            exp_progress = 100. * i / num_active_clients
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                                'Train Epoch (C): {}({:.0f}%)'.format(epoch, exp_progress),
                                'Learning rate: {:.6f}'.format(lr),
                                'ID: {}({}/{})'.format(client_id[i], i + 1, num_active_clients),
                                'Epoch Finished Time: {}'.format(epoch_finished_time),
                                'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
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


if __name__ == "__main__":
    main()
