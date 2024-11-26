import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
import numpy as np
from config import cfg, process_args
from data import (
    fetch_dataset, 
    split_dataset, 
    make_data_loader, 
    separate_dataset, 
    make_batchnorm_dataset, 
    make_batchnorm_stats
)
from metrics import Metric
from models.api import make_batchnorm
from utils.api import (
    save, 
    to_device, 
    process_command, 
    process_dataset,  
    resume, 
    collate
)

from models.api import (
    # CNN,
    create_model,
    make_batchnorm
)

from logger import make_logger

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


def runExperiment():
    result = resume(cfg['model_tag'], load_tag='best', resume_mode=1)
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    print('zzzffff', flush=True)
    
    train_logger = result['logger'] if 'logger' in result else None
    for key in train_logger.history:
        if 'KL' in key or 'QL' in key or 'dp_KL' in key or 'dp_QL' in key:
            temp = np.array(train_logger.history[key])
            print(cfg['model_tag'], key, np.mean(temp), flush=True)
        
    result = {'cfg': cfg, 'logger': {'train': train_logger}}
    save(result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def test(data_loader, model, metric, logger, epoch):
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
