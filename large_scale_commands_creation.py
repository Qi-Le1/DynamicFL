import argparse
import itertools

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--file', default=None, type=str)
parser.add_argument('--data', default=None, type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--log_interval', default=None, type=float)
args = vars(parser.parse_args())


def make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + init_seeds + world_size + num_experiments + resume_mode + log_interval + device + control_names 
    controls = list(itertools.product(*controls))
    # print('---controls', controls)
    return controls

'''
run: train or test
init_seed: 0
world_size: 1
num_experiments: 1
resume_mode: 0
log_interval: 0.25
num_gpus: 12
round: 1
experiment_step: 1
file: train_后面的, 例如privacy_joint
data: ML100K_ML1M_ML10M_ML20M

python create_commands_for_large_scale.py --run train --num_gpus 4 --round 1 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_decoder --data ML100K_ML1M_ML10M_ML20M

control:
  data_name: CIFAR10 (Name of the dataset)
  model_name: resnet9
  num_clients: 100
  data_split_mode: iid / non-iid
  algo_mode: dynamicsgd / fedavg / dynamicfl / dynamicsgd / fedgen / fenensemble / FedProx
  select_client_mode: fix / dynamic
  client_ratio: 0.2-0.3-0.5
  number_of_freq_levels: 2-3-4
  max_local_gradient_update: 50
  
# experiment
num_workers: 0
init_seed: 0
num_experiments: 1
log_interval: 0.25
device: cuda
resume_mode: 0
verbose: False
'''

# test dynamicsgd / fedavg / dynamicfl / dynamicsgd / fedgen / fenensemble / FedProx
# test num_clients: 100 / 300
# test datasets: CIFAR10 / CIFAR100 / FEMNIST
# test data_split_mode: iid / non-iid
# test max_local_gradient_update: 10 / 50 / 100
# for dynamicfl: test select_client_mode: fix / dynamic
# for dynamicfl: test client_ratio: 0.2-0.3-0.5
# for dynamicfl: test number_of_freq_levels: 2-3-4

def main():
    run = args['run']
    num_gpus = args['num_gpus']
    round = args['round']
    world_size = args['world_size']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiments = args['num_experiments']
    resume_mode = args['resume_mode']
    log_interval = args['log_interval']
    device = args['device']
    file = args['file']
    data = args['data'].split('_')
    
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    log_interval = [[log_interval]]
    device = [[device]]
    filename = '{}_{}'.format(run, file)
    
    if file == 'classifier_fl':
        controls = []
        script_name = [['{}_classifier_fl.py'.format(run)]]
        if 'CIFAR10' in data:
            control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                             ['5-1', '4-1', '6-1']]]
            
            CIFAR10_controls_1 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
            controls.extend(CIFAR10_controls_1)

            control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                             ['5-1', '4-1', '6-1']]]
            CIFAR10_controls_2 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
            controls.extend(CIFAR10_controls_2)

            control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4'], ['1-0'], 
                             ['5-1', '4-1']]]
            
            CIFAR10_controls_3 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
            controls.extend(CIFAR10_controls_3)

            control_name = [[['CIFAR10'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4'], 
                             ['5-1', '4-1']]]
            CIFAR10_controls_4 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
            controls.extend(CIFAR10_controls_4)

        if 'CIFAR100' in data:
            control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                             ['5-1', '4-1', '6-1']]]
            CIFAR10_controls_1 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
            controls.extend(CIFAR10_controls_1)

            control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                             ['5-1', '4-1', '6-1']]]
            CIFAR10_controls_2 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
            controls.extend(CIFAR10_controls_2)

            control_name = [[['CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4'], ['1-0'], 
                             ['5-1', '4-1']]]
            
            CIFAR10_controls_3 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
            controls.extend(CIFAR10_controls_3)

            control_name = [[['CIFAR100'], ['resnet18'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4'], 
                             ['5-1', '4-1']]]
            CIFAR10_controls_4 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
            controls.extend(CIFAR10_controls_4)
        if 'FEMNIST' in data:
            control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
                             ['5-1', '4-1', '6-1']]]
            CIFAR10_controls_1 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
            controls.extend(CIFAR10_controls_1)

            control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
                             ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
                             ['5-1', '4-1', '6-1']]]
            CIFAR10_controls_2 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
            controls.extend(CIFAR10_controls_2)
        

        # if 'CIFAR10' in data:
        #     control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                      ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
        #                      ['6-1']]]
            
        #     CIFAR10_controls_1 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
        #     controls.extend(CIFAR10_controls_1)

        #     control_name = [[['CIFAR10'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                      ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
        #                      ['6-1']]]
        #     CIFAR10_controls_2 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
        #     controls.extend(CIFAR10_controls_2)

        # if 'CIFAR100' in data:
        #     control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                      ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
        #                      ['6-1']]]
        #     CIFAR10_controls_1 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
        #     controls.extend(CIFAR10_controls_1)

        #     control_name = [[['CIFAR100'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                      ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
        #                      ['6-1']]]
        #     CIFAR10_controls_2 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
        #     controls.extend(CIFAR10_controls_2)
        # if 'FEMNIST' in data:
        #     control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                      ['dynamicfl'], ['5'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], ['1-0'], 
        #                      ['6-1']]]
        #     CIFAR10_controls_1 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
        #     controls.extend(CIFAR10_controls_1)

        #     control_name = [[['FEMNIST'], ['cnn'], ['0.1'], ['100'], ['non-iid-l-1', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], 
        #                      ['dynamicfl'], ['5'], ['1-0'], ['0.3-0.7', '0.6-0.4', '0.9-0.1'], 
        #                      ['6-1']]]
        #     CIFAR10_controls_2 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name)
        #     controls.extend(CIFAR10_controls_2)
    else:
        raise ValueError('Not valid file')

    print('%$%$$controls', controls)
    # s = '#!/bin/bash\n'
    s = ''
    k = 0

    # s_for_max = '#!/bin/bash\n'
    # k_for_max = 0
    
    for i in range(len(controls)):
        controls[i] = list(controls[i])

        # if 'max' in controls[i][-1]:
        #     s_for_max = s_for_max + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
        #         '--resume_mode {} --log_interval {} --device {} --control_name {}&\n'.format(gpu_ids[k % len(gpu_ids)], *controls[i])

        #     if k_for_max % round == round - 1:
        #         s_for_max = s_for_max[:-2] + '\nwait\n'
        #     k_for_max = k_for_max + 1
        #     continue
        
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
                '--resume_mode {} --log_interval {} --device {} --control_name {}&\n'.format(gpu_ids[k % len(gpu_ids)], *controls[i])
        if k % round == round - 1:
            s = s[:-2] + '\nwait\n'
        k = k + 1

    if s[-5:-1] != 'wait':
        s = s + 'wait\n'
    
    
    # if s_for_max != '#!/bin/bash\n' and s_for_max[-5:-1] != 'wait':
    #     s_for_max = s_for_max + 'wait\n'
    
    # print('@@@@@@@@@@', s)
    
    
    # run_file = open('./{}.sh'.format('large_scale_one_user_per_node_commands'), 'a')
    # run_file.write(s_for_max)
    # run_file.close()

    # server_total = 5
    if run == 'train':
        filename = 'train_server_commands'
        # for i in range(1, server_total):
        #     run_file = open('./{}.sh'.format(f'large_scale_train_server_{i}'), 'a')
        #     run_file.write('#!/bin/bash\n')
        #     run_file.close()

            # run_file.write(s)
            # run_file.close()
        
        run_file = open('./{}.txt'.format(f'large_scale_{filename}_temp'), 'a')
        run_file.write(s)
        run_file.close()
    elif run == 'test':
        filename = 'test_server_commands'
        # for i in range(1, server_total):
        #     run_file = open('./{}.sh'.format(f'large_scale_test_server_{i}'), 'a')
        #     run_file.write('#!/bin/bash\n')
        #     run_file.close()

            # run_file.write(s)
            # run_file.close()

        run_file = open('./{}.txt'.format(f'large_scale_{filename}_temp'), 'a')
        run_file.write(s)
        run_file.close()
    

    print(f'sss: {s}')
    run_file = open('./{}.sh'.format(f'large_scale_{filename}'), 'a')
    run_file.write(s)
    run_file.close()

    # run_file = open('./{}.txt'.format(f'large_scale_{run}'), 'a')
    # run_file.write(s_for_max)
    # run_file.close()

    

        

    # new_s = s.replace('CUDA_VISIBLE_DEVICES="0" ', '!')
    # new_s = new_s.replace('CUDA_VISIBLE_DEVICES="1" ', '!')
    # new_s = new_s.replace('CUDA_VISIBLE_DEVICES="2" ', '!')
    # new_s = new_s.replace('CUDA_VISIBLE_DEVICES="3" ', '!')
    # print('????', new_s)
    # run_file = open('./{}.sh'.format(f'pre_run_large_scale_{filename}'), 'a')
    # run_file.write(new_s)
    # run_file.close()

    # new_s_for_max = s_for_max.replace('CUDA_VISIBLE_DEVICES="0" ', '!')
    # new_s_for_max = new_s_for_max.replace('CUDA_VISIBLE_DEVICES="1" ', '!')
    # new_s_for_max = new_s_for_max.replace('CUDA_VISIBLE_DEVICES="2" ', '!')
    # new_s_for_max = new_s_for_max.replace('CUDA_VISIBLE_DEVICES="3" ', '!')
    # run_file = open('./{}.txt'.format(f'pre_run_large_scale_{run}'), 'a')
    # run_file.write(new_s_for_max)
    # run_file.close()

    # for i in range(4, 4):
    #     run_file = open('./{}.sh'.format(f'pre_run_large_scale_train_server_{i}'), 'a')
    #     run_file.write('#!/bin/bash\n')
    #     run_file.close()

    #     run_file = open('./{}.sh'.format(f'pre_run_large_scale_test_server_{i}'), 'a')
    #     run_file.write('#!/bin/bash\n')
    #     run_file.close()

    return


if __name__ == '__main__':
    main()
