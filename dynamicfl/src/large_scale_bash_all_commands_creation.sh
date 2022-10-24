#!/bin/bash

# run joint
python3 large_scale_commands_creation.py --run train --device cuda --num_gpus 4 --round 8 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file classifier_fl --data CIFAR10
python3 large_scale_commands_creation.py --run test --device cuda --num_gpus 4 --round 8 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file classifier_fl --data CIFAR10

# federate => average all
# python3 large_scale_commands_creation.py --run train --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file classifier_fl --data CIFAR10_CIFAR100_FMNIST
# python3 large_scale_commands_creation.py --run test --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file classifier_fl --data CIFAR10_CIFAR100_FMNIST

# # federate => average decoder
# python3 large_scale_commands_creation.py --run train --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file classifier_fl --data CIFAR10_CIFAR100_FMNIST
# python3 large_scale_commands_creation.py --run test --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file classifier_fl --data CIFAR10_CIFAR100_FMNIST

python3 large_scale_reassign_command.py