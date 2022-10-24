#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python test_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-d-5_fedavg_fix_0.8-0.2_2-1_250&
CUDA_VISIBLE_DEVICES="1" python test_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-2_fedavg_fix_0.8-0.2_2-1_250&
CUDA_VISIBLE_DEVICES="2" python test_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-5_fedavg_fix_0.8-0.2_2-1_250&
CUDA_VISIBLE_DEVICES="3" python test_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-7_fedavg_fix_0.8-0.2_2-1_250&
wait
