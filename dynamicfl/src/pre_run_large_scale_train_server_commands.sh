#!/bin/bash
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_iid_dynamicfl_fix_0.2-0.8_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_iid_dynamicfl_fix_0.5-0.5_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_iid_dynamicfl_fix_0.8-0.2_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-d-0.1_dynamicfl_fix_0.2-0.8_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-d-0.1_dynamicfl_fix_0.5-0.5_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-d-0.1_dynamicfl_fix_0.8-0.2_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-d-1_dynamicfl_fix_0.2-0.8_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-d-1_dynamicfl_fix_0.5-0.5_2-1_250
wait
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-d-1_dynamicfl_fix_0.8-0.2_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-d-5_dynamicfl_fix_0.2-0.8_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-d-5_dynamicfl_fix_0.5-0.5_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-d-5_dynamicfl_fix_0.8-0.2_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-2_dynamicfl_fix_0.2-0.8_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-2_dynamicfl_fix_0.5-0.5_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-2_dynamicfl_fix_0.8-0.2_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-5_dynamicfl_fix_0.2-0.8_2-1_250
wait
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-5_dynamicfl_fix_0.5-0.5_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-5_dynamicfl_fix_0.8-0.2_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-7_dynamicfl_fix_0.2-0.8_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-7_dynamicfl_fix_0.5-0.5_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-7_dynamicfl_fix_0.8-0.2_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_iid_fedavg_fix_0.8-0.2_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-d-0.1_fedavg_fix_0.8-0.2_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-d-1_fedavg_fix_0.8-0.2_2-1_250
wait
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-d-5_fedavg_fix_0.8-0.2_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-2_fedavg_fix_0.8-0.2_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-5_fedavg_fix_0.8-0.2_2-1_250&
!python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet9_100_non-iid-l-7_fedavg_fix_0.8-0.2_2-1_250&
wait
