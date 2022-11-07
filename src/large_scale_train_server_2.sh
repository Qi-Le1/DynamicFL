#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet18_0.1_100_non-iid-d-0.3_dynamicfl_250_nonpre_0.8-0.2_2-1&
CUDA_VISIBLE_DEVICES="1" python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet18_0.1_100_non-iid-d-0.5_dynamicfl_250_nonpre_0.2-0.8_2-1&
CUDA_VISIBLE_DEVICES="2" python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet18_0.1_100_non-iid-d-0.5_dynamicfl_250_nonpre_0.5-0.5_2-1&
CUDA_VISIBLE_DEVICES="3" python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet18_0.1_100_non-iid-d-0.5_dynamicfl_250_nonpre_0.8-0.2_2-1&
CUDA_VISIBLE_DEVICES="0" python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet18_0.1_100_non-iid-d-1_dynamicfl_250_nonpre_0.2-0.8_2-1&
CUDA_VISIBLE_DEVICES="1" python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet18_0.1_100_non-iid-d-1_dynamicfl_250_nonpre_0.5-0.5_2-1&
CUDA_VISIBLE_DEVICES="2" python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet18_0.1_100_non-iid-d-1_dynamicfl_250_nonpre_0.8-0.2_2-1&
CUDA_VISIBLE_DEVICES="3" python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet18_0.1_100_non-iid-l-1_dynamicfl_250_nonpre_0.2-0.8_2-1
wait
CUDA_VISIBLE_DEVICES="0" python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet18_0.1_100_non-iid-d-1_dynamicsgd_250&
CUDA_VISIBLE_DEVICES="1" python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet18_0.1_100_non-iid-l-1_dynamicsgd_250&
CUDA_VISIBLE_DEVICES="2" python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet18_0.1_100_non-iid-l-2_dynamicsgd_250&
CUDA_VISIBLE_DEVICES="3" python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet18_0.1_100_non-iid-l-5_dynamicsgd_250&
CUDA_VISIBLE_DEVICES="0" python train_classifier_fl.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name CIFAR10_resnet18_0.1_100_non-iid-l-7_dynamicsgd_250&
wait
