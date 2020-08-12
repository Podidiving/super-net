#!/usr/bin/env bash

set -e

#CUDA_VISIBLE_DEVICES=0 python3.6 super_net/train_super_net.py -c configs/train_super_net.yml

CUDA_VISIBLE_DEVICES=0 python3.6 super_net/train_sub_net.py -c configs/train_sub_net_conv3x3_conv5x5.yml
CUDA_VISIBLE_DEVICES=0 python3.6 super_net/train_sub_net.py -c configs/train_sub_net_conv5x5_conv3x3.yml
CUDA_VISIBLE_DEVICES=0 python3.6 super_net/train_sub_net.py -c configs/train_sub_net_conv5x5_conv5x5.yml
CUDA_VISIBLE_DEVICES=0 python3.6 super_net/train_sub_net.py -c configs/train_sub_net_conv3x3_conv3x3.yml


CUDA_VISIBLE_DEVICES=0 python3.6 super_net/validate_sampled_nets.py -c configs/train_super_net.yml -d results.txt

CUDA_VISIBLE_DEVICES=0 python3.6 super_net/validate_sub_nets.py -c configs/train_sub_net_conv3x3_conv3x3.yml -d results.txt
CUDA_VISIBLE_DEVICES=0 python3.6 super_net/validate_sub_nets.py -c configs/train_sub_net_conv3x3_conv5x5.yml -d results.txt
CUDA_VISIBLE_DEVICES=0 python3.6 super_net/validate_sub_nets.py -c configs/train_sub_net_conv5x5_conv5x5.yml -d results.txt
CUDA_VISIBLE_DEVICES=0 python3.6 super_net/validate_sub_nets.py -c configs/train_sub_net_conv5x5_conv3x3.yml -d results.txt