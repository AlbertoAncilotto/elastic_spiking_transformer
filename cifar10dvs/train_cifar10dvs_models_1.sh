#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python train.py --batch-size 16 --epochs 200 --lr 1e-3 --num-classes 10 --T 16 --T_train 16 --workers 8 --wandb-project spikformer-hyperparam-search-xisps-cifar10dvs --log-wandb --use-xisps --xisps-elastic --device cuda:0 --depths 1 --embed-dims 256 --mlp-ratios 4 --num-heads 16 --sps-alpha 1.0 --wandb-run-name spikformer_x1_d1_e200 --output-dir ./logs/spikformer_x1_d1_e200 &
CUDA_VISIBLE_DEVICES=8 python train.py --batch-size 16 --epochs 200 --lr 1e-3 --num-classes 10 --T 16 --T_train 16 --workers 8 --wandb-project spikformer-hyperparam-search-xisps-cifar10dvs --log-wandb --use-xisps --xisps-elastic --device cuda:0 --depths 1 --embed-dims 256 --mlp-ratios 4 --num-heads 16 --sps-alpha 2.0 --wandb-run-name spikformer_x2_d1_e200 --output-dir ./logs/spikformer_x2_d1_e200 &

wait
