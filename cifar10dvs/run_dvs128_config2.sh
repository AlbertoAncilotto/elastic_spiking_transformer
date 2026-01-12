#!/bin/bash
# DVS128 Gesture - Config 2: Small XiSPS (no elastic)

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset dvs128gesture \
    --data-path data/dvs128gesture/ \
    --num-classes 11 \
    --batch-size 16 \
    --epochs 200 \
    --lr 1e-3 \
    --T 16 \
    --T_train 16 \
    --workers 8 \
    --device cuda:0 \
    --depths 1 \
    --embed-dims 256 \
    --num-heads 16 \
    --mlp-ratios 4 \
    --sps-alpha 1.0 \
    --use-xisps \
    --log-wandb \
    --wandb-project spikformer-dvs128gesture \
    --wandb-run-name L1_D256_H16_M4_xisps_a1 \
    --output-dir ./logs/dvs128_L1_D256_xisps_a1
