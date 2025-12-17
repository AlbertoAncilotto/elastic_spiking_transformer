#!/bin/bash

set -e

PROJECT="spikformer_ehwgesture-sps"

# ---------------- GPU 5 ----------------
(
  CUDA_VISIBLE_DEVICES=5 python train.py \
    --use-xisps --xisps-elastic --sps-alpha 1.0 \
    --device cuda:0 \
    --wandb-project $PROJECT \
    --wandb-run-name "xisps-elastic_alpha1.0_gpu5"

  CUDA_VISIBLE_DEVICES=5 python train.py \
    --use-xisps --xisps-elastic --sps-alpha 2.0 \
    --device cuda:0 \
    --wandb-project $PROJECT \
    --wandb-run-name "xisps-elastic_alpha2.0_gpu5"
) &

# ---------------- GPU 6 ----------------
(
  CUDA_VISIBLE_DEVICES=6 python train.py \
    --use-xisps --sps-alpha 1.0 \
    --device cuda:0 \
    --wandb-project $PROJECT \
    --wandb-run-name "xisps_alpha1.0_gpu6"

  CUDA_VISIBLE_DEVICES=6 python train.py \
    --use-xisps --sps-alpha 2.0 \
    --device cuda:0 \
    --wandb-project $PROJECT \
    --wandb-run-name "xisps_alpha2.0_gpu6"
) &

# ---------------- GPU 9 ----------------
(
  CUDA_VISIBLE_DEVICES=9 python train.py \
    --sps-alpha 0.5 \
    --device cuda:0 \
    --wandb-project $PROJECT \
    --wandb-run-name "sps_alpha0.5_gpu9"

  CUDA_VISIBLE_DEVICES=9 python train.py \
    --sps-alpha 1.0 \
    --device cuda:0 \
    --wandb-project $PROJECT \
    --wandb-run-name "sps_alpha1.0_gpu9"
) &

# Wait for all GPU blocks to finish
wait

echo "All experiments completed."
