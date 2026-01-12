#!/bin/bash
# DVS128 Gesture training script with 8 configurations
# Testing different hyperparameters: network sizes, xisps, alpha, elastic
# DVS128 Gesture dataset has 11 classes
#
# Usage: ./train_dvs128_models.sh
# Runs 8 experiments in parallel: 2 each on GPU 0, 2, 8, 9

# Common parameters
DATASET="dvs128gesture"
NUM_CLASSES=11
DATA_PATH="data/dvs128gesture/"
EPOCHS=300
LR=1e-3
BATCH_SIZE=16
WORKERS=32
T=16
T_TRAIN=16
WANDB_PROJECT="spikformer-dvs128gesture"

echo "=========================================="
echo "DVS128 Gesture Training - 8 Configurations"
echo "=========================================="
echo "Dataset: $DATASET (${NUM_CLASSES} classes)"
echo "Data path: $DATA_PATH"
echo ""

# ============================================================
# GPU 0 - 2 experiments
# ============================================================

# Config 1: Small network, no XiSPS (baseline)
# L1 D256 H16 M4 - minimal baseline
echo "Starting Config 1 on GPU 0: Small baseline (no XiSPS)"
CUDA_VISIBLE_DEVICES=9 python train.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --num-classes $NUM_CLASSES \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --T $T \
    --T_train $T_TRAIN \
    --workers $WORKERS \
    --device cuda:0 \
    --depths 1 \
    --embed-dims 256 \
    --num-heads 16 \
    --mlp-ratios 4 \
    --sps-alpha 1.0 \
    --log-wandb \
    --wandb-project $WANDB_PROJECT \
    --wandb-run-name "L1_D256_H16_M4_baseline" \
    --output-dir ./logs/dvs128_L1_D256_baseline &

# Config 2: Small network with XiSPS (no elastic)
# L1 D256 H16 M4, alpha=1.0, XiSPS
echo "Starting Config 2 on GPU 0: Small XiSPS (no elastic)"
CUDA_VISIBLE_DEVICES=9 python train.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --num-classes $NUM_CLASSES \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --T $T \
    --T_train $T_TRAIN \
    --workers $WORKERS \
    --device cuda:0 \
    --depths 1 \
    --embed-dims 160 \
    --num-heads 16 \
    --mlp-ratios 4 \
    --sps-alpha 1.0 \
    --use-xisps \
    --log-wandb \
    --wandb-project $WANDB_PROJECT \
    --wandb-run-name "L1_D160_H16_M4_xisps_a1" \
    --output-dir ./logs/dvs128_L1_D160_xisps_a1 &

# ============================================================
# GPU 2 - 2 experiments
# ============================================================

# Config 3: Small network with XiSPS elastic
# L1 D256 H16 M4, alpha=1.0, XiSPS elastic
echo "Starting Config 3 on GPU 2: Small XiSPS elastic"
CUDA_VISIBLE_DEVICES=2 python train.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --num-classes $NUM_CLASSES \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --T $T \
    --T_train $T_TRAIN \
    --workers $WORKERS \
    --device cuda:0 \
    --depths 1 \
    --embed-dims 256 \
    --num-heads 16 \
    --mlp-ratios 4 \
    --sps-alpha 1.0 \
    --use-xisps \
    --xisps-elastic \
    --log-wandb \
    --wandb-project $WANDB_PROJECT \
    --wandb-run-name "L1_D256_H16_M4_xisps_elastic_a1" \
    --output-dir ./logs/dvs128_L1_D256_xisps_elastic_a1 &

# Config 4: Small network with XiSPS elastic, alpha=2.0 (uses XiSPSv2)
# L1 D256 H16 M4, alpha=2.0, XiSPS elastic
echo "Starting Config 4 on GPU 2: Small XiSPSv2 elastic (alpha=2.0)"
CUDA_VISIBLE_DEVICES=2 python train.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --num-classes $NUM_CLASSES \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --T $T \
    --T_train $T_TRAIN \
    --workers $WORKERS \
    --device cuda:0 \
    --depths 1 \
    --embed-dims 256 \
    --num-heads 16 \
    --mlp-ratios 4 \
    --sps-alpha 2.0 \
    --use-xisps \
    --xisps-elastic \
    --log-wandb \
    --wandb-project $WANDB_PROJECT \
    --wandb-run-name "L1_D256_H16_M4_xispsv2_elastic_a2" \
    --output-dir ./logs/dvs128_L1_D256_xispsv2_elastic_a2 &

# ============================================================
# GPU 8 - 2 experiments
# ============================================================

# Config 5: Larger network, no XiSPS (baseline)
# L2 D384 H16 M4 - larger baseline
echo "Starting Config 5 on GPU 8: Large baseline (no XiSPS)"
CUDA_VISIBLE_DEVICES=8 python train.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --num-classes $NUM_CLASSES \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --T $T \
    --T_train $T_TRAIN \
    --workers $WORKERS \
    --device cuda:0 \
    --depths 2 \
    --embed-dims 384 \
    --num-heads 16 \
    --mlp-ratios 4 \
    --sps-alpha 1.0 \
    --log-wandb \
    --wandb-project $WANDB_PROJECT \
    --wandb-run-name "L2_D384_H16_M4_baseline" \
    --output-dir ./logs/dvs128_L2_D384_baseline &

# Config 6: Larger network with XiSPS (no elastic)
# L2 D384 H16 M4, alpha=1.0, XiSPS
echo "Starting Config 6 on GPU 8: Large XiSPS (no elastic)"
CUDA_VISIBLE_DEVICES=8 python train.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --num-classes $NUM_CLASSES \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --T $T \
    --T_train $T_TRAIN \
    --workers $WORKERS \
    --device cuda:0 \
    --depths 2 \
    --embed-dims 384 \
    --num-heads 16 \
    --mlp-ratios 4 \
    --sps-alpha 1.0 \
    --use-xisps \
    --log-wandb \
    --wandb-project $WANDB_PROJECT \
    --wandb-run-name "L2_D384_H16_M4_xisps_a1" \
    --output-dir ./logs/dvs128_L2_D384_xisps_a1 &

# ============================================================
# GPU 9 - 2 experiments
# ============================================================

# Config 7: Larger network with XiSPS elastic
# L2 D384 H16 M4, alpha=1.0, XiSPS elastic
echo "Starting Config 7 on GPU 9: Large XiSPS elastic"
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --num-classes $NUM_CLASSES \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --T $T \
    --T_train $T_TRAIN \
    --workers $WORKERS \
    --device cuda:0 \
    --depths 2 \
    --embed-dims 384 \
    --num-heads 16 \
    --mlp-ratios 4 \
    --sps-alpha 1.0 \
    --use-xisps \
    --xisps-elastic \
    --log-wandb \
    --wandb-project $WANDB_PROJECT \
    --wandb-run-name "L2_D384_H16_M4_xisps_elastic_a1" \
    --output-dir ./logs/dvs128_L2_D384_xisps_elastic_a1 &

# Config 8: Larger network with XiSPS elastic, alpha=2.0 (uses XiSPSv2)
# L2 D384 H16 M4, alpha=2.0, XiSPS elastic
echo "Starting Config 8 on GPU 9: Large XiSPSv2 elastic (alpha=2.0)"
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --num-classes $NUM_CLASSES \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --T $T \
    --T_train $T_TRAIN \
    --workers $WORKERS \
    --device cuda:0 \
    --depths 2 \
    --embed-dims 384 \
    --num-heads 16 \
    --mlp-ratios 4 \
    --sps-alpha 2.0 \
    --use-xisps \
    --xisps-elastic \
    --log-wandb \
    --wandb-project $WANDB_PROJECT \
    --wandb-run-name "L2_D384_H16_M4_xispsv2_elastic_a2" \
    --output-dir ./logs/dvs128_L2_D384_xispsv2_elastic_a2 &

echo ""
echo "All 8 experiments started!"
echo "  GPU 0: Configs 1-2 (small network: baseline & XiSPS)"
echo "  GPU 2: Configs 3-4 (small network: elastic variations)"
echo "  GPU 8: Configs 5-6 (large network: baseline & XiSPS)"
echo "  GPU 9: Configs 7-8 (large network: elastic variations)"
echo ""
echo "Monitor with: watch -n 30 nvidia-smi"
echo "Check wandb project: $WANDB_PROJECT"
echo ""

# Wait for all background jobs to finish
wait

echo "All experiments completed!"