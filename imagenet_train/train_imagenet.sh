#!/bin/bash
# ImageNet training script with multiple configurations targeting >80% accuracy
# Uses XiSPS (not elastic) for multi-granularity inference
# 4 different configurations exploring larger networks

# Common parameters
EPOCHS=300
LR=5e-4
OPT="adamw"
WEIGHT_DECAY=0.05
BATCH_SIZE=24
VAL_BATCH_SIZE=32
DATA_DIR="/home/e3da/code/imagenet"  # Mounted path from docker
WORKERS=8
WARMUP_EPOCHS=20
SCHEDULER="cosine"

# Base config from imagenet.yml (default ~75%): L8, D512, H8, M4 = ~66M params
# Target: >80% accuracy with larger configurations

# ==============================
# Configuration 1: Larger embedding dimension
# L8 D640 H8 M4 = ~102M params
# ==============================
CONFIG1_LAYER=8
CONFIG1_DIM=640
CONFIG1_HEADS=8
CONFIG1_MLP=4

# ==============================
# Configuration 2: More layers with base dim
# L10 D512 H8 M4 = ~82M params
# ==============================
CONFIG2_LAYER=10
CONFIG2_DIM=512
CONFIG2_HEADS=8
CONFIG2_MLP=4

# ==============================
# Configuration 3: Balanced larger model
# L8 D576 H8 M4 = ~83M params
# ==============================
CONFIG3_LAYER=8
CONFIG3_DIM=576
CONFIG3_HEADS=8
CONFIG3_MLP=4

# ==============================
# Configuration 4: Deep with wider MLP
# L10 D512 H8 M5 = ~98M params
# ==============================
CONFIG4_LAYER=10
CONFIG4_DIM=512
CONFIG4_HEADS=8
CONFIG4_MLP=5

# Run configuration based on argument
run_config() {
    local config=$1
    local gpu=$2
    
    case $config in
        1)
            LAYER=$CONFIG1_LAYER
            DIM=$CONFIG1_DIM
            HEADS=$CONFIG1_HEADS
            MLP=$CONFIG1_MLP
            ;;
        2)
            LAYER=$CONFIG2_LAYER
            DIM=$CONFIG2_DIM
            HEADS=$CONFIG2_HEADS
            MLP=$CONFIG2_MLP
            ;;
        3)
            LAYER=$CONFIG3_LAYER
            DIM=$CONFIG3_DIM
            HEADS=$CONFIG3_HEADS
            MLP=$CONFIG3_MLP
            ;;
        4)
            LAYER=$CONFIG4_LAYER
            DIM=$CONFIG4_DIM
            HEADS=$CONFIG4_HEADS
            MLP=$CONFIG4_MLP
            ;;
        *)
            echo "Unknown config: $config"
            exit 1
            ;;
    esac
    
    RUN_NAME="imagenet_L${LAYER}_D${DIM}_H${HEADS}_M${MLP}_xisps"
    OUTPUT_DIR="output/imagenet/${RUN_NAME}"
    
    echo "========================================"
    echo "Running Config $config on GPU $gpu"
    echo "  Layers: $LAYER, Dim: $DIM, Heads: $HEADS, MLP: $MLP"
    echo "  Output: $OUTPUT_DIR"
    echo "========================================"
    
    CUDA_VISIBLE_DEVICES=$gpu python train.py \
        --config imagenet.yml \
        -data-dir "$DATA_DIR" \
        --layer $LAYER \
        --dim $DIM \
        --num_heads $HEADS \
        --mlp-ratio $MLP \
        --use-xisps \
        --sps-alpha 1.0 \
        --use-xi \
        --num-granularities 4 \
        --epochs $EPOCHS \
        --lr $LR \
        --opt $OPT \
        --weight-decay $WEIGHT_DECAY \
        --batch-size $BATCH_SIZE \
        --val-batch-size $VAL_BATCH_SIZE \
        --workers $WORKERS \
        --warmup-epochs $WARMUP_EPOCHS \
        --sched $SCHEDULER \
        --output "$OUTPUT_DIR" \
        --experiment "$RUN_NAME" \
        --log-wandb \
        --wandb-project "elastic_spikformer_imagenet" \
        --wandb-run-name "$RUN_NAME" \
        --amp \
        --pin-mem
}

# Usage examples:
# Run single config: ./train_imagenet.sh 1 0  (config 1 on GPU 0)
# Run all configs in parallel on different GPUs:
#   ./train_imagenet.sh 1 0 &
#   ./train_imagenet.sh 2 1 &
#   ./train_imagenet.sh 3 2 &
#   ./train_imagenet.sh 4 3 &

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config_number> <gpu_id>"
    echo "  config_number: 1-4"
    echo "  gpu_id: GPU device ID"
    echo ""
    echo "Configurations:"
    echo "  1: L8 D640 H8 M4 (~102M params)"
    echo "  2: L10 D512 H8 M4 (~82M params)"
    echo "  3: L8 D576 H8 M4 (~83M params)"
    echo "  4: L10 D512 H8 M5 (~98M params)"
    exit 1
fi

run_config $1 $2
