#!/bin/bash
# ============================================================================
# DVS128 Gesture Final Hyperparameter Search
# ============================================================================
# Based on analysis of previous results.csv:
# 
# KEY FINDINGS:
# 1. L1 (single-layer) >> L2 (two-layer): Best L1=92.01% vs Best L2=85.06%
# 2. Standard SPS outperforms XiSPS on this small dataset
# 3. D256 is optimal, D384 causes instability, D160 is too small
# 4. Elastic training hurts performance on small datasets
# 5. Best config: L1_D256_H16_M4_baseline (92.01%)
#
# OPTIMIZATION STRATEGY:
# - Focus on L1 architecture (proven superior)
# - Test head count variations (H8, H12, H16) for different head_dim
# - Explore embedding dimensions around D256 sweet spot (256, 288, 320)
# - Test MLP ratio 3 vs 4 (parameter efficiency)
# - Include XiSPS non-elastic variants for comparison
# - One L2 with reduced embedding to test if smaller model stabilizes L2
#
# 8 experiments distributed across 4 GPUs (2 per GPU)
# ============================================================================

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
WANDB_PROJECT="spikformer-dvs128-final"

# Function to run training on a specific GPU
run_experiment() {
    local gpu_id=$1
    local experiment_name=$2
    local depths=$3
    local embed_dims=$4
    local num_heads=$5
    local mlp_ratios=$6
    local use_xisps=$7
    local sps_alpha=$8
    
    echo "Starting: $experiment_name on GPU $gpu_id"
    echo "  depths=$depths, embed=$embed_dims, heads=$num_heads, mlp=$mlp_ratios, xisps=$use_xisps, alpha=$sps_alpha"
    
    # Build the xisps flag
    XISPS_FLAG=""
    if [ "$use_xisps" = "true" ]; then
        XISPS_FLAG="--use-xisps"
    fi
    
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
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
        --depths $depths \
        --embed-dims $embed_dims \
        --num-heads $num_heads \
        --mlp-ratios $mlp_ratios \
        --sps-alpha $sps_alpha \
        $XISPS_FLAG \
        --log-wandb \
        --wandb-project $WANDB_PROJECT \
        --wandb-run-name "$experiment_name" \
        --output-dir "./logs/$experiment_name"
    
    echo "Completed: $experiment_name"
}

echo "=========================================="
echo "DVS128 Gesture Final - 8 Optimized Configs"
echo "=========================================="
echo "Dataset: $DATASET (${NUM_CLASSES} classes)"
echo "Epochs: $EPOCHS"
echo ""
echo "All experiments running on GPU 8 in batches of 3"
echo ""

# GPU to use for all experiments
GPU_ID=8

# ============================================================================
# BATCH 1: Configs 1-3 (Head count and embedding variations)
# ============================================================================
run_batch1() {
    echo "=========================================="
    echo "BATCH 1: Starting configs 1-3 in parallel"
    echo "=========================================="
    
    # Config 1: L1_D256_H8_M4_baseline
    run_experiment $GPU_ID "L1_D256_H8_M4_baseline" 1 256 8 4 "false" 1.0 &
    PID1=$!
    
    # Config 2: L1_D256_H12_M4_baseline
    run_experiment $GPU_ID "L1_D256_H12_M4_baseline" 1 256 12 4 "false" 1.0 &
    PID2=$!
    
    # Config 3: L1_D288_H12_M4_baseline
    run_experiment $GPU_ID "L1_D288_H12_M4_baseline" 1 288 12 4 "false" 1.0 &
    PID3=$!
    
    echo "Batch 1 PIDs: $PID1, $PID2, $PID3"
    wait $PID1 $PID2 $PID3
    echo "Batch 1 completed!"
}

# ============================================================================
# BATCH 2: Configs 4-6 (MLP variation and XiSPS variants)
# ============================================================================
run_batch2() {
    echo "=========================================="
    echo "BATCH 2: Starting configs 4-6 in parallel"
    echo "=========================================="
    
    # Config 4: L1_D256_H16_M3_baseline
    run_experiment $GPU_ID "L1_D256_H16_M3_baseline" 1 256 16 3 "false" 1.0 &
    PID4=$!
    
    # Config 5: L1_D256_H8_M4_xisps
    run_experiment $GPU_ID "L1_D256_H8_M4_xisps" 1 256 8 4 "true" 1.0 &
    PID5=$!
    
    # Config 6: L1_D256_H16_M4_xisps
    run_experiment $GPU_ID "L1_D256_H16_M4_xisps" 1 256 16 4 "true" 1.0 &
    PID6=$!
    
    echo "Batch 2 PIDs: $PID4, $PID5, $PID6"
    wait $PID4 $PID5 $PID6
    echo "Batch 2 completed!"
}

# ============================================================================
# BATCH 3: Configs 7-8 (Larger embedding and L2 test)
# ============================================================================
run_batch3() {
    echo "=========================================="
    echo "BATCH 3: Starting configs 7-8 in parallel"
    echo "=========================================="
    
    # Config 7: L1_D320_H16_M4_baseline
    run_experiment $GPU_ID "L1_D320_H16_M4_baseline" 1 320 16 4 "false" 1.0 &
    PID7=$!
    
    # Config 8: L2_D256_H16_M4_baseline
    run_experiment $GPU_ID "L2_D256_H16_M4_baseline" 2 256 16 4 "false" 1.0 &
    PID8=$!
    
    echo "Batch 3 PIDs: $PID7, $PID8"
    wait $PID7 $PID8
    echo "Batch 3 completed!"
}

# ============================================================================
# LAUNCH ALL BATCHES SEQUENTIALLY
# ============================================================================

echo "Running all experiments on GPU $GPU_ID in 3 batches..."
echo ""

{
    echo "Starting at: $(date)"
    echo ""
    
    run_batch1
    echo ""
    
    run_batch2
    echo ""
    
    run_batch3
    echo ""
    
    echo "All batches finished at: $(date)"
} 2>&1 | tee logs_gpu8_final.txt

# ============================================================================
# FINAL REPORT
# ============================================================================
echo ""
echo "=========================================="
echo "ALL TRAINING COMPLETED!"
echo "=========================================="
echo ""
echo "Log file: logs_gpu8_final.txt"
echo ""
echo "WandB Project: $WANDB_PROJECT"
echo "=========================================="
echo ""
echo "EXPERIMENT SUMMARY:"
echo "-------------------"
echo ""
echo "Batch 1 (parallel):"
echo "  1. L1_D256_H8_M4_baseline   - head_dim=32"
echo "  2. L1_D256_H12_M4_baseline  - head_dim~21"
echo "  3. L1_D288_H12_M4_baseline  - D288, head_dim=24"
echo ""
echo "Batch 2 (parallel):"
echo "  4. L1_D256_H16_M3_baseline  - Narrower MLP (M3)"
echo "  5. L1_D256_H8_M4_xisps      - XiSPS + larger head_dim"
echo "  6. L1_D256_H16_M4_xisps     - Standard XiSPS config"
echo ""
echo "Batch 3 (parallel):"
echo "  7. L1_D320_H16_M4_baseline  - Larger D320"
echo "  8. L2_D256_H16_M4_baseline  - L2 with smaller embedding"
echo "=========================================="
