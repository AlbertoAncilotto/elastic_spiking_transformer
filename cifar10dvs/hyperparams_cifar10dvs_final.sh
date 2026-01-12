#!/bin/bash
# ============================================================================
# CIFAR10-DVS Final Hyperparameter Search
# ============================================================================
# Based on analysis of results_cifar10dvs.csv:
#
# KEY FINDINGS:
# 1. XiSPSv2 (alpha=2.0) outperforms XiSPSv1 (alpha=1.0) by 1-3%
# 2. Best config: d1_e256_alpha2.0 = 78.8%
# 3. Shallow networks (d1) > d2 > d3 (d3 crashed with alpha=2.0)
# 4. D256 > D200 (+2% accuracy)
# 5. Elastic regularization works well with CIFAR10-DVS (10k samples)
#
# OPTIMIZATION STRATEGY:
# - Focus on d1 (proven best) and d2 (stable)
# - Explore alpha values: 1.5, 2.0, 2.5
# - Test larger embeddings: D288, D320
# - Compare XiSPS elastic vs non-elastic
# - Include baseline (no XiSPS) for reference
# - Vary head counts (H8, H12, H16) and MLP ratios (M3, M4)
#
# 12 experiments in 6 batches of 2, all on GPU 5
# ============================================================================

# Common parameters
DATASET="cifar10dvs"
NUM_CLASSES=10
DATA_PATH="data/cifar10dvs-python/"
EPOCHS=106
LR=1e-3
BATCH_SIZE=16
WORKERS=8
T=16
T_TRAIN=16
WANDB_PROJECT="spikformer-cifar10dvs-final"

# GPU to use
GPU_ID=5

# Function to run training
run_experiment() {
    local experiment_name=$1
    local depths=$2
    local embed_dims=$3
    local num_heads=$4
    local mlp_ratios=$5
    local use_xisps=$6
    local xisps_elastic=$7
    local sps_alpha=$8
    
    echo "Starting: $experiment_name"
    echo "  depths=$depths, embed=$embed_dims, heads=$num_heads, mlp=$mlp_ratios"
    echo "  xisps=$use_xisps, elastic=$xisps_elastic, alpha=$sps_alpha"
    
    # Build flags
    XISPS_FLAG=""
    if [ "$use_xisps" = "true" ]; then
        XISPS_FLAG="--use-xisps"
    fi
    
    ELASTIC_FLAG=""
    if [ "$xisps_elastic" = "true" ]; then
        ELASTIC_FLAG="--xisps-elastic"
    fi
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
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
        $ELASTIC_FLAG \
        --log-wandb \
        --wandb-project $WANDB_PROJECT \
        --wandb-run-name "$experiment_name" \
        --output-dir "./logs/$experiment_name"
    
    echo "Completed: $experiment_name"
}

echo "============================================================"
echo "CIFAR10-DVS Final - 12 Optimized Configurations"
echo "============================================================"
echo "Dataset: $DATASET (${NUM_CLASSES} classes)"
echo "Epochs: $EPOCHS"
echo "All experiments on GPU $GPU_ID in batches of 2"
echo ""

# ============================================================================
# BATCH 1: Baseline comparison + Best XiSPSv2 with larger embedding
# ============================================================================
run_batch1() {
    echo "=========================================="
    echo "BATCH 1: Baseline + D288 XiSPSv2"
    echo "=========================================="
    
    # Config 1: Standard SPS baseline (no XiSPS) for comparison
    # Rationale: Establish baseline without XiSPS to compare improvements
    run_experiment "L1_D256_H16_M4_baseline" 1 256 16 4 "false" "false" 1.0 &
    PID1=$!
    
    # Config 2: XiSPSv2 with slightly larger embedding
    # Rationale: D256 was best, try D288 for more capacity
    run_experiment "L1_D288_H16_M4_xispsv2_elastic" 1 288 16 4 "true" "true" 2.0 &
    PID2=$!
    
    wait $PID1 $PID2
    echo "Batch 1 completed!"
}

# ============================================================================
# BATCH 2: Alpha exploration (1.5 and 2.5)
# ============================================================================
run_batch2() {
    echo "=========================================="
    echo "BATCH 2: Alpha exploration"
    echo "=========================================="
    
    # Config 3: Test alpha=1.5 (between XiSPSv1 and XiSPSv2)
    # Rationale: Find optimal alpha between 1.0 and 2.0
    run_experiment "L1_D256_H16_M4_xisps_a1.5_elastic" 1 256 16 4 "true" "true" 1.5 &
    PID3=$!
    
    # Config 4: Test alpha=2.5 (beyond XiSPSv2)
    # Rationale: Check if higher alpha provides further improvements
    run_experiment "L1_D256_H16_M4_xisps_a2.5_elastic" 1 256 16 4 "true" "true" 2.5 &
    PID4=$!
    
    wait $PID3 $PID4
    echo "Batch 2 completed!"
}

# ============================================================================
# BATCH 3: Head count variations with XiSPSv2
# ============================================================================
run_batch3() {
    echo "=========================================="
    echo "BATCH 3: Head count variations"
    echo "=========================================="
    
    # Config 5: Fewer heads (H8) for larger head_dim=32
    # Rationale: Larger head dimensions may capture richer patterns
    run_experiment "L1_D256_H8_M4_xispsv2_elastic" 1 256 8 4 "true" "true" 2.0 &
    PID5=$!
    
    # Config 6: Middle ground heads (H12)
    # Rationale: Balance between H8 and H16
    run_experiment "L1_D256_H12_M4_xispsv2_elastic" 1 256 12 4 "true" "true" 2.0 &
    PID6=$!
    
    wait $PID5 $PID6
    echo "Batch 3 completed!"
}

# ============================================================================
# BATCH 4: MLP ratio + Elastic vs Non-elastic
# ============================================================================
run_batch4() {
    echo "=========================================="
    echo "BATCH 4: MLP ratio + Elastic comparison"
    echo "=========================================="
    
    # Config 7: Narrower MLP (M3 instead of M4)
    # Rationale: Fewer parameters, potentially less overfitting
    run_experiment "L1_D256_H16_M3_xispsv2_elastic" 1 256 16 3 "true" "true" 2.0 &
    PID7=$!
    
    # Config 8: XiSPSv2 WITHOUT elastic
    # Rationale: Test if elastic helps or hurts on CIFAR10-DVS
    run_experiment "L1_D256_H16_M4_xispsv2_noelastic" 1 256 16 4 "true" "false" 2.0 &
    PID8=$!
    
    wait $PID7 $PID8
    echo "Batch 4 completed!"
}

# ============================================================================
# BATCH 5: Larger embeddings
# ============================================================================
run_batch5() {
    echo "=========================================="
    echo "BATCH 5: Larger embeddings"
    echo "=========================================="
    
    # Config 9: D320 with XiSPSv2
    # Rationale: Test if more capacity improves accuracy
    run_experiment "L1_D320_H16_M4_xispsv2_elastic" 1 320 16 4 "true" "true" 2.0 &
    PID9=$!
    
    # Config 10: D320 with baseline (no XiSPS)
    # Rationale: Compare larger baseline vs XiSPS
    run_experiment "L1_D320_H16_M4_baseline" 1 320 16 4 "false" "false" 1.0 &
    PID10=$!
    
    wait $PID9 $PID10
    echo "Batch 5 completed!"
}

# ============================================================================
# BATCH 6: L2 depth experiments (stable configurations)
# ============================================================================
run_batch6() {
    echo "=========================================="
    echo "BATCH 6: L2 depth experiments"
    echo "=========================================="
    
    # Config 11: L2 with D288 and alpha=2.0
    # Rationale: L2 was 77.2% with D256, try slightly larger embedding
    run_experiment "L2_D288_H12_M4_xispsv2_elastic" 2 288 12 4 "true" "true" 2.0 &
    PID11=$!
    
    # Config 12: L2 baseline for comparison
    # Rationale: Establish L2 baseline without XiSPS
    run_experiment "L2_D256_H16_M4_baseline" 2 256 16 4 "false" "false" 1.0 &
    PID12=$!
    
    wait $PID11 $PID12
    echo "Batch 6 completed!"
}

# ============================================================================
# LAUNCH ALL BATCHES SEQUENTIALLY
# ============================================================================

echo "Running 12 experiments on GPU $GPU_ID in 6 batches of 2..."
echo ""

{
    echo "============================================================"
    echo "Starting at: $(date)"
    echo "============================================================"
    echo ""
    
    run_batch1
    echo ""
    
    run_batch2
    echo ""
    
    run_batch3
    echo ""
    
    run_batch4
    echo ""
    
    run_batch5
    echo ""
    
    run_batch6
    echo ""
    
    echo "============================================================"
    echo "All batches finished at: $(date)"
    echo "============================================================"
} 2>&1 | tee logs_cifar10dvs_final.txt

# ============================================================================
# FINAL REPORT
# ============================================================================
echo ""
echo "============================================================"
echo "ALL TRAINING COMPLETED!"
echo "============================================================"
echo ""
echo "Log file: logs_cifar10dvs_final.txt"
echo "WandB Project: $WANDB_PROJECT"
echo ""
echo "============================================================"
echo "EXPERIMENT SUMMARY"
echo "============================================================"
echo ""
echo "Batch 1 - Baseline + D288:"
echo "  1. L1_D256_H16_M4_baseline         - Standard SPS, no XiSPS"
echo "  2. L1_D288_H16_M4_xispsv2_elastic  - XiSPSv2, larger embedding"
echo ""
echo "Batch 2 - Alpha Exploration:"
echo "  3. L1_D256_H16_M4_xisps_a1.5_elastic - Alpha=1.5 (between v1 and v2)"
echo "  4. L1_D256_H16_M4_xisps_a2.5_elastic - Alpha=2.5 (beyond v2)"
echo ""
echo "Batch 3 - Head Count Variations:"
echo "  5. L1_D256_H8_M4_xispsv2_elastic   - Fewer heads, head_dim=32"
echo "  6. L1_D256_H12_M4_xispsv2_elastic  - Middle ground heads"
echo ""
echo "Batch 4 - MLP + Elastic Comparison:"
echo "  7. L1_D256_H16_M3_xispsv2_elastic  - Narrower MLP (M3)"
echo "  8. L1_D256_H16_M4_xispsv2_noelastic - XiSPSv2 without elastic"
echo ""
echo "Batch 5 - Larger Embeddings:"
echo "  9. L1_D320_H16_M4_xispsv2_elastic  - D320 with XiSPSv2"
echo " 10. L1_D320_H16_M4_baseline         - D320 baseline"
echo ""
echo "Batch 6 - L2 Depth Experiments:"
echo " 11. L2_D288_H12_M4_xispsv2_elastic  - L2 with larger embedding"
echo " 12. L2_D256_H16_M4_baseline         - L2 baseline"
echo ""
echo "EXPECTED OUTCOMES:"
echo "  - Best d1 XiSPSv2 was 78.8%, target >80%"
echo "  - Alpha 1.5-2.5 range to find optimal value"
echo "  - Head/MLP variations may improve efficiency"
echo "  - Elastic vs non-elastic comparison for CIFAR10-DVS"
echo "  - Baseline comparison to quantify XiSPS benefit"
echo "============================================================"
