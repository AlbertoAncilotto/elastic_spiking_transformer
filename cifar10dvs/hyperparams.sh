#!/bin/bash

# Parallel training script for Spikformer hyperparameter search
# Runs 2 consecutive trainings on each of 4 GPUs (8 total experiments)

# Set common arguments
COMMON_ARGS="--batch-size 16 \
             --epochs 96 \
             --lr 1e-3 \
             --num-classes 10 \
             --T 16 \
             --T_train 16 \
             --workers 8 \
             --wandb-project spikformer-hyperparam-search \
             --log-wandb"

# Function to run training on a specific GPU
run_on_gpu() {
    local gpu_id=$1
    local experiment_name=$2
    local depths=$3
    local embed_dims=$4
    local mlp_ratios=$5
    local num_heads=$6
    
    echo "Starting experiment: $experiment_name on GPU $gpu_id"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        $COMMON_ARGS \
        --device cuda:0 \
        --depths $depths \
        --embed-dims $embed_dims \
        --mlp-ratios $mlp_ratios \
        --num-heads $num_heads \
        --wandb-run-name "$experiment_name" \
        --output-dir "./logs/$experiment_name"
    
    echo "Completed experiment: $experiment_name on GPU $gpu_id"
}

# GPU 4: depths variations (1, 2)
gpu4_experiments() {
    # Experiment 1: Shallow network (depth=1)
    run_on_gpu 4 "spikformer_d1_e256_m4_h16" 1 256 4 16
    
    # Experiment 2: Medium network (depth=2)
    run_on_gpu 4 "spikformer_d2_e256_m4_h16" 2 256 4 16
}

# GPU 5: depths variation (3) and embed_dims variation (196)
gpu5_experiments() {
    # Experiment 1: Deep network (depth=3)
    run_on_gpu 5 "spikformer_d3_e256_m4_h16" 3 256 4 16
    
    # Experiment 2: Small embeddings (196)
    run_on_gpu 5 "spikformer_d2_e196_m4_h14" 2 196 4 14
}

# GPU 6: embed_dims variations (256, 384)
gpu6_experiments() {
    # Experiment 1: Standard embeddings (256) - baseline
    run_on_gpu 6 "spikformer_d2_e256_m4_h16_baseline" 2 256 4 16
    
    # Experiment 2: Large embeddings (384)
    run_on_gpu 6 "spikformer_d2_e384_m4_h24" 2 384 4 24
}

# GPU 9: mlp_ratios variations (2, 4)
gpu9_experiments() {
    # Experiment 1: Narrow MLP (ratio=2)
    run_on_gpu 9 "spikformer_d2_e256_m2_h16" 2 256 2 16
    
    # Experiment 2: Wide MLP (ratio=4) - alternative baseline
    run_on_gpu 9 "spikformer_d2_e256_m4_h16_mlp" 2 256 4 16
}

# Run all experiments in parallel (one per GPU)
echo "=========================================="
echo "Starting parallel training on 4 GPUs"
echo "GPU 4: depths variations (1, 2)"
echo "GPU 5: depths=3, embed_dims=196"
echo "GPU 6: embed_dims variations (256, 384)"
echo "GPU 9: mlp_ratios variations (2, 4)"
echo "=========================================="

# Launch all GPU processes in background
gpu4_experiments > logs_gpu4.txt 2>&1 &
PID4=$!

gpu5_experiments > logs_gpu5.txt 2>&1 &
PID5=$!

gpu6_experiments > logs_gpu6.txt 2>&1 &
PID6=$!

gpu9_experiments > logs_gpu9.txt 2>&1 &
PID9=$!

# Store PIDs for monitoring
echo "GPU 4 process PID: $PID4"
echo "GPU 5 process PID: $PID5"
echo "GPU 6 process PID: $PID6"
echo "GPU 9 process PID: $PID9"

# Function to check if process is still running
check_progress() {
    echo ""
    echo "=========================================="
    echo "Training Progress Check"
    echo "=========================================="
    
    if ps -p $PID4 > /dev/null; then
        echo "GPU 4: Still running ✓"
    else
        echo "GPU 4: Completed ✓"
    fi
    
    if ps -p $PID5 > /dev/null; then
        echo "GPU 5: Still running ✓"
    else
        echo "GPU 5: Completed ✓"
    fi
    
    if ps -p $PID6 > /dev/null; then
        echo "GPU 6: Still running ✓"
    else
        echo "GPU 6: Completed ✓"
    fi
    
    if ps -p $PID9 > /dev/null; then
        echo "GPU 9: Still running ✓"
    else
        echo "GPU 9: Completed ✓"
    fi
    echo "=========================================="
}

# Monitor progress every 30 minutes
while ps -p $PID4 > /dev/null || ps -p $PID5 > /dev/null || ps -p $PID6 > /dev/null || ps -p $PID9 > /dev/null; do
    sleep 1800  # 30 minutes
    check_progress
done

# Wait for all background processes to complete
wait $PID4
EXIT4=$?

wait $PID5
EXIT5=$?

wait $PID6
EXIT6=$?

wait $PID9
EXIT9=$?

# Final status report
echo ""
echo "=========================================="
echo "All training completed!"
echo "=========================================="
echo "GPU 4 exit code: $EXIT4"
echo "GPU 5 exit code: $EXIT5"
echo "GPU 6 exit code: $EXIT6"
echo "GPU 9 exit code: $EXIT9"
echo ""
echo "Check individual logs:"
echo "  - logs_gpu4.txt"
echo "  - logs_gpu5.txt"
echo "  - logs_gpu6.txt"
echo "  - logs_gpu9.txt"
echo ""
echo "View results on WandB:"
echo "  Project: spikformer-hyperparam-search"
echo "=========================================="

# Summary of experiments
echo ""
echo "Experiment Summary:"
echo "-------------------"
echo "GPU 4:"
echo "  1. spikformer_d1_e256_m4_h16 (depths=1)"
echo "  2. spikformer_d2_e256_m4_h16 (depths=2)"
echo ""
echo "GPU 5:"
echo "  1. spikformer_d3_e256_m4_h16 (depths=3)"
echo "  2. spikformer_d2_e196_m4_h14 (embed=196)"
echo ""
echo "GPU 6:"
echo "  1. spikformer_d2_e256_m4_h16_baseline (baseline)"
echo "  2. spikformer_d2_e384_m4_h24 (embed=384)"
echo ""
echo "GPU 9:"
echo "  1. spikformer_d2_e256_m2_h16 (mlp=2)"
echo "  2. spikformer_d2_e256_m4_h16_mlp (mlp=4)"
echo "=========================================="