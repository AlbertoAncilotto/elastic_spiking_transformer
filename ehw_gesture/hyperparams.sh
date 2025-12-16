#!/bin/bash

# Parallel training script for Spikformer hyperparameter search
# Runs 3 consecutive trainings on each of 3 GPUs (9 total experiments)

# Set common arguments
COMMON_ARGS="--batch-size 16 \
             --epochs 110 \
             --lr 1e-3 \
             --num-classes 22 \
             --T 16 \
             --T_train 16 \
             --workers 8 \
             --wandb-project spikformer_ehwgesture-hyperparam-search \
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

# GPU 5: depth / embed variations
gpu5_experiments() {
    run_on_gpu 5 "spikformer_d1_e256_m4_h16" 1 256 4 16
    run_on_gpu 5 "spikformer_d2_e256_m4_h16" 2 256 4 16
    run_on_gpu 5 "spikformer_d3_e256_m4_h16" 3 256 4 16
}

# GPU 6: embed size scaling
gpu6_experiments() {
    run_on_gpu 6 "spikformer_d2_e196_m4_h14" 2 196 4 14
    run_on_gpu 6 "spikformer_d2_e256_m4_h16_baseline" 2 256 4 16
    run_on_gpu 6 "spikformer_d2_e384_m4_h24" 2 384 4 24
}

# GPU 9: MLP ratio variations
gpu9_experiments() {
    run_on_gpu 9 "spikformer_d2_e256_m2_h16" 2 256 2 16
    run_on_gpu 9 "spikformer_d2_e256_m4_h16" 2 256 1 16
    run_on_gpu 9 "spikformer_d2_e256_m6_h16" 2 256 4 16
}

echo "=========================================="
echo "Starting parallel training on 3 GPUs"
echo "GPU 5: depth scaling (1 / 2 / 3)"
echo "GPU 6: embed dims (196 / 256 / 384)"
echo "GPU 9: MLP ratios (2 / 1 / 6)"
echo "Epochs per experiment: 110"
echo "=========================================="

# Launch all GPU processes in background
gpu5_experiments > logs_gpu5.txt 2>&1 &
PID5=$!

gpu6_experiments > logs_gpu6.txt 2>&1 &
PID6=$!

gpu9_experiments > logs_gpu9.txt 2>&1 &
PID9=$!

echo "GPU 5 process PID: $PID5"
echo "GPU 6 process PID: $PID6"
echo "GPU 9 process PID: $PID9"

check_progress() {
    echo ""
    echo "=========================================="
    echo "Training Progress Check"
    echo "=========================================="

    if ps -p $PID5 > /dev/null; then echo "GPU 5: Still running ✓"; else echo "GPU 5: Completed ✓"; fi
    if ps -p $PID6 > /dev/null; then echo "GPU 6: Still running ✓"; else echo "GPU 6: Completed ✓"; fi
    if ps -p $PID9 > /dev/null; then echo "GPU 9: Still running ✓"; else echo "GPU 9: Completed ✓"; fi

    echo "=========================================="
}

# Monitor every 30 minutes
while ps -p $PID5 > /dev/null || ps -p $PID6 > /dev/null || ps -p $PID9 > /dev/null; do
    sleep 1800
    check_progress
done

wait $PID5; EXIT5=$?
wait $PID6; EXIT6=$?
wait $PID9; EXIT9=$?

echo ""
echo "=========================================="
echo "All training completed!"
echo "=========================================="
echo "GPU 5 exit code: $EXIT5"
echo "GPU 6 exit code: $EXIT6"
echo "GPU 9 exit code: $EXIT9"
echo ""
echo "Check logs:"
echo "  - logs_gpu5.txt"
echo "  - logs_gpu6.txt"
echo "  - logs_gpu9.txt"
echo ""
echo "WandB Project:"
echo "  spikformer_ehwgesture-hyperparam-search"
echo "=========================================="

echo ""
echo "Experiment Summary:"
echo "-------------------"
echo "GPU 5:"
echo "  d1 / d2 / d3 @ e256"
echo ""
echo "GPU 6:"
echo "  e196 / e256 / e384 @ d2"
echo ""
echo "GPU 9:"
echo "  mlp 2 / 1 / 4 @ d2"
echo "=========================================="
