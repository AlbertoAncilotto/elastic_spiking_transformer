#!/bin/bash

# Spike Analysis Script for Spikformer on EHW Gesture Dataset
# This script runs spike counting analysis on a trained model

# Basic configuration
MODEL="spikformer"
DATASET="ehwgesture"
NUM_CLASSES=22
DATA_PATH="data/ehwgesture/"
DEVICE="cuda:9"
BATCH_SIZE=16
WORKERS=8
T=16

# Model architecture (should match training)
DEPTHS=2
EMBED_DIMS=256
MLP_RATIOS=4
NUM_HEADS=16
PATCH_SIZE=16
IN_CHANNELS=2
SPS_ALPHA=2.0

# Checkpoint path
CHECKPOINT="logs/xisps2-elastic_alpha2.0_t16/spikformer_b16_T16_Ttrain16_wd0.06_adamw_cnf_ADD/lr0.001/checkpoint_max_test_acc1.pth"

# Output directory
OUTPUT_DIR="./spike_analysis_results"

echo "========================================"
echo "Spike Analysis for Spikformer"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "Device: $DEVICE"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint file not found at $CHECKPOINT"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Run standard evaluation (granularity 0 only)
# echo "Running standard evaluation with spike analysis..."
# python spike_analysis.py \
#   --model $MODEL \
#   --dataset $DATASET \
#   --num-classes $NUM_CLASSES \
#   --data-path $DATA_PATH \
#   --device $DEVICE \
#   --batch-size $BATCH_SIZE \
#   --workers $WORKERS \
#   --T $T \
#   --depths $DEPTHS \
#   --embed-dims $EMBED_DIMS \
#   --mlp-ratios $MLP_RATIOS \
#   --num-heads $NUM_HEADS \
#   --patch-size $PATCH_SIZE \
#   --in-channels $IN_CHANNELS \
#   --checkpoint $CHECKPOINT \
#   --output-dir $OUTPUT_DIR

# echo ""
# echo "========================================"
# echo "Standard evaluation complete!"
# echo "Results saved to: $OUTPUT_DIR/evaluation_results.json"
# echo "========================================"
# echo ""
# echo "To run FULL evaluation (all 64 granularity combinations),"
# echo "uncomment the section below and run again."
# echo ""

# Uncomment the lines below to run full evaluation with all 64 granularity combinations
# WARNING: This will take much longer (64x the time)

echo "Running FULL evaluation (all 64 granularity combinations)..."
python spike_analysis.py \
  --model $MODEL \
  --dataset $DATASET \
  --num-classes $NUM_CLASSES \
  --data-path $DATA_PATH \
  --device $DEVICE \
  --batch-size $BATCH_SIZE \
  --workers $WORKERS \
  --T $T \
  --depths $DEPTHS \
  --embed-dims $EMBED_DIMS \
  --mlp-ratios $MLP_RATIOS \
  --num-heads $NUM_HEADS \
  --patch-size $PATCH_SIZE \
  --in-channels $IN_CHANNELS \
  --checkpoint $CHECKPOINT \
  --output-dir $OUTPUT_DIR \
  --use-xisps \
  --xisps-elastic --sps-alpha $SPS_ALPHA --full-eval

# echo ""
# echo "========================================"
# echo "Full evaluation complete!"
# echo "Results saved to: $OUTPUT_DIR/full_evaluation_results.json"
# echo "========================================"