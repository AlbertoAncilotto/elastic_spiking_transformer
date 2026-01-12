#!/bin/bash
# Hyperparameter sweep for CIFAR-100 targeting ~20M params and >85% accuracy
# Baseline: 78% with 9M params (layer=4, dim=384, heads=12, mlp_ratio=4)
# Scaling strategy: ~2x params via larger dim (~512-640) and/or more layers (6-8)
# Running 2 at a time on GPU 5

echo "=== Starting CIFAR-100 experiments ==="


# # Batch 1: Experiments 1-2
# CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar100_9M.yml \
#     --layer 4 --dim 512 --num_heads 16 --mlp-ratio 4 \
#     --experiment "hp_large1_L4_d512_h16_mlp4" \
#     --wandb-run-name "hp_large1_L4_d512_h16_mlp4" &

# CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar100_9M.yml \
#     --layer 6 --dim 480 --num_heads 16 --mlp-ratio 4 --use-xisps --sps-alpha 1.5 \
#     --experiment "hp_large2_L6_d480_h16_mlp4_xisps_a1.5" \
#     --wandb-run-name "hp_large2_L6_d480_h16_mlp4_xisps_a1.5" &

# wait
# echo "Batch 1 complete"

# # Batch 2: Experiments 3-4
# CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar100_9M.yml \
#     --layer 8 --dim 416 --num_heads 13 --mlp-ratio 4 \
#     --experiment "hp_large3_L8_d416_h13_mlp4" \
#     --wandb-run-name "hp_large3_L8_d416_h13_mlp4" &

# CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar100_9M.yml \
#     --layer 5 --dim 512 --num_heads 16 --mlp-ratio 4 --time-step 6 --use-xisps --sps-alpha 3.0 \
#     --experiment "hp_large4_L5_d512_h16_T6_xisps_a3.0" \
#     --wandb-run-name "hp_large4_L5_d512_h16_T6_xisps_a3.0" &

# wait
# echo "Batch 2 complete"

# # Batch 3: Experiments 5-6
# CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar100_9M.yml \
#     --layer 5 --dim 544 --num_heads 17 --mlp-ratio 4 \
#     --experiment "hp_large5_L5_d544_h17_mlp4" \
#     --wandb-run-name "hp_large5_L5_d544_h17_mlp4" &

# CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar100_9M.yml \
#     --layer 6 --dim 512 --num_heads 16 --mlp-ratio 3 --use-xisps --sps-alpha 1.5 \
#     --experiment "hp_large6_L6_d512_h16_mlp3_xisps_a1.5" \
#     --wandb-run-name "hp_large6_L6_d512_h16_mlp3_xisps_a1.5" &

# wait
# echo "Batch 3 complete"

# # Batch 4: Experiments 7-8
# CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar100_9M.yml \
#     --layer 6 --dim 448 --num_heads 14 --mlp-ratio 5 \
#     --experiment "hp_large7_L6_d448_h14_mlp5" \
#     --wandb-run-name "hp_large7_L6_d448_h14_mlp5" &

# CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar100_9M.yml \
#     --layer 4 --dim 576 --num_heads 18 --mlp-ratio 4 --use-xisps --sps-alpha 3.0 \
#     --experiment "hp_large8_L4_d576_h18_mlp4_xisps_a3.0" \
#     --wandb-run-name "hp_large8_L4_d576_h18_mlp4_xisps_a3.0" &

# wait
# echo "CIFAR-100 experiments complete"

echo ""
echo "=== Starting CIFAR-10 experiments ==="

# CIFAR-10 Batch 1
CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 4 --dim 512 --num_heads 16 --mlp-ratio 4 \
    --experiment "c10_hp_large1_L4_d512_h16_mlp4" \
    --wandb-run-name "c10_hp_large1_L4_d512_h16_mlp4" &

CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 6 --dim 480 --num_heads 16 --mlp-ratio 4 --use-xisps --sps-alpha 1.5 \
    --experiment "c10_hp_large2_L6_d480_h16_mlp4_xisps_a1.5" \
    --wandb-run-name "c10_hp_large2_L6_d480_h16_mlp4_xisps_a1.5" &

wait
echo "CIFAR-10 Batch 1 complete"

# CIFAR-10 Batch 2
CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 8 --dim 416 --num_heads 13 --mlp-ratio 4 \
    --experiment "c10_hp_large3_L8_d416_h13_mlp4" \
    --wandb-run-name "c10_hp_large3_L8_d416_h13_mlp4" &

CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 5 --dim 512 --num_heads 16 --mlp-ratio 4 --time-step 6 --use-xisps --sps-alpha 3.0 \
    --experiment "c10_hp_large4_L5_d512_h16_T6_xisps_a3.0" \
    --wandb-run-name "c10_hp_large4_L5_d512_h16_T6_xisps_a3.0" &

wait
echo "CIFAR-10 Batch 2 complete"

# CIFAR-10 Batch 3
CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 5 --dim 544 --num_heads 17 --mlp-ratio 4 \
    --experiment "c10_hp_large5_L5_d544_h17_mlp4" \
    --wandb-run-name "c10_hp_large5_L5_d544_h17_mlp4" &

CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 6 --dim 512 --num_heads 16 --mlp-ratio 3 --use-xisps --sps-alpha 1.5 \
    --experiment "c10_hp_large6_L6_d512_h16_mlp3_xisps_a1.5" \
    --wandb-run-name "c10_hp_large6_L6_d512_h16_mlp3_xisps_a1.5" &

wait
echo "CIFAR-10 Batch 3 complete"

# CIFAR-10 Batch 4
CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 6 --dim 448 --num_heads 14 --mlp-ratio 5 \
    --experiment "c10_hp_large7_L6_d448_h14_mlp5" \
    --wandb-run-name "c10_hp_large7_L6_d448_h14_mlp5" &

CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 4 --dim 576 --num_heads 18 --mlp-ratio 4 --use-xisps --sps-alpha 3.0 \
    --experiment "c10_hp_large8_L4_d576_h18_mlp4_xisps_a3.0" \
    --wandb-run-name "c10_hp_large8_L4_d576_h18_mlp4_xisps_a3.0" &

wait
echo "CIFAR-10 experiments complete"

echo ""
echo "=== All experiments completed! ==="

# Additional GPU runs if available (uncomment as needed)
# # GPU 6 experiments - exploring extreme configurations
# # Experiment 9: Very wide with moderate depth
# CUDA_VISIBLE_DEVICES=6 python train.py --log-wandb --config cifar100_9M.yml \
#     --layer 5 --dim 576 --num_heads 18 --mlp-ratio 4 \
#     --wandb-run-name "hp_large9_L5_d576_h18_mlp4" &
#
# # Experiment 10: Deep network with time steps
# CUDA_VISIBLE_DEVICES=6 python train.py --log-wandb --config cifar100_9M.yml \
#     --layer 7 --dim 448 --num_heads 14 --mlp-ratio 4 --time-step 6 \
#     --wandb-run-name "hp_large10_L7_d448_h14_T6" &
#
# # Experiment 11: Maximum width
# CUDA_VISIBLE_DEVICES=6 python train.py --log-wandb --config cifar100_9M.yml \
#     --layer 4 --dim 640 --num_heads 20 --mlp-ratio 4 \
#     --wandb-run-name "hp_large11_L4_d640_h20_mlp4" &
#
# # Experiment 12: Deep narrow with high mlp
# CUDA_VISIBLE_DEVICES=6 python train.py --log-wandb --config cifar100_9M.yml \
#     --layer 8 --dim 384 --num_heads 12 --mlp-ratio 6 \
#     --wandb-run-name "hp_large12_L8_d384_h12_mlp6" &
#
# wait

echo "All large hyperparameter experiments completed!"

# Approximate parameter counts (rough estimates):
# hp_large1: L4 d512 h16 mlp4 → ~16M params
# hp_large2: L6 d480 h16 mlp4 → ~20M params
# hp_large3: L8 d416 h13 mlp4 → ~20M params
# hp_large4: L5 d512 h16 mlp4 T6 → ~18M params
# hp_large5: L5 d544 h17 mlp4 → ~22M params
# hp_large6: L6 d512 h16 mlp3 → ~18M params
# hp_large7: L6 d448 h14 mlp5 → ~20M params
# hp_large8: L4 d576 h18 mlp4 → ~20M params
