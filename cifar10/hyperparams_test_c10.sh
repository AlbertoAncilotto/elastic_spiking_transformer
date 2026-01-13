#!/bin/bash
# Hyperparameter sweep for CIFAR-10 targeting ~10M params 

# GPU 5 experiments (4 runs)
# Experiment 1: More layers, slightly smaller dim
CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 5 --dim 352 --num_heads 11 --mlp-ratio 4 \
    --wandb-run-name "c10_hp1_L5_d352_h11_mlp4" &

# Experiment 2: Deeper network with lower mlp ratio
CUDA_VISIBLE_DEVICES=5 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 3 --dim 320 --num_heads 10 --mlp-ratio 4 \
    --wandb-run-name "c10_hp2_L3_d320_h10_mlp4" &

# Experiment 3: Higher time steps for better temporal dynamics
CUDA_VISIBLE_DEVICES=7 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 4 --dim 384 --num_heads 12 --mlp-ratio 4 --time-step 6 \
    --wandb-run-name "c10_hp3_L4_d384_h12_T6" &

# Experiment 4: Larger dim with fewer layers
CUDA_VISIBLE_DEVICES=7 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 3 --dim 448 --num_heads 14 --mlp-ratio 4 \
    --wandb-run-name "c10_hp4_L3_d448_h14_mlp4" &

# wait

# # GPU 8 experiments (4 runs)
# Experiment 5: More heads for better attention capacity
CUDA_VISIBLE_DEVICES=7 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 4 --dim 384 --num_heads 16 --mlp-ratio 4 \
    --wandb-run-name "c10_hp5_L4_d384_h16_mlp4" &

# Experiment 6: Balanced deeper config
CUDA_VISIBLE_DEVICES=7 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 5 --dim 384 --num_heads 12 --mlp-ratio 3 \
    --wandb-run-name "c10_hp6_L5_d384_h12_mlp3" &

# Experiment 7: Higher mlp ratio for more expressive MLP
CUDA_VISIBLE_DEVICES=7 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 4 --dim 352 --num_heads 11 --mlp-ratio 5 \
    --wandb-run-name "c10_hp7_L4_d352_h11_mlp5" &

# Experiment 8: Larger embedding with moderate depth
CUDA_VISIBLE_DEVICES=7 python train.py --log-wandb --config cifar10_8M.yml \
    --layer 4 --dim 416 --num_heads 13 --mlp-ratio 3 \
    --wandb-run-name "c10_hp8_L4_d416_h13_mlp3" &

wait

echo "All hyperparameter experiments completed!"
