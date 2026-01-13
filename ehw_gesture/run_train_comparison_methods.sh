
# # Spikformer Legacy (T=16)
# CUDA_VISIBLE_DEVICES=9 python train.py \
#     --model spikformer_legacy \
#     --batch-size 16 \
#     --epochs 250 \
#     --lr 1e-3 \
#     --num-classes 22 \
#     --T 16 \
#     --T_train 16 \
#     --workers 8 \
#     --log-wandb \
#     --device cuda:0 \
#     --wandb-project "EHW_timesteps_ablation" \
#     --wandb-run-name "spikformer_legacy_t16" \
#     --output-dir "./logs/spikformer_legacy_t16"

# # QKFormer (T=16)
# CUDA_VISIBLE_DEVICES=9 python train.py \
#     --model QKFormer \
#     --batch-size 16 \
#     --epochs 250 \
#     --lr 1e-3 \
#     --num-classes 22 \
#     --T 16 \
#     --T_train 16 \
#     --workers 8 \
#     --log-wandb \
#     --device cuda:0 \
#     --wandb-project "EHW_timesteps_ablation" \
#     --wandb-run-name "qkformer_t16" \
#     --output-dir "./logs/qkformer_t16"

# # Spikformer Legacy (T=16)
# CUDA_VISIBLE_DEVICES=9 python train.py \
#     --model spikformer_legacy \
#     --batch-size 16 \
#     --epochs 250 \
#     --embed-dims 160 \
#     --depths 1 \
#     --lr 1e-3 \
#     --num-classes 22 \
#     --T 16 \
#     --T_train 16 \
#     --workers 8 \
#     --log-wandb \
#     --device cuda:0 \
#     --wandb-project "EHW_timesteps_ablation" \
#     --wandb-run-name "spikformer_legacy_t16_smaller" \
#     --output-dir "./logs/spikformer_legacy_t16_smaller"

# QKFormer (T=16)
CUDA_VISIBLE_DEVICES=9 python train.py \
    --model QKFormer \
    --batch-size 16 \
    --epochs 150 \
    --embed-dims 160 \
    --depths 2 \
    --lr 1e-3 \
    --num-classes 22 \
    --T 16 \
    --T_train 16 \
    --workers 8 \
    --log-wandb \
    --device cuda:0 \
    --wandb-project "EHW_timesteps_ablation" \
    --wandb-run-name "qkformer_t16_smaller" \
    --output-dir "./logs/qkformer_t16_smaller"

# Spikformer Legacy (T=16)
CUDA_VISIBLE_DEVICES=9 python train.py \
    --model spikformer_legacy \
    --batch-size 16 \
    --epochs 150 \
    --embed-dims 320 \
    --depths 3 \
    --lr 1e-3 \
    --num-classes 22 \
    --T 16 \
    --T_train 16 \
    --workers 8 \
    --log-wandb \
    --device cuda:0 \
    --wandb-project "EHW_timesteps_ablation" \
    --wandb-run-name "spikformer_legacy_t16_larger" \
    --output-dir "./logs/spikformer_legacy_t16_larger"

# QKFormer (T=16)
CUDA_VISIBLE_DEVICES=9 python train.py \
    --model QKFormer \
    --batch-size 16 \
    --epochs 150 \
    --embed-dims 320 \
    --depths 5 \
    --lr 1e-3 \
    --num-classes 22 \
    --T 16 \
    --T_train 16 \
    --workers 8 \
    --log-wandb \
    --device cuda:0 \
    --wandb-project "EHW_timesteps_ablation" \
    --wandb-run-name "qkformer_t16_larger" \
    --output-dir "./logs/qkformer_t16_larger"



