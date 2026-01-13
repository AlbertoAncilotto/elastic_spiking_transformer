# CUDA_VISIBLE_DEVICES=9 python train.py \
#     --batch-size 16 \
#     --embed-dims 160 \
#     --num-heads 16 \
#     --epochs 250 \
#     --depths 2 \
#     --lr 1e-3 \
#     --num-classes 22 \
#     --T 16 \
#     --T_train 16 \
#     --workers 8 \
#     --log-wandb \
#     --use-xisps --xisps-elastic --sps-alpha 1.0 \
#     --device cuda:0 \
#     --wandb-project "EHW_timesteps_ablation" \
#     --wandb-run-name "final_xisps1a1_t16_160_d2" \
#     --output-dir "./logs/final_xisps1a1_t16_160_d2" \
#     --attn-lower-heads-limit 4

# CUDA_VISIBLE_DEVICES=9 python train.py \
#     --embed-dims 256 \
#     --num-heads 32 \
#     --depths 2 \
#     --attn-lower-heads-limit 8 \
#     --sps-lower-filter-limit 16 \
#     --batch-size 16 \
#     --epochs 250 \
#     --lr 1e-3 \
#     --num-classes 22 \
#     --T 16 \
#     --T_train 16 \
#     --workers 8 \
#     --log-wandb \
#     --use-xisps --xisps-elastic --sps-alpha 2.0 \
#     --device cuda:0 \
#     --wandb-project "EHW_timesteps_ablation" \
#     --wandb-run-name "final_xisps2a2_t16_h32_256_d2_lfl16" \
#     --output-dir "./logs/final_xisps2a2_t16_h32_256_d2_lfl16"

# CUDA_VISIBLE_DEVICES=9 python train.py \
#     --batch-size 32 \
#     --embed-dims 160 \
#     --num-heads 16 \
#     --epochs 250 \
#     --depths 2 \
#     --lr 1e-3 \
#     --num-classes 22 \
#     --T 8 \
#     --T_train 8 \
#     --workers 8 \
#     --log-wandb \
#     --use-xisps --xisps-elastic --sps-alpha 1.0 \
#     --device cuda:0 \
#     --wandb-project "EHW_timesteps_ablation" \
#     --wandb-run-name "final_xisps1a1_t8_160_d2" \
#     --output-dir "./logs/final_xisps1a1_t8_160_d2" \
#     --attn-lower-heads-limit 4

# CUDA_VISIBLE_DEVICES=9 python train.py \
#     --embed-dims 256 \
#     --num-heads 32 \
#     --depths 2 \
#     --attn-lower-heads-limit 8 \
#     --sps-lower-filter-limit 16 \
#     --batch-size 32 \
#     --epochs 250 \
#     --lr 1e-3 \
#     --num-classes 22 \
#     --T 8 \
#     --T_train 8 \
#     --workers 8 \
#     --log-wandb \
#     --use-xisps --xisps-elastic --sps-alpha 2.0 \
#     --device cuda:0 \
#     --wandb-project "EHW_timesteps_ablation" \
#     --wandb-run-name "final_xisps2a2_t8_h32_256_d2_lfl16" \
#     --output-dir "./logs/final_xisps2a2_t8_h32_256_d2_lfl16"



CUDA_VISIBLE_DEVICES=7 python train.py \
    --embed-dims 256 \
    --num-heads 32 \
    --depths 2 \
    --attn-lower-heads-limit 8 \
    --sps-lower-filter-limit 16 \
    --batch-size 8 \
    --epochs 250 \
    --lr 1e-3 \
    --num-classes 22 \
    --T 32 \
    --T_train 32 \
    --workers 8 \
    --log-wandb \
    --use-xisps --xisps-elastic --sps-alpha 2.0 \
    --device cuda:0 \
    --wandb-project "EHW_timesteps_ablation" \
    --wandb-run-name "final_xisps2a2_t32_h32_256_d2_lfl16" \
    --output-dir "./logs/final_xisps2a2_t32_h32_256_d2_lfl16" &


CUDA_VISIBLE_DEVICES=7 python train.py \
    --batch-size 4 \
    --embed-dims 160 \
    --num-heads 16 \
    --epochs 250 \
    --depths 2 \
    --lr 1e-3 \
    --num-classes 22 \
    --T 64 \
    --T_train 64 \
    --workers 8 \
    --log-wandb \
    --use-xisps --xisps-elastic --sps-alpha 1.0 \
    --device cuda:0 \
    --wandb-project "EHW_timesteps_ablation" \
    --wandb-run-name "final_xisps1a1_t64_160_d2" \
    --output-dir "./logs/final_xisps1a1_t64_160_d2" \
    --attn-lower-heads-limit 4 &

wait

CUDA_VISIBLE_DEVICES=7 python train.py \
    --embed-dims 256 \
    --num-heads 32 \
    --depths 2 \
    --attn-lower-heads-limit 8 \
    --sps-lower-filter-limit 16 \
    --batch-size 4 \
    --epochs 250 \
    --lr 1e-3 \
    --num-classes 22 \
    --T 64 \
    --T_train 64 \
    --workers 8 \
    --log-wandb \
    --use-xisps --xisps-elastic --sps-alpha 2.0 \
    --device cuda:0 \
    --wandb-project "EHW_timesteps_ablation" \
    --wandb-run-name "final_xisps2a2_t64_h32_256_d2_lfl16" \
    --output-dir "./logs/final_xisps2a2_t64_h32_256_d2_lfl16" &

CUDA_VISIBLE_DEVICES=7 python train.py \
    --batch-size 8 \
    --embed-dims 160 \
    --num-heads 16 \
    --epochs 250 \
    --depths 2 \
    --lr 1e-3 \
    --num-classes 22 \
    --T 32 \
    --T_train 32 \
    --workers 8 \
    --log-wandb \
    --use-xisps --xisps-elastic --sps-alpha 1.0 \
    --device cuda:0 \
    --wandb-project "EHW_timesteps_ablation" \
    --wandb-run-name "final_xisps1a1_t32_160_d2_retrain" \
    --output-dir "./logs/final_xisps1a1_t32_160_d2_retrain" \
    --attn-lower-heads-limit 4 &

wait

# # ============================================================
# # Alternative architectures (T=16)
# # ============================================================

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


