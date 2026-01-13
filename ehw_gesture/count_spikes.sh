# Spikformer (elastic with XiSPS)
python spike_analysis_v2.py \
    --model spikformer \
    --checkpoint "logs/final_xisps2a2_t16_h32_256_d2_lfl16/spikformer_b16_T16_Ttrain16_wd0.06_adamw_cnf_ADD/lr0.001/checkpoint_max_test_acc1.pth" \
    --model-name "xispikeformer_t16" \
    --patch-size 16 \
    --embed-dims 256 \
    --num-heads 32 \
    --mlp-ratios 4 \
    --in-channels 2 \
    --depths 2 \
    --sr-ratios 1 \
    --sps-alpha 2.0 \
    --use-xisps \
    --xisps-elastic \
    --num-classes 22 \
    --attn-lower-heads-limit 8 \
    --sps-lower-filter-limit 16 \
    --data-path "data/ehwgesture/" \
    --T 16 \
    --batch-size 16 \
    --workers 4

# Spikformer Legacy
python spike_analysis_v2.py \
    --model spikformer_legacy \
    --checkpoint "logs/spikformer_legacy_t16/spikformer_legacy_b16_T16_Ttrain16_wd0.06_adamw_cnf_ADD/lr0.001/checkpoint_max_test_acc1.pth" \
    --model-name "spikformer_legacy_t16" \
    --num-classes 22 \
    --in-channels 2 \
    --data-path "data/ehwgesture/" \
    --T 16 \
    --batch-size 16 \
    --embed-dims 256 \
    --depths 2 \
    --workers 4

# QKFormer
python spike_analysis_v2.py \
    --model QKFormer \
    --checkpoint "logs/qkformer_t16/QKFormer_b16_T16_Ttrain16_wd0.06_adamw_cnf_ADD/lr0.001/checkpoint_max_test_acc1.pth" \
    --model-name "qkformer_t16" \
    --num-classes 22 \
    --in-channels 2 \
    --data-path "data/ehwgesture/" \
    --T 16 \
    --batch-size 16 \
    --embed-dims 256 \
    --depths 4 \
    --workers 4