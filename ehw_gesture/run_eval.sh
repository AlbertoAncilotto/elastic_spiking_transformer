# Spikformer (elastic with XiSPS)
python full_evaluate_ehw.py \
    --model spikformer \
    --checkpoint "logs/final_xisps2a2_t8_h32_256_d2_lfl16/spikformer_b32_T8_Ttrain8_wd0.06_adamw_cnf_ADD/lr0.001/checkpoint_259.pth" \
    --patch-size 16 \
    --embed-dims 256 \
    --num-heads 32 \
    --mlp-ratios 4 \
    --in-channels 2 \
    --depths 2 \
    --sps-alpha 2.0 \
    --use-xisps \
    --xisps-elastic \
    --num-classes 22 \
    --attn-lower-heads-limit 8 \
    --sps-lower-filter-limit 16 \
    --data-path "data/ehwgesture/" \
    --T 8 \
    --batch-size 16 \
    --workers 4
