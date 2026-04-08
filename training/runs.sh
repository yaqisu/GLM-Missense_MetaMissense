# Final model

# # Terminal 1 — training
# python training/NT2_ref_alt_contrast.py \
#     --train_path data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_training.tsv \
#     --val_path data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_validation.tsv \
#     --output_dir results/NT2_seq12k_BLBvsPLP_ref_alt_contrast_mlp \
#     --gpus 0 1 2 3 \
#     --gpus_per_experiment 4 \
#     --batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --no_eval

# # Terminal 2 — eval running alongside on the same 4 GPUs
# python training/evaluate_checkpoints.py \
#     --checkpoints_dir results/NT2_seq12k_BLBvsPLP_ref_alt_contrast_mlp/exp_1_concat_diff/checkpoints \
#     --train_path data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_training.tsv \
#     --val_path data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_validation.tsv \
#     --num_steps 17000 \
#     --eval_interval 1000 \
#     --gpus 0 1 2 3 \
#     --batch_size 256

# Training & eval in one script; increase eval_batch_size to speed up
python training/NT2_ref_alt_contrast.py \
    --train_path data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_training.tsv \
    --val_path data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_validation.tsv \
    --output_dir results/NT2_seq12k_BLBvsPLP_ref_alt_contrast_mlp \
    --gpus 0 1 2 3 --gpus_per_experiment 4 \
    --batch_size 8 --gradient_accumulation_steps 4 \
    --eval_interval 1000 \
    --eval_batch_size 256 \
    --train_eval_samples 10000