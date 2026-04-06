# Final model
python training/NT2_ref_alt_contrast.py \
    --train_path data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_training.tsv \
    --val_path data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_validation.tsv \
    --output_dir results/NT2_seq12k_BLBvsPLP_ref_alt_contrast_mlp \
    --gpus 0 1 2 3 \
    --gpus_per_experiment 4 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --eval_interval 1000
