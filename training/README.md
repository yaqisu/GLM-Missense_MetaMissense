# Training

Fine-tuning experiments of the architecture exploration and the training of GLM-Missense (Methods 5.2–5.7).
Every command runs from the **repo root**. Inputs are the split files made by `preprocessing/` (Table 2); each run
writes to its own folder under `results/` (`--output_dir`), one `exp_<id>/` per configuration with
`training_metrics.csv`, `summary.json` and `best_model.pt`. The figures are drawn from these folders by
`evaluation/`.

The ensembles (MetaMissense and the ablation ensembles) are trained in `analysis/core.py`, not here.

---

## Experiments

| Manuscript | Experiment | Script | Training / validation data (`data/splits/`) |
|---|---|---|---|
| Fig 2A, Fig S2 | Frozen backbones: NT-2 (12k; 6k for the matched NT-1 comparison) × heads × embeddings | `NT2_phase1_and_unfreezeAll.py` | `ClinVar.251103.missense.hg38.seq{12k,6k}.BvsP_{training,validation}.tsv` |
| Fig 2A, Fig S2 | Frozen backbones: NT-1 (6k) | `NT1_phase1.py` | `ClinVar.251103.missense.hg38.seq6k.BvsP_{training,validation}.tsv` |
| Fig 2A, Fig S2 | Frozen backbones: Caduceus-PS (30k) | `caduceus_phase1.py` | `ClinVar.251103.missense.hg38.seq30k.BvsP_{training,validation}.tsv` |
| Fig 2B, Fig S3 | Local-window mean pooling (±8, ±16, ±32 tokens), frozen NT-2 + CNN | `NT2_window_pooling_sweep.py` | `ClinVar.251103.missense.hg38.seq12k.BvsP_{training,validation}.tsv` |
| Fig 3A | LoRA model selection: 54 configurations (2 embeddings × 3 heads × 3 ranks × 3 learning rates) | `NT2_lora_fullFT_sweep.py` (part 1) | 5,000 / 1,250 subsets (**[TBA-P1]**) |
| Fig S4 | LoRA vs full fine-tuning (5 learning rates) | `NT2_lora_fullFT_sweep.py` (part 2); `NT2_single_mlp_lora32_lr5e5_fulltrain.py` (LoRA) | `ClinVar.251103.missense.hg38.seq12k.BvsP_{training,validation}.tsv` |
| Table S1 | 5-fold chromosome-exclusive CV of the top 10 LoRA configurations | `NT2_ranked_cv.py` (`--ranking_json` from the Fig 3A run) | combined 6,250-variant subset (**[TBA-P1]**) |
| Fig 3B, Fig S5 | Allele-aware inputs: alt, ref-alt, ref-alt-diff, alt-alt-alt | `NT2_ref_alt_input_sweep.py`, `NT2_ref_alt_input_sweep_altTriple.py` (alt-alt-alt) | 5,000 / 1,250 subsets (**[TBA-P1]**) |
| Fig S6 | **GLM-Missense**: NT-2, LoRA r = 32, ref-alt-diff, MLP head | `NT2_ref_alt_contrast.py` (command in `runs.sh`); `evaluate_checkpoints.py` (optional offline checkpoint evaluation) | `ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_{training,validation}.tsv` |
| Fig S10 | **ESM1b-FT**: ESM1b with the GLM-Missense head and settings | `analysis/scoring/fine-tune_ESM1b.py` (see ESM1b-FT below) | same variants as GLM-Missense, as protein sequences |

The released GLM-Missense weights are the selected checkpoint of the GLM-Missense run, copied to
`scoring/GLM-Missense.pt` (see `scoring/README.md`).

---

## GLM-Missense

```bash
bash training/runs.sh
```

Output: `results/NT2_seq12k_BLBvsPLP_ref_alt_contrast_mlp/exp_1_concat_diff/` (`training_metrics.csv`: training and
validation AUROC / AUPRC every 1,000 steps, read by `evaluation/plot_training_curves.py` for Fig S6; checkpoints).

---

## Run commands

`A` = `data/splits/ClinVar.251103.missense.hg38`. `SUB_TRAIN`, `SUB_VAL` and `CV_SET` are the 5,000-variant training
subset, the 1,250-variant validation subset and their 6,250-variant union (**[TBA-P1]** in `preprocessing/`).

```bash
A=data/splits/ClinVar.251103.missense.hg38

# Fig 2A, Fig S2 — frozen backbones
python training/NT2_phase1_and_unfreezeAll.py --train_path $A.seq12k.BvsP_training.tsv \
    --val_path $A.seq12k.BvsP_validation.tsv --output_dir results/NT2_seq12k_BvsP_frozen --gpus 0 1 2 3
python training/NT2_phase1_and_unfreezeAll.py --train_path $A.seq6k.BvsP_training.tsv \
    --val_path $A.seq6k.BvsP_validation.tsv --output_dir results/NT2_seq6k_BvsP_frozen --gpus 0 1 2 3   # matched NT-1 comparison
python training/NT1_phase1.py --train_path $A.seq6k.BvsP_training.tsv \
    --val_path $A.seq6k.BvsP_validation.tsv --output_dir results/NT1_seq6k_BvsP_frozen --gpus 0 1 2 3
python training/caduceus_phase1.py --gpus 0 1 2 3             # -> results_caduceus_phase1/ (paths at the top of the script)

# Fig 2B, Fig S3 — local-window mean pooling, frozen NT-2 + CNN
python training/NT2_window_pooling_sweep.py --train_path $A.seq12k.BvsP_training.tsv \
    --val_path $A.seq12k.BvsP_validation.tsv --output_dir results/NT2_seq12k_BvsP_window_pooling --gpus 0 1 2

# Fig 3A (exp_1-54: LoRA, subsets) and Fig S4 (exp_55-59: full fine-tuning, full split)
python training/NT2_lora_fullFT_sweep.py --train_path_sweep $SUB_TRAIN --val_path_sweep $SUB_VAL \
    --train_path_ft $A.seq12k.BvsP_training.tsv --val_path_ft $A.seq12k.BvsP_validation.tsv \
    --output_dir results/NT2_seq12k_BvsP_lora_fullFT_sweep --gpus 0 1 2 3
# Fig S4 — LoRA run on the full split (variant position, MLP, r = 32, lr 5e-5; 20,000 steps)
python training/NT2_single_mlp_lora32_lr5e5_fulltrain.py --train_path $A.seq12k.BvsP_training.tsv \
    --val_path $A.seq12k.BvsP_validation.tsv --output_dir results/NT2_seq12k_BvsP_lora_fullData --gpus 0

# Table S1 — 5-fold chromosome-exclusive CV of the top 10 LoRA configurations
# (experiment_ranking.json: evaluation/summarize_sweep.py on the Fig 3A run)
python training/NT2_ranked_cv.py --ranking_json results/NT2_seq12k_BvsP_lora_fullFT_sweep/experiment_ranking.json \
    --data_path $CV_SET --top_n 10 --output_dir results/NT2_seq12k_BvsP_ranked_cv --gpus 0 1 2 3

# Fig 3B, Fig S5 — allele-aware inputs (LoRA r = 32, lr 5e-5, variant-position MLP; subsets)
#   alt: exp_36 of the Fig 3A sweep (variant position, MLP, r = 32, lr 5e-5)
python training/NT2_ref_alt_input_sweep.py --train_path $SUB_TRAIN --val_path $SUB_VAL \
    --output_dir results/NT2_seq12k_BvsP_allele_inputs --gpus 0 1     # exp_1_concat_diff (ref-alt-diff), exp_2_concat_only (ref-alt)
python training/NT2_ref_alt_input_sweep_altTriple.py --train_path $SUB_TRAIN --val_path $SUB_VAL \
    --output_dir results/NT2_seq12k_BvsP_alt_triple --gpus 0           # exp_2_alt_triple (alt-alt-alt)
```

### GLM-Missense training

Settings (Methods 5.6): NT-2, LoRA r = 32 (α = 64, dropout 0.1, query and value projections), learning rate
5 × 10⁻⁵ (6% linear warm-up, linear decay over the `num_steps` of `EXPERIMENT_CONFIGS`), ref-alt-diff input, MLP head,
effective batch 32 (8 × 4 gradient accumulation). Every 1,000 steps the model is evaluated on 5,000 randomly drawn
training variants (`--train_eval_samples`) and on a fixed, label-stratified subset of 5,000 validation variants
(`--val_eval_samples`, seed 42), and a checkpoint is saved (`checkpoints/step_<N>.pt`; rerunning the command resumes
from the last one). Training ran for 10,000 steps; `best_model.pt` is the checkpoint with the best validation AUROC
(step 8,000).

```bash
python training/NT2_ref_alt_contrast.py \
    --train_path data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_training.tsv \
    --val_path data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_validation.tsv \
    --output_dir results/NT2_seq12k_BLBvsPLP_ref_alt_contrast_mlp \
    --gpus 0 1 2 3 --gpus_per_experiment 4 --batch_size 8 --gradient_accumulation_steps 4 \
    --eval_interval 1000 --eval_batch_size 256 --train_eval_samples 5000 --val_eval_samples 5000
cp results/NT2_seq12k_BLBvsPLP_ref_alt_contrast_mlp/exp_1_concat_diff/best_model.pt scoring/GLM-Missense.pt
```

### ESM1b-FT training (Fig S10)

ESM1b with the GLM-Missense head and settings (Methods 5.7): LoRA r = 32, α = 64, dropout 0.1; learning rate 5 × 10⁻⁵;
effective batch 32; 10,000 steps; evaluation every 1,000 steps on the same 5,000 validation variants as GLM-Missense.
Inputs are the reference and alternate proteins of the transcript ClinVar named each variant on (MANE Select through the
genome where that transcript cannot be used). `map_variants_to_mane_protein.py` writes every input row
(`status` = `ok` for missense), so the rows stay aligned with the split files.

```bash
# downloads (data/clinvar/, data/reference/)
wget -P data/clinvar https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
wget -P data/clinvar -r -np -nd -A 'MANE.GRCh38.v1.4.refseq_rna.gbff.gz,MANE.GRCh38.v1.4.summary.txt.gz' \
     https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_1.4/
wget -P data/reference https://ftp.ensembl.org/pub/release-104/gtf/homo_sapiens/Homo_sapiens.GRCh38.104.gtf.gz
gunzip data/reference/Homo_sapiens.GRCh38.104.gtf.gz

A=data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP
python analysis/scoring/map_variants_to_mane_protein.py --inputs ${A}_training.tsv ${A}_validation.tsv \
    --variant_summary data/clinvar/variant_summary.txt.gz --gbff data/clinvar/MANE.GRCh38.v1.4.refseq_rna.gbff.gz \
    --gtf data/reference/Homo_sapiens.GRCh38.104.gtf --transcript_ids data/clinvar/MANE.GRCh38.v1.4.summary.txt.gz \
    --genome_fasta data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa --output_dir data/protein
python analysis/scoring/fine-tune_ESM1b.py \
    --train_path data/protein/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_training_protein.tsv \
    --val_path data/protein/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_validation_protein.tsv \
    --output_dir results/models/ESM1b-FT --gpus 0 1 2 3 \
    --eval_interval 1000 --train_eval_samples 5000 --val_eval_samples 5000 --stop_at_step 10000
```

Output: `results/models/ESM1b-FT/exp_1_concat_diff/best_model.pt` (LoRA, projector and head weights; best validation
AUROC at step 3,000), read by `analysis/` to score ClinVar, SGE and VUS variants, and `training_metrics.csv` (same
columns as GLM-Missense), the input of Fig S10 (`evaluation/plot_training_curves.py --name figS10`).
