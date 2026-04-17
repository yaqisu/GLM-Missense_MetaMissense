# Fine-tuning Genomic Language Models for Variant Pathogenicity Prediction

This repository contains the code for *"Fine-tuning genomic language models for variant pathogenicity prediction"* (Su\*, Lin\*, et al.; \*co-first authors; in preparation).

We fine-tune genomic language models (Nucleotide Transformer v2, Caduceus) on ClinVar missense variants to predict pathogenicity. Our best model achieves **0.886 AUC** using NT2 with LoRA (rank=32) and a CNN classifier head.

---

## Quick Start — Score Your Variants

If you just want to score variants using our best pre-trained model:

**1. Download the model weights (link TBD upon publication)** and place at `scoring/model/best_model.pt`

**2. Prepare your input** — a TSV file with the same format as our split files (see [preprocessing](preprocessing/README.md) for how to generate this from a BED file). Required columns:
```
variant_id  chromosome  position  ref_allele  alt_allele
upstream_flank  downstream_flank  ref_sequence  alt_sequence
```

**3. Run scoring:**
```bash
python scoring/score_variants.py \
    --input  your_variants.tsv \
    --model  scoring/model/best_model.pt \
    --output results/predictions/scores.tsv
```

**Output** — a TSV file with one row per variant:
```
variant_id  chromosome  position  ref_allele  alt_allele  pathogenicity_score  predicted_label
```
- `pathogenicity_score`: probability of pathogenicity (0–1), higher = more pathogenic
- `predicted_label`: binary call at 0.5 threshold (0=benign, 1=pathogenic)

If your input has a `label` column, AUC will also be computed and logged automatically.

---

## Repository Structure

```
Fine-tuning-gLM-variant-pathogenicity/
│
├── data/                               # All data files (mostly gitignored)
│   ├── bed/                            # ClinVar BED files (tracked in git)
│   ├── reference/                      # GRCh38 reference genome (gitignored, ~3GB)
│   ├── sequences/                      # Extracted sequences (gitignored)
│   ├── splits/                         # Train/val splits (gitignored)
│   └── vcf/                            # ClinVar VCF files (gitignored)
│
├── preprocessing/                      # Scripts to generate data/ from raw inputs
│   ├── config.tsv                      # Dataset definitions (edit to add timestamps)
│   ├── process_clinvar.sh              # Download ClinVar VCFs → BED files
│   ├── subtract_new_variants.sh        # Extract held-out benchmark variants
│   ├── generate_datasets.sh            # Generate sequences and train/val splits
│   ├── extract_variant_sequences.py    # Extract flanking sequences from FASTA
│   ├── split_data_fixed_chroms.py      # Split by chromosome into train/val
│   └── README.md
│
├── training/                           # Model training scripts
│   ├── NT2_phase1_and_unfreezeAll.py   # NT2 frozen backbone + full fine-tuning
│   ├── NT2_lora_sweep.py               # NT2 LoRA rank/LR/classifier/embedding sweep
│   ├── NT2_fullFT_sweepLR.py           # NT2 full fine-tuning LR sweep
│   ├── NT2_window_pooling_sweep.py     # NT2 frozen backbone window pooling sweep
│   ├── NT2_ranked_cv.py                # 5-fold CV on top-ranked LoRA configs
│   ├── NT2_ref_alt_contrast.py         # Final model: Siamese LoRA + MLP head
│   ├── NT1_phase1.py                   # NT1 frozen backbone experiments
│   ├── caduceus_phase1.py              # Caduceus frozen backbone experiments
│   └── runs.sh                         # Exact commands used in the paper
│
├── scoring/                            # Score new variants using trained model
│   ├── score_variants.py
│   └── model/                          # Best model weights (gitignored)
│       ├── best_model.pt
│       └── model_config.json
│
├── evaluation/                         # Compute metrics from predictions
│   └── README.md
│
└── results/                            # All outputs from training and analysis
    ├── NT2_seq12k_BLBvsPLP_lr3e-5/     # Training outputs (gitignored)
    ├── figures/                         # Final paper figures
    ├── figures.ipynb                    # Figure generation notebook
    └── results.tsv                      # Combined results table
```

---

## Reproducing Paper Results

Follow these steps in order to reproduce our results from scratch.

### 1. Data

Download the reference genome:
```bash
wget https://ftp.ensembl.org/pub/release-104/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
gunzip Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
mv Homo_sapiens.GRCh38.dna.primary_assembly.fa data/reference/
```

ClinVar BED files for both timestamps are already tracked in `data/bed/`. To regenerate them from scratch (e.g. for a new ClinVar release), see Step 2a below.

### 2. Preprocessing

All preprocessing scripts are run from the **repo root**. See [preprocessing](preprocessing/README.md) for full documentation.

**2a. (Optional) Download ClinVar VCFs and regenerate BED files**

Only needed if you want to update to a new ClinVar release or regenerate from scratch. BED files for `251103` and `260309` are already in `data/bed/`.

```bash
bash preprocessing/process_clinvar.sh \
    -t clinvar_20251103,clinvar_20260309 \
    -b /path/to/bcftools
```

> **Note:** If regenerating BED files, also re-run Step 2b to update the held-out benchmark set.

**2b. (Optional) Extract held-out benchmark variants**

Extracts variants present in the `260309` release but not in `251103`, ensuring the benchmark set was never seen during training.

```bash
bash preprocessing/subtract_new_variants.sh
```

**2c. Generate sequences and train/val splits**

```bash
bash preprocessing/generate_datasets.sh -c preprocessing/config.tsv -s 12k
```

This reads `preprocessing/config.tsv` to determine which BED files to use, extracts flanking sequences from the reference genome, and generates train/val splits for labeled datasets. To generate all window sizes used in the paper:

```bash
bash preprocessing/generate_datasets.sh -c preprocessing/config.tsv -s 6k,12k,30k
```

### 3. Training

All training scripts are run from the repo root and read data from `data/splits/`. To reproduce the exact final model from the paper:

```bash
bash training/runs.sh
```

The full experimental pipeline proceeds in four stages:

**Stage 1 — Frozen backbone (classifier and embedding ablation)**

Freeze the backbone and sweep over classifier architectures and embedding strategies to identify the best combination:

| Script | Model | Description |
|--------|-------|-------------|
| `training/NT2_phase1_and_unfreezeAll.py` | NT2 | Frozen backbone experiments; also includes full fine-tuning (unfreeze all) |
| `training/NT1_phase1.py` | NT1 | Frozen backbone experiments |
| `training/caduceus_phase1.py` | Caduceus | Frozen backbone experiments |

**Stage 2 — LoRA and full fine-tuning sweep**

Using the best classifier/embedding combination from Stage 1, sweep over LoRA rank, learning rate, and fine-tuning strategy:

| Script | Description |
|--------|-------------|
| `training/NT2_lora_sweep.py` | 54 LoRA experiments: 2 embedding strategies × 3 classifiers × 3 LoRA ranks (8/16/32) × 3 LRs (1e-5/3e-5/5e-5). Distributed across 4 GPUs (LoRA: 1 GPU each; full FT: 2 GPUs each). |
| `training/NT2_fullFT_sweepLR.py` | Full fine-tuning LR sweep (1e-4, 5e-4) for MLP and CNN classifiers. |
| `training/NT2_window_pooling_sweep.py` | Frozen-backbone CNN experiments varying local mean-pooling window size around the variant position (±8/±16/±32 tokens). |

**Stage 3 — Cross-validation on top configurations**

| Script | Description |
|--------|-------------|
| `training/NT2_ranked_cv.py` | 5-fold chromosome-split cross-validation on top-ranked LoRA configurations from Stage 2, evaluated in rank order. Requires `experiment_ranking.json` produced by `summarize_sweep.py`. |

```bash
# First generate the ranking from sweep results:
python summarize_sweep.py /path/to/sweep_output --csv

# Then run CV in rank order:
python training/NT2_ranked_cv.py \
    --ranking_json /path/to/sweep_output/experiment_ranking.json \
    --data_path data/splits/combined_train_val.tsv \
    --output_dir results/nt2_ranked_cv \
    --gpus 0 1 2 3
```

**Stage 4 — Final model**

| Script | Description |
|--------|-------------|
| `training/NT2_ref_alt_contrast.py` | Siamese LoRA (rank=32) fine-tuning: ref and alt sequences are independently encoded through the shared NT2 backbone; embeddings at the variant position are combined as `[ref, alt, ref − alt]` and passed to a 2-layer MLP classifier. |

### 4. Results

Training outputs are saved to `results/` (one subdirectory per experiment). The combined results table used for figure generation is at `results/results.tsv`.

Generate all paper figures:
```bash
jupyter nbconvert --to notebook --execute results/figures.ipynb
```

Generated figures are saved to `results/figures/`:

| Figure | Description |
|--------|-------------|
| `fig2_classifier_embedding_comparison` | Classifier and embedding strategy comparison |
| `fig3_lora_rank_comparison` | LoRA rank ablation |
| `fig4_learning_rate_comparison` | Learning rate sweep |
| `fig5_lora_vs_full_finetuning` | LoRA vs full fine-tuning comparison |
| `fig6_model_performance_summary` | Summary of all model results |

### 5. Scoring

Score new variants using the best trained model:
```bash
python scoring/score_variants.py \
    --input  your_variants.tsv \
    --model  scoring/model/best_model.pt \
    --output results/predictions/scores.tsv
```

### 6. Evaluation
Evaluate predictions against ground truth labels — see [evaluation](evaluation/README.md) for details.

---

## Models

| Model | Dataset | Seq length | AUC |
|-------|---------|------------|-----|
| NT2 + LoRA (rank=32) + MLP | BvsP | 12k | **0.886** |

---

## Requirements

```bash
conda env create -f environment.yml
conda activate glm-finetune
```

---

## Citation

If you use this code, please cite: Su\*, Lin\*, et al. *Fine-tuning genomic language models for variant pathogenicity prediction*. In preparation.  
\*Co-first authors
