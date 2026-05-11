# A Fine-tuned Genomic Language Model Adds Complementary Nucleotide-Context Information to Missense Variant Interpretation

**Yaqi Su and Yu-Jen Lin** *(co-frist authors)*

---

## Overview

This repository contains all code for reproducing the analyses in the paper.
We fine-tune Nucleotide Transformer v2 (NT2) on ClinVar missense variants using
a Siamese Ref-Alt Contrast architecture with LoRA, producing **GLM-Missense** —
a pathogenicity predictor that captures nucleotide-context signals complementary
to existing tools. We also train **MetaMissense**, an ensemble that combines
GLM-Missense with six established predictors.

---

## Scoring your own variants

If you want to score new variants using our pre-trained models, start with
`scoring/README.md`. The scoring pipeline is self-contained and does not require
running any other part of this repository. At a high level:

1. **GLM-Missense** — prepare input sequences, then score with the fine-tuned model
2. **MetaMissense** — annotate with dbNSFP predictor scores, then run the ensemble

See [`scoring/README.md`](scoring/README.md) for full instructions and download
links for model weights.

---

## Repository structure

```
├── data/             # Input data files (mostly gitignored — see below)
├── preprocessing/    # Generate sequence TSVs from ClinVar BED files
├── training/         # Fine-tuning and ablation experiments
├── scoring/          # Score new variants with GLM-Missense and MetaMissense
├── evaluation/       # Reproduce all paper figures and evaluation results
└── results/          # Training outputs and prediction files (gitignored)
```

---

## Reproducing paper results

Each directory below is self-contained with its own README. Follow them in order
if you want to reproduce everything from scratch, or jump to whichever stage is
relevant.

### `data/` — Input data

ClinVar BED files for both timestamps (`251103` and `260309`) are tracked in
`data/bed/` and are the primary inputs to the preprocessing pipeline. All other
data files (reference genome, extracted sequences, train/val splits, dbNSFP,
SpliceAI, gnomAD constraint) are gitignored due to size — download instructions
are in the relevant README for each step.

### `preprocessing/` — Data preparation

Generates the labeled sequence TSVs used for training and evaluation from ClinVar
VCFs and the GRCh38 reference genome. This includes downloading ClinVar, filtering
to missense variants, extracting flanking sequences, and creating chromosome-split
train/val files. See [`preprocessing/README.md`](preprocessing/README.md).

### `training/` — Model training

Contains all training scripts for the frozen backbone ablations (NT2, NT1,
Caduceus), LoRA and full fine-tuning sweeps, cross-validation, and the final
GLM-Missense model. The commands used for training and evaluating the final GLM-Missense model are in `training/runs.sh`.

### `scoring/` — Scoring variants

Contains `GLM-Missense.py` and `MetaMissense.py` for scoring variants, along
with helper scripts for preparing input sequences and fetching predictor scores
from dbNSFP. See [`scoring/README.md`](scoring/README.md).

### `evaluation/` — Evaluation and figures

Contains the full evaluation pipeline: merging predictions, annotating with
SpliceAI / gnomAD constraint / Grantham distance / exon boundaries, and
reproducing all paper figures. See [`evaluation/README.md`](evaluation/README.md).

### `results/` — Outputs

Training logs, model checkpoints, and prediction files are written here. This
directory is gitignored. The subdirectory structure is created automatically by
the training and evaluation scripts.

---

## Citation

Su Y & Lin YJ. A fine-tuned genomic language model adds complementary
nucleotide-context information to missense variant interpretation. Preprint at bioRxiv. <https://doi.org/10.64898/2026.05.06.723362> (2026).
