# A fine-tuned genomic language model captures nucleotide-level information overlooked by missense variant impact predictors

**Yaqi Su and Yu-Jen Lin** *(co-first authors)*

---

## Overview

This repository contains the code for the analyses in the paper. We fine-tune Nucleotide Transformer v2 (NT-2) on
ClinVar missense variants using a Siamese ref-alt contrast architecture with LoRA, producing **GLM-Missense** — a
pathogenicity predictor that captures nucleotide-level signals complementary to established predictors. We also
train **MetaMissense**, an XGBoost ensemble of GLM-Missense and the seven missense predictors calibrated by ClinGen
for PP3 / BP4 (BayesDel, MutPred2, REVEL, VEST4, AlphaMissense, ESM1b, VARITY_R).

---

## Scoring your own variants

To score new variants with the released models, start with [`scoring/README.md`](scoring/README.md). The scoring
pipeline is self-contained and does not require running any other part of this repository:

1. **GLM-Missense** — prepare input sequences, then score with the fine-tuned model
2. **MetaMissense** — annotate with dbNSFP predictor scores, then run the ensemble

---

## Repository structure

```
├── preprocessing/   ClinVar releases → BED → sequences → chromosome splits (the ClinVar sets of Table 2)
├── training/        architecture exploration and GLM-Missense training (fine-tuning runs)
├── evaluation/      model-development figures: Figs 2, 3, S2–S7, S10, Table S1
├── analysis/        what GLM-Missense adds to established predictors: Figs 4–7, S8, S9, S11–S14, Tables 1, 2, S2
│   └── scoring/     scores and annotations of the analysis tables (ESM1b-FT, SpliceAI, Saluki, sequence features, ...)
├── scoring/         score new variants with the released GLM-Missense and MetaMissense
├── data/            downloaded and generated data (not tracked)
└── results/         model outputs, annotated tables, figures (not tracked)
```

Each folder has its own README with commands, inputs and outputs. Every command runs from the repo root, and every
path below is relative to it.

### How the folders connect

```
preprocessing/ ──► data/splits/*.tsv ──────────────┬──► training/ ──► results/<run>/ ──► evaluation/ ──► results/evaluation/figures/
   (Table 2 sets)                                   │                   (checkpoints, metrics)       (Figs 2, 3, S2–S7, S10)
                                                    │                        │
                                                    │                        ▼
                                                    │              scoring/GLM-Missense.pt (released weights)
                                                    ▼                        │
   later-release splits + VUS ──► scoring/, analysis/scoring/ ◄──────────────┘
                                        │  GLM-Missense, ESM1b-FT, dbNSFP, SpliceAI, ... scores
                                        ▼
                              results/annotated/{clinvar,sge,vus}/   (one central copy of each annotated table)
                                        │
                                        ▼
                                  analysis/ ──► results/analysis/{figures,tables,models}/<ensemble>/
                                                 (Figs 4–7, S8, S9, S11–S14; MetaMissense.joblib → scoring/)
```

---

## Manuscript figures and tables

| Item | Folder | Script |
|---|---|---|
| Fig 1 | — | schematic |
| Fig 2, Fig 3, Figs S2–S5, Table S1 | `training/` (runs), `evaluation/` (figures) | see `training/README.md`, `evaluation/README.md` |
| Fig S6 | `training/`, `evaluation/` | `NT2_ref_alt_contrast.py`; `plot_training_curves.py` |
| Fig S7 | `evaluation/` | `zeroshot_nt.py`, `fig_zeroshot.py` |
| Fig S10 | `training/`, `evaluation/` | ESM1b-FT training; figure **[TBA-E7]** |
| Fig 4, Fig 5, Fig S11 | `analysis/` | `fig_complementarity.py` |
| Fig 6, Fig S13 | `analysis/` | `fig_splicing.py` |
| Fig 7, Fig S14 | `analysis/` | `fig_vus.py` |
| Fig S8, Fig S9 | `analysis/` | `fig_leaderboard.py` |
| Fig S12 | `analysis/` | `fig_circularity.py` |
| Table 1 | `analysis/` | `fig_splicing.py` (`variant_lists/<ensemble>/fig6_resolved`); **[TBA-A7]** |
| Table 2 | `preprocessing/` (sets), `analysis/` (counts) | `table_clinvar_sets.py`, `table_clinvar_subsets.py` |
| Table S2 | `analysis/` | `fig_splicing.py` (`variant_lists/<ensemble>/figS13_resolved`); **[TBA-A8]** |
| Table S3 | `analysis/` | thresholds in `core.py` (`CALIBRATED`, `THRESH`) |

---

## Placeholders

Parts that are still being added are marked **[TBA-xx]** in the READMEs (`P` preprocessing, `T` training,
`E` evaluation, `A` analysis). Each marker states what is to be added, its inputs and where its outputs go, so
that the code that reads them can find them.

---

## Data

All data files are untracked. `preprocessing/` downloads ClinVar and generates `data/bed/`, `data/sequences/` and
`data/splits/`. Other inputs (reference genomes, dbNSFP 5.3.1a, SpliceAI raw scores, ClinGen Evidence Repository,
RAD51D SGE data, Saluki models) are described in the README of the step that uses them.

Data availability: ClinVar (https://ftp.ncbi.nlm.nih.gov/pub/clinvar/); dbNSFP (https://www.dbnsfp.org/); SpliceAI
precomputed scores (Illumina BaseSpace, registration required); RAD51D saturation genome editing data (Casadei et al.
2026); ClinGen Evidence Repository (https://erepo.clinicalgenome.org/evrepo/); codon stability coefficients
(Wu et al. 2019); Saluki models (https://doi.org/10.5281/zenodo.6326409).

---

## Citation

Su Y & Lin YJ. A fine-tuned genomic language model captures nucleotide-level information overlooked by missense
variant impact predictors. Preprint at bioRxiv. <https://doi.org/10.64898/2026.05.06.723362> (2026).
