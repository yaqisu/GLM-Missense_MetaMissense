# Evaluation

This directory contains the full pipeline for reproducing the evaluation results
in the paper. The pipeline takes GLM-Missense scores as input, annotates variants
with additional predictors and genomic features, and produces all figures and
tables reported in the paper.

> Run all commands from the **repo root**.

---

## Directory layout

```
evaluation/
│
│   ── Orchestration ──
├── pipeline_config.tsv             # Single config driving prepare_scores + run_evaluation
├── prepare_scores.py               # Step 1 — zero-shot scoring + dbNSFP annotation
├── run_evaluation.py               # Step 2 — merge all scores + baseline evaluation
│
│   ── Core pipeline scripts ──
├── merge.py                        # Merge all score files into one wide TSV
├── evaluate.py                     # AUROC / AUPRC evaluation (full / filter / stratify modes)
├── extract_subset_ids.py           # Extract variant IDs for a label subset (e.g. P+B only)
│
│   ── Zero-shot scoring ──
├── zeroshot_nt.py                  # Zero-shot NT-1 / NT-2 scoring (6-mer tokenization)
├── zeroshot_caduceus.py            # Zero-shot Caduceus-PS / Ph scoring (single-char tokenization)
│
│   ── Annotation enrichment ──
├── annotate_spliceai.py            # Add SpliceAI delta scores
├── annotate_constraint.py          # Add gnomAD v4.1 LOEUF / pLI per gene
├── annotate_grantham.py            # Add Grantham distance + radical AA change flag (hardcoded)
├── annotate_exon_boundaries.py     # Add distance to nearest exon boundary
├── prepare_exon_boundaries.py      # Parse Ensembl GTF → exon parquet (run once)
│
│   ── Figure analyses ──
├── evaluate_partial_correlation.py           # Fig 4B — partial Spearman r controlling for other methods
├── label_prediction_sets.py                  # Fig 6 prep — binary labels + correctness flags per method
├── glmmissense_correct_analysis_for_fig6.py  # Fig 6  — GLM-Missense-correct subset analysis
└── explained_portion_analysis_for_fig7.py    # Fig 7  — explained portion of GLM-Missense errors
│
└── core/                           # Shared library (metrics, filters, plots)
    ├── metrics.py
    ├── filters.py
    ├── plots.py
    └── __init__.py
```

---

## Pipeline overview

```
[scoring/] GLM-Missense.tsv
        │
        ▼
[Step 1] prepare_scores.py
         ├── scoring/GLM-Missense.py       fine-tuned scoring (called externally)
         ├── zeroshot_nt.py × 3            NT-2 seq12k, NT-2 seq6k, NT-1 seq6k
         ├── zeroshot_caduceus.py × 2      Caduceus-PS seq30k, Caduceus-Ph seq30k
         └── scoring/annotate_dbnsfp.py    all dbNSFP columns (called from scoring/)
        │
        ▼
[Step 2] run_evaluation.py
         ├── merge.py                      merge all score files → merged.tsv
         └── evaluate.py                   AUROC / AUPRC → eval_all/
        │
        ▼
[Step 3] Annotation enrichment             (all edit merged.tsv in place)
         ├── annotate_spliceai.py          SpliceAI delta scores
         ├── annotate_constraint.py        LOEUF / pLI from gnomAD v4.1
         ├── annotate_grantham.py          Grantham distance (hardcoded)
         └── annotate_exon_boundaries.py   distance to nearest exon boundary
        │
        ▼
[Step 4] Figure analyses
         ├── evaluate_partial_correlation.py              Fig 4B
         ├── label_prediction_sets.py                     Fig 6 prep
         ├── glmmissense_correct_analysis_for_fig6.py     Fig 6
         └── explained_portion_analysis_for_fig7.py       Fig 7
```

---

## Datasets

Four ClinVar datasets are evaluated:

| Dataset key | Variant classes | Split |
|---|---|---|
| `ClinVar.251103.BLBvsPLP` | Pathogenic/LP vs Benign/LB | Validation chromosomes only |
| `ClinVar.251103.BvsP` | Pathogenic vs Benign (strict) | Validation chromosomes only |
| `ClinVar.260309only.BLBvsPLP` | Pathogenic/LP vs Benign/LB | All chromosomes |
| `ClinVar.260309only.BvsP` | Pathogenic vs Benign (strict) | All chromosomes |

The `BvsP` datasets are strict subsets of their parent `BLBvsPLP` datasets and
reuse the same score files — no re-scoring is needed. Filtering to P+B variants
happens at evaluation time via `--subset`.

---

## Step 1 — Prepare scores

Runs zero-shot scoring and dbNSFP annotation for all source datasets. All steps
skip gracefully if output already exists — safe to rerun after interruption.

```bash
python evaluation/prepare_scores.py \
    --config  evaluation/pipeline_config.tsv \
    --model   scoring/GLM-Missense.pt \
    --dbnsfp  data/dbnsfp/dbNSFP5.3.1a_grch38.gz
```

For each source dataset this runs, in order: GLM-Missense scoring, five
zero-shot models (NT-2 seq12k, NT-2 seq6k, NT-1 seq6k, Caduceus-PS seq30k,
Caduceus-Ph seq30k), and dbNSFP annotation via `scoring/annotate_dbnsfp.py`.
For derived (`reuse:`) rows, only the subset ID file is generated.

Add `--dry_run` to preview all commands without executing.

> **Note on dbNSFP:** `annotate_dbnsfp.py` lives in `scoring/` and is called
> from there. If you have already run it for the MetaMissense pipeline on the
> same variants, you can copy the resulting `dbnsfp.tsv` into
> `results/predictions/{dataset}/` and the step will be skipped automatically.

---

## Step 2 — Merge and evaluate

Merges all score files and runs the baseline AUROC / AUPRC evaluation for every
dataset in `pipeline_config.tsv`.

```bash
python evaluation/run_evaluation.py \
    --config evaluation/pipeline_config.tsv
```

For each dataset this auto-generates `merge_config.tsv` from whatever `.tsv`
files are present in the predictions directory, runs `merge.py` to produce
`merged.tsv`, then runs `evaluate.py` to produce AUROC / AUPRC tables and
plots under `eval_all/`. Add `--skip_merge` to re-run evaluation only
(useful when `merged.tsv` already exists). Add `--dry_run` to preview.

> `merge_config.tsv` is always regenerated — do not hand-edit it. To include
> or exclude a model, add or remove its score file from the predictions directory.

**Results layout:**
```
results/predictions/
├── ClinVar.260309only.BLBvsPLP/
│   ├── GLM-Missense.tsv
│   ├── zeroshot_NT2_seq12k.tsv
│   ├── zeroshot_CaduceusPS_seq30k.tsv
│   ├── dbnsfp.tsv
│   ├── merge_config.tsv          ← auto-generated
│   ├── merged.tsv
│   └── eval_all/
│
└── ClinVar.260309only.BvsP/
    ├── merge_config.tsv          ← points at BLBvsPLP score files
    ├── merged.tsv
    └── eval_all/                 ← evaluated with --subset BvsP_ids.tsv
```

---

## Step 3 — Annotation enrichment

These four scripts add annotation columns to every `merged.tsv` in place. Run
them in any order after Step 2. All are idempotent — re-running drops and
re-adds their columns.

### 3a. SpliceAI scores

```bash
python evaluation/annotate_spliceai.py \
    --spliceai data/spliceai/spliceai_scores.raw.snv.hg38.vcf.gz \
    --predictions_dir results/predictions
```

Requires the raw SNV VCF bgzipped and tabix-indexed (`tabix -p vcf ...`).
Columns added: `spliceai_DS_AG`, `spliceai_DS_AL`, `spliceai_DS_DG`,
`spliceai_DS_DL`, `spliceai_DS_max`, `spliceai_gene`.

### 3b. gnomAD constraint (LOEUF / pLI)

```bash
python evaluation/annotate_constraint.py \
    --constraint    data/loeuf/gnomad.v4.1.constraint_metrics.tsv \
    --predictions_dir results/predictions
```

Joins on gene name (canonical transcripts only). Columns added: `lof.oe_ci.upper`
(LOEUF), `lof.pLI`, `lof.oe`, `lof.obs`, `lof.exp`, `mis.z_score`, `mis.oe`,
`lof.oe_ci.upper_bin_decile`.

### 3c. Grantham distance

No external data required — values are hardcoded from Grantham (1974).

```bash
python evaluation/annotate_grantham.py \
    --predictions_dir results/predictions
```

Requires `aaref` and `aaalt` columns (provided by dbNSFP). Columns added:
`grantham_distance` (0–215 scale), `is_radical_aa_change` (True if distance > 150).

### 3d. Exon boundary distance

First, prepare the exon boundary parquet from an Ensembl GTF (one-time, ~60 sec):

```bash
python evaluation/prepare_exon_boundaries.py \
    --gtf    data/annotation/Homo_sapiens.GRCh38.113.gtf.gz \
    --output data/annotation/exons_GRCh38.113.parquet
```

Then annotate:

```bash
python evaluation/annotate_exon_boundaries.py \
    --exons data/annotation/exons_GRCh38.113.parquet \
    --predictions_dir results/predictions
```

Columns added: `exon_boundary_dist`, `exon_boundary_5prime`,
`exon_boundary_3prime`, `near_exon_boundary`, `exon_boundary_bin`.

---

## Step 4 — Figure analyses

These scripts operate on the enriched `merged.tsv` files and reproduce the
main paper figures. Step 4a must be run before 4c and 4d, as they depend on
its output.

### 4a. Binary prediction labels and correctness flags  *(prerequisite for Figs 6, 7)*

Binarizes each predictor at its standard threshold, adds per-variant
correctness flags, and generates 7×7 focal-method-correct-while-≤N-others-correct
indicators. The key column used downstream is `GLM-Missense_correct_le{N}`,
which flags variants where GLM-Missense is correct and at most N of the other
six methods are also correct.

```bash
python evaluation/label_prediction_sets.py \
    --predictions_dir results/predictions
```

Outputs per dataset: `merged_prediction_labels_all.tsv` (all variants) and
`merged_prediction_labels_all_overlap.tsv` (variants with all 7 methods scored).

### 4b. Partial correlation analysis  *(Fig 4B)*

Computes partial Spearman r for each predictor controlling for all others,
quantifying the unique predictive signal each method contributes.

```bash
python evaluation/evaluate_partial_correlation.py
```

Outputs `partial_correlation_results.tsv` and `partial_correlation.pdf/png`
alongside each `merged.tsv`.

### 4c. GLM-Missense-correct subset analysis  *(Fig 6)*

Characterises the GLM-Missense-correct subset: variants where GLM-Missense is
correctly classified and at least four of the other six methods are wrong
(`GLM-Missense_correct_le2`). Examines mutation spectrum, SpliceAI scores,
and other genomic features of this subset relative to all other variants.

```bash
python evaluation/glmmissense_correct_analysis_for_fig6.py \
    --input   results/predictions/ClinVar.260309only.BLBvsPLP/merged_prediction_labels_all.tsv \
    --outdir  results/figures/fig6
```

### 4d. Explained portion analysis  *(Fig 7)*

Fits logistic regression models to quantify what fraction of GLM-Missense
prediction errors can be explained by allele frequency, LOEUF, SpliceAI score,
Grantham distance, and exon boundary proximity.

```bash
python evaluation/explained_portion_analysis_for_fig7.py \
    --input   results/predictions/ClinVar.260309only.BLBvsPLP/merged_prediction_labels_all.tsv \
    --outdir  results/figures/fig7
```

---

## Data requirements

| Data | Used by | Where to obtain |
|---|---|---|
| dbNSFP v5.x GRCh38 (bgzipped + tabix index) | `scoring/annotate_dbnsfp.py` | dbNSFP project page |
| gnomAD v4.1 constraint metrics TSV | `annotate_constraint.py` | gnomad.broadinstitute.org |
| SpliceAI raw SNV VCF hg38 (bgzipped + tabix index) | `annotate_spliceai.py` | Illumina BaseSpace |
| Ensembl GRCh38 GTF (any recent release) | `prepare_exon_boundaries.py` | Ensembl FTP |

---

## pipeline_config.tsv columns

| Column | Description |
|---|---|
| `dataset_key` | Unique name; determines output folder under `results/predictions/` |
| `seq12k_input` | Path to seq12k TSV, or `reuse:{parent_key}` for derived datasets, or `premerged:{path}` to skip scoring and merge |
| `seq6k_input` | Path to seq6k TSV (NT zero-shot only) |
| `seq30k_input` | Path to seq30k TSV (Caduceus zero-shot only) |
| `finetune_score` | Output path for GLM-Missense score file |
| `subset_ids` | Path to BvsP IDs file for derived datasets |
| `gpu` | GPU ID for scoring |