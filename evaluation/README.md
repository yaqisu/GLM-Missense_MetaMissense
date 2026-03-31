# Evaluation Pipeline

Modular, dataset-agnostic pipeline for evaluating fine-tuned genomic language
models against dbNSFP baseline methods on held-out ClinVar variants.

---

## Directory layout

```
evaluation/
├── core/                       # Shared library (imported by pipeline scripts)
│   ├── metrics.py              #   metric computation (AUROC, PRAUC, MCC, …)
│   ├── filters.py              #   AF / conservation filtering & stratification
│   ├── plots.py                #   all plotting functions
│   └── __init__.py
│
├── zeroshot_nt.py              # Zero-shot NT-1/NT-2 scoring via masked marginal LLR (6-mer tokenization)
├── zeroshot_caduceus.py        # Zero-shot Caduceus scoring via masked marginal LLR (single-char tokenization)
├── concat_sequences.py         # Concatenate all classes into one dataset TSV
├── extract_subset_ids.py       # Generate subset IDs files
├── annotate_dbnsfp.py          # Annotate variants with dbNSFP via tabix
├── merge_predictions.py        # Merge our scores with dbNSFP annotations
├── evaluate_all.py             # Evaluate all methods on shared subset
├── evaluate_filtered.py        # Evaluate after AF / conservation filter or stratification
└── compare_strategies.py       # Cross-strategy comparison plots & tables
```

---

## Step-by-step usage (example: ClinVar.260309only)

### Step 1 — Concatenate all per-class sequence files

Concatenates all four classes (pathogenic, likely_pathogenic, likely_benign,
benign) into a single TSV. Scoring is run once on this file; class subsets
are handled at evaluation time via `--subset`.

Also generate any subset IDs files you need for evaluation (e.g. P+B only).
These can be created now from the per-class files before concatenation.

```bash
# Concatenate all classes
python evaluation/concat_sequences.py \
    --dataset ClinVar.260309only \
    --datadir data/sequences
# Output: data/sequences/ClinVar.260309only.seq12k.tsv

# Extract P+B variant IDs for use as --subset in eval scripts (default: pathogenic + benign)
python evaluation/extract_subset_ids.py \
    --dataset ClinVar.260309only \
    --datadir data/sequences \
    --outfile data/sequences/ClinVar.260309only.pb_ids.tsv
```

---

### Step 2a — Score variants with our fine-tuned model

```bash
python scoring/score_variants.py \
    --input  data/sequences/ClinVar.260309only.seq12k.tsv \
    --model  scoring/model/best_model.pt \
    --output results/predictions/ClinVar.260309only.seq12k/ours.tsv
```

---

### Step 2b — Annotate variants with dbNSFP (download first if needed)

```bash
# Download (one-time)
curl --http1.1 -C - -o data/dbnsfp/dbNSFP5.3.1a_grch38.gz \
    https://dist.genos.us/academic/yourcode/dbNSFP5.3.1a_grch38.gz
curl --http1.1 -C - -o data/dbnsfp/dbNSFP5.3.1a_grch38.gz.tbi \
    https://dist.genos.us/academic/yourcode/dbNSFP5.3.1a_grch38.gz.tbi

python evaluation/annotate_dbnsfp.py \
    --variants data/sequences/ClinVar.260309only.seq12k.tsv \
    --dbnsfp   data/dbnsfp/dbNSFP5.3.1a_grch38.gz \
    --outdir   results/predictions/ClinVar.260309only.seq12k
```

---

### Step 2c — Zero-shot scoring with pretrained genomic LMs (no fine-tuning)

Scores variants using pretrained model backbones directly, without any fine-tuned
weights. Uses masked marginal log-likelihood ratio as the pathogenicity signal.
Output format is identical to `score_variants.py` so all outputs feed into
`merge_predictions.py` without modification.

Two scripts are provided because NT and Caduceus have different tokenization schemes:
- **`zeroshot_nt.py`**: NT-1 and NT-2 use 6-mer tokenization (`--k 6`); sequences
  must be kmerized before passing to the tokenizer.
- **`zeroshot_caduceus.py`**: Caduceus uses single-character tokenization (1 token
  per nucleotide); sequences are passed as raw strings, no kmerization.

#### NT-1 and NT-2

```bash
# NT-2 (multi-species, 500M) — same backbone as our fine-tuned model — seq12k
python evaluation/zeroshot_nt.py \
    --input  data/sequences/ClinVar.260309only.missense.hg38.seq12k.tsv \
    --output results/predictions/ClinVar.260309only.seq12k/zeroshot_NT2_seq12k.tsv \
    --model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --gpu 2

# NT-2 — seq6k
python evaluation/zeroshot_nt.py \
    --input  data/sequences/ClinVar.260309only.missense.hg38.seq6k.tsv \
    --output results/predictions/ClinVar.260309only.seq6k/zeroshot_NT2_seq6k.tsv \
    --model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --gpu 2

# NT-1 (human-only, 500M) — seq6k
python evaluation/zeroshot_nt.py \
    --input  data/sequences/ClinVar.260309only.missense.hg38.seq6k.tsv \
    --output results/predictions/ClinVar.260309only.seq6k/zeroshot_NT1_seq6k.tsv \
    --model_name InstaDeepAI/nucleotide-transformer-500m-human-ref \
    --gpu 2
```

#### Caduceus

```bash

pip install mamba-ssm --no-build-isolation --break-system-packages

# Caduceus-PS (reverse-complement equivariant) — seq30k
python evaluation/zeroshot_caduceus.py \
    --input  data/sequences/ClinVar.260309only.missense.hg38.seq30k.tsv \
    --output results/predictions/ClinVar.260309only.seq30k/zeroshot_CaduceusPS_seq30k.tsv \
    --model_name kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
    --gpu 2

# Caduceus-Ph (RC augmented) — seq30k
python evaluation/zeroshot_caduceus.py \
    --input  data/sequences/ClinVar.260309only.missense.hg38.seq30k.tsv \
    --output results/predictions/ClinVar.260309only.seq30k/zeroshot_CaduceusPh_seq30k.tsv \
    --model_name kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
    --gpu 2
```

**GPU memory**: NT 500M uses ~2–3 GB; Caduceus uses ~1–2 GB. It is safe to share
a GPU with another running job as long as there is enough free memory — check with
`nvidia-smi` before running. If free memory is tight, use `--gpu -1` to run on
CPU. To restrict to a specific GPU:
```bash
CUDA_VISIBLE_DEVICES=2 python evaluation/zeroshot_nt.py --gpu 0 ...
```

**Checkpointing and resume**: scores are flushed to the output TSV every 100
variants. If the job is interrupted, rerun the exact same command — already-scored
variants are detected by `variant_id` and skipped automatically.

**Output columns** (identical for both scripts):

| Column | Description |
|--------|-------------|
| `variant_id`, `chromosome`, `position`, `ref_allele`, `alt_allele` | Variant identifiers (passed through from input) |
| `pathogenicity_score` | `sigmoid(-(log P(alt\|ctx) - log P(ref\|ctx)))`, range 0–1, higher = more pathogenic |
| `predicted_label` | Binary call at `--threshold` (default 0.5) |
| `true_label` | Copied from input `label` column if present |
| `log_p_alt` | log P(alt token \| masked context) — intermediate, not used by eval scripts |
| `log_p_ref` | log P(ref token \| masked context) — intermediate, not used by eval scripts |
| `log_likelihood_ratio` | `log_p_alt - log_p_ref` — intermediate, not used by eval scripts |

A `.summary.json` file is written alongside each output TSV with model name,
variant count, variant token index, and AUC (if labels present).

---

### Step 3 — Merge predictions

Pass all prediction TSVs you want to compare via `--ours` (can be repeated)
or rely on the directory convention expected by `merge_predictions.py`.

```bash
python evaluation/merge_predictions.py \
    --ours    results/predictions/ClinVar.260309only.seq12k/ours.tsv \
    --zeroshot results/predictions/ClinVar.260309only.seq12k/zeroshot_NT2.tsv \
    --zeroshot results/predictions/ClinVar.260309only.seq12k/zeroshot_NT1.tsv \
    --dbnsfp  results/predictions/ClinVar.260309only.seq12k/dbnsfp.tsv \
    --outdir  results/predictions/ClinVar.260309only.seq12k
# Outputs: merged.tsv, missing_in_dbnsfp.tsv, dbnsfp_column_coverage.tsv
```

---

### Step 4 — Evaluate all variants (shared subset)

```bash
# Four classes (pathogenic + likely pathogenic + likely benign + benign)
python evaluation/evaluate_all.py \
    --merged  results/predictions/ClinVar.260309only.seq12k/merged.tsv \
    --outdir  results/predictions/ClinVar.260309only.seq12k/eval_plpblb

# Restrict to pathogenic + benign only (P+B subset)
python evaluation/evaluate_all.py \
    --merged  results/predictions/ClinVar.260309only.seq12k/merged.tsv \
    --outdir  results/predictions/ClinVar.260309only.seq12k/eval_pb \
    --subset  data/sequences/ClinVar.260309only.pb_ids.tsv
```

Outputs: `all_metrics.tsv`, `summary_comparison.tsv`, `plots/`

---

### Step 5 — Evaluate with filtering or stratification

#### Filter: keep rare variants only (AF < threshold)

```bash
# AF < 1e-3, missing = included (ultra-rare)
python evaluation/evaluate_filtered.py \
    --merged    results/predictions/ClinVar.260309only.seq12k/merged.tsv \
    --outdir    results/predictions/ClinVar.260309only.seq12k/eval_rare_1e-3 \
    --mode      filter \
    --col       gnomAD4.1_joint_AF \
    --threshold 1e-3

# AF < 1e-6, missing = included (ultra-rare)
python evaluation/evaluate_filtered.py \
    --merged    results/predictions/ClinVar.260309only.seq12k/merged.tsv \
    --outdir    results/predictions/ClinVar.260309only.seq12k/eval_rare_1e-6 \
    --mode      filter \
    --col       gnomAD4.1_joint_AF \
    --threshold 1e-6
```

#### Filter: highly conserved sites

```bash
# phyloP100way >= 3
python evaluation/evaluate_filtered.py \
    --merged    results/predictions/ClinVar.260309only.seq12k/merged.tsv \
    --outdir    results/predictions/ClinVar.260309only.seq12k/eval_conserved_phylop \
    --mode      filter \
    --col       phyloP100way_vertebrate \
    --threshold 3.0 \
    --direction above

# GERP++ RS >= 4
python evaluation/evaluate_filtered.py \
    --merged    results/predictions/ClinVar.260309only.seq12k/merged.tsv \
    --outdir    results/predictions/ClinVar.260309only.seq12k/eval_conserved_gerp \
    --mode      filter \
    --col       "GERP++_RS" \
    --threshold 4.0 \
    --direction above
```

#### Stratify: evaluate each AF bin separately

```bash
# Built-in AF strata (not_in_gnomAD, AF=0, AF<1e-6, …, AF>=1e-2)
python evaluation/evaluate_filtered.py \
    --merged  results/predictions/ClinVar.260309only.seq12k/merged.tsv \
    --outdir  results/predictions/ClinVar.260309only.seq12k/eval_strat_af \
    --mode    stratify \
    --col     gnomAD4.1_joint_AF \
    --strata  builtin_af

# GERP strata
python evaluation/evaluate_filtered.py \
    --merged  results/predictions/ClinVar.260309only.seq12k/merged.tsv \
    --outdir  results/predictions/ClinVar.260309only.seq12k/eval_strat_gerp \
    --mode    stratify \
    --col     "GERP++_RS" \
    --strata  builtin_gerp

# (Optional) Custom bins
python evaluation/evaluate_filtered.py \
    --merged  results/predictions/ClinVar.260309only.seq12k/merged.tsv \
    --outdir  results/predictions/ClinVar.260309only.seq12k/eval_strat_custom \
    --mode    stratify \
    --col     gnomAD4.1_joint_AF \
    --strata  'None:1e-6,1e-6:1e-3,1e-3:None'
```

---

### Step 6 — Compare across strategies

```bash
# Compare named eval dirs
python evaluation/compare_strategies.py \
    --dirs \
        "all=results/predictions/ClinVar.260309only.seq12k/eval_all" \
        "rare_1e-3=results/predictions/ClinVar.260309only.seq12k/eval_rare_1e3" \
        "rare_1e-6=results/predictions/ClinVar.260309only.seq12k/eval_rare_1e6" \
        "conserved_gerp=results/predictions/ClinVar.260309only.seq12k/eval_conserved_gerp" \
    --outdir results/predictions/ClinVar.260309only.seq12k/comparison

# Summarize a stratify run (reads subdirectories automatically)
python evaluation/compare_strategies.py \
    --strat_dir results/predictions/ClinVar.260309only.seq12k/eval_strat_af \
    --outdir    results/predictions/ClinVar.260309only.seq12k/comparison_eval_strat_af
```

Outputs:
- `comparison_all_methods.tsv` — full long-format table
- `pivot_auroc.tsv`, `pivot_prauc.tsv` — method × stratum pivots
- `plots/grouped_bar_auroc.png` — grouped bar chart
- `plots/heatmap_auroc.png`     — methods × strata heatmap
- `plots/rank_chart_auroc.png`  — rank stability across strata

---

## Available conservation columns

Defined in `core/filters.py` → `CONSERVATION_COLS`:

| Column | Description |
|---|---|
| `phyloP100way_vertebrate` | PhyloP 100 vertebrates |
| `phyloP470way_mammalian`  | PhyloP 470 mammals |
| `phyloP17way_primate`     | PhyloP 17 primates |
| `phastCons100way_vertebrate` | PhastCons 100 vertebrates |
| `phastCons470way_mammalian`  | PhastCons 470 mammals |
| `phastCons17way_primate`     | PhastCons 17 primates |
| `GERP++_RS`               | GERP++ RS |
| `GERP++_NR`               | GERP++ NR |
| `GERP_92_mammals`         | GERP 92 mammals |
| `bStatistic`              | Background selection B-statistic |

---

## Key design decisions

- **Single input file**: all four ClinVar classes are concatenated once in step 1
  and scored once. Subsetting to P+B (or any class combination) is done at
  evaluation time via `--subset`.
- **Shared evaluation subset**: all methods are compared on the intersection of
  variants with valid REVEL + AlphaMissense scores (configurable via
  `--anchor_cols`), so no method is penalized for missing variants.
- **Score direction**: scores whose naive AUROC < 0.5 are automatically flipped
  (e.g. SIFT, where lower = more damaging).
- **Missing AF = ultra-rare**: in filter mode, variants absent from gnomAD are
  included by default (`--include_missing`; disable with `--no-include_missing`).
- **Multi-column AF filter**: passing multiple `--col` values requires a variant
  to be rare in *all* specified AF populations simultaneously.
- **Uniform scoring format**: `score_variants.py` (fine-tuned), `zeroshot_nt.py` (NT zero-shot),
  and `zeroshot_caduceus.py` (Caduceus zero-shot) all write identical TSV schemas so all
  scoring outputs are interchangeable as inputs to `merge_predictions.py` and downstream
  eval scripts. The scripts differ only in tokenization: NT uses 6-mers, Caduceus uses
  single characters.