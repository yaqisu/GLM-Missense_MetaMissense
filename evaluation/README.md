# Evaluation Pipeline

Modular, dataset-agnostic pipeline for evaluating fine-tuned genomic language
models against dbNSFP baseline methods on held-out ClinVar variants.

Four datasets are evaluated with this pipeline:

| Dataset key | Input source | Variant classes | Chromosomes | Fine-tuned scores? |
|---|---|---|---|---|
| `ClinVar.251103.BLBvsPLP` | `data/splits/` | PLP vs BLB | Val split only | No |
| `ClinVar.251103.BvsP` | Reuses scores from above | P vs B | Val split only | No |
| `ClinVar.260309only` | `data/sequences/` | PLP vs BLB | All chroms | Yes |
| `ClinVar.260309only.BvsP` | Reuses scores from above | P vs B | All chroms | Yes |

> **Why datasets 2 and 4 reuse scores**: `ClinVar.251103.BvsP` (P+B) is a strict
> subset of `ClinVar.251103.BLBvsPLP`, and `ClinVar.260309only.BvsP` is a strict
> subset of the full `ClinVar.260309only` run. Rather than re-running scoring,
> `extract_subset_ids.py` generates a variant ID filter that `evaluate.py`
> applies via `--subset`. Only Steps 1, 3, and 4 are needed for these datasets.

Steps 1–5 below walk through `ClinVar.251103.BLBvsPLP` as the primary demo.
See the subsequent sections for the three additional datasets.

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
├── zeroshot_nt.py              # Zero-shot NT-1/NT-2 scoring (6-mer tokenization)
├── zeroshot_caduceus.py        # Zero-shot Caduceus-PS/Ph scoring (single-char tokenization)
├── merge.py                    # Merge all model predictions + dbNSFP into one wide TSV (config-driven)
├── extract_subset_ids.py       # Generate subset IDs files (e.g. P+B only)
├── annotate_dbnsfp.py          # Annotate variants with dbNSFP via tabix
├── evaluate.py                 # Evaluate all methods — full, filtered, or stratified (config-driven highlighting)
└── compare_strategies.py       # Cross-strategy comparison plots & tables
```

> **Note**: `concat_sequences.py` has been removed — concatenating per-class
> sequence files is now handled automatically by `preprocessing/generate_datasets.sh`
> for all `split=no` rows in the config.

---

## Results directory layout

All prediction outputs for a dataset live in a single flat directory, regardless
of the sequence length used for scoring. dbNSFP annotations are sequence-length
agnostic and belong at the same level.

```
results/predictions/ClinVar.251103.BLBvsPLP/
├── merge_config.tsv                    ← config listing all models to merge (tracked in git)
├── dbnsfp.tsv                          ← dbNSFP annotations (seq-length agnostic)
├── zeroshot_NT2_seq12k.tsv
├── zeroshot_NT2_seq6k.tsv
├── zeroshot_NT1_seq6k.tsv
├── zeroshot_CaduceusPS_seq30k.tsv
├── zeroshot_CaduceusPh_seq30k.tsv
└── merged.tsv                          ← all scores + dbNSFP wide table (Step 3 output)
```

---

## Step-by-step usage (primary: ClinVar.251103.BLBvsPLP)

> **Note — no `score_variants.py`**: `ClinVar.251103` splits were used as the
> validation set during fine-tuning, so fine-tuned model scores are excluded
> to avoid data leakage. This dataset is zero-shot + dbNSFP only.
>
> **Note — no `extract_subset_ids.py`**: the BLBvsPLP validation split is
> already the correct subset. No ID filtering is needed here.

### Step 1 — Annotate variants with dbNSFP

dbNSFP annotations only need chromosome/position/ref/alt — they are
sequence-length agnostic. Any seq-size file works; we use seq12k. Run once per dataset.

```bash
# Download (one-time)
curl --http1.1 -C - -o data/dbnsfp/dbNSFP5.3.1a_grch38.gz \
    https://dist.genos.us/academic/yourcode/dbNSFP5.3.1a_grch38.gz
curl --http1.1 -C - -o data/dbnsfp/dbNSFP5.3.1a_grch38.gz.tbi \
    https://dist.genos.us/academic/yourcode/dbNSFP5.3.1a_grch38.gz.tbi

python evaluation/annotate_dbnsfp.py \
    --variants data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_validation.tsv \
    --dbnsfp   data/dbnsfp/dbNSFP5.3.1a_grch38.gz \
    --outdir   results/predictions/ClinVar.251103.BLBvsPLP
```

---

### Step 2 — Zero-shot scoring with pretrained genomic LMs

Scores variants using pretrained model backbones directly, without any fine-tuned
weights. Uses masked marginal log-likelihood ratio as the pathogenicity signal.

Two scripts are provided because NT and Caduceus have different tokenization:
- **`zeroshot_nt.py`**: 6-mer tokenization; `max_length` is read from
  `tokenizer.model_max_length` automatically (NT-1: 1000 tokens, NT-2: 2048 tokens).
- **`zeroshot_caduceus.py`**: single-character tokenization; no attention mask.
  Requires `mamba-ssm`: `pip install mamba-ssm --no-build-isolation`

Scores are written every 100 variants. If interrupted, rerun the same command
to resume — already-scored variants are skipped automatically by `variant_id`.

#### NT-1 and NT-2

```bash
# NT-2 (multi-species, 500M) — seq12k
python evaluation/zeroshot_nt.py \
    --input  data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_validation.tsv \
    --output results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_NT2_seq12k.tsv \
    --model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --gpu 2

# NT-2 — seq6k
python evaluation/zeroshot_nt.py \
    --input  data/splits/ClinVar.251103.missense.hg38.seq6k.BLBvsPLP_validation.tsv \
    --output results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_NT2_seq6k.tsv \
    --model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --gpu 2

# NT-1 (human-only, 500M) — seq6k
python evaluation/zeroshot_nt.py \
    --input  data/splits/ClinVar.251103.missense.hg38.seq6k.BLBvsPLP_validation.tsv \
    --output results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_NT1_seq6k.tsv \
    --model_name InstaDeepAI/nucleotide-transformer-500m-human-ref \
    --gpu 2
```

#### Caduceus-PS and Caduceus-Ph

```bash
# Caduceus-PS (reverse-complement equivariant) — seq30k
python evaluation/zeroshot_caduceus.py \
    --input  data/splits/ClinVar.251103.missense.hg38.seq30k.BLBvsPLP_validation.tsv \
    --output results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_CaduceusPS_seq30k.tsv \
    --model_name kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
    --gpu 2

# Caduceus-Ph (RC augmented) — seq30k
python evaluation/zeroshot_caduceus.py \
    --input  data/splits/ClinVar.251103.missense.hg38.seq30k.BLBvsPLP_validation.tsv \
    --output results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_CaduceusPh_seq30k.tsv \
    --model_name kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
    --gpu 2
```

**Output columns** (identical for both scripts):

| Column | Description |
|--------|-------------|
| `variant_id`, `chromosome`, `position`, `ref_allele`, `alt_allele` | Variant identifiers |
| `pathogenicity_score` | `sigmoid(-(log P(alt\|ctx) - log P(ref\|ctx)))`, 0–1, higher = more pathogenic |
| `predicted_label` | Binary call at `--threshold` (default 0.5) |
| `true_label` | From input `label` column if present |
| `log_p_alt`, `log_p_ref`, `log_likelihood_ratio` | Intermediate values, ignored by eval scripts |

---

### Step 3 — Merge all predictions into one wide TSV

A config file lists every model and the dbNSFP file to merge. Each model
contributes one `{label}_score` column; dbNSFP adds all its score columns as-is.
The first non-dbnsfp entry is the primary model — its key columns form the
base of the output.

**`merge_config.tsv`** (already committed at
`results/predictions/ClinVar.251103.BLBvsPLP/merge_config.tsv`):

```tsv
label	path	source	highlight
zeroshot_NT2_seq12k	results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_NT2_seq12k.tsv	zeroshot	no
zeroshot_NT2_seq6k	results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_NT2_seq6k.tsv	zeroshot	no
zeroshot_NT1_seq6k	results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_NT1_seq6k.tsv	zeroshot	no
zeroshot_CaduceusPS_seq30k	results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_CaduceusPS_seq30k.tsv	zeroshot	no
zeroshot_CaduceusPh_seq30k	results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_CaduceusPh_seq30k.tsv	zeroshot	no
dbnsfp	results/predictions/ClinVar.251103.BLBvsPLP/dbnsfp.tsv	dbnsfp	no
```

`source` must be `finetune`, `zeroshot`, or `dbnsfp`. Set `highlight=yes` for
any model you want highlighted in plots. Lines starting with `#` are comments.
To add a new model, add one line.

```bash
python evaluation/merge.py \
    --config results/predictions/ClinVar.251103.BLBvsPLP/merge_config.tsv \
    --output results/predictions/ClinVar.251103.BLBvsPLP/merged.tsv
# Outputs: merged.tsv, dbnsfp_column_coverage.tsv
```

---

### Step 4 — Evaluate

`evaluate.py` handles full, filtered, and stratified evaluation. Pass `--config`
pointing to `merge_config.tsv` so the script knows which models to highlight.

#### Full dataset evaluation

```bash
python evaluation/evaluate.py \
    --merged  results/predictions/ClinVar.251103.BLBvsPLP/merged.tsv \
    --config  results/predictions/ClinVar.251103.BLBvsPLP/merge_config.tsv \
    --outdir  results/predictions/ClinVar.251103.BLBvsPLP/eval_all
```

Outputs: `all_metrics.tsv`, `summary_comparison.tsv`, `shared_subset_summary.tsv`, `plots/`

#### Filter: keep rare variants only (AF < threshold)

```bash
python evaluation/evaluate.py \
    --merged    results/predictions/ClinVar.251103.BLBvsPLP/merged.tsv \
    --config    results/predictions/ClinVar.251103.BLBvsPLP/merge_config.tsv \
    --outdir    results/predictions/ClinVar.251103.BLBvsPLP/eval_rare_1e-3 \
    --mode      filter \
    --col       gnomAD4.1_joint_AF \
    --threshold 1e-3

python evaluation/evaluate.py \
    --merged    results/predictions/ClinVar.251103.BLBvsPLP/merged.tsv \
    --config    results/predictions/ClinVar.251103.BLBvsPLP/merge_config.tsv \
    --outdir    results/predictions/ClinVar.251103.BLBvsPLP/eval_rare_1e-6 \
    --mode      filter \
    --col       gnomAD4.1_joint_AF \
    --threshold 1e-6
```

#### Filter: highly conserved sites

```bash
python evaluation/evaluate.py \
    --merged    results/predictions/ClinVar.251103.BLBvsPLP/merged.tsv \
    --config    results/predictions/ClinVar.251103.BLBvsPLP/merge_config.tsv \
    --outdir    results/predictions/ClinVar.251103.BLBvsPLP/eval_conserved_phylop \
    --mode      filter \
    --col       phyloP100way_vertebrate \
    --threshold 3.0 \
    --direction above

python evaluation/evaluate.py \
    --merged    results/predictions/ClinVar.251103.BLBvsPLP/merged.tsv \
    --config    results/predictions/ClinVar.251103.BLBvsPLP/merge_config.tsv \
    --outdir    results/predictions/ClinVar.251103.BLBvsPLP/eval_conserved_gerp \
    --mode      filter \
    --col       "GERP++_RS" \
    --threshold 4.0 \
    --direction above
```

#### Stratify: evaluate each bin separately

```bash
python evaluation/evaluate.py \
    --merged  results/predictions/ClinVar.251103.BLBvsPLP/merged.tsv \
    --config  results/predictions/ClinVar.251103.BLBvsPLP/merge_config.tsv \
    --outdir  results/predictions/ClinVar.251103.BLBvsPLP/eval_strat_af \
    --mode    stratify \
    --col     gnomAD4.1_joint_AF \
    --strata  builtin_af

python evaluation/evaluate.py \
    --merged  results/predictions/ClinVar.251103.BLBvsPLP/merged.tsv \
    --config  results/predictions/ClinVar.251103.BLBvsPLP/merge_config.tsv \
    --outdir  results/predictions/ClinVar.251103.BLBvsPLP/eval_strat_gerp \
    --mode    stratify \
    --col     "GERP++_RS" \
    --strata  builtin_gerp
```

---

### Step 5 — Compare across strategies

```bash
python evaluation/compare_strategies.py \
    --dirs \
        "all=results/predictions/ClinVar.251103.BLBvsPLP/eval_all" \
        "rare_1e-3=results/predictions/ClinVar.251103.BLBvsPLP/eval_rare_1e-3" \
        "rare_1e-6=results/predictions/ClinVar.251103.BLBvsPLP/eval_rare_1e-6" \
        "conserved_gerp=results/predictions/ClinVar.251103.BLBvsPLP/eval_conserved_gerp" \
    --outdir results/predictions/ClinVar.251103.BLBvsPLP/comparison

python evaluation/compare_strategies.py \
    --strat_dir results/predictions/ClinVar.251103.BLBvsPLP/eval_strat_af \
    --outdir    results/predictions/ClinVar.251103.BLBvsPLP/comparison_strat_af
```

Outputs:
- `comparison_all_methods.tsv` — full long-format table
- `pivot_auroc.tsv`, `pivot_prauc.tsv` — method × stratum pivots
- `plots/grouped_bar_auroc.png`, `plots/heatmap_auroc.png`, `plots/rank_chart_auroc.png`

---

## Dataset 2: ClinVar.251103.BvsP (P vs B, val split only)

> **Note — scores reused from Dataset 1**: `ClinVar.251103.BvsP` is a strict
> subset of `ClinVar.251103.BLBvsPLP`. There is no need to re-run dbNSFP
> annotation or zero-shot scoring — the prediction TSVs are shared. The
> `merge_config.tsv` for this dataset points directly at the same files.
> Only subset ID generation, merging, and evaluation are needed.

### Step 1 — Generate subset IDs file

```bash
python evaluation/extract_subset_ids.py \
    --dataset ClinVar.251103.BvsP \
    --datadir data/splits \
    --outfile data/splits/ClinVar.251103.missense.hg38.bvsp_ids.tsv
```

### Step 2 — Merge

**`merge_config.tsv`** (already committed at
`results/predictions/ClinVar.251103.BvsP/merge_config.tsv`):

```tsv
label	path	source	highlight
zeroshot_NT2_seq12k	results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_NT2_seq12k.tsv	zeroshot	no
zeroshot_NT2_seq6k	results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_NT2_seq6k.tsv	zeroshot	no
zeroshot_NT1_seq6k	results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_NT1_seq6k.tsv	zeroshot	no
zeroshot_CaduceusPS_seq30k	results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_CaduceusPS_seq30k.tsv	zeroshot	no
zeroshot_CaduceusPh_seq30k	results/predictions/ClinVar.251103.BLBvsPLP/zeroshot_CaduceusPh_seq30k.tsv	zeroshot	no
dbnsfp	results/predictions/ClinVar.251103.BLBvsPLP/dbnsfp.tsv	dbnsfp	no
```

```bash
python evaluation/merge.py \
    --config results/predictions/ClinVar.251103.BvsP/merge_config.tsv \
    --output results/predictions/ClinVar.251103.BvsP/merged.tsv
```

### Step 3 — Evaluate

```bash
python evaluation/evaluate.py \
    --merged  results/predictions/ClinVar.251103.BvsP/merged.tsv \
    --config  results/predictions/ClinVar.251103.BvsP/merge_config.tsv \
    --outdir  results/predictions/ClinVar.251103.BvsP/eval_all \
    --subset  data/splits/ClinVar.251103.missense.hg38.bvsp_ids.tsv
```

Filtered and stratified evaluation follow the same flags as shown for
`ClinVar.251103.BLBvsPLP` above — just substitute the dataset paths and always
pass `--subset data/splits/ClinVar.251103.missense.hg38.bvsp_ids.tsv`.

---

## Dataset 3: ClinVar.260309only (PLP vs BLB, all chroms)

This dataset uses sequences from `data/sequences/` covering all chromosomes,
and includes fine-tuned model scores alongside zero-shot baselines.

> **Note — concatenated sequence files**: `generate_datasets.sh` has already
> produced `ClinVar.260309only.missense.hg38.seq{size}.tsv` by concatenating
> the per-class `.bed.seq{size}.tsv` files. Use these concatenated files as
> input to all scoring scripts below.

### Step 1 — Generate subset IDs file

Needed now so that Dataset 4 (P+B subset) can reuse it without re-running scoring.

```bash
python evaluation/extract_subset_ids.py \
    --dataset ClinVar.260309only \
    --datadir data/sequences \
    --outfile data/sequences/ClinVar.260309only.missense.hg38.pb_ids.tsv
```

### Step 2a — Score variants with our fine-tuned model

```bash
python scoring/score_variants.py \
    --input  data/sequences/ClinVar.260309only.missense.hg38.seq12k.tsv \
    --model  scoring/model/best_model.pt \
    --output results/predictions/ClinVar.260309only/finetune_NT2_seq12k.tsv
```

### Step 2b — Annotate variants with dbNSFP

```bash
python evaluation/annotate_dbnsfp.py \
    --variants data/sequences/ClinVar.260309only.missense.hg38.seq12k.tsv \
    --dbnsfp   data/dbnsfp/dbNSFP5.3.1a_grch38.gz \
    --outdir   results/predictions/ClinVar.260309only
```

### Step 2c — Zero-shot scoring

#### NT-1 and NT-2

```bash
# NT-2 (multi-species, 500M) — seq12k
python evaluation/zeroshot_nt.py \
    --input  data/sequences/ClinVar.260309only.missense.hg38.seq12k.tsv \
    --output results/predictions/ClinVar.260309only/zeroshot_NT2_seq12k.tsv \
    --model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --gpu 2

# NT-2 — seq6k
python evaluation/zeroshot_nt.py \
    --input  data/sequences/ClinVar.260309only.missense.hg38.seq6k.tsv \
    --output results/predictions/ClinVar.260309only/zeroshot_NT2_seq6k.tsv \
    --model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --gpu 2

# NT-1 (human-only, 500M) — seq6k
python evaluation/zeroshot_nt.py \
    --input  data/sequences/ClinVar.260309only.missense.hg38.seq6k.tsv \
    --output results/predictions/ClinVar.260309only/zeroshot_NT1_seq6k.tsv \
    --model_name InstaDeepAI/nucleotide-transformer-500m-human-ref \
    --gpu 2
```

#### Caduceus-PS and Caduceus-Ph

```bash
# Caduceus-PS (reverse-complement equivariant) — seq30k
python evaluation/zeroshot_caduceus.py \
    --input  data/sequences/ClinVar.260309only.missense.hg38.seq30k.tsv \
    --output results/predictions/ClinVar.260309only/zeroshot_CaduceusPS_seq30k.tsv \
    --model_name kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
    --gpu 2

# Caduceus-Ph (RC augmented) — seq30k
python evaluation/zeroshot_caduceus.py \
    --input  data/sequences/ClinVar.260309only.missense.hg38.seq30k.tsv \
    --output results/predictions/ClinVar.260309only/zeroshot_CaduceusPh_seq30k.tsv \
    --model_name kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
    --gpu 2
```

### Step 3 — Merge

**`merge_config.tsv`** (already committed at
`results/predictions/ClinVar.260309only/merge_config.tsv`):

```tsv
label	path	source	highlight
finetune_NT2_seq12k	results/predictions/ClinVar.260309only/finetune_NT2_seq12k.tsv	finetune	yes
zeroshot_NT2_seq12k	results/predictions/ClinVar.260309only/zeroshot_NT2_seq12k.tsv	zeroshot	no
zeroshot_NT2_seq6k	results/predictions/ClinVar.260309only/zeroshot_NT2_seq6k.tsv	zeroshot	no
zeroshot_NT1_seq6k	results/predictions/ClinVar.260309only/zeroshot_NT1_seq6k.tsv	zeroshot	no
zeroshot_CaduceusPS_seq30k	results/predictions/ClinVar.260309only/zeroshot_CaduceusPS_seq30k.tsv	zeroshot	no
zeroshot_CaduceusPh_seq30k	results/predictions/ClinVar.260309only/zeroshot_CaduceusPh_seq30k.tsv	zeroshot	no
dbnsfp	results/predictions/ClinVar.260309only/dbnsfp.tsv	dbnsfp	no
```

```bash
python evaluation/merge.py \
    --config results/predictions/ClinVar.260309only/merge_config.tsv \
    --output results/predictions/ClinVar.260309only/merged.tsv
# Outputs: merged.tsv, dbnsfp_column_coverage.tsv
```

### Step 4 — Evaluate

#### Full dataset evaluation (all four classes)

```bash
python evaluation/evaluate.py \
    --merged  results/predictions/ClinVar.260309only/merged.tsv \
    --config  results/predictions/ClinVar.260309only/merge_config.tsv \
    --outdir  results/predictions/ClinVar.260309only/eval_all
```

Filtered and stratified evaluation follow the same flags as shown for
`ClinVar.251103.BLBvsPLP` above — just substitute the dataset paths.

### Step 5 — Compare across strategies

```bash
python evaluation/compare_strategies.py \
    --dirs \
        "all=results/predictions/ClinVar.260309only/eval_all" \
        "rare_1e-3=results/predictions/ClinVar.260309only/eval_rare_1e-3" \
        "rare_1e-6=results/predictions/ClinVar.260309only/eval_rare_1e-6" \
        "conserved_gerp=results/predictions/ClinVar.260309only/eval_conserved_gerp" \
    --outdir results/predictions/ClinVar.260309only/comparison

python evaluation/compare_strategies.py \
    --strat_dir results/predictions/ClinVar.260309only/eval_strat_af \
    --outdir    results/predictions/ClinVar.260309only/comparison_strat_af
```

---

## Dataset 4: ClinVar.260309only.BvsP (P vs B, all chroms)

> **Note — scores reused from Dataset 3**: `ClinVar.260309only.BvsP` is a strict
> subset of the full `ClinVar.260309only` run. The subset IDs file was already
> generated in Dataset 3 Step 1. The `merge_config.tsv` for this dataset points
> directly at the same prediction TSVs. Only merging and evaluation are needed.

### Step 1 — Merge

**`merge_config.tsv`** (already committed at
`results/predictions/ClinVar.260309only.BvsP/merge_config.tsv`):

```tsv
label	path	source	highlight
finetune_NT2_seq12k	results/predictions/ClinVar.260309only/finetune_NT2_seq12k.tsv	finetune	yes
zeroshot_NT2_seq12k	results/predictions/ClinVar.260309only/zeroshot_NT2_seq12k.tsv	zeroshot	no
zeroshot_NT2_seq6k	results/predictions/ClinVar.260309only/zeroshot_NT2_seq6k.tsv	zeroshot	no
zeroshot_NT1_seq6k	results/predictions/ClinVar.260309only/zeroshot_NT1_seq6k.tsv	zeroshot	no
zeroshot_CaduceusPS_seq30k	results/predictions/ClinVar.260309only/zeroshot_CaduceusPS_seq30k.tsv	zeroshot	no
zeroshot_CaduceusPh_seq30k	results/predictions/ClinVar.260309only/zeroshot_CaduceusPh_seq30k.tsv	zeroshot	no
dbnsfp	results/predictions/ClinVar.260309only/dbnsfp.tsv	dbnsfp	no
```

```bash
python evaluation/merge.py \
    --config results/predictions/ClinVar.260309only.BvsP/merge_config.tsv \
    --output results/predictions/ClinVar.260309only.BvsP/merged.tsv
```

### Step 2 — Evaluate

```bash
python evaluation/evaluate.py \
    --merged  results/predictions/ClinVar.260309only.BvsP/merged.tsv \
    --config  results/predictions/ClinVar.260309only.BvsP/merge_config.tsv \
    --outdir  results/predictions/ClinVar.260309only.BvsP/eval_all \
    --subset  data/sequences/ClinVar.260309only.missense.hg38.pb_ids.tsv
```

Filtered and stratified evaluation follow the same flags as shown for
`ClinVar.251103.BLBvsPLP` above — just substitute the dataset paths and always
pass `--subset data/sequences/ClinVar.260309only.missense.hg38.pb_ids.tsv`.

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

- **Sequence files produced by preprocessing**: `generate_datasets.sh` now
  concatenates all per-class TSVs for `split=no` rows automatically. No
  separate concat step needed in evaluation; `concat_sequences.py` has been removed.
- **Full filename convention**: sequence files use the complete name
  `ClinVar.260309only.missense.hg38.seq{size}.tsv` consistently. Subset IDs
  follow the same prefix: `ClinVar.260309only.missense.hg38.pb_ids.tsv`.
- **Flat predictions directory**: all scoring outputs for a dataset live under
  one directory regardless of sequence length. dbNSFP annotations are
  seq-length agnostic and live here too.
- **Score reuse across subset datasets**: datasets 2 and 4 point their
  `merge_config.tsv` directly at the prediction TSVs of their parent dataset
  (1 and 3 respectively). Subset filtering happens at evaluation time via
  `--subset`, not at scoring time.
- **dbNSFP is seq-length agnostic**: `annotate_dbnsfp.py` only uses
  chromosome/position/ref/alt. Any seq-size input file works; seq12k is used
  by convention. Run once per source dataset.
- **Single evaluation script**: `evaluate.py` replaces `evaluate_all.py` and
  `evaluate_filtered.py`. Omit `--mode` for full evaluation; pass `--mode filter`
  or `--mode stratify` for filtered/stratified evaluation. `--config` is always
  required and points to `merge_config.tsv`.
- **Highlighted models always shown**: models with `highlight=yes` in `merge_config.tsv`
  are plotted in bold and always included in figures, even if they don't rank in
  the top N dbNSFP methods. The primary model for evaluation is the first
  highlighted model in the config.
- **Config-driven merge**: `merge.py` reads `merge_config.tsv` listing every
  model and dbNSFP file, with `source` (`finetune`, `zeroshot`, `dbnsfp`) and
  `highlight` (`yes`/`no`) columns. Adding a new model requires only one new line.
  Label convention: `{finetune|zeroshot}_{model}_{seqsize}` — these become score
  column names in `merged.tsv` and all downstream outputs.
- **Shared evaluation subset**: all methods are compared on the intersection of
  variants with valid REVEL + AlphaMissense scores (configurable via
  `--anchor_cols`), so no method is penalized for missing variants.
- **Score direction**: scores whose naive AUROC < 0.5 are automatically flipped
  (e.g. SIFT, where lower = more damaging).
- **Missing AF = ultra-rare**: in filter mode, variants absent from gnomAD are
  included by default (`--include_missing`; disable with `--no-include_missing`).