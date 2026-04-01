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
results/predictions/ClinVar.260309only/
├── merge_config.tsv                    ← config listing all models to merge (tracked in git)
├── dbnsfp.tsv                          ← dbNSFP annotations (seq-length agnostic)
├── finetune_NT2_seq12k.tsv             ← fine-tuned model predictions (seq12k input)
├── zeroshot_NT2_seq12k.tsv
├── zeroshot_NT2_seq6k.tsv
├── zeroshot_NT1_seq6k.tsv
├── zeroshot_CaduceusPS_seq30k.tsv
├── zeroshot_CaduceusPh_seq30k.tsv
└── merged.tsv                          ← all scores + dbNSFP wide table (Step 3 output)
```

---

## Step-by-step usage (example: ClinVar.260309only)

### Step 1 — Generate subset IDs file

Sequence files are already produced and concatenated by
`preprocessing/generate_datasets.sh`. The only step needed here is generating
the P+B subset IDs file for evaluation.

```bash
python evaluation/extract_subset_ids.py \
    --dataset ClinVar.260309only \
    --datadir data/sequences \
    --outfile data/sequences/ClinVar.260309only.missense.hg38.pb_ids.tsv
```

---

### Step 2a — Score variants with our fine-tuned model

Name the output file with a descriptive label — this becomes the score column
name in the merged table (`finetune_NT2_seq12k_score`).

```bash
python scoring/score_variants.py \
    --input  data/sequences/ClinVar.260309only.missense.hg38.seq12k.tsv \
    --model  scoring/model/best_model.pt \
    --output results/predictions/ClinVar.260309only/finetune_NT2_seq12k.tsv
```

---

### Step 2b — Annotate variants with dbNSFP

dbNSFP annotations only need chromosome/position/ref/alt — they are
sequence-length agnostic. Run once per dataset.

```bash
# Download (one-time)
curl --http1.1 -C - -o data/dbnsfp/dbNSFP5.3.1a_grch38.gz \
    https://dist.genos.us/academic/yourcode/dbNSFP5.3.1a_grch38.gz
curl --http1.1 -C - -o data/dbnsfp/dbNSFP5.3.1a_grch38.gz.tbi \
    https://dist.genos.us/academic/yourcode/dbNSFP5.3.1a_grch38.gz.tbi

python evaluation/annotate_dbnsfp.py \
    --variants data/sequences/ClinVar.260309only.missense.hg38.seq12k.tsv \
    --dbnsfp   data/dbnsfp/dbNSFP5.3.1a_grch38.gz \
    --outdir   results/predictions/ClinVar.260309only
```

---

### Step 2c — Zero-shot scoring with pretrained genomic LMs

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

**Create the config** (save as `results/predictions/ClinVar.260309only/merge_config.tsv`,
track in git):

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

`source` must be `finetune`, `zeroshot`, or `dbnsfp`. Set `highlight=yes` for
any model you want highlighted in plots — typically your fine-tuned models. If
the `highlight` column is omitted, all `finetune` rows are highlighted by default.
Lines starting with `#` are comments. To add a new model, add one line.

**Run the merge**:

```bash
python evaluation/merge.py \
    --config results/predictions/ClinVar.260309only/merge_config.tsv \
    --output results/predictions/ClinVar.260309only/merged.tsv
# Outputs: merged.tsv, dbnsfp_column_coverage.tsv
```

Downstream eval scripts read `merge_config.tsv` directly via `--config` to
know which columns to highlight — no intermediate JSON file needed.

---

### Step 4 — Evaluate

A single script handles full evaluation, filtered evaluation, and stratified
evaluation. Pass `--config` pointing to `merge_config.tsv` so the script knows
which models to highlight. Models with `highlight=yes` always appear in plots
in bold, regardless of whether they rank in the top N.

#### Full dataset evaluation

```bash
# All four classes
python evaluation/evaluate.py \
    --merged  results/predictions/ClinVar.260309only/merged.tsv \
    --config  results/predictions/ClinVar.260309only/merge_config.tsv \
    --outdir  results/predictions/ClinVar.260309only/eval_all

# Pathogenic + benign only (P+B subset)
python evaluation/evaluate.py \
    --merged  results/predictions/ClinVar.260309only/merged.tsv \
    --config  results/predictions/ClinVar.260309only/merge_config.tsv \
    --outdir  results/predictions/ClinVar.260309only/eval_pb \
    --subset  data/sequences/ClinVar.260309only.missense.hg38.pb_ids.tsv
```

Outputs: `all_metrics.tsv`, `summary_comparison.tsv`, `shared_subset_summary.tsv`, `plots/`

#### Filter: keep rare variants only (AF < threshold)

```bash
python evaluation/evaluate.py \
    --merged    results/predictions/ClinVar.260309only/merged.tsv \
    --config    results/predictions/ClinVar.260309only/merge_config.tsv \
    --outdir    results/predictions/ClinVar.260309only/eval_rare_1e-3 \
    --mode      filter \
    --col       gnomAD4.1_joint_AF \
    --threshold 1e-3

python evaluation/evaluate.py \
    --merged    results/predictions/ClinVar.260309only/merged.tsv \
    --config    results/predictions/ClinVar.260309only/merge_config.tsv \
    --outdir    results/predictions/ClinVar.260309only/eval_rare_1e-6 \
    --mode      filter \
    --col       gnomAD4.1_joint_AF \
    --threshold 1e-6
```

#### Filter: highly conserved sites

```bash
python evaluation/evaluate.py \
    --merged    results/predictions/ClinVar.260309only/merged.tsv \
    --config    results/predictions/ClinVar.260309only/merge_config.tsv \
    --outdir    results/predictions/ClinVar.260309only/eval_conserved_phylop \
    --mode      filter \
    --col       phyloP100way_vertebrate \
    --threshold 3.0 \
    --direction above

python evaluation/evaluate.py \
    --merged    results/predictions/ClinVar.260309only/merged.tsv \
    --config    results/predictions/ClinVar.260309only/merge_config.tsv \
    --outdir    results/predictions/ClinVar.260309only/eval_conserved_gerp \
    --mode      filter \
    --col       "GERP++_RS" \
    --threshold 4.0 \
    --direction above
```

#### Stratify: evaluate each bin separately

```bash
python evaluation/evaluate.py \
    --merged  results/predictions/ClinVar.260309only/merged.tsv \
    --config  results/predictions/ClinVar.260309only/merge_config.tsv \
    --outdir  results/predictions/ClinVar.260309only/eval_strat_af \
    --mode    stratify \
    --col     gnomAD4.1_joint_AF \
    --strata  builtin_af

python evaluation/evaluate.py \
    --merged  results/predictions/ClinVar.260309only/merged.tsv \
    --config  results/predictions/ClinVar.260309only/merge_config.tsv \
    --outdir  results/predictions/ClinVar.260309only/eval_strat_gerp \
    --mode    stratify \
    --col     "GERP++_RS" \
    --strata  builtin_gerp
```

---

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

Outputs:
- `comparison_all_methods.tsv` — full long-format table
- `pivot_auroc.tsv`, `pivot_prauc.tsv` — method × stratum pivots
- `plots/grouped_bar_auroc.png`, `plots/heatmap_auroc.png`, `plots/rank_chart_auroc.png`

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
  one directory (e.g. `results/predictions/ClinVar.260309only/`) regardless of
  sequence length. dbNSFP annotations are seq-length agnostic and live here too.
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