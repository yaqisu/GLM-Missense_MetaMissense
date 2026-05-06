# Scoring

This directory contains everything needed to score new variants with our two
pathogenicity models: **GLM-Missense** and **MetaMissense**.

---

## Directory layout

```
scoring/
├── prepare_glm_input.py             # Convert a variant TSV → seq12k input for GLM-Missense
├── GLM-Missense.py                  # Score variants with the fine-tuned genomic LM
├── GLM-Missense.pt                  # Model weights (not tracked in git — download separately)
├── annotate_dbnsfp.py               # Annotate variants with all dbNSFP scores via tabix
├── prepare_metamissense_input.py    # Prepare input for MetaMissense (fetch + merge predictor scores)
├── MetaMissense.py                  # Score variants with the MetaMissense ensemble model
└── MetaMissense.joblib              # Ensemble model weights
```

---

---

# GLM-Missense

A **Siamese Ref-Alt Contrast** model that independently encodes the reference
and alternate allele sequences through a shared Nucleotide Transformer v2
(500 M) backbone fine-tuned with LoRA, then contrasts the two representations
at the variant position.

| Setting | Value |
|---|---|
| Architecture | NT2_RefAltContrast (Siamese) |
| Fine-tuning | LoRA (rank=32) |
| Projector head | MLPProjector (1024 → 256-d) |
| Classifier head | 2-layer MLPClassifierHead |
| Combine mode | `concat_diff`: [ref, alt, ref − alt] |
| Embedding strategy | Variant-position token (token 1000) |
| Training data | BvsP ClinVar missense variants |
| Sequence length | 12 kb (5,999 bp flanking each side) |

```
ref_sequence → NT2+LoRA → token[1000] → MLPProjector → ref_feat ─┐
                                                                    ├─ concat_diff → MLPHead → logit
alt_sequence → NT2+LoRA → token[1000] → MLPProjector → alt_feat ─┘
```

`GLM-Missense_score` is a probability (0–1) of a variant belonging to the
pathogenic/likely pathogenic class, where higher scores indicate greater
predicted pathogenicity.

The model weights (`GLM-Missense.pt`) are not tracked in git due to file
size. A download link will be provided upon publication. Place the file
directly under `scoring/` after downloading.

---

## Step 1 — Prepare input sequences

Run `prepare_glm_input.py` to generate sequences from your variant table
and the GRCh38 reference genome. Your input requires at minimum four columns:

| Column | Description |
|---|---|
| `chromosome` | Chromosome, e.g. `1`, `X`, or `chr1` |
| `position` | 1-based genomic position |
| `ref_allele` | Reference allele |
| `alt_allele` | Alternate allele |

Two additional columns are passed through if present:

| Column | Description |
|---|---|
| `variant_id` | Unique identifier; auto-generated as `{chrom}:{pos}:{ref}>{alt}` if absent |
| `label` | Integer class label (0 = benign, 1 = pathogenic); enables AUC logging in Step 2 |

```bash
python scoring/prepare_glm_input.py \
    --input  my_variants.tsv \
    --output results/scoring/my_variants.seq12k.tsv \
    --genome data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

The reference genome (GRCh38) must be available at the path given by
`--genome` (default: `data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa`).
The script uses `pyfaidx` for fast random-access sequence extraction.

**Output TSV columns:**

```
variant_id   chromosome   position   ref_allele   alt_allele
upstream_flank   downstream_flank   ref_sequence   alt_sequence
label   (only if label column was present in input)
```

---

## Step 2 — Score variants

```bash
python scoring/GLM-Missense.py \
    --input   results/scoring/my_variants.seq12k.tsv \
    --model   scoring/GLM-Missense.pt \
    --outdir  results/scoring
```

Both `ref_sequence` and `alt_sequence` must be present in the input (produced
by Step 1). If a `label` column is present, AUC is computed and logged to
stdout but not written to the output file.

**Output** (`results/scoring/GLM-Missense.tsv`):

| Column | Description |
|---|---|
| `GLM-Missense_score` | Probability (0–1) of pathogenic/likely pathogenic |

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--input`, `-i` | required | Input seq12k TSV |
| `--model`, `-m` | required | Path to `GLM-Missense.pt` |
| `--outdir`, `-o` | required | Output directory |
| `--batch_size`, `-b` | 128 | Inference batch size |
| `--gpu`, `-g` | 0 | GPU id (−1 for CPU) |

> **If you only need GLM-Missense scores, you are done.**
> The steps below are only required to run MetaMissense.

---

---

# MetaMissense

An **ensemble model** that stacks the GLM-Missense score with six established
predictors (AlphaMissense, ESM1b, REVEL, CADD, PolyPhen-2, SIFT) using a
trained stacking classifier.

`MetaMissense_score` is a probability (0–1) of a variant belonging to the
pathogenic/likely pathogenic class. As with GLM-Missense, scores ≥ 0.5
are predicted pathogenic/likely pathogenic and scores below 0.5 are predicted
benign/likely benign.

---

## Step 1 — Prepare predictor scores as MetaMissense input

MetaMissense requires the GLM-Missense score plus six predictor scores from
external sources: `AlphaMissense_score`, `ESM1b_score`, `REVEL_score`,
`CADD_phred`, `Polyphen2_HVAR_score`, and `SIFT_score`.

We provide `prepare_metamissense_input.py` as a convenience wrapper that
fetches these scores from dbNSFP and merges them with your GLM-Missense
output into a single input file. If you already have these scores from
another source, you can skip this wrapper and format them manually — just
make sure the column names match those listed above. If your column names
differ, update the `input_aliases` dict at the top of `MetaMissense.py`
accordingly.

**Using the wrapper (recommended):**

```bash
python scoring/prepare_metamissense_input.py \
    --glm    results/scoring/GLM-Missense.tsv \
    --dbnsfp data/dbnsfp/dbNSFP5.3.1a_grch38.gz \
    --outdir results/scoring
```

This does two things in sequence:

1. Calls `annotate_dbnsfp.py` to fetch **all** dbNSFP columns for your
   variants via tabix, writing `results/scoring/dbnsfp.tsv`. The full table
   is kept so you can use any other dbNSFP scores for your own analyses
   without re-running the annotation.
2. Merges `dbnsfp.tsv` with `GLM-Missense.tsv` on the variant key columns,
   writing `results/scoring/MetaMissense_input.tsv`.

The dbNSFP annotation step is skipped automatically if `dbnsfp.tsv` already
exists in `--outdir`. Pass `--force` to re-run it.

You can obtain the dbNSFP database and tabix index from the dbNSFP project
page. We used dbNSFP v5.3.1a (GRCh38 build). Run from the repo root.

---

## Step 2 — Score with MetaMissense

```bash
python scoring/MetaMissense.py \
    --input   results/scoring/MetaMissense_input.tsv \
    --model   scoring/MetaMissense.joblib \
    --outdir  results/scoring
```

Output is written to `results/scoring/MetaMissense.tsv`. The output adds a
`MetaMissense_score` column immediately after `GLM-Missense_score`. Rows
where any of the six required predictor scores is missing will receive `NaN`.

---

## Notes

- Output directories are created automatically if they do not exist.
- `GLM-Missense.pt` is gitignored; place it under `scoring/` after downloading.