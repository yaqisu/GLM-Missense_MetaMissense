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
├── prepare_metamissense_input.py    # Merge GLM-Missense scores + dbNSFP → MetaMissense input
├── MetaMissense.py                  # Score variants with the MetaMissense ensemble model
└── MetaMissense.joblib              # Ensemble model weights
```

---

---

# Model 1 — GLM-Missense

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

The model weights (`GLM-Missense.pt`) are not tracked in git due to file
size. A download link will be provided upon publication. Place the file
directly under `scoring/` after downloading.

---

## GLM-Missense step 1 — Prepare input sequences

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

## GLM-Missense step 2 — Score variants

```bash
python scoring/GLM-Missense.py \
    --input  results/scoring/my_variants.seq12k.tsv \
    --model  scoring/GLM-Missense.pt \
    --output results/scoring/GLM-Missense.tsv
```

Both `ref_sequence` and `alt_sequence` must be present in the input (produced
by Step 1). The optional `label` column triggers AUC logging if present.

**Output** (`GLM-Missense.tsv`):

| Column | Description |
|---|---|
| `GLM-Missense_score` | Sigmoid probability (0–1), higher = more pathogenic |
| `predicted_label` | Binary call at threshold (default 0.5) |
| `true_label` | Only present if `label` was in the input |

### All GLM-Missense.py arguments

| Argument | Default | Description |
|---|---|---|
| `--input`, `-i` | required | Input seq12k TSV |
| `--model`, `-m` | required | Path to `GLM-Missense.pt` |
| `--output`, `-o` | required | Output TSV path |
| `--batch_size`, `-b` | 128 | Inference batch size |
| `--gpu`, `-g` | 0 | GPU id (−1 for CPU) |
| `--threshold`, `-t` | 0.5 | Threshold for `predicted_label` |
| `--k` | 6 | K-mer size for tokenization |

> **If you only need GLM-Missense scores, you are done.**
> The steps below are only required to run MetaMissense.

---

---

# Model 2 — MetaMissense

An **ensemble model** that stacks the GLM-Missense score with six established
predictors (AlphaMissense, ESM1b, REVEL, CADD, PolyPhen-2, SIFT) using a
trained stacking classifier. It requires the GLM-Missense score and those six
dbNSFP columns as input — all fetched in the steps below.

---

## MetaMissense step 1 — Annotate with dbNSFP and merge

```bash
python scoring/prepare_metamissense_input.py \
    --glm    results/scoring/GLM-Missense.tsv \
    --dbnsfp data/dbnsfp/dbNSFP5.3.1a_grch38.gz \
    --outdir results/scoring
```

This wrapper does two things in sequence:

1. Calls `annotate_dbnsfp.py` to fetch **all** dbNSFP columns for your
   variants via tabix, writing `results/scoring/dbnsfp.tsv`. MetaMissense
   only uses six of those columns (`AlphaMissense_score`, `ESM1b_score`, 
   `REVEL_score`, `CADD_phred`, `Polyphen2_HVAR_score`, `SIFT_score`), 
   but the full table is kept so you can use any other dbNSFP
   scores for your own analyses without re-running the annotation.
2. Merges `dbnsfp.tsv` with `GLM-Missense.tsv` on the variant key columns,
   writing `results/scoring/MetaMissense_input.tsv`.

The dbNSFP annotation step is skipped automatically if `dbnsfp.tsv` already
exists in `--outdir`. Pass `--force` to re-run it.

You can obtain the dbNSFP database and tabix index from the dbNSFP project
page. We used dbNSFP v5.3.1a (GRCh38 build). Run from the repo root.

---

## MetaMissense step 2 — Score with MetaMissense

```bash
python scoring/MetaMissense.py \
    results/scoring/MetaMissense_input.tsv \
    scoring/MetaMissense.joblib \
    --output results/scoring/MetaMissense.tsv
```

The output adds a `MetaMissense_score` column (0–1 scale, where higher values 
indicate greater predicted pathogenicity) to the input table. 
Rows where any of the six required predictor scores is missing will
receive `NaN`.

---

## Notes

- Output directories are created automatically if they do not exist.
- `GLM-Missense.pt` is gitignored; place them under `scoring/` after downloading.