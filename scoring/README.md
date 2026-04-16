# Scoring

This directory contains the script and model for scoring new variants using our best fine-tuned model.

---

## Contents

```
scoring/
├── score_variants.py       # Main scoring script
└── model/                  # Model weights and config (not tracked in git)
    ├── best_model.pt        # Fine-tuned model weights
    └── model_config.json    # Model architecture config
```

---

## Model

Our best model is a **Siamese Ref-Alt Contrast** model using **Nucleotide Transformer v2 (500M)** as the shared backbone.

| Setting | Value |
|---------|-------|
| Architecture | NT2_RefAltContrast (Siamese) |
| Fine-tuning | LoRA (rank=32) |
| Projector head | MLPProjector (1024 → 256-d) |
| Classifier head | 2-layer MLPClassifierHead |
| Combine mode | `concat_diff`: [ref, alt, ref − alt] |
| Embedding strategy | Variant-position token (token 1000) |
| Training data | BvsP (benign + pathogenic) |
| Sequence length | 12k bp (5,999 bp flanking each side) |

The model weights (`best_model.pt`) are not tracked in git due to file size. A download link will be provided upon publication.

### Architecture overview

Both `ref_sequence` and `alt_sequence` are encoded independently through a **shared** NT2+LoRA backbone. The token at the variant position (token 1000 for 12k sequences with k=6 k-merization) is extracted from each arm and projected down to 256-d via a shared MLPProjector. The two projected features are then combined as `[ref_feat, alt_feat, ref_feat − alt_feat]` and passed to a 2-layer MLP classification head.

```
ref_sequence → NT2+LoRA → token[1000] → MLPProjector → ref_feat ─┐
                                                                    ├─ concat_diff → MLPHead → logit
alt_sequence → NT2+LoRA → token[1000] → MLPProjector → alt_feat ─┘
```

---

## Input Format

The input TSV must have the following columns (same format as files in `data/splits/`):

```
variant_id  chromosome  position  ref_allele  alt_allele
upstream_flank  downstream_flank  ref_sequence  alt_sequence
```

> **Both `ref_sequence` and `alt_sequence` are required.** This model encodes both arms.

An optional `label` column (0=benign, 1=pathogenic) can be included — if present, AUC will be computed and logged automatically.

To generate a properly formatted input from a BED file, see [`preprocessing/README.md`](../preprocessing/README.md).

---

## Output Format

The output TSV contains one row per input variant:

```
variant_id  chromosome  position  ref_allele  alt_allele  pathogenicity_score  predicted_label
```

| Column | Description |
|--------|-------------|
| `pathogenicity_score` | Sigmoid probability (0–1), higher = more pathogenic |
| `predicted_label` | Binary call: 0 (benign) or 1 (pathogenic) at default threshold of 0.5 |

If a `label` column was present in the input, a `true_label` column is also added to the output.

---

## Usage

Run from the **repo root**:

```bash
python scoring/score_variants.py \
    --input  your_variants.tsv \
    --model  scoring/model/best_model.pt \
    --output results/predictions/your_dataset/scores.tsv
```

### All arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`,  `-i` | required | Input TSV file |
| `--model`,  `-m` | required | Path to `best_model.pt` |
| `--output`, `-o` | required | Output TSV file path |
| `--batch_size`, `-b` | 16 | Batch size for inference |
| `--gpu`, `-g` | 0 | GPU id (-1 for CPU) |
| `--threshold`, `-t` | 0.5 | Threshold for `predicted_label` |
| `--k` | 6 | K-mer size for tokenization |

---

## Notes

- The script automatically creates the output directory if it does not exist
- Scoring order matches input order — output rows correspond 1:1 to input rows
- `best_model.pt` is gitignored; place it in `scoring/model/` after downloading
- Batch size 16 is a safe default on a single A100. Lower to 4–8 if running on smaller GPUs, since each forward pass encodes two 12k sequences

---

## Differences from previous model

The previous model (`NT2_FineTune` with CNN classifier) encoded only `alt_sequence` and used a CNN with `full-variant_position` pooling over the full sequence. This model encodes both sequences and explicitly contrasts them at the variant position, which gives the classifier a direct signal about the functional change introduced by the variant.

| | Previous model | This model |
|---|---|---|
| Input sequences | `alt_sequence` only | `ref_sequence` + `alt_sequence` |
| Architecture | NT2 + CNN head | Siamese NT2 + MLPProjector + MLP head |
| Pooling | Full-sequence CNN with variant-position pooling | Variant-position token only |
| Classifier input | Sequence embedding | [ref_feat, alt_feat, ref − alt_feat] |