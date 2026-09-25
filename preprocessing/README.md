# Preprocessing

This directory contains all scripts for generating training data from ClinVar variant annotations and the GRCh38 reference genome. All scripts are run from the **repo root**.

**Design principle:** This folder contains code only. All data files (inputs and outputs) live in `data/`. Dataset configuration lives in `preprocessing/config.tsv`.

Two ClinVar releases are used: **`clinvar_20251103`** (training/validation) and **`clinvar_20260923`** (variants new since 251103 form the held-out temporal test set).

---

## Pipeline Overview

```
ClinVar FTP (NCBI)               Ensembl FTP
        │                             │
        ▼                             ▼
[Step 1] process_clinvar.sh       data/reference/
        │                             │
        ▼                             │
   data/bed/                          │
   ├── ClinVar.251103.*.bed           │
   └── ClinVar.260923.*.bed           │
        │                             │
        ▼                             │
[Step 2] subtract_new_variants.py     │
        │                             │
        ▼                             │
   data/bed/                          │
   └── ClinVar.260923only.*.bed       │
        │                             │
        │   preprocessing/config.tsv  │
        │     (split=yes/no per row)  │
        ▼          │                  ▼
              [Step 3] generate_datasets.sh
                             │
                   ┌─────────┴──────────┐
                   ▼                    ▼
            data/sequences/        data/splits/
          (per-class TSVs)        (train/val by chromosome)
```

---

## Quick Start

```bash
# Step 1 — Download ClinVar VCFs and generate BED files
bash preprocessing/process_clinvar.sh \
    -t clinvar_20251103,clinvar_20260923 \
    -b /path/to/bcftools

# Step 2 — Extract variants new in 260923 that were not in the 251103 data
python preprocessing/subtract_new_variants.py -b /path/to/bcftools

# Step 3 — Generate sequences and train/val splits
bash preprocessing/generate_datasets.sh -c preprocessing/config.tsv -s 6k,12k,30k
```

> Each shell script skips already-existing output files, so it is safe to re-run after interruptions.

---

## Class Definitions

| Dataset | Class 0 (benign) | Class 1 (pathogenic) |
|---------|------------------|----------------------|
| `BvsP` | Benign | Pathogenic |
| `BLBvsPLP` | Benign, Likely_benign, Benign/Likely_benign | Pathogenic, Likely_pathogenic, Pathogenic/Likely_pathogenic |

Classes are assigned by exact match on the ClinVar `CLNSIG` field. `Benign/Likely_benign` and `Pathogenic/Likely_pathogenic` are ClinVar's aggregate classifications when submitters agree on direction but differ in certainty. Conflicting classifications, VUS, and qualified terms (e.g. `Pathogenic|other`, `Likely_pathogenic,_low_penetrance`) are excluded.

---

## Configuration

**File:** `preprocessing/config.tsv`

All dataset definitions live here. Each row defines one output dataset — which BED files go into class 0 (benign), class 1 (pathogenic), or are left unlabeled. Pass it explicitly with `-c`:

```bash
bash preprocessing/generate_datasets.sh -c preprocessing/config.tsv -s 6k,12k,30k
```

To use a different set of datasets (e.g. a new timestamp, a different label scheme), copy `config.tsv`, edit it, and pass the new path with `-c`.

| Column | Description |
|--------|-------------|
| `name` | Output filename stem used in `data/sequences/` and `data/splits/` |
| `class0_files` | Semicolon-separated BED paths assigned label=0 |
| `class1_files` | Semicolon-separated BED paths assigned label=1 |
| `unlabeled_files` | Semicolon-separated BED paths with no label column (benchmark/test data) |
| `split` | `yes`: also generate train/val splits in `data/splits/`; `no`: sequences only |

Leave `class0_files` and `class1_files` empty for unlabeled datasets; leave `unlabeled_files` empty for labeled datasets.

Rows in this project:
```
name                          class0_files                                        class1_files                                                split
251103.BvsP                   ...251103...benign.bed                              ...251103...pathogenic.bed                                  yes
251103.BLBvsPLP               ...benign.bed;...likely_benign.bed;                 ...pathogenic.bed;...likely_pathogenic.bed;                 yes
                              ...benign_likely_benign.bed                         ...pathogenic_likely_pathogenic.bed
260923only.BLBvsPLP           (same classes, 260923only BEDs)                     (same classes, 260923only BEDs)                             yes
```

---

## Step 1 — Download ClinVar VCF → BED

**Script:** `preprocessing/process_clinvar.sh`  
**Requires:** `wget`, `bgzip`, `bcftools`  
**Output:** `data/vcf/` (intermediate), `data/bed/`

```bash
bash preprocessing/process_clinvar.sh \
    -t clinvar_20251103,clinvar_20260923 \
    -b /path/to/bcftools
```

| Flag | Description |
|------|-------------|
| `-t <timestamps>` | Comma-separated ClinVar timestamps (e.g. `clinvar_20251103,clinvar_20260923`) |
| `-b <path>` | Path to `bcftools` binary (default: `bcftools` on `$PATH`) |
| `-h` | Show help |

For each timestamp, the script downloads the VCF from the [NCBI ClinVar FTP](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/) (`weekly/` or the top-level directory, md5-checked), filters to missense variants, then splits by clinical significance (exact `CLNSIG` match) into six BED files:

```
data/bed/ClinVar.{tag}.missense.hg38.pathogenic.bed
data/bed/ClinVar.{tag}.missense.hg38.likely_pathogenic.bed
data/bed/ClinVar.{tag}.missense.hg38.pathogenic_likely_pathogenic.bed
data/bed/ClinVar.{tag}.missense.hg38.benign.bed
data/bed/ClinVar.{tag}.missense.hg38.likely_benign.bed
data/bed/ClinVar.{tag}.missense.hg38.benign_likely_benign.bed
```

Each BED file has six columns: `chr`, `start` (0-based), `end` (start+1), `variant_id` (ClinVar VariationID), `REF`, `ALT`. Only SNVs are included.

> **Note:** BED coordinates are 0-based as per the BED standard. The `position` column in `data/sequences/` TSV output is reported as 1-based for readability.

---

## Step 2 — Subtract New Variants

**Script:** `preprocessing/subtract_new_variants.py`  
**Requires:** BED files for both timestamps in `data/bed/`; optionally both missense VCFs in `data/vcf/` for the gene audit  
**Output:** `data/bed/ClinVar.260923only.missense.hg38.{class}.bed`

```bash
python preprocessing/subtract_new_variants.py -b /path/to/bcftools
```

Keeps only 260923 variants that were not part of the 251103 data. A variant is removed if its `chr + pos + REF + ALT` or ClinVar VariationID appears in **any** of the six 251103 BEDs (pooled across classes, so both resubmissions and reclassifications such as LB→B or B→P are removed), or if it appears in more than one 260923 class. Variants present in 251103 only with other classifications (e.g. VUS, conflicting) are kept, since they were never used for training or validation.

The script prints class × split counts for both releases, the 251103→260923 reclassification table, and a gene-overlap audit confirming that no gene in the held-out test chromosomes appears in the training set.

---

## Step 3 — Generate Sequences and Splits

**Script:** `preprocessing/generate_datasets.sh`  
**Requires:** BED files in `data/bed/`, reference genome in `data/reference/` (see below), `preprocessing/config.tsv`  
**Output:** `data/sequences/`, `data/splits/`

```bash
bash preprocessing/generate_datasets.sh -c preprocessing/config.tsv [-s <sizes>]
```

| Flag | Description |
|------|-------------|
| `-c <config>` | Config file path. Required. Edit to add/change datasets. |
| `-s <sizes>` | Comma-separated window sizes in units of k. Accepts predefined names (`6k`, `12k`, `30k`, `60k`, `130k`) or any custom value like `11.7k` (= 11,700 bp). Default: `12k`. |

For each row in the config, the script runs `extract_variant_sequences.py` for each BED file × the requested window sizes, writing labeled (or unlabeled) TSVs to `data/sequences/`. Then:

- **`split=yes`**: combines all per-class TSVs and splits by chromosome into train/val pairs in `data/splits/`.
- **`split=no`**: concatenates all per-class TSVs into a single file `data/sequences/{name}.seq{size}.tsv` (e.g. for unlabeled benchmark sets). No train/val split is produced.

| Suffix | Flank (each side) | Total window |
|--------|------------------|--------------|
| `seq6k` | 2,999 bp | 5,999 bp |
| `seq12k` | 5,999 bp | 11,999 bp |
| `seq30k` | 14,999 bp | 29,999 bp |
| `seq60k` | 29,999 bp | 59,999 bp |
| `seq130k` | 64,999 bp | 129,999 bp |
| `seq{X}k` | `round(X×1000−1)/2` bp | `round(X×1000)` bp |

Window sizes correspond to the context lengths of Nucleotide Transformer and Caduceus.

Output TSV columns: `variant_id`, `chromosome`, `position`, `ref_allele`, `alt_allele`, `upstream_flank`, `downstream_flank`, `ref_sequence`, `alt_sequence`, `label` (if labeled).

Split output files named `{name}.seq{size}_{training|validation}.tsv` in `data/splits/`.

### Chromosome split

Assignments are hardcoded in `split_data_fixed_chroms.py` for reproducibility and applied identically to both releases:

- **Train:** chr 1–6, 9–10, 12–14, 16–19, 21–22, MT, X, Y (~80%)
- **Val:** chr 7, 8, 11, 15, 20 (~20%)

| Split file | Chromosomes | Use |
|------------|-------------|-----|
| `ClinVar.251103.missense.hg38.seq{size}.{BvsP,BLBvsPLP}_training.tsv` | train | Model training |
| `ClinVar.251103.missense.hg38.seq{size}.{BvsP,BLBvsPLP}_validation.tsv` | val | Validation / model selection |
| `ClinVar.260923only.missense.hg38.seq{size}.BLBvsPLP_training.tsv` | train | New variants in training genes (not used as a test set) |
| `ClinVar.260923only.missense.hg38.seq{size}.BLBvsPLP_validation.tsv` | val | **Held-out temporal test set** — new variants, no genes shared with training |

### Reference genome setup

The reference genome is not tracked in git (~3 GB). Download it once before running Step 3:

```bash
wget https://ftp.ensembl.org/pub/release-104/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
gunzip Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
mv Homo_sapiens.GRCh38.dna.primary_assembly.fa data/reference/
```

> **Note:** GRCh38.104 refers to Ensembl release 104. The `.104` is the Ensembl release number — the underlying DNA sequence is standard GRCh38 (hg38).

---

## Data Directory Reference

| Path | Tracked in git | Notes |
|------|---------------|-------|
| `data/vcf/` | No | Downloaded ClinVar VCFs; regenerate with `process_clinvar.sh` |
| `data/bed/` | No | ClinVar BED files for both timestamps + `260923only`; regenerate with `process_clinvar.sh` and `subtract_new_variants.py` |
| `data/reference/` | No | Reference genome (~3 GB); download instructions in Step 3 |
| `data/sequences/` | No | Per-class TSVs + concatenated files for `split=no` rows; regenerate with `generate_datasets.sh` |
| `data/splits/` | No | Train/val split files (`split=yes` rows only); regenerate with `generate_datasets.sh` |
| `preprocessing/config.tsv` | Yes | Dataset definitions; edit to add new timestamps or datasets |