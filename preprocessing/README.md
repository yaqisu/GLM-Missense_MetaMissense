# Preprocessing

This directory contains all scripts for generating training data from ClinVar variant annotations and the GRCh38 reference genome. All scripts are run from the **repo root**.

**Design principle:** This folder contains code only. All data files (inputs and outputs) live in `data/`. Dataset configuration lives in `preprocessing/config.tsv`.

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
   └── ClinVar.260309.*.bed           │
        │                             │
        ▼                             │
[Step 2] subtract_new_variants.sh     │
        │                             │
        ▼                             │
   data/bed/                          │
   └── ClinVar.260309only.*.bed       │
        │                             │
        │   preprocessing/config.tsv  │
        │     (split=yes/no per row)  │
        ▼          │                  ▼
              [Step 3] generate_datasets.sh
                             │
                   ┌─────────┴──────────┐
                   ▼                    ▼
            data/sequences/        data/splits/
          (all datasets,          (split=yes rows
           labeled + unlabeled)    only, train/val)
```

---

## Quick Start

```bash
# Step 1 — Download ClinVar VCFs and generate BED files
bash preprocessing/process_clinvar.sh \
    -t clinvar_20251103,clinvar_20260309 \
    -b /path/to/bcftools

# Step 2 — Extract variants new in 260309 not seen during training
bash preprocessing/subtract_new_variants.sh

# Step 3 — Generate sequences and train/val splits
bash preprocessing/generate_datasets.sh -c preprocessing/config.tsv -s 6k,12k,30k
```

> Each script skips already-existing output files, so it is safe to re-run after interruptions.

---

## Configuration

**File:** `preprocessing/config.tsv`

All dataset definitions live here. Each row defines one output dataset — which BED files go into class 0 (benign), class 1 (pathogenic), or are left unlabeled. Pass it explicitly to both scripts with `-c`:

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

Example rows for this project:
```
name                  class0_files                                    class1_files                                     unlabeled_files  split
251103_BvsP           data/bed/ClinVar.251103...benign.bed            data/bed/ClinVar.251103...pathogenic.bed                          yes
251103_BLBvsPLP       data/bed/...benign.bed;...likely_benign.bed     data/bed/...pathogenic.bed;...likely_pathogenic.bed               yes
260309only_BLBvsPLP   data/bed/...260309only...benign.bed;...         data/bed/...260309only...pathogenic.bed;...                       no
```

Rows with `split=no` produce sequences only — they are never split into train/val. This is used for the held-out benchmark set (`260309only_BLBvsPLP`).

---

## Step 1 — Download ClinVar VCF → BED

**Script:** `preprocessing/process_clinvar.sh`  
**Requires:** `wget`, `bgzip`, `bcftools`  
**Output:** `data/vcf/` (intermediate), `data/bed/`

```bash
bash preprocessing/process_clinvar.sh \
    -t clinvar_20251103,clinvar_20260309 \
    -b /path/to/bcftools
```

| Flag | Description |
|------|-------------|
| `-t <timestamps>` | Comma-separated ClinVar timestamps (e.g. `clinvar_20251103,clinvar_20260309`) |
| `-b <path>` | Path to `bcftools` binary (default: `bcftools` on `$PATH`) |
| `-h` | Show help |

For each timestamp, the script downloads the VCF from the [NCBI ClinVar FTP weekly archive](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/), filters to missense variants, then splits by clinical significance into four BED files:

```
data/bed/ClinVar.{tag}.missense.hg38.pathogenic.bed
data/bed/ClinVar.{tag}.missense.hg38.likely_pathogenic.bed
data/bed/ClinVar.{tag}.missense.hg38.benign.bed
data/bed/ClinVar.{tag}.missense.hg38.likely_benign.bed
```

Each BED file has six columns: `chr`, `start` (0-based), `end` (start+1), `variant_id`, `REF`, `ALT`. Only SNVs are included.

> **Note:** BED coordinates are 0-based as per the BED standard. The `position` column in `data/sequences/` TSV output is reported as 1-based for readability.

---

## Step 2 — Subtract New Variants

**Script:** `preprocessing/subtract_new_variants.sh`  
**Requires:** BED files for both timestamps in `data/bed/`  
**Output:** `data/bed/ClinVar.260309only.missense.hg38.{class}.bed`

```bash
bash preprocessing/subtract_new_variants.sh
```

Compares `260309` against `251103` using `chr + pos + REF + ALT` as the variant key, and keeps only variants that appear in `260309` but not `251103`. These variants were never seen during training, making them a clean held-out benchmark set.

```
data/bed/ClinVar.260309only.missense.hg38.pathogenic.bed
data/bed/ClinVar.260309only.missense.hg38.likely_pathogenic.bed
data/bed/ClinVar.260309only.missense.hg38.benign.bed
data/bed/ClinVar.260309only.missense.hg38.likely_benign.bed
```

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
- **`split=no`**: concatenates all per-class TSVs into a single file in `data/sequences/` — e.g. `ClinVar.260309only.missense.hg38.seq12k.tsv` — ready for zero-shot scoring or evaluation. No train/val split is produced.

So running with `-s 6k,12k,30k` on the `260309only` config rows produces:
```
data/sequences/ClinVar.260309only.missense.hg38.seq6k.tsv
data/sequences/ClinVar.260309only.missense.hg38.seq12k.tsv
data/sequences/ClinVar.260309only.missense.hg38.seq30k.tsv
```

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

Assignments are hardcoded in `split_data_fixed_chroms.py` for reproducibility:

- **Train:** chr 1–6, 9–10, 12–14, 16–19, 21–22, MT, X, Y (~80%)
- **Val:** chr 7, 8, 11, 15, 20 (~20%)

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
| `data/bed/` | Yes | ClinVar BED files for both timestamps + `260309only` (~few MB each) |
| `data/reference/` | No | Reference genome (~3 GB); download instructions in Step 3 |
| `data/sequences/` | No | Per-class TSVs + concatenated files for `split=no` rows; regenerate with `generate_datasets.sh` |
| `data/splits/` | No | Train/val split files (`split=yes` rows only); regenerate with `generate_datasets.sh` |
| `preprocessing/config.tsv` | Yes | Dataset definitions; edit to add new timestamps or datasets |