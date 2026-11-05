# Preprocessing

Builds the ClinVar sets of **Table 2** of the manuscript from the ClinVar VCFs and the GRCh38 reference genome.
All scripts are run from the **repo root**.

**Design principle:** this folder contains code only. All data files (inputs and outputs) live in `data/`. Dataset
definitions live in `preprocessing/config.tsv`, one row per Table 2 set.

Two ClinVar releases are used: **`clinvar_20251103`** (earlier release) and **`clinvar_20260923`** (later release; the
variants absent from all six benign and pathogenic classes of the earlier release).

---

## Table 2 sets and the files that hold them

| Table 2 set | Release | Mapping | Held-out chromosomes | Sizes | Files (`data/splits/`) | Used by |
|---|---|---|---|---|---|---|
| Architecture exploration | Early | Strict (`BvsP`) | 7, 8, 11, 15, 20 | 6k, 12k, 30k | `ClinVar.251103.missense.hg38.seq{6k,12k,30k}.BvsP_{training,validation}.tsv` | `training/` (Figs 2, 3, S2–S5, Table S1) |
| GLM-Missense training | Early | Expanded (`BLBvsPLP`) | 2, 3, 16, 17 | 12k | `ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_{training,validation}.tsv` | `training/` (GLM-Missense, ESM1b-FT; Figs S6, S10) |
| Later release, classified variants | Late | Expanded (`BLBvsPLP`) | 2, 3, 16, 17 (test) | 12k | `ClinVar.260923only.missense.hg38.seq12k.BLBvsPLP_{training,validation}.tsv` | ensemble training (`_training`) and the held-out test set (`_validation`): `evaluation/` (Fig S7), `analysis/` |
| MetaMissense training; ablation ensembles and benchmarking | Late | Expanded | — | — | subsets of the later-release set with every required score | selected in `analysis/core.py` (no file here) |
| VUS | All (20260923) | VUS | 2, 3, 16, 17 | 12k | see **[TBA-P3]** | `analysis/` (Fig 7, Fig S14) |

The `_validation` file of the later-release set is the **held-out test set** (chromosomes 2, 3, 16, 17, never used to
train GLM-Missense, ESM1b-FT or any ensemble). `analysis/table_clinvar_sets.py` recounts Table 2 from these files.

---

## Pipeline overview

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
        │     (one row per Table 2 set)
        ▼          │                  ▼
              [Step 3] generate_datasets.sh
                             │
                   ┌─────────┴──────────┐
                   ▼                    ▼
            data/sequences/        data/splits/
          (per-class TSVs)        (training / held-out by chromosome)

[Step 4] clinvar_transcripts.py  →  data/clinvar/clinvar_transcripts.tsv
         (the transcript ClinVar named each variant on; used by scoring/annotate_dbnsfp.py --clinvar_tx)
```

---

## Quick start

```bash
# Step 1 — Download ClinVar VCFs and generate BED files
bash preprocessing/process_clinvar.sh \
    -t clinvar_20251103,clinvar_20260923 \
    -b /path/to/bcftools

# Step 2 — Keep the 260923 variants that are absent from the 251103 classes
python preprocessing/subtract_new_variants.py -b /path/to/bcftools

# Step 3 — Sequences and training / held-out splits of every Table 2 set
bash preprocessing/generate_datasets.sh -c preprocessing/config.tsv

# Step 4 — ClinVar transcript of each variant, mapped to Ensembl (downloads: see the script docstring)
python preprocessing/clinvar_transcripts.py \
    --summary data/clinvar/variant_summary.txt.gz \
    --mane data/clinvar/MANE.GRCh38.v1.4.summary.txt.gz \
    --gene2ensembl data/clinvar/gene2ensembl.gz \
    --beds 'data/bed/ClinVar.*.missense.hg38.*.bed' \
    --output data/clinvar/clinvar_transcripts.tsv
```

> Each step skips output files that already exist, so it is safe to re-run after interruptions. To rebuild a split,
> delete its files in `data/splits/` first.

---

## Class definitions

| Mapping | Class 0 (benign) | Class 1 (pathogenic) |
|---------|------------------|----------------------|
| Strict (`BvsP`) | Benign | Pathogenic |
| Expanded (`BLBvsPLP`) | Benign, Likely_benign, Benign/Likely_benign | Pathogenic, Likely_pathogenic, Pathogenic/Likely_pathogenic |

Classes are assigned by exact match on the ClinVar `CLNSIG` field. `Benign/Likely_benign` and
`Pathogenic/Likely_pathogenic` are ClinVar's aggregate classifications when submitters agree on direction but differ
in certainty. Conflicting classifications, VUS, and qualified terms (e.g. `Pathogenic|other`,
`Likely_pathogenic,_low_penetrance`) are excluded.

---

## Configuration

**File:** `preprocessing/config.tsv` — one row per Table 2 set that has its own files.

| Column | Description |
|--------|-------------|
| `name` | Output filename stem used in `data/sequences/` and `data/splits/` |
| `class0_files` | Semicolon-separated BED paths assigned label=0 |
| `class1_files` | Semicolon-separated BED paths assigned label=1 |
| `unlabeled_files` | Semicolon-separated BED paths with no label column |
| `split` | `yes`: also generate training / held-out splits in `data/splits/`; `no`: sequences only |
| `heldout` | Held-out chromosome set of the split: `architecture` (chr 7, 8, 11, 15, 20) or `glm_missense` (chr 2, 3, 16, 17) |
| `sizes` | Comma-separated sequence window sizes generated for this row (e.g. `6k,12k,30k`) |

Leave `class0_files` and `class1_files` empty for unlabeled datasets; leave `unlabeled_files` empty for labeled
datasets. The file must end with a blank line.

---

## Step 1 — Download ClinVar VCF → BED

**Script:** `preprocessing/process_clinvar.sh`
**Requires:** `wget`, `bgzip`, `bcftools`
**Output:** `data/vcf/` (intermediate), `data/bed/`

| Flag | Description |
|------|-------------|
| `-t <timestamps>` | Comma-separated ClinVar timestamps (e.g. `clinvar_20251103,clinvar_20260923`) |
| `-b <path>` | Path to `bcftools` binary (default: `bcftools` on `$PATH`) |
| `-h` | Show help |

For each timestamp, the script downloads the VCF from the
[NCBI ClinVar FTP](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/) (`weekly/` or the top-level directory,
md5-checked), filters to missense variants, then splits by clinical significance (exact `CLNSIG` match) into six BED
files:

```
data/bed/ClinVar.{tag}.missense.hg38.pathogenic.bed
data/bed/ClinVar.{tag}.missense.hg38.likely_pathogenic.bed
data/bed/ClinVar.{tag}.missense.hg38.pathogenic_likely_pathogenic.bed
data/bed/ClinVar.{tag}.missense.hg38.benign.bed
data/bed/ClinVar.{tag}.missense.hg38.likely_benign.bed
data/bed/ClinVar.{tag}.missense.hg38.benign_likely_benign.bed
```

Each BED file has six columns: `chr`, `start` (0-based), `end` (start+1), `variant_id` (ClinVar VariationID), `REF`,
`ALT`. Only SNVs are included.

> **Note:** BED coordinates are 0-based as per the BED standard. The `position` column in `data/sequences/` TSV output
> is reported as 1-based for readability.

---

## Step 2 — Subtract new variants

**Script:** `preprocessing/subtract_new_variants.py`
**Requires:** BED files for both timestamps in `data/bed/`; optionally both missense VCFs in `data/vcf/` for the gene audit
**Output:** `data/bed/ClinVar.260923only.missense.hg38.{class}.bed`

Keeps only 260923 variants that were not part of the 251103 data. A variant is removed if its `chr + pos + REF + ALT`
or ClinVar VariationID appears in **any** of the six 251103 BEDs (pooled across classes, so both resubmissions and
reclassifications such as LB→B or B→P are removed), or if it appears in more than one 260923 class. Variants present
in 251103 only with other classifications (e.g. VUS, conflicting) are kept, since they were never used for training
or validation.

The script prints class × split counts for both releases (split on chr 2, 3, 16, 17), the 251103→260923
reclassification table, and a gene-overlap audit confirming that no gene on the held-out test chromosomes appears in
the training set.

---

## Step 3 — Generate sequences and splits

**Script:** `preprocessing/generate_datasets.sh`
**Requires:** BED files in `data/bed/`, reference genome in `data/reference/` (see below), `preprocessing/config.tsv`
**Output:** `data/sequences/`, `data/splits/`

```bash
bash preprocessing/generate_datasets.sh -c preprocessing/config.tsv [-s <sizes>]
```

| Flag | Description |
|------|-------------|
| `-c <config>` | Config file path. Required. |
| `-s <sizes>` | Optional. Only generate these window sizes (each row still generates only the sizes in its `sizes` column). |

For each row, the script runs `extract_variant_sequences.py` for each BED file × the row's window sizes, writing
labeled (or unlabeled) TSVs to `data/sequences/`. Then:

- **`split=yes`**: combines all per-class TSVs and splits them by the row's `heldout` chromosome set
  (`split_data_fixed_chroms.py --heldout`) into `{prefix}.seq{size}.{mapping}_training.tsv` (all other chromosomes)
  and `{prefix}.seq{size}.{mapping}_validation.tsv` (held-out chromosomes) in `data/splits/`.
- **`split=no`**: concatenates all per-class TSVs into a single file `data/sequences/{name}.seq{size}.tsv`.

| Suffix | Flank (each side) | Total window | Model |
|--------|------------------|--------------|-------|
| `seq6k` | 2,999 bp | 5,999 bp | NT-1; NT-2 in the matched NT-2 vs NT-1 comparison |
| `seq12k` | 5,999 bp | 11,999 bp | NT-2 (GLM-Missense) |
| `seq30k` | 14,999 bp | 29,999 bp | Caduceus-PS |

Output TSV columns: `variant_id`, `chromosome`, `position`, `ref_allele`, `alt_allele`, `upstream_flank`,
`downstream_flank`, `ref_sequence`, `alt_sequence`, `label` (if labeled).

### Chromosome splits

Both held-out sets are hardcoded in `split_data_fixed_chroms.py` (`HELDOUT`); every other chromosome (1–22, X, Y, MT)
is training.

| `heldout` | Held-out chromosomes | Sets |
|---|---|---|
| `architecture` | 7, 8, 11, 15, 20 | Architecture exploration (earlier release, strict) |
| `glm_missense` | 2, 3, 16, 17 | GLM-Missense training (earlier release, expanded); later release (ensemble training / held-out test) |

### Reference genome setup

The reference genome is not tracked in git (~3 GB). Download it once before running Step 3:

```bash
wget https://ftp.ensembl.org/pub/release-104/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
gunzip Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
mv Homo_sapiens.GRCh38.dna.primary_assembly.fa data/reference/
```

> **Note:** GRCh38.104 refers to Ensembl release 104. The `.104` is the Ensembl release number — the underlying DNA
> sequence is standard GRCh38 (hg38).

---

## Step 4 — ClinVar transcripts

**Script:** `preprocessing/clinvar_transcripts.py`
**Output:** `data/clinvar/clinvar_transcripts.tsv`

The ClinVar VCF carries no transcript. This script reads the transcript ClinVar named each variant on from
`variant_summary.txt.gz` and maps it to Ensembl (MANE, then NCBI gene2ensembl). `scoring/annotate_dbnsfp.py
--clinvar_tx` uses it to take each predictor's score for that transcript (Methods: "For transcript-specific scores,
we used the score for the transcript that ClinVar used to name the variant"). Download commands are in the script
docstring.

---

## To be added

> **[TBA-P1] Architecture-exploration subsets.** The LoRA model selection (Fig 3A) and the allele-aware input
> comparison (Fig 3B, Fig S5) used a 5,000-variant label-stratified subset of the architecture-exploration training
> set and a 1,250-variant subset of its validation set; the combined 6,250 variants were repartitioned into five
> chromosome-exclusive folds for cross-validation (Table S1). To add: the script (or command) that draws these subsets,
> with its random seed.
> Input: `data/splits/ClinVar.251103.missense.hg38.seq12k.BvsP_{training,validation}.tsv`.
> Output: `data/splits/` (file names to be listed here), read by `training/NT2_lora_fullFT_sweep.py`
> (`--train_path_sweep`, `--val_path_sweep`), `training/NT2_ranked_cv.py` (`--data_path`) and
> `training/NT2_ref_alt_input_sweep.py`.

> **[TBA-P2] Caduceus-PS input.** `training/caduceus_phase1.py` reads its training and validation files from the
> constants `TRAIN_PATH` / `VAL_PATH` at the top of the script. To add: the step that produces these files from
> `data/splits/ClinVar.251103.missense.hg38.seq30k.BvsP_{training,validation}.tsv` (or point the constants at those
> files if they can be read directly).

> **[TBA-P3] Held-out VUS set (Table 2, VUS row).** All missense variants classified as VUS
> (`Uncertain_significance`) in the 20260923 release on chromosomes 2, 3, 16 and 17, without excluding variants present
> in the earlier release (n = 526,428). To add: the extraction (e.g. one more `CLNSIG` entry in `process_clinvar.sh`
> and a filter to the held-out chromosomes) and, if GLM-Missense scores are computed from this folder, a `config.tsv`
> row with `split=no` and size `12k`.
> Input: `data/vcf/clinvar_20260923_missense.vcf.gz`.
> Output: `data/bed/ClinVar.260923.missense.hg38.uncertain_significance.heldout.bed` (and its 12k sequences in
> `data/sequences/`), the input of the held-out VUS table in `analysis/` (**[TBA-A3]**).

---

## Data directory reference

| Path | Tracked in git | Notes |
|------|---------------|-------|
| `data/vcf/` | No | Downloaded ClinVar VCFs; regenerate with `process_clinvar.sh` |
| `data/bed/` | No | ClinVar BED files for both timestamps + `260923only`; regenerate with `process_clinvar.sh` and `subtract_new_variants.py` |
| `data/reference/` | No | Reference genome (~3 GB); download instructions in Step 3 |
| `data/sequences/` | No | Per-class TSVs; regenerate with `generate_datasets.sh` |
| `data/splits/` | No | Training / held-out split files of the Table 2 sets; regenerate with `generate_datasets.sh` |
| `data/clinvar/` | No | ClinVar `variant_summary`, MANE, gene2ensembl and `clinvar_transcripts.tsv` (Step 4) |
| `preprocessing/config.tsv` | Yes | Dataset definitions (Table 2) |
