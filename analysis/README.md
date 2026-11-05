# analysis/

Manuscript figures on the **late-release held-out test set**: ClinVar 260923-only variants (absent from the 2025-11-03
release) on chromosomes 2, 3, 16, 17, which GLM-Missense never saw. Ensembles are fit on the late-release training
split (other chromosomes) and evaluated on the test split; the RAD51D saturation genome editing (SGE) data give labels
independent of any predictor; held-out ClinVar VUS show where GLM-Missense (and, as a control, ESM1b-FT) disagrees
with protein-level predictors.

Every command runs from the repo root. The pipeline, in order:

1. **Environments** — four conda envs: `lamar`, `saluki`, `alphagenome`, `meme`.
2. **Inputs and scores**, once — copy the tables made elsewhere, compute the VUS scores, build the annotated VUS table.
3. **Figures** — `run_all.sh` (every figure; list in **Figures**). One ensemble version, `bergquist` (**Ensemble**).
4. **Adjusting a run** — fewer bootstraps, one script.

## 1. Environments

| Env | Used for | Main packages |
|---|---|---|
| `lamar` | everything except the two below: figures, `prep_*`, `merge_vus.py`, ESM1b-FT scoring (GPU) | PyTorch + CUDA, `transformers`, `peft`, `xgboost`, `scikit-learn`, `pandas`, `pyarrow`, `scipy`, `matplotlib`, `joblib` |
| `saluki` | `scoring/run_parallel.sh` (Saluki and sequence features) | TensorFlow 2, `pandas`, `numpy` |
| `alphagenome` | `scoring/alphagenome_score.py` (AlphaGenome Atlas); `fig_vus.py` calls it in this env by itself | `alphagenome >= 0.9`, `pandas`, `numpy`, `scipy` |
| `meme` | `scoring/fimo_motifs.py`, `scoring/fimo_score.py`, `run_parallel.sh fimo` (FIMO TF-binding disruption) | MEME suite 5.5.x (`fimo`), `pandas`, `numpy`, `pyjaspar==4.0.0`, `biopython` |

Every command block below starts with the env it needs. A new env gets a name that says what it is for (one tool or
one task), not the analysis it was first made for.

```bash
# alphagenome: build from scratch
conda create -n alphagenome python=3.11 -y
conda activate alphagenome && pip install "alphagenome>=0.9" pandas numpy scipy
# saluki (CPU only; saluki_score.py needs TensorFlow 2 with Keras 2, i.e. TensorFlow <= 2.15):
conda create -n saluki python=3.11 -y
conda activate saluki && pip install "tensorflow-cpu==2.15.*" "numpy<2" pandas
python -c "import tensorflow as tf; print(tf.__version__)"                # 2.15.x
# check that the models load and score: 5 variants of any list with variant_id + TRANSCRIPT_ID
python -u analysis/scoring/saluki_score.py --input <list.tsv> --limit 5 \
    --fasta data/external/ensembl115/Homo_sapiens.GRCh38.115.dna.primary_assembly.fa \
    --gtf data/external/ensembl115/Homo_sapiens.GRCh38.115.gtf \
    --models 'data/external/saluki/train_gru/f*_c*/train/model0_best.h5' \
    --params data/external/saluki/train_gru/params.json -o /tmp/saluki_test.tsv  # saluki_delta for 5 variants
# meme: FIMO (MEME suite) + the JASPAR 2026 motif export
conda create -n meme -c conda-forge -c bioconda meme=5.5 python=3.11 pandas numpy -y
conda activate meme && pip install pyjaspar==4.0.0 biopython && fimo --version          # 5.5.x
# AlphaGenome env under another name: fig_vus.py --ag_env <name>
# lamar: the PyTorch env of training/ ([TBA-A6]). Record each env once, so it can be rebuilt:
mkdir -p analysis/envs
for e in lamar saluki alphagenome meme; do conda env export -n $e --no-builds > analysis/envs/$e.yml; done
# rebuild elsewhere: conda env create -n <env> -f analysis/envs/<env>.yml
```

The AlphaGenome API key lives in a file of its own, readable only by you; the scripts take its path:

```bash
export AG_KEY_FILE=$HOME/.alphagenome_api_key        # holds only the key; chmod 600
```

## 2. Inputs and scores (once)

### 2.1 Copy the inputs made elsewhere

```bash
bash analysis/copy_inputs.sh       # any env; sources in analysis/copy_inputs.local.sh; copies + logs
```

It fills `results/annotated/` (ClinVar main tables, RAD51D SGE table, held-out VUS table, reference tables of the
sequence features), `results/scores/vus_alphagenome_live/` (earlier live AlphaGenome features),
`results/models/ESM1b-FT/` (checkpoint) and `data/external/` (Saluki models, Ensembl GRCh38 release 115 FASTA + GTF),
appends one line per file to `results/annotated/COPY_LOG.tsv` (date, destination, source, md5, user; "already present"
when the file was there) and rewrites `results/annotated/MD5SUMS`. It copies with `cp -pn` (keeps timestamps, never
overwrites) and warns when a file differs from its source. How every input is made: **Inputs** (last section).

### 2.2 Compute the scores

Computed once for the held-out VUS (and the ClinGen overlap of the ClinVar tables), independent of the ensemble
version. Paths used below:

```bash
export FASTA=$PWD/data/external/ensembl115/Homo_sapiens.GRCh38.115.dna.primary_assembly.fa
export GTF=$PWD/data/external/ensembl115/Homo_sapiens.GRCh38.115.gtf
export SALUKI_MODELS="$PWD/data/external/saluki/train_gru/f*_c*/train/model0_best.h5"
export SALUKI_PARAMS=$PWD/data/external/saluki/train_gru/params.json
V=results/annotated/vus/ClinVar_20260923_VUS_missense_heldoutOnly_dbNSFP_GLM-Missense_spliceAI_annot.tsv
mkdir -p results/analysis/logs
```

| Score | Env | Hardware, time | Output |
|---|---|---|---|
| ESM1b-FT | `lamar` | 3 GPUs, hours | `results/scores/vus_esm1bft/vus_esm1bft_scores.tsv` |
| Saluki Δ half-life | `saluki` | 100 CPU cores, hours | `results/scores/vus_saluki/saluki_features_<i>.tsv` |
| Sequence features | `saluki` | CPU, minutes | `results/scores/vus_seq/seq_features_<i>.tsv` |
| FIMO TF-binding disruption | `meme` | CPU, ~4–5 core-hours (minutes on 50 cores) | `results/scores/vus_fimo/fimo_features_<i>.tsv` |
| ClinGen ERepo overlap | `lamar` | minutes | `results/analysis/prep/clingen_erepo/` |
| AlphaGenome (per-track) | `alphagenome` | API; automatic in step 3 | `results/scores/vus_alphagenome_{atlas,live}/alphagenome_features.tsv` |

**ESM1b-FT** (GPU): protein change in one-letter form, mapping to the MANE protein, prediction with the fine-tuned
model. `fine-tune_ESM1b.py` writes its output once every variant is scored (`variant_id`, `score`, `status`: `scored`
or `skipped_invalid_protein_annotation`).

```bash
conda activate lamar
D=results/scores/vus_esm1bft; mkdir -p $D
python analysis/scoring/esm1bft_prepare.py --vus $V --out $D/vus_input.tsv
python analysis/scoring/map_variants_to_mane_protein.py --inputs $D/vus_input.tsv --transcript_column TRANSCRIPT_ID \
    --protein_change_column PROTEIN_CHANGE --gtf $GTF --genome_fasta $FASTA --output_dir $D 2>&1 | tee $D/map.log
CUDA_VISIBLE_DEVICES=1,2,3 python analysis/scoring/fine-tune_ESM1b.py \
    --predict_path $D/vus_input_protein.tsv \
    --checkpoint results/models/ESM1b-FT/exp_1_concat_diff/best_model.pt \
    --predict_out $D/vus_esm1bft_scores.tsv \
    --gpus 0 1 2 --eval_batch_size 1024 2>&1 | tee $D/predict.log
```

**Saluki and sequence features** (CPU): `run_parallel.sh` splits the held-out VUS by transcript into N shards and
runs one process per shard (THREADS threads each). Saluki shards resume when rerun; keep `N` the same when rerunning
(the shards are transcript hash mod N). Each Saluki process loads every model (~1–2 GB of RAM).

```bash
conda activate saluki
N=50 THREADS=2 bash analysis/scoring/run_parallel.sh saluki     # 100 cores
N=50 bash analysis/scoring/run_parallel.sh seq
```

**FIMO TF-binding disruption** (CPU): plain FIMO (MEME suite 5.5; Grant et al. 2011) on the ref and alt allele of
pos ± 32 bp (widest motif − 1) with every JASPAR 2026 CORE vertebrates non-redundant motif (1,019; latest version of
each profile), both strands; per motif the best p of the hits covering the variant (none at p < 1e-3: p = 1e-3). The
motif file is written once from pyJASPAR 4.0.0 (the same matrices as JASPAR's "CORE vertebrates non-redundant, MEME"
download, which works as well: `MOTIFS=<that file>`). `fimo_score.py` takes any variant table (VUS shard, ClinVar main
table, SGE table: `variant_id` as chrN:pos:REF>ALT, or allele columns), so the ClinVar test set and the SGE table can be
annotated the same way:

```bash
conda activate meme
export MOTIFS=data/external/jaspar2026/JASPAR2026_CORE_vertebrates_non-redundant.meme
python analysis/scoring/fimo_motifs.py -o $MOTIFS
N=50 bash analysis/scoring/run_parallel.sh fimo                               # VUS -> merge_vus.py (2.3)
# labeled tables (no figure reads them): fimo_* per variant + a copy of the table with fimo_* added
python analysis/scoring/fimo_score.py --input results/annotated/clinvar/ClinVar.260923only.BLBvsPLP_test.main.tsv \
    --fasta $FASTA --motifs $MOTIFS -o results/scores/clinvar_fimo/fimo_features_test.tsv \
    --annotated results/annotated/clinvar/ClinVar.260923only.BLBvsPLP_test.main.fimo.tsv
S=results/annotated/sge/RAD51D_SGE_missense_annotated.tsv
python analysis/scoring/fimo_score.py --input $S --fasta $FASTA --motifs $MOTIFS \
    -o results/scores/sge_fimo/fimo_features.tsv --annotated ${S%.tsv}.fimo.tsv          # S = core.SGE_FILE
```

**ClinGen Evidence Repository** (figS12):

```bash
conda activate lamar
bash analysis/prep_clingen_download.sh        # skip if data/clingen/ already has the ERepo download; cite its date
python -u analysis/prep_clingen.py 2>&1 | tee results/analysis/logs/prep_clingen.log
```

**AlphaGenome**: per-track scores of 11 scorers (splice sites, splice-site usage, splice junctions, RNA-seq,
polyadenylation, CAGE, PRO-cap, TF and histone ChIP, DNase, ATAC; hundreds of tracks each), summarised across tracks
per variant (`<SCORER>_<summary>`: track count, largest |score| and its track, mean, median, 10th / 90th percentile,
min / max and their tracks, share of tracks with a strong decrease / increase, the same on raw scores; gene-centric
scorers on the variant's own gene; plus the columns the input carries, e.g. gene, transcript, HGVS, scores), as in
the earlier live runs. Two sources, same scorers and summaries, never mixed:
- `atlas` (default): the AlphaGenome Atlas, precomputed per-track scores of every GRCh38 SNV (each scorer as an
  AnnData of tracks with raw and quantile scores) **plus the overall AVI score** (`AVI_score` and its summaries),
  → `results/scores/vus_alphagenome_atlas/`.
- `live`: on-demand model predictions (`score_variant`, 1 Mb window), as the earlier selections were scored; fine for
  the few thousand VUS a figure selects, slow for all 526k → `results/scores/vus_alphagenome_live/` (the copied
  earlier live features there are for `--compare` only).

Every AlphaGenome folder ends with `_atlas` or `_live` (`alphagenome_score.py` refuses an `--outdir` that does not
match `--source`). Score every held-out VUS from the Atlas (~15 h for 526k at 10 variants/s; a rerun of the same
command resumes: scored variants are skipped, failed ones (`failed.tsv`) retried):

```bash
conda activate alphagenome
A=results/scores/vus_alphagenome_atlas
python -u analysis/scoring/alphagenome_score.py --input $V --select none --outdir $A --api_key_file $AG_KEY_FILE \
    --inspect 3                                    # first: AnnData per scorer (shapes, obs / var columns, quantiles)
nohup python -u analysis/scoring/alphagenome_score.py --input $V --select none --outdir $A \
    --api_key_file $AG_KEY_FILE --max_workers 20 > $A/score.log 2>&1 &     # every held-out VUS -> $A/alphagenome_features.tsv
```

The figures do not have to wait for it (step 3): until the full table is merged into the annotated VUS table (2.3),
`fig_vus.py` scores only the VUS it selects, in its own file, and never reads the full table while it is written.
The full scoring is done when `grep -c . $A/alphagenome_features.tsv` = VUS + 1 and a rerun of the same command
reports 0 to score.

### 2.3 Annotated VUS table

One table with every VUS score, read by the VUS figures (`--vus`), once every score of 2.2 is done:

```bash
conda activate lamar
python analysis/scoring/merge_vus.py --alphagenome results/scores/vus_alphagenome_atlas/alphagenome_features.tsv \
    2>&1 | tee results/analysis/logs/merge_vus.log
```

→ `results/annotated/vus/ClinVar_20260923_VUS_heldout_annotated.tsv` (the VUS table + `ESM1b-FT_score`,
`ESM1b-FT_status`, `saluki_*`, `seq_*`, `fimo_*`, AlphaGenome `<SCORER>_<summary>` and `AVI_score`) and
`.coverage.tsv` (variants with each score). It rebuilds the table from the score outputs every time, so rerun this one
command (with `--alphagenome`) whenever a score is recomputed, then rerun the VUS figures (step 3).

**AlphaGenome in the VUS figures, one source per run** (`fig_vus.py` decides at the start and logs it):
- **table**: the annotated VUS table has AlphaGenome features for ≥ 99% of the VUS set (merged above):
  every case is analysed and nothing is scored.
- **run** (fallback, if the table was merged without `--alphagenome`): at most 2,000 cases + controls per model × direction (`--max_variants`), the most confident
  model calls first (1,000 cases + 1,000 controls), and the selected VUS are scored from the Atlas in that run into
  `results/scores/vus_alphagenome_atlas/alphagenome_features_figvus.tsv`, the only AlphaGenome table it reads (a
  partly annotated VUS table is ignored). At most 4 × 2,000 = 8,000 VUS, fewer where a direction has fewer cases;
  later runs reuse what is already in the file.

### Annotated scores

| Score | Column | ClinVar | SGE | VUS | Source |
|---|---|---|---|---|---|
| AlphaMissense, ESM1b, REVEL, CADD, SIFT, PolyPhen-2 HVAR | `AlphaMissense_score`, `ESM1b_score`, `REVEL_score`, `CADD_phred`, `SIFT_score`, `Polyphen2_HVAR_score` | ✓ | ✓ | ✓ | dbNSFP 5.3.1a (copied) |
| BayesDel (no AF), MutPred2, VEST4, VARITY_R | `BayesDel_noAF_score`, `MutPred2_score` / `MutPred_score`, `VEST4_score`, `VARITY_R_LOO_score` / `VARITY_R_score` | ✓ | ✓ | ✓ | dbNSFP 5.3.1a (copied) |
| GERP++, phyloP 100-way / 470-way | `GERP++_RS`, `phyloP100way_vertebrate`, `phyloP470way_mammalian` | ✓ | ✓ | | dbNSFP 5.3.1a (copied) |
| GLM-Missense | `GLM-Missense_score` | ✓ | ✓ | ✓ | copied |
| SpliceAI DS max | `spliceai_DS_max` | ✓ | ✓ | ✓ | copied |
| ESM1b-FT | `ESM1b-FT_score` | ✓ | ✓ | ✓ | ClinVar, SGE: copied; VUS: 2.2 |
| Random | `random_baseline_score` | ✓ | ✓ | | `prep_main_table.py` (Uniform(0, 1), seed 2026) |
| Evo 2 | `evo2_delta` | ✓ | | | optional; not in the manuscript figures |
| AlphaGenome AVI | `AVI_raw` (else `AVI_phred`) | ✓ | | | optional; not in the manuscript figures |
| SGE function class, RNA score | `FUNC_CONSEQUENCE`, `RNA_SCORE` | | ✓ | | copied |
| ClinGen ERepo classification with / without PP3 / BP4 | `erepo_variants.tsv` | ✓ | | | 2.2 |
| Saluki Δ half-life | `saluki_delta` (+ `saluki_status`, …) | | | ✓ | 2.2 |
| Sequence features (exon-junction distance, codon usage / CSC, ESRseq, CpG, transition, GC) | `seq_*` | | | ✓ | 2.2 |
| FIMO TF-binding disruption (largest loss / gain / \|Δ\| of −log10 p over motifs; motifs lost / gained at p < 1e-4, count and any; their names) | `fimo_loss`, `fimo_gain`, `fimo_max_abs_delta`, `fimo_n_loss`, `fimo_n_gain`, `fimo_any_loss`, `fimo_any_gain`, `fimo_lost_tfs`, `fimo_gained_tfs`, `fimo_status` | test (`.fimo.tsv`) | ✓ (`.fimo.tsv`) | ✓ | 2.2 |
| AlphaGenome (11 scorers × summaries across tracks; Atlas: + AVI) | `<SCORER>_<summary>`, `AVI_score` in the AlphaGenome table | | | ✓ | 2.2 / automatic in step 3 |

dbNSFP values are **strict_transcript**: the value of the transcript ClinVar named the variant on (`<col>_fb` with
tier `clinvar_tx`, `clinvar_tx_mapped` or `variant` from the re-annotation; otherwise the entry of `matched_enst` /
`TRANSCRIPT_ID` in `Ensembl_transcriptid`; a single value is variant-level). Orientation: higher = more pathogenic
(ESM1b, SIFT and Evo 2 are negated). `python analysis/core.py` prints the column each score comes from and its coverage.

## 3. Run the figures

```bash
conda activate lamar
export AG_KEY_FILE=$HOME/.alphagenome_api_key
python analysis/core.py                       # check score columns, coverage and variant sets before the long run
bash analysis/run_all.sh                      # every figure (--ensemble bergquist); long: tmux or nohup
```

`run_all.sh` runs `table_clinvar_sets` (the Methods table of the ClinVar sets), `fig_leaderboard` (figS8, figS9; first: it grid-searches MetaMissense, whose
hyperparameters every other ensemble reuses), `fig_complementarity` (fig4, fig5, figS11), `fig_splicing` (fig6,
figS13), `fig_circularity` (figS12) and `fig_vus` (fig7, figS14), each with `--ensemble bergquist`; logs in
`results/analysis/logs/`. `fig_vus.py` reads AlphaGenome from the annotated VUS table or scores the selected VUS in
the run (2.3, "AlphaGenome in the VUS figures"; `alphagenome_score.py` in the env `alphagenome`, through
`conda run`; with `AG_SOURCE=live`, live predictions into `vus_alphagenome_live/`). Without `AG_KEY_FILE` it warns
and tests the VUS without AlphaGenome features (`missing_features` in the manifest). figS10 (ESM1b-FT training curve)
is made outside this folder.

## 4. Adjusting a run

Models are cached in `results/analysis/models/bergquist/`, so nothing is refit unless the inputs or the base changed.

```bash
conda activate lamar
ARGS="--n_boot 100" bash analysis/run_all.sh                               # quick preview (fewer bootstraps)
N_JOBS=20 bash analysis/run_all.sh                                         # fewer CPU cores (default 40)
# one script (figure -> script: see Code)
python -u analysis/fig_splicing.py --ensemble bergquist --figs fig6 --n_jobs 40
python -u analysis/fig_complementarity.py --ensemble bergquist --figs fig5 --n_jobs 40
python -u analysis/fig_vus.py --ensemble bergquist --n_jobs 40             # fig7, figS14 only
python -u analysis/fig_vus.py --ensemble bergquist --select_only           # selection + missing-feature counts only
python -u analysis/fig_vus.py --ensemble bergquist --no_alphagenome_scoring  # no AlphaGenome lookups
python -u analysis/fig_vus.py --ensemble bergquist --max_variants 1000     # cases + controls per model x direction (0 = all)
AG_SOURCE=live bash analysis/run_all.sh                                    # AlphaGenome from live predictions
VUS_ARGS="--max_variants 1000" bash analysis/run_all.sh                   # options for fig_vus.py only
```

`fig_vus.py` AlphaGenome options: `--ag_source` (atlas / live; default `$AG_SOURCE` or atlas), `--ag_env` (default
`alphagenome`), `--ag_key_file` (default `$AG_KEY_FILE`), `--ag_workers` (atlas 20, live 5), `--alphagenome` (the
table of the run-scored selections; default `results/scores/vus_alphagenome_<source>/alphagenome_features_figvus.tsv`).
fig7's three columns (SpliceAI, mean ΔCSC of the six CSC sets `CSC_SETS`, Saluki) are `MAIN_COLS` at the top of `fig_vus.py`; figS14
is the same for ESM1b-FT, same y axes. Every other feature (FIMO too, once `fimo_*` is in the annotated VUS table) is
tested as well (`fig7_tests.tsv`, unadjusted p) but not shown. q: BH over the 6 tests shown (3 features × 2
directions), per model. GLM-Missense vs ESM1b-FT directly: `fig7_specificity.tsv`.

## AlphaGenome: Atlas vs live

Both sources give the same per-track scorers (the Atlas is not only the AVI score: `--inspect` shows each scorer's
tracks with raw and quantile scores), and a figure uses one source only. The earlier VUS selections were scored live
(`--source live`, all 11 scorers, 1 Mb window, `--extreme_q 0.99`, no ontology filter, 5 parallel requests). On 200 of
those VUS (2026-10-23) Atlas and live features correlated but were not identical: median
Spearman 0.95 over 209 features; RNA-seq 10th-percentile log fold change 0.95; RNA-seq share of tracks with a strong
decrease 0.93, with a higher median from the Atlas (0.040 vs 0.014); splice-site usage 0.78. To find out why:

```bash
conda activate alphagenome
A=results/scores/vus_alphagenome_atlas
# the same 20 variants scored both ways today, compared track by track -> $A/diagnose/
python -u analysis/scoring/alphagenome_score.py --input $V --select none --outdir $A --api_key_file $AG_KEY_FILE \
    --diagnose 20
# Atlas vs an earlier live feature table, feature by feature -> $A/atlas_compare/atlas_vs_live.tsv
python -u analysis/scoring/alphagenome_score.py --compare results/scores/vus_alphagenome_live/alphagenome_features_earlier.tsv --outdir $A \
    --api_key_file $AG_KEY_FILE --limit 200
```

`diagnose_summary.tsv` (one row per variant × scorer) separates the possible causes:

| Columns | If they differ | Cause |
|---|---|---|
| `atlas_interval`, `live_interval` | different windows | input sequence length / centring |
| `atlas_only`, `live_only` | tracks only one side returns | track set (filtering, strands) — changes every across-track summary |
| `atlas_genes`, `live_genes` | different genes | gene annotation; own-gene matching of the gene-centric scorers |
| `raw_score_*` on matched tracks | raw scores differ | model weights / version |
| `quantile_score_*` with raw scores equal | only quantiles differ | quantile calibration background (`*_frac_down` uses quantiles) |

`diagnose_versions.txt` records the `alphagenome` package version and each scorer's definition on both sides. If
Atlas and live agree today but both differ from the earlier live table, the earlier run used another model version
or other settings (its package version, date and arguments settle it). Until it is settled, `AG_SOURCE=live` keeps
the AlphaGenome rows of `fig7_tests.tsv` on the same footing as the earlier analysis.

After the full Atlas run, live scores of a batch (or all) VUS and a direct comparison of the two tables:

```bash
conda activate alphagenome
L=results/scores/vus_alphagenome_live
python -u analysis/scoring/alphagenome_score.py --input $V --select none --source live --outdir $L \
    --api_key_file $AG_KEY_FILE --limit 2000 --out_name alphagenome_features_batch.tsv 2>&1 | tee $L/score_batch.log
python -u analysis/scoring/alphagenome_score.py --outdir $L \
    --compare_tables results/scores/vus_alphagenome_atlas/alphagenome_features.tsv $L/alphagenome_features_batch.tsv
```

`--limit N` scores the first N VUS of the table (raise it, or drop it, to extend the same file; reruns resume).

## Code

| File | What it does | Env | Writes |
|---|---|---|---|
| `core.py` | methods, the ensemble version, data loading and variant sets, MetaMissense and ablation ensembles, statistics, figure style. `python analysis/core.py` checks score columns and coverage | `lamar` | `tables/score_coverage.tsv`, `tables/variant_sets.tsv` |
| `fig_leaderboard.py` | figS8, figS9 | `lamar` | |
| `fig_complementarity.py` | fig4, fig5, figS11 | `lamar` | |
| `fig_splicing.py` | fig6, figS13 | `lamar` | |
| `fig_circularity.py` | figS12 | `lamar` | |
| `table_clinvar_sets.py` | Methods table of the ClinVar sets (tab:clinvar_sets): variants per release, mapping and split by ClinVar class, and each split's chromosomes, from the split files and class BEDs (`SETS` at the top) | `lamar` | `tables/table_clinvar_sets.{tsv,tex}` |
| `table_clinvar_subsets.py` | subset rows of that table, by ClinVar class: MetaMissense training, the ablation / benchmarking train and test sets (core.DALL), and the VUS counts (all, scored, fig7 set), with the figures' own filters | `lamar` | `tables/table_clinvar_subsets.{tsv,tex}` |
| `fig_vus.py` | fig7, figS14; scores selected VUS without AlphaGenome features | `lamar` | `prep/vus/missing_alphagenome_bergquist.tsv` |
| `run_all.sh` | every figure | `lamar` | `logs/` |
| `copy_inputs.sh` | copies the inputs made elsewhere and logs each copy | any | `results/annotated/COPY_LOG.tsv`, `MD5SUMS` |
| `prep_main_table.py` | builds the ClinVar main tables the copied ones were made from | `lamar` | `prep/main_table/` |
| `prep_clingen_download.sh`, `prep_clingen.py` | ClinGen Evidence Repository download; match to the ClinVar tables, re-score without PP3 / BP4 | `lamar` | `data/clingen/`, `prep/clingen_erepo/` |
| `scoring/esm1bft_prepare.py`, `map_variants_to_mane_protein.py`, `fine-tune_ESM1b.py` | ESM1b-FT scores of the VUS: input, MANE protein mapping, prediction (model code) | `lamar` | `results/scores/vus_esm1bft/` |
| `scoring/run_parallel.sh` (`saluki_score.py`, `seq_features.py`, `fimo_score.py`) | Saluki Δ half-life, sequence or FIMO features of every held-out VUS, parallel CPU processes | `saluki` (`meme` for fimo) | `results/scores/vus_{saluki,seq,fimo}/` |
| `scoring/alphagenome_score.py` | AlphaGenome features of a list of VUS (`--select none`) from the Atlas (default) or live; `--inspect`, `--diagnose`, `--compare`, `--compare_tables` | `alphagenome` | `results/scores/vus_alphagenome_{atlas,live}/` |
| `scoring/fimo_motifs.py`, `scoring/fimo_score.py` | JASPAR 2026 motif file; FIMO TF-binding disruption of any variant table (`run_parallel.sh fimo` for the VUS) | `meme` | `data/external/jaspar2026/`, `results/scores/vus_fimo/` |
| `scoring/merge_vus.py` | the annotated VUS table | `lamar` | `results/annotated/vus/ClinVar_20260923_VUS_heldout_annotated.tsv` |
| `prep_random_baseline.py` | Random (null) score of the later-release variants, once | `lamar` | `results/scores/random_baseline.tsv` |
| `scoring/annotate_spliceai.py`, `scoring/replace_dbnsfp_columns.py` | SpliceAI columns of any variant table; dbNSFP columns + transcript tiers from `scoring/annotate_dbnsfp.py` (ClinVar main tables' inputs) | `lamar` | the table, in place |

Every figure script takes `--ensemble bergquist` plus the shared options of `core.parser` (input paths, `--vus`,
`--n_boot 1000`, `--seed 0`, `--n_jobs`, `--out results/analysis`); `--figs` picks figures within a script
(`fig_complementarity.py`, `fig_splicing.py`).

## Ensemble

MetaMissense = base + GLM-Missense. One version, `bergquist` (`--ensemble bergquist`; `ENSEMBLES` in `core.py`):

| Version | Base |
|---|---|
| `bergquist` | BayesDel, MutPred2, REVEL, VEST4, AlphaMissense, ESM1b, VARITY_R (Bergquist et al. 2025, *Genet Med*, Table 1) |

SGE figures use the base with ESM1b-FT in place of zero-shot ESM1b. Outputs go to `<dir>/bergquist/` (figures,
tables, variant lists, models).

## Outputs

```
results/
├── annotated/                     analysis inputs: copies (COPY_LOG.tsv, MD5SUMS) + the annotated VUS table
├── scores/vus_<kind>/             VUS scores computed by analysis/scoring/ (esm1bft, saluki, seq, fimo, alphagenome)
├── models/ESM1b-FT/               ESM1b-FT checkpoint (copied)
└── analysis/
    ├── prep/                      inputs built by prep_*.py / fig_vus.py
    │   ├── main_table/            prep_main_table.py
    │   ├── clingen_erepo/         prep_clingen.py: erepo_variants.tsv, erepo_summary.tsv
    │   └── vus/                   fig_vus.py: missing_alphagenome_bergquist.tsv
    ├── figures/<ensemble>/        every manuscript figure, main and supplementary: <fig>.pdf (editable text) + .png
    ├── tables/<ensemble>/         <fig>_<panel>.tsv for every panel + manifest_<fig>.json
    ├── tables/                    score_coverage.tsv, variant_sets.tsv (python analysis/core.py)
    ├── variant_lists/<ensemble>/  variants for manual review, every score, call, feature and source column:
    │                              fig6_resolved, figS13_resolved, fig7_discordant, fig7_candidates
    ├── models/<ensemble>/         MetaMissense.joblib/.json, xgb_params.json, every ablation ensemble (.joblib, .npz
    │                              predictions, .json key); reused when features, variants, hyperparameters and seed match
    └── logs/                      logs of every step
```

`data/external/` holds the Saluki models, Ensembl 115 and the JASPAR 2026 motif file; `data/bed`, `data/sequence` and `data/splits` are
the splits of `preprocessing/` (Table 2) and are only read. **Variant lists** (`variant_lists/<ensemble>/`, for checking variants one by one against other evidence): one row per
variant and group. First the selection (fig6 / figS13 `_resolved`: `resolved_by`, class, focal score, RNA or SpliceAI
score, `reference_wrong` / `reference_correct`; `fig7_discordant`: model, direction, case or control, `match_case`,
focal score; `fig7_candidates`: every GLM-Missense-pathogenic VUS against most comparators, `RNA_test_candidate`), then `oriented_<score>` and `call_<score>` (P / B / empty) for every score,
`feature_<name>` (VUS: AlphaGenome, Saluki, sequence features), and every column of the source table (`src_<column>`:
ClinVar main table, SGE table or annotated VUS table).

`manifest_<fig>.json` records the ensemble version, base, the column each
score came from, input tables, n and variant-ID md5 of the training and evaluation sets, XGBoost hyperparameters and
seed, n_boot (VUS figures: also the VUS and AlphaGenome tables and the missing-feature counts).

## Figures

Every figure is `results/analysis/figures/<ensemble>/<figure>.pdf` (+ `.png`); its tables are
`tables/<ensemble>/<figure>_<table>.tsv`.

| Figure | Script | Data | Panels |
|---|---|---|---|
| fig4 | `fig_complementarity.py` | ClinVar | A pairwise Kendall's τ (base + GLM-Missense); B partial Spearman with the label (+ Random); C drop-column ablation of base + GLM-Missense + Random on all test variants (circle) and on rare ones (triangle; gnomAD AF < `--rare` 1e-4 or absent; same models) |
| fig5 | `fig_complementarity.py` | SGE | fig4 on the RAD51D SGE data, base (ESM1b-FT); C all variants only |
| fig6 | `fig_splicing.py` | SGE | A RNA score of GLM-Missense-resolved vs background variants; B +GLM-Missense \| base (ESM1b-FT) + SpliceAI: ΔAUROC, ΔAUPRC, log-loss reduction (bars, 95% CI, bootstrap p) |
| fig7 | `fig_vus.py` | VUS | discordant VUS, GLM-Missense: A SpliceAI, B mean ΔCSC (six CSC sets), C Saluki Δ half-life; case vs nearest matched control |
| figS8 | `fig_leaderboard.py` | ClinVar | AUROC, AUPRC and log-loss reduction vs the prevalence of MetaMissense and each score alone (base, GLM-Missense, GERP++, phyloP 100/470) |
| figS9 | `fig_leaderboard.py` | ClinVar | what each score (GLM-Missense, GERP++, phyloP 100/470) adds to base + Random |
| figS10 | — | — | ESM1b-FT training curve (made outside `analysis/`) |
| figS11 | `fig_complementarity.py` | ClinVar | fig4 with ESM1b-FT in place of GLM-Missense (protein-model control); C all variants only |
| figS12 | `fig_circularity.py` | ClinVar, ClinGen | label circularity: A ClinGen-classified variants through the filters, by whether the label holds without PP3 / BP4; B PP3 / BP4 applied, by strength; C AUROC of the base on every ClinGen-classified late-release variant (training and test chromosomes) with every base score vs with the PP3 / BP4-dependent labels removed |
| figS13 | `fig_splicing.py` | ClinVar | SpliceAI of A GLM-Missense-resolved and B ESM1b-FT-resolved vs background |
| figS14 | `fig_vus.py` | VUS | fig7 for ESM1b-FT (protein-model control), same y axes |

## Variants

- A missing score is never imputed: a variant without one of the scores an analysis uses is left out.
- **ClinVar figures share one variant set**: training and test variants with every score these figures use (base,
  GLM-Missense, ESM1b-FT, SpliceAI, Random, GERP++, phyloP 100/470). Every ablation ensemble is trained on the
  training part of this set, so each needs all of its features. Exception: figS12 uses every ClinGen-classified
  late-release variant (training and test chromosomes) with every base score, and shows the base only (the
  training-chromosome variants share genes with GLM-Missense's training data).
- **MetaMissense** is trained on every training variant with all of its own inputs (base + GLM-Missense).
- **SGE**: SGE variants with every SGE score (base (ESM1b-FT), GLM-Missense, SpliceAI, Random, conservation); their
  ensembles are trained on the ClinVar training variants with the same scores. Labels: functionally abnormal = 1,
  functionally normal = 0, other classes left out.
- **Rare** (fig4 C): evaluation variants with gnomAD AF < 1e-4 (`--rare`; first value of `--af_col`, default
  `gnomAD4.1_joint_AF`) or absent from gnomAD; the same ensembles as the all-variant rows, evaluated on that subset
  (`<fig>_counts.tsv`: n per stratum, n absent from gnomAD).
- **VUS**: held-out ClinVar missense VUS with every score the discordant selection needs (base, GLM-Missense,
  ESM1b-FT; SpliceAI is not required for this). The three fig7 features (SpliceAI, mean ΔCSC of the six Wu et al.
  2019 CSC sets, Saluki Δ half-life) are required: only VUS with SpliceAI, all six ΔCSC sets and Saluki, filtered
  before cases and controls are chosen, so every panel is on the same pairs (fig7, figS14). The figures show all pairs; the same tests on the
  pairs whose case and control both have SpliceAI < 0.2 are in `<fig>_main_nosplice.tsv` (own q). Control: of the
  unused VUS with the case's gene and number of comparators against, the nearest in genomic position; `--seed`
  (default 0) only breaks ties. Both models are analysed on
  the same VUS. Other features (AlphaGenome, FIMO, ...) are
  never required: each is tested on the selected VUS that have it.

## Methods

- **Ensembles**: XGBoost (`tree_method hist`, `eval_metric logloss`, seed 42). MetaMissense is grid-searched once
  (n_estimators 100/300/500, max_depth 3/4/5, learning_rate 0.01/0.05/0.1, subsample 0.8/1.0; AUROC on 5
  chromosome-exclusive folds) and refit; every other ensemble uses its hyperparameters, fixed, fit once and evaluated
  once.
- **Δ panels**: two ensembles that differ by one score, on the same variants; ΔAUROC, ΔAUPRC and log-loss reduction
  side by side; positive = the score adds information the others do not carry. Drop-column (− X) = full minus without
  X; +X = with X minus without.
- **Statistics**: 1000 class-stratified bootstrap resamples (`default_rng(0)`), the same for every row of a panel; 95%
  percentile CI; two-sided bootstrap p; filled + stars = BH q < 0.05 within the panel. Absolute performance (figS8)
  is bars: solid = not different from the top score (p ≥ 0.05). Single-score log-loss: quantile-normal transform +
  logistic regression on the training split; reduction vs predicting the training prevalence.
- **Calls** (`core.call`; fig6, fig7, figS13). The base tools use the Bergquist et al. 2025 Table 1 Supporting
  intervals — pathogenic at or above the PP3 Supporting lower bound, benign at or below the BP4 Supporting upper bound,
  in between no call: AlphaMissense ≥ 0.792 / ≤ 0.169, ESM1b ≤ −10.7 / ≥ −6.3, REVEL ≥ 0.644 / ≤ 0.290, BayesDel ≥
  0.130 / ≤ −0.180, MutPred2 ≥ 0.737 / ≤ 0.391, VEST4 ≥ 0.764 / ≤ 0.449, VARITY_R ≥ 0.675 / ≤ 0.251 (`CALIBRATED` in
  `core.py`). Scores without a calibration (GLM-Missense, ESM1b-FT): pathogenic at or above 0.5, else benign
  (`THRESH`). **Most of a set** = at least 2/3 of it, rounded up (`core.k_of`): 5 of 7 for the base.
- **Resolved** (fig6, figS13): correct by the focal model while most of its reference predictors are not correct (no
  call = not correct). Reference: ClinVar, the base; SGE, base (ESM1b-FT). Mann–Whitney U
  within class, two-sided, unadjusted (one pre-specified comparison per class).
- **+GLM-Missense | base + SpliceAI** (fig6 B): base (ESM1b-FT) + SpliceAI with vs without GLM-Missense, both
  trained on the ClinVar training variants and evaluated on the SGE variants; one comparison, so p is not adjusted.
- **Discordant VUS** (fig7): case = the model's confident call (≥ 0.8 / ≤ 0.2) opposed by most comparators
  (the base); control = the model's opposite confident call with the same number of comparators against, same gene
  (1 per case, without replacement; cases without one dropped; most confident first, ≤ 2,000 per model × direction);
  two-sided Wilcoxon signed-rank on the pairs; q = Benjamini-Hochberg over the tests the figure shows (its features ×
  both directions, all pairs), per model: 6 in fig7 and figS14; specificity = AUC(GLM-Missense) − AUC(ESM1b-FT)
  (bootstrap CI and p, BH over the same features × directions); RNA-testing candidates. Features: SpliceAI, AlphaGenome Atlas (every scorer ×
  summary across tracks), Saluki Δ half-life, sequence features (ΔCSC: codon stability coefficient of the alt minus
  the ref codon, Wu et al. 2019; fig7 uses the HEK293T ORFeome set, whose reporters share their UTRs, so it measures
  the codon effect alone), FIMO TF-binding disruption (below). Other features (and the subset without splicing signal,
  SpliceAI < 0.2) are tested in `fig7_tests.tsv` with unadjusted p, not reported. Without labels, an enrichment shows what a model responds to, not that its call is
  right.
- **FIMO TF-binding disruption** (`fimo_*`; tested in `fig7_tests.tsv`, not in a figure): FIMO 5.5 (`--text --thresh 1e-3 --max-strand`,
  uniform background, motif pseudocount 0.1) on ref and alt of pos ± 32 bp (GRCh38 + strand, both strands scanned),
  JASPAR 2026 CORE vertebrates non-redundant (1,019 motifs). Per motif, the best p of the hits that cover the variant
  per allele (none at p < 1e-3: censored at 1e-3); Δ = −log10 p(alt) − (−log10 p(ref)). Per variant: largest loss (−min
  Δ, ≥ 0), largest gain (max Δ, ≥ 0), their maximum (`fimo_max_abs_delta`), the Δ of the motif changed most with its
  sign (`fimo_delta`, derived in `fig_vus.py`; < 0 = binding lost), and motifs crossing p < 1e-4
  (bound in ref only = lost; alt only = gained). With ~1,000 motifs about half of random SNVs lose or gain some motif at
  that cut-off, so the count and any-loss features are noisy; the continuous ones carry the signal, if any.
- **Label circularity** (figS12): ERepo classifications matched by ClinVar Variation ID (retracted ones left
  out) and scored with the Bayesian point system (Tavtigian et al. 2020: Supporting 1, Moderate 2, Strong 4, Very
  Strong 8; P ≥ 10, LP 6–9, VUS 0–5, LB −1 to −6, B ≤ −7), with and without PP3 / BP4 (the missense-predictor
  criteria; BP7, a splicing criterion, stays in). A label depends on PP3 / BP4 if, without them, any classification
  of the variant leaves the P/LP or B/LB side of its ClinVar label. C: resamples of the full set, the reduced set taken
  within each resample (paired); resamples where it lacks a class are discarded (`n_boot_used`); BH across scores.
  `figS12_calls.tsv`: share of P/LP called pathogenic and of B/LB called benign by each base predictor (Calls),
  dependent vs other labels, Fisher's exact p (not in the figure).
- **Style**: 10 in wide, Arial (`ARIAL_DIR`, else `../fonts` next to the repo); panel letters 18 pt bold, axis labels
  16 pt, tick labels 14 pt, other text 12 pt. Figures hold data, short axis labels and keys only: no titles or
  footers (n, ensemble, training set, tests and p / q go in the legend; all are in the tables and manifests);
  significance as stars only (ns, *, **, ***); "base predictors" for the base (the seven Bergquist et al. 2025 tools), "Random (null)" for Random.

## Adding an analysis

- One script `<name>.py` that imports `core`, takes `--ensemble`, and draws its figures with
  `core.finish(fig, '<figure name>', tables, ...)`: figures go to `figures/<ensemble>/`, panel tables and manifests
  to `tables/<ensemble>/` next to the other manuscript figures; add it to `run_all.sh`.
- Input preparation in `prep_<name>.py` writing to `results/analysis/prep/<name>/`; new external inputs copied with
  a `copy` line in `copy_inputs.sh` (logged in `COPY_LOG.tsv`; its source in Inputs); new scores computed by a
  script in `scoring/` writing to `results/scores/<name>/`, added to the annotated table by `merge_vus.py` if they
  are VUS scores.
- Models from `core.metamissense()` / `core.ensemble()` (cached in `models/<ensemble>/`), so every analysis uses the
  same MetaMissense and ablation ensembles.

## Inputs

Every figure reads four input tables (plus reference files). Each is either built by the commands below or copied
into place by `copy_inputs.sh` (source folders in `analysis/copy_inputs.local.sh`, not tracked; every copy logged in
`results/annotated/COPY_LOG.tsv`).

| Input | Path (read by) | Made by |
|---|---|---|
| ClinVar main tables | `results/annotated/clinvar/ClinVar.260923only.BLBvsPLP_{train,test}.main.tsv` (`core.py --main_dir`) | `prep_main_table.py` (below) |
| RAD51D SGE table | `core.SGE_FILE` under `results/annotated/sge/` (`core.py --sge`) | **[TBA-A2]** |
| held-out VUS table | `results/annotated/vus/ClinVar_20260923_VUS_missense_heldoutOnly_dbNSFP_GLM-Missense_spliceAI_annot.tsv` (`scoring/merge_vus.py`) | **[TBA-A3]** |
| ClinGen Evidence Repository | `data/clingen/` → `results/analysis/prep/clingen_erepo/` (`fig_circularity.py`) | `prep_clingen_download.sh`, `prep_clingen.py` |
| ESM1b-FT checkpoint | `results/models/ESM1b-FT/exp_1_concat_diff/best_model.pt` | `training/` (**[TBA-T2]**) |
| ESRseq hexamers, codon stability coefficients | `results/annotated/vus/reference/` (`scoring/seq_features.py`) | **[TBA-A4]** |
| Saluki models | `data/external/saluki/train_gru/f*_c*/train/model0_best.h5`, `params.json` | Zenodo 10.5281/zenodo.6326409 (**[TBA-A4]**) |
| Ensembl GRCh38 release 115 FASTA + GTF | `data/external/ensembl115/` | `wget https://ftp.ensembl.org/pub/release-115/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz` and `.../release-115/gtf/homo_sapiens/Homo_sapiens.GRCh38.115.gtf.gz`, then `gunzip` |

### ClinVar main tables

One table per later-release split (training: chromosomes other than 2, 3, 16, 17; test: 2, 3, 16, 17) with every
score the figures use. `prep_main_table.py` joins, per split:

| Column(s) | From | Made by |
|---|---|---|
| dbNSFP 5.3.1a columns (strict values plus `<score>_fb` / `<score>_tier`, `clinvar_match`, `matched_enst`), `GLM-Missense_score`, `spliceai_*` | `results/ClinVar.260923only.missense.hg38.seq12k.BLBvsPLP_{training,validation}_GLM-Missense_annotated_dbNSFPannotated.tsv` | `scoring/GLM-Missense.py`, `scoring/annotate_dbnsfp.py --clinvar_tx`, `analysis/scoring/replace_dbnsfp_columns.py --fallback_tsv`, `analysis/scoring/annotate_spliceai.py --files` (**[TBA-A1]**) |
| `label`, `clinvar_class`, `split` | `data/splits/…BLBvsPLP_{training,validation}.tsv`, `data/bed/ClinVar.260923only.*.bed` | `preprocessing/` |
| `ESM1b-FT_score` | `results/ClinVar.260923only.missense.hg38.seq12k.BLBvsPLP_{training,validation}_ESM1b-FT_annotated.tsv` | **[TBA-A5]** |
| `NT2-ZS_score`, `NT2-ZS_llr` | `results/predictions/ClinVar.260923only.BLBvsPLP[_train]/zeroshot_NT2_seq12k.tsv` | `evaluation/zeroshot_nt.py` |
| `random_baseline_score` | `results/scores/random_baseline.tsv` | `prep_random_baseline.py` |

```bash
conda activate lamar
python analysis/prep_random_baseline.py
for s in train test; do python analysis/prep_main_table.py --split $s --outdir results/annotated/clinvar; done
python analysis/core.py                       # score columns and coverage
```

### To be added

> **[TBA-A1] dbNSFP, GLM-Missense and SpliceAI table of the later-release splits.** The commands that produce
> `results/ClinVar.260923only.missense.hg38.seq12k.BLBvsPLP_{training,validation}_GLM-Missense_annotated_dbNSFPannotated.tsv`
> from the split files: GLM-Missense scores (`scoring/GLM-Missense.py`), dbNSFP 5.3.1a with the ClinVar transcript
> (`scoring/annotate_dbnsfp.py --clinvar_tx data/clinvar/clinvar_transcripts.tsv`), the transcript-tier columns
> (`analysis/scoring/replace_dbnsfp_columns.py --fallback_tsv`) and SpliceAI
> (`analysis/scoring/annotate_spliceai.py --files`; raw SNV scores, `data/spliceai/spliceai_scores.raw.snv.hg38.vcf.gz`).

> **[TBA-A2] RAD51D SGE table (Figs 5, 6).** Source: Casadei et al. (2026) RAD51D saturation genome editing data
> (download and file to be named here, under `data/sge/`). To add: a `prep_sge.py` (or commands) that keeps missense
> SNVs annotated on ENST00000345365.11 and adds dbNSFP (`scoring/annotate_dbnsfp.py`), GLM-Missense
> (`scoring/prepare_glm_input.py`, `scoring/GLM-Missense.py`), ESM1b-FT (as for the VUS, section 2.2) and SpliceAI
> (`analysis/scoring/annotate_spliceai.py`).
> Output: the file `core.SGE_FILE` with the columns `core.sge()` reads: `variant_id`, `position`, `TRANSCRIPT_ID`,
> `FUNC_CONSEQUENCE` (`functionally_abnormal` / `functionally_normal` / other), `RNA_SCORE`, the dbNSFP columns with
> `Ensembl_transcriptid`, `GLM-Missense_score`, `ESM1b-FT_score`, `spliceai_DS_max`.

> **[TBA-A3] Held-out VUS table (Fig 7, Fig S14).** From the held-out VUS set (`preprocessing/`, **[TBA-P3]**): GLM-Missense
> scores, dbNSFP 5.3.1a, SpliceAI, and the ClinVar transcript and protein change.
> Output: `results/annotated/vus/ClinVar_20260923_VUS_missense_heldoutOnly_dbNSFP_GLM-Missense_spliceAI_annot.tsv`
> with `variant_id`, `chromosome`, `position`, `ref_allele`, `alt_allele`, `gene`, `TRANSCRIPT_ID` (Ensembl),
> `HGVS_C`, `PROTEIN_CHANGE` (e.g. `p.Ile293Val`), `clinvar_id`, the dbNSFP columns with `Ensembl_transcriptid`,
> `GLM-Missense_score`, `spliceai_DS_max` (n = 526,428 before filtering for scores).

> **[TBA-A4] Reference files of the sequence features.** Download and file names of the ESRseq hexamer scores
> (Ke et al. 2011) and the six human codon stability coefficient sets (Wu et al. 2019), as read by
> `scoring/seq_features.py`, and the layout of the Saluki model download.

> **[TBA-A5] ESM1b-FT scores of the later-release ClinVar variants.** The commands (as for the VUS, section 2.2) that
> produce `results/ClinVar.260923only.missense.hg38.seq12k.BLBvsPLP_{training,validation}_ESM1b-FT_annotated.tsv`
> (columns `chromosome`, `position`, `ref_allele`, `alt_allele`, `ESM1b-FT_score`).

> **[TBA-A7] Table 1 (RAD51D variants resolved by GLM-Missense).** `fig_splicing.py` writes the functionally
> abnormal resolved variants with their scores to `results/analysis/variant_lists/<ensemble>/fig6_resolved.tsv`. To
> add: the exon number, exon-exon junction distance and ESE hotspot status columns, and the script that writes the
> table (`results/analysis/tables/<ensemble>/table1.tsv`).

> **[TBA-A8] Table S2 (ClinVar variants resolved by GLM-Missense).** As [TBA-A7], from
> `variant_lists/<ensemble>/figS13_resolved.tsv` (pathogenic or likely pathogenic, GLM-Missense-resolved): exon-exon
> junction distance and the ClinVar splicing-evidence status of each variant (`results/analysis/tables/<ensemble>/tableS2.tsv`).

> **[TBA-A6] Environments.** `analysis/envs/<env>.yml` for `lamar` (PyTorch, `transformers`, `peft`; also the
> training env) and `saluki` (section 1: `conda env export -n <env> --no-builds`).
