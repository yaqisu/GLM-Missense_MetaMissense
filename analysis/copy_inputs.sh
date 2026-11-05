#!/usr/bin/env bash
# Copy the analysis inputs made elsewhere into place and record each copy: results/annotated/COPY_LOG.tsv gets one
# line per file (date, destination, source, md5, user, note) and results/annotated/MD5SUMS the checksums of every
# input. cp -p keeps timestamps; -n never overwrites, so a rerun only re-logs and re-checks. Repo root, any env:
#   bash analysis/copy_inputs.sh
# Where each input comes from: README, "Inputs". This script is only for copying inputs that were already made
# elsewhere; the source folders are set in analysis/copy_inputs.local.sh (not tracked in git), e.g.
#   SRC_TABLES=/path/to/tables   # ClinVar main tables (clinvar/), RAD51D SGE table (sge/), ESM1b-FT checkpoint
#   SRC_VUS=/path/to/vus         # held-out VUS table
#   SRC_REF=/path/to/reference   # ESRseq / codon stability tables, earlier live AlphaGenome features
#   SRC_SALUKI=/path/to/train_gru  SRC_ENSEMBL=/path/to/ensembl115
set -euo pipefail

# ── sources ──────────────────────────────────────────────────────────────────
LOCAL=analysis/copy_inputs.local.sh
[ -f $LOCAL ] || { echo "ERROR: $LOCAL not found (sets the SRC_* source folders; see the top of this script)" >&2; exit 1; }
source $LOCAL
: "${SRC_TABLES:?}" "${SRC_VUS:?}" "${SRC_REF:?}" "${SRC_SALUKI:?}" "${SRC_ENSEMBL:?}"
REVISION=$SRC_TABLES
VUS_SRC=$SRC_VUS
REF_SRC=$SRC_REF
SALUKI_SRC=$SRC_SALUKI
ENSEMBL_SRC=$SRC_ENSEMBL
# file names inside the source folders (defaults; override in $LOCAL)
SRC_CLINVAR_DIR=${SRC_CLINVAR_DIR:-$REVISION/clinvar}
SRC_SGE_FILE=${SRC_SGE_FILE:-$REVISION/sge/RAD51D_missense_dbNSFP_annotated_GLM-Missense_annotated_ESM1b-FT_annotated.spliceai_annot.ESCOTT_annot.tsv}
SRC_ESM1BFT=${SRC_ESM1BFT:-$REVISION/ESM1b-FT/exp_1_concat_diff/best_model.pt}
SRC_AG_EARLIER=${SRC_AG_EARLIER:-$REF_SRC/alphagenome_features.tsv}
# ─────────────────────────────────────────────────────────────────────────────

A=results/annotated
V=$A/vus
LOG=$A/COPY_LOG.tsv
mkdir -p $A/clinvar $A/sge $V/reference results/models/ESM1b-FT/exp_1_concat_diff results/scores/vus_alphagenome_live \
         data/external/saluki/train_gru data/external/ensembl115
[ -f $LOG ] || printf 'date\tdestination\tsource\tmd5\tuser\tnote\n' > $LOG

log () {   # log DEST SRC NOTE
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%F)" "$1" "$2" "$( [ -f "$1" ] && md5sum < "$1" | cut -d' ' -f1)" \
         "${USER:-$(id -un)}" "$3" >> $LOG
}
copy () {   # copy SRC DEST_FILE NOTE
  local src=$1 dst=$2 note=$3
  if [ -e "$dst" ]; then note="already present, not copied; $note"
  elif [ -e "$src" ]; then cp -pn "$src" "$dst"
  else echo "WARNING: missing source $src" >&2; note="SOURCE MISSING; $note"; fi
  if [ -e "$src" ] && [ -e "$dst" ] && ! cmp -s "$src" "$dst"; then
    echo "WARNING: $dst differs from $src" >&2; note="DIFFERS from source; $note"; fi
  log "$dst" "$src" "$note"
}

# ClinVar main tables (prep_main_table.py)
for s in train test; do
  f=ClinVar.260923only.BLBvsPLP_$s.main.tsv
  copy $SRC_CLINVAR_DIR/$f $A/clinvar/$f 'ClinVar main table: dbNSFP 5.3.1a, GLM-Missense, ESM1b-FT, SpliceAI, NT2-ZS, Random'
done
# RAD51D SGE missense variants
f=RAD51D_SGE_missense_annotated.tsv
copy $SRC_SGE_FILE $A/sge/$f 'RAD51D SGE: dbNSFP, GLM-Missense, ESM1b-FT, SpliceAI'
# held-out ClinVar VUS
f=ClinVar_20260923_VUS_missense_heldoutOnly_dbNSFP_GLM-Missense_spliceAI_annot.tsv
copy $VUS_SRC/$f $V/$f 'held-out ClinVar 20260923 missense VUS: dbNSFP, GLM-Missense, SpliceAI'
# reference tables of the sequence features
copy $REF_SRC/ESRseq_scores_Ke2011.tsv $V/reference/ESRseq_scores_Ke2011.tsv 'ESRseq hexamers (Ke et al. 2011)'
copy $REF_SRC/ORFeome_CSCs_wu2019.csv $V/reference/ORFeome_CSCs_wu2019.csv 'codon stability (Wu et al. 2019)'
# live AlphaGenome features of the earlier VUS selections (alphagenome_score.py --compare / --compare_tables)
copy $SRC_AG_EARLIER \
     results/scores/vus_alphagenome_live/alphagenome_features_earlier.tsv 'AlphaGenome live per-track features, earlier selections'
# ESM1b-FT checkpoint
copy $SRC_ESM1BFT \
     results/models/ESM1b-FT/exp_1_concat_diff/best_model.pt 'ESM1b-FT checkpoint (scoring/fine-tune_ESM1b.py)'
# Saluki models (Zenodo 10.5281/zenodo.6326409)
if [ -z "$(ls -A data/external/saluki/train_gru)" ]; then
  if rsync -a --include='*/' --include='model0_best.h5' --include='params.json' --exclude='*' \
      $SALUKI_SRC/ data/external/saluki/train_gru/; then note='copied'
  else echo "WARNING: Saluki models not copied" >&2; note='SOURCE MISSING'; fi
else note='already present, not copied'; fi
log data/external/saluki/train_gru/params.json $SALUKI_SRC/ \
    "Saluki models: $(ls data/external/saluki/train_gru/*/train/model0_best.h5 2>/dev/null | wc -l) x model0_best.h5 + params.json; $note"
# Ensembl GRCh38 release 115 (Saluki, sequence features, ESM1b-FT protein mapping)
for f in Homo_sapiens.GRCh38.115.dna.primary_assembly.fa Homo_sapiens.GRCh38.115.gtf; do
  copy $ENSEMBL_SRC/$f data/external/ensembl115/$f 'Ensembl GRCh38 release 115'
done

# checksums of every input the analyses read
md5sum $A/clinvar/*.main.tsv $A/sge/*.tsv $V/ClinVar_20260923_VUS_missense_heldoutOnly_*.tsv $V/reference/* \
    results/scores/vus_alphagenome_live/alphagenome_features_earlier.tsv \
    results/models/ESM1b-FT/exp_1_concat_diff/best_model.pt \
    data/clingen/erepo.tabbed.txt 2>/dev/null > $A/MD5SUMS || true
echo "logged to $LOG; checksums in $A/MD5SUMS"
tail -n 12 $LOG | cut -f1,2,6
