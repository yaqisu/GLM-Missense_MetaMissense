#!/usr/bin/env bash
# Every manuscript figure of the released ensemble version (bergquist). Run from the repo root, conda env lamar, after
# the inputs and scores (README: 2. Inputs and scores):
#   AG_KEY_FILE=$HOME/.alphagenome_api_key bash analysis/run_all.sh   # every figure
#   ARGS="--n_boot 100" bash analysis/run_all.sh                      # quick preview
# Figures (results/analysis/figures/<ensemble>/):
#   main  fig4 ClinVar complementarity (C: all + rare) | fig5 SGE complementarity | fig6 SGE resolved + GLM-Missense | base + SpliceAI |
#   supp  figS8 leaderboard | figS9 added to base + Random | figS10 ESM1b-FT training curve (not made here) |
#         figS11 fig4 with ESM1b-FT | figS12 ClinGen circularity | figS13 ClinVar resolved, SpliceAI |
#         figS14 fig7 for ESM1b-FT (control)
# AG_KEY_FILE: fig_vus.py scores the selected VUS that lack AlphaGenome features (conda env alphagenome); without it
# they are tested without AlphaGenome features. AG_SOURCE=live: live per-track predictions instead of the Atlas.
set -euo pipefail

E=bergquist                                                  # ensemble version (core.ENSEMBLES)
N_JOBS=${N_JOBS:-40}
ARGS=${ARGS:-}                                               # extra options for every script, e.g. --n_boot 100
VUS_ARGS=${VUS_ARGS:-}                                       # extra options for fig_vus.py, e.g. --max_variants 1000
LOG=results/analysis/logs
mkdir -p $LOG

python -u analysis/core.py $ARGS 2>&1 | tee $LOG/core_check.log      # score columns, coverage, variant sets
python -u analysis/table_clinvar_sets.py 2>&1 | tee $LOG/table_clinvar_sets.log   # Methods table of the ClinVar sets
# fig_leaderboard first (figS8, figS9): it grid-searches MetaMissense, whose hyperparameters every other ensemble reuses
for S in fig_leaderboard fig_complementarity fig_splicing fig_circularity; do
  python -u analysis/$S.py --ensemble $E --n_jobs $N_JOBS $ARGS 2>&1 | tee $LOG/${E}_$S.log
done
# fig7 last; it reads the annotated VUS table (scoring/merge_vus.py)
python -u analysis/fig_vus.py --ensemble $E --n_jobs $N_JOBS $ARGS $VUS_ARGS 2>&1 | tee $LOG/${E}_fig_vus.log \
    || echo "fig_vus.py failed: see $LOG/${E}_fig_vus.log"