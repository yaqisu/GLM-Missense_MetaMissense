# Evaluation

Figures of model development: how GLM-Missense was built (Figs 2, 3; Figs S2–S7, S10; Table S1). The training runs
are in `training/`; this folder draws their results and compares GLM-Missense with zero-shot NT-2. The analyses of
what GLM-Missense adds to established predictors (Figs 4–7, S8, S9, S11–S14) are in `analysis/`.

Every command runs from the **repo root**. Figures are written to `results/evaluation/figures/<figure>.pdf`
(editable text, Arial) and `.png`.

---

## Figures

| Figure | Script | Reads |
|---|---|---|
| Fig 1 | — (schematic) | — |
| Fig 2A | `plot_frozen_bar.py` | validation AUROC of each frozen-backbone run (`results.tsv`) |
| Fig 2B | `plot_repr_window_bar.py` (`fig_window`) | local-window pooling runs |
| Fig 3A | `summarize_sweep.py` → `plot_lora_bar_updated.py` | LoRA model-selection runs |
| Fig 3B | `plot_repr_window_bar.py` (`fig_repr`) | allele-aware input runs |
| Fig S2 | `frozen_model_comparison.py` | `results/frozen_results.tsv` |
| Fig S3 | — (drawn from the runs' metrics, below) | local-window pooling runs |
| Fig S4 | `plot_lora_fullFT.py` | full fine-tuning runs (`exp_55`–`exp_59`) and the full-split LoRA run |
| Fig S5 | — (drawn from the runs' metrics, below) | allele-aware input runs |
| Fig S6 | `plot_training_curves.py` | `results/NT2_seq12k_BLBvsPLP_ref_alt_contrast_mlp/exp_1_concat_diff/training_metrics.csv` |
| Fig S7 | `zeroshot_nt.py` → `fig_zeroshot.py` | GLM-Missense and zero-shot NT-2 scores of the held-out test set |
| Fig S10 | `plot_training_curves.py --name figS10` | `results/models/ESM1b-FT/exp_1_concat_diff/training_metrics.csv` |
| Table S1 | `training/NT2_ranked_cv.py` | `results/NT2_seq12k_BvsP_ranked_cv/cv_summary.csv` |

---

## Fig S7 — GLM-Missense vs zero-shot NT-2

Zero-shot NT-2: masked marginal log-likelihood ratio of the pretrained NT-2 on the same 11,999-bp window,
`sigmoid(−[log P(alt) − log P(ref)])` (Methods 5.6).

```bash
D=results/predictions/ClinVar.260923only.BLBvsPLP
S=data/splits/ClinVar.260923only.missense.hg38.seq12k.BLBvsPLP
# GLM-Missense scores of the held-out test set (scoring/README.md)
python scoring/GLM-Missense.py --input ${S}_validation.tsv --model scoring/GLM-Missense.pt --outdir $D
# zero-shot NT-2 (GPU)
python evaluation/zeroshot_nt.py --input ${S}_validation.tsv --output $D/zeroshot_NT2_seq12k.tsv \
    --model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species --gpu 0
# figure: results/evaluation/figures/figS7.{pdf,png,tsv}
python evaluation/fig_zeroshot.py
```

`zeroshot_nt.py` on the training split (`--input ${S}_training.tsv --output ${D}_train/zeroshot_NT2_seq12k.tsv`)
gives the `NT2-ZS` column of the training main table (`analysis/prep_main_table.py`).

## Fig S2 and Fig S6

```bash
python evaluation/frozen_model_comparison.py      # Fig S2 -> results/evaluation/figures/figS2.{pdf,png}
python evaluation/plot_training_curves.py         # Fig S6 -> results/evaluation/figures/figS6.{pdf,png}
```

---

## Fig 2, Fig 3, Fig S4, Fig S10 and Table S1

```bash
# Fig 2A: validation AUROC of every frozen-backbone configuration (NT-2 12k / 6k, NT-1, Caduceus-PS x heads x
# embeddings; values in RECORDS, from results.tsv) -> results/figures/fig_frozen.{pdf,png}
python evaluation/plot_frozen_bar.py

# Fig 3A: rank the 54 LoRA configurations (mean of the AUROC and AUPRC ranks) -> experiment_ranking.json
# (input of training/NT2_ranked_cv.py) and summary_ranked_by_{auroc,auprc,mean_rank}.csv in the sweep folder
python evaluation/summarize_sweep.py results/NT2_seq12k_BvsP_lora_fullFT_sweep --csv
python evaluation/plot_lora_bar_updated.py results/NT2_seq12k_BvsP_lora_fullFT_sweep/summary_ranked_by_mean_rank.csv \
    --fine-tuning lora --out-path results/evaluation/figures/fig3A          # fig3A.{pdf,png}

# Fig 2B (fig_window) and Fig 3B (fig_repr): one table per figure, then both bar charts
python - <<'EOF'
import json, pathlib, pandas as pd
w = pathlib.Path('results/NT2_seq12k_BvsP_window_pooling'); rows = []
for e in sorted(w.glob('exp_*')):
    c = json.load(open(e / 'summary.json'))['config']
    m = pd.read_csv(e / 'training_metrics.csv')
    m['exp_id'], m['embedding_strategy'], m['window_half_size'] = c['exp_id'], c['embedding_strategy'], c['window_half_size']
    rows.append(m)
pd.concat(rows).to_csv(w / 'window_metrics.csv', index=False)
s = json.load(open('results/NT2_seq12k_BvsP_lora_fullFT_sweep/exp_36/summary.json'))      # alt
alt = pd.DataFrame([dict(exp_id=36, combine_mode='alt_only', best_val_auroc=s['best_val_auroc'],
                         best_val_auprc=s['best_val_auprc'])])
ra = pd.read_csv('results/NT2_seq12k_BvsP_allele_inputs/all_experiments_summary.csv')       # ref-alt, ref-alt-diff
pd.concat([alt, ra])[['exp_id', 'combine_mode', 'best_val_auroc', 'best_val_auprc']] \
  .to_csv('results/NT2_seq12k_BvsP_allele_inputs/repr_summary.csv', index=False)
EOF
python evaluation/plot_repr_window_bar.py results/NT2_seq12k_BvsP_allele_inputs/repr_summary.csv \
    results/NT2_seq12k_BvsP_window_pooling/window_metrics.csv --outdir results/evaluation/figures --png
# -> fig_repr.{pdf,png} (Fig 3B), fig_window.{pdf,png} (Fig 2B; the variant-position and full mean-pool bars are
#    the frozen NT-2 CNN values in WINDOW_EXTRA)

# Fig S4: training and validation AUROC / AUPRC of the five full fine-tuning runs and the full-split LoRA run
python - <<'EOF'
import json, pathlib, pandas as pd
d = pathlib.Path('results/NT2_seq12k_BvsP_lora_fullFT_sweep'); rows = []
for e in range(55, 60):
    m = pd.read_csv(d / f'exp_{e}/training_metrics.csv')
    m['exp_id'] = e
    m['learning_rate'] = json.load(open(d / f'exp_{e}/summary.json'))['config']['learning_rate']   # configured LR
    rows.append(m)
pd.concat(rows).to_csv(d / 'fullFT_metrics.csv', index=False)
EOF
python evaluation/plot_lora_fullFT.py results/NT2_seq12k_BvsP_lora_fullFT_sweep/fullFT_metrics.csv \
    results/NT2_seq12k_BvsP_lora_fullData/exp_1/training_metrics.csv -o results/evaluation/figures/figS4.pdf

# Fig S10: as Fig S6, for the ESM1b-FT run (same training_metrics.csv columns; best step 3,000)
python evaluation/plot_training_curves.py --name figS10 --model ESM1b-FT \
    --input results/models/ESM1b-FT/exp_1_concat_diff/training_metrics.csv     # -> figS10.{pdf,png}
```

Table S1: `training/NT2_ranked_cv.py` writes the mean and standard error over the five folds of each configuration to
`results/NT2_seq12k_BvsP_ranked_cv/cv_summary.csv` (per fold: `all_fold_results.csv`).

---

## Fig S2, Fig S3 and Fig S5 inputs

`results/frozen_results.tsv`, the input of Fig S2, is the experiment log of the frozen-backbone runs (`results.tsv`),
compiled by hand: one row per run (`Model`, `Fine-tuning`, `Embedding strategy`, `Classifier head`,
`Classifier head - hidden layers`, `Input length`, ...) with the training and validation AUROC every 1,000 steps
(`Training_Step{N}_AUC`, `Val_Step{N}_AUC`) from the run's `training_metrics.csv`. `frozen_model_comparison.py` keeps
the rows whose `Fine-tuning` contains "frozen".

Fig S3 and Fig S5 (training and validation AUROC / AUPRC every 1,000 steps) were drawn directly from the runs'
`training_metrics.csv` files, without a script in this repository: Fig S3 from
`results/NT2_seq12k_BvsP_window_pooling/exp_*/`, Fig S5 from `results/NT2_seq12k_BvsP_lora_fullFT_sweep/exp_36/` (alt),
`results/NT2_seq12k_BvsP_allele_inputs/exp_*/` (ref-alt, ref-alt-diff) and
`results/NT2_seq12k_BvsP_alt_triple/exp_2_alt_triple/` (alt-alt-alt).
