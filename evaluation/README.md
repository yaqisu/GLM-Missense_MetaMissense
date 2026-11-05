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
| Fig 2A | **[TBA-E1]** | frozen-backbone runs (`training/`) |
| Fig 2B | **[TBA-E2]** | local-window pooling runs |
| Fig 3A | **[TBA-E3]** | LoRA model-selection runs |
| Fig 3B | **[TBA-E4]** | allele-aware input runs |
| Fig S2 | `frozen_model_comparison.py` | `results/frozen_results.tsv` (**[TBA-E1]**) |
| Fig S3 | **[TBA-E2]** | local-window pooling runs |
| Fig S4 | **[TBA-E5]** | LoRA vs full fine-tuning runs |
| Fig S5 | **[TBA-E4]** | allele-aware input runs |
| Fig S6 | `plot_training_curves.py` | `results/NT2_seq12k_BLBvsPLP_ref_alt_contrast_mlp/exp_1_concat_diff/training_metrics.csv` |
| Fig S7 | `zeroshot_nt.py` → `fig_zeroshot.py` | GLM-Missense and zero-shot NT-2 scores of the held-out test set |
| Fig S10 | **[TBA-E7]** | ESM1b-FT training metrics (`training/`, **[TBA-T2]**) |
| Table S1 | **[TBA-E6]** | 5-fold CV runs (`training/NT2_ranked_cv.py`) |

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

## To be added

Each item: the script that draws the figure from the `training/` run folders (named in `training/README.md`,
**[TBA-T1]**), writing `results/evaluation/figures/<figure>.{pdf,png}`, and a row in the table above.

> **[TBA-E1] Fig 2A (frozen backbones).** Best validation AUROC of every frozen-backbone configuration (NT-2, NT-1,
> Caduceus-PS × heads × embeddings). Also: how `results/frozen_results.tsv` (the input of Fig S2) is compiled from the
> runs' `training_metrics.csv` files.

> **[TBA-E2] Fig 2B and Fig S3 (local-window pooling).** Validation AUROC by window size (Fig 2B) and training /
> validation AUROC and AUPRC curves of the ±8, ±16, ±32-token runs (Fig S3).

> **[TBA-E3] Fig 3A (LoRA model selection).** Validation AUROC and AUPRC of the 54 LoRA configurations, ordered by
> mean rank. The same ranking is the `--ranking_json` input of `training/NT2_ranked_cv.py`.

> **[TBA-E4] Fig 3B and Fig S5 (allele-aware inputs).** Validation performance of alt, ref-alt and ref-alt-diff
> (Fig 3B) and their training / validation curves (Fig S5).

> **[TBA-E5] Fig S4 (LoRA vs full fine-tuning).** Training and validation AUROC / AUPRC of the top LoRA run and the
> five full fine-tuning runs.

> **[TBA-E6] Table S1.** Mean and standard error over the five chromosome-exclusive folds of the top 10 LoRA
> configurations (`training/NT2_ranked_cv.py` output).

> **[TBA-E7] Fig S10 (ESM1b-FT training curves).** As Fig S6, for the ESM1b-FT run (best step 3,000). If the ESM1b-FT
> metrics file has the columns of the GLM-Missense `training_metrics.csv`, `plot_training_curves.py` can take its path.
