# Training

Fine-tuning experiments of the architecture exploration and the training of GLM-Missense (Methods 5.2–5.7).
Every command runs from the **repo root**. Inputs are the split files made by `preprocessing/` (Table 2); each run
writes to its own folder under `results/` (`--output_dir`), one `exp_<id>/` per configuration with
`training_metrics.csv`, `summary.json` and `best_model.pt`. The figures are drawn from these folders by
`evaluation/`.

The ensembles (MetaMissense and the ablation ensembles) are trained in `analysis/core.py`, not here.

---

## Experiments

| Manuscript | Experiment | Script | Training / validation data (`data/splits/`) |
|---|---|---|---|
| Fig 2A, Fig S2 | Frozen backbones: NT-2 (12k; 6k for the matched NT-1 comparison) × heads × embeddings | `NT2_phase1_and_unfreezeAll.py` | `ClinVar.251103.missense.hg38.seq{12k,6k}.BvsP_{training,validation}.tsv` |
| Fig 2A, Fig S2 | Frozen backbones: NT-1 (6k) | `NT1_phase1.py` | `ClinVar.251103.missense.hg38.seq6k.BvsP_{training,validation}.tsv` |
| Fig 2A, Fig S2 | Frozen backbones: Caduceus-PS (30k) | `caduceus_phase1.py` | `...seq30k.BvsP_*` (see **[TBA-P2]** in `preprocessing/`) |
| Fig 2B, Fig S3 | Local-window mean pooling (±8, ±16, ±32 tokens), frozen NT-2 + CNN | `NT2_window_pooling_sweep.py` | `ClinVar.251103.missense.hg38.seq12k.BvsP_{training,validation}.tsv` |
| Fig 3A | LoRA model selection: 54 configurations (2 embeddings × 3 heads × 3 ranks × 3 learning rates) | `NT2_lora_fullFT_sweep.py` (part 1) | 5,000 / 1,250 subsets (**[TBA-P1]**) |
| Fig S4 | LoRA vs full fine-tuning (5 learning rates) | `NT2_lora_fullFT_sweep.py` (part 2) | `ClinVar.251103.missense.hg38.seq12k.BvsP_{training,validation}.tsv` |
| Table S1 | 5-fold chromosome-exclusive CV of the top 10 LoRA configurations | `NT2_ranked_cv.py` (`--ranking_json` from the Fig 3A run) | combined 6,250-variant subset (**[TBA-P1]**) |
| Fig 3B, Fig S5 | Allele-aware inputs: alt, ref-alt, ref-alt-diff, alt-alt-alt | `NT2_ref_alt_input_sweep.py` | 5,000 / 1,250 subsets (**[TBA-P1]**) |
| Fig S6 | **GLM-Missense**: NT-2, LoRA r = 32, ref-alt-diff, MLP head | `NT2_ref_alt_contrast.py` (command in `runs.sh`); `evaluate_checkpoints.py` (optional offline checkpoint evaluation); see **[TBA-T3]** | `ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_{training,validation}.tsv` |
| Fig S10 | **ESM1b-FT**: ESM1b with the GLM-Missense head and settings | `analysis/scoring/fine-tune_ESM1b.py` (**[TBA-T2]**) | same variants as GLM-Missense, as protein sequences |

The released GLM-Missense weights are the selected checkpoint of the GLM-Missense run, copied to
`scoring/GLM-Missense.pt` (see `scoring/README.md`).

---

## GLM-Missense

```bash
bash training/runs.sh
```

Output: `results/NT2_seq12k_BLBvsPLP_ref_alt_contrast_mlp/exp_1_concat_diff/` (`training_metrics.csv`: training and
validation AUROC / AUPRC every 1,000 steps, read by `evaluation/plot_training_curves.py` for Fig S6; checkpoints).

---

## To be added

> **[TBA-T1] Run commands of the architecture exploration.** `runs.sh` holds only the GLM-Missense command. To add:
> one command per experiment of the table above (script, data files, GPUs, `--output_dir`), so that each figure's
> folder under `results/` is named here and read by its `evaluation/` script.

> **[TBA-T3] GLM-Missense training run.** The training script and command of the released GLM-Missense model, as in
> Methods 5.6: NT-2, LoRA r = 32 (α = 64, dropout 0.1, query and value projections), learning rate 5 × 10⁻⁵,
> ref-alt-diff input, MLP head; 10,000 steps, effective batch 32; every 1,000 steps evaluated on 5,000 randomly drawn
> training variants and a fixed subset of 5,000 validation variants; checkpoint with the best validation AUROC
> (step 8,000) selected. Replaces `NT2_ref_alt_contrast.py` / `runs.sh` if they differ.
> Input: `data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_{training,validation}.tsv`.
> Output: the run folder under `results/` (metrics read by `evaluation/plot_training_curves.py`, Fig S6) and the
> selected checkpoint, copied to `scoring/GLM-Missense.pt`.

> **[TBA-T2] ESM1b-FT training (Fig S10).** To add: (1) the protein inputs of the GLM-Missense training and validation
> variants (reference and alternate protein sequences of the ClinVar transcript; `analysis/scoring/esm1bft_prepare.py`
> and `analysis/scoring/map_variants_to_mane_protein.py` do this for the VUS), (2) the training command of
> `analysis/scoring/fine-tune_ESM1b.py` (LoRA r = 32, α = 64, dropout 0.1; learning rate 5 × 10⁻⁵; effective batch
> 32; 10,000 steps; evaluation every 1,000 steps on the GLM-Missense validation subset).
> Input: `data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_{training,validation}.tsv` (variant lists).
> Output: `results/models/ESM1b-FT/exp_1_concat_diff/best_model.pt` (read by `analysis/` to score ClinVar, SGE and
> VUS variants) and its training metrics (read by the Fig S10 script, **[TBA-E7]** in `evaluation/`).
