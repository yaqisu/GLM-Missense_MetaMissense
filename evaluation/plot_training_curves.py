"""
Plot train and validation AUROC curves up to the early stopping point.

Input:  results/NT2_seq12k_BLBvsPLP_ref_alt_contrast_mlp/exp_1_concat_diff/training_metrics.csv
Output: results/figures/fig_training_curve.pdf  (and .png)

Run from project root:
    python evaluation/plot_training_curves.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

import matplotlib
matplotlib.use("Agg")

# ── Font and PDF/SVG text settings ─────────────────────────────────────────
import matplotlib.font_manager as _fm
import os as _os

# Register Arial from a user-writable directory — no conda env write access needed
_user_font_dir = "../fonts/"
if _os.path.isdir(_user_font_dir):
    for _f in _os.listdir(_user_font_dir):
        if _f.lower().endswith(".ttf") or _f.lower().endswith(".otf"):
            _fm.fontManager.addfont(_os.path.join(_user_font_dir, _f))

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype":      42,
    "svg.fonttype":      "none",
    "ps.fonttype":       42,
})

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV = "results/NT2_seq12k_BLBvsPLP_ref_alt_contrast_mlp/exp_1_concat_diff/training_metrics.csv"
OUTPUT_DIR = "results/figures"
PATIENCE   = 2          # early stopping patience (in eval steps)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

# ── Apply early stopping logic ────────────────────────────────────────────────
best_val_auroc = -float("inf")
best_step_idx  = 0
no_improve     = 0
stop_idx       = len(df) - 1   # default: use all rows if early stopping never fires

for i, row in df.iterrows():
    if row["val_auroc"] > best_val_auroc:
        best_val_auroc = row["val_auroc"]
        best_step_idx  = i
        no_improve     = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            stop_idx = i
            break

df_plot = df.iloc[: stop_idx + 1].copy()

print(f"Early stopping at step {df_plot['steps'].iloc[-1]} "
      f"(row index {stop_idx}, patience={PATIENCE})")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

metrics = [
    ("train_auroc_subset", "val_auroc", "AUROC", 0.86, 0.97),
    ("train_auprc_subset", "val_auprc", "PRAUC", 0.76, 0.94),
]

for ax, (train_col, val_col, ylabel, ymin, ymax) in zip(axes, metrics):
    ax.plot(df_plot["steps"], df_plot[train_col],
            label=f"Train {ylabel}", color="#2166ac", linewidth=1.8,
            marker="o", markersize=3.5)
    ax.plot(df_plot["steps"], df_plot[val_col],
            label=f"Val {ylabel}",   color="#d6604d", linewidth=1.8,
            marker="o", markersize=3.5)

    best_val  = df_plot[val_col].max()
    best_step = df_plot.loc[df_plot[val_col].idxmax(), "steps"]

    ax.axvline(best_step, color="#d6604d", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.scatter([best_step], [best_val], color="#d6604d", s=60, zorder=5)
    ax.text(best_step, best_val + (ymax - ymin) * 0.025,
            f"best val {ylabel}\n{best_val:.4f}",
            fontsize=7.5, color="#d6604d", ha="center", va="bottom")

    ax.set_xlabel("Training step", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{ylabel} — NT2 ref/alt contrast (MLP)", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
for ext in ("pdf", "png"):
    out_path = os.path.join(OUTPUT_DIR, f"fig_training_curve.{ext}")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.show()