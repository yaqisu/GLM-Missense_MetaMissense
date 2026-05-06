"""
figures/plot_CV_res.py

Cross-validation AUROC and AUPRC comparison across methods.
Horizontal bar chart — style matches plot_feature_importance.py and
evaluate_partial_correlation.py exactly:
  TICK_FONTSIZE=14, LABEL_FONTSIZE=12, TITLE_FONTSIZE=14, BAR_HEIGHT=0.55
  Colors from core.plots.col_color / METHOD_COLORS
  Arial font via core.plots font registration
  pdf.fonttype=42 for editable text in Adobe Illustrator

Reads:  results/ensemble/cv_results.tsv
Writes: results/figures/fig_cv_results.pdf
        results/figures/fig_cv_results.png

Significance bracket: MetaMissense vs REVEL (Welch t-test from summary stats).

Run from project root:
    python figures/plot_CV_res.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ── Project root is one level up from figures/ ────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))

# Importing from core.plots triggers Arial font registration automatically
from core.plots import col_color, METHOD_COLORS

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH = ROOT / "results" / "ensemble" / "cv_results.tsv"
OUT_PATH  = ROOT / "results" / "figures" / "fig_cv_results"

# ── Style — identical to plot_feature_importance.py ────────────────────────
OUR_COL        = "GLM-Missense_score"
TICK_FONTSIZE  = 14
LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 14
BAR_HEIGHT     = 0.55
FIGSIZE        = (10, 4)   # wider to fit 2 panels side by side

N_FOLDS = 5   # number of CV folds used to compute std


# ── Stats helpers ──────────────────────────────────────────────────────────

def welch_ttest(m1, s1, m2, s2, n):
    """Welch t-test from summary statistics (mean, std, n)."""
    se  = np.sqrt(s1**2 / n + s2**2 / n)
    t   = (m1 - m2) / se
    df  = (s1**2 / n + s2**2 / n)**2 / \
          ((s1**2 / n)**2 / (n - 1) + (s2**2 / n)**2 / (n - 1))
    p   = 2 * (1 - stats.t.cdf(abs(t), df))
    return t, p


def sig_label(p):
    """Return 'p = x.xxe-xx' annotation string (no stars — reader can judge)."""
    return f"p = {p:.2e}"


# ── Panel plotting ─────────────────────────────────────────────────────────

def plot_panel(ax, df_sorted, val_col, err_col,
               xlabel, title, p_val,
               ref_method_a, ref_method_b):
    """
    Horizontal bar panel sorted descending by val_col (largest at top).

    Significance bracket between ref_method_a and ref_method_b.
    Label placed to the right of the error bar tip (positive values only —
    all CV metrics are positive).
    """
    methods = df_sorted["method"].tolist()
    means   = df_sorted[val_col].values
    stds    = df_sorted[err_col].values

    x_max    = means.max()
    x_min_v  = max(0.0, means.min() - 0.1)
    vis_range = 1.0 - x_min_v          # visible x range
    xoff      = vis_range * 0.02       # small padding beyond error bar tip

    for i, (method, mean, std) in enumerate(zip(methods, means, stds)):
        # Build the canonical column name used in METHOD_COLORS key matching
        col_key = f"{method.lower()}_score"   # e.g. "glm-missense_score"
        color = col_color(col_key, OUR_COL)

        ax.barh(i, mean,
                xerr=std,
                height=BAR_HEIGHT,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                error_kw=dict(ecolor="black", capsize=3, lw=1.2),
                zorder=3)

        # Value label always outside the error bar tip
        ax.text(mean + std + xoff, i,
                f"{mean:.3f}",
                va="center", ha="left",
                fontsize=TICK_FONTSIZE - 4,
                color="#222222", clip_on=False)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=TICK_FONTSIZE)
    # Color each y-tick label by method color
    for tick, method in zip(ax.get_yticklabels(), methods):
        col_key = f"{method.lower()}_score"
        tick.set_color(col_color(col_key, OUR_COL))

    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title,   fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # x-axis: min = max(0, min_value - 0.1), max = 1.0 (metric ceiling)
    x_min = max(0.0, means.min() - 0.05)
    ax.set_xlim(x_min, 1.0)

    # ── Significance bracket ───────────────────────────────────────────
    if ref_method_a in methods and ref_method_b in methods:
        pos_a   = methods.index(ref_method_a)
        pos_b   = methods.index(ref_method_b)
        tip_a   = means[pos_a] + stds[pos_a]
        tip_b   = means[pos_b] + stds[pos_b]
        # bracket drawn at x = max tip + small gap
        bracket_x  = max(tip_a, tip_b) + xoff * 2.5
        tick_size   = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
        y_mid       = (pos_a + pos_b) / 2

        ax.plot([bracket_x, bracket_x + tick_size, bracket_x + tick_size, bracket_x],
                [pos_a, pos_a, pos_b, pos_b],
                color="black", linewidth=1.2, clip_on=False)
        ax.text(bracket_x + tick_size * 2, y_mid,
                sig_label(p_val),
                va="center", ha="left",
                fontsize=TICK_FONTSIZE - 4,
                color="#222222")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, sep="\t")
    print(f"Loaded {len(df)} methods from {DATA_PATH}")

    # Compute significance: MetaMissense vs REVEL
    row_meta  = df[df["method"] == "MetaMissense"].iloc[0]
    row_revel = df[df["method"] == "REVEL"].iloc[0]

    _, p_auroc = welch_ttest(row_meta["auroc_mean"], row_meta["auroc_std"],
                              row_revel["auroc_mean"], row_revel["auroc_std"],
                              N_FOLDS)
    _, p_auprc = welch_ttest(row_meta["auprc_mean"], row_meta["auprc_std"],
                              row_revel["auprc_mean"], row_revel["auprc_std"],
                              N_FOLDS)
    print(f"  AUROC MetaMissense vs REVEL: p = {p_auroc:.2e}")
    print(f"  AUPRC MetaMissense vs REVEL: p = {p_auprc:.2e}")

    fig, (ax_auroc, ax_auprc) = plt.subplots(1, 2, figsize=FIGSIZE)

    for ax, val_col, err_col, xlabel, title, p_val in [
        (ax_auroc, "auroc_mean", "auroc_std", "CV AUROC", "CV AUROC", p_auroc),
        (ax_auprc, "auprc_mean", "auprc_std", "CV AUPRC", "CV AUPRC", p_auprc),
    ]:
        # Sort descending — largest at top for horizontal bar
        df_sorted = df.sort_values(val_col, ascending=True).reset_index(drop=True)
        plot_panel(ax, df_sorted, val_col, err_col,
                   xlabel=xlabel, title=title, p_val=p_val,
                   ref_method_a="MetaMissense", ref_method_b="REVEL")

    fig.tight_layout()

    for ext in (".pdf", ".png"):
        out = OUT_PATH.with_suffix(ext)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()