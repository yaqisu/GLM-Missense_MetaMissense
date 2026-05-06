"""
Plot XGB permutation feature importance (Mean AUPRC Decrease).
Reads  results/ensemble/feature_importance.csv
Writes results/figures/fig_feature_importance.pdf

Run from project root:
    python evaluation/plot_feature_importance.py
"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ── Project root is one level up from evaluation/ ────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))

# Importing from core.plots triggers Arial font registration automatically
from core.plots import col_color, display_name

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = ROOT / "results" / "ensemble" / "feature_importance.csv"
OUT_PATH  = ROOT / "results" / "figures" / "fig_feature_importance"

# ── Config — match score_correlation figure exactly ───────────────────────────
OUR_COL        = "GLM-missense_score"
TICK_FONTSIZE  = 14
LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 14
BAR_HEIGHT     = 0.55
FIGSIZE        = (4, 4)


def plot_feature_importance(
    data_path: Path = DATA_PATH,
    out_path:  Path = OUT_PATH,
    our_col:   str  = OUR_COL,
    title:     str  = "XGBoost Feature Importance (5-fold CV)",
    subtitle:  str  = "AUROC=0.983±0.001   |   AUPRC=0.950±0.003",
) -> None:

    df = pd.read_csv(data_path)
    # Sort ascending so largest bar appears at top in barh
    df = df.sort_values("auprc_decrease", ascending=True).reset_index(drop=True)

    n = len(df)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    x_max = df["auprc_decrease"].max()
    xoff  = x_max * 0.04   # padding beyond error bar tip, matches partial_correlation

    for i, row in df.iterrows():
        raw_name = row["feature"]
        color    = col_color(raw_name, our_col)
        val      = row["auprc_decrease"]
        ste      = row["ste"]
        ax.barh(
            i,
            val,
            xerr=ste,
            height=BAR_HEIGHT,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            error_kw=dict(ecolor="black", capsize=3, lw=1.2),
        )
        # Label to the right of error bar tip
        ax.text(val + ste + xoff, i,
                f"{val:.3f}",
                va="center", ha="left",
                fontsize=TICK_FONTSIZE - 4,
                color="#222222")

    # Extend x-axis to make room for labels
    ax.set_xlim(0, df["auprc_decrease"].max() + df["ste"].max() + x_max * 0.25)

    # ── Y-axis tick labels ────────────────────────────────────────────────────
    dnames = [display_name(f) for f in df["feature"]]
    dnames = ["GLM-Missense" if d == "GLM-missense_score" else d for d in dnames]
    ax.set_yticks(range(n))
    ax.set_yticklabels(dnames, fontsize=TICK_FONTSIZE)
    for tick, feature in zip(ax.get_yticklabels(), df["feature"]):
        tick.set_color(col_color(feature, our_col))

    ax.set_xlabel("Mean AUPRC Decrease", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)

    # ── Title ─────────────────────────────────────────────────────────────────
    full_title = f"{title}\n{subtitle}" if subtitle else title
    fig.suptitle(full_title, fontsize=TITLE_FONTSIZE, y=1.01, x=0.55)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in [".pdf", ".png"]:
        fig.savefig(out_path.with_suffix(ext), bbox_inches="tight", dpi=300)
        print(f"Saved → {out_path.with_suffix(ext)}")
    plt.close(fig)


if __name__ == "__main__":
    plot_feature_importance()