"""
Two standalone horizontal-bar figures, saved as Illustrator-editable PDFs.

  Figure 1  (fig_repr):    Input representation  — summary CSV (one row per exp,
                           keyed by `combine_mode`).  Panels: AUROC | AUPRC.
  Figure 2  (fig_window):  Pooling window size    — training-log CSV (best
                           checkpoint per exp_id).  Panel: AUROC only.

Usage
-----
    python plot_repr_window_bar.py REPR_CSV WINDOW_CSV [--outdir DIR] [--png]

`REPR_CSV` columns:   exp_id, combine_mode, ..., best_val_auroc, best_val_auprc
`WINDOW_CSV` columns: steps, ..., val_auc, val_auprc, ..., exp_id,
                      embedding_strategy, window_half_size, ...

Text stays editable in Adobe Illustrator because fonts are embedded as TrueType
(pdf.fonttype = 42) rather than outlined.
"""

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["font.family"]  = "Arial"

# -- Display config ------------------------------------------------------------
# combine_mode value (in REPR_CSV)  ->  human-readable bar label.
# Unknown modes fall back to the raw value. Order = bottom -> top of the panel.
REPR_LABEL_MAP = {
    "alt_only":    "alt",
    "concat_only": "ref + alt",
    "concat_diff": "ref \u2212 alt",
}
REPR_ORDER = ["alt_only", "concat_only", "concat_diff"]   # bottom -> top

# Window conditions provided manually (NOT present in the log CSV). Merged with
# the CSV-derived bars and skipped if that window size already appears in the CSV.
#   (window_half_size, embedding_strategy_hint, best_auroc)
WINDOW_EXTRA = [
    (1,    "var_pos",   0.871),   # var-pos (w=1)
    (2000, "mean_pool", 0.778),   # mean-pool (w=2000)
]

REPR_TITLE   = "Input representation\n(NT-2 LoRA, MLP head, 5k subset)"
WINDOW_TITLE = "Pooling window size\n(NT-2 frozen, CNN head)"

# Light -> dark blue gradient endpoints (bars colored light=worst .. dark=best)
GRAD_LIGHT = np.array([0x95, 0xC0, 0xE1]) / 255.0
GRAD_DARK  = np.array([0x06, 0x2E, 0x58]) / 255.0

BAR_HEIGHT = 0.55
TICK_FS    = 9.0
LABEL_FS   = 10.0
TITLE_FS   = 10.5
PANEL_FS   = 12.0


# -- Helpers -------------------------------------------------------------------
def blue_gradient(n):
    """n hex colors interpolated from light to dark blue."""
    if n <= 0:
        return []
    if n == 1:
        return [matplotlib.colors.to_hex(GRAD_DARK)]
    return [matplotlib.colors.to_hex(GRAD_LIGHT + (GRAD_DARK - GRAD_LIGHT) * (i / (n - 1)))
            for i in range(n)]


def auto_axis(values):
    """Return (x_min, x_max, x_step) framing `values` with a little padding."""
    values = [v for v in values if not (v is None or math.isnan(v))]
    lo, hi = float(min(values)), float(max(values))
    span = max(hi - lo, 0.02)
    x_min = math.floor((lo - 0.25 * span) * 100) / 100
    x_max = math.ceil((hi + 0.10 * span) * 100) / 100
    raw_step = (x_max - x_min) / 4.0
    for s in (0.01, 0.02, 0.05, 0.1, 0.2):
        if raw_step <= s:
            return x_min, x_max, s
    return x_min, x_max, 0.5


def collapse_experiments(df):
    """
    Collapse to one row per exp_id with `best_auroc` / `best_auprc`.

    - Training-log files (have a `val_auc` column): pick the best checkpoint per
      exp_id by max val_auc and report that checkpoint's val_auc / val_auprc.
    - Summary files: use the per-row best_val_auroc / best_val_auprc directly.
    """
    if "val_auc" in df.columns and "val_auprc" in df.columns:
        idx = df.groupby("exp_id")["val_auc"].idxmax()
        best = df.loc[idx].copy()
        best["best_auroc"] = best["val_auc"].astype(float)
        best["best_auprc"] = best["val_auprc"].astype(float)
    else:
        best = df.groupby("exp_id", as_index=False).first()
        best["best_auroc"] = best["best_val_auroc"].astype(float)
        best["best_auprc"] = best["best_val_auprc"].astype(float)
    return best


def load_repr_data(csv_path):
    """-> list of (label, auroc, auprc), ordered bottom -> top."""
    df = collapse_experiments(pd.read_csv(csv_path))
    rows = []
    for _, r in df.iterrows():
        mode = str(r["combine_mode"])
        rows.append((REPR_LABEL_MAP.get(mode, mode),
                     float(r["best_auroc"]), float(r["best_auprc"]), mode))
    rank = {m: i for i, m in enumerate(REPR_ORDER)}
    rows.sort(key=lambda x: rank.get(x[3], len(rank)))
    return [(lab, a, p) for (lab, a, p, _) in rows]


def make_window_label(window_half_size, embedding_strategy):
    es = str(embedding_strategy).lower()
    if "var" in es:
        return "var-pos (w=1)"
    if pd.isna(window_half_size):
        return "mean-pool" if "mean" in es else str(embedding_strategy)
    w = int(round(float(window_half_size)))
    if w <= 1:
        return "var-pos (w=1)"
    if w >= 1000:
        return f"mean-pool\n(w={w})"
    return f"w = {w}"


def window_key(window_half_size, embedding_strategy):
    """Numeric sort/dedup key: smaller window -> top of the chart."""
    if "var" in str(embedding_strategy).lower():
        return 1.0
    if pd.isna(window_half_size):
        return 1e9
    return float(window_half_size)


def load_window_data(csv_path):
    """-> list of (label, auroc, auprc), ordered largest window (bottom) -> smallest (top)."""
    df = collapse_experiments(pd.read_csv(csv_path))
    rows, seen = [], set()
    for _, r in df.iterrows():
        w  = r.get("window_half_size", np.nan)
        es = r.get("embedding_strategy", "")
        key = window_key(w, es)
        seen.add(key)
        rows.append((make_window_label(w, es),
                     float(r["best_auroc"]), float(r["best_auprc"]), key))
    for w, es, auroc in WINDOW_EXTRA:
        key = window_key(w, es)
        if key in seen:
            continue
        seen.add(key)
        rows.append((make_window_label(w, es), float(auroc), float("nan"), key))
    rows.sort(key=lambda x: x[3], reverse=True)   # largest window -> bottom
    return [(lab, a, p) for (lab, a, p, _) in rows]


# -- Drawing -------------------------------------------------------------------
def draw_bar_panel(ax, data, colors, metric_idx, xlabel,
                   x_min, x_max, x_step, panel_label, show_ylabels):
    """metric_idx: 1=AUROC, 2=AUPRC. show_ylabels: True for the left panel only."""
    n = len(data)
    for i, (label, auroc, auprc) in enumerate(data):
        val = auroc if metric_idx == 1 else auprc
        if val is None or math.isnan(val):
            continue
        ax.barh(i, val - x_min, left=x_min, height=BAR_HEIGHT,
                color=colors[i], edgecolor="white", linewidth=0.4, zorder=2)
        xoff = (x_max - x_min) * 0.010
        ax.text(val + xoff, i, f"{val:.3f}",
                va="center", ha="left",
                fontsize=TICK_FS - 1.5, color="#222222", zorder=3)

    ax.set_yticks(range(n))
    if show_ylabels:
        ax.set_yticklabels([d[0] for d in data], fontsize=TICK_FS)
        for tick, color in zip(ax.get_yticklabels(), colors):
            tick.set_color(color)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

    ax.set_ylim(-0.6, n - 0.4)
    padding = (x_max - x_min) * 0.20
    ax.set_xlim(x_min, x_max + padding)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not show_ylabels:
        ax.spines["left"].set_visible(False)

    ax.text(-0.02 if show_ylabels else -0.01, 1.10, panel_label,
            transform=ax.transAxes, fontsize=PANEL_FS,
            fontweight="bold", va="top", ha="right")


def make_figure(data, title, out_path, metrics, save_png=False):
    """
    One standalone figure.
    metrics: list of (metric_idx, xlabel, panel_label); one column per entry.
    """
    n = len(data)
    colors = blue_gradient(n)
    ncols = len(metrics)
    fig_h = max(1.8, 0.45 * n + 1.0)
    fig_w = {1: 3.0, 2: 4.6}.get(ncols, 2.3 * ncols)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(wspace=0.08, top=0.78, bottom=0.20, left=0.24, right=0.97)

    for j, (metric_idx, xlabel, panel_label) in enumerate(metrics):
        vals = [(d[1] if metric_idx == 1 else d[2]) for d in data]
        x_min, x_max, x_step = auto_axis(vals)
        draw_bar_panel(axes[0][j], data, colors, metric_idx, xlabel,
                       x_min, x_max, x_step, panel_label, show_ylabels=(j == 0))

    axes[0][0].set_title(title, fontsize=TITLE_FS, pad=6, loc="left", x=0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ([".pdf", ".png"] if save_png else [".pdf"]):
        fig.savefig(out_path.with_suffix(ext), bbox_inches="tight", dpi=300)
        print(f"Saved -> {out_path.with_suffix(ext)}")
    plt.close(fig)


# -- Entrypoint ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repr_csv", help="summary CSV keyed by combine_mode")
    ap.add_argument("window_csv", help="training-log CSV with window_half_size / val_auc")
    ap.add_argument("--outdir", default="results/figures",
                    help="output directory (default: results/figures)")
    ap.add_argument("--png", action="store_true", help="also write a PNG alongside each PDF")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    make_figure(load_repr_data(args.repr_csv), REPR_TITLE, outdir / "fig_repr",
                metrics=[(1, "Validation AUROC", "A"), (2, "Validation AUPRC", "B")],
                save_png=args.png)
    make_figure(load_window_data(args.window_csv), WINDOW_TITLE, outdir / "fig_window",
                metrics=[(1, "Validation AUROC", "A")],
                save_png=args.png)


if __name__ == "__main__":
    main()
