#!/usr/bin/env python3
"""
compare_strategies.py — Compare evaluation results across filter/stratify strategies.

Reads all_metrics.tsv files from a set of eval output directories and produces
a unified comparison using anchor methods only (all 9 predictors defined in
core/plots.py::ANCHOR_COLS). All plots use fixed per-method colors consistent
across every figure in the pipeline.

All plots that show both AUROC and AUPRC display them side by side.
Strata on x-axes are ordered canonically:
  AF:     not_in_gnomAD → AF=0 → AF<1e-6 → ... → AF>=1e-4  (rare → common)
  GERP:   GERP<0 → 0<=GERP<2 → 2<=GERP<4 → GERP>=4
  phyloP: phyloP<0 → 0<=phyloP<1 → ... → phyloP>=6

Usage:
    python evaluation/compare_strategies.py \\
        --strat_dir results/predictions/ClinVar.260309only.BLBvsPLP/eval_strat_af \\
        --outdir    results/predictions/ClinVar.260309only.BLBvsPLP/comparison_strat_af

    python evaluation/compare_strategies.py \\
        --dirs \\
            "all=results/predictions/ClinVar.260309only.BLBvsPLP/eval_all" \\
            "rare_1e-3=results/predictions/ClinVar.260309only.BLBvsPLP/eval_rare_1e-3" \\
        --outdir results/predictions/ClinVar.260309only.BLBvsPLP/comparison
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from core import col_color, display_name
from core.plots import METHOD_COLORS, ANCHOR_ORDER, _anchor_methods, _sort_strata


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--dirs",      nargs="+",
                       help="Named eval directories in 'label=path' format.")
    group.add_argument("--strat_dir", type=Path,
                       help="Root directory of a --mode stratify run. "
                            "Subdirectories are collected automatically.")
    p.add_argument("--outdir",      required=True, type=Path)
    p.add_argument("--metric_file", default="all_metrics.tsv")
    p.add_argument("--our_col",     default=None,
                   help="Our model column name. Inferred from 'finetune_' or "
                        "'GLM-Missense_score' if not set.")
    p.add_argument("--metrics",     default="auroc,prauc",
                   help="Comma-separated metrics to plot (default: auroc,prauc). "
                        "Pairs are plotted side by side.")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────

def collect_from_dirs(dir_specs: list[str], metric_file: str) -> pd.DataFrame:
    frames = []
    for spec in dir_specs:
        if "=" not in spec:
            raise ValueError(f"Expected 'label=path' format, got: '{spec}'")
        label, path_str = spec.split("=", 1)
        mpath = Path(path_str.strip()) / metric_file
        if not mpath.exists():
            print(f"  WARNING: {mpath} not found — skipping '{label}'")
            continue
        df = pd.read_csv(mpath, sep="\t")
        df["stratum"] = label.strip()
        frames.append(df)
        print(f"  Loaded '{label}': {len(df)} methods")
    if not frames:
        raise FileNotFoundError("No metric files found.")
    return pd.concat(frames, ignore_index=True)


def collect_from_strat_dir(strat_dir: Path, metric_file: str) -> pd.DataFrame:
    frames = []
    for subdir in sorted(strat_dir.iterdir()):
        if not subdir.is_dir():
            continue
        mpath = subdir / metric_file
        if not mpath.exists():
            continue
        df = pd.read_csv(mpath, sep="\t")
        df["stratum"] = subdir.name
        frames.append(df)
        print(f"  Loaded stratum '{subdir.name}': {len(df)} methods")
    if not frames:
        raise FileNotFoundError(f"No {metric_file} found under {strat_dir}")
    return pd.concat(frames, ignore_index=True)


def infer_our_col(comparison: pd.DataFrame) -> str:
    """
    Infer our fine-tuned model column. Checks:
      1. "GLM-Missense_score"               (canonical new naming)
      2. Columns starting with "finetune_"  (old naming)
    Pass --our_col explicitly if neither matches.
    """
    cols = comparison["column"].unique()
    for col in cols:
        if col == "GLM-Missense_score":
            return col
    for col in cols:
        if str(col).startswith("finetune_"):
            return col
    return ""


# ── Shared helpers ────────────────────────────────────────────────────────

def _savefig(fig, out_path) -> None:
    """Save figure as both PNG and PDF."""
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"  Saved: {out_path}  +  {pdf_path.name}")


def _metric_label(metric: str) -> str:
    """Return display label: prauc → AUPRC, auroc → AUROC, else uppercase."""
    return "AUPRC" if metric == "prauc" else metric.upper()


def _pos_neg_per_stratum(comparison: pd.DataFrame, strata: list[str]) -> dict[str, str]:
    """
    Build per-stratum 'pos=N  neg=M' strings from all_metrics.tsv data.
    Uses n_positive and n_variants columns (our model row has these).
    Returns dict: stratum → annotation string.
    """
    result = {}
    needed = {"n_variants", "n_positive"}
    if not needed.issubset(comparison.columns):
        return result
    for s in strata:
        sub = comparison[comparison["stratum"] == s].dropna(subset=["n_variants"])
        if sub.empty:
            continue
        row = sub.iloc[0]
        n_total = int(row["n_variants"])
        n_pos   = int(row["n_positive"]) if pd.notna(row.get("n_positive", np.nan)) else None
        if n_pos is not None:
            n_neg = n_total - n_pos
            result[s] = f"pos={n_pos:,}  neg={n_neg:,}"
        else:
            result[s] = f"n={n_total:,}"
    return result


def _prevalence_per_stratum(comparison: pd.DataFrame,
                             strata: list[str]) -> dict[str, float]:
    """Extract prevalence per stratum from all_metrics.tsv data."""
    if "prevalence" not in comparison.columns:
        return {}
    result = {}
    for s in strata:
        sub = comparison[comparison["stratum"] == s]["prevalence"].dropna()
        if not sub.empty:
            result[s] = float(sub.iloc[0])
    return result


# ── Plot functions ─────────────────────────────────────────────────────────

def plot_grouped_bar(comparison: pd.DataFrame, methods: list[str],
                     our_col: str, out_path: Path,
                     metric_pair: tuple[str, str] = ("auroc", "prauc")) -> None:
    """
    Grouped bar chart side by side for metric_pair.
    x = anchor methods (canonical order), groups = strata (canonical order).
    """
    strata  = _sort_strata(comparison["stratum"].unique().tolist())
    methods = [m for m in methods if m in comparison["column"].unique()]
    cmap    = plt.cm.get_cmap("Blues", len(strata) + 2)
    x       = np.arange(len(methods))
    width   = min(0.8 / len(strata), 0.2)
    dnames  = [display_name(m) for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(methods) * 1.2) * 2, 6))

    for ax, metric in zip(axes, metric_pair):
        for i, stratum in enumerate(strata):
            sub  = comparison[comparison["stratum"] == stratum].set_index("column")
            vals = []
            for m in methods:
                if m not in sub.index:
                    vals.append(np.nan)
                else:
                    v = sub.loc[m, metric]
                    vals.append(float(v.iloc[0]) if hasattr(v, "iloc") else float(v))
            offset = i * width - width * (len(strata) - 1) / 2
            ax.bar(x + offset, vals, width, label=stratum,
                   color=cmap(i + 2), alpha=0.88, edgecolor="white", lw=0.4)

        ax.set_xticks(x)
        xlabels = ax.set_xticklabels(dnames, rotation=40, ha="right", fontsize=8)
        for lbl, m in zip(ax.get_xticklabels(), methods):
            lbl.set_color(col_color(m, our_col))
        ax.set_ylabel(_metric_label(metric), fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_title(f"{_metric_label(metric)} by Method and Stratum", fontsize=11)
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", title="Stratum")

    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)


def plot_method_heatmap(comparison: pd.DataFrame, methods: list[str],
                        our_col: str, out_path: Path,
                        metric_pair: tuple[str, str] = ("auroc", "prauc")) -> None:
    """
    Heatmap side by side for metric_pair.
    Rows = anchor methods (canonical), columns = strata (canonical).
    Row labels colored by method color.
    """
    strata  = _sort_strata(comparison["stratum"].unique().tolist())
    methods = [m for m in methods if m in comparison["column"].unique()]
    dnames  = [display_name(m) for m in methods]

    fig, axes = plt.subplots(1, 2,
                             figsize=(max(6, len(strata) * 1.4) * 2,
                                      max(3, len(methods) * 0.55)))

    for ax, metric in zip(axes, metric_pair):
        pivot = (comparison[comparison["column"].isin(methods)]
                 .pivot_table(index="column", columns="stratum",
                              values=metric, aggfunc="first")
                 .reindex(index=methods, columns=strata))
        data = pivot.values.astype(float)
        im   = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
        ax.set_xticks(range(len(strata)))
        ax.set_xticklabels(strata, rotation=40, ha="right", fontsize=9)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(dnames, fontsize=8)
        for i, col in enumerate(methods):
            ax.get_yticklabels()[i].set_color(col_color(col, our_col))
        for i in range(len(methods)):
            for j in range(len(strata)):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=7, color="black")
        plt.colorbar(im, ax=ax, shrink=0.6, label=_metric_label(metric))
        ax.set_title(f"{_metric_label(metric)} — Anchor methods × Strata", fontsize=11)

    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)


def plot_rank_chart(comparison: pd.DataFrame, methods: list[str],
                    our_col: str, out_path: Path,
                    metric_pair: tuple[str, str] = ("auroc", "prauc")) -> None:
    """
    Rank chart side by side for metric_pair.
    x = strata in canonical order, y = rank within that stratum (1 = best).
    Our model line is thicker. Each method uses its fixed color.
    No legend on the chart — methods identified by color/style only.
    """
    strata  = _sort_strata(comparison["stratum"].unique().tolist())
    methods = [m for m in methods if m in comparison["column"].unique()]
    x       = np.arange(len(strata))

    fig, axes = plt.subplots(1, 2,
                             figsize=(max(8, len(strata) * 1.4) * 2, 6))

    for ax, metric in zip(axes, metric_pair):
        ranks_dict: dict[str, dict[str, float]] = {}
        for stratum in strata:
            sub = (comparison[comparison["stratum"] == stratum][["column", metric]]
                   .dropna(subset=[metric])
                   .sort_values(metric, ascending=False)
                   .reset_index(drop=True))
            sub["rank"] = sub.index + 1
            for _, row in sub.iterrows():
                ranks_dict.setdefault(row["column"], {})[stratum] = row["rank"]

        for method in methods:
            color  = col_color(method, our_col)
            lw     = 2.8 if method == our_col else 1.5
            zorder = 5 if method == our_col else 2
            ranks  = [ranks_dict.get(method, {}).get(s, np.nan) for s in strata]
            ax.plot(x, ranks, marker="o", markersize=5,
                    lw=lw, color=color, zorder=zorder)

        ax.set_xticks(x)
        ax.set_xticklabels(strata, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(f"Rank by {_metric_label(metric)} (1 = best)", fontsize=11)
        ax.invert_yaxis()
        ax.set_title(f"Method Rank Across Strata ({_metric_label(metric)})", fontsize=11)
        # No legend here — see stability legend below performance chart

    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)


def plot_performance_chart(comparison: pd.DataFrame, methods: list[str],
                           our_col: str, out_path: Path,
                           metric_pair: tuple[str, str] = ("auroc", "prauc"),
                           pos_neg_info: dict = None,
                           stability_df: pd.DataFrame = None) -> None:
    """
    4-panel figure: 2×2 grid.
      Row 0: performance line curves (AUROC | AUPRC) across strata
      Row 1: across-strata std deviation bars (AUROC std | AUPRC std)

    The std row serves as a visual legend — each bar is colored by method,
    so the curves above need no separate legend.

    Pos/neg counts are annotated below x-tick labels on the curve row.
    """
    strata      = _sort_strata(comparison["stratum"].unique().tolist())
    methods     = [m for m in methods if m in comparison["column"].unique()]
    x           = np.arange(len(strata))
    prevalences = _prevalence_per_stratum(comparison, strata)
    pn_info     = pos_neg_info or _pos_neg_per_stratum(comparison, strata)

    # Compute per-method std across strata if not provided
    if stability_df is None:
        stability_df = compute_strata_stability(comparison, methods,
                                                metric_pair=metric_pair)

    HIGHLIGHT = {"GLM-Missense_score", "MetaMissense_score"}

    fig, axes = plt.subplots(1, 2,
                             figsize=(max(8, len(strata) * 1.4) * 2 * 0.6,
                                      6 * 0.6))

    for col_idx, metric in enumerate(metric_pair):
        ax       = axes[col_idx]
        is_auprc = (metric == "prauc")

        for method in methods:
            color  = col_color(method, our_col)
            lw     = 2 if method in HIGHLIGHT else 1.5
            zorder = 5 if method in HIGHLIGHT else 2
            sub    = comparison[comparison["column"] == method].set_index("stratum")
            vals   = []
            for s in strata:
                if s not in sub.index:
                    vals.append(np.nan)
                else:
                    v = sub.loc[s, metric]
                    vals.append(float(v.iloc[0]) if hasattr(v, "iloc") else float(v))
            ax.plot(x, vals, marker="o", markersize=5,
                    lw=lw, color=color, zorder=zorder, alpha=0.7,
                    label=display_name(method))

        ax.set_xticks(x)
        # Two-line tick labels: stratum name (larger) + pos/neg counts (smaller)
        # Use annotate so each line can have its own fontsize.
        ax.set_xticklabels([])   # hide default labels
        for xi, s in enumerate(strata):
            pn = pn_info.get(s, "")
            # Line 1: stratum name, fontsize=9
            ax.annotate(s,
                        xy=(xi, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -6), textcoords="offset points",
                        ha="right", va="top", fontsize=9,
                        rotation=35, annotation_clip=False)
            if pn:
                # Line 2: pos/neg counts, fontsize=6.5, grey
                ax.annotate(pn,
                            xy=(xi, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, -20), textcoords="offset points",
                            ha="right", va="top", fontsize=6.5,
                            color="#666666", rotation=35,
                            annotation_clip=False)

        ax.set_ylabel(_metric_label(metric), fontsize=11)
        ax.set_title(f"{_metric_label(metric)} Across Strata", fontsize=11)

        if is_auprc and prevalences:
            prev_vals = [prevalences.get(s, np.nan) for s in strata]
            ax.plot(x, prev_vals, marker="", markersize=6, lw=1,
                    color="gray", ls="--", zorder=1, alpha=0.7,
                    label="Random")
            valid_prev = [v for v in prev_vals if not np.isnan(v)]
            if valid_prev:
                ax.set_ylim(max(0.0, min(valid_prev) - 0.05), 1.02)
            # Legend outside right of AUPRC panel
            ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1),
                      loc="upper left", framealpha=0.9)
        else:
            ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
            ax.set_ylim(0.4, 1.02)

    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)


# ── Stability analysis ────────────────────────────────────────────────────

def compute_strata_stability(comparison: pd.DataFrame, methods: list[str],
                              metric_pair: tuple[str, str] = ("auroc", "prauc")
                              ) -> pd.DataFrame:
    """
    For each method, compute across-strata range (max-min) and std
    for each metric in metric_pair.

    Returns a DataFrame with columns:
        column | {metric}_range | {metric}_std | {metric}_min | {metric}_max | {metric}_mean
    for each metric in metric_pair.
    """
    strata  = _sort_strata(comparison["stratum"].unique().tolist())
    methods = [m for m in methods if m in comparison["column"].unique()]
    rows    = []

    for method in methods:
        sub  = comparison[comparison["column"] == method].set_index("stratum")
        row  = {"column": method}
        for metric in set(metric_pair):
            vals = []
            for s in strata:
                if s not in sub.index:
                    continue
                v = sub.loc[s, metric]
                v = float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
                if not np.isnan(v):
                    vals.append(v)
            if len(vals) >= 2:
                row[f"{metric}_range"] = round(max(vals) - min(vals), 4)
                row[f"{metric}_std"]   = round(float(np.std(vals)), 4)
                row[f"{metric}_min"]   = round(min(vals), 4)
                row[f"{metric}_max"]   = round(max(vals), 4)
                row[f"{metric}_mean"]  = round(float(np.mean(vals)), 4)
            else:
                for suffix in ("range", "std", "min", "max", "mean"):
                    row[f"{metric}_{suffix}"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _draw_legend_panel(ax, methods: list[str], our_col: str,
                        stability_df: pd.DataFrame,
                        metric_pair: tuple[str, str]) -> None:
    """
    Draw a legend+range table below the stability chart.
    Each row: color swatch | display name | range (auroc) | range (auprc) | std (auroc) | std (auprc)
    Requirement #9: range/std live here so curve plots can omit their legends.
    """
    ax.axis("off")
    headers = ["Method", "AUROC range", "AUPRC range", "AUROC std", "AUPRC std"]
    col_data = []

    stab = stability_df.set_index("column") if not stability_df.empty else pd.DataFrame()

    for method in methods:
        row = []
        for metric in ["auroc", "prauc"]:
            for suffix in ["range", "std"]:
                key = f"{metric}_{suffix}"
                if not stab.empty and method in stab.index and key in stab.columns:
                    v = stab.loc[method, key]
                    row.append(f"{v:.3f}" if pd.notna(v) else "—")
                else:
                    row.append("—")
        col_data.append(row)

    n_methods = len(methods)
    row_h = 1.0 / max(n_methods + 2, 3)

    # Headers
    for j, hdr in enumerate(headers):
        x_pos = 0.05 + j * 0.19
        ax.text(x_pos, 1.0 - row_h * 0.5, hdr,
                fontsize=8, fontweight="bold", va="center", ha="left",
                transform=ax.transAxes)

    # Rows
    for i, method in enumerate(methods):
        y = 1.0 - row_h * (i + 1.5)
        color = col_color(method, our_col)
        # Color swatch
        patch = mpatches.FancyBboxPatch(
            (0.01, y - row_h * 0.35), 0.025, row_h * 0.7,
            boxstyle="round,pad=0.01", facecolor=color, edgecolor="none",
            transform=ax.transAxes)
        ax.add_patch(patch)
        # Display name
        dname = display_name(method)
        fw = "bold" if method == our_col else "normal"
        ax.text(0.05, y, dname, fontsize=8, va="center", ha="left",
                fontweight=fw, color=color, transform=ax.transAxes)
        # Stats
        for j, val_str in enumerate(col_data[i]):
            ax.text(0.05 + (j + 1) * 0.19, y, val_str,
                    fontsize=8, va="center", ha="left", transform=ax.transAxes)


def plot_strata_stability(stability_df: pd.DataFrame, our_col: str,
                          out_path: Path, comparison: pd.DataFrame = None,
                          metric_pair: tuple[str, str] = ("auroc", "prauc")) -> None:
    """
    Figure showing across-strata stability for each method.

    Layout (3 rows):
      Row 0 (tall): range (max − min) for AUROC and AUPRC side by side
      Row 1 (tall): std               for AUROC and AUPRC side by side
      Row 2 (short): legend+stats table (requirement #9)

    Changes vs original:
      - #5: No black box outline on GLM-Missense (or any method) —
            all bars use plain white edgecolor regardless.
      - #7: PRAUC → AUPRC in all labels.
      - #9: Legend+range table drawn in bottom panel so curve plots
            don't need their own legends.
    """
    metrics  = list(dict.fromkeys(metric_pair))   # deduplicated, order preserved
    measures = [("range", "Range (max − min)"), ("std", "Std dev")]
    methods  = stability_df["column"].tolist() if not stability_df.empty else []

    # 3 rows: 2 bar rows + 1 legend row
    fig, axes = plt.subplots(3, 2,
                             figsize=(16, max(10, len(stability_df) * 0.8)),
                             gridspec_kw={"height_ratios": [3, 3, 2]})

    for row_idx, (measure, measure_label) in enumerate(measures):
        for col_idx, metric in enumerate(metrics):
            ax       = axes[row_idx][col_idx]
            col_key  = f"{metric}_{measure}"
            if col_key not in stability_df.columns:
                ax.set_visible(False)
                continue

            plot_df  = stability_df.dropna(subset=[col_key]).copy()
            colors   = [col_color(c, our_col) for c in plot_df["column"]]
            dnames   = [display_name(c) for c in plot_df["column"]]

            # #5: No black box — all bars use white edgecolor (no special outline)
            bars = ax.barh(dnames, plot_df[col_key],
                           color=colors, edgecolor="white", linewidth=0.5)
            ax.invert_yaxis()
            for bar, val in zip(bars, plot_df[col_key]):
                if not np.isnan(val):
                    ax.text(bar.get_width() + 0.002,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:.3f}", va="center", fontsize=8)
            ax.set_xlabel(measure_label, fontsize=10)
            ax.set_title(f"{_metric_label(metric)} — {measure_label}", fontsize=11)
            ax.axvline(0, color="gray", lw=0.6)
            # Color y-tick labels by method color
            for tick, method in zip(ax.get_yticklabels(), plot_df["column"]):
                tick.set_color(col_color(method, our_col))

    # Row 2: legend + stability table (#9)
    # Merge both legend cells into one wide area using the left axis
    axes[2][1].set_visible(False)
    ax_legend = axes[2][0]
    # Make ax_legend span full width by adjusting its position
    pos0 = axes[2][0].get_position()
    pos1 = axes[2][1].get_position()
    axes[2][0].set_position([pos0.x0, pos0.y0,
                              pos1.x1 - pos0.x0, pos0.height])
    _draw_legend_panel(axes[2][0], methods, our_col, stability_df, metric_pair)

    fig.suptitle(
        "Across-Strata Stability — Anchor methods\n"
        "(shorter bar = more stable performance across strata)",
        fontsize=12)
    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    out_dir   = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Parse metric pairs: plot consecutive pairs side-by-side
    all_metrics = [m.strip() for m in args.metrics.split(",")]
    pairs = []
    for i in range(0, len(all_metrics), 2):
        m1 = all_metrics[i]
        m2 = all_metrics[i + 1] if i + 1 < len(all_metrics) else m1
        pairs.append((m1, m2))

    # ── Collect data ──────────────────────────────────────────────────
    if args.dirs:
        print("Loading named eval directories ...")
        comparison = collect_from_dirs(args.dirs, args.metric_file)
    else:
        print(f"Collecting from stratify directory: {args.strat_dir}")
        comparison = collect_from_strat_dir(args.strat_dir, args.metric_file)

    print(f"\nTotal rows loaded : {len(comparison):,}")
    print(f"Strata            : {comparison['stratum'].unique().tolist()}")
    print(f"Strata (ordered)  : {_sort_strata(comparison['stratum'].unique().tolist())}")

    # ── Infer our model column ────────────────────────────────────────
    our_col = args.our_col or infer_our_col(comparison)
    if our_col:
        print(f"Our model column  : {our_col}")
    else:
        print("  WARNING: could not infer our model column — pass --our_col")

    methods = _anchor_methods(our_col, available_cols=comparison["column"].unique().tolist())
    present = [m for m in methods if m in comparison["column"].unique()]
    missing = [m for m in methods if m not in comparison["column"].unique()]
    print(f"Anchor methods    : {present}")
    if missing:
        print(f"  (not in data)   : {missing}")

    # ── Save full comparison table ────────────────────────────────────
    comp_path = out_dir / "comparison_all_methods.tsv"
    comparison.to_csv(comp_path, sep="\t", index=False)
    print(f"\nSaved: {comp_path}")

    # ── Save anchor pivot tables ──────────────────────────────────────
    strata_ordered = _sort_strata(comparison["stratum"].unique().tolist())
    for m1, m2 in pairs:
        for metric in set([m1, m2]):
            piv = (comparison[comparison["column"].isin(present)]
                   .pivot_table(index="column", columns="stratum",
                                values=metric, aggfunc="first")
                   .reindex(index=present, columns=strata_ordered))
            piv.to_csv(out_dir / f"pivot_{metric}.tsv", sep="\t")
            print(f"  Saved pivot_{metric}.tsv")

    # Precompute pos/neg info for all strata (used in performance chart #8)
    pn_info = _pos_neg_per_stratum(comparison, strata_ordered)

    # ── Strata stability (computed once, embedded in performance chart) ──
    print("\n── Computing strata stability ────────────────────────────────")
    m1, m2 = pairs[0]
    stability_df = compute_strata_stability(comparison, present,
                                            metric_pair=(m1, m2))
    stab_path = out_dir / "strata_stability.tsv"
    stability_df.to_csv(stab_path, sep="\t", index=False)
    print(f"  Saved: {stab_path}")
    print(stability_df.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n── Generating comparison plots ──────────────────────────────")
    for m1, m2 in pairs:
        pair_tag = f"{m1}_{m2}" if m1 != m2 else m1
        plot_grouped_bar(comparison, present, our_col,
                         plots_dir / f"grouped_bar_{pair_tag}.png",
                         metric_pair=(m1, m2))
        plot_method_heatmap(comparison, present, our_col,
                            plots_dir / f"heatmap_{pair_tag}.png",
                            metric_pair=(m1, m2))
        plot_rank_chart(comparison, present, our_col,
                        plots_dir / f"rank_chart_{pair_tag}.png",
                        metric_pair=(m1, m2))
        # performance_chart now embeds the std stability row — no separate
        # strata_stability plot needed
        plot_performance_chart(comparison, present, our_col,
                               plots_dir / f"performance_chart_{pair_tag}.png",
                               metric_pair=(m1, m2),
                               pos_neg_info=pn_info,
                               stability_df=stability_df)

    print(f"\n✓  Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()