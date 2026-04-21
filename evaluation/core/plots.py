"""
Shared plotting utilities for variant pathogenicity evaluation.

Color scheme is fixed per method and applied consistently across all figures.
Methods are ordered by approximate publication year; colors progress from
cool (older) to warm (newer), with our model in crimson.

All figures that show both AUROC and AUPRC display them side-by-side.
Every figure includes a subtitle with variant count and class distribution.

Anchor methods plotted in all figures (edit ANCHOR_COLS / ANCHOR_DISPLAY /
METHOD_COLORS below to add/remove methods or rename/recolor them):
    MetaMissense, GLM-Missense, NT2-Zeroshot,
    AlphaMissense, ESM1b, REVEL, CADD, Polyphen2, SIFT

─────────────────────────────────────────────────────────────────────────────
CENTRAL CONFIGURATION — edit here to rename or recolor any method
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (roc_auc_score, roc_curve,
                              precision_recall_curve, average_precision_score)

from .metrics import try_numeric, flip_if_inverse


# ═══════════════════════════════════════════════════════════════════════════
# CENTRAL METHOD REGISTRY
# Edit ANCHOR_COLS, ANCHOR_DISPLAY, and METHOD_COLORS to rename/reorder/recolor.
# ═══════════════════════════════════════════════════════════════════════════

# Actual column names in the merged TSV — defines canonical plot order.
ANCHOR_COLS = [
    "MetaMissense_score",          # external ensemble
    "GLM-Missense_score",          # our fine-tuned GLM (finetune)
    "AlphaMissense_score",
    "ESM1b_score",
    "REVEL_score",
    "CADD_phred",
    "Polyphen2_HVAR_score",
    "SIFT_score",
]

# Alias so existing code that references ANCHOR_ORDER still works
ANCHOR_ORDER = ANCHOR_COLS

# Methods drawn with thick lines in ROC/PR curves (our fine-tuned + MetaMissense)
HIGHLIGHT_COLS = {
    "GLM-Missense_score",
    "MetaMissense_score",
}

# Human-readable display names shown on plot axes and legends.
ANCHOR_DISPLAY = {
    "MetaMissense_score":         "MetaMissense",
    "GLM-Missense_score":         "GLM-Missense",
    "zeroshot_NT2_seq12k_score":  "NT2-Zeroshot",
    "AlphaMissense_score":        "AlphaMissense",
    "ESM1b_score":                "ESM1b",
    "REVEL_score":                "REVEL",
    "CADD_phred":                 "CADD",
    "Polyphen2_HVAR_score":       "Polyphen2",
    "SIFT_score":                 "SIFT",
}

# Fixed color palette — key is a lowercase substring of the column name.
METHOD_COLORS = {
    "sift":           "#4B0082",   # 2001 — dark indigo
    "polyphen":       "#9467BD",   # 2010 — medium purple
    "cadd":           "#2166AC",   # 2014 — steel blue
    "revel":          "#1A9993",   # 2016 — teal
    "esm1b":          "#D4A017",   # 2022 — golden yellow
    "alphamissense":  "#E6821E",   # 2023 — orange
    "glm-missense":   "#C2185B",   # 2024 — crimson  (our fine-tuned model)
    "metamissense":   "#8B0000",   # 2024 — dark red  (ensemble)
    "nt2_seq12k":     "#A0522D",   # NT2 zeroshot — sienna brown
    "our_model":      "#C2185B",   # fallback "our model" color
    "other":          "#AAAAAA",   # fallback — light grey
    # Legacy zero-shot keys (kept for backward compat)
    "caduceusps":     "#4B0082",
    "caduceuph":      "#9467BD",
    "nt1":            "#1A9993",
    "nt2_seq6k":      "#D4A017",
}

# Strata ordering for rank charts — from least to most common/conserved
STRATA_ORDER = {
    "af": [
        "not_in_gnomAD", "AF=0", "AF<1e-6",
        "1e-6<=AF<1e-5", "AF>=1e-5",
    ],
    "gerp": [
        "GERP<0", "0<=GERP<2", "2<=GERP<4", "GERP>=4",
    ],
    "phylop": [
        "phyloP<0", "0<=phyloP<1", "1<=phyloP<3",
        "3<=phyloP<6", "phyloP>=6",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Display name helper
# ═══════════════════════════════════════════════════════════════════════════

def display_name(col: str) -> str:
    """Return the human-readable display name for a column."""
    return ANCHOR_DISPLAY.get(col, col)


# ═══════════════════════════════════════════════════════════════════════════
# Color helpers
# ═══════════════════════════════════════════════════════════════════════════

def col_color(col: str, our_col: str = "") -> str:
    """Return the fixed color for a method column."""
    if our_col and col == our_col:
        return METHOD_COLORS["our_model"]
    c = col.lower()
    for key, color in METHOD_COLORS.items():
        if key in ("our_model", "other"):
            continue
        if key in c:
            return color
    return METHOD_COLORS["other"]


def zeroshot_col_color(col: str, our_col: str = "") -> str:
    """Return fixed color for a zero-shot model column."""
    if our_col and col == our_col:
        return METHOD_COLORS["our_model"]
    c = col.lower()
    if "caduceusps" in c or ("caduceus" in c and "ps" in c):
        return METHOD_COLORS["caduceusps"]
    if "caduceuph" in c or ("caduceus" in c and "ph" in c):
        return METHOD_COLORS["caduceuph"]
    if "nt2" in c and "seq12k" in c:
        return METHOD_COLORS["nt2_seq12k"]
    if "nt2" in c and "seq6k" in c:
        return METHOD_COLORS["nt2_seq6k"]
    if "nt1" in c:
        return METHOD_COLORS["nt1"]
    return METHOD_COLORS["other"]


# ═══════════════════════════════════════════════════════════════════════════
# Anchor-method helpers
# ═══════════════════════════════════════════════════════════════════════════

def _anchor_methods(our_col: str, available_cols: list = None) -> list[str]:
    """
    Return ANCHOR_COLS in canonical order.
    If available_cols is provided, silently skip any column not present
    (per requirement #10 — graceful degradation when a predictor is absent).
    our_col is moved to front only if it IS one of the ANCHOR_COLS;
    otherwise it is prepended regardless (for backward compat).
    """
    if available_cols is not None:
        order = [m for m in ANCHOR_COLS if m in available_cols]
    else:
        order = ANCHOR_COLS[:]

    # If our_col is not already in the list (e.g. older finetune_ naming),
    # prepend it so it always appears first.
    if our_col and our_col not in order:
        order = [our_col] + order

    return order


def effective_anchor_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the subset of ANCHOR_COLS actually present in df with non-all-NaN values.
    Silently skips missing predictors so the intersection is always maximized.
    Used by evaluate.py to build the shared evaluation subset.
    """
    return [col for col in ANCHOR_COLS
            if col in df.columns and try_numeric(df[col]).notna().any()]


def _sort_strata(strata: list[str]) -> list[str]:
    """Sort strata labels into canonical order; unknown labels appended."""
    for order in STRATA_ORDER.values():
        known   = [s for s in order if s in strata]
        unknown = [s for s in strata if s not in order]
        if known:
            return known + unknown
    return strata


def _class_subtitle(labels: pd.Series) -> str:
    total  = len(labels)
    n_pos  = int((labels == 1).sum())
    n_neg  = int((labels == 0).sum())
    return (f"n={total:,}  |  "
            f"pos: {n_pos/total:.0%}   neg: {n_neg/total:.0%}")


def _strata_counts_text(labels: pd.Series) -> str:
    """Return 'pos=N  neg=M' string for annotating strata figures."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    return f"pos={n_pos:,}  neg={n_neg:,}"


def _add_class_info(ax, base_title: str, subtitle: str, class_info: str) -> None:
    lines = [base_title]
    if subtitle:
        lines.append(subtitle)
    if class_info:
        lines.append(class_info)
    ax.set_title("\n".join(lines), fontsize=10)


def _savefig(fig, out_path) -> None:
    """Save figure as both PNG and PDF."""
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"  Saved: {out_path}  +  {pdf_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Inset bar helper (embedded inside ROC/PR panels)
# ═══════════════════════════════════════════════════════════════════════════

def _draw_bar_inset(fig, ax, metrics_df: pd.DataFrame, our_col: str,
                    metric: str, title_str: str,
                    inset_bounds: list = None) -> None:
    """Draw an AUROC or AUPRC horizontal barplot as an inset axes inside ax."""
    if inset_bounds is None:
        inset_bounds = [0.55, 0.02, 0.44, 0.44]

    avail   = metrics_df["column"].tolist()
    methods = _anchor_methods(our_col, available_cols=avail)
    plot_df = (metrics_df[metrics_df["column"].isin(methods)]
               .drop_duplicates(subset="column")
               .set_index("column").reindex(methods).reset_index())
    colors = [col_color(c, our_col) for c in plot_df["column"]]
    labels = [display_name(c) for c in plot_df["column"]]

    ax_ins = ax.inset_axes(inset_bounds)
    vals   = (plot_df[metric].values.astype(float)
              if metric in plot_df.columns
              else np.full(len(plot_df), np.nan))
    n_methods = len(plot_df)
    y_pos     = np.arange(n_methods)

    bars = ax_ins.barh(y_pos, vals, color=colors, edgecolor="white", lw=0.4)
    valid_vals = vals[~np.isnan(vals)]
    if len(valid_vals):
        ax_ins.set_xlim(max(0.0, valid_vals.min() - 0.02),
                        min(1.0, valid_vals.max() + 0.05))
    ax_ins.set_yticks(y_pos)
    ax_ins.set_yticklabels(labels, fontsize=7.5)
    ax_ins.invert_yaxis()
    ax_ins.tick_params(axis="x", labelbottom=False, length=0)
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax_ins.text(bar.get_width() + 0.001,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=8)
    ax_ins.set_title(title_str, fontsize=7, pad=2)
    ax_ins.patch.set_alpha(0.85)
    ax_ins.set_facecolor("white")
    for spine in ax_ins.spines.values():
        spine.set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════
# ROC + PR curves side by side
# ═══════════════════════════════════════════════════════════════════════════

def plot_roc_curves(df: pd.DataFrame, top_cols: list, our_col: str,
                    label_col: str, out_path: Path, n: int = 20,
                    subtitle: str = "", bold_cols: list = None,
                    metrics_df: pd.DataFrame = None) -> None:
    """ROC and AUPRC curves side by side for anchor methods only.
    Embeds AUROC/AUPRC barplots as insets replacing the legends.
    Pass metrics_df (from build_metrics_df) to enable the insets."""
    labels  = df[label_col]
    avail   = list(df.columns)
    methods = _anchor_methods(our_col, available_cols=avail)
    class_info = _class_subtitle(labels)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(8, 4))

    for col in methods:
        if col not in df.columns:
            continue
        color  = col_color(col, our_col)
        lw     = 2 if col in HIGHLIGHT_COLS else 1.5
        zorder = 5 if col in HIGHLIGHT_COLS else 2
        s      = try_numeric(df[col])
        mask   = s.notna()
        if mask.sum() < 10:
            continue
        s_f, _ = flip_if_inverse(s[mask], labels[mask])
        dname  = display_name(col)
        try:
            fpr, tpr, _ = roc_curve(labels[mask], s_f)
            auroc = roc_auc_score(labels[mask], s_f)
            ax_roc.plot(fpr, tpr, color=color, lw=lw, alpha=0.7,
                        zorder=zorder, label=f"{dname} ({auroc:.3f})")
            rec, prec, _ = precision_recall_curve(labels[mask], s_f)
            auprc = average_precision_score(labels[mask], s_f)
            ax_pr.plot(rec, prec, color=color, lw=lw, alpha=0.7,
                       zorder=zorder, label=f"{dname} ({auprc:.3f})")
        except Exception:
            pass

    # ax_roc.plot([0, 1], [0, 1], color="black", ls="--", lw=0.8)
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    _add_class_info(ax_roc, "ROC Curves", subtitle, class_info)

    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    _add_class_info(ax_pr, "PR Curves", subtitle, class_info)

    if metrics_df is not None:
        _draw_bar_inset(fig, ax_roc, metrics_df, our_col, "auroc", "AUROC",
                        inset_bounds=[0.67, 0.03, 0.22, 0.48])
        _draw_bar_inset(fig, ax_pr, metrics_df, our_col, "prauc", "AUPRC",
                        inset_bounds=[0.30, 0.03, 0.22, 0.48])

    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)


def plot_pr_curves(df, top_cols, our_col, label_col, out_path, n=20,
                   subtitle="", bold_cols=None) -> None:
    """No-op — PR curve is now included in plot_roc_curves side-by-side figure."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# AUROC + AUPRC barplot side by side
# ═══════════════════════════════════════════════════════════════════════════

def plot_auroc_barplot(metrics_df: pd.DataFrame, our_auroc: float,
                       anchor_cols: list, out_path: Path, n: int = 20,
                       our_col: str = "", highlight_cols: list = None,
                       labels: pd.Series = None) -> None:
    """Horizontal barplot for AUROC (left) and AUPRC (right), anchor methods only."""
    avail   = metrics_df["column"].tolist()
    methods = _anchor_methods(our_col, available_cols=avail)
    plot_df = (metrics_df[metrics_df["column"].isin(methods)]
               .drop_duplicates(subset="column")
               .set_index("column").reindex(methods).reset_index())
    colors     = [col_color(c, our_col) for c in plot_df["column"]]
    dnames     = [display_name(c) for c in plot_df["column"]]
    class_info = _class_subtitle(labels) if labels is not None else ""

    fig, axes = plt.subplots(1, 2, figsize=(11, max(4, len(plot_df) * 0.55)))

    for ax, metric, title_str in [
        (axes[0], "auroc", "AUROC"),
        (axes[1], "prauc", "AUPRC"),
    ]:
        vals = (plot_df[metric].values.astype(float)
                if metric in plot_df.columns
                else np.full(len(plot_df), np.nan))
        bars = ax.barh(dnames, vals, color=colors, edgecolor="white", lw=0.5)
        if metric == "auroc":
            ax.axvline(0.5, color="gray", lw=0.8, ls=":")
        ax.set_xlabel(title_str, fontsize=10)
        valid_vals = vals[~np.isnan(vals)]
        if len(valid_vals):
            ax.set_xlim(max(0.0, valid_vals.min() - 0.02),
                        min(1.0, valid_vals.max() + 0.05))
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=8)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_width() + 0.001,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=7)
        title = f"{title_str} — Anchor methods"
        if class_info:
            title += f"\n{class_info}"
        ax.set_title(title, fontsize=9)

    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_metrics_heatmap(metrics_df: pd.DataFrame, out_path: Path, n: int = 20,
                         our_col: str = "", highlight_cols: list = None,
                         labels: pd.Series = None) -> None:
    """Metrics heatmap for anchor methods. Row labels colored by method color."""
    avail       = metrics_df["column"].tolist()
    methods     = _anchor_methods(our_col, available_cols=avail)
    metric_cols = ["auroc", "prauc", "pauroc_fpr10", "mcc", "f1", "balanced_acc"]
    metric_display = {c: ("AUPRC" if c == "prauc" else c.upper())
                      for c in metric_cols}
    plot_df     = (metrics_df[metrics_df["column"].isin(methods)]
                   .drop_duplicates(subset="column")
                   .set_index("column").reindex(methods).reset_index())
    class_info  = _class_subtitle(labels) if labels is not None else ""
    dnames      = [display_name(c) for c in plot_df["column"]]

    data    = plot_df[metric_cols].values.astype(float)
    fig, ax = plt.subplots(figsize=(10, max(3, len(plot_df) * 0.55)))
    im      = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels([metric_display[c] for c in metric_cols],
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(dnames, fontsize=8)
    for i, col in enumerate(plot_df["column"]):
        ax.get_yticklabels()[i].set_color(col_color(col, our_col))
    for i in range(len(plot_df)):
        for j in range(len(metric_cols)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    plt.colorbar(im, ax=ax, shrink=0.6)
    title = "Metrics Heatmap — Anchor methods"
    if class_info:
        title += f"\n{class_info}"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Scatter: AUROC vs N variants
# ═══════════════════════════════════════════════════════════════════════════

def plot_auroc_scatter(metrics_df: pd.DataFrame, our_auroc: float, out_path: Path,
                       our_col: str = "", highlight_cols: list = None,
                       labels: pd.Series = None) -> None:
    """AUROC vs N variants. Anchor methods highlighted with fixed colors."""
    avail      = metrics_df["column"].tolist()
    anchor     = set(_anchor_methods(our_col, available_cols=avail))
    df_all     = metrics_df.dropna(subset=["auroc"])
    df_other   = df_all[~df_all["column"].isin(anchor)]
    df_anch    = df_all[df_all["column"].isin(anchor)]
    class_info = _class_subtitle(labels) if labels is not None else ""

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df_other["n_variants"], df_other["auroc"],
               color=METHOD_COLORS["other"], alpha=0.5,
               edgecolors="none", s=40, label="other methods")
    for _, row in df_anch.iterrows():
        color = col_color(row["column"], our_col)
        lw    = 2.0 if row["column"] == our_col else 1.0
        ax.scatter(row["n_variants"], row["auroc"],
                   color=color, s=120, zorder=5, edgecolors="black", lw=lw)
        ax.annotate(display_name(row["column"]), (row["n_variants"], row["auroc"]),
                    fontsize=7, xytext=(5, 3), textcoords="offset points",
                    color=color)
    ax.axhline(our_auroc, color=METHOD_COLORS["our_model"],
               ls="--", lw=1.5,
               label=f"{display_name(our_col)} ({our_auroc:.3f})")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("N variants with score", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    title = "AUROC vs N variants — Anchor methods highlighted"
    if class_info:
        title += f"\n{class_info}"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Pairwise Kendall τ correlation heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_score_correlation(df: pd.DataFrame, our_col: str, out_path: Path,
                           labels: pd.Series = None) -> None:
    """
    Pairwise Kendall τ correlation heatmap with hierarchical clustering.
    Rows and columns are reordered by Ward linkage on the τ distance matrix.
    Dendrograms are drawn on the top (columns) and left (rows) margins.
    """
    from scipy.stats import kendalltau
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    from matplotlib.gridspec import GridSpec

    avail   = list(df.columns)
    methods = _anchor_methods(our_col, available_cols=avail)
    methods = [m for m in methods if m in df.columns
               and try_numeric(df[m]).notna().sum() >= 10]
    n          = len(methods)
    class_info = _class_subtitle(labels) if labels is not None else ""

    # ── Build full Kendall τ matrix ────────────────────────────────────
    tau_matrix = np.zeros((n, n))
    for i in range(n):
        tau_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            si = try_numeric(df[methods[i]])
            sj = try_numeric(df[methods[j]])
            mask = si.notna() & sj.notna()
            if mask.sum() >= 10:
                tau, _ = kendalltau(si[mask].values, sj[mask].values)
                tau_matrix[i, j] = tau_matrix[j, i] = abs(tau)

    # ── Hierarchical clustering on distance = 1 - τ ───────────────────
    dist_condensed = squareform(1.0 - tau_matrix, checks=False)
    dist_condensed = np.clip(dist_condensed, 0, None)   # numerical safety
    Z = linkage(dist_condensed, method="ward")

    # Get leaf order from dendrogram (suppress plot)
    dend = dendrogram(Z, no_plot=True)
    order = dend["leaves"]   # reordered indices

    # Reorder matrix and labels
    tau_reordered = tau_matrix[np.ix_(order, order)]
    methods_ord   = [methods[i] for i in order]
    dnames_ord    = [display_name(m) for m in methods_ord]

    # ── Figure layout ─────────────────────────────────────────────────
    # Use original square aspect ratio (same as pre-dendrogram version),
    # with small margins for the dendrograms.
    base = max(5, n * 0.3)          # original width == height
    dend_frac_h = 0.05              # top dendrogram = 15% of heatmap height
    dend_frac_w = 0.25              # left dendrogram = 25% of heatmap width
    cbar_frac   = 0.04              # colorbar = 4% of heatmap width

    fig_w = base * (1 + dend_frac_w + cbar_frac) + 7
    fig_h = base * (1 + dend_frac_h) + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(2, 3,
                   width_ratios=[dend_frac_w, 1, cbar_frac],
                   height_ratios=[dend_frac_h, 1],
                   hspace=0.01, wspace=0.8,
                   left=0.25, right=0.92, top=0.90, bottom=0.25)

    ax_top  = fig.add_subplot(gs[0, 1])   # top dendrogram
    ax_left = fig.add_subplot(gs[1, 0])   # left dendrogram
    ax_heat = fig.add_subplot(gs[1, 1])   # heatmap
    ax_cbar = fig.add_subplot(gs[1, 2])   # colorbar

    # ── Top dendrogram ────────────────────────────────────────────────
    # Draw then hide all spines/ticks — axis("off") can leave stray lines,
    # so we do it manually to be safe.
    dendrogram(Z, ax=ax_top, orientation="top",
               link_color_func=lambda k: "#555555",
               no_labels=True)
    ax_top.set_xlim(ax_heat.get_xlim() if hasattr(ax_heat, '_viewLim') else (-0.5, n - 0.5))
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    ax_top.set_xticks([])
    ax_top.set_yticks([])

    # ── Left dendrogram ───────────────────────────────────────────────
    dendrogram(Z, ax=ax_left, orientation="left",
               link_color_func=lambda k: "#555555",
               no_labels=True)
    ax_left.invert_yaxis()
    for spine in ax_left.spines.values():
        spine.set_visible(False)
    ax_left.set_xticks([])
    ax_left.set_yticks([])

    # ── Heatmap ───────────────────────────────────────────────────────
    cmap     = plt.cm.Blues
    vmin_tau = 0.3
    vmax_tau = 0.7
    norm     = plt.Normalize(vmin=vmin_tau, vmax=vmax_tau)

    for i in range(n):
        for j in range(n):
            val   = tau_reordered[i, j]
            color = cmap(norm(min(val, vmax_tau)))
            rect  = plt.Rectangle([j - 0.5, i - 0.5], 1, 1,
                                   facecolor=color, edgecolor="white", lw=0.6)
            ax_heat.add_patch(rect)
            brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            txt_color  = "white" if brightness < 0.5 else "black"
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=max(6, 12 - n // 6), color=txt_color),
                         # fontweight="bold" if i == j else "normal")

    ax_heat.set_xlim(-0.5, n - 0.5)
    ax_heat.set_ylim(-0.5, n - 0.5)
    ax_heat.invert_yaxis()
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(dnames_ord, rotation=45, ha="right", fontsize=14)
    # Shift y-tick labels rightward (positive pad) so they clear the dendrogram
    ax_heat.set_yticklabels(dnames_ord, fontsize=14)
    ax_heat.tick_params(axis="y", pad=2)
    for tick, method in zip(ax_heat.get_xticklabels(), methods_ord):
        tick.set_color(col_color(method, our_col))
    for tick, method in zip(ax_heat.get_yticklabels(), methods_ord):
        tick.set_color(col_color(method, our_col))

    # ── Colorbar ──────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=vmin_tau, vmax=vmax_tau))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label("Kendall |τ|  (capped at 0.7)", fontsize=8)

    title = "Pairwise Kendall |τ|"
    if class_info:
        title += f"\n{class_info}"
    fig.suptitle(title, fontsize=14, y=0.97, x=0.62)
    _savefig(fig, out_path)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Zero-shot plots — no-ops (requirement #6)
# NT2-Zeroshot is now a first-class anchor method in all main figures.
# These stubs prevent import errors in callers.
# ═══════════════════════════════════════════════════════════════════════════

ZEROSHOT_ORDER = [
    "zeroshot_NT2_seq12k_score",
    "zeroshot_NT2_seq6k_score",
    "zeroshot_NT1_seq6k_score",
    "zeroshot_CaduceusPS_seq30k_score",
    "zeroshot_CaduceusPh_seq30k_score",
]


def plot_zeroshot_roc_curves(*args, **kwargs) -> None:
    """No-op — kept for backward compat."""
    pass


def plot_zeroshot_barplot(*args, **kwargs) -> None:
    """No-op — kept for backward compat."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# GLM-Missense vs NT2-Zeroshot ROC+PR curves (requirement #2)
# ═══════════════════════════════════════════════════════════════════════════

GLM_ZEROSHOT_COLS = [
    "GLM-Missense_score",
    "zeroshot_NT2_seq12k_score",
]

def plot_glm_zeroshot_roc_curves(df: pd.DataFrame, our_col: str,
                                  label_col: str, out_path: Path,
                                  subtitle: str = "",
                                  metrics_df: pd.DataFrame = None) -> None:
    """
    ROC + PR curves side by side comparing GLM-Missense vs NT2-Zeroshot only.
    NT2-Zeroshot is drawn in black; GLM-Missense uses its standard crimson.
    """
    labels     = df[label_col]
    class_info = _class_subtitle(labels)
    cols       = [c for c in GLM_ZEROSHOT_COLS if c in df.columns]

    def _glm_zeroshot_color(col):
        if col == "zeroshot_NT2_seq12k_score":
            return "#000000"   # black for NT2-Zeroshot in this plot
        return col_color(col, our_col)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(8, 4))

    for col in cols:
        color  = _glm_zeroshot_color(col)
        lw     = 2.5 if col == our_col else 1.8
        zorder = 5 if col == our_col else 2
        s      = try_numeric(df[col])
        mask   = s.notna()
        if mask.sum() < 10:
            continue
        s_f, _ = flip_if_inverse(s[mask], labels[mask])
        dname  = display_name(col)
        try:
            fpr, tpr, _ = roc_curve(labels[mask], s_f)
            auroc = roc_auc_score(labels[mask], s_f)
            ax_roc.plot(fpr, tpr, color=color, lw=lw, alpha=0.9,
                        zorder=zorder, label=f"{dname} ({auroc:.3f})")
            rec, prec, _ = precision_recall_curve(labels[mask], s_f)
            auprc = average_precision_score(labels[mask], s_f)
            ax_pr.plot(rec, prec, color=color, lw=lw, alpha=0.9,
                       zorder=zorder, label=f"{dname} ({auprc:.3f})")
        except Exception:
            pass

    # ax_roc.plot([0, 1], [0, 1], color="black", ls="--", lw=0.8)
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.legend(fontsize=10, loc="lower right")
    _add_class_info(ax_roc, "ROC Curves", subtitle, class_info)

    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.legend(fontsize=10, loc="lower left")
    _add_class_info(ax_pr, "PR Curves", subtitle, class_info)

    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Stratified comparison bar (used inline in evaluate.py)
# ═══════════════════════════════════════════════════════════════════════════

def plot_comparison_across_strata(comparison_df: pd.DataFrame,
                                  out_path: Path, metric: str = "auroc",
                                  bold_cols: list = None) -> None:
    """Bar chart: x = strata (canonical order), groups = anchor methods."""
    our_col = bold_cols[0] if bold_cols else ""
    avail   = comparison_df["column"].unique().tolist()
    methods = _anchor_methods(our_col, available_cols=avail)
    strata  = _sort_strata(comparison_df["stratum"].unique().tolist())
    dnames  = [display_name(m) for m in methods]
    metric_label = "AUPRC" if metric == "prauc" else metric.upper()

    x     = np.arange(len(strata))
    width = min(0.8 / len(methods), 0.15)

    fig, ax = plt.subplots(figsize=(max(10, len(strata) * 1.2), 6))
    for i, (method, dname) in enumerate(zip(methods, dnames)):
        color  = col_color(method, our_col)
        sub    = comparison_df[comparison_df["column"] == method].set_index("stratum")
        vals   = []
        for s in strata:
            if s not in sub.index:
                vals.append(np.nan)
            else:
                v = sub.loc[s, metric]
                vals.append(float(v.iloc[0]) if hasattr(v, "iloc") else float(v))
        offset = i * width - width * (len(methods) - 1) / 2
        ax.bar(x + offset, vals, width, label=dname,
               color=color, alpha=0.88, edgecolor="white", lw=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(strata, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_title(f"{metric_label} by Stratum — Anchor methods", fontsize=12)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# AF distribution plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_af_distribution(df: pd.DataFrame, af_cols: list, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(af_cols), figsize=(6 * len(af_cols), 4))
    if len(af_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, af_cols):
        if col not in df.columns:
            continue
        vals = try_numeric(df[col]).dropna()
        vals = vals[vals > 0]
        if len(vals) == 0:
            continue
        ax.hist(np.log10(vals + 1e-10), bins=50,
                color=METHOD_COLORS["other"], alpha=0.7)
        ax.set_xlabel("log10(AF)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{col}\nn={len(vals):,}", fontsize=10)
    fig.suptitle("Allele Frequency Distribution", fontsize=12)
    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)