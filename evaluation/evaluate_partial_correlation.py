"""
evaluate_partial_correlation.py

Partial correlation analysis: how much unique information does each predictor
contribute to variant pathogenicity (true_label), after controlling for all
other predictors?

Two panels (A and B only):
  A. Partial Spearman r  (controlling for all other methods)
  B. Bivariate Spearman r  (no control — full signal for comparison)

Runs over ALL results/predictions/*/merged.tsv tables automatically.
One figure per dataset, saved alongside the merged table.

Bar style matches plot_feature_importance.py exactly:
  - Same TICK_FONTSIZE, LABEL_FONTSIZE, TITLE_FONTSIZE, BAR_HEIGHT, FIGSIZE
  - Colors from core.plots.col_color / METHOD_COLORS
  - Arial font via core.plots font registration block
  - pdf.fonttype=42 for editable text in Adobe Illustrator

Run from project root:
    python evaluation/evaluate_partial_correlation.py
"""

import os
import sys
import warnings
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Import color helpers from core (also triggers Arial registration) ──────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
from core.plots import col_color, display_name, METHOD_COLORS

# ── Shared style constants — match plot_feature_importance.py exactly ──────
TICK_FONTSIZE  = 14
LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 14
BAR_HEIGHT     = 0.55
FIGSIZE        = (9, 4)      # wider than feature importance to fit 2 panels

OUR_COL      = "GLM-Missense_score"
TRUE_LABEL   = "true_label"

# Methods to include — col name, score direction
ALL_METHODS: dict[str, tuple[str, str]] = {
    "MetaMissense":  ("MetaMissense_score",    "high"),
    "GLM-Missense":  ("GLM-Missense_score",    "high"),
    "AlphaMissense": ("AlphaMissense_score",   "high"),
    "ESM1b":         ("ESM1b_score",           "low"),
    "REVEL":         ("REVEL_score",           "high"),
    "CADD":          ("CADD_phred",            "high"),
    "Polyphen2":     ("Polyphen2_HVAR_score",  "high"),
    "SIFT":          ("SIFT_score",            "low"),
}


# ── Partial correlation (manual residualisation — no pingouin dependency) ──

def partial_spearman(x, y, covars):
    """
    Partial Spearman correlation between x and y, controlling for covars.
    Returns (r, p, ci_low, ci_high) with Fisher-z 95% CI.
    """
    from scipy.stats import rankdata
    data = np.column_stack([x, y] + list(covars))
    mask = np.all(np.isfinite(data), axis=1)
    data = data[mask]
    if len(data) < 10:
        return np.nan, np.nan, np.nan, np.nan

    ranked = np.apply_along_axis(rankdata, 0, data)
    rx, ry, rc = ranked[:, 0], ranked[:, 1], ranked[:, 2:]

    def ols_resid(target, predictors):
        X = np.column_stack([np.ones(len(target)), predictors])
        coef, *_ = np.linalg.lstsq(X, target, rcond=None)
        return target - X @ coef

    ex = ols_resid(rx, rc)
    ey = ols_resid(ry, rc)
    r, p = stats.pearsonr(ex, ey)

    n  = len(ex)
    z  = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(n - len(covars) - 3)
    return r, p, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def sig_stars(p):
    if np.isnan(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ── Analysis for one merged.tsv ────────────────────────────────────────────

def run_one(merged_path: Path) -> None:
    dataset = merged_path.parent.name
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"File   : {merged_path}")

    df = pd.read_csv(merged_path, sep="\t", low_memory=False)
    print(f"  {len(df):,} rows loaded")

    method_names  = list(ALL_METHODS.keys())
    score_cols    = [ALL_METHODS[m][0] for m in method_names]

    for col in score_cols + [TRUE_LABEL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flip "low" direction so higher = more pathogenic
    df_f = df.copy()
    for method, (sc, direction) in ALL_METHODS.items():
        if sc in df_f.columns and direction == "low":
            df_f[sc] = -df_f[sc]

    available = [c for c in score_cols if c in df_f.columns]
    if TRUE_LABEL not in df_f.columns:
        print(f"  SKIP: '{TRUE_LABEL}' column not found")
        return
    sub = df_f[available + [TRUE_LABEL]].dropna()
    print(f"  {len(sub):,} complete-case rows")
    if len(sub) < 20:
        print("  SKIP: too few complete-case rows")
        return

    # ── Run analysis ──────────────────────────────────────────────────
    records = []
    for focal_method in method_names:
        focal_col = ALL_METHODS[focal_method][0]
        if focal_col not in sub.columns:
            continue
        other_cols = [ALL_METHODS[m][0] for m in method_names
                      if m != focal_method and ALL_METHODS[m][0] in sub.columns]

        r_sp, p_sp = spearmanr(sub[focal_col], sub[TRUE_LABEL])
        r_pc, p_pc, ci_lo, ci_hi = partial_spearman(
            sub[focal_col].values,
            sub[TRUE_LABEL].values,
            [sub[c].values for c in other_cols]
        )
        records.append({
            "method":            focal_method,
            "score_col":         focal_col,
            "n":                 len(sub),
            "spearman_r":        round(r_sp,  4),
            "spearman_p":        round(p_sp,  6),
            "partial_r":         round(r_pc,  4),
            "partial_p":         round(p_pc,  6),
            "partial_CI95_low":  round(ci_lo, 4),
            "partial_CI95_high": round(ci_hi, 4),
        })
        print(f"  {focal_method:15s}  spearman_r={r_sp:+.4f}  "
              f"partial_r={r_pc:+.4f} (p={p_pc:.2e})")

    if not records:
        print("  SKIP: no methods found")
        return

    results_df = pd.DataFrame(records)

    # ── Save TSV ──────────────────────────────────────────────────────
    tab_path = merged_path.parent / "partial_correlation_results.tsv"
    results_df.to_csv(tab_path, sep="\t", index=False)
    print(f"  Table → {tab_path}")

    # ── Plot panels A + B ─────────────────────────────────────────────
    fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=FIGSIZE)

    def hbar(ax, df_sorted, val_col, err_lo=None, err_hi=None,
             xlabel="", title="", pval_col=None):
        methods = df_sorted["method"].tolist()
        yp      = np.arange(len(methods))

        for i, method in enumerate(methods):
            score_col = ALL_METHODS.get(method, (method,))[0]
            color     = col_color(score_col, OUR_COL)
            val       = df_sorted.iloc[i][val_col]

            # Error bar
            xerr = None
            if err_lo and err_hi:
                lo   = val - df_sorted.iloc[i][err_lo]
                hi   = df_sorted.iloc[i][err_hi] - val
                xerr = [[lo], [hi]]

            ax.barh(i, val,
                    xerr=xerr,
                    height=BAR_HEIGHT,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    error_kw=dict(ecolor="black", capsize=3, lw=1.2),
                    zorder=3)

            # Value + significance label
            # For positive bars: place text to the right of the error bar tip
            # For negative bars: place text to the left of the error bar tip
            stars = ""
            label = f"{val:.3f}"
            x_max = df_sorted[val_col].abs().max()
            # Use error bar extent if available, otherwise just the bar value
            if err_lo and err_hi:
                hi_err = df_sorted.iloc[i][err_hi]
                lo_err = df_sorted.iloc[i][err_lo]
                tip_pos = hi_err   # rightmost tip of positive error bar
                tip_neg = lo_err   # leftmost tip of negative error bar
            else:
                tip_pos = val
                tip_neg = val
            xoff = x_max * 0.04   # padding beyond error bar tip
            if val >= 0:
                ax.text(tip_pos + xoff, i, label,
                        va="center", ha="left",
                        fontsize=TICK_FONTSIZE - 4, color="#222222")
            else:
                ax.text(tip_neg - xoff, i, label,
                        va="center", ha="right",
                        fontsize=TICK_FONTSIZE - 4, color="#222222")

        ax.axvline(0, color="#888888", lw=0.8)
        ax.set_yticks(yp)
        dnames = [display_name(ALL_METHODS.get(m, (m,))[0]) for m in methods]
        ax.set_yticklabels(dnames, fontsize=TICK_FONTSIZE)
        for tick, method in zip(ax.get_yticklabels(), methods):
            tick.set_color(col_color(ALL_METHODS.get(method, (method,))[0], OUR_COL))
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel A — partial r
    df_A = results_df.sort_values("partial_r", ascending=True)
    hbar(ax_A, df_A,
         val_col="partial_r",
         err_lo="partial_CI95_low",
         err_hi="partial_CI95_high",
         xlabel="Partial Spearman r",
         title="Partial Spearman r\n(controlling for all other methods)",
         pval_col="partial_p")
    x_range   = results_df["partial_r"].abs().max()
    x_min_val = results_df["partial_r"].min()
    x_max_val = results_df["partial_r"].max()
    # Extra left margin for negative bars + their text labels
    left_pad  = x_range * 0.55 if x_min_val < 0 else x_range * 0.05
    ax_A.set_xlim(x_min_val - left_pad,
                  x_max_val + x_range * 0.55)

    # Panel B — bivariate r
    df_B = results_df.sort_values("spearman_r", ascending=True)
    hbar(ax_B, df_B,
         val_col="spearman_r",
         xlabel="Spearman r",
         title="Bivariate Spearman r\n(no control)",
         pval_col="spearman_p")
    ax_B.set_xlim(0, results_df["spearman_r"].max() + 0.15)

    n = int(results_df["n"].iloc[0]) if len(results_df) else 0
    fig.suptitle(f"{dataset}  |  n={n:,} complete-case variants",
                 fontsize=TITLE_FONTSIZE, y=1.02)
    fig.tight_layout()

    for ext in (".pdf", ".png"):
        out = merged_path.parent / f"partial_correlation{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"  Saved → {out}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    pattern = str(ROOT / "results" / "predictions" / "*" / "merged.tsv")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        print(f"No merged.tsv files found matching: {pattern}")
        sys.exit(1)
    print(f"Found {len(paths)} merged.tsv files")
    for p in paths:
        run_one(Path(p))
    print("\nAll done.")


if __name__ == "__main__":
    main()