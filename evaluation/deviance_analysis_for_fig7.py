#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_RESCUE_COL = "GLM-Missense_correct_le2"
DEFAULT_AF_COL = "gnomAD4.1_joint_AF"
DEFAULT_LOEUF_COL = "lof.oe_ci.upper"
DEFAULT_SPLICEAI_COL = "spliceai_DS_max"
DEFAULT_RADICAL_COL = "is_radical_aa_change"

LABEL_CANDIDATES = ["true_label", "label", "clinvar_label", "truth", "y"]

FEATURE_DISPLAY_NAMES = {
    "log10_AF": "Allele frequency",
    "LOEUF": "LOEUF",
    "SpliceAI_DS_max": "SpliceAI score",
    "radical_AA_change": "Radical AA change",
}

FEATURE_SHORT_NAMES = {
    "log10_AF": "AF",
    "LOEUF": "LOEUF",
    "SpliceAI_DS_max": "SpliceAI",
    "radical_AA_change": "Radical AA",
}


def pick_col(df: pd.DataFrame, candidates: Iterable[str], what: str) -> str:
    """Pick first available column, with case-insensitive fallback."""
    for c in candidates:
        if c in df.columns:
            return c
    lower_to_actual = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_to_actual:
            return lower_to_actual[c.lower()]
    preview = ", ".join(map(str, df.columns[:80]))
    raise ValueError(
        f"Could not find {what}. Tried: {', '.join(candidates)}\n"
        f"Available columns include: {preview} ..."
    )


def parse_bool_series(s: pd.Series) -> pd.Series:
    """Robustly parse bool/int/string columns into bool."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(float) != 0
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "t", "1", "yes", "y"})
    )


def parse_radical_series(s: pd.Series) -> pd.Series:
    """Parse radical AA-change indicator to 0/1 numeric with missing allowed."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": 1.0, "t": 1.0, "yes": 1.0, "y": 1.0, "1": 1.0,
            "false": 0.0, "f": 0.0, "no": 0.0, "n": 0.0, "0": 0.0,
        })
        .astype(float)
    )


def make_model(max_iter: int, c_value: float, random_state: int) -> Pipeline:
    """Unweighted logistic-regression pipeline with median imputation and z-score scaling."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(
                penalty="l2",
                C=c_value,
                solver="lbfgs",
                class_weight=None,
                max_iter=max_iter,
                random_state=random_state,
            )),
        ]
    )


def all_feature_subsets(features: Sequence[str]) -> List[Tuple[str, ...]]:
    subsets: List[Tuple[str, ...]] = []
    for r in range(len(features) + 1):
        for combo in itertools.combinations(features, r):
            subsets.append(tuple(combo))
    return subsets


def safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def oof_predictions_for_subset(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_subset: Sequence[str],
    *,
    folds: int,
    seed: int,
    max_iter: int,
    c_value: float,
    eps: float = 1e-15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return out-of-fold model predictions and null predictions for one feature subset.

    Null prediction is computed fold-wise using only the training-fold prevalence.
    """
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    n_splits = min(folds, n_pos, n_neg)
    if n_splits < 2:
        raise ValueError(f"Not enough classes for CV: n_pos={n_pos}, n_neg={n_neg}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.full(len(y), np.nan)
    null_pred = np.full(len(y), np.nan)

    base = None
    if len(feature_subset) > 0:
        base = make_model(max_iter=max_iter, c_value=c_value, random_state=seed)

    for train_idx, test_idx in skf.split(np.zeros((len(y), 1)), y):
        train_prev = float(np.mean(y[train_idx]))
        train_prev = float(np.clip(train_prev, eps, 1 - eps))
        null_pred[test_idx] = train_prev

        if len(feature_subset) == 0:
            pred[test_idx] = train_prev
        else:
            model = clone(base)
            model.fit(X.iloc[train_idx][list(feature_subset)], y[train_idx])
            pred[test_idx] = model.predict_proba(X.iloc[test_idx][list(feature_subset)])[:, 1]

    return pred, null_pred


def compute_d2(y: np.ndarray, pred: np.ndarray, null_pred: np.ndarray, eps: float = 1e-15) -> Dict[str, float]:
    """Compute cross-validated Bernoulli deviance explained."""
    pred = np.clip(pred.astype(float), eps, 1 - eps)
    null_pred = np.clip(null_pred.astype(float), eps, 1 - eps)

    ll_model = float(log_loss(y, pred, labels=[0, 1]))
    ll_null = float(log_loss(y, null_pred, labels=[0, 1]))
    d2 = 1 - ll_model / ll_null if ll_null > 0 else np.nan

    return {
        "logloss_model": ll_model,
        "logloss_null": ll_null,
        "D2": d2,
        "D2_percent": 100 * d2,
        "AUROC": safe_auroc(y, pred),
        "AUPRC": safe_auprc(y, pred),
        "baseline_AUPRC_prevalence": float(np.mean(y)),
    }


def shapley_decomposition(v: Dict[Tuple[str, ...], float], features: Sequence[str]) -> Dict[str, float]:
    """
    Shapley decomposition of full-model D^2 across features.

    This is saved for interpretability but not stacked in the main figure because
    cross-validated marginal contributions can be slightly negative.
    """
    p = len(features)
    phi = {f: 0.0 for f in features}
    denom = math.factorial(p)

    for f in features:
        others = [x for x in features if x != f]
        for r in range(p):
            for S in itertools.combinations(others, r):
                S = tuple(sorted(S))
                S_plus = tuple(sorted(S + (f,)))
                weight = math.factorial(r) * math.factorial(p - r - 1) / denom
                phi[f] += weight * (v[S_plus] - v[S])
    return phi


def plot_figure7_total_d2(
    summary_df: pd.DataFrame,
    *,
    out_prefix: str,
    strata: Sequence[str],
    title: str,
    fig_width: float,
    fig_height: float,
) -> None:
    """Main Figure 7 plot: total explained vs not explained."""
    plot_df = summary_df.set_index("stratum").reindex(strata).reset_index()

    x = np.arange(len(strata))
    width = 0.55

    total_d2 = plot_df["D2_percent"].to_numpy(dtype=float)
    explained_display = np.maximum(total_d2, 0.0)
    unexplained_display = 100.0 - explained_display

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    explained_color = "#4C78A8"
    unexplained_color = "lightgray"

    ax.bar(
        x,
        explained_display,
        width,
        label="Explained by Figure 6 covariates",
        color=explained_color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar(
        x,
        unexplained_display,
        width,
        bottom=explained_display,
        label="Not explained by these covariates",
        color=unexplained_color,
        alpha=0.90,
        edgecolor="white",
        linewidth=0.8,
    )

    for xi, total, exp_val, unexp_val in zip(x, total_d2, explained_display, unexplained_display):
        # Label explained segment.
        if exp_val >= 1.5:
            ax.text(xi, exp_val / 2, f"{total:.1f}%", ha="center", va="center", fontsize=10, color="white")
        else:
            ax.text(xi, exp_val + 1.0, f"{total:.1f}%", ha="center", va="bottom", fontsize=10)

        # Label unexplained segment.
        if unexp_val >= 10:
            ax.text(xi, exp_val + unexp_val / 2, f"{100.0 - max(total, 0.0):.1f}%", ha="center", va="center", fontsize=10)

        # Total label near bottom/top of explained segment.
        ax.text(
            xi,
            min(exp_val + 2.0, 96),
            f"Total D$^2$={total:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in strata])
    ax.set_ylim(0, 100)
    ax.set_ylabel("GLM-Missense-rescued membership\ndeviance explained (%)")
    ax.set_title(title)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    # Add a note if any total D2 is negative.
    if np.any(total_d2 < 0):
        ax.text(
            0.0,
            -0.16,
            "Note: negative total D$^2$ values are plotted as 0% explained.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

    fig.tight_layout()

    pdf = f"{out_prefix}.figure7_deviance_explained.pdf"
    png = f"{out_prefix}.figure7_deviance_explained.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Wrote main Figure 7 plot: {pdf}")
    print(f"[INFO] Wrote main Figure 7 plot: {png}")


def plot_signed_shapley(
    shapley_df: pd.DataFrame,
    *,
    out_prefix: str,
    strata: Sequence[str],
    fig_width: float,
    fig_height: float,
) -> None:
    """Optional diagnostic plot of signed Shapley contributions."""
    feature_order = ["log10_AF", "LOEUF", "SpliceAI_DS_max", "radical_AA_change"]
    labels = [FEATURE_DISPLAY_NAMES[f] for f in feature_order]
    plot_df = shapley_df[shapley_df["component"] == "feature"].copy()

    max_abs = np.nanmax(np.abs(plot_df["shapley_D2_percent"].to_numpy(dtype=float)))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    xlim = max_abs * 1.25

    fig, axes = plt.subplots(1, len(strata), figsize=(fig_width, fig_height), sharey=True)
    if len(strata) == 1:
        axes = [axes]

    ypos = np.arange(len(feature_order))
    for ax, s in zip(axes, strata):
        sub = plot_df[plot_df["stratum"] == s].set_index("feature")
        vals = [float(sub.loc[f, "shapley_D2_percent"]) if f in sub.index else 0.0 for f in feature_order]
        ax.barh(ypos, vals, alpha=0.75)
        ax.axvline(0, color="black", lw=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(-xlim, xlim)
        ax.set_xlabel("Contribution to D$^2$ (%)")
        ax.set_title(s.capitalize())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        for y, v in zip(ypos, vals):
            if np.isfinite(v):
                ha = "left" if v >= 0 else "right"
                dx = 0.03 * xlim if v >= 0 else -0.03 * xlim
                ax.text(v + dx, y, f"{v:.2f}%", va="center", ha=ha, fontsize=8)

    fig.suptitle("Signed Shapley contributions to cross-validated deviance explained", y=1.03)
    fig.tight_layout()
    pdf = f"{out_prefix}.signed_shapley_contributions.pdf"
    png = f"{out_prefix}.signed_shapley_contributions.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Wrote optional signed Shapley plot: {pdf}")
    print(f"[INFO] Wrote optional signed Shapley plot: {png}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="Input TSV file")
    p.add_argument("--out-prefix", default="glm_rescue_d2_decomposition_v3", help="Output prefix")
    p.add_argument("--sep", default="\t", help="Input delimiter")
    p.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    p.add_argument("--seed", type=int, default=1, help="Random seed")
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--C", type=float, default=1.0)

    p.add_argument("--label-col", default=None, help="Truth-label column; expects 0=benign, 1=pathogenic")
    p.add_argument("--rescue-col", default=DEFAULT_RESCUE_COL)
    p.add_argument("--af-col", default=DEFAULT_AF_COL)
    p.add_argument("--loeuf-col", default=DEFAULT_LOEUF_COL)
    p.add_argument("--spliceai-col", default=DEFAULT_SPLICEAI_COL)
    p.add_argument("--radical-col", default=DEFAULT_RADICAL_COL)
    p.add_argument("--af-pseudocount", type=float, default=1e-8)

    p.add_argument("--run-combined", action="store_true", help="Also run an all-labels combined analysis")
    p.add_argument("--plot-signed-shapley", action="store_true", help="Also output diagnostic signed Shapley plot")

    p.add_argument("--fig-width", type=float, default=6.6, help="Main Figure 7 width in inches")
    p.add_argument("--fig-height", type=float, default=5.0, help="Main Figure 7 height in inches")
    p.add_argument("--shapley-fig-width", type=float, default=8.2)
    p.add_argument("--shapley-fig-height", type=float, default=3.6)
    p.add_argument("--title", default="GLM-Missense rescue explained by Figure 6 covariates")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_prefix = str(args.out_prefix)
    out_dir = Path(out_prefix).parent
    if str(out_dir) not in {"", "."}:
        out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, sep=args.sep, low_memory=False)
    print(f"[INFO] Loaded {len(df):,} rows and {len(df.columns):,} columns")

    label_col = args.label_col or pick_col(df, LABEL_CANDIDATES, "truth-label column")
    required = [label_col, args.rescue_col, args.af_col, args.loeuf_col, args.spliceai_col, args.radical_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work["_true_label"] = pd.to_numeric(work[label_col], errors="coerce")
    work["_rescue_label"] = parse_bool_series(work[args.rescue_col]).astype(int)
    work["log10_AF"] = np.log10(
        np.maximum(pd.to_numeric(work[args.af_col], errors="coerce"), 0) + args.af_pseudocount
    )
    work["LOEUF"] = pd.to_numeric(work[args.loeuf_col], errors="coerce")
    work["SpliceAI_DS_max"] = pd.to_numeric(work[args.spliceai_col], errors="coerce")
    work["radical_AA_change"] = parse_radical_series(work[args.radical_col])

    work = work[work["_true_label"].isin([0, 1])].copy()
    work = work[work["_rescue_label"].isin([0, 1])].copy()

    features = ["log10_AF", "LOEUF", "SpliceAI_DS_max", "radical_AA_change"]

    strata = [(0, "benign"), (1, "pathogenic")]
    if args.run_combined:
        strata.append((None, "combined"))

    subset_rows: List[Dict[str, object]] = []
    shapley_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for label_value, stratum in strata:
        sub = work.copy() if label_value is None else work[work["_true_label"] == label_value].copy()
        X = sub[features].copy()
        y = sub["_rescue_label"].to_numpy(dtype=int)

        n = len(y)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        prevalence = float(np.mean(y)) if n else np.nan

        print(f"[INFO] {stratum}: n={n:,}, rescued={n_pos:,}, prevalence={prevalence:.4g}")

        if n_pos < 2 or n_neg < 2:
            print(f"[WARN] Skipping {stratum}: not enough positives/negatives.", file=sys.stderr)
            continue

        d2_by_subset: Dict[Tuple[str, ...], float] = {}

        for fs in all_feature_subsets(features):
            pred, null_pred = oof_predictions_for_subset(
                X, y, fs,
                folds=args.folds,
                seed=args.seed,
                max_iter=args.max_iter,
                c_value=args.C,
            )
            metrics = compute_d2(y, pred, null_pred)
            fs_sorted = tuple(sorted(fs))
            d2_by_subset[fs_sorted] = metrics["D2"]

            subset_rows.append({
                "stratum": stratum,
                "feature_subset": "+".join(fs) if fs else "NULL",
                "n_features": len(fs),
                "n": n,
                "n_rescued_positive": n_pos,
                "positive_prevalence": prevalence,
                **metrics,
            })

        full_key = tuple(sorted(features))
        total_d2 = d2_by_subset[full_key]
        total_d2_percent = 100.0 * total_d2
        unexplained_percent = 100.0 - total_d2_percent

        # Feature-level decomposition is saved for interpretation/diagnostics.
        phi = shapley_decomposition(d2_by_subset, features)

        # Determine dominant positive contributor, if any.
        positive_phi = {f: 100.0 * v for f, v in phi.items() if 100.0 * v > 0}
        if positive_phi:
            dominant_feature = max(positive_phi, key=positive_phi.get)
            dominant_feature_display = FEATURE_DISPLAY_NAMES[dominant_feature]
            dominant_feature_percent = positive_phi[dominant_feature]
        else:
            dominant_feature = ""
            dominant_feature_display = ""
            dominant_feature_percent = np.nan

        summary_rows.append({
            "stratum": stratum,
            "n": n,
            "n_rescued_positive": n_pos,
            "positive_prevalence": prevalence,
            "D2": total_d2,
            "D2_percent": total_d2_percent,
            "unexplained_percent": unexplained_percent,
            "dominant_positive_feature": dominant_feature,
            "dominant_positive_feature_display": dominant_feature_display,
            "dominant_positive_feature_D2_percent": dominant_feature_percent,
        })

        print(f"[INFO] {stratum}: full model D2={total_d2:.4f} ({total_d2_percent:.2f}%)")

        for f in features:
            shapley_rows.append({
                "stratum": stratum,
                "component": "feature",
                "feature": f,
                "display_feature": FEATURE_DISPLAY_NAMES[f],
                "short_feature": FEATURE_SHORT_NAMES[f],
                "shapley_D2": phi[f],
                "shapley_D2_percent": 100.0 * phi[f],
                "total_D2": total_d2,
                "total_D2_percent": total_d2_percent,
                "unexplained_percent": unexplained_percent,
                "n": n,
                "n_rescued_positive": n_pos,
                "positive_prevalence": prevalence,
            })

        shapley_rows.append({
            "stratum": stratum,
            "component": "unexplained",
            "feature": "unexplained",
            "display_feature": "Not explained by these covariates",
            "short_feature": "Unexplained",
            "shapley_D2": np.nan,
            "shapley_D2_percent": unexplained_percent,
            "total_D2": total_d2,
            "total_D2_percent": total_d2_percent,
            "unexplained_percent": unexplained_percent,
            "n": n,
            "n_rescued_positive": n_pos,
            "positive_prevalence": prevalence,
        })

    subset_df = pd.DataFrame(subset_rows)
    shapley_df = pd.DataFrame(shapley_rows)
    summary_df = pd.DataFrame(summary_rows)

    subset_path = f"{out_prefix}.subset_d2.tsv"
    shapley_path = f"{out_prefix}.shapley_decomposition.tsv"
    summary_path = f"{out_prefix}.full_model_summary.tsv"

    subset_df.to_csv(subset_path, sep="\t", index=False)
    shapley_df.to_csv(shapley_path, sep="\t", index=False)
    summary_df.to_csv(summary_path, sep="\t", index=False)

    print(f"[INFO] Wrote {subset_path}")
    print(f"[INFO] Wrote {shapley_path}")
    print(f"[INFO] Wrote {summary_path}")

    if not summary_df.empty:
        print("\n=== Full model D2 summary ===")
        print(summary_df.to_string(index=False))

    strata_to_plot = [s for _, s in strata if s in {"benign", "pathogenic"}]
    if not summary_df.empty:
        plot_figure7_total_d2(
            summary_df,
            out_prefix=out_prefix,
            strata=strata_to_plot,
            title=args.title,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
        )

    if args.plot_signed_shapley and not shapley_df.empty:
        plot_signed_shapley(
            shapley_df,
            out_prefix=out_prefix,
            strata=strata_to_plot,
            fig_width=args.shapley_fig_width,
            fig_height=args.shapley_fig_height,
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
