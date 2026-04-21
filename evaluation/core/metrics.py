"""
Shared metric computation utilities for variant pathogenicity evaluation.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    matthews_corrcoef, f1_score,
    balanced_accuracy_score, brier_score_loss,
)
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# ── Numeric parsing ────────────────────────────────────────────────────────

def try_numeric(series: pd.Series) -> pd.Series:
    """
    Convert series to numeric. Handles:
      - '.' / 'NA' / '' / NaN → NaN
      - semicolon- or comma-separated multi-values → max numeric value
    """
    def parse_val(x):
        if pd.isna(x) or str(x).strip() in (".", "", "NA", "nan"):
            return np.nan
        parts = [p.strip() for p in str(x).replace(";", ",").split(",")
                 if p.strip() not in (".", "", "NA")]
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                pass
        return max(nums) if nums else np.nan
    return series.apply(parse_val)


def is_valid(series: pd.Series) -> pd.Series:
    """Boolean mask: True where series has a real numeric value."""
    return try_numeric(series).notna()


# ── Score utilities ────────────────────────────────────────────────────────

def flip_if_inverse(scores: pd.Series, labels: pd.Series):
    """Flip score direction if AUC < 0.5 (e.g. SIFT: lower = more pathogenic)."""
    try:
        auc = roc_auc_score(labels, scores)
        if auc < 0.5:
            return -scores, True
        return scores, False
    except Exception:
        return scores, False


# ── Partial AUC helpers ────────────────────────────────────────────────────

def partial_auc(fpr: np.ndarray, tpr: np.ndarray, max_fpr: float = 0.1) -> float:
    mask = fpr <= max_fpr
    if mask.sum() < 2:
        return np.nan
    return np.trapz(tpr[mask], fpr[mask]) / max_fpr


def partial_prauc(recall: np.ndarray, precision: np.ndarray, max_recall: float = 0.1) -> float:
    mask = recall <= max_recall
    if mask.sum() < 2:
        return np.nan
    return np.trapz(precision[mask], recall[mask]) / max_recall


# ── Core metric computation ────────────────────────────────────────────────

def compute_metrics(scores_raw: pd.Series, labels: pd.Series, col_name: str) -> dict | None:
    """
    Compute full metric suite for one score column.
    Returns None if insufficient data.
    """
    scores = try_numeric(scores_raw)
    mask   = scores.notna()

    if mask.sum() < 10 or labels[mask].nunique() < 2:
        return None

    s = scores[mask]
    y = labels[mask]
    s, flipped = flip_if_inverse(s, y)

    def safe(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception:
            return np.nan

    auroc = safe(roc_auc_score, y, s)
    prauc = safe(average_precision_score, y, s)

    pauroc_10 = pauroc_20 = pprauc_10 = np.nan
    try:
        fpr, tpr, _ = roc_curve(y, s)
        pauroc_10   = partial_auc(fpr, tpr, max_fpr=0.1)
        pauroc_20   = partial_auc(fpr, tpr, max_fpr=0.2)
    except Exception:
        pass
    try:
        rec, prec, _ = precision_recall_curve(y, s)
        pprauc_10    = partial_prauc(rec, prec, max_recall=0.1)
    except Exception:
        pass

    mcc = f1 = bacc = brier = np.nan
    try:
        scaler   = MinMaxScaler()
        s_scaled = scaler.fit_transform(s.values.reshape(-1, 1)).ravel()
        y_pred   = (s_scaled >= 0.5).astype(int)
        mcc      = matthews_corrcoef(y, y_pred)
        f1       = f1_score(y, y_pred, zero_division=0)
        bacc     = balanced_accuracy_score(y, y_pred)
        brier    = brier_score_loss(y, s_scaled)
    except Exception:
        pass

    r = lambda x: round(x, 4) if not np.isnan(x) else np.nan

    n_pos      = int(y.sum())
    n_total    = int(mask.sum())
    prevalence = round(n_pos / n_total, 4) if n_total > 0 else np.nan

    return {
        "column":       col_name,
        "n_variants":   n_total,
        "n_positive":   n_pos,
        "prevalence":   prevalence,
        "auroc":        r(auroc),
        "prauc":        r(prauc),
        "pauroc_fpr10": r(pauroc_10),
        "pauroc_fpr20": r(pauroc_20),
        "pprauc_r10":   r(pprauc_10),
        "mcc":          r(mcc),
        "f1":           r(f1),
        "balanced_acc": r(bacc),
        "brier_score":  r(brier),
        "flipped":      flipped,
    }


def evaluate_all_columns(df: pd.DataFrame, labels: pd.Series,
                         skip_cols: set, our_col: str) -> tuple[dict, list]:
    """
    Evaluate our model column + all non-skipped columns.
    Returns (our_metrics_dict, list_of_dbnsfp_metric_dicts).
    """
    our_metrics  = compute_metrics(df[our_col], labels, our_col)
    dbnsfp_cols  = [c for c in df.columns if c not in skip_cols]
    results, n_skipped = [], 0
    for col in dbnsfp_cols:
        r = compute_metrics(df[col], labels, col)
        if r is None:
            n_skipped += 1
        else:
            results.append(r)
    print(f"  Evaluated: {len(results)}, skipped: {n_skipped}")
    return our_metrics, results


def build_metrics_df(our_metrics: dict, dbnsfp_metrics: list) -> pd.DataFrame:
    """Combine ours + dbnsfp metrics, sorted by AUROC descending."""
    dbnsfp_df  = pd.DataFrame(dbnsfp_metrics).sort_values("auroc", ascending=False)
    metrics_df = pd.concat([pd.DataFrame([our_metrics]), dbnsfp_df], ignore_index=True)
    return metrics_df


SUMMARY_COLS = ["column", "n_variants", "n_positive", "prevalence", "auroc", "prauc",
                "pauroc_fpr10", "mcc", "f1", "balanced_acc"]
# Note: the internal key for precision-recall AUC remains "prauc" in all_metrics.tsv.
# All display strings use "AUPRC" — see plots.py _metric_label().


def build_summary(metrics_df: pd.DataFrame, our_col: str, anchor_cols: list, top_n: int = 20) -> pd.DataFrame:
    """Extract anchor rows + top-N others into a summary table."""
    anchor_rows = metrics_df[metrics_df["column"].isin([our_col] + anchor_cols)]
    top_others  = (metrics_df[~metrics_df["column"].isin([our_col] + anchor_cols)]
                   .dropna(subset=["auroc"]).nlargest(top_n, "auroc"))
    return (pd.concat([anchor_rows, top_others])
              .drop_duplicates("column")[SUMMARY_COLS]
              .reset_index(drop=True))