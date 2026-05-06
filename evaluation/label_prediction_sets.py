#!/usr/bin/env python3
"""
label_prediction_sets.py

For each of the 7 anchor predictors, adds:
  1. A binary prediction label  pred_{method}     (1=pathogenic, 0=benign)
  2. A correctness flag         correct_{method}  (1=correct, 0=incorrect)
  3. For every (focal_method, others_correct_max=0..6) pair:
       {focal_method}_correct_le{N}
     = 1 if the focal method is correct AND ≤N of the other 6 methods are correct,
       0 otherwise.
     This gives 7 × 7 = 49 additional indicator columns.

Outputs (per predictions subdirectory):
  - merged_prediction_labels_all.tsv          all variants, all new columns
  - merged_prediction_labels_all_overlap.tsv  same but only variants where
                                               all 7 methods have a non-NaN score

Usage:
    python evaluation/label_prediction_sets.py \
        [--predictions_dir results/predictions]
"""

import argparse
import glob
import os
import sys

import pandas as pd


# ── Method registry ───────────────────────────────────────────────────────────
# Ordered dict: name → (score_col, threshold, direction)
ALL_METHODS: dict[str, tuple[str, float, str]] = {
    "GLM-Missense": ("GLM-Missense_score", 0.5,   "high"),
    "AlphaMissense": ("AlphaMissense_score", 0.564, "high"),
    "ESM1b":         ("ESM1b_score",         -7.5,  "low"),
    "REVEL":         ("REVEL_score",          0.5,  "high"),
    "CADD":          ("CADD_phred",           20.0, "high"),
    "Polyphen2":     ("Polyphen2_HVAR_score", 0.902, "high"),
    "SIFT":          ("SIFT_score",           0.05, "low"),
}
METHOD_NAMES = list(ALL_METHODS.keys())   # fixed order throughout


# ── Helpers ───────────────────────────────────────────────────────────────────

def binarize_score(series: pd.Series, threshold: float, direction: str) -> pd.Series:
    """Return 1 (pathogenic) or 0 (benign) as Int64; NaN where score is missing."""
    series = pd.to_numeric(series, errors="coerce")
    pred = (series >= threshold).astype("Int64") if direction == "high" \
           else (series <= threshold).astype("Int64")
    pred[series.isna()] = pd.NA
    return pred


# ── Main per-file processor ───────────────────────────────────────────────────

def process_one(merged_path: str) -> None:
    pred_dir = os.path.dirname(merged_path)

    print(f"\n{'='*70}")
    print(f"Processing: {merged_path}")
    print(f"{'='*70}")

    df = pd.read_csv(merged_path, sep="\t", low_memory=False)
    print(f"  {len(df)} variants loaded")

    if "true_label" not in df.columns:
        print(f"  [SKIP] 'true_label' column not found — skipping {merged_path}")
        return

    true = df["true_label"].astype(int)

    # ── Step 1: pred_ and correct_ columns for every method ──────────────────
    print("\nAdding binary prediction labels ...")

    for method, (score_col, threshold, direction) in ALL_METHODS.items():
        pred_col    = f"pred_{method}"
        correct_col = f"correct_{method}"

        if score_col not in df.columns:
            print(f"  [skip] {score_col} not found — {method} will be NaN")
            df[pred_col]    = pd.NA
            df[correct_col] = pd.NA
            continue

        df[pred_col] = binarize_score(df[score_col], threshold, direction)
        correct = (df[pred_col] == true).astype("Int64")
        correct[df[pred_col].isna()] = pd.NA
        df[correct_col] = correct

        thr_str = (">=" if direction == "high" else "<=") + str(threshold)
        print(f"  {method:20s} | threshold={thr_str:10s} | "
              f"pred_path={df[pred_col].eq(1).sum():5d} | "
              f"pred_benign={df[pred_col].eq(0).sum():5d} | "
              f"correct={correct.eq(1).sum():5d}/{correct.notna().sum()}")

    # ── Step 2: 7×7 indicator columns ─────────────────────────────────────────
    # For each focal method and each N in 0..6:
    #   {focal}_correct_le{N} = 1 iff focal is correct AND ≤N others are correct
    print("\nAdding focal_correct_leN indicator columns (7 methods × 7 thresholds) ...")

    for focal in METHOD_NAMES:
        focal_correct_col = f"correct_{focal}"
        other_methods     = [m for m in METHOD_NAMES if m != focal]
        other_correct_cols = [
            f"correct_{m}" for m in other_methods
            if f"correct_{m}" in df.columns and df[f"correct_{m}"].notna().any()
        ]

        # Count how many *other* methods are correct per variant
        n_others_correct = (
            df[other_correct_cols]
            .apply(lambda row: row.eq(1).sum(), axis=1)
            .astype(int)
        )

        focal_is_correct = df[focal_correct_col].eq(1)   # NaN → False

        for N in range(7):          # 0, 1, 2, 3, 4, 5, 6
            col_name = f"{focal}_correct_le{N}"
            df[col_name] = (focal_is_correct & n_others_correct.le(N)).fillna(False).astype(int)
            n_flagged = df[col_name].sum()
            print(f"  {col_name:35s}: {n_flagged:5d} variants")

    # ── Step 3: save full annotated table ─────────────────────────────────────
    out_full = os.path.join(pred_dir, "merged_prediction_labels_all.tsv")
    df.to_csv(out_full, sep="\t", index=False)
    print(f"\n  Saved full table ({len(df)} variants, {len(df.columns)} columns): {out_full}")

    # ── Step 4: overlap table (all 7 methods have a score) ────────────────────
    pred_cols   = [f"pred_{m}" for m in METHOD_NAMES]
    present_pred_cols = [c for c in pred_cols if c in df.columns]
    overlap_mask = df[present_pred_cols].notna().all(axis=1)
    overlap = df[overlap_mask].copy()

    out_overlap = os.path.join(pred_dir, "merged_prediction_labels_all_overlap.tsv")
    overlap.to_csv(out_overlap, sep="\t", index=False)
    print(f"  Saved overlap table ({len(overlap)} variants): {out_overlap}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions_dir",
        default="results/predictions",
        help="Root directory containing */merged.tsv subdirectories",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.predictions_dir, "*", "merged.tsv")
    merged_files = sorted(glob.glob(pattern))

    if not merged_files:
        sys.exit(f"[error] No files matched: {pattern}")

    print(f"Found {len(merged_files)} merged.tsv file(s):")
    for f in merged_files:
        print(f"  {f}")

    for merged_path in merged_files:
        process_one(merged_path)

    print(f"\n{'='*70}")
    print("All done.")


if __name__ == "__main__":
    main()