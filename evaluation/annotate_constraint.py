#!/usr/bin/env python3
"""
add_constraint_metrics.py

Annotates all results/predictions/*/merged.tsv files with gnomAD v4.1
constraint metrics (LOEUF, pLI, etc.) joined on gene name.

Usage:
    python evaluation/add_constraint_metrics.py \
        --constraint data/loeuf/gnomad.v4.1.constraint_metrics.tsv \
        --predictions_dir results/predictions \
        [--overwrite]

The script joins on `genename` (in merged.tsv) == `gene` (in constraint
metrics), keeping only canonical transcripts. Columns added:
    lof.oe_ci.upper  (LOEUF)
    lof.pLI
    lof.oe
    lof.obs
    lof.exp
    mis.z_score
    mis.oe
    lof.oe_ci.upper_bin_decile
"""

import argparse
import glob
import os
import sys

import pandas as pd


CONSTRAINT_COLS = [
    "gene",
    "lof.oe_ci.upper",       # LOEUF — primary stratification metric
    "lof.pLI",
    "lof.oe",
    "lof.obs",
    "lof.exp",
    "mis.z_score",
    "mis.oe",
    "lof.oe_ci.upper_bin_decile",
]

# Rename 'gene' -> 'constraint_gene' to avoid collision; join key is genename
RENAME_MAP = {"gene": "constraint_gene"}


def load_constraint(path: str) -> pd.DataFrame:
    """Load gnomAD constraint metrics, keep canonical transcripts only."""
    df = pd.read_csv(path, sep="\t", low_memory=False)

    # Keep canonical transcripts to get one row per gene
    canonical = df[df["canonical"] == True].copy()  # noqa: E712

    # Some genes may still have duplicates (e.g. multiple canonical isoforms);
    # keep the one with the most lof observations as a tiebreaker.
    canonical = (
        canonical
        .sort_values("lof.obs", ascending=False)
        .drop_duplicates(subset="gene", keep="first")
    )

    keep = [c for c in CONSTRAINT_COLS if c in canonical.columns]
    missing = set(CONSTRAINT_COLS) - set(keep)
    if missing:
        print(f"[warn] Constraint file missing columns: {missing}", file=sys.stderr)

    return canonical[keep].rename(columns=RENAME_MAP)


def annotate_merged(merged_path: str, constraint: pd.DataFrame, overwrite: bool) -> None:
    """Add constraint columns to a single merged.tsv."""
    out_path = merged_path  # in-place by default (overwrite=True)
    if not overwrite:
        base, ext = os.path.splitext(merged_path)
        out_path = base + ".constraint" + ext

    df = pd.read_csv(merged_path, sep="\t", low_memory=False)

    if "genename" not in df.columns:
        print(f"[skip] No 'genename' column in {merged_path}", file=sys.stderr)
        return

    # Drop any previously added constraint columns to avoid duplication
    existing_constraint_cols = [c for c in constraint.columns if c in df.columns and c != "genename"]
    if existing_constraint_cols:
        df = df.drop(columns=existing_constraint_cols)

    merged = df.merge(
        constraint,
        left_on="genename",
        right_on="constraint_gene",
        how="left",
    ).drop(columns=["constraint_gene"])

    n_annotated = merged["lof.pLI"].notna().sum()
    n_total = len(merged)
    print(f"  {os.path.relpath(merged_path)}: {n_annotated}/{n_total} variants annotated")

    merged.to_csv(out_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--constraint",
        default="data/loeuf/gnomad.v4.1.constraint_metrics.tsv",
        help="Path to gnomAD constraint metrics TSV",
    )
    parser.add_argument(
        "--predictions_dir",
        default="results/predictions",
        help="Root directory containing */merged.tsv files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite merged.tsv in place (default: True). "
             "Pass --no-overwrite to write *.constraint.tsv instead.",
    )
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    args = parser.parse_args()

    if not os.path.isfile(args.constraint):
        sys.exit(f"[error] Constraint file not found: {args.constraint}")

    print(f"Loading constraint metrics from {args.constraint} ...")
    constraint = load_constraint(args.constraint)
    print(f"  {len(constraint)} canonical genes loaded.")

    pattern = os.path.join(args.predictions_dir, "*", "merged.tsv")
    merged_files = sorted(glob.glob(pattern))

    if not merged_files:
        sys.exit(f"[error] No merged.tsv files found under {args.predictions_dir}/*/")

    print(f"Found {len(merged_files)} merged.tsv file(s). Annotating ...")
    for path in merged_files:
        annotate_merged(path, constraint, overwrite=args.overwrite)

    print("Done.")


if __name__ == "__main__":
    main()