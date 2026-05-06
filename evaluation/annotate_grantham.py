#!/usr/bin/env python3
"""
annotate_grantham.py
Adds Grantham distance and radical amino-acid change annotation to merged.tsv files.

New columns added:
  grantham_distance    : float, Grantham (1974) physicochemical distance between
                         ref and alt amino acid (0–215 scale). NaN for stop/invalid/same AA.
  is_radical_aa_change : bool, True if grantham_distance > radical_threshold (default 150).

The 'Fraction of radical AA change' summary statistic is computed downstream
as mean(is_radical_aa_change) over any desired subset — this script only adds
the per-variant annotations.

Usage:
    # Annotate all merged.tsv files in place (default)
    python evaluation/annotate_grantham.py \
        --predictions_dir results/predictions

    # Write separate .annotated.tsv files instead of overwriting
    python evaluation/annotate_grantham.py \
        --predictions_dir results/predictions \
        --no-overwrite

    # Single file
    python evaluation/annotate_grantham.py \
        --input results/predictions/ClinVar_260309only.testset/merged.tsv

    # Custom radical threshold
    python evaluation/annotate_grantham.py \
        --predictions_dir results/predictions \
        --radical_threshold 100.0
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ── Suppress GIL warnings ────────────────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*global interpreter lock.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*GIL.*", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Annotate variant table(s) with Grantham distance and radical AA change flag."
    )
    p.add_argument(
        "--input", default=None,
        help="Path to a single merged.tsv (mutually exclusive with --predictions_dir)",
    )
    p.add_argument(
        "--predictions_dir", default=None,
        help="Directory to scan for */merged.tsv files (mutually exclusive with --input)",
    )
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Overwrite input file in place (default: True). "
            "Pass --no-overwrite to write <stem>.annotated.tsv instead."
        ),
    )
    p.add_argument(
        "--output", default=None,
        help=(
            "Output path when --no-overwrite and --input are both set. "
            "Ignored when --predictions_dir is used."
        ),
    )
    p.add_argument(
        "--radical_threshold", type=float, default=150.0,
        help=(
            "Grantham distance cutoff for classifying a missense as radical (default: 150.0). "
            "Follows the convention in Grantham (1974) Science 185:862-864."
        ),
    )
    return p.parse_args()


def resolve_output_path(input_path: Path, overwrite: bool, output_arg: str = None) -> Path:
    if overwrite:
        return input_path
    if output_arg:
        return Path(output_arg)
    return input_path.parent / f"{input_path.stem}.annotated.tsv"


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

_ANNOTATION_COLS = [
    'grantham_distance',
    'is_radical_aa_change',
]


def _drop_existing_annotation_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove annotation columns from a previous run to avoid duplicates on re-runs."""
    existing = [c for c in _ANNOTATION_COLS if c in df.columns]
    if existing:
        print(f"  Dropping {len(existing)} existing annotation column(s) from previous run: {existing}")
        df = df.drop(columns=existing)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# GRANTHAM DISTANCE
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_AAS = set("ARNDCQEGHILKMFPSTWYV")

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

_GRANTHAM_RAW = {
    ("A", "C"): 195, ("A", "D"): 126, ("A", "E"): 107, ("A", "F"): 113, ("A", "G"): 60,
    ("A", "H"): 86,  ("A", "I"): 94,  ("A", "K"): 106, ("A", "L"): 96,  ("A", "M"): 84,
    ("A", "N"): 111, ("A", "P"): 27,  ("A", "Q"): 91,  ("A", "R"): 112, ("A", "S"): 99,
    ("A", "T"): 58,  ("A", "V"): 64,  ("A", "W"): 148, ("A", "Y"): 112,
    ("C", "D"): 154, ("C", "E"): 170, ("C", "F"): 205, ("C", "G"): 159, ("C", "H"): 174,
    ("C", "I"): 198, ("C", "K"): 202, ("C", "L"): 198, ("C", "M"): 196, ("C", "N"): 139,
    ("C", "P"): 169, ("C", "Q"): 154, ("C", "R"): 180, ("C", "S"): 112, ("C", "T"): 149,
    ("C", "V"): 192, ("C", "W"): 215, ("C", "Y"): 194,
    ("D", "E"): 45,  ("D", "F"): 177, ("D", "G"): 94,  ("D", "H"): 81,  ("D", "I"): 168,
    ("D", "K"): 101, ("D", "L"): 172, ("D", "M"): 160, ("D", "N"): 23,  ("D", "P"): 108,
    ("D", "Q"): 61,  ("D", "R"): 96,  ("D", "S"): 65,  ("D", "T"): 85,  ("D", "V"): 152,
    ("D", "W"): 181, ("D", "Y"): 160,
    ("E", "F"): 140, ("E", "G"): 98,  ("E", "H"): 40,  ("E", "I"): 134, ("E", "K"): 56,
    ("E", "L"): 138, ("E", "M"): 126, ("E", "N"): 42,  ("E", "P"): 93,  ("E", "Q"): 29,
    ("E", "R"): 54,  ("E", "S"): 80,  ("E", "T"): 65,  ("E", "V"): 121, ("E", "W"): 152,
    ("E", "Y"): 122,
    ("F", "G"): 153, ("F", "H"): 100, ("F", "I"): 21,  ("F", "K"): 102, ("F", "L"): 22,
    ("F", "M"): 28,  ("F", "N"): 158, ("F", "P"): 114, ("F", "Q"): 116, ("F", "R"): 97,
    ("F", "S"): 155, ("F", "T"): 103, ("F", "V"): 50,  ("F", "W"): 40,  ("F", "Y"): 22,
    ("G", "H"): 98,  ("G", "I"): 135, ("G", "K"): 127, ("G", "L"): 138, ("G", "M"): 127,
    ("G", "N"): 80,  ("G", "P"): 42,  ("G", "Q"): 87,  ("G", "R"): 125, ("G", "S"): 56,
    ("G", "T"): 59,  ("G", "V"): 109, ("G", "W"): 184, ("G", "Y"): 147,
    ("H", "I"): 94,  ("H", "K"): 32,  ("H", "L"): 99,  ("H", "M"): 87,  ("H", "N"): 68,
    ("H", "P"): 77,  ("H", "Q"): 24,  ("H", "R"): 29,  ("H", "S"): 89,  ("H", "T"): 47,
    ("H", "V"): 84,  ("H", "W"): 115, ("H", "Y"): 83,
    ("I", "K"): 102, ("I", "L"): 5,   ("I", "M"): 10,  ("I", "N"): 149, ("I", "P"): 95,
    ("I", "Q"): 109, ("I", "R"): 97,  ("I", "S"): 142, ("I", "T"): 89,  ("I", "V"): 29,
    ("I", "W"): 61,  ("I", "Y"): 33,
    ("K", "L"): 107, ("K", "M"): 95,  ("K", "N"): 94,  ("K", "P"): 103, ("K", "Q"): 53,
    ("K", "R"): 26,  ("K", "S"): 121, ("K", "T"): 78,  ("K", "V"): 97,  ("K", "W"): 110,
    ("K", "Y"): 85,
    ("L", "M"): 15,  ("L", "N"): 153, ("L", "P"): 98,  ("L", "Q"): 113, ("L", "R"): 102,
    ("L", "S"): 145, ("L", "T"): 92,  ("L", "V"): 32,  ("L", "W"): 61,  ("L", "Y"): 36,
    ("M", "N"): 142, ("M", "P"): 87,  ("M", "Q"): 101, ("M", "R"): 91,  ("M", "S"): 135,
    ("M", "T"): 81,  ("M", "V"): 21,  ("M", "W"): 67,  ("M", "Y"): 36,
    ("N", "P"): 91,  ("N", "Q"): 46,  ("N", "R"): 86,  ("N", "S"): 46,  ("N", "T"): 65,
    ("N", "V"): 133, ("N", "W"): 174, ("N", "Y"): 143,
    ("P", "Q"): 76,  ("P", "R"): 103, ("P", "S"): 74,  ("P", "T"): 38,  ("P", "V"): 68,
    ("P", "W"): 147, ("P", "Y"): 110,
    ("Q", "R"): 43,  ("Q", "S"): 68,  ("Q", "T"): 42,  ("Q", "V"): 96,  ("Q", "W"): 130,
    ("Q", "Y"): 99,
    ("R", "S"): 110, ("R", "T"): 71,  ("R", "V"): 96,  ("R", "W"): 101, ("R", "Y"): 77,
    ("S", "T"): 58,  ("S", "V"): 124, ("S", "W"): 177, ("S", "Y"): 144,
    ("T", "V"): 69,  ("T", "W"): 128, ("T", "Y"): 92,
    ("V", "W"): 88,  ("V", "Y"): 55,
    ("W", "Y"): 37,
}


def _normalize_aa(x) -> Optional[str]:
    """Convert 1-letter or 3-letter AA code to standard 1-letter. Returns None for stops/invalid."""
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if not s or s in {"*", "X", "TER", "STOP", "NA", "NAN", ".", "-"}:
        return None
    if len(s) == 1 and s in STANDARD_AAS:
        return s
    if len(s) == 3 and s in AA3_TO_1:
        return AA3_TO_1[s]
    return None


def _grantham_distance(aa1, aa2) -> float:
    """
    Symmetric Grantham distance lookup.
    Returns 0.0 for same AA, NaN if either AA is invalid or pair not found.
    """
    a = _normalize_aa(aa1)
    b = _normalize_aa(aa2)
    if a is None or b is None:
        return np.nan
    if a == b:
        return 0.0
    return float(_GRANTHAM_RAW.get((a, b), _GRANTHAM_RAW.get((b, a), np.nan)))


def compute_grantham(df: pd.DataFrame, radical_threshold: float = 150.0) -> pd.DataFrame:
    """
    Add Grantham distance and radical AA change flag to df.

    Columns added:
      grantham_distance    : float [0–215], NaN for stop codons / invalid / same AA
      is_radical_aa_change : bool, True if grantham_distance > radical_threshold

    radical_threshold default of 150 follows the convention in the companion
    plotting script (plot_glm_missense_le2_four_panel_main_radical_aa.py).
    """
    print(f"Computing Grantham distance (radical threshold = {radical_threshold})...")
    df = df.copy()

    distances = df.apply(
        lambda r: _grantham_distance(r['aaref'], r['aaalt']),
        axis=1,
    )

    df['grantham_distance']    = distances
    df['is_radical_aa_change'] = (distances > radical_threshold).fillna(False)

    n_total   = len(df)
    n_valid   = int((distances > 0).sum())       # real missense (distance > 0)
    n_same    = int((distances == 0).sum())       # same AA (synonymous in this table = 0)
    n_nan     = int(distances.isna().sum())       # stop / invalid
    n_radical = int(df['is_radical_aa_change'].sum())

    print(f"  Total variants          : {n_total}")
    print(f"  Valid missense AA pairs : {n_valid}")
    print(f"  Same AA (dist=0)        : {n_same}")
    print(f"  Invalid/stop/NaN        : {n_nan}")
    if n_valid > 0:
        print(f"  Radical (>{radical_threshold})   : {n_radical} / {n_valid} "
              f"({100 * n_radical / n_valid:.1f}% of valid missense)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PER-FILE LOGIC
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_COLS = [
    'grantham_distance',
    'is_radical_aa_change',
]


def annotate_one_file(input_path: Path, output_path: Path, args) -> None:
    """Load, annotate, and save a single merged.tsv."""
    print(f"\n{'='*60}")
    print(f"Processing : {input_path}")
    print(f"Output     : {output_path}")
    print(f"{'='*60}")

    df = pd.read_csv(input_path, sep='\t', low_memory=False)
    print(f"  {len(df)} variants loaded")

    # Drop existing columns to avoid duplicates on re-runs
    df = _drop_existing_annotation_cols(df)

    df = compute_grantham(df, args.radical_threshold)

    print(f"\nWriting to {output_path}...")
    if args.overwrite:
        print("  (overwriting input file in place)")
    df.to_csv(output_path, sep='\t', index=False)
    print("Done.")

    print("\nAnnotation summary:")
    for col in SUMMARY_COLS:
        if col in df.columns:
            print(f"  {col}: {df[col].notna().sum()} non-null values")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Validate mutually exclusive input args ────────────────────────────────
    if args.input and args.predictions_dir:
        raise ValueError("Specify either --input or --predictions_dir, not both.")
    if not args.input and not args.predictions_dir:
        raise ValueError("Must specify one of --input or --predictions_dir.")

    # ── Collect input files ───────────────────────────────────────────────────
    if args.input:
        input_files = [Path(args.input)]
    else:
        preds_dir   = Path(args.predictions_dir)
        input_files = sorted(preds_dir.glob("*/merged.tsv"))
        if not input_files:
            raise FileNotFoundError(
                f"No merged.tsv files found under {preds_dir}/*/merged.tsv"
            )
        print(f"Found {len(input_files)} merged.tsv file(s) under {preds_dir}:")
        for f in input_files:
            print(f"  {f}")

    # ── Process each file ─────────────────────────────────────────────────────
    failed = []
    for input_path in input_files:
        output_arg  = args.output if args.input else None
        output_path = resolve_output_path(input_path, args.overwrite, output_arg)
        try:
            annotate_one_file(input_path, output_path, args)
        except Exception as e:
            print(f"\nERROR processing {input_path}: {e}")
            failed.append((input_path, e))

    # ── Final summary ─────────────────────────────────────────────────────────
    total = len(input_files)
    print(f"\n{'='*60}")
    print(f"Finished. {total - len(failed)}/{total} file(s) annotated successfully.")
    if failed:
        print("Failed files:")
        for path, err in failed:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()