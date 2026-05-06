#!/usr/bin/env python3
"""
add_exon_boundary_info.py

Annotates all results/predictions/*/merged.tsv with exon boundary proximity
metrics, using the pre-computed parquet from prepare_exon_boundaries.py.

Usage:
    python evaluation/add_exon_boundary_info.py \
        --exons data/annotation/exons_GRCh38.113.parquet \
        --predictions_dir results/predictions \
        [--no-overwrite]

Columns added to each merged.tsv:

    exon_boundary_dist      int    distance (bp) to nearest exon boundary
                                   (either start or end of containing exon)
    exon_boundary_5prime    int    distance to 5' splice site of exon
    exon_boundary_3prime    int    distance to 3' splice site of exon
    near_exon_boundary      bool   True if exon_boundary_dist <= 3
                                   (canonical splice site window ±1,2,3)
    exon_boundary_bin       str    "1-3bp", "4-10bp", "11-30bp", ">30bp"
                                   binned distance for stratification

Joins on Ensembl_transcriptid (merged.tsv) → transcript_id (parquet).
Falls back to chromosome+position overlap if transcript match fails.

Requires:
    pip install pandas pyarrow
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


ADDED_COLS = [
    "exon_boundary_dist",
    "exon_boundary_5prime",
    "exon_boundary_3prime",
    "near_exon_boundary",
    "exon_boundary_bin",
]

BOUNDARY_BINS    = [0, 3, 10, 30, np.inf]
BOUNDARY_LABELS  = ["1-3bp", "4-10bp", "11-30bp", ">30bp"]


def load_exons(parquet_path: str) -> pd.DataFrame:
    exons = pd.read_parquet(parquet_path)
    # Build a dict: transcript_id → DataFrame of exon rows for fast lookup
    exons["chromosome"] = exons["chromosome"].astype(str)
    return exons


def build_transcript_index(exons: pd.DataFrame) -> dict:
    """Pre-group exons by transcript_id for O(1) lookup."""
    print("  Building transcript → exon index ...")
    return {tid: grp for tid, grp in exons.groupby("transcript_id")}


def build_chrom_index(exons: pd.DataFrame) -> dict:
    """Pre-group exons by chromosome for fallback positional lookup."""
    print("  Building chromosome → exon index (fallback) ...")
    return {chrom: grp for chrom, grp in exons.groupby("chromosome")}


def compute_distances_for_row(
    chrom: str,
    pos: int,
    transcript_id: str,
    tx_index: dict,
    chrom_index: dict,
) -> dict:
    """
    Compute exon boundary distances for a single variant.
    Tries transcript-level match first, falls back to chromosome-level.
    pos is 1-based genomic coordinate.
    """
    empty = {c: np.nan for c in ADDED_COLS}

    # ── Try transcript-level match ────────────────────────────────────────────
    # Strip version suffix for matching (ENST00000123456.7 → ENST00000123456)
    tx_bare = transcript_id.split(".")[0] if pd.notna(transcript_id) else ""
    exon_rows = None

    for key in [transcript_id, tx_bare]:
        if key in tx_index:
            exon_rows = tx_index[key]
            break

    # ── Fallback: find exons on same chromosome that overlap this position ────
    if exon_rows is None or exon_rows.empty:
        chrom_bare = str(chrom).replace("chr", "")
        if chrom_bare not in chrom_index:
            return empty
        chrom_exons = chrom_index[chrom_bare]
        # Keep only exons that contain this position (1-based pos; exon_start 0-based)
        exon_rows = chrom_exons[
            (chrom_exons["exon_start"] < pos) &   # exon_start 0-based < pos 1-based
            (chrom_exons["exon_end"]   >= pos)
        ]
        if exon_rows.empty:
            return empty

    # ── Compute distances to all boundary positions ───────────────────────────
    dist_5 = (exon_rows["boundary_5prime"] - pos).abs().min()
    dist_3 = (exon_rows["boundary_3prime"] - pos).abs().min()
    dist   = min(dist_5, dist_3)

    bin_label = pd.cut(
        [dist], bins=BOUNDARY_BINS, labels=BOUNDARY_LABELS, right=True
    )[0]

    return {
        "exon_boundary_dist":    int(dist),
        "exon_boundary_5prime":  int(dist_5),
        "exon_boundary_3prime":  int(dist_3),
        "near_exon_boundary":    dist <= 3,
        "exon_boundary_bin":     str(bin_label),
    }


def annotate_merged(
    merged_path: str,
    tx_index: dict,
    chrom_index: dict,
    overwrite: bool,
) -> None:
    df = pd.read_csv(merged_path, sep="\t", low_memory=False)

    # Detect coordinate columns
    if "chromosome" in df.columns and "position" in df.columns:
        chrom_col, pos_col = "chromosome", "position"
    elif "chr" in df.columns and "pos(1-based)" in df.columns:
        chrom_col, pos_col = "chr", "pos(1-based)"
    else:
        print(f"[skip] Cannot find coordinate columns in {merged_path}", file=sys.stderr)
        return

    tx_col = "Ensembl_transcriptid" if "Ensembl_transcriptid" in df.columns else None

    # Drop previously added columns (idempotent)
    df = df.drop(columns=[c for c in ADDED_COLS if c in df.columns])

    results = []
    for _, row in df.iterrows():
        chrom = str(row[chrom_col]).replace("chr", "")
        pos   = int(row[pos_col])
        tx    = str(row[tx_col]) if tx_col else ""
        results.append(compute_distances_for_row(chrom, pos, tx, tx_index, chrom_index))

    annot_df = pd.DataFrame(results, index=df.index)
    df = pd.concat([df, annot_df], axis=1)

    n_annotated = df["exon_boundary_dist"].notna().sum()
    n_total     = len(df)
    print(f"  {os.path.relpath(merged_path)}: {n_annotated}/{n_total} variants annotated")

    out_path = merged_path
    if not overwrite:
        base, ext = os.path.splitext(merged_path)
        out_path  = base + ".exonboundary" + ext

    df.to_csv(out_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exons",
        default="data/annotation/exons_GRCh38.113.parquet",
        help="Path to pre-computed exon boundaries parquet",
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
        help="Overwrite merged.tsv in place (default). "
             "Pass --no-overwrite to write *.exonboundary.tsv instead.",
    )
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    args = parser.parse_args()

    if not os.path.isfile(args.exons):
        sys.exit(
            f"[error] Exon parquet not found: {args.exons}\n"
            f"  Run: python evaluation/prepare_exon_boundaries.py first."
        )

    pattern      = os.path.join(args.predictions_dir, "*", "merged.tsv")
    merged_files = sorted(glob.glob(pattern))
    if not merged_files:
        sys.exit(f"[error] No merged.tsv files found under {args.predictions_dir}/*/")

    print(f"Loading exon boundaries from {args.exons} ...")
    exons       = load_exons(args.exons)
    tx_index    = build_transcript_index(exons)
    chrom_index = build_chrom_index(exons)
    print(f"  {len(exons):,} exon records | {len(tx_index):,} transcripts")

    print(f"\nFound {len(merged_files)} merged.tsv file(s). Annotating ...")
    for path in merged_files:
        annotate_merged(path, tx_index, chrom_index, overwrite=args.overwrite)

    print("Done.")


if __name__ == "__main__":
    main()