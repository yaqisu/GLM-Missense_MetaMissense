#!/usr/bin/env python3
"""
prepare_exon_boundaries.py

Parses an Ensembl GTF file and extracts exon boundary coordinates,
caching the result as a parquet file for fast downstream lookups.

Usage:
    python evaluation/prepare_exon_boundaries.py \
        --gtf data/annotation/Homo_sapiens.GRCh38.113.gtf.gz \
        --output data/annotation/exons_GRCh38.113.parquet

Output columns:
    chromosome      str   e.g. "1", "X" (no "chr" prefix)
    transcript_id   str   Ensembl transcript ID e.g. ENST00000123456.7
    exon_number     int   exon rank within transcript (1-based)
    exon_start      int   0-based start (GTF convention)
    exon_end        int   0-based end (exclusive)
    strand          str   "+" or "-"
    gene_id         str   Ensembl gene ID
    gene_name       str   HGNC gene symbol (if available)

Requires:
    pip install pyranges pandas pyarrow
"""

import argparse
import os
import sys
import time

import pandas as pd


def parse_gtf(gtf_path: str) -> pd.DataFrame:
    """
    Load GTF and extract exon rows with relevant attributes.
    Uses pyranges for robust GTF parsing.
    """
    try:
        import pyranges as pr
    except ImportError:
        sys.exit("[error] pyranges not installed. Run: pip install pyranges")

    print(f"  Reading GTF: {gtf_path}")
    print("  (This may take 30-60 seconds for a full genome GTF...)")
    t0 = time.time()
    gtf = pr.read_gtf(gtf_path, as_df=True)
    print(f"  Loaded in {time.time() - t0:.1f}s — {len(gtf):,} total records")

    exons = gtf[gtf["Feature"] == "exon"].copy()
    print(f"  {len(exons):,} exon records found")

    # Standardize chromosome names — strip "chr" prefix if present
    exons["Chromosome"] = exons["Chromosome"].astype(str).str.replace("^chr", "", regex=True)

    # Select and rename columns
    keep = {
        "Chromosome":    "chromosome",
        "Start":         "exon_start",    # 0-based
        "End":           "exon_end",      # exclusive
        "Strand":        "strand",
        "transcript_id": "transcript_id",
        "exon_number":   "exon_number",
        "gene_id":       "gene_id",
        "gene_name":     "gene_name",
    }
    available = {k: v for k, v in keep.items() if k in exons.columns}
    exons = exons[list(available.keys())].rename(columns=available)

    # exon_number may be missing in some GTFs — fill with rank within transcript
    if "exon_number" not in exons.columns:
        exons["exon_number"] = (
            exons.groupby("transcript_id")["exon_start"].rank(method="first").astype(int)
        )
    else:
        exons["exon_number"] = pd.to_numeric(exons["exon_number"], errors="coerce").fillna(0).astype(int)

    # Ensure correct dtypes
    exons["exon_start"] = exons["exon_start"].astype(int)
    exons["exon_end"]   = exons["exon_end"].astype(int)
    exons = exons.drop_duplicates().reset_index(drop=True)

    return exons


def add_boundary_positions(exons: pd.DataFrame) -> pd.DataFrame:
    """
    Add explicit boundary position columns for fast distance computation:
        boundary_5prime   genomic position of the 5' splice site of this exon
        boundary_3prime   genomic position of the 3' splice site of this exon

    For + strand: 5' boundary = exon_start, 3' boundary = exon_end - 1
    For - strand: 5' boundary = exon_end - 1, 3' boundary = exon_start
    Both are 1-based genomic positions (convert from 0-based GTF coords).
    """
    exons = exons.copy()

    plus  = exons["strand"] == "+"
    minus = exons["strand"] == "-"

    exons["boundary_5prime"] = 0
    exons["boundary_3prime"] = 0

    # + strand: exon runs left→right; start = 5' acceptor side, end = 3' donor side
    exons.loc[plus, "boundary_5prime"] = exons.loc[plus, "exon_start"] + 1  # to 1-based
    exons.loc[plus, "boundary_3prime"] = exons.loc[plus, "exon_end"]        # end is exclusive → last base

    # - strand: exon runs right→left
    exons.loc[minus, "boundary_5prime"] = exons.loc[minus, "exon_end"]
    exons.loc[minus, "boundary_3prime"] = exons.loc[minus, "exon_start"] + 1

    return exons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gtf",
        default="data/annotation/Homo_sapiens.GRCh38.113.gtf.gz",
        help="Path to Ensembl GTF (plain or bgzipped)",
    )
    parser.add_argument(
        "--output",
        default="data/annotation/exons_GRCh38.113.parquet",
        help="Output parquet path",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.gtf):
        sys.exit(f"[error] GTF not found: {args.gtf}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("Parsing GTF ...")
    exons = parse_gtf(args.gtf)

    print("Adding boundary position columns ...")
    exons = add_boundary_positions(exons)

    print(f"Writing parquet: {args.output}")
    exons.to_parquet(args.output, index=False)
    print(f"  Done — {len(exons):,} exon records saved.")
    print(f"\nUnique transcripts : {exons['transcript_id'].nunique():,}")
    print(f"Unique genes       : {exons['gene_id'].nunique():,}")
    print(f"Chromosomes        : {sorted(exons['chromosome'].unique())}")


if __name__ == "__main__":
    main()