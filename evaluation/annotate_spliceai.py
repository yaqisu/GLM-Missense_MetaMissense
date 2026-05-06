#!/usr/bin/env python3
"""
add_spliceai_scores.py

Annotates all results/predictions/*/merged.tsv with SpliceAI delta scores,
looked up efficiently via tabix from the raw SNV VCF.

Usage:
    python evaluation/add_spliceai_scores.py \
        --spliceai data/spliceai/spliceai_scores.raw.snv.hg38.vcf.gz \
        --predictions_dir results/predictions \
        [--no-overwrite]

Requires:
    pip install pysam pandas

SpliceAI INFO field format:
    SpliceAI=ALT|GENE|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL

Columns added to each merged.tsv:
    spliceai_DS_AG   delta score acceptor gain
    spliceai_DS_AL   delta score acceptor loss
    spliceai_DS_DG   delta score donor gain
    spliceai_DS_DL   delta score donor loss
    spliceai_DS_max  max of the four delta scores (primary filter metric)
    spliceai_gene    gene symbol from SpliceAI annotation
"""

import argparse
import glob
import os
import sys
from functools import lru_cache

import pandas as pd
import pysam


SPLICEAI_FIELDS = ["DS_AG", "DS_AL", "DS_DG", "DS_DL", "DP_AG", "DP_AL", "DP_DG", "DP_DL"]
ADDED_COLS = ["spliceai_DS_AG", "spliceai_DS_AL", "spliceai_DS_DG", "spliceai_DS_DL",
              "spliceai_DS_max", "spliceai_gene"]


def parse_spliceai_info(info_str: str) -> dict | None:
    """
    Parse the SpliceAI INFO field into a dict.
    Format: SpliceAI=ALT|GENE|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL
    Returns None if field is absent or malformed.
    """
    for field in info_str.split(";"):
        if field.startswith("SpliceAI="):
            payload = field[len("SpliceAI="):]
            parts = payload.split("|")
            if len(parts) < 6:
                return None
            # parts[0] = ALT allele in annotation (may differ from VCF ALT for multiallelic)
            gene = parts[1]
            try:
                ds_ag, ds_al, ds_dg, ds_dl = (float(x) for x in parts[2:6])
            except ValueError:
                return None
            return {
                "spliceai_gene": gene,
                "spliceai_DS_AG": ds_ag,
                "spliceai_DS_AL": ds_al,
                "spliceai_DS_DG": ds_dg,
                "spliceai_DS_DL": ds_dl,
                "spliceai_DS_max": max(ds_ag, ds_al, ds_dg, ds_dl),
            }
    return None


def build_lookup(tbx: pysam.TabixFile, chrom: str, pos: int, ref: str, alt: str) -> dict | None:
    """
    Query tabix for a single variant. Matches on CHROM, POS, REF, ALT.
    VCF is 1-based; tabix fetch uses 0-based half-open intervals.
    """
    # Normalise chromosome name to match VCF (strip 'chr' prefix if present)
    chrom_query = chrom.replace("chr", "")

    try:
        rows = tbx.fetch(chrom_query, pos - 1, pos)
    except ValueError:
        # Chromosome not in index
        return None

    for row in rows:
        fields = row.split("\t")
        if len(fields) < 8:
            continue
        vcf_pos = int(fields[1])
        vcf_ref = fields[3]
        vcf_alt = fields[4]
        if vcf_pos == pos and vcf_ref == ref and vcf_alt == alt:
            return parse_spliceai_info(fields[7])

    return None


def annotate_merged(merged_path: str, tbx: pysam.TabixFile, overwrite: bool) -> None:
    """Add SpliceAI columns to a single merged.tsv."""
    df = pd.read_csv(merged_path, sep="\t", low_memory=False)

    # Determine coordinate columns
    if "chromosome" in df.columns and "position" in df.columns:
        chrom_col, pos_col, ref_col, alt_col = "chromosome", "position", "ref_allele", "alt_allele"
    elif "chr" in df.columns and "pos(1-based)" in df.columns:
        chrom_col, pos_col, ref_col, alt_col = "chr", "pos(1-based)", "ref", "alt"
    else:
        print(f"[skip] Cannot find coordinate columns in {merged_path}", file=sys.stderr)
        return

    # Drop previously added SpliceAI columns to make idempotent
    df = df.drop(columns=[c for c in ADDED_COLS if c in df.columns])

    results = []
    for _, row in df.iterrows():
        chrom = str(row[chrom_col])
        pos   = int(row[pos_col])
        ref   = str(row[ref_col])
        alt   = str(row[alt_col])
        hit = build_lookup(tbx, chrom, pos, ref, alt)
        results.append(hit if hit else {c: None for c in ADDED_COLS})

    spliceai_df = pd.DataFrame(results, index=df.index)
    df = pd.concat([df, spliceai_df], axis=1)

    n_annotated = df["spliceai_DS_max"].notna().sum()
    n_total = len(df)
    print(f"  {os.path.relpath(merged_path)}: {n_annotated}/{n_total} variants annotated")

    out_path = merged_path
    if not overwrite:
        base, ext = os.path.splitext(merged_path)
        out_path = base + ".spliceai" + ext

    df.to_csv(out_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spliceai",
        default="data/spliceai/spliceai_scores.raw.snv.hg38.vcf.gz",
        help="Path to bgzipped + tabix-indexed SpliceAI VCF",
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
             "Pass --no-overwrite to write *.spliceai.tsv instead.",
    )
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.spliceai):
        sys.exit(f"[error] SpliceAI VCF not found: {args.spliceai}")

    tbi_path = args.spliceai + ".tbi"
    if not os.path.isfile(tbi_path):
        sys.exit(
            f"[error] Tabix index not found: {tbi_path}\n"
            f"  Run: tabix -p vcf {args.spliceai}"
        )

    pattern = os.path.join(args.predictions_dir, "*", "merged.tsv")
    merged_files = sorted(glob.glob(pattern))
    if not merged_files:
        sys.exit(f"[error] No merged.tsv files found under {args.predictions_dir}/*/")

    print(f"Opening SpliceAI VCF: {args.spliceai}")
    tbx = pysam.TabixFile(args.spliceai)

    print(f"Found {len(merged_files)} merged.tsv file(s). Annotating ...")
    for path in merged_files:
        annotate_merged(path, tbx, overwrite=args.overwrite)

    tbx.close()
    print("Done.")


if __name__ == "__main__":
    main()