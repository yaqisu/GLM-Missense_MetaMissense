#!/usr/bin/env python3
"""
prepare_glm_input.py

Prepare the seq12k input TSV required by GLM-Missense.py, starting from a
simple variant table.

Required input columns (tab-separated, any order):
    chromosome   position   ref_allele   alt_allele

Optional columns passed through if present:
    variant_id   — unique ID; auto-generated as {chrom}:{pos}:{ref}>{alt} if absent
    label        — integer class label (0 = benign, 1 = pathogenic)

All other columns in the input are ignored.

Output columns:
    variant_id   chromosome   position   ref_allele   alt_allele
    upstream_flank   downstream_flank   ref_sequence   alt_sequence
    label   (only if label column was present in input)

Usage:
    python scoring/prepare_glm_input.py \\
        --input   my_variants.tsv \\
        --output  results/scoring/my_variants.seq12k.tsv \\
        --genome  data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa

Run from the repo root.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta


FLANK      = 5999   # bp on each side → 11,999 bp total (seq12k)
REQUIRED   = {"chromosome", "position", "ref_allele", "alt_allele"}
DEFAULT_GENOME = "data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa"


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert a variant TSV to seq12k input for GLM-Missense.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",  "-i", required=True,  type=Path,
                   help="Input variant TSV")
    p.add_argument("--output", "-o", required=True,  type=Path,
                   help="Output seq12k TSV path")
    p.add_argument("--genome", "-g", default=DEFAULT_GENOME, type=Path,
                   help=f"GRCh38 reference FASTA (default: {DEFAULT_GENOME})")
    return p.parse_args()


def load_variants(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = df.columns.str.strip()

    missing = REQUIRED - set(df.columns)
    if missing:
        sys.exit(
            f"ERROR: Input TSV is missing required columns: {sorted(missing)}\n"
            f"       Found: {list(df.columns)}"
        )

    # Normalise chromosome: strip 'chr' prefix to match Ensembl reference
    df["chromosome"] = df["chromosome"].str.replace(r"^chr", "", regex=True).str.strip()
    df["position"]   = df["position"].astype(int)

    # Auto-generate variant_id if absent
    if "variant_id" not in df.columns:
        df["variant_id"] = (
            df["chromosome"] + ":" + df["position"].astype(str) + ":" +
            df["ref_allele"] + ">" + df["alt_allele"]
        )

    return df


def extract_sequences(df: pd.DataFrame, genome: Fasta) -> pd.DataFrame:
    rows = []
    n_mismatch = 0

    for _, var in df.iterrows():
        chrom    = str(var["chromosome"])
        pos      = int(var["position"])   # 1-based
        ref      = str(var["ref_allele"]).upper()
        alt      = str(var["alt_allele"]).upper()
        vid      = str(var["variant_id"])

        if chrom not in genome:
            print(f"  WARNING: chromosome '{chrom}' not in genome — skipping {vid}")
            continue

        chrom_len = len(genome[chrom])
        var_0     = pos - 1                           # convert to 0-based for pyfaidx
        seq_start = max(0, var_0 - FLANK)
        seq_end   = min(chrom_len, var_0 + FLANK + 1)

        ref_seq = str(genome[chrom][seq_start:seq_end]).upper()
        offset  = var_0 - seq_start                   # position of variant within extracted window

        # Verify reference allele
        if offset < len(ref_seq) and ref_seq[offset] != ref:
            n_mismatch += 1
            print(f"  WARNING: ref mismatch at {chrom}:{pos} "
                  f"— expected {ref}, found {ref_seq[offset]} — {vid}")

        alt_seq        = ref_seq[:offset] + alt + ref_seq[offset + 1:]
        upstream_flank = offset
        downstream_flank = len(ref_seq) - offset - 1

        row = {
            "variant_id":       vid,
            "chromosome":       chrom,
            "position":         pos,
            "ref_allele":       ref,
            "alt_allele":       alt,
            "upstream_flank":   upstream_flank,
            "downstream_flank": downstream_flank,
            "ref_sequence":     ref_seq,
            "alt_sequence":     alt_seq,
        }
        if "label" in var.index:
            row["label"] = var["label"]

        rows.append(row)

    if n_mismatch:
        print(f"\n  {n_mismatch} reference allele mismatches (see warnings above)")

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    for path, flag in [(args.input, "--input"), (args.genome, "--genome")]:
        if not path.exists():
            sys.exit(f"ERROR: {flag} not found: {path}")

    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print(f"Genome : {args.genome}")
    print(f"Flank  : {FLANK} bp each side ({FLANK * 2 + 1} bp total)\n")

    print("Loading variants ...")
    df = load_variants(args.input)
    print(f"  {len(df):,} variants | "
          f"label column: {'yes' if 'label' in df.columns else 'no'}")

    print("Loading reference genome ...")
    genome = Fasta(str(args.genome))

    print("Extracting sequences ...")
    out_df = extract_sequences(df, genome)
    print(f"  {len(out_df):,} variants written")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, sep="\t", index=False)

    print(f"\nSaved → {args.output}")
    print(f"\nNext step:")
    print(f"  python scoring/GLM-Missense.py \\")
    print(f"      --input  {args.output} \\")
    print(f"      --model  scoring/GLM-Missense.pt \\")
    print(f"      --output results/scoring/GLM-Missense.tsv")


if __name__ == "__main__":
    main()