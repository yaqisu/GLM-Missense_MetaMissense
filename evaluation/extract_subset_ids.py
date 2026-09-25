#!/usr/bin/env python3
"""
Step 0 — Extract variant IDs for a subset of classes into a reusable IDs file.

Reads the per-class seq12k TSVs (before concatenation) and writes a TSV of
variant IDs that can be passed as --subset to any downstream eval script.

The most common use case is extracting Pathogenic + Benign (P+B) IDs so that
evaluation is restricted to the binary classification task, ignoring
likely_pathogenic and likely_benign variants.

Usage:
    # Default: pathogenic + benign (P+B)
    python evaluation/00_extract_subset_ids.py \
        --dataset ClinVar.260923only \
        --datadir data/sequences \
        --outfile data/sequences/ClinVar.260923only.pb_ids.tsv

    # Likely pathogenic + likely benign
    python evaluation/00_extract_subset_ids.py \
        --dataset  ClinVar.260923only \
        --datadir  data/sequences \
        --labels   likely_pathogenic likely_benign \
        --outfile  data/sequences/ClinVar.260923only.plpblb_ids.tsv

    # All four classes (produces an IDs file covering everything)
    python evaluation/00_extract_subset_ids.py \
        --dataset  ClinVar.260923only \
        --datadir  data/sequences \
        --labels   pathogenic likely_pathogenic benign likely_benign \
        --outfile  data/sequences/ClinVar.260923only.all_ids.tsv
"""

import argparse
import glob
import re
from pathlib import Path

import pandas as pd


DEFAULT_PATTERN = "{datadir}/{dataset}.missense.hg38.*.bed.seq12k.tsv"
DEFAULT_LABELS  = ["pathogenic", "benign"]


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument("--dataset",   required=True,
                   help="Dataset name, e.g. ClinVar.260923only")
    p.add_argument("--datadir",   default="data/sequences",
                   help="Directory containing per-class TSV files (default: data/sequences)")
    p.add_argument("--labels",    nargs="+", default=DEFAULT_LABELS,
                   help=f"Class labels to include (matched against filename wildcard). "
                        f"Default: {DEFAULT_LABELS}")
    p.add_argument("--outfile",   required=True, type=Path,
                   help="Output TSV path (columns: variant_id, label, source_label)")
    p.add_argument("--pattern",   default=None,
                   help="Override glob pattern (default uses --datadir and --dataset)")
    p.add_argument("--id_col",    default="variant_id",
                   help="Variant ID column name in input files (default: variant_id)")
    p.add_argument("--label_col", default="label",
                   help="Label column name in input files (default: label)")
    return p.parse_args()


def infer_source_label(filepath: Path, pattern: str) -> str:
    pat_re = re.escape(pattern).replace(r"\*", r"([^/]+)")
    match  = re.search(pat_re, str(filepath))
    return match.group(1) if match else filepath.stem


def main():
    args    = parse_args()
    pattern = args.pattern or DEFAULT_PATTERN.format(
        datadir=args.datadir, dataset=args.dataset)

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    print(f"Dataset : {args.dataset}")
    print(f"Labels  : {args.labels}")
    print(f"Pattern : {pattern}")
    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    all_dfs = []
    for fpath in files:
        src_label = infer_source_label(Path(fpath), pattern)

        if src_label not in args.labels:
            print(f"\n  Skipping '{src_label}' (not in --labels)")
            continue

        print(f"\n  Loading '{src_label}': {fpath}")
        df = pd.read_csv(fpath, sep="\t", low_memory=False)
        print(f"    {len(df):,} rows")

        if args.id_col not in df.columns:
            raise ValueError(f"ID column '{args.id_col}' not found. "
                             f"Available: {list(df.columns)}")

        out = pd.DataFrame()
        out["variant_id"]   = df[args.id_col]
        out["source_label"] = src_label

        if args.label_col in df.columns:
            out["label"] = df[args.label_col]
            print(f"    Label dist: {df[args.label_col].value_counts().to_dict()}")
        else:
            print(f"    WARNING: label column '{args.label_col}' not found — leaving blank")
            out["label"] = None

        all_dfs.append(out)

    if not all_dfs:
        raise RuntimeError("No files loaded. Check --labels match filenames.")

    combined = pd.concat(all_dfs, ignore_index=True)

    n_dupes = combined["variant_id"].duplicated().sum()
    if n_dupes > 0:
        print(f"\nWARNING: {n_dupes:,} duplicate variant_ids across files")

    print(f"\n── Summary ───────────────────────────────────────────────────────")
    print(f"  Total variants : {len(combined):,}")
    for sl, grp in combined.groupby("source_label"):
        print(f"  {sl:30s}: {len(grp):,}")

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.outfile, sep="\t", index=False)
    print(f"\n  Saved to {args.outfile}")


if __name__ == "__main__":
    main()