#!/usr/bin/env python3
"""
Step 2 — Annotate variants with ALL dbNSFP scores via tabix.

Usage:
    python evaluation/02_annotate_dbnsfp.py \
        --variants data/sequences/ClinVar.260309only.seq12k.pb.tsv \
        --dbnsfp   data/dbnsfp/dbNSFP5.3.1a_grch38.gz \
        --outdir   results/predictions/ClinVar.260309only.seq12k.pb

Downloads (if you don't have dbNSFP yet):
    curl --http1.1 -C - -o data/dbnsfp/dbNSFP5.3.1a_grch38.gz \\
         https://dist.genos.us/academic/yourcode/dbNSFP5.3.1a_grch38.gz
    curl --http1.1 -C - -o data/dbnsfp/dbNSFP5.3.1a_grch38.gz.tbi \\
         https://dist.genos.us/academic/yourcode/dbNSFP5.3.1a_grch38.gz.tbi

Notes on semicolon-separated fields:
    dbNSFP stores one value per isoform, semicolon-separated, in parallel across
    all columns (e.g. VEP_canonical="YES;NO;NO", SIFT_score="0.02;0.15;0.08").
    This script resolves each field to the single value corresponding to the
    canonical isoform (VEP_canonical == "YES").  If no canonical isoform is
    flagged, it falls back to the first value.  Fields without semicolons are
    returned as-is.
"""

import argparse
import subprocess
from pathlib import Path

import pandas as pd


REQUIRED_COLS = ["chromosome", "position", "ref_allele", "alt_allele"]
OPTIONAL_COLS = ["variant_id", "label"]

# Column name used by dbNSFP 4.x / 5.x to flag the VEP canonical transcript.
VEP_CANONICAL_COL = "VEP_canonical"


def parse_args():
    p = argparse.ArgumentParser(description="Annotate variants with dbNSFP via tabix.")
    p.add_argument("--variants", required=True, type=Path,
                   help="Input variants TSV (columns: variant_id, chromosome, position, "
                        "ref_allele, alt_allele, label)")
    p.add_argument("--dbnsfp",   required=True, type=Path,
                   help="Path to dbNSFP bgzipped file (e.g. dbNSFP5.3.1a_grch38.gz)")
    p.add_argument("--outdir",   required=True, type=Path,
                   help="Output directory")
    return p.parse_args()


def read_dbnsfp_header(dbnsfp_gz: Path) -> list[str]:
    result = subprocess.run(
        f"zcat {dbnsfp_gz} | head -n1",
        shell=True, capture_output=True, text=True
    )
    return result.stdout.strip().lstrip("#").split("\t")


def query_tabix(dbnsfp_gz: Path, chrom: str, pos: int,
                ref: str, alt: str) -> list[str] | None:
    proc = subprocess.Popen(
        ["tabix", str(dbnsfp_gz), f"{chrom}:{pos}-{pos}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    matched = None
    for raw_line in proc.stdout:
        line   = raw_line.decode("utf-8").rstrip("\n").rstrip("\r")
        fields = line.split("\t")
        if len(fields) > 3 and fields[2] == ref and fields[3] == alt:
            matched = fields
            break
    proc.stdout.close()
    proc.wait()
    return matched


def resolve_canonical_index(fields: list[str], vep_canonical_idx: int | None) -> int:
    """
    Given a matched dbNSFP row, return the isoform index corresponding to the
    VEP canonical transcript (VEP_canonical == "YES").

    Falls back to index 0 if:
      - VEP_canonical column is absent from the header
      - no isoform is flagged as canonical
      - the canonical field itself has no semicolons (single isoform)
    """
    if vep_canonical_idx is None or vep_canonical_idx >= len(fields):
        return 0
    canonical_str = fields[vep_canonical_idx]
    if ";" not in canonical_str:
        return 0
    parts = canonical_str.split(";")
    for i, v in enumerate(parts):
        if v.strip().upper() == "YES":
            return i
    return 0  # fallback: no YES found


def extract_field(raw_value: str, canonical_idx: int) -> str:
    """
    Extract the value at `canonical_idx` from a semicolon-separated string.
    Returns the whole string unchanged if it contains no semicolons.
    Returns "." if the index is out of range.
    """
    if ";" not in raw_value:
        return raw_value
    parts = raw_value.split(";")
    if canonical_idx < len(parts):
        return parts[canonical_idx]
    return parts[0]  # index out of range — fall back to first


def main():
    args    = parse_args()
    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = out_dir / "dbnsfp.tsv"

    # ── Load variants ──────────────────────────────────────────────────────
    print(f"Loading variants from {args.variants}")
    all_cols = pd.read_csv(args.variants, sep="\t", nrows=0).columns.tolist()
    usecols  = REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in all_cols]
    variants = pd.read_csv(args.variants, sep="\t", usecols=usecols)

    # Auto-generate variant_id if not present in input
    if "variant_id" not in variants.columns:
        variants["variant_id"] = (
            variants["chromosome"].astype(str) + ":" +
            variants["position"].astype(str)   + ":" +
            variants["ref_allele"].astype(str) + ">" +
            variants["alt_allele"].astype(str)
        )
        print(f"  'variant_id' column not found — auto-generated as chrom:pos:ref>alt")

    has_label = "label" in variants.columns
    if not has_label:
        print(f"  'label' column not found — will be omitted from output")

    print(f"  {len(variants):,} variants loaded")

    # ── Read dbNSFP header ─────────────────────────────────────────────────
    print(f"Reading dbNSFP header from {args.dbnsfp} ...")
    header  = read_dbnsfp_header(args.dbnsfp)
    col_idx = {name: i for i, name in enumerate(header)}
    print(f"  {len(header)} columns")

    # Index of VEP_canonical within a tabix row — used for isoform resolution
    vep_canonical_idx = col_idx.get(VEP_CANONICAL_COL)
    if vep_canonical_idx is not None:
        print(f"  VEP_canonical column found at index {vep_canonical_idx} "
              f"— will resolve semicolon-separated fields to canonical isoform")
    else:
        print(f"  WARNING: '{VEP_CANONICAL_COL}' not found in header "
              f"— will fall back to first semicolon value for all fields")

    # ── Query tabix per variant ────────────────────────────────────────────
    print("Querying dbNSFP via tabix...")
    rows        = []
    found_count = 0
    not_found   = 0

    for i, var in variants.iterrows():
        chrom = str(var["chromosome"])
        pos   = int(var["position"])
        ref   = str(var["ref_allele"])
        alt   = str(var["alt_allele"])

        matched = query_tabix(args.dbnsfp, chrom, pos, ref, alt)

        record = {
            "variant_id": var["variant_id"],
            "chromosome": chrom,
            "position":   pos,
            "ref_allele": ref,
            "alt_allele": alt,
        }
        if has_label:
            record["label"] = var["label"]

        if matched:
            # Determine which isoform index corresponds to the canonical transcript
            canon_idx = resolve_canonical_index(matched, vep_canonical_idx)

            for col_name, idx in col_idx.items():
                if col_name not in record:
                    raw = matched[idx] if idx < len(matched) else "."
                    record[col_name] = extract_field(raw, canon_idx)
            found_count += 1
        else:
            not_found += 1
            for col_name in header:
                if col_name not in record:
                    record[col_name] = "."

        rows.append(record)

        n_done = i + 1
        if n_done % 1000 == 0:
            pct      = 100 * n_done / len(variants)
            hit_rate = 100 * found_count / n_done
            print(f"  [{n_done:>6,} / {len(variants):,}  {pct:5.1f}%]  "
                  f"found: {found_count:,}  not found: {not_found:,}  "
                  f"({hit_rate:.1f}% hit rate)")

    print(f"\nDone. {found_count:,}/{len(rows):,} variants found in dbNSFP")
    print(f"  {not_found:,} not found (intronic / intergenic / synonymous)")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nSaved to {out_tsv}")
    print(f"Output shape: {out_df.shape[0]:,} rows × {out_df.shape[1]} columns")


if __name__ == "__main__":
    main()