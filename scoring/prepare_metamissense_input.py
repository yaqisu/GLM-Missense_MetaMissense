#!/usr/bin/env python3
"""
prepare_metamissense_input.py

Prepare the input table required by MetaMissense.py by:
  1. Annotating variants with ALL dbNSFP scores via tabix (calls annotate_dbnsfp.py)
  2. Merging the resulting dbnsfp.tsv with the GLM-Missense score file

Both steps take the GLM-Missense score TSV as the single starting point.
The merged output contains all dbNSFP columns plus the GLM-Missense_score,
which MetaMissense.py expects as 'finetune_NT2_score' (aliased automatically).

Usage:
    python scoring/prepare_metamissense_input.py \\
        --glm     results/scoring/GLM-Missense.tsv \\
        --dbnsfp  data/dbnsfp/dbNSFP5.3.1a_grch38.gz \\
        --outdir  results/scoring

Output files written to --outdir:
    dbnsfp.tsv              all dbNSFP annotations (from annotate_dbnsfp.py)
    MetaMissense_input.tsv  merged GLM-Missense + dbNSFP columns

The MetaMissense model only uses six columns from dbNSFP:
    AlphaMissense_score, ESM1b_score, REVEL_score,
    CADD_phred, SIFT_score, Polyphen2_HVAR_score

All other dbNSFP columns are retained in the output so you can use them
for your own downstream analyses without re-running the annotation step.

Prerequisites:
    tabix must be installed and on PATH.
    pip install pandas

Notes:
    - annotate_dbnsfp.py is called as a subprocess and must be present at
      scoring/annotate_dbnsfp.py (run this script from the repo root).
    - If dbnsfp.tsv already exists in --outdir it will NOT be re-run.
      Delete it first or pass --force to re-run.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


KEY_COLS = ["chromosome", "position", "ref_allele", "alt_allele"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare MetaMissense input: annotate with dbNSFP and merge GLM-Missense scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--glm",    "-g", required=True, type=Path,
                   help="GLM-Missense score TSV (output of GLM-Missense.py)")
    p.add_argument("--dbnsfp", "-d", required=True, type=Path,
                   help="Path to dbNSFP bgzipped file (e.g. dbNSFP5.3.1a_grch38.gz)")
    p.add_argument("--outdir", "-o", required=True, type=Path,
                   help="Output directory for dbnsfp.tsv and MetaMissense_input.tsv")
    p.add_argument("--force", action="store_true",
                   help="Re-run dbNSFP annotation even if dbnsfp.tsv already exists")
    return p.parse_args()


def run_dbnsfp_annotation(glm: Path, dbnsfp: Path, outdir: Path) -> Path:
    """
    Run annotate_dbnsfp.py using the GLM-Missense TSV as the variant source.
    Returns the path to the output dbnsfp.tsv.
    """
    dbnsfp_out = outdir / "dbnsfp.tsv"
    print(f"\n── Step 1: dbNSFP annotation ─────────────────────────────────")
    cmd = [
        sys.executable, "scoring/annotate_dbnsfp.py",
        "--variants", str(glm),
        "--dbnsfp",   str(dbnsfp),
        "--outdir",   str(outdir),
    ]
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"ERROR: annotate_dbnsfp.py failed (exit {result.returncode})")
    if not dbnsfp_out.exists():
        sys.exit(f"ERROR: expected output not found: {dbnsfp_out}")
    return dbnsfp_out


def merge_scores(glm_path: Path, dbnsfp_path: Path, outdir: Path) -> Path:
    """
    Left-join GLM-Missense scores onto the dbNSFP annotation table on KEY_COLS,
    write MetaMissense_input.tsv. Returns the path to the merged output.
    """
    print(f"\n── Step 2: Merging GLM-Missense scores with dbNSFP ──────────")

    print(f"  Loading GLM-Missense scores from {glm_path}")
    glm = pd.read_csv(glm_path, sep="\t")
    print(f"    {len(glm):,} variants")

    if "GLM-Missense_score" not in glm.columns:
        sys.exit(
            f"ERROR: 'GLM-Missense_score' column not found in {glm_path}.\n"
            f"       Columns present: {list(glm.columns)}"
        )

    # Keep only key columns + score from the GLM file to avoid collisions
    glm_slim = glm[KEY_COLS + ["GLM-Missense_score"]].copy()

    print(f"  Loading dbNSFP annotations from {dbnsfp_path}")
    dbnsfp = pd.read_csv(dbnsfp_path, sep="\t", low_memory=False)
    print(f"    {len(dbnsfp):,} variants, {dbnsfp.shape[1]} columns")

    # Merge: start from dbNSFP (has all variants), left-join GLM-Missense score
    merged = dbnsfp.merge(glm_slim, on=KEY_COLS, how="left")
    n_matched = merged["GLM-Missense_score"].notna().sum()
    print(f"    {n_matched:,}/{len(merged):,} variants matched GLM-Missense score")

    # Move GLM-Missense_score to be the first score column (after key cols)
    score_col = merged.pop("GLM-Missense_score")
    last_key_pos = max(merged.columns.tolist().index(k) for k in KEY_COLS if k in merged.columns)
    merged.insert(last_key_pos + 1, "GLM-Missense_score", score_col)

    out_path = outdir / "MetaMissense_input.tsv"
    merged.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Saved MetaMissense_input.tsv → {out_path}")
    print(f"  Shape: {merged.shape[0]:,} rows × {merged.shape[1]} columns")
    return out_path


def main():
    args = parse_args()

    for path, label in [(args.glm, "--glm"), (args.dbnsfp, "--dbnsfp")]:
        if not path.exists():
            sys.exit(f"ERROR: {label} file not found: {path}")

    tbi = Path(str(args.dbnsfp) + ".tbi")
    if not tbi.exists():
        sys.exit(
            f"ERROR: tabix index not found: {tbi}\n"
            f"       Run: tabix -s 1 -b 2 -e 2 {args.dbnsfp}"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Step 1: dbNSFP annotation
    dbnsfp_out = args.outdir / "dbnsfp.tsv"
    if dbnsfp_out.exists() and not args.force:
        print(f"  [skip] dbNSFP annotation — {dbnsfp_out} already exists "
              f"(use --force to re-run)")
    else:
        dbnsfp_out = run_dbnsfp_annotation(args.glm, args.dbnsfp, args.outdir)

    # Step 2: merge
    merged_out = merge_scores(args.glm, dbnsfp_out, args.outdir)

    print(f"\n{'='*60}")
    print(f"Done. Next step:")
    print(f"  python scoring/MetaMissense.py \\")
    print(f"      {merged_out} \\")
    print(f"      scoring/MetaMissense.joblib")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()