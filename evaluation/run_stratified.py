#!/usr/bin/env python3
"""
run_stratified.py — Run stratified evaluation across all datasets.

Reads pipeline_config.tsv and runs evaluate.py in stratify mode for every
dataset × stratification combination. Passes --subset automatically for
derived (reuse:) datasets.

Stratification analyses run for every dataset:
  - gnomAD4.1_joint_AF   → eval_strat_af     (builtin_af strata)
  - GERP++_RS            → eval_strat_gerp    (builtin_gerp strata)
  - phyloP100way_vertebrate → eval_strat_phylop (builtin_phylop strata)

After all stratified runs, compare_strategies.py is run for each
dataset × stratification to produce summary plots and tables.

Usage:
    python evaluation/run_stratified.py --config pipeline_config.tsv

    # Skip compare_strategies (useful if you only want the per-stratum eval)
    python evaluation/run_stratified.py --config pipeline_config.tsv --skip_compare

    # Dry run — print commands without executing
    python evaluation/run_stratified.py --config pipeline_config.tsv --dry_run
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ── Stratification definitions ───────────────────────────────────────────────
# Add or remove entries here to change which analyses are run for all datasets.

STRAT_ANALYSES = [
    {
        "name":   "strat_af",
        "col":    "gnomAD4.1_joint_AF",
        "strata": "builtin_af",
    },
    {
        "name":   "strat_gerp",
        "col":    "GERP++_RS",
        "strata": "builtin_gerp",
    },
    {
        "name":   "strat_phylop",
        "col":    "phyloP100way_vertebrate",
        "strata": "builtin_phylop",
    },
    {
        "name":   "strat_loeuf",
        "col":    "lof.oe_ci.upper",
        "strata": "builtin_loeuf",
    },
    {
    "name":   "strat_spliceai",
    "col":    "spliceai_DS_max",
    "strata": "builtin_spliceai",
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list[str], dry_run: bool) -> None:
    print("  $", " ".join(cmd))
    if not dry_run:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nERROR: command failed (exit {result.returncode}). Aborting.")
            sys.exit(result.returncode)


def predictions_dir(dataset_key: str) -> Path:
    return Path("results/predictions") / dataset_key


def resolve_parent_key(seq12k_input: str) -> str | None:
    if str(seq12k_input).startswith("reuse:"):
        return seq12k_input.split("reuse:", 1)[1].strip()
    return None


def resolve_premerged_path(seq12k_input: str) -> str | None:
    """Return external merged.tsv path for 'premerged:' rows, else None."""
    if str(seq12k_input).startswith("premerged:"):
        return seq12k_input.split("premerged:", 1)[1].strip()
    return None


# ── Step functions ───────────────────────────────────────────────────────────

def step_stratify(dataset_key: str, merged: Path, merge_config: Path,
                  analysis: dict, subset_ids: str | None,
                  dry_run: bool) -> Path:
    outdir = predictions_dir(dataset_key) / f"eval_{analysis['name']}"
    print(f"\n── Stratify [{analysis['name']}] → {outdir}")
    cmd = [
        "python", "evaluation/evaluate.py",
        "--merged", str(merged),
        "--config", str(merge_config),
        "--outdir", str(outdir),
        "--mode",   "stratify",
        "--col",    analysis["col"],
        "--strata", analysis["strata"],
    ]
    if subset_ids:
        cmd += ["--subset", subset_ids]
    run(cmd, dry_run)
    return outdir


def step_compare(dataset_key: str, strat_outdir: Path, analysis: dict,
                 dry_run: bool) -> None:
    compare_outdir = predictions_dir(dataset_key) / f"comparison_{analysis['name']}"
    print(f"\n── Compare [{analysis['name']}] → {compare_outdir}")
    cmd = [
        "python", "evaluation/compare_strategies.py",
        "--strat_dir", str(strat_outdir),
        "--outdir",    str(compare_outdir),
    ]
    run(cmd, dry_run)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",        required=True,
                   help="Path to pipeline_config.tsv")
    p.add_argument("--skip_compare",  action="store_true",
                   help="Skip compare_strategies.py after stratified eval")
    p.add_argument("--dry_run",       action="store_true",
                   help="Print commands without executing")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = pd.read_csv(args.config, sep="\t", dtype=str).fillna("")

    print("=" * 70)
    print("run_stratified.py")
    print(f"  Config       : {args.config}")
    print(f"  Analyses     : {[a['name'] for a in STRAT_ANALYSES]}")
    print(f"  Skip compare : {args.skip_compare}")
    print(f"  Dry run      : {args.dry_run}")
    print(f"  Datasets     : {list(cfg['dataset_key'])}")
    print("=" * 70)

    for _, row in cfg.iterrows():
        dataset_key    = row["dataset_key"]
        seq12k_input   = str(row.get("seq12k_input", "")).strip()
        subset_ids_raw = str(row.get("subset_ids", "")).strip()
        subset_ids     = subset_ids_raw if ("/" in subset_ids_raw or "." in subset_ids_raw) else None
        parent_key     = resolve_parent_key(seq12k_input)
        premerged_path = resolve_premerged_path(seq12k_input)
        is_premerged   = premerged_path is not None

        own_dir      = predictions_dir(dataset_key)
        # For premerged rows the merged.tsv lives at the external path;
        # merge_config.tsv is always in own_dir (generated by run_evaluation.py)
        merged       = Path(premerged_path) if is_premerged else own_dir / "merged.tsv"
        merge_config = own_dir / "merge_config.tsv"

        print(f"\n{'='*70}")
        if is_premerged:
            print(f"Dataset : {dataset_key}  (premerged)")
            print(f"Merged  : {merged}")
        else:
            print(f"Dataset : {dataset_key}  ({'derived' if parent_key else 'source'})")
        if subset_ids:
            print(f"Subset  : {subset_ids}")
        print(f"{'='*70}")

        if not merged.exists():
            print(f"  ERROR: {merged} not found — run run_evaluation.py first.")
            sys.exit(1)
        if not merge_config.exists():
            print(f"  ERROR: {merge_config} not found — run run_evaluation.py first.")
            sys.exit(1)

        for analysis in STRAT_ANALYSES:
            strat_outdir = step_stratify(
                dataset_key, merged, merge_config,
                analysis, subset_ids, args.dry_run
            )
            if not args.skip_compare:
                step_compare(dataset_key, strat_outdir, analysis, args.dry_run)

    print(f"\n{'='*70}")
    print("run_stratified.py complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()