#!/usr/bin/env python3
"""
run_evaluation.py — Part 2 of the evaluation pipeline.

For every row in pipeline_config.tsv, auto-generates merge_config.tsv
(always regenerated — never skipped), then runs merge.py and evaluate.py.

For SOURCE rows:
  - predictions dir is results/predictions/{dataset_key}/
  - merge_config.tsv is built from all score files found in that dir

For DERIVED (reuse:X) rows:
  - predictions dir is the PARENT's results/predictions/{parent_key}/
  - merge_config.tsv is still written to results/predictions/{dataset_key}/
    so each dataset has its own config (and can differ from parent's)
  - evaluate.py is passed --subset if subset_ids is set in config

merge_config.tsv format (matches existing evaluation/merge.py convention):
    label    path    source    highlight

  - finetune files → source=finetune, highlight=yes
  - zeroshot files → source=zeroshot, highlight=no
  - dbnsfp.tsv     → source=dbnsfp,   highlight=no

Usage:
    python evaluation/run_evaluation.py \\
        --config pipeline_config.tsv

    # Dry run — print commands without executing
    python evaluation/run_evaluation.py \\
        --config pipeline_config.tsv \\
        --dry_run

    # Skip merge (useful if merged.tsv already exists and you only want eval)
    python evaluation/run_evaluation.py \\
        --config pipeline_config.tsv \\
        --skip_merge
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list[str], dry_run: bool) -> None:
    """Print and optionally execute a shell command."""
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


def classify_score_file(path: Path) -> tuple[str, str, str]:
    """
    Given a score TSV filename, return (label, source, highlight).

    Naming convention:
      finetune_*.tsv  → finetune, highlight=yes
      zeroshot_*.tsv  → zeroshot, highlight=no
      dbnsfp.tsv      → dbnsfp,   highlight=no
    """
    stem = path.stem
    if stem == "dbnsfp":
        return stem, "dbnsfp", "no"
    elif stem.startswith("finetune_"):
        return stem, "finetune", "yes"
    elif stem.startswith("zeroshot_"):
        return stem, "zeroshot", "no"
    else:
        # Unknown — treat as zeroshot, not highlighted
        return stem, "zeroshot", "no"


def build_merge_config(score_dir: Path, merge_config_path: Path) -> None:
    """
    Scan score_dir for .tsv files (excluding merge_config.tsv and merged.tsv)
    and write a merge_config.tsv.

    Files are ordered: finetune first, then zeroshot (alphabetical), then dbnsfp.
    """
    tsv_files = sorted(
        f for f in score_dir.glob("*.tsv")
        if f.name not in ("merge_config.tsv", "merged.tsv")
        and not f.name.startswith("dbnsfp_column")  # coverage report
    )

    finetune_rows = []
    zeroshot_rows = []
    dbnsfp_rows   = []

    for f in tsv_files:
        label, source, highlight = classify_score_file(f)
        row = {"label": label, "path": str(f), "source": source, "highlight": highlight}
        if source == "finetune":
            finetune_rows.append(row)
        elif source == "dbnsfp":
            dbnsfp_rows.append(row)
        else:
            zeroshot_rows.append(row)

    all_rows = finetune_rows + zeroshot_rows + dbnsfp_rows

    if not all_rows:
        print(f"  WARNING: no score files found in {score_dir} — merge_config.tsv will be empty")

    merge_config_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows, columns=["label", "path", "source", "highlight"])
    df.to_csv(merge_config_path, sep="\t", index=False)
    print(f"  Wrote merge_config.tsv ({len(df)} entries) → {merge_config_path}")
    for _, r in df.iterrows():
        print(f"    {r['highlight']:3s}  {r['source']:10s}  {r['label']}")


# ── Step functions ───────────────────────────────────────────────────────────

def step_merge(merge_config_path: Path, merged_output: Path, dry_run: bool) -> None:
    print(f"\n── Merge → {merged_output}")
    cmd = [
        "python", "evaluation/merge.py",
        "--config", str(merge_config_path),
        "--output", str(merged_output),
    ]
    run(cmd, dry_run)


def step_evaluate(merged_path: Path, merge_config_path: Path,
                  eval_outdir: Path, subset_ids: str | None,
                  dry_run: bool) -> None:
    print(f"\n── Evaluate → {eval_outdir}")
    cmd = [
        "python", "evaluation/evaluate.py",
        "--merged", str(merged_path),
        "--config", str(merge_config_path),
        "--outdir", str(eval_outdir),
    ]
    if subset_ids:
        cmd += ["--subset", subset_ids]
    run(cmd, dry_run)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",      required=True,
                   help="Path to pipeline_config.tsv")
    p.add_argument("--dry_run",     action="store_true",
                   help="Print commands without executing")
    p.add_argument("--skip_merge",  action="store_true",
                   help="Skip merge step (use existing merged.tsv)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = pd.read_csv(args.config, sep="\t", dtype=str).fillna("")

    print("=" * 70)
    print("run_evaluation.py")
    print(f"  Config     : {args.config}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"  Skip merge : {args.skip_merge}")
    print(f"  Rows       : {len(cfg)}")
    print("=" * 70)

    for _, row in cfg.iterrows():
        dataset_key  = row["dataset_key"]
        seq12k_input = str(row.get("seq12k_input", "")).strip()
        subset_ids   = str(row.get("subset_ids",   "")).strip() or None
        parent_key   = resolve_parent_key(seq12k_input)
        is_reuse     = parent_key is not None

        # Score files live in parent dir for reuse rows, own dir for source rows
        score_dir    = predictions_dir(parent_key if is_reuse else dataset_key)
        # merge_config and eval outputs always go in dataset's own dir
        own_dir      = predictions_dir(dataset_key)
        own_dir.mkdir(parents=True, exist_ok=True)

        merge_config_path = own_dir / "merge_config.tsv"
        merged_path       = own_dir / "merged.tsv"
        eval_outdir       = own_dir / "eval_all"

        print(f"\n{'='*70}")
        print(f"Dataset    : {dataset_key}  ({'derived/reuse' if is_reuse else 'source'})")
        print(f"Score dir  : {score_dir}")
        print(f"Own dir    : {own_dir}")
        if subset_ids:
            print(f"Subset IDs : {subset_ids}")
        print(f"{'='*70}")

        # Always regenerate merge_config.tsv
        print(f"\n── Building merge_config.tsv from {score_dir}")
        build_merge_config(score_dir, merge_config_path)

        # Merge
        if args.skip_merge:
            if not merged_path.exists():
                print(f"  WARNING: --skip_merge set but {merged_path} does not exist — running merge anyway")
                step_merge(merge_config_path, merged_path, args.dry_run)
            else:
                print(f"  [skip] merge — --skip_merge set and {merged_path} exists")
        else:
            step_merge(merge_config_path, merged_path, args.dry_run)

        # Evaluate
        step_evaluate(merged_path, merge_config_path, eval_outdir,
                      subset_ids, args.dry_run)

    print(f"\n{'='*70}")
    print("run_evaluation.py complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()