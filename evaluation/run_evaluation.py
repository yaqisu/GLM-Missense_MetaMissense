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


def resolve_premerged_path(seq12k_input: str) -> str | None:
    """
    If seq12k_input starts with 'premerged:', return the path to the
    pre-existing merged.tsv. Scoring and merge steps are both skipped.
    The path may be absolute or relative (to the repo root).
    Example:  premerged:results/predictions/ClinVar_260309only.testset/merged.tsv
    """
    if str(seq12k_input).startswith("premerged:"):
        return seq12k_input.split("premerged:", 1)[1].strip()
    return None


def classify_score_file(path: Path) -> tuple[str, str, str]:
    """
    Given a score TSV filename, return (label, source, highlight).

    Naming convention:
      finetune_*.tsv      → finetune, highlight=yes
      GLM-Missense.tsv    → finetune, highlight=yes  (canonical fine-tuned model)
      MetaMissense.tsv    → zeroshot, highlight=no   (external ensemble, not highlighted)
      zeroshot_*.tsv      → zeroshot, highlight=no
      dbnsfp.tsv          → dbnsfp,   highlight=no

    NOTE: The stem is used verbatim as the score column label ({stem}_score).
    Canonical filenames: GLM-Missense.tsv  → GLM-Missense_score
                         MetaMissense.tsv  → MetaMissense_score
    Both match ANCHOR_COLS in core/plots.py.
    """
    stem       = path.stem
    stem_lower = stem.lower()
    if stem == "dbnsfp":
        return stem, "dbnsfp", "no"
    elif stem.startswith("finetune_") or stem_lower == "glm-missense":
        # Normalize to canonical capitalization so column name matches ANCHOR_COLS
        label = "GLM-Missense" if stem_lower == "glm-missense" else stem
        return label, "finetune", "yes"
    elif stem_lower == "metamissense":
        # Normalize capitalization; MetaMissense is an external tool, not highlighted
        return "MetaMissense", "zeroshot", "no"
    elif stem.startswith("zeroshot_"):
        return stem, "zeroshot", "no"
    else:
        # Unknown — treat as zeroshot, not highlighted
        return stem, "zeroshot", "no"


def build_merge_config_from_columns(merged_path: Path,
                                     merge_config_path: Path) -> None:
    """
    Generate a synthetic merge_config.tsv by inspecting the column names of
    an existing merged.tsv. Used for 'premerged:' rows where no score files
    exist on disk.

    Column detection rules (same logic as classify_score_file):
      - Columns ending in '_score': model score cols
        * starts with 'finetune_' or is 'GLM-Missense_score' → finetune, highlight=yes
        * 'MetaMissense_score'                                → zeroshot, highlight=no
        * otherwise                                          → zeroshot, highlight=no
      - All other non-key columns are treated as dbnsfp (one synthetic row).
    """

    KEY_COLS_SET = {"variant_id", "chromosome", "position",
                    "ref_allele", "alt_allele", "true_label"}

    df = pd.read_csv(merged_path, sep="\t", nrows=0)   # header only
    score_cols = [c for c in df.columns if c.endswith("_score")]

    rows = []
    for col in score_cols:
        stem = col[:-len("_score")]          # strip trailing _score
        stem_lower = stem.lower()
        if stem.startswith("finetune_") or stem_lower == "glm-missense":
            label     = "GLM-Missense" if stem_lower == "glm-missense" else stem
            source    = "finetune"
            highlight = "yes"
        elif stem_lower == "metamissense":
            label, source, highlight = "MetaMissense", "zeroshot", "no"
        else:
            label, source, highlight = stem, "zeroshot", "no"
        rows.append({"label": label, "path": str(merged_path),
                     "source": source, "highlight": highlight})

    # One synthetic dbnsfp row so evaluate.py knows there is annotation data
    has_dbnsfp = any(c not in KEY_COLS_SET and not c.endswith("_score")
                     for c in df.columns)
    if has_dbnsfp:
        rows.append({"label": "dbnsfp", "path": str(merged_path),
                     "source": "dbnsfp", "highlight": "no"})

    merge_config_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows, columns=["label", "path", "source", "highlight"])
    out.to_csv(merge_config_path, sep="\t", index=False)
    print(f"  Wrote synthetic merge_config.tsv ({len(out)} entries) → {merge_config_path}")
    for _, r in out.iterrows():
        print(f"    {r['highlight']:3s}  {r['source']:10s}  {r['label']}")


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
        dataset_key    = row["dataset_key"]
        seq12k_input   = str(row.get("seq12k_input", "")).strip()
        subset_ids_raw = str(row.get("subset_ids", "")).strip()
        # Only treat as a subset path if it looks like a file (contains '/' or '.')
        # Guards against gpu column values (e.g. "0", "1") bleeding in
        subset_ids     = subset_ids_raw if ("/" in subset_ids_raw or "." in subset_ids_raw) else None
        parent_key     = resolve_parent_key(seq12k_input)
        premerged_path = resolve_premerged_path(seq12k_input)
        is_reuse       = parent_key is not None
        is_premerged   = premerged_path is not None

        own_dir = predictions_dir(dataset_key)
        own_dir.mkdir(parents=True, exist_ok=True)
        merge_config_path = own_dir / "merge_config.tsv"
        eval_outdir       = own_dir / "eval_all"

        print(f"\n{'='*70}")
        if is_premerged:
            print(f"Dataset    : {dataset_key}  (premerged — skip scoring + merge)")
            print(f"Merged TSV : {premerged_path}")
        elif is_reuse:
            score_dir = predictions_dir(parent_key)
            print(f"Dataset    : {dataset_key}  (derived/reuse from {parent_key})")
            print(f"Score dir  : {score_dir}")
        else:
            score_dir = predictions_dir(dataset_key)
            print(f"Dataset    : {dataset_key}  (source)")
            print(f"Score dir  : {score_dir}")
        print(f"Own dir    : {own_dir}")
        if subset_ids:
            print(f"Subset IDs : {subset_ids}")
        print(f"{'='*70}")

        if is_premerged:
            # ── Premerged path: skip scoring and merge entirely ──────────────
            merged_path = Path(premerged_path)
            if not merged_path.exists():
                print(f"  ERROR: premerged file not found: {merged_path}")
                sys.exit(1)
            print(f"\n── Building synthetic merge_config.tsv from columns of {merged_path}")
            build_merge_config_from_columns(merged_path, merge_config_path)
        else:
            merged_path = own_dir / "merged.tsv"
            # Always regenerate merge_config.tsv from score files on disk
            print(f"\n── Building merge_config.tsv from {score_dir}")
            build_merge_config(score_dir, merge_config_path)

            # Merge (skipped for premerged rows above)
            if args.skip_merge:
                if not merged_path.exists():
                    print(f"  WARNING: --skip_merge set but {merged_path} does not exist — running merge anyway")
                    step_merge(merge_config_path, merged_path, args.dry_run)
                else:
                    print(f"  [skip] merge — --skip_merge set and {merged_path} exists")
            else:
                step_merge(merge_config_path, merged_path, args.dry_run)

        # Evaluate (always runs for all row types)
        step_evaluate(merged_path, merge_config_path, eval_outdir,
                      subset_ids, args.dry_run)

    print(f"\n{'='*70}")
    print("run_evaluation.py complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()