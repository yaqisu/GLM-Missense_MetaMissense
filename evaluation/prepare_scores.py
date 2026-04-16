#!/usr/bin/env python3
"""
prepare_scores.py — Part 1 of the evaluation pipeline.

For each SOURCE row in pipeline_config.tsv (rows where seq12k_input does NOT
start with "reuse:"), runs:
  1. extract_subset_ids.py  — generates BvsP_ids.tsv for derived datasets
  2. score_variants.py      — fine-tuned model scoring (skips if output exists)
  3. zeroshot_nt.py         — NT-2 seq12k, NT-2 seq6k, NT-1 seq6k
  4. zeroshot_caduceus.py   — Caduceus-PS seq30k, Caduceus-Ph seq30k
  5. annotate_dbnsfp.py     — dbNSFP annotations

For DERIVED rows (reuse:X), only generates the BvsP_ids.tsv if not already
present. All scoring is skipped — those rows reuse the parent's prediction files.

All steps skip gracefully if the output file already exists.

Usage:
    python evaluation/prepare_scores.py \\
        --config  pipeline_config.tsv \\
        --model   scoring/model/best_model.pt \\
        --dbnsfp  data/dbnsfp/dbNSFP5.3.1a_grch38.gz

    # Dry run — print commands without executing
    python evaluation/prepare_scores.py \\
        --config  pipeline_config.tsv \\
        --model   scoring/model/best_model.pt \\
        --dbnsfp  data/dbnsfp/dbNSFP5.3.1a_grch38.gz \\
        --dry_run
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ── Model definitions ────────────────────────────────────────────────────────

NT_MODELS = [
    {
        "label":      "zeroshot_NT2_seq12k",
        "script":     "evaluation/zeroshot_nt.py",
        "model_name": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "seq_col":    "seq12k_input",
    },
    {
        "label":      "zeroshot_NT2_seq6k",
        "script":     "evaluation/zeroshot_nt.py",
        "model_name": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "seq_col":    "seq6k_input",
    },
    {
        "label":      "zeroshot_NT1_seq6k",
        "script":     "evaluation/zeroshot_nt.py",
        "model_name": "InstaDeepAI/nucleotide-transformer-500m-human-ref",
        "seq_col":    "seq6k_input",
    },
]

CADUCEUS_MODELS = [
    {
        "label":      "zeroshot_CaduceusPS_seq30k",
        "script":     "evaluation/zeroshot_caduceus.py",
        "model_name": "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
        "seq_col":    "seq30k_input",
    },
    {
        "label":      "zeroshot_CaduceusPh_seq30k",
        "script":     "evaluation/zeroshot_caduceus.py",
        "model_name": "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        "seq_col":    "seq30k_input",
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list[str], dry_run: bool) -> None:
    """Print and optionally execute a shell command."""
    print("  $", " ".join(cmd))
    if not dry_run:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nERROR: command failed (exit {result.returncode}). Aborting.")
            sys.exit(result.returncode)


def skip(path: str | Path, label: str) -> bool:
    """Return True and print a skip message if path already exists."""
    if Path(path).exists():
        print(f"  [skip] {label} — already exists: {path}")
        return True
    return False


def predictions_dir(dataset_key: str) -> Path:
    return Path("results/predictions") / dataset_key


def resolve_parent(seq12k_input: str, config_df: pd.DataFrame) -> str | None:
    """Return the parent dataset_key for a reuse: row, or None."""
    if str(seq12k_input).startswith("reuse:"):
        return seq12k_input.split("reuse:", 1)[1].strip()
    return None


# ── Step functions ───────────────────────────────────────────────────────────

def step_extract_subset_ids(row: pd.Series, dry_run: bool) -> None:
    """Generate BvsP_ids.tsv for a derived (reuse:) row."""
    subset_ids = str(row.get("subset_ids", "")).strip()
    if not subset_ids:
        return

    if skip(subset_ids, "extract_subset_ids"):
        return

    # Infer dataset name and datadir from the ids file path
    ids_path   = Path(subset_ids)
    datadir    = str(ids_path.parent)
    # Dataset name is the stem up to the first "." after the base name
    # e.g. ClinVar.251103.missense.hg38.BvsP_ids.tsv -> ClinVar.251103
    stem_parts = ids_path.stem.split(".")
    # Find the dataset prefix: everything before "missense"
    try:
        missense_idx = stem_parts.index("missense")
        dataset_name = ".".join(stem_parts[:missense_idx])
    except ValueError:
        # Fallback: use everything before the last two parts
        dataset_name = ".".join(stem_parts[:-2])

    print(f"\n── Generating subset IDs: {subset_ids}")
    cmd = [
        "python", "evaluation/extract_subset_ids.py",
        "--dataset", dataset_name,
        "--datadir", datadir,
        "--labels",  "pathogenic", "benign",
        "--outfile", subset_ids,
    ]
    run(cmd, dry_run)


def step_finetune_score(row: pd.Series, model_path: str, dry_run: bool) -> None:
    """Run score_variants.py for a source row."""
    output = str(row["finetune_score"]).strip()
    if not output:
        print("  [skip] finetune scoring — finetune_score not set in config")
        return

    if skip(output, "finetune score"):
        return

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    gpu = str(row.get("gpu", 0))
    print(f"\n── Fine-tuned scoring → {output}")
    cmd = [
        "python", "scoring/score_variants.py",
        "--input",      str(row["seq12k_input"]),
        "--model",      model_path,
        "--output",     output,
        "--batch_size", "128",
        "--gpu",        gpu,
    ]
    run(cmd, dry_run)


def step_zeroshot(row: pd.Series, dry_run: bool) -> None:
    """Run all zero-shot models for a source row."""
    dataset_key = row["dataset_key"]
    outdir      = predictions_dir(dataset_key)
    gpu         = str(row.get("gpu", 0))
    outdir.mkdir(parents=True, exist_ok=True)

    for m in NT_MODELS + CADUCEUS_MODELS:
        seq_input = str(row.get(m["seq_col"], "")).strip()
        if not seq_input:
            print(f"  [skip] {m['label']} — no input file for {m['seq_col']}")
            continue

        output = str(outdir / f"{m['label']}.tsv")
        if skip(output, m["label"]):
            continue

        print(f"\n── Zero-shot {m['label']} → {output}")
        cmd = [
            "python", m["script"],
            "--input",      seq_input,
            "--output",     output,
            "--model_name", m["model_name"],
            "--gpu",        gpu,
        ]
        run(cmd, dry_run)


def step_dbnsfp(row: pd.Series, dbnsfp_path: str, dry_run: bool) -> None:
    """Run annotate_dbnsfp.py for a source row."""
    dataset_key = row["dataset_key"]
    outdir      = predictions_dir(dataset_key)
    output      = outdir / "dbnsfp.tsv"

    if skip(output, "dbnsfp annotation"):
        return

    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n── dbNSFP annotation → {output}")
    cmd = [
        "python", "evaluation/annotate_dbnsfp.py",
        "--variants", str(row["seq12k_input"]),
        "--dbnsfp",   dbnsfp_path,
        "--outdir",   str(outdir),
    ]
    run(cmd, dry_run)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",   required=True,
                   help="Path to pipeline_config.tsv")
    p.add_argument("--model",    required=True,
                   help="Path to best_model.pt for fine-tuned scoring")
    p.add_argument("--dbnsfp",   required=True,
                   help="Path to dbNSFP .gz file")
    p.add_argument("--dry_run",  action="store_true",
                   help="Print commands without executing")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = pd.read_csv(args.config, sep="\t", dtype=str).fillna("")

    print("=" * 70)
    print("prepare_scores.py")
    print(f"  Config : {args.config}")
    print(f"  Model  : {args.model}")
    print(f"  dbNSFP : {args.dbnsfp}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Rows   : {len(cfg)}")
    print("=" * 70)

    for _, row in cfg.iterrows():
        dataset_key  = row["dataset_key"]
        seq12k_input = str(row.get("seq12k_input", "")).strip()
        is_reuse     = seq12k_input.startswith("reuse:")

        print(f"\n{'='*70}")
        print(f"Dataset : {dataset_key}  ({'derived/reuse' if is_reuse else 'source'})")
        print(f"{'='*70}")

        if is_reuse:
            # Derived dataset: only generate subset IDs if needed
            step_extract_subset_ids(row, args.dry_run)
        else:
            # Source dataset: run all scoring steps
            step_finetune_score(row, args.model, args.dry_run)
            step_zeroshot(row, args.dry_run)
            step_dbnsfp(row, args.dbnsfp, args.dry_run)

    print(f"\n{'='*70}")
    print("prepare_scores.py complete.")
    print("Run run_evaluation.py next for merge + evaluate.")
    print("=" * 70)


if __name__ == "__main__":
    main()