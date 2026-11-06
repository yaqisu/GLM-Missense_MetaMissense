#!/usr/bin/env python3
"""
Score a formatted variant table with the fine-tuned ESM-1b ref-alt contrast model
=================================================================================
Reads the TSV written by format_sge_for_esm1b.py, runs the fine-tuned Siamese
ESM-1b (LoRA + projector + MLP head from ESM1b_ref_alt_contrast.py) on every row
with status == "ok", and writes the original table back out with one
new column: the predicted probability of pathogenicity (sigmoid of the logit;
empty for rows that were not scored).

The model definition, 1,022-residue windowing, tokenisation and input checks are
imported from ESM1b_ref_alt_contrast.py, so scores are identical to that script's
own --predict_path mode.

Usage:
    python predict_esm1b_sge.py --input XRCC2_SGE.esm1b_input.tsv \\
        --checkpoint results/ESM1b_ref_alt_contrast/exp_1_concat_diff/best_model.pt \\
        --gpus 0
    # keep the annotation columns (transcript, codons, ref_aa/alt_aa, status, ...):
    python predict_esm1b_sge.py --input ... --checkpoint ... --keep_helper_cols
"""

import sys
import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

# Columns added by format_sge_for_esm1b.py
HELPER_COLS = ["variant_id", "gene_name", "transcript_id", "protein_id", "strand",
               "cds_position", "ref_codon", "alt_codon", "protein_position", "ref_aa",
               "alt_aa", "protein_length", "protein_sequence", "transcript_source",
               "p_agrees", "label", "status"]


def import_training_module(path):
    path = Path(path).resolve()
    if not path.exists():
        sys.exit(f"Cannot find {path}; point --model_script at ESM1b_ref_alt_contrast.py")
    sys.path.insert(0, str(path.parent))   # importable by DataLoader workers too
    return importlib.import_module(path.stem)


def main():
    p = argparse.ArgumentParser(
        description="Add fine-tuned ESM-1b predictions to a formatted variant table")
    p.add_argument("--input", required=True, help="TSV from format_sge_for_esm1b.py")
    p.add_argument("--checkpoint", required=True, help="best_model.pt from training")
    p.add_argument("--output", default=None,
                   help="Output .csv or .tsv (default: <input stem>.ESM1b_pred.csv)")
    p.add_argument("--model_script",
                   default=str(Path(__file__).resolve().parent / "ESM1b_ref_alt_contrast.py"),
                   help="Path to ESM1b_ref_alt_contrast.py (default: next to this script)")
    p.add_argument("--score_col", default="ESM1b_score", help="Name of the prediction column")
    p.add_argument("--gpus", type=int, nargs="+", default=[0],
                   help="GPU IDs; >1 uses nn.DataParallel. Falls back to CPU if no CUDA")
    p.add_argument("--eval_batch_size", type=int, default=64, help="Reduce if OOM")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--keep_helper_cols", action="store_true",
                   help="Keep the columns added by the formatting script "
                        "(protein_sequence is always dropped)")
    args = p.parse_args()

    esm = import_training_module(args.model_script)
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score, average_precision_score

    in_path = Path(args.input)
    if args.output:
        out_path = Path(args.output)
    else:
        stem = in_path.name
        for suffix in (".tsv", ".esm1b_input"):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
        out_path = in_path.with_name(stem + ".ESM1b_pred.csv")

    # ---- Checkpoint / device ----
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    exp_config = ckpt["config"]
    model_name = ckpt.get("model_name", esm.DEFAULT_MODEL)
    max_residues = ckpt.get("max_residues", 1022)
    print(f"Checkpoint: {args.checkpoint} (step {ckpt.get('step')}, "
          f"val AUROC {ckpt.get('val_auroc', float('nan')):.4f}, "
          f"combine_mode {exp_config['combine_mode']}, max_residues {max_residues})")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpus[0]}")
        torch.cuda.set_device(args.gpus[0])
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available -- running on CPU (slow)")

    # ---- Data: raw strings for output, validated table for the model ----
    raw = pd.read_csv(in_path, sep="\t", dtype=str, keep_default_na=False)
    # pandas renames duplicated headers (X -> X.1); restore them when writing
    header = pd.read_csv(in_path, sep="\t", header=None, nrows=1, dtype=str,
                         keep_default_na=False).iloc[0].tolist()
    restore = {m: o for m, o in zip(raw.columns, header) if m != o} \
        if len(header) == raw.shape[1] else {}
    table = esm.load_variant_table(in_path, require_label=False)
    if len(raw) != len(table):
        sys.exit("Row count mismatch while reading the input table")
    scorable = table["_valid"].to_numpy(dtype=bool).copy()
    if "status" in raw.columns:
        scorable = scorable & (raw["status"] == "ok").to_numpy(dtype=bool)
    n_ok = int(scorable.sum())
    print(f"Scoring {n_ok:,} / {len(raw):,} rows")
    if n_ok == 0:
        sys.exit("Nothing to score -- check the status column of the input table")
    sub = table[scorable].reset_index(drop=True)
    n_long = int((sub["_seq"].str.len() > max_residues).sum())
    if n_long:
        print(f"{n_long:,} variants in proteins > {max_residues} aa use a window "
              f"around the mutated residue")

    # ---- Model ----
    tokenizer = esm.AutoTokenizer.from_pretrained(model_name)
    model = esm.build_model(model_name, exp_config)
    esm.load_trainable_state_dict(model, ckpt["model_state_dict"])
    model = model.to(device)
    if device.type == "cuda" and len(args.gpus) > 1:
        model = nn.DataParallel(model, device_ids=args.gpus)
    model.eval()

    loader = esm.make_loader(esm.ProteinPairDataset(sub, tokenizer, max_residues),
                             args.eval_batch_size, False, tokenizer.pad_token_id,
                             num_workers=args.num_workers)
    logits = []
    with torch.no_grad():
        for b, batch in enumerate(loader, 1):
            logits.append(esm.forward_batch(model, batch, device).float().cpu())
            if b % 20 == 0 or b == len(loader):
                print(f"  batch {b}/{len(loader)}", flush=True)
    probs = torch.sigmoid(torch.cat(logits)).numpy()

    # ---- Write original table + prediction column ----
    out = raw.copy()
    if args.keep_helper_cols:
        out = out.drop(columns=["protein_sequence"], errors="ignore")
    else:
        out = out.drop(columns=[c for c in HELPER_COLS if c in out.columns])
        out = out.rename(columns={f"{c}_original": c for c in HELPER_COLS
                                  if f"{c}_original" in out.columns})
    if args.score_col in out.columns:
        sys.exit(f"Column {args.score_col} already exists; choose another --score_col")
    score = np.full(len(out), np.nan)
    score[scorable] = probs
    out[args.score_col] = score

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if out_path.suffix.lower() in (".tsv", ".txt") else ","
    out.rename(columns=restore).to_csv(out_path, sep=sep, index=False,
                                       float_format="%.6g")
    print(f"Wrote {out_path} with new column '{args.score_col}'")

    # ---- Optional evaluation against the formatting script's label ----
    labels = sub["_label"].to_numpy(dtype=float)
    has = ~np.isnan(labels)
    if has.any() and len(np.unique(labels[has])) == 2:
        print(f"vs label: AUROC {roc_auc_score(labels[has], probs[has]):.4f} | "
              f"AUPRC {average_precision_score(labels[has], probs[has]):.4f} "
              f"(n={int(has.sum()):,}; {int(labels[has].sum()):,} positive)")


if __name__ == "__main__":
    main()
