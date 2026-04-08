#!/usr/bin/env python3
"""
evaluate_checkpoints.py — Offline Checkpoint Evaluator
=======================================================
Companion script to NT2_ref_alt_contrast.py.

Polls a checkpoints/ directory every hour for new checkpoint files produced
by the training script. For each new checkpoint found, runs:
    - Evaluation on a random 10k subset of the training set
    - Full evaluation on the complete validation set

Progress is logged every `--log_every` batches so you can track how many
samples have been evaluated so far within each checkpoint eval pass.

Supports multi-GPU evaluation via nn.DataParallel (same pattern as training).
Use --gpus to specify which GPUs to use. With 4 GPUs and batch_size=256,
each eval pass is ~4x faster than single-GPU.

After all expected checkpoints are evaluated, applies early stopping logic
offline to identify the best model for any patience value.

Designed to run in parallel with training in a separate terminal:

    # Terminal 1 — training (no inline eval):
    python training/NT2_ref_alt_contrast.py \\
        --train_path ... --val_path ... --output_dir ... --no_eval

    # Terminal 2 — eval running alongside on same GPUs (safe, eval uses ~7GB):
    python training/evaluate_checkpoints.py \\
        --checkpoints_dir results/.../exp_1_concat_diff/checkpoints \\
        --train_path ... --val_path ... \\
        --num_steps 17000 --eval_interval 1000 \\
        --gpus 0 1 2 3 --batch_size 256

    # Or run after training finishes (all checkpoints already exist):
    python training/evaluate_checkpoints.py \\
        --checkpoints_dir results/.../exp_1_concat_diff/checkpoints \\
        --train_path ... --val_path ... \\
        --num_steps 17000 --eval_interval 1000 \\
        --gpus 0 1 2 3 --batch_size 256 \\
        --poll_interval_hours 0

Output files written to the parent exp_dir (sibling of checkpoints/):
    eval_metrics.csv       — one row per evaluated checkpoint:
                             step, train_auroc, train_auprc, train_loss,
                             val_auroc, val_auprc, val_loss, evaluated_at
    early_stopping.json    — best checkpoint for patience 1, 2, 3
    eval_curves.pdf        — train vs val AUROC/AUPRC plots
    eval.log               — full log including per-batch progress
"""

import sys
import time
import json
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Import model and dataset classes from the training script.
# Assumes evaluate_checkpoints.py lives in the same directory.
sys.path.insert(0, str(Path(__file__).parent))
from NT2_ref_alt_contrast import (
    NT2_RefAltContrast,
    create_dual_data_loader,
)


# ============================================================================
# Model loading
# ============================================================================

def load_model_from_checkpoint(ckpt_path, device, gpu_list):
    """
    Load NT2_RefAltContrast from a checkpoint saved by NT2_ref_alt_contrast.py.
    Wraps with nn.DataParallel if more than one GPU is specified.

    gpu_list : list of GPU IDs, e.g. [0, 1, 2, 3].
               First GPU is the primary device.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']

    base_model = AutoModelForMaskedLM.from_pretrained(
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        trust_remote_code=True,
    )
    model = NT2_RefAltContrast(
        base_model,
        lora_rank=config['lora_rank'],
        combine_mode=config['combine_mode'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    if len(gpu_list) > 1:
        model = nn.DataParallel(model, device_ids=gpu_list)

    model.eval()
    return model, ckpt


# ============================================================================
# Evaluation helpers
# ============================================================================

def evaluate_model(model, dataloader, criterion, device,
                   desc="Eval", log_every=50, logger=None):
    """
    Full eval pass over a dataloader. Returns (avg_loss, auroc, auprc).

    Logs progress every log_every batches so you can track how many samples
    have been processed so far within this eval pass.

    desc     : label shown in progress lines, e.g. "Val (step 1000)"
    log_every: print a progress line every this many batches.
               With batch_size=256 and log_every=50, progress is shown
               every 50*256 = 12,800 samples.
    """
    model.eval()
    total_loss = 0.0
    all_labels, all_preds = [], []
    n_total = len(dataloader.dataset)
    n_done = 0

    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg, flush=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            ref_ids = batch['ref_input_ids'].to(device)
            alt_ids = batch['alt_input_ids'].to(device)
            labels  = batch['labels'].to(device)

            outputs = model(ref_ids, alt_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            n_done += labels.shape[0]

            # Log progress every log_every batches and always on the last batch
            if (batch_idx + 1) % log_every == 0 or n_done == n_total:
                log(
                    f"    [{desc}] {n_done:>7,}/{n_total:>7,} samples "
                    f"({100.0 * n_done / n_total:.1f}%)"
                )

    avg_loss = total_loss / len(dataloader)
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    return avg_loss, auroc, auprc


def evaluate_model_subset(model, dataset, criterion, device,
                           n_samples=10000, batch_size=256,
                           desc="Train subset", log_every=50, logger=None):
    """
    Evaluate on a random subset of the training dataset.
    Returns (avg_loss, auroc, auprc).

    n_samples : random samples drawn from the full training set.
                10000 gives a stable AUROC estimate at ~6% of the cost
                of evaluating all 151k training samples.
    """
    n_samples = min(n_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:n_samples].tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    return evaluate_model(model, loader, criterion, device,
                          desc=desc, log_every=log_every, logger=logger)


# ============================================================================
# Early stopping replay
# ============================================================================

def replay_early_stopping(results_df, patience_values=(1, 2, 3)):
    """
    Replay early stopping logic on completed eval results.

    For each patience value, walks through checkpoints in step order,
    tracks the best val AUROC seen so far, and identifies:
      - best_step      : step with the highest val AUROC
      - stopped_at_step: step where early stopping would have triggered
                         (None if training would have run to completion)

    Returns dict: { patience: { best_step, stopped_at_step, val_auroc, ... } }
    """
    report = {}
    for patience in patience_values:
        best_auroc = 0.0
        best_row = None
        counter = 0
        stopped_at = None

        for _, row in results_df.iterrows():
            if row['val_auroc'] > best_auroc:
                best_auroc = row['val_auroc']
                best_row = row
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    stopped_at = int(row['step'])
                    break

        report[patience] = {
            'best_step': int(best_row['step']) if best_row is not None else None,
            'stopped_at_step': stopped_at,
            'val_auroc': float(best_row['val_auroc']) if best_row is not None else None,
            'val_auprc': float(best_row['val_auprc']) if best_row is not None else None,
            'train_auroc': float(best_row['train_auroc']) if best_row is not None else None,
            'train_auprc': float(best_row['train_auprc']) if best_row is not None else None,
            'checkpoint_file': best_row['checkpoint_file'] if best_row is not None else None,
        }
    return report


# ============================================================================
# Plotting
# ============================================================================

def generate_eval_plots(results_df, output_path):
    """Generate train (subset) vs val AUROC/AUPRC/Loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        'Checkpoint Evaluation: Train (10k subset) vs Val (full)', fontsize=14
    )

    pairs = [
        ('train_loss',  'val_loss',  'Loss'),
        ('train_auroc', 'val_auroc', 'AUROC'),
        ('train_auprc', 'val_auprc', 'AUPRC'),
    ]
    for ax, (tcol, vcol, label) in zip(axes, pairs):
        if tcol in results_df.columns and vcol in results_df.columns:
            ax.plot(results_df['step'], results_df[tcol],
                    label='Train (10k subset)', linewidth=2,
                    marker='o', markersize=4)
            ax.plot(results_df['step'], results_df[vcol],
                    label='Val (full)', linewidth=2,
                    marker='o', markersize=4)
            ax.set_xlabel('Step')
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main polling loop
# ============================================================================

def run_eval_loop(checkpoints_dir, train_path, val_path, gpu_list,
                  k, batch_size, train_eval_samples, log_every,
                  num_steps, eval_interval,
                  poll_interval_hours, max_wait_hours):
    """
    Poll checkpoints_dir for new .pt files and evaluate each one.

    Logs per-batch progress during each eval pass so you always know
    how many samples have been processed vs. total.

    Terminates when:
      (a) all expected checkpoints have been evaluated, or
      (b) max_wait_hours has elapsed (safety timeout for crashed training)

    expected_checkpoints = num_steps // eval_interval
    e.g. num_steps=17000, eval_interval=1000 → 17 checkpoints

    Results are written to eval_metrics.csv incrementally — safe to resume
    if this script crashes. Already-evaluated steps are skipped on restart.

    gpu_list : list of GPU IDs for DataParallel eval, e.g. [0, 1, 2, 3].
               Batch is split across all GPUs automatically.
               batch_size is the total across all GPUs.
    """
    checkpoints_dir = Path(checkpoints_dir)
    exp_dir = checkpoints_dir.parent   # eval outputs written here

    expected_checkpoints = num_steps // eval_interval
    poll_interval_sec = poll_interval_hours * 3600
    primary_gpu = gpu_list[0]
    device = torch.device(f'cuda:{primary_gpu}')

    # ---- Logging ----
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(exp_dir / 'eval.log'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger('EvalLoop')

    logger.info("=" * 80)
    logger.info("Checkpoint Evaluator")
    logger.info(f"  Checkpoints dir   : {checkpoints_dir}")
    logger.info(f"  Output dir        : {exp_dir}")
    logger.info(f"  Expected total    : {expected_checkpoints} checkpoints "
                f"(num_steps={num_steps}, eval_interval={eval_interval})")
    logger.info(f"  GPUs              : {gpu_list} "
                f"({'DataParallel' if len(gpu_list) > 1 else 'single GPU'})")
    logger.info(f"  Batch size        : {batch_size} total "
                f"({batch_size // len(gpu_list)} per GPU)")
    logger.info(f"  Train eval samples: {train_eval_samples}")
    logger.info(f"  Progress log every: {log_every} batches "
                f"(= every {log_every * batch_size:,} samples)")
    logger.info(f"  Poll interval     : {poll_interval_hours}h "
                f"{'(immediate — no waiting)' if poll_interval_hours == 0 else ''}")
    logger.info(f"  Max wait          : {max_wait_hours}h")
    logger.info("=" * 80)

    # ---- Output CSV ----
    eval_csv = exp_dir / 'eval_metrics.csv'
    columns = [
        'step', 'checkpoint_file',
        'train_loss', 'train_auroc', 'train_auprc',
        'val_loss',   'val_auroc',   'val_auprc',
        'evaluated_at',
    ]

    # Resumption: read already-evaluated steps from existing CSV
    evaluated_steps = set()
    if eval_csv.exists():
        existing = pd.read_csv(eval_csv)
        if not existing.empty and 'step' in existing.columns:
            evaluated_steps = set(existing['step'].tolist())
            logger.info(
                f"Resuming — already evaluated {len(evaluated_steps)} steps: "
                f"{sorted(evaluated_steps)}"
            )
    else:
        pd.DataFrame(columns=columns).to_csv(eval_csv, index=False)

    # ---- Load tokenizer and dataloaders once ----
    # Reused across all checkpoint evaluations to avoid re-tokenizing
    # 151k training samples 17 times.
    logger.info(
        "Loading tokenizer and dataloaders "
        "(done once, reused across all checkpoints)..."
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        trust_remote_code=True,
    )
    train_loader = create_dual_data_loader(
        train_path, tokenizer, batch_size,
        max_length=2048, shuffle=False, k=k
    )
    val_loader = create_dual_data_loader(
        val_path, tokenizer, batch_size,
        max_length=2048, shuffle=False, k=k
    )
    criterion = nn.BCEWithLogitsLoss()

    n_train_total = len(train_loader.dataset)
    n_val_total = len(val_loader.dataset)
    logger.info(f"  Train dataset : {n_train_total:,} samples total "
                f"(will eval {train_eval_samples:,} random subset per checkpoint)")
    logger.info(f"  Val dataset   : {n_val_total:,} samples total "
                f"(full eval per checkpoint)")
    logger.info(
        f"  Per checkpoint: ~{train_eval_samples // batch_size} train batches + "
        f"~{n_val_total // batch_size} val batches"
    )

    # ---- Polling loop ----
    start_wall = time.time()

    while True:
        elapsed = time.time() - start_wall

        # Find all checkpoint files, sorted by step number
        ckpt_files = sorted(
            checkpoints_dir.glob('step_*.pt'),
            key=lambda p: int(p.stem.split('_')[1])
        )

        # Evaluate any new checkpoints not yet seen
        for ckpt_path in ckpt_files:
            step = int(ckpt_path.stem.split('_')[1])
            if step in evaluated_steps:
                continue

            logger.info("-" * 60)
            logger.info(f"New checkpoint: {ckpt_path.name}")
            logger.info(
                f"  Checkpoints evaluated so far: "
                f"{len(evaluated_steps)}/{expected_checkpoints}"
            )
            t0 = time.time()

            try:
                model, ckpt = load_model_from_checkpoint(
                    ckpt_path, device, gpu_list
                )

                # ---- Train subset eval ----
                logger.info(
                    f"  [1/2] Train subset eval "
                    f"({train_eval_samples:,} random samples of {n_train_total:,})..."
                )
                train_loss, train_auroc, train_auprc = evaluate_model_subset(
                    model, train_loader.dataset, criterion, device,
                    n_samples=train_eval_samples,
                    batch_size=batch_size,
                    desc=f"Train subset (step {step})",
                    log_every=log_every,
                    logger=logger,
                )

                # ---- Full val eval ----
                logger.info(
                    f"  [2/2] Full val eval "
                    f"({n_val_total:,} samples)..."
                )
                val_loss, val_auroc, val_auprc = evaluate_model(
                    model, val_loader, criterion, device,
                    desc=f"Val (step {step})",
                    log_every=log_every,
                    logger=logger,
                )

                # Free GPU memory before loading next checkpoint
                del model
                torch.cuda.empty_cache()

                elapsed_eval = time.time() - t0
                now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                logger.info(
                    f"  Done in {elapsed_eval:.1f}s | "
                    f"Train AUROC: {train_auroc:.4f}, AUPRC: {train_auprc:.4f} | "
                    f"Val AUROC:   {val_auroc:.4f}, AUPRC: {val_auprc:.4f}"
                )

                # Append result immediately — safe even if script crashes later
                row = {
                    'step': step,
                    'checkpoint_file': ckpt_path.name,
                    'train_loss': train_loss,
                    'train_auroc': train_auroc,
                    'train_auprc': train_auprc,
                    'val_loss': val_loss,
                    'val_auroc': val_auroc,
                    'val_auprc': val_auprc,
                    'evaluated_at': now_str,
                }
                pd.DataFrame([row]).to_csv(
                    eval_csv, mode='a', header=False, index=False
                )
                evaluated_steps.add(step)

            except Exception as e:
                logger.error(f"  Failed to evaluate {ckpt_path.name}: {e}")
                logger.error(traceback.format_exc())
                continue

        n_evaluated = len(evaluated_steps)
        logger.info("=" * 60)
        logger.info(
            f"Status: {n_evaluated}/{expected_checkpoints} checkpoints evaluated | "
            f"elapsed: {elapsed / 3600:.1f}h"
        )

        # ---- Termination conditions ----
        if n_evaluated >= expected_checkpoints:
            logger.info(
                f"All {expected_checkpoints} checkpoints evaluated. Finishing."
            )
            break

        if elapsed > max_wait_hours * 3600:
            logger.warning(
                f"Max wait of {max_wait_hours}h reached with "
                f"{n_evaluated}/{expected_checkpoints} checkpoints evaluated. "
                f"Training may have stopped early or crashed."
            )
            break

        if poll_interval_hours == 0:
            logger.info(
                f"poll_interval_hours=0 and no more checkpoints found. "
                f"Finishing with {n_evaluated}/{expected_checkpoints} evaluated."
            )
            break

        logger.info(
            f"Waiting {poll_interval_hours}h for new checkpoints "
            f"({expected_checkpoints - n_evaluated} remaining)..."
        )
        time.sleep(poll_interval_sec)

    # ---- Final analysis ----
    results_df = pd.read_csv(eval_csv).sort_values('step').reset_index(drop=True)

    if results_df.empty:
        logger.warning("No checkpoints were evaluated.")
        return

    # Generate plots
    plot_path = exp_dir / 'eval_curves.pdf'
    generate_eval_plots(results_df, plot_path)
    logger.info(f"Eval curves saved: {plot_path}")

    # Replay early stopping for patience 1, 2, 3
    es_report = replay_early_stopping(results_df, patience_values=(1, 2, 3))
    es_path = exp_dir / 'early_stopping.json'
    with open(es_path, 'w') as f:
        json.dump(es_report, f, indent=2)

    logger.info("=" * 80)
    logger.info("EARLY STOPPING ANALYSIS")
    logger.info("=" * 80)
    for patience, info in es_report.items():
        logger.info(
            f"  Patience={patience} | "
            f"Best step: {info['best_step']} | "
            f"Would stop at step: {info['stopped_at_step']} | "
            f"Val  AUROC: {info['val_auroc']:.4f}, AUPRC: {info['val_auprc']:.4f} | "
            f"Train AUROC: {info['train_auroc']:.4f} | "
            f"Checkpoint: {info['checkpoint_file']}"
        )

    best_row = results_df.loc[results_df['val_auroc'].idxmax()]
    logger.info("=" * 80)
    logger.info("BEST CHECKPOINT OVERALL (ignoring early stopping)")
    logger.info(f"  Step          : {int(best_row['step'])}")
    logger.info(f"  Val  AUROC    : {best_row['val_auroc']:.4f} | "
                f"AUPRC: {best_row['val_auprc']:.4f}")
    logger.info(f"  Train AUROC   : {best_row['train_auroc']:.4f} | "
                f"AUPRC: {best_row['train_auprc']:.4f}")
    logger.info(f"  Checkpoint    : {best_row['checkpoint_file']}")
    logger.info("=" * 80)
    logger.info(f"Full results  : {eval_csv}")
    logger.info(f"Early stopping: {es_path}")
    logger.info(f"Curves        : {plot_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate NT2_ref_alt_contrast checkpoints offline '
                    'with optional multi-GPU DataParallel support'
    )
    parser.add_argument('--checkpoints_dir', type=str, required=True,
                        help='Path to the checkpoints/ directory produced by '
                             'NT2_ref_alt_contrast.py '
                             '(e.g. results/exp_1_concat_diff/checkpoints)')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training TSV (same as used in training)')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to validation TSV (same as used in training)')
    parser.add_argument('--num_steps', type=int, required=True,
                        help='Total training steps used to compute the expected '
                             'number of checkpoints (e.g. 17000)')
    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Checkpoint save interval used during training '
                             '(default: 1000). '
                             'Expected checkpoints = num_steps // eval_interval')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                        help='GPU IDs for DataParallel evaluation '
                             '(default: 0). Pass multiple for multi-GPU: '
                             '--gpus 0 1 2 3. Batch is split evenly across GPUs.')
    parser.add_argument('--k', type=int, default=6,
                        help='K-mer size (default: 6, must match training)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Total batch size across all GPUs for evaluation '
                             '(default: 256). No gradient storage needed during '
                             'eval so this can be much larger than training batch. '
                             'Should be divisible by number of GPUs.')
    parser.add_argument('--train_eval_samples', type=int, default=10000,
                        help='Number of random training samples for train '
                             'AUROC/AUPRC estimation per checkpoint (default: 10000)')
    parser.add_argument('--log_every', type=int, default=50,
                        help='Log progress every N batches during each eval pass '
                             '(default: 50). With batch_size=256, this logs every '
                             '50 * 256 = 12,800 samples processed.')
    parser.add_argument('--poll_interval_hours', type=float, default=1.0,
                        help='How often to check for new checkpoints in hours '
                             '(default: 1.0). Set to 0 to evaluate immediately '
                             'without waiting — use when training is already done '
                             'and all checkpoints exist.')
    parser.add_argument('--max_wait_hours', type=float, default=48.0,
                        help='Maximum total time to wait before giving up '
                             '(default: 48h). Safety timeout in case training '
                             'crashes before producing all checkpoints.')

    args = parser.parse_args()

    if args.batch_size % len(args.gpus) != 0:
        print(f"WARNING: batch_size={args.batch_size} is not divisible by "
              f"len(gpus)={len(args.gpus)}. DataParallel may drop remainder samples.")

    expected = args.num_steps // args.eval_interval

    print("=" * 80)
    print("NT2 Ref-Alt Contrast — Checkpoint Evaluator")
    print(f"  Checkpoints dir   : {args.checkpoints_dir}")
    print(f"  Expected          : {expected} checkpoints "
          f"({args.num_steps} steps / {args.eval_interval} interval)")
    print(f"  GPUs              : {args.gpus} "
          f"({'DataParallel' if len(args.gpus) > 1 else 'single GPU'})")
    print(f"  Batch size        : {args.batch_size} total "
          f"({args.batch_size // len(args.gpus)} per GPU)")
    print(f"  Train eval samples: {args.train_eval_samples:,}")
    print(f"  Progress log every: {args.log_every} batches "
          f"(= every {args.log_every * args.batch_size:,} samples)")
    print(f"  Poll interval     : {args.poll_interval_hours}h")
    print(f"  Max wait          : {args.max_wait_hours}h")
    print("=" * 80)

    run_eval_loop(
        checkpoints_dir=args.checkpoints_dir,
        train_path=args.train_path,
        val_path=args.val_path,
        gpu_list=args.gpus,
        k=args.k,
        batch_size=args.batch_size,
        train_eval_samples=args.train_eval_samples,
        log_every=args.log_every,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        poll_interval_hours=args.poll_interval_hours,
        max_wait_hours=args.max_wait_hours,
    )


if __name__ == '__main__':
    main()