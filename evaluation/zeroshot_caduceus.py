#!/usr/bin/env python3
"""
zeroshot_caduceus.py
--------------------
Zero-shot variant pathogenicity scoring using a pretrained Caduceus model
via masked marginal log-likelihood ratio.

Caduceus differs from Nucleotide Transformer in two key ways:
  - Tokenization: single character per token (A/T/G/C), NOT k-mers.
    Sequences are passed as raw strings, no kmerize() step.
  - Variant token index: variant is at nucleotide position seq_len//2 (0-indexed),
    +1 for [CLS] token = seq_len//2 + 1.
  - Max length: 131,072 tokens (seq30k = 29,999 tokens, well within limit).

Strategy (masked marginal), same as NT_zeroshot.py:
    For each variant, mask the token at the variant position in the alt
    sequence and compute:

        masked_marginal = log P(alt_token | context) - log P(ref_token | context)

    Pathogenicity score = sigmoid(-masked_marginal), 0-1, higher = more pathogenic.

Output format is identical to scoring/score_variants.py and NT_zeroshot.py so
it feeds into evaluation/merge_predictions.py without modification.

Usage (run from repo root):
    python evaluation/zeroshot_caduceus.py \\
        --input  data/sequences/ClinVar.260309only.missense.hg38.seq30k.tsv \\
        --output results/predictions/ClinVar.260309only.seq30k/zeroshot_CaduceusPS_seq30k.tsv \\
        --gpu 2

    # Caduceus-Ph variant
    python evaluation/zeroshot_caduceus.py \\
        --input  data/sequences/ClinVar.260309only.missense.hg38.seq30k.tsv \\
        --output results/predictions/ClinVar.260309only.seq30k/zeroshot_CaduceusPh_seq30k.tsv \\
        --model_name kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \\
        --gpu 2
"""

import sys
import json
import logging
import argparse
from pathlib import Path

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForMaskedLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('zeroshot_caduceus')


# ============================================================================
# Variant token index
# ============================================================================

def get_variant_token_idx(seq_len):
    """
    Token index (in model input) of the nucleotide at the variant position.
    Caduceus uses single-character tokenization (1 token per nucleotide).
    Sequences are centered on the variant at nucleotide position seq_len//2 (0-indexed).
    +1 for [CLS] token prepended by the tokenizer.
    """
    return (seq_len // 2)  # no CLS prepended; EOS is appended (not before variant)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Zero-shot Caduceus masked marginal scoring.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python evaluation/zeroshot_caduceus.py \\
      --input  data/sequences/ClinVar.260309only.missense.hg38.seq30k.tsv \\
      --output results/predictions/ClinVar.260309only.seq30k/zeroshot_CaduceusPS_seq30k.tsv \\
      --gpu 2
        """
    )
    parser.add_argument('--input',      '-i', required=True,
                        help='Input TSV (same format as score_variants.py input)')
    parser.add_argument('--output',     '-o', required=True,
                        help='Output TSV path')
    parser.add_argument('--model_name', '-m',
                        default='kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16',
                        help='HuggingFace model name (default: Caduceus-PS)')
    parser.add_argument('--gpu',        '-g', type=int, default=0,
                        help='GPU id, -1 for CPU (default: 0)')
    parser.add_argument('--threshold',  '-t', type=float, default=0.5,
                        help='Threshold for predicted_label (default: 0.5)')
    args = parser.parse_args()

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # Load data
    logger.info(f"Loading input from {args.input}")
    df = pd.read_csv(args.input, sep='\t')
    logger.info(f"Loaded {len(df)} variants")

    seq_len = len(df['alt_sequence'].iloc[0])
    variant_token_idx = get_variant_token_idx(seq_len)
    logger.info(f"Sequence length: {seq_len} nt | single-char tokenization | "
                f"variant token index: {variant_token_idx} | note: no CLS prepended")

    # Caduceus takes raw sequences (no kmerization)
    ref_seqs = df['ref_sequence'].tolist()
    alt_seqs = df['alt_sequence'].tolist()

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if output file already exists
    already_scored = set()
    if output_path.exists():
        done = pd.read_csv(output_path, sep='\t', usecols=['variant_id'])
        already_scored = set(done['variant_id'].astype(str))
        logger.info(f"Resuming: {len(already_scored)} variants already scored, skipping them")

    has_label = 'label' in df.columns
    output_cols = [
        'variant_id', 'chromosome', 'position', 'ref_allele', 'alt_allele',
        'pathogenicity_score', 'predicted_label',
        'log_p_alt', 'log_p_ref', 'log_likelihood_ratio',
    ]
    if has_label:
        output_cols.append('true_label')

    # Write header if starting fresh
    write_header = not output_path.exists()
    out_file = open(output_path, 'a')
    if write_header:
        out_file.write('\t'.join(output_cols) + '\n')

    # Score incrementally, writing every checkpoint_every samples
    checkpoint_every = 100
    mask_token_id = tokenizer.mask_token_id
    n = len(df)
    n_written = 0
    batch_rows = []

    logger.info(f"Computing masked marginal scores (checkpoint every {checkpoint_every} samples)...")

    model.eval()
    with torch.no_grad():
        for idx in range(n):
            vid = str(df['variant_id'].iloc[idx])
            if vid in already_scored:
                continue

            ref_seq = ref_seqs[idx]
            alt_seq = alt_seqs[idx]

            # Caduceus: tokenize raw sequence, no kmerization
            # Note: Caduceus tokenizer does not return attention_mask —
            # the SSM architecture does not use it (no padding needed for single sequences)
            ref_enc = tokenizer([ref_seq], add_special_tokens=True,
                                return_tensors='pt', truncation=True,
                                max_length=131072)
            alt_enc = tokenizer([alt_seq], add_special_tokens=True,
                                return_tensors='pt', truncation=True,
                                max_length=131072)

            ref_ids = ref_enc['input_ids'].to(device)
            alt_ids = alt_enc['input_ids'].to(device)

            masked_ids = alt_ids.clone()
            vtok = min(variant_token_idx, masked_ids.shape[1] - 1)
            masked_ids[:, vtok] = mask_token_id

            outputs = model(input_ids=masked_ids)
            log_probs = torch.log_softmax(outputs.logits[0, vtok, :], dim=-1)

            log_p_alt = log_probs[alt_ids[0, vtok]].item()
            log_p_ref = log_probs[ref_ids[0, vtok]].item()
            log_ratio = log_p_alt - log_p_ref
            path_score = float(torch.sigmoid(torch.tensor(-log_ratio)))
            pred_label = int(path_score >= args.threshold)

            row = {
                'variant_id':           vid,
                'chromosome':           df['chromosome'].iloc[idx],
                'position':             df['position'].iloc[idx],
                'ref_allele':           df['ref_allele'].iloc[idx],
                'alt_allele':           df['alt_allele'].iloc[idx],
                'pathogenicity_score':  path_score,
                'predicted_label':      pred_label,
                'log_p_alt':            log_p_alt,
                'log_p_ref':            log_p_ref,
                'log_likelihood_ratio': log_ratio,
            }
            if has_label:
                row['true_label'] = int(df['label'].iloc[idx])

            batch_rows.append(row)
            n_written += 1

            # Flush to disk every checkpoint_every samples
            if len(batch_rows) >= checkpoint_every:
                for r in batch_rows:
                    out_file.write('\t'.join(str(r[c]) for c in output_cols) + '\n')
                out_file.flush()
                batch_rows = []
                logger.info(f"  Scored and saved {len(already_scored) + n_written}/{n} variants")

    # Write any remaining rows
    for r in batch_rows:
        out_file.write('\t'.join(str(r[c]) for c in output_cols) + '\n')
    out_file.flush()
    out_file.close()

    logger.info(f"Saved {n_written} newly scored variants to {output_path}")

    # Final metrics over the complete output file
    full_df = pd.read_csv(output_path, sep='\t')
    n_patho  = (full_df['predicted_label'] == 1).sum()
    n_benign = (full_df['predicted_label'] == 0).sum()
    logger.info(f"Summary (all {len(full_df)} variants): "
                f"{n_patho} predicted pathogenic, {n_benign} predicted benign "
                f"(threshold={args.threshold})")

    auc = None
    if 'true_label' in full_df.columns:
        auc = roc_auc_score(full_df['true_label'], full_df['pathogenicity_score'])
        logger.info(f"AUC on labeled data: {auc:.4f}")

    # Save summary JSON
    summary = {
        'model_name':        args.model_name,
        'input':             args.input,
        'n_variants_total':  len(full_df),
        'n_variants_new':    n_written,
        'tokenization':      'single_char',
        'variant_token_idx': variant_token_idx,
        'auc':               float(auc) if auc is not None else None,
        'threshold':         args.threshold,
    }
    summary_path = output_path.with_suffix('.summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_path}")


if __name__ == '__main__':
    main()