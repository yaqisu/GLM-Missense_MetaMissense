#!/usr/bin/env python3
"""
score_variants.py
-----------------
Score variants from a pre-processed sequence TSV file using the fine-tuned
Nucleotide Transformer 2 (NT2) Siamese Ref-Alt Contrast model.

Architecture: Both ref and alt sequences are encoded through a shared NT2+LoRA
backbone. The variant-position token is extracted from each arm, projected with
a shared MLPProjector, combined as [ref, alt, ref - alt], and passed to a
2-layer MLPClassifierHead.

Input TSV columns (same format as preprocessing pipeline output):
    variant_id  chromosome  position  ref_allele  alt_allele
    upstream_flank  downstream_flank  ref_sequence  alt_sequence  [label]

The label column is optional — if present, AUC will also be computed.

Output TSV columns:
    variant_id  chromosome  position  ref_allele  alt_allele
    pathogenicity_score  predicted_label

    pathogenicity_score : sigmoid probability (0-1), higher = more pathogenic
    predicted_label     : 0 (benign) or 1 (pathogenic) at threshold (default 0.5)

Usage:
    python scoring/score_variants.py \
        --input  data/splits/ClinVar.251103.missense.hg38.seq12k.BvsP_validation.tsv \
        --model  scoring/model/best_model.pt \
        --output results/predictions/my_scores.tsv
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================================
# Hardcoded config for the released best model
# (architecture must match the weights in best_model.pt exactly)
# ============================================================================

MODEL_CONFIG = {
    "combine_mode": "concat_diff",   # [ref, alt, ref - alt] → head input = 3 * proj_dim
    "lora_rank":    32,
    "proj_dim":     256,
    "base_model":   "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('score_variants')


# ============================================================================
# Model Architecture  (must match NT2_ref_alt_contrast.py exactly)
# ============================================================================

class MLPProjector(nn.Module):
    """
    Projects a single variant-position token embedding (1024-d) from the NT2
    backbone down to proj_dim for combination with the other arm.
    Shared weights — used for both ref and alt arms.
    """
    def __init__(self, input_dim=1024, proj_dim=256, dropout=0.1):
        super(MLPProjector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) — single-token embedding at variant position
        Returns:
            (batch, proj_dim)
        """
        return self.projector(x)


class MLPClassifierHead(nn.Module):
    """
    2-layer MLP classification head operating on the combined ref+alt features.

    input_dim depends on combine_mode:
        concat_diff : 3 * proj_dim   ([ref, alt, ref - alt])
        concat_only : 2 * proj_dim   ([ref, alt])
        diff_only   :     proj_dim   (ref - alt)
    """
    def __init__(self, input_dim, dropout=0.1):
        super(MLPClassifierHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(x).squeeze(-1)


class NT2_RefAltContrast(nn.Module):
    """
    Siamese NT2 model with shared LoRA backbone and MLP head.

    Pipeline:
        ref_sequence → NT2+LoRA → hidden states → [variant_position token]
                                                          ↓ MLPProjector
                                                       ref_feat
        alt_sequence → NT2+LoRA → hidden states → [variant_position token]
                                                          ↓ MLPProjector
                                                       alt_feat
                                                          ↓
                                               combine(ref_feat, alt_feat)
                                                          ↓
                                              MLPClassifierHead → logit

    Memory note: the ref hidden state (~128 MB for batch=4) is extracted and
    explicitly freed before the alt forward pass to avoid holding both full
    hidden state tensors in GPU memory simultaneously.
    """

    def __init__(self, base_model, lora_rank=32, combine_mode='concat_diff',
                 proj_dim=256, dropout=0.1):
        super(NT2_RefAltContrast, self).__init__()

        self.combine_mode = combine_mode
        self.proj_dim = proj_dim

        # ---- Shared NT2 backbone with LoRA ----
        self.bert = base_model

        for param in self.bert.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["query", "value"],
            bias="none",
        )
        self.bert = get_peft_model(self.bert, lora_config)
        self.bert.enable_input_require_grads()

        # ---- Shared MLP projector (same weights for both arms) ----
        self.projector = MLPProjector(
            input_dim=1024, proj_dim=proj_dim, dropout=dropout
        )

        # ---- MLP classification head ----
        if combine_mode == 'concat_diff':
            head_input_dim = 3 * proj_dim
        elif combine_mode == 'concat_only':
            head_input_dim = 2 * proj_dim
        elif combine_mode == 'diff_only':
            head_input_dim = proj_dim
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")

        self.classifier = MLPClassifierHead(input_dim=head_input_dim, dropout=dropout)

    def _encode(self, input_ids):
        """Run input_ids through NT2+LoRA. Returns last hidden state (batch, seq_len, 1024)."""
        attention_mask = (input_ids != 1)  # pad token id = 1 for NT2
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs['hidden_states'][-1]  # (batch, seq_len, 1024)

    def forward(self, ref_input_ids, alt_input_ids):
        """
        Args:
            ref_input_ids : (batch, seq_len) — tokenized reference sequence
            alt_input_ids : (batch, seq_len) — tokenized alternate sequence
        Returns:
            logits: (batch,)
        """
        # Variant is centered in a 12 kb window; with k=6 k-merization:
        # 6000 bp / 6 = 1000 tokens upstream + 1 CLS token → position 1000
        variant_position = 1000

        # ---- Ref arm ----
        ref_hidden = self._encode(ref_input_ids)
        seq_len = ref_hidden.shape[1]
        if variant_position >= seq_len:
            variant_position = seq_len // 2
        ref_tok = ref_hidden[:, variant_position, :].clone()  # (batch, 1024)
        del ref_hidden
        torch.cuda.empty_cache()

        # ---- Alt arm ----
        alt_hidden = self._encode(alt_input_ids)
        alt_tok = alt_hidden[:, variant_position, :].clone()  # (batch, 1024)
        del alt_hidden
        torch.cuda.empty_cache()

        # ---- Project both tokens with shared MLP projector ----
        ref_feat = self.projector(ref_tok)   # (batch, proj_dim)
        alt_feat = self.projector(alt_tok)   # (batch, proj_dim)

        # ---- Combine ----
        if self.combine_mode == 'concat_diff':
            combined = torch.cat([ref_feat, alt_feat, ref_feat - alt_feat], dim=1)
        elif self.combine_mode == 'concat_only':
            combined = torch.cat([ref_feat, alt_feat], dim=1)
        elif self.combine_mode == 'diff_only':
            combined = ref_feat - alt_feat
        else:
            raise ValueError(f"Unknown combine_mode: {self.combine_mode}")

        return self.classifier(combined)  # (batch,)


# ============================================================================
# Dataset
# ============================================================================

def kmerize(sequence: str, k: int) -> str:
    """Split a DNA sequence into space-separated k-mers."""
    return ' '.join(sequence[i:i + k] for i in range(0, len(sequence), k))


class DualScoringDataset(Dataset):
    """
    Dataset for scoring — encodes both ref and alt sequences.
    Label column is not required (inference-only).
    """
    def __init__(self, ref_sequences, alt_sequences, tokenizer, max_length=2048):
        self.ref_sequences = ref_sequences.reset_index(drop=True)
        self.alt_sequences = alt_sequences.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ref_sequences)

    def __getitem__(self, idx):
        ref_enc = self.tokenizer.encode_plus(
            self.ref_sequences.iloc[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        alt_enc = self.tokenizer.encode_plus(
            self.alt_sequences.iloc[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return {
            'ref_input_ids': ref_enc['input_ids'].flatten(),
            'alt_input_ids': alt_enc['input_ids'].flatten(),
        }


# ============================================================================
# Main Scoring Logic
# ============================================================================

def load_model(model_path, device):
    """Load Siamese ref-alt contrast model from .pt file using MODEL_CONFIG."""
    config = MODEL_CONFIG
    logger.info(f"Model config: {config}")

    logger.info(f"Loading base NT2 model and tokenizer from {config['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['base_model'], trust_remote_code=True
    )
    base_model = AutoModelForMaskedLM.from_pretrained(
        config['base_model'], trust_remote_code=True
    )

    model = NT2_RefAltContrast(
        base_model,
        lora_rank=config['lora_rank'],
        combine_mode=config['combine_mode'],
        proj_dim=config['proj_dim'],
    )

    logger.info(f"Loading model weights from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    val_auc = checkpoint.get('val_auc', None)
    if val_auc is not None:
        logger.info(f"Model loaded. Checkpoint val AUC: {val_auc:.4f}")
    else:
        logger.info("Model loaded. (No val AUC stored in checkpoint)")

    return model, tokenizer


def score(model, dataloader, device):
    """Run inference and return pathogenicity scores."""
    all_scores = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            ref_ids = batch['ref_input_ids'].to(device)
            alt_ids = batch['alt_input_ids'].to(device)
            logits  = model(ref_ids, alt_ids)
            scores  = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores)
            if (i + 1) % 10 == 0:
                logger.info(
                    f"  Scored {(i + 1) * dataloader.batch_size} / "
                    f"{len(dataloader.dataset)} variants"
                )
    return np.array(all_scores)


def main():
    parser = argparse.ArgumentParser(
        description='Score variants using the NT2 Siamese Ref-Alt Contrast model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scoring/score_variants.py \\
      --input  data/splits/ClinVar.251103.missense.hg38.seq12k.BvsP_validation.tsv \\
      --model  scoring/model/best_model.pt \\
      --output results/predictions/scores.tsv
        """
    )
    parser.add_argument('--input',      '-i', required=True,
                        help='Input TSV file (from preprocessing pipeline)')
    parser.add_argument('--model',      '-m', required=True,
                        help='Path to best_model.pt')
    parser.add_argument('--output',     '-o', required=True,
                        help='Output TSV file path')
    parser.add_argument('--batch_size', '-b', type=int, default=128,
                        help='Batch size for inference (default: 128). Scoring runs '
                             'under torch.no_grad() so no activations are stored — '
                             'safe to use a much larger batch than training. '
                             'Reduce to 32–64 if running on a smaller GPU.')
    parser.add_argument('--gpu',        '-g', type=int, default=0,
                        help='GPU id to use, -1 for CPU (default: 0)')
    parser.add_argument('--threshold',  '-t', type=float, default=0.5,
                        help='Threshold for predicted_label (default: 0.5)')
    parser.add_argument('--k',          type=int, default=6,
                        help='K-mer size for tokenization (default: 6)')

    args = parser.parse_args()

    # ---- Device setup ----
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # ---- Load input data ----
    logger.info(f"Loading input data from {args.input}")
    df = pd.read_csv(args.input, sep='\t')
    logger.info(f"Loaded {len(df)} variants")

    required_cols = {'ref_sequence', 'alt_sequence'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input TSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # ---- K-merize both sequences ----
    ref_seqs = df['ref_sequence'].apply(lambda x: kmerize(x, args.k))
    alt_seqs = df['alt_sequence'].apply(lambda x: kmerize(x, args.k))

    # ---- Load model ----
    model, tokenizer = load_model(args.model, device)

    # ---- Create dataloader ----
    dataset    = DualScoringDataset(ref_seqs, alt_seqs, tokenizer, max_length=2048)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ---- Score ----
    logger.info(f"Scoring {len(df)} variants...")
    scores = score(model, dataloader, device)

    # ---- Build output ----
    output_df = df[['variant_id', 'chromosome', 'position',
                    'ref_allele', 'alt_allele']].copy()
    output_df['pathogenicity_score'] = scores
    output_df['predicted_label']     = (scores >= args.threshold).astype(int)

    # If labels present, compute AUC
    if 'label' in df.columns:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(df['label'], scores)
        logger.info(f"AUC on labeled data: {auc:.4f}")
        output_df['true_label'] = df['label']

    # ---- Save output ----
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, sep='\t', index=False)
    logger.info(f"Saved {len(output_df)} scored variants to {args.output}")

    # ---- Summary ----
    n_patho  = (output_df['predicted_label'] == 1).sum()
    n_benign = (output_df['predicted_label'] == 0).sum()
    logger.info(
        f"Summary: {n_patho} predicted pathogenic, {n_benign} predicted benign "
        f"(threshold={args.threshold})"
    )


if __name__ == '__main__':
    main()