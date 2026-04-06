#!/usr/bin/env python3
"""
NT2 Ref-Alt Contrast — Siamese LoRA Fine-Tuning with MLP Head
==============================================================
Siamese-style architecture: both ref and alt sequences are encoded
through the shared NT2+LoRA backbone. At the variant token position,
embeddings are extracted for each arm, then [ref, alt, ref - alt] are
concatenated and passed to a 2-layer MLP classifier head.

Data format expected (TSV):
    variant_id  chromosome  position  ref_allele  alt_allele
    upstream_flank  downstream_flank
    ref_sequence  alt_sequence  label

Both ref_sequence and alt_sequence are read directly from the TSV.

MLP head follows the same MLPClassifier pattern from NT2_lora_sweep.py:
    Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear(1)

Multi-GPU: uses nn.DataParallel across all specified GPUs (same pattern
as the full fine-tune experiments in NT2_lora_sweep.py).

Combine modes:
    concat_diff  : [ref, alt, ref - alt]  → input_dim = 3 * proj_dim
    concat_only  : [ref, alt]             → input_dim = 2 * proj_dim
    diff_only    : ref - alt              → input_dim = proj_dim

Logging behaviour (controlled by --eval_interval, default 1000):
    Every step         : training loss is logged to console + training_loss.csv
    Every eval_interval: full eval on train+val sets → training_metrics.csv
                         + best_model.pt checkpoint (if val AUROC improved)
    Post-training      : training_metrics_final.csv, training_curves.pdf,
                         summary.json
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
import traceback
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================================
# Experiment Configs
# ============================================================================

EXPERIMENT_CONFIGS = [
    {
        "exp_id": 1,
        "description": "Ref-alt contrast Siamese MLP, combine=concat_diff",
        "combine_mode": "concat_diff",      # [ref, alt, ref - alt]
        "lora_rank": 32,
        "batch_size": 8,                    # 8 total / 4 GPUs = 2 per GPU
        "num_steps": 17000,                 # steps = (epochs × samples) / effective_batch_size = (3.6 × 151,015) / 32 = ~17,000 steps
        "gradient_accumulation_steps": 4,   # effective batch = 8 * 4 = 32
        "learning_rate": 5e-5,
    }
]


# ============================================================================
# MLP Projection Head
# ============================================================================

class MLPProjector(nn.Module):
    """
    Projects a single token embedding (1024-d) from the NT2 backbone
    down to a lower-dimensional feature vector used for combining
    ref and alt arms.

    Follows the same Linear → ReLU → Dropout → Linear pattern
    as MLPClassifier in NT2_lora_sweep.py.
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
    2-layer MLP classification head, identical structure to MLPClassifier
    in NT2_lora_sweep.py. Operates on combined ref+alt features.

    Combine mode determines input_dim:
        concat_diff : 3 * proj_dim
        concat_only : 2 * proj_dim
        diff_only   : proj_dim
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


# ============================================================================
# Dual-Input Model
# ============================================================================

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

    Memory note: the ref hidden state tensor (batch × 2000 × 1024, ~128 MB
    for batch=4) is extracted and explicitly freed before the alt forward pass
    runs. Without this, both full hidden state tensors live in GPU memory
    simultaneously, doubling peak usage. The .clone() before del is required
    because a tensor slice still holds a reference to the original storage —
    del without clone() would not actually free the memory.
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

        print("=" * 80)
        print(f"LoRA Configuration (rank={lora_rank}):")
        self.bert.print_trainable_parameters()
        print("=" * 80)

        # ---- Shared MLP projector (same weights for both arms) ----
        self.projector = MLPProjector(
            input_dim=1024, proj_dim=proj_dim, dropout=dropout
        )

        # ---- MLP classification head ----
        if combine_mode == 'concat_diff':
            head_input_dim = 3 * proj_dim   # [ref, alt, ref - alt]
        elif combine_mode == 'concat_only':
            head_input_dim = 2 * proj_dim   # [ref, alt]
        elif combine_mode == 'diff_only':
            head_input_dim = proj_dim       # ref - alt
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")

        self.classifier = MLPClassifierHead(input_dim=head_input_dim, dropout=dropout)

    def _encode(self, input_ids):
        """Run input_ids through NT2+LoRA. Returns last hidden state (batch, seq_len, 1024)."""
        attention_mask = (input_ids != 1)   # pad token id = 1 for NT2
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
        # Variant is centered in a 12 kb window; with k=6 merization:
        # 6000 bp / 6 = 1000 tokens upstream + 1 CLS token → position 1000
        variant_position = 1000

        # ---- Ref arm ----
        # Extract just the variant-position token, then immediately free the
        # full hidden state (~128 MB) before the alt forward pass allocates.
        # .clone() is required: a plain slice still references the original
        # storage, so del without clone() would not free the memory.
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
            combined = torch.cat(
                [ref_feat, alt_feat, ref_feat - alt_feat], dim=1
            )
        elif self.combine_mode == 'concat_only':
            combined = torch.cat([ref_feat, alt_feat], dim=1)
        elif self.combine_mode == 'diff_only':
            combined = ref_feat - alt_feat
        else:
            raise ValueError(f"Unknown combine_mode: {self.combine_mode}")

        return self.classifier(combined)   # (batch,)


# ============================================================================
# Dataset
# ============================================================================

def kmerize(sequence: str, k: int) -> str:
    """Split a DNA sequence into space-separated k-mers."""
    return ' '.join(sequence[i:i+k] for i in range(0, len(sequence), k))


class DualSequenceDataset(Dataset):
    """
    Reads both ref_sequence and alt_sequence from the TSV.
    Expects columns: ref_sequence, alt_sequence, label.
    Both sequences are k-merized before tokenization.
    """

    def __init__(self, ref_sequences, alt_sequences, labels, tokenizer,
                 max_length=2048):
        self.ref_sequences = ref_sequences.reset_index(drop=True)
        self.alt_sequences = alt_sequences.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ref_seq = self.ref_sequences.iloc[idx]
        alt_seq = self.alt_sequences.iloc[idx]
        label = float(self.labels.iloc[idx])

        ref_enc = self.tokenizer.encode_plus(
            ref_seq,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        alt_enc = self.tokenizer.encode_plus(
            alt_seq,
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
            'labels': torch.tensor(label, dtype=torch.float),
        }


def create_dual_data_loader(file_path, tokenizer, batch_size, max_length,
                             shuffle=False, k=6):
    """
    Load TSV, k-merize ref_sequence and alt_sequence, return a DataLoader.
    Required TSV columns: ref_sequence, alt_sequence, label.
    """
    data = pd.read_csv(file_path, delimiter='\t')

    required = {'ref_sequence', 'alt_sequence', 'label'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"TSV is missing required columns: {missing}\n"
            f"Found columns: {list(data.columns)}"
        )

    ref_sequences = data['ref_sequence'].apply(lambda x: kmerize(x, k))
    alt_sequences = data['alt_sequence'].apply(lambda x: kmerize(x, k))
    labels = data['label']

    dataset = DualSequenceDataset(
        ref_sequences, alt_sequences, labels, tokenizer, max_length
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================================
# Training Utilities
# ============================================================================

class LRSchedulerWithWarmup:
    """Linear warmup + linear decay."""
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            remaining = self.total_steps - self.warmup_steps
            elapsed = self.current_step - self.warmup_steps
            scale = max(0.0, 1.0 - elapsed / remaining)
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base * scale

    def get_last_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]


class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        improved = (
            score > self.best_score + self.min_delta
            if self.mode == 'max'
            else score < self.best_score - self.min_delta
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def evaluate_model(model, dataloader, criterion, device):
    """
    Full eval pass over an entire dataloader. Returns (avg_loss, auroc, auprc).

    This is EXPENSIVE — it runs the full NT2 backbone over every sample in
    the dataset. Do not call at every step. Use eval_interval to control
    how often this is triggered (default 1000 steps).
    """
    model.eval()
    total_loss = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch in dataloader:
            ref_ids = batch['ref_input_ids'].to(device)
            alt_ids = batch['alt_input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(ref_ids, alt_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    return avg_loss, auroc, auprc


# ============================================================================
# Training Loop
# ============================================================================

def train_experiment(exp_config, train_path, val_path, output_dir,
                     gpu_list, batch_size=8, num_steps=8000,
                     learning_rate=3e-5, k=6, gradient_accumulation_steps=4,
                     eval_interval=1000):
    """
    Train one ref-alt contrast experiment across all GPUs in gpu_list
    using nn.DataParallel.

    Output files written per experiment (inside exp_dir):
        training_loss.csv        — one row per optimizer step: step, batch loss, lr
                                   (cheap: reuses loss already computed during training,
                                    no extra forward passes)
        training_metrics.csv     — one row per eval_interval: step, train/val
                                   loss/AUROC/AUPRC, lr, gpu_memory
                                   (expensive: two full forward passes over both datasets)
        best_model.pt            — saved whenever val AUROC improves (at eval points)
        training_metrics_final.csv — same as training_metrics.csv, written at end
        training_curves.pdf      — train vs val loss/AUROC/AUPRC plots
        summary.json             — final best metrics + config
        training.log             — full console log

    gpu_list    : list of GPU IDs, e.g. [0, 1, 2, 3].
                  First GPU is primary; DataParallel splits each batch across all.
    batch_size  : total batch size across all GPUs.
                  Each GPU receives batch_size // len(gpu_list) samples.
    eval_interval : how often (optimizer steps) to run full train+val evaluation.
                  Recommended values:
                    1000 — production runs (default)
                     100 — closer monitoring without too much overhead
                      10 — only for very short debug runs; will slow training
                           significantly on large datasets
    """

    exp_id = exp_config['exp_id']
    combine_mode = exp_config['combine_mode']
    lora_rank = exp_config['lora_rank']
    primary_gpu = gpu_list[0]

    exp_dir = Path(output_dir) / f"exp_{exp_id}_{combine_mode}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ---- Logging ----
    logger = logging.getLogger(f"Exp{exp_id}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for h in [logging.FileHandler(exp_dir / 'training.log'),
              logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt)
        logger.addHandler(h)

    logger.info(f"Experiment {exp_id}: {exp_config['description']}")
    logger.info(f"Combine mode   : {combine_mode}")
    logger.info(f"GPU list       : {gpu_list}  (DataParallel, primary={primary_gpu})")
    logger.info(f"Batch size     : {batch_size} total "
                f"({batch_size // len(gpu_list)} per GPU)")
    logger.info(f"LR             : {learning_rate}")
    logger.info(f"Grad accum     : {gradient_accumulation_steps}")
    logger.info(f"Effective batch: {batch_size * gradient_accumulation_steps}")
    logger.info(f"Eval interval  : every {eval_interval} steps "
                f"(training loss logged every step)")

    # ---- Device setup ----
    device = torch.device(f'cuda:{primary_gpu}')
    torch.cuda.set_device(primary_gpu)
    for g in gpu_list:
        torch.cuda.reset_peak_memory_stats(g)
    torch.cuda.empty_cache()

    start_time = time.time()

    try:
        # ---- Load model ----
        logger.info("Loading NT2 tokenizer and base model...")
        tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
            trust_remote_code=True,
        )
        base_model = AutoModelForMaskedLM.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
            trust_remote_code=True,
        )

        model = NT2_RefAltContrast(
            base_model, lora_rank=lora_rank, combine_mode=combine_mode
        )
        model = model.to(device)

        # ---- Wrap with DataParallel across all GPUs ----
        # Mirrors the full fine-tune pattern in NT2_lora_sweep.py.
        # DataParallel splits each batch across gpu_list and gathers on primary_gpu.
        use_multi_gpu = len(gpu_list) > 1
        if use_multi_gpu:
            model = nn.DataParallel(model, device_ids=gpu_list)
            logger.info(f"DataParallel enabled across GPUs: {gpu_list}")

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {n_trainable:,}")

        memory_after_model = max(
            torch.cuda.memory_allocated(g) / 1024**3 for g in gpu_list
        )
        logger.info(f"Peak GPU memory after model load: {memory_after_model:.2f} GB")

        # ---- Data ----
        logger.info("Creating dual-input dataloaders...")
        train_loader = create_dual_data_loader(
            train_path, tokenizer, batch_size, max_length=2048,
            shuffle=True, k=k
        )
        val_loader = create_dual_data_loader(
            val_path, tokenizer, batch_size, max_length=2048,
            shuffle=False, k=k
        )

        # ---- Optimizer / scheduler ----
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate,
                          eps=1e-8, betas=(0.9, 0.999))

        warmup_steps = int(0.06 * num_steps)
        scheduler = LRSchedulerWithWarmup(optimizer, warmup_steps, num_steps)
        criterion = nn.BCEWithLogitsLoss()
        early_stopping = EarlyStopping(patience=2, mode='max')

        # ---- Output files ----
        #
        # training_loss.csv : written every optimizer step (cheap)
        #   columns: step, train_loss_batch, learning_rate
        #   "train_loss_batch" is the loss on the current mini-batch only,
        #   not the full dataset. Use this to track training progress between
        #   eval checkpoints without paying the cost of a full eval pass.
        #
        # training_metrics.csv : written every eval_interval steps (expensive)
        #   columns: steps, train_loss, train_auroc, train_auprc,
        #            val_loss, val_auroc, val_auprc, learning_rate, gpu_memory_gb
        #   train_loss here is the average over the full training set (not batch).
        #
        train_loss_csv = exp_dir / 'training_loss.csv'
        metrics_csv = exp_dir / 'training_metrics.csv'

        pd.DataFrame(columns=['step', 'train_loss_batch', 'learning_rate']
                     ).to_csv(train_loss_csv, index=False)
        pd.DataFrame(columns=[
            'steps', 'train_loss', 'train_auroc', 'train_auprc',
            'val_loss', 'val_auroc', 'val_auprc',
            'learning_rate', 'gpu_memory_gb',
        ]).to_csv(metrics_csv, index=False)

        # In-memory accumulator for final metrics_final.csv and plots
        eval_metrics = {k_: [] for k_ in [
            'steps', 'train_loss', 'train_auroc', 'train_auprc',
            'val_loss', 'val_auroc', 'val_auprc',
            'learning_rate', 'gpu_memory_gb',
        ]}

        # ---- Training ----
        logger.info(f"Training for {num_steps} steps ({warmup_steps} warmup steps)")
        global_step = 0
        best_val_auroc = 0.0
        best_val_auprc = 0.0
        accum_counter = 0
        train_iter = iter(train_loader)

        while global_step < num_steps:
            model.train()

            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            ref_ids = batch['ref_input_ids'].to(device)
            alt_ids = batch['alt_input_ids'].to(device)
            labels = batch['labels'].to(device)

            if accum_counter == 0:
                optimizer.zero_grad()

            outputs = model(ref_ids, alt_ids)
            loss = criterion(outputs, labels) / gradient_accumulation_steps
            loss.backward()

            accum_counter += 1
            if accum_counter == gradient_accumulation_steps:
                optimizer.step()
                scheduler.step()
                accum_counter = 0
                global_step += 1

                # ---- Log training loss every step (cheap) ----
                # loss was divided by grad_accum above; multiply back for logging
                batch_loss = loss.item() * gradient_accumulation_steps
                current_lr = scheduler.get_last_lr()[0]

                logger.info(
                    f"Step {global_step}/{num_steps} | "
                    f"train loss: {batch_loss:.4f} | lr: {current_lr:.2e}"
                )
                pd.DataFrame(
                    [[global_step, batch_loss, current_lr]],
                    columns=['step', 'train_loss_batch', 'learning_rate']
                ).to_csv(train_loss_csv, mode='a', header=False, index=False)

                # ---- Full eval every eval_interval steps (expensive) ----
                # Runs the entire train and val datasets through the model.
                # Produces AUROC/AUPRC + saves checkpoint if improved.
                # Keep eval_interval >= 100 for large datasets to avoid
                # spending more time evaluating than training.
                if global_step % eval_interval == 0:
                    current_mem = max(
                        torch.cuda.memory_allocated(g) / 1024**3 for g in gpu_list
                    )

                    train_loss, train_auroc, train_auprc = evaluate_model(
                        model, train_loader, criterion, device)
                    val_loss, val_auroc, val_auprc = evaluate_model(
                        model, val_loader, criterion, device)

                    eval_metrics['steps'].append(global_step)
                    eval_metrics['train_loss'].append(train_loss)
                    eval_metrics['train_auroc'].append(train_auroc)
                    eval_metrics['train_auprc'].append(train_auprc)
                    eval_metrics['val_loss'].append(val_loss)
                    eval_metrics['val_auroc'].append(val_auroc)
                    eval_metrics['val_auprc'].append(val_auprc)
                    eval_metrics['learning_rate'].append(current_lr)
                    eval_metrics['gpu_memory_gb'].append(current_mem)

                    row = pd.DataFrame(
                        {k_: [v[-1]] for k_, v in eval_metrics.items()}
                    )
                    row.to_csv(metrics_csv, mode='a', header=False, index=False)

                    logger.info(
                        f"[EVAL] Step {global_step}/{num_steps} | "
                        f"Train AUROC: {train_auroc:.4f}, AUPRC: {train_auprc:.4f} | "
                        f"Val AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f} | "
                        f"GPU mem (max): {current_mem:.2f} GB"
                    )

                    if val_auroc > best_val_auroc:
                        best_val_auroc = val_auroc
                        best_val_auprc = val_auprc
                        # Unwrap DataParallel before saving — mirrors NT2_lora_sweep.py
                        model_to_save = model.module if use_multi_gpu else model
                        torch.save({
                            'step': global_step,
                            'model_state_dict': model_to_save.state_dict(),
                            'val_auroc': val_auroc,
                            'val_auprc': val_auprc,
                            'config': exp_config,
                            'gpu_list': gpu_list,
                        }, exp_dir / 'best_model.pt')
                        logger.info(
                            f"  -> New best: AUROC={val_auroc:.4f}, "
                            f"AUPRC={val_auprc:.4f}"
                        )

                    if early_stopping(val_auroc):
                        logger.info(
                            f"Early stopping triggered at step {global_step}"
                        )
                        break

                    model.train()

        # ---- Post-training ----
        metrics_df = pd.DataFrame(eval_metrics)
        metrics_df.to_csv(exp_dir / 'training_metrics_final.csv', index=False)

        if not metrics_df.empty:
            generate_plots(metrics_df, exp_dir, exp_id, combine_mode)
        else:
            logger.warning(
                "No eval checkpoints recorded — increase num_steps or "
                "decrease eval_interval so at least one eval is triggered."
            )

        peak_gpu = max(
            torch.cuda.max_memory_allocated(g) / 1024**3 for g in gpu_list
        )
        total_time = time.time() - start_time

        summary = {
            'exp_id': exp_id,
            'combine_mode': combine_mode,
            'config': exp_config,
            'gpu_list': gpu_list,
            'best_val_auroc': best_val_auroc,
            'best_val_auprc': best_val_auprc,
            'total_time_hours': total_time / 3600,
            'peak_gpu_memory_gb': peak_gpu,
            'total_steps': global_step,
            'effective_batch_size': batch_size * gradient_accumulation_steps,
            'eval_interval': eval_interval,
        }
        with open(exp_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(
            f"Experiment {exp_id} ({combine_mode}) done: "
            f"AUROC={best_val_auroc:.4f}, AUPRC={best_val_auprc:.4f}, "
            f"time={total_time / 3600:.2f}h, peak GPU={peak_gpu:.2f} GB"
        )
        return summary

    except Exception as e:
        logger.error(f"Error in experiment {exp_id}: {e}")
        logger.error(traceback.format_exc())
        raise


# ============================================================================
# Plotting
# ============================================================================

def generate_plots(metrics_df, exp_dir, exp_id, combine_mode):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Exp {exp_id} ({combine_mode}) Train vs Val', fontsize=14)

    for ax, metric, label in zip(
        axes, ['loss', 'auroc', 'auprc'], ['Loss', 'AUROC', 'AUPRC']
    ):
        tcol, vcol = f'train_{metric}', f'val_{metric}'
        if tcol in metrics_df.columns and vcol in metrics_df.columns:
            ax.plot(metrics_df['steps'], metrics_df[tcol], label='Train', linewidth=2)
            ax.plot(metrics_df['steps'], metrics_df[vcol], label='Val', linewidth=2)
            ax.set_xlabel('Steps')
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Parallel Runner
# ============================================================================

def run_single_experiment(args):
    exp_config, train_path, val_path, output_dir, gpu_list, hyperparams = args
    return train_experiment(
        exp_config, train_path, val_path, output_dir, gpu_list, **hyperparams
    )


def run_all_experiments(train_path, val_path, output_dir,
                        available_gpus=(0, 1, 2, 3), k=6,
                        batch_size_override=None, grad_accum_override=None,
                        eval_interval=1000, gpus_per_experiment=4):
    """
    Run all experiments in EXPERIMENT_CONFIGS.

    gpus_per_experiment : how many GPUs to allocate per experiment via DataParallel.
        4 (default) — all 4 GPUs to one experiment (current single-experiment use case)
        1           — one GPU per experiment, run up to 4 in parallel (original sweep mode)
        2           — two GPUs per experiment, run 2 in parallel

    GPU assignment is round-robin across available_gpus in blocks of gpus_per_experiment.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'main.log'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger('Main')
    logger.info(f"Running {len(EXPERIMENT_CONFIGS)} experiment(s)")
    logger.info(f"Available GPUs     : {list(available_gpus)}")
    logger.info(f"GPUs per experiment: {gpus_per_experiment}")
    logger.info(f"Eval interval      : {eval_interval} steps")

    # Assign a contiguous GPU block to each experiment
    experiment_args = []
    for i, cfg in enumerate(EXPERIMENT_CONFIGS):
        start = (i * gpus_per_experiment) % len(available_gpus)
        gpu_list = [available_gpus[(start + j) % len(available_gpus)]
                    for j in range(gpus_per_experiment)]
        hyperparams = {
            'batch_size': batch_size_override or cfg['batch_size'],
            'num_steps': cfg['num_steps'],
            'learning_rate': cfg['learning_rate'],
            'gradient_accumulation_steps': (grad_accum_override
                                            or cfg['gradient_accumulation_steps']),
            'k': k,
            'eval_interval': eval_interval,
        }
        experiment_args.append(
            (cfg, train_path, val_path, str(output_dir), gpu_list, hyperparams)
        )
        logger.info(
            f"  exp_{cfg['exp_id']} ({cfg['combine_mode']}) → GPUs {gpu_list}"
        )

    # How many experiments can run simultaneously given the GPU budget
    num_parallel = max(1, len(available_gpus) // gpus_per_experiment)
    all_summaries = []

    for i in range(0, len(experiment_args), num_parallel):
        batch = experiment_args[i:i + num_parallel]
        logger.info(
            f"Batch {i // num_parallel + 1}: "
            f"{[a[0]['combine_mode'] for a in batch]}"
        )
        if len(batch) == 1:
            # Single experiment — call directly, no subprocess overhead
            all_summaries.append(run_single_experiment(batch[0]))
        else:
            with mp.Pool(processes=len(batch)) as pool:
                summaries = pool.map(run_single_experiment, batch)
                all_summaries.extend(summaries)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(output_dir / 'all_experiments_summary.csv', index=False)

    logger.info("=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    for s in all_summaries:
        logger.info(
            f"  {s['combine_mode']:15s} | "
            f"AUROC: {s['best_val_auroc']:.4f} | "
            f"AUPRC: {s['best_val_auprc']:.4f} | "
            f"Time: {s['total_time_hours']:.2f}h"
        )
    return summary_df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NT2 Ref-Alt Contrast — Siamese LoRA Fine-Tuning with MLP Head'
    )
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training TSV (needs ref_sequence, alt_sequence, label)')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to validation TSV')
    parser.add_argument('--output_dir', type=str,
                        default='./results/NT2_ref_alt_contrast',
                        help='Output directory')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='GPU IDs to use (default: 0 1 2 3)')
    parser.add_argument('--gpus_per_experiment', type=int, default=4,
                        help='GPUs per experiment via DataParallel (default: 4). '
                             'Set to 1 for one-GPU-per-experiment sweep mode.')
    parser.add_argument('--k', type=int, default=6,
                        help='K-mer size (default: 6)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override total batch size (should be divisible '
                             'by gpus_per_experiment)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                        help='Override gradient accumulation steps')
    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Run full train+val evaluation every N optimizer steps '
                             '(default: 1000). Training loss is logged every step for free. '
                             'Recommended: 1000 for production, 100 for closer monitoring. '
                             'Avoid values < 100 on large datasets — each eval re-runs '
                             'the full NT2 backbone over all training samples.')

    args = parser.parse_args()

    if args.batch_size and args.batch_size % args.gpus_per_experiment != 0:
        print(f"WARNING: batch_size={args.batch_size} is not divisible by "
              f"gpus_per_experiment={args.gpus_per_experiment}.")

    print("=" * 80)
    print("NT2 Ref-Alt Contrast — Siamese MLP Head")
    print(f"  Backbone         : NT2 500M multi-species + LoRA")
    print(f"  Features         : variant-position token → shared MLPProjector")
    print(f"  Head             : 2-layer MLPClassifierHead on combined ref+alt features")
    print(f"  GPUs             : {args.gpus}")
    print(f"  GPUs/experiment  : {args.gpus_per_experiment}")
    print(f"  Eval interval    : every {args.eval_interval} steps")
    print(f"  Experiments      :")
    for cfg in EXPERIMENT_CONFIGS:
        print(f"    exp_{cfg['exp_id']}: {cfg['combine_mode']} — {cfg['description']}")
    print("=" * 80)

    run_all_experiments(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        available_gpus=args.gpus,
        k=args.k,
        batch_size_override=args.batch_size,
        grad_accum_override=args.gradient_accumulation_steps,
        eval_interval=args.eval_interval,
        gpus_per_experiment=args.gpus_per_experiment,
    )

    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()