#!/usr/bin/env python3
"""
Dual-Input (Reference + Alternate) NT2 LoRA Fine-Tuning
========================================================
Siamese-style architecture: both ref and alt sequences are encoded
through the shared NT2+LoRA backbone, variant-position embeddings are
extracted and passed through a shared 2-layer MLP feature extractor,
then [ref, alt, ref - alt] are concatenated and passed to a
classification head.

Fixed config: NT2 + 2-layer MLP + variant_position + LoRA rank 32


"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
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
        "description": "Dual-input (ref+alt) Siamese, combine=[concat, diff]",
        "combine_mode": "concat_diff",  # [ref, alt, ref - alt]
        "lora_rank": 32,
        "batch_size": 2,
        "num_steps": 4000,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-5,
    },
    #{
    #    "exp_id": 2,
    #    "description": "Dual-input (ref+alt) Siamese, combine=[diff only]",
    #    "combine_mode": "diff_only",  # ref - alt
    #    "lora_rank": 32,
    #    "batch_size": 4,
    #    "num_steps": 8000,
    #    "gradient_accumulation_steps": 8,
    #    "learning_rate": 3e-5,
    #},
    {
        "exp_id": 2,
        "description": "Dual-input (ref+alt) Siamese, combine=[concat only]",
        "combine_mode": "concat_only",  # [ref, alt]
        "lora_rank": 32,
        "batch_size": 2,
        "num_steps": 4000,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-5,
    },
]


# ============================================================================
# MLP Feature Extractor (shared between ref and alt arms)
# ============================================================================

class MLPFeatureExtractor(nn.Module):
    """2-layer MLP that extracts features at the variant position.
    Extracts the hidden state at the variant token, then projects
    through two linear layers to produce a feature vector."""
    def __init__(self, input_dim=1024, out_dim=256, dropout=0.1):
        super(MLPFeatureExtractor, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x, variant_position):
        """
        Args:
            x: (batch, seq_len, input_dim)
            variant_position: int, token index of the variant
        Returns:
            features: (batch, out_dim) at the variant position
        """
        # Extract embedding at variant position
        pooled = x[:, variant_position, :]  # (batch, input_dim)
        features = self.feature_extractor(pooled)  # (batch, out_dim)
        return features


# ============================================================================
# Dual-Input Model Wrapper
# ============================================================================

class NT2_DualInput(nn.Module):
    """Siamese NT2 model: shared backbone + MLP for ref and alt sequences,
    combined features fed to a classification head."""

    def __init__(self, base_model, lora_rank=32, combine_mode='concat_diff',
                 mlp_out_dim=256):
        super(NT2_DualInput, self).__init__()

        self.combine_mode = combine_mode
        self.mlp_out_dim = mlp_out_dim

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
            bias="none"
        )
        self.bert = get_peft_model(self.bert, lora_config)
        self.bert.enable_input_require_grads()

        print("=" * 80)
        print(f"LoRA Configuration (rank={lora_rank}):")
        self.bert.print_trainable_parameters()
        print("=" * 80)

        # ---- Shared MLP feature extractor ----
        self.mlp = MLPFeatureExtractor(input_dim=1024, out_dim=mlp_out_dim)

        # ---- Classification head ----
        # Input dim depends on combine mode
        if combine_mode == 'concat_diff':
            # [ref, alt, ref - alt] → 3 * mlp_out_dim
            head_input_dim = 3 * mlp_out_dim
        elif combine_mode == 'concat_only':
            # [ref, alt] → 2 * mlp_out_dim
            head_input_dim = 2 * mlp_out_dim
        elif combine_mode == 'diff_only':
            # ref - alt → mlp_out_dim
            head_input_dim = mlp_out_dim
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")

        self.classifier = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def _encode(self, input_ids):
        """Run a sequence through NT2 backbone, return hidden states."""
        attention_mask = input_ids != 1  # pad token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
        return outputs['hidden_states'][-1]  # (batch, seq_len, 1024)

    def forward(self, ref_input_ids, alt_input_ids):
        """
        Args:
            ref_input_ids: (batch, seq_len) tokenized reference sequence
            alt_input_ids: (batch, seq_len) tokenized alternate sequence
        Returns:
            logits: (batch,)
        """
        # Variant position in tokenized space
        variant_position = 1000  # 5999 // 6 + 1 (CLS token)

        # Encode both sequences through shared backbone
        ref_embed = self._encode(ref_input_ids)  # (batch, seq_len, 1024)
        alt_embed = self._encode(alt_input_ids)   # (batch, seq_len, 1024)

        # Adjust variant position if needed
        seq_len = ref_embed.shape[1]
        if variant_position >= seq_len:
            variant_position = seq_len // 2

        # Extract MLP features at variant position
        ref_features = self.mlp(ref_embed, variant_position)  # (batch, 256)
        alt_features = self.mlp(alt_embed, variant_position)  # (batch, 256)

        # Combine features
        if self.combine_mode == 'concat_diff':
            combined = torch.cat([ref_features, alt_features,
                                  ref_features - alt_features], dim=1)
        elif self.combine_mode == 'concat_only':
            combined = torch.cat([ref_features, alt_features], dim=1)
        elif self.combine_mode == 'diff_only':
            combined = ref_features - alt_features
        else:
            raise ValueError(f"Unknown combine_mode: {self.combine_mode}")

        logits = self.classifier(combined).squeeze(-1)
        return logits


# ============================================================================
# Dataset (dual-sequence)
# ============================================================================

class DualSequenceDataset(Dataset):
    """Dataset that returns both ref and alt sequences."""
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
            ref_seq, add_special_tokens=True, max_length=self.max_length,
            return_token_type_ids=False, padding='max_length',
            return_attention_mask=True, return_tensors='pt', truncation=True
        )
        alt_enc = self.tokenizer.encode_plus(
            alt_seq, add_special_tokens=True, max_length=self.max_length,
            return_token_type_ids=False, padding='max_length',
            return_attention_mask=True, return_tensors='pt', truncation=True
        )

        return {
            'ref_input_ids': ref_enc['input_ids'].flatten(),
            'alt_input_ids': alt_enc['input_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float),
        }


def kmerize(sequence, k):
    return ' '.join([sequence[i:i+k] for i in range(0, len(sequence), k)])


def create_dual_data_loader(file_path, tokenizer, batch_size, max_length,
                             shuffle=False, k=6):
    """Create DataLoader with both ref and alt sequences."""
    data = pd.read_csv(file_path, delimiter='\t')

    ref_sequences = data['ref_sequence'].apply(lambda x: kmerize(x, k))
    alt_sequences = data['alt_sequence'].apply(lambda x: kmerize(x, k))
    labels = data['label']

    dataset = DualSequenceDataset(ref_sequences, alt_sequences, labels,
                                   tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================================
# Training Utilities
# ============================================================================

class LRSchedulerWithWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr_scale = self.current_step / self.warmup_steps
        else:
            remaining = self.total_steps - self.warmup_steps
            elapsed = self.current_step - self.warmup_steps
            lr_scale = max(0.0, 1.0 - elapsed / remaining)
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * lr_scale

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate dual-input model. Returns (loss, auroc, auprc)."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            ref_ids = batch['ref_input_ids'].to(device)
            alt_ids = batch['alt_input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(ref_ids, alt_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probabilities = torch.sigmoid(outputs)
            all_preds.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    return avg_loss, auroc, auprc


# ============================================================================
# Training Loop
# ============================================================================

def train_experiment(exp_config, train_path, val_path, output_dir,
                     gpu_id, batch_size=4, num_steps=4000,
                     learning_rate=3e-5, k=6, gradient_accumulation_steps=1):
    """Train a single dual-input experiment."""

    exp_id = exp_config['exp_id']
    combine_mode = exp_config['combine_mode']
    lora_rank = exp_config['lora_rank']

    exp_dir = Path(output_dir) / f"exp_{exp_id}_{combine_mode}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ---- Logging ----
    logger = logging.getLogger(f"Exp{exp_id}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(logging.FileHandler(exp_dir / 'training.log'))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for h in logger.handlers:
        h.setFormatter(fmt)

    logger.info(f"Starting Experiment {exp_id}: {exp_config['description']}")
    logger.info(f"Combine mode: {combine_mode}")
    logger.info(f"GPU: {gpu_id}, Batch size: {batch_size}, LR: {learning_rate}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}, "
                f"Effective batch size: {batch_size * gradient_accumulation_steps}")

    device = torch.device(f'cuda:{gpu_id}')
    start_time = time.time()

    torch.cuda.set_device(gpu_id)
    torch.cuda.reset_peak_memory_stats(gpu_id)
    torch.cuda.empty_cache()

    try:
        # ---- Load model ----
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
            trust_remote_code=True
        )
        base_model = AutoModelForMaskedLM.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
            trust_remote_code=True
        )

        model = NT2_DualInput(base_model, lora_rank=lora_rank,
                              combine_mode=combine_mode)
        model = model.to(device)

        logger.info(f"Trainable parameters: "
                    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        memory_after_model = torch.cuda.memory_allocated(gpu_id) / 1024**3
        logger.info(f"GPU memory after model loading: {memory_after_model:.2f} GB")

        # ---- Data ----
        logger.info("Creating dual-input dataloaders...")
        train_loader = create_dual_data_loader(
            train_path, tokenizer, batch_size, 2048, shuffle=True, k=k)
        val_loader = create_dual_data_loader(
            val_path, tokenizer, batch_size, 2048, shuffle=False, k=k)

        # ---- Optimizer / Scheduler ----
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate,
                          eps=1e-8, betas=(0.9, 0.999))

        warmup_steps = int(0.06 * num_steps)
        scheduler = LRSchedulerWithWarmup(optimizer, warmup_steps, num_steps)
        criterion = nn.BCEWithLogitsLoss()
        early_stopping = EarlyStopping(patience=2, mode='max')

        # ---- Metrics ----
        metrics = {
            'steps': [], 'train_loss': [], 'train_auroc': [], 'train_auprc': [],
            'val_loss': [], 'val_auroc': [], 'val_auprc': [],
            'learning_rate': [], 'gpu_memory_gb': []
        }
        metrics_csv_path = exp_dir / 'training_metrics.csv'
        pd.DataFrame(columns=list(metrics.keys())).to_csv(metrics_csv_path, index=False)

        # ---- Training ----
        logger.info(f"Training for {num_steps} steps ({warmup_steps} warmup)")
        global_step = 0
        best_val_auroc = 0
        best_val_auprc = 0
        accumulation_counter = 0
        train_iterator = iter(train_loader)

        while global_step < num_steps:
            model.train()

            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            ref_ids = batch['ref_input_ids'].to(device)
            alt_ids = batch['alt_input_ids'].to(device)
            labels = batch['labels'].to(device)

            if accumulation_counter == 0:
                optimizer.zero_grad()

            outputs = model(ref_ids, alt_ids)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            accumulation_counter += 1

            if accumulation_counter == gradient_accumulation_steps:
                optimizer.step()
                scheduler.step()
                accumulation_counter = 0
                global_step += 1

            # Evaluate every 1000 steps
            if global_step > 0 and global_step % 1000 == 0 and accumulation_counter == 0:
                current_memory = torch.cuda.memory_allocated(gpu_id) / 1024**3

                train_loss, train_auroc, train_auprc = evaluate_model(
                    model, train_loader, criterion, device)
                val_loss, val_auroc, val_auprc = evaluate_model(
                    model, val_loader, criterion, device)

                metrics['steps'].append(global_step)
                metrics['train_loss'].append(train_loss)
                metrics['train_auroc'].append(train_auroc)
                metrics['train_auprc'].append(train_auprc)
                metrics['val_loss'].append(val_loss)
                metrics['val_auroc'].append(val_auroc)
                metrics['val_auprc'].append(val_auprc)
                metrics['learning_rate'].append(scheduler.get_last_lr()[0])
                metrics['gpu_memory_gb'].append(current_memory)

                row = pd.DataFrame({k_: [v[-1]] for k_, v in metrics.items()})
                row.to_csv(metrics_csv_path, mode='a', header=False, index=False)

                logger.info(
                    f"Step {global_step}/{num_steps} | "
                    f"Train AUROC: {train_auroc:.4f}, AUPRC: {train_auprc:.4f} | "
                    f"Val AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"GPU Mem: {current_memory:.2f} GB"
                )

                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    best_val_auprc = val_auprc
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'val_auroc': val_auroc,
                        'val_auprc': val_auprc,
                        'config': exp_config,
                    }, exp_dir / 'best_model.pt')
                    logger.info(f"  -> New best: AUROC={val_auroc:.4f}, AUPRC={val_auprc:.4f}")

                if early_stopping(val_auroc):
                    logger.info(f"Early stopping at step {global_step}")
                    break

                model.train()

        # ---- Post-training ----
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(exp_dir / 'training_metrics_final.csv', index=False)

        generate_plots(metrics_df, exp_dir, exp_id, combine_mode)

        peak_gpu = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
        total_time = time.time() - start_time

        summary = {
            'exp_id': exp_id,
            'combine_mode': combine_mode,
            'config': exp_config,
            'best_val_auroc': best_val_auroc,
            'best_val_auprc': best_val_auprc,
            'total_time_hours': total_time / 3600,
            'peak_gpu_memory_gb': peak_gpu,
            'total_steps': global_step,
        }

        with open(exp_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Experiment {exp_id} ({combine_mode}) done: "
                    f"AUROC={best_val_auroc:.4f}, AUPRC={best_val_auprc:.4f}, "
                    f"time={total_time/3600:.2f}h")

        return summary

    except Exception as e:
        logger.error(f"Error in experiment {exp_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# ============================================================================
# Plotting
# ============================================================================

def generate_plots(metrics_df, exp_dir, exp_id, combine_mode):
    """Generate training curves."""
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
            ax.set_title(f'{label}')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Parallel Runner
# ============================================================================

def run_single_experiment(args):
    exp_config, train_path, val_path, output_dir, gpu_id, hyperparams = args
    return train_experiment(
        exp_config, train_path, val_path, output_dir, gpu_id, **hyperparams
    )


def run_all_experiments(train_path, val_path, output_dir,
                        available_gpus=[0, 1, 2, 3], k=6,
                        batch_size_override=None, grad_accum_override=None):
    """Run all 3 combine-mode experiments in parallel."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'main.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('Main')

    logger.info(f"Running {len(EXPERIMENT_CONFIGS)} dual-input experiments")
    logger.info(f"Available GPUs: {available_gpus}")
    if batch_size_override is not None:
        logger.info(f"CLI batch_size override: {batch_size_override}")
    if grad_accum_override is not None:
        logger.info(f"CLI gradient_accumulation_steps override: {grad_accum_override}")

    experiment_args = []
    for i, cfg in enumerate(EXPERIMENT_CONFIGS):
        gpu_id = available_gpus[i % len(available_gpus)]
        hyperparams = {
            'batch_size': batch_size_override if batch_size_override is not None else cfg['batch_size'],
            'num_steps': cfg['num_steps'],
            'learning_rate': cfg['learning_rate'],
            'gradient_accumulation_steps': grad_accum_override if grad_accum_override is not None else cfg['gradient_accumulation_steps'],
            'k': k,
        }
        experiment_args.append((cfg, train_path, val_path,
                               str(output_dir), gpu_id, hyperparams))

    # All 3 experiments can run in parallel (one GPU each)
    num_parallel = min(len(experiment_args), len(available_gpus))
    all_summaries = []

    for i in range(0, len(experiment_args), num_parallel):
        batch = experiment_args[i:i + num_parallel]
        logger.info(f"Batch {i // num_parallel + 1}: "
                    f"{[a[0]['combine_mode'] for a in batch]}")

        with mp.Pool(processes=len(batch)) as pool:
            summaries = pool.map(run_single_experiment, batch)
            all_summaries.extend(summaries)

    # ---- Summary ----
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(output_dir / 'all_experiments_summary.csv', index=False)

    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    for s in all_summaries:
        logger.info(f"  {s['combine_mode']:15s} | "
                    f"AUROC: {s['best_val_auroc']:.4f} | "
                    f"AUPRC: {s['best_val_auprc']:.4f} | "
                    f"Time: {s['total_time_hours']:.2f}h")
    logger.info("=" * 80)

    return summary_df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Dual-Input (ref+alt) Siamese NT2 LoRA Fine-Tuning'
    )
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training data TSV (must have ref_sequence & alt_sequence)')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to validation data TSV')
    parser.add_argument('--output_dir', type=str, default='./nt2_dual_input',
                        help='Output directory')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='GPU IDs (default: 0 1 2 3)')
    parser.add_argument('--k', type=int, default=6, help='K-mer size (default: 6)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size for all experiments (default: use config value)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                        help='Override gradient accumulation steps for all experiments (default: use config value)')

    args = parser.parse_args()

    print("=" * 80)
    print("Dual-Input (Ref + Alt) Siamese NT2 Experiments")
    print("  Backbone: NT2 500M multi-species + LoRA rank 32")
    print("  Feature extractor: 2-layer MLP @ variant position")
    if args.batch_size is not None:
        print(f"  Batch size override: {args.batch_size}")
    if args.gradient_accumulation_steps is not None:
        print(f"  Gradient accumulation steps override: {args.gradient_accumulation_steps}")
    print("  Combine modes tested:")
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
    )

    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
