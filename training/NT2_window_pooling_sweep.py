#!/usr/bin/env python3
"""
Windowed Mean Pooling Sweep for Nucleotide Transformer v2
=========================================================
3 frozen-backbone experiments (CNN classifier only) varying the window size
for local mean pooling around the variant position after CNN convolutions:

  exp_66: ±8 tokens  (48 nt each side,  17 tokens total)
  exp_67: ±16 tokens (96 nt each side,  33 tokens total)
  exp_68: ±32 tokens (192 nt each side, 65 tokens total)

Each experiment uses 1 GPU → 3 concurrent jobs on 3 GPUs.
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
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import psutil
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ============================================================================
# Experiment Configurations
# ============================================================================

def build_experiment_configs():
    """Build windowed mean pooling experiment configs (frozen backbone)."""
    configs = []
    exp_id = 4

    # Window sizes (in tokens): ±8, ±16, ±32 around the variant position
    window_half_sizes = [32]

    for whs in window_half_sizes:
        configs.append({
            "exp_id": exp_id,
            "classifier_type": "cnn",
            "num_layers": 2,
            "embedding_strategy": f"window_{whs}-mean_pool",
            "window_half_size": whs,   # ± this many tokens
            "fine_tuning": "frozen",
        })
        exp_id += 1

    return configs


EXPERIMENT_CONFIGS = build_experiment_configs()


# ============================================================================
# Classifier Heads
# ============================================================================

class MLPClassifier(nn.Module):
    """MLP classifier with 2 hidden layers"""
    def __init__(self, input_dim=1024, num_layers=2, dropout=0.1):
        super(MLPClassifier, self).__init__()

        if num_layers == 2:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, 1)
            )
        elif num_layers == 3:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(128, 1)
            )
        else:
            raise ValueError(f"num_layers must be 2 or 3, got {num_layers}")

    def forward(self, x):
        return self.classifier(x).squeeze(-1)


class CNNClassifier(nn.Module):
    """CNN classifier with 2 conv layers"""
    def __init__(self, input_dim=1024, dropout=0.1, pooling_strategy='mean_pool',
                 window_half_size=None):
        super(CNNClassifier, self).__init__()

        self.pooling_strategy = pooling_strategy
        self.window_half_size = window_half_size
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(128, 1)

    def forward(self, x, variant_position=None, attention_mask=None):
        # x shape: (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # x shape: (batch, 128, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, 128)

        if self.pooling_strategy == 'mean_pool':
            if attention_mask is not None:
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = torch.sum(attention_mask_expanded * x, dim=1)
                sum_mask = torch.sum(attention_mask_expanded, dim=1)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = torch.mean(x, dim=1)
        elif self.pooling_strategy == 'variant_position':
            if variant_position is None:
                raise ValueError("variant_position must be provided for variant_position pooling")
            pooled = x[:, variant_position, :]
        elif self.pooling_strategy == 'window_mean_pool':
            # Mean pool over a local window around the variant position
            # after CNN convolutions have been applied to the full sequence.
            if variant_position is None:
                raise ValueError("variant_position must be provided for window_mean_pool")
            whs = self.window_half_size
            seq_len = x.shape[1]
            win_start = max(0, variant_position - whs)
            win_end = min(seq_len, variant_position + whs + 1)  # exclusive

            window_x = x[:, win_start:win_end, :]  # (batch, window_len, 128)

            if attention_mask is not None:
                window_mask = attention_mask[:, win_start:win_end]
                window_mask_expanded = window_mask.unsqueeze(-1).float()
                sum_embeddings = torch.sum(window_mask_expanded * window_x, dim=1)
                sum_mask = torch.sum(window_mask_expanded, dim=1)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = torch.mean(window_x, dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        logits = self.fc(pooled)
        return logits.squeeze(-1)


class TransformerClassifier(nn.Module):
    """Transformer classifier with 2 encoder layers"""
    def __init__(self, input_dim=1024, embed_dim=128, nhead=2,
                 dim_feedforward=512, dropout=0.1, pooling_strategy='mean_pool'):
        super(TransformerClassifier, self).__init__()

        self.pooling_strategy = pooling_strategy

        # Project input to embed_dim
        self.input_projection = nn.Linear(input_dim, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x, variant_position=None, attention_mask=None):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_projection(x)  # (batch, seq_len, embed_dim)

        x = self.transformer(x)

        if self.pooling_strategy == 'mean_pool':
            if attention_mask is not None:
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = torch.sum(attention_mask_expanded * x, dim=1)
                sum_mask = torch.sum(attention_mask_expanded, dim=1)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = torch.mean(x, dim=1)
        elif self.pooling_strategy == 'variant_position':
            if variant_position is None:
                raise ValueError("variant_position must be provided for variant_position pooling")
            pooled = x[:, variant_position, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        pooled = self.dropout(pooled)

        logits = self.fc(pooled)
        return logits.squeeze(-1)


# ============================================================================
# Model Wrapper
# ============================================================================

class NT2_FineTune(nn.Module):
    """Nucleotide Transformer fine-tuning wrapper with windowed mean pooling."""

    def __init__(self, base_model, classifier_type, num_layers,
                 embedding_strategy, freeze_base=True,
                 window_half_size=None):
        super(NT2_FineTune, self).__init__()

        self.bert = base_model
        self.embedding_strategy = embedding_strategy
        self.classifier_type = classifier_type
        self.window_half_size = window_half_size

        # Freeze base model parameters
        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Initialize classifier head
        input_dim = 1024  # NT2 hidden dimension

        # Determine pooling strategy for CNN/Transformer
        if embedding_strategy.startswith('full-'):
            pooling_strategy = embedding_strategy.split('-', 1)[1]
        elif embedding_strategy.startswith('window_'):
            # e.g. "window_8-mean_pool" → CNN uses window_mean_pool after convolutions
            pooling_strategy = 'window_mean_pool'
        else:
            pooling_strategy = 'mean_pool'  # default

        if classifier_type == "mlp":
            self.classifier = MLPClassifier(input_dim, num_layers)
        elif classifier_type == "cnn":
            self.classifier = CNNClassifier(
                input_dim, pooling_strategy=pooling_strategy,
                window_half_size=window_half_size)
        elif classifier_type == "transformer":
            self.classifier = TransformerClassifier(input_dim, pooling_strategy=pooling_strategy)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def forward(self, input_ids, attention_mask=None):
        # Create attention mask from padding token
        attention_mask = input_ids != 1  # tokenizer.pad_token_id

        # Get embeddings from base model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
        embed = outputs['hidden_states'][-1]  # (batch, seq_len, hidden_dim)

        # Apply embedding strategy
        if self.embedding_strategy == "variant_position":
            variant_position = 1000
            seq_len = embed.shape[1]
            if variant_position >= seq_len:
                variant_position = seq_len // 2
            pooled_embed = embed[:, variant_position, :]
            logits = self.classifier(pooled_embed)

        elif self.embedding_strategy == "mean_pool":
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(attention_mask_expanded * embed, dim=1)
            sum_mask = torch.sum(attention_mask_expanded, dim=1)
            pooled_embed = sum_embeddings / sum_mask
            logits = self.classifier(pooled_embed)

        elif self.embedding_strategy == "full-mean_pool":
            logits = self.classifier(embed, attention_mask=attention_mask)

        elif self.embedding_strategy == "full-variant_position":
            variant_position = 1000
            seq_len = embed.shape[1]
            if variant_position >= seq_len:
                variant_position = seq_len // 2
            logits = self.classifier(embed, variant_position=variant_position,
                                     attention_mask=attention_mask)

        elif self.embedding_strategy.startswith("window_"):
            # Full sequence goes through CNN convolutions, then CNN
            # mean-pools over a local window around the variant position.
            variant_position = 1000
            seq_len = embed.shape[1]
            if variant_position >= seq_len:
                variant_position = seq_len // 2

            logits = self.classifier(embed, variant_position=variant_position,
                                     attention_mask=attention_mask)

        else:
            raise ValueError(f"Unknown embedding strategy: {self.embedding_strategy}")

        return logits


# ============================================================================
# Dataset and DataLoader
# ============================================================================

class SequenceDataset(Dataset):
    """Dataset for DNA sequences"""
    def __init__(self, sequences, labels, tokenizer, max_length=2048):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences.iloc[idx]
        label = float(self.labels.iloc[idx])

        encoding = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float),
        }


def load_data(file_path, k=None):
    """Load data from file"""
    data = pd.read_csv(file_path, delimiter='\t')

    if k:
        sequences = data['alt_sequence'].apply(
            lambda x: ' '.join([x[i:i+k] for i in range(0, len(x), k)])
        )
    else:
        sequences = data['alt_sequence']

    labels = data['label']
    return sequences, labels


def create_data_loader(file_path, tokenizer, batch_size, max_length,
                       shuffle=False, k=None):
    """Create DataLoader"""
    sequences, labels = load_data(file_path, k=k)
    dataset = SequenceDataset(sequences, labels, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================================
# Training Utilities
# ============================================================================

class LRSchedulerWithWarmup:
    """Linear warmup + linear decay scheduler"""
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
            remaining_steps = self.total_steps - self.warmup_steps
            steps_since_warmup = self.current_step - self.warmup_steps
            lr_scale = 1.0 - (steps_since_warmup / remaining_steps)
            lr_scale = max(0.0, lr_scale)

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class EarlyStopping:
    """Early stopping with patience"""
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


# ============================================================================
# Evaluation  (returns AUROC + AUPRC)
# ============================================================================

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on a dataset. Returns (loss, auc, auprc)."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probabilities = torch.sigmoid(outputs)
            all_preds.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    auc_score = roc_auc_score(all_labels, all_preds)
    auprc_score = average_precision_score(all_labels, all_preds)

    return avg_loss, auc_score, auprc_score


# ============================================================================
# Training Loop
# ============================================================================

def train_experiment(exp_config, train_path, val_path, output_dir,
                     gpu_id, batch_size=32, num_steps=5000,
                     learning_rate=3e-5, k=6):
    """Train a single experiment (frozen backbone)."""

    exp_id = exp_config['exp_id']
    exp_dir = Path(output_dir) / f"exp_{exp_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ---- Logging ----
    log_file = exp_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(f"Exp{exp_id}")

    logger.info(f"Starting Experiment {exp_id}")
    logger.info(f"Configuration: {exp_config}")
    logger.info(f"GPU: {gpu_id}")

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

        # Create fine-tuning model
        freeze_base = (exp_config['fine_tuning'] != 'unfreeze')

        model = NT2_FineTune(
            base_model,
            classifier_type=exp_config['classifier_type'],
            num_layers=exp_config.get('num_layers', 2),
            embedding_strategy=exp_config['embedding_strategy'],
            freeze_base=freeze_base,
            window_half_size=exp_config.get('window_half_size', None),
        )
        model = model.to(device)

        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable parameters: "
                    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Record GPU memory
        memory_after_model = torch.cuda.memory_allocated(gpu_id) / 1024**3
        logger.info(f"GPU memory after model loading: {memory_after_model:.2f} GB")

        # ---- Data ----
        logger.info("Creating dataloaders...")
        train_loader = create_data_loader(
            train_path, tokenizer, batch_size, 2048, shuffle=True, k=k)
        val_loader = create_data_loader(
            val_path, tokenizer, batch_size, 2048, shuffle=False, k=k)

        # ---- Optimizer / Scheduler ----
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
                "weight_decay": 0.0
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate,
                          eps=1e-8, betas=(0.9, 0.999))

        total_steps = num_steps
        warmup_steps = int(0.06 * total_steps)

        scheduler = LRSchedulerWithWarmup(optimizer, warmup_steps, total_steps)
        criterion = nn.BCEWithLogitsLoss()
        early_stopping = EarlyStopping(patience=2, mode='max')

        # ---- Metrics storage ----
        metrics = {
            'steps': [], 'train_loss': [], 'train_auc': [], 'train_auprc': [],
            'val_loss': [], 'val_auc': [], 'val_auprc': [],
            'learning_rate': [], 'gpu_memory_gb': []
        }
        metrics_csv_path = exp_dir / 'training_metrics.csv'
        pd.DataFrame(columns=list(metrics.keys())).to_csv(metrics_csv_path, index=False)

        # ---- Training ----
        logger.info(f"Training for {num_steps} steps with {warmup_steps} warmup steps")
        global_step = 0
        best_val_auc = 0
        best_val_auprc = 0
        train_iterator = iter(train_loader)

        while global_step < num_steps:
            model.train()

            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1

            # ---- Evaluate every 1000 steps ----
            if global_step % 1000 == 0:
                current_memory = torch.cuda.memory_allocated(gpu_id) / 1024**3

                train_loss, train_auc, train_auprc = evaluate_model(
                    model, train_loader, criterion, device)
                val_loss, val_auc, val_auprc = evaluate_model(
                    model, val_loader, criterion, device)

                metrics['steps'].append(global_step)
                metrics['train_loss'].append(train_loss)
                metrics['train_auc'].append(train_auc)
                metrics['train_auprc'].append(train_auprc)
                metrics['val_loss'].append(val_loss)
                metrics['val_auc'].append(val_auc)
                metrics['val_auprc'].append(val_auprc)
                metrics['learning_rate'].append(scheduler.get_last_lr()[0])
                metrics['gpu_memory_gb'].append(current_memory)

                # Append to CSV on-the-fly
                current_metrics = pd.DataFrame({
                    'steps': [global_step],
                    'train_loss': [train_loss],
                    'train_auc': [train_auc],
                    'train_auprc': [train_auprc],
                    'val_loss': [val_loss],
                    'val_auc': [val_auc],
                    'val_auprc': [val_auprc],
                    'learning_rate': [scheduler.get_last_lr()[0]],
                    'gpu_memory_gb': [current_memory]
                })
                current_metrics.to_csv(metrics_csv_path, mode='a',
                                       header=False, index=False)

                logger.info(
                    f"Step {global_step}/{num_steps} | "
                    f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, AUPRC: {train_auprc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, AUPRC: {val_auprc:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"GPU Memory: {current_memory:.2f} GB"
                )

                # Save best model
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_val_auprc = val_auprc
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_auc': val_auc,
                        'val_auprc': val_auprc,
                        'config': exp_config,
                    }, exp_dir / 'best_model.pt')
                    logger.info(f"Saved best model with Val AUC: {val_auc:.4f}, AUPRC: {val_auprc:.4f}")

                # Early stopping
                if early_stopping(val_auc):
                    logger.info(f"Early stopping triggered at step {global_step}")
                    break

                model.train()

        # ---- Post-training ----
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(exp_dir / 'training_metrics.csv', index=False)

        generate_plots(metrics_df, exp_dir, exp_id)

        peak_gpu_memory = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
        final_gpu_memory = torch.cuda.memory_allocated(gpu_id) / 1024**3
        total_time = time.time() - start_time

        logger.info(f"Peak GPU memory usage: {peak_gpu_memory:.2f} GB")
        logger.info(f"Final GPU memory usage: {final_gpu_memory:.2f} GB")

        summary = {
            'exp_id': exp_id,
            'config': exp_config,
            'best_val_auc': best_val_auc,
            'best_val_auprc': best_val_auprc,
            'total_time_seconds': total_time,
            'total_time_hours': total_time / 3600,
            'gpu_memory_after_model_gb': memory_after_model,
            'peak_gpu_memory_gb': peak_gpu_memory,
            'final_gpu_memory_gb': final_gpu_memory,
            'total_steps': global_step,
            'warmup_steps': warmup_steps
        }

        with open(exp_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Experiment {exp_id} completed successfully!")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Best validation AUC: {best_val_auc:.4f}, AUPRC: {best_val_auprc:.4f}")
        logger.info(f"Peak GPU memory: {peak_gpu_memory:.2f} GB")

        return summary

    except Exception as e:
        logger.error(f"Error in experiment {exp_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# ============================================================================
# Plotting
# ============================================================================

def generate_plots(metrics_df, exp_dir, exp_id):
    """Generate training curves including AUPRC."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Experiment {exp_id} Training Curves', fontsize=16)

    # Training Loss
    axes[0, 0].plot(metrics_df['steps'], metrics_df['train_loss'],
                    label='Train', linewidth=2)
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Validation Loss
    axes[0, 1].plot(metrics_df['steps'], metrics_df['val_loss'],
                    label='Validation', color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Training AUPRC
    axes[0, 2].plot(metrics_df['steps'], metrics_df['train_auprc'],
                    label='Train', linewidth=2)
    axes[0, 2].set_xlabel('Steps')
    axes[0, 2].set_ylabel('AUPRC')
    axes[0, 2].set_title('Training AUPRC')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Training AUC
    axes[1, 0].plot(metrics_df['steps'], metrics_df['train_auc'],
                    label='Train', linewidth=2)
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_title('Training AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Validation AUC
    axes[1, 1].plot(metrics_df['steps'], metrics_df['val_auc'],
                    label='Validation', color='orange', linewidth=2)
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].set_title('Validation AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Validation AUPRC
    axes[1, 2].plot(metrics_df['steps'], metrics_df['val_auprc'],
                    label='Validation', color='orange', linewidth=2)
    axes[1, 2].set_xlabel('Steps')
    axes[1, 2].set_ylabel('AUPRC')
    axes[1, 2].set_title('Validation AUPRC')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Combined plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Experiment {exp_id} Combined Metrics', fontsize=16)

    # Loss comparison
    axes[0].plot(metrics_df['steps'], metrics_df['train_loss'],
                label='Train', linewidth=2)
    axes[0].plot(metrics_df['steps'], metrics_df['val_loss'],
                label='Validation', linewidth=2)
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUC comparison
    axes[1].plot(metrics_df['steps'], metrics_df['train_auc'],
                label='Train', linewidth=2)
    axes[1].plot(metrics_df['steps'], metrics_df['val_auc'],
                label='Validation', linewidth=2)
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('AUC Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUPRC comparison
    axes[2].plot(metrics_df['steps'], metrics_df['train_auprc'],
                label='Train', linewidth=2)
    axes[2].plot(metrics_df['steps'], metrics_df['val_auprc'],
                label='Validation', linewidth=2)
    axes[2].set_xlabel('Steps')
    axes[2].set_ylabel('AUPRC')
    axes[2].set_title('AUPRC Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / 'combined_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Parallel Experiment Runner
# ============================================================================

def run_single_experiment(args):
    """Wrapper for parallel execution"""
    exp_config, train_path, val_path, output_dir, gpu_id, hyperparams = args
    return train_experiment(
        exp_config, train_path, val_path, output_dir, gpu_id, **hyperparams
    )


def run_all_experiments(train_path, val_path, output_dir,
                       available_gpus=[0, 1, 2], batch_size=32,
                       num_steps=5000, learning_rate=3e-5, k=6):
    """Run all experiments in parallel on available GPUs"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup main logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'main.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('Main')

    logger.info(f"Starting {len(EXPERIMENT_CONFIGS)} experiments")
    logger.info(f"Available GPUs: {available_gpus}")

    # Prepare experiment arguments
    hyperparams = {
        'batch_size': batch_size,
        'num_steps': num_steps,
        'learning_rate': learning_rate,
        'k': k
    }

    # Assign GPUs to experiments
    experiment_args = []
    for i, exp_config in enumerate(EXPERIMENT_CONFIGS):
        gpu_id = available_gpus[i % len(available_gpus)]
        experiment_args.append((
            exp_config, train_path, val_path, output_dir, gpu_id, hyperparams
        ))

    # Run experiments in parallel
    all_summaries = []
    num_parallel = len(available_gpus)

    for i in range(0, len(experiment_args), num_parallel):
        batch = experiment_args[i:i+num_parallel]
        logger.info(f"Running batch {i//num_parallel + 1}: "
                   f"Experiments {[args[0]['exp_id'] for args in batch]}")

        with mp.Pool(processes=len(batch)) as pool:
            summaries = pool.map(run_single_experiment, batch)
            all_summaries.extend(summaries)

    # Create summary report
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(output_dir / 'all_experiments_summary.csv', index=False)

    logger.info("All experiments completed!")
    logger.info(f"\nResults Summary:\n{summary_df.to_string()}")

    return summary_df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Windowed Mean Pooling Sweep: frozen backbone CNN + varying window sizes'
    )
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val_path', type=str, required=True,
                       help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default='./nt2_window_pooling_sweep',
                       help='Output directory for results')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2],
                       help='List of GPU IDs to use')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_steps', type=int, default=5000,
                       help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                       help='Learning rate')
    parser.add_argument('--k', type=int, default=6,
                       help='K-mer size (6 for codons)')

    args = parser.parse_args()

    # Print experiment plan
    configs = EXPERIMENT_CONFIGS
    print("=" * 80)
    print(f"EXPERIMENT PLAN: {len(configs)} total experiments")
    print(f"  batch_size={args.batch_size}, num_steps={args.num_steps}, "
          f"lr={args.learning_rate}, k={args.k}")
    print("=" * 80)
    for c in configs:
        whs = c['window_half_size']
        nt = whs * 6
        print(f"  exp_{c['exp_id']:02d}: {c['embedding_strategy']:25s} | "
              f"{c['classifier_type']:12s} | frozen | "
              f"window=±{whs} tokens (±{nt} nt, {2*whs+1} tokens total)")
    print("=" * 80)

    # Run all experiments
    summary_df = run_all_experiments(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        available_gpus=args.gpus,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        k=args.k
    )

    print("\n" + "=" * 80)
    print("All experiments completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
