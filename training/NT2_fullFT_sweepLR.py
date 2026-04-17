#!/usr/bin/env python3

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
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================================
# Experiment Configurations
# ============================================================================

def build_experiment_configs():
    """Build full fine-tuning experiment configs."""
    configs = []
    exp_id = 62

    # ------------------------------------------------------------------
    # Full fine-tune LR sweep experiments
    # MLP uses variant_position; CNN uses full-variant_position
    # MLP runs before CNN
    # ------------------------------------------------------------------
    full_ft_lrs = [1e-4, 5e-4]
    classifier_embedding_pairs = [
        ("mlp", "variant_position"),
        ("cnn", "full-variant_position"),
    ]
    for classifier_type, embedding_strategy in classifier_embedding_pairs:
        for lr in full_ft_lrs:
            configs.append({
                "exp_id": exp_id,
                "classifier_type": classifier_type,
                "num_layers": 2,
                "embedding_strategy": embedding_strategy,
                "fine_tuning": "unfreeze",
                "lora_rank": None,
                "batch_size": 8,
                "num_steps": 40000,
                "gradient_accumulation_steps": 1,
                "learning_rate": lr,
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
    def __init__(self, input_dim=1024, dropout=0.1, pooling_strategy='mean_pool'):
        super(CNNClassifier, self).__init__()

        self.pooling_strategy = pooling_strategy
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
    """Nucleotide Transformer fine-tuning wrapper supporting frozen, LoRA, and full fine-tuning."""
    def __init__(self, base_model, classifier_type, num_layers,
                 embedding_strategy, freeze_base=True, lora_rank=None):
        super(NT2_FineTune, self).__init__()

        self.bert = base_model
        self.embedding_strategy = embedding_strategy
        self.classifier_type = classifier_type

        # ---- Fine-tuning strategy ----
        if lora_rank is not None:
            # LoRA fine-tuning: freeze base, then add LoRA adapters
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
            print(f"LoRA Configuration Applied (rank={lora_rank}):")
            self.bert.print_trainable_parameters()
            print("=" * 80)

        elif freeze_base:
            # Frozen backbone
            for param in self.bert.parameters():
                param.requires_grad = False
        # else: full fine-tuning — all params remain trainable

        # ---- Classifier head ----
        input_dim = 1024  # NT2 hidden dimension

        if embedding_strategy.startswith('full-'):
            pooling_strategy = embedding_strategy.split('-', 1)[1]
        else:
            pooling_strategy = 'mean_pool'  # default

        if classifier_type == "mlp":
            self.classifier = MLPClassifier(input_dim, num_layers)
        elif classifier_type == "cnn":
            self.classifier = CNNClassifier(input_dim, pooling_strategy=pooling_strategy)
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
# Evaluation  (now returns AUROC + AUPRC)
# ============================================================================

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on a dataset. Returns (loss, auroc, auprc)."""
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
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    return avg_loss, auroc, auprc


# ============================================================================
# Training Loop
# ============================================================================

def train_experiment(exp_config, train_path, val_path, output_dir,
                     gpu_id, batch_size=4, num_steps=40000,
                     learning_rate=3e-5, k=6, gradient_accumulation_steps=8):
    """Train a single experiment."""

    exp_id = exp_config['exp_id']
    exp_dir = Path(output_dir) / f"exp_{exp_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ---- Logging ----
    log_file = exp_dir / "training.log"
    logger = logging.getLogger(f"Exp{exp_id}")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # clear any inherited handlers
    logger.addHandler(logging.FileHandler(log_file))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for h in logger.handlers:
        h.setFormatter(fmt)

    logger.info(f"Starting Experiment {exp_id}")
    logger.info(f"Configuration: {json.dumps(exp_config, indent=2, default=str)}")
    logger.info(f"GPU: {gpu_id}")
    logger.info(f"Batch size: {batch_size}, Grad accum: {gradient_accumulation_steps}, "
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

        # Determine fine-tuning strategy
        is_lora = (exp_config['fine_tuning'] == 'lora')
        is_unfreeze = (exp_config['fine_tuning'] == 'unfreeze')
        freeze_base = not is_unfreeze
        lora_rank = exp_config.get('lora_rank', None) if is_lora else None

        logger.info(f"Fine-tuning mode: {exp_config['fine_tuning']}")
        if lora_rank:
            logger.info(f"LoRA rank: {lora_rank}")

        model = NT2_FineTune(
            base_model,
            classifier_type=exp_config['classifier_type'],
            num_layers=exp_config.get('num_layers', 2),
            embedding_strategy=exp_config['embedding_strategy'],
            freeze_base=freeze_base,
            lora_rank=lora_rank
        )
        model = model.to(device)

        # DataParallel for full fine-tuning
        use_multi_gpu = is_unfreeze
        if use_multi_gpu:
            gpu_list = [gpu_id, (gpu_id + 1) % torch.cuda.device_count()]
            logger.info(f"Using DataParallel with GPUs: {gpu_list}")
            model = nn.DataParallel(model, device_ids=gpu_list)

        logger.info(f"Trainable parameters: "
                    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Record GPU memory
        if use_multi_gpu:
            gpu_list = [gpu_id, (gpu_id + 1) % torch.cuda.device_count()]
            memory_after_model = max(
                [torch.cuda.memory_allocated(g) / 1024**3 for g in gpu_list])
        else:
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
        overfitting_gap_threshold = 0.1

        # ---- Metrics storage ----
        metrics = {
            'steps': [], 'train_loss': [], 'train_auroc': [], 'train_auprc': [],
            'val_loss': [], 'val_auroc': [], 'val_auprc': [],
            'learning_rate': [], 'gpu_memory_gb': []
        }
        metrics_csv_path = exp_dir / 'training_metrics.csv'
        pd.DataFrame(columns=list(metrics.keys())).to_csv(metrics_csv_path, index=False)

        # ---- Training ----
        logger.info(f"Training for {num_steps} steps with {warmup_steps} warmup steps")
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

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if accumulation_counter == 0:
                optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            accumulation_counter += 1

            if accumulation_counter == gradient_accumulation_steps:
                optimizer.step()
                scheduler.step()
                accumulation_counter = 0
                global_step += 1

            # ---- Evaluate every 1000 optimizer steps ----
            if global_step > 0 and global_step % 1000 == 0 and accumulation_counter == 0:
                if use_multi_gpu:
                    gpu_list = [gpu_id, (gpu_id + 1) % torch.cuda.device_count()]
                    current_memory = max(
                        [torch.cuda.memory_allocated(g) / 1024**3 for g in gpu_list])
                else:
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

                # Append to CSV on-the-fly
                row = pd.DataFrame({k: [v[-1]] for k, v in metrics.items()})
                row.to_csv(metrics_csv_path, mode='a', header=False, index=False)

                logger.info(
                    f"Step {global_step}/{num_steps} | "
                    f"Train Loss: {train_loss:.4f}, AUROC: {train_auroc:.4f}, AUPRC: {train_auprc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"GPU Mem: {current_memory:.2f} GB"
                )

                # Save best model (by val AUROC)
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    best_val_auprc = val_auprc
                    model_to_save = model.module if use_multi_gpu else model
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_auroc': val_auroc,
                        'val_auprc': val_auprc,
                        'config': exp_config,
                        'use_multi_gpu': use_multi_gpu
                    }, exp_dir / 'best_model.pt')
                    logger.info(f"  -> New best model: AUROC={val_auroc:.4f}, AUPRC={val_auprc:.4f}")

                # Early stopping
                if early_stopping(val_auroc):
                    logger.info(f"Early stopping triggered at step {global_step}")
                    break

                if train_auroc - val_auroc >= overfitting_gap_threshold:
                    logger.info(
                        f"Overfitting early stopping triggered at step {global_step}: "
                        f"train_auroc={train_auroc:.4f}, val_auroc={val_auroc:.4f}, "
                        f"gap={train_auroc - val_auroc:.4f}"
                    )
                    break

                model.train()

        # ---- Post-training ----
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(exp_dir / 'training_metrics_final.csv', index=False)

        generate_plots(metrics_df, exp_dir, exp_id)

        # GPU memory stats
        if use_multi_gpu:
            gpu_list = [gpu_id, (gpu_id + 1) % torch.cuda.device_count()]
            peak_gpu_memory = max(
                [torch.cuda.max_memory_allocated(g) / 1024**3 for g in gpu_list])
            final_gpu_memory = max(
                [torch.cuda.memory_allocated(g) / 1024**3 for g in gpu_list])
        else:
            peak_gpu_memory = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
            final_gpu_memory = torch.cuda.memory_allocated(gpu_id) / 1024**3

        total_time = time.time() - start_time

        summary = {
            'exp_id': exp_id,
            'config': exp_config,
            'best_val_auroc': best_val_auroc,
            'best_val_auprc': best_val_auprc,
            'total_time_seconds': total_time,
            'total_time_hours': total_time / 3600,
            'gpu_memory_after_model_gb': memory_after_model,
            'peak_gpu_memory_gb': peak_gpu_memory,
            'final_gpu_memory_gb': final_gpu_memory,
            'total_steps': global_step,
            'warmup_steps': warmup_steps,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': batch_size * gradient_accumulation_steps
        }

        with open(exp_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Experiment {exp_id} completed! "
                    f"Time: {total_time/3600:.2f}h, "
                    f"Best AUROC: {best_val_auroc:.4f}, "
                    f"Best AUPRC: {best_val_auprc:.4f}, "
                    f"Peak GPU: {peak_gpu_memory:.2f} GB")

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

    plot_specs = [
        ('train_loss',  'Train Loss',  0, 0),
        ('val_loss',    'Val Loss',    0, 1),
        ('train_auroc', 'Train AUROC', 1, 0),
        ('val_auroc',   'Val AUROC',   1, 1),
        ('train_auprc', 'Train AUPRC', 0, 2),
        ('val_auprc',   'Val AUPRC',   1, 2),
    ]

    for col, title, r, c in plot_specs:
        if col in metrics_df.columns:
            axes[r, c].plot(metrics_df['steps'], metrics_df[col], linewidth=2)
            axes[r, c].set_xlabel('Steps')
            axes[r, c].set_ylabel(col)
            axes[r, c].set_title(title)
            axes[r, c].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Combined train vs val
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Experiment {exp_id} Train vs Val', fontsize=16)

    for ax, metric, label in zip(
        axes, ['loss', 'auroc', 'auprc'], ['Loss', 'AUROC', 'AUPRC']
    ):
        tcol, vcol = f'train_{metric}', f'val_{metric}'
        if tcol in metrics_df.columns and vcol in metrics_df.columns:
            ax.plot(metrics_df['steps'], metrics_df[tcol], label='Train', linewidth=2)
            ax.plot(metrics_df['steps'], metrics_df[vcol], label='Val', linewidth=2)
            ax.set_xlabel('Steps')
            ax.set_ylabel(label)
            ax.set_title(f'{label} Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)

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


def run_all_experiments(train_path_ft, val_path_ft,
                        output_dir, available_gpus=[0, 1, 2, 3], k=6):
    """Run full fine-tune experiments with smart GPU scheduling.

    - Full fine-tune experiments (ft): use train_path_ft / val_path_ft (e.g. 44k samples)
    - Full fine-tune: 2 GPUs each → run len(available_gpus)//2 in parallel
    """
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

    # Full fine-tune experiments only
    unfreeze_configs = [c for c in EXPERIMENT_CONFIGS if c['fine_tuning'] == 'unfreeze']

    logger.info(f"Total experiments: {len(EXPERIMENT_CONFIGS)} "
                f"({len(unfreeze_configs)} full fine-tune)")
    logger.info(f"Available GPUs: {available_gpus}")
    logger.info(f"Full FT data: {train_path_ft} / {val_path_ft}")

    all_summaries = []

    # ---- Run full fine-tune experiments: 2 GPUs each, N_GPU//2 parallel ----
    if unfreeze_configs:
        logger.info("=" * 80)
        logger.info(f"Running {len(unfreeze_configs)} full fine-tune experiments "
                    f"({len(available_gpus) // 2} at a time, 2 GPUs each)")
        logger.info("=" * 80)

        # Assign primary GPUs in pairs: (0,1), (2,3), etc.
        ft_args = []
        gpu_pairs = [(available_gpus[j], available_gpus[j + 1])
                     for j in range(0, len(available_gpus) - 1, 2)]

        for i, cfg in enumerate(unfreeze_configs):
            primary_gpu = gpu_pairs[i % len(gpu_pairs)][0]
            hyperparams = {
                'batch_size': cfg['batch_size'],
                'num_steps': cfg['num_steps'],
                'learning_rate': cfg['learning_rate'],
                'gradient_accumulation_steps': cfg['gradient_accumulation_steps'],
                'k': k,
            }
            ft_args.append((cfg, train_path_ft, val_path_ft,
                           str(output_dir), primary_gpu, hyperparams))

        num_parallel_ft = len(gpu_pairs)
        for i in range(0, len(ft_args), num_parallel_ft):
            batch = ft_args[i:i + num_parallel_ft]
            logger.info(f"Full fine-tune batch {i // num_parallel_ft + 1}: "
                        f"Experiments {[a[0]['exp_id'] for a in batch]}")
            with mp.Pool(processes=len(batch)) as pool:
                summaries = pool.map(run_single_experiment, batch)
                all_summaries.extend(summaries)

    # ---- Final summary ----
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
        description='Reviewer Response Sweep: full fine-tune LR sweep for NT2'
    )
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training data for full fine-tune (TSV, e.g. 44k samples)')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to validation data for full fine-tune (TSV)')
    parser.add_argument('--output_dir', type=str, default='./nt2_reviewer_sweep',
                        help='Output directory')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='List of GPU IDs to use (default: 0 1 2 3)')
    parser.add_argument('--k', type=int, default=6,
                        help='K-mer size (default: 6)')

    args = parser.parse_args()

    # Print experiment plan
    configs = EXPERIMENT_CONFIGS
    print("=" * 80)
    print(f"EXPERIMENT PLAN: {len(configs)} total experiments")
    print("=" * 80)
    for c in configs:
        tag = f"LoRA-r{c['lora_rank']}-lr{c['learning_rate']}" if c['fine_tuning'] == 'lora' else f"FullFT-lr{c['learning_rate']}"
        print(f"  exp_{c['exp_id']:02d}: {c['embedding_strategy']:25s} | "
              f"{c['classifier_type']:12s} | {tag}")
    print("=" * 80)

    summary_df = run_all_experiments(
        train_path_ft=args.train_path,
        val_path_ft=args.val_path,
        output_dir=args.output_dir,
        available_gpus=args.gpus,
        k=args.k
    )

    print("\n" + "=" * 80)
    print("All experiments completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
