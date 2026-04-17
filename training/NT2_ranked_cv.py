#!/usr/bin/env python3
"""
Rank-Ordered 5-Fold Chromosome-Split Cross-Validation for NT2 LoRA Experiments
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
# Chromosome-based 5-Fold Splitting  (from NT2_lora_cv.py)
# ============================================================================

def create_chromosome_folds(data, n_folds=5, seed=42):
    """Split data into n folds by chromosome, balancing variant count."""
    rng = np.random.RandomState(seed)
    chrom_counts = data.groupby('chromosome').size().to_dict()
    chroms_sorted = sorted(chrom_counts.keys(), key=lambda c: -chrom_counts[c])

    fold_chroms = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    for chrom in chroms_sorted:
        smallest_fold = np.argmin(fold_sizes)
        fold_chroms[smallest_fold].append(chrom)
        fold_sizes[smallest_fold] += chrom_counts[chrom]

    print("=" * 80)
    print("Chromosome Fold Assignment:")
    for i, (chroms, size) in enumerate(zip(fold_chroms, fold_sizes)):
        chrom_strs = [str(c) for c in sorted(chroms, key=lambda x: (not str(x).isdigit(), str(x)))]
        print(f"  Fold {i+1}: {size:,} variants | Chromosomes: {', '.join(chrom_strs)}")
    print("=" * 80)

    return fold_chroms


def get_fold_split(data, fold_chroms, fold_idx):
    val_chroms = set(fold_chroms[fold_idx])
    val_mask = data['chromosome'].astype(str).isin([str(c) for c in val_chroms])
    val_data = data[val_mask].reset_index(drop=True)
    train_data = data[~val_mask].reset_index(drop=True)
    return train_data, val_data


# ============================================================================
# Classifier Heads  (from NT2_reviewer_sweep.py)
# ============================================================================

class MLPClassifier(nn.Module):
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
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)

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
                raise ValueError("variant_position required for variant_position pooling")
            pooled = x[:, variant_position, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        logits = self.fc(pooled)
        return logits.squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=128, nhead=2,
                 dim_feedforward=512, dropout=0.1, pooling_strategy='mean_pool'):
        super(TransformerClassifier, self).__init__()
        self.pooling_strategy = pooling_strategy
        self.input_projection = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x, variant_position=None, attention_mask=None):
        x = self.input_projection(x)
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
                raise ValueError("variant_position required for variant_position pooling")
            pooled = x[:, variant_position, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits.squeeze(-1)


# ============================================================================
# Model Wrapper  (from NT2_reviewer_sweep.py, LoRA only)
# ============================================================================

class NT2_FineTune(nn.Module):
    """NT2 fine-tuning wrapper supporting all sweep classifier/embedding combos."""
    def __init__(self, base_model, classifier_type, num_layers,
                 embedding_strategy, lora_rank):
        super(NT2_FineTune, self).__init__()

        self.bert = base_model
        self.embedding_strategy = embedding_strategy
        self.classifier_type = classifier_type

        # Freeze base, apply LoRA
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

        # Classifier head
        input_dim = 1024
        if embedding_strategy.startswith('full-'):
            pooling_strategy = embedding_strategy.split('-', 1)[1]
        else:
            pooling_strategy = 'mean_pool'

        if classifier_type == "mlp":
            self.classifier = MLPClassifier(input_dim, num_layers)
        elif classifier_type == "cnn":
            self.classifier = CNNClassifier(input_dim, pooling_strategy=pooling_strategy)
        elif classifier_type == "transformer":
            self.classifier = TransformerClassifier(input_dim, pooling_strategy=pooling_strategy)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def forward(self, input_ids, attention_mask=None):
        attention_mask = input_ids != 1

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
        embed = outputs['hidden_states'][-1]

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
# Dataset
# ============================================================================

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=2048):
        self.sequences = sequences.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences.iloc[idx]
        label = float(self.labels.iloc[idx])

        encoding = self.tokenizer.encode_plus(
            sequence, add_special_tokens=True, max_length=self.max_length,
            return_token_type_ids=False, padding='max_length',
            return_attention_mask=True, return_tensors='pt', truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float),
        }


def kmerize(sequence, k):
    return ' '.join([sequence[i:i+k] for i in range(0, len(sequence), k)])


def create_data_loader_from_df(data, tokenizer, batch_size, max_length,
                                shuffle=False, k=6):
    sequences = data['alt_sequence'].apply(lambda x: kmerize(x, k))
    labels = data['label']
    dataset = SequenceDataset(sequences, labels, tokenizer, max_length)
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


def evaluate_model(model, dataloader, criterion, device):
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
# Single Fold Training
# ============================================================================

def train_single_fold(fold_config, output_dir, gpu_id,
                      batch_size=4, num_steps=4000, k=6):
    """Train one fold of one experiment configuration."""

    exp_cfg = fold_config['exp_config']
    fold_idx = fold_config['fold_idx']
    train_data = fold_config['train_data']
    val_data = fold_config['val_data']

    classifier_type = exp_cfg['classifier']
    embedding = exp_cfg['embedding']
    lora_rank = exp_cfg['lora_rank']
    learning_rate = exp_cfg['learning_rate']
    exp_id = exp_cfg['exp_id']
    mean_rank = exp_cfg['mean_rank']

    exp_name = f"exp{exp_id}_rank{mean_rank:.1f}_{classifier_type}_{embedding}_r{lora_rank}_lr{learning_rate:.0e}_fold{fold_idx + 1}"
    exp_dir = Path(output_dir) / f"exp_{exp_id}" / f"fold_{fold_idx + 1}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ---- Logging ----
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(logging.FileHandler(exp_dir / 'training.log'))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for h in logger.handlers:
        h.setFormatter(fmt)

    logger.info(f"Starting {exp_name} on GPU {gpu_id}")
    logger.info(f"Config: classifier={classifier_type}, embedding={embedding}, "
                f"lora_rank={lora_rank}, lr={learning_rate}")
    logger.info(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")

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

        model = NT2_FineTune(
            base_model,
            classifier_type=classifier_type,
            num_layers=2,
            embedding_strategy=embedding,
            lora_rank=lora_rank
        )
        model = model.to(device)

        logger.info(f"Trainable parameters: "
                    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # ---- Data ----
        train_loader = create_data_loader_from_df(
            train_data, tokenizer, batch_size, 2048, shuffle=True, k=k)
        val_loader = create_data_loader_from_df(
            val_data, tokenizer, batch_size, 2048, shuffle=False, k=k)

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

        # ---- Metrics ----
        metrics = {
            'steps': [], 'train_loss': [], 'train_auroc': [], 'train_auprc': [],
            'val_loss': [], 'val_auroc': [], 'val_auprc': [],
            'learning_rate': [], 'gpu_memory_gb': []
        }
        metrics_csv_path = exp_dir / 'training_metrics.csv'
        pd.DataFrame(columns=list(metrics.keys())).to_csv(metrics_csv_path, index=False)

        # ---- Training (fixed steps, no early stopping) ----
        logger.info(f"Training for {num_steps} steps ({warmup_steps} warmup)")
        global_step = 0
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

            if global_step % 1000 == 0:
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
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

                model.train()

        # ---- Final evaluation ----
        logger.info("Training complete. Running final evaluation...")
        final_val_loss, final_val_auroc, final_val_auprc = evaluate_model(
            model, val_loader, criterion, device)
        final_train_loss, final_train_auroc, final_train_auprc = evaluate_model(
            model, train_loader, criterion, device)

        logger.info(f"Final Train: AUROC={final_train_auroc:.4f}, AUPRC={final_train_auprc:.4f}")
        logger.info(f"Final Val:   AUROC={final_val_auroc:.4f}, AUPRC={final_val_auprc:.4f}")

        # Save final model
        torch.save({
            'step': global_step,
            'model_state_dict': model.state_dict(),
            'val_auroc': final_val_auroc,
            'val_auprc': final_val_auprc,
            'exp_config': exp_cfg,
            'fold_idx': fold_idx,
        }, exp_dir / 'final_model.pt')

        # Save metrics
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(exp_dir / 'training_metrics_final.csv', index=False)

        total_time = time.time() - start_time
        peak_gpu = torch.cuda.max_memory_allocated(gpu_id) / 1024**3

        summary = {
            'exp_id': exp_id,
            'mean_rank': mean_rank,
            'classifier': classifier_type,
            'embedding': embedding,
            'lora_rank': lora_rank,
            'learning_rate': learning_rate,
            'fold_idx': fold_idx,
            'final_val_auroc': final_val_auroc,
            'final_val_auprc': final_val_auprc,
            'final_train_auroc': final_train_auroc,
            'final_train_auprc': final_train_auprc,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'total_steps': global_step,
            'total_time_hours': total_time / 3600,
            'peak_gpu_memory_gb': peak_gpu,
        }

        with open(exp_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"{exp_name} done: AUROC={final_val_auroc:.4f}, "
                    f"AUPRC={final_val_auprc:.4f}, time={total_time/3600:.2f}h")

        return summary

    except Exception as e:
        logger.error(f"Error in {exp_name}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# ============================================================================
# Parallel Runner
# ============================================================================

def run_single_fold_wrapper(args):
    fold_config, output_dir, gpu_id, hyperparams = args
    return train_single_fold(fold_config, output_dir, gpu_id, **hyperparams)


def run_ranked_cv(ranking_json, data_path, output_dir,
                  available_gpus=[0, 1, 2, 3], k=6,
                  batch_size=4, num_steps=4000, top_n=None):
    """Run 5-fold CV for each experiment config, ordered by mean rank."""

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

    # ---- Load ranking ----
    with open(ranking_json) as f:
        ranking = json.load(f)

    # Filter to LoRA-only experiments
    ranking = [r for r in ranking if r.get('fine_tuning') == 'lora']
    ranking.sort(key=lambda r: r['mean_rank'])

    if top_n is not None:
        ranking = ranking[:top_n]

    logger.info(f"Loaded {len(ranking)} LoRA experiment configs from ranking")

    # ---- Load & split data ----
    logger.info(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path, delimiter='\t')
    logger.info(f"Total samples: {len(data)}")
    logger.info(f"Label distribution:\n{data['label'].value_counts().to_string()}")

    fold_chroms = create_chromosome_folds(data, n_folds=5)

    # ---- Process experiments sequentially by rank ----
    all_summaries = []

    for rank_idx, exp_cfg in enumerate(ranking):
        exp_id = exp_cfg['exp_id']
        mean_rank = exp_cfg['mean_rank']

        logger.info("=" * 80)
        logger.info(f"[{rank_idx + 1}/{len(ranking)}] Starting CV for exp_{exp_id} "
                    f"(mean_rank={mean_rank:.1f}): "
                    f"{exp_cfg['classifier']} / {exp_cfg['embedding']} / "
                    f"r{exp_cfg['lora_rank']} / lr={exp_cfg['learning_rate']:.0e}")
        logger.info("=" * 80)

        # Check if this experiment is already completed (all 5 folds done)
        exp_dir = output_dir / f"exp_{exp_id}"
        existing_folds = 0
        if exp_dir.exists():
            for fi in range(5):
                if (exp_dir / f"fold_{fi + 1}" / "summary.json").exists():
                    existing_folds += 1
        if existing_folds == 5:
            logger.info(f"  exp_{exp_id}: all 5 folds already completed, loading results...")
            fold_summaries = []
            for fi in range(5):
                with open(exp_dir / f"fold_{fi + 1}" / "summary.json") as f:
                    fold_summaries.append(json.load(f))
            all_summaries.extend(fold_summaries)
            _log_exp_cv_summary(logger, exp_cfg, fold_summaries)
            continue

        # Build fold configs for the 5 folds
        fold_jobs = []
        for fold_idx in range(5):
            # Skip already-completed folds
            if (exp_dir / f"fold_{fold_idx + 1}" / "summary.json").exists():
                logger.info(f"  Fold {fold_idx + 1} already completed, skipping")
                with open(exp_dir / f"fold_{fold_idx + 1}" / "summary.json") as f:
                    all_summaries.append(json.load(f))
                continue

            train_data, val_data = get_fold_split(data, fold_chroms, fold_idx)
            fold_jobs.append({
                'exp_config': exp_cfg,
                'fold_idx': fold_idx,
                'train_data': train_data,
                'val_data': val_data,
            })

        if not fold_jobs:
            continue

        # Run remaining folds in parallel across GPUs
        hyperparams = {
            'batch_size': batch_size,
            'num_steps': num_steps,
            'k': k,
        }

        job_args = []
        for i, fc in enumerate(fold_jobs):
            gpu_id = available_gpus[i % len(available_gpus)]
            job_args.append((fc, str(output_dir), gpu_id, hyperparams))

        num_parallel = len(available_gpus)
        for i in range(0, len(job_args), num_parallel):
            batch = job_args[i:i + num_parallel]
            batch_desc = [f"fold{a[0]['fold_idx']+1}" for a in batch]
            logger.info(f"  Running folds: {batch_desc}")

            with mp.Pool(processes=len(batch)) as pool:
                summaries = pool.map(run_single_fold_wrapper, batch)
                all_summaries.extend(summaries)

        # Log CV summary for this experiment
        exp_fold_summaries = [s for s in all_summaries if s['exp_id'] == exp_id]
        _log_exp_cv_summary(logger, exp_cfg, exp_fold_summaries)

        # Save running aggregate
        _save_aggregate(all_summaries, output_dir, logger)

    # ---- Final aggregate ----
    _save_aggregate(all_summaries, output_dir, logger, final=True)
    logger.info("All experiments completed!")


def _log_exp_cv_summary(logger, exp_cfg, fold_summaries):
    """Log mean ± STE for one experiment's 5 folds."""
    aurocs = np.array([s['final_val_auroc'] for s in fold_summaries])
    auprcs = np.array([s['final_val_auprc'] for s in fold_summaries])
    n = len(aurocs)
    if n > 1:
        auroc_ste = np.std(aurocs, ddof=1) / np.sqrt(n)
        auprc_ste = np.std(auprcs, ddof=1) / np.sqrt(n)
    else:
        auroc_ste = auprc_ste = 0.0

    logger.info(
        f"  CV Result for exp_{exp_cfg['exp_id']} "
        f"(mean_rank={exp_cfg['mean_rank']:.1f}): "
        f"AUROC = {np.mean(aurocs):.4f} ± {auroc_ste:.4f}, "
        f"AUPRC = {np.mean(auprcs):.4f} ± {auprc_ste:.4f} "
        f"({n} folds)"
    )


def _save_aggregate(all_summaries, output_dir, logger, final=False):
    """Save running (or final) aggregate CV results."""
    output_dir = Path(output_dir)
    results_df = pd.DataFrame(all_summaries)
    results_df.to_csv(output_dir / 'all_fold_results.csv', index=False)

    # Aggregate by experiment
    agg_rows = []
    for exp_id in results_df['exp_id'].unique():
        exp_results = results_df[results_df['exp_id'] == exp_id]
        if len(exp_results) == 0:
            continue

        aurocs = exp_results['final_val_auroc'].values
        auprcs = exp_results['final_val_auprc'].values
        n = len(aurocs)
        row = exp_results.iloc[0]

        agg_rows.append({
            'exp_id': int(exp_id),
            'mean_rank': row.get('mean_rank', None),
            'classifier': row['classifier'],
            'embedding': row['embedding'],
            'lora_rank': int(row['lora_rank']),
            'learning_rate': row['learning_rate'],
            'n_folds': n,
            'mean_auroc': np.mean(aurocs),
            'ste_auroc': np.std(aurocs, ddof=1) / np.sqrt(n) if n > 1 else 0.0,
            'mean_auprc': np.mean(auprcs),
            'ste_auprc': np.std(auprcs, ddof=1) / np.sqrt(n) if n > 1 else 0.0,
        })

    agg_df = pd.DataFrame(agg_rows).sort_values('mean_rank')
    agg_df.to_csv(output_dir / 'cv_summary.csv', index=False)

    if final:
        logger.info("\n" + "=" * 100)
        logger.info("CROSS-VALIDATION RESULTS (sorted by sweep mean rank)")
        logger.info("=" * 100)
        for _, row in agg_df.iterrows():
            logger.info(
                f"  exp_{int(row['exp_id']):2d} (rank={row['mean_rank']:5.1f}) | "
                f"{row['classifier']:12s} | {row['embedding']:25s} | "
                f"r{int(row['lora_rank']):2d} | lr={row['learning_rate']:.0e} | "
                f"AUROC={row['mean_auroc']:.4f}±{row['ste_auroc']:.4f} | "
                f"AUPRC={row['mean_auprc']:.4f}±{row['ste_auprc']:.4f} | "
                f"({int(row['n_folds'])} folds)"
            )
        logger.info("=" * 100)

        # Generate summary bar plot
        _generate_summary_plot(agg_df, output_dir)


def _generate_summary_plot(agg_df, output_dir):
    """Generate bar plot of CV results ordered by sweep rank."""
    if len(agg_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(agg_df) * 0.8), 6))
    fig.suptitle('5-Fold CV Results (ordered by sweep mean rank)', fontsize=14)

    x = np.arange(len(agg_df))
    labels = [f"exp_{int(r['exp_id'])}\n(rank={r['mean_rank']:.1f})"
              for _, r in agg_df.iterrows()]

    for ax, metric, label in zip(axes, ['auroc', 'auprc'], ['AUROC', 'AUPRC']):
        means = agg_df[f'mean_{metric}'].values
        stes = agg_df[f'ste_{metric}'].values

        bars = ax.bar(x, means, yerr=stes, capsize=3, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'cv_summary_plot.pdf', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Rank-Ordered 5-Fold Chromosome-Split CV for NT2 LoRA Experiments'
    )
    parser.add_argument('--ranking_json', type=str, required=True,
                        help='Path to experiment_ranking.json from summarize_sweep.py')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to combined data TSV (train+val merged, with chromosome column)')
    parser.add_argument('--output_dir', type=str, default='./nt2_ranked_cv',
                        help='Output directory')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='List of GPU IDs (default: 0 1 2 3)')
    parser.add_argument('--k', type=int, default=6, help='K-mer size (default: 6)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--num_steps', type=int, default=4000,
                        help='Training steps per fold (default: 4000)')
    parser.add_argument('--top_n', type=int, default=None,
                        help='Only run CV for the top N ranked experiments (default: all)')

    args = parser.parse_args()

    # Load ranking for display
    with open(args.ranking_json) as f:
        ranking = json.load(f)
    lora_ranking = [r for r in ranking if r.get('fine_tuning') == 'lora']
    lora_ranking.sort(key=lambda r: r['mean_rank'])

    if args.top_n is not None:
        lora_ranking = lora_ranking[:args.top_n]

    print("=" * 80)
    print(f"NT2 Rank-Ordered 5-Fold Chromosome-Split CV")
    print(f"  Experiments: {len(lora_ranking)} LoRA configs (ranked by sweep mean rank)")
    print(f"  Folds: 5 (chromosome-based)")
    print(f"  Total jobs: {len(lora_ranking) * 5}")
    print(f"  GPUs: {args.gpus}")
    print(f"  Steps/fold: {args.num_steps}, Batch size: {args.batch_size}")
    print("-" * 80)
    print("  Execution order:")
    for i, r in enumerate(lora_ranking):
        print(f"    {i+1}. exp_{r['exp_id']:2d} (rank={r['mean_rank']:5.1f}): "
              f"{r['classifier']:12s} | {r['embedding']:25s} | "
              f"r{r['lora_rank']:2d} | lr={r['learning_rate']:.0e}")
    print("=" * 80)

    run_ranked_cv(
        ranking_json=args.ranking_json,
        data_path=args.data_path,
        output_dir=args.output_dir,
        available_gpus=args.gpus,
        k=args.k,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        top_n=args.top_n,
    )


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
