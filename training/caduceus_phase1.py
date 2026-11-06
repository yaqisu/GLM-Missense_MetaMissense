# CRITICAL NOTE ON VARIANT POSITION:
# The input sequences are ALREADY CENTERED with the variant at position 15,000 (1-based).
# The 'position' column in the input dataframe is NOT used - it contains genomic 
# coordinates or other reference information, NOT the position within the input sequence.
# The variant position is HARDCODED as:
#   - VARIANT_POS_1BASED = 15000 (center of 29,999nt sequence, 1-based)
        # Note only [SEP] is added at the end of sequences (no [CLS] at start)
#   - Token position: 14,999 (0-based indexing)
# 
# This is different from NT which uses 6-mer tokenization.

# ============================= imports and setup ==========================================
import os
import sys
import time
import json
import math
import argparse
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ============================= config ==========================================

# using 131k Caduceus model and 30k length ClinVar dataset  
MODEL_NAME = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
TRAIN_PATH = "data/splits/ClinVar.251103.missense.hg38.seq30k.BvsP_training.tsv"
VAL_PATH   = "data/splits/ClinVar.251103.missense.hg38.seq30k.BvsP_validation.tsv"

# Using alt sequence only here
SEQ_COLUMN    = "alt_sequence"
LABEL_COLUMN  = "label"
# Note: position column in dataframe is NOT the variant position in the input sequence
# Variant position is hardcoded based on VARIANT_POS_1BASED below

# Input is 29999nt sequence centered at variant position (variant at position 15000, 1-based)
MAX_LEN       = 29999
VARIANT_POS_1BASED = 15000 

# Downsample to 2048 tokens CENTERED at variant position for CNN/Transformer heads
DOWNSAMPLED_L = 2048

# Train on ALL samples (no subset)
SUBSET_TRAIN  = None
SUBSET_VAL    = None

EMBED_BATCH_SIZE = 25
HEAD_BATCH_SIZE  = 32

# Match NT hyperparameters
N_STEPS_HEAD = 5000
LR_HEAD      = 3e-5
LOG_EVERY    = 1000
WARMUP_RATIO = 0.06  # 6% warmup like NT
WEIGHT_DECAY = 0.01

# Experiment configurations matching NT (frozen only)
EXPERIMENT_CONFIGS = [
    {
        "exp_id": 1,
        "classifier_type": "mlp",
        "num_layers": 2,
        "embedding_strategy": "variant_position",
    },
    {
        "exp_id": 2,
        "classifier_type": "mlp",
        "num_layers": 2,
        "embedding_strategy": "mean_pool",
    },
    {
        "exp_id": 3,
        "classifier_type": "mlp",
        "num_layers": 3,
        "embedding_strategy": "variant_position",
    },
    {
        "exp_id": 5,
        "classifier_type": "transformer",
        "num_layers": 2,
        "embedding_strategy": "full-variant_position",
    },
    {
        "exp_id": 6,
        "classifier_type": "transformer",
        "num_layers": 2,
        "embedding_strategy": "full-mean_pool",
    },
    {
        "exp_id": 7,
        "classifier_type": "cnn",
        "num_layers": 2,
        "embedding_strategy": "full-mean_pool",
    },
    {
        "exp_id": 8,
        "classifier_type": "cnn",
        "num_layers": 2,
        "embedding_strategy": "full-variant_position",
    },
]


def load_clinvar_tsv(path, subset=None):
    df = pd.read_csv(path, sep="\t")
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if subset is not None:
        df = df.iloc[:subset].copy()
    # Only need sequence and label columns - position column is NOT used
    cols = [c for c in [SEQ_COLUMN, LABEL_COLUMN] if c in df.columns]
    df = df[cols].copy()
    df.rename(columns={SEQ_COLUMN: "sequence", LABEL_COLUMN: "label"}, inplace=True)
    return df


class ClinVarSeqDataset(Dataset):
    def __init__(self, df):
        self.seqs = df["sequence"].tolist()
        self.labels = torch.tensor(df["label"].astype(int).values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        return seq, label

def seq_collate_fn(batch):
    seqs, labels = zip(*batch)
    labels = torch.stack(labels, dim=0)
    return list(seqs), labels


def get_hidden_size(model):
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    elif hasattr(cfg, "d_model"):
        return cfg.d_model
    else:
        raise ValueError("Cannot find hidden size in model.config")


def center_downsample_sequence(H, variant_pos, target_len):
    """
    Downsample sequence H (B, L, D) to target_len tokens CENTERED at variant_pos.
    
    Since the input sequences already have variants centered, we simply downsample
    uniformly and the variant remains centered.
    
    Uses adaptive average pooling which preserves information from ALL input tokens
    by averaging regions - better than interpolation for heavy downsampling (30k→2k).
    
    Args:
        H: (B, L, D) tensor of embeddings (e.g., B=32, L=30000, D=256)
        variant_pos: (B,) tensor of variant positions (0-indexed, should be ~L//2)
        target_len: target sequence length after downsampling (e.g., 2048)
    
    Returns:
        H_down: (B, target_len, D) downsampled embeddings
        variant_pos_down: (B,) new variant positions (~target_len//2, 0-indexed)
    """
    B, L, D = H.shape
    
    # Uniform downsampling using adaptive average pooling
    # This averages regions of the input, preserving information from all tokens
    H_perm = H.permute(0, 2, 1)  # (B, D, L)
    H_down = F.adaptive_avg_pool1d(H_perm, output_size=target_len)  # (B, D, target_len)
    H_down = H_down.permute(0, 2, 1)  # (B, target_len, D)
    
    # Scale variant positions proportionally (0-based indexing)
    variant_pos_down = (variant_pos.float() * target_len / L).long()
    variant_pos_down = torch.clamp(variant_pos_down, 0, target_len - 1)
    
    return H_down, variant_pos_down


@torch.no_grad()
def compute_embeddings(df, tokenizer, backbone, device,
                       max_len=MAX_LEN,
                       batch_size=EMBED_BATCH_SIZE,
                       downsample_L=DOWNSAMPLED_L):
    ds = ClinVarSeqDataset(df)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=seq_collate_fn)

    all_mean, all_pos, all_seq, all_labels = [], [], [], []

    backbone.eval()
    for seqs, labels in tqdm(loader, desc="Embedding", ncols=80):
        enc = tokenizer(
            seqs,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            pad_id = 4  # Caduceus PAD token ID
            attention_mask = (enc["input_ids"] != pad_id).long()
        attention_mask = attention_mask.to(device)

        outputs = backbone(input_ids=input_ids, output_hidden_states=True)
        H = outputs.hidden_states[-1]    # (B,L,D)
        B, L, D = H.shape

        # mean-pooled embedding
        mask = attention_mask.unsqueeze(-1)      # (B,L,1)
        H_masked = H * mask                      # zero out pads
        lengths = mask.sum(dim=1).clamp(min=1)   # (B,1)
        mean_emb = H_masked.sum(dim=1) / lengths # (B,D)

        # variant-pos embedding
        # IMPORTANT: Sequences are already centered at the variant
        # Since input is 29,999nt centered at position 15,000 (1-based),
        # the variant is at position 14,999 (0-based) in the sequence
        # With character-level tokenization, this corresponds to token position 14,999
        variant_pos_0based = (VARIANT_POS_1BASED - 1)  # 15000 - 1 = 14999
        pos_idx = torch.full((B,), variant_pos_0based, dtype=torch.long, device=device)
        
        # Ensure position is within bounds after tokenization
        pos_idx = torch.clamp(pos_idx, 0, L - 1)

        pos_emb = H[torch.arange(B, device=device), pos_idx, :]  # (B,D)

        # Downsample full sequence CENTERED at variant position
        # Since variant is already at center (pos_idx), it will remain centered after downsampling
        H_down, downsampled_vpos = center_downsample_sequence(H, pos_idx, downsample_L)

        all_mean.append(mean_emb.cpu().float())
        all_pos.append(pos_emb.cpu().float())
        all_seq.append(H_down.cpu().float())
        all_labels.append(labels)

    mean_emb = torch.cat(all_mean, dim=0)
    pos_emb  = torch.cat(all_pos, dim=0)
    seq_emb  = torch.cat(all_seq, dim=0)
    labels   = torch.cat(all_labels, dim=0)
    
    # Compute downsampled variant positions (will be same for all samples since variant is at center)
    # For simplicity, just compute once
    _, sample_vpos = center_downsample_sequence(
        torch.randn(1, max_len, 1).to(device), 
        torch.tensor([variant_pos_0based], device=device),
        downsample_L
    )
    vpos = sample_vpos.repeat(len(labels)).cpu()
    
    return mean_emb, pos_emb, seq_emb, labels, vpos


# ----------------- Model Heads (Matching NT) ----------------- 

# MLP Classifiers matching NT architecture with dropout
class MLP2(nn.Module):
    """2-layer MLP matching NT architecture"""
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.classifier(x).squeeze(-1)


class MLP3(nn.Module):
    """3-layer MLP matching NT architecture"""
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
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
    
    def forward(self, x):
        return self.classifier(x).squeeze(-1)


# CNN Classifier matching NT architecture
class CNNHead(nn.Module):
    """CNN classifier matching NT architecture"""
    def __init__(self, input_dim, dropout=0.1, pooling_strategy='mean_pool'):
        super().__init__()
        
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
            # Average pooling over sequence length
            if attention_mask is not None:
                # Masked mean pooling
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = torch.sum(attention_mask_expanded * x, dim=1)
                sum_mask = torch.sum(attention_mask_expanded, dim=1)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = torch.mean(x, dim=1) 
        elif self.pooling_strategy == 'variant_position':
            # Extract embedding at variant position
            if variant_position is None:
                raise ValueError("variant_position must be provided for variant_position pooling")
            # Handle both single integer and tensor
            if isinstance(variant_position, int):
                pooled = x[:, variant_position, :]
            else:
                # variant_position is a tensor (B,)
                batch_indices = torch.arange(x.shape[0], device=x.device)
                pooled = x[batch_indices, variant_position, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        logits = self.fc(pooled)
        return logits.squeeze(-1)


# Transformer Classifier matching NT architecture
class TransformerHead(nn.Module):
    """Transformer classifier matching NT architecture"""
    def __init__(self, input_dim, embed_dim=128, nhead=2, 
                 dim_feedforward=512, dropout=0.1, pooling_strategy='mean_pool'):
        super().__init__()
        
        self.pooling_strategy = pooling_strategy
        
        # Project input to embed_dim
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Transformer encoder (2 layers like NT)
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
            # Average pooling over sequence length
            if attention_mask is not None:
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = torch.sum(attention_mask_expanded * x, dim=1)
                sum_mask = torch.sum(attention_mask_expanded, dim=1)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = torch.mean(x, dim=1)  
        elif self.pooling_strategy == 'variant_position':
            # Extract embedding at variant position
            if variant_position is None:
                raise ValueError("variant_position must be provided for variant_position pooling")
            # Handle both single integer and tensor
            if isinstance(variant_position, int):
                pooled = x[:, variant_position, :]
            else:
                # variant_position is a tensor (B,)
                batch_indices = torch.arange(x.shape[0], device=x.device)
                pooled = x[batch_indices, variant_position, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        pooled = self.dropout(pooled)
        
        logits = self.fc(pooled)
        return logits.squeeze(-1)


# =================== Dataset classes ==============================

class VectorDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SeqDataset(Dataset):
    def __init__(self, H, vpos, y):
        self.H, self.vpos, self.y = H, vpos, y
    def __len__(self):
        return self.H.shape[0]
    def __getitem__(self, idx):
        return self.H[idx], self.vpos[idx], self.y[idx]


# =================== Learning Rate Scheduler ==============================

class LRSchedulerWithWarmup:
    """Linear warmup + linear decay scheduler matching NT"""
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr_scale = self.current_step / self.warmup_steps
        else:
            # Linear decay
            remaining_steps = self.total_steps - self.warmup_steps
            steps_since_warmup = self.current_step - self.warmup_steps
            lr_scale = 1.0 - (steps_since_warmup / remaining_steps)
            lr_scale = max(0.0, lr_scale)
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# =================== Training utilities (matching NT) ==============================

def train_vector_steps(model, train_ds, val_ds, device, exp_id,
                       n_steps=N_STEPS_HEAD,
                       batch_size=HEAD_BATCH_SIZE,
                       lr=LR_HEAD,
                       log_every=LOG_EVERY):
    """Training for vector-based models (MLP)"""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss()
    
    # Setup optimizer matching NT (AdamW with weight decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optim = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.999))
    
    # Setup learning rate scheduler with warmup
    warmup_steps = int(WARMUP_RATIO * n_steps)
    scheduler = LRSchedulerWithWarmup(optim, warmup_steps, n_steps)

    # Setup CSV logging
    os.makedirs(f"results_caduceus_phase1/exp_{exp_id}", exist_ok=True)
    csv_path = f"results_caduceus_phase1/exp_{exp_id}/training_metrics.csv"
    metrics_df_init = pd.DataFrame(columns=[
        'steps', 'train_loss', 'train_auc', 'val_loss', 
        'val_auc', 'learning_rate', 'gpu_memory_gb'
    ])
    metrics_df_init.to_csv(csv_path, index=False)

    global_step = 0
    best_auc = 0.0
    train_step_auc = {}
    val_step_auc   = {}

    t0 = time.time()
    train_iterator = iter(train_loader)
    
    while global_step < n_steps:
        model.train()
        
        try:
            X, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            X, y = next(train_iterator)
        
        X = X.to(device).float()
        y = y.to(device).float()
        
        logits = model(X)
        loss = loss_fn(logits, y)
        train_loss = loss.item()
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        
        global_step += 1

        if global_step % log_every == 0:
            # Evaluate on full train batch
            train_loss_full, train_auc = eval_vector(model, train_loader, loss_fn, device)
            val_loss, val_auc = eval_vector(model, val_loader, loss_fn, device)
            
            best_auc = max(best_auc, val_auc)
            train_step_auc[global_step] = train_auc
            val_step_auc[global_step]   = val_auc
            
            lr_current = scheduler.get_last_lr()[0]
            gpu_mem_gb = torch.cuda.memory_allocated(device)/1024**3
            
            # Save metrics to CSV
            current_metrics = pd.DataFrame({
                'steps': [global_step],
                'train_loss': [train_loss_full],
                'train_auc': [train_auc],
                'val_loss': [val_loss],
                'val_auc': [val_auc],
                'learning_rate': [lr_current],
                'gpu_memory_gb': [gpu_mem_gb]
            })
            current_metrics.to_csv(csv_path, mode='a', header=False, index=False)
            
            print(f"[step {global_step}/{n_steps}] train_loss={train_loss_full:.4f}, train_auc={train_auc:.4f}, val_loss={val_loss:.4f}, val_auc={val_auc:.4f}, lr={lr_current:.2e}")

    elapsed_hr = (time.time() - t0)/3600.0
    gpu_mem_gb = torch.cuda.max_memory_allocated(device)/1024**3 if torch.cuda.is_available() else 0.0
    return best_auc, train_step_auc, val_step_auc, elapsed_hr, gpu_mem_gb


def eval_vector(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_trues = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device).float()
            y = y.to(device).float()
            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu())
            all_trues.append(y.cpu())
    total_loss /= max(1, len(loader))
    preds = torch.cat(all_preds).numpy()
    trues = torch.cat(all_trues).numpy()
    auc = roc_auc_score(trues, preds)
    return total_loss, auc


def train_seq_steps(model, train_ds, val_ds, device, exp_id, pooling_strategy='mean_pool',
                    n_steps=N_STEPS_HEAD,
                    batch_size=HEAD_BATCH_SIZE,
                    lr=LR_HEAD,
                    log_every=LOG_EVERY):
    """Training for sequence-based models (CNN/Transformer)"""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss()
    
    # Setup optimizer matching NT
    no_decay = ["bias", "LayerNorm.weight", "bn1.weight", "bn2.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optim = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.999))
    
    # Setup learning rate scheduler with warmup
    warmup_steps = int(WARMUP_RATIO * n_steps)
    scheduler = LRSchedulerWithWarmup(optim, warmup_steps, n_steps)

    # Setup CSV logging
    os.makedirs(f"results_caduceus_phase1/exp_{exp_id}", exist_ok=True)
    csv_path = f"results_caduceus_phase1/exp_{exp_id}/training_metrics.csv"
    metrics_df_init = pd.DataFrame(columns=[
        'steps', 'train_loss', 'train_auc', 'val_loss', 
        'val_auc', 'learning_rate', 'gpu_memory_gb'
    ])
    metrics_df_init.to_csv(csv_path, index=False)

    global_step = 0
    best_auc = 0.0
    train_step_auc = {}
    val_step_auc   = {}
    t0 = time.time()
    
    train_iterator = iter(train_loader)

    while global_step < n_steps:
        model.train()
        
        try:
            H, vpos, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            H, vpos, y = next(train_iterator)
        
        H = H.to(device).float()
        vpos = vpos.to(device).long()
        y = y.to(device).float()
        
        # Call model based on pooling strategy
        if pooling_strategy == 'variant_position':
            logits = model(H, variant_position=vpos)
        else:  # mean_pool
            logits = model(H)
        
        loss = loss_fn(logits, y)
        train_loss = loss.item()
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        
        global_step += 1

        if global_step % log_every == 0:
            # Evaluate on full datasets
            train_loss_full, train_auc = eval_seq(model, train_loader, loss_fn, device, pooling_strategy)
            val_loss, val_auc = eval_seq(model, val_loader, loss_fn, device, pooling_strategy)
            
            best_auc = max(best_auc, val_auc)
            train_step_auc[global_step] = train_auc
            val_step_auc[global_step]   = val_auc
            
            lr_current = scheduler.get_last_lr()[0]
            gpu_mem_gb = torch.cuda.memory_allocated(device)/1024**3
            
            # Save metrics to CSV
            current_metrics = pd.DataFrame({
                'steps': [global_step],
                'train_loss': [train_loss_full],
                'train_auc': [train_auc],
                'val_loss': [val_loss],
                'val_auc': [val_auc],
                'learning_rate': [lr_current],
                'gpu_memory_gb': [gpu_mem_gb]
            })
            current_metrics.to_csv(csv_path, mode='a', header=False, index=False)
            
            print(f"[step {global_step}/{n_steps}] train_loss={train_loss_full:.4f}, train_auc={train_auc:.4f}, val_loss={val_loss:.4f}, val_auc={val_auc:.4f}, lr={lr_current:.2e}")

    elapsed_hr = (time.time() - t0)/3600.0
    gpu_mem_gb = torch.cuda.max_memory_allocated(device)/1024**3 if torch.cuda.is_available() else 0.0
    return best_auc, train_step_auc, val_step_auc, elapsed_hr, gpu_mem_gb


def eval_seq(model, loader, loss_fn, device, pooling_strategy='mean_pool'):
    model.eval()
    total_loss = 0.0
    all_preds, all_trues = [], []
    
    with torch.no_grad():
        for H, vpos, y in loader:
            H = H.to(device).float()
            vpos = vpos.to(device).long()
            y = y.to(device).float()
            
            if pooling_strategy == 'variant_position':
                logits = model(H, variant_position=vpos)
            else:
                logits = model(H)
            
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            
            all_preds.append(torch.sigmoid(logits).cpu())
            all_trues.append(y.cpu())
    
    total_loss /= max(1, len(loader))
    preds = torch.cat(all_preds).numpy()
    trues = torch.cat(all_trues).numpy()
    auc = roc_auc_score(trues, preds)
    return total_loss, auc


# =================== Single Experiment Runner ==============================

def run_single_experiment(args):
    """Run a single experiment on a specified GPU"""
    exp_config, gpu_id, embeddings_dict, hidden_dim = args
    
    # Set device for this process
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)
    
    exp_id = exp_config["exp_id"]
    classifier_type = exp_config["classifier_type"]
    num_layers = exp_config["num_layers"]
    embedding_strategy = exp_config["embedding_strategy"]
    
    print(f"\n{'='*80}")
    print(f"GPU {gpu_id} - Experiment {exp_id}: {classifier_type} (layers={num_layers}, strategy={embedding_strategy})")
    print(f"{'='*80}")
    
    # Get embeddings from dict (already on CPU)
    train_mean = embeddings_dict['train_mean']
    train_pos = embeddings_dict['train_pos']
    train_seq = embeddings_dict['train_seq']
    y_train = embeddings_dict['y_train']
    train_vpos = embeddings_dict['train_vpos']
    
    val_mean = embeddings_dict['val_mean']
    val_pos = embeddings_dict['val_pos']
    val_seq = embeddings_dict['val_seq']
    y_val = embeddings_dict['y_val']
    val_vpos = embeddings_dict['val_vpos']
    
    # Create datasets
    train_vec_pos = VectorDataset(train_pos, y_train)
    train_vec_mean = VectorDataset(train_mean, y_train)
    val_vec_pos = VectorDataset(val_pos, y_val)
    val_vec_mean = VectorDataset(val_mean, y_val)
    
    train_seq_ds = SeqDataset(train_seq, train_vpos, y_train)
    val_seq_ds = SeqDataset(val_seq, val_vpos, y_val)
    
    # Create model based on config
    if classifier_type == "mlp":
        if num_layers == 2:
            model = MLP2(hidden_dim).to(device)
        elif num_layers == 3:
            model = MLP3(hidden_dim).to(device)
        else:
            raise ValueError(f"Unsupported num_layers for MLP: {num_layers}")
        
        # Select dataset based on embedding strategy
        if embedding_strategy == "variant_position":
            train_ds, val_ds = train_vec_pos, val_vec_pos
        elif embedding_strategy == "mean_pool":
            train_ds, val_ds = train_vec_mean, val_vec_mean
        else:
            raise ValueError(f"Unsupported embedding strategy for MLP: {embedding_strategy}")
        
        # Train
        best_auc, t_train, v_train, hours, mem_gb = train_vector_steps(
            model, train_ds, val_ds, device, exp_id
        )
        
    elif classifier_type == "cnn":
        # Determine pooling strategy
        if embedding_strategy.startswith('full-'):
            pooling_strategy = embedding_strategy.split('-', 1)[1]
        else:
            pooling_strategy = 'mean_pool'
        
        model = CNNHead(hidden_dim, pooling_strategy=pooling_strategy).to(device)
        
        # Train
        best_auc, t_train, v_train, hours, mem_gb = train_seq_steps(
            model, train_seq_ds, val_seq_ds, device, exp_id, pooling_strategy=pooling_strategy
        )
        
    elif classifier_type == "transformer":
        # Determine pooling strategy
        if embedding_strategy.startswith('full-'):
            pooling_strategy = embedding_strategy.split('-', 1)[1]
        else:
            pooling_strategy = 'mean_pool'
        
        model = TransformerHead(hidden_dim, pooling_strategy=pooling_strategy).to(device)
        
        # Train
        best_auc, t_train, v_train, hours, mem_gb = train_seq_steps(
            model, train_seq_ds, val_seq_ds, device, exp_id, pooling_strategy=pooling_strategy
        )
        
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    # Store results
    result = {
        "exp_id": exp_id,
        "gpu_id": gpu_id,
        "config": exp_config,
        "best_val_auc": best_auc,
        "train_step_auc": t_train,
        "val_step_auc": v_train,
        "train_time_hr": hours,
        "train_mem_gb": mem_gb,
    }
    
    print(f"GPU {gpu_id} - Experiment {exp_id} completed: best_val_auc={best_auc:.4f}")
    
    return result


# =================== Main Execution ==============================

def run_phase1_caduceus(gpus):
    print("=" * 80)
    print("Phase 1: Frozen Caduceus Fine-tuning (Matching NT Setup)")
    print(f"Using GPUs: {gpus}")
    print("=" * 80)
    
    # Use first GPU for embedding extraction
    embed_device = torch.device(f'cuda:{gpus[0]}')
    
    # Load model and tokenizer (frozen backbone)
    print(f"Loading model on GPU {gpus[0]}: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(embed_device)
    
    # Freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False
    
    # Get hidden dimension
    cfg = backbone.config
    if hasattr(cfg, "hidden_size"):
        config_hidden_dim = cfg.hidden_size
    elif hasattr(cfg, "d_model"):
        config_hidden_dim = cfg.d_model
    else:
        raise ValueError("Cannot find hidden size in model.config")
    print(f"Backbone hidden dim: {config_hidden_dim}")

    # Load data (ALL samples, no subset)
    print(f"\nLoading training data from: {TRAIN_PATH}")
    df_train = load_clinvar_tsv(TRAIN_PATH, subset=SUBSET_TRAIN)
    print(f"Training samples: {len(df_train)}")
    
    print(f"Loading validation data from: {VAL_PATH}")
    df_val = load_clinvar_tsv(VAL_PATH, subset=SUBSET_VAL)
    print(f"Validation samples: {len(df_val)}")

    # Compute embeddings
    print("\n" + "=" * 80)
    print("Computing embeddings...")
    print("=" * 80)
    print("Embedding train...")
    train_mean, train_pos, train_seq, y_train, train_vpos = compute_embeddings(
        df_train, tokenizer, backbone, embed_device
    )
    print(f"Train embeddings: mean={train_mean.shape}, pos={train_pos.shape}, seq={train_seq.shape}")
    
    print("Embedding val...")
    val_mean, val_pos, val_seq, y_val, val_vpos = compute_embeddings(
        df_val, tokenizer, backbone, embed_device
    )
    print(f"Val embeddings: mean={val_mean.shape}, pos={val_pos.shape}, seq={val_seq.shape}")

    # Infer actual hidden dimension from embeddings (may differ from config due to bidirectionality)
    hidden_dim = train_mean.shape[-1]
    print(f"\nActual hidden dim (from embeddings): {hidden_dim}")
    if hidden_dim != config_hidden_dim:
        print(f"Note: Config d_model={config_hidden_dim} but actual output dim={hidden_dim}")
        print(f"      This is expected for bidirectional models (e.g., {config_hidden_dim} × 2 = {hidden_dim})")
    
    # Free up backbone memory
    del backbone
    torch.cuda.empty_cache()
    
    # Store embeddings in a dict for passing to workers
    embeddings_dict = {
        'train_mean': train_mean,
        'train_pos': train_pos,
        'train_seq': train_seq,
        'y_train': y_train,
        'train_vpos': train_vpos,
        'val_mean': val_mean,
        'val_pos': val_pos,
        'val_seq': val_seq,
        'y_val': y_val,
        'val_vpos': val_vpos,
    }

    print("\n" + "=" * 80)
    print("Running Experiments in Parallel")
    print("=" * 80)

    # Prepare arguments for each experiment
    experiment_args = []
    for i, exp_config in enumerate(EXPERIMENT_CONFIGS):
        gpu_id = gpus[i % len(gpus)]  # Round-robin GPU assignment
        experiment_args.append((exp_config, gpu_id, embeddings_dict, hidden_dim))
    
    # Run experiments in parallel
    # Use spawn method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    num_parallel = len(gpus)
    all_results = []
    
    # Run in batches equal to number of GPUs
    for i in range(0, len(experiment_args), num_parallel):
        batch = experiment_args[i:i+num_parallel]
        print(f"\nRunning batch {i//num_parallel + 1}: Experiments {[args[0]['exp_id'] for args in batch]}")
        
        with mp.Pool(processes=len(batch)) as pool:
            results = pool.map(run_single_experiment, batch)
            all_results.extend(results)
    
    # Organize results
    results_dict = {}
    for result in all_results:
        exp_id = result['exp_id']
        config = result['config']
        classifier_type = config['classifier_type']
        embedding_strategy = config['embedding_strategy']
        exp_name = f"Exp{exp_id}_{classifier_type}_{embedding_strategy}"
        results_dict[exp_name] = result

    # Save results
    os.makedirs("results_caduceus_phase1", exist_ok=True)
    
    with open("results_caduceus_phase1/phase1_caduceus_modified.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    with open("results_caduceus_phase1/metadata_modified.json", "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "hidden_dim": hidden_dim,
            "max_len": MAX_LEN,
            "downsample_length": DOWNSAMPLED_L,
            "variant_position_1based": VARIANT_POS_1BASED,
            "n_train_samples": len(df_train),
            "n_val_samples": len(df_val),
            "n_steps": N_STEPS_HEAD,
            "learning_rate": LR_HEAD,
            "batch_size": HEAD_BATCH_SIZE,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "gpus_used": gpus,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("Phase 1 Complete!")
    print("=" * 80)
    print("\nResults Summary:")
    for exp_name, result in sorted(results_dict.items()):
        print(f"{exp_name}: best_val_auc={result['best_val_auc']:.4f} (GPU {result['gpu_id']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run Caduceus Phase 1 experiments across multiple GPUs'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        default=[0],
        help='List of GPU IDs to use (e.g., --gpus 0 1 2)'
    )
    
    args = parser.parse_args()
    
    # Validate GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        sys.exit(1)
    
    num_gpus = torch.cuda.device_count()
    for gpu_id in args.gpus:
        if gpu_id >= num_gpus:
            print(f"ERROR: GPU {gpu_id} is not available. Only {num_gpus} GPU(s) detected.")
            sys.exit(1)
    
    run_phase1_caduceus(args.gpus)
