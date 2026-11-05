#!/usr/bin/env python3
"""
ESM-1b Ref-Alt Contrast — Siamese LoRA Fine-Tuning with MLP Head
=================================================================
Protein-level matched control for GLM-Missense. Mirrors NT2_ref_alt_contrast.py
(the GLM-Missense training script) with the NT-2 backbone swapped for ESM-1b:

    ref protein → ESM-1b+LoRA → [mutated-residue token] → shared MLPProjector → ref_feat
    alt protein → ESM-1b+LoRA → [mutated-residue token] → shared MLPProjector → alt_feat
    combine(ref_feat, alt_feat) → MLPClassifierHead → logit

Training settings copied from GLM-Missense:
    LoRA r=32, alpha=64, dropout=0.1, target modules query/value
    Projector hidden→512→256 (ReLU, dropout 0.1); head 3*256→512→256→1
    combine_mode concat_diff = [ref, alt, ref - alt]
    AdamW lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight decay 0.01
        (no decay on bias / LayerNorm.weight)
    6% linear warmup, then linear decay, scheduled over 17,000 optimizer steps
    batch 8 x grad accum 4 = effective batch 32; BCEWithLogitsLoss; fp32
    eval every 1,000 steps; best checkpoint = highest validation AUROC
    (full val set, or a fixed label-stratified subset with --val_eval_samples)
    train metrics on a random 10k-sample subset at each eval

Implementation differences from the NT-2 script (training settings unchanged):
    - ESM-1b accepts at most 1,022 residues; longer proteins use a 1,022-residue
      window around the mutated residue (shifted inward at protein ends)
    - variant token index varies per sample (residue position + 1 for <cls>),
      and batches are padded dynamically instead of to a fixed length
    - checkpoints store only trainable parameters (LoRA + projector + head);
      frozen ESM-1b weights are reloaded from the hub
    - --val_eval_samples draws its subset like the NT-2 script (same function and
      seed, on ALL rows of the val TSV in file order), then drops rows that are
      invalid here, so a protein TSV row-aligned with the NT-2 val TSV is evaluated
      on the same variants as GLM-Missense


Input TSV (one row per variant; build from the same variant_ids / splits as
the NT-2 TSVs so both models see identical training and validation variants):
    variant_id        : identifier (for merging with GLM-Missense outputs)
    protein_sequence  : full wild-type protein sequence, 1-letter codes
                        (e.g. MANE Select); a trailing '*' is stripped
    protein_position  : 1-based position of the mutated residue
    ref_aa, alt_aa    : 1-letter reference / alternate amino acids
    label             : 0/1 (optional in --predict_path mode)
Rows with a missing sequence, an out-of-range position, a non-standard amino
acid, or ref_aa not matching protein_sequence at protein_position are skipped;
the counts are logged and written to summary.json.

Usage:
    # train
    python ESM1b_ref_alt_contrast.py --train_path train.tsv --val_path val.tsv \
        --gpus 0 1 2 3
    # single GPU (same effective batch of 32)
    python ESM1b_ref_alt_contrast.py --train_path train.tsv --val_path val.tsv \
        --gpus 0 --batch_size 4 --gradient_accumulation_steps 8
    # score a TSV with a trained model
    python ESM1b_ref_alt_contrast.py --predict_path later_release.tsv \
        --checkpoint results/ESM1b_ref_alt_contrast/exp_1_concat_diff/best_model.pt
"""

import sys
import time
import json
import random
import logging
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, EsmModel
from peft import LoraConfig, get_peft_model, TaskType


DEFAULT_MODEL = "facebook/esm1b_t33_650M_UR50S"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
METRIC_COLS = [
    "steps",
    "train_loss_subset", "train_auroc_subset", "train_auprc_subset",
    "val_loss", "val_auroc", "val_auprc",
    "learning_rate", "gpu_memory_gb",
]


# ============================================================================
# Experiment Configs (identical to the GLM-Missense config)
# ============================================================================

EXPERIMENT_CONFIGS = [
    {
        "exp_id": 1,
        "description": "ESM-1b ref-alt contrast Siamese MLP, combine=concat_diff",
        "combine_mode": "concat_diff",
        "lora_rank": 32,
        "batch_size": 16,                    # 8 total / 4 GPUs = 2 per GPU
        "num_steps": 17000,                 # (3.6 epochs x ~151k) / 32
        "gradient_accumulation_steps": 2,   # effective batch = 8 * 4 = 32
        "learning_rate": 5e-5,
    }
]


# ============================================================================
# Model
# ============================================================================

class MLPProjector(nn.Module):
    """Linear → ReLU → Dropout → Linear → ReLU → Dropout (same as NT-2 script)."""
    def __init__(self, input_dim, proj_dim=256, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.projector(x)


class MLPClassifierHead(nn.Module):
    """input → 512 → 256 → 1 (same as NT-2 script)."""
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
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


# Head input size = multiplier x proj_dim
HEAD_INPUT_MULT = {
    "concat_diff": 3,   # [ref, alt, ref - alt]  (GLM-Missense setting)
    "concat_only": 2,   # [ref, alt]
    "diff_only": 1,     # ref - alt
    "alt_only": 1,      # alt                     (control; no ref pass)
    "alt_x3": 3,        # [alt, alt, alt]         (dimension-matched control)
}


class ESM_RefAltContrast(nn.Module):
    """Siamese ESM-1b with a shared LoRA backbone, shared projector and MLP head."""

    def __init__(self, base_model, lora_rank=32, combine_mode="concat_diff",
                 proj_dim=256, dropout=0.1):
        super().__init__()
        if combine_mode not in HEAD_INPUT_MULT:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")
        self.combine_mode = combine_mode

        for param in base_model.parameters():
            param.requires_grad = False
        hidden_size = base_model.config.hidden_size

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["query", "value"],
            bias="none",
        )
        self.backbone = get_peft_model(base_model, lora_config)
        print("=" * 80)
        print(f"LoRA Configuration (rank={lora_rank}):")
        self.backbone.print_trainable_parameters()
        print("=" * 80)

        self.projector = MLPProjector(hidden_size, proj_dim=proj_dim, dropout=dropout)
        self.classifier = MLPClassifierHead(
            HEAD_INPUT_MULT[combine_mode] * proj_dim, dropout=dropout
        )

    def _residue_embedding(self, input_ids, attention_mask, var_pos):
        """Last-layer embedding at the mutated residue token: (batch, hidden)."""
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[idx, var_pos]

    def forward(self, ref_input_ids, alt_input_ids, attention_mask, var_pos):
        mode = self.combine_mode
        alt_feat = self.projector(
            self._residue_embedding(alt_input_ids, attention_mask, var_pos)
        )
        if mode == "alt_only":
            combined = alt_feat
        elif mode == "alt_x3":
            combined = torch.cat([alt_feat, alt_feat, alt_feat], dim=1)
        else:
            ref_feat = self.projector(
                self._residue_embedding(ref_input_ids, attention_mask, var_pos)
            )
            if mode == "concat_diff":
                combined = torch.cat([ref_feat, alt_feat, ref_feat - alt_feat], dim=1)
            elif mode == "concat_only":
                combined = torch.cat([ref_feat, alt_feat], dim=1)
            else:  # diff_only
                combined = ref_feat - alt_feat
        return self.classifier(combined)


def build_model(model_name, exp_config):
    base_model = EsmModel.from_pretrained(model_name, add_pooling_layer=False)
    return ESM_RefAltContrast(
        base_model,
        lora_rank=exp_config["lora_rank"],
        combine_mode=exp_config["combine_mode"],
    )


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def trainable_state_dict(model):
    """State dict restricted to trainable parameters (LoRA + projector + head)."""
    m = unwrap(model)
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in m.state_dict().items() if k in names}


def load_trainable_state_dict(model, state_dict):
    m = unwrap(model)
    _, unexpected = m.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]} ...")
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    not_loaded = trainable - set(state_dict)
    if not_loaded:
        raise RuntimeError(
            f"{len(not_loaded)} trainable parameters missing from checkpoint, "
            f"e.g. {sorted(not_loaded)[:3]}"
        )


# ============================================================================
# Data
# ============================================================================

def window_sequence(seq, pos0, max_residues):
    """Return (window, position of the mutated residue within the window)."""
    length = len(seq)
    if length <= max_residues:
        return seq, pos0
    start = min(max(pos0 - max_residues // 2, 0), length - max_residues)
    return seq[start:start + max_residues], pos0 - start


def load_variant_table(path, require_label=True):
    """
    Read and validate a variant TSV. Adds columns _seq, _pos0, _alt, _valid.
    Invalid rows are kept in the returned frame (so predictions can be merged
    back by variant_id) but flagged _valid=False.
    """
    df = pd.read_csv(path, sep="\t")
    required = {"variant_id", "protein_sequence", "protein_position",
                "ref_aa", "alt_aa"}
    if require_label:
        required.add("label")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}. "
                         f"Found: {list(df.columns)}")

    seqs = (df["protein_sequence"].fillna("").astype(str)
            .str.strip().str.upper().str.rstrip("*"))
    pos = pd.to_numeric(df["protein_position"], errors="coerce")
    ref = df["ref_aa"].fillna("").astype(str).str.strip().str.upper()
    alt = df["alt_aa"].fillna("").astype(str).str.strip().str.upper()
    if "label" in df.columns:
        labels = pd.to_numeric(df["label"], errors="coerce")
    else:
        labels = pd.Series(np.nan, index=df.index)

    valid = []
    for s, p, r, a, y in zip(seqs, pos, ref, alt, labels):
        ok = (
            pd.notna(p) and float(p).is_integer() and 1 <= int(p) <= len(s)
            and r in STANDARD_AA and a in STANDARD_AA and r != a
            and s[int(p) - 1] == r
        )
        if require_label:
            ok = ok and y in (0, 1)
        valid.append(bool(ok))

    df = df.copy()
    df["_seq"] = seqs
    df["_pos0"] = pos.fillna(1).astype(int) - 1
    df["_alt"] = alt
    df["_label"] = labels
    df["_valid"] = valid
    return df


class ProteinPairDataset(Dataset):
    """Builds tokenized ref / alt protein windows and the mutated-token index."""

    def __init__(self, df, tokenizer, max_residues):
        self.seqs = df["_seq"].tolist()
        self.pos0 = df["_pos0"].tolist()
        self.alt = df["_alt"].tolist()
        self.labels = df["_label"].astype(float).tolist()
        self.max_residues = max_residues
        # One token per residue: map characters directly to vocabulary ids so the
        # mutated residue always sits at (position in window + 1).
        self.cls_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.unk_id = tokenizer.unk_token_id
        self.vocab = {c: tokenizer.convert_tokens_to_ids(c)
                      for c in "ACDEFGHIKLMNPQRSTVWYXBZUO"}

    def __len__(self):
        return len(self.seqs)

    def _ids(self, seq):
        return ([self.cls_id]
                + [self.vocab.get(c, self.unk_id) for c in seq]
                + [self.eos_id])

    def __getitem__(self, i):
        win, wp = window_sequence(self.seqs[i], self.pos0[i], self.max_residues)
        alt_win = win[:wp] + self.alt[i] + win[wp + 1:]
        return {
            "ref_ids": self._ids(win),
            "alt_ids": self._ids(alt_win),
            "var_pos": wp + 1,          # +1 for <cls>
            "label": self.labels[i],
        }


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n, max_len = len(batch), max(len(b["ref_ids"]) for b in batch)
        ref = torch.full((n, max_len), self.pad_id, dtype=torch.long)
        alt = torch.full((n, max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros((n, max_len), dtype=torch.long)
        for j, b in enumerate(batch):
            length = len(b["ref_ids"])
            ref[j, :length] = torch.tensor(b["ref_ids"])
            alt[j, :length] = torch.tensor(b["alt_ids"])
            mask[j, :length] = 1
        return {
            "ref_input_ids": ref,
            "alt_input_ids": alt,
            "attention_mask": mask,
            "var_pos": torch.tensor([b["var_pos"] for b in batch], dtype=torch.long),
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.float),
        }


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(dataset, batch_size, shuffle, pad_id, seed=42, num_workers=2):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=PadCollator(pad_id),
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


# ============================================================================
# Training Utilities
# ============================================================================

class LRSchedulerWithWarmup:
    """Linear warmup + linear decay (same as NT-2 script)."""
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            remaining = self.total_steps - self.warmup_steps
            elapsed = self.current_step - self.warmup_steps
            scale = max(0.0, 1.0 - elapsed / remaining)
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base * scale

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {"current_step": self.current_step, "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps, "base_lrs": self.base_lrs}

    def load_state_dict(self, sd):
        self.current_step = sd["current_step"]
        self.warmup_steps = sd["warmup_steps"]
        self.total_steps = sd["total_steps"]
        self.base_lrs = sd["base_lrs"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for h in [logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def forward_batch(model, batch, device):
    return model(
        batch["ref_input_ids"].to(device),
        batch["alt_input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["var_pos"].to(device),
    )


@torch.no_grad()
def run_inference(model, loader, device):
    """Returns (logits, labels) as numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        all_logits.append(forward_batch(model, batch, device).float().cpu())
        all_labels.append(batch["labels"])
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def evaluate(model, loader, device):
    """Returns (loss, auroc, auprc)."""
    logits, labels = run_inference(model, loader, device)
    loss = F.binary_cross_entropy_with_logits(
        torch.from_numpy(logits), torch.from_numpy(labels)).item()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return loss, roc_auc_score(labels, probs), average_precision_score(labels, probs)


VAL_EVAL_SEED = 42  # seed for the fixed --val_eval_samples subset (same as the NT-2 script)


def fixed_subset_indices(labels, n_samples, seed=VAL_EVAL_SEED):
    """
    Indices of a fixed, label-stratified random subset of size ~n_samples.
    Identical to fixed_subset_indices in NT2_ref_alt_contrast_valSubset.py.

    Same seed -> same indices, so val metrics stay comparable across checkpoints
    and after a resume. Stratified by label so the class balance (and hence AUPRC)
    matches the full set. Uses its own numpy RNG, leaving the global torch RNG
    (training shuffle order) untouched.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    frac = n_samples / len(labels)
    picked = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        k = min(len(idx), max(1, round(len(idx) * frac)))
        picked.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(picked)).tolist()


def fixed_val_subset(val_df, n_samples, seed=VAL_EVAL_SEED):
    """
    Positions, within the valid-only val set, of the fixed --val_eval_samples subset.

    The subset is drawn on ALL rows of the val TSV in file order -- what the NT-2
    script draws from its own val TSV -- and then restricted to rows valid here.
    map_variants_to_mane_protein.py writes the protein TSV 1:1 and in order with the
    NT-2 TSV, so both models are evaluated on the same variants.
    Returns (positions, number of rows drawn before dropping invalid ones).
    """
    picked = np.zeros(len(val_df), dtype=bool)
    picked[fixed_subset_indices(val_df["_label"].to_numpy(), n_samples, seed)] = True
    valid = val_df["_valid"].to_numpy(dtype=bool)
    return np.flatnonzero(picked[valid]).tolist(), int(picked.sum())


def generate_plots(metrics_df, exp_dir, exp_id, combine_mode, val_label="Val (full)"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Exp {exp_id} ({combine_mode}) Train subset vs Val", fontsize=14)
    for ax, (tcol, vcol, label) in zip(axes, [
        ("train_loss_subset", "val_loss", "Loss"),
        ("train_auroc_subset", "val_auroc", "AUROC"),
        ("train_auprc_subset", "val_auprc", "AUPRC"),
    ]):
        ax.plot(metrics_df["steps"], metrics_df[tcol], label="Train (random subset)", linewidth=2)
        ax.plot(metrics_df["steps"], metrics_df[vcol], label=val_label, linewidth=2)
        ax.set_xlabel("Steps")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(exp_dir / "training_curves.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================================
# Training
# ============================================================================

def train_experiment(exp_config, args):
    exp_id = exp_config["exp_id"]
    combine_mode = exp_config["combine_mode"]
    num_steps = exp_config["num_steps"]
    batch_size = args.batch_size or exp_config["batch_size"]
    grad_accum = args.gradient_accumulation_steps or exp_config["gradient_accumulation_steps"]
    learning_rate = exp_config["learning_rate"]
    stop_step = min(args.stop_at_step or num_steps, num_steps)
    gpu_list = args.gpus

    exp_dir = Path(args.output_dir) / f"exp_{exp_id}_{combine_mode}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    logger = setup_logger(f"Exp{exp_id}_{combine_mode}", exp_dir / "training.log")

    logger.info(f"Experiment {exp_id}: {exp_config['description']}")
    logger.info(f"Model          : {args.model_name}")
    logger.info(f"Combine mode   : {combine_mode}")
    logger.info(f"Train path     : {args.train_path}")
    logger.info(f"Val path       : {args.val_path}")
    logger.info(f"GPU list       : {gpu_list}")
    logger.info(f"Batch size     : {batch_size} total, grad accum {grad_accum}, "
                f"effective {batch_size * grad_accum}")
    logger.info(f"LR             : {learning_rate}")
    logger.info(f"LR schedule    : {num_steps} steps; training stops at {stop_step}")
    logger.info(f"Max residues   : {args.max_residues}")
    logger.info(f"Seed           : {args.seed}")

    set_seed(args.seed)
    device = torch.device(f"cuda:{gpu_list[0]}")
    torch.cuda.set_device(gpu_list[0])
    for g in gpu_list:
        torch.cuda.reset_peak_memory_stats(g)
    start_time = time.time()

    try:
        # ---- Data ----
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        train_df = load_variant_table(args.train_path, require_label=True)
        val_df = load_variant_table(args.val_path, require_label=True)
        val_eval_pos, n_drawn = None, 0
        if not args.no_eval and args.val_eval_samples and args.val_eval_samples < len(val_df):
            val_eval_pos, n_drawn = fixed_val_subset(val_df, args.val_eval_samples)
        n_dropped = {"train": int((~train_df["_valid"]).sum()),
                     "val": int((~val_df["_valid"]).sum())}
        train_df = train_df[train_df["_valid"]].reset_index(drop=True)
        val_df = val_df[val_df["_valid"]].reset_index(drop=True)
        for name, d in [("train", train_df), ("val", val_df)]:
            n_long = int((d["_seq"].str.len() > args.max_residues).sum())
            logger.info(f"{name:5s}: {len(d):,} variants used, {n_dropped[name]:,} skipped "
                        f"(invalid protein annotation or label); {n_long:,} windowed "
                        f"(protein > {args.max_residues} aa); "
                        f"{int(d['_label'].sum()):,} positive")

        train_ds = ProteinPairDataset(train_df, tokenizer, args.max_residues)
        val_ds = ProteinPairDataset(val_df, tokenizer, args.max_residues)
        pad_id = tokenizer.pad_token_id
        train_loader = make_loader(train_ds, batch_size, True, pad_id,
                                   args.seed, args.num_workers)
        if val_eval_pos is not None:
            val_eval_ds = Subset(val_ds, val_eval_pos)
            n_pos = int(val_df["_label"].iloc[val_eval_pos].sum())
            val_desc = (f"fixed label-stratified subset: {len(val_eval_pos):,} of "
                        f"{len(val_ds):,} samples ({n_pos:,} positive), seed {VAL_EVAL_SEED}, "
                        f"same every checkpoint; {n_drawn:,} rows drawn from the full val "
                        f"TSV, {n_drawn - len(val_eval_pos):,} of them invalid here")
        else:
            val_eval_ds = val_ds
            val_desc = f"full validation set, {len(val_ds):,} samples"
        n_val_eval = None if args.no_eval else len(val_eval_ds)
        val_eval_loader = make_loader(val_eval_ds, args.eval_batch_size, False, pad_id,
                                      args.seed, args.num_workers)
        if not args.no_eval:
            logger.info(f"Val eval       : {val_desc}")

        # ---- Model ----
        model = build_model(args.model_name, exp_config).to(device)
        use_multi_gpu = len(gpu_list) > 1
        if use_multi_gpu:
            model = nn.DataParallel(model, device_ids=gpu_list)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {n_trainable:,} / {n_total:,} "
                    f"({100 * n_trainable / n_total:.2f}%)")

        # ---- Optimizer / scheduler ----
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer = AdamW(
            [
                {"params": [p for n, p in model.named_parameters()
                            if not any(nd in n for nd in no_decay) and p.requires_grad],
                 "weight_decay": 0.01},
                {"params": [p for n, p in model.named_parameters()
                            if any(nd in n for nd in no_decay) and p.requires_grad],
                 "weight_decay": 0.0},
            ],
            lr=learning_rate, eps=1e-8, betas=(0.9, 0.999),
        )
        warmup_steps = int(0.06 * num_steps)
        scheduler = LRSchedulerWithWarmup(optimizer, warmup_steps, num_steps)
        criterion = nn.BCEWithLogitsLoss()

        def save_checkpoint(path, step, running_loss_sum, metrics=None):
            ckpt = {
                "step": step,
                "model_state_dict": trainable_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "running_loss_sum": running_loss_sum,
                "config": exp_config,
                "model_name": args.model_name,
                "max_residues": args.max_residues,
                "seed": args.seed,
                "val_eval_samples": args.val_eval_samples,
            }
            if metrics:
                ckpt.update(metrics)
            torch.save(ckpt, path)

        # ---- Resume ----
        resume_step, resume_running_loss_sum = 0, 0.0
        existing = sorted(checkpoints_dir.glob("step_*.pt"),
                          key=lambda p: int(p.stem.split("_")[1]))
        if existing:
            ckpt = torch.load(existing[-1], map_location="cpu", weights_only=False)
            load_trainable_state_dict(model, ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            resume_step = ckpt["step"]
            resume_running_loss_sum = ckpt.get("running_loss_sum", 0.0)
            if ckpt.get("val_eval_samples", 0) != args.val_eval_samples:
                logger.warning(f"Checkpoint was trained with --val_eval_samples "
                               f"{ckpt.get('val_eval_samples', 0)}, now "
                               f"{args.val_eval_samples}: earlier val metrics and the "
                               f"best-model choice used a different val set")
            logger.info(f"Resumed from {existing[-1].name} (step {resume_step}, "
                        f"lr={scheduler.get_last_lr()[0]:.2e})")
        else:
            logger.info("No existing checkpoints — training from scratch")

        # ---- Output files (truncate anything logged after the resume point) ----
        train_loss_csv = exp_dir / "training_loss.csv"
        if resume_step > 0 and train_loss_csv.exists():
            hist = pd.read_csv(train_loss_csv)
            hist[hist["step"] <= resume_step].to_csv(train_loss_csv, index=False)
        else:
            pd.DataFrame(columns=["step", "train_loss_batch", "train_loss_running_avg",
                                  "learning_rate"]).to_csv(train_loss_csv, index=False)

        metrics_csv = exp_dir / "training_metrics.csv"
        eval_metrics = {c: [] for c in METRIC_COLS}
        best_val_auroc, best_val_auprc, best_step = 0.0, 0.0, None
        if resume_step > 0 and metrics_csv.exists():
            hist = pd.read_csv(metrics_csv)
            hist = hist[hist["steps"] <= resume_step]
            hist.to_csv(metrics_csv, index=False)
            eval_metrics = {c: hist[c].tolist() for c in METRIC_COLS}
            if len(hist):
                best = hist.loc[hist["val_auroc"].idxmax()]
                best_val_auroc, best_val_auprc = best["val_auroc"], best["val_auprc"]
                best_step = int(best["steps"])
                logger.info(f"Restored eval history; best so far AUROC={best_val_auroc:.4f} "
                            f"at step {best_step}")
        elif not args.no_eval:
            pd.DataFrame(columns=METRIC_COLS).to_csv(metrics_csv, index=False)

        # ---- Training loop ----
        logger.info(f"Training from step {resume_step} to {stop_step} "
                    f"({warmup_steps} warmup steps)")
        global_step = resume_step
        running_loss_sum = resume_running_loss_sum
        accum_counter, accum_loss = 0, 0.0
        train_iter = iter(train_loader)
        optimizer.zero_grad()

        with open(train_loss_csv, "a") as loss_fh:
            while global_step < stop_step:
                model.train()
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                logits = forward_batch(model, batch, device)
                loss = criterion(logits, batch["labels"].to(device)) / grad_accum
                loss.backward()
                accum_loss += loss.item()
                accum_counter += 1
                if accum_counter < grad_accum:
                    continue

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                batch_loss, accum_counter, accum_loss = accum_loss, 0, 0.0
                current_lr = scheduler.get_last_lr()[0]
                running_loss_sum += batch_loss
                running_avg = running_loss_sum / global_step
                logger.info(f"Step {global_step}/{stop_step} | loss: {batch_loss:.4f} | "
                            f"running avg: {running_avg:.4f} | lr: {current_lr:.2e}")
                loss_fh.write(f"{global_step},{batch_loss},{running_avg},{current_lr}\n")
                loss_fh.flush()

                if global_step % args.eval_interval != 0 and global_step != stop_step:
                    continue

                ckpt_path = checkpoints_dir / f"step_{global_step}.pt"
                if args.no_eval:
                    save_checkpoint(ckpt_path, global_step, running_loss_sum)
                    logger.info(f"  -> Checkpoint saved: {ckpt_path.name}")
                    continue

                # Train metrics on a random subset; val metrics on the full val set
                # or the fixed --val_eval_samples subset
                rng = np.random.default_rng(args.seed + global_step)
                n_sub = min(args.train_eval_samples, len(train_ds))
                sub_idx = rng.choice(len(train_ds), n_sub, replace=False).tolist()
                sub_loader = make_loader(Subset(train_ds, sub_idx), args.eval_batch_size,
                                         False, pad_id, args.seed, args.num_workers)
                tr_loss, tr_auroc, tr_auprc = evaluate(model, sub_loader, device)
                val_loss, val_auroc, val_auprc = evaluate(model, val_eval_loader, device)
                mem = max(torch.cuda.max_memory_allocated(g) / 1024 ** 3 for g in gpu_list)

                row = [global_step, tr_loss, tr_auroc, tr_auprc,
                       val_loss, val_auroc, val_auprc, current_lr, mem]
                for c, v in zip(METRIC_COLS, row):
                    eval_metrics[c].append(v)
                pd.DataFrame([row], columns=METRIC_COLS).to_csv(
                    metrics_csv, mode="a", header=False, index=False)
                logger.info(f"[EVAL] Step {global_step} | Train (subset) AUROC: {tr_auroc:.4f}, "
                            f"AUPRC: {tr_auprc:.4f} | Val AUROC: {val_auroc:.4f}, "
                            f"AUPRC: {val_auprc:.4f} | peak GPU mem: {mem:.2f} GB")

                metrics = {"val_auroc": val_auroc, "val_auprc": val_auprc,
                           "train_auroc_subset": tr_auroc}
                save_checkpoint(ckpt_path, global_step, running_loss_sum, metrics)
                if val_auroc > best_val_auroc:
                    best_val_auroc, best_val_auprc, best_step = val_auroc, val_auprc, global_step
                    save_checkpoint(exp_dir / "best_model.pt", global_step,
                                    running_loss_sum, metrics)
                    logger.info(f"  -> New best: AUROC={val_auroc:.4f}, AUPRC={val_auprc:.4f}")

        # ---- Post-training ----
        if not args.no_eval and eval_metrics["steps"]:
            metrics_df = pd.DataFrame(eval_metrics)
            metrics_df.to_csv(exp_dir / "training_metrics_final.csv", index=False)
            generate_plots(metrics_df, exp_dir, exp_id, combine_mode,
                           "Val (fixed subset)" if val_eval_pos is not None else "Val (full)")

        total_time = time.time() - start_time
        summary = {
            "exp_id": exp_id,
            "combine_mode": combine_mode,
            "config": exp_config,
            "model_name": args.model_name,
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "val_eval_samples": n_val_eval,
            "n_skipped": n_dropped,
            "max_residues": args.max_residues,
            "seed": args.seed,
            "gpu_list": gpu_list,
            "effective_batch_size": batch_size * grad_accum,
            "scheduled_steps": num_steps,
            "total_steps": global_step,
            "best_step": best_step,
            "best_val_auroc": None if args.no_eval else best_val_auroc,
            "best_val_auprc": None if args.no_eval else best_val_auprc,
            "total_time_hours": total_time / 3600,
            "peak_gpu_memory_gb": max(torch.cuda.max_memory_allocated(g) / 1024 ** 3
                                      for g in gpu_list),
        }
        with open(exp_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Done: {total_time / 3600:.2f} h, best val AUROC={best_val_auroc:.4f} "
                    f"(step {best_step})")
        return summary

    except Exception as e:
        logger.error(f"Error in experiment {exp_id}: {e}")
        logger.error(traceback.format_exc())
        raise


# ============================================================================
# Prediction
# ============================================================================

def predict(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    exp_config = ckpt["config"]
    model_name = ckpt.get("model_name", args.model_name)
    max_residues = ckpt.get("max_residues", args.max_residues)

    device = torch.device(f"cuda:{args.gpus[0]}")
    torch.cuda.set_device(args.gpus[0])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = build_model(model_name, exp_config)
    load_trainable_state_dict(model, ckpt["model_state_dict"])
    model = model.to(device)
    if len(args.gpus) > 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    df = load_variant_table(args.predict_path, require_label=False)
    valid = df[df["_valid"]].reset_index(drop=True)
    print(f"Scoring {len(valid):,} / {len(df):,} variants "
          f"({(~df['_valid']).sum():,} skipped: invalid protein annotation)")
    loader = make_loader(ProteinPairDataset(valid, tokenizer, max_residues),
                         args.eval_batch_size, False, tokenizer.pad_token_id,
                         args.seed, args.num_workers)
    logits, labels = run_inference(model, loader, device)
    probs = 1.0 / (1.0 + np.exp(-logits))

    out = df[["variant_id"]].copy()
    if "label" in df.columns:
        out["label"] = df["label"]
    out["score"] = np.nan
    out.loc[df["_valid"].values, "score"] = probs
    out["status"] = np.where(df["_valid"], "scored", "skipped_invalid_protein_annotation")

    out_path = Path(args.predict_out) if args.predict_out else \
        Path(args.checkpoint).parent / f"predictions_{Path(args.predict_path).stem}.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    print(f"Predictions written to {out_path}")

    has_label = ~np.isnan(labels)
    if has_label.any() and len(np.unique(labels[has_label])) == 2:
        print(f"AUROC: {roc_auc_score(labels[has_label], probs[has_label]):.4f} | "
              f"AUPRC: {average_precision_score(labels[has_label], probs[has_label]):.4f} "
              f"(n={has_label.sum():,})")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ESM-1b Ref-Alt Contrast — Siamese LoRA Fine-Tuning with MLP Head")
    parser.add_argument("--train_path", type=str, help="Training TSV")
    parser.add_argument("--val_path", type=str, help="Validation TSV")
    parser.add_argument("--output_dir", type=str, default="./results/ESM1b_ref_alt_contrast")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="GPU IDs; >1 uses nn.DataParallel (default: 0 1 2 3)")
    parser.add_argument("--combine_mode", type=str, default=None,
                        choices=sorted(HEAD_INPUT_MULT),
                        help="Override combine_mode in EXPERIMENT_CONFIGS")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override total per-step batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Override gradient accumulation steps")
    parser.add_argument("--stop_at_step", type=int, default=None,
                        help="Stop training at this step (LR schedule still spans num_steps)")
    parser.add_argument("--max_residues", type=int, default=1022,
                        help="Max residues per window (ESM-1b limit: 1022)")
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--train_eval_samples", type=int, default=10000)
    parser.add_argument("--val_eval_samples", type=int, default=0,
                        help="Evaluate val on a FIXED label-stratified random subset "
                             "of this many samples, drawn once and reused at every "
                             "checkpoint (default: 0 = full val set). best_model.pt "
                             "is then selected on this subset. Drawn exactly like the "
                             "NT-2 script, so a row-aligned val TSV gives the same variants.")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Batch size for eval/prediction; reduce on a single GPU if OOM")
    parser.add_argument("--no_eval", action="store_true",
                        help="Skip inline eval; checkpoints still saved every eval_interval")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict_path", type=str, default=None,
                        help="Score this TSV with --checkpoint instead of training")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--predict_out", type=str, default=None,
                        help="Output TSV (default: predictions_<input>.tsv next to checkpoint)")
    args = parser.parse_args()

    if args.predict_path:
        if not args.checkpoint:
            parser.error("--predict_path requires --checkpoint")
        predict(args)
        return

    if not (args.train_path and args.val_path):
        parser.error("--train_path and --val_path are required for training")
    if args.batch_size and args.batch_size % len(args.gpus) != 0:
        print(f"WARNING: batch_size={args.batch_size} not divisible by {len(args.gpus)} GPUs")

    summaries = []
    for cfg in EXPERIMENT_CONFIGS:
        cfg = dict(cfg)
        if args.combine_mode:
            cfg["combine_mode"] = args.combine_mode
            cfg["description"] = f"ESM-1b ref-alt contrast Siamese MLP, combine={args.combine_mode}"
        summaries.append(train_experiment(cfg, args))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out_dir / "all_experiments_summary.csv", index=False)
    for s in summaries:
        print(f"{s['combine_mode']:12s} | best val AUROC: {s['best_val_auroc']} | "
              f"AUPRC: {s['best_val_auprc']} | step {s['best_step']}")


if __name__ == "__main__":
    main()
