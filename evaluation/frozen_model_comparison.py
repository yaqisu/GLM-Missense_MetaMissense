"""
evaluation/frozen_model_comparison.py

Classifier Head and Embedding Strategy Comparison
         (Frozen backbone: NT-2, NT-1, Caduceus)

Reads:  results/frozen_results.tsv
        — only frozen-backbone rows (Fine-tuning == "frozen" or
          "frozen-input_leng=6k")

Writes: results/figures/frozen_model_comparison.pdf  (editable in Illustrator)
        results/figures/frozen_model_comparison.png

Font: Arial via runtime addfont — no conda env write access required.
     pdf.fonttype=42 ensures text is editable in Adobe Illustrator.

Run from project root:
    python evaluation/frozen_model_comparison.py
"""

from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Arial font registration (no conda write access needed) ─────────────────
_USER_FONT_DIR = "/n/electric/data/jenniferlin/fonts"
if os.path.isdir(_USER_FONT_DIR):
    for _f in os.listdir(_USER_FONT_DIR):
        if _f.lower().endswith(".ttf") or _f.lower().endswith(".otf"):
            _fm.fontManager.addfont(os.path.join(_USER_FONT_DIR, _f))

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype":    42,      # TrueType — editable text in Illustrator
    "svg.fonttype":    "none",
    "ps.fonttype":     42,
})

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA     = ROOT / "results" / "frozen_results.tsv"
OUT_DIR  = ROOT / "results" / "figures"
OUT_BASE = OUT_DIR / "frozen_model_comparison"

# ── Marker styles ──────────────────────────────────────────────────────────
MARKERS = {
    "NT-2":      "o",   # circle
    "NT-1":      "^",   # triangle
    "Caduceus":  "x",   # x
    "NT-2-6k":   "s",   # square for 6k input length
}


# ── Helper functions ───────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    # Standardize classifier layers column (handles both name variants)
    def get_classifier_layers(row):
        if pd.notna(row.get("Classifier head -  hidden layers")):
            return row["Classifier head -  hidden layers"]
        elif pd.notna(row.get("Classifier head - hidden layers")):
            return row["Classifier head - hidden layers"]
        return None

    df["Classifier_layers"] = df.apply(get_classifier_layers, axis=1)

    # Standardize embedding strategy
    def standardize_embedding(embed_str):
        if pd.isna(embed_str):
            return "unknown"
        embed_lower = str(embed_str).lower().strip()
        if embed_lower.startswith("full-"):
            embed_lower = embed_lower.replace("full-", "", 1)
        if "downsample" in embed_lower:
            if "mean" in embed_lower:
                return "mean_pool"
            elif "variant" in embed_lower:
                return "variant_position"
            return "downsample"
        elif "variant" in embed_lower:
            return "variant_position"
        elif "mean" in embed_lower:
            return "mean_pool"
        return embed_str

    df["Embedding_std"] = df["Embedding strategy"].apply(standardize_embedding)
    return df


def extract_curves(row, prefix="Val_Step", max_steps=14000):
    steps, aucs = [], []
    for step in range(1000, max_steps + 1, 1000):
        col = f"{prefix}{step}_AUC"
        if col in row.index and pd.notna(row[col]):
            try:
                aucs.append(float(row[col]))
                steps.append(step)
            except (ValueError, TypeError):
                pass
    return np.array(steps), np.array(aucs)


def filter_frozen(df, model_name):
    """Keep only frozen-backbone rows for a given model."""
    mask = (df["Model"] == model_name) & (
        df["Fine-tuning"].str.contains("frozen", case=False, na=False)
    )
    return df[mask].copy()


def organize_by_embedding(df_subset, model_type):
    buckets = {
        "variant_position": [],
        "mean_pool":        [],
        "downsample_mean":  [],
        "downsample_variant": [],
    }
    for idx, row in df_subset.iterrows():
        embed = row["Embedding_std"]
        if embed in buckets:
            buckets[embed].append((
                idx,
                row["Classifier head"],
                row["Classifier_layers"],
                model_type,
                row.get("Input length", 12000),
            ))
    return buckets


def get_color(head, layers, embed_type):
    head_lower = str(head).lower() if pd.notna(head) else "unknown"
    if "downsample" in embed_type:
        return "purple" if head_lower == "cnn" else (
            "darkviolet" if head_lower == "transformer" else "mediumorchid")
    if head_lower == "mlp":
        return ("skyblue" if embed_type == "variant_position" else "lightcoral") \
            if str(layers) == "2" else \
            ("C0"     if embed_type == "variant_position" else "C1")
    if head_lower == "transformer":
        return "C2" if embed_type == "variant_position" else "C3"
    if head_lower == "cnn":
        return "C4" if embed_type == "variant_position" else "C5"
    return "C6"


# ── Main plotting function ─────────────────────────────────────────────────

def plot_figure2(df: pd.DataFrame, out_base: Path) -> None:
    frozen_nt2 = filter_frozen(df, "NT-2")
    frozen_nt1 = filter_frozen(df, "NT-1")
    frozen_cad = filter_frozen(df, "Caduceus")

    print(f"  Frozen NT-2: {len(frozen_nt2)} rows")
    print(f"  Frozen NT-1: {len(frozen_nt1)} rows")
    print(f"  Frozen Caduceus: {len(frozen_cad)} rows")

    nt2_models = organize_by_embedding(frozen_nt2, "NT-2")
    nt1_models = organize_by_embedding(frozen_nt1, "NT-1")
    cad_models = organize_by_embedding(frozen_cad, "Caduceus")

    fig = plt.figure(figsize=(18, 8))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.35], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    plotted_models = []

    def plot_models(models_list, embed_type, linestyle="-", alpha=1.0):
        for idx, head, layers, model, input_length in models_list:
            row = df.loc[idx]
            steps_train, aucs_train = extract_curves(row, "Training_Step", 5000)
            steps_val,   aucs_val   = extract_curves(row, "Val_Step",      5000)
            color = get_color(head, layers, embed_type)

            if model == "NT-1":
                marker = MARKERS["NT-1"]
                input_label = ""
            elif model == "NT-2" and input_length == 6000:
                marker = MARKERS["NT-2-6k"]
                input_label = " (6k)"
            else:
                marker = MARKERS.get(model, "o")
                input_label = ""

            head_str  = str(head).upper()  if pd.notna(head)   else "UNK"
            layer_str = str(int(layers))   if pd.notna(layers) else "?"
            embed_label = {
                "variant_position":  "Var",
                "mean_pool":         "Mean",
                "downsample_mean":   "Down+Mean",
                "downsample_variant":"Down+Var",
            }.get(embed_type, embed_type[:4].capitalize())

            model_label = "Cad" if model == "Caduceus" else model
            label = f"{model_label}: {head_str}-{layer_str}L + {embed_label}{input_label}"
            markersize = 7 if model != "Caduceus" else 6

            if len(steps_train) > 0:
                ax1.plot(steps_train, aucs_train,
                         marker=marker, color=color,
                         linewidth=2.5, linestyle=linestyle,
                         markersize=markersize, alpha=alpha)
            if len(steps_val) > 0:
                line, = ax2.plot(steps_val, aucs_val,
                                 marker=marker, color=color,
                                 linewidth=2.5, linestyle=linestyle,
                                 markersize=markersize, alpha=alpha)
                val_auc = aucs_val[-1] if len(aucs_val) > 0 else 0
                plotted_models.append((val_auc, label, line))

    # Variant position — solid lines
    for models_dict, alpha_val in [(nt2_models, 1.0), (nt1_models, 1.0), (cad_models, 0.8)]:
        if models_dict["variant_position"]:
            plot_models(models_dict["variant_position"], "variant_position",
                        linestyle="-", alpha=alpha_val)

    # Mean pool — dashed lines
    for models_dict, alpha_val in [(nt2_models, 1.0), (cad_models, 0.8)]:
        if models_dict["mean_pool"]:
            plot_models(models_dict["mean_pool"], "mean_pool",
                        linestyle="--", alpha=alpha_val)

    # Downsample — dotted lines
    for embed_key in ("downsample_mean", "downsample_variant"):
        if cad_models[embed_key]:
            plot_models(cad_models[embed_key], embed_key,
                        linestyle=":", alpha=0.8)

    # Sort legend by final val AUC descending
    plotted_models.sort(key=lambda x: x[0], reverse=True)

    # ── Axes config ────────────────────────────────────────────────────
    for ax, ylabel, title in [
        (ax1, "Training AUC",   "Training Performance"),
        (ax2, "Validation AUC", "Validation Performance"),
    ]:
        ax.set_xlabel("Training Step", fontsize=20, fontweight="bold")
        ax.set_ylabel(ylabel,          fontsize=20, fontweight="bold")
        ax.set_title(title,            fontsize=20, fontweight="bold")
        ax.set_ylim([0.4, 1.0])
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)

    # ── Tier annotations on validation panel ───────────────────────────
    annotations = [
        (5300, 0.875, "NT-2: Var",  "lightblue",   "navy"),
        (5300, 0.78,  "NT-2: Mean", "lightcoral",   "darkred"),
        (5300, 0.735, "NT-1: Var",  "gray",         "black"),
        (5300, 0.6,   "Cad: Var",   "lightblue",    "navy"),
        (5300, 0.45,  "Cad: Mean",  "lightcoral",   "darkred"),
    ]
    for x, y, txt, fc, ec in annotations:
        ax2.text(x, y, txt, fontsize=16,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=fc,
                           alpha=0.7, edgecolor=ec),
                 verticalalignment="center")

    # ── Legends ────────────────────────────────────────────────────────
    marker_elements = [
        Line2D([0], [0], marker="o", color="black", linestyle="None",
               markersize=8, label="NT-2 (12k)"),
        Line2D([0], [0], marker="s", color="black", linestyle="None",
               markersize=8, label="NT-2 (6k)"),
        Line2D([0], [0], marker="^", color="black", linestyle="None",
               markersize=8, label="NT-1 (6k)"),
        Line2D([0], [0], marker="x", color="black", linestyle="None",
               markersize=6, label="Caduceus (30k)"),
    ]
    linestyle_elements = [
        Line2D([0], [0], color="black", linestyle="-",  linewidth=2, label="Var pos"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Mean pool"),
    ]

    fig.legend(handles=marker_elements,
               loc="upper left", bbox_to_anchor=(0.82, 0.90),
               fontsize=16, title="Model", title_fontsize=16,
               frameon=True, fancybox=True, shadow=True)
    fig.legend(handles=linestyle_elements,
               loc="upper left", bbox_to_anchor=(0.97, 0.90),
               fontsize=16, title="Embedding Strategy", title_fontsize=16,
               frameon=True, fancybox=True, shadow=True)
    fig.legend(
        [x[2] for x in plotted_models],
        [f"{x[1]} ({x[0]:.3f})" for x in plotted_models],
        loc="center left", bbox_to_anchor=(0.82, 0.32),
        fontsize=16, framealpha=0.9,
    )

    plt.suptitle(
        "Frozen Model Comparison: NT-2, NT-1, and Caduceus",
        fontsize=24, fontweight="bold", y=0.98,
    )

    # ── Save ───────────────────────────────────────────────────────────
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        out = out_base.with_suffix(ext)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.close(fig)

    print(f"  ✓ Plotted {len(plotted_models)} models total")
    print(f"  ✓ NT-2: {len(nt2_models['variant_position']) + len(nt2_models['mean_pool'])} models")
    print(f"  ✓ NT-1: {len(nt1_models['variant_position'])} models")
    print(f"  ✓ Caduceus: {sum(len(v) for v in cad_models.values())} models")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Figure 2: Frozen Model Comparison (NT-2, NT-1, Caduceus)")
    print("=" * 70)
    print(f"  Reading: {DATA}")

    if not DATA.exists():
        print(f"ERROR: {DATA} not found.")
        print("  Create results/frozen_results.tsv by filtering results.tsv to")
        print("  rows where Fine-tuning contains 'frozen'.")
        sys.exit(1)

    df = load_data(DATA)
    print(f"  Loaded {len(df)} rows")
    print(f"  Fine-tuning values: {df['Fine-tuning'].value_counts().to_dict()}")

    plot_figure2(df, OUT_BASE)


if __name__ == "__main__":
    main()