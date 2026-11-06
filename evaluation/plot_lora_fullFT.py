import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Editable text in Adobe Illustrator (embed as TrueType, not Type 3)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# Use Arial font
mpl.rcParams['font.family'] = 'Arial'

parser = argparse.ArgumentParser(description='Plot training curves for full FT and LoRA experiments')
parser.add_argument('full_ft_csv', help='CSV file with full fine-tuning experiment metrics')
parser.add_argument('lora_csv', help='CSV file with LoRA experiment metrics')
parser.add_argument('-o', '--output', default='training_curves_comparison.pdf',
                    help='Output figure path (default: training_curves_comparison.pdf)')
args = parser.parse_args()

# Load data
df = pd.read_csv(args.full_ft_csv)
lora_df = pd.read_csv(args.lora_csv)

# Build experiment list: (label, steps, train_auroc, train_auprc, val_auroc, val_auprc)
experiments = []

# Full fine-tuning experiments
for eid in sorted(df['exp_id'].unique()):
    sub = df[df['exp_id'] == eid].sort_values('steps')
    lr = sub['learning_rate'].iloc[0]
    experiments.append((
        f'Full FT (lr={lr:.0e})',
        sub['steps'].values,
        sub['train_auroc'].values,
        sub['train_auprc'].values,
        sub['val_auroc'].values,
        sub['val_auprc'].values,
    ))

# LoRA experiment
lora_df = lora_df.sort_values('steps')
lora_lr = lora_df['learning_rate'].iloc[0]
experiments.append((
    f'LoRA (lr=5e-05)',
    lora_df['steps'].values,
    lora_df['train_auroc'].values,
    lora_df['train_auprc'].values,
    lora_df['val_auroc'].values,
    lora_df['val_auprc'].values,
))

# Color palette
colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
lora_idx = len(experiments) - 1

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# (row, col, metric_index_in_tuple, title)
panel_configs = [
    (0, 0, 2, 'Training AUROC',   'A'),
    (0, 1, 4, 'Validation AUROC', 'B'),
    (1, 0, 3, 'Training AUPRC',   'C'),
    (1, 1, 5, 'Validation AUPRC', 'D'),
]

for row, col, midx, title, panel_label in panel_configs:
    ax = axes[row, col]
    for i, exp in enumerate(experiments):
        label, steps = exp[0], exp[1]
        y = exp[midx]
        if i == lora_idx:
            ax.plot(steps, y, label=label, color='black', linewidth=2.5,
                    linestyle='--', marker='o', markersize=5, zorder=10)
        else:
            ax.plot(steps, y, label=label, color=colors[i], linewidth=1.5,
                    marker='s', markersize=3, alpha=0.85)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(title.split()[-1])
    # Remove top and right spines (no box), no grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.08, 1.05, panel_label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='bottom', ha='right')
    if row == 1:
        ax.set_xlabel('Training Steps')

# Match y-axis ranges: AUROC row (top) and AUPRC row (bottom)
for row_idx in range(2):
    ymin = min(axes[row_idx, 0].get_ylim()[0], axes[row_idx, 1].get_ylim()[0])
    ymax = max(axes[row_idx, 0].get_ylim()[1], axes[row_idx, 1].get_ylim()[1])
    axes[row_idx, 0].set_ylim(ymin, ymax)
    axes[row_idx, 1].set_ylim(ymin, ymax)

# Single legend below the figure
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=14,
           bbox_to_anchor=(0.5, -0.02), frameon=True)


plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig(args.output, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved figure to {args.output}')
