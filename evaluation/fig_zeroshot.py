#!/usr/bin/env python3
"""
fig_zeroshot.py — Fig S7: GLM-Missense vs zero-shot NT-2 on the later-release held-out test set.
ROC (A) and PR (B) curves; AUROC / AUPRC in the legend.

Inputs (defaults; repo root):
  --split     data/splits/ClinVar.260923only.missense.hg38.seq12k.BLBvsPLP_validation.tsv   (variant_id, label)
  --glm       results/predictions/ClinVar.260923only.BLBvsPLP/GLM-Missense.tsv               (scoring/GLM-Missense.py)
  --zeroshot  results/predictions/ClinVar.260923only.BLBvsPLP/zeroshot_NT2_seq12k.tsv        (evaluation/zeroshot_nt.py)
Variants with both scores are used (n = 5,780 in the manuscript).

Output: results/evaluation/figures/figS7.pdf (editable text) + .png, figS7.tsv (n, AUROC, AUPRC)

    python evaluation/fig_zeroshot.py
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for d in [os.environ.get('ARIAL_DIR', ''), str(ROOT.parent / 'fonts')]:      # Arial, if not installed system-wide
    if d and os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith(('.ttf', '.otf')):
                fm.fontManager.addfont(os.path.join(d, f))
matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                            'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

P = 'results/predictions/ClinVar.260923only.BLBvsPLP'
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--split', default='data/splits/ClinVar.260923only.missense.hg38.seq12k.BLBvsPLP_validation.tsv')
ap.add_argument('--glm', default=f'{P}/GLM-Missense.tsv')
ap.add_argument('--zeroshot', default=f'{P}/zeroshot_NT2_seq12k.tsv')
ap.add_argument('--outdir', default='results/evaluation/figures')
a = ap.parse_args()

# (score file, score column, legend name, colour, line width)
METHODS = [(a.glm, 'GLM-Missense_score', 'GLM-Missense', '#C2185B', 2.5),
           (a.zeroshot, 'pathogenicity_score', 'NT2-Zeroshot', '#000000', 1.8)]

df = pd.read_csv(a.split, sep='\t', usecols=['variant_id', 'label'], dtype={'variant_id': str})
for path, col, name, _, _ in METHODS:
    s = pd.read_csv(path, sep='\t', usecols=['variant_id', col], dtype={'variant_id': str}).rename(columns={col: name})
    df = df.merge(s.drop_duplicates('variant_id'), on='variant_id', how='left')
names = [m[2] for m in METHODS]
df = df.dropna(subset=names)
y = df.label.astype(int)
print(f'{len(df):,} test variants with both scores ({y.sum():,} pathogenic)')

fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(8, 4))
rows = []
for _, _, name, color, lw in METHODS:
    s = df[name].astype(float)
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    auroc, auprc = roc_auc_score(y, s), average_precision_score(y, s)
    z = 5 if name == 'GLM-Missense' else 2
    ax_roc.plot(fpr, tpr, color=color, lw=lw, alpha=0.9, zorder=z, label=f'{name} ({auroc:.3f})')
    ax_pr.plot(rec, prec, color=color, lw=lw, alpha=0.9, zorder=z, label=f'{name} ({auprc:.3f})')
    rows.append({'method': name, 'n': len(df), 'n_pathogenic': int(y.sum()), 'AUROC': auroc, 'AUPRC': auprc})

ax_roc.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
ax_pr.set(xlabel='Recall', ylabel='Precision')
ax_roc.legend(fontsize=10, loc='lower right')
ax_pr.legend(fontsize=10, loc='lower left')
for ax, title in [(ax_roc, 'ROC Curves'), (ax_pr, 'PR Curves')]:
    ax.set_title(f'{title}\nn={len(df):,}', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
for ax, letter in [(ax_roc, 'A'), (ax_pr, 'B')]:
    ax.text(-0.18, 1.12, letter, transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')
fig.tight_layout()

out = Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(out / f'figS7.{ext}', dpi=300, bbox_inches='tight')
pd.DataFrame(rows).to_csv(out / 'figS7.tsv', sep='\t', index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(f'Saved {out}/figS7.pdf, .png, .tsv')
