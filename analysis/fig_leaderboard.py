#!/usr/bin/env python3
"""
fig_leaderboard.py — absolute performance.

  figS8   ClinVar test set. AUROC, AUPRC and log-loss reduction vs the training prevalence of MetaMissense and of
          each score alone (base, GLM-Missense, COMPARE: GERP++, phyloP 100/470); bars, solid = not different
          from the top score (p >= 0.05)
  figS9   what each score (GLM-Missense, COMPARE) adds to base + Random (+X | base + Random; drop-column style)

Log-loss of a single score: quantile-normal transform + logistic regression fit on the training split (core.calibrated);
reduction vs predicting the training prevalence for every variant.

    python analysis/fig_leaderboard.py --ensemble bergquist --n_jobs 40
"""
import json

import numpy as np
import pandas as pd

import core as C
from core import COLOR, COMPARE, FS, GLM, MM, NAME, RND, plt


def bars(ax, t, base=0.0, labels=True):
    """Absolute performance: bars from base with 95% CI; solid = not different from the top method."""
    yy = -np.arange(len(t), dtype=float)
    for yi, r in zip(yy, t.itertuples()):
        ax.barh(yi, r.estimate - base, left=base, color=COLOR[r.method], height=0.7,
                alpha=1.0 if r.same_tier_as_top else 0.35)
        ax.errorbar(r.estimate, yi, xerr=C.xerr(r.estimate, r.ci_lo, r.ci_hi), color='black', lw=1, capsize=2)
    ax.axvline(base, color='black', lw=0.8)
    lo, hi = ax.get_xlim()
    for yi, r in zip(yy, t.itertuples()):
        ax.text(max(r.ci_hi, r.estimate, base) + 0.02 * (hi - lo), yi, f'{r.estimate:.3f}', va='center',
                fontsize=FS['note'])
    ax.set_xlim(lo, hi + 0.3 * (hi - lo))
    ax.set_yticks(yy)
    ax.set_yticklabels([NAME[m] for m in t.method] if labels else [], fontsize=FS['tick'])
    C.color_labels(ax.get_yticklabels(), [COLOR[m] for m in t.method])
    ax.set_ylim(yy.min() - 0.6, 0.6)
    ax.tick_params(axis='x', labelsize=FS['tick'])


def leaderboard():
    D = C.DALL
    tr, Sr, te, Se = D
    scores, adds = C.BASE + [GLM] + COMPARE, [GLM] + COMPARE
    y, ytr = te.label.to_numpy(), tr.label.to_numpy()
    meth = [MM] + scores
    prob = {MM: C.mm_prob(Se), **{m: C.calibrated(m, D) for m in scores}}
    p0 = np.clip(ytr.mean(), 1e-7, 1 - 1e-7)
    X = np.column_stack([prob[MM]] + [Se[m].to_numpy(float) for m in scores])

    def single(i):
        yy, out = y[i], []
        ll0 = -np.mean(yy * np.log(p0) + (1 - yy) * np.log(1 - p0))
        for j, m in enumerate(meth):
            pc = np.clip(prob[m][i], 1e-7, 1 - 1e-7)
            out.append([C.auroc(yy, X[i, j]), C.auprc(yy, X[i, j]),
                        ll0 + np.mean(yy * np.log(pc) + (1 - yy) * np.log(1 - pc))])
        return np.array(out)

    est, Bt = single(np.arange(len(y))), np.array([single(i) for i in C.boot_idx(y)])
    K3 = ['AUROC', 'AUPRC', 'logloss_reduction']
    tabs = {k: C.tier_table(meth, est[:, j], Bt[:, :, j]) for j, k in enumerate(K3)}
    order = tabs['AUROC'].method.tolist()
    base = C.BASE + [RND]
    P = {'base': C.ensemble(base, D, 'clinvar')}                          # same models as the drop-column figure
    P.update({x: C.ensemble(base + [x], D, 'clinvar') for x in adds})
    tB = C.compare([(f'+{NAME[x]}', x, 'base', COLOR[x]) for x in adds], P, y)
    h_bar, y_bar = 4.0 * len(meth) / 11, 1.2
    fig = plt.figure(figsize=(10, y_bar + h_bar + 0.3))
    zero = {'AUROC': 0.5, 'AUPRC': y.mean(), 'logloss_reduction': 0.0}
    xl = {'AUROC': 'AUROC', 'AUPRC': 'AUPRC', 'logloss_reduction': 'Log-loss reduction\n(vs. prevalence)'}
    for j, k in enumerate(K3):
        ax = C.axes_in(fig, 2.35 + j * 2.55, y_bar, 2.3, h_bar)
        bars(ax, tabs[k].set_index('method').loc[order].reset_index(), zero[k], labels=(j == 0))
        ax.set_xlabel(xl[k], fontsize=FS['label'])
        if k == 'AUROC':
            ax.set_xlim(0.5, None)
    single_t = pd.concat([t.assign(metric=k) for k, t in tabs.items()]).assign(n=len(y), n_pos=int(y.sum()))
    mm = json.loads((C.DIRS['models'] / 'MetaMissense.json').read_text())
    C.finish(fig, 'figS8', {'single': single_t}, D, metamissense=mm)
    h_def = 0.4 * len(tB)
    fig = plt.figure(figsize=(10, h_def + 1.6))
    C.forest3(fig, tB, 2.4, 0.6, h_def, '', w=2.25, gap=0.28, heading='Score added to\nbase predictors')
    C.finish(fig, 'figS9', {'added_to_base_random': tB}, D, metamissense=mm)


if __name__ == '__main__':
    args = C.parser(__doc__).parse_args()
    C.setup(args)
    C.log('── figS8, figS9')
    leaderboard()
