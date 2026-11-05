#!/usr/bin/env python3
"""
fig_splicing.py — what is the new information? Variants the model resolves, and what the model adds to SpliceAI.

  fig6    RAD51D SGE. A RNA score of GLM-Missense-resolved vs background variants; B what GLM-Missense adds to
          base (ESM1b-FT) + SpliceAI (+GLM-Missense | base + SpliceAI): ΔAUROC, ΔAUPRC, log-loss reduction, bars with
          paired-bootstrap 95% CI, stars for the two-sided p. Base (ESM1b-FT) = the base with ESM1b-FT in place of ESM1b
  figS13  ClinVar test set. SpliceAI Δ score (DS max) of A GLM-Missense-resolved and B ESM1b-FT-resolved vs background

Figures show stars only (ns, *, **, ***); p values, n and the AUROC change are in the tables for the legend.

resolved  correct by the focal model (threshold 0.5) while most of its reference predictors are not correct: at
          least 2/3 of them, rounded up (core.k_of: 4 of 6, 5 of 7, 7 of 10; an uncertain calibrated call is not
          correct). Reference: ClinVar, every predictor of the base; SGE, base (ESM1b-FT) without GLM-Missense (= the
          whole base (ESM1b-FT)). Calls: core.call (bergquist version: Bergquist et al. 2025 Table 1 Supporting
          intervals for the calibrated tools; otherwise one threshold per score). Background = all other variants of
          the same class. Mann–Whitney U, two-sided, unadjusted p (one pre-specified comparison per class).

Variant lists for manual review (variant_lists/<ensemble>/<fig>_resolved.tsv): every resolved variant of the box panels,
with the reference predictors that are wrong, every score and call, and every column of the source table.

    python analysis/fig_splicing.py --ensemble bergquist --figs fig6 figS13 --n_jobs 40
"""

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

import core as C
from core import COLOR, FS, FT, GLM, NAME, SAI, SHORT, plt

GREY = '#9A9A9A'
# one physical bar / box width for fig6 and figS13: boxes 0.32 x-units wide on a BOX_W-inch panel spanning 2 units
BOX_W = 3.2                                                              # box panel width, inches
BAR_IN = 0.32 / 2 * BOX_W                                                # = 0.512 in
GAIN_W, GAIN_SPAN = 3.0, 3.0                                             # fig6 B: width (in), x range (-0.5, 2.5)
SGE_CLASSES = ['Functionally\nnormal', 'Functionally\nabnormal']
CLINVAR_CLASSES = ['Benign / LB', 'Pathogenic / LP']


def correct(d, S, x):
    """Correct call of x (core.call; no call = not correct)."""
    return C.call(S, x) == d.label.to_numpy()


def resolved(d, S, m, ref):
    """Correct by m while at least k_of(len(ref)) of the reference predictors are not correct."""
    return correct(d, S, m) & (sum(~correct(d, S, b) for b in ref) >= C.k_of(len(ref)))


def resolved_list(name, D, value, ref, classes, models):
    """Every variant resolved by each of models -> variant_lists/<ensemble>/<name>_resolved.tsv."""
    te, Se = D[2], D[3]
    rows = []
    for m in models:
        r = ref(m)
        g = resolved(te, Se, m, r)
        wrong = np.column_stack([~correct(te, Se, b) for b in r])
        for i in np.flatnonzero(g):
            rows.append({'variant_id': te.variant_id.iat[i], 'resolved_by': NAME[m], 'class': classes[te.label.iat[i]].replace('\n', ' '),
                         'label': int(te.label.iat[i]), 'focal_score': Se[m].iat[i], value: Se[value].iat[i],
                         'n_reference_wrong': int(wrong[i].sum()), 'n_reference': len(r), 'min_reference_wrong': C.k_of(len(r)),
                         'reference_wrong': ', '.join(NAME[b] for b, w in zip(r, wrong[i]) if w),
                         'reference_correct': ', '.join(NAME[b] for b, w in zip(r, wrong[i]) if not w)})
    sel = pd.DataFrame(rows, columns=['variant_id', 'resolved_by', 'class', 'label', 'focal_score', value,
                                      'n_reference_wrong', 'n_reference', 'min_reference_wrong', 'reference_wrong',
                                      'reference_correct'])
    C.variant_list(f'{name}_resolved', sel, te, Se)


def box_panel(ax, d, groups, value, ylabel, classes):
    """groups: [(name, mask, colour)] x 2, within each class (0, 1); Mann–Whitney U, two-sided. Returns the table and
    the positions of the significance labels [(ax, x, y, class)]."""
    y = d.label.to_numpy()
    rows, data, pos, cols = [], [], [], []
    for s, lab in enumerate(classes):
        vs = [value[mask & (y == s) & ~np.isnan(value)] for _, mask, _ in groups]
        p = mannwhitneyu(*vs, alternative='two-sided')[1] if all(len(v) for v in vs) else np.nan
        for g, ((gname, _, c), v) in enumerate(zip(groups, vs)):
            data.append(v if len(v) else np.array([np.nan]))
            pos.append(s + (g - 0.5) * 0.44)
            cols.append(c)
            rows.append({'class': lab.replace('\n', ' '), 'group': gname, 'n': len(v), 'median': np.median(v) if len(v) else np.nan,
                         'mean': np.mean(v) if len(v) else np.nan, 'mannwhitney_p': p})
    bp = ax.boxplot(data, positions=pos, widths=0.32, patch_artist=True, showfliers=False, showmeans=True,
                    meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 5})
    for patch, c in zip(bp['boxes'], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for med in bp['medians']:
        med.set_color('black')
    wy = np.concatenate([w.get_ydata() for w in bp['whiskers']])
    lo, hi = np.nanmin(wy), np.nanmax(wy)
    span = hi - lo if hi > lo else 1.0
    marks = [(ax, s, hi + 0.06 * span, lab.replace('\n', ' ')) for s, lab in enumerate(classes)]
    for x_, r in zip(pos, rows):
        ax.text(x_, -0.05, f'n={r["n"]:,}', ha='center', va='top', fontsize=FS['note'], transform=ax.get_xaxis_transform())
    ax.set_xlim(-0.5, len(classes) - 0.5)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=FS['tick'])
    ax.tick_params(axis='x', pad=16)
    ax.tick_params(axis='y', labelsize=FS['tick'])
    ax.set_ylabel(ylabel, fontsize=FS['label'])
    ax.set_ylim(lo - 0.05 * span, hi + 0.3 * span)
    ax.legend([Patch(color=c, alpha=0.7) for _, _, c in groups], [g for g, _, _ in groups], fontsize=FS['note'],
              frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2, columnspacing=1.2)
    return pd.DataFrame(rows).assign(value=ylabel), marks


def resolved_boxes(fig, D, value, ylabel, y0, ref, classes, models, x0s, w=BOX_W):
    """One box panel per model (letters A, B, ...): model-resolved vs background (same y range); stars above each
    class (p in the table)."""
    te, Se = D[2], D[3]
    v = Se[value].to_numpy(float)
    axes, tabs, marks = [], {}, []
    for k, (model, L, x0) in enumerate(zip(models, 'AB', x0s)):
        g = resolved(te, Se, model, ref(model))
        ax = C.axes_in(fig, x0, y0, w, 1.9)
        key = f'resolved_{SHORT[model]}_{value}'
        tabs[key], mk = box_panel(ax, te, [(f'{NAME[model]}-resolved', g, COLOR[model]), ('Background', ~g, GREY)],
                                  v, ylabel, classes)
        tabs[key] = tabs[key].assign(reference=', '.join(NAME[b] for b in ref(model)),
                                     min_reference_wrong=C.k_of(len(ref(model))),
                                     calls='Bergquist 2025 Table 1 Supporting (calibrated tools)' if C.ENS in
                                     C.CALIBRATED_VERSIONS else 'single threshold')
        marks += [(key, m_) for m_ in mk]
        C.fig_letter(fig, L, 0.05 if k == 0 else x0 - 0.85, y0 + 1.9 + 0.8)
        axes.append(ax)
    C.same_ylim(axes)
    for k, (ax, x, y, c) in marks:
        pv = tabs[k].loc[tabs[k]['class'] == c, 'mannwhitney_p'].iloc[0]
        ax.text(x, y, C.stars(pv), ha='center', va='bottom', fontsize=FS['note'])
    return tabs


def gain_bars(ax, T):
    """One comparison (first row of T): ΔAUROC, ΔAUPRC, log-loss reduction as bars with 95% CI; filled = p < 0.05;
    stars above (p in the table)."""
    r, col = T.iloc[0], COLOR[GLM]
    est = np.array([r[m] for m in C.DIFF])
    lo, hi = np.array([r[f'{m}_lo'] for m in C.DIFF]), np.array([r[f'{m}_hi'] for m in C.DIFF])
    top = max(0, np.nanmax(hi))
    bot = min(0, np.nanmin(lo))
    for x, m in enumerate(C.DIFF):
        p = r[f'p_{m}']
        ax.bar(x, est[x], BAR_IN * GAIN_SPAN / GAIN_W, color=col if p < 0.05 else 'white', edgecolor=col,
               linewidth=1.6)                                             # BAR_IN wide, as the boxes
        ax.errorbar(x, est[x], yerr=C.xerr(est[x], lo[x], hi[x]), color='#333333', capsize=3, lw=1.1)
        ax.text(x, max(hi[x], 0) + 0.04 * (top - bot), C.stars(p), ha='center', va='bottom', fontsize=FS['note'])
    ax.axhline(0, color='#888888', lw=0.8)
    ax.set_ylim(bot - 0.05 * (top - bot), top + 0.25 * (top - bot))
    ax.set_xlim(-0.5, -0.5 + GAIN_SPAN)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['ΔAUROC', 'ΔAUPRC', 'Log-loss\nreduction'], fontsize=FS['tick'])
    ax.tick_params(axis='y', labelsize=FS['tick'])
    ax.set_ylabel('Gain from adding\nGLM-Missense', fontsize=FS['label'])


def fig6():
    """SGE: A GLM-Missense-resolved vs background (RNA score); B +GLM-Missense | base (ESM1b-FT) + SpliceAI."""
    D = C.sge_data()
    ref = lambda m: [b for b in C.BASE_FT if b != m]                      # noqa: E731
    b = C.BASE_FT
    P = {'b+SAI': C.ensemble(b + [SAI], D, 'sge'), 'b+SAI+GLM': C.ensemble(b + [SAI, GLM], D, 'sge')}
    T = C.compare([(f'+{NAME[GLM]} | base + SpliceAI', 'b+SAI+GLM', 'b+SAI', COLOR[GLM])], P, D[2].label.to_numpy())
    T['base'] = ', '.join(NAME[x] for x in b)
    y0 = 1.0
    fig = plt.figure(figsize=(10, y0 + 1.9 + 1.0))
    tabs = resolved_boxes(fig, D, 'RNA', 'RNA score\n(log$_2$ RNA/DNA)', y0, ref, SGE_CLASSES, [GLM], [1.0])
    ax = C.axes_in(fig, 1.0 + BOX_W + 1.4, y0, GAIN_W, 1.9)
    gain_bars(ax, T)
    ax.set_title('GLM-Missense added to\nbase predictors + SpliceAI', fontsize=FS['note'], pad=8)
    C.fig_letter(fig, 'B', 1.0 + BOX_W + 0.35, y0 + 1.9 + 0.8)
    C.center(fig)
    tabs['added_to_base_spliceai'] = T
    resolved_list('fig6', D, 'RNA', ref, SGE_CLASSES, [GLM])
    C.finish(fig, 'fig6', tabs, D, majority=C.MAJORITY, calibrated_calls=C.ENS in C.CALIBRATED_VERSIONS,
             calls_benign_pathogenic_oriented=C.cutoffs(), sge=C.ARGS.sge, reference='base (ESM1b-FT)')


def figS13():
    """ClinVar: SpliceAI of A GLM-Missense-resolved and B ESM1b-FT-resolved vs background."""
    D = C.DALL
    ref = lambda m: C.BASE                                                # noqa: E731
    y0 = 1.0
    fig = plt.figure(figsize=(10, y0 + 1.9 + 1.0))
    tabs = resolved_boxes(fig, D, SAI, 'SpliceAI Δ score', y0, ref, CLINVAR_CLASSES, [GLM, FT],
                          [1.0, 1.0 + BOX_W + 1.6])
    C.center(fig)
    resolved_list('figS13', D, SAI, ref, CLINVAR_CLASSES, [GLM, FT])
    C.finish(fig, 'figS13', tabs, D, majority=C.MAJORITY, calibrated_calls=C.ENS in C.CALIBRATED_VERSIONS,
             calls_benign_pathogenic_oriented=C.cutoffs(), reference=C.BASE)


if __name__ == '__main__':
    ap = C.parser(__doc__)
    ap.add_argument('--figs', nargs='+', default=['fig6', 'figS13'], choices=['fig6', 'figS13'])
    args = ap.parse_args()
    C.setup(args)
    for f in args.figs:
        C.log(f'── {f}')
        {'fig6': fig6, 'figS13': figS13}[f]()