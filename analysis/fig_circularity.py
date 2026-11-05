#!/usr/bin/env python3
"""
fig_circularity.py — ClinVar label circularity: do predictors look better because labels used them?

ClinGen Evidence Repository (ERepo) expert-panel classifications of the late-release variants (prep_clingen.py) are
re-scored with the Bayesian point system without the computational missense-predictor criteria PP3 / BP4; a variant's
label holds (still P/LP or B/LB) or depends on them (VUS without). BP7 (splicing, synonymous / intronic) is not a
missense-predictor criterion and stays in the points.

  figS12  A the final set (ClinGen-classified late-release variants with every base-predictor score), stacked % of
          each ClinVar class by category: holds (blue; hatched = no PP3 / BP4 applied, solid = applied but not
          needed) or depends (orange), each segment n / (%) in black; B PP3 / BP4 applied to the final set, by
          strength; C AUROC of the base on the final set (black) and with the label-dependent variants removed
          (blue). Every ClinGen-classified late-release variant (before requiring the scores) is counted in the log
          and figS12_steps.tsv only.
Table figS12_variants.tsv: every final-set variant with its category (use), PP3 / BP4 codes and points, to check
the counts of A and B against each other (B counts applications: one variant can carry several, or BP4 instead of
PP3).

C: class-stratified resamples of the full set (default_rng(seed)); both AUROCs are recomputed on each resample, the
reduced set taken from within it (paired); resamples where the reduced set lacks a class are discarded (n_boot_used);
* = BH q < 0.05 (across scores) for removed − all.
Variants: every late-release ClinGen-classified variant (training and test chromosomes) with every base-predictor
score; no other score is required. GLM-Missense is not shown: the training-chromosome variants share genes with its
training data. Table figS12_calls.tsv: share of P/LP called pathogenic and of B/LB called benign (core.call) by each
base predictor, depends vs the rest, with Fisher's exact p (not in the figure).

    python analysis/fig_circularity.py --ensemble bergquist
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import fisher_exact

import core as C
from core import COLOR, FS, NAME, plt

CG_FILL = {'holds': '#92C5DE', 'depends': '#F4A582'}
CG_DOT = {'holds': '#2166AC', 'depends': '#B2182B'}
CG_EDGE = '#4393C3'                 # hatch colour
SEG_LINE = 'black'                  # outline of each segment of A ('white' for gaps instead)
COMP = ['PP3', 'BP4']


def ab_data(ids):
    """ERepo variants by filter step (late release; every base-predictor score = the set of B and C), ClinVar label and
    use of PP3 / BP4 (none; used, label holds without; label depends on them), and the criteria applied to the last
    step."""
    E = C.read_tsv(C.ARGS.erepo).drop_duplicates('variant_id')
    stale = E.comp_codes.str.contains('BP7').any()
    assert not stale, f'{C.ARGS.erepo} counts BP7 as computational; rerun analysis/prep_clingen.py'
    E['use'] = np.select([E.uses_comp == 'False', E.robust_nocomp == 'True'], ['none', 'holds'], 'depends')
    steps = [('ClinGen-classified, late release', E),
             ('With every base-predictor score (final set)', E[E.variant_id.isin(ids)])]
    UC = {'none': ('Holds without PP3/BP4\n(not applied)', CG_FILL['holds'], '///'),
          'holds': ('Holds without PP3/BP4\n(applied, not needed)', CG_FILL['holds'], ''),
          'depends': ('Depends on PP3/BP4\n(VUS without)', CG_FILL['depends'], '')}
    U = pd.concat([pd.crosstab(d.label_side, d.use).reindex(index=['P/LP', 'B/LB'], columns=list(UC), fill_value=0)
                   .assign(step=st) for st, d in steps]).reset_index().rename(columns={'label_side': 'label'})
    C.log('  ClinGen-classified variants by step, ClinVar label and use of PP3 / BP4:\n' + U.to_string(index=False))
    c = steps[-1][1].comp_codes.str.split(r' \| |,').explode()
    c = c[c != ''].str.extract(r'(\w+)\(([+-]\d+)\)')
    c['strength'] = c[1].astype(int).abs().map({1: 'Supporting', 2: 'Moderate', 4: 'Strong', 8: 'Very strong'})
    SC = c.groupby([0, 'strength']).size().rename('n').reset_index().rename(columns={0: 'criterion'})
    SC['row'] = SC.criterion + ' ' + SC.strength
    SC = SC.sort_values(['criterion', 'strength'], key=lambda s: s.map(
        {'PP3': 0, 'BP4': 1, 'Supporting': 0, 'Moderate': 1, 'Strong': 2, 'Very strong': 3})).reset_index(drop=True)
    C.log('  PP3 / BP4 applied, final set:\n' + SC[['row', 'n']].to_string(index=False))
    return E, steps, U, SC, UC


def ab_panels(fig, top, st, U, SC, UC, hA):
    """A (final set: stacked % of each ClinVar class by use of PP3 / BP4, n / (%) in black at the centre of each
    segment; legend to the right) and B (criteria applied to the final set), side by side."""
    W, H = fig.get_size_inches()
    u = U[U.step == st].set_index('label')
    tot = u[list(UC)].sum(axis=1)
    ax = C.axes_in(fig, 1.1, top - hA, 1.9, hA)
    for x, lab in enumerate(['P/LP', 'B/LB']):
        bottom, n_t = 0.0, int(tot[lab])
        for k, (_, col, hatch) in UC.items():
            n = int(u.loc[lab, k])
            pc = 100 * n / n_t if n_t else 0
            ax.bar(x, pc, bottom=bottom, color=col, width=0.7, hatch=hatch, edgecolor=CG_EDGE, linewidth=0)
            ax.bar(x, pc, bottom=bottom, width=0.7, fill=False, edgecolor=SEG_LINE, linewidth=0.8)   # outline only
            if n:
                box = dict(boxstyle='round,pad=0.1', fc=col, ec='none') if hatch else None
                ax.text(x, bottom + pc / 2, f'{n}\n({pc:.0f}%)', ha='center', va='center', fontsize=FS['note'],
                        linespacing=0.9, color='black', bbox=box)
            bottom += pc
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'{lab}\n(n = {int(tot[lab])})' for lab in ['P/LP', 'B/LB']], fontsize=FS['note'])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('% of variants', fontsize=FS['label'])
    ax.tick_params(axis='y', labelsize=FS['tick'])
    items = list(UC.values())[::-1]                                       # top segment first, as stacked
    fig.legend([Patch(facecolor=c_, hatch=h_, edgecolor=CG_EDGE, linewidth=0) for _, c_, h_ in items],
               [lab for lab, _, _ in items], loc='upper left', bbox_to_anchor=(3.2 / W, top / H),
               fontsize=FS['note'], frameon=False, handlelength=1.6, handleheight=1.0, handletextpad=0.4,
               labelspacing=0.8)
    C.fig_letter(fig, 'A', 0.05, top + 0.3)
    # B: to the right, centred on A
    hB = 0.4 * max(len(SC), 1)
    yB = top - hA / 2 - hB / 2
    ax = C.axes_in(fig, 7.7, yB, 1.9, hB)
    if len(SC):
        yy3 = -np.arange(len(SC), dtype=float)
        ax.barh(yy3, SC.n, color=['#555555' if x == 'PP3' else '#A6A6A6' for x in SC.criterion], height=0.7)
        for yi, n in zip(yy3, SC.n):
            ax.text(n + 0.02 * SC.n.max(), yi, f'{n}', va='center', fontsize=FS['note'])
        ax.set_xlim(0, 1.25 * SC.n.max())
        ax.set_yticks(yy3)
        ax.set_yticklabels(SC.row, fontsize=FS['note'])
        ax.set_ylim(yy3.min() - 0.5, 0.5)
    ax.set_xlabel('Times applied', fontsize=FS['label'])
    ax.tick_params(axis='x', labelsize=FS['tick'])
    C.fig_letter(fig, 'B', 6.0, yB + hB + 0.45)


def circularity(name, **manifest):
    """A–B funnel and criteria; C AUROC of the base on every ClinGen-classified late-release variant with every base
    score vs label-dependent ones removed; base calls on dependent vs other labels (table only)."""
    d, S = pd.concat([C.DTR, C.DTE], ignore_index=True), pd.concat([C.STR, C.STE], ignore_index=True)
    ok = d.variant_id.isin(C.read_tsv(C.ARGS.erepo).variant_id).to_numpy() & C.complete(S, C.BASE)
    ok &= ~d.variant_id.duplicated().to_numpy()
    d, S = C.subset(d, S, ok)
    meth = C.BASE
    E, steps, U, SC, UC = ab_data(d.variant_id)
    use = E.set_index('variant_id').use.reindex(d.variant_id.to_numpy()).to_numpy()
    y, keep = d.label.to_numpy().astype(int), use != 'depends'
    X = np.column_stack([S[m].to_numpy(float) for m in meth])
    nm = len(meth)
    nw, nwo = (int(y.sum()), int((y == 0).sum())), (int(y[keep].sum()), int((y[keep] == 0).sum()))
    C.log(f'  all: {nw[0]} P/LP, {nw[1]} B/LB; PP3 / BP4-dependent removed: {nwo[0]} P/LP, {nwo[1]} B/LB')
    assert min(nw) >= 2, 'fewer than 2 ClinGen-classified variants of a class in this set'

    def auc(yy, v):
        return C.auroc(yy, v) if 0 < yy.sum() < len(yy) else np.nan

    def wo(j):
        yy, xx, kk = y[j], X[j], keep[j]
        return np.array([[auc(yy, xx[:, c]) for c in range(nm)], [auc(yy[kk], xx[kk, c]) for c in range(nm)]])

    est, Bt = wo(np.arange(len(y))), np.array([wo(j) for j in C.boot_idx(y)])
    DD = Bt[:, 1] - Bt[:, 0]
    ci = lambda v: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5))   # noqa: E731
    T = pd.DataFrame([{'method': m, 'name': NAME[m], 'n_PLP_all': nw[0], 'n_BLB_all': nw[1], 'n_PLP_removed': nwo[0],
                       'n_BLB_removed': nwo[1], 'auroc_all': est[0, c], 'all_lo': ci(Bt[:, 0, c])[0],
                       'all_hi': ci(Bt[:, 0, c])[1], 'auroc_removed': est[1, c], 'removed_lo': ci(Bt[:, 1, c])[0],
                       'removed_hi': ci(Bt[:, 1, c])[1], 'dAUROC_removed_minus_all': est[1, c] - est[0, c],
                       'dAUROC_lo': ci(DD[:, c])[0], 'dAUROC_hi': ci(DD[:, c])[1],
                       'p_dAUROC': C.p_boot(DD[:, c][~np.isnan(DD[:, c])]), 'n_boot_used': int((~np.isnan(DD[:, c])).sum())}
                      for c, m in enumerate(meth)])
    T['q_dAUROC'] = C.bh(T.p_dAUROC)
    C.log('  AUROC, all vs PP3 / BP4-dependent labels removed:\n' + T[['name', 'auroc_all', 'auroc_removed',
          'dAUROC_removed_minus_all', 'dAUROC_lo', 'dAUROC_hi', 'q_dAUROC']].round(3).to_string(index=False))
    # calls (core.call): share of P/LP called P and of B/LB called B, PP3 / BP4-dependent labels vs the rest
    K = []
    for m in C.BASE:
        cl = np.asarray(C.call(S, m), float)
        for lab, want, cls in ((1, 1.0, 'P/LP'), (0, 0.0, 'B/LB')):
            g = {'depends': (y == lab) & ~keep, 'rest': (y == lab) & keep}
            n = {k: int(v.sum()) for k, v in g.items()}
            ok_ = {k: int((cl[v] == want).sum()) for k, v in g.items()}
            K.append({'method': m, 'name': NAME[m], 'label': cls, **{f'n_{k}': n[k] for k in g},
                      **{f'correct_{k}': ok_[k] for k in g}, **{f'no_call_{k}': int(np.isnan(cl[v]).sum()) for k, v in g.items()},
                      **{f'share_{k}': ok_[k] / n[k] if n[k] else np.nan for k in g},
                      'fisher_p': fisher_exact([[ok_['depends'], n['depends'] - ok_['depends']],
                                                [ok_['rest'], n['rest'] - ok_['rest']]])[1] if min(n.values()) else np.nan})
    K = pd.DataFrame(K)
    C.log('  calls correct, PP3 / BP4-dependent labels vs the rest:\n' + K[['name', 'label', 'correct_depends', 'n_depends',
          'share_depends', 'correct_rest', 'n_rest', 'share_rest', 'fisher_p']].round(3).to_string(index=False))

    h, y0 = 0.42 * nm, 0.6
    hA = 3.0
    top = y0 + h + 1.25 + hA
    fig = plt.figure(figsize=(10, top + 0.45))
    W, H = fig.get_size_inches()
    ab_panels(fig, top, steps[-1][0], U, SC, UC, hA)
    yy = -np.arange(nm, dtype=float)
    ax = C.axes_in(fig, 2.0, y0, 3.6, h)
    col = {'all': 'black', 'removed': CG_DOT['holds']}
    for k, o in (('all', 0.15), ('removed', -0.15)):
        e_ = T[f'auroc_{k}']
        ax.errorbar(e_, yy + o, xerr=C.xerr(e_, T[f'{k}_lo'], T[f'{k}_hi']), fmt='none', ecolor=col[k], lw=1)
        ax.scatter(e_, yy + o, s=26, zorder=3, color=col[k])
    for yi, a, b, q in zip(yy, T.auroc_all, T.auroc_removed, T.q_dAUROC):
        ax.plot([a, b], [yi + 0.15, yi - 0.15], color='#888888', lw=0.8, zorder=2)
        if q < 0.05:
            ax.text(1.0, yi, C.stars(q), transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=FS['note'])
    ax.set_yticks(yy)
    ax.set_yticklabels([NAME[m] for m in meth], fontsize=FS['tick'])
    C.color_labels(ax.get_yticklabels(), [COLOR[m] for m in meth])
    ax.set_ylim(yy.min() - 0.6, 0.6)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel('AUROC', fontsize=FS['label'])
    ax.tick_params(axis='x', labelsize=FS['tick'])
    C.fig_letter(fig, 'C', 0.05, y0 + h + 0.35)
    fig.legend([plt.Line2D([], [], marker='o', ls='', color=col[k]) for k in col],
               [f'All labels ({nw[0]} P/LP, {nw[1]} B/LB)',
                f'PP3/BP4-dependent labels\nremoved ({nwo[0]} P/LP, {nwo[1]} B/LB)'],
               loc='upper left', bbox_to_anchor=(6.0 / W, (y0 + h) / H), fontsize=FS['note'], frameon=False,
               handletextpad=0.3, labelspacing=0.8)
    keep_cols = ['variant_id', 'label_side', 'use', 'comp_codes', 'func_codes', 'points', 'points_nocomp', 'side_erepo',
                 'side_nocomp', 'erepo_agrees', 'n_erepo', 'Assertion', 'Expert Panel', 'Applied Evidence Codes (Met)']
    V = E[E.variant_id.isin(d.variant_id)]
    V = V[[c for c in keep_cols if c in V]].sort_values(['label_side', 'use', 'variant_id'])
    C.finish(fig, name, {'steps': U, 'comp_strength': SC, 'auroc': T, 'calls': K, 'variants': V},
             n_eval=len(d), n_eval_pos=int(y.sum()), eval_variant_ids_md5=C.md5_ids(d.variant_id),
             n_test_chromosomes=int(d.variant_id.isin(C.DTE.variant_id).sum()), erepo=C.ARGS.erepo, erepo_md5=hashlib.md5(Path(C.ARGS.erepo).read_bytes()).hexdigest(),
             computational_criteria=COMP, methods=meth, **manifest)


if __name__ == '__main__':
    ap = C.parser(__doc__)
    ap.add_argument('--erepo', default='results/analysis/prep/clingen_erepo/erepo_variants.tsv',
                    help='ERepo overlap of the ClinVar tables (prep_clingen.py)')
    args = ap.parse_args()
    args.erepo = str(C.ROOT / args.erepo)
    C.setup(args)
    C.log('── figS12')
    circularity('figS12')