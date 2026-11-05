#!/usr/bin/env python3
"""
fig_complementarity.py — does the model add signal the base does not carry?

  fig4    ClinVar test set. A pairwise Kendall's τ (base + GLM-Missense); B partial Spearman with the label (+ Random);
          C drop-column ablation of base + GLM-Missense + Random, two points per predictor removed: all test variants
          (circle) and rare ones (triangle; gnomAD AF < --rare 1e-4 or absent)
  fig5    fig4 on the RAD51D SGE data (functionally abnormal = 1, normal = 0), base with ESM1b-FT in place of ESM1b;
          ensembles trained on the ClinVar training variants (RAD51D is on chr 17, a test chromosome); C all variants
          only
  figS11  fig4 with ESM1b-FT in place of GLM-Missense (protein-model control), all variants only

Drop-column rows (X, under "Predictor removed") = full model minus the model without X (positive = X adds information
the other scores do not carry); ΔAUROC, ΔAUPRC, log-loss reduction with paired bootstrap 95% CI, filled + stars = BH
q < 0.05 within the panel and stratum. Rare rows: the same models (trained once on the ClinVar training variants) on
the rare subset, bootstrapped within it; rows in the order of the all-variant rows (ΔAUROC). gnomAD AF: first value of
--af_col (default: the first of AF_COLS in the table); missing = absent from gnomAD = rare. No titles in the figures: n,
ensemble, training set and the rare definition go in the legend (tables, manifest).

    python analysis/fig_complementarity.py --ensemble bergquist --figs fig4 fig5 figS11 --n_jobs 40
"""
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, rankdata

import core as C
from core import COLOR, DIFF, DLAB, FS, FT, GLM, NAME, RND, plt

AF_COLS = ['gnomAD4.1_joint_AF', 'gnomAD4.1_AF', 'gnomAD_joint_AF']
STRATA = ['All', 'Rare']
KEY = {'All': 'All variants', 'Rare': 'Rare variants'}                 # key of panel C; definition in the legend
MARK = {'All': 'o', 'Rare': '^'}


def kendall_panel(fig, rect, S, methods):
    """Pairwise Kendall's τ: average-linkage clustering on 1 − τ (dendrogram left), scale 0.3–0.7, least concordant
    methods top left. rect = (x0, y0, size) in inches."""
    W, H = fig.get_size_inches()
    x0, y0, size = rect
    n = len(methods)
    K = np.eye(n)
    for i in range(n):
        for j in range(i):
            K[i, j] = K[j, i] = kendalltau(S[methods[i]], S[methods[j]])[0]
    Z = linkage(squareform(1 - K, checks=False), 'average')
    tight = list(np.full(n, np.inf))                                        # less concordant child first
    for i, (a, b, h, _) in enumerate(Z):
        a, b = int(a), int(b)
        if tight[b] > tight[a]:
            Z[i, 0], Z[i, 1] = b, a
        tight.append(min(h, tight[a], tight[b]))
    order = list(leaves_list(Z))
    Ko = K[np.ix_(order, order)]
    names, cols = [NAME[methods[i]] for i in order], [COLOR[methods[i]] for i in order]
    ax = fig.add_axes([x0 / W, y0 / H, size / W, size / H])
    mesh = ax.pcolormesh(Ko, cmap='Blues', vmin=0.3, vmax=0.7, edgecolors='white', linewidth=1.5)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    for i in range(n):
        for j in range(n):
            txt = f'{Ko[i, j]:.2f}' if n < 8 else f'{Ko[i, j]:.2f}'.replace('0.', '.').replace('1.00', '1')
            ax.text(j + 0.5, i + 0.5, txt, ha='center', va='center', fontsize=FS['note'],
                    color='white' if Ko[i, j] > 0.58 else 'black')
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=FS['tick'])
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(names, fontsize=FS['tick'])
    C.color_labels(ax.get_xticklabels(), cols)
    C.color_labels(ax.get_yticklabels(), cols)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color('black')
        sp.set_linewidth(1)
        sp.set_zorder(5)
    dax = fig.add_axes([(x0 - 2.3) / W, y0 / H, 0.55 / W, size / H])
    dendrogram(Z, orientation='left', ax=dax, no_labels=True, color_threshold=0, link_color_func=lambda k: '#555555')
    dax.set_ylim(0, 10 * n)
    dax.invert_yaxis()
    dax.axis('off')
    cax = fig.add_axes([(x0 + size + 0.12) / W, (y0 + size * 0.3) / H, 0.13 / W, size * 0.45 / H])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_ticks([0.3, 0.5, 0.7])
    cb.ax.tick_params(labelsize=FS['tick'])
    cb.set_label("Kendall's τ", fontsize=FS['label'])
    return ax, pd.DataFrame(K, index=methods, columns=methods)


def partial(d, S, methods):
    """Partial Spearman of each method with the label given the other real methods (Random: given all of them);
    95% CI and two-sided p vs 0 over the bootstrap resamples, BH q within the panel."""
    real = [m for m in methods if m != RND]
    y = d.label.to_numpy()
    X, r = S[real].to_numpy(float), S[RND].to_numpy(float)

    def pc(M):
        P = np.linalg.inv(np.corrcoef(M, rowvar=False))
        k = M.shape[1] - 1
        return -P[:k, k] / np.sqrt(np.diag(P)[:k] * P[k, k])

    def part(i):
        R, ry = rankdata(X[i], axis=0), rankdata(y[i])
        return np.r_[pc(np.column_stack([R, ry])), pc(np.column_stack([R, rankdata(r[i]), ry]))[-1]]

    meth = real + [RND]
    est, B = part(np.arange(len(y))), np.array([part(i) for i in C.boot_idx(y)])
    T = pd.DataFrame({'comparison': [NAME[m] for m in meth], 'method': meth, 'color': [COLOR[m] for m in meth],
                      'partial_spearman': est, 'partial_spearman_lo': np.percentile(B, 2.5, axis=0),
                      'partial_spearman_hi': np.percentile(B, 97.5, axis=0),
                      'p_partial_spearman': [C.p_boot(B[:, j]) for j in range(len(meth))]})
    T['q_partial_spearman'] = C.bh(T.p_partial_spearman)
    T = pd.concat([T[T.method != RND].sort_values('partial_spearman', ascending=False), T[T.method == RND]])
    return T.reset_index(drop=True).assign(n=len(y), n_pos=int(y.sum()))


def drop_column(members, D, tag):
    """Full ensemble vs the ensemble without each member, sorted by ΔAUROC."""
    full = '+'.join(C.canon(members))
    P = {full: C.ensemble(members, D, tag)}
    P.update({f'-{m}': C.ensemble([x for x in members if x != m], D, tag) for m in members})
    y = D[2].label.to_numpy()
    T = C.compare([(NAME[m], full, f'-{m}', COLOR[m]) for m in members], P, y)
    T['AUROC_full'] = C.auroc(y, P[full])
    return T.sort_values('dAUROC', ascending=False).reset_index(drop=True)


def rare_mask(te, cut, where):
    """Rare = AF < cut or absent from gnomAD; counts per stratum."""
    want = C.ARGS.af_col
    col = want or next((c for c in AF_COLS if c in te), None)
    if col not in te:
        sys.exit(f'{where}: no gnomAD AF column ({want or AF_COLS}); columns that look related: '
                 f'{[c for c in te.columns if "gnomad" in c.lower()][:30]}')
    af = C.first_num(te, col)
    m = (af.isna() | (af < cut)).to_numpy()
    y = te.label.to_numpy()
    N = pd.DataFrame([{'stratum': k, 'definition': dfn, 'n': int(s.sum()), 'n_pos': int(y[s].sum()),
                       'n_neg': int((y[s] == 0).sum()), 'n_absent_from_gnomAD': int((s & af.isna().to_numpy()).sum())}
                      for k, dfn, s in (('All', 'every variant', np.ones(len(y), bool)),
                                        ('Rare', f'{col} < {cut:g} or absent from gnomAD', m))])
    C.log(f'  {where}, gnomAD AF strata:\n' + N.to_string(index=False))
    return m, col, N


def drop_column_rare(members, D, tag, rare):
    """drop_column on all variants and the same ensembles on the rare ones; both in the order of the all rows."""
    full = '+'.join(C.canon(members))
    P = {full: C.ensemble(members, D, tag)}
    P.update({f'-{m}': C.ensemble([x for x in members if x != m], D, tag) for m in members})
    y = D[2].label.to_numpy()
    pairs = [(NAME[m], full, f'-{m}', COLOR[m]) for m in members]
    T = {}
    for k, s in zip(STRATA, (np.ones(len(y), bool), rare)):
        T[k] = C.compare(pairs, {a: p[s] for a, p in P.items()}, y[s]).assign(stratum=k)
        T[k]['AUROC_full'] = C.auroc(y[s], P[full][s])
    order = T['All'].sort_values('dAUROC', ascending=False).comparison
    return {k: t.set_index('comparison').loc[order].reset_index() for k, t in T.items()}


def forest_pair(ax, T, col, xlabel, labels=True, dy=0.19):
    """core.forest with two points per row: all variants (circle, upper) and rare variants (triangle, lower)."""
    yy = -np.arange(len(T['All']), dtype=float)
    for k, off in zip(STRATA, (dy, -dy)):
        for yi, r in zip(yy + off, T[k].itertuples()):
            v, lo, hi, q = getattr(r, col), getattr(r, f'{col}_lo'), getattr(r, f'{col}_hi'), getattr(r, f'q_{col}')
            ax.errorbar(v, yi, xerr=C.xerr(v, lo, hi), fmt='none', ecolor=r.color, capsize=2, lw=1.2)
            ax.scatter(v, yi, s=32, marker=MARK[k], color=r.color if q < 0.05 else 'white', edgecolor=r.color,
                       zorder=3)
            if q < 0.05:
                ax.text(max(v, hi), yi, ' ' + C.stars(q), color=r.color, va='center', fontsize=FS['note'])
    ax.axvline(0, color='grey', lw=0.8, ls='--')
    ax.locator_params(axis='x', nbins=4)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.15 * (hi - lo))
    ax.set_yticks(yy)
    ax.set_yticklabels(T['All'].comparison if labels else [], fontsize=FS['tick'])
    C.color_labels(ax.get_yticklabels(), T['All'].color)
    ax.set_ylim(yy.min() - 0.6, 0.6)
    ax.set_xlabel(xlabel, fontsize=FS['label'])
    ax.tick_params(axis='x', labelsize=FS['tick'])


def panel_pair(fig, T, x0, y0, h, letter, w=2.45, gap=0.3):
    """ΔAUROC | ΔAUPRC | log-loss reduction (as core.forest3), all and rare variants per row, key above."""
    axes = [C.axes_in(fig, x0 + j * (w + gap), y0, w, h) for j in range(len(DIFF))]
    for j, (ax, col) in enumerate(zip(axes, DIFF)):
        forest_pair(ax, T, col, DLAB[col], labels=(j == 0))
    C.fig_letter(fig, letter, 0.05, y0 + h + 0.45)
    axes[0].text(-0.03, 1.0, 'Predictor removed', transform=axes[0].transAxes, ha='right', va='bottom',
                 fontsize=FS['note'], fontweight='bold')
    keys = [Line2D([], [], ls='none', marker=MARK[k], mfc='0.35', mec='0.35', ms=7, label=KEY[k]) for k in STRATA]
    axes[1].legend(handles=keys, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                   fontsize=FS['note'], handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)


def complementarity(name, D, tag, base, model, label, where, rare=False, **manifest):
    """A Kendall (base + model), B partial Spearman (+ Random), C drop-column (base + model + Random); rare: C also on
    the rare variants (second point per row)."""
    te, Se = D[2], D[3]
    members = base + [model, RND]
    n = len(base) + 1
    side, xB, wB = (2.9, 7.75, 2.0) if n <= 8 else (3.2, 8.0, 1.75)         # > 8 methods: bigger Kendall, narrower B
    hC = (0.38 if rare else 2.6 / 9) * len(members)                          # two points per row: taller C
    yA = 0.6 + hC + 1.75
    fig = plt.figure(figsize=(10, yA + side + 0.35))
    axA, K = kendall_panel(fig, (2.4, yA, side), Se, base + [model])
    C.fig_letter(fig, 'A', 0.05, yA + side + 0.3)
    tB = partial(te, Se, members)
    axB = C.axes_in(fig, xB, yA, wB, side)
    C.forest(axB, tB, 'partial_spearman', 'Partial Spearman ρ\nwith label')
    C.fig_letter(fig, 'B', xB - 1.5, yA + side + 0.3)
    tables = {'kendall': K, 'partial': tB}
    if rare:
        m, col, N = rare_mask(te, C.ARGS.rare, where)
        T = drop_column_rare(members, D, tag, m)
        panel_pair(fig, T, 1.85, 0.6, hC, 'C')
        tables.update(drop_column=T['All'], drop_column_rare=T['Rare'], counts=N)
        manifest.update(af_column=col, rare_cutoff=C.ARGS.rare)
    else:
        tables['drop_column'] = drop_column(members, D, tag)
        C.forest3(fig, tables['drop_column'], 1.85, 0.6, hC, 'C', heading='Predictor removed')
    C.finish(fig, name, tables, D, model=model, figure_base=base, data=where, ensemble_label=label, **manifest)


if __name__ == '__main__':
    ap = C.parser(__doc__)
    ap.add_argument('--figs', nargs='+', default=['fig4', 'fig5', 'figS11'], choices=['fig4', 'fig5', 'figS11'])
    ap.add_argument('--af_col', help=f'gnomAD AF column (default: the first of {", ".join(AF_COLS)})')
    ap.add_argument('--rare', type=float, default=1e-4, help='fig4 rare variants: AF below this, or absent')
    args = ap.parse_args()
    C.setup(args)
    for f in args.figs:
        C.log(f'── {f}')
        if f == 'fig4':
            complementarity('fig4', C.DALL, 'clinvar', C.BASE, GLM, C.BL, 'ClinVar test set', rare=True)
        elif f == 'fig5':
            complementarity('fig5', C.sge_data(), 'sge', C.BASE_FT, GLM, C.BL_FT, 'RAD51D SGE', sge=args.sge)
        else:
            complementarity('figS11', C.DALL, 'clinvar', C.BASE, FT, C.BL, 'ClinVar test set')
