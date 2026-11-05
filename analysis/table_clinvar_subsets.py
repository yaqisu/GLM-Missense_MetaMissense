#!/usr/bin/env python3
"""
table_clinvar_subsets.py — the subset rows of the ClinVar sets table (tab:clinvar_sets): the variants each analysis
used, by ClinVar class, with the same filters as the figures (core.py, fig_vus.py).

  MetaMissense training      later-release training variants with every MetaMissense input (base + GLM-Missense)
  Ablation / benchmarking    core.DALL: training and test variants with every score of the ClinVar figures
                             (core.FIGURE_SCORES)
                             check: the same with Evo 2 also required (logged only)
  VUS                        held-out VUS on the test chromosomes (all); with base, GLM-Missense and ESM1b-FT
                             (core.VUS_SCORES); the fig7 set (+ SpliceAI, the six ΔCSC sets and Saluki)
Class: the clinvar_class column of the ClinVar main tables (prep_main_table.py), or, if absent, the class BEDs of the
later release (col 4 = the main table's variant_id). Checks that the class agrees with the label.

Output: results/analysis/tables/table_clinvar_subsets.tsv and .tex (tabular rows); printed.

    python analysis/table_clinvar_subsets.py --ensemble bergquist
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C  # noqa: E402
import fig_vus as V  # noqa: E402

COLS = ['P', 'LP', 'P/LP', 'B', 'LB', 'B/LB']
BED = 'data/bed/ClinVar.260923only.missense.hg38.{cls}.bed'
BEDCLS = {'pathogenic': 'P', 'likely_pathogenic': 'LP', 'pathogenic_likely_pathogenic': 'P/LP',
          'benign': 'B', 'likely_benign': 'LB', 'benign_likely_benign': 'B/LB'}


def classes(d):
    """ClinVar class per row of a main table."""
    if 'clinvar_class' in d:
        c = d.clinvar_class.astype(str)
    else:
        m = {}
        for f, cls in BEDCLS.items():
            b = pd.read_csv(C.ROOT / BED.format(cls=f), sep='\t', header=None, usecols=[3], dtype=str)
            m.update(dict.fromkeys(b[3], cls))
        c = d.variant_id.astype(str).map(m)
        assert c.notna().all(), f'{c.isna().sum()} variants not in a class BED ({BED}); no clinvar_class column either'
    bad = (c.isin(['P', 'LP', 'P/LP']).astype(int) != d.label.astype(int)).sum()
    assert bad == 0, f'ClinVar class disagrees with the label for {bad} variants'
    return c


def row(name, split, chroms, d):
    n = classes(d).value_counts().reindex(COLS, fill_value=0)
    return {'set': name, 'split': split, 'chromosomes': chroms, **n.to_dict(),
            'P+LP+P/LP': int(n[['P', 'LP', 'P/LP']].sum()), 'B+LB+B/LB': int(n[['B', 'LB', 'B/LB']].sum()),
            'Total': len(d)}


if __name__ == '__main__':
    args = C.parser(__doc__).parse_args()
    C.setup(args)
    held = ', '.join(sorted(C.TEST_CHROMS, key=int))
    mm = C.DTR[C.complete(C.STR, C.BASE + [C.GLM])]
    tr, _, te, _ = C.DALL
    rows = [row('Later release, all variants', 'Train', 'Others', C.DTR),
            row('Later release, all variants', 'Test', held, C.DTE),
            row('MetaMissense training', 'Train', 'Others', mm),
            row('Ablation ensembles and benchmarking', 'Train', 'Others', tr),
            row('Ablation ensembles and benchmarking', 'Test', held, te)]
    T = pd.DataFrame(rows)
    ev = [C.complete(S_, C.FIGURE_SCORES() + [C.EVO2]) for S_ in (C.STR, C.STE)]
    C.log(f'  check, ablation set if Evo 2 were also required: train {int(ev[0].sum()):,} '
          f'({int(C.DTR.label[ev[0]].sum()):,} P/LP), test {int(ev[1].sum()):,} ({int(C.DTE.label[ev[1]].sum()):,} P/LP); '
          f'without it: train {len(tr):,}, test {len(te):,} -> {"identical" if (ev[0].sum(), ev[1].sum()) == (len(tr), len(te)) else "DIFFERENT"}')

    C.log('── VUS')
    d, S = C.vus()
    k1 = C.complete(S, C.VUS_SCORES())
    dv, _ = V.vus_set()                                                  # fig7: + the six ΔCSC sets and Saluki
    vus = pd.DataFrame([{'set': 'Discordant VUS analysis', 'split': s, 'chromosomes': held, 'Total': n} for s, n in (
        ('All held-out VUS', len(d)), ('Base, GLM-Missense, ESM1b-FT', int(k1.sum())),
        ('fig7 set (+ SpliceAI, six ΔCSC sets, Saluki)', len(dv)))])

    out = Path(args.out) / 'tables'
    pd.concat([T, vus], ignore_index=True).to_csv(out / 'table_clinvar_subsets.tsv', sep='\t', index=False)
    fmt = lambda v: f'{int(v):,}' if pd.notna(v) else '--'                # noqa: E731
    tex = [' & '.join([r['set'], 'Late', 'Expanded', r['split'], r['chromosomes']] +
                      [fmt(r[c]) for c in COLS + ['P+LP+P/LP', 'B+LB+B/LB', 'Total']]) + r' \\' for r in rows]
    tex += [' & '.join(['Discordant VUS analysis', 'Sep 2026', 'VUS', r['split'], held] + ['--'] * 8 + [fmt(r['Total'])])
            + r' \\' for r in vus.to_dict('records')]
    (out / 'table_clinvar_subsets.tex').write_text('\n'.join(tex) + '\n')
    with pd.option_context('display.width', 220):
        print('\n' + T.to_string(index=False) + '\n\n' + vus.to_string(index=False))
        print('\nLaTeX rows (also in table_clinvar_subsets.tex):\n' + '\n'.join(tex))
    C.log(f'wrote {out}/table_clinvar_subsets.tsv, .tex')
