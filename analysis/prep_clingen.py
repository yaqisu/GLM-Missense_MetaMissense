#!/usr/bin/env python3
"""prep_clingen.py — ClinGen Evidence Repository (ERepo) overlap: do the ClinVar labels hold without the missense
predictor criteria?

ERepo records, for each expert-panel (VCEP) classification, the ACMG/AMP criteria met. For every variant in the
given tables that ERepo classified, this scores the met criteria with the Bayesian point system (Tavtigian et al.
2020: Supporting 1, Moderate 2, Strong 4, Very Strong 8, BA1 stand-alone -8; P >= 10, LP 6-9, VUS 0-5, LB -1..-6,
B <= -7), once with all criteria and once without the computational missense-predictor criteria (PP3, BP4), and
asks whether the variant stays on the same side (P/LP or B/LB) as its ClinVar label. BP7 (a synonymous or intronic
variant with no predicted splicing impact) is about splicing, not the missense predictors benchmarked here, so it
stays in the points.

Inputs: data/clingen/erepo.tabbed.txt (prep_clingen_download.sh) and tables with variant_id (= ClinVar
VariationID), chromosome and label (1 = P/LP, 0 = B/LB); default = the late-release main tables.
Extra tables as path:tag, e.g. the 251103 held-out-chromosome variants.

Output: results/analysis/prep/clingen_erepo/erepo_variants.tsv (one row per table variant found in ERepo),
erepo_summary.tsv (counts per tag x label), printed.

    python -u analysis/prep_clingen.py
    python -u analysis/prep_clingen.py --tables results/annotated/clinvar/ClinVar.260923only.BLBvsPLP_test.main.tsv:late_test \
        <251103 table>:early_heldout
"""
import argparse
import re
from pathlib import Path

import pandas as pd

M = 'results/annotated/clinvar/ClinVar.260923only.BLBvsPLP_{}.main.tsv'
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--erepo', default='data/clingen/erepo.tabbed.txt')
ap.add_argument('--tables', nargs='+', default=[f'{M.format("test")}:late_test', f'{M.format("train")}:late_train'])
ap.add_argument('--chroms', nargs='*', default=['2', '3', '16', '17'],
                help='held-out chromosomes, flagged in the output (empty = flag nothing)')
ap.add_argument('--outdir', default='results/analysis/prep/clingen_erepo')
args = ap.parse_args()
out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)

COMP = {'PP3', 'BP4'}                            # computational missense-predictor criteria (BP7 = splicing: kept)
FUNC = {'PS3', 'BS3'}                            # functional assay criteria
DEFAULT = {'PVS': 8, 'PS': 4, 'PM': 2, 'PP': 1, 'BA': 8, 'BS': 4, 'BP': 1}
STRENGTH = {'SUPPORTING': 1, 'MODERATE': 2, 'STRONG': 4, 'VERYSTRONG': 8, 'STANDALONE': 8}
unknown = set()


def parse(codes):
    """'PM2_Supporting, PP3_Strong, PS3' -> [('PM2', 1), ('PP3', 4), ('PS3', 4)] with signed points."""
    res = []
    for c in str(codes).split(','):
        c = c.strip().upper().replace(' ', '').replace('-', '')
        if not c:
            continue
        m = re.fullmatch(r'(PVS|PS|PM|PP|BA|BS|BP)(\d+)(?:_(\w+))?', c)
        if not m or (m.group(3) and m.group(3) not in STRENGTH):
            unknown.add(c)
            continue
        pts = STRENGTH[m.group(3)] if m.group(3) else DEFAULT[m.group(1)]
        res.append((m.group(1) + m.group(2), -pts if m.group(1).startswith('B') else pts))
    return res


def side(p):
    return 'P/LP' if p >= 6 else 'B/LB' if p <= -1 else 'VUS'


# ── ERepo ─────────────────────────────────────────────────────────────────────
e = pd.read_csv(args.erepo, sep='\t', dtype=str, keep_default_na=False)
print(f'{args.erepo}: {len(e):,} classifications; example codes:', *e['Applied Evidence Codes (Met)'].head(3), sep='\n  ')
e = e[(e['Retracted'].str.lower() != 'true') & (e['ClinVar Variation Id'] != '')].copy()
e['codes'] = e['Applied Evidence Codes (Met)'].map(parse)
e['points'] = e.codes.map(lambda x: sum(p for _, p in x))
e['points_nocomp'] = e.codes.map(lambda x: sum(p for c, p in x if c not in COMP))
e['comp_codes'] = e.codes.map(lambda x: ','.join(f'{c}({p:+d})' for c, p in x if c in COMP))
e['func_codes'] = e.codes.map(lambda x: ','.join(f'{c}({p:+d})' for c, p in x if c in FUNC))
e['side_erepo'] = e.Assertion.str.lower().map(
    lambda a: 'P/LP' if 'pathogenic' in a else 'B/LB' if 'benign' in a else 'VUS')
e['side_points'] = e.points.map(side)
e['side_nocomp'] = e.points_nocomp.map(side)
if unknown:
    print(f'WARNING: {len(unknown)} unparsed codes (ignored): {sorted(unknown)[:20]}')
agree = (e.side_points == e.side_erepo).mean()
print(f'points reproduce the ERepo assertion side for {agree:.1%} of {len(e):,} classifications '
      '(sanity check of the parser; VCEP-specific rules explain some mismatches)')

# one row per variant; a variant with several VCEP classifications keeps all, joined
cols = ['Assertion', 'Expert Panel', 'Approval Date', 'Applied Evidence Codes (Met)', 'comp_codes', 'func_codes',
        'points', 'points_nocomp', 'side_erepo', 'side_nocomp']
g = e.groupby('ClinVar Variation Id')
ev = g[cols].agg(lambda s: ' | '.join(s.astype(str)))
ev['n_erepo'] = g.size()
ev['uses_comp'] = g.comp_codes.agg(lambda s: (s != '').any())
ev['uses_func'] = g.func_codes.agg(lambda s: (s != '').any())

# ── overlap with the tables ───────────────────────────────────────────────────
rows = []
for spec in args.tables:
    path, tag = spec.rsplit(':', 1)
    t = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, usecols=lambda c: c in
                    {'variant_id', 'chromosome', 'label', 'clinvar_class', 'genename'})
    t['chromosome'] = t.chromosome.str.replace('^chr', '', regex=True)
    print(f'{tag}: {len(t):,} variants, {t.variant_id.isin(ev.index).sum():,} in ERepo')
    rows.append(t.assign(tag=tag).merge(ev, left_on='variant_id', right_index=True))
d = pd.concat(rows, ignore_index=True)
d['label_side'] = d.label.map({'1': 'P/LP', '0': 'B/LB'})
d['heldout_chrom'] = d.chromosome.isin(args.chroms)
d['erepo_agrees'] = d.side_erepo.str.split(' \\| ').map(set) == d.label_side.map(lambda x: {x})
# label is robust if every ERepo classification stays on the label's side without computational criteria
d['robust_nocomp'] = d.apply(lambda r: set(r.side_nocomp.split(' | ')) == {r.label_side}, axis=1)
d.to_csv(out / 'erepo_variants.tsv', sep='\t', index=False)

s = (d.groupby(['tag', 'heldout_chrom', 'label_side'])
     .agg(n_in_erepo=('variant_id', 'size'), erepo_agrees=('erepo_agrees', 'sum'),
          uses_comp=('uses_comp', 'sum'), robust_nocomp=('robust_nocomp', 'sum'),
          uses_func=('uses_func', 'sum'))
     .reset_index())
s.to_csv(out / 'erepo_summary.tsv', sep='\t', index=False)
print(s.to_string(index=False))
print('\ncomputational codes among variants that use them:')
print(d.loc[d.uses_comp, 'comp_codes'].str.split(r' \| |,').explode()
      .loc[lambda x: x != ''].value_counts().to_string())
print(f'\nwrote {out}/erepo_variants.tsv, erepo_summary.tsv')
