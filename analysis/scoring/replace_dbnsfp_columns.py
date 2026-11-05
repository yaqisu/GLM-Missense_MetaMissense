#!/usr/bin/env python3
"""
replace_dbnsfp_columns.py — overwrite every dbNSFP column of an annotated table with
the values from a fresh dbnsfp.tsv (scoring/annotate_dbnsfp.py output), matched on
variant_id. All other columns (GLM-Missense_score, spliceai_*, ...) and the column
order are kept. The table is backed up to <stem>.pre_reannot.bak.tsv first.

With --fallback_tsv (dbnsfp_fallback.tsv from the same run), also appends
clinvar_match, matched_enst, clinvar_hgvsc_agree and, per score, <score>_fb (value
from the first tier that has one) and <score>_tier, replacing them if present.

Usage (repo root):
    python analysis/scoring/replace_dbnsfp_columns.py --table T.tsv --dbnsfp_tsv DIR/dbnsfp.tsv \
        [--fallback_tsv DIR/dbnsfp_fallback.tsv]
"""
import argparse, shutil
from pathlib import Path
import pandas as pd

KEY = ['variant_id', 'chromosome', 'position', 'ref_allele', 'alt_allele', 'label']
SCORES = ['AlphaMissense_score', 'ESM1b_score', 'REVEL_score', 'CADD_phred',
          'SIFT_score', 'Polyphen2_HVAR_score', 'ClinPred_score']

ap = argparse.ArgumentParser()
ap.add_argument('--table', required=True, type=Path)
ap.add_argument('--dbnsfp_tsv', required=True, type=Path)
ap.add_argument('--fallback_tsv', type=Path, default=None)
args = ap.parse_args()

t = pd.read_csv(args.table, sep='\t', dtype=str, keep_default_na=False)
d = pd.read_csv(args.dbnsfp_tsv, sep='\t', dtype=str, keep_default_na=False)
assert t.variant_id.is_unique and d.variant_id.is_unique
assert set(t.variant_id) == set(d.variant_id), 'variant sets differ between table and dbnsfp.tsv'
d = d.set_index('variant_id').loc[t.variant_id]

cols = [c for c in d.columns if c in t.columns and c not in KEY]
new_only = [c for c in d.columns if c not in t.columns and c not in KEY]
print(f'{args.table.name}: n={len(t):,}; replacing {len(cols)} dbNSFP columns'
      + (f'; not in table (skipped): {new_only}' if new_only else ''))

# Keep the table's missing-value token: '' if it never used '.'
use_blank = not (t[[c for c in SCORES if c in t]] == '.').any().any()
new = d[cols].replace('.', '') if use_blank else d[cols]

def n_ok(df):
    return {s: pd.to_numeric(df[s].replace('.', ''), errors='coerce').notna().sum() for s in SCORES if s in df}

before = n_ok(t)
t[cols] = new.to_numpy()
after = n_ok(t)
print(pd.DataFrame({'before': before, 'after': after, 'gained': {s: after[s] - before[s] for s in before}}).to_string())

if args.fallback_tsv:
    fb = pd.read_csv(args.fallback_tsv, sep='\t', dtype=str, keep_default_na=False).set_index('variant_id')
    assert set(fb.index) == set(t.variant_id), 'variant sets differ between table and fallback tsv'
    fb = fb.loc[t.variant_id].rename(columns={s: f'{s}_fb' for s in SCORES})
    add = [c for c in fb.columns if c not in ('chromosome', 'position', 'ref_allele', 'alt_allele', 'clinvar_transcript')]
    if use_blank:
        fb[add] = fb[add].replace('.', '')
    t = t.drop(columns=[c for c in add if c in t])
    t[add] = fb[add].to_numpy()
    print(f'appended {len(add)} fallback columns; clinvar_match: {t.clinvar_match.value_counts().to_dict()}')

bak = args.table.with_name(args.table.stem + '.pre_reannot.bak.tsv')
if not bak.exists():
    shutil.copy2(args.table, bak)
t.to_csv(args.table, sep='\t', index=False)
print(f'saved {args.table} (backup: {bak.name})')
