#!/usr/bin/env python3
"""
prep_main_table.py — one annotated table per late-release split with every score and annotation.

Starts from the re-annotated GLM-Missense table (all dbNSFP 5.3.1a columns, strict values in the
original columns plus <score>_fb / <score>_tier / clinvar_match / matched_enst, SpliceAI columns)
and adds, matched on variant_id unless noted:
  label                  from the split TSV (source of truth; checked against true_label if present)
  clinvar_class          B / LB / B/LB / P / LP / P/LP from the ClinVar class BEDs (col 4 = VariationID)
  split                  test (held-out chr 2, 3, 16, 17) or train
  ESM1b-FT_score         ESM1b-FT table, matched on chromosome / position / ref / alt
  NT2-ZS_score, NT2-ZS_llr   zero-shot NT-2 (evaluation/zeroshot_nt.py): sigmoid(-LLR) and LLR
  random_baseline_score  results/scores/random_baseline.tsv (Uniform(0, 1), seed 2026, created once)
Sequence columns (ref_sequence / alt_sequence) are dropped. Nothing else is changed: values are
read and written as text, so dbNSFP '.' and multi-value fields stay as they are.

Output: results/analysis/prep/main_table/ClinVar.260923only.BLBvsPLP_{test,train}.main.tsv + manifest_{split}.json.
ESCOTT, AlphaGenome AVI and Evo 2 were then added to these tables; the annotated copies are the analysis inputs in
results/annotated/clinvar/ (copy_inputs.sh). Score coverage: python analysis/core.py.

Usage (repo root):
    python analysis/prep_main_table.py --split test
    python analysis/prep_main_table.py --split train
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import md5_ids, log   # noqa: E402

P = 'results/ClinVar.260923only.missense.hg38.seq12k.BLBvsPLP'
S = 'data/splits/ClinVar.260923only.missense.hg38.seq12k.BLBvsPLP'
ZS = 'results/predictions/ClinVar.260923only.BLBvsPLP'
BED = 'data/bed/ClinVar.260923only.missense.hg38.{}.bed'
CLASSES = {'benign': 'B', 'likely_benign': 'LB', 'benign_likely_benign': 'B/LB',
           'pathogenic': 'P', 'likely_pathogenic': 'LP', 'pathogenic_likely_pathogenic': 'P/LP'}
KEYS = ['chromosome', 'position', 'ref_allele', 'alt_allele']

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--split', default='test', choices=['test', 'train'])
ap.add_argument('--table', help=f'default {P}_{{validation,training}}_GLM-Missense_annotated_dbNSFPannotated.tsv')
ap.add_argument('--split_tsv', help=f'default {S}_{{validation,training}}.tsv')
ap.add_argument('--esm1b_ft', help=f'default {P}_{{validation,training}}_ESM1b-FT_annotated.tsv')
ap.add_argument('--nt2_zs', help=f'default {ZS}[_train]/zeroshot_NT2_seq12k.tsv')
ap.add_argument('--random', default='results/scores/random_baseline.tsv')
ap.add_argument('--min_nt2zs_coverage', type=float, default=0.999, help='stop if NT-2 zero-shot covers less of the split')
ap.add_argument('--allow_missing_nt2zs', action='store_true')
ap.add_argument('--outdir', default='results/analysis/prep/main_table')
args = ap.parse_args()
sp = {'test': 'validation', 'train': 'training'}[args.split]
table = args.table or f'{P}_{sp}_GLM-Missense_annotated_dbNSFPannotated.tsv'
split_tsv = args.split_tsv or f'{S}_{sp}.tsv'
esm_path = args.esm1b_ft or f'{P}_{sp}_ESM1b-FT_annotated.tsv'
zs_path = args.nt2_zs or (f'{ZS}/zeroshot_NT2_seq12k.tsv' if args.split == 'test' else f'{ZS}_train/zeroshot_NT2_seq12k.tsv')
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)


def read(path, **kw):
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, low_memory=False, **kw)


def norm_keys(d):
    k = d[KEYS].copy()
    k['chromosome'] = k.chromosome.str.replace('^chr', '', regex=True)
    k['position'] = k.position.astype(int)
    return k


# ── base table ────────────────────────────────────────────────────────────────
df = read(table)
log(f'{table}: {len(df):,} rows, {df.shape[1]} columns')
df = df.drop(columns=[c for c in ('ref_sequence', 'alt_sequence') if c in df])
assert not df.variant_id.duplicated().any(), 'duplicate variant_id in the annotated table'
for c in ('label', 'clinvar_class', 'split', 'ESM1b-FT_score', 'NT2-ZS_score', 'NT2-ZS_llr', 'random_baseline_score'):
    assert c not in df, f'{table} already has a {c} column; pass a clean annotated table'

# label: the split TSV is the source of truth and defines the variant set
lab = read(split_tsv, usecols=['variant_id', 'label'])
assert set(lab.variant_id) == set(df.variant_id), \
    f'variant sets differ: {len(set(lab.variant_id) - set(df.variant_id))} only in split, ' \
    f'{len(set(df.variant_id) - set(lab.variant_id))} only in table'
df['label'] = df.variant_id.map(lab.set_index('variant_id').label).astype(int)
if 'true_label' in df:
    bad = (pd.to_numeric(df.true_label, errors='coerce') != df.label).sum()
    assert bad == 0, f'true_label disagrees with the split label for {bad} variants'

# ClinVar class from the BEDs (col 4 = VariationID), coordinates cross-checked
beds = []
for f, cls in CLASSES.items():
    b = pd.read_csv(BED.format(f), sep='\t', header=None, usecols=[0, 2, 3], dtype=str,
                    names=['chromosome', 'position', 'variant_id'])
    beds.append(b.assign(clinvar_class=cls))
beds = pd.concat(beds).drop_duplicates()
assert not beds.variant_id.duplicated().any(), 'a VariationID appears in more than one ClinVar class BED'
beds = beds.set_index('variant_id')
df['clinvar_class'] = df.variant_id.map(beds.clinvar_class)
assert df.clinvar_class.notna().all(), f'{df.clinvar_class.isna().sum()} variants not in any class BED'
pos_bed = df.variant_id.map(beds.position).astype(int)
assert (pos_bed == df.position.astype(int)).all(), 'BED end != position for some variants (ID / coordinate mismatch)'
cls_lab = df.clinvar_class.isin(['P', 'LP', 'P/LP']).astype(int)
assert (cls_lab == df.label).all(), 'ClinVar class disagrees with the split label'
df['split'] = args.split

# ESM1b-FT (coordinates)
ft = read(esm_path, usecols=KEYS + ['ESM1b-FT_score']).drop_duplicates()
kf = norm_keys(ft)
assert not kf.duplicated().any(), 'conflicting ESM1b-FT scores for the same variant'
ft = pd.concat([kf, ft[['ESM1b-FT_score']]], axis=1)
df['ESM1b-FT_score'] = norm_keys(df).merge(ft, on=KEYS, how='left')['ESM1b-FT_score'].fillna('').to_numpy()

# NT-2 zero-shot (variant_id)
# NT-2 zero-shot (variant_id): scores every variant from sequence alone, so it must cover the whole split
assert Path(zs_path).exists() or args.allow_missing_nt2zs, \
    f'{zs_path} not found (pass --nt2_zs PATH, or --allow_missing_nt2zs to build the table without it)'
if Path(zs_path).exists():
    zs = read(zs_path, usecols=['variant_id', 'pathogenicity_score', 'log_likelihood_ratio'])
    assert not zs.variant_id.duplicated().any(), f'duplicate variant_id in {zs_path} (resumed run written twice?)'
    zs = zs.set_index('variant_id')
    df['NT2-ZS_score'] = df.variant_id.map(zs.pathogenicity_score).fillna('')
    df['NT2-ZS_llr'] = df.variant_id.map(zs.log_likelihood_ratio).fillna('')
    cov = (df['NT2-ZS_score'] != '').mean()
    extra = set(zs.index) - set(df.variant_id)
    log(f'NT-2 zero-shot: {(df["NT2-ZS_score"] != "").sum():,}/{len(df):,} variants ({cov:.2%}) from {zs_path}'
        + (f'; {len(extra):,} variants in the file are not in this split' if extra else ''))
    assert cov >= args.min_nt2zs_coverage or args.allow_missing_nt2zs, \
        f'NT-2 zero-shot covers only {cov:.2%} of the {args.split} split; is the scoring run finished?'
else:
    log(f'WARNING: {zs_path} not found; NT2-ZS columns left empty (--allow_missing_nt2zs)')
    df['NT2-ZS_score'] = df['NT2-ZS_llr'] = ''

# random baseline
rnd = read(args.random).set_index('variant_id').random_baseline_score
df['random_baseline_score'] = df.variant_id.map(rnd).fillna('')
assert (df.random_baseline_score != '').all(), f'{args.random} does not cover every variant'

# ── write ─────────────────────────────────────────────────────────────────────
out = outdir / f'ClinVar.260923only.BLBvsPLP_{args.split}.main.tsv'
df.to_csv(out, sep='\t', index=False)
log(f'wrote {out}: {len(df):,} variants ({df.label.sum():,} P/LP), {df.shape[1]} columns')

print('ClinVar classes:', df.clinvar_class.value_counts().to_dict())
(outdir / f'manifest_{args.split}.json').write_text(json.dumps({
    'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'split': args.split, 'output': str(out),
    'inputs': {'table': table, 'split_tsv': split_tsv, 'esm1b_ft': esm_path, 'nt2_zs': zs_path,
               'random': args.random, 'beds': BED},
    'n': len(df), 'n_pos': int(df.label.sum()), 'variant_ids_md5': md5_ids(df.variant_id),
    'classes': df.clinvar_class.value_counts().to_dict()}, indent=2))
