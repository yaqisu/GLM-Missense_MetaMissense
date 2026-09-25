#!/usr/bin/env python3
"""
Extract variants that are new in the later ClinVar release (260923) relative to the
training release (251103), removing any variant that was part of the training data.

A 260923 variant is removed if:
  1. its chr_pos_ref_alt or ClinVar VariationID is in ANY of the 6 251103 BEDs, pooled
     across classes (resubmissions AND reclassifications, e.g. LB->B, B/LB->B, B->P)
  2. it appears in more than one 260923 class BED
Variants that existed in 251103 but not in its BEDs (e.g. VUS, conflicting) are kept:
they were never used for training or validation.

Output: data/bed/ClinVar.260923only.missense.hg38.{class}.bed (same 6-column format).
These are split by generate_datasets.sh with the same fixed chromosome assignment as
251103 (split_data_fixed_chroms.py):
  train chromosomes -> *_training.tsv   (new variants in genes seen during training)
  val chromosomes   -> *_validation.tsv (held-out test set: no genes shared with training)

Prints class x split counts, the 251103 -> 260923 reclassification table, and a
gene-overlap audit (when both missense VCFs are present in data/vcf/).

Usage (repo root):
  python preprocessing/subtract_new_variants.py [-b /path/to/bcftools]
"""
import argparse
import io
import os
import shutil
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_data_fixed_chroms import TRAIN_CHROMS, VAL_CHROMS

CLASSES = ['benign', 'likely_benign', 'likely_pathogenic', 'pathogenic',
           'benign_likely_benign', 'pathogenic_likely_pathogenic']
BENIGN = {'benign', 'likely_benign', 'benign_likely_benign'}
BED_COLS = ['chr', 'start', 'end', 'variant_id', 'ref', 'alt']
BED = 'data/bed/ClinVar.{tag}.missense.hg38.{cls}.bed'
VCF = 'data/vcf/clinvar_20{tag}_missense.vcf.gz'


def split_of(chrom):
    return 'train' if chrom in TRAIN_CHROMS else 'test' if chrom in VAL_CHROMS else 'excluded'


def read_beds(tag):
    dfs = []
    for c in CLASSES:
        d = pd.read_csv(BED.format(tag=tag, cls=c), sep='\t', header=None, names=BED_COLS, dtype=str)
        if d.empty:
            sys.exit(f'ERROR: {BED.format(tag=tag, cls=c)} is empty. Delete it (and an empty '
                     f'data/vcf/clinvar_20{tag}_missense.vcf.gz) and re-run process_clinvar.sh.')
        d['cls'] = c
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df['key'] = df.chr + '_' + df.start + '_' + df.ref + '_' + df.alt
    df['split'] = df.chr.str.replace('^chr', '', regex=True).map(split_of)
    return df


def read_vcf_genes(path, bcftools):
    fmt = '%CHROM\t%POS\t%REF\t%ALT\t%INFO/GENEINFO\n'
    out = subprocess.run([bcftools, 'query', '-f', fmt, path],
                         capture_output=True, text=True, check=True).stdout
    d = pd.read_csv(io.StringIO(out), sep='\t', header=None, dtype=str,
                    names=['chrom', 'pos', 'ref', 'alt', 'geneinfo'])
    d['key'] = 'chr' + d.chrom + '_' + (d.pos.astype(int) - 1).astype(str) + '_' + d.ref + '_' + d.alt
    d['genes'] = d.geneinfo.fillna('').str.findall(r'([^:|]+):\d+')   # GENE1:id|GENE2:id
    return d


def label_counts(df, title):
    t = df[df.split != 'excluded']
    print(f'\n{title}')
    print(pd.crosstab(t.cls, t.split, margins=True).reindex(CLASSES + ['All']).fillna(0).astype(int).to_string())
    for s in ['train', 'test']:
        m = t[t.split == s]
        print(f'  {s:5s}  BLBvsPLP: {m.cls.isin(BENIGN).sum():>6} B/LB  {(~m.cls.isin(BENIGN)).sum():>6} P/LP   |'
              f'  BvsP: {(m.cls == "benign").sum():>6} B  {(m.cls == "pathogenic").sum():>6} P')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old', default='251103', help='training release tag (default: 251103)')
    ap.add_argument('--new', default='260923', help='later release tag (default: 260923)')
    ap.add_argument('-b', '--bcftools', default='bcftools', help='only needed for the gene audit')
    args = ap.parse_args()

    print(f'Training release: {args.old}   Later release: {args.new}')
    print(f'Train chroms: {sorted(TRAIN_CHROMS)}\nVal   chroms: {sorted(VAL_CHROMS)}')

    old, new = read_beds(args.old), read_beds(args.new)
    label_counts(old, f'{args.old} (train / val chroms)')

    # --- 1. remove anything in the training release (pooled over all classes) ----
    new['in_old'] = new.key.isin(old.key) | new.variant_id.isin(old.variant_id)
    prev = new.key.map(old.drop_duplicates('key').set_index('key').cls)
    prev = prev.fillna(new.variant_id.map(old.drop_duplicates('variant_id').set_index('variant_id').cls))
    print(f'\nIn {args.old} BEDs ({new.in_old.sum()}): {args.old} class (rows) -> {args.new} class (cols)')
    print(pd.crosstab(prev[new.in_old].rename('old_class'), new.cls[new.in_old]).to_string())

    # --- 2. variants in >1 new class ---------------------------------------------
    new['multi_class'] = new.key.duplicated(keep=False)

    drop = new.in_old | new.multi_class
    print(f'\n{args.new}: {len(new)} | in {args.old} {new.in_old.sum()} | '
          f'multi_class {new.multi_class.sum()} | kept {(~drop).sum()}')
    late = new[~drop]

    for c in CLASSES:
        out = BED.format(tag=f'{args.new}only', cls=c)
        late.loc[late.cls == c, BED_COLS].to_csv(out, sep='\t', header=False, index=False)
        print(f'  wrote {out}  ({(late.cls == c).sum()})')
    label_counts(late, f'{args.new}only (train chroms / held-out test chroms)')

    # --- gene-level audit (optional, needs both missense VCFs) --------------------
    old_vcf, new_vcf = VCF.format(tag=args.old), VCF.format(tag=args.new)
    if os.path.exists(old_vcf) and os.path.exists(new_vcf):
        if not (os.path.isfile(args.bcftools) or shutil.which(args.bcftools)):
            sys.exit(f'ERROR: bcftools not runnable at {args.bcftools} (pass the binary, e.g. .../bin/bcftools)')
        ov, nv = read_vcf_genes(old_vcf, args.bcftools), read_vcf_genes(new_vcf, args.bcftools)
        genes = lambda vcf, keys: set(vcf[vcf.key.isin(keys)].genes.explode().dropna())
        old_train = genes(ov, old.key[old.split == 'train'])
        old_val = genes(ov, old.key[old.split == 'test'])
        late_test = genes(nv, late.key[late.split == 'test'])
        late_train = genes(nv, late.key[late.split == 'train'])
        shared = sorted(late_test & old_train)
        print(f'\nGene audit: test genes {len(late_test)} | shared with {args.old} TRAIN {len(shared)}'
              f' {shared[:20]} | shared with {args.old} VAL {len(late_test & old_val)}')
        print(f'            train-chrom genes {len(late_train)} | shared with {args.old} TRAIN '
              f'{len(late_train & old_train)}')
    else:
        print(f'\nGene audit skipped (needs {old_vcf} and {new_vcf})')

    print('\nNext: bash preprocessing/generate_datasets.sh -c preprocessing/config.tsv -s 6k,12k,30k')


if __name__ == '__main__':
    main()