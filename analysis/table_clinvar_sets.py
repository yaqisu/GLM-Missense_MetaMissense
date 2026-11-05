#!/usr/bin/env python3
"""
table_clinvar_sets.py — the ClinVar sets table of the Methods (tab:clinvar_sets): variants per release, label mapping
and split, by ClinVar class (P, LP, P/LP, B, LB, B/LB), and the chromosomes of each split, read from the split files.

Each set (SETS below) = one split TSV per split (variant_id = ClinVar VariationID, label) + the class BEDs of its
release (col 1 chromosome, col 4 VariationID; one BED per class, as in prep_main_table.py). Checks: every variant has
one class, the class agrees with the label, the strict mapping holds only B and P, training and held-out chromosomes
do not overlap. The chromosomes of every split are written out, and a held-out list other than core.TEST_CHROMS is
flagged, so the caption can be checked against the data.

Output: results/analysis/tables/table_clinvar_sets.tsv (counts, chromosomes) and table_clinvar_sets.tex (the
tabular rows and the caption's chromosome sentence).

    python analysis/table_clinvar_sets.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import ROOT, TEST_CHROMS, log, read_tsv   # noqa: E402

CLASSES = {'pathogenic': 'P', 'likely_pathogenic': 'LP', 'pathogenic_likely_pathogenic': 'P/LP',
           'benign': 'B', 'likely_benign': 'LB', 'benign_likely_benign': 'B/LB'}
SPLITS = {'training': 'Train', 'validation': 'Val'}
# release, mapping, split TSV ({split} = training / validation), class BED ({cls} = CLASSES key), primary use per split
SETS = [
    ('Early', 'Strict', 'data/splits/ClinVar.251103.missense.hg38.seq12k.BvsP_{split}.tsv',
     'data/bed/ClinVar.251103.missense.hg38.{cls}.bed',
     {'training': 'Frozen-backbone, LoRA and full fine-tuning', 'validation': 'Frozen-backbone, LoRA and full fine-tuning'}),
    ('Early', 'Expanded', 'data/splits/ClinVar.251103.missense.hg38.seq12k.BLBvsPLP_{split}.tsv',
     'data/bed/ClinVar.251103.missense.hg38.{cls}.bed',
     {'training': 'GLM-Missense training', 'validation': 'GLM-Missense validation'}),
    ('Late', 'Expanded', 'data/splits/ClinVar.260923only.missense.hg38.seq12k.BLBvsPLP_{split}.tsv',
     'data/bed/ClinVar.260923only.missense.hg38.{cls}.bed',
     {'training': r'\makecell[l]{MetaMissense training and cross-validation,\\comparison with established methods}',
      'validation': r'\makecell[l]{MetaMissense validation,\\comparison with established methods}'}),
]
COLS = ['P', 'LP', 'P/LP', 'B', 'LB', 'B/LB']
CHROM_ORDER = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']


def must(path):
    p = ROOT / path
    if not p.exists():
        near = sorted(x.name for x in p.parent.glob('*')) if p.parent.exists() else []
        sys.exit(f'{p} not found; edit SETS in {Path(__file__).name}. Files in {p.parent}:\n  ' + '\n  '.join(near))
    return p


def chrom_ranges(chroms):
    """['1', '4', '5', '6', 'X'] -> '1, 4--6, X' (runs of 3+ consecutive autosomes as ranges)."""
    idx = sorted(CHROM_ORDER.index(c) for c in chroms)
    out, run = [], [idx[0]]
    for i in idx[1:] + [None]:
        if i is not None and i == run[-1] + 1 and i < 22:
            run.append(i)
            continue
        lab = [CHROM_ORDER[j] for j in run]
        out += [f'{lab[0]}--{lab[-1]}'] if len(run) >= 3 else lab
        run = [i] if i is not None else []
    return ', '.join(out)


def classes(bed):
    rows = []
    for f, cls in CLASSES.items():
        b = pd.read_csv(must(bed.format(cls=f)), sep='\t', header=None, usecols=[0, 3], dtype=str,
                        names=['chromosome', 'variant_id'])
        rows.append(b.assign(clinvar_class=cls))
    b = pd.concat(rows).drop_duplicates()
    assert not b.variant_id.duplicated().any(), f'{bed}: a VariationID in more than one class BED'
    b['chromosome'] = b.chromosome.str.replace('^chr', '', regex=True)
    return b.set_index('variant_id')


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--out', default='results/analysis/tables', help='output folder')
args = ap.parse_args()
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)

rows, beds = [], {}
for release, mapping, split_tsv, bed, use in SETS:
    if bed not in beds:
        beds[bed] = classes(bed)
    B = beds[bed]
    chroms = {}
    for sp, sp_lab in SPLITS.items():
        path = must(split_tsv.format(split=sp))
        d = read_tsv(path, usecols=['variant_id', 'label'])
        assert not d.variant_id.duplicated().any(), f'{path}: duplicate variant_id'
        miss = ~d.variant_id.isin(B.index)
        assert not miss.any(), f'{path}: {miss.sum()} variants in no class BED of {bed}'
        d = d.join(B, on='variant_id')
        bad = (d.clinvar_class.isin(['P', 'LP', 'P/LP']).astype(int) != d.label.astype(int)).sum()
        assert bad == 0, f'{path}: ClinVar class disagrees with the label for {bad} variants'
        if mapping == 'Strict':
            assert d.clinvar_class.isin(['P', 'B']).all(), f'{path}: strict set with classes other than B and P'
        chroms[sp] = set(d.chromosome)
        n = d.clinvar_class.value_counts().reindex(COLS, fill_value=0)
        r = {'release': release, 'mapping': mapping, 'split': sp_lab, 'use': use[sp], **n.to_dict(),
             'P+LP+P/LP': int(n[['P', 'LP', 'P/LP']].sum()), 'B+LB+B/LB': int(n[['B', 'LB', 'B/LB']].sum()),
             'total': len(d), 'chromosomes': chrom_ranges(chroms[sp]), 'file': str(path.relative_to(ROOT))}
        rows.append(r)
        log(f'{release} {mapping} {sp_lab}: {len(d):,} variants, chromosomes {r["chromosomes"]}')
    assert not chroms['training'] & chroms['validation'], f'{split_tsv}: training and held-out chromosomes overlap'
    if chroms['validation'] != TEST_CHROMS:
        log(f'WARNING {release} {mapping}: held-out chromosomes {chrom_ranges(chroms["validation"])} are not '
            f'{chrom_ranges(TEST_CHROMS)} (core.TEST_CHROMS)')

T = pd.DataFrame(rows)
T.drop(columns=['use']).to_csv(out / 'table_clinvar_sets.tsv', sep='\t', index=False)


def fmt(r, c):
    return '--' if r['mapping'] == 'Strict' and c in ('LP', 'P/LP', 'LB', 'B/LB', 'P+LP+P/LP', 'B+LB+B/LB') \
        else f'{int(r[c]):,}'


tex = [' & '.join([r['release'], r['mapping'], r['split'], r['use']] +
                  [fmt(r, c) for c in COLS + ['P+LP+P/LP', 'B+LB+B/LB', 'total']]) + r' \\' for r in rows]
splits = {s: sorted({r['chromosomes'] for r in rows if r['split'] == s}) for s in SPLITS.values()}
same = all(len(v) == 1 for v in splits.values())
caption = (f'Both releases used the same chromosome-exclusive split: Train, chromosomes {splits["Train"][0]}; '
           f'Val, chromosomes {splits["Val"][0]}.' if same else
           '% the sets do not share one split; chromosomes per set in table_clinvar_sets.tsv')
(out / 'table_clinvar_sets.tex').write_text('% rows of tab:clinvar_sets (table_clinvar_sets.py)\n' + '\n'.join(tex) +
                                            '\n% caption:\n% ' + caption + '\n')
with pd.option_context('display.width', 250, 'display.max_columns', 30):
    print(T.drop(columns=['use', 'file']).to_string(index=False))
print('\n' + caption)
log(f'wrote {out}/table_clinvar_sets.tsv, table_clinvar_sets.tex')
