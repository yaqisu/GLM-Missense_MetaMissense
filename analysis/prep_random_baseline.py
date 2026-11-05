#!/usr/bin/env python3
"""
prep_random_baseline.py — the Random (null) score of the later-release ClinVar variants.

One Uniform(0, 1) value per variant: numpy.random.default_rng(2026).random(n) over the sorted (string sort) union of
the training and test variant_ids. Written once and never overwritten, so every run uses the same values; a rerun
only checks that the file covers both tables.

Input:  results/ClinVar.260923only.missense.hg38.seq12k.BLBvsPLP_{training,validation}_GLM-Missense_annotated_dbNSFPannotated.tsv
Output: results/scores/random_baseline.tsv (variant_id, random_baseline_score) + random_baseline.json (recipe),
        read by prep_main_table.py

    python analysis/prep_random_baseline.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

P = 'results/ClinVar.260923only.missense.hg38.seq12k.BLBvsPLP'
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--tables', nargs='+', default=[f'{P}_{s}_GLM-Missense_annotated_dbNSFPannotated.tsv'
                                                for s in ('training', 'validation')])
ap.add_argument('--seed', type=int, default=2026)
ap.add_argument('--out', default='results/scores/random_baseline.tsv')
a = ap.parse_args()

ids = sorted(set().union(*[set(pd.read_csv(t, sep='\t', dtype=str, usecols=['variant_id']).variant_id)
                           for t in a.tables]))
out = Path(a.out)
if out.exists():
    have = set(pd.read_csv(out, sep='\t', dtype=str).variant_id)
    missing = set(ids) - have
    assert not missing, f'{len(missing):,} variants not in {out}; it was made for another variant set'
    print(f'{out} exists and covers all {len(ids):,} variants; not changed')
else:
    out.parent.mkdir(parents=True, exist_ok=True)
    col = 'random_baseline_score'
    pd.DataFrame({'variant_id': ids, col: np.random.default_rng(a.seed).random(len(ids))}).to_csv(
        out, sep='\t', index=False)
    out.with_suffix('.json').write_text(json.dumps({
        'column': col, 'seed': a.seed, 'distribution': 'Uniform(0, 1)',
        'generator': 'numpy.random.default_rng(seed).random(n)',
        'order': 'sorted union of train + test variant_id (string sort)', 'n': len(ids), 'numpy': np.__version__,
        'tables': a.tables}, indent=2))
    print(f'wrote {out}: {len(ids):,} variants (seed {a.seed})')
