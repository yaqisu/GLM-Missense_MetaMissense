#!/usr/bin/env python3
"""
make_subsets.py — training / validation subsets of the architecture-exploration split for LoRA model selection
(Fig 3A), the allele-aware input comparison (Fig 3B, Fig S5) and cross-validation (Table S1).

How the manuscript subsets were made: one random draw without replacement from each split file,
    tail -n +2 <training split> | shuf -n 5000        tail -n +2 <validation split> | shuf -n 1250
(GNU coreutils shuf, no --random-source, so the draw used fresh system randomness and cannot be regenerated from a
seed). Labels were not used in the draw. The drawn variant IDs are kept, in draw order, in
preprocessing/architecture_subset_ids.tsv; the cross-validation set is the training subset followed by the
validation subset.

Default: the subsets used in the manuscript (5,000 training and 1,250 validation variants), rebuilt from the variant
IDs in preprocessing/architecture_subset_ids.tsv (a smaller --n_train / --n_val takes the first n listed IDs).
If any listed ID is missing from the input, or more variants are asked for than are listed, that subset is drawn at
random instead, with --seed (default 0).
--random: ignore the ID list and draw every variant at random (numpy default_rng); without --seed the draw is
different on every run, with --seed it is reproducible.

Input:  data/splits/ClinVar.251103.missense.hg38.seq{size}.BvsP_{training,validation}.tsv   (generate_datasets.sh)
Output: data/splits/ClinVar.251103.missense.hg38.seq{size}.BvsP_{training,validation}_subset.tsv
        data/splits/ClinVar.251103.missense.hg38.seq{size}.BvsP_cv_subset.tsv   (training + validation subsets, in
        that order; input of training/NT2_ranked_cv.py)

    python preprocessing/make_subsets.py                          # manuscript subsets (12k)
    python preprocessing/make_subsets.py --size 6k                # same variants, 6k windows
    python preprocessing/make_subsets.py --random --seed 7        # a new random draw
"""
import argparse
from pathlib import Path

import numpy as np

STEM = 'data/splits/ClinVar.251103.missense.hg38'
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--size', default='12k', help='window size of the split files (default 12k)')
ap.add_argument('--n_train', type=int, default=5000)
ap.add_argument('--n_val', type=int, default=1250)
ap.add_argument('--ids', default='preprocessing/architecture_subset_ids.tsv',
                help='variant IDs of the manuscript subsets (columns variant_id, subset)')
ap.add_argument('--random', action='store_true', help='ignore --ids; draw every variant at random')
ap.add_argument('--seed', type=int, default=None,
                help='random seed (default 0 when the ID list cannot be used; none with --random)')
a = ap.parse_args()

seed = a.seed if (a.seed is not None or a.random) else 0
rng = np.random.default_rng(seed)                       # seed None -> fresh OS randomness, unpredictable
listed = {'training': [], 'validation': []}
if not a.random:
    for line in Path(a.ids).read_text().splitlines()[1:]:
        vid, sub = line.split('\t')[:2]
        listed[sub].append(vid)

header, rows = None, {}
for split, n in [('training', a.n_train), ('validation', a.n_val)]:
    src = f'{STEM}.seq{a.size}.BvsP_{split}.tsv'
    with open(src) as f:                                 # pass 1: variant IDs only
        h = f.readline()
        ids = [line.split('\t', 1)[0] for line in f]
    assert header in (None, h), f'{src}: header differs'
    header = h
    assert n <= len(ids), f'{split}: asked for {n:,} variants, {src} has {len(ids):,}'
    have = set(ids)
    miss = sum(v not in have for v in listed[split])
    use_list = not a.random and miss == 0 and n <= len(listed[split])
    if use_list:
        keep, how = listed[split][:n], f'listed IDs ({a.ids})'
    else:                                                # random draw of the whole subset
        keep = [ids[i] for i in rng.choice(len(ids), n, replace=False)]
        why = ('' if a.random else f'; {miss:,} listed IDs not in {src}' if miss
               else f'; {n:,} asked for, {len(listed[split]):,} listed')
        how = f'random ({"unseeded" if seed is None else f"seed {seed}"}){why}'
    want, got = set(keep), {}
    with open(src) as f:                                 # pass 2: rows of the chosen variants
        f.readline()
        for line in f:
            vid = line.split('\t', 1)[0]
            if vid in want:
                got[vid] = line
    rows[split] = [got[v] for v in keep]
    print(f'{split}: {n:,} variants, {how}')

for name, r in [('training', rows['training']), ('validation', rows['validation']),
                ('cv', rows['training'] + rows['validation'])]:
    out = Path(f'{STEM}.seq{a.size}.BvsP_{name}_subset.tsv')
    out.write_text(header + ''.join(r))
    labels = [x.rstrip('\n').rsplit('\t', 1)[1] for x in r]
    print(f'  {out}: {len(r):,} variants ({labels.count("1"):,} pathogenic)')
