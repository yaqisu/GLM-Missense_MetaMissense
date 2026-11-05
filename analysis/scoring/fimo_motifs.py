#!/usr/bin/env python3
"""
fimo_motifs.py — the motif file of fimo_score.py: JASPAR 2026 CORE vertebrates, latest version of each profile (=
JASPAR's "non-redundant" set), from pyJASPAR (v4.0.0 bundles the JASPAR 2026 release), as a MEME motif file
(letter probabilities = PFM counts / column total, uniform background). Same content as the "CORE vertebrates
non-redundant, MEME" download of jaspar.elixir.no; either file works with fimo_score.py --motifs.

    pip install pyjaspar==4.0.0 biopython                                    # env meme
    python analysis/scoring/fimo_motifs.py                                   # -> data/external/jaspar2026/
"""
import argparse
import hashlib
from pathlib import Path

from pyjaspar import jaspardb

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--release', default='JASPAR2026')
ap.add_argument('--tax_group', default='vertebrates')
ap.add_argument('-o', '--out', default=None, help='default: data/external/jaspar2026/<release>_CORE_<tax>_non-redundant.meme')
a = ap.parse_args()
out = Path(a.out or ROOT / f'data/external/{a.release.lower()}/{a.release}_CORE_{a.tax_group}_non-redundant.meme')

ms = jaspardb(release=a.release).fetch_motifs(collection='CORE', tax_group=[a.tax_group], all_versions=False)
lines = ['MEME version 4', '', 'ALPHABET= ACGT', '', 'strands: + -', '',
         'Background letter frequencies', 'A 0.25 C 0.25 G 0.25 T 0.25', '']
for m in sorted(ms, key=lambda m: m.matrix_id):
    cols = [[m.counts[b][i] for b in 'ACGT'] for i in range(len(m))]
    n = round(sum(cols[0]))
    lines += [f'MOTIF {m.matrix_id} {m.name.replace(" ", "_")}', '',
              f'letter-probability matrix: alength= 4 w= {len(m)} nsites= {n} E= 0']
    lines += ['  '.join(f'{x / sum(c):.6f}' for x in c) for c in cols]
    lines += [f'URL http://jaspar.elixir.no/matrix/{m.matrix_id}', '']
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(lines))
md5 = hashlib.md5(out.read_bytes()).hexdigest()
print(f'{a.release} CORE {a.tax_group}: {len(ms):,} motifs (widths {min(map(len, ms))}-{max(map(len, ms))}) -> {out}\nmd5 {md5}')
