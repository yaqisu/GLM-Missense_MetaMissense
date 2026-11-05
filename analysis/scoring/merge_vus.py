#!/usr/bin/env python3
"""
merge_vus.py — one annotated table of the held-out ClinVar VUS: the VUS table (dbNSFP, GLM-Missense, SpliceAI) + the
scores computed in analysis/scoring/: ESM1b-FT (ESM1b-FT_score, ESM1b-FT_status), Saluki (saluki_*), sequence features
(seq_*: seq_junction_dist, codon and context columns, seq_status), FIMO TF-binding disruption (fimo_*, once
fimo_score.py has run: run_parallel.sh fimo) and, with --alphagenome, AlphaGenome features.
Read by core.vus() / fig_vus.py. A score that is missing for a variant stays empty (never imputed); the coverage of
each score is printed and written next to the output (<out stem>.coverage.tsv).

    python analysis/scoring/merge_vus.py                    # defaults below; env lamar
    python analysis/scoring/merge_vus.py --alphagenome results/scores/vus_alphagenome_atlas/alphagenome_features.tsv
                                                            # once the full AlphaGenome scoring is done
"""
import argparse
import glob
from pathlib import Path

import pandas as pd

AG_PREFIXES = ('SPLICE_SITES_', 'SPLICE_SITE_USAGE_', 'SPLICE_JUNCTIONS_', 'RNA_SEQ_', 'POLYADENYLATION_', 'CHIP_TF_',
               'DNASE_', 'ATAC_', 'CHIP_HISTONE_', 'CAGE_', 'PROCAP_', 'AVI')
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--vus', default='results/annotated/vus/ClinVar_20260923_VUS_missense_heldoutOnly_dbNSFP_GLM-Missense_spliceAI_annot.tsv')
ap.add_argument('--esm1bft', default='results/scores/vus_esm1bft/vus_esm1bft_scores.tsv')
ap.add_argument('--saluki', default='results/scores/vus_saluki/saluki_features_*.tsv')
ap.add_argument('--seq', default='results/scores/vus_seq/seq_features_*.tsv')
ap.add_argument('--fimo', default='results/scores/vus_fimo/fimo_features_*.tsv', help='left out (with a note) if no file matches')
ap.add_argument('--alphagenome', default=None, help='the full AlphaGenome table (e.g. results/scores/vus_alphagenome_atlas/alphagenome_features.tsv), once scored for every VUS; fig_vus.py then reads AlphaGenome from this table and keeps every case')
ap.add_argument('--out', default='results/annotated/vus/ClinVar_20260923_VUS_heldout_annotated.tsv')
a = ap.parse_args()


def read(p, **k):
    return pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False, low_memory=False, **k)


def shards(pattern, what):
    files = sorted(glob.glob(pattern))
    assert files, f'no {what} tables match {pattern}'
    t = pd.concat([read(f) for f in files], ignore_index=True).drop_duplicates('variant_id')
    print(f'{what}: {len(t):,} variants from {len(files)} file(s) ({pattern})')
    return t


v = read(a.vus).rename(columns={'variant_id_x': 'variant_id'}).drop_duplicates('variant_id')
print(f'VUS: {len(v):,} from {a.vus}')
ft = read(a.esm1bft, usecols=['variant_id', 'score', 'status']).drop_duplicates('variant_id')
v = v.merge(ft.rename(columns={'score': 'ESM1b-FT_score', 'status': 'ESM1b-FT_status'}), on='variant_id', how='left')
sal = shards(a.saluki, 'Saluki')
v = v.merge(sal[['variant_id'] + [c for c in sal if c.startswith('saluki_')]], on='variant_id', how='left')
seq = shards(a.seq, 'sequence features').drop(columns=['TRANSCRIPT_ID', 'gene', 'genename'], errors='ignore')
v = v.merge(seq.rename(columns={c: f'seq_{c}' for c in seq if c != 'variant_id'}), on='variant_id', how='left')
if glob.glob(a.fimo):
    fimo = shards(a.fimo, 'FIMO')
    v = v.merge(fimo[['variant_id'] + [c for c in fimo if c.startswith('fimo_')]], on='variant_id', how='left')
else:
    print(f'FIMO: no tables match {a.fimo}; left out (run_parallel.sh fimo)')
if a.alphagenome:
    ag = shards(a.alphagenome, 'AlphaGenome')
    ag = ag.rename(columns={'ag_source': 'alphagenome_source'})
    v = v.merge(ag[['variant_id'] + [c for c in ag if c.startswith(AG_PREFIXES) or c == 'alphagenome_source']],
                on='variant_id', how='left')
v = v.fillna('')

num = lambda c: pd.to_numeric(v[c], errors='coerce').notna().sum()          # noqa: E731
cov = pd.DataFrame([{'score': k, 'column': c, 'n': num(c), 'frac': num(c) / len(v)} for k, c in
                    [('GLM-Missense', 'GLM-Missense_score'), ('ESM1b-FT', 'ESM1b-FT_score'), ('SpliceAI', 'spliceai_DS_max'),
                     ('Saluki', 'saluki_delta'), ('sequence features', 'seq_junction_dist'), ('FIMO', 'fimo_max_abs_delta'),
                     ('AlphaGenome', 'RNA_SEQ_n'), ('AlphaGenome AVI', 'AVI_score')] if c in v])
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
v.to_csv(out, sep='\t', index=False)
cov.to_csv(out.with_suffix('.coverage.tsv'), sep='\t', index=False)
print(cov.to_string(index=False))
print(f'wrote {out} ({len(v):,} VUS, {v.shape[1]} columns) and {out.with_suffix(".coverage.tsv")}')
