#!/usr/bin/env python3
"""
fimo_score.py — TF-binding disruption of SNVs by plain FIMO (MEME suite; Grant et al. 2011, Bioinformatics): the ref
and alt sequence of a window of ±(widest motif − 1) bp around the variant (GRCh38, + strand), scanned by FIMO with
every motif of --motifs on both strands; per motif, the best p-value of the hits that cover the variant, for ref and
for alt. An allele without such a hit at p < --thresh gets p = --thresh (FIMO reports nothing above it).

Input: any variant table (VUS shard, ClinVar main table, SGE table). The variant is variant_id when it has the form
chrN:pos:REF>ALT, else chromosome / position / ref_allele / alt_allele (or chrom, pos, ref, alt); the output has one row
per input variant_id (+ fimo_variant = chrN:pos:REF>ALT when variant_id has another form). --annotated also writes the
input table with the fimo_* columns added.

Per variant (-log10 p differences over the motifs, ref − alt for a loss):
  fimo_loss            largest drop of -log10 p from ref to alt (>= 0)
  fimo_gain            largest rise (>= 0)
  fimo_max_abs_delta   max(fimo_loss, fimo_gain)
  fimo_n_loss          motifs bound in ref (p < --p_hit) and not in alt; fimo_n_gain the reverse
  fimo_any_loss        fimo_n_loss > 0 (0/1); fimo_any_gain likewise
  fimo_lost_tfs        names (motif_alt_id) of the lost motifs, comma-separated; fimo_gained_tfs likewise
  fimo_status          ok / bad_variant_id / not_snv / chrom_not_in_fasta / chrom_end / ref_mismatch
                       (features empty unless ok)
--per_motif FILE also writes one row per variant x motif with a hit in either allele (p_ref, p_alt, delta).

FIMO settings: --text (streamed, no q-values, no stored-score cap), --thresh 1e-3, --max-strand, uniform background
(--bfile --uniform--; JASPAR's own MEME files declare it too), default motif pseudocount 0.1. Motifs: JASPAR 2026
CORE vertebrates non-redundant (fimo_motifs.py). Env meme (MEME suite + pandas):

    python analysis/scoring/fimo_score.py --input <list.tsv> --fasta $FASTA --motifs $MOTIFS -o fimo_features.tsv
    python analysis/scoring/fimo_score.py --input <table.tsv> --fasta $FASTA --motifs $MOTIFS -o <scores.tsv> \
        --annotated <table + fimo_*.tsv>
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seq_features import Fasta  # noqa: E402

FIMO_COLS = ['motif_id', 'motif_alt_id', 'sequence_name', 'start', 'stop', 'strand', 'score', 'p', 'q', 'matched']
SNV = re.compile(r'^(?:chr)?([0-9]+|X|Y|MT?):(\d+):([ACGT]+)>([ACGT]+)$')
ALLELE_COLS = {'chrom': ('chromosome', 'chrom', 'chr', '#chr', 'CHROM'), 'pos': ('position', 'pos(1-based)', 'pos', 'POS'),
               'ref': ('ref_allele', 'ref', 'REF'), 'alt': ('alt_allele', 'alt', 'ALT')}
FEATURES = ['fimo_loss', 'fimo_gain', 'fimo_max_abs_delta', 'fimo_n_loss', 'fimo_n_gain', 'fimo_any_loss',
            'fimo_any_gain', 'fimo_lost_tfs', 'fimo_gained_tfs']


def snv_key(d):
    """chrN:pos:REF>ALT per row: variant_id when it has that form, else from the allele columns."""
    v = d.variant_id.astype(str)
    if v.str.match(SNV).all():
        return 'chr' + v.str.replace('^chr', '', regex=True)
    c = {k: next((x for x in cs if x in d), None) for k, cs in ALLELE_COLS.items()}
    assert all(c.values()), f'variant_id is not chrN:pos:REF>ALT and no {[k for k, x in c.items() if not x]} column'
    pos = d[c['pos']].astype(str).str.replace(r'\.0$', '', regex=True)
    return 'chr' + d[c['chrom']].astype(str).str.replace('^chr', '', regex=True) + ':' + pos + ':' + d[c['ref']] + '>' + d[c['alt']]


def motif_widths(path):
    w = [int(x) for x in re.findall(r'letter-probability matrix:.*?\bw=\s*(\d+)', Path(path).read_text())]
    assert w, f'no motifs in {path} (MEME format expected)'
    return w


def windows(df, fa, flank):
    """Per variant: status and the ref / alt sequences of pos ± flank."""
    rows = []
    for i, vid in enumerate(df.variant_id):
        r = {'i': i, 'status': 'ok', 'ref_seq': '', 'alt_seq': ''}
        try:
            chrom, pos, al = vid.split(':')
            ref, alt = al.split('>')
            pos = int(pos)
        except ValueError:
            rows.append(r | {'status': 'bad_variant_id'})
            continue
        if len(ref) != 1 or len(alt) != 1 or ref not in 'ACGT' or alt not in 'ACGT':
            rows.append(r | {'status': 'not_snv'})
            continue
        try:
            s = fa.fetch(chrom, pos - flank, pos + flank)
        except KeyError:
            rows.append(r | {'status': 'chrom_not_in_fasta'})
            continue
        if len(s) != 2 * flank + 1:
            rows.append(r | {'status': 'chrom_end'})
        elif s[flank] != ref:
            rows.append(r | {'status': 'ref_mismatch'})
        else:
            rows.append(r | {'ref_seq': s, 'alt_seq': s[:flank] + alt + s[flank + 1:]})
    return pd.DataFrame(rows, columns=['i', 'status', 'ref_seq', 'alt_seq'])


def run_fimo(a, fasta, centre):
    """FIMO hits covering the variant (1-based position centre in every sequence), streamed: best p per sequence x
    motif. Returns DataFrame i, allele, motif_id, motif_alt_id, p."""
    cmd = [a.fimo, '--text', '--thresh', str(a.thresh), '--max-strand', '--skip-matched-sequence',
           '--bfile', a.bfile, '--verbosity', '1', a.motifs, fasta]
    err = open(Path(fasta).with_suffix('.log'), 'w+')
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
    key = ['sequence_name', 'motif_id', 'motif_alt_id']
    best, n = [pd.DataFrame(columns=key + ['p'])], 0
    try:
        for ch in pd.read_csv(p.stdout, sep='\t', header=None, names=FIMO_COLS, comment='#', chunksize=2_000_000,
                              dtype=str, usecols=key + ['start', 'stop', 'p']):
            ch = ch[ch.motif_id != 'motif_id']                               # header line
            n += len(ch)
            st, sp = pd.to_numeric(ch.start), pd.to_numeric(ch.stop)
            ch = ch[(st <= centre) & (sp >= centre)].assign(p=lambda x: pd.to_numeric(x.p))
            best.append(ch.groupby(key, as_index=False).p.min())
    except pd.errors.EmptyDataError:                                         # no hit at all
        pass
    if p.wait():
        err.seek(0)
        sys.exit(f'fimo failed (exit {p.returncode}): {" ".join(map(str, cmd))}\n{err.read()[-2000:]}')
    h = pd.concat(best, ignore_index=True).astype({'p': float}).groupby(key, as_index=False).p.min()
    h[['i', 'allele']] = h.sequence_name.str.extract(r'^s(\d+)([ra])$')
    print(f'  FIMO: {n:,} hits at p < {a.thresh:g}, {len(h):,} sequence x motif covering the variant', flush=True)
    return h.assign(i=h.i.astype(int), allele=h.allele.map({'r': 'ref', 'a': 'alt'})).drop(columns='sequence_name')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='TSV with variant_id (chrN:pos:REF>ALT, or any id + allele columns)')
    ap.add_argument('--fasta', required=True, help='GRCh38 FASTA, uncompressed (.fai built if missing)')
    ap.add_argument('--motifs', required=True, help='MEME motif file (fimo_motifs.py)')
    ap.add_argument('--thresh', type=float, default=1e-3, help='FIMO output threshold = p of an allele without a hit')
    ap.add_argument('--p_hit', type=float, default=1e-4, help='bound = best p below this (FIMO default threshold)')
    ap.add_argument('--bfile', default='--uniform--', help="FIMO --bfile (default: uniform; '--motif--': the motif file's)")
    ap.add_argument('--fimo', default=shutil.which('fimo') or 'fimo', help='fimo executable')
    ap.add_argument('--per_motif', default=None, help='also write variant x motif p-values (.tsv or .tsv.gz)')
    ap.add_argument('-o', '--output', required=True, help='fimo_* per input variant_id')
    ap.add_argument('--annotated', default=None, help='also write the input table + fimo_* columns here')
    a = ap.parse_args()
    t0 = time.time()

    src = pd.read_csv(a.input, sep='\t', dtype=str, keep_default_na=False, low_memory=False)
    src = src.rename(columns={'variant_id_x': 'variant_id'}) if 'variant_id' not in src else src
    key = snv_key(src)
    df = pd.DataFrame({'variant_id': key.drop_duplicates().to_numpy()})                # scored once per SNV
    w = motif_widths(a.motifs)
    flank = max(w) - 1
    ver = subprocess.run([a.fimo, '--version'], capture_output=True, text=True).stdout.strip()
    print(f'{len(df):,} variants; {len(w):,} motifs ({a.motifs}, widths {min(w)}-{max(w)}: window ± {flank} bp); '
          f'FIMO {ver}, --thresh {a.thresh:g}, bound p < {a.p_hit:g}, --bfile {a.bfile}', flush=True)

    W = windows(df, Fasta(a.fasta), flank)
    ok = W[W.status == 'ok']
    out = pd.DataFrame({'variant_id': df.variant_id, 'fimo_status': W.status})
    if len(ok):
        with tempfile.TemporaryDirectory() as tmp:
            fa = Path(tmp) / 'alleles.fa'
            with open(fa, 'w') as fh:
                for r in ok.itertuples():
                    fh.write(f'>s{r.i}r\n{r.ref_seq}\n>s{r.i}a\n{r.alt_seq}\n')
            h = run_fimo(a, fa, flank + 1)
        P = (h.pivot_table(index=['i', 'motif_id', 'motif_alt_id'], columns='allele', values='p', aggfunc='min')
             .reindex(columns=['ref', 'alt']).fillna(a.thresh).reset_index())
        P['delta'] = np.log10(P.ref) - np.log10(P.alt)                       # -log10 p alt - ref: > 0 gain, < 0 loss
        P['lost'], P['gained'] = (P.ref < a.p_hit) & (P.alt >= a.p_hit), (P.alt < a.p_hit) & (P.ref >= a.p_hit)
        g = P.groupby('i')
        F = pd.DataFrame({'fimo_loss': (-g.delta.min()).clip(lower=0), 'fimo_gain': g.delta.max().clip(lower=0),
                          'fimo_n_loss': g.lost.sum(), 'fimo_n_gain': g.gained.sum()})
        F['fimo_lost_tfs'] = P[P.lost].groupby('i').motif_alt_id.agg(','.join)
        F['fimo_gained_tfs'] = P[P.gained].groupby('i').motif_alt_id.agg(','.join)
        F = F.reindex(ok.i)                                                  # variants without any hit: 0
        num = ['fimo_loss', 'fimo_gain', 'fimo_n_loss', 'fimo_n_gain']
        F[num] = F[num].fillna(0)
        F[['fimo_lost_tfs', 'fimo_gained_tfs']] = F[['fimo_lost_tfs', 'fimo_gained_tfs']].fillna('')
        F['fimo_max_abs_delta'] = F[['fimo_loss', 'fimo_gain']].max(axis=1)
        F['fimo_any_loss'], F['fimo_any_gain'] = (F.fimo_n_loss > 0).astype(int), (F.fimo_n_gain > 0).astype(int)
        F[['fimo_n_loss', 'fimo_n_gain']] = F[['fimo_n_loss', 'fimo_n_gain']].astype(int)
        out = out.join(F[FEATURES])
        if a.per_motif:
            P.assign(variant_id=df.variant_id.to_numpy()[P.i]).drop(columns='i').rename(
                columns={'ref': 'p_ref', 'alt': 'p_alt'}).to_csv(a.per_motif, sep='\t', index=False)
    out = out.reindex(columns=['variant_id', *FEATURES, 'fimo_status'])
    ints = ['fimo_n_loss', 'fimo_n_gain', 'fimo_any_loss', 'fimo_any_gain']
    out[ints] = out[ints].astype('Int64')                                    # empty unless ok
    res = pd.DataFrame({'variant_id': src.variant_id.to_numpy(), 'fimo_variant': key.to_numpy()}).merge(
        out.rename(columns={'variant_id': 'fimo_variant'}), on='fimo_variant', how='left').drop_duplicates('variant_id')
    if (res.variant_id == res.fimo_variant).all():
        res = res.drop(columns='fimo_variant')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.output, sep='\t', index=False)
    if a.annotated:
        Path(a.annotated).parent.mkdir(parents=True, exist_ok=True)
        src.drop(columns=[c for c in res if c.startswith('fimo_') and c in src]).merge(
            res, on='variant_id', how='left').to_csv(a.annotated, sep='\t', index=False)
        print(f'Wrote {a.annotated} (the input table + fimo_*)')
    s = out[out.fimo_status == 'ok']
    print(f'Status: {out.fimo_status.value_counts().to_dict()}\n'
          f'  any loss {int(s.fimo_any_loss.sum()):,}, any gain {int(s.fimo_any_gain.sum()):,} of {len(s):,} scored; '
          f'median max |delta| {s.fimo_max_abs_delta.median():.2f}\nWrote {a.output} ({len(out):,} variants, '
          f'{time.time() - t0:.0f} s; {len(res):,} input variants)', flush=True)


if __name__ == '__main__':
    main()
