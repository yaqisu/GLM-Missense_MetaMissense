#!/usr/bin/env python3
"""
clinvar_transcripts.py — the transcript ClinVar named each missense variant on, mapped
to Ensembl.

The ClinVar VCF carries no transcript. ClinVar's variant_summary.txt.gz does, in its
Name column, e.g.  NM_000546.6(TP53):c.818G>A (p.Arg273His).  This script parses it for
every VariationID in the missense BEDs (BED column 4 = VCF ID = VariationID =
variant_id downstream) and maps the RefSeq transcript to Ensembl with two curated
resources, in this order:
  1. MANE (NCBI/EMBL-EBI): RefSeq_nuc <-> Ensembl_nuc, MANE Select / Plus Clinical
  2. NCBI gene2ensembl: RefSeq RNA <-> Ensembl transcript pairs, human only
Accessions are matched without version suffixes (dbNSFP lists unversioned ENSTs);
annotate_dbnsfp.py checks that the matched Ensembl transcript has the same c. change.

Output columns:
    variant_id chromosome position ref_allele alt_allele
    clinvar_transcript clinvar_hgvsc clinvar_hgvsp
    mane_enst mane_status g2e_enst   (g2e_enst: ';'-separated, unversioned)

Downloads (into data/clinvar/):
    wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
    wget https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2ensembl.gz
    wget -r -np -nd -A 'MANE.GRCh38.v*.summary.txt.gz' \\
         https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/

Usage (repo root):
    python preprocessing/clinvar_transcripts.py \\
        --summary data/clinvar/variant_summary.txt.gz \\
        --mane data/clinvar/MANE.GRCh38.v1.4.summary.txt.gz \\
        --gene2ensembl data/clinvar/gene2ensembl.gz \\
        --beds 'data/bed/ClinVar.*.missense.hg38.*.bed' \\
        --output data/clinvar/clinvar_transcripts.tsv
"""
import argparse
import glob
from pathlib import Path

import pandas as pd

NAME_RE = r'^(?P<tx>[A-Z]{2}_\d+(?:\.\d+)?)(?:\([^)]*\))?:(?P<c>c\.[^ ]+)(?: \((?P<p>p\.[^)]+)\))?'
nover = lambda s: s.str.split('.').str[0]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--summary', required=True)
ap.add_argument('--mane', required=True)
ap.add_argument('--gene2ensembl', required=True)
ap.add_argument('--beds', default='data/bed/ClinVar.*.missense.hg38.*.bed')
ap.add_argument('--output', default='data/clinvar/clinvar_transcripts.tsv', type=Path)
args = ap.parse_args()

# ── VariationIDs in the missense BEDs ─────────────────────────────────────────
beds = sorted(glob.glob(args.beds))
assert beds, f'no BEDs match {args.beds}'
bed = pd.concat([pd.read_csv(b, sep='\t', header=None, dtype=str,
                             names=['chr', 'start', 'end', 'variant_id', 'ref', 'alt']) for b in beds])
bed = bed.drop_duplicates('variant_id')
print(f'{len(beds)} BEDs, {len(bed):,} unique VariationIDs')

# ── ClinVar transcript from variant_summary ───────────────────────────────────
cols = ['VariationID', 'Name', 'Assembly', 'Chromosome', 'PositionVCF', 'ReferenceAlleleVCF', 'AlternateAlleleVCF']
vs = pd.read_csv(args.summary, sep='\t', dtype=str, usecols=cols)
vs = vs[(vs.Assembly == 'GRCh38') & vs.VariationID.isin(set(bed.variant_id))].drop_duplicates('VariationID')
m = vs.Name.str.extract(NAME_RE)
out = pd.DataFrame({'variant_id': vs.VariationID, 'chromosome': vs.Chromosome, 'position': vs.PositionVCF,
                    'ref_allele': vs.ReferenceAlleleVCF, 'alt_allele': vs.AlternateAlleleVCF,
                    'clinvar_transcript': m.tx, 'clinvar_hgvsc': m.c, 'clinvar_hgvsp': m.p})
nm = nover(out.clinvar_transcript.fillna(''))

# ── RefSeq -> Ensembl: MANE ───────────────────────────────────────────────────
mane = pd.read_csv(args.mane, sep='\t', dtype=str)
mane.columns = mane.columns.str.lstrip('#')
assert {'RefSeq_nuc', 'Ensembl_nuc', 'MANE_status'} <= set(mane.columns), f'MANE columns: {list(mane.columns)}'
mane_map = mane.assign(nm=nover(mane.RefSeq_nuc), enst=nover(mane.Ensembl_nuc)).drop_duplicates('nm').set_index('nm')
out['mane_enst'] = nm.map(mane_map.enst)
out['mane_status'] = nm.map(mane_map.MANE_status)

# ── RefSeq -> Ensembl: gene2ensembl (human) ───────────────────────────────────
g2e = pd.concat(c[c['#tax_id'] == '9606'] for c in pd.read_csv(
    args.gene2ensembl, sep='\t', dtype=str, chunksize=1_000_000,
    usecols=['#tax_id', 'RNA_nucleotide_accession.version', 'Ensembl_rna_identifier']))
g2e = g2e[g2e['RNA_nucleotide_accession.version'] != '-']
g2e['nm'], g2e['enst'] = nover(g2e['RNA_nucleotide_accession.version']), nover(g2e.Ensembl_rna_identifier)
g2e_map = g2e.drop_duplicates(['nm', 'enst']).groupby('nm').enst.agg(';'.join)
out['g2e_enst'] = nm.map(g2e_map)

# ── Checks and report ─────────────────────────────────────────────────────────
chk = out.merge(bed, on='variant_id')
bad = ((chk.chr.str.replace('chr', '') + ':' + (chk.start.astype(int) + 1).astype(str) + chk.ref + '>' + chk.alt)
       != (chk.chromosome + ':' + chk.position + chk.ref_allele + '>' + chk.alt_allele)).sum()
has_tx = out.clinvar_transcript.notna()
print(f'  in variant_summary (GRCh38): {len(out):,}/{len(bed):,}; coordinate mismatches vs BED: {bad:,}')
print(f'  ClinVar transcript parsed: {has_tx.sum():,}')
print(f'    mapped by MANE: {out.mane_enst.notna().sum():,} '
      f'({out.mane_status.value_counts().to_dict()})')
print(f'    mapped by gene2ensembl only: {(out.mane_enst.isna() & out.g2e_enst.notna()).sum():,}')
print(f'    not mapped: {(has_tx & out.mane_enst.isna() & out.g2e_enst.isna()).sum():,}')
args.output.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.output, sep='\t', index=False)
print(f'saved {args.output}')
