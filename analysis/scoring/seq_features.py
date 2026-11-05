#!/usr/bin/env python3
"""
seq_features.py — sequence-derived features for coding SNVs on their (MANE) transcript, from a GRCh38 FASTA and a GTF.

Per variant (variant_id = chrN:pos:REF>ALT, transcript from TRANSCRIPT_ID, version ignored):
  checks       cds_pos, ref_matches (REF equals the transcript base), consequence (missense /
               synonymous / stop_gained / stop_lost / start_lost), aa_ref, aa_alt, codon_ref, codon_alt
  codon        codon_position (1-3), RSCU and per-thousand usage of ref/alt codon and their deltas
               (usage computed from the CDS of all MANE Select transcripts in the GTF, or all coding
               transcripts if the GTF has no MANE tags); optional --csc table -> delta_CSC
  splicing     exon_number, n_exons, exon_length, junction_dist (1 = base next to an exon-exon
  geometry     junction; inf = only bordered by a transcript end), nearest_junction (acceptor/donor),
               first_coding_exon, last_coding_exon
  motifs       optional --hexamer_table (e.g. ESRseq): sum and mean of the scores of the six hexamers
               covering the variant, ref and alt, and their delta (transcript-sense strand)
  context      transition, cpg (REF in a CpG), trinucleotide (pyrimidine-centred, e.g. A[C>T]G),
               gc_window (G+C fraction in +/- --gc_flank bp)

Inputs:
  --input    TSV with variant_id and TRANSCRIPT_ID (e.g. selected_variants.tsv)
  --fasta    GRCh38 FASTA, uncompressed, with a .fai index (built here if missing)
  --gtf      GTF (.gz ok) with exon and CDS features: MANE GTF, GENCODE or Ensembl
  --csc      optional codon stability coefficient table (CSV/TSV): a 'codon' column plus one or more
             numeric CSC columns (e.g. Wu et al. 2019 eLife fig1-data2: 293T_endo ... K562_SLAM), or
             a headerless codon <tab> value file; delta_CSC_<column> for each (--csc_cols to choose;
             count columns such as genome / transcriptome are skipped)
  --hexamer_table  optional TSV: hexamer <tab> score (e.g. ESRseq; any header)
Chromosome names are matched across chr1 / 1 / NC_000001.11 styles.

Example:
    python seq_features.py --input VUS_alphagenome/selected_variants.tsv \
        --fasta GRCh38.fa --gtf MANE.GRCh38.v1.4.ensembl_genomic.gtf.gz \
        [--csc csc_human.tsv] [--hexamer_table ESRseq.tsv] -o VUS_seq_features.tsv
"""
import argparse
import gzip
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
BASES = "TCAG"
# standard code, codons ordered TTT, TTC, TTA, ... (first, second, third base from TCAG)
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = {a + b + c: AA[16 * i + 4 * j + k]
               for i, a in enumerate(BASES) for j, b in enumerate(BASES) for k, c in enumerate(BASES)}
TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}


def revcomp(s):
    return s.translate(COMP)[::-1]


def norm_chrom(c):
    """'chr1', '1', 'NC_000001.11' -> '1'; X/Y/MT likewise."""
    c = str(c)
    m = re.match(r"NC_0*(\d+)\.\d+$", c)
    if m:
        c = {"23": "X", "24": "Y", "12920": "MT"}.get(m.group(1), m.group(1))
    c = c[3:] if c.lower().startswith("chr") else c
    return "MT" if c == "M" else c


# ----------------------------------------------------------------------------- FASTA
class Fasta:
    """Random access to an uncompressed FASTA through its .fai index (built if missing)."""

    def __init__(self, path):
        if path.endswith(".gz"):
            sys.exit("--fasta must be uncompressed (or bgzip-free); decompress it or pass the plain .fa")
        self.path = path
        fai = path + ".fai"
        if not os.path.exists(fai):
            print(f"Building {fai} ...", flush=True)
            self._build_fai(fai)
        self.idx = {}
        with open(fai) as fh:
            for line in fh:
                name, length, offset, lb, lw = line.split("\t")[:5]
                self.idx[norm_chrom(name)] = (int(length), int(offset), int(lb), int(lw))
        self.fh = open(path, "rb")

    def _build_fai(self, fai):
        out, name, length, offset, lb, lw, pos = [], None, 0, 0, 0, 0, 0
        with open(self.path, "rb") as fh:
            for line in fh:
                if line.startswith(b">"):
                    if name is not None:
                        out.append(f"{name}\t{length}\t{offset}\t{lb}\t{lw}\n")
                    name = line[1:].split()[0].decode()
                    length, lb, lw, offset = 0, 0, 0, pos + len(line)
                else:
                    if lb == 0:
                        lb, lw = len(line.rstrip(b"\r\n")), len(line)
                    length += len(line.rstrip(b"\r\n"))
                pos += len(line)
            if name is not None:
                out.append(f"{name}\t{length}\t{offset}\t{lb}\t{lw}\n")
        with open(fai, "w") as fh:
            fh.writelines(out)

    def fetch(self, chrom, start, end):
        """1-based inclusive genomic sequence (upper case); clipped at chromosome ends."""
        c = norm_chrom(chrom)
        if c not in self.idx:
            raise KeyError(f"chromosome {chrom} not in FASTA")
        length, offset, lb, lw = self.idx[c]
        start, end = max(1, start), min(length, end)
        if end < start:
            return ""
        s0, e0 = start - 1, end
        b0 = offset + (s0 // lb) * lw + s0 % lb
        b1 = offset + (e0 // lb) * lw + e0 % lb
        self.fh.seek(b0)
        return self.fh.read(b1 - b0).decode().replace("\n", "").replace("\r", "").upper()


# ----------------------------------------------------------------------------- GTF
def read_gtf(path, wanted=None):
    """{tx: {'chrom','strand','exons':[(s,e)],'cds':[(s,e)],'mane':bool}} for wanted (unversioned) ids,
    or all transcripts if wanted is None."""
    pat = re.compile(r'transcript_id "([^".]+)')
    txs = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t", 8)
            if len(f) < 9 or f[2] not in ("exon", "CDS"):
                continue
            m = pat.search(f[8])
            if not m or (wanted is not None and m.group(1) not in wanted):
                continue
            t = txs.setdefault(m.group(1), {"chrom": norm_chrom(f[0]), "strand": f[6], "exons": [], "cds": [],
                                            "mane": False})
            t["exons" if f[2] == "exon" else "cds"].append((int(f[3]), int(f[4])))
            if "MANE_Select" in f[8]:
                t["mane"] = True
    for t in txs.values():
        rev = t["strand"] == "-"
        t["exons"] = sorted(set(t["exons"]), reverse=rev)      # transcript order
        t["cds"] = sorted(set(t["cds"]), reverse=rev)
    return txs


def cds_sequence(fa, t):
    parts = [fa.fetch(t["chrom"], s, e) for s, e in t["cds"]]
    if t["strand"] == "-":
        parts = [revcomp(p) for p in parts]
    return "".join(parts)


def cds_index(t, pos):
    """0-based CDS index of a genomic position, or None if not in the CDS."""
    cum = 0
    for s, e in t["cds"]:
        if s <= pos <= e:
            return cum + (pos - s if t["strand"] == "+" else e - pos)
        cum += e - s + 1
    return None


def exon_geometry(t, pos):
    ex, n, plus = t["exons"], len(t["exons"]), t["strand"] == "+"
    for k, (s, e) in enumerate(ex):
        if s <= pos <= e:
            d5 = (pos - s + 1) if plus else (e - pos + 1)     # from the exon's 5' end (acceptor side)
            d3 = (e - pos + 1) if plus else (pos - s + 1)     # to the exon's 3' end (donor side)
            d5 = d5 if k > 0 else np.inf
            d3 = d3 if k < n - 1 else np.inf
            d = min(d5, d3)
            side = None if not np.isfinite(d) else ("acceptor" if d5 <= d3 else "donor")
            cds_exons = [i for i, (a, b) in enumerate(ex) if any(a <= cs and ce <= b for cs, ce in t["cds"])]
            return {"exon_number": k + 1, "n_exons": n, "exon_length": e - s + 1, "junction_dist": d,
                    "nearest_junction": side,
                    "first_coding_exon": bool(cds_exons) and k == cds_exons[0],
                    "last_coding_exon": bool(cds_exons) and k == cds_exons[-1]}
    return {"exon_number": np.nan}


# ----------------------------------------------------------------------------- codon tables
def codon_usage(fa, txs):
    counts = defaultdict(int)
    n_tx = 0
    for t in txs.values():
        if not t["cds"]:
            continue
        try:
            seq = cds_sequence(fa, t)
        except KeyError:
            continue
        if len(seq) % 3:
            continue
        n_tx += 1
        for i in range(0, len(seq) - 2, 3):
            c = seq[i:i + 3]
            if c in CODON_TABLE:
                counts[c] += 1
    total = sum(counts.values())
    per_k = {c: 1000 * counts[c] / total for c in CODON_TABLE} if total else {}
    rscu = {}
    by_aa = defaultdict(list)
    for c, a in CODON_TABLE.items():
        by_aa[a].append(c)
    for a, cs in by_aa.items():
        mean = np.mean([counts[c] for c in cs])
        for c in cs:
            rscu[c] = counts[c] / mean if mean else np.nan
    return per_k, rscu, n_tx


def read_table(path, upper=True):
    """Two-column table (key, numeric value); header and extra columns ignored."""
    d = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            f = re.split(r"[\t,\s]+", line.strip())
            if len(f) < 2:
                continue
            try:
                v = float(f[1])
            except ValueError:
                continue                                       # header line
            k = f[0].upper().replace("U", "T") if upper else f[0]
            d[k] = v
    return d


def read_csc(path, cols=None):
    """{column: {codon: value}} from a CSC table."""
    sep = "," if path.endswith((".csv", ".csv.gz")) else "\t"
    df = pd.read_csv(path, sep=sep)
    key = next((c for c in df.columns if str(c).lower() == "codon"), None)
    if key is None:                                            # headerless two-column file
        return {"CSC": read_table(path)}
    df[key] = df[key].astype(str).str.upper().str.replace("U", "T")
    num = [c for c in df.columns if c != key and pd.api.types.is_numeric_dtype(df[c])
           and str(c).lower() not in ("genome", "transcriptome")]
    if cols:
        if missing := [c for c in cols if c not in df.columns]:
            sys.exit(f"--csc_cols {missing} not in {list(df.columns)}")
        num = cols
    return {c: dict(zip(df[key], df[c])) for c in num}


def hexamer_scores(ctx, table):
    """ctx = 11-mer with the variant base at index 5; sum and mean over the six covering hexamers."""
    v = [table.get(ctx[i:i + 6]) for i in range(6)]
    v = [x for x in v if x is not None]
    return (sum(v), np.mean(v), len(v)) if v else (np.nan, np.nan, 0)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--csc", default=None, help="codon stability coefficient table (see above)")
    ap.add_argument("--csc_cols", nargs="+", default=None, help="CSC columns to use (default: all)")
    ap.add_argument("--hexamer_table", default=None, help="hexamer <tab> score (e.g. ESRseq)")
    ap.add_argument("--gc_flank", type=int, default=50)
    ap.add_argument("--tx_col", default="TRANSCRIPT_ID")
    ap.add_argument("-o", "--output", default=None, help="default: <input>.seqfeat.tsv")
    a = ap.parse_args()

    df = pd.read_csv(a.input, sep="\t", dtype=str)
    if "variant_id" not in df or a.tx_col not in df:
        sys.exit(f"--input needs variant_id and {a.tx_col}")
    df = df.drop_duplicates("variant_id")
    tx_ids = df[a.tx_col].astype(str).str.split(".").str[0]

    fa = Fasta(a.fasta)
    print("Reading GTF ...", flush=True)
    all_tx = read_gtf(a.gtf)
    coding = {k: v for k, v in all_tx.items() if v["cds"]}
    mane = {k: v for k, v in coding.items() if v["mane"]}
    usage_set = mane or coding
    print(f"  {len(coding):,} coding transcripts ({len(mane):,} tagged MANE_Select); codon usage from "
          f"{'MANE Select' if mane else 'all coding'} transcripts", flush=True)
    per_k, rscu, n_tx = codon_usage(fa, usage_set)
    print(f"  codon usage from {n_tx:,} CDS", flush=True)
    csc = read_csc(a.csc, a.csc_cols) if a.csc else None
    if csc:
        print(f"  CSC columns: {list(csc)}")
    hexa = read_table(a.hexamer_table) if a.hexamer_table else None
    if hexa is not None:
        print(f"  hexamer table: {len(hexa):,} entries")

    rows, cds_cache = [], {}
    for vid, tx in zip(df["variant_id"], tx_ids):
        r = {"variant_id": vid}
        try:
            chrom, pos, al = vid.split(":")
            ref, alt = al.split(">")
            pos = int(pos)
        except ValueError:
            r["status"] = "bad_variant_id"
            rows.append(r)
            continue
        t = all_tx.get(tx)
        if t is None or not t["cds"]:
            r["status"] = "transcript_not_in_gtf"
            rows.append(r)
            continue
        if norm_chrom(chrom) != t["chrom"]:
            r["status"] = "chromosome_mismatch"
            rows.append(r)
            continue
        plus = t["strand"] == "+"

        # genomic context (+ strand)
        flank = fa.fetch(chrom, pos - 1, pos + 1)
        r["genome_ref"] = flank[1] if len(flank) == 3 else ""
        r["ref_matches_genome"] = r["genome_ref"] == ref
        r["transition"] = (ref, alt) in TRANSITIONS
        r["cpg"] = (ref == "C" and flank[2:3] == "G") or (ref == "G" and flank[0:1] == "C")
        tri, ra = (flank, (ref, alt)) if ref in "CT" else (revcomp(flank), (ref.translate(COMP), alt.translate(COMP)))
        r["trinucleotide"] = f"{tri[0]}[{ra[0]}>{ra[1]}]{tri[2]}" if len(tri) == 3 else ""
        w = fa.fetch(chrom, pos - a.gc_flank, pos + a.gc_flank)
        r["gc_window"] = (w.count("G") + w.count("C")) / len(w) if w else np.nan

        # exon geometry
        r.update(exon_geometry(t, pos))

        # codon
        if tx not in cds_cache:
            cds_cache[tx] = cds_sequence(fa, t)
        cds = cds_cache[tx]
        i = cds_index(t, pos)
        if i is None:
            r["status"] = "not_in_cds"
            rows.append(r)
            continue
        tref = ref if plus else ref.translate(COMP)
        talt = alt if plus else alt.translate(COMP)
        r["cds_pos"] = i + 1
        r["ref_matches"] = cds[i] == tref
        k, cp = divmod(i, 3)
        r["codon_position"] = cp + 1
        cref = cds[3 * k: 3 * k + 3]
        calt = cref[:cp] + talt + cref[cp + 1:] if len(cref) == 3 else ""
        r["codon_ref"], r["codon_alt"] = cref, calt
        aref, aalt = CODON_TABLE.get(cref, "?"), CODON_TABLE.get(calt, "?")
        r["aa_ref"], r["aa_alt"] = aref, aalt
        r["consequence"] = ("synonymous" if aref == aalt else "stop_gained" if aalt == "*"
                            else "stop_lost" if aref == "*" else "start_lost" if k == 0 and aref == "M"
                            else "missense" if "?" not in (aref, aalt) else "unknown")
        r["usage_ref"], r["usage_alt"] = per_k.get(cref, np.nan), per_k.get(calt, np.nan)
        r["delta_log2_usage"] = (np.log2(r["usage_alt"] / r["usage_ref"])
                                 if r["usage_ref"] and r["usage_alt"] else np.nan)
        r["rscu_ref"], r["rscu_alt"] = rscu.get(cref, np.nan), rscu.get(calt, np.nan)
        r["delta_rscu"] = r["rscu_alt"] - r["rscu_ref"]
        if csc:
            for col, tab in csc.items():
                r[f"delta_CSC_{col}"] = tab.get(calt, np.nan) - tab.get(cref, np.nan)

        # hexamer motifs on the transcript-sense strand
        if hexa is not None:
            ctx = fa.fetch(chrom, pos - 5, pos + 5)
            if len(ctx) == 11:
                ctx = ctx if plus else revcomp(ctx)
                ctx_alt = ctx[:5] + talt + ctx[6:]
                rs, rm, n = hexamer_scores(ctx, hexa)
                as_, am, _ = hexamer_scores(ctx_alt, hexa)
                r.update(hex_ref_sum=rs, hex_alt_sum=as_, delta_hex_sum=as_ - rs,
                         delta_hex_mean=am - rm, hex_n=n)
        r["status"] = "ok" if r["ref_matches"] else "ref_mismatch"
        rows.append(r)

    out = pd.DataFrame(rows)
    dest = a.output or re.sub(r"\.tsv(\.gz)?$", "", a.input) + ".seqfeat.tsv"
    out.to_csv(dest, sep="\t", index=False)
    print(f"\nStatus: {out['status'].value_counts().to_dict()}")
    if "consequence" in out:
        print(f"Consequence on the transcript: {out['consequence'].value_counts().to_dict()}")
    print(f"Wrote {dest} ({len(out):,} variants)")


if __name__ == "__main__":
    main()
