#!/usr/bin/env python3
"""
Annotate variants with exon number and distance to exon-exon junctions
using an Ensembl GTF (GRCh38).

Distance convention: nucleotides along the mature transcript, where the exonic
base immediately adjacent to a boundary has distance 1 (no zero).

Added columns (prefix gtf_):
  gtf_region                 exon / intron / outside / transcript_not_found
  gtf_exon_number            exon containing the variant (transcript order, 5'->3')
  gtf_n_exons                total exons in the transcript
  gtf_intron_number          intron containing the variant (if intronic)
  gtf_intron_offset          HGVS-style offset to nearest exon (+k donor side, -k acceptor side)
  gtf_exon_length            length of the containing exon
  gtf_dist_exon_5p           distance to the 5' end of the containing exon
  gtf_dist_exon_3p           distance to the 3' end of the containing exon
  gtf_dist_upstream_EEJ      distance to the upstream exon-exon junction (NA in first exon)
  gtf_dist_downstream_EEJ    distance to the downstream exon-exon junction (NA in last exon)
  gtf_dist_nearest_EEJ       min of the two above
  gtf_nearest_EEJ_side       upstream / downstream
  gtf_tx_pos                 position in the mature transcript (cDNA, 1-based)
  gtf_dist_last_EEJ          signed distance to the last exon-exon junction
                             (>0 upstream of it, <0 in the last exon)
  gtf_c_pos                  c. position computed from GTF CDS (CDS/3' region only)
  gtf_c_pos_match            computed c. position equals the one parsed from HGVSc

Usage:
  python annotate_exon_junction.py -i fig10_resolved.tsv \
      -g Homo_sapiens.GRCh38.115.gtf.gz -o fig10_resolved_exon.tsv
"""
import argparse
import gzip
import re
import sys

import pandas as pd

TX_RE = re.compile(r'transcript_id "([^"]+)"')
TXV_RE = re.compile(r'transcript_version "([^"]+)"')
HGVSC_RE = re.compile(r'c\.(\d+)([+-]\d+)?')

OUT_COLS = [
    "gtf_region", "gtf_exon_number", "gtf_n_exons", "gtf_intron_number",
    "gtf_intron_offset", "gtf_exon_length", "gtf_dist_exon_5p", "gtf_dist_exon_3p",
    "gtf_dist_upstream_EEJ", "gtf_dist_downstream_EEJ", "gtf_dist_nearest_EEJ",
    "gtf_nearest_EEJ_side", "gtf_tx_pos", "gtf_dist_last_EEJ", "gtf_c_pos",
    "gtf_c_pos_match",
]


def norm_chrom(c):
    c = str(c).strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    return "MT" if c == "M" else c


def strip_ver(t):
    return str(t).strip().split(".")[0]


def read_gtf(path, wanted):
    """Collect exon and CDS intervals for the wanted (version-stripped) transcripts."""
    opener = gzip.open if path.endswith(".gz") else open
    tx = {}
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in ("exon", "CDS"):
                continue
            m = TX_RE.search(f[8])
            if not m:
                continue
            tid = strip_ver(m.group(1))
            if tid not in wanted:
                continue
            if tid not in tx:
                mv = TXV_RE.search(f[8])
                ver = mv.group(1) if mv else (m.group(1).split(".")[1] if "." in m.group(1) else "")
                tx[tid] = {"chrom": norm_chrom(f[0]), "strand": f[6], "version": ver,
                           "exons": [], "cds": []}
            key = "exons" if f[2] == "exon" else "cds"
            tx[tid][key].append((int(f[3]), int(f[4])))

    for t in tx.values():
        rev = t["strand"] == "-"
        t["exons"].sort(key=lambda x: x[0], reverse=rev)   # transcript order
        t["lens"] = [e - s + 1 for s, e in t["exons"]]
        cum = [0]
        for L in t["lens"]:
            cum.append(cum[-1] + L)
        t["cum"] = cum
        t["cds_start_tx"] = None
        if t["cds"]:
            g = max(e for _, e in t["cds"]) if rev else min(s for s, _ in t["cds"])
            t["cds_start_tx"] = tx_position(g, t)
    return tx


def tx_position(pos, t):
    """1-based mature-transcript position of a genomic position, or None if not exonic."""
    for i, (s, e) in enumerate(t["exons"]):
        if s <= pos <= e:
            d5 = (pos - s) if t["strand"] == "+" else (e - pos)
            return t["cum"][i] + d5 + 1
    return None


def annotate(pos, t):
    out = dict.fromkeys(OUT_COLS, pd.NA)
    exons, lens, cum, strand = t["exons"], t["lens"], t["cum"], t["strand"]
    n = len(exons)
    out["gtf_n_exons"] = n

    for i, (s, e) in enumerate(exons):
        if not (s <= pos <= e):
            continue
        d5 = (pos - s) if strand == "+" else (e - pos)      # 0-based offset from exon 5' end
        dist5 = d5 + 1
        dist3 = lens[i] - d5
        tx_pos = cum[i] + d5 + 1
        out.update(gtf_region="exon", gtf_exon_number=i + 1, gtf_exon_length=lens[i],
                   gtf_dist_exon_5p=dist5, gtf_dist_exon_3p=dist3, gtf_tx_pos=tx_pos)
        cand = []
        if i > 0:
            out["gtf_dist_upstream_EEJ"] = dist5
            cand.append((dist5, "upstream"))
        if i < n - 1:
            out["gtf_dist_downstream_EEJ"] = dist3
            cand.append((dist3, "downstream"))
        if cand:
            d, side = min(cand)
            out["gtf_dist_nearest_EEJ"], out["gtf_nearest_EEJ_side"] = d, side
            last_j = cum[n - 1]        # tx position of last base of penultimate exon
            out["gtf_dist_last_EEJ"] = (last_j - tx_pos + 1) if tx_pos <= last_j else -(tx_pos - last_j)
        if t["cds_start_tx"] is not None and tx_pos >= t["cds_start_tx"]:
            out["gtf_c_pos"] = tx_pos - t["cds_start_tx"] + 1
        return out

    for i in range(n - 1):
        (s1, e1), (s2, e2) = exons[i], exons[i + 1]
        if strand == "+" and e1 < pos < s2:
            donor, acceptor = pos - e1, s2 - pos
        elif strand == "-" and e2 < pos < s1:
            donor, acceptor = s1 - pos, pos - e2
        else:
            continue
        out.update(gtf_region="intron", gtf_intron_number=i + 1,
                   gtf_intron_offset=f"+{donor}" if donor <= acceptor else f"-{acceptor}")
        return out

    out["gtf_region"] = "outside"
    return out


def parse_c_pos(h):
    m = HGVSC_RE.search(str(h))
    if not m or m.group(2):          # skip intronic-offset notation
        return None
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True, help="variant TSV")
    ap.add_argument("-g", "--gtf", required=True, help="Ensembl GTF (.gtf or .gtf.gz), GRCh38")
    ap.add_argument("-o", "--output", help="output TSV (default: <input>_exon.tsv)")
    ap.add_argument("--tx-col", default="src_TRANSCRIPT_ID")
    ap.add_argument("--chrom-col", default="src_chromosome")
    ap.add_argument("--pos-col", default="src_position")
    ap.add_argument("--hgvsc-col", default="src_HGVSc_VEP",
                    help="column with c. notation for the sanity check ('' to skip)")
    a = ap.parse_args()
    out_path = a.output or re.sub(r"\.(tsv|txt)$", "", a.input) + "_exon.tsv"

    df = pd.read_csv(a.input, sep="\t", dtype=str, keep_default_na=False)
    for c in (a.tx_col, a.chrom_col, a.pos_col):
        if c not in df.columns:
            sys.exit(f"Column not found: {c}")

    wanted = set(df[a.tx_col].map(strip_ver))
    print(f"Reading GTF for {len(wanted)} transcript(s)...", file=sys.stderr)
    tx = read_gtf(a.gtf, wanted)

    for tid in sorted(wanted):
        if tid not in tx:
            print(f"WARNING: {tid} not in GTF", file=sys.stderr)
    for full in sorted(set(df[a.tx_col])):
        tid, ver = strip_ver(full), (full.split(".")[1] if "." in full else "")
        if tid in tx and ver and tx[tid]["version"] and ver != tx[tid]["version"]:
            print(f"WARNING: version mismatch {full} vs GTF .{tx[tid]['version']}", file=sys.stderr)

    rows = []
    n_chrom_mismatch = 0
    for _, r in df.iterrows():
        t = tx.get(strip_ver(r[a.tx_col]))
        if t is None:
            o = dict.fromkeys(OUT_COLS, pd.NA)
            o["gtf_region"] = "transcript_not_found"
        else:
            if norm_chrom(r[a.chrom_col]) != t["chrom"]:
                n_chrom_mismatch += 1
            o = annotate(int(float(r[a.pos_col])), t)
            if a.hgvsc_col and a.hgvsc_col in df.columns and pd.notna(o["gtf_c_pos"]):
                parsed = parse_c_pos(r[a.hgvsc_col])
                if parsed is not None:
                    o["gtf_c_pos_match"] = parsed == o["gtf_c_pos"]
        rows.append(o)

    ann = pd.DataFrame(rows, columns=OUT_COLS, index=df.index)
    for c in OUT_COLS:
        if c not in ("gtf_region", "gtf_intron_offset", "gtf_nearest_EEJ_side", "gtf_c_pos_match"):
            ann[c] = ann[c].astype("Int64")
    out = pd.concat([df.drop(columns=[c for c in OUT_COLS if c in df.columns]), ann], axis=1)
    out.to_csv(out_path, sep="\t", index=False, na_rep="NA")

    print(f"Wrote {out_path}", file=sys.stderr)
    print(ann["gtf_region"].value_counts().to_string(), file=sys.stderr)
    if n_chrom_mismatch:
        print(f"WARNING: {n_chrom_mismatch} rows with chromosome mismatch vs GTF", file=sys.stderr)
    chk = ann["gtf_c_pos_match"].dropna()
    if len(chk):
        print(f"c. position check: {int(chk.sum())}/{len(chk)} match HGVSc "
              f"({'OK' if chk.all() else 'CHECK GTF BUILD/RELEASE'})", file=sys.stderr)


if __name__ == "__main__":
    main()
