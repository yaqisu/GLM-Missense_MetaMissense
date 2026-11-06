#!/usr/bin/env python3
"""
Extract missense SNVs with a measured RNA score from SGE datasets and harmonize
them into one GRCh38 table. Each dataset is read only when its option is given;
the manuscript uses RAD51D only (--rad51d).

Datasets
--------
  Findlay2018_BRCA1   Findlay et al. 2018 Nature           (hg19 -> lifted to GRCh38)
  Woo2025_BARD1       Woo et al. 2025 medRxiv (Starita lab)
  Casadei2026_RAD51D  Casadei et al. 2026 bioRxiv (Starita lab)
  Casadei2026_XRCC2   Casadei et al. 2026 bioRxiv (Starita lab)

RNA score thresholds (log2 RNA/gDNA)
------------------------------------
  RAD51D  -2.7856  low if < mean - 3 SD of synonymous RNA scores
  XRCC2   -0.6603  low if < mean - 3 SD of synonymous RNA scores
  BARD1   -1.244   normal if > mean + 1 SD of nonsense RNA scores
                   (bottom 97.5%, excluding exons 1, 2, 11)
  BRCA1   -2.0     (default; --findlay-rna-threshold) 4-fold depletion used for
                   "depleted in RNA" in Figs 4-5. Supp. Note 2 uses > -1 to define
                   variants unlikely to act through splicing. Findlay et al. did not
                   report a per-variant RNA class, so RNA_CONSEQUENCE is NA for BRCA1.

Fitness/function scores are on different scales across studies (Findlay: log2
ratio normalized to synonymous/nonsense medians; Starita lab: log2 fold change
per day), so compare FUNC_SCORE only within a dataset. FUNC_CLASS is harmonized
to normal / abnormal / indeterminate; FUNC_CONSEQUENCE keeps the source label.

Requirements: pandas (+ openpyxl and pyliftover for --findlay; pyfaidx for --ref-fasta)

Example
-------
  # RAD51D only (manuscript)
  python extract_sge_missense_rna.py --rad51d Starita_2026_RAD51D.csv \
      --out RAD51D_missense_rna.tsv --ref-fasta GRCh38.fa
  # all four datasets
  python extract_sge_missense_rna.py \
      --findlay Findlay_BRCA1_source.xlsx \
      --bard1 Starita_2025_BARD1_score.csv \
      --rad51d Starita_2026_RAD51D.csv \
      --xrcc2 Starita_2026_XRCC2.csv \
      --out sge_missense_rna.tsv \
      --chain hg19ToHg38.over.chain.gz \
      --ref-fasta GRCh38.fa
"""

import argparse
import sys

import numpy as np
import pandas as pd

COMP = str.maketrans("ACGT", "TGCA")

# Gene-level metadata. Transcripts are the ones each study annotated against.
GENE_INFO = {
    "BRCA1":  {"gene_id": "ENSG00000012048", "strand": "-"},
    "BARD1":  {"gene_id": "ENSG00000138376", "strand": "-", "transcript": "ENST00000260947.9"},
    "RAD51D": {"gene_id": "ENSG00000185379", "strand": "-"},
    "XRCC2":  {"gene_id": "ENSG00000196584", "strand": "-"},
}

STARITA_DATASETS = {
    # key: (gene, study, RNA threshold, threshold rule)
    "bard1": ("BARD1", "Woo2025_BARD1", -1.244,
              "normal if RNA score > mean + 1 SD of nonsense RNA scores "
              "(bottom 97.5%, excl. exons 1,2,11); Woo et al. 2025"),
    "rad51d": ("RAD51D", "Casadei2026_RAD51D", -2.7856,
               "low if RNA score < mean - 3 SD of synonymous RNA scores; Casadei et al. 2026"),
    "xrcc2": ("XRCC2", "Casadei2026_XRCC2", -0.6603,
              "low if RNA score < mean - 3 SD of synonymous RNA scores; Casadei et al. 2026"),
}

FUNC_CLASS_MAP = {
    "functionally_normal": "normal",
    "functionally_abnormal": "abnormal",
    "indeterminate": "indeterminate",
    "FUNC": "normal",
    "LOF": "abnormal",
    "INT": "indeterminate",
}

OUT_COLS = [
    "DATASET", "GENE", "GENE_ID", "TRANSCRIPT_ID",
    "CHR", "POS", "REF_ALLELE", "ALT_ALLELE", "GENOME_BUILD",
    "SOURCE_BUILD", "SOURCE_POS",
    "HGVS_C", "PROTEIN_CHANGE",
    "RNA_SCORE", "RNA_CONSEQUENCE", "RNA_THRESHOLD", "RNA_THRESHOLD_RULE",
    "FUNC_SCORE", "FUNC_CONSEQUENCE", "FUNC_CLASS",
]


def check_rna_labels(df, threshold, name):
    """Warn if reported RNA classes disagree with the stated threshold."""
    low_bad = ((df["RNA_CONSEQUENCE"] == "low") & (df["RNA_SCORE"] >= threshold)).sum()
    norm_bad = ((df["RNA_CONSEQUENCE"] == "normal") & (df["RNA_SCORE"] < threshold)).sum()
    if low_bad or norm_bad:
        print(f"[WARN] {name}: {low_bad} 'low' >= threshold, "
              f"{norm_bad} 'normal' < threshold ({threshold})", file=sys.stderr)


def load_starita(path, key):
    gene, dataset, thr, rule = STARITA_DATASETS[key]
    df = pd.read_csv(path, low_memory=False)
    df = df[(df["consequence"] == "missense_variant")
            & (df["var_type"] == "snv")
            & df["RNAscore"].notna()].copy()

    if "hgvs_c" in df.columns and df["hgvs_c"].notna().all():
        hgvs_c = df["hgvs_c"]
        transcript = hgvs_c.str.split(":").str[0]
    else:
        # BARD1 file has no hgvs_c: rebuild it from CDS position on the MANE transcript
        transcript = pd.Series(GENE_INFO[gene]["transcript"], index=df.index)
        ref, alt = df["ref"], df["alt"]
        if GENE_INFO[gene]["strand"] == "-":
            ref, alt = ref.str.translate(COMP), alt.str.translate(COMP)
        hgvs_c = transcript + ":c." + df["CDS_position"].astype(str) + ref + ">" + alt

    out = pd.DataFrame({
        "DATASET": dataset,
        "GENE": gene,
        "GENE_ID": GENE_INFO[gene]["gene_id"],
        "TRANSCRIPT_ID": transcript,
        "CHR": df["chrom"].astype(str),
        "POS": df["pos"].astype(int),
        "REF_ALLELE": df["ref"],
        "ALT_ALLELE": df["alt"],
        "GENOME_BUILD": "GRCh38",
        "SOURCE_BUILD": "GRCh38",
        "SOURCE_POS": df["pos"].astype(int),
        "HGVS_C": hgvs_c,
        "PROTEIN_CHANGE": df["amino_acid_change"],
        "RNA_SCORE": df["RNAscore"].astype(float),
        "RNA_CONSEQUENCE": df["RNA_consequence"],
        "RNA_THRESHOLD": thr,
        "RNA_THRESHOLD_RULE": rule,
        "FUNC_SCORE": df["score"].astype(float),
        "FUNC_CONSEQUENCE": df["functional_consequence"],
    })
    check_rna_labels(out, thr, dataset)
    return out


def get_liftover(chain):
    try:
        from pyliftover import LiftOver
    except ImportError:
        sys.exit("pyliftover is required for the BRCA1 hg19 -> GRCh38 liftover "
                 "(pip install pyliftover)")
    try:
        return LiftOver(chain) if chain else LiftOver("hg19", "hg38")
    except Exception as e:
        sys.exit(f"Could not load hg19->hg38 chain ({e}). "
                 "Download hg19ToHg38.over.chain.gz from UCSC and pass --chain.")


def load_findlay(path, lifter, rna_thr, keep_start_codon):
    df = pd.read_excel(path, header=2, na_values=["NA"])
    df = df[(df["consequence"] == "Missense") & df["mean.rna.score"].notna()].copy()
    if not keep_start_codon:
        # Findlay labels p.M1 changes as Missense; VEP (used by the other datasets)
        # calls them start_lost, so drop them for consistency.
        df = df[df["aa_pos"] != 1]

    chrom19 = "chr" + df["chromosome"].astype(str)
    pos19 = df["position (hg19)"].astype(int)
    ref, alt = df["reference"].astype(str), df["alt"].astype(str)

    pos38, chr38, keep = [], [], []
    for i, (c, p, r, a) in enumerate(zip(chrom19, pos19, ref, alt)):
        hits = lifter.convert_coordinate(c, p - 1)  # pyliftover is 0-based
        if not hits or len(hits) != 1:
            print(f"[WARN] BRCA1 {c}:{p} {r}>{a} failed liftover "
                  f"({0 if not hits else len(hits)} hits); dropped", file=sys.stderr)
            keep.append(False); pos38.append(np.nan); chr38.append(None)
            continue
        hc, hp, hs, _ = hits[0]
        if hs != "+":
            print(f"[WARN] BRCA1 {c}:{p} lifted to minus strand; dropped", file=sys.stderr)
            keep.append(False); pos38.append(np.nan); chr38.append(None)
            continue
        keep.append(True); pos38.append(hp + 1); chr38.append(hc)

    keep = np.array(keep)
    df, chrom19, pos19 = df[keep], chrom19[keep], pos19[keep]
    pos38 = pd.Series(np.array(pos38)[keep].astype(int), index=df.index)
    chr38 = pd.Series(np.array(chr38, dtype=object)[keep], index=df.index)

    offsets = sorted(int(o) for o in (pos38 - pos19).unique())
    print(f"[INFO] BRCA1 hg19->GRCh38 position offsets: {offsets}", file=sys.stderr)

    rule = (f"{rna_thr} (log2), user-set; Findlay 2018 used -2 (4-fold) for 'depleted in RNA' "
            "(Figs 4-5) and > -1 for 'unlikely to act via splicing' (Supp. Note 2); "
            "no per-variant RNA class reported")
    aa_pos = df["aa_pos"].astype(int).astype(str)

    return pd.DataFrame({
        "DATASET": "Findlay2018_BRCA1",
        "GENE": "BRCA1",
        "GENE_ID": GENE_INFO["BRCA1"]["gene_id"],
        "TRANSCRIPT_ID": df["transcript_ID"],
        "CHR": chr38,
        "POS": pos38,
        "REF_ALLELE": df["reference"],
        "ALT_ALLELE": df["alt"],
        "GENOME_BUILD": "GRCh38",
        "SOURCE_BUILD": "GRCh37",
        "SOURCE_POS": pos19,
        "HGVS_C": df["transcript_ID"] + ":" + df["transcript_variant"],
        "PROTEIN_CHANGE": df["aa_ref"] + aa_pos + df["aa_alt"],
        "RNA_SCORE": df["mean.rna.score"].astype(float),
        "RNA_CONSEQUENCE": np.nan,
        "RNA_THRESHOLD": rna_thr,
        "RNA_THRESHOLD_RULE": rule,
        "FUNC_SCORE": df["function.score.mean"].astype(float),
        "FUNC_CONSEQUENCE": df["func.class"],
    })


def check_ref_fasta(df, fasta_path):
    try:
        from pyfaidx import Fasta
    except ImportError:
        sys.exit("pyfaidx is required for --ref-fasta (pip install pyfaidx)")
    fa = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
    names = set(fa.keys())

    def contig(c):
        if c in names:
            return c
        alt_name = c[3:] if c.startswith("chr") else "chr" + c
        return alt_name if alt_name in names else None

    bad = []
    for r in df.itertuples():
        c = contig(r.CHR)
        base = fa[c][r.POS - 1] if c else None
        if base != r.REF_ALLELE:
            bad.append((r.DATASET, r.CHR, r.POS, r.REF_ALLELE, base))
    if bad:
        print(f"[WARN] {len(bad)} variants whose REF does not match the FASTA, e.g. {bad[:5]}",
              file=sys.stderr)
    else:
        print(f"[INFO] REF alleles match {fasta_path} for all {len(df)} variants",
              file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--findlay", default=None, help="Findlay et al. 2018 BRCA1 source table (.xlsx)")
    ap.add_argument("--bard1", default=None, help="BARD1 SGE scores (.csv)")
    ap.add_argument("--rad51d", default=None, help="RAD51D SGE scores (.csv), Casadei et al. 2026")
    ap.add_argument("--xrcc2", default=None, help="XRCC2 SGE scores (.csv)")
    ap.add_argument("--out", default="sge_missense_rna.tsv",
                    help="output path; .csv writes comma-separated, anything else TSV")
    ap.add_argument("--chain", default=None,
                    help="local hg19ToHg38 chain file; if omitted pyliftover downloads it from UCSC")
    ap.add_argument("--findlay-rna-threshold", type=float, default=-2.0)
    ap.add_argument("--keep-findlay-start-codon", action="store_true",
                    help="keep BRCA1 p.M1 variants (labelled Missense by Findlay)")
    ap.add_argument("--ref-fasta", default=None,
                    help="optional GRCh38 FASTA to verify REF alleles")
    args = ap.parse_args()

    frames = [load_starita(getattr(args, k), k) for k in ("bard1", "rad51d", "xrcc2")
              if getattr(args, k)]
    if args.findlay:
        lifter = get_liftover(args.chain)
        frames.append(load_findlay(args.findlay, lifter, args.findlay_rna_threshold,
                                   args.keep_findlay_start_codon))
    if not frames:
        ap.error("give at least one dataset (--rad51d, --bard1, --xrcc2, --findlay)")

    out = pd.concat(frames, ignore_index=True)
    out["FUNC_CLASS"] = out["FUNC_CONSEQUENCE"].map(FUNC_CLASS_MAP)
    unmapped = out.loc[out["FUNC_CLASS"].isna(), "FUNC_CONSEQUENCE"].unique()
    if len(unmapped):
        print(f"[WARN] unmapped functional labels: {unmapped}", file=sys.stderr)

    key = ["CHR", "POS", "REF_ALLELE", "ALT_ALLELE"]
    n_dup = out.duplicated(key).sum()
    if n_dup:
        print(f"[WARN] {n_dup} duplicated CHR/POS/REF/ALT keys", file=sys.stderr)

    if args.ref_fasta:
        check_ref_fasta(out, args.ref_fasta)

    out = out[OUT_COLS].sort_values(["GENE", "POS", "ALT_ALLELE"]).reset_index(drop=True)
    sep = "," if args.out.endswith(".csv") else "\t"
    out.to_csv(args.out, sep=sep, index=False, na_rep="NA")

    summary = out.groupby("DATASET").agg(
        n_missense=("POS", "size"),
        rna_low=("RNA_CONSEQUENCE", lambda s: (s == "low").sum()),
        func_abnormal=("FUNC_CLASS", lambda s: (s == "abnormal").sum()),
    )
    print(summary.to_string(), file=sys.stderr)
    print(f"[INFO] wrote {len(out)} variants to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
