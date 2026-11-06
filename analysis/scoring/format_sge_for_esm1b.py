#!/usr/bin/env python3
"""
Format an SGE / MAVE variant table for ESM1b_ref_alt_contrast.py (RefSeq gbff route)
=====================================================================================
Mirrors the ClinVar mode of map_variants_to_mane_protein.py, with the table's hgvs_c
in place of the ClinVar Name, so SGE variants are annotated exactly like the training
data:
    1. hgvs_c (e.g. ENST00000359321.2:c.620G>A) gives the transcript, the CDS position
       and the transcript-strand alleles.
    2. An Ensembl transcript is mapped to its MANE RefSeq partner (NM_) with the NCBI
       MANE summary (--mane_summary). MANE Ensembl/RefSeq pairs are sequence-identical,
       so the c. numbering carries over. NM_/XM_ ids in hgvs_c are used directly.
    3. CDS and protein (/translation) come from the RefSeq GenBank file(s) (--gbff),
       e.g. MANE.GRCh38.v1.4.refseq_rna.gbff.gz.
    4. Checks: c. alleles == table ref/alt or their complements (gives the strand);
       transcript base at c. == c. ref allele; codon translation == protein residue;
       computed change == hgvs_p (or amino_acid_change).
GenBank parsing and codon classification are imported from
map_variants_to_mane_protein.py (--mapper_script, default: next to this script).

Output: every input row and column (unchanged), plus
    variant_id, gene_name, transcript_id, protein_id, strand, cds_position, ref_codon,
    alt_codon, protein_position, ref_aa, alt_aa, protein_length, protein_sequence
    (filled only when status == ok), transcript_source, p_agrees, label, status

status values:
    ok / synonymous / stop_gained / stop_lost / start_lost / selenocysteine_codon /
        protein_mismatch      from the codon change, as for the training data
    invalid_allele            ref/alt not single A/C/G/T bases
    hgvs_c_unparsed           hgvs_c is not <transcript>:c.N X>Y
    noncoding_position        c.-N, c.*N or c.N+/-M (UTR / intronic)
    allele_mismatch           c. alleles are neither ref/alt nor their complements
    transcript_not_in_mane    ENST (version) not in the MANE summary
    transcript_not_in_gbff    RefSeq transcript (version) not in the --gbff file(s)
    cds_incomplete / cpos_out_of_range / transcript_ref_mismatch
    p_mismatch                missense, but the computed change disagrees with hgvs_p

label (optional, for evaluation) from --label_col (functional_consequence):
    contains "abnormal" -> 1, contains "normal" -> 0, otherwise empty

Usage:
    python format_sge_for_esm1b.py --input XRCC2_SGE.csv \\
        --gbff MANE.GRCh38.v1.4.refseq_rna.gbff.gz \\
        --mane_summary MANE.GRCh38.v1.4.summary.txt.gz
    # writes XRCC2_SGE.esm1b_input.tsv next to the input (or --output)
"""

import re
import sys
import argparse
import importlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


HGVS_C_RE = re.compile(
    r"^(?P<tx>(?:ENST\d+|[NX]M_\d+)(?:\.\d+)?)(?:\((?P<gene>[^)]*)\))?"
    r":c\.(?P<cpos>[-*]?\d+(?:[+-]\d+)?)(?P<cref>[ACGT])>(?P<calt>[ACGT])$")
P1_RE = re.compile(r"^([A-Z*])(\d+)([A-Z*=?])$")          # 1-letter, e.g. R207Q
ANNOT_COLS = ["gene_name", "transcript_id", "protein_id", "strand", "cds_position",
              "ref_codon", "alt_codon", "protein_position", "ref_aa", "alt_aa",
              "protein_length", "protein_sequence", "transcript_source", "p_agrees"]
NEW_COLS = ["variant_id"] + ANNOT_COLS + ["label", "status"]   # predict script drops these
INT_COLS = ["cds_position", "protein_position", "protein_length"]


def import_module_from(path):
    path = Path(path).resolve()
    if not path.exists():
        sys.exit(f"Cannot find {path}; point --mapper_script at map_variants_to_mane_protein.py")
    sys.path.insert(0, str(path.parent))
    return importlib.import_module(path.stem)


def guess_sep(path):
    suffixes = [s.lower() for s in Path(path).suffixes]
    return "\t" if (".tsv" in suffixes or ".txt" in suffixes) else ","


def read_mane_summary(path, strip_version):
    """Ensembl_nuc -> {nm, symbol}; exact versions plus an unversioned index."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.lstrip("#") for c in df.columns]
    if not {"Ensembl_nuc", "RefSeq_nuc"} <= set(df.columns):
        sys.exit(f"{path} lacks Ensembl_nuc / RefSeq_nuc columns (NCBI MANE summary expected)")
    exact, by_base = {}, defaultdict(list)
    for r in df.to_dict("records"):
        if r["Ensembl_nuc"] and r["RefSeq_nuc"]:
            info = {"nm": r["RefSeq_nuc"], "symbol": r.get("symbol", "")}
            exact[r["Ensembl_nuc"]] = info
            by_base[strip_version(r["Ensembl_nuc"])].append(info)
    print(f"Read {len(exact):,} Ensembl->RefSeq pairs from {path}")
    return exact, by_base


def expected_change(hgvs_p, aa_change, mvp):
    """(ref, pos, alt) from hgvs_p (3-letter) or amino_acid_change (1-letter); alt may be '?'."""
    for text in (hgvs_p, aa_change):
        t = str(text or "").strip().split(":")[-1].replace("%3D", "=").replace("%3d", "=")
        t = (t[2:] if t.startswith("p.") else t).strip("()")
        m = mvp.P_RE.match(t)
        if m:
            ref = mvp.THREE_TO_ONE.get(m["ref"])
            alt = {"=": ref, "?": "?"}.get(m["alt"]) or mvp.THREE_TO_ONE.get(m["alt"])
            if ref and alt:
                return ref, int(m["pos"]), alt
        m = P1_RE.match(t)
        if m:
            return m[1], int(m[2]), (m[1] if m[3] == "=" else m[3])
    return None


def annotate(ref, alt, hgvs_c, hgvs_p, aa_change, mane, gbff, versions, allow_other, mvp):
    ref, alt = str(ref).strip().upper(), str(alt).strip().upper()
    if not mvp.valid_snv(ref, alt):
        return {"status": "invalid_allele"}
    m = HGVS_C_RE.match(str(hgvs_c).strip())
    if not m:
        return {"status": "hgvs_c_unparsed"}
    if not m["cpos"].isdigit():
        return {"status": "noncoding_position"}
    c, cref, calt = int(m["cpos"]), m["cref"], m["calt"]
    out = {}
    if (cref, calt) == (ref, alt):
        out["strand"] = "+"
    elif (cref, calt) == (ref.translate(mvp.COMPLEMENT), alt.translate(mvp.COMPLEMENT)):
        out["strand"] = "-"
    else:
        return {**out, "status": "allele_mismatch"}

    # ---- transcript -> RefSeq record ----
    tx = m["tx"]
    if tx.startswith("ENST"):
        exact, by_base = mane
        info, source = exact.get(tx), "mane_refseq"
        if info is None and allow_other and by_base.get(mvp.strip_version(tx)):
            info, source = by_base[mvp.strip_version(tx)][0], "mane_refseq_other_ensembl_version"
        if info is None:
            return {**out, "transcript_id": tx, "status": "transcript_not_in_mane"}
        nm, out["gene_name"] = info["nm"], info["symbol"]
    else:
        nm, source = tx, "refseq"
    rec = gbff.get(nm)
    if rec is None and allow_other and versions.get(mvp.strip_version(nm)):
        nm = max(versions[mvp.strip_version(nm)], key=mvp.version_number)
        rec, source = gbff[nm], source + "_other_refseq_version"
    out.update(transcript_id=nm, transcript_source=source)
    if rec is None:
        return {**out, "status": "transcript_not_in_gbff"}
    out["protein_id"] = rec["protein_id"]
    if rec["gene"]:
        out["gene_name"] = rec["gene"]
    if not rec["complete"]:
        return {**out, "status": "cds_incomplete"}
    cds, protein = rec["cds"], rec["protein"]
    if c > len(cds):
        return {**out, "status": "cpos_out_of_range"}
    if cds[c - 1] != cref:
        return {**out, "status": "transcript_ref_mismatch"}

    # ---- codon change + p. cross-check (same logic as the training data) ----
    fields, status = mvp.classify_codon_change(cds, protein, c - 1, calt)
    out.update(fields)
    exp = expected_change(hgvs_p, aa_change, mvp)
    agrees = None
    if exp:
        agrees = (exp[1] == fields["protein_position"] and exp[0] == fields["ref_aa"]
                  and exp[2] in ("?", fields["alt_aa"]))
    out["p_agrees"] = agrees
    if status == "ok" and agrees is False:
        status = "p_mismatch"
    if status == "ok":
        out["protein_sequence"] = protein
    out["status"] = status
    return out


def main():
    p = argparse.ArgumentParser(
        description="Format an SGE/MAVE table for ESM1b_ref_alt_contrast.py via RefSeq gbff")
    p.add_argument("--input", required=True, help="SGE/MAVE CSV or TSV (optionally .gz)")
    p.add_argument("--gbff", nargs="+", required=True,
                   help="RefSeq GenBank RNA file(s), e.g. MANE.GRCh38.v1.4.refseq_rna.gbff.gz")
    p.add_argument("--mane_summary", default=None,
                   help="NCBI MANE summary (MANE.GRCh38.vX.summary.txt.gz); required when "
                        "hgvs_c uses ENST ids")
    p.add_argument("--allow_other_versions", action="store_true",
                   help="If the exact ENST / NM version is missing, use another version "
                        "(still fully checked)")
    p.add_argument("--output", default=None,
                   help="Output TSV (default: <input stem>.esm1b_input.tsv)")
    p.add_argument("--sep", default=None, help="Input delimiter (default: from extension)")
    p.add_argument("--mapper_script",
                   default=str(Path(__file__).resolve().parent / "map_variants_to_mane_protein.py"),
                   help="Path to map_variants_to_mane_protein.py (default: next to this script)")
    p.add_argument("--hgvs_c_col", default="hgvs_c")
    p.add_argument("--hgvs_p_col", default="hgvs_p")
    p.add_argument("--aa_change_col", default="amino_acid_change")
    p.add_argument("--consequence_col", default="consequence",
                   help="Only used for the status-vs-consequence summary")
    p.add_argument("--label_col", default="functional_consequence",
                   help="Column to derive a 0/1 label from; 'none' to skip")
    args = p.parse_args()

    mvp = import_module_from(args.mapper_script)

    in_path = Path(args.input)
    stem = in_path.name
    for suffix in (".gz", ".csv", ".tsv", ".txt"):
        if stem.lower().endswith(suffix):
            stem = stem[:-len(suffix)]
    out_path = Path(args.output) if args.output else in_path.with_name(stem + ".esm1b_input.tsv")

    sep = args.sep or guess_sep(in_path)
    df = pd.read_csv(in_path, sep=sep, dtype=str, keep_default_na=False)
    # pandas renames duplicated headers (X -> X.1); restore them when writing
    header = pd.read_csv(in_path, sep=sep, header=None, nrows=1, dtype=str,
                         keep_default_na=False).iloc[0].tolist()
    restore = {m: o for m, o in zip(df.columns, header) if m != o} \
        if len(header) == df.shape[1] else {}
    n = len(df)
    print(f"Read {n:,} rows x {df.shape[1]} columns from {in_path}")
    missing = {"chrom", "pos", "ref", "alt", args.hgvs_c_col} - set(df.columns)
    if missing:
        sys.exit(f"Input is missing columns: {sorted(missing)}")
    col = lambda c: df[c].tolist() if c in df.columns else [""] * n

    # ---- transcripts named in hgvs_c -> RefSeq accessions to load ----
    named = Counter(m["tx"] for m in (HGVS_C_RE.match(h.strip()) for h in df[args.hgvs_c_col])
                    if m)
    ensts = [t for t in named if t.startswith("ENST")]
    print("hgvs_c transcripts (coding/UTR/intronic SNVs): "
          + ", ".join(f"{t} ({k:,})" for t, k in named.most_common(10)))
    mane = ({}, {})
    if ensts:
        if not args.mane_summary:
            p.error("hgvs_c uses Ensembl transcripts; give --mane_summary to map them to RefSeq")
        mane = read_mane_summary(args.mane_summary, mvp.strip_version)
    wanted = {t for t in named if not t.startswith("ENST")}
    for t in ensts:
        infos = ([mane[0][t]] if t in mane[0] else []) + \
            (mane[1].get(mvp.strip_version(t), []) if args.allow_other_versions else [])
        wanted |= {i["nm"] for i in infos}
        if t in mane[0]:
            print(f"  {t} -> {mane[0][t]['nm']} ({mane[0][t]['symbol']})")
        else:
            print(f"  {t}: not in the MANE summary with this version"
                  + (" (trying another version)" if args.allow_other_versions
                     else " (see --allow_other_versions)"))
    gbff, versions = mvp.read_gbff(args.gbff, wanted)
    print(f"RefSeq transcripts needed: {len(wanted):,}; "
          f"{sum(w in gbff for w in wanted):,} found with the exact version in the gbff file(s)")

    # ---- annotate ----
    rows = [annotate(r, a, hc, hp, aac, mane, gbff, versions, args.allow_other_versions, mvp)
            for r, a, hc, hp, aac in zip(df["ref"], df["alt"], df[args.hgvs_c_col],
                                         col(args.hgvs_p_col), col(args.aa_change_col))]
    ann = pd.DataFrame(rows, columns=ANNOT_COLS + ["status"], index=df.index)
    for c in INT_COLS:
        ann[c] = pd.to_numeric(ann[c]).astype("Int64")

    vid = df["chrom"] + ":" + df["pos"] + ":" + df["ref"] + ":" + df["alt"]
    dup = vid.groupby(vid).cumcount()
    if (dup > 0).any():
        print(f"{int((dup > 0).sum()):,} duplicated chrom:pos:ref:alt ids suffixed _1, _2, ...")
    vid = vid.where(dup == 0, vid + "_" + dup.astype(str))

    out = df.copy()
    clashes = [c for c in NEW_COLS if c in out.columns]
    if clashes:
        out = out.rename(columns={c: f"{c}_original" for c in clashes})
        print(f"Renamed existing columns {clashes} -> *_original")
    out["variant_id"] = vid.values
    for c in ANNOT_COLS:
        out[c] = ann[c]
    if args.label_col.lower() != "none" and args.label_col in df.columns:
        lab = df[args.label_col].str.lower()
        label = np.where(lab.str.contains("abnormal"), 1.0,
                         np.where(lab.str.contains("normal"), 0.0, np.nan))
        out["label"] = pd.Series(label, index=out.index).astype("Int64")
    elif args.label_col.lower() != "none":
        print(f"Label column '{args.label_col}' not found; no label written")
    out["status"] = ann["status"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.rename(columns=restore).to_csv(out_path, sep="\t", index=False)

    # ---- summary ----
    status = ann["status"]
    ok = status == "ok"
    print("\nstatus:")
    for k, v in status.value_counts().items():
        print(f"  {k:26s} {v:>7,}  ({100 * v / n:.1f}%)")
    if args.consequence_col in df.columns:
        print("\nTable consequence vs computed status:")
        print(pd.crosstab(df[args.consequence_col], status).to_string())
    agrees = ann["p_agrees"].map({True: "agree", False: "disagree"}).fillna("no p.")
    print(f"\nhgvs_p vs computed change (coding SNVs): "
          f"{agrees[ann['cds_position'].notna()].value_counts().to_dict()}")
    print(f"transcript_source among ok rows: "
          f"{ann.loc[ok, 'transcript_source'].value_counts().to_dict()}")
    if "label" in out.columns:
        lab_ok = out.loc[ok, "label"]
        print(f"Labels among ok rows: {int((lab_ok == 1).sum()):,} abnormal (1), "
              f"{int((lab_ok == 0).sum()):,} normal (0), {int(lab_ok.isna().sum()):,} other")
    print(f"\nWrote {out_path} ({int(ok.sum()):,} / {n:,} rows scorable)")


if __name__ == "__main__":
    main()
