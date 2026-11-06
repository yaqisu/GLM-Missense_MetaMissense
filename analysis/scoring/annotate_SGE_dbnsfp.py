#!/usr/bin/env python3
"""
Step 2 — Annotate variants with ALL dbNSFP scores via tabix.

Usage:
    python annotate_dbnsfp.py \
        --variants SGE_combined_missenseSNVs.tsv \
        --dbnsfp   data/dbnsfp/dbNSFP5.3.1a_grch38.gz \
        --outdir   results/SGE_dbNSFP
    Optional: --transcript-col TRANSCRIPT_ID  --protein-col PROTEIN_CHANGE  (defaults)

Downloads (if you don't have dbNSFP yet):
    curl --http1.1 -C - -o data/dbnsfp/dbNSFP5.3.1a_grch38.gz \\
         https://dist.genos.us/academic/yourcode/dbNSFP5.3.1a_grch38.gz
    curl --http1.1 -C - -o data/dbnsfp/dbNSFP5.3.1a_grch38.gz.tbi \\
         https://dist.genos.us/academic/yourcode/dbNSFP5.3.1a_grch38.gz.tbi

Input / output:
    Input TSV needs chromosome, position, ref_allele, alt_allele. Every input column
    is kept unchanged; the dbNSFP columns are appended after them (a dbNSFP column
    whose name already exists in the input is prefixed with "dbNSFP_").
    Output: <outdir>/dbnsfp.tsv

Transcript resolution:
    dbNSFP stores transcript-specific values semicolon-separated, in the same order
    as Ensembl_transcriptid. For each variant the value for the input's own
    transcript (--transcript-col, e.g. ENST00000269305.9; version ignored) is used.
    If that transcript is not listed for the variant, transcript-specific fields
    are set to "." (see dbNSFP_status). If the input has no transcript column, the
    VEP_canonical transcript is used instead (the previous behaviour).

    Which fields are transcript-specific is decided from the data: a field counts
    as per-transcript if, in most matched rows, its number of ';'-entries equals
    the number of transcripts. In a row where it doesn't, the value is set to "."
    rather than guessed (counted in the summary). Fields keyed on other ID lists
    (MutationTaster, VEST3, InterPro, POPMAX; list from Ensembl's VEP dbNSFP
    plugin) and other multi-valued fields are kept as the full raw string and
    listed at the end of the run.

Added columns (right after the input columns):
    dbNSFP_status         matched | transcript_not_in_dbNSFP | variant_not_in_dbNSFP
                          (canonical, when no transcript column is given)
    dbNSFP_aa_consistent  True/False: dbNSFP aaref/aapos/aaalt for the chosen
                          transcript equal --protein-col (e.g. R175H); NA otherwise

Chromosome names:
    dbNSFP names chromosomes without a prefix ("17"); inputs written as "chr17"
    are mapped to the index's naming before querying. The input's own names are
    kept in the output.
"""

import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd


REQUIRED_COLS = ["chromosome", "position", "ref_allele", "alt_allele"]
TX_COL = "Ensembl_transcriptid"
VEP_CANONICAL_COL = "VEP_canonical"

# ';'-separated fields NOT ordered like Ensembl_transcriptid
# (Ensembl VEP_plugins dbNSFP.pm, %NON_TRANSCRIPT_SPECIFIC_FIELDS; plus all MutationTaster_*)
NON_TRANSCRIPT_FIELDS = {
    "Interpro_domain", "MutPred_Top5features", "Transcript_id_VEST3",
    "Transcript_var_VEST3", "VEST3_score", "dbNSFP_POPMAX_AC", "dbNSFP_POPMAX_POP",
    "AllofUs_POPMAX_AC", "AllofUs_POPMAX_AN", "AllofUs_POPMAX_POP",
}
KEY_SCORES = ["SIFT_score", "SIFT4G_score", "Polyphen2_HDIV_score", "Polyphen2_HVAR_score",
              "REVEL_score", "AlphaMissense_score", "ESM1b_score", "CADD_raw", "CADD_phred"]
PROTEIN_RE = re.compile(r"([A-Z*])(\d+)([A-Z*])")


def parse_args():
    p = argparse.ArgumentParser(description="Annotate variants with dbNSFP via tabix.")
    p.add_argument("--variants", required=True, type=Path,
                   help="Input variants TSV (needs chromosome, position, ref_allele, "
                        "alt_allele; all columns are kept)")
    p.add_argument("--dbnsfp", required=True, type=Path,
                   help="Path to dbNSFP bgzipped file (e.g. dbNSFP5.3.1a_grch38.gz)")
    p.add_argument("--outdir", required=True, type=Path, help="Output directory")
    p.add_argument("--transcript-col", default="TRANSCRIPT_ID",
                   help="Input column with the Ensembl transcript to take scores for "
                        "(default: TRANSCRIPT_ID; falls back to VEP_canonical if absent)")
    p.add_argument("--protein-col", default="PROTEIN_CHANGE",
                   help="Input column with the protein change (e.g. R175H) used to check "
                        "the chosen transcript (default: PROTEIN_CHANGE; skipped if absent)")
    return p.parse_args()


def is_non_transcript(col: str) -> bool:
    return col.startswith("MutationTaster_") or col in NON_TRANSCRIPT_FIELDS


def strip_version(tx: str) -> str:
    return tx.strip().split(".")[0]


def read_dbnsfp_header(dbnsfp_gz: Path) -> list[str]:
    result = subprocess.run(
        f"zcat {dbnsfp_gz} | head -n1",
        shell=True, capture_output=True, text=True
    )
    return result.stdout.strip().lstrip("#").split("\t")


def tabix_seqnames(dbnsfp_gz: Path) -> set[str]:
    result = subprocess.run(["tabix", "-l", str(dbnsfp_gz)],
                            capture_output=True, text=True)
    return set(result.stdout.split())


def match_chrom(chrom: str, seqnames: set[str]) -> str:
    """Map an input chromosome name onto the index's naming ("chr17" <-> "17")."""
    if chrom in seqnames:
        return chrom
    alt = chrom[3:] if chrom.lower().startswith("chr") else "chr" + chrom
    return alt if alt in seqnames else chrom


def query_tabix(dbnsfp_gz: Path, chrom: str, pos: int, ref: str, alt: str) -> list[str]:
    """All dbNSFP lines (raw strings) at chrom:pos with matching ref/alt."""
    proc = subprocess.run(["tabix", str(dbnsfp_gz), f"{chrom}:{pos}-{pos}"],
                          capture_output=True, text=True)
    hits = []
    for line in proc.stdout.splitlines():
        line = line.rstrip("\r")
        f = line.split("\t", 4)
        if len(f) > 3 and f[2] == ref and f[3] == alt:
            hits.append(line)
    return hits


def canonical_index(fields: list[str], canon_i: int | None) -> int:
    """Index of the VEP_canonical == YES transcript (0 if none flagged)."""
    if canon_i is None or canon_i >= len(fields):
        return 0
    for i, v in enumerate(fields[canon_i].split(";")):
        if v.strip().upper() == "YES":
            return i
    return 0


def classify_columns(lines: list[str | None], header: list[str], tx_i: int):
    """
    Decide which columns run parallel to Ensembl_transcriptid: in rows where the
    column is informative, its entry count must equal the transcript count in the
    majority of them. Returns (aligned column indices, names of other columns that
    contain ';' and are therefore kept raw).
    """
    n = len(header)
    parallel, other, multi = [0] * n, [0] * n, [False] * n
    for line in lines:
        if line is None:
            continue
        f = line.split("\t")
        k = f[tx_i].count(";") + 1
        for j, v in enumerate(f[:n]):
            c = v.count(";") + 1
            if c > 1:
                multi[j] = True
            elif k == 1 or v == ".":
                continue  # single value in a single-transcript row: uninformative
            if c == k:
                parallel[j] += 1
            else:
                other[j] += 1
    aligned = {j for j in range(n) if parallel[j] > other[j]
               and not is_non_transcript(header[j])}
    kept_raw = sorted(header[j] for j in range(n) if multi[j] and j not in aligned)
    return aligned, kept_raw


def main():
    args    = parse_args()
    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = out_dir / "dbnsfp.tsv"

    # ── Load variants (all columns, values kept exactly as in the file) ─────
    print(f"Loading variants from {args.variants}")
    variants = pd.read_csv(args.variants, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLS if c not in variants.columns]
    if missing:
        raise SystemExit(f"Input is missing required columns: {missing}")
    if "variant_id" not in variants.columns:
        variants.insert(0, "variant_id",
                        variants["chromosome"] + ":" + variants["position"] + ":" +
                        variants["ref_allele"] + ">" + variants["alt_allele"])
        print("  'variant_id' column not found — auto-generated as chrom:pos:ref>alt")
    n = len(variants)
    print(f"  {n:,} variants loaded, {variants.shape[1]} input columns kept")

    use_tx = args.transcript_col in variants.columns
    if use_tx:
        print(f"  transcript-specific values taken for '{args.transcript_col}' (version ignored)")
    else:
        print(f"  '{args.transcript_col}' not found — using the VEP_canonical transcript")
    check_aa = args.protein_col in variants.columns

    # ── Read dbNSFP header ─────────────────────────────────────────────────
    print(f"Reading dbNSFP header from {args.dbnsfp} ...")
    header  = read_dbnsfp_header(args.dbnsfp)
    col_idx = {name: i for i, name in enumerate(header)}
    print(f"  {len(header)} columns")
    tx_i = col_idx.get(TX_COL)
    if tx_i is None:
        raise SystemExit(f"'{TX_COL}' not found in the dbNSFP header")
    canon_i = col_idx.get(VEP_CANONICAL_COL)
    aa_i = [col_idx.get(c) for c in ("aaref", "aapos", "aaalt")]
    if check_aa and None in aa_i:
        print("  aaref/aapos/aaalt not all in dbNSFP header — protein check skipped")
        check_aa = False

    # Match input chromosome names to the names used in the tabix index
    seqnames  = tabix_seqnames(args.dbnsfp)
    chrom_map = {c: match_chrom(c, seqnames) for c in variants["chromosome"].unique()}
    renamed   = {k: v for k, v in chrom_map.items() if k != v}
    if renamed:
        print(f"  chromosome names mapped to dbNSFP naming: {renamed}")
    absent = sorted(v for v in chrom_map.values() if v not in seqnames)
    if absent:
        print(f"  WARNING: chromosomes not present in the dbNSFP index: {absent}")

    chroms = variants["chromosome"].tolist()
    poss   = variants["position"].tolist()
    refs   = variants["ref_allele"].tolist()
    alts   = variants["alt_allele"].tolist()
    txs    = ([strip_version(t) if t else None for t in variants[args.transcript_col]]
              if use_tx else [None] * n)

    # ── Pass 1: query tabix, keep the dbNSFP row that lists our transcript ──
    print("Querying dbNSFP via tabix...")
    chosen, found_count = [None] * n, 0
    for i in range(n):
        hits = query_tabix(args.dbnsfp, chrom_map[chroms[i]], int(float(poss[i])),
                           refs[i], alts[i])
        if hits:
            pick = hits[0]
            if len(hits) > 1 and txs[i]:
                for h in hits:
                    if txs[i] in {strip_version(t) for t in h.split("\t")[tx_i].split(";")}:
                        pick = h
                        break
            chosen[i] = pick
            found_count += 1
        if (i + 1) % 1000 == 0:
            print(f"  [{i + 1:>6,} / {n:,}  {100 * (i + 1) / n:5.1f}%]  "
                  f"found: {found_count:,}  not found: {i + 1 - found_count:,}")
    print(f"\nDone. {found_count:,}/{n:,} variants found in dbNSFP")

    # ── Pass 2: which columns are per-transcript ───────────────────────────
    aligned, kept_raw = classify_columns(chosen, header, tx_i)
    unresolved = Counter()  # rows where an aligned column's count didn't match
    print(f"  {len(aligned)} columns resolved per transcript, "
          f"{len(kept_raw)} multi-valued columns kept raw")

    # ── Pass 3: extract values for the chosen transcript ───────────────────
    empty = ["."] * len(header)
    db_rows, status, aa_ok = [], [], []
    for i in range(n):
        line = chosen[i]
        if line is None:
            db_rows.append(empty)
            status.append("variant_not_in_dbNSFP")
            aa_ok.append("NA")
            continue
        f = line.split("\t")[:len(header)]
        f += ["."] * (len(header) - len(f))
        tx_list = [strip_version(t) for t in f[tx_i].split(";")]
        k = len(tx_list)
        if use_tx:
            idx = tx_list.index(txs[i]) if txs[i] in tx_list else None
            status.append("matched" if idx is not None else "transcript_not_in_dbNSFP")
        else:
            idx = canonical_index(f, canon_i)
            status.append("canonical")
        vals = []
        for j, v in enumerate(f):
            if j in aligned:
                parts = v.split(";")
                if idx is None:
                    v = "."
                elif len(parts) == k:
                    v = parts[idx]
                elif v != ".":
                    v = "."  # entries don't line up with transcripts: don't guess
                    unresolved[header[j]] += 1
            vals.append(v)
        db_rows.append(vals)

        ok = "NA"
        if check_aa and idx is not None:
            m = PROTEIN_RE.fullmatch(variants.at[i, args.protein_col])
            if m:
                r, p, a = (vals[k].replace("X", "*") for k in aa_i)
                ok = str((r, p, a) == (m.group(1), m.group(2), m.group(3)))
        aa_ok.append(ok)

    # ── Assemble: input columns + status + dbNSFP columns ──────────────────
    db_cols = [f"dbNSFP_{c}" if c in variants.columns else c for c in header]
    db = pd.DataFrame(db_rows, columns=db_cols, index=variants.index)
    added = pd.DataFrame({"dbNSFP_status": status, "dbNSFP_aa_consistent": aa_ok},
                         index=variants.index)
    out_df = pd.concat([variants, added, db], axis=1)
    out_df.to_csv(out_tsv, sep="\t", index=False)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\nStatus:")
    for k, v in Counter(status).most_common():
        print(f"  {k:26s} {v:,}")
    if check_aa:
        c = Counter(aa_ok)
        print(f"Protein change agrees with dbNSFP for the chosen transcript: "
              f"{c['True']:,} yes, {c['False']:,} no, {c['NA']:,} not checkable")
        if c["False"]:
            bad = out_df[out_df["dbNSFP_aa_consistent"] == "False"]
            aa_names = [db_cols[k] for k in aa_i]
            print(bad[["variant_id", args.protein_col] + aa_names].head(10).to_string(index=False))
    print("Key predictor columns:")
    for col in KEY_SCORES:
        j = col_idx.get(col)
        how = ("not in this dbNSFP file" if j is None else
               "per transcript" if j in aligned else
               "multi-valued but NOT per transcript — kept raw" if col in kept_raw else
               "one value per variant")
        if unresolved[col]:
            how += f"; {unresolved[col]:,} row(s) set to '.' (entries didn't match transcripts)"
        print(f"  {col:22s} {how}")
    if kept_raw:
        names = ", ".join(kept_raw[:25]) + (" ..." if len(kept_raw) > 25 else "")
        print(f"Multi-valued columns kept raw ({len(kept_raw)}): {names}")

    print(f"\nSaved to {out_tsv}")
    print(f"Output shape: {out_df.shape[0]:,} rows × {out_df.shape[1]} columns")


if __name__ == "__main__":
    main()
