#!/usr/bin/env python3
"""
Step 2 — Annotate variants with ALL dbNSFP scores via tabix.

Usage:
    python evaluation/02_annotate_dbnsfp.py \
        --variants data/sequences/ClinVar.260923only.seq12k.pb.tsv \
        --dbnsfp   data/dbnsfp/dbNSFP5.3.1a_grch38.gz \
        --outdir   results/predictions/ClinVar.260923only.seq12k.pb

Downloads (if you don't have dbNSFP yet):
    curl --http1.1 -C - -o data/dbnsfp/dbNSFP5.3.1a_grch38.gz \\
         https://dist.genos.us/academic/yourcode/dbNSFP5.3.1a_grch38.gz
    curl --http1.1 -C - -o data/dbnsfp/dbNSFP5.3.1a_grch38.gz.tbi \\
         https://dist.genos.us/academic/yourcode/dbNSFP5.3.1a_grch38.gz.tbi

Notes on semicolon-separated fields:
    dbNSFP stores one value per isoform, semicolon-separated, in parallel across
    all columns (e.g. VEP_canonical="YES;NO;NO", SIFT_score="0.02;0.15;0.08").
    This script resolves each field to the single value corresponding to the
    canonical isoform (VEP_canonical == "YES").  If no canonical isoform is
    flagged, it falls back to the first value.  Fields without semicolons are
    returned as-is.

Optional --clinvar_tx (preprocessing/clinvar_transcripts.py): use the transcript
ClinVar named the variant on, mapped to Ensembl by MANE, then NCBI gene2ensembl.
    dbnsfp.tsv           same format. The per-transcript predictor scores in
                         TX_SCORES come only from the ClinVar-matched transcript
                         ("." if none matches); all other fields use that
                         transcript when matched, else the canonical isoform.
                         Without --clinvar_tx, identical to the original output.
    dbnsfp_fallback.tsv  always written. For each score in FALLBACK_SCORES, the value
                         from the first tier that has one, and <score>_tier:
                           clinvar_tx > canonical > first_isoform > any_transcript
                         (first_isoform only when no isoform is flagged canonical;
                         any_transcript = most damaging value across isoforms: max,
                         min for SIFT and ESM1b). Variant-level scores (CADD,
                         ClinPred) get tier "variant". A tier-k value implies no
                         value at any tier above k, so downstream code can choose how
                         far to relax without re-annotating. Also: clinvar_match
                         (mane / gene2ensembl / no_ensembl_match / no_mapping /
                         no_clinvar_tx / no_dbnsfp_row) and clinvar_hgvsc_agree
                         (matched transcript's HGVSc_VEP == ClinVar's c.).
"""

import argparse
import subprocess
from pathlib import Path

import pandas as pd


REQUIRED_COLS = ["chromosome", "position", "ref_allele", "alt_allele"]
OPTIONAL_COLS = ["variant_id", "label"]

# Column name used by dbNSFP 4.x / 5.x to flag the VEP canonical transcript.
VEP_CANONICAL_COL = "VEP_canonical"

# Per-transcript predictor scores and how to pick the most damaging value.
TX_SCORES = {"AlphaMissense_score": max, "ESM1b_score": min, "REVEL_score": max,
             "SIFT_score": min, "Polyphen2_HVAR_score": max}
VARIANT_SCORES = ["CADD_phred", "ClinPred_score"]       # one value per variant
FALLBACK_SCORES = list(TX_SCORES) + VARIANT_SCORES
TIERS = ["clinvar_tx", "canonical", "first_isoform", "any_transcript", "variant", "missing"]


def parse_args():
    p = argparse.ArgumentParser(description="Annotate variants with dbNSFP via tabix.")
    p.add_argument("--variants", required=True, type=Path,
                   help="Input variants TSV (columns: variant_id, chromosome, position, "
                        "ref_allele, alt_allele, label)")
    p.add_argument("--dbnsfp",   required=True, type=Path,
                   help="Path to dbNSFP bgzipped file (e.g. dbNSFP5.3.1a_grch38.gz)")
    p.add_argument("--outdir",   required=True, type=Path,
                   help="Output directory")
    p.add_argument("--clinvar_tx", type=Path, default=None,
                   help="Optional ClinVar transcript table keyed by variant_id "
                        "(preprocessing/clinvar_transcripts.py)")
    return p.parse_args()


def read_dbnsfp_header(dbnsfp_gz: Path) -> list[str]:
    result = subprocess.run(
        f"zcat {dbnsfp_gz} | head -n1",
        shell=True, capture_output=True, text=True
    )
    return result.stdout.strip().lstrip("#").split("\t")


def query_tabix(dbnsfp_gz: Path, chrom: str, pos: int,
                ref: str, alt: str) -> list[str] | None:
    chrom = "M" if chrom == "MT" else chrom          # dbNSFP names mitochondria "M"
    proc = subprocess.Popen(
        ["tabix", str(dbnsfp_gz), f"{chrom}:{pos}-{pos}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    matched = None
    for raw_line in proc.stdout:
        line   = raw_line.decode("utf-8").rstrip("\n").rstrip("\r")
        fields = line.split("\t")
        if len(fields) > 3 and fields[2] == ref and fields[3] == alt:
            matched = fields
            break
    proc.stdout.close()
    proc.wait()
    return matched


def resolve_canonical_index(fields: list[str], vep_canonical_idx: int | None) -> int:
    """
    Given a matched dbNSFP row, return the isoform index corresponding to the
    VEP canonical transcript (VEP_canonical == "YES").

    Falls back to index 0 if:
      - VEP_canonical column is absent from the header
      - no isoform is flagged as canonical
      - the canonical field itself has no semicolons (single isoform)
    """
    if vep_canonical_idx is None or vep_canonical_idx >= len(fields):
        return 0
    canonical_str = fields[vep_canonical_idx]
    if ";" not in canonical_str:
        return 0
    parts = canonical_str.split(";")
    for i, v in enumerate(parts):
        if v.strip().upper() == "YES":
            return i
    return 0  # fallback: no YES found


def extract_field(raw_value: str, canonical_idx: int) -> str:
    """
    Extract the value at `canonical_idx` from a semicolon-separated string.
    Returns the whole string unchanged if it contains no semicolons.
    Returns "." if the index is out of range.
    """
    if ";" not in raw_value:
        return raw_value
    parts = raw_value.split(";")
    if canonical_idx < len(parts):
        return parts[canonical_idx]
    return parts[0]  # index out of range — fall back to first


def flagged_canonical(fields: list[str], vep_canonical_idx: int | None) -> int | None:
    """Index of the isoform with VEP_canonical == YES, or None if none is flagged."""
    if vep_canonical_idx is None:
        return None
    for i, v in enumerate(fields[vep_canonical_idx].split(";")):
        if v.strip().upper() == "YES":
            return i
    return None


def is_num(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def aligned_value(raw_value: str, idxs: list[int], n_isoforms: int) -> str | None:
    """First numeric value among isoforms idxs, only if the field has one value per isoform."""
    parts = raw_value.split(";")
    if len(parts) != n_isoforms:
        return None
    for i in idxs:
        if i < len(parts) and is_num(parts[i]):
            return parts[i]
    return None


def clinvar_isoforms(enst: list[str], cv: dict | None) -> tuple[list[int], str]:
    """dbNSFP isoform indices matching ClinVar's transcript, and how they were matched."""
    if not cv or not cv.get("clinvar_transcript"):
        return [], "no_clinvar_tx"
    if not cv.get("mane_enst") and not cv.get("g2e_enst"):
        return [], "no_mapping"
    for method, ids in [("mane", [cv.get("mane_enst", "")]),
                        ("gene2ensembl", cv.get("g2e_enst", "").split(";"))]:
        idx = [i for i, e in enumerate(enst) if e and e in ids]
        if idx:
            return idx, method
    return [], "no_ensembl_match"


def main():
    args    = parse_args()
    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = out_dir / "dbnsfp.tsv"

    # ── Load variants ──────────────────────────────────────────────────────
    print(f"Loading variants from {args.variants}")
    all_cols = pd.read_csv(args.variants, sep="\t", nrows=0).columns.tolist()
    usecols  = REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in all_cols]
    variants = pd.read_csv(args.variants, sep="\t", usecols=usecols)

    # Auto-generate variant_id if not present in input
    if "variant_id" not in variants.columns:
        variants["variant_id"] = (
            variants["chromosome"].astype(str) + ":" +
            variants["position"].astype(str)   + ":" +
            variants["ref_allele"].astype(str) + ">" +
            variants["alt_allele"].astype(str)
        )
        print(f"  'variant_id' column not found — auto-generated as chrom:pos:ref>alt")

    has_label = "label" in variants.columns
    if not has_label:
        print(f"  'label' column not found — will be omitted from output")

    print(f"  {len(variants):,} variants loaded")

    # ── Read dbNSFP header ─────────────────────────────────────────────────
    print(f"Reading dbNSFP header from {args.dbnsfp} ...")
    header  = read_dbnsfp_header(args.dbnsfp)
    col_idx = {name: i for i, name in enumerate(header)}
    print(f"  {len(header)} columns")

    # Index of VEP_canonical within a tabix row — used for isoform resolution
    vep_canonical_idx = col_idx.get(VEP_CANONICAL_COL)
    if vep_canonical_idx is not None:
        print(f"  VEP_canonical column found at index {vep_canonical_idx} "
              f"— will resolve semicolon-separated fields to canonical isoform")
    else:
        print(f"  WARNING: '{VEP_CANONICAL_COL}' not found in header "
              f"— will fall back to first semicolon value for all fields")

    # Truncated-file check: the last chromosome in the index must be readable
    last_chrom = subprocess.run(["tabix", "-l", str(args.dbnsfp)],
                                capture_output=True, text=True).stdout.split()[-1]
    probe = subprocess.run(f"tabix {args.dbnsfp} {last_chrom} | head -n1",
                           shell=True, capture_output=True, text=True)
    if not probe.stdout.strip() or "truncated" in probe.stderr:
        print(f"  WARNING: no rows readable for chromosome {last_chrom} "
              f"— dbNSFP file may be truncated. {probe.stderr.strip()}")

    for c in FALLBACK_SCORES + ["Ensembl_transcriptid", "HGVSc_VEP"]:
        if c not in col_idx:
            print(f"  WARNING: '{c}' not in dbNSFP header")
    tx_idx  = col_idx.get("Ensembl_transcriptid")
    hgv_idx = col_idx.get("HGVSc_VEP")

    clinvar = {}
    if args.clinvar_tx is not None:
        clinvar = (pd.read_csv(args.clinvar_tx, sep="\t", dtype=str).fillna("")
                   .drop_duplicates("variant_id").set_index("variant_id").to_dict("index"))
        n_cv = variants["variant_id"].astype(str).isin(clinvar).sum()
        print(f"  ClinVar transcripts: {len(clinvar):,} loaded, {n_cv:,}/{len(variants):,} input variants found "
              f"— TX_SCORES in dbnsfp.tsv taken from the ClinVar-matched transcript only")
        if n_cv == 0:
            print("  WARNING: no input variant_id in --clinvar_tx (variant_id must be the ClinVar VariationID)")

    # ── Query tabix per variant ────────────────────────────────────────────
    print("Querying dbNSFP via tabix...")
    rows        = []
    fallback    = []
    unaligned   = {s: 0 for s in TX_SCORES}
    found_count = 0
    not_found   = 0

    for i, var in variants.iterrows():
        chrom = str(var["chromosome"])
        pos   = int(var["position"])
        ref   = str(var["ref_allele"])
        alt   = str(var["alt_allele"])

        matched = query_tabix(args.dbnsfp, chrom, pos, ref, alt)

        record = {
            "variant_id": var["variant_id"],
            "chromosome": chrom,
            "position":   pos,
            "ref_allele": ref,
            "alt_allele": alt,
        }
        if has_label:
            record["label"] = var["label"]

        fb = {k: record[k] for k in ("variant_id", "chromosome", "position", "ref_allele", "alt_allele")}
        cv = clinvar.get(str(var["variant_id"]))
        fb["clinvar_transcript"] = cv.get("clinvar_transcript", "") if cv else ""

        if matched:
            # Determine which isoform index corresponds to the canonical transcript
            canon_idx = resolve_canonical_index(matched, vep_canonical_idx)
            flag_idx  = flagged_canonical(matched, vep_canonical_idx)
            enst      = [e.split(".")[0] for e in matched[tx_idx].split(";")] if tx_idx is not None else []
            n_iso     = max(len(enst), 1)
            cv_idx, method = clinvar_isoforms(enst, cv)
            fb["clinvar_match"] = method
            fb["matched_enst"]  = enst[cv_idx[0]] if cv_idx else ""
            fb["clinvar_hgvsc_agree"] = ""
            if cv_idx and hgv_idx is not None and cv.get("clinvar_hgvsc"):
                hg = matched[hgv_idx].split(";")
                h  = hg[cv_idx[0]].split(":")[-1] if len(hg) == n_iso else ""
                fb["clinvar_hgvsc_agree"] = "yes" if h == cv["clinvar_hgvsc"] else "no"

            for col_name, idx in col_idx.items():
                if col_name not in record:
                    raw = matched[idx] if idx < len(matched) else "."
                    if args.clinvar_tx is None:
                        record[col_name] = extract_field(raw, canon_idx)            # original behaviour
                    elif col_name in TX_SCORES:
                        record[col_name] = aligned_value(raw, cv_idx, n_iso) or "."
                    else:
                        record[col_name] = extract_field(raw, cv_idx[0] if cv_idx else canon_idx)

            # Tiered fallback: first tier with a value wins
            for s in FALLBACK_SCORES:
                if s not in col_idx:
                    continue
                raw = matched[col_idx[s]]
                if s in TX_SCORES and len(raw.split(";")) != n_iso and raw not in (".", ""):
                    unaligned[s] += 1
                val, tier = None, "missing"
                if s in VARIANT_SCORES:
                    nums = [x for x in raw.split(";") if is_num(x)]
                    val, tier = (nums[0], "variant") if nums else (None, "missing")
                else:
                    for t, idxs in [("clinvar_tx", cv_idx),
                                    ("canonical", [flag_idx] if flag_idx is not None else []),
                                    ("first_isoform", [0] if flag_idx is None else [])]:
                        val = aligned_value(raw, idxs, n_iso)
                        if val is not None:
                            tier = t
                            break
                    if val is None:
                        nums = [x for x in raw.split(";") if is_num(x)]
                        if nums:
                            val, tier = TX_SCORES[s](nums, key=float), "any_transcript"
                fb[s], fb[f"{s}_tier"] = (val if val is not None else "."), tier
            found_count += 1
        else:
            not_found += 1
            for col_name in header:
                if col_name not in record:
                    record[col_name] = "."
            fb.update({"clinvar_match": "no_dbnsfp_row", "matched_enst": "", "clinvar_hgvsc_agree": ""})
            for s in FALLBACK_SCORES:
                fb[s], fb[f"{s}_tier"] = ".", "missing"

        rows.append(record)
        fallback.append(fb)

        n_done = i + 1
        if n_done % 1000 == 0:
            pct      = 100 * n_done / len(variants)
            hit_rate = 100 * found_count / n_done
            print(f"  [{n_done:>6,} / {len(variants):,}  {pct:5.1f}%]  "
                  f"found: {found_count:,}  not found: {not_found:,}  "
                  f"({hit_rate:.1f}% hit rate)")

    print(f"\nDone. {found_count:,}/{len(rows):,} variants found in dbNSFP")
    print(f"  {not_found:,} not found (intronic / intergenic / synonymous)")

    fb_df = pd.DataFrame(fallback)
    print("\nClinVar transcript match:", fb_df["clinvar_match"].value_counts().to_dict())
    agree = fb_df["clinvar_hgvsc_agree"]
    if (agree != "").any():
        print(f"  matched transcript has ClinVar's c. change: {(agree == 'yes').sum():,}/{(agree != '').sum():,}")
    tiers = pd.DataFrame({s: fb_df[f"{s}_tier"].value_counts() for s in FALLBACK_SCORES if f"{s}_tier" in fb_df}).T
    tiers = tiers.reindex(columns=TIERS).fillna(0).astype(int)
    print("\nScore tier (first tier with a value):")
    print(tiers.to_string())
    print("\nCumulative coverage when allowing tiers up to ...:")
    print(tiers[TIERS[:4]].cumsum(axis=1).add(tiers["variant"], axis=0).to_string())
    if any(unaligned.values()):
        print(f"\n  values not one-per-isoform (only usable as any_transcript): {dict(unaligned)}")
    fb_df.to_csv(out_dir / "dbnsfp_fallback.tsv", sep="\t", index=False)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nSaved to {out_tsv}")
    print(f"Tiered fallback saved to {out_dir / 'dbnsfp_fallback.tsv'}")
    print(f"Output shape: {out_df.shape[0]:,} rows × {out_df.shape[1]} columns")


if __name__ == "__main__":
    main()