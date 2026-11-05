#!/usr/bin/env python3
"""esm1bft_prepare.py — ESM1b-FT scoring, step 1: slim variant table for map_variants_to_mane_protein.py.

Keeps variant_id, coordinates, the ClinVar transcript (TRANSCRIPT_ID) and the protein change, converted to the
one-letter form the mapping script compares against (p.Ile293Val -> I293V; anything else -> empty, i.e. not
checked).

Usage: python esm1bft_prepare.py --vus <VUS table> --out vus_input.tsv
"""
import argparse
import re

import pandas as pd

AA = dict(Ala="A", Arg="R", Asn="N", Asp="D", Cys="C", Gln="Q", Glu="E", Gly="G", His="H", Ile="I",
          Leu="L", Lys="K", Met="M", Phe="F", Pro="P", Ser="S", Thr="T", Trp="W", Tyr="Y", Val="V")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--vus", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def one_letter(p):
    m = re.match(r"^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$", str(p).strip())
    return f"{AA[m[1]]}{m[2]}{AA[m[3]]}" if m and m[1] in AA and m[3] in AA else None


cols = ["chromosome", "position", "ref_allele", "alt_allele", "TRANSCRIPT_ID", "PROTEIN_CHANGE"]
t = pd.read_csv(a.vus, sep="\t", dtype=str, usecols=lambda c: c in cols + ["variant_id", "variant_id_x"])
t = t.rename(columns={"variant_id_x": "variant_id"}).drop_duplicates("variant_id")
t["PROTEIN_CHANGE"] = t["PROTEIN_CHANGE"].map(one_letter)
print(f"{len(t):,} variants; PROTEIN_CHANGE converted for {t.PROTEIN_CHANGE.notna().sum():,}")
t[["variant_id"] + cols].to_csv(a.out, sep="\t", index=False)
print(f"Wrote {a.out}")
