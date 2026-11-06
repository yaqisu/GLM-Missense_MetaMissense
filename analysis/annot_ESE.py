#!/usr/bin/env python3
"""
Add HGVSc, FUNC_SCORE, RNA_SCORE and SpliceAI DS max to RAD51D_variants.csv
by matching variant_id against one or more Casadei-style TSV datasets, and
fill "In ESE hotspot" with True (c. position within --hotspot) or NA.

Column names are found with or without the "src_" prefix, so both the raw
dataset and the *_resolved.tsv files work.

Rows whose field count does not match the header (e.g. merged tabs from
copy/paste) are parsed by anchoring on recognizable values instead:
  HGVS_C          first field like ENST...:c.
  RNA_SCORE       HGVS_C + 2
  FUNC_SCORE      HGVS_C + 6
  spliceai_DS_max field just before the last occurrence of the gene symbol
                  (spliceai_gene)

"In ESE hotspot" is filled with True/NA from the HGVSc c. position: True when the
variant is exonic (no intronic +/- offset) on the hotspot's transcript and inside
the c. interval. Default hotspot: RAD51D exon 3 c.202_235 on ENST00000345365
(Bueno-Martinez et al. 2021); override or add more with --hotspot.

Usage:
  python add_scores_to_csv.py -c RAD51D_variants.csv -d dataset.txt [dataset2.tsv ...] \
      -o RAD51D_variants_annotated.csv [--hotspot ENST00000345365:202-235 ...]
"""
import argparse
import csv
import re
import sys

NEW_COLS = ["HGVSc", "FUNC_SCORE", "RNA_SCORE", "SpliceAI_DS_max"]
HGVS_RE = re.compile(r"^ENST[\d.]+:c\.")
CPOS_RE = re.compile(r"^(ENST\d+)(?:\.\d+)?:c\.(\d+)([+-]\d+)?")
HOTSPOT_COL = "In ESE hotspot"


def parse_hotspots(specs):
    out = []
    for s in specs:
        m = re.fullmatch(r"(ENST\d+)(?:\.\d+)?:(\d+)-(\d+)", s.strip())
        if not m:
            sys.exit(f"Bad --hotspot '{s}' (expected ENST00000345365:202-235)")
        out.append((m.group(1), int(m.group(2)), int(m.group(3))))
    return out


def in_hotspot(hgvsc, hotspots):
    m = CPOS_RE.match(hgvsc or "")
    if not m or m.group(3):             # no c. notation, or intronic offset (c.263+5 etc.)
        return "NA"
    tx, c = m.group(1), int(m.group(2))
    return "True" if any(tx == t and lo <= c <= hi for t, lo, hi in hotspots) else "NA"


def col_index(header, name):
    for n in (name, "src_" + name):
        if n in header:
            return header.index(n)
    return None


def parse_aligned(row, idx):
    get = lambda k: row[idx[k]].strip() if idx[k] is not None else ""
    return {"HGVSc": get("HGVS_C") or get("HGVSc_VEP"),
            "FUNC_SCORE": get("FUNC_SCORE"),
            "RNA_SCORE": get("RNA_SCORE"),
            "SpliceAI_DS_max": get("spliceai_DS_max")}


def parse_anchored(row, gene):
    out = dict.fromkeys(NEW_COLS, "")
    h = next((i for i, v in enumerate(row) if HGVS_RE.match(v.strip())), None)
    if h is not None:
        out["HGVSc"] = row[h].strip()
        if h + 6 < len(row):
            out["RNA_SCORE"] = row[h + 2].strip()
            out["FUNC_SCORE"] = row[h + 6].strip()
    if gene:
        g = max((i for i, v in enumerate(row) if v.strip() == gene), default=None)
        if g is not None and g > 0:
            out["SpliceAI_DS_max"] = row[g - 1].strip()
    return out


def load_dataset(path, lookup):
    n_aligned = n_anchored = 0
    with open(path, newline="") as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = [h.strip() for h in next(rd)]
        idx = {k: col_index(header, k) for k in
               ("variant_id", "HGVS_C", "HGVSc_VEP", "RNA_SCORE", "FUNC_SCORE",
                "spliceai_DS_max", "GENE", "genename")}
        if idx["variant_id"] is None:
            sys.exit(f"{path}: no variant_id column")
        for row in rd:
            if not row or not row[0].strip():
                continue
            vid = row[idx["variant_id"]].strip()
            if len(row) == len(header):
                rec = parse_aligned(row, idx)
                n_aligned += 1
            else:
                gi = idx["GENE"] if idx["GENE"] is not None else idx["genename"]
                gene = row[gi].strip() if gi is not None and gi < len(row) else ""
                rec = parse_anchored(row, gene)
                n_anchored += 1
            if vid in lookup and lookup[vid] != rec:
                print(f"WARNING: {vid} appears again in {path} with different values; keeping first",
                      file=sys.stderr)
                continue
            lookup.setdefault(vid, rec)
    print(f"{path}: {n_aligned} rows parsed by header, {n_anchored} by anchoring "
          f"(field count != header)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--csv", required=True, help="RAD51D_variants.csv")
    ap.add_argument("-d", "--data", required=True, nargs="+", help="dataset TSV/TXT file(s)")
    ap.add_argument("-o", "--output", help="output CSV (default: <csv>_annotated.csv)")
    ap.add_argument("--hotspot", nargs="+", default=["ENST00000345365:202-235"],
                    help="ESE hotspot(s) as TRANSCRIPT:cstart-cend (default: ENST00000345365:202-235)")
    a = ap.parse_args()
    hotspots = parse_hotspots(a.hotspot)
    out_path = a.output or re.sub(r"\.csv$", "", a.csv) + "_annotated.csv"

    lookup = {}
    for p in a.data:
        load_dataset(p, lookup)

    with open(a.csv, newline="") as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        fields = [f for f in rd.fieldnames if f not in NEW_COLS]

    # HGVSc right after variant_id; scores after the existing columns
    out_fields = fields[:1] + ["HGVSc"] + fields[1:] + ["FUNC_SCORE", "RNA_SCORE", "SpliceAI_DS_max"]
    if HOTSPOT_COL not in out_fields:
        out_fields.insert(out_fields.index("FUNC_SCORE"), HOTSPOT_COL)
    missing, rna_mismatch = [], []
    for r in rows:
        rec = lookup.get(r["variant_id"].strip())
        if rec is None:
            missing.append(r["variant_id"])
            rec = dict.fromkeys(NEW_COLS, "")
        r.update(rec)
        r[HOTSPOT_COL] = in_hotspot(rec["HGVSc"], hotspots)
        old = r.get("RNA score", "")
        try:
            if old and rec["RNA_SCORE"] and abs(float(old) - float(rec["RNA_SCORE"])) > 1e-4:
                rna_mismatch.append(r["variant_id"])
        except ValueError:
            pass

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_path}: {len(rows) - len(missing)}/{len(rows)} variants matched, "
          f"{sum(r[HOTSPOT_COL] == 'True' for r in rows)} in ESE hotspot", file=sys.stderr)
    if missing:
        print(f"Not found in dataset(s): {', '.join(missing)}", file=sys.stderr)
    if rna_mismatch:
        print(f"WARNING: RNA_SCORE differs from existing 'RNA score' for: {', '.join(rna_mismatch)}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
