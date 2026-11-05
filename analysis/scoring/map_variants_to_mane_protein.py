#!/usr/bin/env python3
"""
Map NT-2 variant TSVs to protein-level TSVs for ESM1b_ref_alt_contrast.py
==========================================================================
ClinVar mode (--variant_summary + --gbff): each variant is placed on the exact transcript
ClinVar used to name it.
    1. variant_id is looked up in ClinVar's variant_summary (GRCh38 rows). Whether it is a
       VariationID or an AlleleID is detected from agreement with the TSV locus
       (the two ID spaces overlap numerically, so match rate alone is not enough).
    2. The ClinVar Name, e.g. NM_007294.4(BRCA1):c.5123C>A (p.Ala1708Glu), gives the
       transcript (accession.version), the CDS position and the transcript-strand alleles.
    3. The transcript's mRNA, CDS and protein (/translation) come from RefSeq GenBank files
       (--gbff), e.g. MANE.GRCh38.v1.4.refseq_rna.gbff.gz. That file only has MANE
       transcripts; add the full RefSeq GRCh38 *_rna.gbff.gz to cover the rest.
    4. Checks: ClinVar locus == TSV locus; c. alleles == TSV alleles or their complements
       (which gives the strand); transcript base at c. == c. ref allele; codon translation
       == protein residue; computed change == ClinVar p. notation.
    Optional MANE fallback (--gtf + --genome_fasta [+ --transcript_ids]): variants whose
    ClinVar transcript can't be used (not in ClinVar, Name not a coding c. SNV, transcript
    not in the gbff files) are mapped to MANE Select through the genome instead.
    transcript_source records which route was used; clinvar_status keeps the ClinVar result.

MANE mode (no --variant_summary): every variant is mapped to the MANE Select transcript
whose CDS contains it, using a GTF + genome FASTA (see transcript selection below).

Given-transcript mode (--transcript_column): each variant is mapped to the transcript named
in that input column (e.g. TRANSCRIPT_ID already resolved upstream, as for MFASS), using the
GTF + genome FASTA. Only those transcripts are loaded from the GTF (any tag; IDs matched
without version). With --protein_change_column (e.g. PROTEIN_CHANGE = "P394L"), the computed
change is checked against it and disagreements get status input_protein_change_mismatch.

Output: every input row is written (1:1 with the NT-2 TSV). The large sequence columns
(upstream_flank, downstream_flank, ref_sequence, alt_sequence) are dropped; all other input
columns are kept, and these are added:
    protein_sequence   protein of the assigned transcript (filled only when status == ok)
    protein_position   1-based residue position;  ref_aa, alt_aa: 1-letter amino acids
    gene_name, transcript_id, protein_id, strand, cds_position, ref_codon, alt_codon,
    protein_length, transcript_source (clinvar / clinvar_other_version / mane_fallback /
    mane), clinvar_name, clinvar_p_agrees, clinvar_status, n_mane_hits, mane_hits, status

status values (ESM1b_ref_alt_contrast.py uses only status == ok):
    ok                          missense on the assigned transcript
    synonymous / stop_gained / stop_lost / start_lost / selenocysteine_codon
    invalid_allele              ref/alt not single A/C/G/T bases
    ClinVar mode:
    not_in_clinvar              variant_id not found among GRCh38 variant_summary rows
    clinvar_locus_mismatch      ClinVar chrom/pos/ref/alt differ from the TSV row
    clinvar_name_unparsed       Name is not an NM_/XM_ c.N X>Y SNV (e.g. genomic-only name)
    clinvar_noncoding_position  Name places the variant outside the CDS (c.-N, c.*N, c.N+M)
    clinvar_allele_mismatch     c. alleles are neither the TSV alleles nor their complements
    transcript_not_in_gbff      named transcript version not in the --gbff files
    cds_incomplete              partial CDS, codon_start != 1, or length not a multiple of 3
    clinvar_cpos_out_of_range   c. position beyond the CDS
    transcript_ref_mismatch     transcript base at c. != c. ref allele
    protein_mismatch            codon translation != protein residue
    clinvar_p_mismatch          computed change disagrees with ClinVar's p. notation
    MANE mode / fallback:
    not_in_mane_cds, genome_ref_mismatch, ref_mismatch, protein_mismatch,
    protein_fasta_mismatch, unknown_chromosome, cds_incomplete
    Given-transcript mode (same as MANE mode, plus):
    transcript_not_in_gtf       the named transcript has no CDS in --gtf
    not_in_transcript_cds       the variant is outside that transcript's CDS
    input_protein_change_mismatch  computed change != --protein_change_column

MANE transcript selection (MANE mode or fallback):
    --gtf Homo_sapiens.GRCh38.104.gtf --transcript_ids MANE.GRCh38.vX.summary.txt.gz
        MANE Select IDs from the NCBI summary, CDS models from Ensembl 104 (IDs matched
        without version; --strict_version keeps exact versions only)
    --gtf MANE.GRCh38.vX.ensembl_genomic.gtf.gz          (tag MANE_Select, the default)
    --gtf <pre-filtered GTF> --transcript_tag all
The genome FASTA must be uncompressed or bgzip-compressed (pyfaidx builds a .fai index).

Requires: pandas (+ pyfaidx for MANE mode / fallback)

Usage (ClinVar transcripts, MANE fallback for the rest):
    python map_variants_to_mane_protein.py \
        --inputs train.tsv val.tsv \
        --variant_summary variant_summary_2025-11.txt.gz \
        --gbff MANE.GRCh38.v1.4.refseq_rna.gbff.gz \
        --gtf Homo_sapiens.GRCh38.104.gtf --transcript_ids MANE.GRCh38.v1.4.summary.txt.gz \
        --genome_fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa
    # writes train_protein.tsv, val_protein.tsv next to the inputs (or --output_dir)

Usage (transcripts already resolved, e.g. MFASS on MANE Select via dbNSFP):
    python map_variants_to_mane_protein.py \
        --inputs MFASS_missense_main.tsv \
        --transcript_column TRANSCRIPT_ID --protein_change_column PROTEIN_CHANGE \
        --gtf MANE.GRCh38.v1.4.ensembl_genomic.gtf.gz \
        --genome_fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa
"""

import re
import sys
import gzip
import argparse
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd


BASES = "TCAG"
AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = {
    a + b + c: AMINO[16 * i + 4 * j + k]
    for i, a in enumerate(BASES) for j, b in enumerate(BASES) for k, c in enumerate(BASES)
}
THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V", "Sec": "U",
    "Ter": "*", "Xaa": "X",
}
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
NAME_RE = re.compile(
    r"^(?P<tx>[NX][MR]_\d+(?:\.\d+)?)(?:\((?P<gene>[^)]*)\))?"
    r":c\.(?P<cpos>[-*]?\d+(?:[+-]\d+)?)(?P<cref>[ACGT])>(?P<calt>[ACGT])"
    r"(?:\s*\(p\.(?P<p>[^)]*)\))?")
P_RE = re.compile(r"^(?P<ref>[A-Z][a-z]{2})(?P<pos>\d+)(?P<alt>[A-Z][a-z]{2}|=|\?)$")
FALLBACK_STATUSES = {"not_in_clinvar", "clinvar_name_unparsed", "transcript_not_in_gbff"}
SEQUENCE_COLUMNS = ["upstream_flank", "downstream_flank", "ref_sequence", "alt_sequence"]
REQUIRED_COLUMNS = ["variant_id", "chromosome", "position", "ref_allele", "alt_allele"]
OUTPUT_COLUMNS = [
    "gene_name", "transcript_id", "protein_id", "strand", "cds_position",
    "ref_codon", "alt_codon", "protein_position", "ref_aa", "alt_aa",
    "protein_length", "protein_sequence", "transcript_source", "clinvar_name",
    "clinvar_p_agrees", "clinvar_status", "n_mane_hits", "mane_hits",
    "input_protein_change_agrees", "status",
]
CLINVAR_COLUMNS = {"#AlleleID", "VariationID", "Name", "GeneSymbol", "Assembly",
                   "Chromosome", "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF"}
BIN_SIZE = 100_000


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def norm_chrom(chrom):
    c = str(chrom).strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    return "MT" if c.upper() in ("M", "MT") else c


def strip_version(identifier):
    return identifier.split(".")[0] if identifier else identifier


def version_number(identifier):
    parts = identifier.split(".")
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0


def with_version(identifier, version):
    if identifier and version and "." not in identifier:
        return f"{identifier}.{version}"
    return identifier


def revcomp(seq):
    return seq.translate(COMPLEMENT)[::-1]


def translate(seq):
    return "".join(CODON_TABLE.get(seq[i:i + 3], "X")
                   for i in range(0, len(seq) - len(seq) % 3, 3))


def valid_snv(ref, alt):
    return len(ref) == 1 and len(alt) == 1 and ref in "ACGT" and alt in "ACGT" and ref != alt


def classify_codon_change(cds, protein, ci, tx_alt):
    """
    cds: coding sequence (transcript strand, stop codon optional); protein: without the
    terminal '*'; ci: 0-based CDS index of the variant; tx_alt: transcript-strand alt base.
    Returns (fields, status).
    """
    k, frame = divmod(ci, 3)
    ref_codon = cds[3 * k:3 * k + 3]
    alt_codon = ref_codon[:frame] + tx_alt + ref_codon[frame + 1:]
    ref_aa, alt_aa = CODON_TABLE.get(ref_codon, "X"), CODON_TABLE.get(alt_codon, "X")
    fields = {"cds_position": ci + 1, "ref_codon": ref_codon, "alt_codon": alt_codon,
              "protein_position": k + 1, "ref_aa": ref_aa, "alt_aa": alt_aa,
              "protein_length": len(protein)}
    if k >= len(protein):                                # variant in the stop codon
        status = "synonymous" if alt_aa == "*" else "stop_lost"
    elif ref_aa == "*":                                  # internal TGA = selenocysteine
        status = "selenocysteine_codon"
        fields["ref_aa"] = "U"
    elif alt_aa == "*":
        status = "stop_gained"
    elif alt_aa == ref_aa:
        status = "synonymous"
    elif k == 0:
        status = "start_lost"
    elif protein[k] != ref_aa:
        status = "protein_mismatch"
    else:
        status = "ok"
    return fields, status


# ============================================================================
# ClinVar variant_summary
# ============================================================================

def clinvar_locus(row):
    try:
        return (norm_chrom(row["Chromosome"]), int(row["PositionVCF"]),
                str(row["ReferenceAlleleVCF"]).upper(), str(row["AlternateAlleleVCF"]).upper())
    except (KeyError, TypeError, ValueError):
        return None


class ClinVarIndex:
    """GRCh38 variant_summary rows for the input variant_ids."""

    def __init__(self, path, loci, id_column="auto", chunksize=500_000):
        ids = set(loci)
        log(f"Reading ClinVar variant_summary (GRCh38 rows for {len(ids):,} input IDs): {path}")
        parts = []
        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize,
                                 usecols=lambda c: c in CLINVAR_COLUMNS):
            chunk = chunk[chunk["Assembly"] == "GRCh38"]
            keep = pd.Series(False, index=chunk.index)
            for col in ("VariationID", "#AlleleID"):
                if col in chunk.columns:
                    keep |= chunk[col].isin(ids)
            parts.append(chunk[keep])
        cv = pd.concat(parts, ignore_index=True)
        self.has_locus = {"Chromosome", "PositionVCF", "ReferenceAlleleVCF",
                          "AlternateAlleleVCF"} <= set(cv.columns)

        choices = {"VariationID": "VariationID", "AlleleID": "#AlleleID"}
        candidates = list(choices.values()) if id_column == "auto" else [choices[id_column]]
        stats = {}
        for col in candidates:
            if col not in cv.columns:
                continue
            sub = cv[cv[col].isin(ids)].drop_duplicates(col)
            agree = (sum(clinvar_locus(r) == loci[r[col]] for r in sub.to_dict("records"))
                     if self.has_locus else None)
            stats[col] = (len(sub), agree)
            log(f"  as {col}: {len(sub):,}/{len(ids):,} IDs found"
                + (f", {agree:,} with matching chrom/pos/ref/alt" if agree is not None else ""))
        if not stats:
            sys.exit("variant_summary has neither VariationID nor #AlleleID columns")
        self.id_column = max(stats, key=lambda c: stats[c][1] if self.has_locus else stats[c][0])
        log(f"  -> variant_id interpreted as {self.id_column}")
        if not self.has_locus:
            log("  WARNING: no PositionVCF columns; ID type chosen by match count only and "
                "loci are not cross-checked")

        sub = cv[cv[self.id_column].isin(ids)]
        n_dup = int(sub[self.id_column].duplicated().sum())
        if n_dup:
            log(f"  {n_dup:,} duplicate GRCh38 rows for the same ID; keeping the first")
        self.rows = sub.drop_duplicates(self.id_column).set_index(self.id_column).to_dict("index")

    def transcripts(self):
        out = set()
        for row in self.rows.values():
            m = NAME_RE.match(str(row.get("Name") or ""))
            if m:
                out.add(m["tx"])
        return out


# ============================================================================
# RefSeq GenBank (gbff)
# ============================================================================

def parse_location(loc):
    """'71..4495', 'join(1..10,20..30)', '<1..>900' → ([(start, end), ...], partial)."""
    loc = loc.replace(" ", "")
    if "complement" in loc or "order" in loc:
        return None, True
    partial = "<" in loc or ">" in loc
    inner = loc[5:-1] if loc.startswith("join(") and loc.endswith(")") else loc
    ranges = []
    for part in inner.replace("<", "").replace(">", "").split(","):
        if ".." in part:
            s, e = part.split("..")
            ranges.append((int(s), int(e)))
        elif part.isdigit():
            ranges.append((int(part), int(part)))
        else:
            return None, True
    return ranges, partial


def parse_genbank_record(lines):
    """First CDS of a GenBank mRNA record → dict(cds, protein, protein_id, gene, complete)."""
    try:
        f_idx = next(i for i, l in enumerate(lines) if l.startswith("FEATURES"))
        o_idx = next(i for i, l in enumerate(lines) if l.startswith("ORIGIN"))
    except StopIteration:
        return None
    seq = "".join(ch for l in lines[o_idx + 1:] for ch in l if ch.isalpha()).upper()

    cds_feature, current = None, None
    for l in lines[f_idx + 1:o_idx]:
        l = l.rstrip("\n")
        if len(l) > 5 and l.startswith("     ") and l[5] != " ":
            if cds_feature is not None:
                break
            current = {"key": l[5:21].strip(), "loc": [l[21:].strip()], "quals": []}
            if current["key"] == "CDS":
                cds_feature = current
        elif current is not None and l.startswith(" " * 21):
            text = l[21:]
            if text.startswith("/"):
                current["quals"].append(text)
            elif current["quals"]:
                current["quals"][-1] += "\n" + text
            else:
                current["loc"].append(text.strip())
    if cds_feature is None:
        return None

    quals = {}
    for q in cds_feature["quals"]:
        key, _, value = q[1:].partition("=")
        value = value.strip()
        if value.startswith('"'):
            value = value[1:-1] if value.endswith('"') and len(value) > 1 else value[1:]
        value = (re.sub(r"\s+", "", value) if key == "translation"
                 else re.sub(r"\s*\n\s*", " ", value))
        quals.setdefault(key, value)

    ranges, partial = parse_location("".join(cds_feature["loc"]))
    cds = "".join(seq[s - 1:e] for s, e in ranges) if ranges else ""
    codon_start = int(quals.get("codon_start", "1") or 1)
    protein = quals.get("translation", "").upper()
    if not protein:
        protein = translate(cds)
        protein = (protein[:-1] if protein.endswith("*") else protein).replace("*", "U")
    return {
        "cds": cds,
        "protein": protein,
        "protein_id": quals.get("protein_id", ""),
        "gene": quals.get("gene", ""),
        "complete": bool(ranges) and not partial and codon_start == 1
                    and len(cds) % 3 == 0 and len(cds) >= 3,
    }


def read_gbff(paths, wanted_accessions):
    """Parse only records whose accession (version-stripped) is wanted."""
    wanted = {strip_version(a) for a in wanted_accessions}
    records = {}
    for path in paths:
        log(f"Reading GenBank records: {path}")
        n_seen = 0
        state, buf, acc = None, [], None
        with open_text(path) as fh:
            for line in fh:
                if state is None:
                    if line.startswith("LOCUS"):
                        state, buf, acc = "pending", [line], None
                    continue
                if line.startswith("//"):
                    n_seen += 1
                    if state == "keep":
                        rec = parse_genbank_record(buf)
                        if rec is not None and acc not in records:
                            records[acc] = rec
                    state, buf = None, []
                    continue
                if state == "skip":
                    continue
                buf.append(line)
                if state == "pending" and line.startswith("VERSION"):
                    acc = line.split()[1]
                    state = "keep" if strip_version(acc) in wanted else "skip"
                    if state == "skip":
                        buf = []
        log(f"  {n_seen:,} records scanned")
    versions = defaultdict(list)
    for acc in records:
        versions[strip_version(acc)].append(acc)
    return records, versions


def annotate_with_clinvar(row, chrom, pos, ref, alt, gbff, versions, has_locus,
                          allow_other_versions=False):
    out = {"transcript_source": "clinvar"}
    if row is None:
        return {**out, "clinvar_status": "not_in_clinvar"}
    name = str(row.get("Name") or "")
    out["clinvar_name"] = name
    out["gene_name"] = str(row.get("GeneSymbol") or "").split(";")[0]
    if has_locus and clinvar_locus(row) != (norm_chrom(chrom), pos, ref, alt):
        return {**out, "clinvar_status": "clinvar_locus_mismatch"}

    m = NAME_RE.match(name)
    if not m:
        return {**out, "clinvar_status": "clinvar_name_unparsed"}
    out["transcript_id"] = m["tx"]
    if not m["cpos"].isdigit():
        return {**out, "clinvar_status": "clinvar_noncoding_position"}
    c, cref, calt = int(m["cpos"]), m["cref"], m["calt"]
    if (cref, calt) == (ref, alt):
        out["strand"] = "+"
    elif (cref, calt) == (ref.translate(COMPLEMENT), alt.translate(COMPLEMENT)):
        out["strand"] = "-"
    else:
        return {**out, "clinvar_status": "clinvar_allele_mismatch"}

    tx = m["tx"]
    rec = gbff.get(tx)
    if rec is None and allow_other_versions and versions.get(strip_version(tx)):
        tx = max(versions[strip_version(tx)], key=version_number)
        rec = gbff[tx]
        out["transcript_source"] = "clinvar_other_version"
        out["transcript_id"] = tx
    if rec is None:
        return {**out, "clinvar_status": "transcript_not_in_gbff"}
    out["protein_id"] = rec["protein_id"]
    if rec["gene"]:
        out["gene_name"] = rec["gene"]
    if not rec["complete"]:
        return {**out, "clinvar_status": "cds_incomplete"}
    cds, protein = rec["cds"], rec["protein"]
    if c > len(cds):
        return {**out, "clinvar_status": "clinvar_cpos_out_of_range"}
    if cds[c - 1] != cref:
        return {**out, "clinvar_status": "transcript_ref_mismatch"}

    fields, status = classify_codon_change(cds, protein, c - 1, calt)
    out.update(fields)
    agrees = None
    pm = P_RE.match((m["p"] or "").strip())
    if pm:
        p_ref = THREE_TO_ONE.get(pm["ref"])
        p_alt = p_ref if pm["alt"] == "=" else THREE_TO_ONE.get(pm["alt"])
        agrees = (int(pm["pos"]) == fields["protein_position"]
                  and p_ref == fields["ref_aa"]
                  and (pm["alt"] == "?" or p_alt == fields["alt_aa"]))
    out["clinvar_p_agrees"] = agrees
    if status == "ok" and agrees is False:
        status = "clinvar_p_mismatch"
    if status == "ok":
        out["protein_sequence"] = protein
    out["clinvar_status"] = status
    return out


# ============================================================================
# MANE Select via GTF + genome (MANE mode and fallback)
# ============================================================================

def read_protein_fasta(path):
    """Protein sequences keyed by every ENSP/ENST ID in the header (version stripped)."""
    proteins, keys, buf = {}, [], []

    def flush():
        if keys:
            seq = "".join(buf).upper().rstrip("*")
            for k in keys:
                proteins[k] = seq

    with open_text(path) as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                keys = [strip_version(k) for k in re.findall(r"ENS[PT]\d+(?:\.\d+)?", line)]
                buf = []
            else:
                buf.append(line.strip())
    flush()
    return proteins


def read_transcript_ids(path, mane_status="MANE Select"):
    """
    ENST IDs to keep, as {unversioned: versioned}. Accepts the NCBI MANE summary file
    (rows with MANE_status == mane_status, column Ensembl_nuc) or a plain list with one
    ID per line (first whitespace-separated field).
    """
    with open_text(path) as fh:
        header = fh.readline().lstrip("#").rstrip("\n").split("\t")
    if "Ensembl_nuc" in header and "MANE_status" in header:
        df = pd.read_csv(path, sep="\t", dtype=str)
        df.columns = [c.lstrip("#") for c in df.columns]
        ids = df.loc[df["MANE_status"] == mane_status, "Ensembl_nuc"].dropna().tolist()
        source = f"MANE summary ({mane_status})"
    else:
        with open_text(path) as fh:
            ids = [ln.split()[0] for ln in fh if ln.strip() and not ln.startswith("#")]
        source = "ID list"
    ids = {strip_version(i): i for i in ids if i.startswith("ENST")}
    log(f"Read {len(ids):,} transcript IDs from {source}: {path}")
    return ids


class MANEAnnotator:
    """Genomic SNV → MANE Select codon / amino-acid change, via GTF CDS models + genome."""

    def __init__(self, gtf_path, genome_fasta, transcript_tag="MANE_Select",
                 protein_fasta=None, transcript_ids=None, strict_version=False):
        from pyfaidx import Fasta
        log(f"Loading genome FASTA: {genome_fasta}")
        self.genome = Fasta(str(genome_fasta), as_raw=True, sequence_always_upper=True)
        self.genome_keys = {norm_chrom(k): k for k in self.genome.keys()}

        if transcript_ids is not None:
            selection = "listed"
        elif transcript_tag.lower() == "all":
            selection = "all"
        else:
            selection = f"'{transcript_tag}'-tagged"
        log(f"Parsing GTF ({selection} transcripts): {gtf_path}")
        self.tx = self._load_gtf(gtf_path, transcript_tag, transcript_ids, strict_version)
        self.index = defaultdict(list)
        self._build_index()
        self.by_stripped = {strip_version(tid): tid for tid in self.tx}
        log(f"  {len(self.tx):,} {selection} transcripts with CDS; "
            f"{sum(len(t['intervals']) for t in self.tx.values()):,} CDS intervals")

        self.fasta_proteins = None
        if protein_fasta:
            log(f"Loading protein FASTA for cross-check: {protein_fasta}")
            self.fasta_proteins = read_protein_fasta(protein_fasta)
        self._seq_cache = {}

    # ---------------------------------------------------------------- GTF
    @staticmethod
    def _load_gtf(gtf_path, transcript_tag, transcript_ids=None, strict_version=False):
        tags = defaultdict(set)
        info = {}
        intervals = defaultdict(list)
        with open_text(gtf_path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 9 or fields[2] not in ("transcript", "CDS", "stop_codon"):
                    continue
                attrs = fields[8]
                m = re.search(r'transcript_id "([^"]+)"', attrs)
                if not m:
                    continue
                tid = m.group(1)
                tags[tid].update(re.findall(r'tag "([^"]+)"', attrs))
                if fields[2] == "transcript":
                    continue
                intervals[tid].append((int(fields[3]), int(fields[4])))
                if tid not in info:
                    a = dict(re.findall(r'(\S+) "([^"]*)"', attrs))
                    info[tid] = {
                        "chrom": norm_chrom(fields[0]),
                        "strand": fields[6],
                        "gene_name": a.get("gene_name") or a.get("gene_id", ""),
                        "transcript_id": with_version(tid, a.get("transcript_version")),
                        "protein_id": with_version(a.get("protein_id", ""),
                                                   a.get("protein_version")),
                    }
                elif not info[tid]["protein_id"] and 'protein_id "' in attrs:
                    a = dict(re.findall(r'(\S+) "([^"]*)"', attrs))
                    info[tid]["protein_id"] = with_version(a.get("protein_id", ""),
                                                           a.get("protein_version"))

        selected = {}
        n_version_match, n_version_diff = 0, 0
        for tid, t in info.items():
            if transcript_ids is not None:
                listed = transcript_ids.get(strip_version(tid))
                if listed is None:
                    continue
                same_version = listed == t["transcript_id"] or "." not in listed
                n_version_match += same_version
                n_version_diff += not same_version
                if strict_version and not same_version:
                    continue
            elif transcript_tag.lower() != "all" and transcript_tag not in tags.get(tid, ()):
                continue
            merged = []
            for s, e in sorted(intervals[tid]):          # merge overlaps (CDS ∪ stop_codon)
                if merged and s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            t["intervals"] = merged                      # ascending genomic order
            selected[tid] = t

        if transcript_ids is not None:
            n_found = n_version_match + n_version_diff
            log(f"  {n_found:,}/{len(transcript_ids):,} listed IDs have a CDS in the GTF "
                f"({n_version_match:,} same version, {n_version_diff:,} different version"
                f"{', dropped (--strict_version)' if strict_version else ''}); "
                f"{len(transcript_ids) - n_found:,} not in the GTF or non-coding there")
        if not selected and transcript_ids is not None:
            sys.exit("None of the listed transcript IDs were found with a CDS in the GTF.")
        if not selected:
            seen = Counter(tag for ts in tags.values() for tag in ts)
            sys.exit(f"No transcripts with tag '{transcript_tag}' found in {gtf_path}. "
                     f"Tags present: {seen.most_common(10)}. For the Ensembl 104 GTF use "
                     f"--transcript_ids <MANE summary>; for a pre-filtered GTF use "
                     f"--transcript_tag all; or use the NCBI MANE GTF (tag MANE_Select).")
        return selected

    def _build_index(self):
        for tid, t in self.tx.items():
            order = t["intervals"] if t["strand"] == "+" else t["intervals"][::-1]
            offset = 0
            for s, e in order:                           # offsets in transcript order
                for b in range(s // BIN_SIZE, e // BIN_SIZE + 1):
                    self.index[(t["chrom"], b)].append((s, e, tid, offset))
                offset += e - s + 1
            t["cds_length"] = offset

    # ----------------------------------------------------------- sequences
    def _sequences(self, tid):
        if tid in self._seq_cache:
            return self._seq_cache[tid]
        t = self.tx[tid]
        key = self.genome_keys.get(t["chrom"])
        if key is None:
            result = None
        else:
            cds = "".join(self.genome[key][s - 1:e] for s, e in t["intervals"])
            if t["strand"] == "-":
                cds = revcomp(cds)
            protein = translate(cds)
            if protein.endswith("*"):
                protein = protein[:-1]
            protein = protein.replace("*", "U")          # internal TGA = selenocysteine
            fasta_ok = None
            if self.fasta_proteins is not None:
                ref_protein = (self.fasta_proteins.get(strip_version(t["protein_id"]))
                               or self.fasta_proteins.get(strip_version(tid)))
                fasta_ok = ref_protein == protein
            result = {"cds": cds, "protein": protein,
                      "complete": len(cds) % 3 == 0, "fasta_ok": fasta_ok}
        self._seq_cache[tid] = result
        return result

    # ----------------------------------------------------------- annotate
    def _annotate_hit(self, tid, start, end, offset, pos, ref, alt):
        t = self.tx[tid]
        out = {"gene_name": t["gene_name"], "transcript_id": t["transcript_id"],
               "protein_id": t["protein_id"], "strand": t["strand"]}
        seqs = self._sequences(tid)
        if seqs is None:
            return {**out, "status": "unknown_chromosome"}
        if not seqs["complete"]:
            return {**out, "status": "cds_incomplete"}

        plus = t["strand"] == "+"
        ci = offset + (pos - start) if plus else offset + (end - pos)
        tx_ref = ref if plus else ref.translate(COMPLEMENT)
        tx_alt = alt if plus else alt.translate(COMPLEMENT)
        cds, protein = seqs["cds"], seqs["protein"]
        if cds[ci] != tx_ref:
            return {**out, "status": "ref_mismatch"}

        fields, status = classify_codon_change(cds, protein, ci, tx_alt)
        out.update(fields)
        if status == "ok" and seqs["fasta_ok"] is False:
            status = "protein_fasta_mismatch"
        if status == "ok":
            out["protein_sequence"] = protein
        out["status"] = status
        return out

    def annotate_transcript(self, chrom, pos, ref, alt, transcript):
        """Map an SNV onto one named transcript (any version of the ID)."""
        ref, alt = str(ref).strip().upper(), str(alt).strip().upper()
        if not valid_snv(ref, alt):
            return {"status": "invalid_allele"}
        tid = self.by_stripped.get(strip_version(str(transcript))) if transcript else None
        if tid is None:
            return {"transcript_id": transcript, "status": "transcript_not_in_gtf"}
        chrom = norm_chrom(chrom)
        key = self.genome_keys.get(chrom)
        if key is None:
            return {"status": "unknown_chromosome"}
        if self.genome[key][pos - 1] != ref:
            return {"status": "genome_ref_mismatch"}
        hits = [(s, e, off) for s, e, t, off in self.index.get((chrom, pos // BIN_SIZE), ())
                if t == tid and s <= pos <= e]
        if not hits:
            t = self.tx[tid]
            return {"gene_name": t["gene_name"], "transcript_id": t["transcript_id"],
                    "protein_id": t["protein_id"], "strand": t["strand"],
                    "status": "not_in_transcript_cds"}
        s, e, off = hits[0]
        return self._annotate_hit(tid, s, e, off, pos, ref, alt)

    def annotate(self, chrom, pos, ref, alt, gene=None):
        ref, alt = str(ref).strip().upper(), str(alt).strip().upper()
        if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT" or ref == alt:
            return {"status": "invalid_allele"}
        chrom = norm_chrom(chrom)
        key = self.genome_keys.get(chrom)
        if key is None:
            return {"status": "unknown_chromosome"}
        if self.genome[key][pos - 1] != ref:
            return {"status": "genome_ref_mismatch"}

        hits = [(s, e, tid, off) for s, e, tid, off in self.index.get((chrom, pos // BIN_SIZE), ())
                if s <= pos <= e]
        if not hits:
            return {"status": "not_in_mane_cds", "n_mane_hits": 0}

        results = [self._annotate_hit(tid, s, e, off, pos, ref, alt) for s, e, tid, off in hits]
        pool = [r for r in results if r["status"] == "ok"] or results
        if gene is not None and not pd.isna(gene):
            same_gene = [r for r in pool if r["gene_name"] == str(gene)]
            pool = same_gene or pool
        chosen = dict(min(pool, key=lambda r: r["transcript_id"]))
        chosen["n_mane_hits"] = len(results)
        if len(results) > 1:
            chosen["mane_hits"] = ";".join(
                f"{r['gene_name']}:{r['transcript_id']}:{r['status']}"
                for r in sorted(results, key=lambda r: r["transcript_id"]))
        return chosen


# ============================================================================
# Driver
# ============================================================================

class VariantMapper:
    def __init__(self, clinvar=None, gbff=None, versions=None, mane=None,
                 allow_other_versions=False, given_transcripts=False):
        self.clinvar, self.gbff, self.versions = clinvar, gbff or {}, versions or {}
        self.mane = mane
        self.allow_other_versions = allow_other_versions
        self.given_transcripts = given_transcripts

    def annotate(self, vid, chrom, pos, ref, alt, gene=None, transcript=None):
        ref, alt = str(ref).strip().upper(), str(alt).strip().upper()
        if not valid_snv(ref, alt):
            return {"status": "invalid_allele",
                    "clinvar_status": "invalid_allele" if self.clinvar else None}
        if self.given_transcripts:
            tx = None if transcript is None or pd.isna(transcript) else str(transcript)
            return {**self.mane.annotate_transcript(chrom, pos, ref, alt, tx),
                    "transcript_source": "input_column"}
        if self.clinvar is None:
            return {**self.mane.annotate(chrom, pos, ref, alt, gene),
                    "transcript_source": "mane"}

        res = annotate_with_clinvar(self.clinvar.rows.get(str(vid)), chrom, pos, ref, alt,
                                    self.gbff, self.versions, self.clinvar.has_locus,
                                    self.allow_other_versions)
        if res["clinvar_status"] in FALLBACK_STATUSES and self.mane is not None:
            fb = self.mane.annotate(chrom, pos, ref, alt, gene or res.get("gene_name") or None)
            return {**fb, "transcript_source": "mane_fallback",
                    "clinvar_name": res.get("clinvar_name"),
                    "clinvar_status": res["clinvar_status"]}
        return {**res, "status": res["clinvar_status"]}


def collect_input_loci(paths, chunksize):
    loci = {}
    for path in paths:
        for chunk in pd.read_csv(path, sep="\t", usecols=REQUIRED_COLUMNS, dtype=str,
                                 chunksize=chunksize):
            for vid, c, p, r, a in zip(chunk["variant_id"], chunk["chromosome"],
                                       chunk["position"], chunk["ref_allele"],
                                       chunk["alt_allele"]):
                loci[str(vid)] = (norm_chrom(c), int(float(p)), str(r).upper(), str(a).upper())
    return loci


def collect_input_transcripts(paths, column, chunksize):
    ids = set()
    for path in paths:
        for chunk in pd.read_csv(path, sep="\t", usecols=[column], dtype=str, chunksize=chunksize):
            ids.update(strip_version(t) for t in chunk[column].dropna())
    log(f"{len(ids):,} distinct transcripts in column '{column}'")
    return {t: t for t in ids}


def protein_change_agrees(row, given):
    if given is None or pd.isna(given) or pd.isna(row.get("protein_position")):
        return None
    return f"{row['ref_aa']}{int(row['protein_position'])}{row['alt_aa']}" == str(given).strip()


def process_file(path, mapper, output_path, gene_column=None, chunksize=20000,
                 transcript_column=None, protein_change_column=None):
    log(f"\nProcessing {path}")
    out_chunks = []
    n_seq, n_seq_mismatch = 0, 0
    upstream_lengths = Counter()
    warned = False

    reader = pd.read_csv(path, sep="\t", chunksize=chunksize,
                         dtype={"variant_id": str, "chromosome": str,
                                "ref_allele": str, "alt_allele": str})
    for chunk in reader:
        missing = set(REQUIRED_COLUMNS) - set(chunk.columns)
        if missing:
            sys.exit(f"{path} is missing columns: {sorted(missing)}")

        # Sanity check of the NT-2 inputs: the variant base should sit at
        # index len(upstream_flank) of ref_sequence (5,999 for an 11,999-bp window).
        if "ref_sequence" in chunk.columns:
            ups = chunk["upstream_flank"] if "upstream_flank" in chunk.columns else None
            for i, (seq, ref) in enumerate(zip(chunk["ref_sequence"], chunk["ref_allele"])):
                if not isinstance(seq, str) or not seq:
                    continue
                up = ups.iloc[i] if ups is not None else None
                idx = len(up) if isinstance(up, str) else len(seq) // 2
                if isinstance(up, str):
                    upstream_lengths[len(up)] += 1
                n_seq += 1
                n_seq_mismatch += idx >= len(seq) or seq[idx].upper() != str(ref).upper()

        chunk = chunk.drop(columns=[c for c in SEQUENCE_COLUMNS if c in chunk.columns])
        previous = [c for c in OUTPUT_COLUMNS if c in chunk.columns]
        if previous:                       # input is an earlier output: re-annotate from scratch
            if not warned:
                log(f"  NOTE: input already has annotation columns {previous[:4]}...; "
                    f"they are replaced")
                warned = True
            chunk = chunk.drop(columns=previous)
        genes = (chunk[gene_column] if gene_column and gene_column in chunk.columns
                 else [None] * len(chunk))
        txs = chunk[transcript_column] if transcript_column else [None] * len(chunk)
        rows = [mapper.annotate(v, c, int(float(p)), r, a, g, t) for v, c, p, r, a, g, t in zip(
            chunk["variant_id"], chunk["chromosome"], chunk["position"],
            chunk["ref_allele"], chunk["alt_allele"], genes, txs)]
        if protein_change_column:
            for row, given in zip(rows, chunk[protein_change_column]):
                agrees = protein_change_agrees(row, given)
                row["input_protein_change_agrees"] = agrees
                if agrees is False and row.get("status") == "ok":
                    row["status"] = "input_protein_change_mismatch"
                    row.pop("protein_sequence", None)
        ann = pd.DataFrame(rows, columns=OUTPUT_COLUMNS, index=chunk.index)
        out_chunks.append(pd.concat([chunk, ann], axis=1))
        log(f"  {sum(len(c) for c in out_chunks):,} variants processed")

    out = pd.concat(out_chunks, ignore_index=True)
    for col in ["cds_position", "protein_position", "protein_length", "n_mane_hits"]:
        out[col] = out[col].astype("Int64")
    out.to_csv(output_path, sep="\t", index=False)

    n = len(out)
    log(f"Wrote {output_path}  ({n:,} rows)")
    log("  final status:")
    for status, k in out["status"].value_counts().items():
        log(f"    {status:28s} {k:>9,}  ({100 * k / n:.2f}%)")
    if out["clinvar_status"].notna().any():
        log("  ClinVar-transcript status (before any MANE fallback):")
        for status, k in out["clinvar_status"].value_counts().items():
            log(f"    {status:28s} {k:>9,}  ({100 * k / n:.2f}%)")
        ok = out["status"] == "ok"
        log(f"  transcript_source among ok rows: "
            f"{out.loc[ok, 'transcript_source'].value_counts().to_dict()}")
        agrees = out["clinvar_p_agrees"].map({True: "agree", False: "disagree"}).fillna("no p.")
        log(f"  ClinVar p. notation vs computed change: {agrees.value_counts().to_dict()}")
    if protein_change_column:
        ag = out["input_protein_change_agrees"].map({True: "agree", False: "disagree"}).fillna("n/a")
        log(f"  computed change vs '{protein_change_column}': {ag.value_counts().to_dict()}")
    n_multi = int((out["n_mane_hits"].fillna(0) > 1).sum())
    if n_multi:
        log(f"  {n_multi:,} variants overlap >1 MANE Select CDS (see mane_hits)")
    for col in ("label", "SDV"):
        if col not in out.columns:
            continue
        ok = out["status"] == "ok"
        log(f"  {col} counts, all rows: {out[col].value_counts().to_dict()}; "
            f"status==ok rows: {out.loc[ok, col].value_counts().to_dict()}")
    if n_seq:
        log(f"  NT-2 ref_sequence check: {n_seq - n_seq_mismatch:,}/{n_seq:,} rows have "
            f"ref_allele at the expected index")
    if upstream_lengths:
        log(f"  upstream_flank lengths: {dict(upstream_lengths.most_common(5))}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Map NT-2 variant TSVs to protein-level TSVs (ClinVar transcript or MANE)")
    parser.add_argument("--inputs", nargs="+", required=True, help="NT-2 variant TSV(s)")
    g = parser.add_argument_group("ClinVar transcripts")
    g.add_argument("--variant_summary", default=None,
                   help="ClinVar variant_summary.txt.gz (e.g. variant_summary_2025-11.txt.gz)")
    g.add_argument("--gbff", nargs="+", default=None,
                   help="RefSeq GenBank RNA file(s), e.g. MANE.GRCh38.v1.4.refseq_rna.gbff.gz "
                        "and the full RefSeq GRCh38 *_rna.gbff.gz")
    g.add_argument("--id_column", default="auto", choices=["auto", "VariationID", "AlleleID"],
                   help="What variant_id holds (default: detect from locus agreement)")
    g.add_argument("--allow_other_versions", action="store_true",
                   help="If the named transcript version is missing, use the highest available "
                        "version of the same accession (still fully checked)")
    m = parser.add_argument_group("MANE Select via genome (MANE mode, or fallback in ClinVar mode)")
    m.add_argument("--gtf", default=None)
    m.add_argument("--genome_fasta", default=None,
                   help="GRCh38 genome FASTA (uncompressed or bgzipped)")
    m.add_argument("--transcript_ids", default=None,
                   help="NCBI MANE summary file or plain ENST ID list selecting transcripts from --gtf")
    m.add_argument("--transcript_tag", default="MANE_Select",
                   help="GTF tag selecting transcripts, or 'all'. Ignored with --transcript_ids")
    m.add_argument("--mane_status", default="MANE Select")
    m.add_argument("--strict_version", action="store_true")
    m.add_argument("--protein_fasta", default=None,
                   help="Optional protein FASTA to cross-check MANE translations")
    m.add_argument("--transcript_column", default=None,
                   help="Input column holding an already-resolved transcript ID per variant "
                        "(e.g. TRANSCRIPT_ID); maps each variant to that transcript instead of "
                        "searching MANE Select. Needs --gtf + --genome_fasta")
    m.add_argument("--protein_change_column", default=None,
                   help="Input column with the expected change (e.g. PROTEIN_CHANGE = P394L) "
                        "to cross-check against (given-transcript mode)")
    m.add_argument("--gene_column", default=None,
                   help="Input column with gene symbols to break MANE ties "
                        "(ClinVar mode uses ClinVar's GeneSymbol by default)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--suffix", default="_protein")
    parser.add_argument("--chunksize", type=int, default=20000)
    args = parser.parse_args()

    if bool(args.variant_summary) != bool(args.gbff):
        parser.error("--variant_summary and --gbff must be given together")
    use_mane = bool(args.gtf and args.genome_fasta)
    if not args.variant_summary and not use_mane:
        parser.error("give --variant_summary + --gbff, and/or --gtf + --genome_fasta")
    if args.transcript_column and (args.variant_summary or not use_mane):
        parser.error("--transcript_column needs --gtf + --genome_fasta and no --variant_summary")

    clinvar, gbff, versions = None, None, None
    if args.variant_summary:
        loci = collect_input_loci(args.inputs, args.chunksize)
        clinvar = ClinVarIndex(args.variant_summary, loci, args.id_column)
        named = clinvar.transcripts()
        gbff, versions = read_gbff(args.gbff, named)
        exact = sum(t in gbff for t in named)
        other = sum(t not in gbff and strip_version(t) in versions for t in named)
        log(f"ClinVar-named transcripts: {len(named):,}; {exact:,} found with the exact version, "
            f"{other:,} only with another version"
            f"{' (used)' if args.allow_other_versions else ' (see --allow_other_versions)'}, "
            f"{len(named) - exact - other:,} absent from the gbff file(s)")

    mane = None
    if use_mane:
        if args.transcript_column:
            transcript_ids = collect_input_transcripts(args.inputs, args.transcript_column,
                                                       args.chunksize)
        else:
            transcript_ids = (read_transcript_ids(args.transcript_ids, args.mane_status)
                              if args.transcript_ids else None)
        mane = MANEAnnotator(args.gtf, args.genome_fasta, args.transcript_tag,
                             args.protein_fasta, transcript_ids, args.strict_version)
        if mane.fasta_proteins is not None:
            checked = [mane._sequences(tid) for tid in mane.tx]
            n_bad = sum(1 for s in checked if s and s["fasta_ok"] is False)
            log(f"Protein FASTA cross-check: {len(checked) - n_bad:,}/{len(checked):,} "
                f"MANE translations match")
    if clinvar is not None:
        log("MANE fallback " + ("enabled" if mane else "disabled (no --gtf/--genome_fasta)"))

    mapper = VariantMapper(clinvar, gbff, versions, mane, args.allow_other_versions,
                           given_transcripts=bool(args.transcript_column))
    for path in args.inputs:
        path = Path(path)
        out_dir = Path(args.output_dir) if args.output_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{path.stem}{args.suffix}.tsv"
        if output_path.resolve() == path.resolve():
            sys.exit(f"Output would overwrite the input {path}; use --output_dir or --suffix")
        process_file(path, mapper, output_path, args.gene_column, args.chunksize,
                     args.transcript_column, args.protein_change_column)


if __name__ == "__main__":
    main()
