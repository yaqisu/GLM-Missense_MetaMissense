#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, mannwhitneyu
try:
    import pysam
except ImportError:
    pysam = None

# Keep PDF text editable in Illustrator as text rather than paths where possible.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["axes.unicode_minus"] = False


# -----------------------------
# Constants and column candidates
# -----------------------------

METHOD_COL_DEFAULT = "GLM-Missense_correct_le2"

LABEL_TO_NAME = {0: "Benign", 1: "Pathogenic"}
STRATA = [0, 1]

BASES = {"A", "C", "G", "T"}
PYRIMIDINES = {"C", "T"}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
SBS6_TYPES = {"C>A", "C>G", "C>T", "T>A", "T>C", "T>G"}

COORD_CANDIDATES = {
    "chrom": ["chromosome", "chrom", "chr", "CHROM", "#CHROM"],
    "pos": ["position", "pos", "pos(1-based)", "POS"],
    "ref": ["ref_allele", "ref", "REF"],
    "alt": ["alt_allele", "alt", "ALT"],
}

LABEL_CANDIDATES = ["true_label", "label"]
AF_CANDIDATES = ["gnomAD4.1_joint_AF", "gnomAD4_1_joint_AF", "gnomAD_joint_AF"]
LOEUF_CANDIDATES = ["lof.oe_ci.upper", "LOEUF", "loeuf"]
SPLICEAI_CANDIDATES = ["spliceai_DS_max", "SpliceAI_DS_max", "spliceAI_DS_max"]

AA_REF_CANDIDATES = [
    "aaref", "aaref_x", "aaref_y",
    "aa_ref", "aa_ref_x", "aa_ref_y",
    "ref_aa", "ref_aa_x", "ref_aa_y",
    "protein_ref", "protein_ref_x", "protein_ref_y",
    "AA_ref", "AA_ref_x", "AA_ref_y",
]

AA_ALT_CANDIDATES = [
    "aaalt", "aaalt_x", "aaalt_y",
    "aa_alt", "aa_alt_x", "aa_alt_y",
    "alt_aa", "alt_aa_x", "alt_aa_y",
    "protein_alt", "protein_alt_x", "protein_alt_y",
    "AA_alt", "AA_alt_x", "AA_alt_y",
]

STANDARD_AAS = set("ARNDCQEGHILKMFPSTWYV")

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

_GRANTHAM_RAW = {
    ("A", "C"): 195, ("A", "D"): 126, ("A", "E"): 107, ("A", "F"): 113, ("A", "G"): 60,
    ("A", "H"): 86, ("A", "I"): 94, ("A", "K"): 106, ("A", "L"): 96, ("A", "M"): 84,
    ("A", "N"): 111, ("A", "P"): 27, ("A", "Q"): 91, ("A", "R"): 112, ("A", "S"): 99,
    ("A", "T"): 58, ("A", "V"): 64, ("A", "W"): 148, ("A", "Y"): 112,
    ("C", "D"): 154, ("C", "E"): 170, ("C", "F"): 205, ("C", "G"): 159, ("C", "H"): 174,
    ("C", "I"): 198, ("C", "K"): 202, ("C", "L"): 198, ("C", "M"): 196, ("C", "N"): 139,
    ("C", "P"): 169, ("C", "Q"): 154, ("C", "R"): 180, ("C", "S"): 112, ("C", "T"): 149,
    ("C", "V"): 192, ("C", "W"): 215, ("C", "Y"): 194,
    ("D", "E"): 45, ("D", "F"): 177, ("D", "G"): 94, ("D", "H"): 81, ("D", "I"): 168,
    ("D", "K"): 101, ("D", "L"): 172, ("D", "M"): 160, ("D", "N"): 23, ("D", "P"): 108,
    ("D", "Q"): 61, ("D", "R"): 96, ("D", "S"): 65, ("D", "T"): 85, ("D", "V"): 152,
    ("D", "W"): 181, ("D", "Y"): 160,
    ("E", "F"): 140, ("E", "G"): 98, ("E", "H"): 40, ("E", "I"): 134, ("E", "K"): 56,
    ("E", "L"): 138, ("E", "M"): 126, ("E", "N"): 42, ("E", "P"): 93, ("E", "Q"): 29,
    ("E", "R"): 54, ("E", "S"): 80, ("E", "T"): 65, ("E", "V"): 121, ("E", "W"): 152,
    ("E", "Y"): 122,
    ("F", "G"): 153, ("F", "H"): 100, ("F", "I"): 21, ("F", "K"): 102, ("F", "L"): 22,
    ("F", "M"): 28, ("F", "N"): 158, ("F", "P"): 114, ("F", "Q"): 116, ("F", "R"): 97,
    ("F", "S"): 155, ("F", "T"): 103, ("F", "V"): 50, ("F", "W"): 40, ("F", "Y"): 22,
    ("G", "H"): 98, ("G", "I"): 135, ("G", "K"): 127, ("G", "L"): 138, ("G", "M"): 127,
    ("G", "N"): 80, ("G", "P"): 42, ("G", "Q"): 87, ("G", "R"): 125, ("G", "S"): 56,
    ("G", "T"): 59, ("G", "V"): 109, ("G", "W"): 184, ("G", "Y"): 147,
    ("H", "I"): 94, ("H", "K"): 32, ("H", "L"): 99, ("H", "M"): 87, ("H", "N"): 68,
    ("H", "P"): 77, ("H", "Q"): 24, ("H", "R"): 29, ("H", "S"): 89, ("H", "T"): 47,
    ("H", "V"): 84, ("H", "W"): 115, ("H", "Y"): 83,
    ("I", "K"): 102, ("I", "L"): 5, ("I", "M"): 10, ("I", "N"): 149, ("I", "P"): 95,
    ("I", "Q"): 109, ("I", "R"): 97, ("I", "S"): 142, ("I", "T"): 89, ("I", "V"): 29,
    ("I", "W"): 61, ("I", "Y"): 33,
    ("K", "L"): 107, ("K", "M"): 95, ("K", "N"): 94, ("K", "P"): 103, ("K", "Q"): 53,
    ("K", "R"): 26, ("K", "S"): 121, ("K", "T"): 78, ("K", "V"): 97, ("K", "W"): 110,
    ("K", "Y"): 85,
    ("L", "M"): 15, ("L", "N"): 153, ("L", "P"): 98, ("L", "Q"): 113, ("L", "R"): 102,
    ("L", "S"): 145, ("L", "T"): 92, ("L", "V"): 32, ("L", "W"): 61, ("L", "Y"): 36,
    ("M", "N"): 142, ("M", "P"): 87, ("M", "Q"): 101, ("M", "R"): 91, ("M", "S"): 135,
    ("M", "T"): 81, ("M", "V"): 21, ("M", "W"): 67, ("M", "Y"): 36,
    ("N", "P"): 91, ("N", "Q"): 46, ("N", "R"): 86, ("N", "S"): 46, ("N", "T"): 65,
    ("N", "V"): 133, ("N", "W"): 174, ("N", "Y"): 143,
    ("P", "Q"): 76, ("P", "R"): 103, ("P", "S"): 74, ("P", "T"): 38, ("P", "V"): 68,
    ("P", "W"): 147, ("P", "Y"): 110,
    ("Q", "R"): 43, ("Q", "S"): 68, ("Q", "T"): 42, ("Q", "V"): 96, ("Q", "W"): 130,
    ("Q", "Y"): 99,
    ("R", "S"): 110, ("R", "T"): 71, ("R", "V"): 96, ("R", "W"): 101, ("R", "Y"): 77,
    ("S", "T"): 58, ("S", "V"): 124, ("S", "W"): 177, ("S", "Y"): 144,
    ("T", "V"): 69, ("T", "W"): 128, ("T", "Y"): 92,
    ("V", "W"): 88, ("V", "Y"): 55,
    ("W", "Y"): 37,
}


@dataclass
class CoordinateColumns:
    chrom: str
    pos: str
    ref: str
    alt: str


# -----------------------------
# Utilities
# -----------------------------

def pick_col(
    df: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
    what: str = "column",
) -> Optional[str]:
    """Pick the first candidate column present in df, with case-insensitive fallback."""
    for c in candidates:
        if c in df.columns:
            return c

    lower_to_actual = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_to_actual:
            return lower_to_actual[c.lower()]

    if required:
        preview = ", ".join(map(str, df.columns[:80]))
        raise ValueError(
            f"Could not find {what}. Tried: {', '.join(candidates)}\n"
            f"Available columns include: {preview} ..."
        )
    return None


def resolve_coord_cols(df: pd.DataFrame) -> CoordinateColumns:
    return CoordinateColumns(
        chrom=pick_col(df, COORD_CANDIDATES["chrom"], what="chromosome column"),
        pos=pick_col(df, COORD_CANDIDATES["pos"], what="position column"),
        ref=pick_col(df, COORD_CANDIDATES["ref"], what="reference allele column"),
        alt=pick_col(df, COORD_CANDIDATES["alt"], what="alternate allele column"),
    )


def parse_bool_series(s: pd.Series) -> pd.Series:
    """Robustly parse boolean/int/string subset columns."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(float) != 0
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "t", "1", "yes", "y"})
    )


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan).dropna()


def get_stratum(df: pd.DataFrame, label_value: int) -> pd.DataFrame:
    return df.loc[df["_true_label"] == label_value].copy()


def format_p(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.3g}"


def significance_stars(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def safe_fisher(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float]:
    if n1 <= 0 or n2 <= 0:
        return np.nan, np.nan
    table = np.array([[k1, n1 - k1], [k2, n2 - k2]], dtype=int)
    try:
        odds_ratio, p = fisher_exact(table, alternative="two-sided")
    except Exception:
        odds_ratio, p = np.nan, np.nan
    return odds_ratio, p


def safe_mannwhitney(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    try:
        u, p = mannwhitneyu(x, y, alternative="two-sided")
    except Exception:
        u, p = np.nan, np.nan
    return u, p


# -----------------------------
# CpG/SBS context functions
# -----------------------------

def clean_allele(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def candidate_fasta_chroms(chrom: str) -> List[str]:
    chrom = str(chrom).strip()
    if chrom.endswith(".0"):
        chrom = chrom[:-2]
    candidates = [chrom]
    if chrom.lower().startswith("chr"):
        candidates.append(chrom[3:])
    else:
        candidates.append("chr" + chrom)
    if chrom in {"M", "MT", "chrM", "chrMT"}:
        candidates.extend(["MT", "M", "chrM", "chrMT"])

    out = []
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def fetch_trinuc(fasta: pysam.FastaFile, chrom: str, pos_1based: int) -> Optional[str]:
    if pos_1based <= 1:
        return None
    for c in candidate_fasta_chroms(chrom):
        try:
            seq = fasta.fetch(c, pos_1based - 2, pos_1based + 1).upper()
            if len(seq) == 3:
                return seq
        except Exception:
            continue
    return None


def reverse_complement_seq(seq: str) -> str:
    return "".join(COMPLEMENT[b] for b in reversed(seq))


def canonical_context_and_substitution(trinuc: str, ref: str, alt: str) -> Optional[Tuple[str, str]]:
    """Return canonical pyrimidine trinucleotide context and SBS substitution."""
    if len(trinuc) != 3 or any(b not in BASES for b in trinuc):
        return None
    if len(ref) != 1 or len(alt) != 1 or ref not in BASES or alt not in BASES or ref == alt:
        return None

    if ref in PYRIMIDINES:
        context = trinuc
        sub = f"{ref}>{alt}"
    else:
        context = reverse_complement_seq(trinuc)
        sub = f"{COMPLEMENT[ref]}>{COMPLEMENT[alt]}"

    if sub not in SBS6_TYPES:
        return None
    return context, sub


def is_cpg_context(context: str, sub: str) -> bool:
    """CpG after pyrimidine canonicalization. For C>* substitutions, CpG = NCG."""
    return sub.startswith("C>") and len(context) == 3 and context[1] == "C" and context[2] == "G"


def add_cpg_flags(
    df: pd.DataFrame,
    cols: CoordinateColumns,
    fasta: pysam.FastaFile,
    *,
    strict_ref_match: bool = True,
    max_mismatch_warnings: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy().reset_index(drop=True)

    valid_context = []
    is_ct = []
    is_cpg = []
    is_ct_cpg = []

    qc = {
        "n_rows": len(out),
        "n_valid_context": 0,
        "n_non_snv_or_bad_allele": 0,
        "n_fetch_fail_or_N": 0,
        "n_ref_mismatch": 0,
    }
    mismatch_examples = []

    for _, row in out.iterrows():
        chrom = row[cols.chrom]
        pos = pd.to_numeric(row[cols.pos], errors="coerce")
        ref = clean_allele(row[cols.ref])
        alt = clean_allele(row[cols.alt])

        if pd.isna(pos) or len(ref) != 1 or len(alt) != 1 or ref not in BASES or alt not in BASES:
            valid_context.append(False); is_ct.append(False); is_cpg.append(False); is_ct_cpg.append(False)
            qc["n_non_snv_or_bad_allele"] += 1
            continue

        trinuc = fetch_trinuc(fasta, str(chrom), int(pos))
        if trinuc is None or len(trinuc) != 3 or "N" in trinuc:
            valid_context.append(False); is_ct.append(False); is_cpg.append(False); is_ct_cpg.append(False)
            qc["n_fetch_fail_or_N"] += 1
            continue

        if trinuc[1] != ref:
            qc["n_ref_mismatch"] += 1
            if len(mismatch_examples) < max_mismatch_warnings:
                mismatch_examples.append(f"{chrom}:{int(pos)} genome={trinuc[1]} file_ref={ref}")
            if strict_ref_match:
                valid_context.append(False); is_ct.append(False); is_cpg.append(False); is_ct_cpg.append(False)
                continue

        canonical = canonical_context_and_substitution(trinuc, ref, alt)
        if canonical is None:
            valid_context.append(False); is_ct.append(False); is_cpg.append(False); is_ct_cpg.append(False)
            qc["n_non_snv_or_bad_allele"] += 1
            continue

        context, sub = canonical
        ct = sub == "C>T"
        cpg = is_cpg_context(context, sub)

        valid_context.append(True)
        is_ct.append(ct)
        is_cpg.append(cpg)
        is_ct_cpg.append(ct and cpg)
        qc["n_valid_context"] += 1

    out["_valid_context"] = valid_context
    out["_is_ct"] = is_ct
    out["_is_cpg"] = is_cpg
    out["_is_ct_cpg"] = is_ct_cpg

    if mismatch_examples:
        print("[WARN] Reference mismatches observed. First examples:", file=sys.stderr)
        for ex in mismatch_examples:
            print(f"       {ex}", file=sys.stderr)
        if strict_ref_match:
            print("       These variants were skipped for CpG analysis. Use --allow-ref-mismatch to keep them.", file=sys.stderr)

    return out, qc



# -----------------------------
# Amino-acid change / Grantham functions
# -----------------------------

def normalize_aa(x) -> Optional[str]:
    """Return one-letter amino-acid code for standard amino acids, otherwise None."""
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if not s:
        return None
    if s in {"*", "X", "TER", "STOP", "NA", "NAN", ".", "-"}:
        return None
    if len(s) == 1 and s in STANDARD_AAS:
        return s
    if len(s) == 3 and s in AA3_TO_1:
        return AA3_TO_1[s]
    return None


def grantham_distance(aa1, aa2) -> Optional[float]:
    """Symmetric Grantham distance for two standard amino acids."""
    a = normalize_aa(aa1)
    b = normalize_aa(aa2)
    if a is None or b is None:
        return None
    if a == b:
        return 0.0
    return float(_GRANTHAM_RAW.get((a, b), _GRANTHAM_RAW.get((b, a), np.nan)))


def add_aa_change_flags(
    df: pd.DataFrame,
    aa_ref_col: str,
    aa_alt_col: str,
    *,
    radical_threshold: float = 150.0,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Add Grantham distance and radical-change flags to a dataframe."""
    out = df.copy()
    distances = []
    valid = []
    radical = []

    for ref_aa, alt_aa in zip(out[aa_ref_col], out[aa_alt_col]):
        r = normalize_aa(ref_aa)
        a = normalize_aa(alt_aa)
        if r is None or a is None:
            distances.append(np.nan)
            valid.append(False)
            radical.append(False)
            continue
        d = grantham_distance(r, a)
        if d is None or pd.isna(d) or r == a:
            distances.append(np.nan if d is None else d)
            valid.append(False)
            radical.append(False)
            continue
        distances.append(float(d))
        valid.append(True)
        radical.append(bool(d > radical_threshold))

    out["_grantham"] = distances
    out["_valid_aa_change"] = valid
    out["_is_radical_aa_change"] = radical

    qc = {
        "aa_ref_col": aa_ref_col,
        "aa_alt_col": aa_alt_col,
        "radical_threshold": radical_threshold,
        "n_rows": len(out),
        "n_valid_aa_change": int(np.sum(valid)),
        "n_radical_aa_change": int(np.sum(radical)),
        "n_invalid_or_non_missense": int(len(out) - np.sum(valid)),
    }
    return out, qc

# -----------------------------
# Plotting functions
# -----------------------------

def annotate_pairwise_p(ax, x_pos: float, y: float, p: float, extra: Optional[str] = None) -> None:
    text = f"{significance_stars(p)}\np={format_p(p)}"
    if extra:
        text += f"\n{extra}"
    ax.text(x_pos, y, text, ha="center", va="bottom", fontsize=7)


def add_panel_label(ax, letter: str) -> None:
    ax.text(
        -0.16,
        1.10,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )


def draw_two_group_boxplot(
    ax,
    subset_df: pd.DataFrame,
    background_df: pd.DataFrame,
    *,
    value_col: str,
    ylabel: str,
    title: str,
    transform=None,
    subset_color=None,
    bg_color=None,
) -> None:
    """Paired boxplots for subset/background within Benign and Pathogenic strata on a provided axis."""
    labels = [LABEL_TO_NAME[s] for s in STRATA]
    x_centers = np.arange(len(STRATA))
    offset = 0.18
    width = 0.30

    data = []
    positions = []
    group_tags = []
    pvals = []

    for i, label_value in enumerate(STRATA):
        g = numeric_series(get_stratum(subset_df, label_value), value_col)
        b = numeric_series(get_stratum(background_df, label_value), value_col)
        if transform is not None:
            g = transform(g)
            b = transform(b)
            g = pd.Series(g).replace([np.inf, -np.inf], np.nan).dropna()
            b = pd.Series(b).replace([np.inf, -np.inf], np.nan).dropna()

        gx = g.to_numpy(dtype=float)
        bx = b.to_numpy(dtype=float)
        data.append(gx if len(gx) else np.array([np.nan]))
        data.append(bx if len(bx) else np.array([np.nan]))
        positions.extend([x_centers[i] - offset, x_centers[i] + offset])
        group_tags.extend(["subset", "background"])
        _, p = safe_mannwhitney(gx, bx)
        pvals.append(p)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 3.5},
    )

    if subset_color is None or bg_color is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        subset_color = default_colors[0]
        bg_color = default_colors[1]

    for patch, tag in zip(bp["boxes"], group_tags):
        patch.set_facecolor(subset_color if tag == "subset" else bg_color)
        patch.set_alpha(0.65)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax > ymin else 1.0
    ax.set_ylim(ymin, ymax + yrange * 0.22)
    for i, p in enumerate(pvals):
        annotate_pairwise_p(ax, x_centers[i], ymax + yrange * 0.04, p)


def proportion_summary(
    subset_df: pd.DataFrame,
    background_df: pd.DataFrame,
    *,
    metric_col: str,
    denominator_filter,
) -> pd.DataFrame:
    rows = []
    for label_value in STRATA:
        g = get_stratum(subset_df, label_value)
        b = get_stratum(background_df, label_value)
        g_den = g.loc[denominator_filter(g)].copy()
        b_den = b.loc[denominator_filter(b)].copy()

        n_g = int(len(g_den))
        n_b = int(len(b_den))
        k_g = int(g_den[metric_col].sum()) if n_g else 0
        k_b = int(b_den[metric_col].sum()) if n_b else 0
        ci_g = wilson_ci(k_g, n_g)
        ci_b = wilson_ci(k_b, n_b)
        or_, p = safe_fisher(k_g, n_g, k_b, n_b)
        rows.append({
            "label": LABEL_TO_NAME[label_value],
            "subset_n": n_g,
            "subset_count": k_g,
            "subset_fraction": k_g / n_g if n_g else np.nan,
            "subset_ci_low": ci_g[0],
            "subset_ci_high": ci_g[1],
            "background_n": n_b,
            "background_count": k_b,
            "background_fraction": k_b / n_b if n_b else np.nan,
            "background_ci_low": ci_b[0],
            "background_ci_high": ci_b[1],
            "odds_ratio": or_,
            "fisher_p": p,
        })
    return pd.DataFrame(rows)


def draw_two_group_barplot(
    ax,
    summary: pd.DataFrame,
    *,
    ylabel: str,
    title: str,
    subset_color=None,
    bg_color=None,
) -> None:
    labels = [LABEL_TO_NAME[s] for s in STRATA]
    summary = summary.set_index("label").reindex(labels).reset_index()

    x = np.arange(len(labels))
    width = 0.34
    subset_frac = summary["subset_fraction"].to_numpy(dtype=float)
    bg_frac = summary["background_fraction"].to_numpy(dtype=float)

    subset_yerr = np.vstack([
        subset_frac - summary["subset_ci_low"].to_numpy(dtype=float),
        summary["subset_ci_high"].to_numpy(dtype=float) - subset_frac,
    ])
    bg_yerr = np.vstack([
        bg_frac - summary["background_ci_low"].to_numpy(dtype=float),
        summary["background_ci_high"].to_numpy(dtype=float) - bg_frac,
    ])

    if subset_color is None or bg_color is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        subset_color = default_colors[0]
        bg_color = default_colors[1]

    ax.bar(x - width / 2, subset_frac, width, yerr=subset_yerr, capsize=3, label="GLM-Missense le2 subset", color=subset_color, alpha=0.85)
    ax.bar(x + width / 2, bg_frac, width, yerr=bg_yerr, capsize=3, label="Background", color=bg_color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    valid_vals = np.concatenate([subset_frac[~np.isnan(subset_frac)], bg_frac[~np.isnan(bg_frac)]])
    upper = np.max(valid_vals) if len(valid_vals) else 0.05
    ax.set_ylim(0, max(0.05, upper * 1.65))
    ymax = ax.get_ylim()[1]
    for i, row in summary.iterrows():
        if pd.isna(row["subset_fraction"]) or pd.isna(row["background_fraction"]):
            continue
        or_ = row["odds_ratio"]
        extra = f"OR={or_:.2g}" if not pd.isna(or_) else None
        y = max(row["subset_fraction"], row["background_fraction"]) + ymax * 0.07
        annotate_pairwise_p(ax, i, y, row["fisher_p"], extra)


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--merged-set", required=True, help="Merged TSV containing GLM-Missense_correct_le2 and feature columns")
    p.add_argument("--ref", default=None, help="Unused in this radical-AA version; kept for backward compatibility")
    p.add_argument("--out-prefix", default="glm_missense_le2", help="Output prefix for the combined PDF figure")

    p.add_argument("--method-col", default=METHOD_COL_DEFAULT, help="Boolean subset column for GLM-Missense unique-correct set")
    p.add_argument(
        "--background-mode",
        choices=["remaining", "full"],
        default="remaining",
        help="remaining = all variants excluding GLM subset; full = literal full set including GLM subset",
    )

    p.add_argument("--label-col", default=None, help="Override true label column; expects 0=benign, 1=pathogenic")
    p.add_argument("--af-col", default=None, help="Override AF column")
    p.add_argument("--loeuf-col", default=None, help="Override LOEUF column")
    p.add_argument("--spliceai-col", default=None, help="Override spliceAI DS max column")
    p.add_argument("--aaref-col", default=None, help="Override reference amino-acid column, e.g. aaref")
    p.add_argument("--aaalt-col", default=None, help="Override alternate amino-acid column, e.g. aaalt")
    p.add_argument(
        "--radical-threshold",
        type=float,
        default=150.0,
        help="Classify amino-acid changes with Grantham distance greater than this threshold as radical",
    )
    p.add_argument("--af-pseudocount", type=float, default=1e-8, help="Pseudocount for log10(AF + pseudocount)")

    p.add_argument("--chrom-col", default=None, help="Override chromosome column")
    p.add_argument("--pos-col", default=None, help="Override 1-based position column")
    p.add_argument("--ref-col", default=None, help="Override reference allele column")
    p.add_argument("--alt-col", default=None, help="Override alternate allele column")
    p.add_argument(
        "--allow-ref-mismatch",
        action="store_true",
        help="Keep variants whose file REF does not match FASTA base. Default skips them for CpG panel.",
    )
    p.add_argument("--fig-width", type=float, default=8.6, help="Combined figure width in inches")
    p.add_argument("--fig-height", type=float, default=7.0, help="Combined figure height in inches")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_prefix = str(args.out_prefix)
    out_dir = Path(out_prefix).parent
    if str(out_dir) not in {"", "."}:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Reading merged TSV...")
    df = pd.read_csv(args.merged_set, sep="\t", low_memory=False)
    print(f"[INFO] Rows: {len(df):,}; columns: {len(df.columns):,}")

    if args.method_col not in df.columns:
        raise ValueError(f"Subset column not found: {args.method_col}")
    subset_mask = parse_bool_series(df[args.method_col])
    subset = df.loc[subset_mask].copy()
    background = df.copy() if args.background_mode == "full" else df.loc[~subset_mask].copy()
    print(f"[INFO] GLM subset rows: {len(subset):,}")
    print(f"[INFO] Background rows: {len(background):,} ({args.background_mode})")

    label_col = args.label_col or pick_col(df, LABEL_CANDIDATES, what="true label column")
    for x in (subset, background):
        x["_true_label"] = pd.to_numeric(x[label_col], errors="coerce")
    print(f"[INFO] Label column: {label_col} (0=Benign, 1=Pathogenic)")

    af_col = args.af_col or pick_col(df, AF_CANDIDATES, what="gnomAD4.1 joint AF column")
    loeuf_col = args.loeuf_col or pick_col(df, LOEUF_CANDIDATES, what="LOEUF column")
    spliceai_col = args.spliceai_col or pick_col(df, SPLICEAI_CANDIDATES, what="spliceAI DS max column")
    print(f"[INFO] AF column: {af_col}")
    print(f"[INFO] LOEUF column: {loeuf_col}")
    print(f"[INFO] spliceAI column: {spliceai_col}")

    # Radical amino-acid-change panel data.
    aa_ref_col = args.aaref_col or pick_col(df, AA_REF_CANDIDATES, what="reference amino-acid column")
    aa_alt_col = args.aaalt_col or pick_col(df, AA_ALT_CANDIDATES, what="alternate amino-acid column")
    print(f"[INFO] amino-acid columns: ref={aa_ref_col}; alt={aa_alt_col}")
    subset_aa, qc_subset_aa = add_aa_change_flags(
        subset,
        aa_ref_col,
        aa_alt_col,
        radical_threshold=args.radical_threshold,
    )
    background_aa, qc_bg_aa = add_aa_change_flags(
        background,
        aa_ref_col,
        aa_alt_col,
        radical_threshold=args.radical_threshold,
    )
    print(f"[INFO] AA-change QC subset: {qc_subset_aa}")
    print(f"[INFO] AA-change QC background: {qc_bg_aa}")

    radical_aa_summary = proportion_summary(
        subset_aa,
        background_aa,
        metric_col="_is_radical_aa_change",
        denominator_filter=lambda d: d["_valid_aa_change"].astype(bool),
    )

    # Combined figure.
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    subset_color = colors[0]
    bg_color = colors[1]

    fig, axes = plt.subplots(2, 2, figsize=(args.fig_width, args.fig_height))
    axes_flat = axes.ravel()

    af_transform = lambda s: np.log10(np.maximum(pd.to_numeric(s, errors="coerce"), 0) + args.af_pseudocount)
    draw_two_group_boxplot(
        axes_flat[0],
        subset,
        background,
        value_col=af_col,
        ylabel=f"log10(AF + {args.af_pseudocount:g})",
        title="Allele frequency",
        transform=af_transform,
        subset_color=subset_color,
        bg_color=bg_color,
    )
    add_panel_label(axes_flat[0], "A")

    draw_two_group_boxplot(
        axes_flat[1],
        subset,
        background,
        value_col=loeuf_col,
        ylabel="LOEUF",
        title="Gene constraint: LOEUF",
        subset_color=subset_color,
        bg_color=bg_color,
    )
    add_panel_label(axes_flat[1], "B")

    draw_two_group_boxplot(
        axes_flat[2],
        subset,
        background,
        value_col=spliceai_col,
        ylabel="SpliceAI DS max",
        title="SpliceAI score",
        subset_color=subset_color,
        bg_color=bg_color,
    )
    add_panel_label(axes_flat[2], "C")

    draw_two_group_barplot(
        axes_flat[3],
        radical_aa_summary,
        ylabel="Fraction of valid AA changes",
        title=f"Radical AA changes (Grantham > {args.radical_threshold:g})",
        subset_color=subset_color,
        bg_color=bg_color,
    )
    add_panel_label(axes_flat[3], "D")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=subset_color, alpha=0.65),
        plt.Rectangle((0, 0), 1, 1, color=bg_color, alpha=0.65),
    ]
    fig.legend(
        handles,
        ["GLM-Missense le2 subset", "Background"],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        columnspacing=2.0,
        handlelength=1.8,
    )

    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=2.0, w_pad=1.8)
    output_pdf = f"{out_prefix}.four_panel_main.pdf"
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print("[INFO] Wrote one combined four-panel PDF figure:")
    print(f"       {output_pdf}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
