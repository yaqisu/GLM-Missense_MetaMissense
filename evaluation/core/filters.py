"""
Filtering and stratification utilities for variant evaluation.

Supports two modes:
  - filter:    keep only variants satisfying a condition (e.g. AF < threshold)
  - stratify:  bin variants into labeled ranges and return each bin separately

Conservation score columns (from dbNSFP) are defined here for easy reference.
"""

import numpy as np
import pandas as pd
from .metrics import try_numeric


# ── Conservation score columns available in dbNSFP ────────────────────────
#
# Grouped by type for convenience. Pass any subset to evaluate_all / compare.

CONSERVATION_COLS = {
    # Vertebrate / mammalian phylogenetic scores
    "phyloP100way_vertebrate":        "PhyloP 100 vertebrates",
    "phyloP470way_mammalian":         "PhyloP 470 mammals",
    "phyloP17way_primate":            "PhyloP 17 primates",
    "phastCons100way_vertebrate":     "PhastCons 100 vertebrates",
    "phastCons470way_mammalian":      "PhastCons 470 mammals",
    "phastCons17way_primate":         "PhastCons 17 primates",
    # GERP
    "GERP++_RS":                      "GERP++ RS",
    "GERP++_NR":                      "GERP++ NR",
    "GERP_92_mammals":                "GERP 92 mammals",
    # Background selection
    "bStatistic":                     "B-statistic (background selection)",
}

# Commonly used AF columns
AF_COLS = {
    "gnomAD4.1_joint_AF":                    "gnomAD v4.1 joint",
    "gnomAD2.1.1_exomes_controls_AF":         "gnomAD v2.1.1 exomes (controls)",
    "gnomAD2.1.1_exomes_non_neuro_AF":        "gnomAD v2.1.1 exomes (non-neuro)",
    "gnomAD4.1_joint_POPMAX_AF":             "gnomAD v4.1 POPMAX",
    "gnomAD2.1.1_exomes_controls_POPMAX_AF":  "gnomAD v2.1.1 POPMAX",
    "dbNSFP_POPMAX_AF":                       "dbNSFP POPMAX",
}


# ── Filter mode ────────────────────────────────────────────────────────────

def apply_af_filter(df: pd.DataFrame, af_cols: list[str],
                    threshold: float, include_missing: bool = True) -> pd.Series:
    """
    Return boolean mask: True if variant passes the AF filter.

    include_missing=True (default): variants absent from gnomAD are treated
    as ultra-rare and included.
    """
    mask = pd.Series(True, index=df.index)
    for col in af_cols:
        if col not in df.columns:
            print(f"  WARNING: AF column '{col}' not in dataframe — skipping")
            continue
        numeric = try_numeric(df[col])
        if include_missing:
            col_mask = numeric.isna() | (numeric < threshold)
        else:
            col_mask = numeric < threshold
        n_rare    = (numeric < threshold).sum()
        n_missing = numeric.isna().sum()
        n_common  = (~col_mask).sum()
        print(f"  {col}: rare={n_rare:,}  missing(included)={n_missing:,}  "
              f"excluded(common)={n_common:,}")
        mask &= col_mask
    return mask


def apply_conservation_filter(df: pd.DataFrame, cons_col: str,
                               threshold: float, direction: str = "above") -> pd.Series:
    """
    Filter by a conservation score.

    direction='above': keep variants with score >= threshold (conserved sites)
    direction='below': keep variants with score <  threshold
    """
    if cons_col not in df.columns:
        raise ValueError(f"Conservation column '{cons_col}' not found in dataframe. "
                         f"Available: {[c for c in df.columns if c in CONSERVATION_COLS]}")
    numeric = try_numeric(df[cons_col])
    if direction == "above":
        mask = numeric.notna() & (numeric >= threshold)
    else:
        mask = numeric.notna() & (numeric < threshold)
    print(f"  {cons_col} {direction} {threshold}: {mask.sum():,} / {len(df):,} variants kept")
    return mask


# ── Stratify mode ──────────────────────────────────────────────────────────

# Pre-built AF strata — from ultra-rare to common
# Last stratum is AF>=1e-4 (variants seen at >= 1/10,000 frequency in gnomAD)
AF_STRATA_DEFAULT = [
    ("not_in_gnomAD",   None,    None),     # missing AF entirely
    ("AF=0",            0.0,     0.0),      # observed but AF exactly 0
    ("AF<1e-6",         0.0,     1e-6),
    ("1e-6<=AF<1e-5",   1e-6,    1e-5),
    ("AF>=1e-5",        1e-5,    None),     # merged: 1e-5<=AF<1e-4 + AF>=1e-4
]

# Pre-built conservation strata (GERP++ RS)
GERP_STRATA_DEFAULT = [
    ("GERP<0",      None, 0.0),
    ("0<=GERP<2",   0.0,  2.0),
    ("2<=GERP<4",   2.0,  4.0),
    ("GERP>=4",     4.0,  None),
]

PHYLOP_STRATA_DEFAULT = [
    ("phyloP<0",       None, 0.0),
    ("0<=phyloP<1",    0.0,  1.0),
    ("1<=phyloP<3",    1.0,  3.0),
    ("3<=phyloP<6",    3.0,  6.0),
    ("phyloP>=6",      6.0,  None),
]

# Gene constraint strata — LOEUF upper bound (lower = more constrained)
LOEUF_STRATA_DEFAULT = [
    ("LOEUF<0.35",        None,  0.35),   # most constrained
    ("0.35<=LOEUF<0.6",   0.35,  0.6),
    ("0.60<=LOEUF<0.85",  0.6,   0.85),
    ("LOEUF>=0.85",       0.85,  None),   # least constrained
]

# SpliceAI strata — stratify by max delta score (higher = more splice impact)
SPLICEAI_STRATA_DEFAULT = [
    ("spliceAI<0.1",        None,  0.1),   # likely no splice effect
    ("0.1<=spliceAI<0.2",   0.1,   0.2),   # low
    ("0.2<=spliceAI<0.5",   0.2,   0.5),   # moderate
    ("spliceAI>=0.5",       0.5,   None),   # high splice impact
]

def stratify_by_column(df: pd.DataFrame, col: str,
                       strata: list[tuple]) -> dict[str, pd.DataFrame]:
    """
    Split df into named strata based on (name, lo, hi) tuples.

    lo=None → no lower bound
    hi=None → no upper bound
    Both None → missing values (the 'not_in_gnomAD' stratum)

    Returns dict mapping stratum_name → subset DataFrame.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. "
                         f"Available AF cols: {[c for c in df.columns if 'AF' in c or 'GERP' in c]}")

    numeric = try_numeric(df[col])
    result  = {}

    for name, lo, hi in strata:
        if lo is None and hi is None:
            # Missing stratum
            mask = numeric.isna()
        elif lo is None:
            mask = numeric.notna() & (numeric < hi)
        elif hi is None:
            mask = numeric.notna() & (numeric >= lo)
        elif lo == hi:
            mask = numeric.notna() & (numeric == lo)
        else:
            mask = numeric.notna() & (numeric >= lo) & (numeric < hi)

        subset = df[mask].copy().reset_index(drop=True)
        result[name] = subset
        print(f"  Stratum '{name}': {len(subset):,} variants")

    return result


def parse_custom_strata(spec: str) -> list[tuple]:
    """
    Parse a user-supplied strata specification string.

    Format: 'lo1:hi1,lo2:hi2,...'  where lo/hi can be floats or 'None'.
    The label is auto-generated.

    Example:  'None:1e-6,1e-6:1e-4,1e-4:None'
    """
    strata = []
    for part in spec.split(","):
        lo_str, hi_str = part.strip().split(":")
        lo = None if lo_str.strip().lower() == "none" else float(lo_str)
        hi = None if hi_str.strip().lower() == "none" else float(hi_str)
        if lo is None and hi is not None:
            name = f"<{hi}"
        elif hi is None and lo is not None:
            name = f">={lo}"
        elif lo is None and hi is None:
            name = "missing"
        else:
            name = f"{lo}-{hi}"
        strata.append((name, lo, hi))
    return strata


# ── Anchor column filter (shared evaluation subset) ────────────────────────

def apply_anchor_filter(df: pd.DataFrame, anchor_cols: list[str],
                        our_col: str) -> pd.Series:
    """
    Require valid values in anchor columns (e.g. REVEL + AlphaMissense)
    AND our model score. Used to define the 'shared evaluation subset'.
    """
    mask = pd.Series(True, index=df.index)
    for col in anchor_cols:
        if col not in df.columns:
            print(f"  WARNING: anchor column '{col}' not found — skipping")
            continue
        col_mask = df[col].apply(
            lambda x: str(x).strip() not in (".", "", "NA", "nan") and not pd.isna(x))
        print(f"  {col}: {col_mask.sum():,} / {len(df):,} annotated")
        mask &= col_mask
    mask &= df[our_col].notna()
    return mask