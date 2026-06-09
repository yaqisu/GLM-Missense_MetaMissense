#!/usr/bin/env python3
"""
Multi-method RESCUE analysis with per-method own-backgrounds.

For each method M, define:
  - rescue_M    = variants in <Method>_correct_le<k> (or synthesized for SpAI)
  - background_M:
      * If M is one of the 6 dbNSFP COMPARATOR methods (everything except
        GLM-Missense and SpliceAI-DS-AG):
            background_M = (stratum \ rescue_M) \ rescue_SpAI
        i.e. exclude both M's own rescue subset AND the SpliceAI-DS-AG rescue
        subset, so the protein-effect comparison d is not contaminated by
        splice-driven variants.
      * Otherwise (GLM-Missense, SpliceAI-DS-AG, or when SpAI is not in
        the figure):
            background_M = stratum minus rescue_M

Each panel pairs each method's rescue against its own background, with
Mann-Whitney (or Fisher exact for the radical-AA panel) testing rescue vs
own-background within method. n is annotated under every box.

This is the "method-correct vs method-specific background" framing,
extending the published Fig 6 (which used GLM rescue vs GLM background)
symmetrically to all methods.

Outputs go to --out-dir (default evaluation/results/multi_method_rescue_ownbg/).

Conventions match evaluation/glmmissense_correct_analysis_for_fig6.py:
PDF font 42, Arial, column-name auto-detection, Bonferroni within panel.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, mannwhitneyu

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["axes.unicode_minus"] = False


# -----------------------------
# Constants
# -----------------------------

DBNSFP_METHODS = ["GLM-Missense", "AlphaMissense", "CADD", "ESM1b",
                  "Polyphen2", "REVEL", "SIFT"]
COMPARATOR_METHODS = ["AlphaMissense", "CADD", "ESM1b",
                      "Polyphen2", "REVEL", "SIFT"]
PLOT_ORDER = ["GLM-Missense", "AlphaMissense", "ESM1b", "REVEL",
              "CADD", "Polyphen2", "SIFT", "SpliceAI-DS-AG"]

LABEL_TO_NAME = {0: "Benign", 1: "Pathogenic"}
STRATA = [0, 1]

LABEL_CANDIDATES = ["true_label", "label"]
AF_CANDIDATES = ["gnomAD4.1_joint_AF", "gnomAD4_1_joint_AF", "gnomAD_joint_AF"]
LOEUF_CANDIDATES = ["lof.oe_ci.upper", "LOEUF", "loeuf"]
SPLICEAI_MAX_CANDIDATES = ["spliceai_DS_max", "SpliceAI_DS_max"]
SPLICEAI_DSAG_CANDIDATES = ["spliceai_DS_AG", "SpliceAI_DS_AG"]
AA_REF_CANDIDATES = ["aaref", "aaref_x", "aaref_y", "aa_ref", "ref_aa",
                     "protein_ref", "AA_ref"]
AA_ALT_CANDIDATES = ["aaalt", "aaalt_x", "aaalt_y", "aa_alt", "alt_aa",
                     "protein_alt", "AA_alt"]

STANDARD_AAS = set("ARNDCQEGHILKMFPSTWYV")
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
_GRANTHAM_RAW = {
    ("A","C"):195,("A","D"):126,("A","E"):107,("A","F"):113,("A","G"):60,
    ("A","H"):86,("A","I"):94,("A","K"):106,("A","L"):96,("A","M"):84,
    ("A","N"):111,("A","P"):27,("A","Q"):91,("A","R"):112,("A","S"):99,
    ("A","T"):58,("A","V"):64,("A","W"):148,("A","Y"):112,
    ("C","D"):154,("C","E"):170,("C","F"):205,("C","G"):159,("C","H"):174,
    ("C","I"):198,("C","K"):202,("C","L"):198,("C","M"):196,("C","N"):139,
    ("C","P"):169,("C","Q"):154,("C","R"):180,("C","S"):112,("C","T"):149,
    ("C","V"):192,("C","W"):215,("C","Y"):194,
    ("D","E"):45,("D","F"):177,("D","G"):94,("D","H"):81,("D","I"):168,
    ("D","K"):101,("D","L"):172,("D","M"):160,("D","N"):23,("D","P"):108,
    ("D","Q"):61,("D","R"):96,("D","S"):65,("D","T"):85,("D","V"):152,
    ("D","W"):181,("D","Y"):160,
    ("E","F"):140,("E","G"):98,("E","H"):40,("E","I"):134,("E","K"):56,
    ("E","L"):138,("E","M"):126,("E","N"):42,("E","P"):93,("E","Q"):29,
    ("E","R"):54,("E","S"):80,("E","T"):65,("E","V"):121,("E","W"):152,
    ("E","Y"):122,
    ("F","G"):153,("F","H"):100,("F","I"):21,("F","K"):102,("F","L"):22,
    ("F","M"):28,("F","N"):158,("F","P"):114,("F","Q"):116,("F","R"):97,
    ("F","S"):155,("F","T"):103,("F","V"):50,("F","W"):40,("F","Y"):22,
    ("G","H"):98,("G","I"):135,("G","K"):127,("G","L"):138,("G","M"):127,
    ("G","N"):80,("G","P"):42,("G","Q"):87,("G","R"):125,("G","S"):56,
    ("G","T"):59,("G","V"):109,("G","W"):184,("G","Y"):147,
    ("H","I"):94,("H","K"):32,("H","L"):99,("H","M"):87,("H","N"):68,
    ("H","P"):77,("H","Q"):24,("H","R"):29,("H","S"):89,("H","T"):47,
    ("H","V"):84,("H","W"):115,("H","Y"):83,
    ("I","K"):102,("I","L"):5,("I","M"):10,("I","N"):149,("I","P"):95,
    ("I","Q"):109,("I","R"):97,("I","S"):142,("I","T"):89,("I","V"):29,
    ("I","W"):61,("I","Y"):33,
    ("K","L"):107,("K","M"):95,("K","N"):94,("K","P"):103,("K","Q"):53,
    ("K","R"):26,("K","S"):121,("K","T"):78,("K","V"):97,("K","W"):110,
    ("K","Y"):85,
    ("L","M"):15,("L","N"):153,("L","P"):98,("L","Q"):113,("L","R"):102,
    ("L","S"):145,("L","T"):92,("L","V"):32,("L","W"):61,("L","Y"):36,
    ("M","N"):142,("M","P"):87,("M","Q"):101,("M","R"):91,("M","S"):135,
    ("M","T"):81,("M","V"):21,("M","W"):67,("M","Y"):36,
    ("N","P"):91,("N","Q"):46,("N","R"):86,("N","S"):46,("N","T"):65,
    ("N","V"):133,("N","W"):174,("N","Y"):143,
    ("P","Q"):76,("P","R"):103,("P","S"):74,("P","T"):38,("P","V"):68,
    ("P","W"):147,("P","Y"):110,
    ("Q","R"):43,("Q","S"):68,("Q","T"):42,("Q","V"):96,("Q","W"):130,
    ("Q","Y"):99,
    ("R","S"):110,("R","T"):71,("R","V"):96,("R","W"):101,("R","Y"):77,
    ("S","T"):58,("S","V"):124,("S","W"):177,("S","Y"):144,
    ("T","V"):69,("T","W"):128,("T","Y"):92,
    ("V","W"):88,("V","Y"):55,
    ("W","Y"):37,
}

# Rescue colors (dark) and matching backgrounds (lighter shade of same hue).
COLOR_RESCUE = {
    "GLM-Missense":    "#1f77b4",
    "SpliceAI-DS-AG":  "#ff7f0e",
    "_comparator":     "#555555",
}
COLOR_BG = {
    "GLM-Missense":    "#a8c7df",   # lighter blue
    "SpliceAI-DS-AG":  "#ffc78a",   # lighter orange
    "_comparator":     "#cccccc",   # lighter grey
}


# -----------------------------
# Utilities
# -----------------------------

def pick_col(df, candidates, *, required=True, what="column"):
    for c in candidates:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        preview = ", ".join(map(str, list(df.columns)[:80]))
        raise ValueError(
            f"Could not find {what}. Tried: {', '.join(candidates)}\n"
            f"Available columns include: {preview} ..."
        )
    return None


def parse_bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(float) != 0
    return (
        s.astype(str).str.strip().str.lower()
         .isin({"true", "t", "1", "yes", "y"})
    )


def numeric_series(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def format_p(p):
    if pd.isna(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.3g}"


def significance_stars(p):
    if pd.isna(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def wilson_ci(k, n, z=1.96):
    if n <= 0:
        return np.nan, np.nan
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


# -----------------------------
# AA / Grantham
# -----------------------------

def normalize_aa(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if not s or s in {"*", "X", "TER", "STOP", "NA", "NAN", ".", "-"}:
        return None
    if len(s) == 1 and s in STANDARD_AAS:
        return s
    if len(s) == 3 and s in AA3_TO_1:
        return AA3_TO_1[s]
    return None


def grantham(a, b):
    a, b = normalize_aa(a), normalize_aa(b)
    if a is None or b is None:
        return None
    if a == b:
        return 0.0
    return float(_GRANTHAM_RAW.get((a, b), _GRANTHAM_RAW.get((b, a), np.nan)))


def add_aa_flags(df, aa_ref_col, aa_alt_col, radical_threshold=150.0):
    out = df.copy()
    valid, radical = [], []
    for r, a in zip(out[aa_ref_col], out[aa_alt_col]):
        d = grantham(r, a)
        if d is None or (isinstance(d, float) and np.isnan(d)):
            valid.append(False); radical.append(False)
        else:
            valid.append(True); radical.append(d > radical_threshold)
    out["_valid_aa_change"] = valid
    out["_is_radical_aa_change"] = radical
    return out


# -----------------------------
# Rescue subsets
# -----------------------------

def build_rescue_masks(df, *, le_threshold, include_spliceai,
                       spliceai_dsag_col, spliceai_threshold, label_col):
    masks = {}
    for m in DBNSFP_METHODS:
        col = f"{m}_correct_le{le_threshold}"
        if col not in df.columns:
            raise ValueError(f"Expected column {col} not found in merged TSV.")
        masks[m] = parse_bool_series(df[col])

    if include_spliceai:
        dsag = numeric_series(df[spliceai_dsag_col])
        y = pd.to_numeric(df[label_col], errors="coerce")
        spliceai_call = (dsag >= spliceai_threshold).fillna(False).astype(int)
        spliceai_correct = (spliceai_call == y.fillna(-1).astype(int)) & y.notna()
        anchor_methods = [m for m in DBNSFP_METHODS if m != "GLM-Missense"]
        anchors_correct = sum(
            parse_bool_series(df[f"correct_{m}"]).astype(int)
            if f"correct_{m}" in df.columns
            else pd.Series(0, index=df.index)
            for m in anchor_methods
        )
        masks["SpliceAI-DS-AG"] = (
            spliceai_correct & (anchors_correct <= le_threshold)
        ).astype(bool)
    return masks


def methods_in_order(masks):
    return [m for m in PLOT_ORDER if m in masks]


def own_background_mask(method, masks, *, df_index, exclude_spai_from_comparators):
    """Build background mask for method M, aligned to df_index.

    - For comparator methods (everything except GLM and SpAI):
        background = NOT in M_rescue AND NOT in SpAI_rescue (if SpAI present).
    - For GLM and SpAI: background = NOT in M_rescue.
    """
    rescue = masks[method].reindex(df_index, fill_value=False)
    bg = ~rescue
    if (exclude_spai_from_comparators
        and method in COMPARATOR_METHODS
        and "SpliceAI-DS-AG" in masks):
        bg = bg & ~masks["SpliceAI-DS-AG"].reindex(df_index, fill_value=False)
    return bg


def rescue_color(method):
    if method in ("GLM-Missense", "SpliceAI-DS-AG"):
        return COLOR_RESCUE[method]
    return COLOR_RESCUE["_comparator"]


def bg_color(method):
    if method in ("GLM-Missense", "SpliceAI-DS-AG"):
        return COLOR_BG[method]
    return COLOR_BG["_comparator"]


def short_label(method):
    """Display name for x-tick labels. Kept named 'short_label' for backwards
    compatibility, but now returns the full method name."""
    return {
        "GLM-Missense": "GLM-Missense",
        "AlphaMissense": "AlphaMissense",
        "CADD": "CADD",
        "ESM1b": "ESM1b",
        "Polyphen2": "PolyPhen-2",
        "REVEL": "REVEL",
        "SIFT": "SIFT",
        "SpliceAI-DS-AG": "SpliceAI-DS-AG",
    }.get(method, method)


# -----------------------------
# Plotting: paired rescue + own-background per method
# -----------------------------

PAIR_GAP = 0.30      # gap between rescue and own-bg within a method pair
GROUP_GAP = 1.20     # gap between methods


def _method_positions(n_methods):
    """Two positions per method; returns (rescue_positions, bg_positions, centers)."""
    pos_r, pos_b, centers = [], [], []
    cursor = 0.0
    for _ in range(n_methods):
        r = cursor
        b = cursor + PAIR_GAP
        pos_r.append(r); pos_b.append(b); centers.append((r + b) / 2)
        cursor += PAIR_GAP + GROUP_GAP
    return pos_r, pos_b, centers


def draw_paired_boxplot(ax, *, df_stratum, masks, methods, value_col, transform,
                        ylabel, title, exclude_spai_from_comparators):
    pos_r, pos_b, centers = _method_positions(len(methods))

    data, positions, colors, ns = [], [], [], []
    method_results = []
    for i, m in enumerate(methods):
        rescue_mask = masks[m].reindex(df_stratum.index, fill_value=False)
        bg_mask = own_background_mask(
            m, masks, df_index=df_stratum.index,
            exclude_spai_from_comparators=exclude_spai_from_comparators,
        ).reindex(df_stratum.index, fill_value=False)

        r_vals = transform(df_stratum.loc[rescue_mask, value_col]).dropna().to_numpy()
        b_vals = transform(df_stratum.loc[bg_mask, value_col]).dropna().to_numpy()

        data.append(r_vals if len(r_vals) else np.array([np.nan]))
        positions.append(pos_r[i]); colors.append(rescue_color(m))
        ns.append(len(r_vals))
        data.append(b_vals if len(b_vals) else np.array([np.nan]))
        positions.append(pos_b[i]); colors.append(bg_color(m))
        ns.append(len(b_vals))

        if len(r_vals) and len(b_vals):
            try:
                _, p = mannwhitneyu(r_vals, b_vals, alternative="two-sided")
            except Exception:
                p = np.nan
        else:
            p = np.nan
        method_results.append((m, len(r_vals), len(b_vals), p))

    bp = ax.boxplot(
        data, positions=positions, widths=PAIR_GAP * 0.85,
        patch_artist=True, showfliers=False, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 3},
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.80)
    for med in bp["medians"]:
        med.set_color("black")

    ax.set_xticks(centers)
    ax.set_xticklabels([short_label(m) for m in methods],
                       fontsize=8, rotation=45, ha="right")
    # Push tick labels down so the rotated n's can sit between axis and labels.
    ax.tick_params(axis="x", which="major", pad=16)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Bonferroni across the N methods within this panel.
    n_tests = len(methods)
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax > ymin else 1.0
    ax.set_ylim(ymin, ymax + yrange * 0.18)
    ystar = ymax + yrange * 0.05
    for i, (_, _, _, p) in enumerate(method_results):
        p_adj = min(p * n_tests, 1.0) if not pd.isna(p) else np.nan
        s = significance_stars(p_adj)
        if s:
            ax.text(centers[i], ystar, s, ha="center", va="bottom", fontsize=8)

    # n annotations: rotated 90 deg, placed JUST BELOW the x-axis (above the
    # tick labels), one per box (rescue under dark box, bg under light box),
    # color-matched to box color, no "r:"/"bg:" prefix since color encodes it.
    for i, m in enumerate(methods):
        rcol = rescue_color(m)
        bcol = bg_color(m)
        ax.annotate(f"{ns[2*i]}", xy=(pos_r[i], -0.03),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="top", rotation=90,
                    fontsize=6.0, color=rcol)
        ax.annotate(f"{ns[2*i+1]}", xy=(pos_b[i], -0.03),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="top", rotation=90,
                    fontsize=6.0, color=bcol)

    ax.set_title(title, fontsize=9, pad=6)

    # Return per-method test info
    out_rows = []
    for (m, nr, nb, p_raw) in method_results:
        out_rows.append({
            "method": m, "rescue_n": nr, "own_background_n": nb,
            "mw_p_raw": p_raw,
            "mw_p_bonf": min(p_raw * n_tests, 1.0) if not pd.isna(p_raw) else np.nan,
        })
    return {"title": title, "rows": out_rows}


def draw_paired_barplot_radical(ax, *, df_stratum, masks, methods,
                                ylabel, title, exclude_spai_from_comparators):
    pos_r, pos_b, centers = _method_positions(len(methods))

    def frac_and_ci(sub):
        v = sub.loc[sub["_valid_aa_change"].astype(bool)]
        n = len(v); k = int(v["_is_radical_aa_change"].sum()) if n else 0
        f = k / n if n else np.nan
        lo, hi = wilson_ci(k, n)
        return n, k, f, lo, hi

    positions, fracs, los, his, ns, colors = [], [], [], [], [], []
    method_results = []
    for i, m in enumerate(methods):
        rescue_mask = masks[m].reindex(df_stratum.index, fill_value=False)
        bg_mask = own_background_mask(
            m, masks, df_index=df_stratum.index,
            exclude_spai_from_comparators=exclude_spai_from_comparators,
        ).reindex(df_stratum.index, fill_value=False)

        rescue_df = df_stratum.loc[rescue_mask]
        bg_df = df_stratum.loc[bg_mask]
        nr, kr, fr, lor, hir = frac_and_ci(rescue_df)
        nb, kb, fb, lob, hib = frac_and_ci(bg_df)

        positions += [pos_r[i], pos_b[i]]
        fracs += [fr, fb]; los += [lor, lob]; his += [hir, hib]
        ns += [nr, nb]; colors += [rescue_color(m), bg_color(m)]

        if nr > 0 and nb > 0:
            try:
                _, p = fisher_exact(
                    [[kr, nr - kr], [kb, nb - kb]], alternative="two-sided"
                )
            except Exception:
                p = np.nan
        else:
            p = np.nan
        method_results.append((m, nr, nb, p))

    fracs_np = np.array(fracs, dtype=float)
    los_np = np.array(los, dtype=float)
    his_np = np.array(his, dtype=float)
    yerr_lo = np.where(np.isnan(fracs_np - los_np), 0, fracs_np - los_np)
    yerr_hi = np.where(np.isnan(his_np - fracs_np), 0, his_np - fracs_np)
    yerr_lo = np.clip(yerr_lo, 0, None)
    yerr_hi = np.clip(yerr_hi, 0, None)
    yerr = np.vstack([yerr_lo, yerr_hi])

    fracs_plot = np.where(np.isnan(fracs_np), 0, fracs_np)
    ax.bar(positions, fracs_plot, width=PAIR_GAP * 0.85, yerr=yerr, capsize=2,
           color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_xticks(centers)
    ax.set_xticklabels([short_label(m) for m in methods],
                       fontsize=8, rotation=45, ha="right")
    ax.tick_params(axis="x", which="major", pad=16)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    valid = fracs_np[~np.isnan(fracs_np)]
    upper = float(np.max(valid)) if len(valid) else 0.05
    ax.set_ylim(0, max(0.1, upper * 1.55))

    n_tests = len(methods)
    ymax = ax.get_ylim()[1]
    for i, (_, _, _, p) in enumerate(method_results):
        p_adj = min(p * n_tests, 1.0) if not pd.isna(p) else np.nan
        s = significance_stars(p_adj)
        if s:
            r_height = max(fracs_plot[2*i], fracs_plot[2*i+1])
            ax.text(centers[i], r_height + ymax * 0.04, s,
                    ha="center", va="bottom", fontsize=8)

    for i, m in enumerate(methods):
        rcol = rescue_color(m)
        bcol = bg_color(m)
        ax.annotate(f"{ns[2*i]}", xy=(pos_r[i], -0.03),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="top", rotation=90,
                    fontsize=6.0, color=rcol)
        ax.annotate(f"{ns[2*i+1]}", xy=(pos_b[i], -0.03),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="top", rotation=90,
                    fontsize=6.0, color=bcol)
    ax.set_title(title, fontsize=9, pad=6)

    out_rows = []
    for (m, nr, nb, p_raw) in method_results:
        out_rows.append({
            "method": m, "rescue_n": nr, "own_background_n": nb,
            "fisher_p_raw": p_raw,
            "fisher_p_bonf": min(p_raw * n_tests, 1.0) if not pd.isna(p_raw) else np.nan,
        })
    return {"title": title, "rows": out_rows}


# -----------------------------
# Main figure
# -----------------------------

@dataclass
class Cols:
    label: str
    af: str
    loeuf: str
    spliceai_max: str
    spliceai_dsag: str
    aa_ref: str
    aa_alt: str


def build_figure(df, *, masks, methods, cols, af_pseudocount,
                 radical_threshold, le_threshold,
                 exclude_spai_from_comparators, out_pdf):
    fig, axes = plt.subplots(2, 4, figsize=(17, 9.5))
    af_transform = lambda s: np.log10(
        np.maximum(pd.to_numeric(s, errors="coerce"), 0) + af_pseudocount
    )
    id_transform = lambda s: pd.to_numeric(s, errors="coerce")
    records = []

    for row_i, stratum in enumerate(STRATA):
        strat_df = df.loc[df["_true_label"] == stratum]
        sname = LABEL_TO_NAME[stratum]
        for col_i, (key, vc, tx, ylab, ttl) in enumerate([
            ("SpliceAI", cols.spliceai_max, id_transform,
             "SpliceAI DS max", f"{sname}: SpliceAI score"),
            ("AF", cols.af, af_transform,
             f"log10(AF + {af_pseudocount:g})",
             f"{sname}: Allele frequency"),
            ("LOEUF", cols.loeuf, id_transform,
             "LOEUF", f"{sname}: LOEUF"),
        ]):
            rec = draw_paired_boxplot(
                axes[row_i, col_i], df_stratum=strat_df, masks=masks,
                methods=methods, value_col=vc, transform=tx,
                ylabel=ylab, title=ttl,
                exclude_spai_from_comparators=exclude_spai_from_comparators,
            )
            rec["panel_key"] = key; rec["stratum"] = sname
            records.append(rec)

        rec_rad = draw_paired_barplot_radical(
            axes[row_i, 3], df_stratum=strat_df, masks=masks, methods=methods,
            ylabel="Fraction radical AA",
            title=f"{sname}: Radical AA (Grantham>{radical_threshold:g})",
            exclude_spai_from_comparators=exclude_spai_from_comparators,
        )
        rec_rad["panel_key"] = "RadicalAA"; rec_rad["stratum"] = sname
        records.append(rec_rad)

    bg_note = ("comparator backgrounds exclude SpAI rescue"
               if exclude_spai_from_comparators else
               "each method's background = stratum \\ rescue_M")
    fig.suptitle(
        f"Per-method rescue vs own background (le{le_threshold}; {bg_note})\n"
        f"Each method pair: rescue (dark) vs own background (light); "
        f"stars = MW/Fisher rescue-vs-own-bg, Bonferroni within panel",
        fontsize=10, y=1.00,
    )

    # Compact legend; SpAI entries only if SpliceAI-DS-AG is actually plotted.
    include_spai = "SpliceAI-DS-AG" in methods
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_RESCUE["GLM-Missense"], alpha=0.8),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BG["GLM-Missense"], alpha=0.8),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_RESCUE["_comparator"], alpha=0.8),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BG["_comparator"], alpha=0.8),
    ]
    labels = ["GLM rescue", "GLM own bg",
              "Comparator rescue", "Comparator own bg"]
    if include_spai:
        handles += [
            plt.Rectangle((0, 0), 1, 1, color=COLOR_RESCUE["SpliceAI-DS-AG"], alpha=0.8),
            plt.Rectangle((0, 0), 1, 1, color=COLOR_BG["SpliceAI-DS-AG"], alpha=0.8),
        ]
        labels += ["SpAI rescue", "SpAI own bg"]
    fig.legend(
        handles, labels,
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.005),
        ncol=len(labels), columnspacing=2.0, handlelength=1.6, fontsize=8,
    )

    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return records


# -----------------------------
# Stats consolidation
# -----------------------------

def records_to_dataframe(records):
    rows = []
    for r in records:
        for row in r["rows"]:
            d = {"stratum": r["stratum"], "panel": r["title"],
                 "panel_key": r["panel_key"]}
            d.update(row)
            rows.append(d)
    return pd.DataFrame(rows)


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Multi-method rescue analysis with per-method "
                    "own-backgrounds (option A).",
    )
    p.add_argument("--merged-set", required=True)
    p.add_argument("--out-dir",
                   default="evaluation/results/multi_method_rescue_ownbg")
    p.add_argument("--out-prefix", default="rescue_ownbg")
    p.add_argument("--le-threshold", type=int, default=2,
                   choices=[0, 1, 2, 3, 4, 5, 6])
    p.add_argument("--include-spliceai", action="store_true", default=True)
    p.add_argument("--no-spliceai", dest="include_spliceai",
                   action="store_false")
    p.add_argument("--exclude-spai-from-comparators",
                   action="store_true", default=True,
                   help="If set (default), comparator methods' backgrounds "
                        "exclude SpliceAI-DS-AG rescue variants.")
    p.add_argument("--no-exclude-spai-from-comparators",
                   dest="exclude_spai_from_comparators", action="store_false")
    p.add_argument("--spliceai-threshold", type=float, default=0.2)
    p.add_argument("--radical-threshold", type=float, default=150.0)
    p.add_argument("--af-pseudocount", type=float, default=1e-8)
    p.add_argument("--label-col", default=None)
    p.add_argument("--af-col", default=None)
    p.add_argument("--loeuf-col", default=None)
    p.add_argument("--spliceai-max-col", default=None)
    p.add_argument("--spliceai-dsag-col", default=None)
    p.add_argument("--aaref-col", default=None)
    p.add_argument("--aaalt-col", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading {args.merged_set}")
    df = pd.read_csv(args.merged_set, sep="\t", low_memory=False)
    print(f"[INFO] {len(df):,} rows × {len(df.columns):,} columns")

    label_col = args.label_col or pick_col(df, LABEL_CANDIDATES, what="label column")
    af_col = args.af_col or pick_col(df, AF_CANDIDATES, what="AF column")
    loeuf_col = args.loeuf_col or pick_col(df, LOEUF_CANDIDATES, what="LOEUF column")
    spliceai_max_col = args.spliceai_max_col or pick_col(
        df, SPLICEAI_MAX_CANDIDATES, what="SpliceAI DS max column")
    spliceai_dsag_col = args.spliceai_dsag_col or pick_col(
        df, SPLICEAI_DSAG_CANDIDATES, what="SpliceAI DS_AG column")
    aa_ref_col = args.aaref_col or pick_col(df, AA_REF_CANDIDATES, what="aa ref")
    aa_alt_col = args.aaalt_col or pick_col(df, AA_ALT_CANDIDATES, what="aa alt")

    cols = Cols(label_col, af_col, loeuf_col, spliceai_max_col,
                spliceai_dsag_col, aa_ref_col, aa_alt_col)
    df["_true_label"] = pd.to_numeric(df[label_col], errors="coerce")
    df = add_aa_flags(df, aa_ref_col, aa_alt_col,
                      radical_threshold=args.radical_threshold)

    print(f"[INFO] Building rescue masks at le_threshold={args.le_threshold} "
          f"(include_spliceai={args.include_spliceai}, "
          f"spliceai_threshold={args.spliceai_threshold})")
    masks = build_rescue_masks(
        df, le_threshold=args.le_threshold,
        include_spliceai=args.include_spliceai,
        spliceai_dsag_col=spliceai_dsag_col,
        spliceai_threshold=args.spliceai_threshold,
        label_col=label_col,
    )
    methods = methods_in_order(masks)

    # Subset sizes (rescue + own-background) per method × stratum
    rows = []
    for m in methods:
        for stratum in STRATA:
            sname = LABEL_TO_NAME[stratum]
            strat_idx = df.index[df["_true_label"] == stratum]
            rescue_n = int(masks[m].reindex(strat_idx, fill_value=False).sum())
            bg = own_background_mask(
                m, masks, df_index=df.index,
                exclude_spai_from_comparators=args.exclude_spai_from_comparators,
            )
            bg_n = int(bg.reindex(strat_idx, fill_value=False).sum())
            rows.append({
                "method": m, "stratum": sname,
                "rescue_n": rescue_n, "own_background_n": bg_n,
            })
    sizes_df = pd.DataFrame(rows)
    sizes_path = (out_dir
                  / f"{args.out_prefix}.le{args.le_threshold}.subset_sizes.tsv")
    sizes_df.to_csv(sizes_path, sep="\t", index=False)
    print(f"[INFO] wrote {sizes_path}")
    print(sizes_df.to_string(index=False))

    fig_pdf = (out_dir
               / f"{args.out_prefix}.le{args.le_threshold}.own_background.pdf")
    records = build_figure(
        df, masks=masks, methods=methods, cols=cols,
        af_pseudocount=args.af_pseudocount,
        radical_threshold=args.radical_threshold,
        le_threshold=args.le_threshold,
        exclude_spai_from_comparators=args.exclude_spai_from_comparators,
        out_pdf=fig_pdf,
    )
    print(f"[INFO] wrote {fig_pdf}")

    stats_path = (out_dir
                  / f"{args.out_prefix}.le{args.le_threshold}.stats.tsv")
    records_to_dataframe(records).to_csv(stats_path, sep="\t", index=False)
    print(f"[INFO] wrote {stats_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()