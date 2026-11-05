#!/usr/bin/env python3
"""
alphagenome_score.py — AlphaGenome per-track scores of variants, aggregated into per-variant features (the features
of fig_vus.py). Used in analysis/ with --select none --input <list of variants> (e.g. missing_alphagenome_*.tsv or
the whole VUS table); the selection modes below are from the original discordant-VUS analysis.

--source atlas (default): precomputed scores from the AlphaGenome Atlas (every GRCh38 SNV; same API key; a larger
  query rate than on-demand prediction, per-key quotas unpublished), one lookup per variant, --max_workers in
  parallel, the same scorers and the same per-track summaries as below. Check first:
    --inspect 3            print what the Atlas returns for 3 variants (AnnData per scorer: shapes, obs / var
                           columns, layers) and how it is turned into per-track rows, then exit
    --compare <table>      score the first --limit variants of a feature table scored live (--source live) with the
                           Atlas and report agreement per feature (<outdir>/atlas_compare/atlas_vs_live.tsv); mix
                           Atlas and live features only if they agree
    --compare_tables A L   compare an Atlas and a live feature table already scored, per feature (no API calls)
    --diagnose N           score the first N variants of --input both ways now (Atlas and live, same scorers) and
                           compare per track: which tracks / genes each returns, the scored interval, raw and
                           quantile scores of matched tracks (<outdir>/diagnose/), i.e. where Atlas and live differ
  --atlas_gene_filter asks the Atlas for the variant's gene only (smaller replies; gene-centric scorers are
  restricted to the variant's gene anyway).
--source live: on-demand predictions (dna_client.score_variants, --sequence_length), as originally run.
--outdir must end with _atlas or _live (the --source), so the two kinds of features never share a folder; a second
--out_name in the same folder (e.g. fig_vus.py's alphagenome_features_figvus.tsv while a full run writes
alphagenome_features.tsv) keeps its own checkpoint, failure list and selected_variants_<name>.tsv.

Variant selection (from scores_all.tsv.gz of ensemble_discordant_VUS.py):
  --select confident   confident discordant variants (default)
  --select discordant  all discordant variants
  --select none        every row of --input (already filtered)
  --extend_controls N adds N more controls per case to an existing selection (same rule, existing
  controls kept and not reused), so only the new ones are scored.
  plus --n_controls concordant variants per selected variant: the nearest neighbours on the
  without-GLM probability within +/- --caliper, from the same gene when enough are available,
  otherwise from any gene; variants without enough controls inside the caliper are reported.

Scorers (AlphaGenome RECOMMENDED_VARIANT_SCORERS keys; default all below, CONTACT_MAPS left out; --source atlas also
adds the Atlas's overall AVI score: <AVI scorer>_* summaries and AVI_score):
  SPLICE_SITES, SPLICE_SITE_USAGE, SPLICE_JUNCTIONS   splicing (gene-centric)
  RNA_SEQ                                              expression log fold change (gene-centric, signed)
  POLYADENYLATION                                      polyA site usage (gene-centric)
  CHIP_TF, DNASE, ATAC, CHIP_HISTONE, CAGE, PROCAP     centre-mask (variant-centric)

Per variant and scorer (quantile_score where available, else raw_score):
  <S>_n          tracks scored
  <S>_maxabs     largest |score| across tracks
  <S>_signed     signed score of that track
  <S>_mean       mean score across tracks (signed; most useful for RNA_SEQ)
  <S>_top_track  biosample (and TF / histone mark) of that track
  <S>_min / <S>_max            most negative / most positive score across tracks (strongest decrease /
                               increase, e.g. RNA_SEQ: largest predicted loss / gain of expression)
  <S>_min_track / <S>_max_track  biosample of those tracks
  <S>_n_down / <S>_n_up        tracks with quantile_score <= -q / >= q (q = --extreme_q, default 0.99;
  <S>_frac_down / <S>_frac_up  only when quantile scores are used) and their share of the tracks scored
  <S>_median, <S>_p10, <S>_p90 median and 10th / 90th percentile of the same scores across tracks (robust
                               typical effect and strong effect in a subset of tracks, without relying on
                               the single most extreme track)
  <S>_raw_mean, _raw_median, _raw_p10, _raw_p90, _raw_min, _raw_max
                               the same on raw_score (RNA_SEQ: log fold change of mean exon coverage,
                               alt vs ref), i.e. the size of the predicted change, which quantile scores
                               compress in the tails; only when quantile scores are also present
--save_tidy keeps every per-track row (own gene only for gene-centric scorers, plus input_variant_id) in
tidy_<out_name stem>/batch_*.tsv.gz, for offline summaries such as tissue-matched tracks.
Gene-centric scorers use only rows for the variant's own gene (--gene_col, matched on symbol, or
Ensembl gene id via --gene_id_col); <S>_own_gene says whether it was found. With --ontology, the same
aggregates are also computed over tracks of those ontology terms (<S>_tis_*).

Runs in batches with a checkpoint: re-running the same command skips variants already in
alphagenome_features.tsv (or --out_name). Failed variants go to failed.tsv (failed_<out_name>) and are
retried on the next run. To add new columns for variants that are already scored, re-score them into a
new file with the saved selection, e.g. --outdir <same> --out_name alphagenome_features_rnaseq.tsv
--scorers RNA_SEQ (only the scorers you need, to save API time), and pass that file to --features too.

API key: --api_key_file, or the ALPHAGENOME_API_KEY environment variable.

Example (pilot of 50 variants first):
    export ALPHAGENOME_API_KEY=...
    python alphagenome_score.py \
        --input VUS_discordant/scores_all.tsv.gz --select confident --n_controls 2 \
        --outdir VUS_alphagenome --limit 50
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SCORERS = ["SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS", "RNA_SEQ", "POLYADENYLATION",
                   "CHIP_TF", "DNASE", "ATAC", "CHIP_HISTONE", "CAGE", "PROCAP"]
GENE_COLS = ("gene", "GENE", "genename")
CARRY = ["variant_id", "gene", "genename", "TRANSCRIPT_ID", "HGVS_C", "PROTEIN_CHANGE", "p_with", "p_without",
         "call_with", "call_without", "logit_shift", "discordant", "direction", "confident",
         "GLM-Missense_score", "AlphaMissense_score", "spliceai_DS_max", "CADD_phred", "dbNSFP_POPMAX_AF"]
OUT_NAME = "alphagenome_features.tsv"


# ----------------------------------------------------------------------------- selection
def as_bool(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def nearest_unused(vals, ids, x, k, caliper, used):
    """Up to k ids from sorted vals nearest to x (|diff| <= caliper), skipping used ones."""
    pos = np.searchsorted(vals, x)
    lo, hi, got = pos - 1, pos, []
    while len(got) < k:
        dl = x - vals[lo] if lo >= 0 else np.inf
        dh = vals[hi] - x if hi < len(vals) else np.inf
        if min(dl, dh) > caliper:
            break
        if dl <= dh:
            i, lo = ids[lo], lo - 1
        else:
            i, hi = ids[hi], hi + 1
        if i not in used:
            got.append(i)
    return got


def extend_controls(df, sel, n_extra, match_gene, gene_col, seed, caliper):
    """Add n_extra matched controls per case to an existing selection: same rule (nearest unused
    p_without within the caliper, same gene first); controls already selected are kept and never reused."""
    p0 = pd.to_numeric(df["p_without"], errors="coerce")
    disc = as_bool(df["discordant"])
    ok = p0.notna() & pd.to_numeric(df["p_with"], errors="coerce").notna()
    taken = set(sel["variant_id"])
    pool = df.index[~disc & ok & ~df["variant_id"].isin(taken)]
    order = np.argsort(p0[pool].to_numpy(), kind="stable")
    all_vals, all_ids = p0[pool].to_numpy()[order], pool.to_numpy()[order]
    by_gene = {}
    if match_gene and gene_col:
        g = df.loc[pool, gene_col].to_numpy()[order]
        for gene, idx in pd.Series(range(len(g))).groupby(g).groups.items():
            idx = np.asarray(idx)
            by_gene[gene] = (all_vals[idx], all_ids[idx])
    row_of = pd.Series(df.index, index=df["variant_id"])
    cases = sel[sel["ag_group"] != "matched_concordant"]
    direction = dict(zip(cases["variant_id"], cases.get("case_direction", cases.get("direction"))))
    rng = np.random.default_rng(seed + 1)
    used, picks, owner, n_gene, short = set(), [], [], 0, 0
    for vid in rng.permutation(cases["variant_id"].to_numpy()):
        idx = row_of[vid]
        x = p0[idx]
        got = []
        if by_gene and df.at[idx, gene_col] in by_gene:
            v, i = by_gene[df.at[idx, gene_col]]
            got = nearest_unused(v, i, x, n_extra, caliper, used)
        n_gene += len(got)
        if len(got) < n_extra:
            got += nearest_unused(all_vals, all_ids, x, n_extra - len(got), caliper, used | set(got))
        short += len(got) < n_extra
        used.update(got)
        picks += got
        owner += [vid] * len(got)
    new = df.loc[picks].assign(ag_group="matched_concordant", match_case=owner,
                               case_direction=[direction[v] for v in owner])
    print(f"Extended: {len(new):,} additional controls ({n_gene:,} from the same gene); "
          f"{short:,} cases got fewer than {n_extra} extra")
    return new


def select_variants(df, how, n_controls, match_gene, gene_col, seed, caliper):
    if how == "none":
        return df.assign(ag_group="input")
    for c in ("discordant", "confident", "p_without"):
        if c not in df:
            sys.exit(f"--select {how} needs column '{c}' (use scores_all.tsv.gz from ensemble_discordant_VUS.py)")
    disc = as_bool(df["discordant"])
    sel = as_bool(df["confident"]) if how == "confident" else disc
    p0 = pd.to_numeric(df["p_without"], errors="coerce")
    ok = p0.notna() & pd.to_numeric(df["p_with"], errors="coerce").notna()
    cases = df[sel & ok].assign(ag_group=f"{how}_discordant")
    print(f"Selected {len(cases):,} {how} discordant variants")
    cases["match_case"] = cases["variant_id"]
    cases["case_direction"] = cases["direction"] if "direction" in cases else ""
    if n_controls <= 0:
        return cases

    pool = df.index[~disc & ok]
    order = np.argsort(p0[pool].to_numpy(), kind="stable")
    all_vals, all_ids = p0[pool].to_numpy()[order], pool.to_numpy()[order]
    by_gene = {}
    if match_gene and gene_col:
        g = df.loc[pool, gene_col].to_numpy()[order]
        for gene, idx in pd.Series(range(len(g))).groupby(g).groups.items():
            idx = np.asarray(idx)
            by_gene[gene] = (all_vals[idx], all_ids[idx])

    rng = np.random.default_rng(seed)
    used, picks, owner, n_gene, short = set(), [], [], 0, 0
    for idx in rng.permutation(cases.index.to_numpy()):          # random order: no case gets priority
        x = p0[idx]
        got = []
        if by_gene and df.at[idx, gene_col] in by_gene:
            v, i = by_gene[df.at[idx, gene_col]]
            got = nearest_unused(v, i, x, n_controls, caliper, used)
        n_gene += len(got)
        if len(got) < n_controls:
            got += nearest_unused(all_vals, all_ids, x, n_controls - len(got), caliper, used | set(got))
        short += len(got) < n_controls
        used.update(got)
        picks += got
        owner += [idx] * len(got)
    ctrl = df.loc[picks].assign(ag_group="matched_concordant",
                                match_case=cases.loc[owner, "variant_id"].to_numpy(),
                                case_direction=cases.loc[owner, "case_direction"].to_numpy())
    p_case = p0[owner].to_numpy()
    flipped_side = int(((p0[picks].to_numpy() >= 0.5) != (p_case >= 0.5)).sum())
    print(f"Controls on the other side of 0.5 from their case (p_without): {flipped_side}")
    d = np.abs(p0[picks].to_numpy().mean() - p0[cases.index].mean())
    print(f"Matched {len(ctrl):,} concordant controls by nearest p_without within +/-{caliper} "
          f"({n_gene:,} from the same gene); {short:,} variants got fewer than {n_controls}; "
          f"mean |difference| of group means {d:.4f}")
    return pd.concat([cases, ctrl])


# ----------------------------------------------------------------------------- aggregation
def track_label(df):
    lab = df.get("biosample_name", pd.Series("", index=df.index)).astype(str)
    for extra in ("transcription_factor", "histone_mark", "Assay title"):
        if extra in df:
            v = df[extra].astype(str)
            lab = np.where(v.isin(["nan", "None", ""]), lab, lab + "|" + v)
            lab = pd.Series(lab, index=df.index)
    return lab


def summarise(t, prefix, extreme_q=0.99):
    """Aggregate tidy rows of one scorer for one variant."""
    if t is None or t.empty:
        return {f"{prefix}_n": 0}
    use_q = "quantile_score" in t and t["quantile_score"].notna().any()
    s = t["quantile_score"] if use_q else t["raw_score"]
    s = pd.to_numeric(s, errors="coerce")
    ok = s.notna()
    if not ok.any():
        return {f"{prefix}_n": 0}
    s = s[ok]
    i, lo, hi = s.abs().idxmax(), s.idxmin(), s.idxmax()
    out = {f"{prefix}_n": int(ok.sum()), f"{prefix}_maxabs": float(abs(s[i])), f"{prefix}_signed": float(s[i]),
           f"{prefix}_mean": float(s.mean()), f"{prefix}_top_track": track_label(t.loc[[i]]).iloc[0],
           f"{prefix}_min": float(s[lo]), f"{prefix}_min_track": track_label(t.loc[[lo]]).iloc[0],
           f"{prefix}_max": float(s[hi]), f"{prefix}_max_track": track_label(t.loc[[hi]]).iloc[0]}
    out.update({f"{prefix}_median": float(s.median()), f"{prefix}_p10": float(s.quantile(0.1)),
                f"{prefix}_p90": float(s.quantile(0.9))})
    if use_q:
        nd, nu = int((s <= -extreme_q).sum()), int((s >= extreme_q).sum())
        out.update({f"{prefix}_n_down": nd, f"{prefix}_n_up": nu,
                    f"{prefix}_frac_down": nd / len(s), f"{prefix}_frac_up": nu / len(s)})
        if "raw_score" in t:
            r = pd.to_numeric(t.loc[s.index, "raw_score"], errors="coerce").dropna()
            if len(r):
                out.update({f"{prefix}_raw_mean": float(r.mean()), f"{prefix}_raw_median": float(r.median()),
                            f"{prefix}_raw_p10": float(r.quantile(0.1)), f"{prefix}_raw_p90": float(r.quantile(0.9)),
                            f"{prefix}_raw_min": float(r.min()), f"{prefix}_raw_max": float(r.max())})
    return out


def safe_tidy(tidy, adata):
    """tidy_scores for one scorer's AnnData; None when it holds no scores (tidy_scores raises a
    KeyError on empty results, e.g. no polyA site or no qualifying gene in the window)."""
    if adata is None or getattr(adata, "n_obs", 1) == 0 or getattr(adata, "n_vars", 1) == 0:
        return None
    try:
        t = tidy([[adata]])
    except KeyError:
        return None
    return None if t is None or t.empty else t


def own_gene_mask(t, gene, gene_id):
    """Rows of a gene-centric tidy table that belong to the variant's gene (symbol or Ensembl id)."""
    m = pd.Series(False, index=t.index)
    if gene_id and "gene_id" in t:
        m |= t["gene_id"].astype(str).str.split(".").str[0] == str(gene_id).split(".")[0]
    if gene:
        m |= t["gene_name"].astype(str) == str(gene)
    return m


def is_gene_centric(t):
    return t is not None and "gene_name" in t and t["gene_name"].notna().any()


def features_for_variant(results, keys, scorer_objs, tidy, gene, gene_id, ontology, extreme_q=0.99):
    row = {}
    for key, adata in zip(keys, results):
        t = safe_tidy(tidy, adata)
        if is_gene_centric(t):
            m = own_gene_mask(t, gene, gene_id)
            row[f"{key}_own_gene"] = bool(m.any())
            t = t[m]
        row.update(summarise(t, key, extreme_q))
        if "AVI" in key.upper():                     # the overall AVI score itself (one value per variant / gene)
            row["AVI_score"] = row.get(f"{key}_signed", np.nan)
        if ontology and t is not None and "ontology_curie" in t:
            row.update(summarise(t[t["ontology_curie"].isin(ontology)], f"{key}_tis", extreme_q))
    return row


# ----------------------------------------------------------------------------- Atlas
def dense(x):
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, float)


def anndata_to_tidy(ad):
    """One row per obs (variant x gene / junction) x var (track): obs and var columns, raw_score (X) and
    quantile_score (a quantile layer, if any); columns renamed to the tidy_scores names used by summarise()."""
    X = dense(ad.X).reshape(ad.n_obs, ad.n_vars)
    i, j = np.repeat(np.arange(ad.n_obs), ad.n_vars), np.tile(np.arange(ad.n_vars), ad.n_obs)
    obs = ad.obs.reset_index().rename(columns={"index": "obs_id", "strand": "gene_strand"}).iloc[i].reset_index(drop=True)
    var = ad.var.reset_index().rename(columns={"index": "track_id", "name": "track_name", "strand": "track_strand"})
    var = var.iloc[j].reset_index(drop=True)
    t = pd.concat([obs, var.drop(columns=[c for c in var if c in obs])], axis=1)
    t["raw_score"] = X.ravel()
    q = next((k for k in ("quantiles", "quantile", "quantile_score", "quantile_scores") if k in ad.layers), None)
    if q:
        t["quantile_score"] = dense(ad.layers[q]).reshape(ad.n_obs, ad.n_vars).ravel()
    return t.rename(columns={k: v for k, v in {"gene_symbol": "gene_name", "biosample": "biosample_name",
                                                "ontology_term": "ontology_curie"}.items() if k in t and v not in t})


def atlas_tidy(nested):
    """tidy_scores when it accepts the Atlas AnnData (same layout as score_variants), else anndata_to_tidy."""
    ad = nested[0][0]
    try:
        from alphagenome.models import variant_scorers
        t = variant_scorers.tidy_scores([[ad]])
        if t is not None and len(t):
            return t
    except Exception:                                                   # noqa: BLE001
        pass
    return anndata_to_tidy(ad)


def describe(key, ad):
    """--inspect: what the Atlas returned for one scorer and how it is read."""
    if ad is None:
        print(f"  {key}: nothing returned")
        return
    print(f"  {key}: AnnData {ad.n_obs} x {ad.n_vars}; obs {list(ad.obs.columns)}; var {list(ad.var.columns)[:12]}; "
          f"layers {list(ad.layers)}; uns {list(ad.uns)}")
    try:
        from alphagenome.models import variant_scorers
        t, how = variant_scorers.tidy_scores([[ad]]), "tidy_scores"
    except Exception as e:                                              # noqa: BLE001
        t, how = anndata_to_tidy(ad), f"anndata_to_tidy (tidy_scores failed: {type(e).__name__})"
    print(f"    read with {how}: {len(t)} rows; columns {list(t.columns)[:16]}")
    print(f"    raw_score {pd.to_numeric(t.get('raw_score'), errors='coerce').describe().round(4).to_dict()}")
    if "quantile_score" in t:
        print(f"    quantile_score {pd.to_numeric(t['quantile_score'], errors='coerce').describe().round(4).to_dict()}")


def compare_tables(atlas_path, live_path, out):
    """Agreement per feature between Atlas and live features of the same variants."""
    from scipy.stats import spearmanr
    A = pd.read_csv(atlas_path, sep="\t", dtype=str).drop_duplicates("variant_id").set_index("variant_id")
    L = pd.read_csv(live_path, sep="\t", dtype=str).drop_duplicates("variant_id").set_index("variant_id")
    ids = A.index.intersection(L.index)
    rows = []
    for c in [c for c in A if c in L and c.startswith(tuple(f"{k}_" for k in DEFAULT_SCORERS))]:
        x, y = pd.to_numeric(A.loc[ids, c], errors="coerce"), pd.to_numeric(L.loc[ids, c], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < 5:
            continue
        d = (x[ok] - y[ok]).abs()
        if x[ok].nunique() < 2 or y[ok].nunique() < 2:                     # constant: no correlation defined
            rho = np.nan
        else:
            rho = spearmanr(x[ok], y[ok])[0]
        rows.append({"feature": c, "n": int(ok.sum()), "spearman": rho,
                     "median_abs_diff": d.median(), "frac_equal": float((d <= 1e-6 * np.maximum(1, y[ok].abs())).mean()),
                     "atlas_median": x[ok].median(), "live_median": y[ok].median()})
    if not rows:
        print(f"Atlas vs live: {len(ids):,} variants in both tables; no feature with >= 5 values in both")
        return
    R = pd.DataFrame(rows).sort_values("spearman")
    R.to_csv(out, sep="\t", index=False)
    key = R[R.feature.isin(["RNA_SEQ_frac_down", "RNA_SEQ_raw_p10", "RNA_SEQ_mean", "SPLICE_SITES_maxabs",
                            "SPLICE_SITE_USAGE_maxabs", "SPLICE_JUNCTIONS_maxabs"])]
    print(f"\nAtlas vs live on {len(ids):,} variants ({len(R)} features): median Spearman {R.spearman.median():.3f}; "
          f"lowest:\n{R.head(8).round(4).to_string(index=False)}\nfeatures used in the figures:\n"
          f"{key.round(4).to_string(index=False)}\nwrote {out}")


def diagnose(a, atlas, live, keys, sel, variant_of, gcol, outdir):
    """--diagnose: the same variants scored by the Atlas and live today, compared track by track."""
    import importlib.metadata as md
    from alphagenome.models import dna_client, variant_scorers
    out = outdir
    rec = variant_scorers.RECOMMENDED_VARIANT_SCORERS
    seq_len = dna_client.SUPPORTED_SEQUENCE_LENGTHS[f"SEQUENCE_LENGTH_{a.sequence_length}"]
    meta = atlas.scorer_metadata()
    try:
        ver = md.version("alphagenome")
    except md.PackageNotFoundError:
        ver = "unknown"
    lines = [f"alphagenome package {ver}; live sequence length {a.sequence_length}; "
             f"{min(a.diagnose, len(sel))} variants"]
    lines += [f"Atlas scorer {k}: {meta.get(k)!r}"[:600] for k in keys]
    lines += [f"live scorer {k}: {rec[k]!r}"[:600] for k in keys if k in rec]
    keys = [k for k in keys if k in rec]             # live has no AVI score: compare the scorers both sides have
    rows, summ = [], []
    for _, r in sel.head(a.diagnose).iterrows():
        v = variant_of(r["variant_id"])
        g = r.get(gcol) if gcol else None
        A = atlas.query_variant(v, requested_scorers=keys)
        L = live.score_variant(interval=v.reference_interval.resize(seq_len), variant=v,
                               variant_scorers=[rec[k] for k in keys])
        for k, la in zip(keys, L):
            ta, tl = safe_tidy(atlas_tidy, A.get(k)), safe_tidy(variant_scorers.tidy_scores, la)
            for t in (ta, tl):
                if t is not None and "gene_id" in t:
                    t["gene_id"] = t["gene_id"].map(lambda x: x.split(".")[0] if isinstance(x, str) else x)
            on = [c for c in ("gene_id", "junction_Start", "junction_End", "track_name", "track_strand")
                  if ta is not None and tl is not None and c in ta and c in tl]
            s = {"variant_id": r["variant_id"], "gene": g, "scorer": k,
                 "atlas_rows": 0 if ta is None else len(ta), "live_rows": 0 if tl is None else len(tl),
                 "match_on": "+".join(on),
                 "atlas_interval": "" if ta is None else ";".join(map(str, pd.unique(ta.get("interval", pd.Series(dtype=str))))),
                 "live_interval": "" if tl is None else ";".join(map(str, pd.unique(tl.get("scored_interval", pd.Series(dtype=str))))),
                 "atlas_genes": "" if ta is None or "gene_name" not in ta else ";".join(map(str, sorted(pd.unique(ta.gene_name.astype(str))))),
                 "live_genes": "" if tl is None or "gene_name" not in tl else ";".join(map(str, sorted(pd.unique(tl.gene_name.astype(str)))))}
            if on:
                m = ta.drop_duplicates(on).merge(tl.drop_duplicates(on), on=on, how="outer", suffixes=("_atlas", "_live"),
                                                 indicator=True)
                s.update(matched=int((m._merge == "both").sum()), atlas_only=int((m._merge == "left_only").sum()),
                         live_only=int((m._merge == "right_only").sum()))
                b = m[m._merge == "both"]
                for sc in ("raw_score", "quantile_score"):
                    if f"{sc}_atlas" in b and f"{sc}_live" in b and len(b) > 2:
                        x, y = (pd.to_numeric(b[f"{sc}_{w}"], errors="coerce") for w in ("atlas", "live"))
                        ok = x.notna() & y.notna()
                        s[f"{sc}_spearman"] = x[ok].corr(y[ok], method="spearman") if ok.sum() > 2 else np.nan
                        s[f"{sc}_median_abs_diff"] = (x[ok] - y[ok]).abs().median()
                        s[f"{sc}_frac_equal"] = float(((x[ok] - y[ok]).abs() <= 1e-4 * np.maximum(1, y[ok].abs())).mean())
                rows.append(m.drop(columns="_merge").assign(variant_id=r["variant_id"], scorer=k, side=m._merge.astype(str)))
            summ.append(s)
    S = pd.DataFrame(summ)
    S.to_csv(out / "diagnose_summary.tsv", sep="\t", index=False)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out / "diagnose_tracks.tsv.gz", sep="\t", index=False)
    (out / "diagnose_versions.txt").write_text("\n".join(lines) + "\n")
    num = [c for c in S if c.endswith(("_rows", "matched", "_only", "_spearman", "_abs_diff", "_equal"))]
    print("\n".join(lines))
    print(f"\nper scorer (median over variants):\n{S.groupby('scorer')[num].median().round(4).to_string()}")
    print(f"\nwrote {out}/diagnose_summary.tsv (per variant x scorer: rows, intervals, genes, matched tracks, "
          f"agreement), diagnose_tracks.tsv.gz (every track of both), diagnose_versions.txt")


# ----------------------------------------------------------------------------- API
def make_client(api_key, timeout):
    from alphagenome.models import dna_client
    return dna_client.create(api_key, timeout=timeout)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="TSV with variant_id (and gene); not needed with --compare")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--select", choices=["confident", "discordant", "none"], default="confident")
    ap.add_argument("--n_controls", type=int, default=2, help="matched concordant controls per variant (0 = none)")
    ap.add_argument("--no_gene_match", action="store_true", help="ignore gene when matching controls")
    ap.add_argument("--extend_controls", type=int, default=0, metavar="N",
                    help="add N more matched controls per case to the saved selection (existing controls "
                         "and scores are kept; only the new controls are scored)")
    ap.add_argument("--caliper", type=float, default=0.01,
                    help="max |p_without difference| between a variant and its controls (default 0.01)")
    ap.add_argument("--scorers", nargs="+", default=None,
                    help="default: the 11 scorers below, plus the AVI score with --source atlas")
    ap.add_argument("--sequence_length", default="1MB", choices=["16KB", "100KB", "500KB", "1MB"])
    ap.add_argument("--ontology", nargs="+", default=None,
                    help="also aggregate over tracks of these ontology terms (e.g. UBERON:0000310 for breast)")
    ap.add_argument("--gene_col", default=None, help="gene symbol column (default: gene/GENE/genename)")
    ap.add_argument("--gene_id_col", default=None, help="Ensembl gene id column (optional)")
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--source", choices=["atlas", "live"], default="atlas",
                    help="atlas: precomputed AlphaGenome Atlas scores (default); live: on-demand score_variants")
    ap.add_argument("--inspect", type=int, default=0, metavar="N",
                    help="atlas: print what the Atlas returns for the first N variants, then exit")
    ap.add_argument("--compare", default=None, metavar="TABLE",
                    help="atlas: score the first --limit (default 200) variants of a live-scored feature table with "
                         "the Atlas and report agreement per feature (in <outdir>/atlas_compare/), then exit")
    ap.add_argument("--diagnose", type=int, default=0, metavar="N",
                    help="score the first N variants of --input with the Atlas and live now and compare track by "
                         "track (<outdir>/diagnose/), then exit")
    ap.add_argument("--compare_tables", nargs=2, metavar=("ATLAS", "LIVE"), default=None,
                    help="compare two feature tables already scored (Atlas, live) per feature -> <outdir>/atlas_vs_live.tsv")
    ap.add_argument("--atlas_gene_filter", action="store_true",
                    help="atlas: request only the variant's gene (gene_names) for smaller replies")
    ap.add_argument("--max_workers", type=int, default=10,
                    help="parallel requests (atlas: lookups in flight; live: score_variants workers)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=None, help="API timeout per request (s)")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N variants (pilot run)")
    ap.add_argument("--save_tidy", action="store_true",
                    help="also save per-track rows (own gene for gene-centric scorers) to tidy/batch_*.tsv.gz")
    ap.add_argument("--extreme_q", type=float, default=0.99,
                    help="quantile-score threshold for <S>_n_down / _n_up / _frac_down / _frac_up (default 0.99)")
    ap.add_argument("--out_name", default=OUT_NAME,
                    help=f"feature file in --outdir (default {OUT_NAME}); a new name re-scores everything "
                         "in the saved selection into that file")
    ap.add_argument("--api_key_file", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.compare_tables:
        Path(a.outdir).mkdir(parents=True, exist_ok=True)
        compare_tables(a.compare_tables[0], a.compare_tables[1], Path(a.outdir) / "atlas_vs_live.tsv")
        return
    if not a.input and not a.compare:
        ap.error("--input is required (unless --compare)")
    if a.diagnose or a.compare:
        a.source = "atlas"
    if not Path(a.outdir).name.endswith(f"_{a.source}"):           # Atlas and live features never share a folder
        ap.error(f"--outdir must end with _{a.source} (--source {a.source}), e.g. results/scores/vus_alphagenome_{a.source}")
    if a.diagnose:                                   # own folder; the scoring run's files stay untouched
        a.select, a.source, a.outdir = "none", "atlas", str(Path(a.outdir) / "diagnose")
    if a.compare:                                    # Atlas features of variants already scored live, own folder
        a.input, a.select, a.source = a.compare, "none", "atlas"
        a.outdir, a.limit = str(Path(a.outdir) / "atlas_compare"), a.limit or 200

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / a.out_name
    fail_path = outdir / ("failed.tsv" if a.out_name == OUT_NAME else f"failed_{Path(a.out_name).stem}.tsv")

    # ---- variants (selection is saved, so a re-run scores the same set)
    sel_path = outdir / "selected_variants.tsv"
    if a.select == "none" and a.out_name != OUT_NAME:  # a second list scored into the same folder keeps its own record
        sel_path = outdir / f"selected_variants_{Path(a.out_name).stem}.tsv"
    if sel_path.exists() and a.extend_controls:
        sel = pd.read_csv(sel_path, sep="\t", dtype=str)
        if "match_case" not in sel:
            sys.exit("--extend_controls needs a selection with match_case (re-run the selection first)")
        df = pd.read_csv(a.input, sep="\t", dtype=str, low_memory=False)
        if "variant_id" not in df and "variant_id_x" in df:
            df = df.rename(columns={"variant_id_x": "variant_id"})
        df = df.drop_duplicates("variant_id").reset_index(drop=True)
        gcol = a.gene_col or next((c for c in GENE_COLS if c in df), None)
        new = extend_controls(df, sel, a.extend_controls, not a.no_gene_match, gcol, a.seed, a.caliper)
        new = new[[c for c in sel.columns if c in new.columns]]
        backup = outdir / f"selected_variants.before_extend_{len(sel)}.tsv"
        sel.to_csv(backup, sep="\t", index=False)
        sel = pd.concat([sel, new], ignore_index=True)
        sel.to_csv(sel_path, sep="\t", index=False)
        per = sel[sel.ag_group == "matched_concordant"].groupby("match_case").size()
        print(f"Wrote {sel_path} ({len(sel):,} variants; controls per case: "
              f"{per.value_counts().sort_index().to_dict()}); previous selection kept as {backup.name}")
    elif sel_path.exists() and a.select != "none":     # --select none: always the rows of --input
        sel = pd.read_csv(sel_path, sep="\t", dtype=str)
        print(f"Using the saved selection {sel_path} ({len(sel):,} variants)")
    else:
        df = pd.read_csv(a.input, sep="\t", dtype=str, low_memory=False)
        if "variant_id" not in df and "variant_id_x" in df:
            df = df.rename(columns={"variant_id_x": "variant_id"})
        df = df.drop_duplicates("variant_id")
        gcol = a.gene_col or next((c for c in GENE_COLS if c in df), None)
        sel = select_variants(df, a.select, a.n_controls, not a.no_gene_match, gcol, a.seed, a.caliper)
        keep = [c for c in CARRY if c in sel] + [c for c in (a.gene_col, a.gene_id_col) if c and c in sel]
        sel = sel[list(dict.fromkeys(keep + [c for c in ("ag_group", "match_case", "case_direction") if c in sel]))]
        sel.to_csv(sel_path, sep="\t", index=False)
        print(f"Wrote {sel_path}")
    gcol = a.gene_col or next((c for c in GENE_COLS if c in sel), None)

    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path, sep="\t", usecols=["variant_id"], dtype=str)["variant_id"])
    todo = sel[~sel["variant_id"].isin(done)]
    if a.limit:
        todo = todo.head(max(a.limit - len(done), 0))
    print(f"{len(done):,} already scored; {len(todo):,} to score")
    if todo.empty and not (a.inspect or a.diagnose):
        if a.compare:
            compare_tables(out_path, a.compare, outdir / "atlas_vs_live.tsv")
        return

    # ---- AlphaGenome
    key = None
    if a.api_key_file:
        key = Path(a.api_key_file).read_text().strip()
    key = key or os.environ.get("ALPHAGENOME_API_KEY")
    if not key:
        sys.exit("No API key: set ALPHAGENOME_API_KEY or pass --api_key_file")
    from alphagenome.data import genome
    from alphagenome.models import variant_scorers
    keys = [s.upper() for s in a.scorers] if a.scorers else list(DEFAULT_SCORERS)

    def variant_of(vid):
        chrom, pos, alleles = vid.split(":")
        ref, alt = alleles.split(">")
        chrom = chrom if chrom.startswith("chr") else "chr" + chrom
        return genome.Variant(chromosome=chrom, position=int(pos), reference_bases=ref, alternate_bases=alt,
                              name=vid)

    if a.source == "atlas":
        try:
            from alphagenome.atlas import atlas as ag_atlas
        except ImportError:
            import importlib.metadata as md
            sys.exit(f"alphagenome {md.version('alphagenome')} has no Atlas client (alphagenome.atlas): "
                     "pip install -U 'alphagenome>=0.9' (or --source live)")
        client = ag_atlas.create(key, **({"timeout": a.timeout} if a.timeout else {}))
        avail = set(client.scorer_metadata())
        if not a.scorers:                            # default: + the Atlas's overall AVI score
            avi = sorted(k for k in avail if "AVI" in k.upper())
            if avi:
                keys.append(avi[0])
            else:
                print(f"WARNING: no AVI scorer in the Atlas ({sorted(avail)}); scoring the 11 scorers only", file=sys.stderr)
        if bad := [k for k in keys if k not in avail]:
            sys.exit(f"Scorers not in the Atlas: {bad}; available: {sorted(avail)}")
        scorer_objs, tidy_fn = keys, atlas_tidy
        print(f"AlphaGenome Atlas; scorers: {keys}; {a.max_workers} lookups in parallel")

        def lookup(r):
            g = r.get(gcol) if gcol else None
            kw = {"gene_names": [g]} if a.atlas_gene_filter and g else {}
            for attempt in range(a.retries):
                try:
                    res = client.query_variant(variant_of(r["variant_id"]), requested_scorers=keys, **kw)
                    return [res.get(k) for k in keys]
                except Exception as e:                                   # noqa: BLE001
                    wait = 10 * (attempt + 1)
                    print(f"  {r['variant_id']} failed ({type(e).__name__}: {str(e)[:150]}); retry in {wait}s",
                          file=sys.stderr)
                    time.sleep(wait)
            return None

        def score_batch_atlas(rows):                 # per variant: list of AnnData (one per scorer), or None
            with ThreadPoolExecutor(a.max_workers) as ex:
                return list(ex.map(lookup, [r for _, r in rows.iterrows()]))

        if a.diagnose:
            diagnose(a, client, make_client(key, a.timeout), keys, sel, variant_of, gcol, outdir)
            return
        if a.inspect:
            for _, r in sel.head(a.inspect).iterrows():
                res = client.query_variant(variant_of(r["variant_id"]), requested_scorers=keys)
                print(f"\n== {r['variant_id']} ({gcol} = {r.get(gcol) if gcol else None}); returned: {list(res)}")
                for k in keys:
                    describe(k, res.get(k))
            return
    else:
        from alphagenome.models import dna_client
        rec = variant_scorers.RECOMMENDED_VARIANT_SCORERS
        if bad := [k for k in keys if k not in rec]:
            sys.exit(f"Unknown scorers {bad}; available: {list(rec)}")
        scorer_objs, tidy_fn = [rec[k] for k in keys], variant_scorers.tidy_scores
        seq_len = dna_client.SUPPORTED_SEQUENCE_LENGTHS[f"SEQUENCE_LENGTH_{a.sequence_length}"]
        client = make_client(key, a.timeout)
        print(f"Scorers: {keys}; sequence length {a.sequence_length}")

    def score_batch_live(rows):
        variants = [variant_of(v) for v in rows["variant_id"]]
        intervals = [v.reference_interval.resize(seq_len) for v in variants]
        for attempt in range(a.retries):
            try:
                return client.score_variants(intervals=intervals, variants=variants, variant_scorers=scorer_objs,
                                             progress_bar=False, max_workers=a.max_workers)
            except Exception as e:                                       # noqa: BLE001
                wait = 30 * (attempt + 1)
                print(f"  batch failed ({type(e).__name__}: {str(e)[:200]}); retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
        return None

    score_batch = score_batch_atlas if a.source == "atlas" else score_batch_live

    tidy_dir = outdir / ("tidy" if a.out_name == OUT_NAME else f"tidy_{Path(a.out_name).stem}")
    if a.save_tidy:
        tidy_dir.mkdir(exist_ok=True)
    t0, n_done = time.time(), 0
    for b0 in range(0, len(todo), a.batch_size):
        rows = todo.iloc[b0:b0 + a.batch_size]
        res = score_batch(rows)
        if res is None:                                      # fall back to one variant at a time
            res = []
            for _, r in rows.iterrows():
                one = score_batch(r.to_frame().T)
                res.append(one[0] if one else None)
        feats, failed, tidy_rows = [], [], []
        for (_, r), result in zip(rows.iterrows(), res):
            if result is None:
                failed.append(r["variant_id"])
                continue
            f = features_for_variant(result, keys, scorer_objs, tidy_fn,
                                     r.get(gcol) if gcol else None,
                                     r.get(a.gene_id_col) if a.gene_id_col else None, a.ontology, a.extreme_q)
            feats.append({"variant_id": r["variant_id"], "ag_source": a.source, **f})
            if a.save_tidy:
                parts = []
                for key, ad in zip(keys, result):
                    t = safe_tidy(tidy_fn, ad)
                    if t is None:
                        continue
                    if is_gene_centric(t):
                        t = t[own_gene_mask(t, r.get(gcol) if gcol else None,
                                            r.get(a.gene_id_col) if a.gene_id_col else None)]
                    if len(t):
                        parts.append(t.assign(input_variant_id=r["variant_id"], scorer_key=key))
                if parts:
                    tidy_rows.append(pd.concat(parts, ignore_index=True))
        if feats:
            f = pd.DataFrame(feats)
            f = rows[[c for c in rows.columns]].merge(f, on="variant_id", how="inner")
            if out_path.exists():                       # keep a stable column set across batches
                cols = pd.read_csv(out_path, sep="\t", nrows=0).columns.tolist()
                new = [c for c in f.columns if c not in cols]
                if new:
                    old = pd.read_csv(out_path, sep="\t", dtype=str)
                    pd.concat([old, f], ignore_index=True).to_csv(out_path, sep="\t", index=False)
                else:
                    f.reindex(columns=cols).to_csv(out_path, sep="\t", index=False, mode="a", header=False)
            else:
                f.to_csv(out_path, sep="\t", index=False)
        if failed:
            pd.DataFrame({"variant_id": failed}).to_csv(fail_path, sep="\t", index=False, mode="a",
                                                        header=not fail_path.exists())
        if a.save_tidy and tidy_rows:
            tb = pd.concat(tidy_rows, ignore_index=True)
            tb.to_csv(tidy_dir / f"batch_{len(done) + b0:07d}.tsv.gz", sep="\t", index=False)
        n_done += len(rows)
        rate = (time.time() - t0) / n_done
        print(f"  {n_done:,}/{len(todo):,} variants ({len(failed)} failed in batch); "
              f"{rate:.1f} s/variant, ~{rate * (len(todo) - n_done) / 3600:.1f} h left", flush=True)
    print(f"\nWrote {out_path}" + (f"; failures in {fail_path}" if fail_path.exists() else ""))
    if a.compare:
        compare_tables(out_path, a.compare, outdir / "atlas_vs_live.tsv")


if __name__ == "__main__":
    main()
