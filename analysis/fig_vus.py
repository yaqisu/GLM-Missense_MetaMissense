#!/usr/bin/env python3
"""
fig_vus.py — on which VUS does GLM-Missense disagree with the established predictors, and do those VUS carry
nucleotide-level signals (splicing, codon-level mRNA stability) that protein-level predictors cannot see? ESM1b-FT
(fine-tuned on the same ClinVar variants as GLM-Missense, protein sequence only) is the negative control: a signal both
models show comes from the fine-tuning data; one only GLM-Missense shows is its own. Without labels, an enrichment
shows what a model responds to, not that its call is right.

VUS: held-out ClinVar missense VUS (chr 2, 3, 16, 17) with every score the selection needs (core.VUS_SCORES: base,
GLM-Missense, ESM1b-FT); both models are analysed on the same VUS. The three fig7 features (SpliceAI, mean ΔCSC of
the six CSC sets, Saluki Δ half-life) are required: only VUS with SpliceAI, all six ΔCSC sets and Saluki, filtered
before cases and controls are chosen, so every panel is on the same pairs.
Figures: all pairs, one row (A–C). q: BH over the 6 tests shown, per model (main). Table main_nosplice: the same on the
pairs whose case and control both have SpliceAI < SPLICE_CUT (0.2), its own q. Other features (AlphaGenome, FIMO, ...)
are not required: each is tested on the selected VUS that have it.

Discordant VUS. Comparators = the base, called by core.call (bergquist version: Bergquist et al. 2025 Table 1
Supporting intervals, in between = no call). Per model and direction: case = the model's confident call (>= P_THR
pathogenic, <= B_THR benign) opposed by most comparators (core.k_of: >= 2/3 of the base, e.g. 5 of 7); control = the
model's opposite confident call with the same number of comparators against, from the case's gene: of those not yet
used, the one nearest to the case in genomic position (1 per case, without replacement; a case without one is dropped;
most confident cases matched first, up to --max_variants / 2 cases; --seed only breaks ties in score or distance, and
analysis/reviewer/fig7_seeds.py checks 100 seeds). Each
case vs its own control: Wilcoxon signed-rank (continuous) or exact McNemar (binary). Subsets: all, and no splicing
signal (SpliceAI < 0.2; tables only). Figure q: Benjamini-Hochberg over the tests the figure shows (its features x both
directions, all pairs), per model: 6 tests in fig7 and figS14.

  fig7    GLM-Missense, all pairs: A SpliceAI DS max, B mean ΔCSC (codon stability coefficient, alt − ref, Wu et al.
          2019; mean of the six sets CSC_SETS: 293T, HeLa, RPE endogenous, 293T and K562 ORFeome, K562 SLAM-seq), C
          Saluki Δ half-life; case vs control mean with 95% CI, stars of the q of the pair test (q in fig7_main.tsv);
          no title or footer
  figS14  fig7 for ESM1b-FT (control), same y axes
Tables (tables/<ensemble>/fig7_*.tsv; fig7 and figS14): main (bars with p and q), selection(_summary), tests (every
feature, both subsets, unadjusted p), groups, specificity (AUC(GLM-Missense) − AUC(ESM1b-FT), bootstrap CI
and p, BH over the figure's features x directions: is the enrichment specific to GLM-Missense?), candidates
(RNA-testing candidates: every VUS GLM-Missense calls pathogenic against most comparators; flagged if SpliceAI >= 0.2 or
<= 3 nt from a junction). Variant lists (variant_lists/<ensemble>/): fig7_discordant.tsv (cases and matched controls),
fig7_candidates.tsv; each with every score and call, every feature and every VUS-table column.

Features: SpliceAI DS max (and >= 0.2), AlphaGenome (every scorer x summary across tracks, + the AVI score (Atlas)),
Saluki Δ half-life (and |Δ|), sequence features (exon-junction distance (log10; <= 3 nt), codon usage / CSC, ESRseq,
CpG, transition, GC), FIMO TF-binding disruption (largest loss / gain / |Δ| of -log10 p over JASPAR 2026 motifs, motifs
lost / gained at p < 1e-4). Saluki, sequence and FIMO features come from the annotated VUS table (--vus;
scoring/merge_vus.py).
AlphaGenome, one source per run: (table) the annotated VUS table holds AlphaGenome features for >= 99% of the version's
VUS set (merge_vus.py --alphagenome): every case, nothing scored here; (run) otherwise: at most --max_variants (2,000)
cases + controls per model x direction, the most extreme cases first, and the selected VUS without features are listed
in results/analysis/prep/vus/missing_alphagenome_<version>.tsv and scored in this run into --alphagenome
(scoring/alphagenome_score.py --source <ag_source: atlas (default) or live> in the conda env alphagenome; needs the API
key file: --ag_key_file or AG_KEY_FILE; without it, or with --no_alphagenome_scoring, they are tested without
AlphaGenome features). Each feature is tested on the selected VUS that have it. --select_only selects, counts the
missing features and stops. Env lamar:

    python analysis/fig_vus.py --ensemble bergquist --select_only        # how many selected VUS lack features
    AG_KEY_FILE=$HOME/.alphagenome_api_key python analysis/fig_vus.py --ensemble bergquist --n_jobs 40
"""
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import binomtest, fisher_exact, mannwhitneyu, rankdata, wilcoxon

import core as C
from core import COLOR, FS, FT, GLM, NAME, SAI, plt

MODELS = [GLM, FT]                                                       # first = main, second = control
P_THR, B_THR = 0.8, 0.2                                                  # confident calls
MAX_VARIANTS, AG_COMPLETE = 2000, 0.99            # 'run' mode: cases + controls per model x direction
SPLICE_CUT, JUNCTION_NT, MIN_N = 0.2, 3, 10
AG_PREFIXES = ('SPLICE_SITES_', 'SPLICE_SITE_USAGE_', 'SPLICE_JUNCTIONS_', 'RNA_SEQ_', 'POLYADENYLATION_', 'CHIP_TF_',
               'DNASE_', 'ATAC_', 'CHIP_HISTONE_', 'CAGE_', 'PROCAP_', 'AVI')
NOT_FEATURES = re.compile(r'(_track|_top_track|_own_gene|_n|_n_down|_n_up)$|^(saluki_(status|tx_length|tx_pos|trimmed|'
                          r'ref|alt|delta_sd|n_models)|fimo_(status|lost_tfs|gained_tfs|variant)|status|cds_pos|exon_number|'
                          r'n_exons|ref_matches|ref_matches_genome|'
                          r'consequence|hex_n)$')
DIRS = ['to_pathogenic', 'to_benign']
DIR_COL = {'to_pathogenic': '#2B7BD6', 'to_benign': '#EC6A33'}
# ΔCSC sets of the VUS table (seq_features.py --csc, Wu et al. 2019); fig7 B = their mean (one scale: correlations)
CSC_SETS = ['delta_CSC_293T_endo', 'delta_CSC_293T_ORFome', 'delta_CSC_HeLa_endo', 'delta_CSC_K562_ORFome',
            'delta_CSC_K562_SLAM', 'delta_CSC_RPE_endo']
# fig7 columns: (title, feature, y label, scale)
MAIN_COLS = [('Predicted splicing\ndisruption', 'spliceai_DS_max', 'SpliceAI Δ score', 1),
             ('Codon stability', 'delta_CSC_mean', 'Mean ΔCSC\n(×10$^{-2}$)', 1e2),
             ('Predicted mRNA\nhalf-life', 'saluki_delta', 'Δ mRNA half-life\n(×10$^{-3}$)', 1e3)]
KINDS = ['alphagenome', 'saluki', 'seq', 'fimo']
FIG7 = 'fig7'


# ── VUS, features ─────────────────────────────────────────────────────────────
def vus_set():
    """VUS with every score the selection needs (core.VUS_SCORES) and every fig7 feature (MAIN_COLS:
    SpliceAI, all six ΔCSC sets of the mean, Saluki), so all three panels are on the same VUS."""
    d, S = C.vus()
    keep = C.complete(S, C.VUS_SCORES())
    C.log(f'  VUS with {", ".join(NAME[m] for m in C.VUS_SCORES())}: {int(keep.sum()):,} of {len(d):,}')
    assert keep.sum(), f'no VUS with every score (is ESM1b-FT_score in {C.ARGS.vus}? scoring/merge_vus.py)'
    head = set(C.read_tsv(C.ARGS.vus, nrows=0).columns)
    raw = [g for _, f, _, _ in MAIN_COLS if f != 'spliceai_DS_max'
           for g in (CSC_SETS if f == 'delta_CSC_mean' else [f])]
    need = {f: next((c for c in (f, f'seq_{f}') if c in head), None) for f in raw}
    miss = [f for f, c in need.items() if c is None]
    assert not miss, f'{C.ARGS.vus} has no column for {miss} (scoring/merge_vus.py)'
    has = S[SAI].notna().to_numpy()                                   # SpliceAI: the oriented score of core.vus
    C.log(f'  spliceai_DS_max: {int((keep & ~has).sum()):,} of these VUS lack it')
    keep = keep & has
    t = C.read_tsv(C.ARGS.vus, usecols=lambda c: c in {'variant_id', 'variant_id_x', *need.values()})
    t = t.rename(columns={'variant_id_x': 'variant_id'}).drop_duplicates('variant_id').set_index('variant_id')
    for f, c in need.items():
        has = to_num(t[c]).reindex(d.variant_id).notna().to_numpy()
        C.log(f'  {f}: {int((keep & ~has).sum()):,} of these VUS lack it')
        keep = keep & has
    C.log(f'  VUS with every fig7 feature (spliceai_DS_max, {", ".join(need)}): {int(keep.sum()):,}')
    assert keep.sum(), 'no VUS with every fig7 feature'
    return C.subset(d, S, keep)


def to_num(s):
    return pd.to_numeric(s.replace({'True': '1', 'False': '0', 'true': '1', 'false': '0'}), errors='coerce')


def features(ids, S_by_id):
    """Numeric features of the VUS ids + derived ones: Saluki (saluki_*), sequence features (seq_*) and FIMO (fimo_*)
    from the annotated VUS table (--vus, scoring/merge_vus.py), AlphaGenome from one source (alphagenome()).
    Returns X and, per kind, the ids that have that kind of feature."""
    head = C.read_tsv(C.ARGS.vus, nrows=0).columns
    sal = [c for c in head if c.startswith('saluki_') and not NOT_FEATURES.search(c)]
    seq = [c for c in head if c.startswith('seq_') and not NOT_FEATURES.search(c.removeprefix('seq_'))]
    fim = [c for c in head if c.startswith('fimo_') and not NOT_FEATURES.search(c)]
    t = C.read_tsv(C.ARGS.vus, usecols=lambda c: c in {'variant_id', 'variant_id_x', 'saluki_status', 'seq_status',
                                                       'fimo_status', *sal, *seq, *fim})
    t = t.rename(columns={'variant_id_x': 'variant_id'})
    t = t[t.variant_id.isin(ids)].drop_duplicates('variant_id').set_index('variant_id')
    have = {k: set(t.index[t[f'{k}_status'].fillna('') != '']) if f'{k}_status' in t else set()
            for k in ('saluki', 'seq', 'fimo')}
    C.log(f'  Saluki / sequence / FIMO features: {len(have["saluki"]):,} / {len(have["seq"]):,} / {len(have["fimo"]):,} '
          f'selected VUS in {C.ARGS.vus}')
    X = t[sal + seq + fim].rename(columns={c: c.removeprefix('seq_') for c in seq})
    X = X.reindex(pd.Index(sorted(ids), name='variant_id'))
    A = alphagenome(ids)
    have['alphagenome'] = set(A.index)
    return derive(X.join(A, how='left').apply(to_num), S_by_id[SAI]), have


def derive(X, sai):
    """Numeric features X (index variant_id) + SpliceAI (sai, by variant_id) and the derived features; empty, constant
    and duplicate columns dropped."""
    X = X.loc[:, X.notna().any()].copy()
    X['spliceai_DS_max'] = sai.reindex(X.index)
    X['spliceai_ge_cut'] = (X.spliceai_DS_max >= SPLICE_CUT).astype(float).where(X.spliceai_DS_max.notna())
    if set(CSC_SETS) <= set(X):                                    # mean of the six; none unless all six
        X['delta_CSC_mean'] = X[CSC_SETS].mean(axis=1, skipna=False)
    if 'junction_dist' in X:
        X['junction_le_nt'] = (X.junction_dist <= JUNCTION_NT).astype(float).where(X.junction_dist.notna())
        X['junction_dist'] = np.log10(X.junction_dist.replace(np.inf, np.nan))
    if 'saluki_delta' in X:
        X['abs_saluki_delta'] = X.saluki_delta.abs()
    if {'fimo_loss', 'fimo_gain'} <= set(X):                    # signed: the larger of loss and gain (loss negative)
        X['fimo_delta'] = np.where(X.fimo_gain >= X.fimo_loss, X.fimo_gain, -X.fimo_loss)
        X['fimo_delta'] = X.fimo_delta.where(X.fimo_loss.notna() & X.fimo_gain.notna())
    X = X.loc[:, X.nunique() > 1]
    return X.loc[:, ~X.T.duplicated()]                # identical columns (e.g. summaries of a single AVI value) once


def ag_columns():
    """AlphaGenome feature columns of the annotated VUS table (merge_vus.py --alphagenome) and the marker column whose
    value says a variant was scored (<scorer>_n)."""
    head = C.read_tsv(C.ARGS.vus, nrows=0).columns
    cols = [c for c in head if c.startswith(AG_PREFIXES)]
    marker = next((c for c in ('RNA_SEQ_n', 'SPLICE_SITES_n') if c in cols), cols[0] if cols else None)
    return [c for c in cols if not NOT_FEATURES.search(c)], marker


def ag_mode(ids):
    """'table' when the annotated VUS table holds AlphaGenome features for >= AG_COMPLETE of the VUS set (the full
    precomputed scores, merged by merge_vus.py --alphagenome): every VUS case, no scoring here. Otherwise 'run': the
    selected VUS are scored in this run into --alphagenome (one file, one source); AlphaGenome columns of a partly
    annotated VUS table are not used, so sources are never mixed."""
    cols, marker = ag_columns()
    if not marker:
        C.log(f'  AlphaGenome: none in {C.ARGS.vus}; scored for the selected VUS in this run -> {C.ARGS.alphagenome}')
        return 'run'
    t = C.read_tsv(C.ARGS.vus, usecols=lambda c: c in ('variant_id', 'variant_id_x', marker))
    t = t.rename(columns={'variant_id_x': 'variant_id'})
    cov = t.loc[t[marker] != '', 'variant_id'].isin(ids).sum() / max(len(ids), 1)
    if cov >= AG_COMPLETE:
        C.log(f'  AlphaGenome: from {C.ARGS.vus} ({len(cols)} features; {cov:.1%} of the VUS set); every case')
        return 'table'
    C.log(f'  AlphaGenome: {C.ARGS.vus} covers only {cov:.1%} of the VUS set (< {AG_COMPLETE:.0%}): not used; scored for '
          f'the selected VUS in this run -> {C.ARGS.alphagenome}')
    return 'run'


def alphagenome(ids):
    """AlphaGenome features of the VUS ids from the source of this run (C.ARGS.ag_mode)."""
    if C.ARGS.ag_mode == 'table':
        cols, marker = ag_columns()
        a = C.read_tsv(C.ARGS.vus, usecols=lambda c: c in {'variant_id', 'variant_id_x', marker, *cols})
        a = a.rename(columns={'variant_id_x': 'variant_id'})
        a = a[a.variant_id.isin(ids) & (a[marker] != '')].drop_duplicates('variant_id').set_index('variant_id')
        src = C.ARGS.vus
    else:
        f = Path(C.ARGS.alphagenome)
        a = C.read_tsv(f) if f.exists() and f.stat().st_size else pd.DataFrame(columns=['variant_id'])
        a = a.rename(columns={'variant_id_x': 'variant_id'})
        a = a[a.variant_id.isin(ids)].drop_duplicates('variant_id').set_index('variant_id')
        cols, src = [c for c in a if c.startswith(AG_PREFIXES) and not NOT_FEATURES.search(c)], f
    C.log(f'  AlphaGenome: {len(a):,} selected VUS in {src}')
    return a[cols]


def missing_lists(sel_ids, cand_ids, have, d):
    """Count the selected VUS without each kind of feature; those without AlphaGenome features are listed in
    prep/vus/missing_alphagenome_<version>.tsv (the input of score_alphagenome)."""
    out = Path(C.ARGS.out) / 'prep' / 'vus'
    out.mkdir(parents=True, exist_ok=True)
    info = d.set_index('variant_id')[[c for c in ('gene', 'genename', 'TRANSCRIPT_ID') if c in d]]
    counts = {}
    for kind in KINDS:
        need = sorted((sel_ids | (cand_ids if kind == 'seq' else set())) - have[kind])
        counts[kind] = len(need)
        C.log(f'  {kind}: {len(need):,} selected VUS without features')
        if kind == 'alphagenome':
            info.loc[need].reset_index().to_csv(out / f'missing_alphagenome_{C.ENS}.tsv', sep='\t', index=False)
    return counts


def score_alphagenome(n_missing):
    """'run' mode: score the selected VUS that lack AlphaGenome features (scoring/alphagenome_score.py --source
    --ag_source in the conda env --ag_env) into --alphagenome. Returns True if it ran."""
    if not n_missing or C.ARGS.no_alphagenome_scoring or C.ARGS.ag_mode == 'table':
        return False
    key = C.ARGS.ag_key_file or os.environ.get('AG_KEY_FILE')
    if not key or not Path(key).is_file():
        C.log(f'WARNING: {n_missing:,} selected VUS without AlphaGenome features and no API key file (--ag_key_file or '
              f'AG_KEY_FILE): their AlphaGenome features are tested without them')
        return False
    lst = Path(C.ARGS.out) / 'prep' / 'vus' / f'missing_alphagenome_{C.ENS}.tsv'
    cmd = [C.ARGS.conda, 'run', '--no-capture-output', '-n', C.ARGS.ag_env, 'python', '-u',
           str(Path(__file__).resolve().parent / 'scoring' / 'alphagenome_score.py'), '--input', str(lst),
           '--select', 'none', '--source', C.ARGS.ag_source, '--outdir', str(Path(C.ARGS.alphagenome).parent),
           '--api_key_file', key, '--max_workers', str(C.ARGS.ag_workers), '--retries', '6',
           '--out_name', Path(C.ARGS.alphagenome).name]
    C.log(f'── AlphaGenome {C.ARGS.ag_source} scores for {n_missing:,} selected VUS (conda env {C.ARGS.ag_env}, '
          f'-> {C.ARGS.alphagenome}; ~{n_missing * 0.3 / 3600:.1f} h at 0.3 s per variant)')
    subprocess.run(cmd, check=True, cwd=C.ROOT)
    return True


# ── selection ─────────────────────────────────────────────────────────────────
def info_table(d, S, extra):
    t = d[[c for c in ('variant_id', 'gene', 'TRANSCRIPT_ID', 'HGVS_C', 'PROTEIN_CHANGE', 'clinvar_id') if c in d]].copy()
    for m in MODELS + [SAI]:
        t[f'{m}_score'] = S[m].to_numpy()
    return pd.concat([t, extra], axis=1)


def select_threshold(d, S):
    """fig7 cases and controls; candidates (GLM-Missense pathogenic against most comparators)."""
    base, K = C.BASE, C.k_of(len(C.BASE))
    calls = pd.DataFrame({m: C.call(S, m) for m in base}, index=d.index)
    nP, nB = (calls == 1).sum(axis=1).to_numpy(), (calls == 0).sum(axis=1).to_numpy()
    extra = pd.DataFrame({'n_comp_P': nP, 'n_comp_B': nB, **{f'call_{m}': calls[m].map({1.0: 'P', 0.0: 'B'}).fillna('')
                                                             for m in base}}, index=d.index)
    info = info_table(d, S, extra)
    C.log(f'  comparators {", ".join(NAME[m] for m in base)}; most = >= {K} of {len(base)}; ' + '; '.join(
        f'{NAME[m]} {np.nanmean(calls[m]):.1%} P, {calls[m].isna().mean():.1%} no call' for m in base))
    gene, rng = d.gene.to_numpy(), np.random.default_rng(C.ARGS.seed)
    pos = d.variant_id.str.split(':').str[1].astype(int).to_numpy()
    rows, summary, cand = [], [], None
    for m in MODELS:
        s = S[m].to_numpy(float)
        for direction, case_m, pool_m, cnt in (('to_pathogenic', s >= P_THR, s <= B_THR, nB),
                                               ('to_benign', s <= B_THR, s >= P_THR, nP)):
            cases, pool = np.flatnonzero(case_m & (cnt >= K)), np.flatnonzero(pool_m & (cnt >= K))
            if m == GLM and direction == 'to_pathogenic':
                cand = info.iloc[cases].assign(focal_score=s[cases]).sort_values('focal_score', ascending=False)
            n_all, cases = len(cases), rng.permutation(cases)
            cases = cases[np.argsort(-s[cases] if direction == 'to_pathogenic' else s[cases], kind='stable')]
            cap = C.ARGS.max_variants // 2 if C.ARGS.max_variants else None          # 1 control per case
            by_gs = {}
            for i in pool:
                by_gs.setdefault((gene[i], cnt[i]), []).append(i)
            by_gs = {k: np.array(v) for k, v in by_gs.items()}
            kept, dropped, dist = [], 0, []
            for i in cases:                                               # most confident first, until the cap
                if cap and len(kept) >= cap:
                    break
                cand_j = by_gs.get((gene[i], cnt[i]))
                if cand_j is None or not len(cand_j):
                    dropped += 1
                    continue
                dd = np.abs(pos[cand_j] - pos[i])                         # nearest unused control; ties: random
                ties = np.flatnonzero(dd == dd.min())
                k = ties[rng.integers(len(ties))] if len(ties) > 1 else ties[0]
                j = cand_j[k]
                by_gs[(gene[i], cnt[i])] = np.delete(cand_j, k)
                kept.append(s[i])
                dist.append(int(dd[k]))
                rows += [(m, direction, 'case', i, i, cnt[i], s[i]), (m, direction, 'control', j, i, cnt[j], s[j])]
            summary.append(dict(model=m, direction=direction, k=K, n_comparators=len(base), eligible_cases=n_all,
                                cap=cap or 0, cases=len(kept), cases_dropped_no_control=dropped,
                                control_pool=len(pool), case_score_min=min(kept, default=np.nan),
                                case_score_max=max(kept, default=np.nan),
                                median_distance_bp=float(np.median(dist)) if dist else np.nan))
    sel = pd.DataFrame(rows, columns=['model', 'direction', 'group', 'i', 'm', 'n_against', 'focal_score'])
    sel = sel.assign(variant_id=d.variant_id.to_numpy()[sel.i], match_case=d.variant_id.to_numpy()[sel.m])
    sel = sel.drop(columns=['i', 'm']).merge(info, on='variant_id', how='left')
    T = pd.DataFrame(summary)
    C.log('  selection:\n' + T.to_string(index=False))
    return sel, T, cand


# ── tests ─────────────────────────────────────────────────────────────────────
def auc(x1, x0):
    r = rankdata(np.concatenate([x1, x0]))
    return (r[:len(x1)].sum() - len(x1) * (len(x1) + 1) / 2) / (len(x1) * len(x0))


def tests(sel, X, paired):
    """Every feature, cases vs controls per model x direction x subset. Returns tests, group means, raw values."""
    binary = {c for c in X if set(X[c].dropna().unique()) <= {0.0, 1.0}}
    nosplice = (X.spliceai_DS_max < SPLICE_CUT).fillna(False)                 # SpliceAI only
    subsets = {'all': pd.Series(True, index=X.index), 'nosplice': nosplice}
    res, groups, data = [], [], {}
    for (m, dn), g in sel.groupby(['model', 'direction'], sort=False):
        for sub, keep in subsets.items():
            pr = g.loc[g.group == 'control', ['match_case', 'variant_id']]
            if paired:
                pr = pr[keep.reindex(pr.match_case).to_numpy() & keep.reindex(pr.variant_id).to_numpy()]
            ids1 = g.loc[g.group == 'case', 'variant_id']
            ids1, ids0 = ids1[keep.reindex(ids1).to_numpy()], pr.variant_id[keep.reindex(pr.variant_id).to_numpy()]
            for c in X:
                if sub == 'nosplice' and c == 'spliceai_ge_cut':
                    continue
                if paired:
                    y1, y0 = X.loc[pr.match_case, c].to_numpy(), X.loc[pr.variant_id, c].to_numpy()
                    ok = np.isfinite(y1) & np.isfinite(y0)
                    x1, x0 = y1[ok], y0[ok]
                else:
                    x1, x0 = X.loc[ids1, c].dropna().to_numpy(), X.loc[ids0, c].dropna().to_numpy()
                if len(x1) < MIN_N or len(x0) < MIN_N:
                    continue
                data[(m, dn, sub, c)] = (x1, x0)
                if paired and c in binary:                                # exact McNemar on discordant pairs
                    n10, n01 = int(((x1 == 1) & (x0 == 0)).sum()), int(((x1 == 0) & (x0 == 1)).sum())
                    p, stat, test = (binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0), \
                        (n10 / n01 if n01 else np.inf), 'mcnemar_exact'
                elif paired:
                    dd = x1 - x0
                    p, stat, test = (wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0), np.nan, 'wilcoxon_paired'
                elif c in binary:
                    stat, p = fisher_exact([[int(x1.sum()), len(x1) - int(x1.sum())], [int(x0.sum()), len(x0) - int(x0.sum())]])
                    test = 'fisher'
                else:
                    stat, p, test = np.nan, mannwhitneyu(x1, x0, alternative='two-sided').pvalue, 'mwu'
                for grp, x in (('case', x1), ('control', x0)):
                    se = x.std(ddof=1) / np.sqrt(len(x))
                    groups.append(dict(model=m, direction=dn, subset=sub, feature=c, group=grp, n=len(x), mean=x.mean(),
                                       ci_lo=x.mean() - 1.96 * se, ci_hi=x.mean() + 1.96 * se, median=np.median(x)))
                res.append(dict(model=m, direction=dn, subset=sub, feature=c, binary=c in binary, n_cases=len(x1),
                                n_controls=len(x0), case_mean=x1.mean(), control_mean=x0.mean(),
                                case_median=np.median(x1), control_median=np.median(x0),
                                median_paired_diff=np.median(x1 - x0) if paired else np.nan, auc=auc(x1, x0),
                                odds_ratio=stat, test=test, p=p))
    R = pd.DataFrame(res)
    if R.empty:                                                           # every group below MIN_N
        C.log(f'  no feature with >= {MIN_N} cases and controls')
        return R, pd.DataFrame(groups), data
    return R.sort_values(['subset', 'model', 'direction', 'p']), pd.DataFrame(groups), data


def specificity(data, cols):
    """Per feature of cols x direction (all pairs): AUC(GLM-Missense) − AUC(ESM1b-FT), bootstrap CI and p; q = BH over
    these tests."""
    rng, rows, sub = np.random.default_rng(C.ARGS.seed), [], 'all'
    for c, dn in [(col[1], dn) for col in cols for dn in DIRS]:
        k1, k0 = (GLM, dn, sub, c), (FT, dn, sub, c)
        if k1 not in data or k0 not in data:
            continue
        (a1, a0), (b1, b0) = data[k1], data[k0]
        boot = np.array([auc(rng.choice(a1, len(a1)), rng.choice(a0, len(a0))) - auc(rng.choice(b1, len(b1)),
                         rng.choice(b0, len(b0))) for _ in range(C.ARGS.n_boot)])
        rows.append(dict(direction=dn, subset=sub, feature=c, auc_GLM=auc(a1, a0), auc_ESM1bFT=auc(b1, b0),
                         delta_auc=auc(a1, a0) - auc(b1, b0), ci_lo=np.quantile(boot, 0.025),
                         ci_hi=np.quantile(boot, 0.975), p_delta=C.p_boot(boot)))
    S = pd.DataFrame(rows)
    if len(S):
        S['q_delta'] = C.bh(S.p_delta)
    return S


# ── figures ───────────────────────────────────────────────────────────────────
def main_table(R, G, cols=MAIN_COLS, subset='all'):
    """fig7 bars: every model x cols feature x direction x group of one subset ('all' pairs, or 'nosplice': case and
    control SpliceAI < SPLICE_CUT), with the test's p and q = BH over that row's tests (cols x directions), per model."""
    rows = []
    for m in MODELS:
        for c, (_, feat, _, scale) in enumerate(cols):
            for dn in DIRS:
                t = R[(R.model == m) & (R.subset == subset) & (R.feature == feat) & (R.direction == dn)]
                for grp in ('case', 'control'):
                    x = G[(G.model == m) & (G.subset == subset) & (G.feature == feat) & (G.direction == dn)
                          & (G.group == grp)]
                    if len(x) == 1 and len(t):
                        rows.append(dict(model=m, col=c, feature=feat, direction=dn, group=grp, scale=scale,
                                         **x[['n', 'mean', 'ci_lo', 'ci_hi']].iloc[0].to_dict(), test=t.test.iloc[0],
                                         p=t.p.iloc[0]))
                    else:
                        C.log(f'  not in the results, bar left empty: {m} / {feat} / {dn} / {grp}')
    T = pd.DataFrame(rows, columns=['model', 'col', 'feature', 'direction', 'group', 'scale', 'n', 'mean', 'ci_lo',
                                    'ci_hi', 'test', 'p'])
    k = ['model', 'feature', 'direction']                                 # one test per model x feature x direction
    u = T.drop_duplicates(k)[k + ['p']].copy()
    u['q'] = u.groupby('model').p.transform(lambda p: C.bh(p.to_numpy()))
    return T.merge(u.drop(columns='p'), on=k, how='left')


def qfmt(q):
    return f'{q:.2g}' if q >= 0.001 else '<1e-10' if q < 1e-10 else f'{q:.0e}'.replace('e-0', 'e-')


def bars_figure(d, m, cols=MAIN_COLS, rows=None):
    """One model, one panel per column of cols and per row (A, B, ... across, then down): discordant VUS (filled) vs
    matched control (open) mean with 95% CI per direction, stars of q above (q in the table). rows: [(heading, table)]
    or [(heading, table, model)] to show another model in that row (default: one row, d, no heading); a heading is
    drawn above its row. The y range of each panel is shared by both models (the table's rows of that column). No title
    or footer: model, n, tests and q go in the legend."""
    rows = [r if len(r) == 3 else (*r, m) for r in (rows or [('', d)])]
    nr, nc, wide = len(rows), len(cols), len(cols) > 3
    ph, gap_r, bot = 2.05, 1.35, 1.3 + (0.25 if wide else 0)                 # inches: panel, between rows, bottom
    top = 0.95 + (0.45 if rows[0][0] else 0)                                 # room for a heading above row 1
    H = top + nr * ph + (nr - 1) * gap_r + bot
    gap = 1.0                                                            # inches between panels (y ticks + label)
    w = (8.7 - gap * (nc - 1)) / nc
    fig, axes = plt.subplots(nr, nc, figsize=(10, H), squeeze=False,
                             gridspec_kw=dict(wspace=gap / w, hspace=gap_r / ph))
    fig.subplots_adjust(left=0.12, right=0.975, top=1 - top / H, bottom=bot / H)
    W, X = 0.4, (0, 1.6)                                                  # bar width; x of the two directions
    for r, (heading, dd, mr) in enumerate(rows):
        for c, (ctitle, _, ylab, sc) in enumerate(cols):
            ax, s = axes[r, c], dd[(dd.model == mr) & (dd.col == c)]
            a = dd[dd.col == c]
            lo, hi = min(0, (a.ci_lo * sc).min()), max(0, (a.ci_hi * sc).max())
            n = {}
            for k, dn in enumerate(DIRS):
                for j, grp in enumerate(('case', 'control')):
                    x = s[(s.direction == dn) & (s.group == grp)]
                    if not len(x):
                        continue
                    x = x.iloc[0]
                    n[dn] = int(x.n)
                    mu, lo_, hi_ = x['mean'] * sc, x.ci_lo * sc, x.ci_hi * sc
                    ax.bar(X[k] + (j - 0.5) * W, mu, W * 0.95, color=DIR_COL[dn] if grp == 'case' else 'white',
                           edgecolor=DIR_COL[dn], linewidth=1.6)
                    ax.errorbar(X[k] + (j - 0.5) * W, mu, yerr=[[mu - lo_], [hi_ - mu]], color='#333333', capsize=3,
                                lw=1.1)
                q = s[s.direction == dn].q
                if len(q):
                    ax.text(X[k], 0.98, C.stars(q.iloc[0]), transform=ax.get_xaxis_transform(), ha='center',
                            va='top', fontsize=FS['note'])
            if len(a):
                ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.2 * (hi - lo))
            if lo < 0:
                ax.axhline(0, color='#888888', lw=0.8)
            ax.set_xticks(X)
            lab = ('To\npathogenic', 'To\nbenign') if wide else ('To pathogenic', 'To benign')
            ax.set_xticklabels([(f'{t}\n({n[dn]:,})' if wide else f'{t}\nn = {n[dn]:,} pairs') if dn in n else t
                                for dn, t in zip(DIRS, lab)], fontsize=FS['note'])
            C.color_labels(ax.get_xticklabels(), [DIR_COL[x] for x in DIRS])
            ax.set_xlim(-0.55, 2.15)
            ax.tick_params(axis='y', labelsize=FS['tick'])
            ax.set_ylabel(ylab, fontsize=FS['note'])
            ax.text(-0.85 / w, 1.0, 'ABCDEFGHIJKL'[r * nc + c], transform=ax.transAxes, fontsize=FS['panel'],
                    fontweight='bold', va='bottom', ha='left')
            if r == 0:
                p_ = ax.get_position()
                fig.text((p_.x0 + p_.x1) / 2, 1 - 0.08 / H, ctitle, ha='center', va='top', fontsize=FS['tick'],
                         fontweight='bold')
        if heading:
            y_top = axes[r, 0].get_position().y1
            fig.text(0.5, y_top + 0.45 / H, heading, ha='center', va='bottom', fontsize=FS['tick'], fontweight='bold',
                     color=COLOR[mr] if len({x[2] for x in rows}) > 1 else 'black')   # models differ by row
    fig.legend(handles=[Patch(facecolor='#555555', edgecolor='#555555', label='Discordant VUS'),
                        Patch(facecolor='white', edgecolor='#555555', label='Matched control')],
               loc='lower center', ncol=2, frameon=False, fontsize=FS['note'], bbox_to_anchor=(0.5, 0.0))
    return fig


def fig7(d, S, X, sel, summary, cand, missing):
    R, G, data = tests(sel, X, paired=True)
    K, n = C.k_of(len(C.BASE)), len(C.BASE)
    md, mn = main_table(R, G), main_table(R, G, subset='nosplice')      # figure: all pairs; table: SpliceAI < cut
    figs = {m: bars_figure(md, m) for m in MODELS}                        # fig7 GLM-Missense, figS14 ESM1b-FT
    tabs = {'main': md, 'main_nosplice': mn, 'specificity': specificity(data, MAIN_COLS)}
    cand = cand.copy()
    jd = X.junction_dist.reindex(cand.variant_id) if 'junction_dist' in X else pd.Series(np.nan, index=cand.variant_id)
    cand['junction_dist'] = (10 ** jd).round().to_numpy()
    cand['RNA_test_candidate'] = (cand[f'{SAI}_score'] >= SPLICE_CUT).to_numpy() | (cand.junction_dist <= JUNCTION_NT).to_numpy()
    cand['ESM1b-FT_call'] = np.select([cand['ESM1b-FT_score'] >= P_THR, cand['ESM1b-FT_score'] <= B_THR], ['P', 'B'], '')
    C.log(f'  RNA-testing candidates: {len(cand):,} GLM-Missense-pathogenic VUS against most comparators, '
          f'{int(cand.RNA_test_candidate.sum()):,} flagged')
    keep = ['model', 'direction', 'group', 'variant_id', 'match_case', 'n_against', 'focal_score', 'n_comp_P', 'n_comp_B']
    C.variant_list(f'{FIG7}_discordant', sel[keep], d, S, X, 'cases and their matched controls')
    C.variant_list(f'{FIG7}_candidates', cand[['variant_id', 'focal_score', 'junction_dist', 'RNA_test_candidate',
                                            'n_comp_P', 'n_comp_B']], d, S, X, 'RNA-testing candidates')
    tabs.update(selection_summary=summary, selection=sel, tests=R, groups=G, candidates=cand)
    C.save(figs[FT], C.DIRS['figures'] / FIG7.replace('fig7', 'figS14'))
    C.finish(figs[GLM], FIG7, tabs, None, n_vus=len(d), k=K, n_comparators=n, comparators=C.BASE, p_thr=P_THR,
             control='nearest unused VUS in genomic position (same gene, same number of comparators against)',
             csc_sets=CSC_SETS,
             b_thr=B_THR, max_variants=C.ARGS.max_variants,
             alphagenome_mode=C.ARGS.ag_mode,
             calls_benign_pathogenic_oriented={m: C.cutoffs()[m] for m in C.BASE}, main_columns=MAIN_COLS,
             missing_features=missing, vus_table=C.ARGS.vus, alphagenome_source=C.ARGS.ag_source,
             alphagenome_table=C.ARGS.vus if C.ARGS.ag_mode == 'table' else C.ARGS.alphagenome)


if __name__ == '__main__':
    ap = C.parser(__doc__)
    ap.add_argument('--ag_source', choices=['atlas', 'live'], default=os.environ.get('AG_SOURCE', 'atlas'),
                    help='AlphaGenome per-track scores from the Atlas or live predictions (default: $AG_SOURCE or atlas)')
    ap.add_argument('--alphagenome', default=None,
                    help="'run' mode: the AlphaGenome table of this analysis' selections (default: "
                         "results/scores/vus_alphagenome_<ag_source>/alphagenome_features_figvus.tsv)")
    ap.add_argument('--ag_env', default='alphagenome', help='conda env with alphagenome >= 0.9 (default: alphagenome)')
    ap.add_argument('--ag_key_file', default=None, help='AlphaGenome API key file (default: $AG_KEY_FILE)')
    ap.add_argument('--ag_workers', type=int, default=None, help='parallel requests (default: atlas 20, live 5)')
    ap.add_argument('--conda', default=os.environ.get('CONDA_EXE', 'conda'), help='conda executable')
    ap.add_argument('--no_alphagenome_scoring', action='store_true',
                    help='do not score selected VUS without AlphaGenome features (their AlphaGenome features are tested '
                         'without them)')
    ap.add_argument('--max_variants', type=int, default=None,
                    help=f'cases + controls per model x direction, most extreme cases first (default: {MAX_VARIANTS} '
                         f"when AlphaGenome is scored in the run, no cap when the VUS table has it; 0 = no cap)")
    ap.add_argument('--select_only', action='store_true', help='select, count missing features and stop')
    args = ap.parse_args()
    args.alphagenome = str(C.ROOT / (args.alphagenome or
                                     f'results/scores/vus_alphagenome_{args.ag_source}/alphagenome_features_figvus.tsv'))
    assert Path(args.alphagenome).parent.name.endswith(f'_{args.ag_source}'), \
        f'--alphagenome must be in a folder ending with _{args.ag_source}'
    args.ag_workers = args.ag_workers or {'atlas': 20, 'live': 5}[args.ag_source]
    C.setup(args)
    C.log('── VUS')
    d, S = vus_set()
    args.ag_mode = ag_mode(set(d.variant_id))
    if args.max_variants is None:
        args.max_variants = 0 if args.ag_mode == 'table' else MAX_VARIANTS
    C.log('  cases per model x direction: ' + (f'the most extreme, at most {args.max_variants:,} cases + controls'
                                                if args.max_variants else 'all'))
    sel, summary, cand = select_threshold(d, S)
    sel_ids, cand_ids = set(sel.variant_id), set(cand.variant_id)
    X, have = features(sel_ids | cand_ids, S.set_index(d.variant_id))
    missing = missing_lists(sel_ids, cand_ids, have, d)
    if not args.select_only and score_alphagenome(missing['alphagenome']):
        X, have = features(sel_ids | cand_ids, S.set_index(d.variant_id))
        missing = missing_lists(sel_ids, cand_ids, have, d)
    if args.select_only:
        C.log('--select_only: stopping after the missing-feature lists')
    else:
        C.log('── fig7')
        fig7(d, S, X, sel, summary, cand, missing)