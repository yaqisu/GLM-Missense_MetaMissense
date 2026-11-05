#!/usr/bin/env python3
"""
core.py — shared by every analysis script: methods, ensemble version, data, ensembles, statistics, figure style.

Ensemble version (--ensemble bergquist; MetaMissense = base + GLM-Missense)
  bergquist         Bergquist et al. 2025 (Genet Med), Table 1: BayesDel, MutPred2, REVEL, VEST4, AlphaMissense, ESM1b,
                    VARITY_R
  ENSEMBLES holds the released version only; another base is one more entry (name: (base, label)).

Scores: oriented so that higher = more pathogenic (ESM1b, SIFT, Evo 2 negated). dbNSFP scores are strict_transcript:
  the value of the transcript ClinVar named the variant on: <col>_fb where <col>_tier is clinvar_tx(_mapped) or variant
  (re-annotation by scoring/annotate_dbnsfp.py) when those columns exist; otherwise the entry of matched_enst in
  Ensembl_transcriptid (SGE: TRANSCRIPT_ID), and a single value (variant-level score) as is. Column names: the first of
  each method's candidates found in the table (python analysis/core.py prints which).
Variants: a missing score is never imputed; a variant without one of the scores an analysis uses is left out.
  ClinVar figures share one variant set: training and test variants with every score these figures use
  (FIGURE_SCORES: base + GLM-Missense, ESM1b-FT, SpliceAI, Random, COMPARE).
  MetaMissense is trained on every training variant with all of its own inputs. SGE: variants with every SGE score.
Ensembles: MetaMissense is grid-searched once (5 chromosome-exclusive folds, AUROC) and cached; every other
  ensemble (drop-column, +X | base) is XGBoost with MetaMissense's hyperparameters, fixed, seed 42, fit once on the
  training split of its variant set and evaluated once.
Calls: pathogenic / benign calls of single predictors (CALLS; resolved variants of fig6 / figS13, comparators of
  fig7). The tools calibrated by Bergquist et al. 2025 (Table 1) use their Supporting intervals (pathogenic >= PP3
  Supporting lower bound, benign <= BP4 Supporting upper bound, in between no call); uncalibrated scores (GLM-Missense,
  ESM1b-FT): one threshold (THRESH). "Most predictors disagree" = at least MAJORITY (2/3, rounded up) of the set: 5 of
  7 for the base (k_of).
VUS (fig7): the annotated held-out ClinVar VUS table (--vus, scoring/merge_vus.py); dbNSFP values of the ClinVar
  transcript (TRANSCRIPT_ID) by the same strict rule; VUS with every score of the VUS set (VUS_SCORES).
Statistics: --n_boot class-stratified bootstrap resamples from default_rng(--seed), the same for every row of a panel;
  two-sided bootstrap p; BH within a panel.
Style: 10 in wide, Arial; panel letters 18 pt bold, axis labels 16 pt, tick labels 14 pt, other text 12 pt; editable PDF.

Output (under --out, default results/analysis/)
  figures/<ensemble>/   every manuscript figure, main and supplementary (PDF + PNG)
  tables/<ensemble>/    one TSV per panel (<fig>_<panel>.tsv) + manifest_<fig>.json
  models/<ensemble>/    MetaMissense + every ablation ensemble (joblib, predictions, json), xgb_params.json
  prep/                 inputs built by prep_*.py
  tables/score_coverage.tsv, tables/variant_sets.tsv

    python analysis/core.py      # which column each score comes from, coverage, variant-set sizes
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''      # CPU only: XGBoost never opens a CUDA context

import joblib  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.preprocessing import QuantileTransformer  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]                               # repo root
MAIN = 'ClinVar.260923only.BLBvsPLP_{split}.main.tsv'
SGE_FILE = ('results/annotated/sge/RAD51D_missense_dbNSFP_annotated_GLM-Missense_annotated_ESM1b-FT_annotated'
            '.spliceai_annot.ESCOTT_annot.tsv')
TIERS = ['clinvar_tx', 'clinvar_tx_mapped', 'variant']                   # strict_transcript
TEST_CHROMS = {'2', '3', '16', '17'}

# ── methods ───────────────────────────────────────────────────────────────────
GLM, FT, SAI, RND, MM, EVO2, AVI = 'GLM-Missense', 'ESM1b-FT', 'SpliceAI', 'Random', 'MetaMissense', 'Evo2', 'AVI'
# key: (column candidates, display name, lower score = more pathogenic, colour)
METHODS = {
    GLM:             (['GLM-Missense_score'], 'GLM-Missense', False, '#C2185B'),
    FT:              (['ESM1b-FT_score'], 'ESM1b-FT', False, '#854F0B'),
    'AlphaMissense': (['AlphaMissense_score'], 'AlphaMissense', False, '#E6821E'),
    'ESM1b':         (['ESM1b_score'], 'ESM1b', True, '#D4A017'),
    'REVEL':         (['REVEL_score'], 'REVEL', False, '#1A9993'),
    'CADD':          (['CADD_phred'], 'CADD', False, '#2166AC'),
    'SIFT':          (['SIFT_score'], 'SIFT', True, '#4B0082'),
    'PolyPhen-2':    (['Polyphen2_HVAR_score'], 'PolyPhen-2', False, '#9467BD'),
    'BayesDel':      (['BayesDel_noAF_score'], 'BayesDel', False, '#253494'),           # without allele frequency
    'MutPred2':      (['MutPred2_score', 'MutPred_score'], 'MutPred2', False, '#7570B3'),  # dbNSFP >= 5.2: MutPred2
    'VEST4':         (['VEST4_score'], 'VEST4', False, '#17BECF'),
    'VARITY_R':      (['VARITY_R_LOO_score', 'VARITY_R_score'], 'VARITY_R', False, '#FB8072'),  # LOO if present
    SAI:             (['spliceai_DS_max'], 'SpliceAI', False, '#2E7D32'),
    RND:             (['random_baseline_score'], 'Random (null)', False, '#B4B2A9'),
    'GERP++':        (['GERP++_RS'], 'GERP++', False, '#8C8C8C'),
    'phyloP100':     (['phyloP100way_vertebrate'], 'phyloP (100 vert.)', False, '#8C8C8C'),
    'phyloP470':     (['phyloP470way_mammalian'], 'phyloP (470 mamm.)', False, '#8C8C8C'),
    EVO2:            (['evo2_delta'], 'Evo 2', True, '#8C8C8C'),                        # LL(alt) − LL(ref); reviewer analyses only
    AVI:             (['AVI_raw', 'AVI_phred'], 'AlphaGenome AVI', False, '#8C8C8C'),   # raw first; reviewer analyses only
}
DBNSFP = ['AlphaMissense', 'ESM1b', 'REVEL', 'CADD', 'SIFT', 'PolyPhen-2', 'BayesDel', 'MutPred2', 'VEST4', 'VARITY_R']
NAME = {k: v[1] for k, v in METHODS.items()} | {MM: 'MetaMissense'}
COLOR = {k: v[3] for k, v in METHODS.items()} | {MM: '#8B0000'}
SHORT = {GLM: 'GLM', FT: 'ESM1bFT', 'AlphaMissense': 'AM', 'ESM1b': 'ESM1b', 'REVEL': 'REVEL', 'CADD': 'CADD',
         'SIFT': 'SIFT', 'PolyPhen-2': 'PP2', 'BayesDel': 'BayesDel', 'MutPred2': 'MutPred2', 'VEST4': 'VEST4',
         'VARITY_R': 'VARITY', SAI: 'SAI', RND: 'Random', 'GERP++': 'GERP', 'phyloP100': 'phyloP100',
         'phyloP470': 'phyloP470', EVO2: 'Evo2', AVI: 'AVI'}
ORDER = list(METHODS)                                                    # fixed feature order of every ensemble
CONS = ['GERP++', 'phyloP100', 'phyloP470']
COMPARE = CONS                                                           # figS8 / figS9 comparison scores, never in a base

# ── ensemble versions ─────────────────────────────────────────────────────────
BERGQUIST = ['BayesDel', 'MutPred2', 'REVEL', 'VEST4', 'AlphaMissense', 'ESM1b', 'VARITY_R']   # 2025, Table 1
ENSEMBLES = {'bergquist': (BERGQUIST, 'Bergquist')}                    # the released version

MM_GRID = {'n_estimators': [100, 300, 500], 'max_depth': [3, 4, 5], 'learning_rate': [0.01, 0.05, 0.1],
           'subsample': [0.8, 1.0]}
XGB_SEED = 42
DIFF = ['dAUROC', 'dAUPRC', 'logloss_reduction']
DLAB = {'dAUROC': 'ΔAUROC', 'dAUPRC': 'ΔAUPRC', 'logloss_reduction': 'Log-loss reduction'}
SGE_CLASS = {'functionally_abnormal': 1, 'functionally_normal': 0}       # other SGE classes are left out
VUS_FILE = 'results/annotated/vus/ClinVar_20260923_VUS_heldout_annotated.tsv'          # scoring/merge_vus.py

# ── pathogenic / benign calls ─────────────────────────────────────────────────
# one threshold, oriented scores (ESM1b, SIFT negated): pathogenic at or above, else benign
THRESH = {GLM: 0.5, FT: 0.5, 'AlphaMissense': 0.564, 'ESM1b': 7.5, 'REVEL': 0.5, 'CADD': 20.0, 'PolyPhen-2': 0.902,
          'SIFT': -0.05, 'BayesDel': -0.0570105, 'MutPred2': 0.5, 'VEST4': 0.5, 'VARITY_R': 0.5}
# Bergquist et al. 2025, Table 1, raw scores: (benign = at or below the BP4 Supporting upper bound,
#                                              pathogenic = at or above the PP3 Supporting lower bound; ESM1b: lower = pathogenic)
CALIBRATED = {'BayesDel': (-0.180, 0.130), 'MutPred2': (0.391, 0.737), 'REVEL': (0.290, 0.644),
              'VEST4': (0.449, 0.764), 'AlphaMissense': (0.169, 0.792), 'ESM1b': (-6.3, -10.7), 'VARITY_R': (0.251, 0.675)}
CALIBRATED_VERSIONS = {'bergquist'}                                     # versions whose calls use CALIBRATED
MAJORITY = 2 / 3                                                         # "most predictors disagree": share of a set

# set by setup()
ARGS = ENS = BASE = BASE_FT = BL = BL_FT = DIRS = None
DTR = STR = DTE = STE = DALL = None
MAIN_TABLES = []
_MM, _SGE, _VUS = {}, {}, {}


# ── style ─────────────────────────────────────────────────────────────────────
for _d in [os.environ.get('ARIAL_DIR', ''), str(ROOT.parent / 'fonts')]:
    if _d and os.path.isdir(_d):
        for _f in os.listdir(_d):
            if _f.lower().endswith(('.ttf', '.otf')):
                fm.fontManager.addfont(os.path.join(_d, _f))
matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                            'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                            'axes.spines.top': False, 'axes.spines.right': False})
FS = {'panel': 18, 'label': 16, 'tick': 14, 'note': 12}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ── setup ─────────────────────────────────────────────────────────────────────
def parser(doc, ensemble=True):
    """Arguments shared by every script (paths relative to the repo root; absolute paths also work)."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    if ensemble:
        ap.add_argument('--ensemble', required=True, choices=list(ENSEMBLES), help='ensemble version')
    ap.add_argument('--main_dir', default='results/annotated/clinvar', help=f'ClinVar main tables ({MAIN})')
    ap.add_argument('--sge', default=SGE_FILE, help='RAD51D SGE table')
    ap.add_argument('--sge_label', default='FUNC_CONSEQUENCE', help=f'SGE class column ({", ".join(SGE_CLASS)})')
    ap.add_argument('--sge_rna', default='RNA_SCORE', help='SGE RNA score column')
    ap.add_argument('--gene_col', default='genename', help='gene symbol column of the ClinVar main tables')
    ap.add_argument('--vus', default=VUS_FILE, help='annotated held-out ClinVar VUS table (scoring/merge_vus.py)')
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_jobs', type=int, default=8)
    ap.add_argument('--out', default='results/analysis', help='output root')
    return ap


def setup(args):
    """Paths, the version's bases, the ClinVar tables and the shared figure variant set (DALL)."""
    global ARGS, ENS, BASE, BASE_FT, BL, BL_FT, DIRS, DTR, STR, DTE, STE, DALL
    for a in ('main_dir', 'sge', 'out', 'vus'):
        setattr(args, a, str(ROOT / getattr(args, a)))
    ARGS, ENS = args, args.ensemble
    BASE, BL = ENSEMBLES[ENS]
    BASE_FT = [FT if m == 'ESM1b' else m for m in BASE]                  # SGE base: ESM1b-FT in place of ESM1b
    BL_FT = f'{BL} (ESM1b-FT)'
    DIRS = {k: Path(args.out) / k / ENS for k in ('figures', 'tables', 'models', 'variant_lists')}
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)
    if 'arial' not in Path(fm.findfont('Arial')).name.lower():
        log('WARNING: Arial is not installed here; text falls back to ' + Path(fm.findfont('Arial')).name)
    log(f'ensemble version {ENS}: base = {", ".join(NAME[m] for m in BASE)}')
    DTE, STE = clinvar('test')
    DTR, STR = clinvar('train')
    assert not set(DTR.chromosome) & set(DTE.chromosome), 'training and test splits share chromosomes'
    miss = [m for m in FIGURE_SCORES() if STE[m].isna().all()]
    assert not miss, f'no values for {", ".join(miss)} in {args.main_dir} (python analysis/core.py shows the columns)'
    log('── ClinVar figure variant set')
    DALL = data(FIGURE_SCORES())


def FIGURE_SCORES():
    """Every score of the ClinVar figures of this version: they share one variant set."""
    return list(dict.fromkeys(BASE + [GLM, FT, SAI, RND] + COMPARE))


def SGE_SCORES():
    return list(dict.fromkeys(BASE_FT + [GLM, SAI, RND] + CONS))


def VUS_SCORES():
    """Scores the discordant-VUS selection needs (comparators = base, the two models); SpliceAI is a fig7 feature,
    also required by fig_vus.py (vus_set), with the other fig7 features."""
    return list(dict.fromkeys(BASE + [GLM, FT]))


def k_of(n):
    """'Most of n predictors': at least MAJORITY of them, rounded up (4 of 6, 5 of 7, 7 of 10)."""
    return int(np.ceil(MAJORITY * n - 1e-9))


def cutoffs():
    """Per score: (benign at or below, pathogenic at or above) on the oriented scores of this version."""
    out = {m: (t, t) for m, t in THRESH.items()}
    if ENS in CALIBRATED_VERSIONS:
        for m, (b, p) in CALIBRATED.items():
            out[m] = (-b, -p) if METHODS[m][2] else (b, p)               # ESM1b: oriented = −raw
    return out


def call(S, m):
    """1 = pathogenic, 0 = benign, NaN = no call (score missing, or between the calibrated intervals)."""
    v, (b, p) = S[m].to_numpy(float), cutoffs()[m]
    return np.select([v >= p, v <= b], [1.0, 0.0], np.nan)


# ── data ──────────────────────────────────────────────────────────────────────
def read_tsv(path, **kw):
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, low_memory=False, **kw)


def num(s):
    return pd.to_numeric(s, errors='coerce')                             # '.' and '' -> NaN


def first_num(d, c):
    """First value of a ';'-separated column, as a number."""
    return num(d[c].astype(str).str.split(';').str[0])


def column(d, m):
    """The first column candidate of method m found in d (or its _fb version), else None."""
    return next((c for c in METHODS[m][0] if c in d or f'{c}_fb' in d), None)


def tx_value(d, col, tx):
    """Strict per-transcript dbNSFP value: the entry of transcript tx when col lists one value per
    Ensembl_transcriptid entry; a single value as is (variant-level score); otherwise NaN."""
    vals = d[col].astype(str).str.split(';')
    txs = (d['Ensembl_transcriptid'].astype(str).str.split(';') if 'Ensembl_transcriptid' in d
           else vals.map(lambda _: []))
    out = []
    for v, t, x in zip(vals, txs, tx):
        v = [e.strip() for e in v]
        t = [e.strip().split('.')[0] for e in t]
        out.append(v[0] if len(v) == 1 else v[t.index(x)] if len(t) == len(v) and x in t else '')
    return num(pd.Series(out, index=d.index))


def score(d, m, tx):
    """Oriented strict score of method m (NaN where missing; all NaN if no column)."""
    col = column(d, m)
    if col is None:
        return pd.Series(np.nan, index=d.index)
    if f'{col}_fb' in d:
        v = num(d[f'{col}_fb']).where(d[f'{col}_tier'].isin(TIERS))
    elif m in DBNSFP:
        v = tx_value(d, col, tx)
    else:
        v = first_num(d, col)
    return -v if METHODS[m][2] else v


def clinvar(split):
    """ClinVar main table + oriented scores of every method."""
    path = Path(ARGS.main_dir) / MAIN.format(split=split)
    d = read_tsv(path)
    d['label'] = d.label.astype(int)
    d['chromosome'] = d.chromosome.astype(str).str.replace('^chr', '', regex=True)
    tx = d.matched_enst.astype(str).str.split('.').str[0] if 'matched_enst' in d else pd.Series('', index=d.index)
    S = pd.DataFrame({m: score(d, m, tx) for m in METHODS}, index=d.index)
    MAIN_TABLES.append(str(path))
    log(f'{path}: {len(d):,} variants ({d.label.sum():,} P/LP)')
    return d, S


def sge():
    """SGE table: label from --sge_label (functionally_abnormal = 1, functionally_normal = 0; other classes left out),
    oriented scores (dbNSFP: the value of the variant's TRANSCRIPT_ID), RNA score. Random: Uniform(0, 1) from
    default_rng(2026), one value per variant_id in sorted order, unless random_baseline_score is present."""
    if 'd' in _SGE:
        return _SGE['d'], _SGE['S']
    d = read_tsv(ARGS.sge)
    for c in (ARGS.sge_label, ARGS.sge_rna, 'variant_id', 'position', 'TRANSCRIPT_ID'):
        assert c in d, f'{ARGS.sge}: no column {c}'
    rcol = METHODS[RND][0][0]
    ids = np.sort(d.variant_id.unique())
    rnd = num(d[rcol]) if rcol in d else d.variant_id.map(pd.Series(np.random.default_rng(2026).uniform(size=len(ids)), index=ids))
    y = d[ARGS.sge_label].str.strip().map(SGE_CLASS)
    log(f'SGE {ARGS.sge}: {len(d):,} variants; {ARGS.sge_label} {d[ARGS.sge_label].value_counts().to_dict()}')
    keep = y.notna().to_numpy()
    d, rnd = d[keep].reset_index(drop=True), rnd[keep].reset_index(drop=True)
    d['label'] = y[keep].astype(int).to_numpy()
    d['tx'] = d.TRANSCRIPT_ID.str.split('.').str[0]
    S = pd.DataFrame({m: score(d, m, d.tx) for m in METHODS if m != RND}, index=d.index)
    S[RND] = rnd.to_numpy()
    S['RNA'] = first_num(d, ARGS.sge_rna)
    _SGE.update(d=d, S=S)
    return d, S


def subset(d, S, mask):
    return d[mask].reset_index(drop=True), S[mask].reset_index(drop=True)


def complete(S, cols):
    return S[cols].notna().all(axis=1).to_numpy()


def data(cols, keep_tr=True, keep_te=True, test=None):
    """(train, train scores, eval, eval scores): ClinVar training variants with every score in cols, and the
    evaluation variants (ClinVar test split, or test = (d, S), e.g. SGE) with every score in cols."""
    cols = list(dict.fromkeys(cols))
    tr, Sr = subset(DTR, STR, complete(STR, cols) & keep_tr)
    d, S = test or (DTE, STE)
    te, Se = subset(d, S, complete(S, cols) & keep_te)
    log(f'  variants with {", ".join(NAME[c] for c in cols)}: train {len(tr):,} ({tr.label.sum():,} P/LP), '
        f'{"SGE" if test else "test"} {len(te):,} ({te.label.sum():,} positive)')
    return tr, Sr, te, Se


def sge_data():
    """SGE figures: ClinVar training variants and SGE variants with every SGE score."""
    if 'D' not in _SGE:
        log('── SGE variant set')
        _SGE['D'] = data(SGE_SCORES(), test=sge())
    return _SGE['D']


def vus():
    """Annotated held-out ClinVar VUS (--vus, test chromosomes only) and their oriented scores; dbNSFP values of the
    ClinVar transcript (TRANSCRIPT_ID)."""
    if 'd' in _VUS:
        return _VUS['d'], _VUS['S']
    want = {c for m in METHODS for c0 in METHODS[m][0] for c in (c0, f'{c0}_fb', f'{c0}_tier')} | {
        'variant_id', 'variant_id_x', 'chromosome', 'gene', 'genename', 'TRANSCRIPT_ID', 'HGVS_C', 'PROTEIN_CHANGE',
        'clinvar_id', 'Ensembl_transcriptid'}
    d = read_tsv(ARGS.vus, usecols=lambda c: c in want).rename(columns={'variant_id_x': 'variant_id'})
    d = d.drop_duplicates('variant_id').reset_index(drop=True)
    chrom = (d.chromosome if 'chromosome' in d else d.variant_id.str.split(':').str[0]).astype(str)
    d = d[chrom.str.replace('^chr', '', regex=True).isin(TEST_CHROMS).to_numpy()].reset_index(drop=True)
    if 'gene' not in d:
        d['gene'] = d.get('genename', pd.Series('', index=d.index)).astype(str).str.split(';').str[0]
    tx = d.TRANSCRIPT_ID.astype(str).str.split('.').str[0]
    S = pd.DataFrame({m: score(d, m, tx) for m in METHODS if m != RND}, index=d.index)
    log(f'VUS {ARGS.vus}: {len(d):,} on test chromosomes')
    _VUS.update(d=d, S=S)
    return d, S


# ── ensembles ─────────────────────────────────────────────────────────────────
def md5_ids(ids):
    return hashlib.md5('\n'.join(sorted(map(str, ids))).encode()).hexdigest()


def canon(feats):
    return sorted(set(feats), key=ORDER.index)


def xgb(params, n_jobs):
    """The one XGBoost setup of MetaMissense, its grid search and every ablation ensemble."""
    return XGBClassifier(**params, eval_metric='logloss', tree_method='hist', device='cpu', n_jobs=n_jobs,
                         random_state=XGB_SEED)


def chrom_folds(chroms, k=5):
    """Chromosome-exclusive folds, largest chromosome first into the lightest fold."""
    counts, load_, c2f = chroms.value_counts(), np.zeros(k), {}
    for c in counts.index:
        c2f[c] = int(np.argmin(load_))
        load_[c2f[c]] += counts[c]
    fold = chroms.map(c2f).to_numpy()
    return [(np.flatnonzero(fold != j), np.flatnonzero(fold == j)) for j in range(k)]


def metamissense():
    """MetaMissense = BASE + GLM-Missense on every ClinVar training variant with all of these scores: grid search once,
    refit, cache. Returns {'model', 'features', 'params'}; its params are used, fixed, by every other ensemble."""
    if _MM:
        return _MM
    feats = canon(BASE + [GLM])
    tr, Sr = subset(DTR, STR, complete(STR, feats))
    key = {'features': feats, 'train_md5': md5_ids(tr.variant_id), 'grid': MM_GRID, 'seed': XGB_SEED}
    f_job, f_json = DIRS['models'] / 'MetaMissense.joblib', DIRS['models'] / 'MetaMissense.json'
    if f_job.exists() and f_json.exists() and json.loads(f_json.read_text()).get('key') == key:
        log(f'MetaMissense (cached): {json.loads(f_json.read_text())["params"]}')
        _MM.update(joblib.load(f_job))
        return _MM
    log(f'── MetaMissense: grid search on {len(tr):,} training variants with {", ".join(NAME[f] for f in feats)}')
    X, y = Sr[feats].to_numpy(float), tr.label.to_numpy()
    gs = GridSearchCV(xgb({}, 1), MM_GRID, scoring='roc_auc', cv=chrom_folds(tr.chromosome), n_jobs=ARGS.n_jobs,
                      refit=False, error_score='raise').fit(X, y)
    m = xgb(gs.best_params_, ARGS.n_jobs).fit(X, y)
    te, Se = subset(DTE, STE, complete(STE, feats))
    p = m.predict_proba(Se[feats].to_numpy(float))[:, 1]
    bundle = {'model': m, 'features': feats, 'params': gs.best_params_}
    joblib.dump(bundle, f_job)
    info = {'key': key, 'params': gs.best_params_, 'cv_auroc': gs.best_score_, 'n_train': len(tr), 'n_test': len(te),
            'test_auroc': auroc(te.label.to_numpy(), p)}
    f_json.write_text(json.dumps(info, indent=2))
    (DIRS['models'] / 'xgb_params.json').write_text(json.dumps(
        {'ensemble': ENS, 'features': feats, 'params': gs.best_params_, 'seed': XGB_SEED, 'grid': MM_GRID,
         'cv_auroc': gs.best_score_}, indent=2))
    log(f'  MetaMissense: {gs.best_params_}, CV AUROC {gs.best_score_:.4f}, test AUROC {info["test_auroc"]:.4f} '
        f'(n train {len(tr):,}, test with its inputs {len(te):,})')
    _MM.update(bundle)
    return _MM


def mm_prob(Se):
    """MetaMissense probabilities for the variants of score table Se."""
    M = metamissense()
    return M['model'].predict_proba(Se[M['features']].to_numpy(float))[:, 1]


def ensemble(feats, D, tag):
    """XGBoost with MetaMissense's hyperparameters on feats: fit on D's training variants, predictions for D's
    evaluation variants (cached in models/<ensemble>/)."""
    tr, Sr, te, Se = D
    feats, params = canon(feats), metamissense()['params']
    name = f'{tag}__' + '_'.join(SHORT[f] for f in feats)
    key = {'features': feats, 'train_md5': md5_ids(tr.variant_id), 'test_md5': md5_ids(te.variant_id),
           'params': params, 'seed': XGB_SEED}
    f_npz, f_json = DIRS['models'] / f'{name}.npz', DIRS['models'] / f'{name}.json'
    if f_npz.exists() and f_json.exists() and json.loads(f_json.read_text()).get('key') == key:
        return np.load(f_npz)['p']
    m = xgb(params, ARGS.n_jobs).fit(Sr[feats].to_numpy(float), tr.label.to_numpy())
    p = m.predict_proba(Se[feats].to_numpy(float))[:, 1]
    np.savez(f_npz, p=p, variant_id=te.variant_id.to_numpy())
    joblib.dump({'model': m, 'features': feats, 'params': params}, DIRS['models'] / f'{name}.joblib')
    auc_ = auroc(te.label.to_numpy(), p) if 'label' in te else None              # unlabelled (VUS): none
    f_json.write_text(json.dumps({'key': key, 'auroc': auc_}, indent=2))
    log(f'  fitted {name}' + (f': AUROC {auc_:.4f}' if auc_ is not None else f' ({len(te):,} unlabelled variants)'))
    return p


def calibrated(m, D):
    """Probability of score m alone for D's evaluation variants: quantile-normal transform + logistic regression fit
    on D's training variants (used for the log-loss of single scores)."""
    tr, Sr, _, Se = D
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=ARGS.seed).fit(Sr[[m]])
    return LogisticRegression(max_iter=2000).fit(qt.transform(Sr[[m]]), tr.label.to_numpy()).predict_proba(
        qt.transform(Se[[m]]))[:, 1]


# ── statistics ────────────────────────────────────────────────────────────────
def auroc(y, s):
    r = rankdata(s)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def auprc(y, s):
    """Average precision with ties handled as in sklearn (one threshold per distinct score)."""
    o = np.argsort(-s, kind='mergesort')
    y, s = y[o], s[o]
    idx = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[idx]
    return np.sum(np.diff(np.r_[0, tp / tp[-1]]) * tp / (idx + 1))


def metrics(p, y):
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return np.array([auroc(y, p), auprc(y, p), -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))])


def boot_idx(y, seed=None):
    """Class-stratified resamples (the same for every row of a panel); seed default --seed."""
    rng = np.random.default_rng(ARGS.seed if seed is None else seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(ARGS.n_boot)]


def p_boot(d):
    """Two-sided bootstrap p for H0: difference = 0 (floor 1/n_boot)."""
    d = np.asarray(d)
    return max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / len(d))


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        pv = p[ok]
        o = np.argsort(pv)
        r = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
        q[np.flatnonzero(ok)[o]] = np.minimum(np.minimum.accumulate(r[::-1])[::-1], 1)
    return q


def xerr(e, lo, hi):
    """Error-bar lengths for matplotlib; 0 where the point estimate falls outside its percentile CI."""
    e, lo, hi = (np.atleast_1d(np.asarray(v, float)) for v in (e, lo, hi))
    return np.clip(np.vstack([e - lo, hi - e]), 0, None)


def compare(pairs, P, y):
    """pairs: [(label, key A, key B, colour)]; A − B per metric (positive = A better; log-loss: B − A), paired
    bootstrap 95% CI and two-sided p, BH q within the call."""
    keys = sorted({k for _, a, b, _ in pairs for k in (a, b)})
    boots = boot_idx(y)
    est = {k: metrics(P[k], y) for k in keys}
    B = {k: np.array([metrics(P[k][i], y[i]) for i in boots]) for k in keys}
    sign = np.array([1, 1, -1])
    rows = []
    for lab, a, b, c in pairs:
        d, db = sign * (est[a] - est[b]), sign * (B[a] - B[b])
        r = {'comparison': lab, 'A': a, 'B': b, 'color': c, 'AUROC_A': est[a][0], 'AUROC_B': est[b][0],
             'AUPRC_A': est[a][1], 'AUPRC_B': est[b][1]}
        for j, m in enumerate(DIFF):
            r.update({m: d[j], f'{m}_lo': np.percentile(db[:, j], 2.5), f'{m}_hi': np.percentile(db[:, j], 97.5),
                      f'p_{m}': p_boot(db[:, j])})
        rows.append(r)
    T = pd.DataFrame(rows)
    for m in DIFF:
        T[f'q_{m}'] = bh(T[f'p_{m}'])
    return T.assign(n=len(y), n_pos=int(y.sum()))


def tier_table(methods, est, boots, alpha=0.05):
    """Estimate + 95% CI per method, and whether it differs from the top method (paired: boots on the same
    resamples). same_tier_as_top = two-sided bootstrap p of top − method >= alpha (not shown equivalent).
    Random is never the top."""
    est, boots = np.asarray(est), np.asarray(boots)
    cand = [i for i, m in enumerate(methods) if m != RND]
    top = cand[int(np.argmax(est[cand]))]
    rows = []
    for j, m in enumerate(methods):
        d = boots[:, top] - boots[:, j]
        rows.append({'method': m, 'name': NAME[m], 'estimate': est[j],
                     'ci_lo': np.percentile(boots[:, j], 2.5), 'ci_hi': np.percentile(boots[:, j], 97.5),
                     'top': methods[top], 'diff_top_minus': est[top] - est[j] if j != top else 0.0,
                     'diff_lo': np.percentile(d, 2.5) if j != top else np.nan,
                     'diff_hi': np.percentile(d, 97.5) if j != top else np.nan,
                     'p_vs_top': p_boot(d) if j != top else np.nan})
    t = pd.DataFrame(rows)
    t['same_tier_as_top'] = (t.method == methods[top]) | (t.p_vs_top >= alpha)
    t['rank'] = t.estimate.rank(ascending=False, method='min').astype(int)
    return t.sort_values('estimate', ascending=False).reset_index(drop=True)


# ── drawing ───────────────────────────────────────────────────────────────────
def stars(q):
    return '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else 'ns'


def fig_letter(fig, s, x_in, y_in):
    W, H = fig.get_size_inches()
    fig.text(x_in / W, y_in / H, s, fontsize=FS['panel'], fontweight='bold', va='top', ha='left')


def axes_in(fig, x0, y0, w, h):
    W, H = fig.get_size_inches()
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


def color_labels(labels, colors):
    for lab, c in zip(labels, colors):
        lab.set_color(c)


def forest(ax, T, col, xlabel, labels=True):
    """One row per line of T: point + 95% CI; filled + stars if BH q < 0.05; dashed line at 0."""
    yy = -np.arange(len(T), dtype=float)
    for yi, r in zip(yy, T.itertuples()):
        v, lo, hi, q = getattr(r, col), getattr(r, f'{col}_lo'), getattr(r, f'{col}_hi'), getattr(r, f'q_{col}')
        ax.errorbar(v, yi, xerr=xerr(v, lo, hi), fmt='none', ecolor=r.color, capsize=2, lw=1.2)
        ax.scatter(v, yi, s=36, color=r.color if q < 0.05 else 'white', edgecolor=r.color, zorder=3)
        if q < 0.05:
            ax.text(max(v, hi), yi, ' ' + stars(q), color=r.color, va='center', fontsize=FS['note'])
    ax.axvline(0, color='grey', lw=0.8, ls='--')
    ax.locator_params(axis='x', nbins=4)                                   # tick labels never collide
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.15 * (hi - lo))                                 # room for the stars
    ax.set_yticks(yy)
    ax.set_yticklabels(T.comparison if labels else [], fontsize=FS['tick'])
    color_labels(ax.get_yticklabels(), T.color)
    ax.set_ylim(yy.min() - 0.6, 0.6)
    ax.set_xlabel(xlabel, fontsize=FS['label'])
    ax.tick_params(axis='x', labelsize=FS['tick'])


def forest3(fig, T, x0, y0, h, letters, title='', w=2.45, gap=0.3, heading=''):
    """ΔAUROC | ΔAUPRC | log-loss reduction, side by side, one row per comparison. letters: one per column
    ('CDE') or one for the row ('B'); title above the middle column (none in the manuscript figures); heading above
    the row labels (e.g. 'Predictor removed'). Returns the three axes."""
    axes, top = [], y0 + h + (0.75 if title else 0.45)
    for j, col in enumerate(DIFF):
        ax = axes_in(fig, x0 + j * (w + gap), y0, w, h)
        forest(ax, T, col, DLAB[col], labels=(j == 0))
        if j < len(letters):
            fig_letter(fig, letters[j], 0.05 if j == 0 else x0 + j * (w + gap) - 0.3, top)
        if j == 1 and title:
            ax.set_title(title, fontsize=FS['note'], pad=8)
        if j == 0 and heading:
            ax.text(-0.03, 1.0, heading, transform=ax.transAxes, ha='right', va='bottom', fontsize=FS['note'],
                    fontweight='bold')
        axes.append(ax)
    return axes


def same_ylim(axes):
    lo, hi = min(a.get_ylim()[0] for a in axes), max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)


def center(fig):
    """Shift every axes and figure text sideways so the drawn content is centred in the figure width."""
    W = fig.get_size_inches()[0]
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    dx = (W - bb.x0 - bb.x1) / 2 / W
    for ax in fig.axes:
        p = ax.get_position()
        ax.set_position([p.x0 + dx, p.y0, p.width, p.height])
    for t in fig.texts:
        x, y = t.get_position()
        t.set_position((x + dx, y))


def save(fig, path):
    """PDF (editable text) + PNG at exactly 10 in wide; the height is trimmed to the content."""
    W, _ = fig.get_size_inches()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    if bb.x0 < -0.01 or bb.x1 > W + 0.01:
        log(f'WARNING {path.name}: content spans {bb.x0:.2f}–{bb.x1:.2f} in, outside the {W:g}-in width')
    box = Bbox([[0, bb.y0 - 0.05], [W, bb.y1 + 0.05]])
    for ext in ('.pdf', '.png'):
        fig.savefig(path.with_suffix(ext), dpi=300, bbox_inches=box)
    plt.close(fig)
    log(f'saved {path.with_suffix(".pdf")} ({W:g} x {box.height:.2f} in)')


def finish(fig, name, tables, D=None, **manifest):
    """Figure -> figures/<ensemble>/<name>.pdf/png; tables -> tables/<ensemble>/<name>_<key>.tsv;
    manifest_<name>.json (variant sets, ensemble version, hyperparameters, inputs)."""
    save(fig, DIRS['figures'] / name)
    for k, t in tables.items():
        t.drop(columns=['color'], errors='ignore').to_csv(DIRS['tables'] / f'{name}_{k}.tsv', sep='\t',
                                                           index=(k == 'kendall'))
    if D is not None:
        tr, _, te, _ = D
        manifest = dict(n_eval=len(te), n_eval_pos=int(te.label.sum()), n_train=len(tr),
                        eval_variant_ids_md5=md5_ids(te.variant_id), train_variant_ids_md5=md5_ids(tr.variant_id),
                        **manifest)
    (DIRS['tables'] / f'manifest_{name}.json').write_text(json.dumps(dict(
        created=time.strftime('%Y-%m-%d %H:%M:%S'), script=Path(sys.argv[0]).name, figure=name, ensemble=ENS,
        base=BASE, columns={m: column(DTE, m) for m in METHODS}, main_tables=MAIN_TABLES,
        xgb=_MM.get('params'), xgb_seed=XGB_SEED, n_boot=ARGS.n_boot, seed=ARGS.seed, **manifest),
        indent=2, default=str))


def variant_list(name, sel, d, S, X=None, note=''):
    """Variants of an analysis, for manual review -> variant_lists/<ensemble>/<name>.tsv: the selection columns (sel,
    one row per variant and group, with variant_id), every oriented score (oriented_<score>; higher = more pathogenic)
    and call of this version (call_<score>: P / B / empty = no call), the features X (index variant_id; feature_<name>)
    if any, then every column of the source table d."""
    oriented = S[[m for m in METHODS if m in S and S[m].notna().any()]]
    calls = pd.DataFrame({f'call_{NAME[m]}': pd.Series(call(S, m), index=S.index).map({1.0: 'P', 0.0: 'B'}).fillna('')
                          for m in cutoffs() if m in S}, index=S.index)
    ann = pd.concat([d[['variant_id']], oriented.rename(columns=lambda m: f'oriented_{NAME.get(m, m)}'), calls], axis=1)
    if X is not None:
        ann = ann.merge(X.add_prefix('feature_').reset_index(), on='variant_id', how='left')
    src = d.drop(columns=['label'], errors='ignore').rename(columns=lambda c: c if c == 'variant_id' else f'src_{c}')
    ann = ann.drop_duplicates('variant_id').merge(src.drop_duplicates('variant_id'), on='variant_id', how='left')
    out = sel.merge(ann, on='variant_id', how='left', suffixes=('', '_ann'))
    f = DIRS['variant_lists'] / f'{name}.tsv'
    out.to_csv(f, sep='\t', index=False)
    log(f'  variant list: {len(out):,} rows ({out.variant_id.nunique():,} variants) -> {f}' + (f' ({note})' if note else ''))
    return out


def ens_title(T, D, label, members, base, where):
    extra = [m for m in members if m not in base]
    return (f'{label[0].upper() + label[1:]} + {" + ".join(NAME[m] for m in extra)} (AUROC {T.AUROC_full.iloc[0]:.3f})\n'
            f'{where} n = {len(D[2]):,}; XGBoost trained on ClinVar training variants (n = {len(D[0]):,})')


# ── check: python analysis/core.py ────────────────────────────────────────────
if __name__ == '__main__':
    args = parser(__doc__, ensemble=False).parse_args()
    args.ensemble = next(iter(ENSEMBLES))
    setup(args)
    rows = []
    for split, (d, S) in (('train', (DTR, STR)), ('test', (DTE, STE))):
        for m in METHODS:
            col = column(d, m)
            multi = (d[col].astype(str).str.contains(';').mean() if col in d else np.nan) if col else np.nan
            rows.append({'split': split, 'method': m, 'column': col or '(none)',
                         'source': '' if not col else '_fb/_tier' if f'{col}_fb' in d else
                         'per transcript' if m in DBNSFP else 'variant', 'raw_multi_valued': multi,
                         'n': int(S[m].notna().sum()), 'frac': S[m].notna().mean()})
    cov = pd.DataFrame(rows)
    sz = []
    for e, (base, _) in ENSEMBLES.items():
        fs = list(dict.fromkeys(base + [GLM, FT, SAI, RND] + COMPARE))
        sz.append({'ensemble': e, 'set': 'MetaMissense training', 'n': int(complete(STR, base + [GLM]).sum()),
                   'n_PLP': int(DTR.label[complete(STR, base + [GLM])].sum())})
        for split, (d, S) in (('train', (DTR, STR)), ('test', (DTE, STE))):
            k = complete(S, fs)
            sz.append({'ensemble': e, 'set': f'figure set, {split}', 'n': int(k.sum()), 'n_PLP': int(d.label[k].sum())})
    if Path(args.vus).exists():
        dv, Sv = vus()
        for m in METHODS:
            if m in Sv:
                col = column(dv, m)
                rows.append({'split': 'VUS', 'method': m, 'column': col or '(none)',
                             'source': '' if not col else '_fb/_tier' if f'{col}_fb' in dv else
                             'per transcript' if m in DBNSFP else 'variant',
                             'raw_multi_valued': dv[col].astype(str).str.contains(';').mean() if col in dv else np.nan,
                             'n': int(Sv[m].notna().sum()), 'frac': Sv[m].notna().mean()})
        for e, (base, _) in ENSEMBLES.items():
            sz.append({'ensemble': e, 'set': 'VUS set (base, GLM-Missense, ESM1b-FT)',
                       'n': int(complete(Sv, list(dict.fromkeys(base + [GLM, FT]))).sum()), 'n_PLP': np.nan})
        cov = pd.DataFrame(rows)
    sz = pd.DataFrame(sz)
    with pd.option_context('display.width', 200, 'display.max_rows', 100):
        print(cov.pivot_table(index=['method', 'column', 'source'], columns='split', values=['n', 'raw_multi_valued'],
                              sort=False).round(3).to_string())
        print(sz.to_string(index=False))
    out = Path(args.out) / 'tables'
    cov.to_csv(out / 'score_coverage.tsv', sep='\t', index=False)
    sz.to_csv(out / 'variant_sets.tsv', sep='\t', index=False)
    log(f'wrote {out}/score_coverage.tsv, variant_sets.tsv')