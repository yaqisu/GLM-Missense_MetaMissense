#!/usr/bin/env python3
"""
MetaMissense.py — score variants with the MetaMissense ensemble.

MetaMissense is an XGBoost ensemble of GLM-Missense and the seven predictors calibrated by ClinGen for PP3 / BP4
(Bergquist et al. 2025, Table 1): BayesDel (no AF), MutPred2, REVEL, VEST4, AlphaMissense, ESM1b and VARITY_R.
The model file (MetaMissense.joblib) holds {'model', 'features', 'params'}; 'features' are the method names below,
in the order the model expects. Scores are oriented so that higher = more pathogenic (ESM1b is negated), as in
training. A variant missing any input score gets NaN (missing scores are never imputed).

Usage (repo root):
    python scoring/MetaMissense.py --input results/scoring/MetaMissense_input.tsv \\
        --model scoring/MetaMissense.joblib --outdir results/scoring
Output: <outdir>/MetaMissense.tsv = the input table + MetaMissense_score (after GLM-Missense_score).
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

# method name in the model -> (input column candidates, negate so that higher = more pathogenic)
INPUTS = {
    'GLM-Missense':  (['GLM-Missense_score'], False),
    'AlphaMissense': (['AlphaMissense_score'], False),
    'ESM1b':         (['ESM1b_score'], True),
    'REVEL':         (['REVEL_score'], False),
    'BayesDel':      (['BayesDel_noAF_score'], False),
    'MutPred2':      (['MutPred2_score', 'MutPred_score'], False),
    'VEST4':         (['VEST4_score'], False),
    'VARITY_R':      (['VARITY_R_LOO_score', 'VARITY_R_score'], False),
}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--input', '-i', required=True, help='Input TSV (output of prepare_metamissense_input.py)')
ap.add_argument('--model', '-m', required=True, help='MetaMissense model (.joblib)')
ap.add_argument('--outdir', '-o', required=True, help='Output directory; MetaMissense.tsv is written here')
args = ap.parse_args()

bundle = joblib.load(args.model)
model, features = bundle['model'], bundle['features']
print(f'Loaded {args.model}: features {features}; hyperparameters {bundle.get("params")}')
unknown = [f for f in features if f not in INPUTS]
if unknown:
    sys.exit(f'ERROR: no input column defined for model features {unknown} (INPUTS in {__file__})')

df = pd.read_csv(args.input, sep='\t', na_values='.', low_memory=False)
print(f'Loaded {len(df):,} variants from {args.input}')

X = pd.DataFrame(index=df.index)
missing = []
for f in features:
    cols, negate = INPUTS[f]
    col = next((c for c in cols if c in df.columns), None)
    if col is None:
        missing.append(f'{f} ({" or ".join(cols)})')
        continue
    v = pd.to_numeric(df[col].astype(str).str.split(';').str[0], errors='coerce')   # one value per variant
    X[f] = -v if negate else v
    print(f'  {f:<14} <- {col}{" (negated)" if negate else ""}')
if missing:
    sys.exit(f'ERROR: input columns not found: {missing}')

valid = X.notna().all(axis=1).to_numpy()
scores = np.full(len(df), np.nan)
if valid.any():
    scores[valid] = model.predict_proba(X.loc[valid, features].to_numpy(float))[:, 1]
df['MetaMissense_score'] = scores
if 'GLM-Missense_score' in df.columns:                       # place it right after GLM-Missense_score
    df.insert(df.columns.get_loc('GLM-Missense_score') + 1, 'MetaMissense_score', df.pop('MetaMissense_score'))

os.makedirs(args.outdir, exist_ok=True)
out = os.path.join(args.outdir, 'MetaMissense.tsv')
df.to_csv(out, sep='\t', index=False)
print(f'Saved {out}: {valid.sum():,} scored, {(~valid).sum():,} with a missing input score (NaN)')
