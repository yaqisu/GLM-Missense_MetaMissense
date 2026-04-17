import argparse
import pandas as pd
import numpy as np
import joblib
import sys

parser = argparse.ArgumentParser(
    description='Predict MetaMissense scores using a saved ensemble model')
parser.add_argument('input_tsv', help='TSV with individual method scores')
parser.add_argument('model_joblib', help='Path to saved MetaMissense model (.joblib)')
parser.add_argument('-o', '--output', default=None,
                    help='Output TSV path (default: input with .MetaMissense appended)')
args = parser.parse_args()

# Column mapping: possible input names -> training feature names
# We'll match against the model's actual expected features
input_aliases = {
    'AlphaMissense_score': 'AlphaMissense_score',
    'ESM1b_score': 'ESM1b_score',
    'REVEL_score': 'REVEL_score',
    'CADD_phred': 'CADD_phred',
    'SIFT_score': 'SIFT_score',
    'Polyphen2_HVAR_score': 'Polyphen2_HVAR_score',
    'GLM-Missense_score': 'finetune_NT2_score',
    'GLM-missense_score': 'finetune_NT2_score',
    'finetune_NT2_score': 'finetune_NT2_score',
}

# Load model bundle
bundle = joblib.load(args.model_joblib)
model = bundle['model']
scaler = bundle['scaler']
train_features = bundle['features']
print(f'Loaded model: {bundle["classifier"]}')
print(f'Expected features: {train_features}')
print(f'Best params: {bundle["best_params"]}')

# Load input data
df = pd.read_csv(args.input_tsv, sep='\t', na_values='.')
print(f'\nLoaded {len(df)} variants from {args.input_tsv}')

# Build mapping: training feature name -> actual input column name
# Uses the alias dict, plus case-insensitive fallback
input_cols_lower = {c.lower(): c for c in df.columns}
reverse_aliases = {}
for input_name, train_name in input_aliases.items():
    reverse_aliases.setdefault(train_name, []).append(input_name)

feature_to_col = {}
missing = []
for feat in train_features:
    # Try exact match in input
    if feat in df.columns:
        feature_to_col[feat] = feat
        continue
    # Try known aliases
    found = False
    for alias in reverse_aliases.get(feat, []):
        if alias in df.columns:
            feature_to_col[feat] = alias
            found = True
            break
        if alias.lower() in input_cols_lower:
            feature_to_col[feat] = input_cols_lower[alias.lower()]
            found = True
            break
    if found:
        continue
    # Case-insensitive fallback on feature name
    if feat.lower() in input_cols_lower:
        feature_to_col[feat] = input_cols_lower[feat.lower()]
        continue
    missing.append(feat)

if missing:
    sys.exit(f'ERROR: Cannot find input columns for features: {missing}')

print(f'\nColumn mapping:')
for feat, col in feature_to_col.items():
    tag = '' if feat == col else f'  <- "{col}"'
    print(f'  {feat}{tag}')

# Extract and rename to training feature names
X = df[[feature_to_col[f] for f in train_features]].copy()
X.columns = train_features

# Track which rows have complete data
valid_mask = X.notna().all(axis=1)
n_missing = (~valid_mask).sum()
if n_missing > 0:
    print(f'WARNING: {n_missing} variants have missing scores, '
          f'MetaMissense_score will be NaN for these')

# Predict
scores = np.full(len(df), np.nan)
if valid_mask.any():
    X_valid = X.loc[valid_mask].values.astype(float)
    if scaler is not None:
        X_valid = scaler.transform(X_valid)
    scores[valid_mask] = model.predict_proba(X_valid)[:, 1]

# Add predictions to dataframe
df['MetaMissense_score'] = scores

# Output
if args.output is None:
    base, ext = args.input_tsv.rsplit('.', 1)
    args.output = f'{base}.MetaMissense.{ext}'

df.to_csv(args.output, sep='\t', index=False)
print(f'\nSaved {len(df)} variants to {args.output}')
print(f'  {valid_mask.sum()} scored, {n_missing} missing')
