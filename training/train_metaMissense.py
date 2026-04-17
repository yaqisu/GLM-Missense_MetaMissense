import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print('WARNING: xgboost not installed, skipping XGBoost models.\n')

parser = argparse.ArgumentParser(
    description='Train ensemble meta-classifiers with chromosome-based test holdout + CV')
parser.add_argument('input_csv', help='CSV/TSV with score columns, true_label, and chromosome')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds (default: 5)')
parser.add_argument('--test_frac', type=float, default=0.25,
                    help='Fraction of data held out for final test (default: 0.25)')
parser.add_argument('--outdir', type=str, default='trained_models',
                    help='Directory to save trained models (default: trained_models)')
args = parser.parse_args()

# ── Load and clean ──────────────────────────────────────────────────────────
all_score_cols = ['AlphaMissense_score', 'ESM1b_score', 'REVEL_score',
                  'CADD_phred', 'SIFT_score', 'Polyphen2_HVAR_score',
                  'GLM-missense_score']
required_cols = all_score_cols + ['true_label', 'chromosome']

df = pd.read_csv(args.input_csv, sep=None, engine='python', na_values='.')
for c in required_cols:
    assert c in df.columns, f'Missing column: {c}'

df = df.dropna(subset=all_score_cols + ['true_label']).reset_index(drop=True)
df = df[df['chromosome'].astype(str) != 'Y'].reset_index(drop=True)
print(f'Loaded {len(df)} variants after dropping NaNs and chrY')

# ── Stage 0: Chromosome-based test holdout ──────────────────────────────────
chroms = df['chromosome'].unique()
np.random.seed(args.seed)
np.random.shuffle(chroms)

# Greedily assign chromosomes to dev set until we reach (1 - test_frac)
cumulative = 0
split_idx = 0
for i, chrom in enumerate(chroms):
    cumulative += (df['chromosome'] == chrom).sum()
    if cumulative / len(df) >= (1 - args.test_frac):
        split_idx = i + 1
        break

dev_chroms = set(chroms[:split_idx])
test_chroms = set(chroms[split_idx:])

dev_df = df[df['chromosome'].isin(dev_chroms)].reset_index(drop=True)
test_df = df[df['chromosome'].isin(test_chroms)].reset_index(drop=True)

print(f'\n{"="*60}')
print('Stage 0: Test Holdout')
print(f'{"="*60}')
print(f'  Dev  chromosomes ({len(dev_chroms)}): {sorted(dev_chroms, key=str)}')
print(f'  Test chromosomes ({len(test_chroms)}): {sorted(test_chroms, key=str)}')
print(f'  Dev:  {len(dev_df)} variants | Test: {len(test_df)} variants '
      f'({len(test_df)/len(df)*100:.1f}% test)')
print(f'  Dev  label dist: {dict(dev_df["true_label"].value_counts())}')
print(f'  Test label dist: {dict(test_df["true_label"].value_counts())}')

# ── Stage 0b: Chromosome-based K-fold split within dev set ──────────────────
dev_chroms_list = dev_df['chromosome'].unique()
np.random.seed(args.seed + 1)  # different seed from test split
np.random.shuffle(dev_chroms_list)

fold_counts = [0] * args.n_folds
chrom_to_fold = {}
for chrom in dev_chroms_list:
    n = (dev_df['chromosome'] == chrom).sum()
    smallest_fold = np.argmin(fold_counts)
    chrom_to_fold[chrom] = smallest_fold
    fold_counts[smallest_fold] += n

dev_df = dev_df.copy()
dev_df['fold'] = dev_df['chromosome'].map(chrom_to_fold)

print(f'\n  {args.n_folds}-fold CV split within dev set:')
for fold_i in range(args.n_folds):
    fold_chroms = sorted([c for c, f in chrom_to_fold.items() if f == fold_i], key=str)
    fold_n = (dev_df['fold'] == fold_i).sum()
    fold_pos = ((dev_df['fold'] == fold_i) & (dev_df['true_label'] == 1)).sum()
    print(f'    Fold {fold_i}: {fold_n:>5} variants ({fold_pos:>4} pos) | '
          f'chroms: {fold_chroms}')

# ── Define feature sets ─────────────────────────────────────────────────────
feature_sets = {
    'all_scores': all_score_cols,
    'no_NT2':     [c for c in all_score_cols if c != 'GLM-missense_score'],
}

# ── Define classifiers ──────────────────────────────────────────────────────
classifiers = {
    'LR': {
        'model': LogisticRegression(max_iter=1000, random_state=args.seed),
        'param_grid': {
            'C': [0.01, 0.1, 1, 10, 100],
        },
        'needs_scaling': True,
    },
    'RF': {
        'model': RandomForestClassifier(random_state=args.seed),
        'param_grid': {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7, None],
            'min_samples_leaf': [1, 5, 10],
        },
        'needs_scaling': False,
    },
}

if HAS_XGB:
    classifiers['XGB'] = {
        'model': XGBClassifier(
            eval_metric='logloss', tree_method='hist', device='cpu',
            random_state=args.seed),
        'param_grid': {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
        },
        'needs_scaling': False,
    }

# ── Helper ──────────────────────────────────────────────────────────────────
def mean_ste(values):
    """Return mean and standard error."""
    m = np.mean(values)
    s = np.std(values, ddof=1) / np.sqrt(len(values))
    return m, s

os.makedirs(args.outdir, exist_ok=True)
cv_results = {}
test_results = {}

# ════════════════════════════════════════════════════════════════════════════
# Stage 1: Hyperparameter selection via GridSearchCV on dev folds
# Stage 2: CV evaluation with fixed params
# Stage 3: Retrain final model on full dev set, evaluate on test set
# ════════════════════════════════════════════════════════════════════════════

for feat_name, features in feature_sets.items():
    for clf_name, clf_cfg in classifiers.items():
        key = f'{clf_name}_{feat_name}'

        print(f'\n{"="*60}')
        print(f'{clf_name} | Features: {feat_name} ({len(features)})')
        print(f'{"="*60}')

        # ── Stage 1: Global grid search on dev folds ───────────────────
        print(f'\n  [Stage 1] Hyperparameter selection...')
        X_dev = dev_df[features].values.astype(float)
        y_dev = dev_df['true_label'].values
        cv_splits = []
        for fold_i in range(args.n_folds):
            train_idx = np.where(dev_df['fold'] != fold_i)[0]
            val_idx = np.where(dev_df['fold'] == fold_i)[0]
            cv_splits.append((train_idx, val_idx))

        if clf_cfg['needs_scaling']:
            pipe = Pipeline([('scaler', StandardScaler()),
                             ('clf', clone(clf_cfg['model']))])
            pipe_param_grid = {f'clf__{k}': v
                               for k, v in clf_cfg['param_grid'].items()}
        else:
            pipe = clone(clf_cfg['model'])
            pipe_param_grid = clf_cfg['param_grid']

        gs = GridSearchCV(
            pipe, pipe_param_grid,
            scoring='roc_auc', cv=cv_splits, n_jobs=-1, refit=False)
        gs.fit(X_dev, y_dev)

        if clf_cfg['needs_scaling']:
            best_params = {k.replace('clf__', ''): v
                           for k, v in gs.best_params_.items()}
        else:
            best_params = gs.best_params_
        print(f'  Best params: {best_params}')
        print(f'  GridSearchCV AUROC: {gs.best_score_:.4f}')

        # ── Stage 2: CV evaluation with fixed params ───────────────────
        print(f'\n  [Stage 2] {args.n_folds}-fold CV evaluation...')
        fold_aurocs, fold_auprcs = [], []
        fold_importances_auroc = []
        fold_importances_auprc = []

        for fold_i in range(args.n_folds):
            train_mask = dev_df['fold'] != fold_i
            val_mask = dev_df['fold'] == fold_i

            X_train_raw = dev_df.loc[train_mask, features].values.astype(float)
            y_train = dev_df.loc[train_mask, 'true_label'].values
            X_val_raw = dev_df.loc[val_mask, features].values.astype(float)
            y_val = dev_df.loc[val_mask, 'true_label'].values

            scaler = None
            if clf_cfg['needs_scaling']:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train_raw)
                X_val = scaler.transform(X_val_raw)
            else:
                X_train = X_train_raw
                X_val = X_val_raw

            model = clone(clf_cfg['model']).set_params(**best_params)
            model.fit(X_train, y_train)

            val_probs = model.predict_proba(X_val)[:, 1]
            auroc = roc_auc_score(y_val, val_probs)
            auprc = average_precision_score(y_val, val_probs)
            fold_aurocs.append(auroc)
            fold_auprcs.append(auprc)

            # Feature importances
            if clf_name == 'LR':
                coefs = np.abs(model.coef_[0])
                fold_importances_auroc.append(coefs)
                fold_importances_auprc.append(coefs)
            else:
                perm_auroc = permutation_importance(
                    model, X_val, y_val, scoring='roc_auc',
                    n_repeats=10, random_state=args.seed, n_jobs=-1)
                perm_auprc = permutation_importance(
                    model, X_val, y_val, scoring='average_precision',
                    n_repeats=10, random_state=args.seed, n_jobs=-1)
                fold_importances_auroc.append(perm_auroc.importances_mean)
                fold_importances_auprc.append(perm_auprc.importances_mean)

            print(f'    Fold {fold_i}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}')

        auroc_mean, auroc_ste = mean_ste(fold_aurocs)
        auprc_mean, auprc_ste = mean_ste(fold_auprcs)

        # Aggregate importances
        imp_auroc_mean = np.mean(fold_importances_auroc, axis=0)
        imp_auroc_ste = np.std(fold_importances_auroc, axis=0, ddof=1) / np.sqrt(args.n_folds)
        imp_auprc_mean = np.mean(fold_importances_auprc, axis=0)
        imp_auprc_ste = np.std(fold_importances_auprc, axis=0, ddof=1) / np.sqrt(args.n_folds)

        cv_results[key] = {
            'classifier': clf_name, 'feature_set': feat_name,
            'auroc_mean': auroc_mean, 'auroc_ste': auroc_ste,
            'auprc_mean': auprc_mean, 'auprc_ste': auprc_ste,
            'fold_aurocs': fold_aurocs, 'fold_auprcs': fold_auprcs,
            'best_params': best_params,
        }

        print(f'\n  CV AUROC: {auroc_mean:.4f} ± {auroc_ste:.4f}')
        print(f'  CV AUPRC: {auprc_mean:.4f} ± {auprc_ste:.4f}')

        # Print feature importances
        if clf_name == 'LR':
            print(f'\n  {"Feature":<28} {"| Coef | Mean":>14} {"± STE":>10}')
            print(f'  {"-"*52}')
            for feat, m, s in zip(features, imp_auroc_mean, imp_auroc_ste):
                print(f'  {feat:<28} {m:>14.4f} {s:>10.4f}')
        else:
            print(f'\n  {"Feature":<28} {"AUROC decr":>12} {"± STE":>8}'
                  f'  {"AUPRC decr":>12} {"± STE":>8}')
            print(f'  {"-"*68}')
            for feat, ma, sa, mp, sp in zip(features,
                    imp_auroc_mean, imp_auroc_ste,
                    imp_auprc_mean, imp_auprc_ste):
                print(f'  {feat:<28} {ma:>12.4f} {sa:>8.4f}'
                      f'  {mp:>12.4f} {sp:>8.4f}')

        # Feature importance plots
        imp_configs = [
            (imp_auroc_mean, imp_auroc_ste, 'AUROC', '|Coefficient|' if clf_name == 'LR' else 'Mean AUROC Decrease'),
            (imp_auprc_mean, imp_auprc_ste, 'AUPRC', '|Coefficient|' if clf_name == 'LR' else 'Mean AUPRC Decrease'),
        ]
        for imp_mean, imp_ste, metric_name, ylabel in imp_configs:
            sorted_idx = np.argsort(imp_mean)
            fig, ax = plt.subplots(figsize=(8, max(3, len(features) * 0.5)))
            ax.barh(np.arange(len(features)), imp_mean[sorted_idx],
                    xerr=imp_ste[sorted_idx], color='steelblue',
                    capsize=3, ecolor='gray')
            ax.set_yticks(np.arange(len(features)))
            ax.set_yticklabels([features[i] for i in sorted_idx])
            ax.set_xlabel(ylabel)
            ax.set_title(f'{clf_name} — {feat_name}\n'
                         f'CV AUROC={auroc_mean:.4f}±{auroc_ste:.4f}  '
                         f'AUPRC={auprc_mean:.4f}±{auprc_ste:.4f}')
            plt.tight_layout()
            fig_path = os.path.join(args.outdir,
                                   f'{key}_feature_importance_{metric_name}.pdf')
            fig.savefig(fig_path, bbox_inches='tight')
            plt.close(fig)
            print(f'  Feature importance plot ({metric_name}): {fig_path}')

        # ── Stage 3: Final model on full dev set, evaluate on test ─────
        print(f'\n  [Stage 3] Final model (train on full dev, evaluate on test)...')
        X_dev_raw = dev_df[features].values.astype(float)
        y_dev_all = dev_df['true_label'].values
        X_test_raw = test_df[features].values.astype(float)
        y_test = test_df['true_label'].values

        final_scaler = None
        if clf_cfg['needs_scaling']:
            final_scaler = StandardScaler()
            X_dev_final = final_scaler.fit_transform(X_dev_raw)
            X_test_final = final_scaler.transform(X_test_raw)
        else:
            X_dev_final = X_dev_raw
            X_test_final = X_test_raw

        final_model = clone(clf_cfg['model']).set_params(**best_params)
        final_model.fit(X_dev_final, y_dev_all)

        test_probs = final_model.predict_proba(X_test_final)[:, 1]
        test_auroc = roc_auc_score(y_test, test_probs)
        test_auprc = average_precision_score(y_test, test_probs)

        test_results[key] = {
            'classifier': clf_name, 'feature_set': feat_name,
            'test_auroc': test_auroc, 'test_auprc': test_auprc,
            'best_params': best_params,
        }

        print(f'  Test AUROC: {test_auroc:.4f}')
        print(f'  Test AUPRC: {test_auprc:.4f}')

        # Save final model
        save_path = os.path.join(args.outdir, f'{key}_final.joblib')
        joblib.dump({
            'model': final_model,
            'scaler': final_scaler,
            'features': features,
            'classifier': clf_name,
            'best_params': best_params,
            'dev_chroms': sorted(dev_chroms, key=str),
            'test_chroms': sorted(test_chroms, key=str),
            'chrom_to_fold': chrom_to_fold,
            'cv_metrics': cv_results[key],
            'test_metrics': test_results[key],
        }, save_path)
        print(f'  Final model saved to {save_path}')

        # Save test set predictions for downstream stratified analysis
        pred_path = os.path.join(args.outdir, f'{key}_test_predictions.tsv')
        pred_df = test_df.copy()
        pred_df[f'{key}_prob'] = test_probs
        pred_df.to_csv(pred_path, sep='\t', index=False)
        print(f'  Test predictions saved to {pred_path}')

# ── Individual score evaluation: CV + test ─────────────────────────────────
print(f'\n{"="*60}')
print(f'Individual Score Performance')
print(f'{"="*60}')

indiv_cv_results = {}
indiv_test_results = {}

for score_col in all_score_cols:
    # CV on dev folds
    fold_aurocs, fold_auprcs = [], []
    for fold_i in range(args.n_folds):
        val_mask = dev_df['fold'] == fold_i
        val_scores = dev_df.loc[val_mask, score_col].values.astype(float)
        y_val = dev_df.loc[val_mask, 'true_label'].values
        if score_col in ('SIFT_score', 'ESM1b_score'):
            val_scores = 1 - val_scores
        fold_aurocs.append(roc_auc_score(y_val, val_scores))
        fold_auprcs.append(average_precision_score(y_val, val_scores))

    auroc_mean, auroc_ste = mean_ste(fold_aurocs)
    auprc_mean, auprc_ste = mean_ste(fold_auprcs)
    indiv_cv_results[score_col] = {
        'auroc_mean': auroc_mean, 'auroc_ste': auroc_ste,
        'auprc_mean': auprc_mean, 'auprc_ste': auprc_ste,
    }

    # Test set
    test_scores = test_df[score_col].values.astype(float)
    y_test = test_df['true_label'].values
    if score_col in ('SIFT_score', 'ESM1b_score'):
        test_scores = 1 - test_scores
    test_auroc = roc_auc_score(y_test, test_scores)
    test_auprc = average_precision_score(y_test, test_scores)
    indiv_test_results[score_col] = {
        'test_auroc': test_auroc, 'test_auprc': test_auprc,
    }

print(f'\n  {"Score":<28} {"CV AUROC":>18} {"CV AUPRC":>18}'
      f'    {"Test AUROC":>10} {"Test AUPRC":>10}')
print(f'  {"-"*94}')
for score_col in all_score_cols:
    rc = indiv_cv_results[score_col]
    rt = indiv_test_results[score_col]
    print(f'  {score_col:<28} {rc["auroc_mean"]:.4f} ± {rc["auroc_ste"]:.4f}'
          f'    {rc["auprc_mean"]:.4f} ± {rc["auprc_ste"]:.4f}'
          f'    {rt["test_auroc"]:>10.4f} {rt["test_auprc"]:>10.4f}')

# ── Summary ─────────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print('Summary: All Methods')
print(f'{"="*60}')
print(f'  {"Method":<28} {"CV AUROC":>18} {"CV AUPRC":>18}'
      f'    {"Test AUROC":>10} {"Test AUPRC":>10}')
print(f'  {"-"*94}')

for key in sorted(cv_results.keys()):
    rc = cv_results[key]
    rt = test_results[key]
    print(f'  {key:<28} {rc["auroc_mean"]:.4f} ± {rc["auroc_ste"]:.4f}'
          f'    {rc["auprc_mean"]:.4f} ± {rc["auprc_ste"]:.4f}'
          f'    {rt["test_auroc"]:>10.4f} {rt["test_auprc"]:>10.4f}')

print(f'  {"---":<28} {"---":>18} {"---":>18}'
      f'    {"---":>10} {"---":>10}')

for score_col in all_score_cols:
    rc = indiv_cv_results[score_col]
    rt = indiv_test_results[score_col]
    print(f'  {score_col:<28} {rc["auroc_mean"]:.4f} ± {rc["auroc_ste"]:.4f}'
          f'    {rc["auprc_mean"]:.4f} ± {rc["auprc_ste"]:.4f}'
          f'    {rt["test_auroc"]:>10.4f} {rt["test_auprc"]:>10.4f}')
