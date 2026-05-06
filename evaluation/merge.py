#!/usr/bin/env python3
"""
merge.py
------------
Merge all model predictions and dbNSFP annotations into a single wide TSV,
driven by a config file.

The config TSV has four columns (tab-separated):
    label      path      source      highlight

    label     : name used as the score column prefix ({label}_score)
                For source=dbnsfp, this field is ignored — all dbNSFP columns
                are added as-is.
    path      : path to the prediction or annotation TSV file
    source    : one of: finetune | zeroshot | dbnsfp
    highlight : yes | no — whether to highlight in plots (e.g. our fine-tuned models)
                If this column is omitted, all finetune rows are highlighted by default.

The first non-dbnsfp entry is the PRIMARY model — its key columns
(variant_id, chromosome, position, ref_allele, alt_allele, true_label)
form the base of the output. All other models are joined on variant_id.
dbnsfp is joined on all KEY_COLS.

Downstream eval scripts read merge_config.tsv directly (via --config) to
determine which columns to highlight — no separate JSON file needed.

Usage:
    python evaluation/merge.py \\
        --config  results/predictions/ClinVar.260309only/merge_config.tsv \\
        --output  results/predictions/ClinVar.260309only/merged.tsv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

KEY_COLS  = ['variant_id', 'chromosome', 'position', 'ref_allele', 'alt_allele']
LABEL_COL = 'true_label'
VALID_SOURCES = {'finetune', 'zeroshot', 'dbnsfp'}


def parse_args():
    p = argparse.ArgumentParser(
        description='Merge all model predictions and dbNSFP into one wide TSV.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument('--config', required=True, type=Path,
                   help='Config TSV with columns: label, path, source')
    p.add_argument('--output', required=True, type=Path,
                   help='Output merged TSV path')
    p.add_argument('--top_coverage', default=30, type=int,
                   help='Print top-N best-covered dbNSFP columns (default: 30)')
    return p.parse_args()


def load_config(config_path):
    cfg = pd.read_csv(config_path, sep='\t', comment='#')
    cfg.columns = cfg.columns.str.strip()
    required = {'label', 'path', 'source'}
    missing = required - set(cfg.columns)
    if missing:
        print(f"ERROR: config missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    invalid = set(cfg['source']) - VALID_SOURCES
    if invalid:
        print(f"ERROR: unknown source values: {invalid}. Must be one of {VALID_SOURCES}",
              file=sys.stderr)
        sys.exit(1)
    cfg['path']   = cfg['path'].str.strip()
    cfg['label']  = cfg['label'].str.strip()
    cfg['source'] = cfg['source'].str.strip()

    # highlight column is optional — default: all finetune models are highlighted
    if 'highlight' in cfg.columns:
        cfg['highlight'] = cfg['highlight'].str.strip().str.lower().eq('yes')
    else:
        cfg['highlight'] = cfg['source'] == 'finetune'

    return cfg


def main():
    args = parse_args()

    print(f"Loading config from {args.config}")
    cfg = load_config(args.config)
    print(f"  {len(cfg)} entries: "
          f"{(cfg['source']=='finetune').sum()} finetune, "
          f"{(cfg['source']=='zeroshot').sum()} zeroshot, "
          f"{(cfg['source']=='dbnsfp').sum()} dbnsfp")

    # Separate dbnsfp from model entries
    model_rows  = cfg[cfg['source'] != 'dbnsfp'].reset_index(drop=True)
    dbnsfp_rows = cfg[cfg['source'] == 'dbnsfp'].reset_index(drop=True)

    if len(model_rows) == 0:
        print("ERROR: no finetune or zeroshot entries in config", file=sys.stderr)
        sys.exit(1)

    # ── Load primary model (first non-dbnsfp entry) ────────────────────────
    primary = model_rows.iloc[0]
    primary_path = Path(primary['path'])
    primary_label = primary['label']

    print(f"\nLoading primary model '{primary_label}' [{primary['source']}]")
    print(f"  from {primary_path}")
    if not primary_path.exists():
        print(f"ERROR: file not found: {primary_path}", file=sys.stderr)
        sys.exit(1)

    df_primary = pd.read_csv(primary_path, sep='\t')
    print(f"  {len(df_primary):,} variants")

    # Build base: key columns + label + primary score
    base_cols = KEY_COLS.copy()
    if LABEL_COL in df_primary.columns:
        base_cols.append(LABEL_COL)

    merged = df_primary[base_cols].copy()
    # Accept either the new canonical column name or the legacy name used by
    # zero-shot scripts (pathogenicity_score).
    if f'{primary_label}_score' in df_primary.columns:
        merged[f'{primary_label}_score'] = df_primary[f'{primary_label}_score'].values
    elif 'pathogenicity_score' in df_primary.columns:
        merged[f'{primary_label}_score'] = df_primary['pathogenicity_score'].values
    else:
        print(f"  WARNING: neither '{primary_label}_score' nor 'pathogenicity_score' "
              f"found in primary model", file=sys.stderr)

    # ── Join all other model files on variant_id ───────────────────────────
    for _, row in model_rows.iloc[1:].iterrows():
        label = row['label']
        path  = Path(row['path'])
        source = row['source']

        print(f"\nMerging '{label}' [{source}] from {path}")
        if not path.exists():
            print(f"  WARNING: file not found, skipping", file=sys.stderr)
            continue

        # Accept either the canonical column name or the legacy pathogenicity_score
        score_col = f'{label}_score' if f'{label}_score' in pd.read_csv(path, sep='\t', nrows=0).columns \
                    else 'pathogenicity_score'
        df = pd.read_csv(path, sep='\t', usecols=['variant_id', score_col])
        df = df.rename(columns={score_col: f'{label}_score'})

        n_before = len(merged)
        merged = merged.merge(df, on='variant_id', how='left')
        n_matched = merged[f'{label}_score'].notna().sum()
        print(f"  {n_matched:,}/{n_before:,} variants matched")

    # ── Join dbNSFP (on KEY_COLS, adds many columns) ───────────────────────
    for _, row in dbnsfp_rows.iterrows():
        path = Path(row['path'])
        print(f"\nMerging dbNSFP from {path}")
        if not path.exists():
            print(f"  WARNING: file not found, skipping", file=sys.stderr)
            continue

        dbnsfp = pd.read_csv(path, sep='\t', low_memory=False)
        print(f"  {len(dbnsfp):,} variants, {dbnsfp.shape[1]} columns")

        # Missing variant stats
        our_ids    = set(merged['variant_id'])
        dbnsfp_ids = set(dbnsfp['variant_id'])
        in_both    = our_ids & dbnsfp_ids
        only_ours  = our_ids - dbnsfp_ids
        print(f"  In both: {len(in_both):,}  |  Only in ours (no dbNSFP): {len(only_ours):,}")

        # Drop label/annotation cols that clash with what we already have
        dbnsfp_for_merge = dbnsfp.drop(
            columns=['label', 'true_label', '_n_ann'], errors='ignore'
        )

        n_before = len(merged)
        merged = merged.merge(dbnsfp_for_merge, on=KEY_COLS, how='left',
                              suffixes=('', '_dbnsfp'))
        print(f"  Merged shape: {merged.shape[0]:,} rows × {merged.shape[1]} columns")

    # ── Coverage report for dbNSFP columns ────────────────────────────────
    known_cols = set(KEY_COLS) | {LABEL_COL}
    score_cols = [c for c in merged.columns if c.endswith('_score')]
    known_cols.update(score_cols)
    extra_cols = [c for c in merged.columns if c not in known_cols]

    if extra_cols:
        print(f"\n── dbNSFP column coverage (top {args.top_coverage}) ──────────────────")
        cov_rows = []
        for col in extra_cols:
            n_total   = len(merged)
            n_missing = merged[col].isna().sum() + (merged[col] == '.').sum()
            cov_rows.append({
                'column':   col,
                'n_present': n_total - n_missing,
                'coverage': round((n_total - n_missing) / n_total, 4),
            })
        cov_df = (pd.DataFrame(cov_rows)
                  .sort_values('coverage', ascending=False)
                  .reset_index(drop=True))

        cov_path = args.output.parent / 'dbnsfp_column_coverage.tsv'
        cov_df.to_csv(cov_path, sep='\t', index=False)
        print(cov_df.head(args.top_coverage).to_string(index=False))
        print(f"\n  Full coverage table saved to {cov_path}")

    # ── Save ───────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, sep='\t', index=False)

    highlighted_labels = cfg[cfg['highlight'] & (cfg['source'] != 'dbnsfp')]['label'].tolist()
    highlighted_cols   = [f'{l}_score' for l in highlighted_labels]
    all_score_cols     = [c for c in merged.columns if c.endswith('_score')]

    print(f"\n── Summary ───────────────────────────────────────────────────────")
    print(f"  Output:           {args.output}")
    print(f"  Shape:            {merged.shape[0]:,} rows × {merged.shape[1]} columns")
    print(f"  All score cols:   {all_score_cols}")
    print(f"  Highlighted cols: {highlighted_cols}")


if __name__ == '__main__':
    main()