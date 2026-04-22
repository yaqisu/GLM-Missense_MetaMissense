#!/usr/bin/env python3
"""
evaluate.py — Evaluate all methods on merged variant predictions.

Reads merge_config.tsv to identify which score columns to highlight in plots
(models with highlight=yes). Highlighted models always appear in figures in
bold, even if they don't rank in the top N.

Three modes:
  (no --mode)       Full dataset evaluation — same as old evaluate_all.py
  --mode filter     Keep variants passing a condition (AF < threshold, conservation >= threshold)
  --mode stratify   Bin variants and evaluate each bin separately

Usage:
    # ── Full evaluation ──────────────────────────────────────────────────
    python evaluation/evaluate.py \\
        --merged  results/predictions/ClinVar.260309only/merged.tsv \\
        --config  results/predictions/ClinVar.260309only/merge_config.tsv \\
        --outdir  results/predictions/ClinVar.260309only/eval_all

    # P+B subset only
    python evaluation/evaluate.py \\
        --merged  results/predictions/ClinVar.260309only/merged.tsv \\
        --config  results/predictions/ClinVar.260309only/merge_config.tsv \\
        --outdir  results/predictions/ClinVar.260309only/eval_pb \\
        --subset  data/sequences/ClinVar.260309only.missense.hg38.pb_ids.tsv

    # ── Filter mode ──────────────────────────────────────────────────────
    python evaluation/evaluate.py \\
        --merged    results/predictions/ClinVar.260309only/merged.tsv \\
        --config    results/predictions/ClinVar.260309only/merge_config.tsv \\
        --outdir    results/predictions/ClinVar.260309only/eval_rare_1e-3 \\
        --mode      filter \\
        --col       gnomAD4.1_joint_AF \\
        --threshold 1e-3

    # ── Stratify mode ────────────────────────────────────────────────────
    python evaluation/evaluate.py \\
        --merged  results/predictions/ClinVar.260309only/merged.tsv \\
        --config  results/predictions/ClinVar.260309only/merge_config.tsv \\
        --outdir  results/predictions/ClinVar.260309only/eval_strat_af \\
        --mode    stratify \\
        --col     gnomAD4.1_joint_AF \\
        --strata  builtin_af
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from core import (
    evaluate_all_columns, build_metrics_df, build_summary,
    apply_anchor_filter, apply_af_filter, apply_conservation_filter,
    stratify_by_column, parse_custom_strata,
    AF_STRATA_DEFAULT, GERP_STRATA_DEFAULT, PHYLOP_STRATA_DEFAULT,
    LOEUF_STRATA_DEFAULT, SPLICEAI_STRATA_DEFAULT,
    CONSERVATION_COLS,
    ANCHOR_COLS,
    effective_anchor_cols,
    plot_roc_curves, plot_pr_curves, plot_auroc_barplot,
    plot_metrics_heatmap, plot_auroc_scatter, plot_af_distribution,
    plot_comparison_across_strata, plot_score_correlation,
    plot_zeroshot_roc_curves, plot_zeroshot_barplot,
    plot_glm_zeroshot_roc_curves,
)

SKIP_COLS = {"variant_id", "chromosome", "position", "ref_allele", "alt_allele",
             "predicted_label"}

BUILTIN_STRATA = {
    "builtin_af":     AF_STRATA_DEFAULT,
    "builtin_gerp":   GERP_STRATA_DEFAULT,
    "builtin_phylop": PHYLOP_STRATA_DEFAULT,
    "builtin_loeuf":  LOEUF_STRATA_DEFAULT,
    "builtin_spliceai": SPLICEAI_STRATA_DEFAULT,
}


# ============================================================================
# Config loading
# ============================================================================

def load_merge_config(config_path: Path) -> dict:
    """
    Read merge_config.tsv and return a dict with:
        highlighted_cols : list of score column names with highlight=yes
        our_col          : first highlighted col (primary model for evaluate_all_columns)
        all_model_cols   : all {label}_score columns (finetune + zeroshot)
    """
    cfg = pd.read_csv(config_path, sep='\t', comment='#')
    cfg.columns = cfg.columns.str.strip()
    cfg['label']  = cfg['label'].str.strip()
    cfg['source'] = cfg['source'].str.strip()

    if 'highlight' in cfg.columns:
        cfg['highlight'] = cfg['highlight'].str.strip().str.lower().eq('yes')
    else:
        cfg['highlight'] = cfg['source'] == 'finetune'

    model_rows = cfg[cfg['source'] != 'dbnsfp']
    highlighted = model_rows[model_rows['highlight']]['label'].tolist()
    all_models  = model_rows['label'].tolist()

    highlighted_cols = [f'{l}_score' for l in highlighted]
    all_model_cols   = [f'{l}_score' for l in all_models]

    # Primary model = first highlighted; fall back to first model overall
    our_col = highlighted_cols[0] if highlighted_cols else all_model_cols[0]

    return {
        'highlighted_cols': highlighted_cols,
        'all_model_cols':   all_model_cols,
        'our_col':          our_col,
    }


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate all methods on merged variant predictions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument('--merged',    required=True,  type=Path,
                   help='Merged TSV from merge_all.py')
    p.add_argument('--config',    required=True,  type=Path,
                   help='merge_config.tsv — used to identify highlighted models')
    p.add_argument('--outdir',    required=True,  type=Path)
    p.add_argument('--subset',    default=None,   type=Path,
                   help='Optional TSV with variant_id column to restrict evaluation')

    # Anchor / evaluation settings
    p.add_argument('--label_col',   default='true_label')
    p.add_argument('--anchor_cols',
                   default=','.join(ANCHOR_COLS),
                   help='Comma-separated cols required to be non-missing in the '
                        'shared evaluation subset. Defaults to all 9 anchor predictors. '
                        'Missing cols in the actual table are silently skipped.')
    p.add_argument('--top_n',       default=20, type=int,
                   help='Number of top dbNSFP methods to include in plots (default: 20)')

    # Filter / stratify (all optional — omit for full evaluation)
    p.add_argument('--mode',      default=None, choices=['filter', 'stratify'],
                   help='filter: keep variants passing a condition. '
                        'stratify: evaluate each bin separately. '
                        'Omit for full dataset evaluation.')
    p.add_argument('--col',       default=None,
                   help='[filter/stratify] Column to filter or stratify on. '
                        'Comma-separated for multi-AF filter.')
    p.add_argument('--threshold', default=1e-3, type=float,
                   help='[filter] Numeric threshold (default: 1e-3)')
    p.add_argument('--direction', default='below', choices=['below', 'above'],
                   help='[filter] below: col < threshold (AF). above: col >= threshold '
                        '(conservation). Default: below')
    p.add_argument('--include_missing', default=True,
                   action=argparse.BooleanOptionalAction,
                   help='[filter AF] Include variants absent from gnomAD (default: True)')
    p.add_argument('--strata',    default='builtin_af',
                   help='[stratify] builtin_af | builtin_gerp | builtin_phylop | '
                    '   builtin_loeuf | '
                        "custom spec 'lo:hi,lo:hi,...' (default: builtin_af)")
    return p.parse_args()


# ============================================================================
# Core evaluation runner
# ============================================================================

def run_evaluation(shared: pd.DataFrame,
                   labels: pd.Series,
                   skip: set,
                   our_col: str,
                   highlighted_cols: list,
                   anchor_cols: list,
                   out_dir: Path,
                   top_n: int,
                   subtitle: str = '') -> dict:
    """
    Evaluate all score columns in `shared`, generate plots, and save metrics.

    highlighted_cols: these models always appear in plots in bold, even if they
                      don't rank in the top N dbNSFP methods.
    """
    plots_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    if len(shared) < 20 or labels.nunique() < 2:
        print(f'  SKIP: too few variants ({len(shared)}) or only one class.')
        return {}

    our_metrics, other_metrics = evaluate_all_columns(shared, labels, skip, our_col)
    print(f'  {our_col}: AUROC={our_metrics["auroc"]:.4f}  n={our_metrics["n_variants"]:,}')

    metrics_df = build_metrics_df(our_metrics, other_metrics)
    metrics_df.to_csv(out_dir / 'all_metrics.tsv', sep='\t', index=False)

    summary = build_summary(metrics_df, our_col, anchor_cols, top_n)
    summary.to_csv(out_dir / 'summary_comparison.tsv', sep='\t', index=False)
    print(summary.to_string(index=False))

    # Top N non-highlighted methods by AUROC
    non_highlighted = [c for c in metrics_df['column']
                       if c not in highlighted_cols and c != our_col]
    top_cols = (metrics_df[metrics_df['column'].isin(non_highlighted)]
                .dropna(subset=['auroc'])
                .nlargest(top_n, 'auroc')['column'].tolist())

    # Always include highlighted models in plots, in bold
    # plot_cols = highlighted first (bold), then top N others
    extra_highlighted = [c for c in highlighted_cols if c != our_col]
    plot_cols = extra_highlighted + top_cols

    our_auroc = our_metrics['auroc']

    plot_roc_curves(shared, plot_cols, our_col, labels.name,
                    plots_dir / 'roc_pr_curves.png',
                    subtitle=subtitle,
                    bold_cols=highlighted_cols,
                    metrics_df=metrics_df)
    plot_auroc_barplot(metrics_df, our_auroc, anchor_cols,
                       plots_dir / 'auroc_prauc_barplot.png',
                       our_col=our_col, highlight_cols=highlighted_cols,
                       labels=labels)
    plot_metrics_heatmap(metrics_df, plots_dir / 'metrics_heatmap.png',
                         our_col=our_col, highlight_cols=highlighted_cols,
                         labels=labels)
    plot_auroc_scatter(metrics_df, our_auroc, plots_dir / 'auroc_scatter.png',
                       our_col=our_col, highlight_cols=highlighted_cols,
                       labels=labels)
    plot_score_correlation(shared, our_col,
                           plots_dir / 'score_correlation.png',
                           labels=labels)
    # no-ops kept for compat
    plot_zeroshot_roc_curves(shared, our_col, labels.name,
                             plots_dir / 'zeroshot_roc_pr_curves.png',
                             subtitle=subtitle, labels=labels,
                             metrics_df=metrics_df)
    plot_zeroshot_barplot(metrics_df, our_col,
                          plots_dir / 'zeroshot_auroc_prauc_barplot.png',
                          labels=labels)
    # GLM-Missense vs NT2-Zeroshot dedicated comparison (#2)
    plot_glm_zeroshot_roc_curves(shared, our_col, labels.name,
                                 plots_dir / 'glm_vs_zeroshot_roc_pr.png',
                                 subtitle=subtitle,
                                 metrics_df=metrics_df)

    return our_metrics


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # Load config to get highlighted columns
    print(f'Loading config from {args.config}')
    cfg_info = load_merge_config(args.config)
    our_col          = cfg_info['our_col']
    highlighted_cols = cfg_info['highlighted_cols']
    print(f'  Primary model (our_col): {our_col}')
    print(f'  Highlighted cols: {highlighted_cols}')

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ── Load merged TSV ────────────────────────────────────────────────────
    print(f'\nLoading {args.merged}')
    merged = pd.read_csv(args.merged, sep='\t', low_memory=False)
    print(f'  {merged.shape[0]:,} variants × {merged.shape[1]} columns')

    # Optional variant subset — only apply if path looks like a real file
    if args.subset and str(args.subset).strip() not in ("", "0", "None"):
        sub  = pd.read_csv(args.subset, sep='\t', low_memory=False)
        ids  = set(sub['variant_id'].dropna().astype(str))
        n_before = len(merged)
        merged = merged[merged['variant_id'].astype(str).isin(ids)].copy().reset_index(drop=True)
        print(f'  After subset filter: {len(merged):,} / {n_before:,} variants')

    # Parse anchor cols from args, then filter to those actually present in merged.
    # Silently skips any predictor not available in this table (#10).
    requested_anchors = [c.strip() for c in args.anchor_cols.split(',') if c.strip()]
    anchor_cols = [c for c in requested_anchors
                   if c in merged.columns and merged[c].notna().any()]
    if not anchor_cols:
        anchor_cols = effective_anchor_cols(merged)
    print(f'  Effective anchor cols ({len(anchor_cols)}): {anchor_cols}')
    skip        = SKIP_COLS | {args.label_col}

    # ── Anchor filter (shared evaluation subset) ───────────────────────────
    print(f'\n── Anchor filter: {anchor_cols} ────────────────────────────────')
    anchor_mask = apply_anchor_filter(merged, anchor_cols, our_col)
    shared_base = merged[anchor_mask].copy().reset_index(drop=True)
    print(f'  Shared base: {len(shared_base):,} variants')
    print(f'  Label dist:  {shared_base[args.label_col].value_counts().to_dict()}')

    if len(shared_base) < 50:
        print('ERROR: shared subset too small (<50). Check --anchor_cols match column names.')
        for c in merged.columns:
            if 'revel' in c.lower() or 'alpha' in c.lower():
                print(f'  {c}')
        return

    # ══════════════════════════════════════════════════════════════════════
    # FULL EVALUATION (no --mode)
    # ══════════════════════════════════════════════════════════════════════
    if args.mode is None:
        labels = shared_base[args.label_col]
        labels.name = args.label_col

        # Save shared subset summary
        save_cols = (['variant_id', 'chromosome', 'position', 'ref_allele', 'alt_allele',
                      our_col, args.label_col] + anchor_cols)
        save_cols = [c for c in save_cols if c in shared_base.columns]
        shared_base[save_cols].to_csv(
            args.outdir / 'shared_subset_summary.tsv', sep='\t', index=False)

        run_evaluation(shared_base, labels, skip, our_col, highlighted_cols,
                       anchor_cols, args.outdir, args.top_n)

        print(f'\n✓  Done.  Outputs in {args.outdir}')

    # ══════════════════════════════════════════════════════════════════════
    # FILTER MODE
    # ══════════════════════════════════════════════════════════════════════
    elif args.mode == 'filter':
        if args.col is None:
            print('ERROR: --col is required for --mode filter', file=sys.stderr)
            sys.exit(1)

        filter_cols    = [c.strip() for c in args.col.split(',')]
        is_conservation = filter_cols[0] in CONSERVATION_COLS or args.direction == 'above'

        print(f'\n── Filter mode ──────────────────────────────────────────────')
        if is_conservation:
            assert len(filter_cols) == 1, 'Conservation filter supports only one column.'
            col  = filter_cols[0]
            mask = apply_conservation_filter(shared_base, col, args.threshold, args.direction)
            subtitle = f'{col} {args.direction} {args.threshold}  n={mask.sum():,}'
        else:
            print(f'  AF threshold: {args.threshold}  include_missing={args.include_missing}')
            mask     = apply_af_filter(shared_base, filter_cols, args.threshold,
                                       include_missing=args.include_missing)
            subtitle = (f'AF < {args.threshold} ({", ".join(filter_cols)})  '
                        f'n={mask.sum():,}')

        filtered = shared_base[mask].copy().reset_index(drop=True)
        labels   = filtered[args.label_col]
        labels.name = args.label_col

        print(f'  Filtered subset: {len(filtered):,} variants')
        print(f'  Label dist: {labels.value_counts().to_dict()}')

        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / 'plots').mkdir(exist_ok=True)

        if not is_conservation:
            plot_af_distribution(filtered, filter_cols,
                                 args.outdir / 'plots' / 'af_distribution.png')

        save_cols = (['variant_id', 'chromosome', 'position', 'ref_allele', 'alt_allele',
                      our_col, args.label_col] + filter_cols + anchor_cols)
        save_cols = [c for c in save_cols if c in filtered.columns]
        filtered[save_cols].to_csv(
            args.outdir / 'filtered_subset_summary.tsv', sep='\t', index=False)

        run_evaluation(filtered, labels, skip, our_col, highlighted_cols,
                       anchor_cols, args.outdir, args.top_n, subtitle=subtitle)

        print(f'\n✓  Done.  Outputs in {args.outdir}')

    # ══════════════════════════════════════════════════════════════════════
    # STRATIFY MODE
    # ══════════════════════════════════════════════════════════════════════
    elif args.mode == 'stratify':
        if args.col is None:
            print('ERROR: --col is required for --mode stratify', file=sys.stderr)
            sys.exit(1)

        filter_cols = [c.strip() for c in args.col.split(',')]
        assert len(filter_cols) == 1, 'Stratify supports only one column.'
        col = filter_cols[0]

        if args.strata in BUILTIN_STRATA:
            strata = BUILTIN_STRATA[args.strata]
        else:
            strata = parse_custom_strata(args.strata)

        print(f'\n── Stratify mode  —  column: {col} ──────────────────────────')
        print(f'  Strata: {[s[0] for s in strata]}')

        stratum_dfs = stratify_by_column(shared_base, col, strata)
        all_summaries = []

        for stratum_name, sub_df in stratum_dfs.items():
            print(f'\n──── Stratum: "{stratum_name}"  ({len(sub_df):,} variants) ────')
            if len(sub_df) < 20:
                print(f'  SKIP: too few variants.')
                continue

            labels = sub_df[args.label_col]
            labels.name = args.label_col
            stratum_dir = args.outdir / stratum_name.replace(' ', '_').replace('/', '_')

            our_m = run_evaluation(sub_df, labels, skip, our_col, highlighted_cols,
                                   anchor_cols, stratum_dir, args.top_n,
                                   subtitle=f'{col} = "{stratum_name}" | n={len(sub_df):,}')

            if our_m:
                metrics_path = stratum_dir / 'all_metrics.tsv'
                if metrics_path.exists():
                    mdf = pd.read_csv(metrics_path, sep='\t')
                    mdf['stratum'] = stratum_name
                    all_summaries.append(mdf)

        # Cross-stratum comparison
        if all_summaries:
            print('\n── Cross-stratum comparison ────────────────────────────────')
            comparison = pd.concat(all_summaries, ignore_index=True)
            comp_path  = args.outdir / 'stratification_comparison.tsv'
            comparison.to_csv(comp_path, sep='\t', index=False)
            print(f'  Saved to {comp_path}')

            top_methods = (comparison[comparison['column'].isin(
                               [c for c in comparison['column'] if c not in highlighted_cols])]
                           .dropna(subset=['auroc'])
                           .groupby('column')['auroc'].mean()
                           .nlargest(args.top_n).index.tolist())
            # Always include highlighted cols in cross-stratum plot
            focus_cols = highlighted_cols + [c for c in top_methods if c not in highlighted_cols]
            focus = comparison[comparison['column'].isin(focus_cols)]

            for metric in ['auroc', 'prauc']:
                plot_comparison_across_strata(
                    focus, args.outdir / f'stratification_{metric}.png',
                    metric=metric, bold_cols=highlighted_cols)

        print(f'\n✓  Done.  Outputs in {args.outdir}')


if __name__ == '__main__':
    main()