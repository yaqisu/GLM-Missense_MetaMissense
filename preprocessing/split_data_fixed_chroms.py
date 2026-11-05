#!/usr/bin/env python3
"""
Split a combined labeled sequence TSV into training and held-out sets by chromosome.
The input TSV must already contain a 'label' column (extract_variant_sequences.py --label).

Two fixed held-out chromosome sets are used (Methods; Table 2):
  architecture   chr 7, 8, 11, 15, 20   earlier release, strict mapping (BvsP): architecture exploration
  glm_missense   chr 2, 3, 16, 17       earlier release, expanded mapping (BLBvsPLP): GLM-Missense / ESM1b-FT
                                        training and validation; later release: ensemble training and test set
Every other chromosome (1-22, X, Y, MT) goes to the training set.
"""

import argparse

import pandas as pd

ALL_CHROMS = {str(c) for c in range(1, 23)} | {'X', 'Y', 'MT'}
HELDOUT = {
    'architecture': {'7', '8', '11', '15', '20'},
    'glm_missense': {'2', '3', '16', '17'},
}
# held-out set of the GLM-Missense split, used by subtract_new_variants.py for its reports
VAL_CHROMS = HELDOUT['glm_missense']
TRAIN_CHROMS = ALL_CHROMS - VAL_CHROMS


def split_and_save(df, heldout, train_out, val_out):
    df['chromosome'] = df['chromosome'].astype(str).str.replace('^chr', '', regex=True)
    train_chroms = ALL_CHROMS - heldout

    unknown = set(df['chromosome'].unique()) - ALL_CHROMS
    if unknown:
        print(f"  WARNING: Unknown chromosomes will be excluded: {sorted(unknown)}")

    train_df = df[df['chromosome'].isin(train_chroms)]
    val_df = df[df['chromosome'].isin(heldout)]

    print(f"  Held-out chromosomes: {sorted(heldout, key=lambda c: (len(c), c))}")
    print(f"  Total:  {len(df):>6} variants")
    print(f"  Train:  {len(train_df):>6} ({len(train_df)/len(df)*100:.1f}%)"
          f" | label counts: {dict(train_df['label'].value_counts().sort_index())}")
    print(f"  Val:    {len(val_df):>6} ({len(val_df)/len(df)*100:.1f}%)"
          f" | label counts: {dict(val_df['label'].value_counts().sort_index())}")

    train_df.to_csv(train_out, sep='\t', index=False)
    val_df.to_csv(val_out, sep='\t', index=False)
    print(f"  Saved:  {train_out}")
    print(f"  Saved:  {val_out}")


def main():
    parser = argparse.ArgumentParser(
        description='Split a labeled sequence TSV into training / held-out sets by chromosome.',
        epilog='Held-out sets: ' + '; '.join(f"{k} = chr {', '.join(sorted(v, key=int))}"
                                             for k, v in HELDOUT.items()))
    parser.add_argument('--input', '-i', required=True, help='Combined labeled sequence TSV')
    parser.add_argument('--train', '-t', required=True, help='Output training TSV')
    parser.add_argument('--val', '-v', required=True, help='Output held-out (validation / test) TSV')
    parser.add_argument('--heldout', '-H', required=True, choices=list(HELDOUT),
                        help='Held-out chromosome set (see Table 2 of the manuscript)')
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep='\t')
    if 'label' not in df.columns:
        raise ValueError("Input TSV must contain a 'label' column. "
                         "Generate sequences with --label flag first.")

    split_and_save(df, HELDOUT[args.heldout], args.train, args.val)


if __name__ == '__main__':
    main()
