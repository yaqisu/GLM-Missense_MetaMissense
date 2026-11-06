#!/usr/bin/env python3
"""
Summarize NT2 LoRA sweep results into three ranked tables:
  1. Ranked by AUROC
  2. Ranked by AUPRC
  3. Ranked by mean rank (average of AUROC rank and AUPRC rank)

Reads summary.json from each exp_* directory.
Also saves a JSON file (experiment_ranking.json) for downstream CV scripts.

Usage:
    python summarize_sweep.py /path/to/sweep_output
    python summarize_sweep.py /path/to/sweep_output --csv  # also save CSVs
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def load_summaries(output_dir):
    output_dir = Path(output_dir)
    rows = []
    for exp_dir in sorted(output_dir.glob("exp_*"), key=lambda p: int(p.name.split("_")[1])):
        summary_path = exp_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            s = json.load(f)

        cfg = s.get("config", {})
        rows.append({
            "exp_id": s.get("exp_id", cfg.get("exp_id")),
            "classifier": cfg.get("classifier_type"),
            "embedding": cfg.get("embedding_strategy"),
            "fine_tuning": cfg.get("fine_tuning"),
            "lora_rank": cfg.get("lora_rank"),
            "learning_rate": cfg.get("learning_rate"),
            "best_val_auroc": s.get("best_val_auroc"),
            "best_val_auprc": s.get("best_val_auprc"),
            "total_steps": s.get("total_steps"),
            "time_hours": round(s.get("total_time_hours", 0), 2),
            "peak_gpu_gb": round(s.get("peak_gpu_memory_gb", 0), 1),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize NT2 sweep results")
    parser.add_argument("output_dir", type=str, help="Sweep output directory containing exp_* folders")
    parser.add_argument("--csv", action="store_true", help="Also save tables as CSVs")
    args = parser.parse_args()

    df = load_summaries(args.output_dir)

    if df.empty:
        print("No summary.json files found.")
        return

    print(f"Found {len(df)} experiments\n")

    # Format learning rate for display
    df["lr"] = df["learning_rate"].apply(lambda x: f"{x:.0e}" if x is not None else "N/A")

    # Compute ranks
    df["auroc_rank"] = df["best_val_auroc"].rank(ascending=False, method="min").astype(int)
    df["auprc_rank"] = df["best_val_auprc"].rank(ascending=False, method="min").astype(int)
    df["mean_rank"] = (df["auroc_rank"] + df["auprc_rank"]) / 2.0

    display_cols_12 = [
        "rank", "exp_id", "classifier", "embedding", "fine_tuning",
        "lora_rank", "lr", "best_val_auroc", "best_val_auprc",
        "total_steps", "time_hours", "peak_gpu_gb",
    ]
    display_cols_3 = [
        "mean_rank", "auroc_rank", "auprc_rank",
        "exp_id", "classifier", "embedding", "fine_tuning",
        "lora_rank", "lr", "best_val_auroc", "best_val_auprc",
        "total_steps", "time_hours", "peak_gpu_gb",
    ]

    # Table 1: ranked by AUROC
    df_auroc = df.sort_values("best_val_auroc", ascending=False).reset_index(drop=True)
    df_auroc["rank"] = range(1, len(df_auroc) + 1)
    print("=" * 130)
    print("TABLE 1: Ranked by Best Validation AUROC")
    print("=" * 130)
    print(df_auroc[display_cols_12].to_string(index=False))
    print()

    # Table 2: ranked by AUPRC
    df_auprc = df.sort_values("best_val_auprc", ascending=False).reset_index(drop=True)
    df_auprc["rank"] = range(1, len(df_auprc) + 1)
    print("=" * 130)
    print("TABLE 2: Ranked by Best Validation AUPRC")
    print("=" * 130)
    print(df_auprc[display_cols_12].to_string(index=False))
    print()

    # Table 3: ranked by mean rank
    df_mean = df.sort_values("mean_rank", ascending=True).reset_index(drop=True)
    print("=" * 150)
    print("TABLE 3: Ranked by Mean Rank (avg of AUROC rank + AUPRC rank)")
    print("=" * 150)
    print(df_mean[display_cols_3].to_string(index=False))

    # Save experiment_ranking.json for downstream CV scripts
    out = Path(args.output_dir)
    ranking_records = []
    for _, row in df_mean.iterrows():
        ranking_records.append({
            "exp_id": int(row["exp_id"]),
            "mean_rank": float(row["mean_rank"]),
            "auroc_rank": int(row["auroc_rank"]),
            "auprc_rank": int(row["auprc_rank"]),
            "classifier": row["classifier"],
            "embedding": row["embedding"],
            "fine_tuning": row["fine_tuning"],
            "lora_rank": int(row["lora_rank"]) if pd.notna(row["lora_rank"]) else None,
            "learning_rate": float(row["learning_rate"]) if pd.notna(row["learning_rate"]) else None,
            "best_val_auroc": float(row["best_val_auroc"]),
            "best_val_auprc": float(row["best_val_auprc"]),
        })
    with open(out / "experiment_ranking.json", "w") as f:
        json.dump(ranking_records, f, indent=2)
    print(f"\nRanking saved to {out / 'experiment_ranking.json'}")

    if args.csv:
        df_auroc[display_cols_12].to_csv(out / "summary_ranked_by_auroc.csv", index=False)
        df_auprc[display_cols_12].to_csv(out / "summary_ranked_by_auprc.csv", index=False)
        df_mean[display_cols_3].to_csv(out / "summary_ranked_by_mean_rank.csv", index=False)
        print(f"CSVs saved to {out}")


if __name__ == "__main__":
    main()
