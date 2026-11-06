"""
LoRA fine-tuning: side-by-side AUROC and AUPRC bars from a CSV results table.

Expected CSV columns:
    mean_rank,classifier,embedding,lora_rank,lr,best_val_auroc,best_val_auprc

The CSV may contain additional columns, such as exp_id, auroc_rank, auprc_rank,
fine_tuning, total_steps, time_hours, and peak_gpu_gb. They are ignored unless
you use --fine-tuning to filter rows.

Examples:
    python plot_lora_bar_updated.py results/lora_ranked.csv
    python plot_lora_bar_updated.py results/lora_ranked.csv --top-n 20 --out-path results/figures/fig_lora_top20

Clean layout notes:
    - CSV input replaces the old hardcoded DATA list.
    - full_mean_pool / full-mean-pool / Full Mean Pool are all shown as Mean-pool.
    - Row metadata is drawn in a separate left-side label panel, away from the bars.
    - Numeric values are labeled immediately to the right of every bar by default.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib import font_manager

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["hatch.linewidth"] = 0.8

FONT_FAMILY = (
    "Arial"
    if any(font.name == "Arial" for font in font_manager.fontManager.ttflist)
    else "DejaVu Sans"
)
matplotlib.rcParams["font.family"] = FONT_FAMILY

DEFAULT_OUT_PATH = Path("results") / "figures" / "fig_lora"

REQUIRED_COLUMNS = {
    "mean_rank",
    "classifier",
    "embedding",
    "lora_rank",
    "lr",
    "best_val_auroc",
    "best_val_auprc",
}

CLF_COLOR = {
    "mlp": "#0A3F6E",
    "transformer": "#5EA4CE",
    "cnn": "#A8CDE8",
}

EMBEDDING_HATCH = {
    "variant-position": "",
    "mean-pool": "////",
}
FALLBACK_HATCHES = ["", "////", "....", "\\\\\\\\", "xxxx", "----", "++++"]
FALLBACK_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]

# Layout defaults for dense 40-60 row figures.
BAR_HEIGHT = 0.72
TICK_FONTSIZE = 7.5
LABEL_FONTSIZE = 9.5
TITLE_FONTSIZE = 10.5
MIN_FIG_HEIGHT = 3.2
DEFAULT_ROW_HEIGHT = 0.18
DEFAULT_FIG_WIDTH = 10.0
DEFAULT_X_STEP = 0.05


@dataclass(frozen=True)
class Row:
    mean_rank: float
    classifier: str
    embedding: str
    lora_rank: int
    lr: str
    auroc: float
    auprc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot side-by-side AUROC and AUPRC bars from a ranked LoRA CSV table."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help=(
            "CSV file with columns such as mean_rank, classifier, embedding, "
            "lora_rank, lr, best_val_auroc, best_val_auprc."
        ),
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"Output path stem or filename. Default: {DEFAULT_OUT_PATH}",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="One or more output formats. Default: pdf png",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Plot only the top N rows after sorting.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["mean_rank", "best_val_auroc", "best_val_auprc", "input"],
        default="mean_rank",
        help="How to sort rows before plotting. Default: mean_rank.",
    )
    parser.add_argument(
        "--fine-tuning",
        default=None,
        help="Optional value for filtering the fine_tuning column, e.g. --fine-tuning lora.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title. By default no title is drawn, keeping dense plots cleaner.",
    )
    parser.add_argument(
        "--x-min-auroc",
        type=float,
        default=0.60,
        help="Preferred AUROC x-axis minimum. The script lowers this automatically if needed.",
    )
    parser.add_argument(
        "--x-min-auprc",
        type=float,
        default=0.55,
        help="Preferred AUPRC x-axis minimum. The script lowers this automatically if needed.",
    )
    parser.add_argument(
        "--x-step",
        type=float,
        default=DEFAULT_X_STEP,
        help=f"Major tick step for both x-axes. Default: {DEFAULT_X_STEP}",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=DEFAULT_FIG_WIDTH,
        help=f"Figure width in inches before tight bounding box. Default: {DEFAULT_FIG_WIDTH}",
    )
    parser.add_argument(
        "--row-height",
        type=float,
        default=DEFAULT_ROW_HEIGHT,
        help=f"Figure height per row in inches. Default: {DEFAULT_ROW_HEIGHT}",
    )
    parser.add_argument(
        "--value-labels",
        choices=["on", "off"],
        default="on",
        help="Show numeric values to the right of every bar. Default: on.",
    )
    parser.add_argument(
        "--value-decimals",
        type=int,
        default=3,
        help="Number of decimals for value labels. Default: 3.",
    )
    parser.add_argument(
        "--legend",
        choices=["right", "bottom", "none"],
        default="right",
        help="Legend placement. Default: right, in its own reserved panel.",
    )
    return parser.parse_args()


def require_columns(fieldnames: Iterable[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("CSV appears to be empty or has no header row.")

    missing = REQUIRED_COLUMNS.difference(fieldnames)
    if missing:
        raise ValueError(
            "CSV is missing required column(s): " + ", ".join(sorted(missing))
        )


def normalize_classifier(value: str) -> str:
    return value.strip().lower()


def normalize_embedding(value: str) -> str:
    raw = value.strip().lower().replace("_", "-").replace(" ", "-")

    variant_aliases = {
        "variant-position",
        "variant-pos",
        "var-position",
        "var-pos",
        "full-variant-position",
        "full-var-position",
        "full-var-pos",
        "variant-position-embedding",
        "full-variant-position-embedding",
    }
    mean_pool_aliases = {
        "mean-pool",
        "meanpool",
        "mean-pooling",
        "mean-pool-embedding",
        "full-mean-pool",
        "full-meanpool",
        "full-mean-pooling",
        "full-mean-pool-embedding",
    }

    if raw in variant_aliases:
        return "variant-position"
    if raw in mean_pool_aliases:
        return "mean-pool"
    return raw


def embedding_label(value: str) -> str:
    labels = {
        "variant-position": "Variant-position",
        "mean-pool": "Mean-pool",
    }
    return labels.get(value, value.replace("-", " ").title())


def classifier_label(value: str) -> str:
    if value == "mlp":
        return "MLP"
    return value.replace("_", " ").replace("-", " ").title()


def format_lr(value: str) -> str:
    text = value.strip()
    try:
        formatted = f"{float(text):.0e}"
        return formatted.replace("e-0", "e-").replace("e+0", "e+")
    except ValueError:
        return text


def parse_int_like(value: str) -> int:
    return int(float(value.strip()))


def parse_float(value: str, column: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Could not parse {column}={value!r} as a float.") from exc


def load_rows(csv_path: Path, fine_tuning: str | None = None) -> list[Row]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        require_columns(reader.fieldnames)

        rows: list[Row] = []
        for line_no, raw in enumerate(reader, start=2):
            if fine_tuning is not None:
                if "fine_tuning" not in raw:
                    raise ValueError("--fine-tuning was provided, but CSV has no fine_tuning column.")
                if raw["fine_tuning"].strip().lower() != fine_tuning.strip().lower():
                    continue

            try:
                rows.append(
                    Row(
                        mean_rank=parse_float(raw["mean_rank"], "mean_rank"),
                        classifier=normalize_classifier(raw["classifier"]),
                        embedding=normalize_embedding(raw["embedding"]),
                        lora_rank=parse_int_like(raw["lora_rank"]),
                        lr=format_lr(raw["lr"]),
                        auroc=parse_float(raw["best_val_auroc"], "best_val_auroc"),
                        auprc=parse_float(raw["best_val_auprc"], "best_val_auprc"),
                    )
                )
            except Exception as exc:
                raise ValueError(f"Error while parsing CSV line {line_no}: {exc}") from exc

    if not rows:
        raise ValueError("No rows found after applying filters.")

    return rows


def sort_rows(rows: list[Row], sort_by: str) -> list[Row]:
    rows = list(rows)
    if sort_by == "input":
        return rows
    if sort_by == "mean_rank":
        return sorted(rows, key=lambda r: r.mean_rank)
    if sort_by == "best_val_auroc":
        return sorted(rows, key=lambda r: r.auroc, reverse=True)
    if sort_by == "best_val_auprc":
        return sorted(rows, key=lambda r: r.auprc, reverse=True)
    raise ValueError(f"Unsupported sort key: {sort_by}")


def first_seen(values: Iterable[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def build_classifier_colors(rows: list[Row]) -> dict[str, str]:
    colors = dict(CLF_COLOR)
    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", FALLBACK_COLORS)
    if not fallback_colors:
        fallback_colors = FALLBACK_COLORS
    fallback_idx = 0

    for clf in first_seen(row.classifier for row in rows):
        if clf in colors:
            continue
        colors[clf] = fallback_colors[fallback_idx % len(fallback_colors)]
        fallback_idx += 1

    return colors


def build_embedding_hatches(rows: list[Row]) -> dict[str, str]:
    hatches = dict(EMBEDDING_HATCH)
    fallback_idx = 0

    for emb in first_seen(row.embedding for row in rows):
        if emb in hatches:
            continue
        hatches[emb] = FALLBACK_HATCHES[fallback_idx % len(FALLBACK_HATCHES)]
        fallback_idx += 1

    return hatches


def choose_x_min(values: list[float], preferred_min: float, step: float) -> float:
    observed_min = min(values)
    if observed_min >= preferred_min:
        return preferred_min

    return math.floor(observed_min / step) * step


def embedding_runs(rows_in_plot_order: list[Row]) -> list[tuple[int, int, str]]:
    runs: list[tuple[int, int, str]] = []
    start = 0

    for i in range(1, len(rows_in_plot_order) + 1):
        if i == len(rows_in_plot_order) or rows_in_plot_order[i].embedding != rows_in_plot_order[start].embedding:
            runs.append((start, i - 1, rows_in_plot_order[start].embedding))
            start = i

    return runs


def save_figure(fig, out_path: Path, formats: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ext in formats:
        suffix = ext if ext.startswith(".") else f".{ext}"
        path = out_path.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved -> {path}")


def draw_label_panel(ax, rows_plot: list[Row], clf_color: dict[str, str]) -> None:
    n = len(rows_plot)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Column positions in axis-fraction coordinates.
    x_clf, x_emb, x_rank, x_lr = 0.01, 0.34, 0.70, 0.86
    header_y = 1.006
    header_kwargs = dict(
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=TICK_FONTSIZE,
        color="#444444",
        fontweight="bold",
        clip_on=False,
    )
    ax.text(x_clf, header_y, "Clf", **header_kwargs)
    ax.text(x_emb, header_y, "Embedding", **header_kwargs)
    ax.text(x_rank, header_y, "LoRA r", **header_kwargs)
    ax.text(x_lr, header_y, "LR", **header_kwargs)

    # Light vertical separators between the text columns.
    for x in (0.30, 0.66, 0.82):
        ax.axvline(x, color="#d0d0d0", linewidth=0.6, zorder=0)

    blended = ax.get_yaxis_transform()
    for idx, row in enumerate(rows_plot):
        color = clf_color[row.classifier]
        row_kwargs = dict(
            transform=blended,
            va="center",
            ha="left",
            fontsize=TICK_FONTSIZE,
            color=color,
            clip_on=False,
        )
        ax.text(x_clf, idx, classifier_label(row.classifier), **row_kwargs)
        ax.text(x_emb, idx, embedding_label(row.embedding), **row_kwargs)
        ax.text(x_rank, idx, f"r={row.lora_rank}", **row_kwargs)
        ax.text(x_lr, idx, row.lr, **row_kwargs)


def draw_metric_axis(
    ax,
    rows_plot: list[Row],
    metric: str,
    preferred_x_min: float,
    xlabel: str,
    x_step: float,
    clf_color: dict[str, str],
    embedding_hatch: dict[str, str],
    runs: list[tuple[int, int, str]],
    show_value_labels: bool,
    value_decimals: int,
) -> None:
    n = len(rows_plot)
    values = [getattr(row, metric) for row in rows_plot]
    x_min = choose_x_min(values, preferred_x_min, x_step)
    x_max_d = max(values)
    x_range = max(x_max_d - x_min, 1e-6)
    label_pad = max(0.020, x_range * 0.12) if show_value_labels else max(0.008, x_range * 0.05)
    xoff = max(0.003, x_range * 0.010)

    for idx, row in enumerate(rows_plot):
        val = getattr(row, metric)
        ax.barh(
            idx,
            val - x_min,
            left=x_min,
            height=BAR_HEIGHT,
            color=clf_color[row.classifier],
            hatch=embedding_hatch[row.embedding],
            edgecolor="white",
            linewidth=0.3,
            zorder=2,
        )
        if show_value_labels:
            ax.text(
                val + xoff,
                idx,
                f"{val:.{value_decimals}f}",
                va="center",
                ha="left",
                fontsize=max(5.5, TICK_FONTSIZE - 1.5),
                color="#222222",
                zorder=3,
                clip_on=False,
            )

    ax.set_xlim(x_min, x_max_d + label_pad)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_yticks(range(n))
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_step / 2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Light separator lines between consecutive embedding blocks, but no text over bars.
    for _, end, _ in runs[:-1]:
        ax.axhline(end + 0.5, color="#888888", linewidth=0.9, linestyle="-", zorder=1)
    ax.axhline(n - 0.5, color="#aaaaaa", linewidth=0.6, zorder=1)


def draw_legends(
    fig,
    legend_ax,
    rows: list[Row],
    clf_color: dict[str, str],
    embedding_hatch: dict[str, str],
    placement: str,
) -> None:
    if placement == "none":
        legend_ax.set_axis_off()
        return

    clf_patches = [
        mpatches.Patch(color=clf_color[clf], label=classifier_label(clf))
        for clf in first_seen(row.classifier for row in rows)
    ]
    emb_patches = [
        mpatches.Patch(
            facecolor="#000000",
            hatch=embedding_hatch[emb],
            edgecolor="white",
            label=embedding_label(emb),
        )
        for emb in first_seen(row.embedding for row in rows)
    ]

    if placement == "right":
        legend_ax.set_axis_off()
        leg1 = legend_ax.legend(
            handles=clf_patches,
            title="Classifier head",
            title_fontsize=TICK_FONTSIZE,
            fontsize=TICK_FONTSIZE - 0.5,
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            edgecolor="#cccccc",
            prop={"family": FONT_FAMILY},
        )
        legend_ax.add_artist(leg1)
        legend_ax.legend(
            handles=emb_patches,
            title="Embedding",
            title_fontsize=TICK_FONTSIZE,
            fontsize=TICK_FONTSIZE - 0.5,
            loc="upper left",
            bbox_to_anchor=(0, 0.68),
            frameon=True,
            framealpha=0.92,
            edgecolor="#cccccc",
            prop={"family": FONT_FAMILY},
        )
        return

    # Bottom placement uses figure-level legends, not the reserved legend axis.
    legend_ax.set_axis_off()
    fig.legend(
        handles=emb_patches,
        title="Embedding",
        title_fontsize=TICK_FONTSIZE,
        fontsize=TICK_FONTSIZE - 0.5,
        loc="lower center",
        bbox_to_anchor=(0.42, -0.02),
        ncol=max(1, len(emb_patches)),
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        prop={"family": FONT_FAMILY},
    )
    fig.legend(
        handles=clf_patches,
        title="Classifier head",
        title_fontsize=TICK_FONTSIZE,
        fontsize=TICK_FONTSIZE - 0.5,
        loc="lower center",
        bbox_to_anchor=(0.72, -0.02),
        ncol=max(1, len(clf_patches)),
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        prop={"family": FONT_FAMILY},
    )


def plot(rows: list[Row], args: argparse.Namespace) -> None:
    n = len(rows)
    rows_plot = list(reversed(rows))
    clf_color = build_classifier_colors(rows)
    embedding_hatch = build_embedding_hatches(rows)
    runs = embedding_runs(rows_plot)

    fig_height = max(MIN_FIG_HEIGHT, args.row_height * n)
    legend_ratio = 0.72 if args.legend == "right" else 0.05
    fig = plt.figure(figsize=(args.fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[2.45, 1.0, 1.0, legend_ratio],
        wspace=0.05,
    )
    ax_labels = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax1)
    ax_legend = fig.add_subplot(gs[0, 3])

    draw_label_panel(ax_labels, rows_plot, clf_color)

    draw_metric_axis(
        ax1,
        rows_plot,
        metric="auroc",
        preferred_x_min=args.x_min_auroc,
        xlabel="Validation AUROC",
        x_step=args.x_step,
        clf_color=clf_color,
        embedding_hatch=embedding_hatch,
        runs=runs,
        show_value_labels=args.value_labels == "on",
        value_decimals=args.value_decimals,
    )
    draw_metric_axis(
        ax2,
        rows_plot,
        metric="auprc",
        preferred_x_min=args.x_min_auprc,
        xlabel="Validation AUPRC",
        x_step=args.x_step,
        clf_color=clf_color,
        embedding_hatch=embedding_hatch,
        runs=runs,
        show_value_labels=args.value_labels == "on",
        value_decimals=args.value_decimals,
    )

    draw_legends(fig, ax_legend, rows, clf_color, embedding_hatch, args.legend)

    if args.title:
        fig.suptitle(args.title, fontsize=TITLE_FONTSIZE)

    save_figure(fig, args.out_path, args.formats)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    rows = load_rows(args.csv_path, fine_tuning=args.fine_tuning)
    rows = sort_rows(rows, args.sort_by)
    if args.top_n is not None:
        rows = rows[: args.top_n]
    if not rows:
        raise ValueError("No rows to plot.")

    plot(rows, args)


if __name__ == "__main__":
    main()
