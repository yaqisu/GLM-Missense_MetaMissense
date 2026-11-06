"""
Fig. 2 → horizontal bar chart of best validation AUC, v4.
Colors: navy NT-2, lavender NT-1, gray Caduceus.
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["font.family"]  = "Arial"

OUT_PATH = Path("results") / "figures" / "fig_frozen"

RECORDS = [
    ("NT-2", "12k", "CNN-2L",   "Var pos",   0.871),
    ("NT-2", "12k", "MLP-3L",   "Var pos",   0.869),
    ("NT-2", "12k", "TRANS-2L", "Var pos",   0.869),
    ("NT-2", "12k", "MLP-2L",   "Var pos",   0.868),
    ("NT-2", "6k",  "CNN-2L",   "Var pos",   0.863),
    ("NT-2", "12k", "CNN-2L",   "Mean pool", 0.778),
    ("NT-2", "12k", "TRANS-2L", "Mean pool", 0.775),
    ("NT-2", "12k", "MLP-2L",   "Mean pool", 0.751),
    ("NT-1", "6k",  "CNN-2L",   "Var pos",   0.732),
    ("Cad",  "30k", "CNN-2L",   "Var pos",   0.754),
    ("Cad",  "30k", "TRANS-2L", "Var pos",   0.700),
    ("Cad",  "30k", "MLP-3L",   "Var pos",   0.600),
    ("Cad",  "30k", "MLP-2L",   "Var pos",   0.597),
    ("Cad",  "30k", "CNN-2L",   "Mean pool", 0.663),
    ("Cad",  "30k", "TRANS-2L", "Mean pool", 0.556),
    ("Cad",  "30k", "MLP-2L",   "Mean pool", 0.459),
]

RECORDS_PLOT = list(reversed(RECORDS))

BACKBONE_COLOR = {
    ("NT-2", "12k"): "#0A3F6E",
    ("NT-2", "6k"):  "#5EA4CE",
    ("NT-1", "6k"):  "#9B7FC7",
    ("Cad",  "30k"): "#888580",
}

HATCH = {"Var pos": "", "Mean pool": "////"}

BAR_HEIGHT     = 0.62
TICK_FONTSIZE  = 9.5
LABEL_FONTSIZE = 10.5
TITLE_FONTSIZE = 11.5
FIGSIZE        = (4, 5.4)

SEP = " │ "

def make_label(backbone, length, head, embed):
    return f"{backbone}{SEP}{length}{SEP}{head}{SEP}{embed}"

fig, ax = plt.subplots(figsize=FIGSIZE)

x_min   = 0.40
x_max_d = max(r[4] for r in RECORDS_PLOT)
xoff    = (x_max_d - x_min) * 0.010

for idx, (backbone, length, head, embed, auc) in enumerate(RECORDS_PLOT):
    color = BACKBONE_COLOR[(backbone, length)]
    ax.barh(idx, auc - x_min, left=x_min, height=BAR_HEIGHT,
            color=color, hatch=HATCH[embed],
            edgecolor="white", linewidth=0.4, zorder=2)
    ax.text(auc + xoff, idx, f"{auc:.3f}",
            va="center", ha="left",
            fontsize=TICK_FONTSIZE - 1, color="#222222", zorder=3)

ylabels = [make_label(*r[:4]) for r in RECORDS_PLOT]
ax.set_yticks(range(len(RECORDS_PLOT)))
ax.set_yticklabels(ylabels, fontsize=TICK_FONTSIZE, fontfamily="Arial", ha="right")
for tick, (backbone, length, *_) in zip(ax.get_yticklabels(), RECORDS_PLOT):
    tick.set_color(BACKBONE_COLOR[(backbone, length)])

header_str = f"Backbone{SEP}Input{SEP}Head{SEP}Embedding"
n = len(RECORDS_PLOT)
header_text = ax.text(-0.01, 1.012, header_str, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=TICK_FONTSIZE,
        fontfamily="Arial", color="#444444", fontweight="bold")

ax.axhline(n - 0.5, color="#aaaaaa", linewidth=0.6, zorder=1)
for sep_idx in [3, 7, 8, 11]:
    ax.axhline(sep_idx - 0.5, color="#dddddd", linewidth=0.5,
               linestyle="--", zorder=1)

padding = (x_max_d - x_min) * 0.16
ax.set_xlim(x_min, x_max_d + padding)
ax.set_xlabel("Best Validation AUC", fontsize=LABEL_FONTSIZE)
ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

backbone_patches = [
    mpatches.Patch(color=BACKBONE_COLOR[("NT-2", "12k")], label="NT-2  (12k) — 500M params"),
    mpatches.Patch(color=BACKBONE_COLOR[("NT-2", "6k")],  label="NT-2  (6k)  — 500M params"),
    mpatches.Patch(color=BACKBONE_COLOR[("NT-1", "6k")],  label="NT-1  (6k)  — 500M params"),
    mpatches.Patch(color=BACKBONE_COLOR[("Cad",  "30k")], label="Caduceus (30k) — 7.7M params"),
]
embed_patches = [
    mpatches.Patch(facecolor="#888888", hatch="",     edgecolor="white", label="Variant-position embedding"),
    mpatches.Patch(facecolor="#888888", hatch="////", edgecolor="white", label="Mean-pool embedding"),
]

leg1 = ax.legend(handles=backbone_patches, title="Backbone  (input length)",
                 title_fontsize=TICK_FONTSIZE - 0.5, fontsize=TICK_FONTSIZE - 1,
                 loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True,
                 framealpha=0.92, edgecolor="#cccccc", prop={"family": "Arial"})
ax.add_artist(leg1)
leg2 = ax.legend(handles=embed_patches, title="Embedding strategy",
                 title_fontsize=TICK_FONTSIZE - 0.5, fontsize=TICK_FONTSIZE - 1,
                 loc="upper left", bbox_to_anchor=(1.02, 0.62),
                 frameon=True, framealpha=0.92, edgecolor="#cccccc",
                 prop={"family": "Arial"})

suptitle = fig.suptitle("Best Validation AUC: Frozen-Backbone Comparison",
                        fontsize=TITLE_FONTSIZE, x=0.55, y=1.03)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
for ext in [".pdf", ".png"]:
    fig.savefig(OUT_PATH.with_suffix(ext), bbox_inches="tight",
                bbox_extra_artists=(leg1, leg2, suptitle, header_text), dpi=300)
    print(f"Saved → {OUT_PATH.with_suffix(ext)}")
plt.close(fig)