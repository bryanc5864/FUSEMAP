"""
Figure 1: FUSEMAP Training Corpus — Dataset overview infographic.
Stylized card-row layout with proportional bars, assay badges, and kingdom grouping.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": "#222222",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
})

# ─── Dataset specs (verified from actual data) ───────────────────────────────
datasets = [
    # (label, species_italic, cell_type, color, assay, assay_color, n_seqs, seq_len, n_out, out_labels, kingdom)
    ("K562",        "Homo sapiens",              "K562",   "#C0392B", "lentiMPRA",  "#0077B6", 196_664,   230, 1, ["activity"],     "Mammalia"),
    ("HepG2",       "Homo sapiens",              "HepG2",  "#7B241C", "lentiMPRA",  "#0077B6", 122_926,   230, 1, ["activity"],     "Mammalia"),
    ("WTC11",       "Homo sapiens",              "WTC11",  "#1ABC9C", "lentiMPRA",  "#0077B6",  46_128,   230, 1, ["activity"],     "Mammalia"),
    ("S2",          "Drosophila melanogaster",   "S2",     "#E67E22", "STARR-seq",  "#E67E22", 484_052,   249, 2, ["Dev", "Hk"],    "Arthropoda"),
    ("Arabidopsis", "Arabidopsis thaliana",      "Leaf",   "#27AE60", "STARR-seq",  "#27AE60",  13_462,   170, 2, ["leaf", "proto"],"Plantae"),
    ("Maize",       "Zea mays",                  "Leaf",   "#1E8449", "STARR-seq",  "#27AE60",  24_604,   170, 2, ["leaf", "proto"],"Plantae"),
    ("Sorghum",     "Sorghum bicolor",           "Leaf",   "#6C8C3C", "STARR-seq",  "#27AE60",  19_673,   170, 2, ["leaf", "proto"],"Plantae"),
    ("Yeast",       "Saccharomyces cerevisiae",  "",       "#D4AC0D", "DREAM",      "#D4AC0D", 6_810_361, 110, 1, ["expression"],   "Fungi"),
]

total_seqs = sum(d[6] for d in datasets)
max_seqs = max(d[6] for d in datasets)

# ─── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

n_rows = len(datasets)
row_h = 0.072
top_start = 0.85
gap = 0.008

# Column positions (left edges, as fraction of figure width)
col_species_x = 0.02
col_assay_x = 0.30
col_bar_x = 0.42
col_bar_end = 0.82
col_len_x = 0.84
col_out_x = 0.92

# ─── Title ────────────────────────────────────────────────────────────────────
ax.text(0.5, 0.96, "FUSEMAP Training Corpus", fontsize=20, fontweight="bold",
        ha="center", va="center", color="#1B3A5C")
ax.text(0.5, 0.925, f"{total_seqs:,} CRE–Activity Pairs Across 4 Kingdoms",
        fontsize=13, ha="center", va="center", color="#555555")

# ─── Column headers ──────────────────────────────────────────────────────────
header_y = top_start + 0.025
ax.text(col_species_x, header_y, "Species / Cell Type", fontsize=9,
        fontweight="bold", color="#666666", va="center")
ax.text(col_assay_x, header_y, "Assay", fontsize=9,
        fontweight="bold", color="#666666", va="center")
ax.text((col_bar_x + col_bar_end) / 2, header_y, "Sequences", fontsize=9,
        fontweight="bold", color="#666666", va="center", ha="center")
ax.text(col_len_x, header_y, "Length", fontsize=9,
        fontweight="bold", color="#666666", va="center")
ax.text(col_out_x, header_y, "Outputs", fontsize=9,
        fontweight="bold", color="#666666", va="center")

# ─── Draw rows ────────────────────────────────────────────────────────────────
for i, (label, species, cell, color, assay, assay_col, n_seq, seq_len, n_out, out_labels, kingdom) in enumerate(datasets):
    y = top_start - i * (row_h + gap)

    # Alternating row background
    if i % 2 == 0:
        bg = FancyBboxPatch((0.01, y - row_h / 2 - 0.002), 0.97, row_h + 0.004,
                            boxstyle="round,pad=0.003", facecolor="#F7F9FB",
                            edgecolor="none", zorder=0)
        ax.add_patch(bg)

    # Col 1: Species dot + name
    ax.plot(col_species_x + 0.008, y, "o", color=color, markersize=8, zorder=3)
    if cell:
        species_text = f"$\\it{{{species.replace(' ', '~')}}}$ — {cell}"
    else:
        species_text = f"$\\it{{{species.replace(' ', '~')}}}$"
    ax.text(col_species_x + 0.025, y, species_text, fontsize=9.5,
            va="center", color="#222222")

    # Col 2: Assay badge
    badge_w = 0.08
    badge_h = 0.025
    badge = FancyBboxPatch((col_assay_x, y - badge_h / 2), badge_w, badge_h,
                           boxstyle="round,pad=0.004", facecolor=assay_col,
                           edgecolor="none", alpha=0.15, zorder=1)
    ax.add_patch(badge)
    ax.text(col_assay_x + badge_w / 2, y, assay, fontsize=8, fontweight="bold",
            ha="center", va="center", color=assay_col)

    # Col 3: Proportional bar
    bar_max_w = col_bar_end - col_bar_x
    # Use log scale for bar width to handle the yeast outlier, but cap minimum
    log_max = np.log10(max_seqs)
    log_val = np.log10(max(n_seq, 1))
    bar_w = (log_val / log_max) * bar_max_w * 0.95
    bar_rect = FancyBboxPatch((col_bar_x, y - 0.013), bar_w, 0.026,
                               boxstyle="round,pad=0.003", facecolor=color,
                               edgecolor="none", alpha=0.75, zorder=2)
    ax.add_patch(bar_rect)
    # Count label
    count_str = f"{n_seq:,}"
    ax.text(col_bar_x + bar_w + 0.008, y, count_str, fontsize=8.5,
            fontweight="bold", va="center", color="#333333")

    # Col 4: Sequence length
    ax.text(col_len_x, y, f"{seq_len} bp", fontsize=9, va="center",
            color="#444444", fontweight="medium")

    # Col 5: Output dimensionality
    if n_out == 1:
        ax.text(col_out_x, y, "1", fontsize=10, va="center", ha="center",
                color="#444444", fontweight="bold")
    else:
        out_str = f"{n_out}  " + ", ".join(out_labels)
        ax.text(col_out_x, y - 0.001, out_str, fontsize=8.5, va="center",
                color="#444444")

# ─── Kingdom brackets (right side) ───────────────────────────────────────────
bracket_x = 0.985

def draw_bracket(ax, y_top, y_bot, label, x=bracket_x):
    mid = (y_top + y_bot) / 2
    ax.plot([x, x], [y_top, y_bot], color="#999999", linewidth=0.8, clip_on=False)
    ax.plot([x - 0.005, x], [y_top, y_top], color="#999999", linewidth=0.8, clip_on=False)
    ax.plot([x - 0.005, x], [y_bot, y_bot], color="#999999", linewidth=0.8, clip_on=False)
    ax.text(x + 0.005, mid, label, fontsize=7.5, va="center", color="#777777",
            fontstyle="italic", rotation=-90, clip_on=False)

y0 = lambda i: top_start - i * (row_h + gap)
draw_bracket(ax, y0(0) + 0.02, y0(2) - 0.02, "Mammalia")
draw_bracket(ax, y0(3) + 0.02, y0(3) - 0.02, "Arthropoda")
draw_bracket(ax, y0(4) + 0.02, y0(6) - 0.02, "Plantae")
draw_bracket(ax, y0(7) + 0.02, y0(7) - 0.02, "Fungi")

# Outer kingdom bracket
outer_x = 0.998
draw_bracket(ax, y0(0) + 0.025, y0(3) - 0.025, "Animalia", x=outer_x)

# ─── Summary stats strip ─────────────────────────────────────────────────────
strip_y = 0.08
stats = [
    ("7", "species"),
    ("8", "datasets"),
    (f"{total_seqs / 1e6:.1f}M", "sequences"),
    ("110–249", "bp range"),
]
stat_xs = np.linspace(0.15, 0.85, len(stats))
for x_pos, (val, desc) in zip(stat_xs, stats):
    ax.text(x_pos, strip_y + 0.02, val, fontsize=22, fontweight="bold",
            ha="center", va="center", color="#1B3A5C")
    ax.text(x_pos, strip_y - 0.02, desc, fontsize=10, ha="center",
            va="center", color="#777777")

# Thin separator line above stats
ax.plot([0.05, 0.95], [strip_y + 0.055, strip_y + 0.055],
        color="#DDDDDD", linewidth=0.8)

# ─── Sources ──────────────────────────────────────────────────────────────────
ax.text(0.5, 0.015,
        "Sources: ENCODE lentiMPRA (Tewhey et al.), DeepSTARR (de Almeida et al., 2022), "
        "Plant STARR-seq (Jores et al., 2021), DREAM Challenge (Schreiber et al., 2022)",
        fontsize=7, ha="center", va="center", color="#AAAAAA")

fig.savefig(os.path.join(OUT, "figure_dataset_overview.png"), facecolor="white")
fig.savefig(os.path.join(OUT, "figure_dataset_overview.pdf"), facecolor="white")
plt.close(fig)
print("Done: figure_dataset_overview")
