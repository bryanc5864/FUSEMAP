"""
Figure 1: FUSEMAP Training Corpus — Dataset overview table.
Nimbus Sans (Helvetica). Clean text table, no colored bars.
Large text for presentation readability.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

FONT = "Nimbus Sans"
plt.rcParams.update({
    "font.family": FONT,
    "text.color": "#111111",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# ─── Dataset specs ────────────────────────────────────────────────────────────
datasets = [
    ("Homo sapiens  \u2014  K562",          "#C0392B", "lentiMPRA",  196_664,   230, 1, ""),
    ("Homo sapiens  \u2014  HepG2",         "#922B21", "lentiMPRA",  122_926,   230, 1, ""),
    ("Homo sapiens  \u2014  WTC11",         "#17A589", "lentiMPRA",   46_128,   230, 1, ""),
    ("D. melanogaster  \u2014  S2",         "#E67E22", "STARR-seq",  484_052,   249, 2, "Dev, Hk"),
    ("A. thaliana  \u2014  Leaf",           "#27AE60", "STARR-seq",   13_462,   170, 2, "leaf, proto"),
    ("Zea mays  \u2014  Leaf",             "#1E8449", "STARR-seq",   24_604,   170, 2, "leaf, proto"),
    ("Sorghum bicolor  \u2014  Leaf",      "#6C8C3C", "STARR-seq",   19_673,   170, 2, "leaf, proto"),
    ("S. cerevisiae",                       "#C5A028", "FACS-seq",  6_810_361,  110, 1, ""),
]

total_seqs = sum(d[3] for d in datasets)

# ─── Layout math ──────────────────────────────────────────────────────────────
n_rows = len(datasets)
row_h = 1.10                    # taller rows for bigger text
row_gap = 0.10
row_stride = row_h + row_gap    # 1.20 per row
table_height = n_rows * row_stride
title_block = 2.8               # more room for large title + subtitle
header_gap = 1.0                # gap between header line and first row
stats_block = 2.6               # bigger stats strip
bottom_pad = 1.0                # sources + margin
top_pad = 0.5

H = top_pad + title_block + header_gap + table_height + 0.8 + stats_block + bottom_pad
W = 22

fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ax.set_aspect("equal")

# Y positions (top to bottom)
title_y = H - top_pad - 0.7
subtitle_y = title_y - 1.1
header_y = subtitle_y - 1.2
header_line_y = header_y - 0.40
first_row_y = header_line_y - 0.65

# Column x-positions (wider figure)
cx_dot = 0.50
cx_name = 0.95
cx_assay = 9.0
cx_seqs = 13.2
cx_len = 16.2
cx_out = 19.2

# ─── Title ────────────────────────────────────────────────────────────────────
ax.text(W / 2, title_y, "FUSEMAP Training Corpus",
        fontsize=52, fontweight="bold", ha="center", va="center", color="#111111")
ax.text(W / 2, subtitle_y,
        f"{total_seqs:,} CRE\u2013activity pairs  \u00b7  7 species  \u00b7  4 kingdoms",
        fontsize=26, ha="center", va="center", color="#333333")

# ─── Column headers ──────────────────────────────────────────────────────────
hdr = dict(fontsize=24, fontweight="bold", color="#222222", va="center")
ax.text(cx_name, header_y, "Species / Cell Type", **hdr)
ax.text(cx_assay, header_y, "Assay", ha="center", **hdr)
ax.text(cx_seqs, header_y, "Sequences", ha="right", **hdr)
ax.text(cx_len, header_y, "Length", ha="center", **hdr)
ax.text(cx_out, header_y, "Outputs", ha="center", **hdr)

ax.plot([0.35, W - 1.5], [header_line_y, header_line_y],
        color="#bbbbbb", linewidth=1.2)

# ─── Draw rows ────────────────────────────────────────────────────────────────
assay_colors = {"lentiMPRA": "#0077B6", "STARR-seq": "#E67E22", "FACS-seq": "#C5A028"}

for i, (name, color, assay, n_seq, seq_len, n_out, out_labels) in enumerate(datasets):
    yc = first_row_y - i * row_stride

    if i % 2 == 0:
        bg = FancyBboxPatch((0.30, yc - row_h / 2 + 0.04),
                            W - 1.8, row_h - 0.08,
                            boxstyle="round,pad=0.06", facecolor="#f2f5f9",
                            edgecolor="none", zorder=0)
        ax.add_patch(bg)

    ax.plot(cx_dot, yc, "o", color=color, markersize=18, zorder=3)

    ax.text(cx_name, yc, name, fontsize=24, va="center", color="#111111",
            fontstyle="italic")

    ac = assay_colors.get(assay, color)
    bw, bh = 2.2, 0.68
    badge = FancyBboxPatch((cx_assay - bw / 2, yc - bh / 2), bw, bh,
                           boxstyle="round,pad=0.06", facecolor=ac,
                           edgecolor="none", alpha=0.13, zorder=1)
    ax.add_patch(badge)
    ax.text(cx_assay, yc, assay, fontsize=22, fontweight="bold",
            ha="center", va="center", color=ac)

    ax.text(cx_seqs, yc, f"{n_seq:,}",
            fontsize=24, fontweight="bold", va="center", ha="right",
            color="#111111")

    ax.text(cx_len, yc, f"{seq_len} bp",
            fontsize=24, va="center", ha="center", color="#111111")

    if n_out == 1:
        ax.text(cx_out, yc, "1", fontsize=24, fontweight="bold",
                va="center", ha="center", color="#111111")
    else:
        ax.text(cx_out, yc, f"{n_out}  ({out_labels})",
                fontsize=21, va="center", ha="center", color="#111111")

# ─── Kingdom brackets ────────────────────────────────────────────────────────
bx = W - 0.90

def draw_bracket(y_top, y_bot, label):
    mid = (y_top + y_bot) / 2
    ax.plot([bx, bx], [y_top, y_bot], color="#555555", linewidth=1.5, clip_on=False)
    ax.plot([bx - 0.15, bx], [y_top, y_top], color="#555555", linewidth=1.5, clip_on=False)
    ax.plot([bx - 0.15, bx], [y_bot, y_bot], color="#555555", linewidth=1.5, clip_on=False)
    ax.text(bx + 0.20, mid, label, fontsize=18, va="center", color="#333333",
            fontstyle="italic", rotation=-90, clip_on=False)

y_of = lambda i: first_row_y - i * row_stride
draw_bracket(y_of(0) + 0.35, y_of(2) - 0.35, "Mammalia")
draw_bracket(y_of(3) + 0.35, y_of(3) - 0.35, "Arthropoda")
draw_bracket(y_of(4) + 0.35, y_of(6) - 0.35, "Plantae")
draw_bracket(y_of(7) + 0.35, y_of(7) - 0.35, "Fungi")

# ─── Summary stats strip ─────────────────────────────────────────────────────
last_row_y = first_row_y - (n_rows - 1) * row_stride
strip_center = last_row_y - row_h / 2 - 1.3

stats = [("7", "species"), ("8", "datasets"),
         (f"{total_seqs / 1e6:.1f}M", "sequences"), ("110\u2013249", "bp range")]
stat_xs = np.linspace(3.5, W - 3.5, len(stats))

ax.plot([0.5, W - 0.5], [strip_center + 0.80, strip_center + 0.80],
        color="#bbbbbb", linewidth=1.0)

for x_pos, (val, desc) in zip(stat_xs, stats):
    ax.text(x_pos, strip_center + 0.22, val, fontsize=52, fontweight="bold",
            ha="center", va="center", color="#111111")
    ax.text(x_pos, strip_center - 0.48, desc, fontsize=24, ha="center",
            va="center", color="#333333")

# ─── Sources ──────────────────────────────────────────────────────────────────
ax.text(W / 2, strip_center - 1.15,
        "Sources: ENCODE lentiMPRA (Tewhey et al.), DeepSTARR (de Almeida et al., 2022), "
        "Plant STARR-seq (Jores et al., 2021), DREAM Challenge (Schreiber et al., 2022)",
        fontsize=16, ha="center", va="center", color="#555555")

fig.savefig(os.path.join(OUT, "figure_dataset_overview.png"), facecolor="white")
fig.savefig(os.path.join(OUT, "figure_dataset_overview.pdf"), facecolor="white")
plt.close(fig)
print("Done: figure_dataset_overview")
