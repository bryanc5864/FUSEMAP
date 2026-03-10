"""
Figure 1: FUSEMAP Training Corpus — Dataset overview table.
Nimbus Sans (Helvetica). No colored bars — just clean text table.
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
    "savefig.pad_inches": 0.1,
})

# ─── Dataset specs ────────────────────────────────────────────────────────────
datasets = [
    # (display_name, color, assay, n_seqs, seq_len, n_out, out_labels)
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

# ─── Figure ───────────────────────────────────────────────────────────────────
W, H = 15, 9
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ax.set_aspect("equal")

n_rows = len(datasets)
row_h = 0.72
row_gap = 0.06
table_top = 7.2

# Column x-positions
cx_dot = 0.35
cx_name = 0.65
cx_assay = 5.8
cx_seqs = 8.8
cx_len = 11.0
cx_out = 13.0

# ─── Title ────────────────────────────────────────────────────────────────────
ax.text(W / 2, 8.6, "FUSEMAP Training Corpus",
        fontsize=34, fontweight="bold", ha="center", va="center", color="#111111")
ax.text(W / 2, 8.0,
        f"{total_seqs:,} CRE\u2013activity pairs  \u00b7  7 species  \u00b7  4 kingdoms",
        fontsize=17, ha="center", va="center", color="#333333")

# ─── Column headers ──────────────────────────────────────────────────────────
hy = table_top + 0.42
hdr = dict(fontsize=15, fontweight="bold", color="#222222", va="center")
ax.text(cx_name, hy, "Species / Cell Type", **hdr)
ax.text(cx_assay, hy, "Assay", ha="center", **hdr)
ax.text(cx_seqs, hy, "Sequences", ha="right", **hdr)
ax.text(cx_len, hy, "Length", ha="center", **hdr)
ax.text(cx_out, hy, "Outputs", ha="center", **hdr)

ax.plot([0.25, W - 1.0], [table_top + 0.18, table_top + 0.18],
        color="#bbbbbb", linewidth=1.0)

# ─── Draw rows ────────────────────────────────────────────────────────────────
assay_colors = {"lentiMPRA": "#0077B6", "STARR-seq": "#E67E22", "FACS-seq": "#C5A028"}

for i, (name, color, assay, n_seq, seq_len, n_out, out_labels) in enumerate(datasets):
    yc = table_top - i * (row_h + row_gap)

    # Alternating background
    if i % 2 == 0:
        bg = FancyBboxPatch((0.2, yc - row_h / 2 + 0.03),
                            W - 1.2, row_h - 0.06,
                            boxstyle="round,pad=0.05", facecolor="#f2f5f9",
                            edgecolor="none", zorder=0)
        ax.add_patch(bg)

    # Color dot
    ax.plot(cx_dot, yc, "o", color=color, markersize=13, zorder=3)

    # Species + cell type
    ax.text(cx_name, yc, name, fontsize=16, va="center", color="#111111",
            fontstyle="italic")

    # Assay badge
    ac = assay_colors.get(assay, color)
    bw, bh = 1.5, 0.44
    badge = FancyBboxPatch((cx_assay - bw / 2, yc - bh / 2), bw, bh,
                           boxstyle="round,pad=0.05", facecolor=ac,
                           edgecolor="none", alpha=0.13, zorder=1)
    ax.add_patch(badge)
    ax.text(cx_assay, yc, assay, fontsize=14, fontweight="bold",
            ha="center", va="center", color=ac)

    # Sequence count (right-aligned, just the number)
    ax.text(cx_seqs, yc, f"{n_seq:,}",
            fontsize=16, fontweight="bold", va="center", ha="right",
            color="#111111")

    # Length
    ax.text(cx_len, yc, f"{seq_len} bp",
            fontsize=16, va="center", ha="center", color="#111111")

    # Outputs
    if n_out == 1:
        ax.text(cx_out, yc, "1", fontsize=16, fontweight="bold",
                va="center", ha="center", color="#111111")
    else:
        ax.text(cx_out, yc, f"{n_out}  ({out_labels})",
                fontsize=14, va="center", ha="center", color="#111111")

# ─── Kingdom brackets ────────────────────────────────────────────────────────
bx = W - 0.65

def draw_bracket(y_top, y_bot, label):
    mid = (y_top + y_bot) / 2
    ax.plot([bx, bx], [y_top, y_bot], color="#555555", linewidth=1.2, clip_on=False)
    ax.plot([bx - 0.12, bx], [y_top, y_top], color="#555555", linewidth=1.2, clip_on=False)
    ax.plot([bx - 0.12, bx], [y_bot, y_bot], color="#555555", linewidth=1.2, clip_on=False)
    ax.text(bx + 0.15, mid, label, fontsize=12, va="center", color="#333333",
            fontstyle="italic", rotation=-90, clip_on=False)

y_of = lambda i: table_top - i * (row_h + row_gap)
draw_bracket(y_of(0) + 0.25, y_of(2) - 0.25, "Mammalia")
draw_bracket(y_of(3) + 0.25, y_of(3) - 0.25, "Arthropoda")
draw_bracket(y_of(4) + 0.25, y_of(6) - 0.25, "Plantae")
draw_bracket(y_of(7) + 0.25, y_of(7) - 0.25, "Fungi")

# ─── Summary stats strip ─────────────────────────────────────────────────────
strip_y = 0.7
stats = [("7", "species"), ("8", "datasets"),
         (f"{total_seqs / 1e6:.1f}M", "sequences"), ("110\u2013249", "bp range")]
stat_xs = np.linspace(2.5, W - 2.5, len(stats))

ax.plot([0.4, W - 0.4], [strip_y + 0.60, strip_y + 0.60],
        color="#bbbbbb", linewidth=0.8)

for x_pos, (val, desc) in zip(stat_xs, stats):
    ax.text(x_pos, strip_y + 0.22, val, fontsize=34, fontweight="bold",
            ha="center", va="center", color="#111111")
    ax.text(x_pos, strip_y - 0.22, desc, fontsize=15, ha="center",
            va="center", color="#333333")

# ─── Sources ──────────────────────────────────────────────────────────────────
ax.text(W / 2, 0.12,
        "Sources: ENCODE lentiMPRA (Tewhey et al.), DeepSTARR (de Almeida et al., 2022), "
        "Plant STARR-seq (Jores et al., 2021), DREAM Challenge (Schreiber et al., 2022)",
        fontsize=10, ha="center", va="center", color="#555555")

fig.savefig(os.path.join(OUT, "figure_dataset_overview.png"), facecolor="white")
fig.savefig(os.path.join(OUT, "figure_dataset_overview.pdf"), facecolor="white")
plt.close(fig)
print("Done: figure_dataset_overview")
