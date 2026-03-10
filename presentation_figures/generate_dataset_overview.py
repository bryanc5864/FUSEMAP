"""
Figure 1: FUSEMAP Training Corpus — Dataset overview infographic.
Clean table layout. Nimbus Sans font (Helvetica equivalent). All dark text.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ─── Style ────────────────────────────────────────────────────────────────────
FONT = "Nimbus Sans"
plt.rcParams.update({
    "font.family": FONT,
    "text.color": "#111111",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})

# ─── Dataset specs ────────────────────────────────────────────────────────────
datasets = [
    # (display_name, color, assay, n_seqs, seq_len, n_out, out_labels)
    ("Homo sapiens  \u2014  K562",              "#C0392B", "lentiMPRA",  196_664,   230, 1, ""),
    ("Homo sapiens  \u2014  HepG2",             "#922B21", "lentiMPRA",  122_926,   230, 1, ""),
    ("Homo sapiens  \u2014  WTC11",             "#17A589", "lentiMPRA",   46_128,   230, 1, ""),
    ("D. melanogaster  \u2014  S2",              "#E67E22", "STARR-seq",  484_052,   249, 2, "Dev, Hk"),
    ("Arabidopsis thaliana  \u2014  Leaf",      "#27AE60", "STARR-seq",   13_462,   170, 2, "leaf, proto"),
    ("Zea mays  \u2014  Leaf",                  "#1E8449", "STARR-seq",   24_604,   170, 2, "leaf, proto"),
    ("Sorghum bicolor  \u2014  Leaf",           "#6C8C3C", "STARR-seq",   19_673,   170, 2, "leaf, proto"),
    ("S. cerevisiae",                            "#C5A028", "FACS-seq",  6_810_361,  110, 1, ""),
]

total_seqs = sum(d[3] for d in datasets)
max_seqs = max(d[3] for d in datasets)

# ─── Figure ───────────────────────────────────────────────────────────────────
W, H = 18, 11.5
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ax.set_aspect("equal")

n_rows = len(datasets)
row_h = 0.78
row_gap = 0.12
table_top = 8.6

# ─── Title ────────────────────────────────────────────────────────────────────
ax.text(W / 2, 10.8, "FUSEMAP Training Corpus",
        fontsize=36, fontweight="bold", ha="center", va="center", color="#111111")
ax.text(W / 2, 10.1,
        f"{total_seqs:,} CRE\u2013activity pairs  \u00b7  7 species  \u00b7  4 kingdoms",
        fontsize=19, ha="center", va="center", color="#333333")

# ─── Column headers ──────────────────────────────────────────────────────────
hy = table_top + 0.50
hdr_kw = dict(fontsize=17, fontweight="bold", color="#333333", va="center")
ax.text(0.7, hy, "Species / Cell Type", **hdr_kw)
ax.text(6.3, hy, "Assay", ha="center", **hdr_kw)
ax.text(9.5, hy, "Sequences", ha="center", **hdr_kw)
ax.text(13.5, hy, "Length", ha="center", **hdr_kw)
ax.text(15.8, hy, "Outputs", ha="center", **hdr_kw)

# Header underline
ax.plot([0.3, W - 0.8], [table_top + 0.22, table_top + 0.22],
        color="#bbbbbb", linewidth=1.2, solid_capstyle="round")

# ─── Draw rows ────────────────────────────────────────────────────────────────
assay_colors = {"lentiMPRA": "#0077B6", "STARR-seq": "#E67E22", "FACS-seq": "#C5A028"}
bar_start = 7.6
bar_end = 11.6

for i, (name, color, assay, n_seq, seq_len, n_out, out_labels) in enumerate(datasets):
    yc = table_top - i * (row_h + row_gap)

    # Alternating background
    if i % 2 == 0:
        bg = FancyBboxPatch((0.25, yc - row_h / 2 + 0.02),
                            W - 1.0, row_h - 0.04,
                            boxstyle="round,pad=0.06", facecolor="#f0f3f7",
                            edgecolor="none", zorder=0)
        ax.add_patch(bg)

    # Color dot
    ax.plot(0.5, yc, "o", color=color, markersize=14, zorder=3)

    # Species + cell type
    ax.text(0.9, yc, name, fontsize=18, va="center", color="#111111",
            fontstyle="italic")

    # Assay badge
    ac = assay_colors.get(assay, color)
    bw, bh = 1.6, 0.48
    badge = FancyBboxPatch((6.3 - bw / 2, yc - bh / 2), bw, bh,
                           boxstyle="round,pad=0.06", facecolor=ac,
                           edgecolor="none", alpha=0.15, zorder=1)
    ax.add_patch(badge)
    ax.text(6.3, yc, assay, fontsize=16, fontweight="bold",
            ha="center", va="center", color=ac)

    # Proportional bar (log scale)
    bar_max_w = bar_end - bar_start
    log_frac = np.log10(max(n_seq, 1)) / np.log10(max_seqs)
    bar_w = log_frac * bar_max_w * 0.75
    bar_rect = FancyBboxPatch((bar_start, yc - 0.20), bar_w, 0.40,
                               boxstyle="round,pad=0.05", facecolor=color,
                               edgecolor="none", alpha=0.70, zorder=2)
    ax.add_patch(bar_rect)

    # Count label
    ax.text(12.2, yc, f"{n_seq:,}",
            fontsize=17, fontweight="bold", va="center", ha="right",
            color="#111111")

    # Length
    ax.text(13.5, yc, f"{seq_len} bp",
            fontsize=17, va="center", ha="center", color="#222222")

    # Outputs
    if n_out == 1:
        ax.text(15.8, yc, "1", fontsize=18, fontweight="bold",
                va="center", ha="center", color="#222222")
    else:
        ax.text(15.8, yc, f"{n_out}  ({out_labels})",
                fontsize=15, va="center", ha="center", color="#222222")

# ─── Kingdom brackets ────────────────────────────────────────────────────────
bx = W - 0.4

def draw_bracket(y_top, y_bot, label):
    mid = (y_top + y_bot) / 2
    ax.plot([bx, bx], [y_top, y_bot], color="#666666", linewidth=1.2, clip_on=False)
    ax.plot([bx - 0.15, bx], [y_top, y_top], color="#666666", linewidth=1.2, clip_on=False)
    ax.plot([bx - 0.15, bx], [y_bot, y_bot], color="#666666", linewidth=1.2, clip_on=False)
    ax.text(bx + 0.18, mid, label, fontsize=14, va="center", color="#444444",
            fontstyle="italic", rotation=-90, clip_on=False)

y_of = lambda i: table_top - i * (row_h + row_gap)
draw_bracket(y_of(0) + 0.28, y_of(2) - 0.28, "Mammalia")
draw_bracket(y_of(3) + 0.28, y_of(3) - 0.28, "Arthropoda")
draw_bracket(y_of(4) + 0.28, y_of(6) - 0.28, "Plantae")
draw_bracket(y_of(7) + 0.28, y_of(7) - 0.28, "Fungi")

# ─── Summary stats strip ─────────────────────────────────────────────────────
strip_y = 1.0
stats = [("7", "species"), ("8", "datasets"),
         (f"{total_seqs / 1e6:.1f}M", "sequences"), ("110\u2013249", "bp range")]
stat_xs = np.linspace(3.0, W - 3.0, len(stats))

ax.plot([0.5, W - 0.5], [strip_y + 0.70, strip_y + 0.70],
        color="#bbbbbb", linewidth=0.8)

for x_pos, (val, desc) in zip(stat_xs, stats):
    ax.text(x_pos, strip_y + 0.30, val, fontsize=36, fontweight="bold",
            ha="center", va="center", color="#111111")
    ax.text(x_pos, strip_y - 0.25, desc, fontsize=17, ha="center",
            va="center", color="#333333")

# ─── Sources ──────────────────────────────────────────────────────────────────
ax.text(W / 2, 0.15,
        "Sources: ENCODE lentiMPRA (Tewhey et al.), DeepSTARR (de Almeida et al., 2022), "
        "Plant STARR-seq (Jores et al., 2021), DREAM Challenge (Schreiber et al., 2022)",
        fontsize=12, ha="center", va="center", color="#555555")

fig.savefig(os.path.join(OUT, "figure_dataset_overview.png"), facecolor="white")
fig.savefig(os.path.join(OUT, "figure_dataset_overview.pdf"), facecolor="white")
plt.close(fig)
print("Done: figure_dataset_overview")
