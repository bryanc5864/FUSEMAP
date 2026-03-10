"""
Figure 1: FUSEMAP Training Corpus — Dataset overview infographic.
Clean table layout using Cantarell font, inches-based coordinate system.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ─── Font & style ─────────────────────────────────────────────────────────────
FONT = "Cantarell"
plt.rcParams.update({
    "font.family": FONT,
    "text.color": "#1a1a2e",
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
    ("Drosophila melanogaster  \u2014  S2",     "#E67E22", "STARR-seq",  484_052,   249, 2, "Dev, Hk"),
    ("Arabidopsis thaliana  \u2014  Leaf",      "#27AE60", "STARR-seq",   13_462,   170, 2, "leaf, proto"),
    ("Zea mays  \u2014  Leaf",                  "#1E8449", "STARR-seq",   24_604,   170, 2, "leaf, proto"),
    ("Sorghum bicolor  \u2014  Leaf",           "#6C8C3C", "STARR-seq",   19_673,   170, 2, "leaf, proto"),
    ("Saccharomyces cerevisiae",                "#C5A028", "FACS-seq",  6_810_361,  110, 1, ""),
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
        fontsize=34, fontweight="bold", ha="center", va="center", color="#0f2744")
ax.text(W / 2, 10.15,
        f"{total_seqs:,} CRE\u2013activity pairs  \u00b7  7 species  \u00b7  4 kingdoms",
        fontsize=18, ha="center", va="center", color="#5a6a7a")

# ─── Column headers ──────────────────────────────────────────────────────────
hy = table_top + 0.50
hdr_kw = dict(fontsize=16, fontweight="bold", color="#5a6a7a", va="center")
ax.text(0.7, hy, "Species / Cell Type", **hdr_kw)
ax.text(6.3, hy, "Assay", ha="center", **hdr_kw)
ax.text(9.8, hy, "Sequences", ha="center", **hdr_kw)
ax.text(14.0, hy, "Length", ha="center", **hdr_kw)
ax.text(16.2, hy, "Outputs", ha="center", **hdr_kw)

# Header underline
ax.plot([0.3, W - 0.8], [table_top + 0.22, table_top + 0.22],
        color="#d0d8e0", linewidth=1.0, solid_capstyle="round")

# ─── Draw rows ────────────────────────────────────────────────────────────────
assay_colors = {"lentiMPRA": "#0077B6", "STARR-seq": "#E67E22", "FACS-seq": "#C5A028"}
bar_start = 7.8
bar_end = 12.0

for i, (name, color, assay, n_seq, seq_len, n_out, out_labels) in enumerate(datasets):
    yc = table_top - i * (row_h + row_gap)

    # Alternating background
    if i % 2 == 0:
        bg = FancyBboxPatch((0.25, yc - row_h / 2 + 0.02),
                            W - 1.0, row_h - 0.04,
                            boxstyle="round,pad=0.06", facecolor="#f4f7fa",
                            edgecolor="none", zorder=0)
        ax.add_patch(bg)

    # Color dot
    ax.plot(0.5, yc, "o", color=color, markersize=14, zorder=3)

    # Species + cell type (plain text, no LaTeX)
    ax.text(0.9, yc, name, fontsize=17, va="center", color="#1a1a2e",
            fontstyle="italic")

    # Assay badge
    ac = assay_colors.get(assay, color)
    bw, bh = 1.6, 0.48
    badge = FancyBboxPatch((6.3 - bw / 2, yc - bh / 2), bw, bh,
                           boxstyle="round,pad=0.06", facecolor=ac,
                           edgecolor="none", alpha=0.12, zorder=1)
    ax.add_patch(badge)
    ax.text(6.3, yc, assay, fontsize=15, fontweight="bold",
            ha="center", va="center", color=ac)

    # Proportional bar (log scale)
    bar_max_w = bar_end - bar_start
    log_frac = np.log10(max(n_seq, 1)) / np.log10(max_seqs)
    bar_w = log_frac * bar_max_w * 0.75
    bar_rect = FancyBboxPatch((bar_start, yc - 0.20), bar_w, 0.40,
                               boxstyle="round,pad=0.05", facecolor=color,
                               edgecolor="none", alpha=0.70, zorder=2)
    ax.add_patch(bar_rect)

    # Count label (right-aligned to fixed position)
    ax.text(12.8, yc, f"{n_seq:,}",
            fontsize=16, fontweight="bold", va="center", ha="right",
            color="#2a3a4a")

    # Length
    ax.text(14.0, yc, f"{seq_len} bp",
            fontsize=16, va="center", ha="center", color="#3a4a5a")

    # Outputs
    if n_out == 1:
        ax.text(16.2, yc, "1", fontsize=17, fontweight="bold",
                va="center", ha="center", color="#3a4a5a")
    else:
        ax.text(16.2, yc, f"{n_out}  ({out_labels})",
                fontsize=14, va="center", ha="center", color="#3a4a5a")

# ─── Kingdom brackets ────────────────────────────────────────────────────────
bx = W - 0.4

def draw_bracket(y_top, y_bot, label):
    mid = (y_top + y_bot) / 2
    ax.plot([bx, bx], [y_top, y_bot], color="#8a9aaa", linewidth=1.2, clip_on=False)
    ax.plot([bx - 0.15, bx], [y_top, y_top], color="#8a9aaa", linewidth=1.2, clip_on=False)
    ax.plot([bx - 0.15, bx], [y_bot, y_bot], color="#8a9aaa", linewidth=1.2, clip_on=False)
    ax.text(bx + 0.18, mid, label, fontsize=13, va="center", color="#7a8a9a",
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
        color="#d0d8e0", linewidth=0.8)

for x_pos, (val, desc) in zip(stat_xs, stats):
    ax.text(x_pos, strip_y + 0.30, val, fontsize=34, fontweight="bold",
            ha="center", va="center", color="#0f2744")
    ax.text(x_pos, strip_y - 0.25, desc, fontsize=16, ha="center",
            va="center", color="#7a8a9a")

# ─── Sources ──────────────────────────────────────────────────────────────────
ax.text(W / 2, 0.15,
        "Sources: ENCODE lentiMPRA (Tewhey et al.), DeepSTARR (de Almeida et al., 2022), "
        "Plant STARR-seq (Jores et al., 2021), DREAM Challenge (Schreiber et al., 2022)",
        fontsize=11, ha="center", va="center", color="#a0aab4")

fig.savefig(os.path.join(OUT, "figure_dataset_overview.png"), facecolor="white")
fig.savefig(os.path.join(OUT, "figure_dataset_overview.pdf"), facecolor="white")
plt.close(fig)
print("Done: figure_dataset_overview")
