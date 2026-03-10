"""
Figure 2: CRE Activity Distributions — Overlapping ridgeline plot from real FUSEMAP data.
Uses raw (unnormalized) activity values. Nimbus Sans font (Helvetica equivalent).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
import os, warnings
warnings.filterwarnings("ignore")

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(OUT), "data")

# ─── Style ────────────────────────────────────────────────────────────────────
FONT = "Nimbus Sans"
sns.set_theme(style="white", font=FONT, font_scale=1.4)
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 22,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.linewidth": 0.5,
    "text.color": "#111111",
})

# ─── Load datasets ────────────────────────────────────────────────────────────
print("Loading datasets...")

def load_tsv(path, col, sep="\t", subsample=None):
    df = pd.read_csv(path, sep=sep, usecols=[col])
    vals = df[col].dropna().values.astype(float)
    if subsample and len(vals) > subsample:
        rng = np.random.RandomState(42)
        vals = rng.choice(vals, subsample, replace=False)
    return vals

def load_splits(cell_type, col="activity"):
    """Load raw (unnormalized) human lentiMPRA data from splits/."""
    sdir = f"{DATA}/lentiMPRA_data/{cell_type}/splits"
    parts = []
    for f in ["train.tsv", "val.tsv", "test.tsv", "calibration.tsv"]:
        fpath = os.path.join(sdir, f)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, sep="\t", usecols=[col])
            parts.append(df[col].dropna().values.astype(float))
    return np.concatenate(parts)

# Human lentiMPRA — RAW unnormalized log2(RNA/DNA)
k562  = load_splits("K562")
hepg2 = load_splits("HepG2")
wtc11 = load_splits("WTC11")

# Drosophila S2
s2_files = [f"{DATA}/S2_data/splits/{s}.tsv" for s in ["train", "val", "test"]]
s2_dev = np.concatenate([load_tsv(f, "Dev_log2_enrichment") for f in s2_files])
s2_hk  = np.concatenate([load_tsv(f, "Hk_log2_enrichment")  for f in s2_files])

# Plant data
arab = load_tsv(f"{DATA}/plant_data/jores2021/processed/arabidopsis/arabidopsis_train.tsv",
                "enrichment_leaf")
maize = load_tsv(f"{DATA}/plant_data/jores2021/processed/maize/maize_train.tsv",
                 "enrichment_leaf")
sorghum = load_tsv(f"{DATA}/plant_data/jores2021/processed/sorghum/sorghum_train.tsv",
                   "enrichment_leaf")

# Yeast (subsample for speed)
yeast = load_tsv(f"{DATA}/yeast_data/yeast_train.txt", "label", subsample=500_000)

print("All datasets loaded.")

# ─── Ridge data ───────────────────────────────────────────────────────────────
ridge_data = [
    ("K562",        "Human",      k562,    "#C0392B", len(k562)),
    ("HepG2",       "Human",      hepg2,   "#922B21", len(hepg2)),
    ("WTC11",       "Human",      wtc11,   "#17A589", len(wtc11)),
    ("S2 Dev",      "Drosophila", s2_dev,  "#E67E22", len(s2_dev)),
    ("S2 Hk",       "Drosophila", s2_hk,   "#D4880F", len(s2_hk)),
    ("Arabidopsis", "Plant",      arab,    "#27AE60", len(arab)),
    ("Maize",       "Plant",      maize,   "#1E8449", len(maize)),
    ("Sorghum",     "Plant",      sorghum, "#6C8C3C", len(sorghum)),
    ("Yeast",       "Fungi",      yeast,   "#C5A028", 6_705_562),
]

# ─── Compute KDEs ─────────────────────────────────────────────────────────────
x_min, x_max = -7, 14
x_grid = np.linspace(x_min, x_max, 800)

densities = []
for name, group, vals, color, n in ridge_data:
    clipped = vals[(vals > x_min) & (vals < x_max)]
    try:
        kde = gaussian_kde(clipped, bw_method=0.15)
        density = kde(x_grid)
    except Exception:
        density = np.zeros_like(x_grid)
    densities.append(density)

global_max = max(d.max() for d in densities if d.max() > 0)

# ─── Figure layout ────────────────────────────────────────────────────────────
n_ridges = len(ridge_data)
ridge_spacing = 0.45  # Tight overlap — peaks bleed into adjacent rows

fig = plt.figure(figsize=(15, 13))
gs = gridspec.GridSpec(2, 1, height_ratios=[n_ridges, 3.0], hspace=0.20,
                       figure=fig, top=0.93, bottom=0.05, left=0.15, right=0.87)

ax_main = fig.add_subplot(gs[0])
gs_bottom = gs[1].subgridspec(1, 3, wspace=0.28)

# ─── Title ────────────────────────────────────────────────────────────────────
fig.text(0.51, 0.97,
         "Regulatory Activity Distributions Across the FUSEMAP Training Corpus",
         fontsize=24, fontweight="bold", ha="center", va="center",
         color="#111111")
fig.text(0.51, 0.945,
         r"Activity measured as $\log_2$(RNA/DNA) from MPRA / STARR-seq / FACS-seq",
         fontsize=15, ha="center", va="center", color="#444444")

# ─── Main ridgeline panel ────────────────────────────────────────────────────
for i, ((name, group, vals, color, n), density) in enumerate(zip(ridge_data, densities)):
    y_offset = (n_ridges - 1 - i) * ridge_spacing
    scaled = density / global_max * 1.0  # Taller than spacing → visible overlap

    ax_main.fill_between(x_grid, y_offset, y_offset + scaled,
                         color=color, alpha=0.55, zorder=n_ridges - i + 1)
    ax_main.plot(x_grid, y_offset + scaled, color=color, linewidth=1.1,
                 alpha=0.9, zorder=n_ridges - i + 2)

    # Baseline
    ax_main.plot([x_min, x_max], [y_offset, y_offset],
                 color="#ddd", linewidth=0.3, zorder=0)

    # Left label
    if n >= 1_000_000:
        n_str = f"n = {n/1e6:.1f}M"
    elif n >= 1000:
        n_str = f"n = {n/1e3:.0f}K"
    else:
        n_str = f"n = {n}"

    ax_main.text(x_min - 0.5, y_offset + ridge_spacing * 0.45,
                 name, fontsize=16, fontweight="bold",
                 ha="right", va="center", color=color)
    ax_main.text(x_min - 0.5, y_offset + ridge_spacing * 0.10,
                 f"({group})", fontsize=12, ha="right", va="center", color="#333333")
    ax_main.text(x_min - 0.5, y_offset - ridge_spacing * 0.18,
                 n_str, fontsize=11, ha="right", va="center", color="#555555")

    # Right side: stats
    clipped_vals = vals[(vals > x_min) & (vals < x_max)]
    mean = np.mean(clipped_vals)
    std = np.std(clipped_vals)

    # Median tick
    median = np.median(clipped_vals)
    med_idx = np.argmin(np.abs(x_grid - median))
    med_height = y_offset + scaled[med_idx]
    ax_main.plot([median, median], [y_offset, med_height],
                 color="#222222", linewidth=1.0, zorder=n_ridges + 5, alpha=0.5)

    ax_main.text(x_max + 0.5, y_offset + ridge_spacing * 0.20,
                 f"\u03bc = {mean:.2f}", fontsize=13, fontweight="bold",
                 va="center", color="#222222")
    ax_main.text(x_max + 0.5, y_offset - ridge_spacing * 0.12,
                 f"\u03c3 = {std:.2f}", fontsize=12,
                 va="center", color="#333333")

# Zero line
ax_main.axvline(0, color="#999999", linewidth=0.9, linestyle="--", zorder=0, alpha=0.5)

# Light gridlines
for xv in range(-6, 15, 2):
    ax_main.axvline(xv, color="#eee", linewidth=0.4, zorder=0)

ax_main.set_xlim(x_min, x_max)
ax_main.set_ylim(-0.1, (n_ridges - 1) * ridge_spacing + 1.15)
ax_main.set_yticks([])
ax_main.tick_params(axis="x", labelsize=14, colors="#222222")
ax_main.set_xlabel(r"Regulatory Activity  ($\log_2$ RNA/DNA)", fontsize=18,
                   labelpad=8, color="#111111")
ax_main.spines["bottom"].set_color("#888888")

# ─── Bottom comparison panels ─────────────────────────────────────────────────
panel_specs = [
    {
        "title": "Within-kingdom similarity",
        "subtitle": "Plant distributions overlap\n\u2192 physics transfer works",
        "datasets": [
            ("Arabidopsis", arab, "#27AE60"),
            ("Maize", maize, "#1E8449"),
            ("Sorghum", sorghum, "#6C8C3C"),
        ],
        "x_range": (-5, 6),
    },
    {
        "title": "Within-species, across cell types",
        "subtitle": "Human cell types differ in spread\n\u2192 cell-type TF usage dominates",
        "datasets": [
            ("K562", k562, "#C0392B"),
            ("HepG2", hepg2, "#922B21"),
            ("WTC11", wtc11, "#17A589"),
        ],
        "x_range": (-4, 4),
    },
    {
        "title": "Cross-kingdom gap",
        "subtitle": "Different assays & promoter architecture\n\u2192 cross-kingdom transfer challenging",
        "datasets": [
            ("K562 (Human)", k562, "#C0392B"),
            ("Maize (Plant)", maize, "#1E8449"),
            ("Yeast (Fungi)", yeast, "#C5A028"),
        ],
        "x_range": (-5, 14),
    },
]

for j, spec in enumerate(panel_specs):
    ax_in = fig.add_subplot(gs_bottom[0, j])

    xr = spec["x_range"]
    x_sub = np.linspace(xr[0], xr[1], 300)

    for lname, vals, col in spec["datasets"]:
        clipped = vals[(vals > xr[0]) & (vals < xr[1])]
        try:
            kde = gaussian_kde(clipped, bw_method=0.2)
            d = kde(x_sub)
            d = d / d.max()
        except Exception:
            d = np.zeros_like(x_sub)
        ax_in.fill_between(x_sub, d, alpha=0.30, color=col, label=lname)
        ax_in.plot(x_sub, d, color=col, linewidth=1.3, alpha=0.85)

    ax_in.set_xlim(*xr)
    ax_in.set_ylim(0, 1.18)
    ax_in.set_yticks([])
    ax_in.spines["top"].set_visible(False)
    ax_in.spines["right"].set_visible(False)
    ax_in.spines["left"].set_visible(False)
    ax_in.spines["bottom"].set_color("#888888")
    ax_in.tick_params(axis="x", labelsize=12, colors="#222222")
    ax_in.set_xlabel(r"$\log_2$(RNA/DNA)", fontsize=13, labelpad=3, color="#222222")

    ax_in.set_title(spec["title"], fontsize=15, fontweight="bold",
                    color="#111111", pad=8)
    ax_in.text(0.5, -0.38, spec["subtitle"], transform=ax_in.transAxes,
               fontsize=11, ha="center", color="#444444", linespacing=1.3)

    leg = ax_in.legend(fontsize=11, frameon=False, loc="upper right",
                       handlelength=1.0, handletextpad=0.4)
    for t in leg.get_texts():
        t.set_color("#222222")

fig.savefig(os.path.join(OUT, "figure_activity_distributions.png"), facecolor="white")
fig.savefig(os.path.join(OUT, "figure_activity_distributions.pdf"), facecolor="white")
plt.close(fig)
print("Done: figure_activity_distributions")
