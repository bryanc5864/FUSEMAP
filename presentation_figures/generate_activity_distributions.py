"""
Figure 2: CRE Activity Distributions — Ridgeline plot from real FUSEMAP data.
Uses seaborn theming + matplotlib ridgeline with Cantarell font.
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
FONT = "Cantarell"
sns.set_theme(style="white", font=FONT, font_scale=1.3)
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.linewidth": 0.5,
    "text.color": "#1a1a2e",
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

k562   = load_tsv(f"{DATA}/lentiMPRA_data/K562/fold_splits_with_seq/all_folds.tsv",
                  "Observed log2(RNA/DNA)")
hepg2  = load_tsv(f"{DATA}/lentiMPRA_data/HepG2/fold_splits_with_seq/all_folds.tsv",
                  "Observed log2(RNA/DNA)")
wtc11  = load_tsv(f"{DATA}/lentiMPRA_data/WTC11/fold_splits_with_seq/all_folds.tsv",
                  "Observed log2(RNA/DNA)")

s2_files = [f"{DATA}/S2_data/splits/{s}.tsv" for s in ["train", "val", "test"]]
s2_dev = np.concatenate([load_tsv(f, "Dev_log2_enrichment") for f in s2_files])
s2_hk  = np.concatenate([load_tsv(f, "Hk_log2_enrichment")  for f in s2_files])

arab = load_tsv(f"{DATA}/plant_data/jores2021/processed/arabidopsis/arabidopsis_train.tsv",
                "enrichment_leaf")
maize = load_tsv(f"{DATA}/plant_data/jores2021/processed/maize/maize_train.tsv",
                 "enrichment_leaf")
sorghum = load_tsv(f"{DATA}/plant_data/jores2021/processed/sorghum/sorghum_train.tsv",
                   "enrichment_leaf")

yeast = load_tsv(f"{DATA}/yeast_data/yeast_train.txt", "label", subsample=500_000)

print("All datasets loaded.")

# ─── Ridge data ───────────────────────────────────────────────────────────────
ridge_data = [
    ("K562\n(Human)",        k562,    "#C0392B", len(k562)),
    ("HepG2\n(Human)",       hepg2,   "#922B21", len(hepg2)),
    ("WTC11\n(Human)",       wtc11,   "#17A589", len(wtc11)),
    ("S2 Dev\n(Drosophila)", s2_dev,  "#E67E22", len(s2_dev)),
    ("S2 Hk\n(Drosophila)",  s2_hk,   "#D4880F", len(s2_hk)),
    ("Arabidopsis\n(Plant)", arab,    "#27AE60", len(arab)),
    ("Maize\n(Plant)",       maize,   "#1E8449", len(maize)),
    ("Sorghum\n(Plant)",     sorghum, "#6C8C3C", len(sorghum)),
    ("Yeast\n(Fungi)",       yeast,   "#C5A028", 6_705_562),
]

# ─── Compute KDEs ─────────────────────────────────────────────────────────────
x_min, x_max = -6, 11
x_grid = np.linspace(x_min, x_max, 600)

densities = []
for label, vals, color, n in ridge_data:
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
ridge_spacing = 1.0

fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(2, 1, height_ratios=[n_ridges, 3.0], hspace=0.22,
                       figure=fig, top=0.92, bottom=0.06, left=0.14, right=0.88)

ax_main = fig.add_subplot(gs[0])
gs_bottom = gs[1].subgridspec(1, 3, wspace=0.30)

# ─── Title (figure-level) ────────────────────────────────────────────────────
fig.text(0.51, 0.97,
         "Regulatory Activity Distributions Across the FUSEMAP Training Corpus",
         fontsize=22, fontweight="bold", ha="center", va="center",
         color="#0f2744")
fig.text(0.51, 0.94,
         "Activity measured as log\u2082(RNA/DNA) from MPRA / STARR-seq / FACS-seq",
         fontsize=14, ha="center", va="center", color="#7a8a9a")

# ─── Main ridgeline panel ────────────────────────────────────────────────────
for i, ((label, vals, color, n), density) in enumerate(zip(ridge_data, densities)):
    y_offset = (n_ridges - 1 - i) * ridge_spacing
    scaled = density / global_max * ridge_spacing * 0.88

    ax_main.fill_between(x_grid, y_offset, y_offset + scaled,
                         color=color, alpha=0.6, zorder=n_ridges - i + 1)
    ax_main.plot(x_grid, y_offset + scaled, color=color, linewidth=1.0,
                 alpha=0.85, zorder=n_ridges - i + 2)

    # Baseline
    ax_main.plot([x_min, x_max], [y_offset, y_offset],
                 color="#e0e4e8", linewidth=0.4, zorder=0)

    # Left label: name + count
    if n >= 1_000_000:
        n_str = f"n = {n/1e6:.1f}M"
    elif n >= 1000:
        n_str = f"n = {n/1e3:.0f}K"
    else:
        n_str = f"n = {n}"
    ax_main.text(x_min - 0.4, y_offset + ridge_spacing * 0.30,
                 label.split("\n")[0], fontsize=15, fontweight="bold",
                 ha="right", va="center", color=color)
    ax_main.text(x_min - 0.4, y_offset + ridge_spacing * 0.08,
                 label.split("\n")[1] if "\n" in label else "",
                 fontsize=12, ha="right", va="center", color="#7a8a9a")
    ax_main.text(x_min - 0.4, y_offset - ridge_spacing * 0.10,
                 n_str, fontsize=11, ha="right", va="center", color="#9aaaba")

    # Right side: stats
    clipped_vals = vals[(vals > x_min) & (vals < x_max)]
    median = np.median(clipped_vals)
    mean = np.mean(clipped_vals)
    std = np.std(clipped_vals)

    # Median tick
    med_idx = np.argmin(np.abs(x_grid - median))
    med_height = y_offset + scaled[med_idx]
    ax_main.plot([median, median], [y_offset, med_height],
                 color="#2a2a3e", linewidth=0.9, zorder=n_ridges + 5, alpha=0.5)

    ax_main.text(x_max + 0.4, y_offset + ridge_spacing * 0.15,
                 f"\u03bc = {mean:.2f}  \u00b1  {std:.2f}",
                 fontsize=12, va="center", color="#5a6a7a")

# Zero line
ax_main.axvline(0, color="#b0b8c0", linewidth=0.9, linestyle="--", zorder=0, alpha=0.5)

# Light gridlines
for xv in range(-6, 12, 2):
    ax_main.axvline(xv, color="#eef0f2", linewidth=0.4, zorder=0)

ax_main.set_xlim(x_min - 0.2, x_max + 0.2)
ax_main.set_ylim(-0.2, n_ridges * ridge_spacing + 0.15)
ax_main.set_yticks([])
ax_main.tick_params(axis="x", labelsize=14)
ax_main.set_xlabel("Regulatory Activity  (log\u2082 RNA/DNA)", fontsize=16, labelpad=8)
ax_main.spines["bottom"].set_color("#c0c8d0")
ax_main.spines["bottom"].set_linewidth(0.6)

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
        "subtitle": "Human cell types differ in tails\n\u2192 cell-type TF usage dominates",
        "datasets": [
            ("K562", k562, "#C0392B"),
            ("HepG2", hepg2, "#922B21"),
            ("WTC11", wtc11, "#17A589"),
        ],
        "x_range": (-5, 8),
    },
    {
        "title": "Cross-kingdom gap",
        "subtitle": "Different assays & promoter architecture\n\u2192 cross-kingdom transfer challenging",
        "datasets": [
            ("K562 (Human)", k562, "#C0392B"),
            ("Maize (Plant)", maize, "#1E8449"),
            ("Yeast (Fungi)", yeast, "#C5A028"),
        ],
        "x_range": (-5, 8),
    },
]

for j, spec in enumerate(panel_specs):
    ax_in = fig.add_subplot(gs_bottom[0, j])

    xr = spec["x_range"]
    x_sub = np.linspace(xr[0], xr[1], 300)

    for name, vals, col in spec["datasets"]:
        clipped = vals[(vals > xr[0]) & (vals < xr[1])]
        try:
            kde = gaussian_kde(clipped, bw_method=0.2)
            d = kde(x_sub)
            d = d / d.max()
        except Exception:
            d = np.zeros_like(x_sub)
        ax_in.fill_between(x_sub, d, alpha=0.30, color=col, label=name)
        ax_in.plot(x_sub, d, color=col, linewidth=1.2, alpha=0.8)

    ax_in.set_xlim(*xr)
    ax_in.set_ylim(0, 1.18)
    ax_in.set_yticks([])
    ax_in.spines["top"].set_visible(False)
    ax_in.spines["right"].set_visible(False)
    ax_in.spines["left"].set_visible(False)
    ax_in.spines["bottom"].set_color("#c0c8d0")
    ax_in.tick_params(axis="x", labelsize=11)
    ax_in.set_xlabel("log\u2082(RNA/DNA)", fontsize=12, labelpad=3)

    ax_in.set_title(spec["title"], fontsize=14, fontweight="bold",
                    color="#1a2a3e", pad=8)
    ax_in.text(0.5, -0.42, spec["subtitle"], transform=ax_in.transAxes,
               fontsize=10, ha="center", color="#7a8a9a", linespacing=1.3)

    leg = ax_in.legend(fontsize=10, frameon=False, loc="upper right",
                       handlelength=1.0, handletextpad=0.4)

fig.savefig(os.path.join(OUT, "figure_activity_distributions.png"), facecolor="white")
fig.savefig(os.path.join(OUT, "figure_activity_distributions.pdf"), facecolor="white")
plt.close(fig)
print("Done: figure_activity_distributions")
