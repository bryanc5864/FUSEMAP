"""
Figure 2: CRE Activity Distributions — Ridgeline plot from real FUSEMAP data.
Loads actual datasets and plots activity distributions as overlapping density ridges.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import os, warnings
warnings.filterwarnings("ignore")

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(OUT), "data")

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 13,
    "axes.titlesize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.linewidth": 0.6,
    "text.color": "#222222",
})

# ─── Load datasets ────────────────────────────────────────────────────────────
print("Loading datasets...")

def load_tsv(path, col, sep="\t", subsample=None):
    """Load a single column from a TSV, optionally subsampling."""
    df = pd.read_csv(path, sep=sep, usecols=[col])
    vals = df[col].dropna().values.astype(float)
    if subsample and len(vals) > subsample:
        rng = np.random.RandomState(42)
        vals = rng.choice(vals, subsample, replace=False)
    return vals

# Human lentiMPRA
k562   = load_tsv(f"{DATA}/lentiMPRA_data/K562/fold_splits_with_seq/all_folds.tsv",
                  "Observed log2(RNA/DNA)")
hepg2  = load_tsv(f"{DATA}/lentiMPRA_data/HepG2/fold_splits_with_seq/all_folds.tsv",
                  "Observed log2(RNA/DNA)")
wtc11  = load_tsv(f"{DATA}/lentiMPRA_data/WTC11/fold_splits_with_seq/all_folds.tsv",
                  "Observed log2(RNA/DNA)")

# Drosophila S2 — combine train+val+test
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

# ─── Dataset metadata ─────────────────────────────────────────────────────────
ridge_data = [
    ("K562 (Human)",            k562,    "#C0392B", len(k562)),
    ("HepG2 (Human)",           hepg2,   "#7B241C", len(hepg2)),
    ("WTC11 (Human)",           wtc11,   "#1ABC9C", len(wtc11)),
    ("S2 Dev (Drosophila)",     s2_dev,  "#E67E22", len(s2_dev)),
    ("S2 Hk (Drosophila)",      s2_hk,   "#F0A04B", len(s2_hk)),
    ("Arabidopsis (Plant)",     arab,    "#27AE60", len(arab)),
    ("Maize (Plant)",           maize,   "#1E8449", len(maize)),
    ("Sorghum (Plant)",         sorghum, "#6C8C3C", len(sorghum)),
    ("Yeast (Fungi)",           yeast,   "#D4AC0D", 6_705_562),  # full count
]

# ─── Compute KDEs ─────────────────────────────────────────────────────────────
x_min, x_max = -6, 10
x_grid = np.linspace(x_min, x_max, 600)

densities = []
for label, vals, color, n in ridge_data:
    # Clip extreme outliers for KDE stability
    clipped = vals[(vals > x_min) & (vals < x_max)]
    try:
        kde = gaussian_kde(clipped, bw_method=0.15)
        density = kde(x_grid)
    except Exception:
        density = np.zeros_like(x_grid)
    densities.append(density)

# Normalize all densities to same max height for visual consistency
global_max = max(d.max() for d in densities if d.max() > 0)

# ─── Figure: Ridgeline plot ──────────────────────────────────────────────────
n_ridges = len(ridge_data)
ridge_spacing = 1.0
fig_h = 2.0 + n_ridges * 0.85 + 3.5  # extra space for bottom panel

fig, (ax_main, ax_bottom) = plt.subplots(
    2, 1, figsize=(14, fig_h),
    gridspec_kw={"height_ratios": [n_ridges, 2.8], "hspace": 0.25}
)

# Main ridgeline panel
for i, ((label, vals, color, n), density) in enumerate(zip(ridge_data, densities)):
    y_offset = (n_ridges - 1 - i) * ridge_spacing
    scaled = density / global_max * ridge_spacing * 0.85

    ax_main.fill_between(x_grid, y_offset, y_offset + scaled,
                         color=color, alpha=0.65, zorder=n_ridges - i + 1)
    ax_main.plot(x_grid, y_offset + scaled, color=color, linewidth=0.8,
                 alpha=0.9, zorder=n_ridges - i + 2)

    # Baseline
    ax_main.plot([x_min, x_max], [y_offset, y_offset],
                 color="#DDDDDD", linewidth=0.3, zorder=0)

    # Label (left side)
    if n >= 1_000_000:
        n_str = f"n={n/1e6:.1f}M"
    elif n >= 1000:
        n_str = f"n={n/1e3:.0f}K"
    else:
        n_str = f"n={n}"
    ax_main.text(x_min - 0.3, y_offset + ridge_spacing * 0.25,
                 label, fontsize=10, fontweight="bold",
                 ha="right", va="center", color=color)
    ax_main.text(x_min - 0.3, y_offset + ridge_spacing * 0.02,
                 n_str, fontsize=8, ha="right", va="center", color="#888888")

    # Stats annotation (right side)
    median = np.median(vals[(vals > x_min) & (vals < x_max)])
    mean = np.mean(vals[(vals > x_min) & (vals < x_max)])
    std = np.std(vals[(vals > x_min) & (vals < x_max)])

    # Median tick on the ridge
    med_idx = np.argmin(np.abs(x_grid - median))
    med_height = y_offset + scaled[med_idx]
    ax_main.plot([median, median], [y_offset, med_height],
                 color="black", linewidth=0.8, zorder=n_ridges + 5, alpha=0.6)

    ax_main.text(x_max + 0.3, y_offset + ridge_spacing * 0.15,
                 f"\u03bc={mean:.2f} \u00b1 {std:.2f}",
                 fontsize=7.5, va="center", color="#666666")

# Zero line
ax_main.axvline(0, color="#BBBBBB", linewidth=0.8, linestyle="--", zorder=0, alpha=0.6)

# Gridlines
for xv in range(-6, 11, 2):
    ax_main.axvline(xv, color="#EEEEEE", linewidth=0.4, zorder=0)

ax_main.set_xlim(x_min - 0.2, x_max + 0.2)
ax_main.set_ylim(-0.15, n_ridges * ridge_spacing + 0.1)
ax_main.set_yticks([])
ax_main.set_xlabel("")
ax_main.tick_params(axis="x", labelsize=10)
ax_main.set_title("Regulatory Activity Distributions Across the FUSEMAP Training Corpus",
                   fontsize=16, fontweight="bold", color="#1B3A5C", pad=12)
# Subtitle
ax_main.text(0.5, 1.02,
             "Activity measured as log\u2082(RNA/DNA) from MPRA/STARR-seq",
             transform=ax_main.transAxes, fontsize=10, ha="center",
             color="#777777")

# X-axis label on main
ax_main.set_xlabel("Regulatory Activity (log\u2082 RNA/DNA)", fontsize=12, labelpad=8)

# ─── Bottom comparison panel ──────────────────────────────────────────────────
ax_bottom.set_xlim(x_min, x_max)
ax_bottom.set_ylim(0, 1.1)
ax_bottom.set_yticks([])
ax_bottom.axis("off")

# Three subpanels side by side
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
            ("HepG2", hepg2, "#7B241C"),
            ("WTC11", wtc11, "#1ABC9C"),
        ],
        "x_range": (-5, 8),
    },
    {
        "title": "Cross-kingdom gap",
        "subtitle": "Different assays & promoter architecture\n\u2192 cross-kingdom transfer is challenging",
        "datasets": [
            ("K562 (Human)", k562, "#C0392B"),
            ("Maize (Plant)", maize, "#1E8449"),
            ("Yeast (Fungi)", yeast, "#D4AC0D"),
        ],
        "x_range": (-5, 8),
    },
]

# Create inset axes for each comparison
inset_width = 0.27
inset_height = 0.7
inset_y = 0.15
for j, spec in enumerate(panel_specs):
    inset_x = 0.06 + j * 0.33
    ax_in = fig.add_axes([inset_x, 0.02, inset_width, 0.16])

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
        ax_in.fill_between(x_sub, d, alpha=0.35, color=col, label=name)
        ax_in.plot(x_sub, d, color=col, linewidth=0.8, alpha=0.8)

    ax_in.set_xlim(*xr)
    ax_in.set_ylim(0, 1.15)
    ax_in.set_yticks([])
    ax_in.spines["top"].set_visible(False)
    ax_in.spines["right"].set_visible(False)
    ax_in.spines["left"].set_visible(False)
    ax_in.tick_params(axis="x", labelsize=7)
    ax_in.set_xlabel("log\u2082(RNA/DNA)", fontsize=7, labelpad=2)

    # Title above inset
    ax_in.set_title(spec["title"], fontsize=9, fontweight="bold",
                    color="#333333", pad=4)
    # Subtitle below title
    ax_in.text(0.5, -0.45, spec["subtitle"], transform=ax_in.transAxes,
               fontsize=7, ha="center", color="#888888", linespacing=1.3)

    # Small legend
    leg = ax_in.legend(fontsize=6.5, frameon=False, loc="upper right",
                       handlelength=1, handletextpad=0.4)

fig.savefig(os.path.join(OUT, "figure_activity_distributions.png"), facecolor="white")
fig.savefig(os.path.join(OUT, "figure_activity_distributions.pdf"), facecolor="white")
plt.close(fig)
print("Done: figure_activity_distributions")
