#!/usr/bin/env python3
"""
Figure 14. Physics-activity relationships across systems.

5-panel composite figure:
  (A) Heatmap of top 50 physics features correlated with activity across datasets
  (B) Top feature |r| by dataset (scatter/lollipop) for Maize (bending) vs K562 (entropy)
  (C) Feature importance by category across datasets
  (D) Physics-only R² across datasets (plants vs animals)
  (E) ElasticNet coefficient profiles for human cell types
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
})

FUSEMAP_ROOT = Path('/home/bcheng/sequence_optimization/FUSEMAP')
PHYSICS_RESULTS = FUSEMAP_ROOT / 'physics/results'

PLANT_COLOR = '#5BB75B'
ANIMAL_COLOR = '#4A90D9'

CATEGORY_COLORS = {
    'Bending': '#66BB6A',
    'Thermo': '#FF7043',
    'Entropy': '#42A5F5',
    'Stiffness': '#AB47BC',
    'Advanced': '#FFA726',
    'PWM': '#EF5350',
    'Other': '#BDBDBD',
}

# ---------------------------------------------------------------------------
# Helper: map feature prefix to category
# ---------------------------------------------------------------------------
def feature_to_category(feat_name):
    if feat_name.startswith('bend_'):
        return 'Bending'
    elif feat_name.startswith('thermo_'):
        return 'Thermo'
    elif feat_name.startswith('entropy_'):
        return 'Entropy'
    elif feat_name.startswith('stiff_'):
        return 'Stiffness'
    elif feat_name.startswith('advanced_'):
        return 'Advanced'
    elif feat_name.startswith('pwm_'):
        return 'PWM'
    else:
        return 'Other'

# ---------------------------------------------------------------------------
# Load univariate correlation data for all datasets
# ---------------------------------------------------------------------------
dataset_configs = {
    'K562':      PHYSICS_RESULTS / 'human/01_univariate_stability/univariate_K562.csv',
    'HepG2':     PHYSICS_RESULTS / 'human/01_univariate_stability/univariate_HepG2.csv',
    'WTC11':     PHYSICS_RESULTS / 'human/01_univariate_stability/univariate_WTC11.csv',
    'S2 Dev':    PHYSICS_RESULTS / 'drosophila_dev/01_univariate_stability/univariate_S2.csv',
    'S2 Hk':     PHYSICS_RESULTS / 'drosophila_hk/01_univariate_stability/univariate_S2.csv',
    'Maize':     PHYSICS_RESULTS / 'maize_leaf/01_univariate_stability/univariate_maize.csv',
    'Sorghum':   PHYSICS_RESULTS / 'sorghum_leaf/01_univariate_stability/univariate_sorghum.csv',
    'Arabidopsis': PHYSICS_RESULTS / 'arabidopsis_leaf/01_univariate_stability/univariate_arabidopsis.csv',
}

print("Loading univariate correlation data...")
all_corr = {}
all_features = set()

for dset_name, csv_path in dataset_configs.items():
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        corr_map = dict(zip(df['feature'], df['pearson_r']))
        all_corr[dset_name] = corr_map
        all_features.update(df['feature'].tolist())
        print(f"  {dset_name}: {len(df)} features")
    else:
        print(f"  {dset_name}: NOT FOUND at {csv_path}")

# ---------------------------------------------------------------------------
# Panel A: Heatmap of top 50 physics features
# ---------------------------------------------------------------------------
# Find top 50 by max absolute correlation across datasets (exclude PWM)
print("Computing top features for heatmap...")
feature_max_abs = {}
for feat in all_features:
    cat = feature_to_category(feat)
    if cat == 'PWM':  # exclude PWM for cleaner visualization
        continue
    max_r = 0
    for dset_name in all_corr:
        r = abs(all_corr[dset_name].get(feat, 0))
        if r > max_r:
            max_r = r
    feature_max_abs[feat] = max_r

# Sort by max |r| and take top 50
top_features = sorted(feature_max_abs, key=lambda x: -feature_max_abs[x])[:50]

# Build heatmap matrix
dataset_order = ['K562', 'HepG2', 'WTC11', 'S2 Dev', 'S2 Hk', 'Maize', 'Sorghum', 'Arabidopsis']
heatmap_data = np.zeros((len(top_features), len(dataset_order)))
for i, feat in enumerate(top_features):
    for j, dset in enumerate(dataset_order):
        heatmap_data[i, j] = all_corr.get(dset, {}).get(feat, 0)

# Feature category labels for coloring
feat_categories = [feature_to_category(f) for f in top_features]
# Shorten feature names for display
def shorten_name(name):
    for prefix in ['bend_', 'thermo_', 'entropy_', 'stiff_', 'advanced_']:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

short_names = [shorten_name(f) for f in top_features]

# ---------------------------------------------------------------------------
# Panel C data: Feature category importance by dataset
# From paper Table 24 and extended analysis
# ---------------------------------------------------------------------------
# Top feature and dominant category by dataset (from paper Table 24)
dataset_labels_c = ['K562', 'HepG2', 'WTC11', 'S2 Dev', 'S2 Hk', 'Arabidopsis', 'Sorghum', 'Maize']
# Count significant features by category for each dataset
print("Computing category importance...")
category_counts = {}
for dset_name in dataset_order:
    if dset_name not in all_corr:
        continue
    cat_sum = {}
    for feat, r in all_corr[dset_name].items():
        cat = feature_to_category(feat)
        if cat not in cat_sum:
            cat_sum[cat] = 0
        cat_sum[cat] += abs(r)
    total = sum(cat_sum.values()) if sum(cat_sum.values()) > 0 else 1
    category_counts[dset_name] = {k: v/total for k, v in cat_sum.items()}

# ---------------------------------------------------------------------------
# Panel D data: Physics-only R²
# ---------------------------------------------------------------------------
datasets_d = ['Maize\nLeaf', 'Sorghum\nLeaf', 'Arab.\nLeaf',
              'S2 Dev', 'S2 Hk', 'K562', 'HepG2', 'WTC11']
r2_physics = [0.464, 0.451, 0.279, 0.142, 0.116, 0.070, 0.061, 0.143]
kingdoms_d = ['Plant', 'Plant', 'Plant', 'Animal', 'Animal', 'Animal', 'Animal', 'Animal']

# ---------------------------------------------------------------------------
# Panel E data: ElasticNet coefficients (from report.txt)
# ---------------------------------------------------------------------------
# Top features per cell type
k562_features = [
    ('gc_entropy_w30_mean', +0.726, 'Entropy'),
    ('dH_p90', -0.652, 'Thermo'),
    ('gc_entropy_w50_mean', +0.538, 'Entropy'),
    ('global_kmer5_entropy', -0.423, 'Entropy'),
    ('conditional_entropy', +0.386, 'Entropy'),
    ('dS_p5', +0.327, 'Thermo'),
    ('stacking_std_energy', -0.275, 'Advanced'),
    ('bending_energy_var', +0.265, 'Bending'),
]

hepg2_features = [
    ('curvature_var_w5', -0.841, 'Bending'),
    ('narrow_groove_frac', +0.640, 'Advanced'),
    ('shannon_w10_mean', +0.574, 'Entropy'),
    ('entropy_rate', -0.541, 'Entropy'),
    ('kmer2_entropy_w30', +0.532, 'Entropy'),
    ('global_kmer4_ent', -0.493, 'Entropy'),
    ('var_dG', +0.477, 'Thermo'),
    ('local_opening_rate', -0.440, 'Advanced'),
]

wtc11_features = [
    ('gc_content_global', -1.500, 'Stiffness'),
    ('mean_mgw', +1.125, 'Advanced'),
    ('soft_min_melting_dG', +1.068, 'Advanced'),
    ('var_dG', +0.912, 'Thermo'),
    ('stress_mean_opening', -0.908, 'Advanced'),
    ('bending_energy_var', +0.786, 'Bending'),
    ('std_melting_dG', -0.645, 'Advanced'),
    ('curvature_var_w9', -0.547, 'Bending'),
]

# ---------------------------------------------------------------------------
# Create figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(22, 18))
gs = gridspec.GridSpec(3, 2, hspace=0.42, wspace=0.28,
                       left=0.08, right=0.96, top=0.92, bottom=0.04,
                       height_ratios=[1.3, 1.0, 1.0])

# ===== Panel A: Heatmap ====================================================
ax_a = fig.add_subplot(gs[0, :])  # span full width
im = ax_a.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', vmin=-0.4, vmax=0.4)

ax_a.set_xticks(range(len(dataset_order)))
ax_a.set_xticklabels(dataset_order, fontsize=10, fontweight='bold', rotation=30, ha='right')
ax_a.set_yticks(range(len(short_names)))
ax_a.set_yticklabels(short_names, fontsize=7, fontweight='bold')

# Color the y-axis labels by category
for i, cat in enumerate(feat_categories):
    color = CATEGORY_COLORS.get(cat, '#333333')
    ax_a.get_yticklabels()[i].set_color(color)

# Colorbar
cbar = fig.colorbar(im, ax=ax_a, shrink=0.6, pad=0.02, label='Pearson r')
cbar.ax.tick_params(labelsize=9)

# Kingdom divider line
ax_a.axvline(4.5, color='black', linewidth=2, linestyle='-')
ax_a.text(2.0, -2.0, 'Animals', fontsize=11, fontweight='bold', ha='center',
          color=ANIMAL_COLOR)
ax_a.text(6.5, -2.0, 'Plants', fontsize=11, fontweight='bold', ha='center',
          color=PLANT_COLOR)

ax_a.set_title('Top 50 Physics Features × Activity Correlation Across Datasets', fontsize=13, pad=15)
ax_a.text(-0.03, 1.04, '(A)', transform=ax_a.transAxes, fontsize=16, fontweight='bold', va='top')

# ===== Panel C: Category importance stacked bar ============================
ax_c = fig.add_subplot(gs[1, 0])
cats_order = ['Bending', 'Thermo', 'Entropy', 'Stiffness', 'Advanced', 'PWM']
bar_datasets = ['K562', 'HepG2', 'WTC11', 'S2 Dev', 'S2 Hk', 'Maize', 'Sorghum', 'Arabidopsis']
x_c = np.arange(len(bar_datasets))
bottom = np.zeros(len(bar_datasets))

for cat in cats_order:
    vals = []
    for dset in bar_datasets:
        if dset in category_counts:
            vals.append(category_counts[dset].get(cat, 0))
        else:
            vals.append(0)
    vals = np.array(vals)
    ax_c.bar(x_c, vals, bottom=bottom, color=CATEGORY_COLORS[cat],
             label=cat, width=0.7, edgecolor='white', linewidth=0.5)
    bottom += vals

ax_c.set_xticks(x_c)
ax_c.set_xticklabels(bar_datasets, fontsize=9, fontweight='bold', rotation=30, ha='right')
ax_c.set_ylabel('Relative Importance\n(Sum |r| fraction)', fontsize=10)
ax_c.set_ylim(0, 1.05)
ax_c.set_title('Feature Category Importance by Dataset', fontsize=12)
ax_c.legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.98),
            framealpha=0.9)

# Kingdom labels
ax_c.axvline(4.5, color='black', linewidth=1.5, linestyle='--', alpha=0.5)

ax_c.text(-0.08, 1.06, '(B)', transform=ax_c.transAxes, fontsize=16, fontweight='bold', va='top')

# ===== Panel D: Physics-only R² bar chart ==================================
ax_d = fig.add_subplot(gs[1, 1])
bar_colors_d = [PLANT_COLOR if k == 'Plant' else ANIMAL_COLOR for k in kingdoms_d]
x_d = np.arange(len(datasets_d))
bars_d = ax_d.bar(x_d, r2_physics, color=bar_colors_d, edgecolor='white',
                  linewidth=0.8, width=0.7, zorder=3)

for xi, val in zip(x_d, r2_physics):
    ax_d.text(xi, val + 0.008, f'{val:.3f}', ha='center', va='bottom',
              fontsize=8, fontweight='bold')

plant_mean = np.mean([0.464, 0.451, 0.279])
animal_mean = np.mean([0.142, 0.116, 0.070, 0.061, 0.143])
fold = plant_mean / animal_mean

ax_d.axhline(plant_mean, xmin=0.02, xmax=0.40, color=PLANT_COLOR,
             ls='--', lw=2.0, zorder=4)
ax_d.axhline(animal_mean, xmin=0.42, xmax=0.98, color=ANIMAL_COLOR,
             ls='--', lw=2.0, zorder=4)

ax_d.text(0.8, plant_mean + 0.012, f'Plant = {plant_mean:.3f}',
          fontsize=9, fontweight='bold', color=PLANT_COLOR)
ax_d.text(5.0, animal_mean + 0.012, f'Animal = {animal_mean:.3f}',
          fontsize=9, fontweight='bold', color=ANIMAL_COLOR)

# Fold annotation
mid_y = (plant_mean + animal_mean) / 2
ax_d.annotate('', xy=(7.6, plant_mean), xytext=(7.6, animal_mean),
              arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5))
ax_d.text(7.75, mid_y, f'{fold:.1f}×', fontsize=12, fontweight='bold',
          color='#D32F2F', va='center')

ax_d.set_xticks(x_d)
ax_d.set_xticklabels(datasets_d, fontsize=9, fontweight='bold')
ax_d.set_ylabel('Physics-Only R²', fontsize=10)
ax_d.set_ylim(0, 0.56)
ax_d.set_title('Variance Explained by Physics Features', fontsize=12)

plant_patch = mpatches.Patch(color=PLANT_COLOR, label='Plant')
animal_patch = mpatches.Patch(color=ANIMAL_COLOR, label='Animal')
ax_d.legend(handles=[plant_patch, animal_patch], loc='upper right',
            framealpha=0.9, fontsize=10)

ax_d.text(-0.08, 1.06, '(C)', transform=ax_d.transAxes, fontsize=16, fontweight='bold', va='top')

# ===== Panel E: ElasticNet coefficients for 3 human cell types =============
# Three horizontal bar subplots
gs_e = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, :], wspace=0.35)

for idx, (cell_type, features, cell_color) in enumerate([
    ('K562', k562_features, '#4A90D9'),
    ('HepG2', hepg2_features, '#E67E22'),
    ('WTC11', wtc11_features, '#27AE60'),
]):
    ax = fig.add_subplot(gs_e[0, idx])
    names = [f[0] for f in features]
    coeffs = [f[1] for f in features]
    cats = [f[2] for f in features]
    colors = [CATEGORY_COLORS.get(c, '#999999') for c in cats]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, coeffs, color=colors, edgecolor='white',
                   linewidth=0.5, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8, fontweight='bold')
    ax.set_xlabel('ElasticNet Coefficient', fontsize=9)
    ax.set_title(f'{cell_type}', fontsize=11, fontweight='bold', color=cell_color)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.invert_yaxis()

    # Annotate values
    for j, (coeff, name) in enumerate(zip(coeffs, names)):
        ha = 'left' if coeff >= 0 else 'right'
        offset = 0.02 if coeff >= 0 else -0.02
        ax.text(coeff + offset, j, f'{coeff:+.3f}', va='center', ha=ha,
                fontsize=7, fontweight='bold')

    if idx == 0:
        ax.text(-0.15, 1.10, '(D)', transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

# ===== Category legend for panel E =========================================
legend_patches = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c)
                  for c in ['Bending', 'Thermo', 'Entropy', 'Stiffness', 'Advanced']]
fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=10,
           framealpha=0.9, bbox_to_anchor=(0.5, 0.005))

# ===== Suptitle =============================================================
fig.suptitle(
    'Figure 14. Physics-Activity Relationships Across Systems',
    fontsize=16, fontweight='bold', y=0.97)

# ===== Save =================================================================
out_base = str(FUSEMAP_ROOT / 'presentation_figures/figure14_physics_activity_correlations')
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(out_base + '.pdf', bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f'Saved: {out_base}.png')
print(f'Saved: {out_base}.pdf')
