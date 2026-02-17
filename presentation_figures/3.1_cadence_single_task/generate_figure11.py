#!/usr/bin/env python3
"""
Figure 11. S2A Zero-Shot Cross-Species Activity Transfer

4-panel composite figure showing:
  (A) Zero-shot Spearman rho by transfer scenario
  (B) Leave-one-out holdout results across species
  (C) Calibration curve for Maize holdout (Spearman + R^2)
  (D) Key insight: within-plant vs within-human fold-change comparison
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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
    'figure.facecolor': 'white',
})

fig = plt.figure(figsize=(18, 14), facecolor='white')
fig.suptitle(
    'Figure 11. S2A Zero-Shot Cross-Species Activity Transfer',
    fontsize=20, fontweight='bold', y=0.97
)

# ===== PANEL A — Bar chart of zero-shot Spearman rho by transfer scenario ===
ax_a = fig.add_subplot(2, 2, 1)

scenarios = ['Within\nPlant', 'Within\nHuman', 'Plant\u2192\nAnimal', 'Animal\u2192\nPlant']
rho_vals = [0.700, 0.260, 0.125, -0.321]
colors_a = ['#5BB75B', '#4A90D9', '#E8833A', '#E74C3C']

bars_a = ax_a.bar(scenarios, rho_vals, color=colors_a, edgecolor='black',
                  linewidth=0.8, width=0.60, zorder=3)

# Value labels on bars
for bar, val in zip(bars_a, rho_vals):
    y = val
    va = 'bottom' if val >= 0 else 'top'
    offset = 0.02 if val >= 0 else -0.02
    ax_a.text(bar.get_x() + bar.get_width() / 2, y + offset,
              f'{val:.3f}', ha='center', va=va, fontweight='bold', fontsize=12)

# Red warning annotation for anti-correlation
ax_a.annotate(
    'Anti-correlated!\n(\u03c1 = \u22120.32)',
    xy=(3, -0.321), xytext=(2.2, -0.50),
    fontsize=10, fontweight='bold', color='#E74C3C',
    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2),
    bbox=dict(boxstyle='round,pad=0.4', fc='#FDEDED', ec='#E74C3C', lw=1.5),
    ha='center'
)

ax_a.axhline(0, color='black', linewidth=0.8, zorder=2)
ax_a.set_ylabel('Spearman \u03c1 (zero-shot)', fontsize=13)
ax_a.set_title('Zero-Shot Transfer Scenarios', fontsize=14)
ax_a.set_ylim(-0.65, 0.90)
ax_a.text(-0.12, 1.05, '(A)', transform=ax_a.transAxes,
          fontsize=16, fontweight='bold', va='top')


# ===== PANEL B — Leave-one-out holdout results ==============================
ax_b = fig.add_subplot(2, 2, 2)

holdouts = ['Maize', 'Sorghum', 'Arab.', 'WTC11', 'K562', 'HepG2', 'S2']
rho_b = [0.700, 0.370, 0.308, 0.184, 0.050, 0.045, -0.085]
kingdoms = ['Plant', 'Plant', 'Plant', 'Human', 'Human', 'Human', 'Dros.']
kingdom_colors = ['#5BB75B', '#5BB75B', '#5BB75B',
                  '#4A90D9', '#4A90D9', '#4A90D9', '#9B59B6']

bars_b = ax_b.barh(np.arange(len(holdouts)), rho_b,
                   color=kingdom_colors, edgecolor='black', linewidth=0.8,
                   height=0.60, zorder=3)
ax_b.set_yticks(np.arange(len(holdouts)))
ax_b.set_yticklabels(holdouts, fontsize=12, fontweight='bold')
ax_b.invert_yaxis()

# Value labels
for i, (bar, val) in enumerate(zip(bars_b, rho_b)):
    x_pos = val + 0.015 if val >= 0 else val - 0.015
    ha = 'left' if val >= 0 else 'right'
    color = '#E74C3C' if val < 0 else 'black'
    ax_b.text(x_pos, i, f'{val:.3f}', ha=ha, va='center',
              fontweight='bold', fontsize=11, color=color)

ax_b.axvline(0, color='black', linewidth=0.8, zorder=2)
ax_b.set_xlabel('Spearman \u03c1 (holdout)', fontsize=13)
ax_b.set_title('Leave-One-Out Holdout Performance', fontsize=14)
ax_b.set_xlim(-0.20, 0.85)

# Kingdom legend
legend_handles = [
    mpatches.Patch(color='#5BB75B', label='Plant'),
    mpatches.Patch(color='#4A90D9', label='Human'),
    mpatches.Patch(color='#9B59B6', label='Drosophila'),
]
ax_b.legend(handles=legend_handles, loc='lower right', fontsize=10,
            framealpha=0.9, edgecolor='gray')

ax_b.text(-0.12, 1.05, '(B)', transform=ax_b.transAxes,
          fontsize=16, fontweight='bold', va='top')


# ===== PANEL C — Calibration curve for Maize holdout =======================
ax_c = fig.add_subplot(2, 2, 3)

n_samples = [0, 10, 20, 50, 100, 200, 500]
spearman_c = [0.700, 0.701, 0.700, 0.701, 0.700, 0.703, 0.702]
r2_c = [0.351, 0.351, 0.439, 0.468, 0.475, 0.476, 0.483]

color_spearman = '#2C7FB8'
color_r2 = '#D95F02'

ln1 = ax_c.plot(n_samples, spearman_c, 'o-', color=color_spearman,
                linewidth=2.5, markersize=8, label='Spearman \u03c1', zorder=4)
ax_c.fill_between(n_samples, [s - 0.005 for s in spearman_c],
                  [s + 0.005 for s in spearman_c],
                  color=color_spearman, alpha=0.12, zorder=2)
ax_c.set_xlabel('Number of Maize calibration samples', fontsize=13)
ax_c.set_ylabel('Spearman \u03c1', fontsize=13, color=color_spearman)
ax_c.tick_params(axis='y', labelcolor=color_spearman)
ax_c.set_ylim(0.55, 0.80)

# Second y-axis for R^2
ax_c2 = ax_c.twinx()
ax_c2.spines['top'].set_visible(False)
ln2 = ax_c2.plot(n_samples, r2_c, 's--', color=color_r2,
                 linewidth=2.5, markersize=8, label='R\u00b2', zorder=4)
ax_c2.fill_between(n_samples, [r - 0.01 for r in r2_c],
                   [r + 0.01 for r in r2_c],
                   color=color_r2, alpha=0.12, zorder=2)
ax_c2.set_ylabel('R\u00b2', fontsize=13, color=color_r2)
ax_c2.tick_params(axis='y', labelcolor=color_r2)
ax_c2.set_ylim(0.25, 0.60)

# Annotation: R^2 improves, Spearman is flat
ax_c.annotate(
    'Spearman \u03c1 stable\n(\u0394 < 0.003)',
    xy=(250, 0.702), xytext=(350, 0.76),
    fontsize=10, fontweight='bold', color=color_spearman,
    arrowprops=dict(arrowstyle='->', color=color_spearman, lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', fc='#E8F0FE', ec=color_spearman, lw=1),
    ha='center'
)
ax_c2.annotate(
    'R\u00b2 improves\n+37.6%',
    xy=(300, 0.478), xytext=(400, 0.34),
    fontsize=10, fontweight='bold', color=color_r2,
    arrowprops=dict(arrowstyle='->', color=color_r2, lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', fc='#FFF0E0', ec=color_r2, lw=1),
    ha='center'
)

# Combined legend
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax_c.legend(lns, labs, loc='center left', fontsize=11, framealpha=0.9,
            edgecolor='gray')

ax_c.set_title('Maize Holdout: Calibration Curve', fontsize=14)
ax_c.text(-0.12, 1.05, '(C)', transform=ax_c.transAxes,
          fontsize=16, fontweight='bold', va='top')


# ===== PANEL D — Key insight: within-plant vs within-human ==================
ax_d = fig.add_subplot(2, 2, 4)

# Comparative bars
categories = ['Within-Plant', 'Within-Human']
values = [0.700, 0.260]
bar_colors = ['#5BB75B', '#4A90D9']

bars_d = ax_d.bar(categories, values, color=bar_colors, edgecolor='black',
                  linewidth=1.0, width=0.50, zorder=3)

# Value labels
for bar, val in zip(bars_d, values):
    ax_d.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
              f'\u03c1 = {val:.2f}', ha='center', va='bottom',
              fontweight='bold', fontsize=14)

# Fold-change arrow between the two bars
x0 = bars_d[0].get_x() + bars_d[0].get_width() / 2
x1 = bars_d[1].get_x() + bars_d[1].get_width() / 2
y_arrow = 0.72

# Horizontal connector with arrow
ax_d.annotate(
    '', xy=(x1, 0.62), xytext=(x0, 0.62),
    arrowprops=dict(arrowstyle='<->', color='#333333', lw=2.5,
                    connectionstyle='arc3,rad=-0.15')
)

# Fold-change label
x_mid = (x0 + x1) / 2
ax_d.text(x_mid, 0.74, '2.7\u00d7', ha='center', va='center',
          fontsize=26, fontweight='bold', color='#C0392B',
          bbox=dict(boxstyle='round,pad=0.4', fc='#FDEDED', ec='#C0392B',
                    lw=2, alpha=0.95))

# Descriptive annotation
ax_d.text(x_mid, 0.88,
          'Plant promoters encode\nmore transferable sequence\u2013activity signals',
          ha='center', va='center', fontsize=11, fontweight='bold',
          fontstyle='italic', color='#2C3E50',
          bbox=dict(boxstyle='round,pad=0.5', fc='#F9F9F0', ec='#BDC3C7',
                    lw=1.2, alpha=0.9))

ax_d.set_ylabel('Spearman \u03c1 (zero-shot)', fontsize=13)
ax_d.set_title('Key Insight: Kingdom-Specific Transferability', fontsize=14)
ax_d.set_ylim(0, 1.02)
ax_d.text(-0.12, 1.05, '(D)', transform=ax_d.transAxes,
          fontsize=16, fontweight='bold', va='top')

# ---------------------------------------------------------------------------
# Final layout and save
# ---------------------------------------------------------------------------
fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=4, w_pad=4)

out_base = '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/figure11_s2a_transfer'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(out_base + '.pdf', dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_base}.png')
print(f'Saved: {out_base}.pdf')
plt.close(fig)
