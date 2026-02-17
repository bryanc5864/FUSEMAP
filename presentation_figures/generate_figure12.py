#!/usr/bin/env python3
"""
Figure 12. Physics Explains 4x More Activity Variance in Plants Than Animals

4-panel composite figure:
  (A) Bar chart of R² by dataset, colored by kingdom
  (B) Side-by-side pie charts for Maize vs K562 feature contributions
  (C) Scatter of R²(physics) vs S2A transfer rho with regression line
  (D) Mechanistic summary diagram
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

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

PLANT_COLOR = '#5BB75B'
ANIMAL_COLOR = '#4A90D9'

# ---------------------------------------------------------------------------
# Data (Table 17)
# ---------------------------------------------------------------------------
datasets = ['Maize\nLeaf', 'Sorghum\nLeaf', 'Arab.\nLeaf',
            'S2 Dev', 'S2 Hk', 'K562', 'HepG2', 'WTC11']
r2_physics = [0.464, 0.451, 0.279, 0.142, 0.116, 0.070, 0.061, 0.143]
kingdoms = ['Plant', 'Plant', 'Plant',
            'Animal', 'Animal', 'Animal', 'Animal', 'Animal']
bar_colors = [PLANT_COLOR if k == 'Plant' else ANIMAL_COLOR for k in kingdoms]

plant_mean = np.mean([0.464, 0.451, 0.279])   # 0.398
animal_mean = np.mean([0.142, 0.116, 0.070, 0.061, 0.143])  # 0.1064
fold = plant_mean / animal_mean

# Panel C data
scatter_labels = ['Maize', 'Sorghum', 'Arab.', 'WTC11', 'K562', 'HepG2', 'S2']
scatter_r2   = [0.464, 0.451, 0.279, 0.143, 0.070, 0.061, 0.142]
scatter_rho  = [0.700, 0.370, 0.308, 0.184, 0.050, 0.045, -0.085]
scatter_king = ['Plant', 'Plant', 'Plant', 'Animal', 'Animal', 'Animal', 'Animal']
scatter_colors = [PLANT_COLOR if k == 'Plant' else ANIMAL_COLOR for k in scatter_king]

# Pie data
pie_cats   = ['Bending', 'Thermo', 'Entropy', 'Stiffness', 'Advanced', 'Other']
pie_colors = ['#66BB6A', '#FF7043', '#42A5F5', '#AB47BC', '#FFA726', '#BDBDBD']
maize_vals = [51, 15, 12, 10, 8, 4]
k562_vals  = [10, 20, 28, 15, 22, 5]

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                       left=0.07, right=0.95, top=0.90, bottom=0.06)

# ===== Panel A: Bar chart ===================================================
ax_a = fig.add_subplot(gs[0, 0])
x = np.arange(len(datasets))
bars = ax_a.bar(x, r2_physics, color=bar_colors, edgecolor='white',
                linewidth=0.8, width=0.70, zorder=3)

# Value labels on bars
for xi, val in zip(x, r2_physics):
    ax_a.text(xi, val + 0.008, f'{val:.3f}', ha='center', va='bottom',
              fontsize=9, fontweight='bold')

# Mean lines
ax_a.axhline(plant_mean, xmin=0.02, xmax=0.40, color=PLANT_COLOR,
             ls='--', lw=2.0, zorder=4)
ax_a.axhline(animal_mean, xmin=0.40, xmax=0.98, color=ANIMAL_COLOR,
             ls='--', lw=2.0, zorder=4)

# Mean labels
ax_a.text(0.8, plant_mean + 0.012, f'Plant mean = {plant_mean:.3f}',
          fontsize=10, fontweight='bold', color=PLANT_COLOR)
ax_a.text(4.5, animal_mean + 0.012, f'Animal mean = {animal_mean:.3f}',
          fontsize=10, fontweight='bold', color=ANIMAL_COLOR)

# Fold-difference annotation
mid_y = (plant_mean + animal_mean) / 2
ax_a.annotate('', xy=(7.6, plant_mean), xytext=(7.6, animal_mean),
              arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.8))
ax_a.text(7.75, mid_y, f'{fold:.1f}x', fontsize=13, fontweight='bold',
          color='#333333', va='center')

ax_a.set_xticks(x)
ax_a.set_xticklabels(datasets, fontsize=10, fontweight='bold')
ax_a.set_ylabel('Physics Feature R$^2$', fontsize=12)
ax_a.set_ylim(0, 0.56)
ax_a.set_title('R$^2$ of Physics Features by Dataset', fontsize=13)

# Legend
plant_patch = mpatches.Patch(color=PLANT_COLOR, label='Plant')
animal_patch = mpatches.Patch(color=ANIMAL_COLOR, label='Animal')
ax_a.legend(handles=[plant_patch, animal_patch], loc='upper right',
            framealpha=0.9, fontsize=11)

ax_a.text(-0.08, 1.06, '(A)', transform=ax_a.transAxes,
          fontsize=16, fontweight='bold', va='top')

# ===== Panel B: Pie charts ==================================================
ax_b_left  = fig.add_subplot(gs[0, 1])

# We'll split the right column cell into two halves using inset axes
ax_b_left.set_axis_off()
ax_b_left.text(-0.02, 1.06, '(B)', transform=ax_b_left.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='left')
ax_b_left.set_title('Feature Category Contributions', fontsize=13,
                     pad=18)

# Maize pie (left half)
ax_pie1 = ax_b_left.inset_axes([0.02, 0.08, 0.44, 0.80])
wedges1, texts1, autotexts1 = ax_pie1.pie(
    maize_vals, labels=pie_cats, colors=pie_colors, autopct='%1.0f%%',
    startangle=140, pctdistance=0.78, textprops={'fontsize': 9, 'fontweight': 'bold'})
for at in autotexts1:
    at.set_fontsize(8)
    at.set_fontweight('bold')
ax_pie1.set_title('Maize Leaf (Plant)', fontsize=11, fontweight='bold',
                   color=PLANT_COLOR, pad=8)

# K562 pie (right half)
ax_pie2 = ax_b_left.inset_axes([0.54, 0.08, 0.44, 0.80])
wedges2, texts2, autotexts2 = ax_pie2.pie(
    k562_vals, labels=pie_cats, colors=pie_colors, autopct='%1.0f%%',
    startangle=140, pctdistance=0.78, textprops={'fontsize': 9, 'fontweight': 'bold'})
for at in autotexts2:
    at.set_fontsize(8)
    at.set_fontweight('bold')
ax_pie2.set_title('K562 (Animal)', fontsize=11, fontweight='bold',
                   color=ANIMAL_COLOR, pad=8)

# ===== Panel C: Scatter =====================================================
ax_c = fig.add_subplot(gs[1, 0])

for i in range(len(scatter_r2)):
    ax_c.scatter(scatter_r2[i], scatter_rho[i], c=scatter_colors[i],
                 s=120, edgecolors='white', linewidth=1.2, zorder=5)

# Labels with small offsets to avoid overlap
offsets = {
    'Maize':   ( 0.012,  0.025),
    'Sorghum': ( 0.012, -0.045),
    'Arab.':   ( 0.012,  0.025),
    'WTC11':   ( 0.012,  0.025),
    'K562':    ( 0.012,  0.025),
    'HepG2':   ( 0.012, -0.040),
    'S2':      ( 0.012,  0.025),
}
for i, lbl in enumerate(scatter_labels):
    dx, dy = offsets[lbl]
    ax_c.annotate(lbl, (scatter_r2[i], scatter_rho[i]),
                  xytext=(scatter_r2[i]+dx, scatter_rho[i]+dy),
                  fontsize=10, fontweight='bold',
                  color=scatter_colors[i])

# Regression line
slope, intercept, r_val, p_val, _ = stats.linregress(scatter_r2, scatter_rho)
x_line = np.linspace(-0.02, 0.52, 100)
y_line = slope * x_line + intercept
ax_c.plot(x_line, y_line, '--', color='#555555', lw=1.8, zorder=2)
ax_c.text(0.30, -0.06, f'r = {r_val:.2f}', fontsize=12, fontweight='bold',
          color='#333333')

ax_c.set_xlabel('Physics Feature R$^2$', fontsize=12)
ax_c.set_ylabel('S2A Transfer $\\rho$', fontsize=12)
ax_c.set_title('Physics R$^2$ vs Cross-Kingdom Transfer', fontsize=13)
ax_c.set_xlim(-0.02, 0.52)
ax_c.set_ylim(-0.15, 0.80)

ax_c.legend(handles=[plant_patch, animal_patch], loc='upper left',
            framealpha=0.9, fontsize=11)

ax_c.text(-0.08, 1.06, '(C)', transform=ax_c.transAxes,
          fontsize=16, fontweight='bold', va='top')

# ===== Panel D: Mechanistic summary diagram ==================================
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_xlim(0, 10)
ax_d.set_ylim(0, 10)
ax_d.set_axis_off()

ax_d.text(0.0, 1.06, '(D)', transform=ax_d.transAxes,
          fontsize=16, fontweight='bold', va='top')
ax_d.set_title('Mechanistic Summary', fontsize=13, pad=18)

# ---------- PLANTS box (left) ----------
pcx = 2.4  # plant center x
plant_box = mpatches.FancyBboxPatch(
    (0.3, 1.0), 4.2, 8.0, boxstyle="round,pad=0.3",
    facecolor='#E8F5E9', edgecolor=PLANT_COLOR, linewidth=2.5)
ax_d.add_patch(plant_box)
ax_d.text(pcx, 8.5, 'Plants', fontsize=14, fontweight='bold',
          ha='center', color=PLANT_COLOR)

# DNA Physics box
ax_d.text(pcx, 7.4, 'DNA Physics', fontsize=11, fontweight='bold',
          ha='center', va='center',
          bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#388E3C', lw=1.5))

# Thick arrow: DNA Physics -> Regulatory Activity (strong)
ax_d.annotate('', xy=(pcx, 5.75), xytext=(pcx, 6.75),
              arrowprops=dict(arrowstyle='->', color='#2E7D32',
                              lw=5.0, shrinkA=2, shrinkB=2))
ax_d.text(pcx + 1.0, 6.25, 'Strong', fontsize=10, fontweight='bold',
          ha='left', va='center', color='#2E7D32', style='italic')

# Regulatory Activity box
ax_d.text(pcx, 5.0, 'Regulatory\nActivity', fontsize=11, fontweight='bold',
          ha='center', va='center',
          bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#388E3C', lw=1.5))

# Thin arrow: TF Binding -> Regulatory Activity (moderate)
ax_d.annotate('', xy=(pcx, 4.25), xytext=(pcx, 3.25 + 0.55),
              arrowprops=dict(arrowstyle='->', color='#81C784',
                              lw=1.8, shrinkA=2, shrinkB=2))
ax_d.text(pcx + 1.0, 3.9, 'Moderate', fontsize=9, fontweight='bold',
          ha='left', va='center', color='#81C784', style='italic')

# TF Binding box
ax_d.text(pcx, 3.0, 'TF Binding', fontsize=11, fontweight='bold',
          ha='center', va='center',
          bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#388E3C', lw=1.5))

# R2 annotation
ax_d.text(pcx, 1.6, f'R$^2$ = {plant_mean:.3f}', fontsize=13,
          fontweight='bold', ha='center', color=PLANT_COLOR,
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=PLANT_COLOR, lw=1.5))

# ---------- ANIMALS box (right) ----------
acx = 7.6  # animal center x
animal_box = mpatches.FancyBboxPatch(
    (5.5, 1.0), 4.2, 8.0, boxstyle="round,pad=0.3",
    facecolor='#E3F2FD', edgecolor=ANIMAL_COLOR, linewidth=2.5)
ax_d.add_patch(animal_box)
ax_d.text(acx, 8.5, 'Animals', fontsize=14, fontweight='bold',
          ha='center', color=ANIMAL_COLOR)

# DNA Physics box
ax_d.text(acx, 7.4, 'DNA Physics', fontsize=11, fontweight='bold',
          ha='center', va='center',
          bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#1565C0', lw=1.5))

# Thin arrow: DNA Physics -> Regulatory Activity (weak)
ax_d.annotate('', xy=(acx, 5.75), xytext=(acx, 6.75),
              arrowprops=dict(arrowstyle='->', color='#90CAF9',
                              lw=1.8, shrinkA=2, shrinkB=2))
ax_d.text(acx - 1.2, 6.25, 'Weak', fontsize=9, fontweight='bold',
          ha='right', va='center', color='#64B5F6', style='italic')

# Regulatory Activity box
ax_d.text(acx, 5.0, 'Regulatory\nActivity', fontsize=11, fontweight='bold',
          ha='center', va='center',
          bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#1565C0', lw=1.5))

# Thick arrow: TF Binding -> Regulatory Activity (strong)
ax_d.annotate('', xy=(acx, 4.25), xytext=(acx, 3.25 + 0.55),
              arrowprops=dict(arrowstyle='->', color='#1565C0',
                              lw=5.0, shrinkA=2, shrinkB=2))
ax_d.text(acx - 1.2, 3.9, 'Strong', fontsize=10, fontweight='bold',
          ha='right', va='center', color='#1565C0', style='italic')

# TF Binding box
ax_d.text(acx, 3.0, 'TF Binding', fontsize=11, fontweight='bold',
          ha='center', va='center',
          bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#1565C0', lw=1.5))

# R2 annotation
ax_d.text(acx, 1.6, f'R$^2$ = {animal_mean:.3f}', fontsize=13,
          fontweight='bold', ha='center', color=ANIMAL_COLOR,
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=ANIMAL_COLOR, lw=1.5))

# Central fold annotation between the two boxes
ax_d.annotate('', xy=(5.35, 5.0), xytext=(4.65, 5.0),
              arrowprops=dict(arrowstyle='<->', color='#333333', lw=2.0))
ax_d.text(5.0, 5.60, f'{fold:.1f}x', fontsize=14, fontweight='bold',
          ha='center', va='center', color='#D32F2F',
          bbox=dict(boxstyle='round,pad=0.2', fc='#FFF9C4', ec='#D32F2F', lw=1.5))

# ===== Suptitle ==============================================================
fig.suptitle(
    'Figure 12. Physics Explains 4x More Activity Variance in Plants Than Animals',
    fontsize=16, fontweight='bold', y=0.96)

# ===== Save ==================================================================
out_base = '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/figure12_variance_explained'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(out_base + '.pdf', bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f'Saved: {out_base}.png')
print(f'Saved: {out_base}.pdf')
print(f'Plant mean R² = {plant_mean:.3f}')
print(f'Animal mean R² = {animal_mean:.3f}')
print(f'Fold difference = {fold:.2f}x')
print(f'Scatter correlation r = {r_val:.3f}')
