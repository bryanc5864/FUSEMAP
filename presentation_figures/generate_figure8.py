#!/usr/bin/env python3
"""
Figure 8. Universal Physics Hierarchy: Feature Transfer Across Evolutionary Distance
4-panel composite figure for the paper.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# =============================================================================
# DATA (Table 12 - Feature Category Transfer)
# =============================================================================
categories = ['Bending', 'Advanced\nStructural', 'Entropy/\nComplexity', 'Stiffness/\nMechanics', 'PWM/TF\nBinding']
categories_short = ['Bending', 'Adv. Structural', 'Entropy/Complex.', 'Stiffness/Mech.', 'PWM/TF Binding']
n_features = [43, 38, 62, 36, 149]
within_human = [0.98, 0.94, 0.84, 0.43, 0.94]
cross_species = [0.94, 0.91, 0.78, 0.46, 0.11]
cross_kingdom = [0.92, 0.91, 0.68, 0.45, 0.03]
delta_pct = [-6, -3, -19, +5, -97]

scenarios = ['Within Human', 'Cross-Species\n(\u2192Drosophila)', 'Cross-Kingdom\n(\u2192Plants)']
scenarios_short = ['Within\nHuman', 'Cross-Species', 'Cross-Kingdom']
overall_r = [0.85, 0.73, 0.67]

# Colors
cat_colors = {
    'Bending': '#27AE60',
    'Advanced\nStructural': '#3498DB',
    'Entropy/\nComplexity': '#F39C12',
    'Stiffness/\nMechanics': '#E74C3C',
    'PWM/TF\nBinding': '#8E44AD',
}
scenario_colors = ['#4A90D9', '#E8833A', '#5BB75B']

# =============================================================================
# FIGURE SETUP
# =============================================================================
fig = plt.figure(figsize=(18, 14), facecolor='white')
fig.suptitle(
    'Figure 8. Universal Physics Hierarchy: Feature Transfer Across Evolutionary Distance',
    fontsize=16, fontweight='bold', y=0.97
)

gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30,
                       left=0.07, right=0.95, top=0.91, bottom=0.06)

# =============================================================================
# Helper: remove top/right spines, add grid
# =============================================================================
def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)


# =============================================================================
# PANEL A  -  Overall transfer degradation bar chart
# =============================================================================
ax_a = fig.add_subplot(gs[0, 0])
style_ax(ax_a)

bars_a = ax_a.bar(
    np.arange(len(scenarios)), overall_r,
    color=scenario_colors, edgecolor='white', linewidth=1.2, width=0.55,
    zorder=3
)

# Value labels
for bar, val in zip(bars_a, overall_r):
    ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
              f'r = {val:.2f}', ha='center', va='bottom',
              fontsize=13, fontweight='bold')

# Degradation arrows between bars
for i in range(len(overall_r) - 1):
    drop = overall_r[i] - overall_r[i + 1]
    pct = drop / overall_r[i] * 100
    mid_x = (i + i + 1) / 2
    mid_y = max(overall_r[i], overall_r[i + 1]) + 0.06
    ax_a.annotate('', xy=(i + 1, overall_r[i + 1] + 0.03),
                  xytext=(i, overall_r[i] + 0.03),
                  arrowprops=dict(arrowstyle='->', color='#555555',
                                  lw=1.8, connectionstyle='arc3,rad=-0.15'))
    ax_a.text(mid_x, mid_y, f'\u2013{pct:.0f}%',
              ha='center', va='bottom', fontsize=10, color='#555555',
              fontweight='bold')

ax_a.set_xticks(np.arange(len(scenarios)))
ax_a.set_xticklabels(scenarios, fontsize=11)
ax_a.set_ylabel('Pearson r (overall)', fontsize=12, fontweight='bold')
ax_a.set_ylim(0, 1.08)
ax_a.set_title('(A)  Overall Transfer Degradation', fontsize=13,
               fontweight='bold', loc='left', pad=10)

# =============================================================================
# PANEL B  -  Grouped bar chart: 5 categories x 3 scenarios
# =============================================================================
ax_b = fig.add_subplot(gs[0, 1])
style_ax(ax_b)

x = np.arange(len(categories))
bar_w = 0.24
data_matrix = [within_human, cross_species, cross_kingdom]
scenario_labels = ['Within Human', 'Cross-Species', 'Cross-Kingdom']

for s_idx, (vals, color, label) in enumerate(zip(data_matrix, scenario_colors, scenario_labels)):
    # Compute alpha intensities: full for Within, lighter for farther transfer
    alphas = [1.0, 0.75, 0.55]
    offset = (s_idx - 1) * bar_w
    bars = ax_b.bar(x + offset, vals, bar_w,
                    color=color, alpha=alphas[s_idx],
                    edgecolor='white', linewidth=0.8,
                    label=label, zorder=3)
    # Small value labels on top of each bar
    for bar, val in zip(bars, vals):
        fontsize = 7.5 if val > 0.15 else 7
        ax_b.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.01,
                  f'{val:.2f}', ha='center', va='bottom',
                  fontsize=fontsize, color='#333333')

# Highlight PWM collapse with red box
pwm_idx = 4
rect_x = pwm_idx - 0.42
rect = mpatches.FancyBboxPatch(
    (rect_x, -0.02), 0.84, 1.02,
    boxstyle='round,pad=0.02', linewidth=2.2,
    edgecolor='#C0392B', facecolor='none', linestyle='--', zorder=5
)
ax_b.add_patch(rect)
ax_b.annotate('PWM collapse\n(\u221297%)', xy=(pwm_idx, 0.03),
              xytext=(pwm_idx - 0.1, -0.18),
              fontsize=9.5, fontweight='bold', color='#C0392B',
              ha='center', va='top',
              arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5))

ax_b.set_xticks(x)
ax_b.set_xticklabels(categories, fontsize=9.5)
ax_b.set_ylabel('Pearson r', fontsize=12, fontweight='bold')
ax_b.set_ylim(-0.22, 1.12)
ax_b.set_title('(B)  Feature Category Transfer by Scenario', fontsize=13,
               fontweight='bold', loc='left', pad=10)
ax_b.legend(fontsize=9.5, loc='upper right', framealpha=0.9,
            edgecolor='#CCCCCC')

# =============================================================================
# PANEL C  -  Heatmap: categories (rows) x scenarios (cols)
# =============================================================================
ax_c = fig.add_subplot(gs[1, 0])

heatmap_data = np.array([
    within_human,
    cross_species,
    cross_kingdom,
]).T  # shape (5 categories, 3 scenarios)

# Custom green-yellow-red colormap
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    'GreenRed', ['#C0392B', '#E74C3C', '#F5B041', '#58D68D', '#27AE60'], N=256
)

im = ax_c.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=1.0)

# Text annotations
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = heatmap_data[i, j]
        text_color = 'white' if val < 0.35 else 'black'
        fontw = 'bold'
        ax_c.text(j, i, f'{val:.2f}', ha='center', va='center',
                  fontsize=13, fontweight=fontw, color=text_color)

ax_c.set_xticks(np.arange(3))
ax_c.set_xticklabels(['Within\nHuman', 'Cross-Species\n(\u2192Drosophila)',
                       'Cross-Kingdom\n(\u2192Plants)'],
                      fontsize=10)
ax_c.set_yticks(np.arange(5))
ax_c.set_yticklabels(categories_short, fontsize=10)
ax_c.set_title('(C)  Transfer Heatmap (Pearson r)', fontsize=13,
               fontweight='bold', loc='left', pad=10)

# Colorbar
cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04, shrink=0.85)
cbar.set_label('Pearson r', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=9)

# Highlight PWM row
rect_hm = mpatches.FancyBboxPatch(
    (-0.48, 3.55), 2.96, 0.9,
    boxstyle='round,pad=0.05', linewidth=2.5,
    edgecolor='#C0392B', facecolor='none', linestyle='-', zorder=5
)
ax_c.add_patch(rect_hm)

# =============================================================================
# PANEL D  -  Summary insight panel
# =============================================================================
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_xlim(0, 10)
ax_d.set_ylim(0, 10)
ax_d.axis('off')

ax_d.set_title('(D)  Key Insights: Physics vs. Regulatory Transfer',
               fontsize=13, fontweight='bold', loc='left', pad=10)

# Background box
bg_rect = mpatches.FancyBboxPatch(
    (0.3, 0.3), 9.4, 9.2,
    boxstyle='round,pad=0.3', linewidth=1.5,
    edgecolor='#BDC3C7', facecolor='#F8F9FA', zorder=0
)
ax_d.add_patch(bg_rect)

# --- Section 1: Physics features ---
ax_d.text(0.8, 9.0, 'PHYSICS-BASED FEATURES', fontsize=12,
          fontweight='bold', color='#27AE60', va='top')
ax_d.text(0.8, 8.35, '(Bending, Structural, Entropy, Stiffness)',
          fontsize=9.5, color='#555555', va='top', style='italic')

# Individual category results
y_pos = 7.6
results = [
    ('Bending', '#27AE60', '\u22126%', 'Near-perfect conservation (0.98 \u2192 0.92)'),
    ('Adv. Structural', '#3498DB', '\u22123%', 'Highly conserved (0.94 \u2192 0.91)'),
    ('Entropy/Complexity', '#F39C12', '\u221219%', 'Moderate degradation (0.84 \u2192 0.68)'),
    ('Stiffness/Mechanics', '#E74C3C', '+5%', 'Slight improvement cross-species (0.43 \u2192 0.45)'),
]

for name, color, delta, desc in results:
    # Colored dot
    ax_d.plot(1.1, y_pos, 'o', color=color, markersize=9, zorder=5)
    ax_d.text(1.6, y_pos, f'{name}:', fontsize=10, fontweight='bold',
              va='center', color='#2C3E50')
    ax_d.text(4.2, y_pos, delta, fontsize=10, fontweight='bold',
              va='center', color=color,
              bbox=dict(boxstyle='round,pad=0.2', facecolor=color,
                        alpha=0.15, edgecolor=color, linewidth=0.8))
    ax_d.text(5.2, y_pos, desc, fontsize=8.8, va='center', color='#555555')
    y_pos -= 0.65

# Divider line
ax_d.plot([0.8, 9.2], [y_pos + 0.15, y_pos + 0.15], '-',
          color='#BDC3C7', linewidth=1.2)
y_pos -= 0.35

# --- Section 2: Regulatory features ---
ax_d.text(0.8, y_pos, 'REGULATORY FEATURES', fontsize=12,
          fontweight='bold', color='#C0392B', va='top')
y_pos -= 0.6
ax_d.plot(1.1, y_pos, 'o', color='#8E44AD', markersize=9, zorder=5)
ax_d.text(1.6, y_pos, 'PWM/TF Binding:', fontsize=10, fontweight='bold',
          va='center', color='#2C3E50')
ax_d.text(4.8, y_pos, '\u221297%', fontsize=11, fontweight='bold',
          va='center', color='#C0392B',
          bbox=dict(boxstyle='round,pad=0.25', facecolor='#FDEDEC',
                    edgecolor='#C0392B', linewidth=1.5))
ax_d.text(6.0, y_pos, 'Near-total collapse (0.94 \u2192 0.03)',
          fontsize=9, va='center', color='#C0392B', fontweight='bold')

# Divider
y_pos -= 0.7
ax_d.plot([0.8, 9.2], [y_pos + 0.15, y_pos + 0.15], '-',
          color='#BDC3C7', linewidth=1.2)
y_pos -= 0.35

# --- Section 3: Key takeaway box ---
takeaway_rect = mpatches.FancyBboxPatch(
    (0.8, y_pos - 1.65), 8.4, 1.85,
    boxstyle='round,pad=0.2', linewidth=2,
    edgecolor='#2C3E50', facecolor='#EBF5FB', zorder=2
)
ax_d.add_patch(takeaway_rect)

ax_d.text(5.0, y_pos - 0.15, 'KEY INSIGHT', fontsize=11,
          fontweight='bold', color='#2C3E50', ha='center', va='top')
ax_d.text(5.0, y_pos - 0.65,
          'DNA physical properties are conserved across\n'
          'evolutionary distance, while regulatory grammar\n'
          'is species-specific.',
          fontsize=10, color='#2C3E50', ha='center', va='top',
          linespacing=1.4)

# --- Bottom annotation: the highlight comparison ---
y_bot = 0.7
ax_d.text(5.0, y_bot,
          'Bending: \u22126%  vs.  PWM: \u221297%',
          fontsize=14, fontweight='bold', ha='center', va='center',
          color='#2C3E50',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEF9E7',
                    edgecolor='#F39C12', linewidth=2))

# =============================================================================
# SAVE
# =============================================================================
out_base = '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/figure8_feature_category_transfer'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(out_base + '.pdf', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out_base}.png')
print(f'Saved: {out_base}.pdf')
