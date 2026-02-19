#!/usr/bin/env python3
"""
Pres Fig 6: S2A Transfer Learning
==================================
3 panels:
  (A) Methodology flowchart — S2A pipeline from DNA to predictions
  (B) Per-dataset zero-shot Spearman rho (bar chart)
  (C) Transfer scenario comparison (bar chart)
"""
import sys
import csv

sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/pres_figures')
from pres_style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

apply_pres_style()

BASE = '/home/bcheng/sequence_optimization/FUSEMAP'

# ── LOAD S2A DATA ────────────────────────────────────────────────────────────
# Per-dataset results
per_dataset = []
with open(f'{BASE}/results/s2a/full_evaluation/per_dataset_results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        per_dataset.append(row)

print("Per-dataset results:")
for row in per_dataset:
    print(f"  {row['dataset_name']}: zeroshot_spearman={row['zeroshot_spearman']}")

# Transfer comparison results
comparison = []
with open(f'{BASE}/results/s2a/comparison/transfer_comparison.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        comparison.append(row)

print("\nComparison scenarios:")
for row in comparison:
    print(f"  {row['scenario']}: zeroshot_spearman={row['zeroshot_spearman']}")

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 6.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.3, 0.9], wspace=0.32)

# ══════════════════════════════════════════════════════════════════════════════
# Panel A: S2A Methodology Flowchart
# ══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0])
ax_a.set_xlim(-0.5, 12.0)
ax_a.set_ylim(-0.3, 10.5)
ax_a.axis('off')


def draw_box(ax, cx, cy, w, h, text, fc, ec='white', fontsize=None,
             text_color='white', alpha=0.9, pad=0.15):
    """Draw a FancyBboxPatch centered at (cx, cy) with text inside."""
    if fontsize is None:
        fontsize = FONTS['flowchart']
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f'round,pad={pad}', facecolor=fc,
        edgecolor=ec, linewidth=1.5, alpha=alpha, zorder=3
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color, zorder=4)
    return box


def draw_arrow(ax, x_start, y_start, x_end, y_end, color=COLORS['text'],
               lw=1.8, style='->', shrinkA=4, shrinkB=4):
    """Draw a simple arrow between two points."""
    arrow = FancyArrowPatch(
        (x_start, y_start), (x_end, y_end),
        arrowstyle=style, color=color, lw=lw,
        shrinkA=shrinkA, shrinkB=shrinkB, zorder=2,
        mutation_scale=12
    )
    ax.add_patch(arrow)
    return arrow


# --- Title above flowchart ---
ax_a.set_title('S2A Pipeline', fontsize=FONTS['subtitle'],
               fontweight='bold', pad=12, color=COLORS['text'])

# --- Main pipeline boxes (vertical flow) ---
box_x = 4.5
box_w = 5.0
box_h = 1.15

# Box 1: DNA Sequences
y1 = 8.7
draw_box(ax_a, box_x, y1, box_w, box_h,
         'DNA Sequences\n(any species)', COLORS['primary'],
         ec=COLORS['border'], fontsize=FONTS['flowchart'] + 1)

# Box 2: 268 Physics Features
y2 = 6.95
draw_box(ax_a, box_x, y2, box_w, box_h,
         '268 Physics Features\n(biophysical properties)',
         COLORS['accent1'], ec=COLORS['border'], fontsize=FONTS['flowchart'])

# Box 3: Z-Score Normalization
y3 = 5.2
draw_box(ax_a, box_x, y3, box_w, box_h,
         'Z-Score Normalize\n(across datasets)',
         COLORS['blue1'], ec=COLORS['border'], fontsize=FONTS['flowchart'] + 1)

# Box 4: Ridge Regression (leave-one-out)
y4 = 3.45
draw_box(ax_a, box_x, y4, box_w, box_h,
         'Ridge Regression\n(leave-one-out)',
         COLORS['blue2'], ec=COLORS['border'], fontsize=FONTS['flowchart'] + 1)

# Box 5: Predictions
y5 = 1.7
draw_box(ax_a, box_x, y5, box_w, box_h,
         'Zero-Shot Prediction\n(held-out dataset)',
         COLORS['plant'], ec=COLORS['border'], fontsize=FONTS['flowchart'] + 1)

# --- Arrows between main boxes ---
arrow_color = COLORS['text_light']
for ya, yb in [(y1, y2), (y2, y3), (y3, y4), (y4, y5)]:
    draw_arrow(ax_a, box_x, ya - box_h / 2, box_x, yb + box_h / 2,
               color=arrow_color, shrinkA=6, shrinkB=6)

# --- Side annotation: PWM excluded ---
pwm_x = 9.5
pwm_y = 6.95
ax_a.text(pwm_x, pwm_y, 'PWM excluded\n(species-specific)',
          fontsize=FONTS['flowchart_detail'] + 1, fontweight='bold', ha='center', va='center',
          color=COLORS['red'], style='italic',
          bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFEBEE',
                    edgecolor=COLORS['red'], alpha=0.9, linewidth=1.2),
          zorder=5)
# Small arrow from side note pointing toward physics features box
draw_arrow(ax_a, pwm_x - 0.8, pwm_y, box_x + box_w / 2 + 0.15, y2,
           color=COLORS['red'], lw=1.4, style='->', shrinkA=20, shrinkB=8)


add_panel_label(ax_a, 'A', x=-0.03)

# ══════════════════════════════════════════════════════════════════════════════
# Panel B: Per-dataset zero-shot Spearman rho
# ══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[1])

dataset_order = ['K562', 'HepG2', 'WTC11', 'S2_dev', 'arabidopsis_leaf',
                 'sorghum_leaf', 'maize_leaf']
display_names = ['K562', 'HepG2', 'WTC11', 'S2 Dev', 'Arabid.', 'Sorghum', 'Maize']

# Map dataset names to organism group for coloring
org_map = {
    'K562': 'human', 'HepG2': 'human', 'WTC11': 'human',
    'S2_dev': 'drosophila',
    'arabidopsis_leaf': 'plant', 'sorghum_leaf': 'plant', 'maize_leaf': 'plant',
}
org_color_map = {
    'human': COLORS['human'],
    'drosophila': COLORS['drosophila'],
    'plant': COLORS['plant'],
}

# Build ordered values and colors
spearman_vals = []
bar_colors = []
for dname in dataset_order:
    for row in per_dataset:
        if row['dataset_name'] == dname:
            spearman_vals.append(float(row['zeroshot_spearman']))
            bar_colors.append(org_color_map[org_map[dname]])
            break

x_b = np.arange(len(display_names))
bars_b = ax_b.bar(x_b, spearman_vals, color=bar_colors, edgecolor='white',
                  linewidth=0.5, width=0.6, zorder=3)

# Value labels above/below bars
for i, v in enumerate(spearman_vals):
    if v >= 0:
        y_pos = v + 0.015
        va = 'bottom'
    else:
        y_pos = v - 0.015
        va = 'top'
    ax_b.text(i, y_pos, f'{v:.3f}', ha='center', va=va,
              fontsize=FONTS['bar_label'], fontweight='bold', color=bar_colors[i])

# Baseline at zero
ax_b.axhline(0, color=COLORS['text'], lw=0.8, zorder=1)

ax_b.set_xticks(x_b)
ax_b.set_xticklabels(display_names, fontsize=FONTS['tick'],
                     rotation=20, ha='right')
ax_b.set_ylim(-0.15, 0.45)

# Legend for organism groups
legend_handles = [
    mpatches.Patch(facecolor=COLORS['human'], label='Human'),
    mpatches.Patch(facecolor=COLORS['drosophila'], label='Drosophila'),
    mpatches.Patch(facecolor=COLORS['plant'], label='Plant'),
]
ax_b.legend(handles=legend_handles, loc='upper left', framealpha=0.9,
            fontsize=FONTS['legend'] - 1, edgecolor=COLORS['border'])

style_axis(ax_b, ylabel='Spearman $\\rho$')
ax_b.set_title('S2A Zero-Shot Spearman $\\rho$ by Dataset', fontsize=FONTS['subtitle'],
               fontweight='bold', pad=12, color=COLORS['text'])
add_panel_label(ax_b, 'B', x=-0.10)

# ══════════════════════════════════════════════════════════════════════════════
# Panel C: Transfer scenario comparison
# ══════════════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[2])

# Define scenario display order, labels, and colors
scenario_spec = {
    'within_human':    ('Within\nHuman',     COLORS['human']),
    'within_plant':    ('Within\nPlant',     COLORS['plant']),
    'animal_to_plant': ('Animal \u2192\nPlant',  COLORS['accent2']),
    'plant_to_animal': ('Plant \u2192\nAnimal',  COLORS['accent3']),
}
scenario_order = ['within_human', 'within_plant', 'animal_to_plant', 'plant_to_animal']

scenarios = []
scenario_rho = []
scenario_colors = []
for sname in scenario_order:
    label, color = scenario_spec[sname]
    for row in comparison:
        if row['scenario'] == sname:
            scenarios.append(label)
            scenario_rho.append(float(row['zeroshot_spearman']))
            scenario_colors.append(color)
            break

x_c = np.arange(len(scenarios))
bars_c = ax_c.bar(x_c, scenario_rho, color=scenario_colors, edgecolor='white',
                  linewidth=0.5, width=0.55, zorder=3)

# Value labels
for i, v in enumerate(scenario_rho):
    if v >= 0:
        y_pos = v + 0.02
        va = 'bottom'
    else:
        y_pos = v - 0.02
        va = 'top'
    ax_c.text(i, y_pos, f'{v:.3f}', ha='center', va=va,
              fontsize=FONTS['bar_label'], fontweight='bold',
              color=scenario_colors[i])

ax_c.axhline(0, color=COLORS['text'], lw=0.8, zorder=1)
ax_c.set_xticks(x_c)
ax_c.set_xticklabels(scenarios, fontsize=FONTS['tick'])
ax_c.set_ylim(-0.45, 0.85)

style_axis(ax_c, ylabel='Spearman $\\rho$')
ax_c.set_title('S2A Transfer Comparison', fontsize=FONTS['subtitle'],
               fontweight='bold', pad=12, color=COLORS['text'])
add_panel_label(ax_c, 'C', x=-0.10)

# ── SAVE ─────────────────────────────────────────────────────────────────────
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.90, wspace=0.32)
save_pres_fig(fig, 'pres_fig6_s2a')
print('Done.')
