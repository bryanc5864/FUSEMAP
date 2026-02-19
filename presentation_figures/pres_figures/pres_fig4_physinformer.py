#!/usr/bin/env python3
"""
Pres Fig 4: PhysInformer — Biophysical Feature Prediction & Transfer
3 panels: (A) Architecture flowchart, (B) Transfer heatmap, (C) Feature universality bars

Improvements over v2:
  - Panel A: Added "521 Biophysical Features" annotation to clarify families vs total count
  - Panel A: Wider spacing for property boxes to eliminate overlap
  - Panel B: Informative title "Cross-Species Transfer Performance (Pearson r)"
  - Panel B: Better spacing for group brackets, summary text repositioned
  - Panel C: Clearer title "Feature Category Transferability (K562 -> Others)"
  - Panel C: PWM collapse annotation positioned to avoid all bar/label overlap
  - All panel labels at consistent vertical position
  - General overlap and clipping fixes throughout
"""
import sys
sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/pres_figures')
from pres_style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as mtransforms

apply_pres_style()

# ── DATA (hardcoded, verified) ──────────────────────────────────────────────
sources = ['K562', 'HepG2', 'WTC11']
targets = ['K562', 'HepG2', 'WTC11', 'S2', 'Maize', 'Sorghum', 'Arab.']

matrix = np.array([
    [np.nan, 0.847, 0.839, 0.729, 0.680, 0.679, 0.656],
    [0.657, np.nan, 0.647, 0.464, 0.444, 0.434, 0.420],
    [0.832, 0.829, np.nan, 0.649, 0.382, 0.336, 0.106],
])

within_human = 0.775
cross_species = 0.614
cross_kingdom = 0.460

categories = ['Bend.', 'Adv.\nStr.', 'Entr.', 'Stiff.', 'PWM']
within_r  = [0.98, 0.94, 0.84, 0.43, 0.94]
cross_sp  = [0.94, 0.91, 0.78, 0.46, 0.11]
cross_kg  = [0.92, 0.91, 0.68, 0.45, 0.03]

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 6.4))
gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.3, 1.05],
                      wspace=0.32, left=0.03, right=0.97,
                      top=0.84, bottom=0.10)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel A: Architecture flowchart (improved property head fan-out)
# ═══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0])
ax_a.set_xlim(-1.5, 11.5)
ax_a.set_ylim(-0.2, 15.0)
ax_a.axis('off')

# Main pipeline blocks — evenly spaced vertically
blocks = [
    (5, 1.0,  'Sequence Input',         COLORS['blue3']),
    (5, 2.8,  'PWMConvStem',            COLORS['blue2']),
    (5, 4.6,  'State Space Model',      COLORS['accent2']),
    (5, 6.4,  'FeaturePyramid',         COLORS['accent1']),
    (5, 8.2,  'PhysicsRouters',         COLORS['periwinkle']),
    (5, 10.0, 'PropertyHeads',          COLORS['accent3']),
]

# Draw blocks
box_h = 1.1
box_w = 5.2
for x, y, text, color in blocks:
    bbox = FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                           boxstyle='round,pad=0.15', facecolor=color,
                           edgecolor='white', linewidth=1.5, alpha=0.88)
    ax_a.add_patch(bbox)
    ax_a.text(x, y, text, ha='center', va='center', fontsize=FONTS['flowchart'],
              fontweight='bold', color='white', linespacing=1.15)

# Connecting arrows between sequential blocks
for i in range(len(blocks) - 1):
    y_start = blocks[i][1] + box_h / 2
    y_end = blocks[i + 1][1] - box_h / 2
    ax_a.annotate('', xy=(5, y_end), xytext=(5, y_start),
                  arrowprops=dict(arrowstyle='->', color=COLORS['primary'],
                                  lw=1.8, shrinkA=1, shrinkB=1))

# Property head fan-out — 5 output boxes below PropertyHeads
properties = ['Bend', 'Therm', 'Entr', 'Stiff', 'PWM']
prop_colors = [COLORS['bending'], COLORS['thermo'], COLORS['entropy'],
               COLORS['stiffness'], COLORS['pwm']]

prop_y = 12.2
prop_box_w = 1.5
prop_box_h = 0.80
n_props = len(properties)
# Center the 5 boxes, use wider spacing to avoid all overlap
prop_spacing = 2.3
total_span = (n_props - 1) * prop_spacing
x_start = 5 - total_span / 2

for i, (prop, pc) in enumerate(zip(properties, prop_colors)):
    px = x_start + i * prop_spacing
    # Draw a rounded box for each property output
    pbox = FancyBboxPatch((px - prop_box_w / 2, prop_y - prop_box_h / 2),
                           prop_box_w, prop_box_h,
                           boxstyle='round,pad=0.10', facecolor='white',
                           edgecolor=pc, linewidth=1.8)
    ax_a.add_patch(pbox)
    ax_a.text(px, prop_y, prop, fontsize=FONTS['flowchart'] - 1.5, ha='center',
              va='center', fontweight='bold', color=pc)
    # Arrow from PropertyHeads block down to each property box
    ax_a.annotate('', xy=(px, prop_y - prop_box_h / 2),
                  xytext=(5, blocks[-1][1] + box_h / 2),
                  arrowprops=dict(arrowstyle='->', color=pc, lw=1.2,
                                  alpha=0.7, shrinkA=2, shrinkB=2,
                                  connectionstyle='arc3,rad=0'))

# ── CRITICAL: Annotation showing these are FAMILIES, not individual features ──
# Place a callout box above the fan-out boxes, centered
annotation_y = 13.8
ax_a.text(5, annotation_y, '5 Feature Families  (521 total features)',
          ha='center', va='center', fontsize=FONTS['annotation'] - 1.5,
          fontweight='bold', color=COLORS['primary'],
          bbox=dict(boxstyle='round,pad=0.30', facecolor=COLORS['bg_light'],
                    edgecolor=COLORS['border'], linewidth=1.3, alpha=0.95))

ax_a.set_title('PhysInformer Architecture',
               fontsize=FONTS['subtitle'], fontweight='bold',
               pad=12, color=COLORS['text'])
add_panel_label(ax_a, 'A', x=-0.02)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel B: Transfer heatmap (improved readability)
# ═══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[1])

# Custom colormap: light lavender -> purple -> deep blue (better contrast)
heatmap_cmap = LinearSegmentedColormap.from_list(
    'pres_transfer', [
        (0.0,  '#F3E5F5'),   # very light lavender
        (0.3,  '#CE93D8'),   # lavender
        (0.55, '#7B1FA2'),   # purple
        (0.75, '#1565C0'),   # blue
        (1.0,  '#0D47A1'),   # deep blue
    ])

display_matrix = matrix.copy()
im = ax_b.imshow(display_matrix, cmap=heatmap_cmap, vmin=0.05, vmax=0.90,
                 aspect='auto')

# Gray out self-transfer (NaN) cells
for i in range(len(sources)):
    for j in range(len(targets)):
        if np.isnan(matrix[i, j]):
            ax_b.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                           facecolor='#EEEEEE', edgecolor='white', lw=1.5,
                           zorder=2))

# Cell text — use .2f for cleaner reading, bold
for i in range(len(sources)):
    for j in range(len(targets)):
        v = matrix[i, j]
        if np.isnan(v):
            txt = 'self'
            c = COLORS['text_light']
            fw = 'normal'
        else:
            txt = f'{v:.2f}'
            c = 'white' if v > 0.45 else COLORS['text']
            fw = 'bold'
        ax_b.text(j, i, txt, ha='center', va='center',
                  fontsize=FONTS['annotation'] - 0.5, fontweight=fw, color=c)

# Axis ticks
ax_b.set_xticks(range(len(targets)))
ax_b.set_xticklabels(targets, fontsize=FONTS['tick'] - 1, rotation=40, ha='right')
ax_b.set_yticks(range(len(sources)))
ax_b.set_yticklabels(sources, fontsize=FONTS['tick'])

# Group separators — white vertical lines between human / insect / plant
ax_b.axvline(2.5, color='white', lw=2.5)
ax_b.axvline(3.5, color='white', lw=2.5)

# Group bracket labels above the heatmap, below the title
for span, label, lbl_color in [
    ((0, 2),  'Human',  COLORS['human']),
    ((3, 3),  'Insect', COLORS['drosophila']),
    ((4, 6),  'Plant',  COLORS['plant']),
]:
    x_mid = (span[0] + span[1]) / 2
    x_left = span[0] - 0.4
    x_right = span[1] + 0.4
    # Bracket line above heatmap
    bracket_y = -0.72
    ax_b.plot([x_left, x_right], [bracket_y, bracket_y],
              color=lbl_color, lw=2.0, clip_on=False, solid_capstyle='round')
    ax_b.text(x_mid, bracket_y - 0.20, label, ha='center', va='top',
              fontsize=FONTS['annotation'] - 1, fontweight='bold',
              color=lbl_color, clip_on=False)

# Colorbar
cb = fig.colorbar(im, ax=ax_b, shrink=0.72, pad=0.03)
cb.set_label('Pearson r', fontsize=FONTS['annotation'], fontweight='bold')
cb.ax.tick_params(labelsize=FONTS['tick'] - 1)
cb.outline.set_linewidth(0.5)

# Summary annotation below heatmap
summary_text = (f'Within-Human: r = {within_human:.3f}   |   '
                f'Cross-Species: r = {cross_species:.3f}   |   '
                f'Cross-Kingdom: r = {cross_kingdom:.3f}')
ax_b.text(0.5, -0.32, summary_text, transform=ax_b.transAxes,
          fontsize=FONTS['caption'] - 1, color=COLORS['text_light'],
          fontstyle='italic', ha='center', va='top')

# Apply axis styling — informative title
style_axis(ax_b, xlabel='Target Cell Type / Species', ylabel='Source', grid_y=False)
ax_b.set_title('Cross-Species Transfer Performance\n(Pearson r)',
               fontsize=FONTS['subtitle'], fontweight='bold',
               pad=12, color=COLORS['text'])
add_panel_label(ax_b, 'B', x=-0.10)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel C: Feature universality grouped bars (PWM collapse fix)
# ═══════════════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[2])
x = np.arange(len(categories))
w = 0.24

bars_within = ax_c.bar(x - w, within_r, w, color=COLORS['within'],
                        edgecolor='white', linewidth=0.5,
                        label='Within-Human', zorder=3)
bars_cross  = ax_c.bar(x, cross_sp, w, color=COLORS['cross_sp'],
                        edgecolor='white', linewidth=0.5,
                        label='Cross-Species', zorder=3)
bars_plant  = ax_c.bar(x + w, cross_kg, w, color=COLORS['cross_kg'],
                        edgecolor='white', linewidth=0.5,
                        label='Cross-Kingdom', zorder=3)

# Cross-kingdom value labels on top of bars — selective to avoid crowding
# Only label Bend, Adv.Str, Entr (skip Stiff and PWM — handled by annotation / too close)
for i, v in enumerate(cross_kg):
    if i <= 2:  # Bend, Adv.Str, Entr only
        ax_c.text(i + w, v + 0.02, f'{v:.2f}', ha='center', va='bottom',
                  fontsize=FONTS['bar_label'] - 1, color=COLORS['cross_kg'],
                  fontweight='bold', clip_on=False)

# PWM collapse annotation — positioned between Stiff and PWM groups, below bars
# at y=0.18 between x=3.5 (between Stiff at x=3 and PWM at x=4)
pwm_bar_x = 4 + w         # x center of the PWM cross-kingdom bar
pwm_bar_top = cross_kg[-1] # top of the bar (0.03)

ax_c.annotate(
    'PWM collapse\n  -97%',
    xy=(pwm_bar_x, pwm_bar_top + 0.01),
    xytext=(3.5, 0.22),
    fontsize=FONTS['annotation'] - 1.5,
    fontweight='bold',
    color=COLORS['red'],
    ha='center', va='center',
    bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFEBEE',
              edgecolor=COLORS['red'], linewidth=1.2, alpha=0.9),
    arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5,
                    connectionstyle='arc3,rad=-0.15',
                    shrinkA=4, shrinkB=2),
    zorder=5,
)

ax_c.set_xticks(x)
ax_c.set_xticklabels(categories, fontsize=FONTS['tick'] - 1)
ax_c.set_ylim(-0.02, 1.12)

# Informative title for Panel C — two lines for clarity
style_axis(ax_c, ylabel='Pearson r')
ax_c.set_title('Feature Category Transferability\n(K562 $\\rightarrow$ Other Species)',
               fontsize=FONTS['subtitle'], fontweight='bold',
               pad=12, color=COLORS['text'])
ax_c.legend(fontsize=FONTS['legend'] - 2, loc='upper right', framealpha=0.92,
            edgecolor=COLORS['border'], borderpad=0.4, labelspacing=0.35)
add_panel_label(ax_c, 'C', x=-0.12)

# ── SAVE ─────────────────────────────────────────────────────────────────────
save_pres_fig(fig, 'pres_fig4_physinformer')
print('Done.')
