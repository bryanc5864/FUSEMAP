#!/usr/bin/env python3
"""
Poster Fig 5: PhysInformer Transfer Matrix + Physics Universality
1x2: (A) Transfer heatmap | (B) Physics vs PWM bar chart
Data from Figure 9 and Figure 8 generation scripts (hardcoded from actual experiments).
"""
import sys
sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/poster_figures')
from poster_style import *

apply_poster_style()

# ── DATA (from verified experimental results) ────────────────────────────────
# Transfer matrix (Pearson r, from PhysInformer zero-shot experiments)
sources = ['K562', 'HepG2', 'WTC11']
targets = ['K562', 'HepG2', 'WTC11', 'S2', 'Maize', 'Sorghum', 'Arab.']

# Matrix values [source x target], NaN = self or not tested
matrix = np.array([
    [np.nan, 0.847, 0.839, 0.729, 0.680, 0.679, 0.656],
    [0.657, np.nan, np.nan, 0.464, np.nan, np.nan, np.nan],
    [0.832, 0.829, np.nan, 0.649, np.nan, np.nan, np.nan],
])

# Block means
within_human = 0.801
cross_species = 0.614
cross_kingdom = 0.672

# Physics feature transfer by category (from Figure 8 experiments)
categories = ['Bending', 'Adv.\nStruct.', 'Entropy', 'Stiffness', 'PWM/TF']
within_r  = [0.98, 0.94, 0.84, 0.43, 0.94]
cross_sp  = [0.94, 0.91, 0.78, 0.46, 0.11]
cross_kg  = [0.92, 0.91, 0.68, 0.45, 0.03]

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0),
                          gridspec_kw={'width_ratios': [1.3, 1]})

# Panel A: Transfer heatmap
ax = axes[0]
# Create display matrix with text
display_matrix = matrix.copy()
im = ax.imshow(display_matrix, cmap='RdYlGn', vmin=0.3, vmax=0.9,
               aspect='auto')

for i in range(len(sources)):
    for j in range(len(targets)):
        v = matrix[i, j]
        if np.isnan(v):
            txt = 'self' if sources[i] == targets[j] else 'N/A'
            c = COLORS['text_light']
        else:
            txt = f'{v:.3f}'
            c = 'white' if v < 0.5 else COLORS['text']
        ax.text(j, i, txt, ha='center', va='center',
                fontsize=FONTS['annotation'] - 0.5, fontweight='bold', color=c)

ax.set_xticks(range(len(targets)))
ax.set_xticklabels(targets, fontsize=FONTS['tick'] - 1, rotation=30, ha='right')
ax.set_yticks(range(len(sources)))
ax.set_yticklabels(sources, fontsize=FONTS['tick'])
ax.set_xlabel('Target Dataset', fontsize=FONTS['axis_label'], fontweight='bold')
ax.set_ylabel('Source Dataset', fontsize=FONTS['axis_label'], fontweight='bold')

# Vertical separators for groups
ax.axvline(2.5, color='white', lw=2)
ax.axvline(3.5, color='white', lw=2)

cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('Pearson r', fontsize=8)
cb.ax.tick_params(labelsize=7)

# Block means annotation
ax.text(0.02, -0.18, f'Within-Human: r={within_human:.3f}  |  '
        f'Cross-Species: r={cross_species:.3f}  |  '
        f'Cross-Kingdom: r={cross_kingdom:.3f}',
        transform=ax.transAxes, fontsize=FONTS['caption'],
        color=COLORS['text_light'], fontstyle='italic')

ax.set_title('PhysInformer Transfer Matrix', fontsize=FONTS['title'],
             fontweight='bold', pad=8)
add_panel_label(ax, 'A')

# Panel B: Physics vs PWM transfer
ax = axes[1]
x = np.arange(len(categories))
w = 0.25
ax.bar(x - w, within_r, w, color=COLORS['within'], edgecolor='white',
       linewidth=0.5, label='Within-Human', zorder=3)
ax.bar(x, cross_sp, w, color=COLORS['cross_sp'], edgecolor='white',
       linewidth=0.5, label='Cross-Species', zorder=3)
ax.bar(x + w, cross_kg, w, color=COLORS['cross_kg'], edgecolor='white',
       linewidth=0.5, label='Cross-Kingdom', zorder=3)

# Highlight PWM collapse
ax.annotate('PWM\ncollapse\n-97%', xy=(4 + w, cross_kg[-1]),
            xytext=(4 + w + 0.3, 0.25),
            fontsize=FONTS['annotation'] - 1, fontweight='bold',
            color=COLORS['accent'], ha='center',
            arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.2))

# Bar labels for cross-kingdom
for i, v in enumerate(cross_kg):
    ax.text(i + w, v + 0.02, f'{v:.2f}', ha='center', va='bottom',
            fontsize=FONTS['bar_label'] - 1, color=COLORS['cross_kg'])

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=FONTS['tick'] - 0.5)
ax.set_ylim(0, 1.1)
style_axis(ax, title='Feature Transfer by Category', ylabel='Pearson r')
ax.legend(fontsize=FONTS['legend'] - 1, loc='upper right', framealpha=0.8,
          ncol=1)
add_panel_label(ax, 'B')

fig.suptitle('Fig 5.  Physics Features Transfer Across Species; PWM Does Not',
             fontsize=FONTS['title'], fontweight='bold', y=1.02, color=COLORS['text'])

plt.tight_layout()
save_poster_fig(fig, 'poster_fig5_physinformer_transfer')
print('Done.')
