#!/usr/bin/env python3
"""
Poster Fig 6: Physics Explains 4x More Activity Variance in Plants
1x2: (A) R^2 by dataset (plants vs animals) | (B) Mechanistic diagram
Data from verified experimental results.
"""
import sys
sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/poster_figures')
from poster_style import *

apply_poster_style()

# ── DATA (from verified experimental results) ────────────────────────────────
datasets = ['Maize\nLeaf', 'Sorghum\nLeaf', 'Arab.\nLeaf',
            'S2 Dev', 'S2 Hk', 'K562', 'HepG2', 'WTC11']
r2_vals = [0.464, 0.451, 0.279, 0.142, 0.116, 0.070, 0.061, 0.143]
is_plant = [True, True, True, False, False, False, False, False]
plant_mean = np.mean([0.464, 0.451, 0.279])
animal_mean = np.mean([0.142, 0.116, 0.070, 0.061, 0.143])

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0),
                          gridspec_kw={'width_ratios': [1.5, 1]})

# Panel A: R^2 bars
ax = axes[0]
colors = [COLORS['plant'] if p else COLORS['human'] for p in is_plant]
x = np.arange(len(datasets))
bars = ax.bar(x, r2_vals, color=colors, edgecolor='white', linewidth=0.5,
              width=0.65, zorder=3)

for i, v in enumerate(r2_vals):
    ax.text(i, v + 0.008, f'{v:.3f}', ha='center', va='bottom',
            fontsize=FONTS['bar_label'] - 0.5, fontweight='bold',
            color=colors[i])

# Group means
ax.axhline(plant_mean, color=COLORS['plant'], ls='--', lw=1.2, alpha=0.6,
           xmin=0, xmax=0.38)
ax.axhline(animal_mean, color=COLORS['human'], ls='--', lw=1.2, alpha=0.6,
           xmin=0.38, xmax=1.0)
ax.text(1, plant_mean + 0.01, f'Plant mean = {plant_mean:.3f}',
        fontsize=FONTS['annotation'] - 1, fontweight='bold',
        color=COLORS['plant'], va='bottom')
ax.text(6, animal_mean + 0.01, f'Animal mean = {animal_mean:.3f}',
        fontsize=FONTS['annotation'] - 1, fontweight='bold',
        color=COLORS['human'], va='bottom')

# 3.7x annotation
mid_x = 4.5
ax.annotate('', xy=(mid_x, plant_mean), xytext=(mid_x, animal_mean),
            arrowprops=dict(arrowstyle='<->', color=COLORS['text'], lw=1.5))
ax.text(mid_x + 0.3, (plant_mean + animal_mean) / 2,
        f'{plant_mean/animal_mean:.1f}x', fontsize=FONTS['annotation'] + 1,
        fontweight='bold', color=COLORS['accent'], va='center')

# Separator
ax.axvline(2.5, color=COLORS['border'], ls='--', lw=0.8, alpha=0.5)

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=FONTS['tick'] - 1)
ax.tick_params(axis='x', length=0, pad=6)
ax.set_ylim(0, 0.55)

# Group labels
ax.text(1, -0.16, 'Plants', fontsize=FONTS['annotation'], fontweight='bold',
        ha='center', va='top', transform=ax.get_xaxis_transform(),
        color=COLORS['plant'])
ax.text(5.5, -0.16, 'Animals', fontsize=FONTS['annotation'], fontweight='bold',
        ha='center', va='top', transform=ax.get_xaxis_transform(),
        color=COLORS['human'])

style_axis(ax, title='Physics-Only Variance Explained ($R^2$)',
           ylabel='$R^2$ (Physics Features)')
add_panel_label(ax, 'A')

# Panel B: Mechanistic summary (text-based diagram)
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Plants box
rect_p = plt.Rectangle((0.5, 5.5), 4, 4, fill=True,
                         facecolor=matplotlib.colors.to_rgba(COLORS['plant'], 0.1),
                         edgecolor=COLORS['plant'], linewidth=1.5, zorder=2)
ax.add_patch(rect_p)
ax.text(2.5, 9.2, 'Plants', fontsize=FONTS['title'], fontweight='bold',
        ha='center', color=COLORS['plant'])
ax.text(2.5, 8.3, 'DNA Physics', fontsize=FONTS['annotation'], fontweight='bold',
        ha='center', color=COLORS['text'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=COLORS['plant'], lw=1))
ax.annotate('', xy=(2.5, 6.8), xytext=(2.5, 7.7),
            arrowprops=dict(arrowstyle='->', color=COLORS['plant'], lw=2))
ax.text(2.5, 7.2, 'Strong', fontsize=FONTS['annotation'] - 1,
        fontweight='bold', ha='center', color=COLORS['plant'], fontstyle='italic')
ax.text(2.5, 6.2, 'Regulatory\nActivity', fontsize=FONTS['annotation'],
        fontweight='bold', ha='center', color=COLORS['text'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=COLORS['plant'], lw=1))
ax.text(2.5, 5.8, f'$R^2$ = {plant_mean:.3f}', fontsize=FONTS['annotation'],
        fontweight='bold', ha='center', color=COLORS['plant'])

# Animals box
rect_a = plt.Rectangle((5.5, 5.5), 4, 4, fill=True,
                         facecolor=matplotlib.colors.to_rgba(COLORS['human'], 0.1),
                         edgecolor=COLORS['human'], linewidth=1.5, zorder=2)
ax.add_patch(rect_a)
ax.text(7.5, 9.2, 'Animals', fontsize=FONTS['title'], fontweight='bold',
        ha='center', color=COLORS['human'])
ax.text(7.5, 8.3, 'DNA Physics', fontsize=FONTS['annotation'], fontweight='bold',
        ha='center', color=COLORS['text'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=COLORS['human'], lw=1))
ax.annotate('', xy=(7.5, 6.8), xytext=(7.5, 7.7),
            arrowprops=dict(arrowstyle='->', color=COLORS['human'], lw=1,
                            linestyle='--'))
ax.text(7.5, 7.2, 'Weak', fontsize=FONTS['annotation'] - 1,
        fontweight='bold', ha='center', color=COLORS['human'], fontstyle='italic')
ax.text(7.5, 6.2, 'Regulatory\nActivity', fontsize=FONTS['annotation'],
        fontweight='bold', ha='center', color=COLORS['text'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=COLORS['human'], lw=1))
ax.text(7.5, 5.8, f'$R^2$ = {animal_mean:.3f}', fontsize=FONTS['annotation'],
        fontweight='bold', ha='center', color=COLORS['human'])

# Central annotation
ax.text(5, 4.5, f'{plant_mean/animal_mean:.1f}x', fontsize=18,
        fontweight='bold', ha='center', va='center', color=COLORS['accent'],
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDECEA',
                  edgecolor=COLORS['accent'], lw=1.5))

# Key finding
ax.text(5, 2.5, 'Key Finding', fontsize=FONTS['annotation'] + 1,
        fontweight='bold', ha='center', color=COLORS['text'])
ax.text(5, 1.5, 'Physics features explain ~4x more\nexpression variance in plants\n'
        'than in animals across all models.',
        fontsize=FONTS['annotation'] - 0.5, ha='center', color=COLORS['text_light'],
        linespacing=1.4)

ax.set_title('Mechanistic Summary', fontsize=FONTS['title'],
             fontweight='bold', pad=8, color=COLORS['text'])
add_panel_label(ax, 'B', x=-0.02)

fig.suptitle('Fig 6.  Physics Features Explain ~4x More Activity Variance in Plants',
             fontsize=FONTS['title'], fontweight='bold', y=1.02, color=COLORS['text'])

plt.tight_layout()
save_poster_fig(fig, 'poster_fig6_physics_activity')
print('Done.')
