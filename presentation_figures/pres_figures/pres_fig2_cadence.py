#!/usr/bin/env python3
"""
Pres Fig 2: CADENCE Architecture & Performance
===============================================
3 panels:
  (A) Architecture flowchart
  (B) Performance bar chart — test Pearson r (paper values)
  (C) Training convergence — val Pearson r only (all curves go up)
"""
import sys, json, re, os
sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/pres_figures')
from pres_style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

apply_pres_style()

BASE = '/home/bcheng/sequence_optimization/FUSEMAP'

# ── PERFORMANCE DATA (paper-verified values) ────────────────────────────────
# These are test Pearson r values from the paper
datasets = ['K562', 'HepG2', 'WTC11', 'DS Dev', 'DS Hk',
            'Maize', 'Sorghum', 'Arabid.', 'Yeast']
pearson_r = [0.809, 0.786, 0.698, 0.909, 0.920,
             0.796, 0.782, 0.618, 0.958]

# Organism grouping
organism_idx = [0, 0, 0, 1, 1, 2, 2, 2, 3]
org_colors = [COLORS['human'], COLORS['drosophila'], COLORS['plant'], COLORS['yeast']]
org_names = ['Human', 'Drosophila', 'Plants', 'Yeast']

# ── PARSE TRAINING CURVES ────────────────────────────────────────────────────

def parse_training_log_loss(log_path):
    """Parse epoch-level validation loss (NLL) from a CADENCE training log."""
    epochs, val_loss = [], []
    try:
        with open(log_path) as f:
            for line in f:
                if '[VAL]   NLL:' in line:
                    m = re.search(r'\[VAL\]\s+NLL:\s*([\d.e+-]+)', line)
                    if m:
                        val_loss.append(float(m.group(1)))
                        epochs.append(len(epochs))
    except FileNotFoundError:
        print(f'  Warning: training log not found: {log_path}')
    return {'epochs': epochs, 'val_loss': val_loss}


# Parse training logs — all human cell types + all plant species
curves = {
    'K562': parse_training_log_loss(
        f'{BASE}/results/cadence_k562_v2/training.log'),
    'HepG2': parse_training_log_loss(
        f'{BASE}/results/cadence_hepg2_v2/training.log'),
    'WTC11': parse_training_log_loss(
        f'{BASE}/results/cadence_wtc11_v2/training.log'),
    'Maize': parse_training_log_loss(
        f'{BASE}/training/results/cadence_maize_v1/training.log'),
    'Sorghum': parse_training_log_loss(
        f'{BASE}/training/results/cadence_sorghum_v1/training.log'),
    'Arabidopsis': parse_training_log_loss(
        f'{BASE}/training/results/cadence_arabidopsis_v1/training.log'),
}

curve_colors = {
    'K562':        COLORS['human'],
    'HepG2':       COLORS['blue2'],
    'WTC11':       COLORS['periwinkle'],
    'Maize':       COLORS['plant'],
    'Sorghum':     '#2E7D32',       # darker green
    'Arabidopsis': '#81C784',       # lighter green
}

curve_styles = {
    'K562':        '-',
    'HepG2':       '--',
    'WTC11':       '-.',
    'Maize':       '-',
    'Sorghum':     '--',
    'Arabidopsis': '-.',
}

# ── FIGURE SETUP ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.4, 1.0], wspace=0.38)

# ==============================================================================
# Panel A: Architecture Flowchart
# ==============================================================================

ax_a = fig.add_subplot(gs[0])
ax_a.set_xlim(0, 10)
ax_a.set_ylim(-0.2, 12.5)
ax_a.invert_yaxis()
ax_a.axis('off')

arch_blocks = [
    (5.0,  0.8,  'Input [4 x L]',               COLORS['blue3'],      5.8, 0.90),
    (5.0,  2.5,  'Conv Stem\n(Conv + BN + SiLU)', COLORS['blue2'],     5.8, 1.10),
    (5.0,  4.3,  '4x ResidualConcat\n+ EffBlock', COLORS['accent2'],   5.8, 1.10),
    (5.0,  6.1,  '4x LocalBlock\n+ MaxPool',      COLORS['accent1'],   5.8, 1.10),
    (5.0,  7.9,  'MapperBlock\n+ AdaptivePool',   COLORS['periwinkle'], 5.8, 1.10),
    (5.0,  9.7,  'Per-Task Heads',                COLORS['accent3'],   5.8, 0.90),
    (5.0, 11.4,  'Activity Predictions',          COLORS['blue1'],     5.8, 0.90),
]

for x, y, text, color, w, h in arch_blocks:
    bbox = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle='round,pad=0.12',
        facecolor=color, edgecolor='white',
        linewidth=1.8, alpha=0.88, zorder=3,
    )
    ax_a.add_patch(bbox)
    ax_a.text(x, y, text, ha='center', va='center',
              fontsize=FONTS['flowchart'] - 1, fontweight='bold', color='white',
              linespacing=1.15, zorder=4)

for i in range(len(arch_blocks) - 1):
    y_start = arch_blocks[i][1] + arch_blocks[i][5] / 2 + 0.02
    y_end = arch_blocks[i + 1][1] - arch_blocks[i + 1][5] / 2 - 0.02
    ax_a.annotate('', xy=(5.0, y_end), xytext=(5.0, y_start),
                  arrowprops=dict(arrowstyle='->', color=COLORS['primary'],
                                  lw=1.8, shrinkA=0, shrinkB=0), zorder=2)

# "1.4M params" callout
callout_y = (arch_blocks[2][1] + arch_blocks[3][1]) / 2
ax_a.annotate(
    '1.4M\nparams',
    xy=(5.0 + 5.8 / 2, callout_y), xytext=(8.8, callout_y),
    fontsize=FONTS['annotation'], fontweight='bold',
    ha='center', va='center', color=COLORS['accent1'],
    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg_light'],
              edgecolor=COLORS['accent1'], linewidth=1.0),
    arrowprops=dict(arrowstyle='->', color=COLORS['accent1'],
                    lw=1.0, connectionstyle='arc3,rad=0.15'),
    zorder=5,
)

ax_a.set_title('CADENCE Architecture',
               fontsize=FONTS['subtitle'], fontweight='bold', pad=12, color=COLORS['text'])
add_panel_label(ax_a, 'A', x=-0.05)

# ==============================================================================
# Panel B: Performance Bar Chart
# ==============================================================================

ax_b = fig.add_subplot(gs[1])

bar_w = 0.58
group_gap = 0.6

# Compute x-positions with gaps between organism groups
x_pos = []
cx = 0.0
for i in range(len(datasets)):
    if i > 0 and organism_idx[i] != organism_idx[i - 1]:
        cx += group_gap
    x_pos.append(cx)
    cx += 1.0
x_pos = np.array(x_pos)

# Draw bars
for i in range(len(datasets)):
    c = org_colors[organism_idx[i]]
    ax_b.bar(x_pos[i], pearson_r[i], width=bar_w,
             color=c, edgecolor='white', linewidth=0.6, zorder=3, alpha=0.90)

    # Value label — place inside bar, offset from top to avoid overlap
    val_str = f'{pearson_r[i]:.2f}'
    lbl_fs = FONTS['bar_label'] - 1  # slightly smaller to avoid crowding
    if pearson_r[i] > 0.75:
        # Inside the bar, well below top to avoid clipping
        ax_b.text(x_pos[i], pearson_r[i] - 0.04, val_str,
                  ha='center', va='top',
                  fontsize=lbl_fs, fontweight='bold', color='black')
    else:
        ax_b.text(x_pos[i], pearson_r[i] + 0.015, val_str,
                  ha='center', va='bottom',
                  fontsize=lbl_fs, fontweight='bold', color='black')

# Group separators
group_bounds = [(0, 3), (3, 5), (5, 8), (8, 9)]
for g in range(1, len(group_bounds)):
    i_prev = group_bounds[g - 1][1] - 1
    i_curr = group_bounds[g][0]
    sep = (x_pos[i_prev] + x_pos[i_curr]) / 2
    ax_b.axvline(sep, color=COLORS['border'], ls='--', lw=0.8, alpha=0.5, zorder=1)

# Organism group labels below x-axis
for g, (s, e) in enumerate(group_bounds):
    cx_group = np.mean(x_pos[s:e])
    ax_b.text(cx_group, -0.26, org_names[g],
              fontsize=FONTS['annotation'], fontweight='bold',
              ha='center', va='top', transform=ax_b.get_xaxis_transform(),
              color=org_colors[g])

ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(datasets, fontsize=FONTS['tick'] - 1,
                     fontweight='bold', rotation=35, ha='right')
ax_b.tick_params(axis='x', length=0, pad=6)
ax_b.set_ylim(0.55, 1.05)
ax_b.set_xlim(x_pos[0] - 0.5, x_pos[-1] + 0.6)
style_axis(ax_b, ylabel='Test Pearson r')
ax_b.set_title('Single-Task Test Pearson r Across 9 Datasets',
               fontsize=FONTS['subtitle'], fontweight='bold', pad=12, color=COLORS['text'])
add_panel_label(ax_b, 'B')

# ==============================================================================
# Panel C: Training Convergence — validation loss (NLL) for all datasets
# ==============================================================================

ax_c = fig.add_subplot(gs[2])

for name, data in curves.items():
    if data['epochs']:
        ax_c.plot(data['epochs'], data['val_loss'],
                  color=curve_colors[name], lw=1.8,
                  ls=curve_styles.get(name, '-'),
                  label=name, zorder=3, alpha=0.85)

# Set a sensible y-range (cap to avoid early-epoch spikes dominating)
ax_c.set_ylim(0.2, 2.5)

ax_c.legend(fontsize=FONTS['legend'] - 2, loc='upper right',
            framealpha=0.95, edgecolor=COLORS['border'],
            borderpad=0.4, handlelength=1.5, ncol=2)

style_axis(ax_c, xlabel='Epoch', ylabel='Val Loss (NLL)')
ax_c.set_title('Validation Loss During Training',
               fontsize=FONTS['subtitle'], fontweight='bold', pad=12, color=COLORS['text'])
add_panel_label(ax_c, 'C')

# ── SAVE ─────────────────────────────────────────────────────────────────────

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    plt.tight_layout()
save_pres_fig(fig, 'pres_fig2_cadence')
print('Done.')
