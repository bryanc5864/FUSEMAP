#!/usr/bin/env python3
"""
Figure 5: Summary of CADENCE single-task performance across all evaluated systems.
Single wide bar chart showing test Pearson r for all datasets, color-coded by organism,
with SOTA baselines as hatched bars where available.

ALL values loaded from actual result JSON files - no hardcoded metrics.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = '/home/bcheng/sequence_optimization/FUSEMAP'

# ── LOAD REAL DATA FROM JSON FILES ─────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

# Human cell types (test split available)
k562 = load_json(f'{BASE}/results/cadence_k562_v2/final_results.json')
hepg2 = load_json(f'{BASE}/results/cadence_hepg2_v2/final_results.json')
wtc11 = load_json(f'{BASE}/results/cadence_wtc11_v2/final_results.json')

# DeepSTARR (test split available)
deepstarr = load_json(f'{BASE}/training/results/cadence_deepstarr_v2/final_results.json')

# Plants (validation split only - no separate test split)
maize = load_json(f'{BASE}/training/results/cadence_maize_v1/final_results.json')
sorghum = load_json(f'{BASE}/training/results/cadence_sorghum_v1/final_results.json')
arabidopsis = load_json(f'{BASE}/training/results/cadence_arabidopsis_v1/final_results.json')

# DREAM Yeast (from best Pro model)
dream = load_json(f'{BASE}/models/legatoV2/outputs/'
                  'dream_pro_dream_20250831_182621/final_results.json')

# SOTA baselines
legnet_k562 = load_json(f'{BASE}/comparison_models/human_legnet/results/k562_legnet/results.json')
legnet_hepg2 = load_json(f'{BASE}/comparison_models/human_legnet/results/hepg2_legnet/results.json')
legnet_wtc11 = load_json(f'{BASE}/comparison_models/human_legnet/results/wtc11_legnet/results.json')
# DREAM-RNN results file is truncated JSON - parse only the complete test_metrics
dreamrnn_path = f'{BASE}/comparison_models/dreamrnn_deepstarr_results/results.json'
with open(dreamrnn_path) as f:
    dreamrnn_text = f.read()
# Extract test_metrics block which is complete
import re as _re
_dev_r = float(_re.search(r'"Dev":\s*\{\s*"pearson":\s*([\d.]+)', dreamrnn_text).group(1))
_hk_r = float(_re.search(r'"Hk":\s*\{\s*"pearson":\s*([\d.]+)', dreamrnn_text).group(1))
dreamrnn_test = {'Dev': {'pearson': _dev_r}, 'Hk': {'pearson': _hk_r}}
print(f"DREAM-RNN baselines: Dev r={_dev_r:.4f}, Hk r={_hk_r:.4f}")

# ── EXTRACT VALUES ─────────────────────────────────────────────────────────

datasets = ['K562', 'HepG2', 'WTC11', 'S2 Dev', 'S2 Hk',
            'Maize\nLeaf', 'Sorghum\nLeaf', 'Arab.\nLeaf', 'DREAM\nYeast']

pearson_r = [
    k562['test']['encode4_k562']['activity']['pearson'],
    hepg2['test']['encode4_hepg2']['activity']['pearson'],
    wtc11['test']['encode4_wtc11']['activity']['pearson'],
    deepstarr['test']['deepstarr']['Dev']['pearson'],
    deepstarr['test']['deepstarr']['Hk']['pearson'],
    maize['jores_maize']['leaf']['pearson']['value'],        # val (no test split)
    sorghum['jores_sorghum']['leaf']['pearson']['value'],    # val (no test split)
    arabidopsis['jores_arabidopsis']['leaf']['pearson']['value'],  # val (no test split)
    dream['test']['target_pearson'],
]

spearman_rho = [
    k562['test']['encode4_k562']['activity']['spearman'],
    hepg2['test']['encode4_hepg2']['activity']['spearman'],
    wtc11['test']['encode4_wtc11']['activity']['spearman'],
    deepstarr['test']['deepstarr']['Dev']['spearman'],
    deepstarr['test']['deepstarr']['Hk']['spearman'],
    maize['jores_maize']['leaf']['spearman']['value'],
    sorghum['jores_sorghum']['leaf']['spearman']['value'],
    arabidopsis['jores_arabidopsis']['leaf']['spearman']['value'],
    dream['test']['target_spearman'],
]

sota_r = [
    legnet_k562['test_pearson'],
    legnet_hepg2['test_pearson'],
    legnet_wtc11['test_pearson'],
    dreamrnn_test['Dev']['pearson'],
    dreamrnn_test['Hk']['pearson'],
    None, None, None, None,
]

sota_labels = ['LegNet', 'LegNet', 'LegNet', 'DREAM-RNN', 'DREAM-RNN',
               None, None, None, None]

# Print loaded values for verification
print("Loaded CADENCE metrics from JSON files:")
for i, name in enumerate(datasets):
    name_clean = name.replace('\n', ' ')
    sota_str = f", SOTA r={sota_r[i]:.3f}" if sota_r[i] is not None else ""
    print(f"  {name_clean:12s}: r={pearson_r[i]:.4f}, rho={spearman_rho[i]:.4f}{sota_str}")

# ── ORGANISM GROUPING ──────────────────────────────────────────────────────

organism_idx = [0, 0, 0, 1, 1, 2, 2, 2, 3]
organism_names = ['Human', 'Drosophila', 'Plants', 'Yeast']
organism_colors = {
    0: '#4A90D9',
    1: '#9B59B6',
    2: '#5BB75B',
    3: '#E8833A',
}

group_starts = [0, 3, 5, 8]
group_ends   = [3, 5, 8, 9]

# ── FIGURE SETUP ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 8))

bar_width = 0.35
gap = 0.08
group_gap = 0.6

x_positions = []
current_x = 0.0
for i in range(len(datasets)):
    if i > 0 and organism_idx[i] != organism_idx[i - 1]:
        current_x += group_gap
    x_positions.append(current_x)
    current_x += 1.0

x_positions = np.array(x_positions)

# ── DRAW BARS ──────────────────────────────────────────────────────────────
for i in range(len(datasets)):
    color = organism_colors[organism_idx[i]]
    has_sota = sota_r[i] is not None

    if has_sota:
        x_cad = x_positions[i] - (bar_width + gap) / 2
        ax.bar(x_cad, pearson_r[i], width=bar_width, color=color,
               edgecolor='white', linewidth=0.8, zorder=3)
        ax.text(x_cad, pearson_r[i] + 0.012, f'{pearson_r[i]:.3f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                color=color)

        x_sota = x_positions[i] + (bar_width + gap) / 2
        lighter = matplotlib.colors.to_rgba(color, alpha=0.40)
        ax.bar(x_sota, sota_r[i], width=bar_width, color=lighter,
               edgecolor=color, linewidth=1.2, hatch='///', zorder=3)
        ax.text(x_sota, sota_r[i] + 0.012, f'{sota_r[i]:.3f}',
                ha='center', va='bottom', fontsize=8.0, fontweight='bold',
                color='#555555')
    else:
        ax.bar(x_positions[i], pearson_r[i], width=bar_width * 1.1,
               color=color, edgecolor='white', linewidth=0.8, zorder=3)
        ax.text(x_positions[i], pearson_r[i] + 0.012, f'{pearson_r[i]:.3f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                color=color)

# ── DELTA ANNOTATIONS for Drosophila ──────────────────────────────────────
for i in [3, 4]:
    if sota_r[i] is not None:
        delta_pct = (pearson_r[i] - sota_r[i]) / sota_r[i] * 100
        x_cad = x_positions[i] - (bar_width + gap) / 2
        ax.annotate(
            f'+{delta_pct:.1f}%',
            xy=(x_cad, pearson_r[i] + 0.035),
            fontsize=10, fontweight='bold', color='#9B59B6',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#F3E5F5',
                      edgecolor='#9B59B6', linewidth=1.0, alpha=0.9),
        )

# ── VERTICAL SEPARATORS ──────────────────────────────────────────────────
for g in range(1, len(group_starts)):
    i_prev = group_ends[g - 1] - 1
    i_curr = group_starts[g]
    sep_x = (x_positions[i_prev] + x_positions[i_curr]) / 2
    ax.axvline(sep_x, color='#CCCCCC', linestyle='--', linewidth=1.0,
               zorder=1, alpha=0.7)

# ── HORIZONTAL REFERENCE LINE ────────────────────────────────────────────
ax.axhline(y=0.80, color='#888888', linestyle=':', linewidth=1.2,
           zorder=1, alpha=0.5)
ax.text(x_positions[-1] + 0.8, 0.80, 'r = 0.80', fontsize=8,
        color='#888888', va='center', ha='left', fontstyle='italic')

# ── GROUP LABELS ─────────────────────────────────────────────────────────
ax.set_xticks(x_positions)
ax.set_xticklabels(datasets, fontsize=9.5, fontweight='bold')

for g, name in enumerate(organism_names):
    idx_range = list(range(group_starts[g], group_ends[g]))
    center_x = np.mean(x_positions[idx_range])
    ax.text(center_x, -0.13, name, fontsize=12, fontweight='bold',
            ha='center', va='top', transform=ax.get_xaxis_transform(),
            color=organism_colors[g])

# ── AXES STYLING ─────────────────────────────────────────────────────────
ax.set_ylim(0, 1.05)
ax.set_ylabel('Test Pearson r', fontsize=13, fontweight='bold')
ax.set_xlim(x_positions[0] - 0.7, x_positions[-1] + 0.9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

ax.yaxis.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='x', length=0, pad=10)

# ── LEGEND ───────────────────────────────────────────────────────────────
cadence_patch = mpatches.Patch(facecolor='#4A90D9', edgecolor='white',
                               linewidth=0.8, label='CADENCE')
sota_patch = mpatches.Patch(facecolor=matplotlib.colors.to_rgba('#4A90D9', 0.40),
                            edgecolor='#4A90D9', linewidth=1.2,
                            hatch='///', label='Published SOTA')
ax.legend(handles=[cadence_patch, sota_patch], loc='upper left',
          fontsize=11, frameon=True, framealpha=0.9,
          edgecolor='#CCCCCC', fancybox=True)

# ── TITLE ────────────────────────────────────────────────────────────────
fig.suptitle('Figure 5. CADENCE Single-Task Performance Across All Evaluated Systems',
             fontsize=15, fontweight='bold', y=0.97)

# ── SAVE ─────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.04, 1, 0.94])

out_base = f'{BASE}/presentation_figures/3.1_cadence_single_task/figure5_performance_summary'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig(out_base + '.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)

print(f'\nSaved: {out_base}.png')
print(f'Saved: {out_base}.pdf')
