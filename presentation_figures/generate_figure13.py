#!/usr/bin/env python3
"""
Figure 13: CADENCE Cross-Species Transfer Learning Results

6-panel composite figure loading ALL data from real JSON result files.
  (A) Mouse ESC data efficiency curves (full fine-tune, all sources)
  (B) S2 Drosophila data efficiency curves (full fine-tune, all sources)
  (C) Frozen vs full fine-tune comparison at 25% data
  (D) Source model heatmap (Spearman rho, 25% full fine-tune)
  (E) Scratch vs best transfer at each data fraction
  (F) Cross-kingdom model advantage over single-species

Data source: external_validation/results/comprehensive_validation/cadence/*.json
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

BASE = '/home/bcheng/sequence_optimization/FUSEMAP'
DATA_DIR = f'{BASE}/external_validation/results/comprehensive_validation/cadence'

# =============================================================================
# Load ALL transfer result JSON files
# =============================================================================
results = []
for fname in sorted(os.listdir(DATA_DIR)):
    if fname.endswith('.json'):
        with open(os.path.join(DATA_DIR, fname)) as f:
            d = json.load(f)
        results.append(d)

print(f"Loaded {len(results)} transfer experiment results")

# Organize into nested dict: [source][target][fraction][strategy] = metrics
data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for r in results:
    src = r['source_model']
    tgt = r['target_dataset']
    frac = r['data_fraction']
    strat = r['strategy']
    data[src][tgt][frac][strat] = r['test_metrics']

# Define display order and colors
source_models = [
    'cadence_k562_v2', 'cadence_hepg2_v2', 'cadence_wtc11_v2',
    'cadence_deepstarr_v2',
    'cadence_arabidopsis_v1', 'cadence_maize_v1', 'cadence_sorghum_v1',
    'cadence_yeast_v1',
    'config4_cross_kingdom_v1', 'config5_universal_no_yeast',
    'scratch',
]

source_labels = {
    'cadence_k562_v2': 'K562',
    'cadence_hepg2_v2': 'HepG2',
    'cadence_wtc11_v2': 'WTC11',
    'cadence_deepstarr_v2': 'DeepSTARR',
    'cadence_arabidopsis_v1': 'Arabidopsis',
    'cadence_maize_v1': 'Maize',
    'cadence_sorghum_v1': 'Sorghum',
    'cadence_yeast_v1': 'Yeast',
    'config4_cross_kingdom_v1': 'Cross-Kingdom',
    'config5_universal_no_yeast': 'Universal',
    'scratch': 'Scratch',
}

source_colors = {
    'cadence_k562_v2': '#4A90D9',
    'cadence_hepg2_v2': '#2166AC',
    'cadence_wtc11_v2': '#72B5E8',
    'cadence_deepstarr_v2': '#9B59B6',
    'cadence_arabidopsis_v1': '#27AE60',
    'cadence_maize_v1': '#5BB75B',
    'cadence_sorghum_v1': '#82D882',
    'cadence_yeast_v1': '#E8833A',
    'config4_cross_kingdom_v1': '#E74C3C',
    'config5_universal_no_yeast': '#C0392B',
    'scratch': '#888888',
}

fractions = [0.01, 0.05, 0.1, 0.25]
frac_labels = ['1%', '5%', '10%', '25%']

targets = ['mouse_esc', 's2_drosophila']
target_labels = {'mouse_esc': 'Mouse ESC', 's2_drosophila': 'S2 Drosophila'}

# =============================================================================
# Figure setup
# =============================================================================
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
})

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                      left=0.06, right=0.97, top=0.90, bottom=0.06)

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# =============================================================================
# Panel A: Mouse ESC data efficiency (full fine-tune)
# =============================================================================
ax_a = fig.add_subplot(gs[0, 0])
despine(ax_a)

for src in source_models:
    if src not in data or 'mouse_esc' not in data[src]:
        continue
    vals = []
    valid_fracs = []
    for f in fractions:
        strat_key = 'full_finetune' if src != 'scratch' else 'scratch'
        if f in data[src]['mouse_esc'] and strat_key in data[src]['mouse_esc'][f]:
            vals.append(data[src]['mouse_esc'][f][strat_key]['spearman_r'])
            valid_fracs.append(f)
    if vals:
        style = '--' if src == 'scratch' else '-'
        marker = 'x' if src == 'scratch' else 'o'
        lw = 2.5 if src.startswith('config') else 1.5
        ax_a.plot(valid_fracs, vals, f'{marker}{style}',
                  color=source_colors.get(src, '#333'),
                  label=source_labels.get(src, src),
                  linewidth=lw, markersize=6, alpha=0.85)

ax_a.set_xlabel('Data Fraction')
ax_a.set_ylabel('Spearman $\\rho$')
ax_a.set_xscale('log')
ax_a.set_xticks(fractions)
ax_a.set_xticklabels(frac_labels)
ax_a.grid(True, alpha=0.3, linestyle='--')
ax_a.legend(fontsize=7, ncol=2, loc='upper left', framealpha=0.9)
ax_a.set_title('(A)  Mouse ESC Transfer (Full Fine-Tune)', loc='left', pad=8)

# =============================================================================
# Panel B: S2 Drosophila data efficiency (full fine-tune)
# =============================================================================
ax_b = fig.add_subplot(gs[0, 1])
despine(ax_b)

for src in source_models:
    if src not in data or 's2_drosophila' not in data[src]:
        continue
    vals = []
    valid_fracs = []
    for f in fractions:
        strat_key = 'full_finetune' if src != 'scratch' else 'scratch'
        if f in data[src]['s2_drosophila'] and strat_key in data[src]['s2_drosophila'][f]:
            vals.append(data[src]['s2_drosophila'][f][strat_key]['spearman_r'])
            valid_fracs.append(f)
    if vals:
        style = '--' if src == 'scratch' else '-'
        marker = 'x' if src == 'scratch' else 'o'
        lw = 2.5 if src.startswith('config') else 1.5
        ax_b.plot(valid_fracs, vals, f'{marker}{style}',
                  color=source_colors.get(src, '#333'),
                  label=source_labels.get(src, src),
                  linewidth=lw, markersize=6, alpha=0.85)

ax_b.set_xlabel('Data Fraction')
ax_b.set_ylabel('Spearman $\\rho$')
ax_b.set_xscale('log')
ax_b.set_xticks(fractions)
ax_b.set_xticklabels(frac_labels)
ax_b.grid(True, alpha=0.3, linestyle='--')
ax_b.legend(fontsize=7, ncol=2, loc='upper left', framealpha=0.9)
ax_b.set_title('(B)  S2 Drosophila Transfer (Full Fine-Tune)', loc='left', pad=8)

# =============================================================================
# Panel C: Frozen vs Full Fine-Tune at 25%
# =============================================================================
ax_c = fig.add_subplot(gs[0, 2])
despine(ax_c)

# Collect frozen vs full for each source on mouse_esc at 25%
frozen_vals = []
full_vals = []
bar_labels = []
bar_colors_list = []

for src in source_models:
    if src == 'scratch':
        continue
    if src in data and 'mouse_esc' in data[src]:
        frac = 0.25
        if frac in data[src]['mouse_esc']:
            f_val = data[src]['mouse_esc'][frac].get('frozen', {}).get('spearman_r', None)
            ff_val = data[src]['mouse_esc'][frac].get('full_finetune', {}).get('spearman_r', None)
            if f_val is not None and ff_val is not None:
                frozen_vals.append(f_val)
                full_vals.append(ff_val)
                bar_labels.append(source_labels.get(src, src))
                bar_colors_list.append(source_colors.get(src, '#333'))

x = np.arange(len(bar_labels))
width = 0.35
bars1 = ax_c.bar(x - width/2, frozen_vals, width, color=[c for c in bar_colors_list],
                 alpha=0.5, edgecolor='white', label='Frozen')
bars2 = ax_c.bar(x + width/2, full_vals, width, color=bar_colors_list,
                 edgecolor='white', label='Full Fine-Tune')

ax_c.set_xticks(x)
ax_c.set_xticklabels(bar_labels, fontsize=7.5, rotation=45, ha='right')
ax_c.set_ylabel('Spearman $\\rho$ (Mouse ESC)')
ax_c.legend(fontsize=8, framealpha=0.9)
ax_c.grid(True, axis='y', alpha=0.3, linestyle='--')
ax_c.set_title('(C)  Frozen vs Fine-Tune (25% data)', loc='left', pad=8)

# =============================================================================
# Panel D: Source model heatmap at 25% full fine-tune
# =============================================================================
ax_d = fig.add_subplot(gs[1, 0])

heatmap_sources = [s for s in source_models if s != 'scratch']
heatmap_labels_y = [source_labels[s] for s in heatmap_sources]

matrix = np.full((len(heatmap_sources), len(targets)), np.nan)
for i, src in enumerate(heatmap_sources):
    for j, tgt in enumerate(targets):
        if src in data and tgt in data[src]:
            if 0.25 in data[src][tgt] and 'full_finetune' in data[src][tgt][0.25]:
                matrix[i, j] = data[src][tgt][0.25]['full_finetune']['spearman_r']

im = ax_d.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.5)
ax_d.set_xticks([0, 1])
ax_d.set_xticklabels([target_labels[t] for t in targets], fontsize=9)
ax_d.set_yticks(range(len(heatmap_labels_y)))
ax_d.set_yticklabels(heatmap_labels_y, fontsize=8)

# Annotate cells
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        if not np.isnan(val):
            color = 'white' if val < 0.15 or val > 0.4 else 'black'
            ax_d.text(j, i, f'{val:.3f}', ha='center', va='center',
                     fontsize=8, fontweight='bold', color=color)

cb = fig.colorbar(im, ax=ax_d, shrink=0.8, pad=0.02)
cb.set_label('Spearman $\\rho$', fontsize=9)
ax_d.set_title('(D)  Transfer Heatmap (25% Full Fine-Tune)', loc='left', pad=8)

# =============================================================================
# Panel E: Best transfer vs scratch at each data fraction (Mouse ESC)
# =============================================================================
ax_e = fig.add_subplot(gs[1, 1])
despine(ax_e)

scratch_vals = []
best_single_vals = []
best_multi_vals = []

for f in fractions:
    # Scratch
    sv = data.get('scratch', {}).get('mouse_esc', {}).get(f, {}).get('scratch', {}).get('spearman_r', np.nan)
    scratch_vals.append(sv)

    # Best single-species
    best_s = -999
    for src in ['cadence_k562_v2', 'cadence_hepg2_v2', 'cadence_wtc11_v2',
                'cadence_deepstarr_v2', 'cadence_arabidopsis_v1',
                'cadence_maize_v1', 'cadence_sorghum_v1', 'cadence_yeast_v1']:
        v = data.get(src, {}).get('mouse_esc', {}).get(f, {}).get('full_finetune', {}).get('spearman_r', -999)
        if v > best_s:
            best_s = v
    best_single_vals.append(best_s if best_s > -999 else np.nan)

    # Best multi-species
    best_m = -999
    for src in ['config4_cross_kingdom_v1', 'config5_universal_no_yeast']:
        v = data.get(src, {}).get('mouse_esc', {}).get(f, {}).get('full_finetune', {}).get('spearman_r', -999)
        if v > best_m:
            best_m = v
    best_multi_vals.append(best_m if best_m > -999 else np.nan)

x_pos = np.arange(len(fractions))
width = 0.25
ax_e.bar(x_pos - width, scratch_vals, width, color='#888888', edgecolor='white',
         label='Scratch', alpha=0.8)
ax_e.bar(x_pos, best_single_vals, width, color='#4A90D9', edgecolor='white',
         label='Best Single-Species', alpha=0.8)
ax_e.bar(x_pos + width, best_multi_vals, width, color='#E74C3C', edgecolor='white',
         label='Best Multi-Species', alpha=0.8)

ax_e.set_xticks(x_pos)
ax_e.set_xticklabels(frac_labels)
ax_e.set_xlabel('Data Fraction')
ax_e.set_ylabel('Spearman $\\rho$ (Mouse ESC)')
ax_e.legend(fontsize=8, framealpha=0.9)
ax_e.grid(True, axis='y', alpha=0.3, linestyle='--')
ax_e.set_title('(E)  Transfer vs Scratch (Mouse ESC)', loc='left', pad=8)

# =============================================================================
# Panel F: Cross-kingdom advantage — difference from scratch
# =============================================================================
ax_f = fig.add_subplot(gs[1, 2])
despine(ax_f)

# For each source and target at 25% full fine-tune, compute delta from scratch
deltas_mouse = {}
deltas_s2 = {}

scratch_mouse_25 = data.get('scratch', {}).get('mouse_esc', {}).get(0.25, {}).get('scratch', {}).get('spearman_r', 0)
scratch_s2_25 = data.get('scratch', {}).get('s2_drosophila', {}).get(0.25, {}).get('scratch', {}).get('spearman_r', 0)

for src in source_models:
    if src == 'scratch':
        continue
    label = source_labels[src]
    # Mouse ESC
    v = data.get(src, {}).get('mouse_esc', {}).get(0.25, {}).get('full_finetune', {}).get('spearman_r', None)
    if v is not None:
        deltas_mouse[label] = v - scratch_mouse_25
    # S2
    v = data.get(src, {}).get('s2_drosophila', {}).get(0.25, {}).get('full_finetune', {}).get('spearman_r', None)
    if v is not None:
        deltas_s2[label] = v - scratch_s2_25

# Plot grouped bar of deltas
labels_f = list(deltas_mouse.keys())
mouse_deltas = [deltas_mouse.get(l, 0) for l in labels_f]
s2_deltas = [deltas_s2.get(l, 0) for l in labels_f]

x = np.arange(len(labels_f))
width = 0.35
ax_f.bar(x - width/2, mouse_deltas, width, color='#3498DB', edgecolor='white',
         label='Mouse ESC', alpha=0.8)
ax_f.bar(x + width/2, s2_deltas, width, color='#E67E22', edgecolor='white',
         label='S2 Drosophila', alpha=0.8)

ax_f.axhline(0, color='black', linewidth=0.8, linestyle='-')
ax_f.set_xticks(x)
ax_f.set_xticklabels(labels_f, fontsize=7, rotation=45, ha='right')
ax_f.set_ylabel('$\\Delta$ Spearman $\\rho$ vs Scratch')
ax_f.legend(fontsize=8, framealpha=0.9)
ax_f.grid(True, axis='y', alpha=0.3, linestyle='--')
ax_f.set_title('(F)  Transfer Advantage over Scratch (25%)', loc='left', pad=8)

# =============================================================================
# Suptitle
# =============================================================================
fig.suptitle(
    'Figure 13. CADENCE Cross-Species Transfer Learning Results',
    fontsize=16, fontweight='bold', y=0.96
)
fig.text(
    0.5, 0.925,
    f'{len(results)} experiments | 10 source models | 2 external targets | '
    f'4 data fractions | 2 strategies',
    ha='center', fontsize=11, fontstyle='italic', color='#444444'
)

# =============================================================================
# Save
# =============================================================================
out_base = f'{BASE}/presentation_figures/figure13_transfer_learning'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(out_base + '.pdf', bbox_inches='tight', facecolor='white')
print(f"\nSaved: {out_base}.png")
print(f"Saved: {out_base}.pdf")
plt.close(fig)

# Print summary statistics
print("\n--- Summary ---")
for tgt in targets:
    print(f"\n{target_labels[tgt]} at 25% full fine-tune:")
    for src in source_models:
        v = data.get(src, {}).get(tgt, {}).get(0.25, {})
        strat = 'full_finetune' if src != 'scratch' else 'scratch'
        rho = v.get(strat, {}).get('spearman_r', None)
        if rho is not None:
            print(f"  {source_labels[src]:20s}: rho={rho:.4f}")
