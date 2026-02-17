#!/usr/bin/env python3
"""
Generate Figure 17: FUSEMAP Comprehensive Benchmark Summary

6-panel figure:
  A) CADENCE performance vs SOTA across all datasets
  B) Improvement over baseline methods
  C) Multi-task configuration comparison (breadth vs depth)
  D) Computational efficiency (performance vs training time)
  E) Transfer learning summary (best per source-target pair)
  F) Overall FUSEMAP validation statistics

Data sources:
  - results/cadence_*/final_results.json
  - training/results/cadence_*/final_results.json
  - cadence_place/config*/original_results.json
  - comparison_models/human_legnet/results/*/results.json
  - external_validation/results/
  - COMPREHENSIVE_RESULTS.md
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ============================================================================
# DATA: CADENCE Single-Task vs SOTA
# ============================================================================
datasets = [
    'K562', 'HepG2', 'WTC11', 'DeepSTARR\nDev', 'DeepSTARR\nHk',
    'Maize', 'Sorghum', 'Arabid.', 'Yeast\nDREAM'
]
cadence_r = [0.809, 0.786, 0.698, 0.909, 0.920, 0.796, 0.782, 0.618, 0.958]
sota_r    = [0.811, 0.783, 0.698, 0.708, 0.779, None,  None,  None,  None]
dataset_colors = [
    '#2196F3', '#2196F3', '#2196F3',  # Human (blue)
    '#FF9800', '#FF9800',              # Drosophila (orange)
    '#4CAF50', '#4CAF50', '#4CAF50',   # Plants (green)
    '#9C27B0',                         # Yeast (purple)
]

# ============================================================================
# DATA: Multi-task configs
# ============================================================================
config_names = ['Single-\nCell', 'Config2\n(Multi-H)', 'Config3\n(Cross-A)', 'Config4\n(Cross-K)', 'Config5\n(Universal)']
# Average Pearson r across common datasets
config_k562    = [0.809, 0.514, 0.692, 0.717, 0.624]
config_hepg2   = [0.786, 0.667, 0.689, 0.696, 0.634]
config_wtc11   = [0.698, 0.556, 0.667, 0.655, 0.585]
config_ds_dev  = [0.909, None,  0.707, 0.660, 0.637]
config_maize   = [0.796, None,  None,  0.713, 0.779]

# ============================================================================
# DATA: Transfer learning best results
# ============================================================================
transfer_pairs = [
    ('K562→S2',     0.556, 'Full FT 25%'),
    ('HepG2→S2',    0.524, 'Full FT 25%'),
    ('WTC11→S2',    0.495, 'Full FT 25%'),
    ('WTC11→Mouse', 0.281, 'Full FT 25%'),
    ('K562→Mouse',  0.216, 'Full FT 25%'),
    ('Maize→S2',    0.501, 'Full FT 25%'),
    ('Arab→S2',     0.478, 'Full FT 25%'),
    ('DS→S2',       0.976, 'Frozen 1%'),
]

# ============================================================================
# DATA: Computational efficiency
# ============================================================================
model_names_eff = ['CADENCE', 'CADENCE\nPro', 'LegNet', 'DREAM-\nRNN', 'Ridge']
perf_eff = [0.809, 0.958, 0.811, 0.708, 0.310]
time_eff = [8, 213, 8, 24, 0.02]  # hours
params_eff = [2.0, 4.5, 2.0, 5.0, 0.01]  # millions
eff_colors = ['#2196F3', '#9C27B0', '#607D8B', '#FF9800', '#795548']

# ============================================================================
# FIGURE LAYOUT
# ============================================================================
fig = plt.figure(figsize=(18, 13))
gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.35,
                       left=0.06, right=0.96, top=0.92, bottom=0.06)

panel_kw = dict(fontsize=16, fontweight='bold', va='top', ha='left')

# ---------------------------------------------------------------------------
# Panel A: CADENCE vs SOTA
# ---------------------------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 0])

x_a = np.arange(len(datasets))
bar_width = 0.35
bars_cadence = ax_a.bar(x_a - bar_width/2, cadence_r, bar_width,
                         color=dataset_colors, edgecolor='white', linewidth=0.8,
                         label='CADENCE', alpha=0.85)

# SOTA bars (where available)
sota_vals = [v if v is not None else 0 for v in sota_r]
sota_mask = [v is not None for v in sota_r]
sota_colors = ['#90A4AE' if m else 'none' for m in sota_mask]
sota_edges = ['white' if m else 'none' for m in sota_mask]
bars_sota = ax_a.bar(x_a + bar_width/2, sota_vals, bar_width,
                      color=sota_colors, edgecolor=sota_edges, linewidth=0.8,
                      label='Published SOTA', alpha=0.7)

# Value labels
for i, v in enumerate(cadence_r):
    ax_a.text(i - bar_width/2, v + 0.008, f'{v:.3f}', ha='center', va='bottom',
              fontsize=6.5, fontweight='bold', rotation=0)
for i, v in enumerate(sota_r):
    if v is not None:
        ax_a.text(i + bar_width/2, v + 0.008, f'{v:.3f}', ha='center', va='bottom',
                  fontsize=6.5, color='#546E7A', rotation=0)

# "SOTA" labels for plant/yeast
for i in [5, 6, 7, 8]:
    ax_a.text(i + bar_width/2, 0.02, 'First\nbenchmark', ha='center', va='bottom',
              fontsize=5.5, color='gray', fontstyle='italic')

ax_a.set_xticks(x_a)
ax_a.set_xticklabels(datasets, fontsize=7.5)
ax_a.set_ylabel('Test Pearson r', fontsize=10)
ax_a.set_ylim(0, 1.05)
ax_a.set_title('CADENCE vs Published SOTA', fontsize=11, fontweight='bold')
ax_a.legend(fontsize=8, loc='upper left')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.text(-0.12, 1.05, 'A', transform=ax_a.transAxes, **panel_kw)

# ---------------------------------------------------------------------------
# Panel B: Improvement Over Baseline
# ---------------------------------------------------------------------------
ax_b = fig.add_subplot(gs[0, 1])

improvement_datasets = ['K562\nvs LegNet', 'HepG2\nvs LegNet', 'WTC11\nvs LegNet',
                         'DS Dev\nvs DREAM-RNN', 'DS Hk\nvs DREAM-RNN']
improvement_pct = [
    (0.809 - 0.811) / 0.811 * 100,
    (0.786 - 0.783) / 0.783 * 100,
    (0.698 - 0.698) / 0.698 * 100,
    (0.909 - 0.708) / 0.708 * 100,
    (0.920 - 0.779) / 0.779 * 100,
]
imp_colors = ['#F44336' if v < 0 else '#4CAF50' for v in improvement_pct]

bars_b = ax_b.barh(range(len(improvement_datasets)), improvement_pct,
                    color=imp_colors, edgecolor='white', alpha=0.85, height=0.6)
for i, v in enumerate(improvement_pct):
    offset = 0.5 if v >= 0 else -0.5
    ha = 'left' if v >= 0 else 'right'
    ax_b.text(v + offset, i, f'{v:+.1f}%', ha=ha, va='center', fontsize=9, fontweight='bold')

ax_b.axvline(x=0, color='black', linewidth=0.8)
ax_b.set_yticks(range(len(improvement_datasets)))
ax_b.set_yticklabels(improvement_datasets, fontsize=8)
ax_b.set_xlabel('Improvement (%)', fontsize=10)
ax_b.set_title('Improvement Over Baselines', fontsize=11, fontweight='bold')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.set_xlim(-5, 35)
ax_b.text(-0.18, 1.05, 'B', transform=ax_b.transAxes, **panel_kw)

# ---------------------------------------------------------------------------
# Panel C: Multi-Task Configuration Comparison
# ---------------------------------------------------------------------------
ax_c = fig.add_subplot(gs[0, 2])

x_c = np.arange(len(config_names))
w = 0.15
datasets_mt = ['K562', 'HepG2', 'WTC11', 'DS Dev', 'Maize']
mt_data = [config_k562, config_hepg2, config_wtc11, config_ds_dev, config_maize]
mt_colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#8BC34A']

for j, (data, color, name) in enumerate(zip(mt_data, mt_colors, datasets_mt)):
    vals = [v if v is not None else 0 for v in data]
    alpha_vals = [0.85 if v is not None else 0 for v in data]
    offset = (j - 2) * w
    bars = ax_c.bar(x_c + offset, vals, w, color=color, label=name,
                     edgecolor='white', linewidth=0.5, alpha=0.85)
    # Make None bars invisible
    for k, v in enumerate(data):
        if v is None:
            bars[k].set_alpha(0)

ax_c.set_xticks(x_c)
ax_c.set_xticklabels(config_names, fontsize=7.5)
ax_c.set_ylabel('Test Pearson r', fontsize=10)
ax_c.set_ylim(0, 1.0)
ax_c.set_title('Multi-Task Config Comparison\n(Breadth vs Depth)', fontsize=11, fontweight='bold')
ax_c.legend(fontsize=7, loc='upper right', ncol=2)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.text(-0.12, 1.05, 'C', transform=ax_c.transAxes, **panel_kw)

# ---------------------------------------------------------------------------
# Panel D: Computational Efficiency
# ---------------------------------------------------------------------------
ax_d = fig.add_subplot(gs[1, 0])

for i, (name, perf, time, params, color) in enumerate(
        zip(model_names_eff, perf_eff, time_eff, params_eff, eff_colors)):
    ax_d.scatter(time + 0.1, perf, s=params * 80, c=color, alpha=0.7,
                 edgecolors='white', linewidth=1, zorder=3)
    # Label
    offset_x = 1.1 if time > 1 else 0.005
    ax_d.text(time + offset_x, perf + 0.015, name, fontsize=7.5,
              ha='left' if time > 0.1 else 'center', va='bottom')

ax_d.set_xscale('log')
ax_d.set_xlabel('Training Time (hours, log scale)', fontsize=10)
ax_d.set_ylabel('Test Pearson r', fontsize=10)
ax_d.set_title('Performance vs Computational Cost', fontsize=11, fontweight='bold')
ax_d.set_xlim(0.01, 500)
ax_d.set_ylim(0.2, 1.05)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
ax_d.axhline(y=0.8, color='gray', linestyle=':', alpha=0.3)
ax_d.text(0.02, 0.81, 'Strong prediction\nthreshold', fontsize=7, color='gray')

# Size legend
for s_val, s_label in [(0.8, '1M'), (3.2, '4M')]:
    ax_d.scatter([], [], s=s_val * 80, c='gray', alpha=0.5, label=f'{s_label} params')
ax_d.legend(fontsize=7, loc='lower right', title='Model Size', title_fontsize=7)
ax_d.text(-0.12, 1.05, 'D', transform=ax_d.transAxes, **panel_kw)

# ---------------------------------------------------------------------------
# Panel E: Transfer Learning Summary
# ---------------------------------------------------------------------------
ax_e = fig.add_subplot(gs[1, 1])

transfer_names = [p[0] for p in transfer_pairs]
transfer_vals = [p[1] for p in transfer_pairs]
transfer_notes = [p[2] for p in transfer_pairs]

# Color by transfer type
t_colors = []
for name in transfer_names:
    if 'DS→' in name:
        t_colors.append('#4CAF50')   # Same species
    elif '→Mouse' in name:
        t_colors.append('#F44336')   # Hard transfer
    elif 'Arab' in name or 'Maize' in name:
        t_colors.append('#8BC34A')   # Cross-kingdom
    else:
        t_colors.append('#2196F3')   # Cross-species

bars_e = ax_e.barh(range(len(transfer_names)), transfer_vals,
                    color=t_colors, edgecolor='white', alpha=0.85, height=0.6)

for i, (v, note) in enumerate(zip(transfer_vals, transfer_notes)):
    ax_e.text(v + 0.01, i, f'{v:.3f} ({note})', ha='left', va='center',
              fontsize=7, fontweight='bold')

ax_e.set_yticks(range(len(transfer_names)))
ax_e.set_yticklabels(transfer_names, fontsize=8)
ax_e.set_xlabel('Best Spearman ρ', fontsize=10)
ax_e.set_title('Best Transfer Performance\nper Source→Target', fontsize=11, fontweight='bold')
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)
ax_e.set_xlim(0, 1.15)
ax_e.axvline(x=0.5, color='gray', linestyle=':', alpha=0.3)
ax_e.text(-0.18, 1.05, 'E', transform=ax_e.transAxes, **panel_kw)

# ---------------------------------------------------------------------------
# Panel F: Overall FUSEMAP Validation Statistics
# ---------------------------------------------------------------------------
ax_f = fig.add_subplot(gs[1, 2])
ax_f.axis('off')

stats = [
    ('FUSEMAP Validation Summary', '', '', True),
    ('', '', '', False),
    ('Total Experiments', '486+', '', False),
    ('Species Covered', '9', '(4 kingdoms)', False),
    ('Experimental Systems', '7', '', False),
    ('', '', '', False),
    ('CADENCE Single-Task', '', '', True),
    ('  Best Pearson r', '0.958', '(Yeast DREAM)', False),
    ('  Mean across datasets', '0.797', '', False),
    ('  vs SOTA Human', '0.0%', '(matched)', False),
    ('  vs SOTA Drosophila', '+23.2%', '(new SOTA)', False),
    ('', '', '', False),
    ('Transfer Learning', '', '', True),
    ('  Best cross-species', 'ρ=0.556', '(K562→S2)', False),
    ('  Best cross-kingdom', 'ρ=0.501', '(Maize→S2)', False),
    ('  S2A zero-shot plants', 'ρ=0.700', '(Maize LOO)', False),
    ('', '', '', False),
    ('Physics Models', '', '', True),
    ('  TileFormer R²', '>0.95', '(10,000× speedup)', False),
    ('  PhysInformer transfer', 'r=0.847', '(within-human)', False),
    ('  Bending universality', '-6%', '(across kingdoms)', False),
]

y = 0.98
for label, value, note, is_header in stats:
    if is_header:
        ax_f.text(0.02, y, label, fontsize=9, fontweight='bold',
                  transform=ax_f.transAxes, color='#1565C0')
        ax_f.plot([0.02, 0.98], [y - 0.01, y - 0.01], color='#BBDEFB',
                  linewidth=1, transform=ax_f.transAxes, clip_on=False)
    elif label:
        ax_f.text(0.02, y, label, fontsize=8, transform=ax_f.transAxes)
        ax_f.text(0.55, y, value, fontsize=8.5, fontweight='bold',
                  transform=ax_f.transAxes, color='#1B5E20')
        if note:
            ax_f.text(0.75, y, note, fontsize=7, transform=ax_f.transAxes,
                      color='gray', fontstyle='italic')
    y -= 0.045

ax_f.set_title('Overall Statistics', fontsize=11, fontweight='bold', pad=10)
ax_f.text(-0.05, 1.05, 'F', transform=ax_f.transAxes, **panel_kw)

# ---------------------------------------------------------------------------
# Main title
# ---------------------------------------------------------------------------
fig.suptitle('Figure 17: FUSEMAP Comprehensive Benchmark Summary',
             fontsize=14, fontweight='bold', y=0.97)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_base = '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/figure17_benchmark_summary'
fig.savefig(f'{out_base}.png', dpi=200, bbox_inches='tight')
fig.savefig(f'{out_base}.pdf', bbox_inches='tight')
print(f"Saved: {out_base}.png and .pdf")
plt.close(fig)
