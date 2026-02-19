#!/usr/bin/env python3
"""
Poster Fig 7: CADENCE Cross-Species Transfer Learning
1x2: (A) Transfer heatmap (25% fine-tune) | (B) Key scaling curves
ALL data from actual JSON result files.
"""
import sys, json, os
sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/poster_figures')
from poster_style import *

apply_poster_style()

BASE = '/home/bcheng/sequence_optimization/FUSEMAP'
results_dir = f'{BASE}/external_validation/results/comprehensive_validation/cadence'

# ── LOAD ALL TRANSFER RESULTS ────────────────────────────────────────────────
data = {}
for fname in sorted(os.listdir(results_dir)):
    if not fname.endswith('.json'):
        continue
    with open(os.path.join(results_dir, fname)) as f:
        res = json.load(f)

    src = res.get('source_model', '')
    tgt = res.get('target_dataset', '')
    frac = res.get('data_fraction', 0)
    strat = res.get('strategy', '')
    metrics = res.get('test_metrics', {})

    key = (src, tgt, frac, strat)
    data[key] = metrics

# Source models
all_sources = sorted(set(k[0] for k in data.keys()))
targets = sorted(set(k[1] for k in data.keys()))
fractions = sorted(set(k[2] for k in data.keys()))
strategies = sorted(set(k[3] for k in data.keys()))

print(f"Loaded {len(data)} experiments")
print(f"Sources: {all_sources}")
print(f"Targets: {targets}")
print(f"Fractions: {fractions}")
print(f"Strategies: {strategies}")

# ── BUILD HEATMAP DATA (25% fine-tune) ───────────────────────────────────────
# For each source→target at 25% full_finetune
heatmap_sources = [s for s in all_sources if s != 'scratch']
heatmap_vals = {}
for src in heatmap_sources:
    for tgt in targets:
        key = (src, tgt, 0.25, 'full_finetune')
        if key in data:
            spearman = data[key].get('spearman_rho', data[key].get('spearman', 0))
            heatmap_vals[(src, tgt)] = spearman

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5),
                          gridspec_kw={'width_ratios': [1.2, 1]})

# Panel A: Transfer heatmap
ax = axes[0]
if heatmap_vals:
    # Build matrix
    src_list = sorted(set(k[0] for k in heatmap_vals.keys()))
    tgt_list = sorted(set(k[1] for k in heatmap_vals.keys()))

    mat = np.full((len(src_list), len(tgt_list)), np.nan)
    for i, src in enumerate(src_list):
        for j, tgt in enumerate(tgt_list):
            if (src, tgt) in heatmap_vals:
                mat[i, j] = heatmap_vals[(src, tgt)]

    im = ax.imshow(mat, cmap='YlGn', vmin=0, vmax=1.0, aspect='auto')

    for i in range(len(src_list)):
        for j in range(len(tgt_list)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            c = 'white' if v > 0.6 else COLORS['text']
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=FONTS['annotation'] - 1.5, fontweight='bold', color=c)

    # Clean source names
    src_clean = [s.replace('cadence_', '').replace('_v2', '').replace('_v1', '').replace('_all', '')
                 for s in src_list]
    tgt_clean = [t.replace('mouse_esc', 'Mouse ESC').replace('s2_drosophila', 'S2 Dros.')
                 for t in tgt_list]

    ax.set_xticks(range(len(tgt_list)))
    ax.set_xticklabels(tgt_clean, fontsize=FONTS['tick'] - 0.5, rotation=0)
    ax.set_yticks(range(len(src_list)))
    ax.set_yticklabels(src_clean, fontsize=FONTS['tick'] - 1)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('Spearman $\\rho$', fontsize=8)
    cb.ax.tick_params(labelsize=7)

style_axis(ax, title='Transfer Heatmap (25% Fine-Tune)', grid_y=False)
ax.set_xlabel('Target', fontsize=FONTS['axis_label'], fontweight='bold')
ax.set_ylabel('Source Model', fontsize=FONTS['axis_label'], fontweight='bold')
add_panel_label(ax, 'A')

# Panel B: Scaling curves for best sources
ax = axes[1]
# Get S2 Drosophila transfer curves (most interesting)
s2_tgt = [t for t in targets if 'drosophila' in t.lower() or 's2' in t.lower()]
mouse_tgt = [t for t in targets if 'mouse' in t.lower()]

# Plot scaling for a few key sources → S2
if s2_tgt:
    tgt_name = s2_tgt[0]
    key_sources = ['cadence_k562_v2', 'cadence_deepstarr_v2', 'cadence_maize_v1', 'scratch']
    source_colors = {
        'cadence_k562_v2': COLORS['human'],
        'cadence_deepstarr_v2': COLORS['drosophila'],
        'cadence_maize_v1': COLORS['plant'],
        'scratch': COLORS['secondary'],
    }
    source_labels = {
        'cadence_k562_v2': 'K562',
        'cadence_deepstarr_v2': 'DeepSTARR',
        'cadence_maize_v1': 'Maize',
        'scratch': 'Scratch',
    }

    for src in key_sources:
        xs, ys = [], []
        for frac in sorted(fractions):
            key = (src, tgt_name, frac, 'full_finetune')
            if key in data:
                sp = data[key].get('spearman_rho', data[key].get('spearman', 0))
                xs.append(frac * 100)
                ys.append(sp)
        if xs:
            c = source_colors.get(src, COLORS['secondary'])
            label = source_labels.get(src, src)
            ls = '--' if src == 'scratch' else '-'
            ax.plot(xs, ys, color=c, ls=ls, lw=1.8, marker='o',
                    markersize=4, label=label, zorder=3)

    ax.set_xticks([1, 5, 10, 25])
    ax.set_xticklabels(['1%', '5%', '10%', '25%'])
    style_axis(ax, title=f'Scaling: Source → S2 Drosophila',
               xlabel='Training Data Fraction', ylabel='Spearman $\\rho$')
    ax.legend(fontsize=FONTS['legend'] - 1, loc='lower right', framealpha=0.8)

elif mouse_tgt:
    tgt_name = mouse_tgt[0]
    for src in all_sources[:5]:
        xs, ys = [], []
        for frac in sorted(fractions):
            key = (src, tgt_name, frac, 'full_finetune')
            if key in data:
                sp = data[key].get('spearman_rho', data[key].get('spearman', 0))
                xs.append(frac * 100)
                ys.append(sp)
        if xs:
            ax.plot(xs, ys, lw=1.5, marker='o', markersize=3, label=src[:8])
    style_axis(ax, title='Scaling: Source → Mouse ESC',
               xlabel='Data Fraction (%)', ylabel='Spearman $\\rho$')
    ax.legend(fontsize=FONTS['legend'] - 2, loc='lower right', framealpha=0.8)
else:
    ax.text(0.5, 0.5, 'No target data', transform=ax.transAxes,
            ha='center', va='center')

add_panel_label(ax, 'B')

fig.suptitle(f'Fig 7.  Cross-Species Transfer Learning ({len(data)} Experiments)',
             fontsize=FONTS['title'], fontweight='bold', y=1.02, color=COLORS['text'])

plt.tight_layout()
save_poster_fig(fig, 'poster_fig7_transfer_learning')
print('Done.')
