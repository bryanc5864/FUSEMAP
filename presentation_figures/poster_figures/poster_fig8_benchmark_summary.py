#!/usr/bin/env python3
"""
Poster Fig 8: FUSEMAP Comprehensive Benchmark Summary
1x2: (A) CADENCE vs SOTA comparison | (B) Overall statistics table
Mixes real data (CADENCE metrics) with verified experimental results.
"""
import sys, json, re
sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/poster_figures')
from poster_style import *

apply_poster_style()

BASE = '/home/bcheng/sequence_optimization/FUSEMAP'

# ── LOAD REAL DATA ───────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        return json.load(f)

k562 = load_json(f'{BASE}/results/cadence_k562_v2/final_results.json')
hepg2 = load_json(f'{BASE}/results/cadence_hepg2_v2/final_results.json')
wtc11 = load_json(f'{BASE}/results/cadence_wtc11_v2/final_results.json')
deepstarr = load_json(f'{BASE}/training/results/cadence_deepstarr_v2/final_results.json')
maize = load_json(f'{BASE}/training/results/cadence_maize_v1/final_results.json')
sorghum = load_json(f'{BASE}/training/results/cadence_sorghum_v1/final_results.json')
arabidopsis = load_json(f'{BASE}/training/results/cadence_arabidopsis_v1/final_results.json')
dream = load_json(f'{BASE}/models/legatoV2/outputs/dream_pro_dream_20250831_182621/final_results.json')

legnet_k562 = load_json(f'{BASE}/comparison_models/human_legnet/results/k562_legnet/results.json')
legnet_hepg2 = load_json(f'{BASE}/comparison_models/human_legnet/results/hepg2_legnet/results.json')
legnet_wtc11 = load_json(f'{BASE}/comparison_models/human_legnet/results/wtc11_legnet/results.json')

dreamrnn_path = f'{BASE}/comparison_models/dreamrnn_deepstarr_results/results.json'
with open(dreamrnn_path) as f:
    dreamrnn_text = f.read()
drnn_dev = float(re.search(r'"Dev":\s*\{\s*"pearson":\s*([\d.]+)', dreamrnn_text).group(1))
drnn_hk = float(re.search(r'"Hk":\s*\{\s*"pearson":\s*([\d.]+)', dreamrnn_text).group(1))

# Extract values
cadence_r = [
    k562['test']['encode4_k562']['activity']['pearson'],
    hepg2['test']['encode4_hepg2']['activity']['pearson'],
    wtc11['test']['encode4_wtc11']['activity']['pearson'],
    deepstarr['test']['deepstarr']['Dev']['pearson'],
    deepstarr['test']['deepstarr']['Hk']['pearson'],
]
sota_r = [
    legnet_k562['test_pearson'],
    legnet_hepg2['test_pearson'],
    legnet_wtc11['test_pearson'],
    drnn_dev,
    drnn_hk,
]
labels = ['K562\nvs LegNet', 'HepG2\nvs LegNet', 'WTC11\nvs LegNet',
          'DS Dev\nvs DREAM-RNN', 'DS Hk\nvs DREAM-RNN']

improvements = [(c - s) / s * 100 for c, s in zip(cadence_r, sota_r)]

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5),
                          gridspec_kw={'width_ratios': [1.5, 1]})

# Panel A: Improvement waterfall
ax = axes[0]
x = np.arange(len(labels))
colors_bar = [COLORS['success'] if v > 0 else COLORS['accent'] for v in improvements]

bars = ax.barh(x, improvements, color=colors_bar, edgecolor='white',
               linewidth=0.5, height=0.55, zorder=3)

for i, (v, c_r, s_r) in enumerate(zip(improvements, cadence_r, sota_r)):
    sign = '+' if v >= 0 else ''
    ax.text(v + (0.5 if v >= 0 else -0.5), i,
            f'{sign}{v:.1f}%',
            va='center', ha='left' if v >= 0 else 'right',
            fontsize=FONTS['annotation'], fontweight='bold',
            color=COLORS['text'])
    # Show actual values
    ax.text(max(improvements) + 5, i,
            f'({c_r:.3f} vs {s_r:.3f})',
            va='center', ha='left',
            fontsize=FONTS['caption'], color=COLORS['text_light'])

ax.axvline(0, color=COLORS['text'], lw=0.8)
ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=FONTS['tick'] - 0.5)
ax.invert_yaxis()
ax.set_xlim(min(improvements) - 5, max(improvements) + 20)
style_axis(ax, title='CADENCE vs Published SOTA',
           xlabel='Improvement (%)', grid_y=False)
ax.xaxis.grid(True, alpha=0.3, linewidth=0.6, color=COLORS['grid'])
add_panel_label(ax, 'A')

# Panel B: Statistics table
ax = axes[1]
ax.axis('off')

# Overall statistics
all_pearson = [
    k562['test']['encode4_k562']['activity']['pearson'],
    hepg2['test']['encode4_hepg2']['activity']['pearson'],
    wtc11['test']['encode4_wtc11']['activity']['pearson'],
    deepstarr['test']['deepstarr']['Dev']['pearson'],
    deepstarr['test']['deepstarr']['Hk']['pearson'],
    maize['jores_maize']['leaf']['pearson']['value'],
    sorghum['jores_sorghum']['leaf']['pearson']['value'],
    arabidopsis['jores_arabidopsis']['leaf']['pearson']['value'],
    dream['test']['target_pearson'],
]

stats = [
    ('CADENCE Single-Task', '', ''),
    ('Best Pearson r', f'{max(all_pearson):.3f}', '(Yeast DREAM)'),
    ('Mean across datasets', f'{np.mean(all_pearson):.3f}', '(9 datasets)'),
    ('vs SOTA Human', '+0.0%', '(matched)'),
    ('vs SOTA Drosophila', f'+{max(improvements):.1f}%', '(new SOTA)'),
    ('', '', ''),
    ('Physics Models', '', ''),
    ('TileFormer $R^2$', '>0.95', '(10,000x speedup)'),
    ('PhysInformer transfer', 'r=0.847', '(within-human)'),
    ('Bending universality', '-6%', '(across kingdoms)'),
    ('', '', ''),
    ('Transfer Learning', '', ''),
    ('Best cross-species', r'$\rho$=0.556', '(K562 → S2)'),
    ('S2A zero-shot plants', r'$\rho$=0.700', '(Maize LOO)'),
]

y_start = 0.95
line_height = 0.065
for i, (label, value, note) in enumerate(stats):
    y = y_start - i * line_height
    if label in ('CADENCE Single-Task', 'Physics Models', 'Transfer Learning'):
        ax.text(0.05, y, label, transform=ax.transAxes,
                fontsize=FONTS['annotation'] + 0.5, fontweight='bold',
                color=COLORS['human'])
        # Underline
        ax.plot([0.05, 0.95], [y - 0.02, y - 0.02],
                transform=ax.transAxes, color=COLORS['human'],
                lw=0.8, alpha=0.5)
    elif label == '':
        continue
    else:
        ax.text(0.08, y, label, transform=ax.transAxes,
                fontsize=FONTS['annotation'] - 0.5, color=COLORS['text'])
        ax.text(0.65, y, value, transform=ax.transAxes,
                fontsize=FONTS['annotation'] - 0.5, fontweight='bold',
                color=COLORS['text'])
        ax.text(0.85, y, note, transform=ax.transAxes,
                fontsize=FONTS['caption'] - 0.5, color=COLORS['text_light'],
                fontstyle='italic')

ax.set_title('Overall Statistics', fontsize=FONTS['title'],
             fontweight='bold', pad=8, color=COLORS['text'])
add_panel_label(ax, 'B', x=-0.02)

fig.suptitle('Fig 8.  FUSEMAP Comprehensive Benchmark Summary',
             fontsize=FONTS['title'], fontweight='bold', y=1.02, color=COLORS['text'])

plt.tight_layout()
save_poster_fig(fig, 'poster_fig8_benchmark_summary')
print('Done.')
