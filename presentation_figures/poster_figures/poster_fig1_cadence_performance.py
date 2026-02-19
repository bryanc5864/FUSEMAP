#!/usr/bin/env python3
"""
Poster Fig 1: CADENCE Single-Task Performance Summary
Compact bar chart: 9 datasets, color-coded by organism, with SOTA baselines.
ALL values from actual JSON files.
"""
import sys, json, re
sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/poster_figures')
from poster_style import *

apply_poster_style()

BASE = '/home/bcheng/sequence_optimization/FUSEMAP'

# ── LOAD DATA ────────────────────────────────────────────────────────────────
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
_dev_r = float(re.search(r'"Dev":\s*\{\s*"pearson":\s*([\d.]+)', dreamrnn_text).group(1))
_hk_r = float(re.search(r'"Hk":\s*\{\s*"pearson":\s*([\d.]+)', dreamrnn_text).group(1))

# ── EXTRACT VALUES ───────────────────────────────────────────────────────────
datasets = ['K562', 'HepG2', 'WTC11', 'S2\nDev', 'S2\nHk',
            'Maize', 'Sorghum', 'Arab.', 'DREAM\nYeast']

pearson_r = [
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

sota_r = [
    legnet_k562['test_pearson'], legnet_hepg2['test_pearson'],
    legnet_wtc11['test_pearson'], _dev_r, _hk_r,
    None, None, None, None,
]

organism_idx = [0, 0, 0, 1, 1, 2, 2, 2, 3]
org_colors = [COLORS['human'], COLORS['drosophila'], COLORS['plant'], COLORS['yeast']]

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.0, 4.2))

bar_w = 0.32
gap = 0.06
group_gap = 0.5

x_pos = []
cx = 0.0
for i in range(len(datasets)):
    if i > 0 and organism_idx[i] != organism_idx[i-1]:
        cx += group_gap
    x_pos.append(cx)
    cx += 1.0
x_pos = np.array(x_pos)

for i in range(len(datasets)):
    c = org_colors[organism_idx[i]]
    has_sota = sota_r[i] is not None

    if has_sota:
        xc = x_pos[i] - (bar_w + gap) / 2
        ax.bar(xc, pearson_r[i], width=bar_w, color=c, edgecolor='white',
               linewidth=0.5, zorder=3)
        ax.text(xc, pearson_r[i] + 0.01, f'{pearson_r[i]:.3f}',
                ha='center', va='bottom', fontsize=FONTS['bar_label'],
                fontweight='bold', color=c)

        xs = x_pos[i] + (bar_w + gap) / 2
        lighter = matplotlib.colors.to_rgba(c, alpha=0.35)
        ax.bar(xs, sota_r[i], width=bar_w, color=lighter, edgecolor=c,
               linewidth=0.8, hatch='///', zorder=3)
        ax.text(xs, sota_r[i] + 0.01, f'{sota_r[i]:.3f}',
                ha='center', va='bottom', fontsize=FONTS['bar_label'] - 0.5,
                color=COLORS['text_light'])
    else:
        ax.bar(x_pos[i], pearson_r[i], width=bar_w * 1.1, color=c,
               edgecolor='white', linewidth=0.5, zorder=3)
        ax.text(x_pos[i], pearson_r[i] + 0.01, f'{pearson_r[i]:.3f}',
                ha='center', va='bottom', fontsize=FONTS['bar_label'],
                fontweight='bold', color=c)

# Delta annotations for Drosophila
for i in [3, 4]:
    if sota_r[i] is not None:
        delta = (pearson_r[i] - sota_r[i]) / sota_r[i] * 100
        xc = x_pos[i] - (bar_w + gap) / 2
        ax.annotate(f'+{delta:.0f}%', xy=(xc, pearson_r[i] + 0.025),
                    fontsize=FONTS['annotation'] - 1, fontweight='bold',
                    color=COLORS['drosophila'], ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#F3E5F5',
                              edgecolor=COLORS['drosophila'], linewidth=0.8, alpha=0.9))

# Group separators
group_bounds = [(0,3), (3,5), (5,8), (8,9)]
for g in range(1, len(group_bounds)):
    i_prev = group_bounds[g-1][1] - 1
    i_curr = group_bounds[g][0]
    sep = (x_pos[i_prev] + x_pos[i_curr]) / 2
    ax.axvline(sep, color=COLORS['border'], ls='--', lw=0.8, alpha=0.5, zorder=1)

# Reference line
ax.axhline(0.80, color=COLORS['text_light'], ls=':', lw=0.8, alpha=0.4, zorder=1)
ax.text(x_pos[-1] + 0.6, 0.80, 'r = 0.80', fontsize=7, color=COLORS['text_light'],
        va='center', fontstyle='italic')

# Group labels
org_names = ['Human', 'Drosophila', 'Plants', 'Yeast']
for g, (s, e) in enumerate(group_bounds):
    cx = np.mean(x_pos[s:e])
    ax.text(cx, -0.12, org_names[g], fontsize=FONTS['annotation'],
            fontweight='bold', ha='center', va='top',
            transform=ax.get_xaxis_transform(), color=org_colors[g])

ax.set_xticks(x_pos)
ax.set_xticklabels(datasets, fontsize=FONTS['tick'] - 0.5, fontweight='bold')
ax.tick_params(axis='x', length=0, pad=8)
ax.set_ylim(0, 1.05)
ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.7)
ax.set_yticks(np.arange(0, 1.1, 0.2))

style_axis(ax, ylabel='Test Pearson r')

# Legend
cadence_p = mpatches.Patch(facecolor=COLORS['human'], edgecolor='white',
                            linewidth=0.5, label='CADENCE')
sota_p = mpatches.Patch(facecolor=matplotlib.colors.to_rgba(COLORS['human'], 0.35),
                         edgecolor=COLORS['human'], linewidth=0.8,
                         hatch='///', label='Published SOTA')
ax.legend(handles=[cadence_p, sota_p], loc='upper left', fontsize=FONTS['legend'],
          frameon=True, framealpha=0.9, edgecolor=COLORS['border'])

fig.suptitle('Fig 1.  CADENCE Performance Across 9 Datasets (4 Kingdoms)',
             fontsize=FONTS['title'], fontweight='bold', y=0.98, color=COLORS['text'])

plt.tight_layout(rect=[0, 0.04, 1, 0.94])
save_poster_fig(fig, 'poster_fig1_cadence_performance')
print('Done.')
