#!/usr/bin/env python3
"""
Poster Fig 2: CADENCE Training Dynamics (Compact)
Compact 1x3: Pearson r curves for Human (K562), Drosophila (DeepSTARR Dev/Hk), Plants (Maize).
Shows convergence and final test performance.
"""
import sys, re, json
sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/poster_figures')
from poster_style import *

apply_poster_style()

BASE = '/home/bcheng/sequence_optimization/FUSEMAP'

# ── PARSE TRAINING LOGS ──────────────────────────────────────────────────────
def parse_log(path, dataset_key, multi_output=False):
    """Parse training log for Pearson r curves."""
    epochs, train_r, val_r = [], [], []
    val_sub1, val_sub2 = [], []

    epoch_re = re.compile(r'EPOCH (\d+) SUMMARY')
    train_re = re.compile(r'\[TRAIN\] NLL: [\d.]+ \| MSE: [\d.]+ \| r: ([\d.]+)')
    val_re = re.compile(r'\[VAL\]\s+NLL: [\d.]+ \| MSE: [\d.]+ \| r: ([\d.]+)')

    if multi_output:
        sub1_re = re.compile(r'Dev: r=([\d.]+)')
        sub2_re = re.compile(r'Hk: r=([\d.]+)')
    else:
        sub1_re = sub2_re = None

    current_epoch = None
    with open(path) as f:
        for line in f:
            m = epoch_re.search(line)
            if m:
                current_epoch = int(m.group(1))
                continue
            if current_epoch is not None:
                m = train_re.search(line)
                if m:
                    epochs.append(current_epoch)
                    train_r.append(float(m.group(1)))
                    continue
                m = val_re.search(line)
                if m:
                    val_r.append(float(m.group(1)))
                    continue
                if sub1_re:
                    m = sub1_re.search(line)
                    if m:
                        val_sub1.append(float(m.group(1)))
                        continue
                    m = sub2_re.search(line)
                    if m:
                        val_sub2.append(float(m.group(1)))
                        continue

    n = len(epochs)
    return {
        'epochs': np.array(epochs),
        'train_r': np.array(train_r[:n]),
        'val_r': np.array(val_r[:n]),
        'val_sub1': np.array(val_sub1[:n]) if val_sub1 else None,
        'val_sub2': np.array(val_sub2[:n]) if val_sub2 else None,
    }

# Parse K562
k562_log = f'{BASE}/results/cadence_k562_v2/training.log'
k562 = parse_log(k562_log, 'k562')

# Parse DeepSTARR (has Dev/Hk sub-outputs)
ds_log = f'{BASE}/training/results/cadence_deepstarr_v2/training.log'
ds = parse_log(ds_log, 'deepstarr', multi_output=True)

# Parse Maize
maize_log = f'{BASE}/training/results/cadence_maize_v1/training.log'
maize = parse_log(maize_log, 'maize')

# Load final test results
k562_res = json.load(open(f'{BASE}/results/cadence_k562_v2/final_results.json'))
ds_res = json.load(open(f'{BASE}/training/results/cadence_deepstarr_v2/final_results.json'))
maize_res = json.load(open(f'{BASE}/training/results/cadence_maize_v1/final_results.json'))

k562_test_r = k562_res['test']['encode4_k562']['activity']['pearson']
ds_dev_test = ds_res['test']['deepstarr']['Dev']['pearson']
ds_hk_test = ds_res['test']['deepstarr']['Hk']['pearson']
maize_val_r = maize_res['jores_maize']['leaf']['pearson']['value']

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.5))

LW = 1.8

# Panel A: K562
ax = axes[0]
ax.plot(k562['epochs'], k562['train_r'], color=COLORS['human'], ls='-',
        lw=LW, alpha=0.5, label='Train')
ax.plot(k562['epochs'], k562['val_r'], color=COLORS['human'], ls='--',
        lw=LW, label='Val')
ax.axhline(k562_test_r, color=COLORS['human'], ls=':', lw=1.0, alpha=0.6)
ax.text(k562['epochs'][-1] * 0.95, k562_test_r + 0.005,
        f'Test r={k562_test_r:.3f}', fontsize=FONTS['annotation'] - 1,
        fontweight='bold', color=COLORS['human'], ha='right', va='bottom')
idx = np.argmax(k562['val_r'])
ax.scatter(k562['epochs'][idx], k562['val_r'][idx], marker='*', s=100,
           color=COLORS['human'], zorder=5, edgecolors='black', linewidths=0.3)
style_axis(ax, title='K562 (Human)', ylabel='Pearson r', xlabel='Epoch')
ax.legend(fontsize=FONTS['legend'] - 1, loc='lower right', framealpha=0.8)
ax.set_ylim(0, 1.0)
add_panel_label(ax, 'A')

# Panel B: DeepSTARR
ax = axes[1]
ax.plot(ds['epochs'], ds['train_r'], color=COLORS['secondary'], ls='-',
        lw=LW - 0.5, alpha=0.4, label='Train (agg.)')
if ds['val_sub1'] is not None:
    ax.plot(ds['epochs'], ds['val_sub1'], color=COLORS['drosophila'], ls='--',
            lw=LW, label='Val Dev')
if ds['val_sub2'] is not None:
    ax.plot(ds['epochs'], ds['val_sub2'], color='#1ABC9C', ls='--',
            lw=LW, label='Val Hk')
ax.axhline(ds_dev_test, color=COLORS['drosophila'], ls=':', lw=1.0, alpha=0.6)
ax.axhline(ds_hk_test, color='#1ABC9C', ls=':', lw=1.0, alpha=0.6)
ax.text(ds['epochs'][-1] * 0.95, ds_dev_test + 0.005,
        f'Test Dev={ds_dev_test:.3f}', fontsize=FONTS['annotation'] - 1.5,
        fontweight='bold', color=COLORS['drosophila'], ha='right', va='bottom')
ax.text(ds['epochs'][-1] * 0.95, ds_hk_test + 0.005,
        f'Test Hk={ds_hk_test:.3f}', fontsize=FONTS['annotation'] - 1.5,
        fontweight='bold', color='#1ABC9C', ha='right', va='bottom')
style_axis(ax, title='DeepSTARR (Drosophila)', xlabel='Epoch')
ax.legend(fontsize=FONTS['legend'] - 1, loc='lower right', framealpha=0.8)
ax.set_ylim(0, 1.0)
add_panel_label(ax, 'B')

# Panel C: Maize
ax = axes[2]
ax.plot(maize['epochs'], maize['train_r'], color=COLORS['plant'], ls='-',
        lw=LW, alpha=0.5, label='Train')
ax.plot(maize['epochs'], maize['val_r'], color=COLORS['plant'], ls='--',
        lw=LW, label='Val')
ax.axhline(maize_val_r, color=COLORS['plant'], ls=':', lw=1.0, alpha=0.6)
ax.text(maize['epochs'][-1] * 0.95, maize_val_r + 0.005,
        f'Val r={maize_val_r:.3f}', fontsize=FONTS['annotation'] - 1,
        fontweight='bold', color=COLORS['plant'], ha='right', va='bottom')
idx = np.argmax(maize['val_r'])
ax.scatter(maize['epochs'][idx], maize['val_r'][idx], marker='*', s=100,
           color=COLORS['plant'], zorder=5, edgecolors='black', linewidths=0.3)
style_axis(ax, title='Maize Leaf (Plant)', xlabel='Epoch')
ax.legend(fontsize=FONTS['legend'] - 1, loc='lower right', framealpha=0.8)
ax.set_ylim(0, 1.0)
add_panel_label(ax, 'C')

fig.suptitle('Fig 2.  CADENCE Training Dynamics Across Kingdoms',
             fontsize=FONTS['title'], fontweight='bold', y=1.02, color=COLORS['text'])

plt.tight_layout()
save_poster_fig(fig, 'poster_fig2_training_dynamics')
print('Done.')
