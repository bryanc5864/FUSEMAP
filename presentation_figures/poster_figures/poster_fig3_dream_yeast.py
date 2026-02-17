#!/usr/bin/env python3
"""
Poster Fig 3: DREAM Yeast Results (Compact 1x2)
Panel A: Training curve  |  Panel B: Per-subset performance bars
ALL values from actual files (epoch_log.txt TSV + dream_subset_metrics.txt TSV).
"""
import sys, json, csv
sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/poster_figures')
from poster_style import *

apply_poster_style()

BASE = '/home/bcheng/sequence_optimization/FUSEMAP'
model_dir = f'{BASE}/models/legatoV2/outputs/dream_pro_dream_20250831_182621'

# ── LOAD TRAINING CURVES FROM epoch_log.txt (TSV) ───────────────────────────
log_path = f'{model_dir}/logs/epoch_log.txt'
epochs, train_r, val_r, test_r_curve = [], [], [], []

with open(log_path) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        epochs.append(int(row['epoch']))
        train_r.append(float(row['train_pearson']))
        val_r.append(float(row['val_pearson']))
        tr = row.get('test_pearson', 'nan')
        test_r_curve.append(float(tr) if tr != 'nan' else np.nan)

epochs = np.array(epochs)
train_r = np.array(train_r)
val_r = np.array(val_r)
test_r_curve = np.array(test_r_curve)

# Final results
with open(f'{model_dir}/final_results.json') as f:
    results = json.load(f)
test_r = results['test']['target_pearson']
test_rho = results['test']['target_spearman']

# ── LOAD PER-SUBSET METRICS FROM TSV ────────────────────────────────────────
subset_path = f'{model_dir}/logs/dream_subset_metrics.txt'
# Keep only final epoch values per subset
last_epoch = epochs[-1]
subset_names = []
subset_r = []

with open(subset_path) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        ep = int(row['epoch'])
        name = row['subset']
        if name == 'all':
            continue
        r_val = float(row['pearson'])
        # Keep last occurrence (highest epoch)
        found = False
        for idx, n in enumerate(subset_names):
            if n == name:
                subset_r[idx] = r_val
                found = True
                break
        if not found:
            subset_names.append(name)
            subset_r.append(r_val)

print(f"Training: {len(epochs)} epochs (1 to {epochs[-1]})")
print(f"Test: r={test_r:.3f}, rho={test_rho:.3f}")
print(f"Subsets ({len(subset_names)}): {list(zip(subset_names, [f'{v:.3f}' for v in subset_r]))}")

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))

# Panel A: Training curve
ax = axes[0]
ax.plot(epochs, train_r, color=COLORS['yeast'], ls='-', lw=1.8,
        alpha=0.6, label='Train r')
ax.plot(epochs, val_r, color=COLORS['yeast'], ls='--', lw=1.8,
        label='Val r')
# Plot test r where available
valid_test = ~np.isnan(test_r_curve)
if valid_test.any():
    ax.plot(epochs[valid_test], test_r_curve[valid_test],
            color=COLORS['accent'], ls=':', lw=1.2, alpha=0.5, label='Test r')

ax.axhline(test_r, color=COLORS['yeast'], ls=':', lw=1.0, alpha=0.6)
ax.text(epochs[-1] * 0.95, test_r + 0.003,
        f'Final test r={test_r:.3f}', fontsize=FONTS['annotation'] - 1,
        fontweight='bold', color=COLORS['yeast'], ha='right', va='bottom')

# Mark best val epoch
best_idx = np.argmax(val_r)
ax.scatter(epochs[best_idx], val_r[best_idx], marker='*', s=100,
           color=COLORS['yeast'], zorder=5, edgecolors='black', linewidths=0.3)

style_axis(ax, title='Training Convergence', ylabel='Pearson r', xlabel='Epoch')
ax.legend(fontsize=FONTS['legend'] - 1, loc='lower right', framealpha=0.8)
ax.set_ylim(0.6, 1.0)
add_panel_label(ax, 'A')

# Panel B: Per-subset bars
ax = axes[1]
if subset_names:
    order = np.argsort(subset_r)[::-1]
    names_sorted = [subset_names[i] for i in order]
    vals_sorted = [subset_r[i] for i in order]

    colors_bar = [COLORS['yeast'] if v > 0.5 else COLORS['warning'] for v in vals_sorted]
    bars = ax.barh(range(len(names_sorted)), vals_sorted, color=colors_bar,
                   edgecolor='white', linewidth=0.5, height=0.6)

    for i, (v, name) in enumerate(zip(vals_sorted, names_sorted)):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=FONTS['bar_label'],
                fontweight='bold', color=COLORS['text'])

    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=FONTS['tick'] - 0.5)
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()

style_axis(ax, title='Per-Subset Pearson r (Final Epoch)', xlabel='Pearson r', grid_y=False)
ax.xaxis.grid(True, alpha=0.3, linewidth=0.6, color=COLORS['grid'])
add_panel_label(ax, 'B')

fig.suptitle(f'Fig 3.  DREAM 2022 Yeast Challenge  |  r = {test_r:.3f}, $\\rho$ = {test_rho:.3f}',
             fontsize=FONTS['title'], fontweight='bold', y=1.02, color=COLORS['text'])

plt.tight_layout()
save_poster_fig(fig, 'poster_fig3_dream_yeast')
print('Done.')
