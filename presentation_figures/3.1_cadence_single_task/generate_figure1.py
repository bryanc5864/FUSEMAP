#!/usr/bin/env python3
"""
Generate Figure 1: CADENCE Training Dynamics for Human lentiMPRA Datasets.

3x3 composite figure:
  Rows: K562, HepG2, WTC11
  Columns: NLL Loss, Pearson r, Spearman rho
"""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Parse training logs
# ──────────────────────────────────────────────────────────────────────────────

def parse_log(filepath):
    """Parse a CADENCE training log and return per-epoch metrics."""
    train_nll, val_nll = [], []
    train_r, val_r = [], []
    train_rho, val_rho = [], []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Match TRAIN summary lines
        m_train = re.search(
            r'\[TRAIN\]\s+NLL:\s+([\d.]+)\s+\|.*?\|\s+r:\s+([\d.]+)\s+\|\s+rho:\s+([\d.]+)',
            line
        )
        if m_train:
            train_nll.append(float(m_train.group(1)))
            train_r.append(float(m_train.group(2)))
            train_rho.append(float(m_train.group(3)))

        # Match VAL summary lines (the main summary, not per-dataset)
        m_val = re.search(
            r'\[VAL\]\s+NLL:\s+([\d.]+)\s+\|.*?\|\s+r:\s+([\d.]+)\s+\|\s+rho:\s+([\d.]+)',
            line
        )
        if m_val:
            val_nll.append(float(m_val.group(1)))
            val_r.append(float(m_val.group(2)))
            val_rho.append(float(m_val.group(3)))

    return {
        'train_nll': np.array(train_nll),
        'val_nll': np.array(val_nll),
        'train_r': np.array(train_r),
        'val_r': np.array(val_r),
        'train_rho': np.array(train_rho),
        'val_rho': np.array(val_rho),
    }


log_paths = {
    'K562':  '/home/bcheng/sequence_optimization/FUSEMAP/results/cadence_k562_v2/training.log',
    'HepG2': '/home/bcheng/sequence_optimization/FUSEMAP/results/cadence_hepg2_v2/training.log',
    'WTC11': '/home/bcheng/sequence_optimization/FUSEMAP/results/cadence_wtc11_v2/training.log',
}

data = {}
for name, path in log_paths.items():
    data[name] = parse_log(path)
    n_epochs = len(data[name]['train_nll'])
    print(f"{name}: parsed {n_epochs} epochs")

# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

BLUE = '#4A90D9'
RED  = '#E74C3C'
LW   = 2

cell_types = ['K562', 'HepG2', 'WTC11']
col_titles = ['NLL Loss', 'Pearson $r$', r'Spearman $\rho$']

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
plt.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.06,
                    hspace=0.38, wspace=0.30)

for row_idx, ct in enumerate(cell_types):
    d = data[ct]
    epochs = np.arange(len(d['train_nll']))

    # Columns: NLL, Pearson r, Spearman rho
    metric_pairs = [
        ('train_nll', 'val_nll'),
        ('train_r', 'val_r'),
        ('train_rho', 'val_rho'),
    ]

    for col_idx, (train_key, val_key) in enumerate(metric_pairs):
        ax = axes[row_idx, col_idx]
        train_vals = d[train_key]
        val_vals = d[val_key]

        # Plot curves
        ax.plot(epochs, train_vals, color=BLUE, linewidth=LW, label='Train')
        ax.plot(epochs, val_vals, color=RED, linewidth=LW, label='Validation')

        # Find best validation epoch
        if col_idx == 0:
            # For NLL, best = minimum
            best_epoch = int(np.argmin(val_vals))
        else:
            # For Pearson r and Spearman rho, best = maximum
            best_epoch = int(np.argmax(val_vals))

        # Mark best val epoch with a star
        ax.plot(best_epoch, val_vals[best_epoch], marker='*', markersize=14,
                color=RED, markeredgecolor='black', markeredgewidth=0.8,
                zorder=5)

        # Annotate the best value -- use axes transform offsets to stay in bounds
        y_range = np.max(val_vals) - np.min(val_vals)
        if y_range == 0:
            y_range = 0.1
        if col_idx == 0:
            # NLL: annotate below the star
            txt_y = val_vals[best_epoch] + 0.06 * y_range
            va = 'bottom'
        else:
            # r / rho: annotate above the star
            txt_y = val_vals[best_epoch] - 0.06 * y_range
            va = 'top'

        # Shift text left if best_epoch is near right edge
        txt_x = best_epoch + 1.0
        ha = 'left'
        if best_epoch > len(epochs) - 5:
            txt_x = best_epoch - 1.0
            ha = 'right'

        ax.annotate(
            f'{val_vals[best_epoch]:.4f}',
            xy=(best_epoch, val_vals[best_epoch]),
            xytext=(txt_x, txt_y),
            fontsize=8, fontweight='bold', color=RED,
            ha=ha, va=va,
            clip_on=True,
        )

        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # X-axis
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_xlim(-0.5, len(epochs) - 0.5)

        # Y-axis label
        if col_idx == 0:
            ylabel = 'NLL Loss'
        elif col_idx == 1:
            ylabel = 'Pearson $r$'
        else:
            ylabel = r'Spearman $\rho$'
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')

        # Column title (top row only)
        if row_idx == 0:
            ax.set_title(col_titles[col_idx], fontsize=14, fontweight='bold', pad=10)

        # Row label (left column only)
        if col_idx == 0:
            ax.annotate(
                ct, xy=(-0.30, 0.5), xycoords='axes fraction',
                fontsize=15, fontweight='bold', ha='center', va='center',
                rotation=90,
            )

        # Legend
        if row_idx == 0 and col_idx == 2:
            ax.legend(loc='lower right', fontsize=10, framealpha=0.9,
                      edgecolor='gray')

        # Tick styling
        ax.tick_params(axis='both', labelsize=9)

# Suptitle
fig.suptitle(
    'Figure 1. CADENCE Training Dynamics \u2014 Human lentiMPRA',
    fontsize=16, fontweight='bold', y=0.97,
)

# Save
out_base = '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/3.1_cadence_single_task/figure1_human_training_curves'
fig.savefig(f'{out_base}.png', dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(f'{out_base}.pdf', bbox_inches='tight', facecolor='white')
print(f"\nSaved: {out_base}.png")
print(f"Saved: {out_base}.pdf")
