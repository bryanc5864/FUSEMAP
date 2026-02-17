#!/usr/bin/env python3
"""
Generate Figure 2: CADENCE Training Dynamics for Drosophila DeepSTARR dataset.
2x2 figure: (A) NLL Loss, (B) MSE, (C) Pearson r, (D) Spearman rho.
"""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 1. Parse the training log
# ---------------------------------------------------------------------------
log_path = '/home/bcheng/sequence_optimization/FUSEMAP/training/results/cadence_deepstarr_v2/training.log'

# Storage
epochs = []
train_nll, train_mse, train_r, train_rho = [], [], [], []
val_nll, val_mse, val_r, val_rho = [], [], [], []
val_dev_r, val_dev_rho = [], []
val_hk_r, val_hk_rho = [], []

# We also need NLL/MSE per-class for val.  The log only has aggregate val NLL/MSE,
# but let's also grab per-class RMSE to compute per-class MSE if available.
# Actually, the log gives per-class RMSE but not per-class NLL/MSE directly.
# For the NLL and MSE panels we'll show aggregate train vs aggregate val (single pair),
# and for correlation panels we'll show per-class val lines.

# Regex patterns
epoch_re = re.compile(r'EPOCH (\d+) SUMMARY')
train_re = re.compile(
    r'\[TRAIN\] NLL: ([\d.]+) \| MSE: ([\d.]+) \| r: ([\d.]+) \| rho: ([\d.]+)'
)
val_re = re.compile(
    r'\[VAL\]\s+NLL: ([\d.]+) \| MSE: ([\d.]+) \| r: ([\d.]+) \| rho: ([\d.]+)'
)
dev_re = re.compile(
    r'Dev: r=([\d.]+) \| rho=([\d.]+) \| R2=([\d.]+) \| RMSE=([\d.]+)'
)
hk_re = re.compile(
    r'Hk: r=([\d.]+) \| rho=([\d.]+) \| R2=([\d.]+) \| RMSE=([\d.]+)'
)

current_epoch = None

with open(log_path, 'r') as f:
    for line in f:
        m = epoch_re.search(line)
        if m:
            current_epoch = int(m.group(1))
            continue

        if current_epoch is not None:
            m = train_re.search(line)
            if m:
                epochs.append(current_epoch)
                train_nll.append(float(m.group(1)))
                train_mse.append(float(m.group(2)))
                train_r.append(float(m.group(3)))
                train_rho.append(float(m.group(4)))
                continue

            m = val_re.search(line)
            if m:
                val_nll.append(float(m.group(1)))
                val_mse.append(float(m.group(2)))
                val_r.append(float(m.group(3)))
                val_rho.append(float(m.group(4)))
                continue

            m = dev_re.search(line)
            if m:
                val_dev_r.append(float(m.group(1)))
                val_dev_rho.append(float(m.group(2)))
                continue

            m = hk_re.search(line)
            if m:
                val_hk_r.append(float(m.group(1)))
                val_hk_rho.append(float(m.group(2)))
                continue

# Convert to numpy
epochs = np.array(epochs)
train_nll = np.array(train_nll)
train_mse = np.array(train_mse)
train_r = np.array(train_r)
train_rho = np.array(train_rho)
val_nll = np.array(val_nll)
val_mse = np.array(val_mse)
val_r = np.array(val_r)
val_rho = np.array(val_rho)
val_dev_r = np.array(val_dev_r[:len(epochs)])
val_dev_rho = np.array(val_dev_rho[:len(epochs)])
val_hk_r = np.array(val_hk_r[:len(epochs)])
val_hk_rho = np.array(val_hk_rho[:len(epochs)])

print(f"Parsed {len(epochs)} epochs (0 to {epochs[-1]})")
print(f"  train_nll range: {train_nll.min():.4f} - {train_nll.max():.4f}")
print(f"  val_nll range:   {val_nll.min():.4f} - {val_nll.max():.4f}")
print(f"  val_dev_r range: {val_dev_r.min():.4f} - {val_dev_r.max():.4f}")
print(f"  val_hk_r range:  {val_hk_r.min():.4f} - {val_hk_r.max():.4f}")

# ---------------------------------------------------------------------------
# 2. Identify best validation epochs
# ---------------------------------------------------------------------------
# Best = lowest val NLL for loss panels, highest val r for correlation panels
best_val_nll_epoch = epochs[np.argmin(val_nll)]
best_val_mse_epoch = epochs[np.argmin(val_mse)]
best_val_r_epoch = epochs[np.argmax(val_r)]
best_val_rho_epoch = epochs[np.argmax(val_rho)]

# Per-class best val epochs for correlation panels
best_dev_r_epoch = epochs[np.argmax(val_dev_r)]
best_hk_r_epoch = epochs[np.argmax(val_hk_r)]
best_dev_rho_epoch = epochs[np.argmax(val_dev_rho)]
best_hk_rho_epoch = epochs[np.argmax(val_hk_rho)]

print(f"\nBest val NLL epoch: {best_val_nll_epoch} ({val_nll.min():.4f})")
print(f"Best val MSE epoch: {best_val_mse_epoch} ({val_mse.min():.4f})")
print(f"Best val Dev r epoch: {best_dev_r_epoch} ({val_dev_r.max():.4f})")
print(f"Best val Hk r epoch: {best_hk_r_epoch} ({val_hk_r.max():.4f})")
print(f"Best val Dev rho epoch: {best_dev_rho_epoch} ({val_dev_rho.max():.4f})")
print(f"Best val Hk rho epoch: {best_hk_rho_epoch} ({val_hk_rho.max():.4f})")

# ---------------------------------------------------------------------------
# 3. Create figure
# ---------------------------------------------------------------------------
DEV_COLOR = '#9B59B6'   # purple
HK_COLOR = '#1ABC9C'    # teal
TRAIN_ALPHA = 1.0
VAL_ALPHA = 1.0
LW = 2
STAR_SIZE = 180

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

def style_ax(ax, title, ylabel, xlabel='Epoch'):
    """Apply clean style to axis."""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_xlim(-1, epochs[-1] + 1)
    ax.tick_params(labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

# ---- Panel A: NLL Loss ----
ax = axes[0, 0]
ax.plot(epochs, train_nll, color=DEV_COLOR, ls='-', lw=LW, alpha=0.7, label='Train NLL')
ax.plot(epochs, val_nll, color=DEV_COLOR, ls='--', lw=LW, label='Val NLL')
# Mark best val epoch
idx_best = np.argmin(val_nll)
ax.scatter(epochs[idx_best], val_nll[idx_best], marker='*', s=STAR_SIZE,
           color=DEV_COLOR, zorder=5, edgecolors='black', linewidths=0.5)
ax.annotate(f'Best: {val_nll[idx_best]:.4f}\n(epoch {epochs[idx_best]})',
            xy=(epochs[idx_best], val_nll[idx_best]),
            xytext=(epochs[idx_best]+8, val_nll[idx_best]+0.04),
            fontsize=8, fontweight='bold', color=DEV_COLOR,
            arrowprops=dict(arrowstyle='->', color=DEV_COLOR, lw=1.2))
style_ax(ax, '(A) NLL Loss', 'NLL Loss')
ax.legend(fontsize=9, framealpha=0.9, loc='upper right',
          prop={'weight': 'bold'})

# ---- Panel B: MSE ----
ax = axes[0, 1]
ax.plot(epochs, train_mse, color=HK_COLOR, ls='-', lw=LW, alpha=0.7, label='Train MSE')
ax.plot(epochs, val_mse, color=HK_COLOR, ls='--', lw=LW, label='Val MSE')
idx_best = np.argmin(val_mse)
ax.scatter(epochs[idx_best], val_mse[idx_best], marker='*', s=STAR_SIZE,
           color=HK_COLOR, zorder=5, edgecolors='black', linewidths=0.5)
ax.annotate(f'Best: {val_mse[idx_best]:.4f}\n(epoch {epochs[idx_best]})',
            xy=(epochs[idx_best], val_mse[idx_best]),
            xytext=(epochs[idx_best]+8, val_mse[idx_best]+0.04),
            fontsize=8, fontweight='bold', color=HK_COLOR,
            arrowprops=dict(arrowstyle='->', color=HK_COLOR, lw=1.2))
style_ax(ax, '(B) MSE', 'MSE')
ax.legend(fontsize=9, framealpha=0.9, loc='upper right',
          prop={'weight': 'bold'})

# ---- Panel C: Pearson r ----
ax = axes[1, 0]
# Aggregate train
ax.plot(epochs, train_r, color='#7F8C8D', ls='-', lw=LW-0.5, alpha=0.5, label='Train r (agg.)')
# Per-class val
ax.plot(epochs, val_dev_r, color=DEV_COLOR, ls='--', lw=LW, label='Val Dev r')
ax.plot(epochs, val_hk_r, color=HK_COLOR, ls='--', lw=LW, label='Val Hk r')
# Stars for best val epoch per class
idx_dev = np.argmax(val_dev_r)
idx_hk = np.argmax(val_hk_r)
ax.scatter(epochs[idx_dev], val_dev_r[idx_dev], marker='*', s=STAR_SIZE,
           color=DEV_COLOR, zorder=5, edgecolors='black', linewidths=0.5)
ax.scatter(epochs[idx_hk], val_hk_r[idx_hk], marker='*', s=STAR_SIZE,
           color=HK_COLOR, zorder=5, edgecolors='black', linewidths=0.5)
# Annotate final test performance
ax.axhline(y=0.909, color=DEV_COLOR, ls=':', alpha=0.5, lw=1)
ax.axhline(y=0.920, color=HK_COLOR, ls=':', alpha=0.5, lw=1)
ax.text(epochs[-1]-2, 0.909+0.004, 'Test r=0.909', fontsize=8, fontweight='bold',
        color=DEV_COLOR, ha='right', va='bottom')
ax.text(epochs[-1]-2, 0.920+0.004, 'Test r=0.920', fontsize=8, fontweight='bold',
        color=HK_COLOR, ha='right', va='bottom')
style_ax(ax, '(C) Pearson r', 'Pearson r')
ax.legend(fontsize=9, framealpha=0.9, loc='lower right',
          prop={'weight': 'bold'})

# ---- Panel D: Spearman rho ----
ax = axes[1, 1]
# Aggregate train
ax.plot(epochs, train_rho, color='#7F8C8D', ls='-', lw=LW-0.5, alpha=0.5, label=r'Train $\rho$ (agg.)')
# Per-class val
ax.plot(epochs, val_dev_rho, color=DEV_COLOR, ls='--', lw=LW, label=r'Val Dev $\rho$')
ax.plot(epochs, val_hk_rho, color=HK_COLOR, ls='--', lw=LW, label=r'Val Hk $\rho$')
# Stars for best val epoch per class
idx_dev = np.argmax(val_dev_rho)
idx_hk = np.argmax(val_hk_rho)
ax.scatter(epochs[idx_dev], val_dev_rho[idx_dev], marker='*', s=STAR_SIZE,
           color=DEV_COLOR, zorder=5, edgecolors='black', linewidths=0.5)
ax.scatter(epochs[idx_hk], val_hk_rho[idx_hk], marker='*', s=STAR_SIZE,
           color=HK_COLOR, zorder=5, edgecolors='black', linewidths=0.5)
# Annotate final test performance
ax.axhline(y=0.867, color=DEV_COLOR, ls=':', alpha=0.5, lw=1)
ax.axhline(y=0.879, color=HK_COLOR, ls=':', alpha=0.5, lw=1)
ax.text(epochs[-1]-2, 0.867+0.004, r'Test $\rho$=0.867', fontsize=8, fontweight='bold',
        color=DEV_COLOR, ha='right', va='bottom')
ax.text(epochs[-1]-2, 0.879+0.004, r'Test $\rho$=0.879', fontsize=8, fontweight='bold',
        color=HK_COLOR, ha='right', va='bottom')
style_ax(ax, r'(D) Spearman $\rho$', r'Spearman $\rho$')
ax.legend(fontsize=9, framealpha=0.9, loc='lower right',
          prop={'weight': 'bold'})

# ---------------------------------------------------------------------------
# 4. Suptitle and subtitle
# ---------------------------------------------------------------------------
fig.suptitle('Figure 2. CADENCE Training Dynamics \u2014 Drosophila DeepSTARR',
             fontsize=16, fontweight='bold', y=0.98)
fig.text(0.5, 0.935,
         'Dev r=0.909, Hk r=0.920  |  Training: 100 epochs, 43.1 hours',
         ha='center', fontsize=11, fontweight='bold', color='#555555')

plt.tight_layout(rect=[0, 0, 1, 0.92])

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
out_dir = '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/3.1_cadence_single_task'
png_path = f'{out_dir}/figure2_deepstarr_training_curves.png'
pdf_path = f'{out_dir}/figure2_deepstarr_training_curves.pdf'

fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {png_path}")
print(f"Saved: {pdf_path}")
plt.close()
