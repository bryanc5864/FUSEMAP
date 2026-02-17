#!/usr/bin/env python3
"""
Figure 7: TileFormer Training Dynamics and Per-Metric Convergence.

Parses real per-epoch metrics from TileFormer training run (run_20250819_063725)
and per-step training losses. Produces a 2x3 panel figure showing:
  (A) Training loss curve (mean per-step loss per epoch)
  (B) Validation MSE per epoch
  (C) Train vs Val loss on log scale
  (D) Per-metric R^2 across epochs
  (E) Per-metric Pearson r across epochs
  (F) Overall validation MSE vs per-metric R^2 summary
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import re

# ── CONFIGURATION ──────────────────────────────────────────────────────────
CHECKPOINT_DIR = '/home/bcheng/sequence_optimization/FUSEMAP/physics/TileFormer/checkpoints/run_20250819_063725'
OUTPUT_DIR = '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures'
N_EPOCHS = 25

METRIC_NAMES = [
    'STD_PSI_MIN', 'STD_PSI_MAX', 'STD_PSI_MEAN',
    'ENH_PSI_MIN', 'ENH_PSI_MAX', 'ENH_PSI_MEAN',
]
METRIC_LABELS = [
    'STD PSI Min', 'STD PSI Max', 'STD PSI Mean',
    'ENH PSI Min', 'ENH PSI Max', 'ENH PSI Mean',
]

# Colors for per-metric lines
METRIC_COLORS = [
    '#4A90D9',  # blue
    '#E74C3C',  # red
    '#2ECC71',  # green
    '#F39C12',  # amber
    '#9B59B6',  # purple
    '#1ABC9C',  # teal
]

TRAIN_COLOR = '#4A90D9'
VAL_COLOR = '#E74C3C'

# ── PARSE EPOCH METRICS ───────────────────────────────────────────────────
def parse_metrics_file(filepath):
    """Parse a metrics_epoch_N.txt file and return a dict of dicts."""
    metrics = {}
    current_section = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Detect section headers like "STD_PSI_MIN Metrics:"
            if line.endswith('Metrics:') and not line.startswith('='):
                current_section = line.replace(' Metrics:', '')
            # Parse key-value lines
            m = re.match(r'(\w+)\s*:\s*(.+)', line)
            if m and current_section:
                key = m.group(1)
                try:
                    val = float(m.group(2))
                except ValueError:
                    continue
                if current_section not in metrics:
                    metrics[current_section] = {}
                metrics[current_section][key] = val
    return metrics


def parse_step_losses(filepath):
    """Parse a step_losses_epoch_N.txt file and return array of floats."""
    losses = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    losses.append(float(line))
                except ValueError:
                    pass
    return np.array(losses)


# Collect per-epoch data
epochs = list(range(1, N_EPOCHS + 1))

# Per-metric R^2 and Pearson r, and overall MSE
per_metric_r2 = {name: [] for name in METRIC_NAMES}
per_metric_pearson = {name: [] for name in METRIC_NAMES}
overall_mse_val = []

# Training step losses  -> mean per epoch
train_loss_per_epoch = []

for ep in epochs:
    # --- Validation metrics ---
    mf = os.path.join(CHECKPOINT_DIR, f'metrics_epoch_{ep}.txt')
    if os.path.exists(mf):
        m = parse_metrics_file(mf)
        for name in METRIC_NAMES:
            if name in m:
                per_metric_r2[name].append(m[name].get('r2', np.nan))
                per_metric_pearson[name].append(m[name].get('pearson_r', np.nan))
            else:
                per_metric_r2[name].append(np.nan)
                per_metric_pearson[name].append(np.nan)
        if 'OVERALL' in m:
            overall_mse_val.append(m['OVERALL'].get('mse', np.nan))
        else:
            overall_mse_val.append(np.nan)
    else:
        for name in METRIC_NAMES:
            per_metric_r2[name].append(np.nan)
            per_metric_pearson[name].append(np.nan)
        overall_mse_val.append(np.nan)

    # --- Training step losses ---
    sf = os.path.join(CHECKPOINT_DIR, f'step_losses_epoch_{ep}.txt')
    if os.path.exists(sf):
        losses = parse_step_losses(sf)
        if len(losses) > 0:
            train_loss_per_epoch.append(np.mean(losses))
        else:
            train_loss_per_epoch.append(np.nan)
    else:
        train_loss_per_epoch.append(np.nan)

epochs = np.array(epochs)
train_loss = np.array(train_loss_per_epoch)
val_mse = np.array(overall_mse_val)

# ── FIGURE ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Shared styling helper
def style_ax(ax, xlabel, ylabel, title, panel_label=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.tick_params(labelsize=10)
    if panel_label:
        ax.text(-0.10, 1.05, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')


# ── (A) Training Loss per Epoch ───────────────────────────────────────────
ax = axes[0, 0]
ax.plot(epochs, train_loss, '-o', color=TRAIN_COLOR, markersize=5,
        linewidth=2.0, markeredgecolor='white', markeredgewidth=0.8,
        label='Training loss')
ax.fill_between(epochs, train_loss, alpha=0.12, color=TRAIN_COLOR)
style_ax(ax, 'Epoch', 'Mean Step Loss', 'Training Loss', 'A')
ax.set_xlim(0.5, 25.5)
ax.legend(fontsize=10, framealpha=0.9, loc='upper right')

# ── (B) Validation MSE per Epoch ──────────────────────────────────────────
ax = axes[0, 1]
ax.plot(epochs, val_mse, '-s', color=VAL_COLOR, markersize=5,
        linewidth=2.0, markeredgecolor='white', markeredgewidth=0.8,
        label='Validation MSE')
ax.fill_between(epochs, val_mse, alpha=0.12, color=VAL_COLOR)
style_ax(ax, 'Epoch', 'Overall MSE', 'Validation Loss', 'B')
ax.set_xlim(0.5, 25.5)
ax.legend(fontsize=10, framealpha=0.9, loc='upper right')

# ── (C) Train vs Val loss on Log Scale ────────────────────────────────────
ax = axes[0, 2]
ax.semilogy(epochs, train_loss, '-o', color=TRAIN_COLOR, markersize=5,
            linewidth=2.0, markeredgecolor='white', markeredgewidth=0.8,
            label='Train loss')
ax.semilogy(epochs, val_mse, '-s', color=VAL_COLOR, markersize=5,
            linewidth=2.0, markeredgecolor='white', markeredgewidth=0.8,
            label='Val MSE')
style_ax(ax, 'Epoch', 'Loss (log scale)', 'Train vs. Validation (Log Scale)', 'C')
ax.set_xlim(0.5, 25.5)
ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
# Add annotation for convergence region
ax.axhspan(5e-5, 2e-4, alpha=0.08, color='green', zorder=0)
ax.text(20, 1.1e-4, 'converged', fontsize=9, color='#2d7d2d', fontstyle='italic',
        ha='center', va='center')

# ── (D) Per-Metric R^2 across Epochs ─────────────────────────────────────
ax = axes[1, 0]
for i, name in enumerate(METRIC_NAMES):
    r2_vals = np.array(per_metric_r2[name])
    ax.plot(epochs, r2_vals, '-', color=METRIC_COLORS[i], linewidth=2.0,
            label=METRIC_LABELS[i], alpha=0.9)
style_ax(ax, 'Epoch', r'$R^2$', r'Per-Metric $R^2$ Convergence', 'D')
ax.set_xlim(0.5, 25.5)
ax.set_ylim(-0.3, 1.02)
ax.axhline(y=0.96, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(1.5, 0.965, r'$R^2 = 0.96$', fontsize=8.5, color='gray', fontstyle='italic')
ax.legend(fontsize=8.5, framealpha=0.9, ncol=2, loc='lower right',
          borderaxespad=0.8)

# ── (E) Per-Metric Pearson r across Epochs ────────────────────────────────
ax = axes[1, 1]
for i, name in enumerate(METRIC_NAMES):
    pr_vals = np.array(per_metric_pearson[name])
    ax.plot(epochs, pr_vals, '-', color=METRIC_COLORS[i], linewidth=2.0,
            label=METRIC_LABELS[i], alpha=0.9)
style_ax(ax, 'Epoch', 'Pearson r', 'Per-Metric Pearson r Convergence', 'E')
ax.set_xlim(0.5, 25.5)
ax.set_ylim(-0.1, 1.02)
ax.axhline(y=0.98, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(1.5, 0.983, r'$r = 0.98$', fontsize=8.5, color='gray', fontstyle='italic')
ax.legend(fontsize=8.5, framealpha=0.9, ncol=2, loc='lower right',
          borderaxespad=0.8)

# ── (F) Final Epoch Summary: R^2 Bar Chart + Convergence Rate ────────────
ax = axes[1, 2]

# Get final epoch metrics
final_r2 = [per_metric_r2[name][-1] for name in METRIC_NAMES]
final_pearson = [per_metric_pearson[name][-1] for name in METRIC_NAMES]

x_bar = np.arange(len(METRIC_NAMES))
bar_w = 0.35

bars_r2 = ax.bar(x_bar - bar_w/2, final_r2, bar_w, color=METRIC_COLORS,
                  edgecolor='white', linewidth=0.8, label=r'$R^2$', alpha=0.85)
bars_pr = ax.bar(x_bar + bar_w/2, final_pearson, bar_w,
                  color=METRIC_COLORS, edgecolor='black', linewidth=0.8,
                  label='Pearson r', hatch='///', alpha=0.55)

# Value labels
for j, (r2_v, pr_v) in enumerate(zip(final_r2, final_pearson)):
    ax.text(x_bar[j] - bar_w/2, r2_v + 0.005, f'{r2_v:.3f}',
            ha='center', va='bottom', fontsize=7.5, fontweight='bold',
            color=METRIC_COLORS[j])
    ax.text(x_bar[j] + bar_w/2, pr_v + 0.005, f'{pr_v:.3f}',
            ha='center', va='bottom', fontsize=7.5, fontweight='bold',
            color='#333333')

style_ax(ax, '', 'Score', 'Final Epoch Metrics (Epoch 25)', 'F')
ax.set_xticks(x_bar)
ax.set_xticklabels([l.replace(' ', '\n') for l in METRIC_LABELS],
                    fontsize=8.5, fontweight='bold')
ax.set_ylim(0.90, 1.005)
ax.axhline(y=0.96, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# Custom legend for R^2 vs Pearson
import matplotlib.patches as mpatches
r2_patch = mpatches.Patch(facecolor='#aaaaaa', edgecolor='white',
                           label=r'$R^2$')
pr_patch = mpatches.Patch(facecolor='#aaaaaa', edgecolor='black',
                           hatch='///', alpha=0.55, label='Pearson r')
ax.legend(handles=[r2_patch, pr_patch], fontsize=9.5, framealpha=0.9,
          loc='lower left')

# ── SUPTITLE AND SAVE ─────────────────────────────────────────────────────
fig.suptitle('Figure 7. TileFormer Training Dynamics and Per-Metric Convergence',
             fontsize=16, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_png = os.path.join(OUTPUT_DIR, 'figure7_tileformer_training.png')
out_pdf = os.path.join(OUTPUT_DIR, 'figure7_tileformer_training.pdf')
fig.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_png}')
print(f'Saved: {out_pdf}')
plt.close(fig)
