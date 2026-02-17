#!/usr/bin/env python3
"""
Generate Figure 3: CADENCE Training Dynamics -- Plant MPRA Datasets
3x3 composite figure (3 species x 3 metrics)
"""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Parse training logs
# =============================================================================

def parse_log(filepath):
    """Parse a CADENCE training log and extract per-epoch metrics."""
    epochs = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        # Look for EPOCH summary lines
        match = re.search(r'EPOCH (\d+) SUMMARY', line)
        if match:
            epoch_num = int(match.group(1))
            is_best = '(BEST)' in line
            # Next lines contain TRAIN and VAL metrics
            train_data = None
            val_data = None
            for j in range(i+1, min(i+10, len(lines))):
                tline = lines[j]
                # Parse TRAIN line
                tmatch = re.search(
                    r'\[TRAIN\]\s+NLL:\s+([\d.]+)\s+\|\s+MSE:\s+([\d.-]+)\s+\|\s+r:\s+([\d.]+)\s+\|\s+rho:\s+([\d.]+)',
                    tline
                )
                if tmatch:
                    train_data = {
                        'nll': float(tmatch.group(1)),
                        'r': float(tmatch.group(3)),
                        'rho': float(tmatch.group(4)),
                    }
                # Parse VAL line (with extra spaces before NLL)
                vmatch = re.search(
                    r'\[VAL\]\s+NLL:\s+([\d.]+)\s+\|\s+MSE:\s+([\d.-]+)\s+\|\s+r:\s+([\d.]+)\s+\|\s+rho:\s+([\d.]+)',
                    tline
                )
                if vmatch:
                    val_data = {
                        'nll': float(vmatch.group(1)),
                        'r': float(vmatch.group(3)),
                        'rho': float(vmatch.group(4)),
                    }
            if train_data and val_data:
                epochs.append({
                    'epoch': epoch_num,
                    'is_best': is_best,
                    'train': train_data,
                    'val': val_data,
                })
        i += 1
    return epochs


# Parse all logs
maize_data = parse_log('/home/bcheng/sequence_optimization/FUSEMAP/training/results/cadence_maize_v1/training.log')
sorghum_data = parse_log('/home/bcheng/sequence_optimization/FUSEMAP/training/results/cadence_sorghum_v1/training.log')
arabidopsis_data = parse_log('/home/bcheng/sequence_optimization/FUSEMAP/training/results/cadence_arabidopsis_v1/training.log')

print(f"Maize: {len(maize_data)} epochs")
print(f"Sorghum: {len(sorghum_data)} epochs")
print(f"Arabidopsis: {len(arabidopsis_data)} epochs")

# =============================================================================
# Extract arrays
# =============================================================================

def extract_arrays(data):
    epochs = [d['epoch'] for d in data]
    train_nll = [d['train']['nll'] for d in data]
    val_nll = [d['val']['nll'] for d in data]
    train_r = [d['train']['r'] for d in data]
    val_r = [d['val']['r'] for d in data]
    train_rho = [d['train']['rho'] for d in data]
    val_rho = [d['val']['rho'] for d in data]
    # Find best val epoch (lowest val NLL)
    best_idx = np.argmin(val_nll)
    return {
        'epochs': epochs,
        'train_nll': train_nll, 'val_nll': val_nll,
        'train_r': train_r, 'val_r': val_r,
        'train_rho': train_rho, 'val_rho': val_rho,
        'best_idx': best_idx,
    }


maize = extract_arrays(maize_data)
sorghum = extract_arrays(sorghum_data)
arabidopsis = extract_arrays(arabidopsis_data)

# Print best epochs
for name, d in [('Maize', maize), ('Sorghum', sorghum), ('Arabidopsis', arabidopsis)]:
    bi = d['best_idx']
    print(f"{name} best val epoch: {d['epochs'][bi]} "
          f"(NLL={d['val_nll'][bi]:.4f}, r={d['val_r'][bi]:.4f}, rho={d['val_rho'][bi]:.4f})")

# =============================================================================
# Plot
# =============================================================================

species_info = [
    ('Maize (n=2,461)', maize, '#5BB75B', {'r': 0.796, 'rho': 0.799}),
    ('Sorghum (n=1,968)', sorghum, '#2E8B57', {'r': 0.782, 'rho': 0.777}),
    ('Arabidopsis (n=1,347)', arabidopsis, '#6B8E23', {'r': 0.618, 'rho': 0.591}),
]

col_titles = ['NLL Loss', 'Pearson $r$', 'Spearman $\\rho$']

fig, axes = plt.subplots(3, 3, figsize=(16, 14))

for row_idx, (row_label, data, color, test_perf) in enumerate(species_info):
    for col_idx in range(3):
        ax = axes[row_idx, col_idx]

        epochs = data['epochs']
        bi = data['best_idx']

        if col_idx == 0:
            train_y = data['train_nll']
            val_y = data['val_nll']
            ylabel = 'NLL Loss'
            # Best is minimum for NLL
            best_val = val_y[bi]
        elif col_idx == 1:
            train_y = data['train_r']
            val_y = data['val_r']
            ylabel = 'Pearson $r$'
            best_val = val_y[bi]
        else:
            train_y = data['train_rho']
            val_y = data['val_rho']
            ylabel = 'Spearman $\\rho$'
            best_val = val_y[bi]

        # Plot lines
        ax.plot(epochs, train_y, color=color, linewidth=2, linestyle='-',
                label='Train', alpha=0.9)
        ax.plot(epochs, val_y, color=color, linewidth=2, linestyle='--',
                label='Val', alpha=0.9)

        # Mark best val epoch with star
        ax.plot(epochs[bi], best_val, marker='*', markersize=14,
                color=color, markeredgecolor='black', markeredgewidth=0.8,
                zorder=5)

        # Annotate test performance on Pearson and Spearman panels
        if col_idx == 1:
            ax.annotate(
                f"Test $r$ = {test_perf['r']:.3f}",
                xy=(0.97, 0.06), xycoords='axes fraction',
                fontsize=10, fontweight='bold', color='#333333',
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.85, linewidth=1.2),
            )
        elif col_idx == 2:
            ax.annotate(
                f"Test $\\rho$ = {test_perf['rho']:.3f}",
                xy=(0.97, 0.06), xycoords='axes fraction',
                fontsize=10, fontweight='bold', color='#333333',
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.85, linewidth=1.2),
            )

        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linestyle='-')
        ax.set_axisbelow(True)

        # Column titles (top row only)
        if row_idx == 0:
            ax.set_title(col_titles[col_idx], fontsize=15, fontweight='bold', pad=10)

        # Row labels (left column only)
        if col_idx == 0:
            ax.set_ylabel(row_label, fontsize=13, fontweight='bold', labelpad=10)

        # X label (bottom row only)
        if row_idx == 2:
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')

        # Tick label sizes
        ax.tick_params(axis='both', labelsize=10)

        # Legend (only once per row, in NLL column)
        if col_idx == 0:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.85,
                      edgecolor='gray')

# Suptitle
fig.suptitle(
    'Figure 3. CADENCE Training Dynamics \u2014 Plant MPRA Datasets',
    fontsize=17, fontweight='bold', y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
outdir = '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/3.1_cadence_single_task'
fig.savefig(f'{outdir}/figure3_plant_training_curves.png', dpi=200, bbox_inches='tight',
            facecolor='white')
fig.savefig(f'{outdir}/figure3_plant_training_curves.pdf', dpi=200, bbox_inches='tight',
            facecolor='white')
print(f"\nSaved to {outdir}/figure3_plant_training_curves.png")
print(f"Saved to {outdir}/figure3_plant_training_curves.pdf")
plt.close()
