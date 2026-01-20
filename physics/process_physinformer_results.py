#!/usr/bin/env python3
"""
Comprehensive script to process and organize PhysInformer training results.
Creates organized results folders with all relevant information, metrics, and visualizations.
"""

import os
import shutil
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from collections import defaultdict

def parse_epoch_results(epoch_file):
    """Parse epoch_results.txt to extract metrics for all epochs."""
    with open(epoch_file, 'r') as f:
        content = f.read()

    epochs = []
    epoch_blocks = content.split('=' * 80)

    for block in epoch_blocks:
        if 'EPOCH' not in block:
            continue

        epoch_data = {}

        # Extract epoch number
        epoch_match = re.search(r'EPOCH (\d+):', block)
        if epoch_match:
            epoch_data['epoch'] = int(epoch_match.group(1))

        # Extract losses
        train_loss_match = re.search(r'Train Total Loss.*?:\s+([\d.]+)', block)
        val_loss_match = re.search(r'Val Total Loss:\s+([\d.]+)', block)
        if train_loss_match:
            epoch_data['train_loss'] = float(train_loss_match.group(1))
        if val_loss_match:
            epoch_data['val_loss'] = float(val_loss_match.group(1))

        # Extract Pearson correlations
        train_pearson_match = re.search(r'Train Overall Pearson:\s+([\d.]+)', block)
        val_pearson_match = re.search(r'Val Overall Pearson:\s+([\d.]+)', block)
        if train_pearson_match:
            epoch_data['train_pearson'] = float(train_pearson_match.group(1))
        if val_pearson_match:
            epoch_data['val_pearson'] = float(val_pearson_match.group(1))

        # Extract descriptor Pearson distribution
        train_desc_match = re.search(
            r'Train: mean=([\d.-]+), median=([\d.-]+), range=\[([\d.-]+),\s+([\d.]+)\]',
            block
        )
        val_desc_match = re.search(
            r'Val: mean=([\d.-]+), median=([\d.-]+), range=\[([\d.-]+),\s+([\d.]+)\]',
            block
        )

        if train_desc_match:
            epoch_data['train_desc_mean'] = float(train_desc_match.group(1))
            epoch_data['train_desc_median'] = float(train_desc_match.group(2))
            epoch_data['train_desc_min'] = float(train_desc_match.group(3))
            epoch_data['train_desc_max'] = float(train_desc_match.group(4))

        if val_desc_match:
            epoch_data['val_desc_mean'] = float(val_desc_match.group(1))
            epoch_data['val_desc_median'] = float(val_desc_match.group(2))
            epoch_data['val_desc_min'] = float(val_desc_match.group(3))
            epoch_data['val_desc_max'] = float(val_desc_match.group(4))

        # Extract auxiliary head metrics
        aux_a_match = re.search(r'Head A.*?Pearson=([\d.]+).*?R²=([\d.]+).*?MSE=([\d.]+)', block)
        aux_b_match = re.search(r'Head B.*?Pearson=([\d.]+).*?R²=([\d.]+).*?MSE=([\d.]+)', block)

        if aux_a_match:
            epoch_data['aux_a_pearson'] = float(aux_a_match.group(1))
            epoch_data['aux_a_r2'] = float(aux_a_match.group(2))
            epoch_data['aux_a_mse'] = float(aux_a_match.group(3))

        if aux_b_match:
            epoch_data['aux_b_pearson'] = float(aux_b_match.group(1))
            epoch_data['aux_b_r2'] = float(aux_b_match.group(2))
            epoch_data['aux_b_mse'] = float(aux_b_match.group(3))

        if epoch_data:
            epochs.append(epoch_data)

    return sorted(epochs, key=lambda x: x.get('epoch', 0))

def create_loss_curves(epochs, output_path):
    """Create comprehensive loss curve plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Progress - Loss and Metrics', fontsize=16, fontweight='bold')

    epoch_nums = [e['epoch'] for e in epochs if 'epoch' in e]

    # Plot 1: Total Loss
    ax = axes[0, 0]
    train_losses = [e['train_loss'] for e in epochs if 'train_loss' in e]
    val_losses = [e['val_loss'] for e in epochs if 'val_loss' in e]

    if train_losses:
        ax.plot(epoch_nums, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    if val_losses:
        ax.plot(epoch_nums, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Overall Pearson Correlation
    ax = axes[0, 1]
    train_pearsons = [e['train_pearson'] for e in epochs if 'train_pearson' in e]
    val_pearsons = [e['val_pearson'] for e in epochs if 'val_pearson' in e]

    if train_pearsons:
        ax.plot(epoch_nums, train_pearsons, 'b-o', label='Train Pearson', linewidth=2, markersize=4)
    if val_pearsons:
        ax.plot(epoch_nums, val_pearsons, 'r-s', label='Val Pearson', linewidth=2, markersize=4)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Overall Pearson Correlation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 3: Descriptor Pearson Distribution (Mean/Median)
    ax = axes[1, 0]
    train_desc_means = [e['train_desc_mean'] for e in epochs if 'train_desc_mean' in e]
    val_desc_means = [e['val_desc_mean'] for e in epochs if 'val_desc_mean' in e]
    train_desc_medians = [e['train_desc_median'] for e in epochs if 'train_desc_median' in e]
    val_desc_medians = [e['val_desc_median'] for e in epochs if 'val_desc_median' in e]

    if train_desc_means:
        ax.plot(epoch_nums, train_desc_means, 'b-o', label='Train Mean', linewidth=2, markersize=4)
    if val_desc_means:
        ax.plot(epoch_nums, val_desc_means, 'r-s', label='Val Mean', linewidth=2, markersize=4)
    if train_desc_medians:
        ax.plot(epoch_nums, train_desc_medians, 'b--^', label='Train Median', linewidth=2, markersize=4, alpha=0.7)
    if val_desc_medians:
        ax.plot(epoch_nums, val_desc_medians, 'r--v', label='Val Median', linewidth=2, markersize=4, alpha=0.7)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Descriptor Pearson', fontsize=12)
    ax.set_title('Descriptor-Level Pearson Statistics', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Auxiliary Head Performance
    ax = axes[1, 1]
    aux_a_pearsons = [e['aux_a_pearson'] for e in epochs if 'aux_a_pearson' in e]
    aux_b_pearsons = [e['aux_b_pearson'] for e in epochs if 'aux_b_pearson' in e]

    if aux_a_pearsons:
        ax.plot(epoch_nums, aux_a_pearsons, 'g-o', label='Aux Head A (Seq+Feat)', linewidth=2, markersize=4)
    if aux_b_pearsons:
        ax.plot(epoch_nums, aux_b_pearsons, 'm-s', label='Aux Head B (Feat Only)', linewidth=2, markersize=4)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Auxiliary Heads Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created loss curves: {output_path}")

def create_pearson_evolution_plot(epochs, output_path):
    """Create plot showing evolution of Pearson distribution across epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Evolution of Descriptor Pearson Distribution', fontsize=16, fontweight='bold')

    epoch_nums = [e['epoch'] for e in epochs if 'epoch' in e]

    for idx, split in enumerate(['train', 'val']):
        ax = axes[idx]

        means = [e[f'{split}_desc_mean'] for e in epochs if f'{split}_desc_mean' in e]
        medians = [e[f'{split}_desc_median'] for e in epochs if f'{split}_desc_median' in e]
        mins = [e[f'{split}_desc_min'] for e in epochs if f'{split}_desc_min' in e]
        maxs = [e[f'{split}_desc_max'] for e in epochs if f'{split}_desc_max' in e]

        # Plot mean and median
        if means:
            ax.plot(epoch_nums, means, 'b-o', label='Mean', linewidth=2.5, markersize=6)
        if medians:
            ax.plot(epoch_nums, medians, 'r-s', label='Median', linewidth=2.5, markersize=6)

        # Fill between min and max
        if mins and maxs:
            ax.fill_between(epoch_nums, mins, maxs, alpha=0.2, color='gray', label='Min-Max Range')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Pearson Correlation', fontsize=12)
        ax.set_title(f'{split.capitalize()} Set Descriptor Pearson Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created Pearson evolution plot: {output_path}")

def extract_final_descriptor_scores(run_dir, output_path):
    """Extract final Pearson scores for each descriptor from the last epoch."""
    plots_dir = os.path.join(run_dir, 'plots')

    # Find the last epoch feature scores file
    feature_score_files = sorted(
        [f for f in os.listdir(plots_dir) if f.startswith('feature_scores_epoch_') and f.endswith('_val.txt')],
        key=lambda x: int(re.search(r'epoch_(\d+)_', x).group(1))
    )

    if not feature_score_files:
        print(f"No feature score files found in {plots_dir}")
        return

    last_file = os.path.join(plots_dir, feature_score_files[-1])
    epoch_num = re.search(r'epoch_(\d+)_', feature_score_files[-1]).group(1)

    # Parse the file
    descriptors = []
    with open(last_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Rank') or line.startswith('===') or line.startswith('---'):
                continue

            # Parse line format: "  123. descriptor_name: 0.1234"
            match = re.match(r'\s*\d+\.\s+(.+?):\s+([-\d.]+)', line)
            if match:
                desc_name = match.group(1).strip()
                pearson = float(match.group(2))
                descriptors.append({'descriptor': desc_name, 'pearson': pearson})

    # Save to file
    with open(output_path, 'w') as f:
        f.write(f"Final Descriptor Pearson Scores (Epoch {epoch_num} - Validation Set)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Descriptors: {len(descriptors)}\n\n")

        if descriptors:
            pearsons = [d['pearson'] for d in descriptors]
            f.write(f"Statistics:\n")
            f.write(f"  Mean: {np.mean(pearsons):.4f}\n")
            f.write(f"  Median: {np.median(pearsons):.4f}\n")
            f.write(f"  Std Dev: {np.std(pearsons):.4f}\n")
            f.write(f"  Min: {np.min(pearsons):.4f}\n")
            f.write(f"  Max: {np.max(pearsons):.4f}\n")
            f.write(f"  Q1 (25th percentile): {np.percentile(pearsons, 25):.4f}\n")
            f.write(f"  Q3 (75th percentile): {np.percentile(pearsons, 75):.4f}\n\n")

        f.write("\nAll Descriptors (sorted by Pearson correlation):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Descriptor':<60} {'Pearson':>10}\n")
        f.write("-" * 80 + "\n")

        # Sort by Pearson (descending)
        descriptors_sorted = sorted(descriptors, key=lambda x: x['pearson'], reverse=True)
        for rank, desc in enumerate(descriptors_sorted, 1):
            f.write(f"{rank:<6} {desc['descriptor']:<60} {desc['pearson']:>10.6f}\n")

    print(f"Extracted final descriptor scores to: {output_path}")
    return descriptors

def create_performance_summary(epochs, run_dir, output_path):
    """Create comprehensive performance summary document."""
    if not epochs:
        return

    last_epoch = epochs[-1]
    best_val_epoch = max(epochs, key=lambda x: x.get('val_pearson', 0))

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PHYSINFORMER MODEL PERFORMANCE SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Training configuration
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 80 + "\n")

        # Try to extract from training.log
        log_file = os.path.join(run_dir, 'training.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as log:
                for line in log:
                    if 'Arguments:' in line:
                        f.write(f"Arguments: {line.split('Arguments:')[1].strip()}\n")
                        break

        f.write(f"Total Epochs Trained: {len(epochs)}\n")
        f.write(f"Run Directory: {run_dir}\n\n")

        # Final epoch performance
        f.write("FINAL EPOCH PERFORMANCE (Epoch {}):\n".format(last_epoch.get('epoch', '?')))
        f.write("-" * 80 + "\n")
        f.write("Overall Metrics:\n")
        train_loss = last_epoch.get('train_loss', 'N/A')
        val_loss = last_epoch.get('val_loss', 'N/A')
        f.write(f"  Train Loss: {train_loss if isinstance(train_loss, str) else f'{train_loss:.4f}'}\n")
        f.write(f"  Val Loss: {val_loss if isinstance(val_loss, str) else f'{val_loss:.4f}'}\n")

        train_pearson = last_epoch.get('train_pearson', 'N/A')
        val_pearson = last_epoch.get('val_pearson', 'N/A')
        f.write(f"  Train Pearson: {train_pearson if isinstance(train_pearson, str) else f'{train_pearson:.4f}'}\n")
        f.write(f"  Val Pearson: {val_pearson if isinstance(val_pearson, str) else f'{val_pearson:.4f}'}\n\n")

        f.write("Descriptor-Level Performance:\n")
        for key, label in [('train_desc_mean', 'Train Mean Pearson'),
                           ('train_desc_median', 'Train Median Pearson'),
                           ('val_desc_mean', 'Val Mean Pearson'),
                           ('val_desc_median', 'Val Median Pearson')]:
            val = last_epoch.get(key, 'N/A')
            f.write(f"  {label}: {val if isinstance(val, str) else f'{val:.4f}'}\n")

        train_min = last_epoch.get('train_desc_min', 'N/A')
        train_max = last_epoch.get('train_desc_max', 'N/A')
        val_min = last_epoch.get('val_desc_min', 'N/A')
        val_max = last_epoch.get('val_desc_max', 'N/A')

        f.write(f"  Train Range: [")
        f.write(f"{train_min if isinstance(train_min, str) else f'{train_min:.4f}'}, ")
        f.write(f"{train_max if isinstance(train_max, str) else f'{train_max:.4f}'}]\n")
        f.write(f"  Val Range: [")
        f.write(f"{val_min if isinstance(val_min, str) else f'{val_min:.4f}'}, ")
        f.write(f"{val_max if isinstance(val_max, str) else f'{val_max:.4f}'}]\n\n")

        f.write("Auxiliary Head Performance (Val):\n")
        aux_a_p = last_epoch.get('aux_a_pearson', 'N/A')
        aux_a_r = last_epoch.get('aux_a_r2', 'N/A')
        aux_a_m = last_epoch.get('aux_a_mse', 'N/A')
        aux_b_p = last_epoch.get('aux_b_pearson', 'N/A')
        aux_b_r = last_epoch.get('aux_b_r2', 'N/A')
        aux_b_m = last_epoch.get('aux_b_mse', 'N/A')

        f.write(f"  Head A (Seq+Feat): ")
        f.write(f"Pearson={aux_a_p if isinstance(aux_a_p, str) else f'{aux_a_p:.4f}'}, ")
        f.write(f"R²={aux_a_r if isinstance(aux_a_r, str) else f'{aux_a_r:.4f}'}, ")
        f.write(f"MSE={aux_a_m if isinstance(aux_a_m, str) else f'{aux_a_m:.4f}'}\n")
        f.write(f"  Head B (Feat Only): ")
        f.write(f"Pearson={aux_b_p if isinstance(aux_b_p, str) else f'{aux_b_p:.4f}'}, ")
        f.write(f"R²={aux_b_r if isinstance(aux_b_r, str) else f'{aux_b_r:.4f}'}, ")
        f.write(f"MSE={aux_b_m if isinstance(aux_b_m, str) else f'{aux_b_m:.4f}'}\n\n")

        # Best validation performance
        f.write("BEST VALIDATION PERFORMANCE (Epoch {}):\n".format(best_val_epoch.get('epoch', '?')))
        f.write("-" * 80 + "\n")
        best_val_p = best_val_epoch.get('val_pearson', 'N/A')
        best_val_l = best_val_epoch.get('val_loss', 'N/A')
        best_desc_mean = best_val_epoch.get('val_desc_mean', 'N/A')
        best_desc_median = best_val_epoch.get('val_desc_median', 'N/A')

        f.write(f"  Val Pearson: {best_val_p if isinstance(best_val_p, str) else f'{best_val_p:.4f}'}\n")
        f.write(f"  Val Loss: {best_val_l if isinstance(best_val_l, str) else f'{best_val_l:.4f}'}\n")
        f.write(f"  Val Descriptor Mean Pearson: {best_desc_mean if isinstance(best_desc_mean, str) else f'{best_desc_mean:.4f}'}\n")
        f.write(f"  Val Descriptor Median Pearson: {best_desc_median if isinstance(best_desc_median, str) else f'{best_desc_median:.4f}'}\n\n")

        # Training progress summary
        f.write("TRAINING PROGRESS SUMMARY:\n")
        f.write("-" * 80 + "\n")

        first_epoch = epochs[0]
        f.write(f"Improvement from Epoch 1 to Epoch {last_epoch.get('epoch', '?')}:\n")

        if 'val_pearson' in first_epoch and 'val_pearson' in last_epoch:
            improvement = last_epoch['val_pearson'] - first_epoch['val_pearson']
            f.write(f"  Val Pearson: {first_epoch['val_pearson']:.4f} → {last_epoch['val_pearson']:.4f} ")
            f.write(f"(+{improvement:.4f}, +{improvement/first_epoch['val_pearson']*100:.1f}%)\n")

        if 'val_desc_mean' in first_epoch and 'val_desc_mean' in last_epoch:
            improvement = last_epoch['val_desc_mean'] - first_epoch['val_desc_mean']
            f.write(f"  Val Desc Mean: {first_epoch['val_desc_mean']:.4f} → {last_epoch['val_desc_mean']:.4f} ")
            f.write(f"(+{improvement:.4f}, +{improvement/first_epoch['val_desc_mean']*100:.1f}%)\n")

        if 'val_loss' in first_epoch and 'val_loss' in last_epoch:
            improvement = first_epoch['val_loss'] - last_epoch['val_loss']
            f.write(f"  Val Loss: {first_epoch['val_loss']:.2f} → {last_epoch['val_loss']:.2f} ")
            f.write(f"(-{improvement:.2f}, {improvement/first_epoch['val_loss']*100:.1f}% reduction)\n")

        f.write("\n")

        # Epoch-by-epoch summary
        f.write("EPOCH-BY-EPOCH SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epoch':<7} {'Train Loss':<12} {'Val Loss':<12} {'Train Pearson':<15} {'Val Pearson':<15}\n")
        f.write("-" * 80 + "\n")

        for epoch in epochs[::5]:  # Every 5th epoch to keep it manageable
            f.write(f"{epoch.get('epoch', '?'):<7} ")
            f.write(f"{epoch.get('train_loss', 'N/A'):<12.2f} " if 'train_loss' in epoch else "N/A          ")
            f.write(f"{epoch.get('val_loss', 'N/A'):<12.2f} " if 'val_loss' in epoch else "N/A          ")
            f.write(f"{epoch.get('train_pearson', 'N/A'):<15.4f} " if 'train_pearson' in epoch else "N/A             ")
            f.write(f"{epoch.get('val_pearson', 'N/A'):<15.4f}\n" if 'val_pearson' in epoch else "N/A\n")

        # Add last epoch if not already included
        if len(epochs) % 5 != 1:
            epoch = last_epoch
            f.write(f"{epoch.get('epoch', '?'):<7} ")
            f.write(f"{epoch.get('train_loss', 'N/A'):<12.2f} " if 'train_loss' in epoch else "N/A          ")
            f.write(f"{epoch.get('val_loss', 'N/A'):<12.2f} " if 'val_loss' in epoch else "N/A          ")
            f.write(f"{epoch.get('train_pearson', 'N/A'):<15.4f} " if 'train_pearson' in epoch else "N/A             ")
            f.write(f"{epoch.get('val_pearson', 'N/A'):<15.4f}\n" if 'val_pearson' in epoch else "N/A\n")

    print(f"Created performance summary: {output_path}")

def process_physinformer_model(cell_type, run_dir, output_dir):
    """Process a single PhysInformer model and organize results."""
    print(f"\n{'='*80}")
    print(f"Processing {cell_type} PhysInformer Model")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Copy training logs
    print("\n1. Copying training logs...")
    for log_file in ['training.log', 'batch_log.txt', 'epoch_results.txt']:
        src = os.path.join(run_dir, log_file)
        if os.path.exists(src):
            dst = os.path.join(output_dir, log_file)
            shutil.copy2(src, dst)
            print(f"   Copied: {log_file}")

    # 2. Parse epoch results
    print("\n2. Parsing epoch results...")
    epoch_file = os.path.join(run_dir, 'epoch_results.txt')
    if os.path.exists(epoch_file):
        epochs = parse_epoch_results(epoch_file)
        print(f"   Parsed {len(epochs)} epochs")

        # Save parsed data as JSON
        with open(os.path.join(output_dir, 'parsed_epochs.json'), 'w') as f:
            json.dump(epochs, f, indent=2)
    else:
        print("   epoch_results.txt not found!")
        epochs = []

    # 3. Generate loss curves
    print("\n3. Generating loss curves...")
    if epochs:
        create_loss_curves(epochs, os.path.join(output_dir, 'loss_curves.png'))

    # 4. Generate Pearson evolution plot
    print("\n4. Generating Pearson evolution plot...")
    if epochs:
        create_pearson_evolution_plot(epochs, os.path.join(output_dir, 'pearson_evolution.png'))

    # 5. Copy existing plots
    print("\n5. Copying existing training plots...")
    plots_dir = os.path.join(run_dir, 'plots')
    if os.path.exists(plots_dir):
        output_plots_dir = os.path.join(output_dir, 'training_plots')
        os.makedirs(output_plots_dir, exist_ok=True)

        # Copy training progress and metrics table
        for plot_file in ['training_progress.png', 'metrics_table_current.png']:
            src = os.path.join(plots_dir, plot_file)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_plots_dir, plot_file))
                print(f"   Copied: {plot_file}")

        # Copy Pearson distribution plots
        pearson_plots = [f for f in os.listdir(plots_dir) if 'pearson_distribution' in f and f.endswith('.png')]

        # Copy first, middle, and last epoch Pearson distributions
        if pearson_plots:
            pearson_plots_sorted = sorted(pearson_plots, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))

            # First epoch
            for p in pearson_plots_sorted[:2]:  # train and val
                shutil.copy2(os.path.join(plots_dir, p), os.path.join(output_plots_dir, p))
                print(f"   Copied: {p}")

            # Middle epoch
            mid_idx = len(pearson_plots_sorted) // 2
            if mid_idx > 2:
                for p in pearson_plots_sorted[mid_idx:mid_idx+2]:
                    shutil.copy2(os.path.join(plots_dir, p), os.path.join(output_plots_dir, p))
                    print(f"   Copied: {p}")

            # Last epoch
            for p in pearson_plots_sorted[-2:]:
                shutil.copy2(os.path.join(plots_dir, p), os.path.join(output_plots_dir, p))
                print(f"   Copied: {p}")

    # 6. Extract final descriptor scores
    print("\n6. Extracting final descriptor Pearson scores...")
    extract_final_descriptor_scores(run_dir, os.path.join(output_dir, 'final_descriptor_scores.txt'))

    # 7. Create performance summary
    print("\n7. Creating performance summary...")
    if epochs:
        create_performance_summary(epochs, run_dir, os.path.join(output_dir, 'PERFORMANCE_SUMMARY.txt'))

    # 8. Copy best model info
    print("\n8. Copying best model...")
    best_model_src = os.path.join(run_dir, 'best_model.pt')
    if os.path.exists(best_model_src):
        # Don't copy the actual model file (too large), just create a reference
        with open(os.path.join(output_dir, 'BEST_MODEL_LOCATION.txt'), 'w') as f:
            f.write(f"Best model location: {best_model_src}\n")
            f.write(f"Model size: {os.path.getsize(best_model_src) / (1024*1024):.2f} MB\n")
        print(f"   Created best model reference")

    print(f"\nCompleted processing {cell_type}!")

def main():
    """Main processing function for all PhysInformer models."""
    base_dir = '/home/bcheng/sequence_optimization/mainproject/PhysiFormer/physpreprocess'
    results_base = os.path.join(base_dir, 'results')

    models = {
        'S2': 'PhysInformer/runs/S2_20250831_025144',
        'WTC11': 'PhysInformer/runs/WTC11_20250829_095738',
        'HepG2': 'PhysInformer/runs/HepG2_20250829_095749',
        'K562': 'PhysInformer/runs/K562_20250829_095741'
    }

    for cell_type, run_path in models.items():
        run_dir = os.path.join(base_dir, run_path)
        output_dir = os.path.join(results_base, f'PhysInformer_{cell_type}')

        if os.path.exists(run_dir):
            process_physinformer_model(cell_type, run_dir, output_dir)
        else:
            print(f"Warning: Run directory not found for {cell_type}: {run_dir}")

    print("\n" + "="*80)
    print("ALL PHYSINFORMER MODELS PROCESSED SUCCESSFULLY!")
    print("="*80)

if __name__ == '__main__':
    main()
