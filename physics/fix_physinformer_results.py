#!/usr/bin/env python3
"""
FIXED script to properly process PhysInformer training results.
Correctly parses all metrics from epoch_results.txt and includes every single epoch.
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

def parse_epoch_results_correctly(epoch_file):
    """Parse epoch_results.txt with correct parsing for ALL fields."""
    with open(epoch_file, 'r') as f:
        content = f.read()

    epochs = []

    # Split by the epoch headers
    epoch_blocks = re.split(r'={80,}\nEPOCH (\d+):\n={80,}', content)

    # Skip the first empty block
    for i in range(1, len(epoch_blocks), 2):
        if i+1 >= len(epoch_blocks):
            break

        epoch_num = int(epoch_blocks[i])
        block = epoch_blocks[i+1]

        epoch_data = {'epoch': epoch_num}

        # Parse LOSSES section
        train_total_loss = re.search(r'Train Total Loss.*?:\s+([\d.]+)', block)
        val_total_loss = re.search(r'Val Total Loss:\s+([\d.]+)', block)
        train_desc_loss = re.search(r'Train Desc Loss.*?:\s+([\d.]+)', block)
        val_desc_loss = re.search(r'Val Desc Loss:\s+([\d.]+)', block)
        train_aux_a = re.search(r'Train Aux A Loss:\s+([\d.]+)', block)
        train_aux_b = re.search(r'Train Aux B Loss:\s+([\d.]+)', block)
        val_aux_a = re.search(r'Val Aux A Loss:\s+([\d.]+)', block)
        val_aux_b = re.search(r'Val Aux B Loss:\s+([\d.]+)', block)

        if train_total_loss:
            epoch_data['train_total_loss'] = float(train_total_loss.group(1))
        if val_total_loss:
            epoch_data['val_total_loss'] = float(val_total_loss.group(1))
        if train_desc_loss:
            epoch_data['train_desc_loss'] = float(train_desc_loss.group(1))
        if val_desc_loss:
            epoch_data['val_desc_loss'] = float(val_desc_loss.group(1))
        if train_aux_a:
            epoch_data['train_aux_a_loss'] = float(train_aux_a.group(1))
        if train_aux_b:
            epoch_data['train_aux_b_loss'] = float(train_aux_b.group(1))
        if val_aux_a:
            epoch_data['val_aux_a_loss'] = float(val_aux_a.group(1))
        if val_aux_b:
            epoch_data['val_aux_b_loss'] = float(val_aux_b.group(1))

        # Parse OVERALL METRICS
        train_pearson = re.search(r'Train Overall Pearson:\s+([\d.]+)', block)
        val_pearson = re.search(r'Val Overall Pearson:\s+([\d.]+)', block)

        if train_pearson:
            epoch_data['train_pearson'] = float(train_pearson.group(1))
        if val_pearson:
            epoch_data['val_pearson'] = float(val_pearson.group(1))

        # Parse DESCRIPTOR PEARSON DISTRIBUTION
        train_dist = re.search(r'Train:\s+mean=([\d.-]+),\s+median=([\d.-]+),\s+range=\[([\d.-]+),\s+([\d.]+)\]', block)
        val_dist = re.search(r'Val:\s+mean=([\d.-]+),\s+median=([\d.-]+),\s+range=\[([\d.-]+),\s+([\d.]+)\]', block)

        if train_dist:
            epoch_data['train_desc_mean'] = float(train_dist.group(1))
            epoch_data['train_desc_median'] = float(train_dist.group(2))
            epoch_data['train_desc_min'] = float(train_dist.group(3))
            epoch_data['train_desc_max'] = float(train_dist.group(4))

        if val_dist:
            epoch_data['val_desc_mean'] = float(val_dist.group(1))
            epoch_data['val_desc_median'] = float(val_dist.group(2))
            epoch_data['val_desc_min'] = float(val_dist.group(3))
            epoch_data['val_desc_max'] = float(val_dist.group(4))

        # Parse AUXILIARY HEAD METRICS
        aux_a = re.search(r'Head A \(Seq\+Feat\):\s+Pearson=([\d.]+),\s+R²=([\d.]+),\s+MSE=([\d.]+)', block)
        aux_b = re.search(r'Head B \(Feat only\):\s+Pearson=([\d.]+),\s+R²=([\d.]+),\s+MSE=([\d.]+)', block)

        if aux_a:
            epoch_data['aux_a_pearson'] = float(aux_a.group(1))
            epoch_data['aux_a_r2'] = float(aux_a.group(2))
            epoch_data['aux_a_mse'] = float(aux_a.group(3))

        if aux_b:
            epoch_data['aux_b_pearson'] = float(aux_b.group(1))
            epoch_data['aux_b_r2'] = float(aux_b.group(2))
            epoch_data['aux_b_mse'] = float(aux_b.group(3))

        # Extract top 5 best features
        top_features = re.findall(r'\d+\.\s+([^:]+):\s+([\d.]+)',
                                   re.search(r'TOP 5 BEST PREDICTED FEATURES.*?\n(.*?)(?=\nBOTTOM|\nLearning)', block, re.DOTALL).group(1) if re.search(r'TOP 5 BEST PREDICTED FEATURES.*?\n(.*?)(?=\nBOTTOM|\nLearning)', block, re.DOTALL) else '')
        if top_features:
            epoch_data['top_5_features'] = [(name.strip(), float(score)) for name, score in top_features[:5]]

        # Extract bottom 5 worst features
        bottom_features = re.findall(r'\d+\.\s+([^:]+):\s+([-\d.]+)',
                                       re.search(r'BOTTOM 5 WORST PREDICTED FEATURES.*?\n(.*?)(?=\nLearning|\n\n|$)', block, re.DOTALL).group(1) if re.search(r'BOTTOM 5 WORST PREDICTED FEATURES.*?\n(.*?)(?=\nLearning|\n\n|$)', block, re.DOTALL) else '')
        if bottom_features:
            epoch_data['bottom_5_features'] = [(name.strip(), float(score)) for name, score in bottom_features[:5]]

        # Learning rate
        lr = re.search(r'Learning Rate:\s+([\d.]+)', block)
        if lr:
            epoch_data['learning_rate'] = float(lr.group(1))

        epochs.append(epoch_data)

    return sorted(epochs, key=lambda x: x['epoch'])

def create_loss_curves(epochs, output_path):
    """Create comprehensive loss curve plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Progress - Loss and Metrics', fontsize=16, fontweight='bold')

    epoch_nums = [e['epoch'] for e in epochs]

    # Plot 1: Total Loss
    ax = axes[0, 0]
    train_losses = [e.get('train_total_loss') for e in epochs]
    val_losses = [e.get('val_total_loss') for e in epochs]

    # Clean None values
    train_losses_clean = [x for x in train_losses if x is not None]
    val_losses_clean = [x for x in val_losses if x is not None]

    if train_losses_clean:
        ax.plot(epoch_nums, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    if val_losses_clean:
        ax.plot(epoch_nums, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    if train_losses_clean or val_losses_clean:
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Overall Pearson Correlation
    ax = axes[0, 1]
    train_pearsons = [e.get('train_pearson') for e in epochs]
    val_pearsons = [e.get('val_pearson') for e in epochs]

    train_pearsons_clean = [x for x in train_pearsons if x is not None]
    val_pearsons_clean = [x for x in val_pearsons if x is not None]

    if train_pearsons_clean:
        ax.plot(epoch_nums, train_pearsons, 'b-o', label='Train Pearson', linewidth=2, markersize=4)
    if val_pearsons_clean:
        ax.plot(epoch_nums, val_pearsons, 'r-s', label='Val Pearson', linewidth=2, markersize=4)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Overall Pearson Correlation', fontsize=14, fontweight='bold')
    if train_pearsons_clean or val_pearsons_clean:
        ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 3: Descriptor Pearson Distribution (Mean/Median)
    ax = axes[1, 0]
    train_desc_means = [e.get('train_desc_mean') for e in epochs]
    val_desc_means = [e.get('val_desc_mean') for e in epochs]
    train_desc_medians = [e.get('train_desc_median') for e in epochs]
    val_desc_medians = [e.get('val_desc_median') for e in epochs]

    has_data = False
    if any(x is not None for x in train_desc_means):
        ax.plot(epoch_nums, train_desc_means, 'b-o', label='Train Mean', linewidth=2, markersize=4)
        has_data = True
    if any(x is not None for x in val_desc_means):
        ax.plot(epoch_nums, val_desc_means, 'r-s', label='Val Mean', linewidth=2, markersize=4)
        has_data = True
    if any(x is not None for x in train_desc_medians):
        ax.plot(epoch_nums, train_desc_medians, 'b--^', label='Train Median', linewidth=2, markersize=4, alpha=0.7)
        has_data = True
    if any(x is not None for x in val_desc_medians):
        ax.plot(epoch_nums, val_desc_medians, 'r--v', label='Val Median', linewidth=2, markersize=4, alpha=0.7)
        has_data = True

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Descriptor Pearson', fontsize=12)
    ax.set_title('Descriptor-Level Pearson Statistics', fontsize=14, fontweight='bold')
    if has_data:
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Auxiliary Head Performance
    ax = axes[1, 1]
    aux_a_pearsons = [e.get('aux_a_pearson') for e in epochs]
    aux_b_pearsons = [e.get('aux_b_pearson') for e in epochs]

    has_data = False
    if any(x is not None for x in aux_a_pearsons):
        ax.plot(epoch_nums, aux_a_pearsons, 'g-o', label='Aux Head A (Seq+Feat)', linewidth=2, markersize=4)
        has_data = True
    if any(x is not None for x in aux_b_pearsons):
        ax.plot(epoch_nums, aux_b_pearsons, 'm-s', label='Aux Head B (Feat Only)', linewidth=2, markersize=4)
        has_data = True

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Auxiliary Heads Performance', fontsize=14, fontweight='bold')
    if has_data:
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

    epoch_nums = [e['epoch'] for e in epochs]

    for idx, split in enumerate(['train', 'val']):
        ax = axes[idx]

        means = [e.get(f'{split}_desc_mean') for e in epochs]
        medians = [e.get(f'{split}_desc_median') for e in epochs]
        mins = [e.get(f'{split}_desc_min') for e in epochs]
        maxs = [e.get(f'{split}_desc_max') for e in epochs]

        has_data = False
        # Plot mean and median
        if any(x is not None for x in means):
            ax.plot(epoch_nums, means, 'b-o', label='Mean', linewidth=2.5, markersize=6)
            has_data = True
        if any(x is not None for x in medians):
            ax.plot(epoch_nums, medians, 'r-s', label='Median', linewidth=2.5, markersize=6)
            has_data = True

        # Fill between min and max
        if any(x is not None for x in mins) and any(x is not None for x in maxs):
            # Replace None with nan for plotting
            mins_clean = [x if x is not None else np.nan for x in mins]
            maxs_clean = [x if x is not None else np.nan for x in maxs]
            ax.fill_between(epoch_nums, mins_clean, maxs_clean, alpha=0.2, color='gray', label='Min-Max Range')
            has_data = True

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Pearson Correlation', fontsize=12)
        ax.set_title(f'{split.capitalize()} Set Descriptor Pearson Evolution', fontsize=14, fontweight='bold')
        if has_data:
            ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created Pearson evolution plot: {output_path}")

def extract_final_descriptor_scores(run_dir, output_path):
    """Extract final Pearson scores for top/bottom descriptors from the last epoch."""
    plots_dir = os.path.join(run_dir, 'plots')

    # Find the last epoch feature scores file
    feature_score_files = sorted(
        [f for f in os.listdir(plots_dir) if f.startswith('feature_scores_epoch_') and f.endswith('_val.txt')],
        key=lambda x: int(re.search(r'epoch_(\d+)_', x).group(1))
    )

    if not feature_score_files:
        print(f"No feature score files found in {plots_dir}")
        return None

    last_file = os.path.join(plots_dir, feature_score_files[-1])
    epoch_num = re.search(r'epoch_(\d+)_', feature_score_files[-1]).group(1)

    # Parse the file - new format with spaces
    top_descriptors = []
    bottom_descriptors = []

    with open(last_file, 'r') as f:
        content = f.read()

        # Extract overall statistics from header
        stats_match = re.search(r'Mean:\s+([\d.]+)\s*\|\s*Median:\s+([\d.]+)\s*\|\s*Std:\s+([\d.]+)\s*\|\s*Range:\s*\[([\d.]+),\s*([\d.]+)\]', content)
        if stats_match:
            overall_mean = float(stats_match.group(1))
            overall_median = float(stats_match.group(2))
            overall_std = float(stats_match.group(3))
            overall_min = float(stats_match.group(4))
            overall_max = float(stats_match.group(5))
        else:
            overall_mean = overall_median = overall_std = overall_min = overall_max = None

        # Extract top features
        top_section = re.search(r'Top \d+ Best Predicted Features:(.*?)(?=Bottom|$)', content, re.DOTALL)
        if top_section:
            # Match format: "  1. descriptor_name                                        0.9994"
            for match in re.finditer(r'\s*\d+\.\s+(\S.*?)\s+([\d.]+)\s*$', top_section.group(1), re.MULTILINE):
                desc_name = match.group(1).strip()
                pearson = float(match.group(2))
                top_descriptors.append({'descriptor': desc_name, 'pearson': pearson, 'category': 'top'})

        # Extract bottom features
        bottom_section = re.search(r'Bottom \d+ Worst Predicted Features:(.*?)$', content, re.DOTALL)
        if bottom_section:
            for match in re.finditer(r'\s*\d+\.\s+(\S.*?)\s+([\d.]+)\s*$', bottom_section.group(1), re.MULTILINE):
                desc_name = match.group(1).strip()
                pearson = float(match.group(2))
                bottom_descriptors.append({'descriptor': desc_name, 'pearson': pearson, 'category': 'bottom'})

    # Save to file
    with open(output_path, 'w') as f:
        f.write(f"Final Descriptor Pearson Scores (Epoch {epoch_num} - Validation Set)\n")
        f.write("=" * 80 + "\n\n")

        f.write("OVERALL STATISTICS (Across All Descriptors):\n")
        f.write("-" * 80 + "\n")
        if overall_mean is not None:
            f.write(f"  Mean:   {overall_mean:.6f}\n")
            f.write(f"  Median: {overall_median:.6f}\n")
            f.write(f"  Std Dev: {overall_std:.6f}\n")
            f.write(f"  Range:  [{overall_min:.6f}, {overall_max:.6f}]\n\n")

        f.write(f"NOTE: Full descriptor list not saved in logs.\n")
        f.write(f"Showing top {len(top_descriptors)} best and bottom {len(bottom_descriptors)} worst only.\n\n")

        f.write(f"\nTOP {len(top_descriptors)} BEST PREDICTED DESCRIPTORS:\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Descriptor':<60} {'Pearson':>10}\n")
        f.write("-" * 80 + "\n")

        for rank, desc in enumerate(top_descriptors, 1):
            f.write(f"{rank:<6} {desc['descriptor']:<60} {desc['pearson']:>10.6f}\n")

        f.write(f"\n\nBOTTOM {len(bottom_descriptors)} WORST PREDICTED DESCRIPTORS:\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Descriptor':<60} {'Pearson':>10}\n")
        f.write("-" * 80 + "\n")

        for rank, desc in enumerate(bottom_descriptors, 1):
            f.write(f"{rank:<6} {desc['descriptor']:<60} {desc['pearson']:>10.6f}\n")

    print(f"Extracted final descriptor scores to: {output_path}")
    return top_descriptors + bottom_descriptors

def create_performance_summary(epochs, run_dir, output_path):
    """Create comprehensive performance summary document with ALL epochs."""
    if not epochs:
        return

    last_epoch = epochs[-1]
    best_val_epoch = max(epochs, key=lambda x: x.get('val_pearson', -999))

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
        f.write(f"FINAL EPOCH PERFORMANCE (Epoch {last_epoch['epoch']}):\n")
        f.write("-" * 80 + "\n")
        f.write("Overall Metrics:\n")
        for key, label in [
            ('train_total_loss', 'Train Loss'),
            ('val_total_loss', 'Val Loss'),
            ('train_pearson', 'Train Pearson'),
            ('val_pearson', 'Val Pearson')
        ]:
            val = last_epoch.get(key)
            if val is not None:
                f.write(f"  {label}: {val:.6f}\n")
        f.write("\n")

        f.write("Descriptor-Level Performance:\n")
        for key, label in [
            ('train_desc_mean', 'Train Mean Pearson'),
            ('train_desc_median', 'Train Median Pearson'),
            ('val_desc_mean', 'Val Mean Pearson'),
            ('val_desc_median', 'Val Median Pearson')
        ]:
            val = last_epoch.get(key)
            if val is not None:
                f.write(f"  {label}: {val:.6f}\n")

        train_min = last_epoch.get('train_desc_min')
        train_max = last_epoch.get('train_desc_max')
        val_min = last_epoch.get('val_desc_min')
        val_max = last_epoch.get('val_desc_max')

        if train_min is not None and train_max is not None:
            f.write(f"  Train Range: [{train_min:.6f}, {train_max:.6f}]\n")
        if val_min is not None and val_max is not None:
            f.write(f"  Val Range: [{val_min:.6f}, {val_max:.6f}]\n")
        f.write("\n")

        f.write("Auxiliary Head Performance (Val):\n")
        for key, label in [
            ('aux_a_pearson', 'Head A Pearson'),
            ('aux_a_r2', 'Head A R²'),
            ('aux_a_mse', 'Head A MSE'),
            ('aux_b_pearson', 'Head B Pearson'),
            ('aux_b_r2', 'Head B R²'),
            ('aux_b_mse', 'Head B MSE')
        ]:
            val = last_epoch.get(key)
            if val is not None:
                f.write(f"  {label}: {val:.6f}\n")
        f.write("\n")

        # Best validation performance
        f.write(f"BEST VALIDATION PERFORMANCE (Epoch {best_val_epoch['epoch']}):\n")
        f.write("-" * 80 + "\n")
        for key, label in [
            ('val_pearson', 'Val Pearson'),
            ('val_total_loss', 'Val Loss'),
            ('val_desc_mean', 'Val Descriptor Mean Pearson'),
            ('val_desc_median', 'Val Descriptor Median Pearson')
        ]:
            val = best_val_epoch.get(key)
            if val is not None:
                f.write(f"  {label}: {val:.6f}\n")
        f.write("\n")

        # Training progress summary
        f.write("TRAINING PROGRESS SUMMARY:\n")
        f.write("-" * 80 + "\n")

        first_epoch = epochs[0]
        f.write(f"Improvement from Epoch 1 to Epoch {last_epoch['epoch']}:\n")

        for key, label in [
            ('val_pearson', 'Val Pearson'),
            ('val_desc_mean', 'Val Desc Mean'),
            ('val_total_loss', 'Val Loss')
        ]:
            first_val = first_epoch.get(key)
            last_val = last_epoch.get(key)

            if first_val is not None and last_val is not None:
                if key == 'val_total_loss':
                    improvement = first_val - last_val
                    pct = (improvement / first_val * 100) if first_val != 0 else 0
                    f.write(f"  {label}: {first_val:.6f} → {last_val:.6f} ")
                    f.write(f"(-{improvement:.6f}, {pct:.1f}% reduction)\n")
                else:
                    improvement = last_val - first_val
                    pct = (improvement / first_val * 100) if first_val != 0 else 0
                    f.write(f"  {label}: {first_val:.6f} → {last_val:.6f} ")
                    f.write(f"(+{improvement:.6f}, +{pct:.1f}%)\n")

        f.write("\n")

        # COMPLETE Epoch-by-epoch summary - ALL EPOCHS
        f.write("COMPLETE EPOCH-BY-EPOCH SUMMARY (ALL EPOCHS):\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Epoch':<7} {'Train Loss':<13} {'Val Loss':<13} {'Train Pear':<12} {'Val Pear':<12} {'Val Desc Mean':<13}\n")
        f.write("=" * 80 + "\n")

        # Output EVERY single epoch
        for epoch in epochs:
            f.write(f"{epoch['epoch']:<7} ")

            train_loss = epoch.get('train_total_loss')
            f.write(f"{f'{train_loss:.4f}' if train_loss is not None else 'N/A':<13} ")

            val_loss = epoch.get('val_total_loss')
            f.write(f"{f'{val_loss:.4f}' if val_loss is not None else 'N/A':<13} ")

            train_pearson = epoch.get('train_pearson')
            f.write(f"{f'{train_pearson:.4f}' if train_pearson is not None else 'N/A':<12} ")

            val_pearson = epoch.get('val_pearson')
            f.write(f"{f'{val_pearson:.4f}' if val_pearson is not None else 'N/A':<12} ")

            val_desc_mean = epoch.get('val_desc_mean')
            f.write(f"{f'{val_desc_mean:.4f}' if val_desc_mean is not None else 'N/A':<13}\n")

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

    # 2. Parse epoch results CORRECTLY
    print("\n2. Parsing epoch results correctly...")
    epoch_file = os.path.join(run_dir, 'epoch_results.txt')
    if os.path.exists(epoch_file):
        epochs = parse_epoch_results_correctly(epoch_file)
        print(f"   Parsed {len(epochs)} epochs successfully")

        # Save parsed data as JSON
        with open(os.path.join(output_dir, 'parsed_epochs.json'), 'w') as f:
            json.dump(epochs, f, indent=2)
            print(f"   Saved parsed data to JSON")
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
    print("\n8. Creating best model reference...")
    best_model_src = os.path.join(run_dir, 'best_model.pt')
    if os.path.exists(best_model_src):
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
    print("ALL PHYSINFORMER MODELS RE-PROCESSED SUCCESSFULLY!")
    print("="*80)

if __name__ == '__main__':
    main()
