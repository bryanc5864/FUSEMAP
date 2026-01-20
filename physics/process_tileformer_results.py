#!/usr/bin/env python3
"""
Script to process and organize TileFormer training results.
Creates organized results folder with metrics, plots, and performance summaries.
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

def parse_metrics_file(metrics_file):
    """Parse a metrics_epoch_X.txt file."""
    metrics = {}
    with open(metrics_file, 'r') as f:
        content = f.read()

        # Extract OVERALL metrics section (has simple aggregated values)
        overall_section = re.search(r'OVERALL Metrics:.*?(?=\n\n[A-Z]|\n\n$|$)', content, re.DOTALL)

        if overall_section:
            section_text = overall_section.group(0)
            # Parse format: "  mse                      : 0.002362"
            metrics_patterns = {
                'mse': r'mse\s*:\s+([\d.]+(?:e[+-]?\d+)?)',
                'rmse': r'rmse\s*:\s+([\d.]+)',
                'mae': r'mae\s*:\s+([\d.]+)',
                'median_ae': r'median_ae\s*:\s+([\d.]+)',
            }

            for key, pattern in metrics_patterns.items():
                match = re.search(pattern, section_text)
                if match:
                    metrics[key] = float(match.group(1))

        # Also try to extract r2, pearson, spearman from STD_PSI_MEAN section (most representative)
        std_psi_mean = re.search(r'STD_PSI_MEAN Metrics:.*?(?=\n\n[A-Z]|\n\n|$)', content, re.DOTALL)
        if std_psi_mean:
            section_text = std_psi_mean.group(0)
            extra_patterns = {
                'r2': r'r2\s*:\s+([-\d.]+)',
                'explained_variance': r'explained_variance\s*:\s+([-\d.]+)',
                'pearson_r': r'pearson_r\s*:\s+([-\d.]+)',
                'spearman_r': r'spearman_r\s*:\s+([-\d.]+)',
            }

            for key, pattern in extra_patterns.items():
                match = re.search(pattern, section_text)
                if match:
                    metrics[key] = float(match.group(1))

    return metrics

def collect_all_metrics(checkpoint_dir):
    """Collect metrics from all epochs."""
    epochs_data = []

    # Get all metrics files
    metrics_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith('metrics_epoch_') and f.endswith('.txt')],
        key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1))
    )

    for metrics_file in metrics_files:
        epoch_num = int(re.search(r'epoch_(\d+)', metrics_file).group(1))
        filepath = os.path.join(checkpoint_dir, metrics_file)

        metrics = parse_metrics_file(filepath)
        metrics['epoch'] = epoch_num

        epochs_data.append(metrics)

    return epochs_data

def create_comprehensive_loss_curves(epochs_data, output_path):
    """Create comprehensive loss and metrics curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('TileFormer Training Progress', fontsize=16, fontweight='bold')

    epoch_nums = [e['epoch'] for e in epochs_data]

    # Plot 1: MSE
    ax = axes[0, 0]
    mse_vals = [e.get('mse') for e in epochs_data]
    if any(v is not None for v in mse_vals):
        ax.plot(epoch_nums, mse_vals, 'b-o', linewidth=2, markersize=5, label='MSE')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Mean Squared Error', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: RMSE
    ax = axes[0, 1]
    rmse_vals = [e.get('rmse') for e in epochs_data]
    if any(v is not None for v in rmse_vals):
        ax.plot(epoch_nums, rmse_vals, 'r-s', linewidth=2, markersize=5, label='RMSE')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: MAE
    ax = axes[0, 2]
    mae_vals = [e.get('mae') for e in epochs_data]
    if any(v is not None for v in mae_vals):
        ax.plot(epoch_nums, mae_vals, 'g-^', linewidth=2, markersize=5, label='MAE')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Pearson R
    ax = axes[1, 0]
    pearson_vals = [e.get('pearson_r') for e in epochs_data]
    if any(v is not None for v in pearson_vals):
        ax.plot(epoch_nums, pearson_vals, 'm-o', linewidth=2, markersize=5, label='Pearson R')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pearson R', fontsize=12)
    ax.set_title('Pearson Correlation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.2, 1.0])

    # Plot 5: Spearman R
    ax = axes[1, 1]
    spearman_vals = [e.get('spearman_r') for e in epochs_data]
    if any(v is not None for v in spearman_vals):
        ax.plot(epoch_nums, spearman_vals, 'c-s', linewidth=2, markersize=5, label='Spearman R')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Spearman R', fontsize=12)
    ax.set_title('Spearman Correlation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.2, 1.0])

    # Plot 6: R²
    ax = axes[1, 2]
    r2_vals = [e.get('r2') for e in epochs_data]
    if any(v is not None for v in r2_vals):
        ax.plot(epoch_nums, r2_vals, 'y-^', linewidth=2, markersize=5, label='R²')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² Score', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.2, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created comprehensive loss curves: {output_path}")

def create_performance_summary(epochs_data, checkpoint_dir, output_path):
    """Create comprehensive performance summary document."""
    if not epochs_data:
        return

    last_epoch = epochs_data[-1]

    # Find best epoch by Pearson R
    valid_epochs = [e for e in epochs_data if 'pearson_r' in e]
    if valid_epochs:
        best_epoch = max(valid_epochs, key=lambda x: x.get('pearson_r', -999))
    else:
        best_epoch = last_epoch

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TILEFORMER MODEL PERFORMANCE SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Training configuration
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 80 + "\n")

        # Try to extract from training.log
        log_file = os.path.join(checkpoint_dir, 'training.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as log:
                for line in log:
                    if 'Arguments:' in line:
                        args_str = line.split('Arguments:')[1].strip()
                        f.write(f"Arguments: {args_str}\n")
                        break

        f.write(f"Total Epochs Trained: {len(epochs_data)}\n")
        f.write(f"Run Directory: {checkpoint_dir}\n\n")

        # Final epoch performance
        f.write(f"FINAL EPOCH PERFORMANCE (Epoch {last_epoch.get('epoch', '?')}):\n")
        f.write("-" * 80 + "\n")
        for metric in ['mse', 'rmse', 'mae', 'median_ae', 'r2', 'explained_variance', 'pearson_r', 'spearman_r']:
            val = last_epoch.get(metric, 'N/A')
            metric_name = metric.replace('_', ' ').title()
            if isinstance(val, float):
                if metric == 'mse' and val < 0.01:
                    f.write(f"  {metric_name}: {val:.6f}\n")
                else:
                    f.write(f"  {metric_name}: {val:.4f}\n")
            else:
                f.write(f"  {metric_name}: {val}\n")
        f.write("\n")

        # Best validation performance
        f.write(f"BEST VALIDATION PERFORMANCE (Epoch {best_epoch.get('epoch', '?')}):\n")
        f.write("-" * 80 + "\n")
        for metric in ['pearson_r', 'spearman_r', 'r2', 'mse', 'rmse', 'mae']:
            val = best_epoch.get(metric, 'N/A')
            metric_name = metric.replace('_', ' ').title()
            if isinstance(val, float):
                if metric == 'mse' and val < 0.01:
                    f.write(f"  {metric_name}: {val:.6f}\n")
                else:
                    f.write(f"  {metric_name}: {val:.4f}\n")
            else:
                f.write(f"  {metric_name}: {val}\n")
        f.write("\n")

        # Training progress
        if len(epochs_data) > 1:
            first_epoch = epochs_data[0]
            f.write("TRAINING PROGRESS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Improvement from Epoch 1 to Epoch {last_epoch.get('epoch', '?')}:\n")

            for metric in ['pearson_r', 'mse', 'mae']:
                if metric in first_epoch and metric in last_epoch:
                    first_val = first_epoch[metric]
                    last_val = last_epoch[metric]

                    if metric == 'mse' or metric == 'mae':
                        improvement = first_val - last_val
                        pct = (improvement / first_val * 100) if first_val != 0 else 0
                        f.write(f"  {metric.replace('_', ' ').title()}: {first_val:.6f} → {last_val:.6f} ")
                        f.write(f"(-{improvement:.6f}, {pct:.1f}% reduction)\n")
                    else:
                        improvement = last_val - first_val
                        if first_val != 0:
                            pct = (improvement / abs(first_val) * 100) if first_val > 0 else 0
                        else:
                            pct = 0
                        f.write(f"  {metric.replace('_', ' ').title()}: {first_val:.4f} → {last_val:.4f} ")
                        f.write(f"(+{improvement:.4f})\n")

            f.write("\n")

        # Epoch-by-epoch summary
        f.write("EPOCH-BY-EPOCH SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epoch':<7} {'MSE':<12} {'RMSE':<10} {'MAE':<10} {'Pearson R':<12} {'Spearman R':<12}\n")
        f.write("-" * 80 + "\n")

        for epoch in epochs_data:
            f.write(f"{epoch.get('epoch', '?'):<7} ")
            mse = epoch.get('mse', np.nan)
            f.write(f"{mse if np.isnan(mse) else f'{mse:.6f}':<12} ")
            rmse = epoch.get('rmse', np.nan)
            f.write(f"{rmse if np.isnan(rmse) else f'{rmse:.4f}':<10} ")
            mae = epoch.get('mae', np.nan)
            f.write(f"{mae if np.isnan(mae) else f'{mae:.4f}':<10} ")
            pearson = epoch.get('pearson_r', np.nan)
            f.write(f"{pearson if np.isnan(pearson) else f'{pearson:.4f}':<12} ")
            spearman = epoch.get('spearman_r', np.nan)
            f.write(f"{spearman if np.isnan(spearman) else f'{spearman:.4f}':<12}\n")

    print(f"Created performance summary: {output_path}")

def parse_final_evaluation_report(report_file, output_path):
    """Parse final evaluation report and create formatted summary."""
    if not os.path.exists(report_file):
        print(f"Final evaluation report not found: {report_file}")
        return

    with open(report_file, 'r') as f:
        content = f.read()

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TILEFORMER FINAL EVALUATION - ELECTROSTATIC POTENTIAL METRICS\n")
        f.write("=" * 80 + "\n\n")

        f.write("This report contains comprehensive evaluation metrics for TileFormer's\n")
        f.write("prediction of electrostatic potential (PSI) features for DNA sequences.\n\n")

        # Extract PSI metrics sections
        psi_features = [
            'STD_PSI_MIN', 'STD_PSI_MAX', 'STD_PSI_MEAN',
            'ENH_PSI_MIN', 'ENH_PSI_MAX', 'ENH_PSI_MEAN'
        ]

        f.write("ELECTROSTATIC POTENTIAL (PSI) FEATURE PREDICTIONS:\n")
        f.write("-" * 80 + "\n\n")

        for psi_feature in psi_features:
            # Find the section for this feature
            pattern = f"{psi_feature} Metrics:.*?(?=\n\n[A-Z_]+ Metrics:|\n\nOVERALL|$)"
            match = re.search(pattern, content, re.DOTALL)

            if match:
                f.write(f"{psi_feature}:\n")
                section = match.group(0)

                # Extract metrics
                metrics_to_extract = ['mse', 'rmse', 'mae', 'r2', 'pearson_r', 'spearman_r']
                for metric in metrics_to_extract:
                    metric_pattern = f"{metric}\\s+:\\s+([\\d.e+-]+)"
                    metric_match = re.search(metric_pattern, section)
                    if metric_match:
                        value = float(metric_match.group(1))
                        metric_name = metric.replace('_', ' ').upper()
                        if 'mse' in metric.lower() or 'e-' in str(value).lower():
                            f.write(f"  {metric_name:<20}: {value:.6e}\n")
                        else:
                            f.write(f"  {metric_name:<20}: {value:.6f}\n")
                f.write("\n")

        # Overall metrics
        overall_match = re.search(r'OVERALL Metrics:.*?(?=\n\n[A-Z]+ Metrics:|\n\n|$)', content, re.DOTALL)
        if overall_match:
            f.write("OVERALL PERFORMANCE (Across All PSI Features):\n")
            f.write("-" * 80 + "\n")
            overall_section = overall_match.group(0)
            for line in overall_section.split('\n')[2:]:
                if ':' in line:
                    f.write(f"  {line.strip()}\n")
            f.write("\n")

        # Calibration metrics
        cal_match = re.search(r'CALIBRATION Metrics:.*?(?=\n\n[A-Z]+ Metrics:|\n\n|$)', content, re.DOTALL)
        if cal_match:
            f.write("MODEL CALIBRATION:\n")
            f.write("-" * 80 + "\n")
            cal_section = cal_match.group(0)
            for line in cal_section.split('\n')[2:]:
                if ':' in line:
                    f.write(f"  {line.strip()}\n")
            f.write("\n")

        # Ranking metrics
        rank_match = re.search(r'RANKING Metrics:.*?(?=\n\n[A-Z]+ Metrics:|\n\n|$)', content, re.DOTALL)
        if rank_match:
            f.write("RANKING PERFORMANCE:\n")
            f.write("-" * 80 + "\n")
            rank_section = rank_match.group(0)
            for line in rank_section.split('\n')[2:]:
                if ':' in line:
                    f.write(f"  {line.strip()}\n")

    print(f"Created electrostatic potential metrics summary: {output_path}")

def process_tileformer_results(checkpoint_dir, output_dir):
    """Process TileFormer results and organize them."""
    print("\n" + "="*80)
    print("Processing TileFormer Model")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Copy training logs
    print("\n1. Copying training logs...")
    for log_file in ['training.log', 'final_evaluation_report.txt', 'training_results.json']:
        src = os.path.join(checkpoint_dir, log_file)
        if os.path.exists(src):
            dst = os.path.join(output_dir, log_file)
            shutil.copy2(src, dst)
            print(f"   Copied: {log_file}")

    # 2. Collect metrics from all epochs
    print("\n2. Collecting metrics from all epochs...")
    epochs_data = collect_all_metrics(checkpoint_dir)
    print(f"   Collected metrics from {len(epochs_data)} epochs")

    # Save as JSON
    with open(os.path.join(output_dir, 'parsed_epochs.json'), 'w') as f:
        json.dump(epochs_data, f, indent=2)

    # 3. Generate comprehensive loss curves
    print("\n3. Generating comprehensive loss curves...")
    if epochs_data:
        create_comprehensive_loss_curves(epochs_data, os.path.join(output_dir, 'comprehensive_metrics.png'))

    # 4. Copy existing plots
    print("\n4. Copying existing plots...")
    plots_src_dir = os.path.join(checkpoint_dir, 'plots')
    if os.path.exists(plots_src_dir):
        plots_dst_dir = os.path.join(output_dir, 'training_plots')
        os.makedirs(plots_dst_dir, exist_ok=True)

        for plot_file in os.listdir(plots_src_dir):
            if plot_file.endswith('.png') or plot_file.endswith('.txt'):
                src = os.path.join(plots_src_dir, plot_file)
                dst = os.path.join(plots_dst_dir, plot_file)
                shutil.copy2(src, dst)
                print(f"   Copied: {plot_file}")

    # 5. Create performance summary
    print("\n5. Creating performance summary...")
    if epochs_data:
        create_performance_summary(epochs_data, checkpoint_dir, os.path.join(output_dir, 'PERFORMANCE_SUMMARY.txt'))

    # 6. Parse final evaluation report for electrostatic metrics
    print("\n6. Extracting electrostatic potential metrics...")
    final_eval = os.path.join(checkpoint_dir, 'final_evaluation_report.txt')
    if os.path.exists(final_eval):
        parse_final_evaluation_report(final_eval, os.path.join(output_dir, 'ELECTROSTATIC_METRICS.txt'))

    # 7. Copy best model reference
    print("\n7. Creating best model reference...")
    best_model_src = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_src):
        with open(os.path.join(output_dir, 'BEST_MODEL_LOCATION.txt'), 'w') as f:
            f.write(f"Best model location: {best_model_src}\n")
            f.write(f"Model size: {os.path.getsize(best_model_src) / (1024*1024):.2f} MB\n")
        print(f"   Created best model reference")

    print("\nCompleted processing TileFormer!")

def main():
    """Main processing function for TileFormer."""
    base_dir = '/home/bcheng/sequence_optimization/mainproject/PhysiFormer/physpreprocess'
    results_base = os.path.join(base_dir, 'results')

    checkpoint_dir = os.path.join(base_dir, 'TileFormer_model/checkpoints/run_20250819_063725')
    output_dir = os.path.join(results_base, 'TileFormer')

    if os.path.exists(checkpoint_dir):
        process_tileformer_results(checkpoint_dir, output_dir)
    else:
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")

    print("\n" + "="*80)
    print("TILEFORMER PROCESSING COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == '__main__':
    main()
