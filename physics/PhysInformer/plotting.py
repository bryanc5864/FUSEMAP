import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

def create_training_plots(
    epoch_histories: Dict[str, List[Dict]], 
    current_epoch: int,
    save_dir: Path,
    cell_type: str
):
    """
    Create tables showing train vs val metrics side by side
    
    Args:
        epoch_histories: Dict with 'train' and 'val' lists of metric dictionaries
        current_epoch: Current epoch number
        save_dir: Directory to save plots
        cell_type: Cell type being trained
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Get the latest metrics
    if not epoch_histories['train'] or not epoch_histories['val']:
        print("No metrics to plot yet")
        return
        
    train_metrics = epoch_histories['train'][-1]
    val_metrics = epoch_histories['val'][-1]
    
    # Create figure with tables
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(f'{cell_type} Training Metrics - Epoch {current_epoch}', fontsize=20, fontweight='bold')
    
    # Define sections and their metrics
    sections = [
        {
            'title': 'Main Task - Descriptors Prediction',
            'metrics': [
                ('descriptors_pearson_mean', 'Pearson (mean)', '.4f'),
                ('descriptors_spearman_mean', 'Spearman (mean)', '.4f'),
                ('descriptors_r2_mean', 'R² (mean)', '.4f'),
                ('descriptors_mse_mean', 'MSE (mean)', '.6f'),
                ('descriptors_mae_mean', 'MAE (mean)', '.6f'),
                ('descriptors_pearson_std', 'Pearson (std)', '.4f'),
                ('descriptors_spearman_std', 'Spearman (std)', '.4f'),
                ('descriptors_r2_std', 'R² (std)', '.4f'),
                ('descriptors_mse_std', 'MSE (std)', '.6f'),
                ('descriptors_mae_std', 'MAE (std)', '.6f'),
            ]
        },
        {
            'title': 'Overall Metrics',
            'metrics': [
                ('overall_pearson', 'Overall Pearson', '.4f'),
                ('overall_spearman', 'Overall Spearman', '.4f'),
                ('overall_r2', 'Overall R²', '.4f'),
                ('overall_mse', 'Overall MSE', '.6f'),
                ('overall_mae', 'Overall MAE', '.6f'),
                ('overall_rmse', 'Overall RMSE', '.6f'),
            ]
        },
        {
            'title': 'Loss Values',
            'metrics': [
                ('total_loss', 'Total Loss', '.6f'),
                ('desc_loss', 'Descriptor Loss (Sum)', '.6f'),
                ('desc_loss_mean', 'Descriptor Loss (Mean)', '.6f'),
            ]
        }
    ]
    
    # Create subplots for each section
    n_sections = len(sections)
    for idx, section in enumerate(sections):
        ax = plt.subplot(n_sections, 1, idx + 1)
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data for table
        table_data = []
        for metric_key, metric_name, fmt in section['metrics']:
            train_val = train_metrics.get(metric_key, 0.0)
            val_val = val_metrics.get(metric_key, 0.0)
            
            # Format values
            train_str = f'{train_val:{fmt}}' if train_val != 0 else 'N/A'
            val_str = f'{val_val:{fmt}}' if val_val != 0 else 'N/A'
            
            # Calculate difference and color
            if train_val != 0 and val_val != 0:
                diff = val_val - train_val
                diff_str = f'{diff:+{fmt}}'
                
                # Determine if improvement (for correlation metrics, higher is better)
                if 'pearson' in metric_key or 'spearman' in metric_key or 'r2' in metric_key:
                    better = val_val > train_val
                else:  # For loss/error metrics, lower is better
                    better = val_val < train_val
            else:
                diff_str = 'N/A'
                better = None
                
            table_data.append([metric_name, train_str, val_str, diff_str, better])
        
        # Create table
        col_labels = ['Metric', 'Train', 'Val', 'Δ (Val-Train)', 'Status']
        
        # Create colors for the table
        cell_colors = []
        for row in table_data:
            row_colors = ['white', 'lightblue', 'lightcoral', 'white', 'white']
            if row[4] is not None:  # If we have a comparison
                if row[4]:  # Better
                    row_colors[3] = 'lightgreen'
                    row_colors[4] = 'lightgreen'
                else:  # Worse
                    row_colors[3] = 'lightyellow'
                    row_colors[4] = 'lightyellow'
            cell_colors.append(row_colors)
            
        # Remove the status column from display but keep colors
        display_data = [row[:4] for row in table_data]
        display_colors = [row[:4] for row in cell_colors]
        
        # Add status indicators
        for i, row in enumerate(display_data):
            if table_data[i][4] is not None:
                if table_data[i][4]:
                    display_data[i][3] += ' ✓'
                else:
                    display_data[i][3] += ' ⚠'
        
        table = ax.table(cellText=display_data,
                        colLabels=col_labels[:4],
                        cellColours=display_colors,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.3, 0.2, 0.2, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the header
        for i in range(len(col_labels[:4])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Add section title
        ax.text(0.5, 1.05, section['title'], transform=ax.transAxes,
                fontsize=14, fontweight='bold', ha='center')
    
    plt.tight_layout()
    # Use fixed filename so each epoch replaces the previous one
    filename = save_dir / f'metrics_table_current.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Also create a progress plot showing key metrics over time
    if len(epoch_histories['train']) > 1:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{cell_type} - Training Progress (Epoch {current_epoch})', fontsize=16)
        
        epochs = list(range(1, len(epoch_histories['train']) + 1))
        
        # Key metrics to track over time
        progress_metrics = [
            ('total_loss', 'Total Loss', True),
            ('overall_pearson', 'Overall Pearson', False),
            ('descriptors_pearson_mean', 'Descriptors Pearson', False),
            ('desc_loss_mean', 'Mean Feature Loss', True),
            ('descriptors_r2_mean', 'Descriptors R²', False),
        ]
        
        for idx, (metric, title, use_log) in enumerate(progress_metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Extract values
            train_vals = [h.get(metric, np.nan) for h in epoch_histories['train']]
            val_vals = [h.get(metric, np.nan) for h in epoch_histories['val']]
            
            # Filter out NaN values
            train_epochs = [e for e, v in zip(epochs, train_vals) if not np.isnan(v)]
            train_vals_clean = [v for v in train_vals if not np.isnan(v)]
            val_epochs = [e for e, v in zip(epochs, val_vals) if not np.isnan(v)]
            val_vals_clean = [v for v in val_vals if not np.isnan(v)]
            
            if train_vals_clean and val_vals_clean:
                ax.plot(train_epochs, train_vals_clean, 'o-', label='Train', linewidth=2, markersize=6)
                ax.plot(val_epochs, val_vals_clean, 's-', label='Val', linewidth=2, markersize=6)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(title)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                if use_log and min(min(train_vals_clean), min(val_vals_clean)) > 0:
                    ax.set_yscale('log')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
        
        plt.tight_layout()
        # Use fixed filename so each epoch replaces the previous one
        filename = save_dir / f'training_progress.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

def create_final_summary_plot(
    epoch_histories: Dict[str, List[Dict]],
    save_dir: Path,
    cell_type: str
):
    """Create a final summary table with all metrics"""
    save_dir = Path(save_dir)
    
    if not epoch_histories['train'] or not epoch_histories['val']:
        print("No metrics for final summary")
        return
    
    # Get final epoch metrics
    final_train = epoch_histories['train'][-1]
    final_val = epoch_histories['val'][-1]
    
    # Create summary DataFrame
    summary_data = {
        'Metric': [],
        'Final Train': [],
        'Final Val': [],
        'Best Val': [],
        'Best Epoch': []
    }
    
    # Track best values
    key_metrics = [
        ('overall_pearson', 'Overall Pearson', True, '.4f'),
        ('overall_mse', 'Overall MSE', False, '.6f'),
        ('descriptors_pearson_mean', 'Descriptors Pearson', True, '.4f'),
        ('descriptors_mse_mean', 'Descriptors MSE', False, '.6f'),
        ('descriptors_r2_mean', 'Descriptors R²', True, '.4f'),
        ('total_loss', 'Total Loss', False, '.6f'),
    ]
    
    for metric_key, metric_name, higher_better, fmt in key_metrics:
        # Get all values
        val_history = [h.get(metric_key, np.nan) for h in epoch_histories['val']]
        valid_vals = [(i+1, v) for i, v in enumerate(val_history) if not np.isnan(v)]
        
        if valid_vals:
            if higher_better:
                best_epoch, best_val = max(valid_vals, key=lambda x: x[1])
            else:
                best_epoch, best_val = min(valid_vals, key=lambda x: x[1])
            
            summary_data['Metric'].append(metric_name)
            summary_data['Final Train'].append(f'{final_train.get(metric_key, 0):{fmt}}')
            summary_data['Final Val'].append(f'{final_val.get(metric_key, 0):{fmt}}')
            summary_data['Best Val'].append(f'{best_val:{fmt}}')
            summary_data['Best Epoch'].append(best_epoch)
    
    # Create figure with summary table
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    df = pd.DataFrame(summary_data)
    
    # Create table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color cells based on performance
    for i in range(1, len(df) + 1):
        # Highlight best epoch
        if df.iloc[i-1]['Best Epoch'] == len(epoch_histories['val']):
            table[(i, 3)].set_facecolor('lightgreen')
        
    plt.title(f'{cell_type} - Final Training Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_dir / 'final_summary_table.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_pearson_distribution_plot(
    metrics: Dict,
    epoch: int,
    save_dir: Path,
    cell_type: str,
    split: str = 'val'
):
    """
    Create a distribution plot of Pearson correlations across features
    
    Args:
        metrics: Metrics dictionary containing 'descriptors_feature_scores' 
        epoch: Current epoch number
        save_dir: Directory to save plots
        cell_type: Cell type being trained
        split: 'train' or 'val'
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Get feature scores if available
    if 'descriptors_feature_scores' not in metrics:
        return
    
    feature_scores = metrics['descriptors_feature_scores']
    if not feature_scores:
        return
    
    # Extract scores and names
    names, scores = zip(*feature_scores)
    scores = np.array(scores)
    
    # Create figure with distribution plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Main histogram
    ax1.hist(scores, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.4f}')
    ax1.axvline(np.median(scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.4f}')
    ax1.set_xlabel('Pearson Correlation', fontsize=12)
    ax1.set_ylabel('Number of Features', fontsize=12)
    ax1.set_title(f'{cell_type} - Epoch {epoch} - {split.upper()} Pearson Distribution\n'
                 f'Range: [{np.min(scores):.4f}, {np.max(scores):.4f}] | '
                 f'Std: {np.std(scores):.4f}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot for quartiles
    ax2.boxplot(scores, vert=False, widths=0.5, 
                showmeans=True, meanline=True,
                patch_artist=True,  # Enable filled boxes
                boxprops=dict(facecolor='lightblue', edgecolor='black'),
                medianprops=dict(color='green', linewidth=2),
                meanprops=dict(color='red', linewidth=2))
    ax2.set_xlabel('Pearson Correlation', fontsize=12)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add text annotations for quartiles
    q1, q2, q3 = np.percentile(scores, [25, 50, 75])
    ax2.text(q1, 1.3, f'Q1: {q1:.3f}', ha='center', fontsize=10)
    ax2.text(q2, 1.3, f'Q2: {q2:.3f}', ha='center', fontsize=10)
    ax2.text(q3, 1.3, f'Q3: {q3:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'pearson_distribution_epoch_{epoch}_{split}.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Also save the sorted feature scores to a text file
    sorted_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
    with open(save_dir / f'feature_scores_epoch_{epoch}_{split}.txt', 'w') as f:
        f.write(f"Feature Pearson Correlations - {cell_type} - Epoch {epoch} - {split.upper()}\n")
        f.write("="*80 + "\n")
        f.write(f"Mean: {np.mean(scores):.4f} | Median: {np.median(scores):.4f} | ")
        f.write(f"Std: {np.std(scores):.4f} | Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]\n")
        f.write("="*80 + "\n\n")
        
        f.write("Top 20 Best Predicted Features:\n")
        for i, (name, score) in enumerate(sorted_scores[:20], 1):
            f.write(f"{i:3d}. {name:60s} {score:8.4f}\n")
        
        f.write("\nBottom 20 Worst Predicted Features:\n")
        for i, (name, score) in enumerate(sorted_scores[-20:], 1):
            f.write(f"{i:3d}. {name:60s} {score:8.4f}\n")