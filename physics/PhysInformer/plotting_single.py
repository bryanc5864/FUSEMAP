import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
from pathlib import Path
from typing import Dict, List
import pandas as pd

def create_single_feature_plots(
    epoch_histories: Dict[str, List[Dict]], 
    current_epoch: int,
    save_dir: Path,
    target_feature: str
):
    """
    Create training plots for single feature prediction
    
    Args:
        epoch_histories: Dict with 'train' and 'val' lists of metric dictionaries
        current_epoch: Current epoch number
        save_dir: Directory to save plots
        target_feature: Target feature being predicted
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
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{target_feature} Training Metrics - Epoch {current_epoch}', fontsize=20, fontweight='bold')
    
    # Define sections and their metrics for single feature prediction
    sections = [
        {
            'title': f'Main Task - {target_feature} Prediction',
            'metrics': [
                ('main_pearson', 'Pearson', '.4f'),
                ('main_spearman', 'Spearman', '.4f'),
                ('main_r2', 'R²', '.4f'),
                ('main_mse', 'MSE', '.6f'),
                ('main_mae', 'MAE', '.6f'),
                ('main_rmse', 'RMSE', '.6f'),
            ]
        },
        {
            'title': 'Loss Values',
            'metrics': [
                ('total_loss', 'Total Loss', '.6f'),
                ('main_loss', 'Main Loss', '.6f'),
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
                        colWidths=[0.4, 0.2, 0.2, 0.2])
        
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
        fig.suptitle(f'{target_feature} - Training Progress (Epoch {current_epoch})', fontsize=16)
        
        epochs = list(range(1, len(epoch_histories['train']) + 1))
        
        # Key metrics to track over time
        progress_metrics = [
            ('total_loss', 'Total Loss', True),
            ('main_pearson', 'Main Pearson', False),
            ('main_mse', 'Main MSE', False),
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
                
                if use_log and len(train_vals_clean) > 0 and len(val_vals_clean) > 0:
                    if min(min(train_vals_clean), min(val_vals_clean)) > 0:
                        ax.set_yscale('log')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
        
        plt.tight_layout()
        # Use fixed filename so each epoch replaces the previous one
        filename = save_dir / f'training_progress.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

def create_single_feature_summary(
    epoch_histories: Dict[str, List[Dict]],
    save_dir: Path,
    target_feature: str
):
    """Create a final summary table for single feature prediction"""
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
        ('main_pearson', 'Main Pearson', True, '.4f'),
        ('main_mse', 'Main MSE', False, '.6f'),
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
        
    plt.title(f'{target_feature} - Final Training Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_dir / 'final_summary_table.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()