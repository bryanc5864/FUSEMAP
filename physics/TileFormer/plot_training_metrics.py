#!/usr/bin/env python3
"""
Plot training metrics and losses from TileFormer training runs
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingPlotter:
    def __init__(self, checkpoint_dir: str):
        """Initialize with checkpoint directory."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_file = self.checkpoint_dir / 'training_results.json'
        
        # Load training results
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
        else:
            # Try to load from individual epoch files
            self.results = self._load_from_epoch_files()
        
        self.plot_dir = self.checkpoint_dir / 'plots'
        self.plot_dir.mkdir(exist_ok=True)
    
    def _load_from_epoch_files(self):
        """Load metrics from individual epoch files if main results file doesn't exist."""
        print("Loading from individual epoch files...")
        results = {'train_loss': [], 'val_metrics': []}
        
        # Find all metrics files
        metrics_files = sorted(self.checkpoint_dir.glob('metrics_epoch_*.txt'))
        
        # Extract epoch numbers and load step losses
        for i in range(1, len(metrics_files) + 1):
            step_loss_file = self.checkpoint_dir / f'step_losses_epoch_{i}.txt'
            if step_loss_file.exists():
                step_losses = np.loadtxt(step_loss_file)
                results[f'step_losses_epoch_{i}'] = step_losses.tolist()
                results['train_loss'].append(float(np.mean(step_losses)))
        
        return results
    
    def plot_loss_curves(self):
        """Plot training and validation loss curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Training loss per epoch
        if 'train_loss' in self.results and self.results['train_loss']:
            epochs = range(1, len(self.results['train_loss']) + 1)
            axes[0, 0].plot(epochs, self.results['train_loss'], 'b-o', label='Training Loss', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('MSE Loss')
            axes[0, 0].set_title('Training Loss per Epoch')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # 2. Validation loss per epoch
        if 'val_metrics' in self.results and self.results['val_metrics']:
            val_losses = [epoch_metrics['overall']['mse'] for epoch_metrics in self.results['val_metrics']]
            epochs = range(1, len(val_losses) + 1)
            axes[0, 1].plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MSE Loss')
            axes[0, 1].set_title('Validation Loss per Epoch')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # 3. Training vs Validation loss
        if 'train_loss' in self.results and 'val_metrics' in self.results:
            train_epochs = range(1, len(self.results['train_loss']) + 1)
            val_epochs = range(1, len(val_losses) + 1)
            
            axes[1, 0].plot(train_epochs, self.results['train_loss'], 'b-o', label='Training', linewidth=2)
            axes[1, 0].plot(val_epochs, val_losses, 'r-s', label='Validation', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MSE Loss')
            axes[1, 0].set_title('Training vs Validation Loss')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            axes[1, 0].set_yscale('log')
        
        # 4. Step-wise loss for latest epoch
        latest_epoch = len(self.results['train_loss']) if 'train_loss' in self.results else 1
        step_key = f'step_losses_epoch_{latest_epoch}'
        if step_key in self.results:
            step_losses = self.results[step_key]
            steps = range(1, len(step_losses) + 1)
            axes[1, 1].plot(steps, step_losses, 'g-', alpha=0.7, linewidth=1)
            
            # Add moving average
            window = max(1, len(step_losses) // 20)
            moving_avg = pd.Series(step_losses).rolling(window=window).mean()
            axes[1, 1].plot(steps, moving_avg, 'orange', linewidth=2, label=f'Moving Avg (window={window})')
            
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('MSE Loss')
            axes[1, 1].set_title(f'Step-wise Loss (Epoch {latest_epoch})')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved loss curves to {self.plot_dir / 'loss_curves.png'}")
    
    def plot_validation_metrics(self):
        """Plot validation metrics across epochs."""
        if 'val_metrics' not in self.results or not self.results['val_metrics']:
            print("No validation metrics found!")
            return
        
        # Extract metrics for each ABPS value
        psi_columns = ['std_psi_min', 'std_psi_max', 'std_psi_mean', 'enh_psi_min', 'enh_psi_max', 'enh_psi_mean']
        metrics_to_plot = ['pearson_r', 'spearman_r', 'r2', 'explained_variance', 'mse', 'mae']
        
        epochs = range(1, len(self.results['val_metrics']) + 1)
        
        # Create subplot grid
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            for psi_col in psi_columns:
                metric_values = []
                for epoch_metrics in self.results['val_metrics']:
                    if psi_col in epoch_metrics and metric in epoch_metrics[psi_col]:
                        metric_values.append(epoch_metrics[psi_col][metric])
                    else:
                        metric_values.append(np.nan)
                
                if len(metric_values) > 0 and not all(np.isnan(metric_values)):
                    ax.plot(epochs[:len(metric_values)], metric_values, 'o-', label=psi_col, linewidth=2, markersize=4)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Validation {metric.replace("_", " ").title()} Across Epochs')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'validation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved validation metrics to {self.plot_dir / 'validation_metrics.png'}")
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix for final epoch predictions."""
        if 'val_metrics' not in self.results or not self.results['val_metrics']:
            return
        
        # Get latest validation metrics
        latest_metrics = self.results['val_metrics'][-1]
        psi_columns = ['std_psi_min', 'std_psi_max', 'std_psi_mean', 'enh_psi_min', 'enh_psi_max', 'enh_psi_mean']
        
        # Extract correlation values
        correlation_data = []
        for psi_col in psi_columns:
            if psi_col in latest_metrics:
                correlation_data.append({
                    'Target': psi_col,
                    'Pearson R': latest_metrics[psi_col].get('pearson_r', np.nan),
                    'Spearman R': latest_metrics[psi_col].get('spearman_r', np.nan),
                    'R²': latest_metrics[psi_col].get('r2', np.nan),
                    'MSE': latest_metrics[psi_col].get('mse', np.nan),
                    'MAE': latest_metrics[psi_col].get('mae', np.nan)
                })
        
        if correlation_data:
            df = pd.DataFrame(correlation_data)
            df = df.set_index('Target')
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Correlation heatmap
            corr_metrics = df[['Pearson R', 'Spearman R', 'R²']]
            sns.heatmap(corr_metrics, annot=True, cmap='RdYlBu_r', center=0.5, 
                       ax=axes[0], cbar_kws={'label': 'Correlation'})
            axes[0].set_title('Correlation Metrics (Final Epoch)')
            
            # Error metrics heatmap
            error_metrics = df[['MSE', 'MAE']]
            sns.heatmap(error_metrics, annot=True, cmap='Reds', 
                       ax=axes[1], cbar_kws={'label': 'Error'})
            axes[1].set_title('Error Metrics (Final Epoch)')
            
            plt.tight_layout()
            plt.savefig(self.plot_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved correlation matrix to {self.plot_dir / 'correlation_matrix.png'}")
    
    def plot_learning_curves_comparison(self):
        """Plot learning curves showing overfitting analysis."""
        if 'train_loss' not in self.results or 'val_metrics' not in self.results:
            return
        
        train_losses = self.results['train_loss']
        val_losses = [epoch_metrics['overall']['mse'] for epoch_metrics in self.results['val_metrics']]
        
        epochs = range(1, min(len(train_losses), len(val_losses)) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Learning curves
        axes[0].plot(epochs, train_losses[:len(epochs)], 'b-o', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, val_losses[:len(epochs)], 'r-s', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].set_title('Learning Curves')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_yscale('log')
        
        # Overfitting ratio (val_loss / train_loss)
        overfitting_ratio = [v/t for v, t in zip(val_losses[:len(epochs)], train_losses[:len(epochs)])]
        axes[1].plot(epochs, overfitting_ratio, 'g-^', linewidth=2, markersize=6)
        axes[1].axhline(y=1.0, color='orange', linestyle='--', label='Perfect fit line')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Loss / Training Loss')
        axes[1].set_title('Overfitting Analysis')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'learning_curves_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved learning curves analysis to {self.plot_dir / 'learning_curves_analysis.png'}")
    
    def create_all_plots(self):
        """Create all plots."""
        print(f"Creating plots for training run: {self.checkpoint_dir}")
        
        self.plot_loss_curves()
        self.plot_validation_metrics()
        self.plot_correlation_matrix()
        self.plot_learning_curves_comparison()
        
        print(f"All plots saved to: {self.plot_dir}")
        
        # Create summary report
        self._create_summary_report()
    
    def _create_summary_report(self):
        """Create a summary report of training progress."""
        if 'train_loss' not in self.results:
            return
        
        summary_file = self.plot_dir / 'training_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("TileFormer Training Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic info
            f.write(f"Run directory: {self.checkpoint_dir}\n")
            f.write(f"Total epochs: {len(self.results['train_loss'])}\n\n")
            
            # Loss summary
            final_train_loss = self.results['train_loss'][-1]
            f.write(f"Final training loss: {final_train_loss:.6f}\n")
            
            if 'val_metrics' in self.results and self.results['val_metrics']:
                final_val_loss = self.results['val_metrics'][-1]['overall']['mse']
                f.write(f"Final validation loss: {final_val_loss:.6f}\n")
                f.write(f"Overfitting ratio: {final_val_loss/final_train_loss:.4f}\n\n")
                
                # Best metrics
                best_val_loss = min([m['overall']['mse'] for m in self.results['val_metrics']])
                best_epoch = [m['overall']['mse'] for m in self.results['val_metrics']].index(best_val_loss) + 1
                f.write(f"Best validation loss: {best_val_loss:.6f} (Epoch {best_epoch})\n\n")
                
                # Final correlations
                final_metrics = self.results['val_metrics'][-1]
                f.write("Final Validation Correlations:\n")
                for psi_col in ['std_psi_min', 'std_psi_max', 'std_psi_mean']:
                    if psi_col in final_metrics:
                        pearson_r = final_metrics[psi_col].get('pearson_r', 'N/A')
                        r2 = final_metrics[psi_col].get('r2', 'N/A')
                        f.write(f"  {psi_col}: Pearson R = {pearson_r:.4f}, R² = {r2:.4f}\n")
        
        print(f"Training summary saved to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot TileFormer training metrics")
    parser.add_argument('checkpoint_dir', type=str, help='Path to checkpoint directory')
    parser.add_argument('--live', action='store_true', help='Update plots continuously')
    
    args = parser.parse_args()
    
    plotter = TrainingPlotter(args.checkpoint_dir)
    
    if args.live:
        import time
        print("Live plotting mode - updating every 30 seconds...")
        while True:
            try:
                plotter = TrainingPlotter(args.checkpoint_dir)  # Reload data
                plotter.create_all_plots()
                time.sleep(30)
            except KeyboardInterrupt:
                print("Stopped live plotting")
                break
    else:
        plotter.create_all_plots()

if __name__ == "__main__":
    main()