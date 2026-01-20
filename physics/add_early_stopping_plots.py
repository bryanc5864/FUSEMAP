import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_early_stopping_point(val_losses, patience=5, min_delta=0.01):
    """
    Find early stopping point based on validation loss plateau.

    Args:
        val_losses: List of validation losses
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement

    Returns:
        Epoch index of early stopping point (or None if should train to end)
    """
    if len(val_losses) < patience + 1:
        return None

    best_loss = float('inf')
    best_epoch = 0
    wait = 0

    for epoch, loss in enumerate(val_losses):
        if loss < best_loss - min_delta:
            best_loss = loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                return best_epoch

    return None  # No early stopping needed

def create_early_stop_plots_physinformer(cell_type):
    """Create training plots with early stopping indicators for PhysInformer."""

    base_dir = Path(f'results/PhysInformer_{cell_type}')
    json_path = base_dir / 'parsed_epochs.json'

    if not json_path.exists():
        print(f"No data found for {cell_type}")
        return

    with open(json_path) as f:
        epochs = json.load(f)

    if not epochs:
        print(f"Empty data for {cell_type}")
        return

    print(f"\nProcessing {cell_type}...")

    # Extract data
    epoch_nums = [e['epoch'] for e in epochs]

    # Loss data
    train_total_loss = [e.get('train_total_loss') for e in epochs]
    val_total_loss = [e.get('val_total_loss') for e in epochs]

    train_desc_loss = [e.get('train_descriptor_loss') for e in epochs]
    val_desc_loss = [e.get('val_descriptor_loss') for e in epochs]

    train_aux_loss = [e.get('train_auxiliary_loss') for e in epochs]
    val_aux_loss = [e.get('val_auxiliary_loss') for e in epochs]

    # Pearson data
    train_pearson = [e.get('train_pearson') for e in epochs]
    val_pearson = [e.get('val_pearson') for e in epochs]

    train_desc_mean = [e.get('train_descriptor_mean_pearson') for e in epochs]
    val_desc_mean = [e.get('val_descriptor_mean_pearson') for e in epochs]

    # Find early stopping points
    val_total_clean = [v for v in val_total_loss if v is not None]
    val_desc_clean = [v for v in val_desc_loss if v is not None]

    early_stop_total = find_early_stopping_point(val_total_clean, patience=5, min_delta=0.5)
    early_stop_desc = find_early_stopping_point(val_desc_clean, patience=5, min_delta=0.5)

    if early_stop_total:
        print(f"  Early stopping for total loss at epoch: {early_stop_total + 1}")
    if early_stop_desc:
        print(f"  Early stopping for descriptor loss at epoch: {early_stop_desc + 1}")

    # Create loss curves with early stopping
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{cell_type} Training Progress - With Early Stopping Indicators', fontsize=16, fontweight='bold')

    # Total Loss
    ax = axes[0, 0]
    ax.plot(epoch_nums, train_total_loss, 'b-o', label='Train Loss', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(epoch_nums, val_total_loss, 'r-s', label='Val Loss', linewidth=2, markersize=4, alpha=0.7)
    if early_stop_total:
        ax.axvline(x=early_stop_total + 1, color='green', linestyle='--', linewidth=2,
                   label=f'Early Stop (epoch {early_stop_total + 1})', alpha=0.8)
        ax.scatter([early_stop_total + 1], [val_total_clean[early_stop_total]],
                  color='green', s=200, marker='*', zorder=5, edgecolors='black', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Descriptor Loss
    ax = axes[0, 1]
    ax.plot(epoch_nums, train_desc_loss, 'b-o', label='Train Descriptor Loss', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(epoch_nums, val_desc_loss, 'r-s', label='Val Descriptor Loss', linewidth=2, markersize=4, alpha=0.7)
    if early_stop_desc:
        ax.axvline(x=early_stop_desc + 1, color='green', linestyle='--', linewidth=2,
                   label=f'Early Stop (epoch {early_stop_desc + 1})', alpha=0.8)
        ax.scatter([early_stop_desc + 1], [val_desc_clean[early_stop_desc]],
                  color='green', s=200, marker='*', zorder=5, edgecolors='black', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Descriptor Loss', fontsize=12)
    ax.set_title('Descriptor Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Auxiliary Loss
    ax = axes[1, 0]
    ax.plot(epoch_nums, train_aux_loss, 'b-o', label='Train Auxiliary Loss', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(epoch_nums, val_aux_loss, 'r-s', label='Val Auxiliary Loss', linewidth=2, markersize=4, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Auxiliary Loss', fontsize=12)
    ax.set_title('Auxiliary Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Overall Pearson
    ax = axes[1, 1]
    ax.plot(epoch_nums, train_pearson, 'b-o', label='Train Pearson', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(epoch_nums, val_pearson, 'r-s', label='Val Pearson', linewidth=2, markersize=4, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Overall Pearson Correlation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(base_dir / 'loss_curves_early_stop.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: loss_curves_early_stop.png")

    # Create Pearson evolution with early stopping
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{cell_type} Pearson Evolution - With Early Stopping Indicators', fontsize=16, fontweight='bold')

    # Find early stopping for Pearson (look for plateau in improvement)
    val_pearson_clean = [v for v in val_pearson if v is not None]
    # For Pearson, we want maximum, so invert the logic
    val_pearson_inverted = [-v for v in val_pearson_clean]
    early_stop_pearson = find_early_stopping_point(val_pearson_inverted, patience=5, min_delta=0.001)

    if early_stop_pearson:
        print(f"  Early stopping for Pearson at epoch: {early_stop_pearson + 1}")

    # Overall Pearson
    ax = axes[0, 0]
    ax.plot(epoch_nums, train_pearson, 'b-o', label='Train Overall Pearson', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(epoch_nums, val_pearson, 'r-s', label='Val Overall Pearson', linewidth=2, markersize=4, alpha=0.7)
    if early_stop_pearson:
        ax.axvline(x=early_stop_pearson + 1, color='green', linestyle='--', linewidth=2,
                   label=f'Early Stop (epoch {early_stop_pearson + 1})', alpha=0.8)
        ax.scatter([early_stop_pearson + 1], [val_pearson_clean[early_stop_pearson]],
                  color='green', s=200, marker='*', zorder=5, edgecolors='black', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Overall Pearson Correlation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])

    # Descriptor Mean Pearson
    ax = axes[0, 1]
    ax.plot(epoch_nums, train_desc_mean, 'b-o', label='Train Descriptor Mean', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(epoch_nums, val_desc_mean, 'r-s', label='Val Descriptor Mean', linewidth=2, markersize=4, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Pearson', fontsize=12)
    ax.set_title('Descriptor Mean Pearson', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])

    # Descriptor Median Pearson
    train_desc_median = [e.get('train_descriptor_median_pearson') for e in epochs]
    val_desc_median = [e.get('val_descriptor_median_pearson') for e in epochs]

    ax = axes[1, 0]
    ax.plot(epoch_nums, train_desc_median, 'b-o', label='Train Descriptor Median', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(epoch_nums, val_desc_median, 'r-s', label='Val Descriptor Median', linewidth=2, markersize=4, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Median Pearson', fontsize=12)
    ax.set_title('Descriptor Median Pearson', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])

    # Auxiliary Head Performance
    aux_a_train = [e.get('train_auxiliary_head_a_pearson') for e in epochs]
    aux_a_val = [e.get('val_auxiliary_head_a_pearson') for e in epochs]

    ax = axes[1, 1]
    ax.plot(epoch_nums, aux_a_train, 'b-o', label='Train Aux Head A', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(epoch_nums, aux_a_val, 'r-s', label='Val Aux Head A', linewidth=2, markersize=4, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Auxiliary Head A Pearson', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(base_dir / 'pearson_evolution_early_stop.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: pearson_evolution_early_stop.png")

def create_early_stop_plots_tileformer():
    """Create training plots with early stopping indicators for TileFormer."""

    base_dir = Path('results/TileFormer')
    json_path = base_dir / 'parsed_epochs.json'

    if not json_path.exists():
        print("No TileFormer data found")
        return

    with open(json_path) as f:
        epochs = json.load(f)

    if not epochs:
        print("Empty TileFormer data")
        return

    print("\nProcessing TileFormer...")

    # Extract data
    epoch_nums = [e['epoch'] for e in epochs]
    mse = [e.get('mse') for e in epochs]
    rmse = [e.get('rmse') for e in epochs]
    mae = [e.get('mae') for e in epochs]
    pearson_r = [e.get('pearson_r') for e in epochs]
    spearman_r = [e.get('spearman_r') for e in epochs]
    r2 = [e.get('r2') for e in epochs]

    # Find early stopping points
    mse_clean = [v for v in mse if v is not None]
    early_stop_mse = find_early_stopping_point(mse_clean, patience=3, min_delta=0.00001)

    if early_stop_mse:
        print(f"  Early stopping at epoch: {early_stop_mse + 1}")

    # Create comprehensive metrics plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('TileFormer Training Progress - With Early Stopping Indicators', fontsize=16, fontweight='bold')

    # MSE
    ax = axes[0, 0]
    ax.plot(epoch_nums, mse, 'b-o', label='MSE', linewidth=2, markersize=6, alpha=0.7)
    if early_stop_mse:
        ax.axvline(x=early_stop_mse + 1, color='green', linestyle='--', linewidth=2,
                   label=f'Early Stop (epoch {early_stop_mse + 1})', alpha=0.8)
        ax.scatter([early_stop_mse + 1], [mse_clean[early_stop_mse]],
                  color='green', s=200, marker='*', zorder=5, edgecolors='black', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Mean Squared Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # RMSE
    ax = axes[0, 1]
    ax.plot(epoch_nums, rmse, 'r-o', label='RMSE', linewidth=2, markersize=6, alpha=0.7)
    if early_stop_mse:
        rmse_at_stop = [v for v in rmse if v is not None][early_stop_mse]
        ax.axvline(x=early_stop_mse + 1, color='green', linestyle='--', linewidth=2,
                   label=f'Early Stop (epoch {early_stop_mse + 1})', alpha=0.8)
        ax.scatter([early_stop_mse + 1], [rmse_at_stop],
                  color='green', s=200, marker='*', zorder=5, edgecolors='black', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Root Mean Squared Error', fontsize=12)
    ax.set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # MAE
    ax = axes[0, 2]
    ax.plot(epoch_nums, mae, 'g-o', label='MAE', linewidth=2, markersize=6, alpha=0.7)
    if early_stop_mse:
        mae_at_stop = [v for v in mae if v is not None][early_stop_mse]
        ax.axvline(x=early_stop_mse + 1, color='green', linestyle='--', linewidth=2,
                   label=f'Early Stop (epoch {early_stop_mse + 1})', alpha=0.8)
        ax.scatter([early_stop_mse + 1], [mae_at_stop],
                  color='green', s=200, marker='*', zorder=5, edgecolors='black', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Pearson R
    ax = axes[1, 0]
    ax.plot(epoch_nums, pearson_r, 'm-o', label='Pearson R', linewidth=2, markersize=6, alpha=0.7)
    if early_stop_mse:
        pearson_at_stop = [v for v in pearson_r if v is not None][early_stop_mse]
        ax.axvline(x=early_stop_mse + 1, color='green', linestyle='--', linewidth=2,
                   label=f'Early Stop (epoch {early_stop_mse + 1})', alpha=0.8)
        ax.scatter([early_stop_mse + 1], [pearson_at_stop],
                  color='green', s=200, marker='*', zorder=5, edgecolors='black', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Pearson Correlation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.2, 1.0])

    # Spearman R
    ax = axes[1, 1]
    ax.plot(epoch_nums, spearman_r, 'c-o', label='Spearman R', linewidth=2, markersize=6, alpha=0.7)
    if early_stop_mse:
        spearman_at_stop = [v for v in spearman_r if v is not None][early_stop_mse]
        ax.axvline(x=early_stop_mse + 1, color='green', linestyle='--', linewidth=2,
                   label=f'Early Stop (epoch {early_stop_mse + 1})', alpha=0.8)
        ax.scatter([early_stop_mse + 1], [spearman_at_stop],
                  color='green', s=200, marker='*', zorder=5, edgecolors='black', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title('Spearman Correlation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.2, 1.0])

    # R²
    ax = axes[1, 2]
    ax.plot(epoch_nums, r2, 'y-o', label='R² Score', linewidth=2, markersize=6, alpha=0.7)
    if early_stop_mse:
        r2_at_stop = [v for v in r2 if v is not None][early_stop_mse]
        ax.axvline(x=early_stop_mse + 1, color='green', linestyle='--', linewidth=2,
                   label=f'Early Stop (epoch {early_stop_mse + 1})', alpha=0.8)
        ax.scatter([early_stop_mse + 1], [r2_at_stop],
                  color='green', s=200, marker='*', zorder=5, edgecolors='black', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² Score', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(base_dir / 'comprehensive_metrics_early_stop.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: comprehensive_metrics_early_stop.png")

if __name__ == '__main__':
    print("="*80)
    print("CREATING TRAINING PLOTS WITH EARLY STOPPING INDICATORS")
    print("="*80)

    # Process all PhysInformer models
    for cell_type in ['S2', 'WTC11', 'HepG2', 'K562']:
        create_early_stop_plots_physinformer(cell_type)

    # Process TileFormer
    create_early_stop_plots_tileformer()

    print("\n" + "="*80)
    print("COMPLETE! New plots saved with '_early_stop' suffix")
    print("="*80)
