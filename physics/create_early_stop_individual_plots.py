import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_early_stopping_point(val_losses, patience=10, min_delta=0.001):
    """
    Find early stopping point based on validation loss plateau.
    More conservative - waits longer to detect true plateau.

    Args:
        val_losses: List of validation losses
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement (relative to best)

    Returns:
        Epoch index of early stopping point (or None if should train to end)
    """
    if len(val_losses) < patience + 5:
        return None

    best_loss = float('inf')
    best_epoch = 0
    wait = 0

    for epoch, loss in enumerate(val_losses):
        # Calculate relative improvement
        if best_loss != float('inf'):
            improvement = (best_loss - loss) / abs(best_loss) if best_loss != 0 else (best_loss - loss)
        else:
            improvement = float('inf')

        if loss < best_loss and improvement > min_delta:
            best_loss = loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                # Found plateau - return best epoch
                return best_epoch

    return None  # No early stopping needed

def create_individual_early_stop_plots(cell_type):
    """Create 6 individual plots for each PhysInformer model, ending at early stop."""

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

    # Pearson data
    train_pearson = [e.get('train_pearson') for e in epochs]
    val_pearson = [e.get('val_pearson') for e in epochs]

    # Auxiliary head losses (both train and val)
    train_aux_a_loss = [e.get('train_aux_a_loss') for e in epochs]
    val_aux_a_loss = [e.get('val_aux_a_loss') for e in epochs]
    train_aux_b_loss = [e.get('train_aux_b_loss') for e in epochs]
    val_aux_b_loss = [e.get('val_aux_b_loss') for e in epochs]

    # Auxiliary head Pearson (validation only in the parsed JSON)
    aux_a_val_pearson = [e.get('aux_a_pearson') for e in epochs]
    aux_b_val_pearson = [e.get('aux_b_pearson') for e in epochs]

    # Find early stopping point based on validation loss
    val_total_clean = [v for v in val_total_loss if v is not None]
    early_stop = find_early_stopping_point(val_total_clean, patience=10, min_delta=0.001)

    if early_stop is None:
        early_stop = len(epoch_nums) - 1
        print(f"  No early stopping detected, using all {len(epoch_nums)} epochs")
    else:
        print(f"  Early stopping at epoch: {early_stop + 1}")
        print(f"    Best val loss: {val_total_clean[early_stop]:.4f}")

    # Truncate data at early stop point
    cutoff = early_stop + 1
    epoch_nums_cut = epoch_nums[:cutoff]
    train_total_loss_cut = train_total_loss[:cutoff]
    val_total_loss_cut = val_total_loss[:cutoff]
    train_pearson_cut = train_pearson[:cutoff]
    val_pearson_cut = val_pearson[:cutoff]

    # Aux Losses
    train_aux_a_loss_cut = train_aux_a_loss[:cutoff]
    val_aux_a_loss_cut = val_aux_a_loss[:cutoff]
    train_aux_b_loss_cut = train_aux_b_loss[:cutoff]
    val_aux_b_loss_cut = val_aux_b_loss[:cutoff]

    # Aux Pearson
    aux_a_val_pearson_cut = aux_a_val_pearson[:cutoff]
    aux_b_val_pearson_cut = aux_b_val_pearson[:cutoff]

    # === PLOT 1: Train vs Val Loss ===
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(epoch_nums_cut, train_total_loss_cut, 'b-o', label='Train Loss',
            linewidth=2.5, markersize=5, alpha=0.8)
    ax.plot(epoch_nums_cut, val_total_loss_cut, 'r-s', label='Val Loss',
            linewidth=2.5, markersize=5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Loss', fontsize=14, fontweight='bold')
    ax.set_title(f'{cell_type} - Total Loss (Epochs 1-{early_stop + 1})',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(base_dir / f'{cell_type}_loss_early_stop.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {cell_type}_loss_early_stop.png")

    # === PLOT 2: Train vs Val Overall Pearson ===
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(epoch_nums_cut, train_pearson_cut, 'b-o', label='Train Overall Pearson',
            linewidth=2.5, markersize=5, alpha=0.8)
    ax.plot(epoch_nums_cut, val_pearson_cut, 'r-s', label='Val Overall Pearson',
            linewidth=2.5, markersize=5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pearson Correlation', fontsize=14, fontweight='bold')
    ax.set_title(f'{cell_type} - Overall Pearson (Epochs 1-{early_stop + 1})',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    plt.tight_layout()
    plt.savefig(base_dir / f'{cell_type}_pearson_overall_early_stop.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {cell_type}_pearson_overall_early_stop.png")

    # === PLOT 3: Aux Head A Loss ===
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(epoch_nums_cut, train_aux_a_loss_cut, 'b-o', label='Train Aux A Loss',
            linewidth=2.5, markersize=5, alpha=0.8)
    ax.plot(epoch_nums_cut, val_aux_a_loss_cut, 'r-s', label='Val Aux A Loss',
            linewidth=2.5, markersize=5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title(f'{cell_type} - Auxiliary Head A Loss (Epochs 1-{early_stop + 1})',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(base_dir / f'{cell_type}_aux_head_a_loss_early_stop.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {cell_type}_aux_head_a_loss_early_stop.png")

    # === PLOT 4: Aux Head B Loss ===
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(epoch_nums_cut, train_aux_b_loss_cut, 'b-o', label='Train Aux B Loss',
            linewidth=2.5, markersize=5, alpha=0.8)
    ax.plot(epoch_nums_cut, val_aux_b_loss_cut, 'r-s', label='Val Aux B Loss',
            linewidth=2.5, markersize=5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title(f'{cell_type} - Auxiliary Head B Loss (Epochs 1-{early_stop + 1})',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(base_dir / f'{cell_type}_aux_head_b_loss_early_stop.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {cell_type}_aux_head_b_loss_early_stop.png")

    # === PLOT 5: Aux Head A Pearson ===
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(epoch_nums_cut, aux_a_val_pearson_cut, 'r-s', label='Val Aux Head A Pearson',
            linewidth=2.5, markersize=5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pearson Correlation', fontsize=14, fontweight='bold')
    ax.set_title(f'{cell_type} - Auxiliary Head A Pearson (Epochs 1-{early_stop + 1})',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)

    # Set y-limits based on actual data
    aux_a_clean = [v for v in aux_a_val_pearson_cut if v is not None]
    if aux_a_clean:
        y_min = min(aux_a_clean) - 0.05
        y_max = max(aux_a_clean) + 0.05
        ax.set_ylim([max(0, y_min), min(1.0, y_max)])

    plt.tight_layout()
    plt.savefig(base_dir / f'{cell_type}_aux_head_a_pearson_early_stop.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {cell_type}_aux_head_a_pearson_early_stop.png")

    # === PLOT 6: Aux Head B Pearson ===
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(epoch_nums_cut, aux_b_val_pearson_cut, 'r-s', label='Val Aux Head B Pearson',
            linewidth=2.5, markersize=5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pearson Correlation', fontsize=14, fontweight='bold')
    ax.set_title(f'{cell_type} - Auxiliary Head B Pearson (Epochs 1-{early_stop + 1})',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)

    # Set y-limits based on actual data
    aux_b_clean = [v for v in aux_b_val_pearson_cut if v is not None]
    if aux_b_clean:
        y_min = min(aux_b_clean) - 0.05
        y_max = max(aux_b_clean) + 0.05
        ax.set_ylim([max(0, y_min), min(1.0, y_max)])

    plt.tight_layout()
    plt.savefig(base_dir / f'{cell_type}_aux_head_b_pearson_early_stop.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {cell_type}_aux_head_b_pearson_early_stop.png")

def create_tileformer_early_stop_plot():
    """Create plots for TileFormer, ending at early stop."""

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
    pearson_r = [e.get('pearson_r') for e in epochs]

    # Find early stopping point - use more conservative settings
    mse_clean = [v for v in mse if v is not None]
    early_stop = find_early_stopping_point(mse_clean, patience=8, min_delta=0.0001)

    if early_stop is None:
        early_stop = len(epoch_nums) - 1
        print(f"  No early stopping detected, using all {len(epoch_nums)} epochs")
    else:
        print(f"  Early stopping at epoch: {early_stop + 1}")
        print(f"    Best MSE: {mse_clean[early_stop]:.6f}")

    # Truncate data
    cutoff = early_stop + 1
    epoch_nums_cut = epoch_nums[:cutoff]
    mse_cut = mse[:cutoff]
    pearson_r_cut = pearson_r[:cutoff]

    # === PLOT 1: MSE ===
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(epoch_nums_cut, mse_cut, 'b-o', label='MSE',
            linewidth=2.5, markersize=5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=14, fontweight='bold')
    ax.set_title(f'TileFormer - MSE (Epochs 1-{early_stop + 1})',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(base_dir / 'TileFormer_mse_early_stop.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: TileFormer_mse_early_stop.png")

    # === PLOT 2: Pearson R ===
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(epoch_nums_cut, pearson_r_cut, 'b-o', label='Pearson R',
            linewidth=2.5, markersize=5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pearson Correlation', fontsize=14, fontweight='bold')
    ax.set_title(f'TileFormer - Pearson R (Epochs 1-{early_stop + 1})',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.2, 1.0])
    plt.tight_layout()
    plt.savefig(base_dir / 'TileFormer_pearson_early_stop.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: TileFormer_pearson_early_stop.png")

if __name__ == '__main__':
    print("="*80)
    print("CREATING INDIVIDUAL EARLY STOP PLOTS")
    print("Plots will END at early stopping point (no green line)")
    print("="*80)

    # Process all PhysInformer models (6 plots each)
    for cell_type in ['S2', 'WTC11', 'HepG2', 'K562']:
        create_individual_early_stop_plots(cell_type)

    # Process TileFormer (2 plots)
    create_tileformer_early_stop_plot()

    print("\n" + "="*80)
    print("COMPLETE!")
    print("  PhysInformer: 6 plots per cell type × 4 cell types = 24 plots")
    print("    - Total Loss (train vs val)")
    print("    - Overall Pearson (train vs val)")
    print("    - Aux Head A Loss (train vs val)")
    print("    - Aux Head B Loss (train vs val)")
    print("    - Aux Head A Pearson (val only)")
    print("    - Aux Head B Pearson (val only)")
    print("  TileFormer: 2 plots")
    print("  Total: 26 individual early stop plots")
    print("="*80)
