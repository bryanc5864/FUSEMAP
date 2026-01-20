#!/usr/bin/env python3
"""
TileFormer Training Orchestrator
Trains the model on ABPS-labeled electrostatic potential data
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Tuple, Optional, List

# Add models directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.tileformer_architecture import TileFormer, TileFormerWithMetadata
from models.evaluation_metrics import ComprehensiveEvaluator

# Try to import plotting (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib not available - plots will not be generated")

# Setup logging will be configured in main() after args are parsed
logger = logging.getLogger(__name__)

# Nucleotide to index mapping
NUCLEOTIDE_MAP = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}

class TileDataset(Dataset):
    def __init__(self, data: pd.DataFrame, max_len: int = 200):
        self.data = data
        self.max_len = max_len
        
        # Data should already be fixed, but verify no NaN values
        nan_counts = {
            'gc_content': data['gc_content'].isna().sum(),
            'cpg_density': data['cpg_density'].isna().sum(), 
            'minor_groove_score': data['minor_groove_score'].isna().sum()
        }
        if any(count > 0 for count in nan_counts.values()):
            logger.warning(f"Found unexpected NaN values: {nan_counts}")
            raise ValueError("Dataset contains NaN values. Please run fix_nan_values.py first.")
        
        # Extract features
        self.sequences = data['sequence'].values
        self.metadata = data[['gc_content', 'cpg_density', 'minor_groove_score']].values.astype(np.float32)
        
        # Extract targets (ABPS values)
        target_cols = ['std_psi_min', 'std_psi_max', 'std_psi_mean',
                      'enh_psi_min', 'enh_psi_max', 'enh_psi_mean']
        self.targets = data[target_cols].values.astype(np.float32)
        
        # Store full dataframe for subset analysis
        self.full_data = data
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Convert sequence to indices
        seq = self.sequences[idx]
        seq_indices = [NUCLEOTIDE_MAP.get(nt, 4) for nt in seq.upper()]
        
        # Pad or truncate to max_len
        if len(seq_indices) < self.max_len:
            seq_indices = seq_indices + [4] * (self.max_len - len(seq_indices))
        else:
            seq_indices = seq_indices[:self.max_len]
        
        return {
            'sequence': torch.tensor(seq_indices, dtype=torch.long),
            'metadata': torch.tensor(self.metadata[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32)
        }

def load_presplit_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-split train/val/test datasets."""
    data_dir = Path(data_dir)
    
    train_path = data_dir / 'train.tsv'
    val_path = data_dir / 'val.tsv'
    test_path = data_dir / 'test.tsv'
    
    logger.info(f"Loading pre-split datasets from {data_dir}")
    
    # Load datasets
    train_data = pd.read_csv(train_path, sep='\t')
    val_data = pd.read_csv(val_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')
    
    logger.info(f"Loaded datasets - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Verify ABPS columns are present
    target_cols = ['std_psi_min', 'std_psi_max', 'std_psi_mean',
                  'enh_psi_min', 'enh_psi_max', 'enh_psi_mean']
    for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
        missing_cols = [col for col in target_cols if col not in dataset.columns]
        if missing_cols:
            raise ValueError(f"Missing ABPS columns in {name} set: {missing_cols}")
    
    return train_data, val_data, test_data

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    checkpoint_dir: Path = None
) -> Tuple[float, List[float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    step_losses = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        sequences = batch['sequence'].to(device)
        metadata = batch['metadata'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences, metadata)
        predictions = outputs['psi']
        
        # Compute loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        step_losses.append(loss.item())
        
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Log every 100 steps
        if batch_idx % 100 == 0 and batch_idx > 0:
            logger.info(f"Epoch {epoch}, Step {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
    
    # Save step losses to file
    if checkpoint_dir:
        step_loss_file = checkpoint_dir / f'step_losses_epoch_{epoch}.txt'
        np.savetxt(step_loss_file, step_losses)
    
    return total_loss / n_batches, step_losses

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    evaluator: ComprehensiveEvaluator,
    device: torch.device,
    dataset_name: str = "Validation"
) -> Dict:
    """Validate model and compute comprehensive metrics."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    metadata_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            sequences = batch['sequence'].to(device)
            metadata = batch['metadata'].to(device)
            targets = batch['target']
            
            outputs = model(sequences, metadata)
            predictions = outputs['psi'].cpu().numpy()
            
            all_predictions.append(predictions)
            all_targets.append(targets.numpy())
            
            if 'uncertainty' in outputs:
                all_uncertainties.append(outputs['uncertainty'].cpu().numpy())
            
            # Store metadata for subset analysis
            metadata_list.append(metadata.cpu().numpy())
    
    # Concatenate all batches
    y_pred = np.vstack(all_predictions)
    y_true = np.vstack(all_targets)
    uncertainty = np.vstack(all_uncertainties) if all_uncertainties else None
    
    # Create metadata dataframe for subset analysis
    metadata_array = np.vstack(metadata_list)
    metadata_df = pd.DataFrame(
        metadata_array,
        columns=['gc_content', 'cpg_density', 'minor_groove_score']
    )
    
    # Compute comprehensive metrics
    metrics = evaluator.compute_all_metrics(y_true, y_pred, uncertainty, metadata_df)
    
    return metrics

def _create_training_plots(history: Dict, checkpoint_dir: Path, current_epoch: int):
    """Create training plots at checkpoints."""
    if not PLOTTING_AVAILABLE:
        return
    
    plot_dir = checkpoint_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Loss curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        if 'train_loss' in history and history['train_loss']:
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0, 0].plot(epochs, history['train_loss'], 'b-o', linewidth=2)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('MSE Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Validation loss
        if 'val_metrics' in history and history['val_metrics']:
            val_losses = [m['overall']['mse'] for m in history['val_metrics']]
            epochs = range(1, len(val_losses) + 1)
            axes[0, 1].plot(epochs, val_losses, 'r-s', linewidth=2)
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MSE Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Training vs Validation
        if 'train_loss' in history and 'val_metrics' in history:
            train_epochs = range(1, len(history['train_loss']) + 1)
            val_epochs = range(1, len(val_losses) + 1)
            axes[1, 0].plot(train_epochs, history['train_loss'], 'b-o', label='Training')
            axes[1, 0].plot(val_epochs, val_losses, 'r-s', label='Validation')
            axes[1, 0].set_title('Training vs Validation Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MSE Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # Pearson correlation for std_psi_min
        if 'val_metrics' in history and history['val_metrics']:
            pearson_r = []
            for m in history['val_metrics']:
                if 'std_psi_min' in m and 'pearson_r' in m['std_psi_min']:
                    pearson_r.append(m['std_psi_min']['pearson_r'])
                else:
                    pearson_r.append(np.nan)
            
            if pearson_r:
                epochs = range(1, len(pearson_r) + 1)
                axes[1, 1].plot(epochs, pearson_r, 'g-^', linewidth=2)
                axes[1, 1].set_title('Validation Pearson R (std_psi_min)')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Pearson R')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / f'training_progress_epoch_{current_epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {plot_dir}")
        
    except Exception as e:
        logger.warning(f"Failed to create plots: {e}")

def main(args):
    # Create checkpoint directory first
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to save in the checkpoint directory
    log_file = checkpoint_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ],
        force=True  # Override any existing configuration
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting training run: {checkpoint_dir}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set device and GPU
    if torch.cuda.is_available():
        if args.gpu_id is not None:
            device = torch.device(f'cuda:{args.gpu_id}')
            torch.cuda.set_device(args.gpu_id)
            logger.info(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
        else:
            device = torch.device('cuda')
            logger.info(f"Using default GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.warning("CUDA not available, using CPU")
    
    logger.info(f"Device: {device}")
    
    # Load pre-split data
    train_data, val_data, test_data = load_presplit_data(args.data_dir)
    
    # Create datasets
    train_dataset = TileDataset(train_data, max_len=args.max_len)
    val_dataset = TileDataset(val_data, max_len=args.max_len)
    test_dataset = TileDataset(test_data, max_len=args.max_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = TileFormerWithMetadata(
        vocab_size=5,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
        output_dim=6,
        predict_uncertainty=args.predict_uncertainty,
        metadata_dim=3
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Training history
    history = {
        'train_loss': [],
        'val_metrics': [],
        'test_metrics': None
    }
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, step_losses = train_epoch(model, train_loader, criterion, optimizer, device, epoch, checkpoint_dir)
        history['train_loss'].append(train_loss)
        history[f'step_losses_epoch_{epoch}'] = step_losses
        logger.info(f"Train Loss: {train_loss:.6f}, Min step loss: {min(step_losses):.6f}, Max step loss: {max(step_losses):.6f}")
        
        # Validate
        val_metrics = validate(model, val_loader, evaluator, device, "Validation")
        history['val_metrics'].append(val_metrics)
        
        # Also evaluate on small train subset for overfitting detection (every 5 epochs)
        if epoch % 5 == 0:
            logger.info("Evaluating on train subset for overfitting check...")
            train_subset = train_dataset.data.sample(n=min(1000, len(train_dataset.data)))
            train_subset_dataset = TileDataset(train_subset, max_len=args.max_len)
            train_subset_loader = DataLoader(train_subset_dataset, batch_size=args.batch_size, shuffle=False)
            train_subset_metrics = validate(model, train_subset_loader, evaluator, device, "Train Subset")
            
            train_mse = train_subset_metrics['overall']['mse']
            logger.info(f"Train Subset MSE: {train_mse:.6f}")
            logger.info(f"Overfitting ratio (val/train): {val_loss/train_mse:.4f}")
        
        val_loss = val_metrics['overall']['mse']
        logger.info(f"Validation MSE: {val_loss:.6f}")
        logger.info(f"Validation RMSE: {val_metrics['overall']['rmse']:.6f}")
        logger.info(f"Validation MAE: {val_metrics['overall']['mae']:.6f}")
        
        # Log comprehensive metrics for first output (std_psi_min as example)
        if 'std_psi_min' in val_metrics:
            logger.info(f"Validation Pearson R: {val_metrics['std_psi_min']['pearson_r']:.4f}")
            logger.info(f"Validation Spearman R: {val_metrics['std_psi_min']['spearman_r']:.4f}")
            logger.info(f"Validation R²: {val_metrics['std_psi_min']['r2']:.4f}")
            logger.info(f"Validation Explained Variance: {val_metrics['std_psi_min']['explained_variance']:.4f}")
        
        # Log calibration metrics if available
        if 'calibration' in val_metrics:
            logger.info(f"Validation 1-sigma coverage: {val_metrics['calibration'].get('1_sigma_coverage', 0):.4f}")
            logger.info(f"Validation 2-sigma coverage: {val_metrics['calibration'].get('2_sigma_coverage', 0):.4f}")
        
        # Save detailed metrics report for this epoch
        metrics_report_file = checkpoint_dir / f'metrics_epoch_{epoch}.txt'
        with open(metrics_report_file, 'w') as f:
            f.write(evaluator.format_metrics_report(val_metrics))
        logger.info(f"Detailed metrics saved to {metrics_report_file}")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint and create plots
        if epoch % 5 == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
            
            # Create interim plots
            if PLOTTING_AVAILABLE:
                _create_training_plots(history, checkpoint_dir, epoch)
    
    # Load best model for final evaluation
    logger.info("\nLoading best model for final evaluation")
    checkpoint = torch.load(checkpoint_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test set evaluation
    logger.info("Evaluating on test set")
    test_metrics = validate(model, test_loader, evaluator, device, "Test")
    history['test_metrics'] = test_metrics
    
    # Print comprehensive test report
    report = evaluator.format_metrics_report(test_metrics)
    logger.info("\n" + report)
    
    # Save results
    results_path = checkpoint_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_history = convert_to_serializable(history)
        json.dump(serializable_history, f, indent=2)
    
    logger.info(f"\nTraining complete! Results saved to {results_path}")
    
    # Save the final report as text
    report_path = checkpoint_dir / 'final_evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Evaluation report saved to {report_path}")
    
    # Create final comprehensive plots
    if PLOTTING_AVAILABLE:
        logger.info("Creating final training plots...")
        _create_training_plots(history, checkpoint_dir, args.epochs)
        
        # Run the comprehensive plotting script
        try:
            from plot_training_metrics import TrainingPlotter
            plotter = TrainingPlotter(str(checkpoint_dir))
            plotter.create_all_plots()
        except Exception as e:
            logger.warning(f"Failed to create comprehensive plots: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TileFormer model")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                       default='/home/bcheng/TileFormer/data/splits',
                       help='Directory containing pre-split train.tsv, val.tsv, test.tsv files')
    parser.add_argument('--max_len', type=int, default=200,
                       help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=1024,
                       help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--predict_uncertainty', action='store_true',
                       help='Predict uncertainty estimates')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Other arguments
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='GPU ID to use (e.g., 1 or 9). If not specified, uses default GPU.')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default=f'checkpoints/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    main(args)