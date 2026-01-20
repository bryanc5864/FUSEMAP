import torch
import torch.optim as optim
import numpy as np
import os
import logging
import json
import random
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict

from physics_aware_model import create_physics_aware_model, PhysicsAwareLoss
from dataset import create_dataloaders, create_plant_dataloaders
from metrics import MetricsCalculator, compute_feature_statistics
from plotting import create_training_plots, create_final_summary_plot

# Define plant cell types
PLANT_CELL_TYPES = ['arabidopsis', 'sorghum', 'maize']

def set_seed(seed: int):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def convert_to_serializable(obj):
    """Convert numpy and torch types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('PhysInformer')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_dir / 'training.log')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def save_checkpoint(
    model, optimizer, scheduler, epoch, loss, metrics, 
    checkpoint_path: Path, cell_type: str = None, 
    normalization_stats: Dict = None, seed: int = None, is_best: bool = False
):
    """Save model checkpoint with normalization stats and seed"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metrics': metrics,
        'cell_type': cell_type,
        'normalization_stats': normalization_stats,
        'seed': seed,
    }
    
    torch.save(checkpoint, checkpoint_path / f'checkpoint_epoch_{epoch}.pt')
    
    if is_best:
        torch.save(checkpoint, checkpoint_path / 'best_model.pt')

def train_epoch(model, dataloader, optimizer, scheduler, device, metrics_calculator, epoch, batch_log_file, loss_fn=None, feature_stats=None, max_grad_norm=1.0, cell_type=None, aux_optimizer=None, aux_loss_fn=None):
    """Train for one epoch with detailed batch logging"""
    model.train()
    total_losses = {'total': 0.0, 'desc': 0.0, 'aux_a': 0.0, 'aux_b': 0.0}
    all_predictions = {'descriptors': [], 'aux_head_a': [], 'aux_head_b': []}
    all_targets = {'descriptors': [], 'activities': []}
    all_per_feature_losses = []
    
    # Use PhysicsAwareLoss if not provided
    if loss_fn is None:
        loss_fn = PhysicsAwareLoss()
    
    # Check if dataset has activities
    has_activities = hasattr(dataloader.dataset, 'has_activities') and dataloader.dataset.has_activities
    
    # Load electrostatic data if available
    import numpy as np
    import os
    if cell_type is None:
        cell_type = 'HepG2'  # Fallback
    elec_data = None

    # Different paths for plant vs human cell types
    if cell_type in PLANT_CELL_TYPES:
        # Plant electrostatic data is in the processed data directory
        elec_file = f'../data/plant_data/jores2021/processed/{cell_type}/{cell_type}_train_tileformer_tileformer.npz'
    else:
        elec_file = f'../output/{cell_type}_train_tileformer_tileformer_vectors.npz'

    if os.path.exists(elec_file):
        npz_data = np.load(elec_file)
        # Handle different key names in npz files
        if 'predictions' in npz_data:
            elec_data = npz_data['predictions']
        elif 'arr_0' in npz_data:
            elec_data = npz_data['arr_0']
        else:
            # Try first available key
            keys = list(npz_data.keys())
            if keys:
                elec_data = npz_data[keys[0]]
    
    for batch_idx, batch in enumerate(dataloader):
        sequences = batch['sequence'].to(device)
        desc_targets = batch['descriptors'].to(device)
        
        # Get activities if available
        activities = None
        if has_activities and 'activities' in batch:
            activities = batch['activities'].to(device)
        
        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()
        
        # Forward pass - pass real features and activities for auxiliary heads
        if has_activities and activities is not None:
            predictions = model(sequences, real_features=desc_targets.detach(), real_activities=activities)
        else:
            predictions = model(sequences)
        
        # Prepare targets for physics-aware loss
        targets = {}
        # Use the actual descriptor columns from the dataset (after filtering)
        feature_cols = dataloader.dataset.descriptor_cols
        
        # Map descriptor columns to individual feature targets for physics loss
        for i, col in enumerate(feature_cols):
            # Physics model expects feature targets as feature_i
            feature_key = f'feature_{i}'
            targets[feature_key] = desc_targets[:, i]
        
        # Add electrostatic targets if available
        if elec_data is not None:
            # Get batch indices
            start_idx = batch_idx * dataloader.batch_size
            end_idx = min(start_idx + len(sequences), len(elec_data))
            batch_elec = elec_data[start_idx:end_idx]
            # Reshape to (batch, n_windows, features_per_window)
            # Dynamically determine n_windows from data shape (features / 6)
            n_features_per_window = 6
            n_windows = elec_data.shape[1] // n_features_per_window
            targets['electrostatic_windows'] = torch.tensor(
                batch_elec.reshape(len(sequences), n_windows, n_features_per_window),
                dtype=torch.float32, device=device
            )
        
        # Calculate loss
        losses = loss_fn(predictions, targets)
        
        # Handle case where losses is a dict
        if isinstance(losses, dict):
            total_loss = losses.get('total_loss', losses.get('total', torch.tensor(0.0, device=device)))
        else:
            total_loss = losses
        
        # Calculate auxiliary losses if available
        aux_losses = {}
        if has_activities and activities is not None and aux_loss_fn is not None:
            # Debug: Check what's in predictions
            has_aux_a = 'aux_activity_seq_feat' in predictions
            has_aux_b = 'aux_activity_feat_only' in predictions
            
            if has_aux_a or has_aux_b:
                aux_losses = aux_loss_fn(predictions, activities)
                # Add auxiliary losses for logging
                if 'aux_seq_feat_loss' in aux_losses:
                    total_losses['aux_a'] += aux_losses['aux_seq_feat_loss'].item()
                if 'aux_feat_only_loss' in aux_losses:
                    total_losses['aux_b'] += aux_losses['aux_feat_only_loss'].item()
            elif batch_idx == 0:
                # Debug print on first batch only
                print(f"DEBUG: No auxiliary predictions found. Keys in predictions: {list(predictions.keys())[:5]}...")
        
        # Backward pass for main model
        total_loss.backward(retain_graph=has_activities)
        
        # Backward pass for auxiliary heads if available
        if aux_losses and 'aux_total_loss' in aux_losses and aux_optimizer is not None:
            aux_losses['aux_total_loss'].backward()
        
        # Calculate gradient norm (before clipping)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        if aux_optimizer is not None:
            aux_optimizer.step()
        
        # Scheduler step (OneCycleLR needs per-batch update)
        scheduler.step()
        
        # Accumulate losses
        total_losses['total'] += total_loss.item()
        if isinstance(losses, dict):
            desc_loss = losses.get('desc_loss', total_loss)
            if hasattr(desc_loss, 'item'):
                total_losses['desc'] += desc_loss.item()
            else:
                total_losses['desc'] += float(desc_loss) if desc_loss is not None else total_loss.item()
            # Collect per-feature losses if available
            if 'per_feature_losses' in losses:
                all_per_feature_losses.append(losses['per_feature_losses'].detach())
        else:
            total_losses['desc'] += total_loss.item()
        
        # Per-batch logging (every 10 batches)
        if batch_idx % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            
            # Console logging
            total_loss_optimized = total_loss.item()
            desc_avg_loss_logged = total_loss.item()  # Simplified for physics loss
            
            # Add weight info if available
            weight_str = ""
            if isinstance(losses, dict) and "min_weight" in losses:
                weight_str = f' | Weights: [{losses["min_weight"]:.2f}-{losses["max_weight"]:.2f}]'
            
            # Add auxiliary loss info if available
            aux_str = ""
            if has_activities:
                # Show current batch auxiliary losses if available
                if aux_losses:
                    current_aux_a = aux_losses.get('aux_seq_feat_loss', 0)
                    current_aux_b = aux_losses.get('aux_feat_only_loss', 0)
                    if hasattr(current_aux_a, 'item'):
                        current_aux_a = current_aux_a.item()
                    if hasattr(current_aux_b, 'item'):
                        current_aux_b = current_aux_b.item()
                    aux_str = f' | Aux A: {current_aux_a:.4f} | Aux B: {current_aux_b:.4f}'
            
            print(f'Epoch {epoch} | Batch {batch_idx:4d}/{len(dataloader)} | '
                  f'Total Loss (optimized): {total_loss_optimized:.6f} | '
                  f'Avg Desc Loss (logged): {desc_avg_loss_logged:.6f}{aux_str} | '
                  f'Grad: {total_norm:.4f} | LR: {lr:.2e}{weight_str}')
            
            # File logging with all weights
            with open(batch_log_file, 'a') as f:
                # Base metrics
                f.write(f'{epoch},{batch_idx},{total_loss_optimized:.6f},'
                       f'{desc_avg_loss_logged:.6f},'
                       f'{total_norm:.6f},{lr:.6e}')
                
                # Add weight statistics and all individual weights
                if 'all_weights' in losses:
                    all_weights = losses['all_weights']
                    f.write(f',{losses["min_weight"]:.6f},{losses["max_weight"]:.6f},{losses["mean_weight"]:.6f}')
                    # Write all 529 weights
                    weights_str = ','.join([f'{w:.6f}' for w in all_weights])
                    f.write(f',{weights_str}')
                else:
                    # No weights available (adaptive weights disabled)
                    f.write(',,,')  # Empty weight stats
                    f.write(',' * 529)  # Empty individual weights
                    
                f.write('\n')
        
        # Collect predictions for metrics evaluation (sample 25% of batches)
        if batch_idx % 4 == 0:  # Sample every 4th batch = 25%
            # Reconstruct descriptor predictions from physics model output
            if 'descriptors' in predictions:
                all_predictions['descriptors'].append(predictions['descriptors'].detach())
            else:
                # Reconstruct from physics model individual predictions
                feature_preds = []
                n_features = desc_targets.shape[1]
                
                # Collect individual feature predictions
                for i in range(n_features):
                    feature_key = f'feature_{i}_mean'
                    if feature_key in predictions:
                        feature_preds.append(predictions[feature_key].detach().unsqueeze(1))
                    else:
                        # Fallback to zeros if feature not found
                        feature_preds.append(torch.zeros(desc_targets.shape[0], 1, device=desc_targets.device))
                
                if feature_preds:
                    reconstructed = torch.cat(feature_preds, dim=1)
                    all_predictions['descriptors'].append(reconstructed)
            
            # Collect auxiliary predictions if available
            if has_activities and activities is not None:
                if 'aux_activity_seq_feat' in predictions:
                    all_predictions['aux_head_a'].append(predictions['aux_activity_seq_feat'].detach())
                if 'aux_activity_feat_only' in predictions:
                    all_predictions['aux_head_b'].append(predictions['aux_activity_feat_only'].detach())
                all_targets['activities'].append(activities.detach())
                    
            all_targets['descriptors'].append(desc_targets.detach())
    
    # Calculate metrics on sampled data (25% of training data)
    if len(all_targets['descriptors']) > 0 and len(all_predictions['descriptors']) > 0:
        target_tensors = {
            'descriptors': torch.cat(all_targets['descriptors'], dim=0),
        }
        pred_tensors = {
            'descriptors': torch.cat(all_predictions['descriptors'], dim=0),
        }
        # Calculate per-feature losses if available
        if all_per_feature_losses:
            pred_tensors['per_feature_losses'] = torch.stack(all_per_feature_losses, dim=0).mean(dim=0)
        
        metrics = metrics_calculator.calculate_metrics(pred_tensors, target_tensors)
        
        # Calculate auxiliary metrics if available
        if has_activities and len(all_targets['activities']) > 0:
            from metrics import compute_auxiliary_metrics
            aux_predictions = {}
            if len(all_predictions['aux_head_a']) > 0:
                aux_predictions['aux_head_a'] = torch.cat(all_predictions['aux_head_a'], dim=0)
            if len(all_predictions['aux_head_b']) > 0:
                aux_predictions['aux_head_b'] = torch.cat(all_predictions['aux_head_b'], dim=0)
            
            if aux_predictions:
                activity_targets = torch.cat(all_targets['activities'], dim=0)
                aux_metrics = compute_auxiliary_metrics(aux_predictions, activity_targets)
                # Add to main metrics (aux_metrics already have 'aux_' prefix)
                metrics.update(aux_metrics)
    else:
        metrics = {'overall_pearson': 0.0}  # Fallback
    
    # Add loss information to metrics
    for key, val in total_losses.items():
        metrics[f'{key}_loss'] = val / len(dataloader)
    
    return metrics

def evaluate_subset(model, dataloader, device, metrics_calculator, feature_stats=None, subset_fraction=1.0, aux_loss_fn=None):
    """Evaluate on a subset or full dataset
    
    Args:
        subset_fraction: Fraction of dataset to evaluate (1.0 = full dataset, 0.25 = 25%)
        aux_loss_fn: Optional auxiliary loss function for activity evaluation
    """
    model.eval()
    total_losses = {'total': 0.0, 'desc': 0.0, 'aux_a': 0.0, 'aux_b': 0.0}
    all_predictions = {'descriptors': [], 'aux_head_a': [], 'aux_head_b': []}
    all_targets = {'descriptors': [], 'activities': []}
    all_per_feature_losses = []
    
    # Check if dataset has activities
    has_activities = hasattr(dataloader.dataset, 'has_activities') and dataloader.dataset.has_activities
    
    # Calculate total batches to evaluate
    total_batches = len(dataloader)
    batches_to_eval = max(1, int(total_batches * subset_fraction))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Stop if we've evaluated enough batches for the subset
            if batch_idx >= batches_to_eval:
                break
                
            sequences = batch['sequence'].to(device)
            desc_targets = batch['descriptors'].to(device)
            
            # Get activities if available
            activities = None
            if has_activities and 'activities' in batch:
                activities = batch['activities'].to(device)
            
            # Forward pass - include auxiliary predictions if activities available
            if has_activities and activities is not None:
                predictions = model(sequences, real_features=desc_targets.detach(), real_activities=activities)
            else:
                predictions = model(sequences)
            
            # Use physics-aware loss for evaluation
            loss_fn = PhysicsAwareLoss()
            targets = {}
            
            # Prepare targets using descriptor columns for physics loss
            feature_cols = dataloader.dataset.descriptor_cols
            for i, col in enumerate(feature_cols):
                feature_key = f'feature_{i}'
                targets[feature_key] = desc_targets[:, i]
            
            losses = loss_fn(predictions, targets)
            
            # Handle physics loss output format
            if isinstance(losses, dict):
                total_loss = losses.get('total_loss', losses.get('total', torch.tensor(0.0)))
                desc_loss = losses.get('desc_loss', total_loss)
            else:
                total_loss = losses
                desc_loss = losses
            
            # Accumulate losses
            total_losses['total'] += total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
            total_losses['desc'] += desc_loss.item() if hasattr(desc_loss, 'item') else float(desc_loss)
            
            # Calculate auxiliary losses if available
            if has_activities and activities is not None and aux_loss_fn is not None:
                if 'aux_activity_seq_feat' in predictions or 'aux_activity_feat_only' in predictions:
                    aux_losses = aux_loss_fn(predictions, activities)
                    if 'aux_seq_feat_loss' in aux_losses:
                        total_losses['aux_a'] += aux_losses['aux_seq_feat_loss'].item()
                    if 'aux_feat_only_loss' in aux_losses:
                        total_losses['aux_b'] += aux_losses['aux_feat_only_loss'].item()
            
            # Collect per-feature losses if available
            if isinstance(losses, dict) and 'per_feature_losses' in losses:
                all_per_feature_losses.append(losses['per_feature_losses'].detach())
            
            # Reconstruct descriptor predictions from physics model output
            # Physics model outputs individual feature predictions with specific keys
            if 'descriptors' in predictions:
                all_predictions['descriptors'].append(predictions['descriptors'])
            else:
                # Reconstruct from physics model individual predictions
                feature_preds = []
                n_features = desc_targets.shape[1]
                
                # Collect individual feature predictions
                for i in range(n_features):
                    feature_key = f'feature_{i}_mean'
                    if feature_key in predictions:
                        feature_preds.append(predictions[feature_key].unsqueeze(1))
                    else:
                        # Fallback to zeros if feature not found
                        feature_preds.append(torch.zeros(desc_targets.shape[0], 1, device=desc_targets.device))
                
                if feature_preds:
                    reconstructed = torch.cat(feature_preds, dim=1)
                    all_predictions['descriptors'].append(reconstructed)
            
            # Collect auxiliary predictions if available
            if has_activities and activities is not None:
                if 'aux_activity_seq_feat' in predictions:
                    all_predictions['aux_head_a'].append(predictions['aux_activity_seq_feat'].detach())
                if 'aux_activity_feat_only' in predictions:
                    all_predictions['aux_head_b'].append(predictions['aux_activity_feat_only'].detach())
                all_targets['activities'].append(activities.detach())
            
            all_targets['descriptors'].append(desc_targets)
    
    # Calculate metrics on evaluated data
    if len(all_predictions['descriptors']) > 0 and len(all_targets['descriptors']) > 0:
        pred_tensors = {
            'descriptors': torch.cat(all_predictions['descriptors'], dim=0),
        }
        if all_per_feature_losses:
            pred_tensors['per_feature_losses'] = torch.stack(all_per_feature_losses, dim=0).mean(dim=0)
        
        target_tensors = {
            'descriptors': torch.cat(all_targets['descriptors'], dim=0),
        }
        
        metrics = metrics_calculator.calculate_metrics(pred_tensors, target_tensors)
        
        # Calculate auxiliary metrics if available
        if has_activities and len(all_targets['activities']) > 0:
            from metrics import compute_auxiliary_metrics
            aux_predictions = {}
            if len(all_predictions['aux_head_a']) > 0:
                aux_predictions['aux_head_a'] = torch.cat(all_predictions['aux_head_a'], dim=0)
            if len(all_predictions['aux_head_b']) > 0:
                aux_predictions['aux_head_b'] = torch.cat(all_predictions['aux_head_b'], dim=0)
            
            if aux_predictions:
                activity_targets = torch.cat(all_targets['activities'], dim=0)
                aux_metrics = compute_auxiliary_metrics(aux_predictions, activity_targets)
                # Add to main metrics (aux_metrics already have 'aux_' prefix)
                metrics.update(aux_metrics)
    else:
        metrics = {'overall_pearson': 0.0}
    
    # Add loss information to metrics
    for key, val in total_losses.items():
        metrics[f'{key}_loss'] = val / batches_to_eval
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Train PhysInformer')
    parser.add_argument('--cell_type', type=str, required=True,
                      choices=['HepG2', 'K562', 'WTC11', 'S2', 'arabidopsis', 'sorghum', 'maize'],
                      help='Cell type to train on')
    parser.add_argument('--data_dir', type=str, default='../output',
                      help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='./runs',
                      help='Output directory for logs and checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='Learning rate')
    parser.add_argument('--epochs', type=int, default=25,
                      help='Number of epochs')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from latest checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup directories
    if args.resume:
        # Find the latest run directory for this cell type
        output_path = Path(args.output_dir)
        existing_runs = sorted([d for d in output_path.glob(f"{args.cell_type}_*") if d.is_dir()])
        if not existing_runs:
            print(f"ERROR: No existing runs found for {args.cell_type} to resume from")
            return
        run_dir = existing_runs[-1]
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = Path(args.output_dir) / f"{args.cell_type}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(run_dir)
    
    if args.resume:
        logger.info(f"Resuming training from: {run_dir}")
    else:
        logger.info(f"Starting new training run: {run_dir}")
    logger.info(f"Starting PhysInformer training for {args.cell_type}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders - use plant-specific loader for plant cell types
    logger.info("Loading datasets...")
    if args.cell_type in PLANT_CELL_TYPES:
        logger.info(f"Using plant dataloader for {args.cell_type}")
        dataloaders = create_plant_dataloaders(
            args.cell_type,
            args.data_dir,
            args.batch_size,
            args.num_workers
        )
    else:
        dataloaders = create_dataloaders(
            args.cell_type,
            args.data_dir,
            args.batch_size,
            args.num_workers
        )
    
    # Get feature names and normalization stats
    sample_dataset = dataloaders['train'].dataset
    feature_names = sample_dataset.get_feature_names()
    dataset_norm_stats = sample_dataset.get_normalization_stats()  # For saving in checkpoint
    
    # Create model - get actual number of features from dataset
    actual_n_features = len(dataloaders['train'].dataset.descriptor_cols)
    descriptor_names = dataloaders['train'].dataset.descriptor_cols
    logger.info(f"Creating physics-aware model with {actual_n_features} descriptor features...")
    model = create_physics_aware_model(args.cell_type, n_descriptor_features=actual_n_features, descriptor_names=descriptor_names)
    model = model.to(device)
    
    # Check if dataset has activities and enable auxiliary heads
    has_activities = hasattr(sample_dataset, 'has_activities') and sample_dataset.has_activities
    aux_optimizer = None
    aux_loss_fn = None
    
    if has_activities:
        # Get number of activities
        n_activities = len(sample_dataset.activity_cols)
        logger.info(f"Dataset has activities: {sample_dataset.activity_cols} ({n_activities} dimensions)")
        
        # Enable auxiliary heads
        model.enable_auxiliary_heads(n_real_features=actual_n_features, n_activities=n_activities)
        logger.info("Enabled auxiliary heads for activity prediction")
        
        # Import auxiliary loss
        from physics_aware_model import AuxiliaryLoss
        aux_loss_fn = AuxiliaryLoss()
        
        # Create separate optimizer for auxiliary heads
        aux_optimizer = optim.AdamW(model.get_auxiliary_parameters(), lr=args.learning_rate * 0.1, weight_decay=0.01)
        logger.info("Created separate optimizer for auxiliary heads")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup optimizer - only for main model parameters if we have auxiliary heads
    if has_activities:
        optimizer = optim.AdamW(model.get_main_parameters(), lr=args.learning_rate, weight_decay=0.01)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Calculate total training steps for OneCycleLR
    steps_per_epoch = len(dataloaders['train'])
    total_steps = steps_per_epoch * args.epochs
    
    # Initialize training variables
    start_epoch = 1
    best_val_loss = float('inf')
    epoch_histories = {'train': [], 'val': []}
    
    # Load checkpoint if resuming
    if args.resume:
        checkpoint_files = sorted(run_dir.glob('checkpoint_epoch_*.pt'), 
                                key=lambda x: int(x.stem.split('_')[-1]))
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            logger.info(f"Loading checkpoint: {latest_checkpoint}")
            
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if aux_optimizer is not None and 'aux_optimizer_state_dict' in checkpoint:
                aux_optimizer.load_state_dict(checkpoint['aux_optimizer_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            logger.info(f"Resumed from epoch {checkpoint['epoch']}, best val loss: {best_val_loss:.6f}")
            
            # Adjust scheduler to resume from correct step
            completed_steps = steps_per_epoch * (start_epoch - 1)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.learning_rate * 10,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos',
                cycle_momentum=False,
                div_factor=25,
                final_div_factor=10000,
                last_epoch=completed_steps - 1  # Resume from correct step
            )
        else:
            logger.warning("No checkpoint found, starting from scratch")
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.learning_rate * 10,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos',
                cycle_momentum=False,
                div_factor=25,
                final_div_factor=10000
            )
    else:
        # Normal training from scratch
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate * 10,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=25,
            final_div_factor=10000
        )
    
    # Setup metrics calculator
    metrics_calculator = MetricsCalculator(feature_names)
    
    # Get feature statistics for loss normalization from dataset
    # Don't compute from normalized data - get original stats instead
    logger.info("Getting feature statistics for loss normalization...")
    if dataset_norm_stats is not None:
        # Use the original data statistics for loss normalization
        feature_stats = {
            'desc_std': dataset_norm_stats['desc_std'].to(device),
            'desc_mean': dataset_norm_stats['desc_mean'].to(device)
        }
        logger.info(f"Using original data statistics for loss normalization")
        logger.info(f"Descriptor std range: [{feature_stats['desc_std'].min().item():.4f}, {feature_stats['desc_std'].max().item():.4f}]")
    else:
        # Fallback: compute from data (but this won't work well if data is normalized)
        feature_stats = compute_feature_statistics(dataloaders['train'], device)
        logger.warning("Computing statistics from normalized data - loss scaling may be incorrect!")
    
    # Setup logging files
    metrics_log_file = run_dir / 'metrics_log.txt'
    epoch_results_file = run_dir / 'epoch_results.txt'
    batch_log_file = run_dir / 'batch_log.txt'
    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Initialize batch log file with header
    # Get feature names for weight column headers
    feature_names = dataloaders['train'].dataset.get_feature_names()
    weight_columns = ','.join([f'weight_{name}' for name in feature_names['descriptors']])
    
    # Handle batch log file for resume
    if args.resume and batch_log_file.exists():
        # Batch log already exists and has been cleaned, open in append mode
        logger.info(f"Appending to existing batch log: {batch_log_file}")
    else:
        # Create new batch log with header
        with open(batch_log_file, 'w') as f:
            f.write(f'epoch,batch,total_loss_optimized,desc_avg_loss_logged,grad_norm,learning_rate,min_weight,max_weight,mean_weight,{weight_columns}\n')
    
    # Training loop
    train_metrics_history = []
    val_metrics_history = []
    
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, dataloaders['train'], optimizer, scheduler, device, 
            metrics_calculator, epoch, batch_log_file, feature_stats=feature_stats, cell_type=args.cell_type,
            aux_optimizer=aux_optimizer, aux_loss_fn=aux_loss_fn
        )
        
        # Per-epoch evaluation: 25% of train set + 100% of validation set
        logger.info("Evaluating on 25% of training set...")
        train_eval_metrics = evaluate_subset(
            model, dataloaders['train'], device, metrics_calculator, 
            feature_stats=feature_stats, subset_fraction=0.25, aux_loss_fn=aux_loss_fn
        )
        
        logger.info("Evaluating on full validation set...")
        val_metrics = evaluate_subset(
            model, dataloaders['val'], device, metrics_calculator, 
            feature_stats=feature_stats, subset_fraction=1.0, aux_loss_fn=aux_loss_fn
        )
        
        # Log metrics (use train_eval_metrics for proper evaluation on 25% of train)
        metrics_calculator.log_metrics(train_eval_metrics, epoch, 'train', logger)
        metrics_calculator.log_metrics(val_metrics, epoch, 'val', logger)
        
        # Log epoch results to file with detailed metrics
        with open(epoch_results_file, 'a') as f:
            f.write(f"{'='*80}\n")
            f.write(f"EPOCH {epoch}:\n")
            f.write(f"{'='*80}\n")
            
            # Basic losses
            f.write(f"\nLOSSES:\n")
            f.write(f"  Train Total Loss (25% eval): {train_eval_metrics['total_loss']:.6f}\n")
            f.write(f"  Train Desc Loss (25% eval): {train_eval_metrics['desc_loss']:.6f}\n")
            f.write(f"  Val Total Loss: {val_metrics['total_loss']:.6f}\n")
            f.write(f"  Val Desc Loss: {val_metrics['desc_loss']:.6f}\n")
            
            # Auxiliary losses if available
            if 'aux_a_loss' in train_eval_metrics:
                f.write(f"  Train Aux A Loss: {train_eval_metrics['aux_a_loss']:.6f}\n")
                f.write(f"  Train Aux B Loss: {train_eval_metrics['aux_b_loss']:.6f}\n")
            if 'aux_a_loss' in val_metrics:
                f.write(f"  Val Aux A Loss: {val_metrics['aux_a_loss']:.6f}\n")
                f.write(f"  Val Aux B Loss: {val_metrics['aux_b_loss']:.6f}\n")
            
            # Overall metrics
            f.write(f"\nOVERALL METRICS:\n")
            f.write(f"  Train Overall Pearson: {train_eval_metrics['overall_pearson']:.4f}\n")
            f.write(f"  Val Overall Pearson: {val_metrics['overall_pearson']:.4f}\n")
            
            # Descriptor Pearson distribution
            f.write(f"\nDESCRIPTOR PEARSON DISTRIBUTION:\n")
            f.write(f"  Train: mean={train_eval_metrics.get('descriptors_pearson_mean', 0):.4f}, "
                   f"median={train_eval_metrics.get('descriptors_pearson_median', 0):.4f}, "
                   f"range=[{train_eval_metrics.get('descriptors_pearson_min', 0):.4f}, "
                   f"{train_eval_metrics.get('descriptors_pearson_max', 0):.4f}]\n")
            f.write(f"  Val: mean={val_metrics.get('descriptors_pearson_mean', 0):.4f}, "
                   f"median={val_metrics.get('descriptors_pearson_median', 0):.4f}, "
                   f"range=[{val_metrics.get('descriptors_pearson_min', 0):.4f}, "
                   f"{val_metrics.get('descriptors_pearson_max', 0):.4f}]\n")
            
            # Auxiliary metrics if available
            if 'aux_seq_feat_pearson' in val_metrics:
                f.write(f"\nAUXILIARY HEAD METRICS (Val):\n")
                f.write(f"  Head A (Seq+Feat): Pearson={val_metrics['aux_seq_feat_pearson']:.4f}, "
                       f"R²={val_metrics['aux_seq_feat_r2']:.4f}, "
                       f"MSE={val_metrics['aux_seq_feat_mse']:.6f}\n")
                f.write(f"  Head B (Feat only): Pearson={val_metrics['aux_feat_only_pearson']:.4f}, "
                       f"R²={val_metrics['aux_feat_only_r2']:.4f}, "
                       f"MSE={val_metrics['aux_feat_only_mse']:.6f}\n")
            
            # Top and bottom features for validation
            if 'descriptors_feature_scores' in val_metrics:
                feature_scores = val_metrics['descriptors_feature_scores']
                if feature_scores:
                    sorted_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
                    f.write(f"\nTOP 5 BEST PREDICTED FEATURES (Val):\n")
                    for i, (name, score) in enumerate(sorted_scores[:5], 1):
                        f.write(f"  {i}. {name}: {score:.4f}\n")
                    f.write(f"\nBOTTOM 5 WORST PREDICTED FEATURES (Val):\n")
                    for i, (name, score) in enumerate(sorted_scores[-5:], 1):
                        f.write(f"  {i}. {name}: {score:.4f}\n")
            
            f.write(f"\nLearning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")
            f.write(f"\n")
        
        # Store metrics for plotting (use train_eval_metrics for consistent evaluation)
        train_metrics_history.append(train_eval_metrics)
        val_metrics_history.append(val_metrics)
        
        # Generate plots for this epoch
        epoch_histories = {'train': train_metrics_history, 'val': val_metrics_history}
        create_training_plots(epoch_histories, epoch, plots_dir, args.cell_type)
        
        # Generate Pearson distribution plots
        from plotting import create_pearson_distribution_plot
        create_pearson_distribution_plot(train_eval_metrics, epoch, plots_dir, args.cell_type, 'train')
        create_pearson_distribution_plot(val_metrics, epoch, plots_dir, args.cell_type, 'val')
        
        # Save checkpoint
        val_loss = val_metrics['total_loss']
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {val_loss:.6f}")
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss, val_metrics, 
            run_dir, cell_type=args.cell_type, 
            normalization_stats=dataset_norm_stats, seed=args.seed, is_best=is_best
        )
    
    # Final evaluation on 100% of all datasets
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION ON ALL DATASETS (100%)")
    logger.info("="*60)
    
    # Load best model
    best_checkpoint = torch.load(run_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Evaluate on 100% of each dataset
    logger.info("\nEvaluating on full training set...")
    final_train_metrics = evaluate_subset(
        model, dataloaders['train'], device, metrics_calculator,
        feature_stats=feature_stats, subset_fraction=1.0, aux_loss_fn=aux_loss_fn
    )
    metrics_calculator.log_metrics(final_train_metrics, args.epochs, 'train_final', logger)
    
    logger.info("\nEvaluating on full validation set...")
    final_val_metrics = evaluate_subset(
        model, dataloaders['val'], device, metrics_calculator,
        feature_stats=feature_stats, subset_fraction=1.0, aux_loss_fn=aux_loss_fn
    )
    metrics_calculator.log_metrics(final_val_metrics, args.epochs, 'val_final', logger)
    
    logger.info("\nEvaluating on full test set...")
    test_metrics = evaluate_subset(
        model, dataloaders['test'], device, metrics_calculator,
        feature_stats=feature_stats, subset_fraction=1.0, aux_loss_fn=aux_loss_fn
    )
    metrics_calculator.log_metrics(test_metrics, args.epochs, 'test', logger)
    
    # Create final summary plot
    epoch_histories = {'train': train_metrics_history, 'val': val_metrics_history}
    create_final_summary_plot(epoch_histories, plots_dir, args.cell_type)
    
    # Save final results
    results = {
        'args': vars(args),
        'model_info': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        },
        'training_summary': {
            'best_val_loss': float(best_val_loss),
            'final_train_metrics': convert_to_serializable(final_train_metrics),
            'final_val_metrics': convert_to_serializable(final_val_metrics),
            'test_metrics': convert_to_serializable(test_metrics),
            'total_epochs': args.epochs
        },
        'test_metrics': convert_to_serializable(test_metrics)
    }
    
    # Save config and results
    with open(run_dir / 'config.json', 'w') as f:
        config = convert_to_serializable(vars(args))
        json.dump(config, f, indent=2)
    
    with open(run_dir / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed final report
    with open(run_dir / 'final_report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PHYSINFORMER TRAINING FINAL REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Cell Type: {args.cell_type}\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Epochs: {args.epochs}\n")
        f.write(f"Model Parameters: {total_params:,}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
        f.write(f"Final Test Total Loss: {test_metrics['total_loss']:.6f}\n")
        f.write(f"Final Test Desc Loss: {test_metrics['desc_loss']:.6f}\n")
        
        f.write("FINAL TEST METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall MSE: {test_metrics['overall_mse']:.6f}\n")
        f.write(f"Overall MAE: {test_metrics['overall_mae']:.6f}\n")
        f.write(f"Overall RMSE: {test_metrics['overall_rmse']:.6f}\n")
        f.write(f"Overall Pearson: {test_metrics['overall_pearson']:.4f}\n")
        f.write(f"Overall Spearman: {test_metrics['overall_spearman']:.4f}\n\n")
        
        f.write("BIOPHYSICAL DESCRIPTORS:\n")
        f.write(f"  Pearson: {test_metrics['descriptors_pearson_mean']:.4f} ± {test_metrics['descriptors_pearson_std']:.4f}\n")
        f.write(f"  Spearman: {test_metrics['descriptors_spearman_mean']:.4f} ± {test_metrics['descriptors_spearman_std']:.4f}\n")
        f.write(f"  MSE: {test_metrics['descriptors_mse_mean']:.6f} ± {test_metrics['descriptors_mse_std']:.6f}\n")
        f.write(f"  MAE: {test_metrics['descriptors_mae_mean']:.6f} ± {test_metrics['descriptors_mae_std']:.6f}\n\n")
        
        
        f.write("=" * 80 + "\n")
    
    logger.info(f"Training completed. Results saved to {run_dir}")
    logger.info(f"Files created:")
    logger.info(f"  - config.json: Training configuration")
    logger.info(f"  - final_results.json: Final metrics and results")
    logger.info(f"  - final_report.txt: Human-readable final report")
    logger.info(f"  - best_model.pt: Best model checkpoint")
    logger.info(f"  - training.log: Complete training log")
    logger.info(f"  - batch_log.txt: Per-batch training metrics")
    logger.info(f"  - epoch_results.txt: Per-epoch summary")
    logger.info(f"  - plots/: Training progression plots (PNG)")
    logger.info(f"  - plots/final_training_summary.png: Final summary plot")

if __name__ == '__main__':
    main()