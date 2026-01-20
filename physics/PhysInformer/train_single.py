import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from tqdm import tqdm
import argparse

from tileformer_single import create_tileformer_single
from metrics_single import SingleFeatureMetricsCalculator, calculate_single_feature_loss
from plotting_single import create_single_feature_plots
from dataset import create_dataloaders

def setup_logging(run_dir: Path):
    """Setup logging configuration"""
    log_file = run_dir / 'training.log'
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup logger
    logger = logging.getLogger('tileformer_single')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def train_epoch(model, dataloader, optimizer, scheduler, device, target_feature, logger):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    main_loss_sum = 0.0
    n_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        sequences = batch['sequence'].to(device)
        descriptors = batch['descriptors'].to(device)
        # sequence_scores = batch['sequence_score'].to(device)  # Not available in data
        
        # Extract target feature - entropy_mi_d5 is at index 479 in descriptor columns
        target_values = descriptors[:, 479]  # entropy_mi_d5
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(sequences)
        
        # Prepare targets (no sequence_score available)
        targets = {
            target_feature: target_values
            # 'sequence_score': sequence_scores  # Removed - not in data
        }
        
        # Calculate loss
        losses = calculate_single_feature_loss(predictions, targets, target_feature)
        
        # Backward pass
        losses['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += losses['total_loss'].item()
        main_loss_sum += losses['main_loss'].item()
        n_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{losses['total_loss'].item():.4f}",
            'Main': f"{losses['main_loss'].item():.4f}"
        })
        
        # Log detailed batch info periodically
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}: Total={losses['total_loss'].item():.6f}, "
                       f"Main={losses['main_loss'].item():.6f}")
    
    # Step scheduler
    if scheduler is not None:
        scheduler.step()
    
    # Return epoch metrics
    return {
        'total_loss': total_loss / n_batches,
        'main_loss': main_loss_sum / n_batches,
    }

def evaluate_epoch(model, dataloader, device, target_feature, metrics_calculator, logger):
    """Evaluate for one epoch"""
    model.eval()
    
    all_predictions = {
        'target_feature': []
    }
    all_targets = {
        target_feature: []
        # 'sequence_score': []  # Removed - not needed
    }
    
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for batch in progress_bar:
            # Move data to device
            sequences = batch['sequence'].to(device)
            descriptors = batch['descriptors'].to(device)
            # sequence_scores = batch['sequence_score'].to(device)  # Not available in data
            
            # Extract target feature - entropy_mi_d5 is at index 479
            target_values = descriptors[:, 479]  # entropy_mi_d5
            
            # Forward pass
            predictions = model(sequences)
            
            # Prepare targets (no sequence_score available)
            targets = {
                target_feature: target_values
                # 'sequence_score': sequence_scores  # Removed - not in data
            }
            
            # Calculate loss
            losses = calculate_single_feature_loss(predictions, targets, target_feature)
            total_loss += losses['total_loss'].item()
            n_batches += 1
            
            # Collect predictions and targets
            all_predictions['target_feature'].append(predictions['target_feature'])
            all_targets[target_feature].append(target_values)
            # REMOVED sequence_score - we're predicting ONE feature from sequence!
    
    # Concatenate all predictions and targets
    for key in all_predictions:
        all_predictions[key] = torch.cat(all_predictions[key], dim=0)
    for key in all_targets:
        all_targets[key] = torch.cat(all_targets[key], dim=0)
    
    # Calculate metrics
    metrics = metrics_calculator.calculate_metrics(all_predictions, all_targets)
    metrics['total_loss'] = total_loss / n_batches
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_type', type=str, default='HepG2', choices=['HepG2', 'K562', 'WTC11'])
    parser.add_argument('--target_feature', type=str, default='entropy_mi_d5')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dir', type=str, default='../output')
    
    args = parser.parse_args()
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"TileFormerSingle_{args.cell_type}_{args.target_feature}_{timestamp}"
    run_dir = Path(f"runs/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(run_dir)
    logger.info(f"Starting TileFormer Single Feature Training: {args.target_feature}")
    logger.info(f"Args: {vars(args)}")
    
    # Save configuration
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    logger.info("Loading data...")
    try:
        dataloaders = create_dataloaders(
            cell_type=args.cell_type,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=4
        )
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        logger.info(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Create model
    logger.info("Creating model...")
    model = create_tileformer_single(
        target_feature=args.target_feature,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1
    )
    
    model = model.to(args.device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Setup metrics
    metrics_calculator = SingleFeatureMetricsCalculator(args.target_feature)
    
    # Training history
    epoch_histories = {
        'train': [],
        'val': []
    }
    
    best_val_pearson = -1.0
    best_epoch = 0
    
    logger.info("Starting training...")
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, args.device, 
            args.target_feature, logger
        )
        
        # Validation
        val_metrics = evaluate_epoch(
            model, val_loader, args.device, args.target_feature,
            metrics_calculator, logger
        )
        
        # Log metrics
        logger.info(f"\nTraining Loss: {train_metrics['total_loss']:.6f}")
        metrics_calculator.log_metrics(val_metrics, epoch, 'val', logger)
        
        # Save history
        epoch_histories['train'].append(train_metrics)
        epoch_histories['val'].append(val_metrics)
        
        # Check for best model
        current_val_pearson = val_metrics.get('main_pearson', 0.0)
        if current_val_pearson > best_val_pearson:
            best_val_pearson = current_val_pearson
            best_epoch = epoch
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_pearson': current_val_pearson,
                'config': vars(args)
            }, run_dir / 'best_model.pt')
            
            logger.info(f"New best model saved! Pearson: {current_val_pearson:.4f}")
        
        # Save checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'epoch_histories': epoch_histories,
                'config': vars(args)
            }, run_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Create plots
        try:
            create_single_feature_plots(epoch_histories, epoch, run_dir, args.target_feature)
        except Exception as e:
            logger.warning(f"Failed to create plots: {e}")
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best validation Pearson: {best_val_pearson:.4f} at epoch {best_epoch}")
    
    # Save final results
    final_results = {
        'best_epoch': best_epoch,
        'best_val_pearson': best_val_pearson,
        'final_train_metrics': epoch_histories['train'][-1],
        'final_val_metrics': epoch_histories['val'][-1],
        'config': vars(args)
    }
    
    with open(run_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info("Training finished successfully!")

if __name__ == "__main__":
    main()