"""
Training script for PhysicsVAE.

Comprehensive logging of:
- Per-batch: loss components, gradients, learning rate
- Per-epoch: train/val metrics, checkpoints
- Final: test evaluation, summary plots, config

Usage:
    python train.py --cell_type K562 --epochs 100 --batch_size 64
"""

import torch
import torch.optim as optim
import numpy as np
import os
import logging
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.physics_vae import PhysicsVAE, create_physics_vae
from models.losses import CombinedVAELoss
from data.dataset import create_dataloaders, get_physics_feature_count
from data.multi_dataset import create_multi_dataloaders, load_config


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('PhysicsVAE')
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_dir / 'training.log')
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def save_config(args, run_dir: Path, model_info: Dict):
    """Save comprehensive configuration."""
    config = {
        'training_args': vars(args),
        'model_info': model_info,
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    return config


def save_checkpoint(
    model, optimizer, scheduler, epoch, metrics,
    checkpoint_path: Path, config: Dict = None, is_best: bool = False
):
    """Save model checkpoint with full state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config
    }

    torch.save(checkpoint, checkpoint_path / f'checkpoint_epoch_{epoch}.pt')

    if is_best:
        torch.save(checkpoint, checkpoint_path / 'best_model.pt')


def compute_gradient_stats(model) -> Dict[str, float]:
    """Compute gradient statistics for monitoring."""
    total_norm = 0.0
    max_norm = 0.0
    min_norm = float('inf')
    n_params = 0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            max_norm = max(max_norm, param_norm)
            min_norm = min(min_norm, param_norm)
            n_params += 1

    total_norm = total_norm ** 0.5

    return {
        'grad_norm_total': total_norm,
        'grad_norm_max': max_norm,
        'grad_norm_min': min_norm if n_params > 0 else 0.0,
        'n_params_with_grad': n_params
    }


def compute_metrics(
    model,
    dataloader,
    device,
    loss_fn,
    detailed: bool = False
) -> Dict[str, float]:
    """Compute validation/test metrics."""
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    # For detailed analysis
    all_mu = []
    all_logvar = []

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequence'].to(device)
            physics = batch['physics'].to(device)

            # Forward pass
            outputs = model(sequences, physics)

            # Compute loss
            losses = loss_fn(
                outputs['logits'],
                sequences,
                outputs['mu'],
                outputs['logvar'],
                physics,
                compute_physics=False
            )

            total_loss += losses['total_loss'].item()
            total_recon += losses['recon_loss'].item()
            total_kl += losses['kl_loss'].item()

            # Compute reconstruction accuracy
            preds = outputs['logits'].argmax(dim=-1)
            correct = (preds == sequences).sum().item()
            total_correct += correct
            total_tokens += sequences.numel()

            n_batches += 1

            if detailed:
                all_mu.append(outputs['mu'].cpu())
                all_logvar.append(outputs['logvar'].cpu())

    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'kl_loss': total_kl / n_batches,
        'accuracy': total_correct / total_tokens,
        'perplexity': np.exp(total_recon / n_batches / dataloader.dataset.seq_length if hasattr(dataloader.dataset, 'seq_length') else total_recon / n_batches / 200)
    }

    if detailed and len(all_mu) > 0:
        all_mu = torch.cat(all_mu, dim=0)
        all_logvar = torch.cat(all_logvar, dim=0)

        # Latent space statistics
        metrics['latent_mu_mean'] = all_mu.mean().item()
        metrics['latent_mu_std'] = all_mu.std().item()
        metrics['latent_var_mean'] = all_logvar.exp().mean().item()
        metrics['latent_active_dims'] = (all_logvar.exp().mean(dim=0) > 0.1).sum().item()

    return metrics


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    loss_fn,
    epoch: int,
    logger: logging.Logger,
    batch_log_file: Path,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch with comprehensive logging."""
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = len(dataloader)

    global_step = (epoch - 1) * n_batches

    # Gradient statistics accumulator
    grad_norms = []

    for batch_idx, batch in enumerate(dataloader):
        sequences = batch['sequence'].to(device)
        physics = batch['physics'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(sequences, physics)

        # Compute loss
        step = global_step + batch_idx
        losses = loss_fn(
            outputs['logits'],
            sequences,
            outputs['mu'],
            outputs['logvar'],
            physics,
            step=step,
            compute_physics=False
        )

        loss = losses['total_loss']

        # Backward pass
        loss.backward()

        # Compute gradient stats before clipping
        grad_stats = compute_gradient_stats(model)
        grad_norms.append(grad_stats['grad_norm_total'])

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Track metrics
        total_loss += loss.item()
        total_recon += losses['recon_loss'].item()
        total_kl += losses['kl_loss'].item()

        # Accuracy
        preds = outputs['logits'].argmax(dim=-1)
        correct = (preds == sequences).sum().item()
        total_correct += correct
        total_tokens += sequences.numel()

        # Get current learning rate
        lr = optimizer.param_groups[0]['lr']

        # Log to batch file
        with open(batch_log_file, 'a') as f:
            f.write(f"{epoch},{batch_idx},{loss.item():.6f},"
                   f"{losses['recon_loss'].item():.6f},"
                   f"{losses['kl_loss'].item():.6f},"
                   f"{losses['beta'].item():.6f},"
                   f"{grad_stats['grad_norm_total']:.6f},"
                   f"{grad_stats['grad_norm_max']:.6f},"
                   f"{lr:.8f},"
                   f"{correct/sequences.numel():.6f}\n")

        # Console logging every 50 batches
        if batch_idx % 50 == 0:
            acc = correct / sequences.numel()
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx:4d}/{n_batches} | "
                f"Loss: {loss.item():.4f} | Recon: {losses['recon_loss'].item():.4f} | "
                f"KL: {losses['kl_loss'].item():.4f} | Beta: {losses['beta'].item():.4f} | "
                f"Grad: {grad_stats['grad_norm_total']:.4f} | LR: {lr:.2e} | Acc: {acc:.4f}"
            )

    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'kl_loss': total_kl / n_batches,
        'accuracy': total_correct / total_tokens,
        'grad_norm_mean': np.mean(grad_norms),
        'grad_norm_max': np.max(grad_norms),
        'grad_norm_std': np.std(grad_norms)
    }


def log_epoch_results(
    epoch: int,
    train_metrics: Dict,
    val_metrics: Dict,
    lr: float,
    results_file: Path
):
    """Log detailed epoch results to file."""
    with open(results_file, 'a') as f:
        f.write(f"{'='*80}\n")
        f.write(f"EPOCH {epoch}\n")
        f.write(f"{'='*80}\n\n")

        f.write("TRAINING METRICS:\n")
        f.write(f"  Total Loss: {train_metrics['loss']:.6f}\n")
        f.write(f"  Reconstruction Loss: {train_metrics['recon_loss']:.6f}\n")
        f.write(f"  KL Divergence: {train_metrics['kl_loss']:.6f}\n")
        f.write(f"  Accuracy: {train_metrics['accuracy']:.4f}\n")
        f.write(f"  Gradient Norm (mean): {train_metrics['grad_norm_mean']:.4f}\n")
        f.write(f"  Gradient Norm (max): {train_metrics['grad_norm_max']:.4f}\n")
        f.write(f"  Gradient Norm (std): {train_metrics['grad_norm_std']:.4f}\n\n")

        f.write("VALIDATION METRICS:\n")
        f.write(f"  Total Loss: {val_metrics['loss']:.6f}\n")
        f.write(f"  Reconstruction Loss: {val_metrics['recon_loss']:.6f}\n")
        f.write(f"  KL Divergence: {val_metrics['kl_loss']:.6f}\n")
        f.write(f"  Accuracy: {val_metrics['accuracy']:.4f}\n")
        f.write(f"  Perplexity: {val_metrics['perplexity']:.4f}\n")

        if 'latent_mu_mean' in val_metrics:
            f.write(f"\nLATENT SPACE:\n")
            f.write(f"  Mean of mu: {val_metrics['latent_mu_mean']:.4f}\n")
            f.write(f"  Std of mu: {val_metrics['latent_mu_std']:.4f}\n")
            f.write(f"  Mean variance: {val_metrics['latent_var_mean']:.4f}\n")
            f.write(f"  Active dimensions: {val_metrics['latent_active_dims']}\n")

        f.write(f"\nLearning Rate: {lr:.8f}\n")
        f.write(f"\n")


def create_training_plots(
    train_history: List[Dict],
    val_history: List[Dict],
    plots_dir: Path,
    cell_type: str
):
    """Create comprehensive training plots."""
    epochs = range(1, len(train_history) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'PhysicsVAE Training - {cell_type}', fontsize=14)

    # Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, [m['loss'] for m in train_history], 'b-', label='Train')
    ax.plot(epochs, [m['loss'] for m in val_history], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reconstruction Loss
    ax = axes[0, 1]
    ax.plot(epochs, [m['recon_loss'] for m in train_history], 'b-', label='Train')
    ax.plot(epochs, [m['recon_loss'] for m in val_history], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recon Loss')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # KL Divergence
    ax = axes[0, 2]
    ax.plot(epochs, [m['kl_loss'] for m in train_history], 'b-', label='Train')
    ax.plot(epochs, [m['kl_loss'] for m in val_history], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Loss')
    ax.set_title('KL Divergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, [m['accuracy'] for m in train_history], 'b-', label='Train')
    ax.plot(epochs, [m['accuracy'] for m in val_history], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reconstruction Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient Norm
    ax = axes[1, 1]
    ax.plot(epochs, [m['grad_norm_mean'] for m in train_history], 'g-', label='Mean')
    ax.fill_between(epochs,
                    [m['grad_norm_mean'] - m['grad_norm_std'] for m in train_history],
                    [m['grad_norm_mean'] + m['grad_norm_std'] for m in train_history],
                    alpha=0.3, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm (mean ± std)')
    ax.grid(True, alpha=0.3)

    # Perplexity
    ax = axes[1, 2]
    ax.plot(epochs, [m['perplexity'] for m in val_history], 'r-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.set_title('Validation Perplexity')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'training_curves.png', dpi=150)
    plt.close()


def create_final_report(
    run_dir: Path,
    config: Dict,
    train_history: List[Dict],
    val_history: List[Dict],
    test_metrics: Dict,
    best_epoch: int
):
    """Create comprehensive final report."""
    with open(run_dir / 'final_report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PHYSICSVAE TRAINING FINAL REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Config
        f.write("CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        args = config['training_args']
        f.write(f"  Cell Type: {args['cell_type']}\n")
        f.write(f"  Epochs: {args['epochs']}\n")
        f.write(f"  Batch Size: {args['batch_size']}\n")
        f.write(f"  Learning Rate: {args['learning_rate']}\n")
        f.write(f"  Latent Dim: {args['latent_dim']}\n")
        f.write(f"  Beta (KL weight): {args['beta']}\n")
        f.write(f"  Beta Annealing: {args['beta_annealing']}\n")
        f.write(f"  Seed: {args['seed']}\n\n")

        # Model info
        f.write("MODEL INFO:\n")
        f.write("-" * 40 + "\n")
        model_info = config['model_info']
        f.write(f"  Total Parameters: {model_info['n_params']:,}\n")
        f.write(f"  Trainable Parameters: {model_info['n_trainable']:,}\n")
        f.write(f"  Sequence Length: {model_info['seq_length']}\n")
        f.write(f"  Physics Features: {model_info['n_physics']}\n\n")

        # Training summary
        f.write("TRAINING SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Best Epoch: {best_epoch}\n")
        f.write(f"  Best Val Loss: {min(m['loss'] for m in val_history):.6f}\n")
        f.write(f"  Final Train Loss: {train_history[-1]['loss']:.6f}\n")
        f.write(f"  Final Val Loss: {val_history[-1]['loss']:.6f}\n")
        f.write(f"  Final Train Accuracy: {train_history[-1]['accuracy']:.4f}\n")
        f.write(f"  Final Val Accuracy: {val_history[-1]['accuracy']:.4f}\n\n")

        # Test results
        if test_metrics:
            f.write("TEST RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Loss: {test_metrics['loss']:.6f}\n")
            f.write(f"  Reconstruction Loss: {test_metrics['recon_loss']:.6f}\n")
            f.write(f"  KL Divergence: {test_metrics['kl_loss']:.6f}\n")
            f.write(f"  Accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"  Perplexity: {test_metrics['perplexity']:.4f}\n")
            if 'latent_active_dims' in test_metrics:
                f.write(f"  Active Latent Dims: {test_metrics['latent_active_dims']}\n")

        f.write("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train PhysicsVAE')
    parser.add_argument('--cell_type', type=str, default=None,
                        choices=['K562', 'HepG2', 'WTC11', 'S2', 'arabidopsis', 'sorghum', 'maize'],
                        help='Cell type to train on (single dataset mode)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config for multi-dataset training')
    parser.add_argument('--data_dir', type=str, default='../output',
                        help='Directory containing descriptor files')
    parser.add_argument('--output_dir', type=str, default='./runs',
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--beta', type=float, default=0.001,
                        help='KL weight (beta-VAE), spec default=0.001 for weak regularization')
    parser.add_argument('--beta_annealing', action='store_true',
                        help='Use beta annealing')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--n_decoder_layers', type=int, default=4,
                        help='Number of transformer decoder layers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early_stopping', type=int, default=0,
                        help='Stop if val loss does not improve for N epochs (0=disabled)')

    args = parser.parse_args()

    # Validate arguments
    if args.config is None and args.cell_type is None:
        parser.error("Either --cell_type or --config must be specified")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Determine run name and mode
    use_config = args.config is not None
    if use_config:
        yaml_config = load_config(args.config)
        run_name = yaml_config.get('name', Path(args.config).stem)
    else:
        run_name = args.cell_type
        yaml_config = None

    # Setup directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(args.output_dir) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Setup logging
    logger = setup_logging(run_dir)
    logger.info(f"Starting PhysicsVAE training for {run_name}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Arguments: {vars(args)}")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create dataloaders
    logger.info("Loading datasets...")
    if use_config:
        dataloaders, yaml_config = create_multi_dataloaders(
            args.config,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        logger.info(f"Multi-dataset mode: {yaml_config['_computed']['cell_types']}")
    else:
        dataloaders = create_dataloaders(
            args.cell_type,
            args.data_dir,
            args.batch_size,
            args.num_workers
        )

    logger.info(f"Train batches: {len(dataloaders['train'])}")
    logger.info(f"Val batches: {len(dataloaders['val'])}")
    if 'test' in dataloaders:
        logger.info(f"Test batches: {len(dataloaders['test'])}")

    # Get actual physics feature count and sequence length from data
    # (dataset removes constant features, so get count from actual batch)
    sample_batch = next(iter(dataloaders['train']))
    seq_length = sample_batch['sequence'].size(1)
    n_physics = sample_batch['physics'].size(1)  # Actual features after filtering

    logger.info(f"Sequence length: {seq_length}")
    logger.info(f"Physics features: {n_physics}")

    # Create model (per spec architecture)
    logger.info("Creating PhysicsVAE model...")
    model = PhysicsVAE(
        seq_length=seq_length,
        n_physics_features=n_physics,
        latent_dim=args.latent_dim,
        physics_cond_dim=64,       # Spec: z_physics dim = 64
        n_decoder_layers=args.n_decoder_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    logger.info(f"Dropout: {args.dropout}, Decoder layers: {args.n_decoder_layers}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Save config
    model_info = {
        'n_params': n_params,
        'n_trainable': n_trainable,
        'seq_length': seq_length,
        'n_physics': n_physics,
        'latent_dim': args.latent_dim
    }
    config = save_config(args, run_dir, model_info)

    # Setup loss function
    loss_fn = CombinedVAELoss(
        beta=args.beta,
        gamma=0.0,
        beta_annealing=args.beta_annealing,
        annealing_steps=len(dataloaders['train']) * 10
    )

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    logger.info(f"Weight decay: {args.weight_decay}")

    total_steps = len(dataloaders['train']) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate * 10,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # Initialize logging files
    batch_log_file = run_dir / 'batch_log.csv'
    with open(batch_log_file, 'w') as f:
        f.write('epoch,batch,total_loss,recon_loss,kl_loss,beta,grad_norm,grad_norm_max,learning_rate,accuracy\n')

    epoch_results_file = run_dir / 'epoch_results.txt'

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    train_history = []
    val_history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, dataloaders['train'], optimizer, scheduler,
            device, loss_fn, epoch, logger, batch_log_file,
            max_grad_norm=args.max_grad_norm
        )
        train_history.append(train_metrics)

        # Validate (detailed metrics every 10 epochs)
        detailed = (epoch % 10 == 0) or (epoch == args.epochs)
        val_metrics = compute_metrics(model, dataloaders['val'], device, loss_fn, detailed=detailed)
        val_history.append(val_metrics)

        # Log summary
        lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Recon: {train_metrics['recon_loss']:.4f}, "
            f"KL: {train_metrics['kl_loss']:.4f}, "
            f"Acc: {train_metrics['accuracy']:.4f}, "
            f"Grad: {train_metrics['grad_norm_mean']:.4f}"
        )
        logger.info(
            f"Val   - Loss: {val_metrics['loss']:.4f}, "
            f"Recon: {val_metrics['recon_loss']:.4f}, "
            f"KL: {val_metrics['kl_loss']:.4f}, "
            f"Acc: {val_metrics['accuracy']:.4f}, "
            f"PPL: {val_metrics['perplexity']:.2f}"
        )

        # Log to epoch results file
        log_epoch_results(epoch, train_metrics, val_metrics, lr, epoch_results_file)

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            logger.info(f"*** New best validation loss: {best_val_loss:.4f} ***")

        if is_best or (epoch % args.save_every == 0) or (epoch == args.epochs):
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                run_dir, config=config, is_best=is_best
            )

        # Early stopping check
        if args.early_stopping > 0:
            epochs_without_improvement = epoch - best_epoch
            if epochs_without_improvement >= args.early_stopping:
                logger.info(f"\n*** Early stopping triggered after {epochs_without_improvement} epochs without improvement ***")
                logger.info(f"*** Best epoch was {best_epoch} with val loss {best_val_loss:.4f} ***")
                break

        # Create plots every 10 epochs
        if epoch % 10 == 0:
            create_training_plots(train_history, val_history, plots_dir, run_name)

    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    # Load best model
    best_checkpoint = torch.load(run_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    logger.info(f"Loaded best model from epoch {best_epoch}")

    # Test set evaluation
    if 'test' in dataloaders:
        logger.info("\nEvaluating on test set...")
        test_metrics = compute_metrics(model, dataloaders['test'], device, loss_fn, detailed=True)
        logger.info(
            f"Test - Loss: {test_metrics['loss']:.4f}, "
            f"Recon: {test_metrics['recon_loss']:.4f}, "
            f"KL: {test_metrics['kl_loss']:.4f}, "
            f"Acc: {test_metrics['accuracy']:.4f}, "
            f"PPL: {test_metrics['perplexity']:.2f}"
        )
    else:
        test_metrics = None
        logger.info("No test set available")

    # Create final plots
    create_training_plots(train_history, val_history, plots_dir, run_name)

    # Create final report
    create_final_report(run_dir, config, train_history, val_history, test_metrics, best_epoch)

    # Save all results
    results = {
        'config': config,
        'train_history': train_history,
        'val_history': val_history,
        'test_metrics': test_metrics,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch
    }

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {run_dir}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info("\nOutput files:")
    logger.info(f"  - config.json: Training configuration")
    logger.info(f"  - results.json: Full training results")
    logger.info(f"  - final_report.txt: Human-readable summary")
    logger.info(f"  - training.log: Complete training log")
    logger.info(f"  - batch_log.csv: Per-batch metrics")
    logger.info(f"  - epoch_results.txt: Per-epoch summaries")
    logger.info(f"  - plots/: Training curves")
    logger.info(f"  - best_model.pt: Best model checkpoint")


if __name__ == '__main__':
    main()
