#!/usr/bin/env python3
"""
Progressive Transfer Learning Experiment

Tests how model performance improves with increasing amounts of target domain data.
Compares:
- Pre-trained CADENCE → fine-tuned (frozen backbone)
- Pre-trained CADENCE → fine-tuned (full)
- Training from scratch

Data fractions: 1%, 5%, 10%, 25%
"""

import sys
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Import CADENCE
from training.models import MultiSpeciesCADENCE, initialize_weights
from training.config import ModelConfig

# Paths
BASE_DIR = Path("external_validation")
DATA_FILE = BASE_DIR / "processed" / "mouse_esc_sequences.csv"
OUTPUT_DIR = BASE_DIR / "results" / "progressive_transfer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment settings
DATA_FRACTIONS = [0.01, 0.05, 0.10, 0.25]
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training settings
BATCH_SIZE = 128
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
LR_PRETRAINED = 1e-4  # Lower LR for fine-tuning
LR_SCRATCH = 1e-3     # Higher LR for from-scratch


class MouseESCDataset(Dataset):
    """Dataset for mouse ESC STARR-seq sequences."""

    def __init__(self, sequences, activities, target_len=256):
        self.sequences = sequences
        self.activities = activities
        self.target_len = target_len
        self.mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        activity = self.activities[idx]

        # Pad/trim to target length
        if len(seq) < self.target_len:
            pad_left = (self.target_len - len(seq)) // 2
            pad_right = self.target_len - len(seq) - pad_left
            seq = 'N' * pad_left + seq + 'N' * pad_right
        elif len(seq) > self.target_len:
            start = (len(seq) - self.target_len) // 2
            seq = seq[start:start+self.target_len]

        # One-hot encode
        encoded = torch.zeros(4, self.target_len, dtype=torch.float32)
        for i, base in enumerate(seq):
            if base in self.mapping:
                encoded[self.mapping[base], i] = 1.0

        return encoded, torch.tensor(activity, dtype=torch.float32)


class SimpleCADENCE(nn.Module):
    """Simplified CADENCE model for single-task fine-tuning."""

    def __init__(self, stem_ch=64, stem_ks=11, block_channels=[80, 96, 112, 128],
                 block_kernel=9, expand_ratio=4):
        super().__init__()
        from models.CADENCE.cadence import LocalBlock, EffBlock, ResidualConcat, MapperBlock

        # Stem
        self.stem = LocalBlock(in_ch=4, out_ch=stem_ch, ks=stem_ks, activation=nn.SiLU)

        # Main blocks
        blocks = []
        in_ch = stem_ch
        pool_sizes = [2, 2, 2, 2]
        for pool_sz, out_ch in zip(pool_sizes, block_channels):
            blc = nn.Sequential(
                ResidualConcat(
                    EffBlock(in_ch=in_ch, out_ch=in_ch, ks=block_kernel,
                            resize_factor=expand_ratio, activation=nn.SiLU)
                ),
                LocalBlock(in_ch=in_ch * 2, out_ch=out_ch, ks=block_kernel, activation=nn.SiLU),
                nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity()
            )
            in_ch = out_ch
            blocks.append(blc)
        self.main = nn.Sequential(*blocks)

        # Mapper
        final_ch = block_channels[-1]
        self.mapper = MapperBlock(in_features=final_ch, out_features=final_ch * 2)

        # Head
        head_dim = final_ch * 2
        self.head = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.BatchNorm1d(head_dim),
            nn.SiLU(),
            nn.Linear(head_dim, 1)
        )

        self.apply(initialize_weights)

    def forward(self, x):
        x = self.stem(x)
        x = self.main(x)
        x = self.mapper(x)
        x = nn.functional.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        pred = self.head(x).squeeze(-1)
        return pred

    def freeze_backbone(self):
        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.main.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.stem.parameters():
            param.requires_grad = True
        for param in self.main.parameters():
            param.requires_grad = True
        for param in self.mapper.parameters():
            param.requires_grad = True


def create_model(from_pretrained=None, freeze_backbone=False):
    """Create a CADENCE model for mouse ESC prediction.

    Args:
        from_pretrained: Path to pretrained model checkpoint, or None for random init
        freeze_backbone: Whether to freeze backbone (only train head)
    """
    # Create simplified model
    model = SimpleCADENCE()

    if from_pretrained is not None:
        print(f"  Loading pretrained weights from: {from_pretrained}")
        checkpoint = torch.load(from_pretrained, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Load backbone weights (stems, main, mapper)
        model_dict = model.state_dict()
        pretrained_dict = {}

        for k, v in state_dict.items():
            # Skip head weights and embedding weights
            if 'heads.' in k or '_embed' in k:
                continue
            # Handle kingdom/species stems - map to shared stem
            if 'kingdom_stems' in k or 'species_stems' in k:
                # Map first kingdom/species stem to shared stem
                if '.0.' in k:
                    new_k = k.replace('kingdom_stems.0.', 'stem.').replace('species_stems.0.', 'stem.')
                    if new_k in model_dict and model_dict[new_k].shape == v.shape:
                        pretrained_dict[new_k] = v
                continue
            # Direct match
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict[k] = v

        print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from pretrained model")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # Re-initialize head for new task
        for module in model.head.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)

    if freeze_backbone:
        model.freeze_backbone()
        print("  Backbone frozen - only training head")

    return model


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward - SimpleCADENCE returns tensor directly
        pred = model(x)

        # MSE loss
        loss = nn.functional.mse_loss(pred, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, data_loader, device):
    """Evaluate model on dataset."""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            # SimpleCADENCE returns tensor directly
            pred = model(x)

            loss = nn.functional.mse_loss(pred, y)
            total_loss += loss.item()
            n_batches += 1

            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # Compute metrics
    spearman_r, _ = spearmanr(preds, targets)
    pearson_r, _ = pearsonr(preds, targets)

    # AUROC for top 25%
    threshold = np.percentile(targets, 75)
    labels = (targets >= threshold).astype(int)
    try:
        auroc = roc_auc_score(labels, preds)
    except:
        auroc = 0.5

    return {
        'loss': total_loss / n_batches,
        'spearman_r': float(spearman_r) if not np.isnan(spearman_r) else 0,
        'pearson_r': float(pearson_r) if not np.isnan(pearson_r) else 0,
        'auroc': float(auroc),
    }


def run_experiment(train_seqs, train_activities, val_seqs, val_activities,
                   test_seqs, test_activities, experiment_name,
                   from_pretrained=None, freeze_backbone=False, lr=1e-3):
    """Run a single training experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}")
    print(f"  Train samples: {len(train_seqs)}")
    print(f"  Val samples: {len(val_seqs)}")
    print(f"  Test samples: {len(test_seqs)}")
    print(f"  Learning rate: {lr}")

    # Create datasets
    train_dataset = MouseESCDataset(train_seqs, train_activities)
    val_dataset = MouseESCDataset(val_seqs, val_activities)
    test_dataset = MouseESCDataset(test_seqs, test_activities)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Create model
    model = create_model(from_pretrained=from_pretrained, freeze_backbone=freeze_backbone)
    model.to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # Training loop
    best_val_spearman = -float('inf')
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })

        if val_metrics['spearman_r'] > best_val_spearman:
            best_val_spearman = val_metrics['spearman_r']
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                  f"val_spearman={val_metrics['spearman_r']:.4f}, "
                  f"val_auroc={val_metrics['auroc']:.4f}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, DEVICE)

    print(f"\n  Best epoch: {best_epoch+1}")
    print(f"  Test Results:")
    print(f"    Spearman ρ: {test_metrics['spearman_r']:.4f}")
    print(f"    Pearson r:  {test_metrics['pearson_r']:.4f}")
    print(f"    AUROC:      {test_metrics['auroc']:.4f}")

    return {
        'experiment_name': experiment_name,
        'n_train': len(train_seqs),
        'n_val': len(val_seqs),
        'n_test': len(test_seqs),
        'best_epoch': best_epoch,
        'best_val_spearman': best_val_spearman,
        'test_metrics': test_metrics,
        'history': history,
        'from_pretrained': str(from_pretrained) if from_pretrained else None,
        'freeze_backbone': freeze_backbone,
        'learning_rate': lr,
    }


def main():
    print("=" * 70)
    print("PROGRESSIVE TRANSFER LEARNING EXPERIMENT")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")

    # Load data
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  Total sequences: {len(df)}")

    # Use 2iL condition as the target activity
    sequences = df['sequence'].tolist()
    activities = df['activity_2iL'].values

    # Split into train (70%), val (15%), test (15%)
    np.random.seed(SEED)
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)

    n_train = int(0.70 * len(sequences))
    n_val = int(0.15 * len(sequences))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    train_seqs = [sequences[i] for i in train_idx]
    train_activities = activities[train_idx]
    val_seqs = [sequences[i] for i in val_idx]
    val_activities = activities[val_idx]
    test_seqs = [sequences[i] for i in test_idx]
    test_activities = activities[test_idx]

    print(f"\n  Train set: {len(train_seqs)}")
    print(f"  Val set: {len(val_seqs)}")
    print(f"  Test set: {len(test_seqs)}")

    # Available pretrained models
    pretrained_models = {
        'cadence_k562': Path('results/cadence_k562_v2/best_model.pt'),
        'config4_cross_kingdom': Path('results/config4_cross_kingdom_v1/best_model.pt'),
        'config5_universal': Path('results/config5_universal_no_yeast_20260114_204533/best_model.pt'),
    }

    all_results = {}

    # Run experiments for each data fraction
    for frac in DATA_FRACTIONS:
        n_samples = int(frac * len(train_seqs))
        frac_name = f"{int(frac*100)}pct"

        print(f"\n{'#'*70}")
        print(f"DATA FRACTION: {frac*100:.0f}% ({n_samples} training samples)")
        print(f"{'#'*70}")

        # Subsample training data
        np.random.seed(SEED)
        subset_idx = np.random.choice(len(train_seqs), n_samples, replace=False)
        subset_seqs = [train_seqs[i] for i in subset_idx]
        subset_activities = train_activities[subset_idx]

        results_for_frac = {}

        # 1. Train from scratch
        result = run_experiment(
            subset_seqs, subset_activities,
            val_seqs, val_activities,
            test_seqs, test_activities,
            experiment_name=f"scratch_{frac_name}",
            from_pretrained=None,
            freeze_backbone=False,
            lr=LR_SCRATCH,
        )
        results_for_frac['from_scratch'] = result

        # 2. Fine-tune pretrained models
        for model_name, model_path in pretrained_models.items():
            if not model_path.exists():
                print(f"  Skipping {model_name} - not found")
                continue

            # Fine-tune with frozen backbone
            result = run_experiment(
                subset_seqs, subset_activities,
                val_seqs, val_activities,
                test_seqs, test_activities,
                experiment_name=f"{model_name}_frozen_{frac_name}",
                from_pretrained=model_path,
                freeze_backbone=True,
                lr=LR_PRETRAINED,
            )
            results_for_frac[f'{model_name}_frozen'] = result

            # Fine-tune full model
            result = run_experiment(
                subset_seqs, subset_activities,
                val_seqs, val_activities,
                test_seqs, test_activities,
                experiment_name=f"{model_name}_full_{frac_name}",
                from_pretrained=model_path,
                freeze_backbone=False,
                lr=LR_PRETRAINED,
            )
            results_for_frac[f'{model_name}_full'] = result

        all_results[frac_name] = results_for_frac

        # Clean up GPU memory
        torch.cuda.empty_cache()

    # Save results
    output_file = OUTPUT_DIR / "progressive_transfer_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'experiment': 'progressive_transfer_learning',
            'timestamp': datetime.now().isoformat(),
            'dataset': 'Mouse ESC STARR-seq (2iL condition)',
            'data_fractions': DATA_FRACTIONS,
            'total_train': len(train_seqs),
            'total_val': len(val_seqs),
            'total_test': len(test_seqs),
            'results': all_results,
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Progressive Transfer Learning Results")
    print("=" * 80)
    print(f"{'Method':<30} {'1%':>10} {'5%':>10} {'10%':>10} {'25%':>10}")
    print("-" * 80)

    methods = ['from_scratch', 'cadence_k562_frozen', 'cadence_k562_full',
               'config4_cross_kingdom_frozen', 'config4_cross_kingdom_full']

    for method in methods:
        row = f"{method:<30}"
        for frac in DATA_FRACTIONS:
            frac_name = f"{int(frac*100)}pct"
            if frac_name in all_results and method in all_results[frac_name]:
                spearman = all_results[frac_name][method]['test_metrics']['spearman_r']
                row += f" {spearman:>9.4f}"
            else:
                row += f" {'N/A':>9}"
        print(row)

    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
