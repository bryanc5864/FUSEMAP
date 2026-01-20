#!/usr/bin/env python3
"""
Run config5_universal transfer learning experiments only.
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

from training.models import initialize_weights

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
LR_PRETRAINED = 1e-4


class MouseESCDataset(Dataset):
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

        if len(seq) < self.target_len:
            pad_left = (self.target_len - len(seq)) // 2
            pad_right = self.target_len - len(seq) - pad_left
            seq = 'N' * pad_left + seq + 'N' * pad_right
        elif len(seq) > self.target_len:
            start = (len(seq) - self.target_len) // 2
            seq = seq[start:start+self.target_len]

        encoded = torch.zeros(4, self.target_len, dtype=torch.float32)
        for i, base in enumerate(seq):
            if base in self.mapping:
                encoded[self.mapping[base], i] = 1.0

        return encoded, torch.tensor(activity, dtype=torch.float32)


class SimpleCADENCE(nn.Module):
    def __init__(self, stem_ch=64, stem_ks=11, block_channels=[80, 96, 112, 128],
                 block_kernel=9, expand_ratio=4):
        super().__init__()
        from models.CADENCE.cadence import LocalBlock, EffBlock, ResidualConcat, MapperBlock

        self.stem = LocalBlock(in_ch=4, out_ch=stem_ch, ks=stem_ks, activation=nn.SiLU)

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

        final_ch = block_channels[-1]
        self.mapper = MapperBlock(in_features=final_ch, out_features=final_ch * 2)

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


def create_model(from_pretrained=None, freeze_backbone=False):
    model = SimpleCADENCE()

    if from_pretrained is not None:
        print(f"  Loading pretrained weights from: {from_pretrained}")
        checkpoint = torch.load(from_pretrained, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        model_dict = model.state_dict()
        pretrained_dict = {}

        for k, v in state_dict.items():
            if 'heads.' in k or '_embed' in k:
                continue
            if 'kingdom_stems' in k or 'species_stems' in k:
                if '.0.' in k:
                    new_k = k.replace('kingdom_stems.0.', 'stem.').replace('species_stems.0.', 'stem.')
                    if new_k in model_dict and model_dict[new_k].shape == v.shape:
                        pretrained_dict[new_k] = v
                continue
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict[k] = v

        print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from pretrained model")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

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
    model.train()
    total_loss = 0
    n_batches = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    spearman_r, _ = spearmanr(preds, targets)
    pearson_r, _ = pearsonr(preds, targets)

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
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}")
    print(f"  Train samples: {len(train_seqs)}")

    train_dataset = MouseESCDataset(train_seqs, train_activities)
    val_dataset = MouseESCDataset(val_seqs, val_activities)
    test_dataset = MouseESCDataset(test_seqs, test_activities)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = create_model(from_pretrained=from_pretrained, freeze_backbone=freeze_backbone)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_val_spearman = -float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        if val_metrics['spearman_r'] > best_val_spearman:
            best_val_spearman = val_metrics['spearman_r']
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                  f"val_spearman={val_metrics['spearman_r']:.4f}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, DEVICE)

    print(f"\n  Best epoch: {best_epoch+1}")
    print(f"  Test Results:")
    print(f"    Spearman ρ: {test_metrics['spearman_r']:.4f}")
    print(f"    AUROC:      {test_metrics['auroc']:.4f}")

    return {
        'experiment_name': experiment_name,
        'n_train': len(train_seqs),
        'best_epoch': best_epoch,
        'test_metrics': test_metrics,
    }


def main():
    print("=" * 70)
    print("CONFIG5 UNIVERSAL TRANSFER LEARNING EXPERIMENT")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    # Load data
    df = pd.read_csv(DATA_FILE)
    sequences = df['sequence'].tolist()
    activities = df['activity_2iL'].values

    # Split
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

    model_path = Path('results/config5_universal_no_yeast_20260114_204533/best_model.pt')
    all_results = {}

    for frac in DATA_FRACTIONS:
        n_samples = int(frac * len(train_seqs))
        frac_name = f"{int(frac*100)}pct"

        print(f"\n{'#'*70}")
        print(f"DATA FRACTION: {frac*100:.0f}% ({n_samples} training samples)")
        print(f"{'#'*70}")

        np.random.seed(SEED)
        subset_idx = np.random.choice(len(train_seqs), n_samples, replace=False)
        subset_seqs = [train_seqs[i] for i in subset_idx]
        subset_activities = train_activities[subset_idx]

        results_for_frac = {}

        # Frozen backbone
        result = run_experiment(
            subset_seqs, subset_activities,
            val_seqs, val_activities,
            test_seqs, test_activities,
            experiment_name=f"config5_universal_frozen_{frac_name}",
            from_pretrained=model_path,
            freeze_backbone=True,
            lr=LR_PRETRAINED,
        )
        results_for_frac['config5_universal_frozen'] = result

        # Full fine-tune
        result = run_experiment(
            subset_seqs, subset_activities,
            val_seqs, val_activities,
            test_seqs, test_activities,
            experiment_name=f"config5_universal_full_{frac_name}",
            from_pretrained=model_path,
            freeze_backbone=False,
            lr=LR_PRETRAINED,
        )
        results_for_frac['config5_universal_full'] = result

        all_results[frac_name] = results_for_frac
        torch.cuda.empty_cache()

    # Save results
    output_file = OUTPUT_DIR / "config5_transfer_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'experiment': 'config5_universal_transfer',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Config5 Universal Transfer Learning")
    print("=" * 70)
    print(f"{'Method':<30} {'1%':>10} {'5%':>10} {'10%':>10} {'25%':>10}")
    print("-" * 70)

    for method in ['config5_universal_frozen', 'config5_universal_full']:
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
