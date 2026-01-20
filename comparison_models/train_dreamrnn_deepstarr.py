#!/usr/bin/env python3
"""
Train DREAM-RNN on DeepSTARR (S2) data for comparison with CADENCE.

This script trains the DREAM-RNN model from the DREAM Challenge on the
DeepSTARR dataset and evaluates it for comparison with CADENCE models.
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

# Add prixfixe to path
PRIXFIXE_PATH = Path("/home/bcheng/sequence_optimization/FUSEMAP/comparison_models/random-promoter-dream-challenge-2022/benchmarks/drosophila")
sys.path.insert(0, str(PRIXFIXE_PATH))

from prixfixe.autosome import AutosomeFinalLayersBlock
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.prixfixe import PrixFixeNet

# Data paths
DEEPSTARR_DATA_DIR = Path("/home/bcheng/sequence_optimization/mainproject/processed_data/DeepSTARR_data")
OUTPUT_DIR = Path("/home/bcheng/sequence_optimization/FUSEMAP/comparison_models/dreamrnn_deepstarr_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def set_seed(seed: int = 42):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fasta(fasta_path):
    """Load sequences from FASTA file."""
    sequences = []
    current_seq = ""
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append(current_seq)
                current_seq = ""
            else:
                current_seq += line
        if current_seq:
            sequences.append(current_seq)
    return sequences


def one_hot_encode(seq):
    """One-hot encode DNA sequence with reverse complement indicator."""
    mapping = {
        'A': [1, 0, 0, 0],
        'G': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }
    encoded = np.zeros((5, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in mapping:
            encoded[:4, i] = mapping[base]
    # Channel 5 is reverse complement indicator (0 for forward)
    return encoded


def reverse_complement(seq):
    """Get reverse complement of sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq.upper()))


class DeepSTARRDataset(Dataset):
    """DeepSTARR dataset for DREAM-RNN."""

    def __init__(self, split="train", augment=True, normalize_stats=None):
        self.split = split
        self.augment = augment and (split == "train")

        # Load sequences
        fasta_map = {"train": "Train", "val": "Val", "test": "Test"}
        fasta_file = DEEPSTARR_DATA_DIR / f"Sequences_{fasta_map[split]}.fa"
        self.sequences = load_fasta(fasta_file)

        # Load activities
        activity_file = DEEPSTARR_DATA_DIR / f"Sequences_activity_{fasta_map[split]}.txt"
        df = pd.read_csv(activity_file, sep='\t')

        self.dev_activity = df['Dev_log2_enrichment'].values.astype(np.float32)
        self.hk_activity = df['Hk_log2_enrichment'].values.astype(np.float32)

        # Normalize
        if normalize_stats is None:
            self.dev_mean = self.dev_activity.mean()
            self.dev_std = self.dev_activity.std()
            self.hk_mean = self.hk_activity.mean()
            self.hk_std = self.hk_activity.std()
        else:
            self.dev_mean, self.dev_std, self.hk_mean, self.hk_std = normalize_stats

        self.dev_normalized = (self.dev_activity - self.dev_mean) / self.dev_std
        self.hk_normalized = (self.hk_activity - self.hk_mean) / self.hk_std

        print(f"{split}: {len(self.sequences)} samples")
        print(f"  Dev: mean={self.dev_mean:.3f}, std={self.dev_std:.3f}")
        print(f"  Hk: mean={self.hk_mean:.3f}, std={self.hk_std:.3f}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        dev = self.dev_normalized[idx]
        hk = self.hk_normalized[idx]

        # Augment with reverse complement
        is_rev = 0
        if self.augment and random.random() > 0.5:
            seq = reverse_complement(seq)
            is_rev = 1

        # One-hot encode
        x = one_hot_encode(seq)
        x[4, :] = is_rev  # Set reverse indicator channel

        return torch.from_numpy(x), torch.tensor([dev, hk])

    def get_normalize_stats(self):
        return (self.dev_mean, self.dev_std, self.hk_mean, self.hk_std)


def build_dream_rnn(seq_size=249, generator=None):
    """Build DREAM-RNN model."""
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(42)

    first = BHIFirstLayersBlock(
        in_channels=5,
        out_channels=320,
        seqsize=seq_size,
        kernel_sizes=[9, 15],
        pool_size=1,
        dropout=0.2
    )

    core = BHICoreBlock(
        in_channels=first.out_channels,
        out_channels=320,
        seqsize=first.infer_outseqsize(),
        lstm_hidden_channels=320,
        kernel_sizes=[9, 15],
        pool_size=1,
        dropout1=0.2,
        dropout2=0.5
    )

    final = AutosomeFinalLayersBlock(in_channels=core.out_channels)

    model = PrixFixeNet(
        first=first,
        core=core,
        final=final,
        generator=generator
    )

    return model


def compute_metrics(pred_dev, pred_hk, true_dev, true_hk):
    """Compute Pearson and Spearman correlations."""
    dev_pearson = pearsonr(pred_dev, true_dev)[0]
    dev_spearman = spearmanr(pred_dev, true_dev)[0]
    hk_pearson = pearsonr(pred_hk, true_hk)[0]
    hk_spearman = spearmanr(pred_hk, true_hk)[0]

    return {
        "Dev": {"pearson": dev_pearson, "spearman": dev_spearman},
        "Hk": {"pearson": hk_pearson, "spearman": hk_spearman}
    }


def train_epoch(model, loader, optimizer, scheduler, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(loader, desc="Train")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        with autocast():
            pred = model(x)
            pred_combined = torch.stack(pred, dim=1).squeeze(-1)  # (batch, 2)
            loss = nn.functional.mse_loss(pred_combined, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device, normalize_stats):
    """Evaluate model."""
    model.eval()
    all_pred_dev, all_pred_hk = [], []
    all_true_dev, all_true_hk = [], []

    dev_mean, dev_std, hk_mean, hk_std = normalize_stats

    for x, y in tqdm(loader, desc="Eval"):
        x, y = x.to(device), y.to(device)

        with autocast():
            pred = model(x)

        # Unnormalize predictions
        pred_dev = pred[0].cpu().numpy().flatten() * dev_std + dev_mean
        pred_hk = pred[1].cpu().numpy().flatten() * hk_std + hk_mean

        # Unnormalize targets
        true_dev = y[:, 0].cpu().numpy() * dev_std + dev_mean
        true_hk = y[:, 1].cpu().numpy() * hk_std + hk_mean

        all_pred_dev.extend(pred_dev)
        all_pred_hk.extend(pred_hk)
        all_true_dev.extend(true_dev)
        all_true_hk.extend(true_hk)

    return compute_metrics(
        np.array(all_pred_dev), np.array(all_pred_hk),
        np.array(all_true_dev), np.array(all_true_hk)
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Training DREAM-RNN on DeepSTARR Data")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    # Create datasets
    print("\n1. Loading datasets...")
    train_ds = DeepSTARRDataset(split="train", augment=True)
    normalize_stats = train_ds.get_normalize_stats()

    val_ds = DeepSTARRDataset(split="val", augment=False, normalize_stats=normalize_stats)
    test_ds = DeepSTARRDataset(split="test", augment=False, normalize_stats=normalize_stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Create model
    print("\n2. Creating DREAM-RNN model...")
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    model = build_dream_rnn(seq_size=249, generator=generator)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr / 25,
        weight_decay=0.01
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25.0,
        three_phase=False
    )

    scaler = GradScaler()

    # Training loop
    print("\n3. Training...")
    best_val_dev = -1
    best_epoch = 0

    results = {
        "config": vars(args),
        "train_loss": [],
        "val_metrics": [],
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        val_metrics = evaluate(model, val_loader, device, normalize_stats)

        results["train_loss"].append(train_loss)
        results["val_metrics"].append(val_metrics)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Dev r: {val_metrics['Dev']['pearson']:.4f}, Val Hk r: {val_metrics['Hk']['pearson']:.4f}")

        # Save best model (by Dev pearson)
        if val_metrics['Dev']['pearson'] > best_val_dev:
            best_val_dev = val_metrics['Dev']['pearson']
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, OUTPUT_DIR / "best_model.pt")
            print(f"  -> New best model!")

    # Load best model and evaluate on test set
    print("\n4. Evaluating on test set...")
    checkpoint = torch.load(OUTPUT_DIR / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, normalize_stats)

    results["best_epoch"] = best_epoch
    results["best_val_dev_pearson"] = best_val_dev
    results["test_metrics"] = test_metrics

    print(f"\n" + "=" * 70)
    print("FINAL RESULTS - DREAM-RNN on DeepSTARR")
    print("=" * 70)
    print(f"Best epoch: {best_epoch+1}")
    print(f"Test Dev r: {test_metrics['Dev']['pearson']:.4f}")
    print(f"Test Hk r: {test_metrics['Hk']['pearson']:.4f}")

    # Save results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")

    # Print comparison with CADENCE
    print(f"\n" + "=" * 70)
    print("COMPARISON WITH CADENCE")
    print("=" * 70)

    cadence_results_file = Path("/home/bcheng/sequence_optimization/FUSEMAP/training/results/cadence_deepstarr_v2/final_results.json")
    if cadence_results_file.exists():
        with open(cadence_results_file) as f:
            cadence_data = json.load(f)

        if "test" in cadence_data:
            cadence_test = cadence_data["test"]["deepstarr"]
            print(f"\nCADENCE DeepSTARR v2:")
            print(f"  Dev Pearson: {cadence_test['Dev']['pearson']:.4f}")
            print(f"  Hk Pearson: {cadence_test['Hk']['pearson']:.4f}")

            print(f"\nDREAM-RNN:")
            print(f"  Dev Pearson: {test_metrics['Dev']['pearson']:.4f}")
            print(f"  Hk Pearson: {test_metrics['Hk']['pearson']:.4f}")

            dev_diff = cadence_test['Dev']['pearson'] - test_metrics['Dev']['pearson']
            hk_diff = cadence_test['Hk']['pearson'] - test_metrics['Hk']['pearson']

            print(f"\nDifference (CADENCE - DREAM-RNN):")
            print(f"  Dev: {dev_diff:+.4f}")
            print(f"  Hk: {hk_diff:+.4f}")


if __name__ == "__main__":
    main()
