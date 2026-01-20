#!/usr/bin/env python3
"""Evaluate trained DREAM-RNN model on DeepSTARR test set."""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

# Add prixfixe to path
PRIXFIXE_PATH = Path("/home/bcheng/sequence_optimization/FUSEMAP/comparison_models/random-promoter-dream-challenge-2022/benchmarks/drosophila")
sys.path.insert(0, str(PRIXFIXE_PATH))

from prixfixe.autosome import AutosomeFinalLayersBlock
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.prixfixe import PrixFixeNet

DEEPSTARR_DATA_DIR = Path("/home/bcheng/sequence_optimization/mainproject/processed_data/DeepSTARR_data")
OUTPUT_DIR = Path("/home/bcheng/sequence_optimization/FUSEMAP/comparison_models/dreamrnn_deepstarr_results")


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
    """One-hot encode DNA sequence."""
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
    return encoded


class DeepSTARRDataset(Dataset):
    def __init__(self, split="test", normalize_stats=None):
        fasta_map = {"train": "Train", "val": "Val", "test": "Test"}
        fasta_file = DEEPSTARR_DATA_DIR / f"Sequences_{fasta_map[split]}.fa"
        self.sequences = load_fasta(fasta_file)

        activity_file = DEEPSTARR_DATA_DIR / f"Sequences_activity_{fasta_map[split]}.txt"
        df = pd.read_csv(activity_file, sep='\t')

        self.dev_activity = df['Dev_log2_enrichment'].values.astype(np.float32)
        self.hk_activity = df['Hk_log2_enrichment'].values.astype(np.float32)

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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = one_hot_encode(self.sequences[idx])
        return torch.from_numpy(x), torch.tensor([self.dev_normalized[idx], self.hk_normalized[idx]])

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

        pred_dev = pred[0].cpu().numpy().flatten() * dev_std + dev_mean
        pred_hk = pred[1].cpu().numpy().flatten() * hk_std + hk_mean

        true_dev = y[:, 0].cpu().numpy() * dev_std + dev_mean
        true_hk = y[:, 1].cpu().numpy() * hk_std + hk_mean

        all_pred_dev.extend(pred_dev)
        all_pred_hk.extend(pred_hk)
        all_true_dev.extend(true_dev)
        all_true_hk.extend(true_hk)

    pred_dev = np.array(all_pred_dev, dtype=np.float64)
    pred_hk = np.array(all_pred_hk, dtype=np.float64)
    true_dev = np.array(all_true_dev, dtype=np.float64)
    true_hk = np.array(all_true_hk, dtype=np.float64)

    return {
        "Dev": {
            "pearson": float(pearsonr(pred_dev, true_dev)[0]),
            "spearman": float(spearmanr(pred_dev, true_dev)[0])
        },
        "Hk": {
            "pearson": float(pearsonr(pred_hk, true_hk)[0]),
            "spearman": float(spearmanr(pred_hk, true_hk)[0])
        }
    }


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Evaluating DREAM-RNN on DeepSTARR Test Set")
    print("=" * 70)

    # Load training stats
    train_ds = DeepSTARRDataset(split="train")
    normalize_stats = train_ds.get_normalize_stats()
    del train_ds

    # Load test dataset
    test_ds = DeepSTARRDataset(split="test", normalize_stats=normalize_stats)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    # Build model
    print("\nLoading model...")
    generator = torch.Generator()
    generator.manual_seed(42)
    model = build_dream_rnn(seq_size=249, generator=generator)

    # Load checkpoint
    checkpoint = torch.load(OUTPUT_DIR / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")

    # Evaluate
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device, normalize_stats)

    print(f"\n" + "=" * 70)
    print("DREAM-RNN Test Results")
    print("=" * 70)
    print(f"Dev Pearson: {test_metrics['Dev']['pearson']:.4f}")
    print(f"Dev Spearman: {test_metrics['Dev']['spearman']:.4f}")
    print(f"Hk Pearson: {test_metrics['Hk']['pearson']:.4f}")
    print(f"Hk Spearman: {test_metrics['Hk']['spearman']:.4f}")

    # Save results
    results = {
        "model": "DREAM-RNN",
        "dataset": "DeepSTARR",
        "best_epoch": checkpoint['epoch'] + 1,
        "test_metrics": test_metrics,
        "val_metrics": checkpoint.get('val_metrics', {})
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Compare with CADENCE
    print(f"\n" + "=" * 70)
    print("COMPARISON WITH CADENCE")
    print("=" * 70)

    cadence_results_file = Path("/home/bcheng/sequence_optimization/FUSEMAP/training/results/cadence_deepstarr_v2/final_results.json")
    if cadence_results_file.exists():
        with open(cadence_results_file) as f:
            cadence_data = json.load(f)

        if "test" in cadence_data:
            cadence_test = cadence_data["test"]["deepstarr"]

            print(f"\n{'Model':<20} {'Dev Pearson':>12} {'Hk Pearson':>12}")
            print("-" * 46)
            print(f"{'CADENCE v2':<20} {cadence_test['Dev']['pearson']:>12.4f} {cadence_test['Hk']['pearson']:>12.4f}")
            print(f"{'DREAM-RNN':<20} {test_metrics['Dev']['pearson']:>12.4f} {test_metrics['Hk']['pearson']:>12.4f}")

            dev_diff = cadence_test['Dev']['pearson'] - test_metrics['Dev']['pearson']
            hk_diff = cadence_test['Hk']['pearson'] - test_metrics['Hk']['pearson']

            print("-" * 46)
            print(f"{'Diff (CADENCE-RNN)':<20} {dev_diff:>+12.4f} {hk_diff:>+12.4f}")

            # Save comparison
            comparison = {
                "CADENCE_v2": {
                    "Dev_pearson": cadence_test['Dev']['pearson'],
                    "Hk_pearson": cadence_test['Hk']['pearson']
                },
                "DREAM_RNN": {
                    "Dev_pearson": test_metrics['Dev']['pearson'],
                    "Hk_pearson": test_metrics['Hk']['pearson']
                },
                "difference_CADENCE_minus_RNN": {
                    "Dev": dev_diff,
                    "Hk": hk_diff
                }
            }

            with open(OUTPUT_DIR / "comparison.json", "w") as f:
                json.dump(comparison, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
