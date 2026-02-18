#!/usr/bin/env python3
"""
Generate training curve data for LaTeX/TikZ.
Extracts real epoch-by-epoch metrics from CADENCE training logs.
"""

import re
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/home/bcheng/sequence_optimization/FUSEMAP")
TRAINING_DIR = BASE_DIR / "training/results"
OUTPUT_DIR = BASE_DIR / "paper/scripts/output"

def parse_training_log(log_file):
    """Parse training.log file to extract epoch metrics."""
    epochs = []

    with open(log_file) as f:
        content = f.read()

    # Pattern for epoch summaries
    # Example: [VAL]   NLL: 2.3456 | MSE: 0.1234 | r: 0.8901 | rho: 0.8765 | R2: 0.7890 | RMSE: 0.3514
    train_pattern = r'\[TRAIN\]\s*NLL:\s*([\d.]+)\s*\|\s*MSE:\s*([\d.]+)\s*\|\s*r:\s*([\d.-]+)\s*\|\s*rho:\s*([\d.-]+)'
    val_pattern = r'\[VAL\]\s*NLL:\s*([\d.]+)\s*\|\s*MSE:\s*([\d.]+)\s*\|\s*r:\s*([\d.-]+)\s*\|\s*rho:\s*([\d.-]+)'

    train_matches = re.findall(train_pattern, content)
    val_matches = re.findall(val_pattern, content)

    # Pair train and val metrics by epoch
    for i, (train, val) in enumerate(zip(train_matches, val_matches)):
        epochs.append({
            'epoch': i + 1,
            'train_nll': float(train[0]),
            'train_mse': float(train[1]),
            'train_pearson': float(train[2]),
            'train_spearman': float(train[3]),
            'val_nll': float(val[0]),
            'val_mse': float(val[1]),
            'val_pearson': float(val[2]),
            'val_spearman': float(val[3]),
        })

    return epochs

def output_tikz_curves(epochs, output_file, model_name):
    """Write training curves in TikZ-compatible format."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(f"% Training curves for {model_name}\n")
        f.write(f"% Epochs: {len(epochs)}\n\n")

        # Training loss curve
        f.write("% Training NLL (loss)\n")
        f.write("\\addplot[blue, thick] coordinates {\n")
        for e in epochs:
            f.write(f"  ({e['epoch']},{e['train_nll']:.4f})\n")
        f.write("};\n\\addlegendentry{Train Loss}\n\n")

        # Validation loss curve
        f.write("% Validation NLL (loss)\n")
        f.write("\\addplot[red, thick, dashed] coordinates {\n")
        for e in epochs:
            f.write(f"  ({e['epoch']},{e['val_nll']:.4f})\n")
        f.write("};\n\\addlegendentry{Val Loss}\n\n")

        # Spearman correlation curve
        f.write("% Validation Spearman correlation\n")
        f.write("\\addplot[green!60!black, thick] coordinates {\n")
        for e in epochs:
            f.write(f"  ({e['epoch']},{e['val_spearman']:.4f})\n")
        f.write("};\n\\addlegendentry{Val $\\rho$}\n")

    print(f"Wrote {len(epochs)} epochs to {output_file}")

def find_best_training_logs():
    """Find training logs with the most data."""
    logs = []

    for model_dir in TRAINING_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        log_file = model_dir / "training.log"
        if log_file.exists():
            size = log_file.stat().st_size
            logs.append((model_dir.name, log_file, size))

    # Sort by size (larger = more epochs/data)
    logs.sort(key=lambda x: x[2], reverse=True)
    return logs

def generate_representative_curves():
    """Generate training curves for representative models."""
    logs = find_best_training_logs()

    print(f"Found {len(logs)} training logs:")
    for name, path, size in logs[:5]:
        print(f"  {name}: {size/1024:.1f} KB")

    # Process top models
    for name, log_file, size in logs[:3]:
        print(f"\nProcessing {name}...")

        epochs = parse_training_log(log_file)
        if not epochs:
            print(f"  No epochs found, skipping")
            continue

        print(f"  Found {len(epochs)} epochs")

        # Output TikZ
        safe_name = name.replace('_', '-')
        output_file = OUTPUT_DIR / f"training_curves_{safe_name}.tex"
        output_tikz_curves(epochs, output_file, name)

        # Also save as CSV
        df = pd.DataFrame(epochs)
        csv_file = OUTPUT_DIR / f"training_curves_{safe_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"  Also saved to {csv_file}")

def print_final_metrics():
    """Print final validation metrics for each model."""
    print("\n=== Final Validation Metrics ===")

    for model_dir in sorted(TRAINING_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        results_file = model_dir / "final_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)

            print(f"\n{model_dir.name}:")
            # Print first dataset's metrics
            for dataset, tissues in results.items():
                for tissue, metrics in tissues.items():
                    if 'spearman' in metrics:
                        spearman = metrics['spearman']['value']
                        pearson = metrics['pearson']['value']
                        print(f"  {dataset}/{tissue}: ρ={spearman:.3f}, r={pearson:.3f}")
                    break
                break

if __name__ == "__main__":
    print("Generating training curves from real logs...")

    generate_representative_curves()
    print_final_metrics()
