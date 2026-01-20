"""
Bidirectional ISM Optimization Experiment
==========================================

Optimizes sequences in both directions:
1. High-performing natural sequences -> maximize activity (push higher)
2. Low-performing natural sequences -> minimize activity (push lower)

Uses validation set sequences only (not training set).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import time
import json
from datetime import datetime
from typing import List, Tuple

# Setup paths
FUSEMAP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FUSEMAP_DIR))
sys.path.insert(0, str(FUSEMAP_DIR / "results" / "optimization"))

from real_optimization import ISMOptimizer, ISMConfig
from real_model_loader import load_model_suite, DATASET_CONFIGS


# Data paths
DATA_PATHS = {
    'K562': FUSEMAP_DIR / "data/human_mpra/human_mpra/K562_clean.tsv",
    'HepG2': FUSEMAP_DIR / "data/human_mpra/human_mpra/HepG2_clean.tsv",
    'WTC11': FUSEMAP_DIR / "data/human_mpra/human_mpra/WTC11_clean.tsv",
}

# Use val split file if available, otherwise use percentile-based split
VAL_SPLIT_PATHS = {
    'K562': FUSEMAP_DIR / "data/human_mpra/human_mpra/K562_val.tsv",
    'HepG2': FUSEMAP_DIR / "data/human_mpra/human_mpra/HepG2_val.tsv",
    'WTC11': FUSEMAP_DIR / "data/human_mpra/human_mpra/WTC11_val.tsv",
}


def load_sequences_by_activity(
    cell_type: str,
    n_high: int = 50,
    n_low: int = 50,
    high_percentile: float = 90,
    low_percentile: float = 10,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Load high and low activity sequences from validation set.

    Returns:
        (high_seqs, low_seqs) - Lists of (sequence, activity) tuples
    """
    # Try to load val split first
    val_path = VAL_SPLIT_PATHS.get(cell_type)
    data_path = DATA_PATHS.get(cell_type)

    if val_path and val_path.exists():
        print(f"Loading validation set from {val_path}")
        df = pd.read_csv(val_path, sep='\t')
    elif data_path and data_path.exists():
        print(f"Loading from {data_path}")
        df = pd.read_csv(data_path, sep='\t')

        # Use fold column for validation if available (folds 1-2 for val, others for train)
        if 'fold' in df.columns:
            print("Using fold column for train/val split (folds 1-2 = val)")
            df = df[df['fold'].isin([1, 2])]  # Use folds 1-2 as validation
        else:
            # Random 20% holdout
            print("Using random 20% holdout as validation")
            np.random.seed(42)
            val_mask = np.random.rand(len(df)) < 0.2
            df = df[val_mask]
    else:
        raise FileNotFoundError(f"No data found for {cell_type}")

    # Get sequence length
    expected_len = DATASET_CONFIGS[cell_type]['sequence_length']

    # Identify sequence and activity columns
    seq_col = 'seq' if 'seq' in df.columns else 'sequence'
    act_col = 'mean_value' if 'mean_value' in df.columns else 'activity'

    # Filter by sequence length
    df = df[df[seq_col].str.len() == expected_len].copy()
    print(f"Found {len(df)} sequences of length {expected_len}")

    # Get high and low activity thresholds
    high_thresh = np.percentile(df[act_col], high_percentile)
    low_thresh = np.percentile(df[act_col], low_percentile)

    print(f"Activity thresholds: low < {low_thresh:.3f}, high > {high_thresh:.3f}")

    # Select high activity sequences
    high_df = df[df[act_col] >= high_thresh].nlargest(n_high, act_col)
    high_seqs = [(row[seq_col], float(row[act_col])) for _, row in high_df.iterrows()]

    # Select low activity sequences
    low_df = df[df[act_col] <= low_thresh].nsmallest(n_low, act_col)
    low_seqs = [(row[seq_col], float(row[act_col])) for _, row in low_df.iterrows()]

    print(f"Selected {len(high_seqs)} high-activity sequences (mean={np.mean([s[1] for s in high_seqs]):.3f})")
    print(f"Selected {len(low_seqs)} low-activity sequences (mean={np.mean([s[1] for s in low_seqs]):.3f})")

    return high_seqs, low_seqs


def run_bidirectional_optimization(
    cell_type: str,
    n_high: int = 50,
    n_low: int = 50,
    max_iterations: int = 100,
    device: str = "cuda",
    output_dir: Path = None,
):
    """Run bidirectional ISM optimization."""

    print("=" * 70)
    print(f"BIDIRECTIONAL ISM OPTIMIZATION - {cell_type}")
    print("=" * 70)

    # Setup output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = FUSEMAP_DIR / "oracle_check" / "bidirectional_results" / f"{cell_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model suite
    print("\nLoading model suite...")
    model_suite = load_model_suite(cell_type, device=device)

    # Load sequences
    print("\nLoading sequences from validation set...")
    high_seqs, low_seqs = load_sequences_by_activity(cell_type, n_high, n_low)

    results = {
        'cell_type': cell_type,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_high': n_high,
            'n_low': n_low,
            'max_iterations': max_iterations,
        },
        'high_to_higher': [],
        'low_to_lower': [],
    }

    # =========================================================================
    # Part 1: Optimize HIGH -> HIGHER (maximize)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: HIGH -> HIGHER (Maximizing activity)")
    print("=" * 70)

    max_config = ISMConfig(
        max_iterations=max_iterations,
        early_stop_patience=15,
        minimize=False,
    )
    max_optimizer = ISMOptimizer(model_suite, config=max_config)

    for i, (seq, data_activity) in enumerate(high_seqs):
        print(f"\n[{i+1}/{len(high_seqs)}] High seq (data_label={data_activity:.4f})")
        start_time = time.time()

        # Use verbose=True to see ISM iterations
        history = max_optimizer.optimize(seq, verbose=(i < 3))  # Verbose for first 3
        elapsed = time.time() - start_time

        if history:
            model_initial = history[0].activity
            model_final = history[-1].activity
            final_seq = history[-1].sequence
            n_mutations = sum(1 for a, b in zip(seq, final_seq) if a != b)
            improvement = model_final - model_initial

            print(f"    Model: {model_initial:.4f} -> {model_final:.4f} "
                  f"({'+' if improvement >= 0 else ''}{improvement:.4f}, {n_mutations} mutations, {elapsed:.1f}s)")

            results['high_to_higher'].append({
                'initial_seq': seq,
                'final_seq': final_seq,
                'data_activity': data_activity,
                'model_initial': model_initial,
                'model_final': model_final,
                'improvement': improvement,
                'n_mutations': n_mutations,
                'n_iterations': len(history),
                'elapsed_seconds': elapsed,
            })

    # =========================================================================
    # Part 2: Optimize LOW -> LOWER (minimize)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: LOW -> LOWER (Minimizing activity)")
    print("=" * 70)

    min_config = ISMConfig(
        max_iterations=max_iterations,
        early_stop_patience=15,
        minimize=True,
    )
    min_optimizer = ISMOptimizer(model_suite, config=min_config)

    for i, (seq, data_activity) in enumerate(low_seqs):
        print(f"\n[{i+1}/{len(low_seqs)}] Low seq (data_label={data_activity:.4f})")
        start_time = time.time()

        # Use verbose=True to see ISM iterations
        history = min_optimizer.optimize(seq, verbose=(i < 3))  # Verbose for first 3
        elapsed = time.time() - start_time

        if history:
            model_initial = history[0].activity
            model_final = history[-1].activity
            final_seq = history[-1].sequence
            n_mutations = sum(1 for a, b in zip(seq, final_seq) if a != b)
            reduction = model_initial - model_final

            print(f"    Model: {model_initial:.4f} -> {model_final:.4f} "
                  f"({'-' if reduction >= 0 else '+'}{abs(reduction):.4f}, {n_mutations} mutations, {elapsed:.1f}s)")

            results['low_to_lower'].append({
                'initial_seq': seq,
                'final_seq': final_seq,
                'data_activity': data_activity,
                'model_initial': model_initial,
                'model_final': model_final,
                'reduction': reduction,
                'n_mutations': n_mutations,
                'n_iterations': len(history),
                'elapsed_seconds': elapsed,
            })

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results['high_to_higher']:
        high_improvements = [r['improvement'] for r in results['high_to_higher']]
        high_initial = [r['model_initial'] for r in results['high_to_higher']]
        high_final = [r['model_final'] for r in results['high_to_higher']]
        print(f"\nHIGH -> HIGHER (Maximizing):")
        print(f"  N sequences: {len(results['high_to_higher'])}")
        print(f"  Model initial: {np.mean(high_initial):.4f} ± {np.std(high_initial):.4f}")
        print(f"  Model final: {np.mean(high_final):.4f} ± {np.std(high_final):.4f}")
        print(f"  Mean improvement: {np.mean(high_improvements):.4f} ± {np.std(high_improvements):.4f}")
        print(f"  Max improvement: {np.max(high_improvements):.4f}")

    if results['low_to_lower']:
        low_reductions = [r['reduction'] for r in results['low_to_lower']]
        low_initial = [r['model_initial'] for r in results['low_to_lower']]
        low_final = [r['model_final'] for r in results['low_to_lower']]
        print(f"\nLOW -> LOWER (Minimizing):")
        print(f"  N sequences: {len(results['low_to_lower'])}")
        print(f"  Model initial: {np.mean(low_initial):.4f} ± {np.std(low_initial):.4f}")
        print(f"  Model final: {np.mean(low_final):.4f} ± {np.std(low_final):.4f}")
        print(f"  Mean reduction: {np.mean(low_reductions):.4f} ± {np.std(low_reductions):.4f}")
        print(f"  Max reduction: {np.max(low_reductions):.4f}")

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Save sequences
    high_seqs_file = output_dir / "high_optimized_sequences.tsv"
    if results['high_to_higher']:
        high_df = pd.DataFrame(results['high_to_higher'])
        high_df.to_csv(high_seqs_file, sep='\t', index=False)
        print(f"High sequences saved to {high_seqs_file}")

    low_seqs_file = output_dir / "low_optimized_sequences.tsv"
    if results['low_to_lower']:
        low_df = pd.DataFrame(results['low_to_lower'])
        low_df.to_csv(low_seqs_file, sep='\t', index=False)
        print(f"Low sequences saved to {low_seqs_file}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run bidirectional ISM optimization")
    parser.add_argument("--cell-type", type=str, default="K562",
                        help="Cell type (K562, HepG2, WTC11)")
    parser.add_argument("--n-high", type=int, default=50,
                        help="Number of high-activity sequences")
    parser.add_argument("--n-low", type=int, default=50,
                        help="Number of low-activity sequences")
    parser.add_argument("--max-iterations", type=int, default=100,
                        help="Max ISM iterations per sequence")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None

    run_bidirectional_optimization(
        cell_type=args.cell_type,
        n_high=args.n_high,
        n_low=args.n_low,
        max_iterations=args.max_iterations,
        device=args.device,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
