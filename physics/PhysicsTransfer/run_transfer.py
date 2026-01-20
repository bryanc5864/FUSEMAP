#!/usr/bin/env python3
"""
CLI for running PhysicsTransfer experiments.

Usage:
    python -m PhysicsTransfer.run_transfer --experiment human_to_drosophila
    python -m PhysicsTransfer.run_transfer --source WTC11 --target S2_dev
    python -m PhysicsTransfer.run_transfer --list-experiments
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from .config import (
    TransferConfig, DATASETS, EXPERIMENTS, get_fusemap_root
)
from .data_loader import PhysicsDataLoader
from .protocols import (
    ZeroShotTransfer,
    PhysicsAnchoredFineTuning,
    MultiSpeciesJointTraining,
    run_experiment
)
from .evaluation import TransferEvaluator


def list_datasets():
    """Print available datasets."""
    print("\nAvailable Datasets:")
    print("-" * 70)
    for name, config in DATASETS.items():
        cadence_marker = " [CADENCE]" if config.has_cadence else ""
        print(f"  {name:20s} - {config.description}{cadence_marker}")
        print(f"                       Species: {config.species}, Kingdom: {config.kingdom}")
    print()


def list_experiments():
    """Print available experiments."""
    print("\nAvailable Experiments:")
    print("-" * 70)
    for name, config in EXPERIMENTS.items():
        print(f"\n  {name}:")
        print(f"    {config.description}")
        print(f"    Source: {', '.join(config.source_datasets)}")
        print(f"    Target: {config.target_dataset}")
        print(f"    Fine-tune sizes: {config.fine_tune_sizes}")
    print()


def check_data_availability():
    """Check which datasets have data available."""
    print("\nData Availability Check:")
    print("-" * 70)
    loader = PhysicsDataLoader()

    for name, config in DATASETS.items():
        try:
            data_path = loader._find_data_file(config, 'train')
            if data_path and data_path.exists():
                print(f"  [OK] {name:20s} - {data_path.name}")
            else:
                print(f"  [--] {name:20s} - Data file not found")
        except Exception as e:
            print(f"  [!!] {name:20s} - Error: {e}")
    print()


def run_custom_transfer(
    source_datasets: list,
    target_dataset: str,
    protocols: list = None,
    fine_tune_sizes: list = None,
    output_dir: str = None
):
    """
    Run custom transfer experiment.

    Args:
        source_datasets: List of source dataset names
        target_dataset: Target dataset name
        protocols: Which protocols to run (1=zero-shot, 2=fine-tune, 3=joint)
        fine_tune_sizes: Sample sizes for fine-tuning
        output_dir: Output directory
    """
    protocols = protocols or [1, 2, 3]
    fine_tune_sizes = fine_tune_sizes or [1000, 5000, 10000]

    config = TransferConfig()
    evaluator = TransferEvaluator(config)

    results = {
        'experiment': f"custom_{source_datasets[0]}_to_{target_dataset}",
        'description': f"Custom transfer: {source_datasets} → {target_dataset}",
        'protocols': {}
    }

    # Protocol 1: Zero-Shot
    if 1 in protocols:
        print("\n" + "=" * 70)
        print("Running Protocol 1: Zero-Shot Transfer")
        print("=" * 70)
        try:
            zero_shot = ZeroShotTransfer(config)
            zs_result = zero_shot.run(source_datasets, target_dataset)
            results['protocols']['zero_shot'] = zs_result
        except Exception as e:
            print(f"  Error in zero-shot: {e}")

    # Protocol 2: Fine-Tuning
    if 2 in protocols:
        print("\n" + "=" * 70)
        print("Running Protocol 2: Physics-Anchored Fine-Tuning")
        print("=" * 70)
        try:
            fine_tuning = PhysicsAnchoredFineTuning(config)
            ft_results = fine_tuning.run(
                source_datasets, target_dataset,
                fine_tune_sizes=fine_tune_sizes
            )
            results['protocols']['fine_tuning'] = ft_results
        except Exception as e:
            print(f"  Error in fine-tuning: {e}")

    # Protocol 3: Joint Training
    if 3 in protocols:
        print("\n" + "=" * 70)
        print("Running Protocol 3: Multi-Species Joint Training")
        print("=" * 70)
        try:
            all_datasets = list(set(source_datasets + [target_dataset]))
            joint = MultiSpeciesJointTraining(config)
            joint_results = joint.run(all_datasets, holdout_dataset=target_dataset)
            results['protocols']['joint_training'] = joint_results
        except Exception as e:
            print(f"  Error in joint training: {e}")

    # Generate report and export
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    report = evaluator.generate_report(results)
    print(report)

    if output_dir:
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = output_path / f"{results['experiment']}_{timestamp}"
        evaluator.export_results(results, exp_dir)
        print(f"\nResults exported to: {exp_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='PhysicsTransfer: Cross-Species Transfer Learning via Physics Features'
    )

    # Listing options
    parser.add_argument('--list-datasets', action='store_true',
                        help='List available datasets')
    parser.add_argument('--list-experiments', action='store_true',
                        help='List pre-configured experiments')
    parser.add_argument('--check-data', action='store_true',
                        help='Check data availability')

    # Pre-configured experiment
    parser.add_argument('--experiment', '-e', type=str,
                        help='Run a pre-configured experiment')

    # Custom transfer
    parser.add_argument('--source', '-s', type=str, nargs='+',
                        help='Source dataset(s)')
    parser.add_argument('--target', '-t', type=str,
                        help='Target dataset')

    # Protocol selection
    parser.add_argument('--protocols', '-p', type=int, nargs='+',
                        default=[1, 2, 3],
                        help='Protocols to run (1=zero-shot, 2=fine-tune, 3=joint)')

    # Fine-tuning options
    parser.add_argument('--fine-tune-sizes', type=int, nargs='+',
                        default=[1000, 5000, 10000],
                        help='Sample sizes for fine-tuning')

    # Output
    parser.add_argument('--output', '-o', type=str,
                        default=None,
                        help='Output directory (default: physics/PhysicsTransfer/results)')

    # Model options
    parser.add_argument('--probe-type', type=str, default='ridge',
                        choices=['elastic_net', 'ridge', 'lasso', 'mlp'],
                        help='Type of physics probe model (ridge recommended)')

    args = parser.parse_args()

    # Handle listing commands
    if args.list_datasets:
        list_datasets()
        return

    if args.list_experiments:
        list_experiments()
        return

    if args.check_data:
        check_data_availability()
        return

    # Set output directory
    if args.output is None:
        args.output = get_fusemap_root() / 'physics/PhysicsTransfer/results'

    # Run pre-configured experiment
    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"Error: Unknown experiment '{args.experiment}'")
            print(f"Available: {list(EXPERIMENTS.keys())}")
            sys.exit(1)

        config = TransferConfig(probe_type=args.probe_type)
        results = run_experiment(args.experiment, config)

        # Export results
        evaluator = TransferEvaluator(config)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(args.output) / f"{args.experiment}_{timestamp}"
        evaluator.export_results(results, exp_dir)
        print(f"\nResults exported to: {exp_dir}")
        return

    # Run custom transfer
    if args.source and args.target:
        # Validate datasets
        for ds in args.source + [args.target]:
            if ds not in DATASETS:
                print(f"Error: Unknown dataset '{ds}'")
                list_datasets()
                sys.exit(1)

        run_custom_transfer(
            source_datasets=args.source,
            target_dataset=args.target,
            protocols=args.protocols,
            fine_tune_sizes=args.fine_tune_sizes,
            output_dir=args.output
        )
        return

    # No command specified
    parser.print_help()


if __name__ == '__main__':
    main()
