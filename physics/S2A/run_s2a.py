#!/usr/bin/env python
"""
CLI entry point for S2A (Sequence-to-Activity) zero-shot prediction.

Usage:
    # Train and save a universal head
    python run_s2a.py train --datasets K562 HepG2 WTC11 arabidopsis_leaf sorghum_leaf maize_leaf \
        --holdout S2_dev --output-dir results/s2a/

    # Zero-shot inference
    python run_s2a.py predict --checkpoint results/s2a/ \
        --input new_species_descriptors.tsv --output predictions.tsv --mode zscore

    # Full leave-one-out evaluation
    python run_s2a.py evaluate --datasets K562 HepG2 WTC11 S2_dev arabidopsis_leaf sorghum_leaf maize_leaf \
        --output-dir results/s2a/leave_one_out/

    # Compare transfer scenarios
    python run_s2a.py compare --output-dir results/s2a/comparison/
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from physics.S2A.config import S2AConfig, S2A_DATASETS, S2A_DATASET_GROUPS
from physics.S2A.universal_features import UniversalFeatureExtractor, count_universal_vs_total_features
from physics.S2A.training import UniversalS2ATrainer
from physics.S2A.inference import S2APredictor, predict_from_descriptors_file
from physics.S2A.evaluation import S2AEvaluator, compare_transfer_scenarios


def cmd_train(args):
    """Train a universal S2A head."""
    print("\n" + "="*60)
    print("S2A TRAINING")
    print("="*60)

    # Parse datasets
    if args.datasets:
        datasets = args.datasets
    elif args.dataset_group:
        if args.dataset_group in S2A_DATASET_GROUPS:
            datasets = list(S2A_DATASET_GROUPS[args.dataset_group])
        else:
            print(f"Unknown dataset group: {args.dataset_group}")
            print(f"Available: {list(S2A_DATASET_GROUPS.keys())}")
            return 1
    else:
        datasets = list(S2A_DATASETS.keys())

    # Configure
    config = S2AConfig(
        head_type=args.head_type,
        head_alpha=args.alpha,
        output_dir=args.output_dir
    )

    trainer = UniversalS2ATrainer(config)

    if args.holdout:
        # Leave-one-out training
        if args.holdout not in datasets:
            datasets.append(args.holdout)

        head, extractor, results = trainer.train_leave_one_out(
            datasets,
            args.holdout,
            train_split=args.train_split,
            test_split=args.test_split,
            verbose=True
        )
        checkpoint_name = f'universal_s2a_holdout_{args.holdout}'
    else:
        # Train on all datasets
        head, extractor, results = trainer.train(
            datasets,
            split=args.train_split,
            verbose=True
        )
        checkpoint_name = 'universal_s2a'

    # Save checkpoint
    trainer.save_checkpoint(
        head, extractor, results,
        args.output_dir, checkpoint_name
    )

    print(f"\nCheckpoint saved to {args.output_dir}")
    return 0


def cmd_predict(args):
    """Run zero-shot or calibrated prediction."""
    print("\n" + "="*60)
    print("S2A PREDICTION")
    print("="*60)

    # Load predictor
    predictor = S2APredictor.from_checkpoint(args.checkpoint)

    # Run prediction
    result_df = predict_from_descriptors_file(
        predictor,
        args.input,
        args.output,
        mode=args.mode,
        calibration_file=args.calibration_file,
        calibration_activity_col=args.calibration_col,
        calibration_n_samples=args.calibration_samples
    )

    print(f"\nPredictions saved to {args.output}")
    print(f"  N samples: {len(result_df)}")
    print(f"  Mode: {args.mode}")

    return 0


def cmd_evaluate(args):
    """Run full leave-one-out evaluation."""
    print("\n" + "="*60)
    print("S2A EVALUATION")
    print("="*60)

    # Parse datasets
    if args.datasets:
        datasets = args.datasets
    elif args.dataset_group:
        if args.dataset_group in S2A_DATASET_GROUPS:
            datasets = list(S2A_DATASET_GROUPS[args.dataset_group])
        else:
            print(f"Unknown dataset group: {args.dataset_group}")
            return 1
    else:
        # Default to a reasonable subset
        datasets = ['K562', 'HepG2', 'WTC11', 'S2_dev',
                   'arabidopsis_leaf', 'sorghum_leaf', 'maize_leaf']

    # Configure
    config = S2AConfig(
        head_type=args.head_type,
        head_alpha=args.alpha
    )

    evaluator = S2AEvaluator(config)

    # Run evaluation
    results = evaluator.run_full_evaluation(
        datasets=datasets,
        train_split=args.train_split,
        test_split=args.test_split,
        calibration_n_samples=args.calibration_samples,
        verbose=True
    )

    # Save results
    results.save(args.output_dir)

    return 0


def cmd_compare(args):
    """Compare different transfer scenarios."""
    print("\n" + "="*60)
    print("S2A TRANSFER SCENARIO COMPARISON")
    print("="*60)

    config = S2AConfig(
        head_type=args.head_type,
        head_alpha=args.alpha
    )

    evaluator = S2AEvaluator(config)
    comparison_df = compare_transfer_scenarios(evaluator, verbose=True)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(output_dir / 'transfer_comparison.csv', index=False)
    print(f"\nResults saved to {output_dir / 'transfer_comparison.csv'}")

    return 0


def cmd_info(args):
    """Show information about available datasets and features."""
    print("\n" + "="*60)
    print("S2A DATASET INFORMATION")
    print("="*60)

    print("\nAvailable Datasets:")
    print("-" * 60)

    for name, config in S2A_DATASETS.items():
        print(f"\n{name}:")
        print(f"  Species: {config.species}")
        print(f"  Kingdom: {config.kingdom}")
        print(f"  Activity: {config.activity_col}")
        print(f"  {config.description}")

    print("\n\nDataset Groups:")
    print("-" * 60)
    for group, datasets in S2A_DATASET_GROUPS.items():
        print(f"\n{group}: {', '.join(sorted(datasets))}")

    # Count features if requested
    if args.count_features:
        print("\n\nFeature Counts:")
        print("-" * 60)
        print(f"{'Dataset':<20} {'Universal':>10} {'Total':>10} {'PWM':>10}")
        print("-" * 60)

        for name in S2A_DATASETS.keys():
            try:
                n_universal, n_total, n_pwm = count_universal_vs_total_features(name)
                print(f"{name:<20} {n_universal:>10} {n_total:>10} {n_pwm:>10}")
            except Exception as e:
                print(f"{name:<20} {'Error':>10} - {str(e)[:30]}")

    return 0


def cmd_calibration_curve(args):
    """Analyze calibration curve for a dataset."""
    print("\n" + "="*60)
    print("S2A CALIBRATION CURVE ANALYSIS")
    print("="*60)

    # Parse datasets
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = ['K562', 'HepG2', 'WTC11', 'S2_dev',
                   'arabidopsis_leaf', 'sorghum_leaf', 'maize_leaf']

    config = S2AConfig(
        head_type=args.head_type,
        head_alpha=args.alpha
    )

    evaluator = S2AEvaluator(config)

    sample_sizes = [10, 20, 50, 100, 200, 500]

    results = evaluator.evaluate_calibration_curve(
        datasets,
        args.holdout,
        sample_sizes=sample_sizes,
        n_repeats=args.n_repeats,
        verbose=True
    )

    # Save results
    import pandas as pd
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for n, metrics in results.items():
        record = {'n_samples': n}
        record.update(metrics)
        records.append(record)

    df = pd.DataFrame(records)
    output_file = output_dir / f'calibration_curve_{args.holdout}.csv'
    df.to_csv(output_file, index=False)

    print(f"\nResults saved to {output_file}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='S2A: Zero-Shot Sequence-to-Activity Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a universal S2A head')
    train_parser.add_argument('--datasets', nargs='+', help='Datasets to train on')
    train_parser.add_argument('--dataset-group', type=str, help='Use predefined dataset group')
    train_parser.add_argument('--holdout', type=str, help='Dataset to hold out for testing')
    train_parser.add_argument('--head-type', type=str, default='ridge',
                             choices=['ridge', 'elastic_net', 'mlp'])
    train_parser.add_argument('--alpha', type=float, default=1.0,
                             help='Regularization strength')
    train_parser.add_argument('--train-split', type=str, default='train')
    train_parser.add_argument('--test-split', type=str, default='test')
    train_parser.add_argument('--output-dir', type=str, default='results/s2a/',
                             help='Output directory for checkpoint')
    train_parser.set_defaults(func=cmd_train)

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run inference on new data')
    predict_parser.add_argument('--checkpoint', type=str, required=True,
                               help='Path to checkpoint directory')
    predict_parser.add_argument('--input', type=str, required=True,
                               help='Input descriptors TSV file')
    predict_parser.add_argument('--output', type=str, required=True,
                               help='Output predictions TSV file')
    predict_parser.add_argument('--mode', type=str, default='zscore',
                               choices=['zscore', 'calibrated', 'ranking'])
    predict_parser.add_argument('--calibration-file', type=str,
                               help='TSV file with calibration samples (for calibrated mode)')
    predict_parser.add_argument('--calibration-col', type=str,
                               help='Activity column name in calibration file')
    predict_parser.add_argument('--calibration-samples', type=int, default=50,
                               help='Number of calibration samples to use')
    predict_parser.set_defaults(func=cmd_predict)

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run leave-one-out evaluation')
    eval_parser.add_argument('--datasets', nargs='+', help='Datasets to evaluate')
    eval_parser.add_argument('--dataset-group', type=str, help='Use predefined dataset group')
    eval_parser.add_argument('--head-type', type=str, default='ridge',
                            choices=['ridge', 'elastic_net', 'mlp'])
    eval_parser.add_argument('--alpha', type=float, default=1.0)
    eval_parser.add_argument('--train-split', type=str, default='train')
    eval_parser.add_argument('--test-split', type=str, default='test')
    eval_parser.add_argument('--calibration-samples', type=int, default=50)
    eval_parser.add_argument('--output-dir', type=str, default='results/s2a/evaluation/',
                            help='Output directory for results')
    eval_parser.set_defaults(func=cmd_evaluate)

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare transfer scenarios')
    compare_parser.add_argument('--head-type', type=str, default='ridge',
                               choices=['ridge', 'elastic_net', 'mlp'])
    compare_parser.add_argument('--alpha', type=float, default=1.0)
    compare_parser.add_argument('--output-dir', type=str, default='results/s2a/comparison/')
    compare_parser.set_defaults(func=cmd_compare)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show dataset information')
    info_parser.add_argument('--count-features', action='store_true',
                            help='Count features in each dataset')
    info_parser.set_defaults(func=cmd_info)

    # Calibration curve command
    cal_parser = subparsers.add_parser('calibration-curve',
                                       help='Analyze calibration curve')
    cal_parser.add_argument('--datasets', nargs='+', help='Source datasets')
    cal_parser.add_argument('--holdout', type=str, required=True,
                           help='Holdout dataset for calibration analysis')
    cal_parser.add_argument('--head-type', type=str, default='ridge')
    cal_parser.add_argument('--alpha', type=float, default=1.0)
    cal_parser.add_argument('--n-repeats', type=int, default=10,
                           help='Number of random repeats per sample size')
    cal_parser.add_argument('--output-dir', type=str, default='results/s2a/calibration/')
    cal_parser.set_defaults(func=cmd_calibration_curve)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
