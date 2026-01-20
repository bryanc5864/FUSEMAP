#!/usr/bin/env python3
"""
FUSEMAP Training Coordinator

Main entry point for running multi-species training experiments.

Usage:
    # Run a single dataset baseline
    python coordinator.py --config single_celltype --dataset encode4_k562

    # Run multi-cell-type human
    python coordinator.py --config multi_celltype_human

    # Run cross-animal transfer
    python coordinator.py --config cross_animal

    # Run cross-kingdom transfer
    python coordinator.py --config cross_kingdom

    # Run universal foundation model
    python coordinator.py --config universal

    # Resume from checkpoint
    python coordinator.py --config cross_animal --resume checkpoints/checkpoint.pt
"""

import argparse
import os
import sys
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import (
    ConfigurationType,
    ExperimentConfig,
    get_config,
    get_config1_single_celltype,
    get_config2_multi_celltype_human,
    get_config3_cross_animal,
    get_config4_cross_kingdom,
    get_config5_universal,
    get_config5_universal_no_yeast,
    DATASET_CATALOG,
)
from training.trainer import Trainer, MultiPhaseTrainer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FUSEMAP Training Coordinator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Configuration selection
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=[
            "single_celltype",
            "multi_celltype_human",
            "cross_animal",
            "cross_kingdom",
            "universal",
            "universal_no_yeast",
        ],
        help="Configuration type to run",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (required for single_celltype config)",
    )

    # Training overrides
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override sampling temperature",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not specified)",
    )

    # Checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Flags
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode with smaller dataset",
    )

    parser.add_argument(
        "--mse",
        action="store_true",
        help="Use MSE loss instead of Gaussian NLL",
    )

    # Extreme value weighting
    parser.add_argument(
        "--no-extreme-weights",
        action="store_true",
        help="Disable extreme value weighting in loss",
    )

    parser.add_argument(
        "--extreme-alpha",
        type=float,
        default=None,
        help="Extreme weighting alpha (higher = more emphasis on extremes)",
    )

    parser.add_argument(
        "--extreme-beta",
        type=float,
        default=None,
        help="Extreme weighting beta (power for z-score, default 2.0)",
    )

    # Extreme-aware sampling
    parser.add_argument(
        "--no-extreme-sampling",
        action="store_true",
        help="Disable extreme-aware sampling (oversample tail values)",
    )

    parser.add_argument(
        "--sampling-extreme-alpha",
        type=float,
        default=None,
        help="Sampling extreme alpha (controls how much to oversample tails)",
    )

    # Balanced activity sampling
    parser.add_argument(
        "--no-balanced-sampling",
        action="store_true",
        help="Disable balanced activity sampling",
    )

    parser.add_argument(
        "--balanced-bins",
        type=int,
        default=None,
        help="Number of activity bins for balanced sampling (default 10)",
    )

    # CADENCE module selection
    parser.add_argument(
        "--all-modules",
        action="store_true",
        help="Enable all optional CADENCE modules",
    )

    parser.add_argument(
        "--no-rc-stem",
        action="store_true",
        help="Disable RC-equivariant stem",
    )

    parser.add_argument(
        "--cluster-space",
        action="store_true",
        help="Enable ClusterSpace module (dilated convolutions)",
    )

    parser.add_argument(
        "--grammar",
        action="store_true",
        help="Enable Grammar layer (BiGRU with FiLM)",
    )

    parser.add_argument(
        "--micromotif",
        action="store_true",
        help="Enable MicroMotif processor",
    )

    parser.add_argument(
        "--correlator",
        action="store_true",
        help="Enable Motif correlator",
    )

    return parser.parse_args()


def get_experiment_config(args) -> ExperimentConfig:
    """Get experiment configuration based on arguments."""

    use_mse = getattr(args, 'mse', False)

    if args.config == "single_celltype":
        if args.dataset is None:
            raise ValueError("--dataset required for single_celltype config")
        if args.dataset not in DATASET_CATALOG:
            available = ", ".join(DATASET_CATALOG.keys())
            raise ValueError(f"Unknown dataset: {args.dataset}. Available: {available}")
        config = get_config1_single_celltype(args.dataset, use_mse=use_mse)

    elif args.config == "multi_celltype_human":
        config = get_config2_multi_celltype_human()

    elif args.config == "cross_animal":
        config = get_config3_cross_animal()

    elif args.config == "cross_kingdom":
        config = get_config4_cross_kingdom()

    elif args.config == "universal":
        config = get_config5_universal()

    elif args.config == "universal_no_yeast":
        config = get_config5_universal_no_yeast()

    else:
        raise ValueError(f"Unknown config: {args.config}")

    # Apply overrides
    if args.epochs is not None:
        config.training.max_epochs = args.epochs

    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    if args.lr is not None:
        config.training.learning_rate = args.lr

    if args.temperature is not None:
        config.training.sampling_temperature = args.temperature

    if args.no_amp:
        config.training.use_amp = False

    # Extreme value weighting options
    if getattr(args, 'no_extreme_weights', False):
        config.training.use_extreme_weights = False

    if getattr(args, 'extreme_alpha', None) is not None:
        config.training.extreme_alpha = args.extreme_alpha

    if getattr(args, 'extreme_beta', None) is not None:
        config.training.extreme_beta = args.extreme_beta

    # Extreme-aware sampling options
    if getattr(args, 'no_extreme_sampling', False):
        config.training.use_extreme_sampling = False

    if getattr(args, 'sampling_extreme_alpha', None) is not None:
        config.training.sampling_extreme_alpha = args.sampling_extreme_alpha

    # Balanced activity sampling options
    if getattr(args, 'no_balanced_sampling', False):
        config.training.use_balanced_sampling = False

    if getattr(args, 'balanced_bins', None) is not None:
        config.training.balanced_sampling_bins = args.balanced_bins

    # CADENCE module selection
    if getattr(args, 'all_modules', False):
        config.model.use_rc_stem = True
        config.model.use_cluster_space = True
        config.model.use_grammar = True
        config.model.use_micromotif = True
        config.model.use_motif_correlator = True
    else:
        if getattr(args, 'no_rc_stem', False):
            config.model.use_rc_stem = False
        if getattr(args, 'cluster_space', False):
            config.model.use_cluster_space = True
        if getattr(args, 'grammar', False):
            config.model.use_grammar = True
        if getattr(args, 'micromotif', False):
            config.model.use_micromotif = True
        if getattr(args, 'correlator', False):
            config.model.use_motif_correlator = True

    if args.name is not None:
        config.name = args.name
    else:
        # Auto-generate name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.name = f"{config.name}_{timestamp}"

    config.output_dir = args.output_dir
    config.seed = args.seed

    # Debug mode: reduce dataset sizes
    if args.debug:
        config.training.max_epochs = 2
        config.training.samples_per_epoch = 1000
        config.training.log_every_n_steps = 10
        config.name = f"debug_{config.name}"

    return config


def print_config_summary(config: ExperimentConfig):
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("FUSEMAP Training Coordinator")
    print("=" * 60)
    print(f"\nExperiment: {config.name}")
    print(f"Config Type: {config.config_type.value}")
    print(f"\nDatasets ({len(config.datasets)}):")
    for ds in config.datasets:
        info = DATASET_CATALOG.get(ds)
        if info:
            print(f"  - {ds}: {info.sequence_length}bp, {info.num_outputs} outputs, {info.species}")
        else:
            print(f"  - {ds}: (not in catalog)")

    print(f"\nModel:")
    print(f"  - Stem: {config.model.stem_channels}ch, kernel={config.model.stem_kernel_size}")
    print(f"  - Blocks: {config.model.block_channels}")
    print(f"  - Loss: {'MSE' if not config.model.use_uncertainty else 'Gaussian NLL'}")
    print(f"  - Length embedding: {config.model.use_length_embedding}")
    print(f"  - RC stem: {getattr(config.model, 'use_rc_stem', True)}")
    print(f"  - ClusterSpace: {getattr(config.model, 'use_cluster_space', False)}")
    print(f"  - Grammar: {getattr(config.model, 'use_grammar', False)}")
    print(f"  - MicroMotif: {getattr(config.model, 'use_micromotif', False)}")
    print(f"  - Correlator: {getattr(config.model, 'use_motif_correlator', False)}")

    print(f"\nTraining:")
    print(f"  - Max epochs: {config.training.max_epochs}")
    print(f"  - Batch size: {config.training.batch_size}")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Weight decay: {config.training.weight_decay}")
    print(f"  - Scheduler: {config.training.scheduler}")
    if config.training.scheduler == "onecycle":
        init_lr = config.training.learning_rate / config.training.onecycle_div_factor
        print(f"  - OneCycle initial LR: {init_lr:.6f}")
    print(f"  - Early stopping patience: {config.training.patience}")
    print(f"  - Mixed precision: {config.training.use_amp}")

    if config.training_phases:
        print(f"\nTraining Phases ({len(config.training_phases)}):")
        for phase in config.training_phases:
            print(f"  - {phase['name']}: {phase['epochs']} epochs, lr={phase['lr']}")

    print(f"\nOutput: {config.output_dir}/{config.name}")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    # Get configuration
    config = get_experiment_config(args)

    # Print summary
    print_config_summary(config)

    # Set seed
    set_seed(config.seed)

    # Create trainer
    if config.config_type == ConfigurationType.UNIVERSAL and config.training_phases:
        trainer = MultiPhaseTrainer(
            config=config,
            device=args.device,
            resume_from=args.resume,
        )
    else:
        trainer = Trainer(
            config=config,
            device=args.device,
            resume_from=args.resume,
        )

    # Run training
    try:
        results = trainer.train()
        print("\nTraining completed successfully!")
        print(f"Results saved to: {config.output_dir}/{config.name}")
        return 0

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_all_baselines(output_dir: str = "results/baselines"):
    """Run all single cell-type baselines."""
    datasets = [
        "encode4_k562",
        "encode4_hepg2",
        "encode4_wtc11",
        "deepstarr",
        "dream_yeast",
        "jores_arabidopsis",
        "jores_maize",
        "jores_sorghum",
    ]

    results = {}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Running baseline for: {dataset}")
        print(f"{'='*60}\n")

        config = get_config1_single_celltype(dataset)
        config.output_dir = output_dir

        set_seed(config.seed)

        trainer = Trainer(
            config=config,
            device="cuda",
        )

        try:
            result = trainer.train()
            results[dataset] = result
        except Exception as e:
            print(f"Failed on {dataset}: {e}")
            results[dataset] = {"error": str(e)}

    # Save summary
    summary_path = Path(output_dir) / "baseline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAll baselines complete. Summary saved to {summary_path}")
    return results


def run_experiment_suite():
    """Run the complete experiment suite (all 5 configurations)."""
    configs = [
        ("config2", get_config2_multi_celltype_human),
        ("config3", get_config3_cross_animal),
        ("config4", get_config4_cross_kingdom),
        ("config5", get_config5_universal),
    ]

    results = {}

    for name, config_fn in configs:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}\n")

        config = config_fn()
        config.output_dir = f"results/{name}"

        set_seed(config.seed)

        if name == "config5":
            trainer = MultiPhaseTrainer(config=config, device="cuda")
        else:
            trainer = Trainer(config=config, device="cuda")

        try:
            result = trainer.train()
            results[name] = result
        except Exception as e:
            print(f"Failed on {name}: {e}")
            results[name] = {"error": str(e)}

    return results


if __name__ == "__main__":
    sys.exit(main())
