"""
Transfer Learning Protocols for PhysicsTransfer.

Implements three transfer protocols:
1. Zero-Shot Transfer: Apply source model directly to target
2. Physics-Anchored Fine-Tuning: Fine-tune with frozen physics encoder
3. Multi-Species Joint Training: Shared physics encoder, species-specific heads
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import warnings
import json

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

from .config import TransferConfig, DATASETS, EXPERIMENTS, get_fusemap_root
from .data_loader import PhysicsDataLoader
from .physics_probe import PhysicsActivityProbe, ProbeResults
from .logging_utils import ExperimentLogger, Timer, get_config_dict


@dataclass
class TransferResult:
    """Results from a transfer experiment."""
    protocol: str
    source_datasets: List[str]
    target_dataset: str

    # Source performance (in-domain)
    source_pearson: float
    source_spearman: float

    # Target performance (transfer)
    target_pearson: float
    target_spearman: float

    # Fine-tuning performance (if applicable)
    fine_tuned_pearson: Optional[float] = None
    fine_tuned_spearman: Optional[float] = None
    fine_tune_n_samples: Optional[int] = None

    # Additional metrics
    transfer_efficiency: Optional[float] = None  # target / source ratio
    n_common_features: int = 0
    feature_contributions: Dict[str, float] = field(default_factory=dict)


class ZeroShotTransfer:
    """
    Protocol 1: Physics-Bridge Zero-Shot Transfer.

    Train physics→activity probe on source species, apply directly to target.
    Tests whether physics features capture universal regulatory principles.
    """

    def __init__(self, config: TransferConfig = None, output_dir: str = None):
        self.config = config or TransferConfig()
        self.data_loader = PhysicsDataLoader(self.config)
        self.probe: Optional[PhysicsActivityProbe] = None
        self.common_features: List[str] = []
        self.output_dir = output_dir
        self.logger: Optional[ExperimentLogger] = None

    def run(
        self,
        source_datasets: List[str],
        target_dataset: str,
        source_split: str = 'train',
        target_split: str = 'test'
    ) -> TransferResult:
        """
        Run zero-shot transfer from source to target.

        Args:
            source_datasets: List of source dataset names
            target_dataset: Target dataset name
            source_split: Split to use for source training
            target_split: Split to use for target evaluation

        Returns:
            TransferResult with metrics
        """
        # Initialize logger
        exp_name = f"zero_shot_{'_'.join(source_datasets)}_to_{target_dataset}"
        self.logger = ExperimentLogger(exp_name, self.output_dir)

        # Log hyperparameters
        self.logger.log_hyperparams({
            'protocol': 'zero_shot',
            'source_datasets': source_datasets,
            'target_dataset': target_dataset,
            'source_split': source_split,
            'target_split': target_split,
            'probe_type': self.config.probe_type,
            'probe_alpha': self.config.probe_alpha,
            'n_folds': self.config.n_folds,
            'random_seed': self.config.random_seed
        })

        # Load source data
        self.logger.info("Loading source datasets...")
        with Timer('data_load', self.logger):
            X_source, y_source, source_features, source_labels = self.data_loader.load_multiple_datasets(
                source_datasets, split=source_split
            )

        # Load target data
        self.logger.info("Loading target dataset...")
        _, X_target, y_target, target_features = self.data_loader.load_dataset(
            target_dataset, split=target_split
        )

        # Align features
        self.logger.info("Aligning features...")
        X_source_aligned, X_target_aligned, self.common_features = self.data_loader.align_features(
            X_source, source_features, X_target, target_features
        )

        # Log data info
        self.logger.log_data_info(
            n_train=len(y_source),
            n_test=len(y_target),
            n_features=len(self.common_features),
            feature_names=self.common_features,
            source_datasets=source_datasets,
            target_dataset=target_dataset
        )

        # Train probe on source with CV
        self.logger.info("Training physics probe on source (with CV)...")
        self.probe = PhysicsActivityProbe(self.config)

        with Timer('train', self.logger):
            source_results = self.probe.fit_evaluate_cv(
                X_source_aligned, y_source, self.common_features,
                logger=self.logger
            )

        # Apply to target (zero-shot)
        self.logger.info("Applying to target (zero-shot)...")
        with Timer('eval', self.logger):
            target_results = self.probe.evaluate(X_target_aligned, y_target)

        # Compute transfer efficiency
        transfer_eff = target_results.pearson_r / source_results.pearson_r if source_results.pearson_r > 0 else 0

        # Log transfer results
        self.logger.log_transfer_result(
            source_pearson=source_results.pearson_r,
            target_pearson=target_results.pearson_r,
            transfer_efficiency=transfer_eff
        )

        # Get and log feature contributions
        family_indices = self.data_loader.get_physics_family_indices(self.common_features)
        contributions = self.probe.get_physics_family_contributions(family_indices)

        self.logger.info("Feature family contributions:")
        for family, contrib in sorted(contributions.items(), key=lambda x: -x[1]):
            self.logger.info(f"  {family}: {contrib:.1f}%")

        # Log top features
        if source_results.feature_importances:
            self.logger.log_feature_importance(source_results.feature_importances, top_n=15)

        # Save logs
        self.logger.save()

        return TransferResult(
            protocol='zero_shot',
            source_datasets=source_datasets,
            target_dataset=target_dataset,
            source_pearson=source_results.pearson_r,
            source_spearman=source_results.spearman_r,
            target_pearson=target_results.pearson_r,
            target_spearman=target_results.spearman_r,
            transfer_efficiency=transfer_eff,
            n_common_features=len(self.common_features),
            feature_contributions=contributions
        )

    def get_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions for aligned features."""
        if self.probe is None:
            raise ValueError("Run transfer first.")
        return self.probe.predict(X)


class PhysicsAnchoredFineTuning:
    """
    Protocol 2: Physics-Anchored Fine-Tuning.

    Start with source-trained physics probe, then fine-tune on limited
    target data. Tests few-shot learning with physics anchoring.
    """

    def __init__(self, config: TransferConfig = None, output_dir: str = None):
        self.config = config or TransferConfig()
        self.data_loader = PhysicsDataLoader(self.config)
        self.source_probe: Optional[PhysicsActivityProbe] = None
        self.fine_tuned_probe: Optional[PhysicsActivityProbe] = None
        self.common_features: List[str] = []
        self.output_dir = output_dir
        self.logger: Optional[ExperimentLogger] = None

    def run(
        self,
        source_datasets: List[str],
        target_dataset: str,
        fine_tune_sizes: List[int] = None,
        source_split: str = 'train',
        target_train_split: str = 'train',
        target_test_split: str = 'test'
    ) -> List[TransferResult]:
        """
        Run fine-tuning transfer with varying amounts of target data.

        Args:
            source_datasets: Source dataset names
            target_dataset: Target dataset name
            fine_tune_sizes: List of fine-tuning sample sizes to test
            source_split: Source data split
            target_train_split: Target training split
            target_test_split: Target test split

        Returns:
            List of TransferResults for each fine-tune size
        """
        fine_tune_sizes = fine_tune_sizes or self.config.fine_tune_sizes

        # Initialize logger
        exp_name = f"fine_tune_{'_'.join(source_datasets)}_to_{target_dataset}"
        self.logger = ExperimentLogger(exp_name, self.output_dir)

        # Log hyperparameters
        self.logger.log_hyperparams({
            'protocol': 'physics_anchored_fine_tuning',
            'source_datasets': source_datasets,
            'target_dataset': target_dataset,
            'fine_tune_sizes': fine_tune_sizes,
            'probe_type': self.config.probe_type,
            'probe_alpha': self.config.probe_alpha,
            'n_folds': self.config.n_folds,
            'random_seed': self.config.random_seed
        })

        # Load source data
        self.logger.info("Loading source datasets...")
        with Timer('data_load', self.logger):
            X_source, y_source, source_features, _ = self.data_loader.load_multiple_datasets(
                source_datasets, split=source_split
            )

        # Load target data (both train and test)
        self.logger.info("Loading target datasets...")
        _, X_target_train, y_target_train, target_train_features = self.data_loader.load_dataset(
            target_dataset, split=target_train_split
        )
        _, X_target_test, y_target_test, target_test_features = self.data_loader.load_dataset(
            target_dataset, split=target_test_split
        )

        # Align features across all datasets
        all_features = set(source_features) & set(target_train_features) & set(target_test_features)
        self.common_features = sorted(all_features)
        self.logger.info(f"Common features: {len(self.common_features)}")

        # Align source
        source_idx = [source_features.index(f) for f in self.common_features]
        X_source_aligned = X_source[:, source_idx]

        # Align target train
        train_idx = [target_train_features.index(f) for f in self.common_features]
        X_target_train_aligned = X_target_train[:, train_idx]

        # Align target test
        test_idx = [target_test_features.index(f) for f in self.common_features]
        X_target_test_aligned = X_target_test[:, test_idx]

        # Log data info
        self.logger.log_data_info(
            n_train=len(y_source),
            n_val=len(y_target_train),
            n_test=len(y_target_test),
            n_features=len(self.common_features),
            source_datasets=source_datasets,
            target_dataset=target_dataset
        )

        # Train source probe
        self.logger.info("Training source physics probe...")
        self.source_probe = PhysicsActivityProbe(self.config)
        with Timer('train', self.logger):
            source_results = self.source_probe.fit_evaluate_cv(
                X_source_aligned, y_source, self.common_features,
                logger=self.logger
            )

        # Zero-shot baseline on target test
        zero_shot_results = self.source_probe.evaluate(X_target_test_aligned, y_target_test)
        self.logger.info(f"Zero-shot on target test: Pearson r = {zero_shot_results.pearson_r:.4f}")

        # Fine-tune with different amounts of target data
        results = []
        for n_samples in fine_tune_sizes:
            self.logger.info(f"\n{'='*40}")
            self.logger.info(f"Fine-tuning with {n_samples} samples")
            self.logger.info(f"{'='*40}")

            # Subsample target training data
            X_ft, y_ft, _ = self.data_loader.subsample(
                X_target_train_aligned, y_target_train,
                n_samples, random_state=self.config.random_seed
            )

            # Fine-tune on target data
            self.fine_tuned_probe = PhysicsActivityProbe(self.config)
            ft_train_results = self.fine_tuned_probe.fit_evaluate_cv(
                X_ft, y_ft, self.common_features,
                n_folds=min(5, n_samples // 100) if n_samples >= 500 else 3
            )

            # Evaluate on target test
            ft_test_results = self.fine_tuned_probe.evaluate(X_target_test_aligned, y_target_test)

            improvement = ft_test_results.pearson_r - zero_shot_results.pearson_r
            self.logger.info(f"Fine-tuned test Pearson r: {ft_test_results.pearson_r:.4f} (improvement: {improvement:+.4f})")

            # Compute transfer efficiency
            transfer_eff = ft_test_results.pearson_r / source_results.pearson_r if source_results.pearson_r > 0 else 0

            # Get feature contributions
            family_indices = self.data_loader.get_physics_family_indices(self.common_features)
            contributions = self.fine_tuned_probe.get_physics_family_contributions(family_indices)

            result = TransferResult(
                protocol='physics_anchored_fine_tuning',
                source_datasets=source_datasets,
                target_dataset=target_dataset,
                source_pearson=source_results.pearson_r,
                source_spearman=source_results.spearman_r,
                target_pearson=zero_shot_results.pearson_r,
                target_spearman=zero_shot_results.spearman_r,
                fine_tuned_pearson=ft_test_results.pearson_r,
                fine_tuned_spearman=ft_test_results.spearman_r,
                fine_tune_n_samples=n_samples,
                transfer_efficiency=transfer_eff,
                n_common_features=len(self.common_features),
                feature_contributions=contributions
            )
            results.append(result)

        # Save logs
        self.logger.save()

        return results


class MultiSpeciesJointTraining:
    """
    Protocol 3: Multi-Species Joint Training.

    Train a shared physics encoder on multiple species simultaneously,
    with species-specific output heads. Tests whether shared representation
    improves transfer.
    """

    def __init__(self, config: TransferConfig = None, output_dir: str = None):
        self.config = config or TransferConfig()
        self.data_loader = PhysicsDataLoader(self.config)
        self.shared_probe: Optional[PhysicsActivityProbe] = None
        self.species_heads: Dict[str, PhysicsActivityProbe] = {}
        self.common_features: List[str] = []
        self.output_dir = output_dir
        self.logger: Optional[ExperimentLogger] = None

    def run(
        self,
        datasets: List[str],
        holdout_dataset: str = None,
        split: str = 'train',
        test_split: str = 'test'
    ) -> Dict[str, TransferResult]:
        """
        Run multi-species joint training.

        Args:
            datasets: All datasets to include in training
            holdout_dataset: Optional dataset to hold out for transfer testing
            split: Training split
            test_split: Test split

        Returns:
            Dict mapping dataset name to TransferResult
        """
        # Initialize logger
        exp_name = f"joint_{'_'.join(datasets[:3])}"
        if holdout_dataset:
            exp_name += f"_holdout_{holdout_dataset}"
        self.logger = ExperimentLogger(exp_name, self.output_dir)

        # Log hyperparameters
        self.logger.log_hyperparams({
            'protocol': 'multi_species_joint_training',
            'datasets': datasets,
            'holdout_dataset': holdout_dataset,
            'probe_type': self.config.probe_type,
            'probe_alpha': self.config.probe_alpha,
            'n_folds': self.config.n_folds,
            'random_seed': self.config.random_seed
        })

        # Load all datasets
        self.logger.info("Loading all datasets...")
        all_data = {}
        all_features = None

        with Timer('data_load', self.logger):
            for name in datasets:
                try:
                    _, X, y, features = self.data_loader.load_dataset(name, split=split)
                    all_data[name] = {'X': X, 'y': y, 'features': features}
                    self.logger.info(f"  Loaded {name}: {len(y)} samples, {len(features)} features")

                    if all_features is None:
                        all_features = set(features)
                    else:
                        all_features &= set(features)
                except FileNotFoundError as e:
                    self.logger.warning(f"Could not load {name}: {e}")
                    continue

        self.common_features = sorted(all_features)
        self.logger.info(f"Common features across all datasets: {len(self.common_features)}")

        # Align all datasets
        for name in all_data:
            features = all_data[name]['features']
            idx = [features.index(f) for f in self.common_features]
            all_data[name]['X_aligned'] = all_data[name]['X'][:, idx]

        # Combine for joint training (excluding holdout if specified)
        train_datasets = [d for d in datasets if d != holdout_dataset and d in all_data]

        X_combined = []
        y_combined = []
        labels_combined = []

        for name in train_datasets:
            X_combined.append(all_data[name]['X_aligned'])
            y_combined.append(all_data[name]['y'])
            labels_combined.extend([name] * len(all_data[name]['y']))

        X_combined = np.vstack(X_combined)
        y_combined = np.concatenate(y_combined)

        # Log data info
        self.logger.log_data_info(
            n_train=len(y_combined),
            n_features=len(self.common_features),
            source_datasets=train_datasets
        )

        self.logger.info(f"Joint training on {len(train_datasets)} datasets, {len(y_combined):,} total samples")

        # Train shared probe
        self.logger.info("Training shared physics probe...")
        self.shared_probe = PhysicsActivityProbe(self.config)
        with Timer('train', self.logger):
            shared_results = self.shared_probe.fit_evaluate_cv(
                X_combined, y_combined, self.common_features,
                logger=self.logger
            )

        # Evaluate on each dataset
        self.logger.info("\nEvaluating on each dataset...")
        results = {}

        for name in all_data:
            is_holdout = (name == holdout_dataset)
            self.logger.info(f"\n{'[HOLDOUT] ' if is_holdout else ''}Evaluating on {name}...")

            # Load test split
            try:
                _, X_test, y_test, test_features = self.data_loader.load_dataset(name, split=test_split)
                test_idx = [test_features.index(f) for f in self.common_features if f in test_features]
                common_test = [f for f in self.common_features if f in test_features]

                if len(common_test) < len(self.common_features):
                    self.logger.warning(f"Only {len(common_test)}/{len(self.common_features)} features in test")
                    continue

                X_test_aligned = X_test[:, test_idx]
            except FileNotFoundError:
                # Use train split if no test available
                X_test_aligned = all_data[name]['X_aligned']
                y_test = all_data[name]['y']

            test_results = self.shared_probe.evaluate(X_test_aligned, y_test)

            transfer_eff = test_results.pearson_r / shared_results.pearson_r if shared_results.pearson_r > 0 else 0

            self.logger.info(f"  Pearson r: {test_results.pearson_r:.4f}, Transfer efficiency: {transfer_eff:.1%}")

            # Get feature contributions
            family_indices = self.data_loader.get_physics_family_indices(self.common_features)
            contributions = self.shared_probe.get_physics_family_contributions(family_indices)

            results[name] = TransferResult(
                protocol='multi_species_joint',
                source_datasets=train_datasets,
                target_dataset=name,
                source_pearson=shared_results.pearson_r,
                source_spearman=shared_results.spearman_r,
                target_pearson=test_results.pearson_r,
                target_spearman=test_results.spearman_r,
                transfer_efficiency=transfer_eff,
                n_common_features=len(self.common_features),
                feature_contributions=contributions
            )

        # Save logs
        self.logger.save()

        return results


def run_experiment(
    experiment_name: str,
    config: TransferConfig = None
) -> Dict[str, Any]:
    """
    Run a pre-configured transfer experiment.

    Args:
        experiment_name: Name of experiment from EXPERIMENTS config
        config: Optional TransferConfig override

    Returns:
        Dict with all results from the experiment
    """
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {list(EXPERIMENTS.keys())}")

    exp_config = EXPERIMENTS[experiment_name]
    config = config or TransferConfig()

    print(f"\n{'#'*70}")
    print(f"# Running Experiment: {exp_config.name}")
    print(f"# {exp_config.description}")
    print(f"{'#'*70}")

    results = {
        'experiment': experiment_name,
        'description': exp_config.description,
        'protocols': {}
    }

    # Protocol 1: Zero-Shot
    print("\n" + "="*70)
    print("PROTOCOL 1: Zero-Shot Transfer")
    print("="*70)
    zero_shot = ZeroShotTransfer(config)
    zs_result = zero_shot.run(
        exp_config.source_datasets,
        exp_config.target_dataset
    )
    results['protocols']['zero_shot'] = zs_result

    # Protocol 2: Fine-Tuning
    print("\n" + "="*70)
    print("PROTOCOL 2: Physics-Anchored Fine-Tuning")
    print("="*70)
    fine_tuning = PhysicsAnchoredFineTuning(config)
    ft_results = fine_tuning.run(
        exp_config.source_datasets,
        exp_config.target_dataset,
        fine_tune_sizes=exp_config.fine_tune_sizes
    )
    results['protocols']['fine_tuning'] = ft_results

    # Protocol 3: Joint Training
    print("\n" + "="*70)
    print("PROTOCOL 3: Multi-Species Joint Training")
    print("="*70)
    all_datasets = list(set(exp_config.source_datasets + [exp_config.target_dataset]))
    joint = MultiSpeciesJointTraining(config)
    joint_results = joint.run(
        all_datasets,
        holdout_dataset=exp_config.target_dataset
    )
    results['protocols']['joint_training'] = joint_results

    return results
