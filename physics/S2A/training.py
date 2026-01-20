"""
Training module for Universal S2A Head.

Handles:
- Multi-species data loading and alignment
- Per-dataset z-score normalization
- Universal head training
- Leave-one-out training protocol
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import time

from .config import S2AConfig, S2A_DATASETS, get_fusemap_root
from .universal_features import UniversalFeatureExtractor
from .universal_head import UniversalS2AHead, EnsembleS2AHead, HeadEvaluationResults


@dataclass
class TrainingResults:
    """Results from training a universal S2A head."""
    # Training info
    source_datasets: List[str]
    n_source_samples: int
    n_features: int
    head_type: str

    # Training metrics
    train_spearman: float
    train_pearson: float
    train_r2: float

    # Timing
    fit_time_seconds: float

    def to_dict(self) -> Dict:
        """Convert to dictionary with JSON-serializable types."""
        d = asdict(self)
        # Convert numpy types to Python types
        for k, v in d.items():
            if hasattr(v, 'item'):  # numpy scalar
                d[k] = v.item()
        return d


@dataclass
class LeaveOneOutResults:
    """Results from leave-one-out evaluation."""
    holdout_dataset: str
    source_datasets: List[str]

    # Zero-shot performance
    zeroshot_spearman: float
    zeroshot_pearson: float
    zeroshot_r2: float
    zeroshot_mse: float

    # Test set info
    n_test_samples: int

    def to_dict(self) -> Dict:
        """Convert to dictionary with JSON-serializable types."""
        d = asdict(self)
        # Convert numpy types to Python types
        for k, v in d.items():
            if hasattr(v, 'item'):  # numpy scalar
                d[k] = v.item()
        return d


class UniversalS2ATrainer:
    """
    Trainer for Universal S2A Head.

    Handles multi-species data loading, z-score normalization,
    feature alignment, and head training.
    """

    def __init__(self, config: S2AConfig = None):
        """
        Initialize trainer.

        Args:
            config: S2AConfig with training parameters
        """
        self.config = config or S2AConfig()
        self.feature_extractor = UniversalFeatureExtractor(config)

        # Cached data
        self._datasets_cache: Dict[str, Dict] = {}
        self._common_features: Optional[List[str]] = None

    def load_datasets(
        self,
        dataset_names: List[str],
        split: str = 'train',
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Load multiple datasets and cache them.

        Args:
            dataset_names: List of dataset names to load
            split: Data split
            verbose: Print progress

        Returns:
            Dict mapping dataset_name to {'X': features, 'y': activity, 'features': names}
        """
        datasets = {}

        for name in dataset_names:
            if name in self._datasets_cache and split in self._datasets_cache.get(name, {}):
                if verbose:
                    print(f"Using cached {name} {split}")
                datasets[name] = self._datasets_cache[name][split]
                continue

            try:
                X, y, feature_names = self.feature_extractor.load_dataset_features(
                    name, split, return_activity=True
                )

                datasets[name] = {
                    'X': X,
                    'y': y,
                    'features': feature_names,
                    'n_samples': len(y) if y is not None else len(X)
                }

                # Cache
                if name not in self._datasets_cache:
                    self._datasets_cache[name] = {}
                self._datasets_cache[name][split] = datasets[name]

            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load {name}: {e}")

        return datasets

    def prepare_training_data(
        self,
        source_datasets: List[str],
        split: str = 'train',
        z_score_per_dataset: bool = True,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Prepare training data from multiple source datasets.

        Args:
            source_datasets: List of source dataset names
            split: Data split
            z_score_per_dataset: Z-score normalize activity per dataset
            verbose: Print progress

        Returns:
            Tuple of (X_combined, y_combined, feature_names, dataset_labels)
        """
        if verbose:
            print(f"\n=== Loading {len(source_datasets)} source datasets ===")

        datasets = self.load_datasets(source_datasets, split, verbose)

        if len(datasets) == 0:
            raise ValueError("No datasets could be loaded!")

        # Find common features
        common_features = None
        for name, data in datasets.items():
            if common_features is None:
                common_features = set(data['features'])
            else:
                common_features &= set(data['features'])

        common_features = sorted(common_features)
        self._common_features = common_features

        if verbose:
            print(f"\nCommon features: {len(common_features)}")

        # Combine datasets
        all_X = []
        all_y = []
        all_labels = []

        for name, data in datasets.items():
            X = data['X']
            y = data['y']
            features = data['features']

            if y is None:
                if verbose:
                    print(f"  Skipping {name}: no activity values")
                continue

            # Align to common features
            feature_idx = [features.index(f) for f in common_features]
            X_aligned = X[:, feature_idx]

            # Z-score normalize activity
            if z_score_per_dataset:
                y_mean = np.mean(y)
                y_std = np.std(y)
                if y_std > 1e-8:
                    y_normalized = (y - y_mean) / y_std
                else:
                    y_normalized = y - y_mean
            else:
                y_normalized = y

            all_X.append(X_aligned)
            all_y.append(y_normalized)
            all_labels.extend([name] * len(y))

            if verbose:
                print(f"  {name}: {len(y)} samples (y_mean={np.mean(y):.3f}, "
                      f"y_std={np.std(y):.3f})")

        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)

        if verbose:
            print(f"\nCombined: {len(y_combined)} samples, {len(common_features)} features")

        return X_combined, y_combined, common_features, all_labels

    def train(
        self,
        source_datasets: List[str],
        split: str = 'train',
        verbose: bool = True
    ) -> Tuple[UniversalS2AHead, UniversalFeatureExtractor, TrainingResults]:
        """
        Train a universal S2A head on multiple source datasets.

        Args:
            source_datasets: List of source dataset names
            split: Data split to use
            verbose: Print progress

        Returns:
            Tuple of (fitted_head, fitted_extractor, training_results)
        """
        start_time = time.time()

        # Prepare data
        X, y, features, labels = self.prepare_training_data(
            source_datasets, split, z_score_per_dataset=True, verbose=verbose
        )

        # Fit feature scaler
        if verbose:
            print("\nFitting feature extractor...")
        self.feature_extractor.fit(X, features)

        # Scale features
        X_scaled = self.feature_extractor.transform(X)

        # Train head
        if verbose:
            print(f"\nTraining {self.config.head_type} head...")

        head = UniversalS2AHead(self.config)
        head.fit(X_scaled, y, features)

        # Evaluate on training data
        train_results = head.evaluate(X_scaled, y)

        fit_time = time.time() - start_time

        if verbose:
            print(f"\nTraining Results:")
            print(f"  Spearman ρ: {train_results.spearman_rho:.4f}")
            print(f"  Pearson r: {train_results.pearson_r:.4f}")
            print(f"  R²: {train_results.r2:.4f}")
            print(f"  Fit time: {fit_time:.1f}s")

        results = TrainingResults(
            source_datasets=source_datasets,
            n_source_samples=len(y),
            n_features=len(features),
            head_type=self.config.head_type,
            train_spearman=train_results.spearman_rho,
            train_pearson=train_results.pearson_r,
            train_r2=train_results.r2,
            fit_time_seconds=fit_time
        )

        return head, self.feature_extractor, results

    def train_ensemble(
        self,
        source_datasets: List[str],
        split: str = 'train',
        head_types: List[str] = None,
        verbose: bool = True
    ) -> Tuple[EnsembleS2AHead, UniversalFeatureExtractor, TrainingResults]:
        """
        Train an ensemble of S2A heads.

        Args:
            source_datasets: List of source dataset names
            split: Data split
            head_types: Types of heads for ensemble
            verbose: Print progress

        Returns:
            Tuple of (fitted_ensemble, fitted_extractor, training_results)
        """
        head_types = head_types or ['ridge', 'elastic_net']
        start_time = time.time()

        # Prepare data
        X, y, features, labels = self.prepare_training_data(
            source_datasets, split, z_score_per_dataset=True, verbose=verbose
        )

        # Fit feature scaler
        self.feature_extractor.fit(X, features)
        X_scaled = self.feature_extractor.transform(X)

        # Train ensemble
        if verbose:
            print(f"\nTraining ensemble with: {head_types}")

        ensemble = EnsembleS2AHead(self.config)
        ensemble.fit(X_scaled, y, features, head_types=head_types)

        # Evaluate
        train_results = ensemble.evaluate(X_scaled, y)
        fit_time = time.time() - start_time

        if verbose:
            print(f"\nEnsemble Training Results:")
            print(f"  Spearman ρ: {train_results.spearman_rho:.4f}")
            print(f"  Pearson r: {train_results.pearson_r:.4f}")
            print(f"  Weights: {ensemble.weights}")

        results = TrainingResults(
            source_datasets=source_datasets,
            n_source_samples=len(y),
            n_features=len(features),
            head_type='ensemble_' + '_'.join(head_types),
            train_spearman=train_results.spearman_rho,
            train_pearson=train_results.pearson_r,
            train_r2=train_results.r2,
            fit_time_seconds=fit_time
        )

        return ensemble, self.feature_extractor, results

    def train_leave_one_out(
        self,
        all_datasets: List[str],
        holdout: str,
        train_split: str = 'train',
        test_split: str = 'test',
        verbose: bool = True
    ) -> Tuple[UniversalS2AHead, UniversalFeatureExtractor, LeaveOneOutResults]:
        """
        Train with leave-one-out: train on all except holdout, test on holdout.

        Args:
            all_datasets: All dataset names
            holdout: Dataset to hold out for testing
            train_split: Split to use for training
            test_split: Split to use for testing holdout
            verbose: Print progress

        Returns:
            Tuple of (head, extractor, leave_one_out_results)
        """
        if holdout not in all_datasets:
            raise ValueError(f"Holdout '{holdout}' not in dataset list")

        source_datasets = [d for d in all_datasets if d != holdout]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Leave-One-Out: Holdout = {holdout}")
            print(f"Sources: {source_datasets}")
            print(f"{'='*60}")

        # Train on sources
        head, extractor, train_results = self.train(
            source_datasets, train_split, verbose=verbose
        )

        # Load holdout test data
        if verbose:
            print(f"\nLoading holdout test data: {holdout} {test_split}")

        X_test, y_test, test_features = extractor.load_dataset_features(
            holdout, test_split, return_activity=True
        )

        if y_test is None:
            raise ValueError(f"No activity values for holdout {holdout}")

        # Align features
        common_features = extractor.feature_names
        test_feature_idx = [test_features.index(f) for f in common_features]
        X_test_aligned = X_test[:, test_feature_idx]

        # Scale
        X_test_scaled = extractor.transform(X_test_aligned)

        # Evaluate zero-shot
        test_results = head.evaluate(X_test_scaled, y_test)

        if verbose:
            print(f"\nZero-Shot Results on {holdout}:")
            print(f"  Spearman ρ: {test_results.spearman_rho:.4f}")
            print(f"  Pearson r: {test_results.pearson_r:.4f}")
            print(f"  R²: {test_results.r2:.4f}")
            print(f"  MSE: {test_results.mse:.4f}")
            print(f"  N samples: {test_results.n_samples}")

        results = LeaveOneOutResults(
            holdout_dataset=holdout,
            source_datasets=source_datasets,
            zeroshot_spearman=test_results.spearman_rho,
            zeroshot_pearson=test_results.pearson_r,
            zeroshot_r2=test_results.r2,
            zeroshot_mse=test_results.mse,
            n_test_samples=test_results.n_samples
        )

        return head, extractor, results

    def save_checkpoint(
        self,
        head: UniversalS2AHead,
        extractor: UniversalFeatureExtractor,
        results: TrainingResults,
        output_dir: str,
        name: str = 'universal_s2a'
    ):
        """
        Save trained model checkpoint.

        Args:
            head: Fitted S2A head
            extractor: Fitted feature extractor
            results: Training results
            output_dir: Output directory
            name: Checkpoint name prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save head
        head_path = output_dir / f'{name}_head.pkl'
        head.save(str(head_path))

        # Save extractor stats
        extractor_path = output_dir / f'{name}_extractor.json'
        extractor.save_stats(str(extractor_path))

        # Save results
        results_path = output_dir / f'{name}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)

        print(f"Saved checkpoint to {output_dir}")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: str,
        name: str = 'universal_s2a',
        config: S2AConfig = None
    ) -> Tuple[UniversalS2AHead, UniversalFeatureExtractor]:
        """
        Load a saved checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint
            name: Checkpoint name prefix
            config: Optional config override

        Returns:
            Tuple of (head, extractor)
        """
        checkpoint_dir = Path(checkpoint_dir)

        # Load head
        head_path = checkpoint_dir / f'{name}_head.pkl'
        head = UniversalS2AHead.load(str(head_path))

        # Load extractor
        config = config or S2AConfig()
        extractor = UniversalFeatureExtractor(config)
        extractor_path = checkpoint_dir / f'{name}_extractor.json'
        extractor.load_stats(str(extractor_path))

        return head, extractor
