"""
Data loader for PhysicsTransfer experiments.

Loads physics features, activity values, and optional electrostatics
from pre-computed descriptor files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

from .config import DatasetConfig, TransferConfig, DATASETS, get_fusemap_root


class PhysicsDataLoader:
    """
    Load and prepare physics features for transfer learning experiments.

    Handles:
    - Loading physics features from descriptor files
    - Filtering to physics-only features (excluding PWM if specified)
    - Loading electrostatics features from TileFormer
    - Standardization and missing value handling
    """

    def __init__(self, config: TransferConfig = None):
        """
        Initialize data loader.

        Args:
            config: TransferConfig with feature settings
        """
        self.config = config or TransferConfig()
        self.root = get_fusemap_root()
        self._feature_stats: Dict[str, Dict] = {}  # For standardization

    def load_dataset(
        self,
        dataset_name: str,
        split: str = 'train',
        include_pwm: bool = None
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
        """
        Load a dataset with physics features and activity.

        Args:
            dataset_name: Name of dataset (e.g., 'K562', 'S2_dev')
            split: Data split ('train', 'val', 'test')
            include_pwm: Whether to include PWM features (overrides config)

        Returns:
            Tuple of (full_df, X_physics, y_activity, feature_names)
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

        dataset_config = DATASETS[dataset_name]
        include_pwm = include_pwm if include_pwm is not None else self.config.include_pwm_in_transfer

        # Find the data file
        data_path = self._find_data_file(dataset_config, split)
        if data_path is None:
            raise FileNotFoundError(f"Could not find data file for {dataset_name} {split}")

        print(f"Loading {dataset_name} {split} from {data_path.name}")

        # Load dataframe
        df = pd.read_csv(data_path, sep='\t')
        print(f"  Loaded {len(df)} sequences, {len(df.columns)} columns")

        # Get activity column
        activity_col = dataset_config.activity_col
        if activity_col not in df.columns:
            # Try alternative names
            alt_cols = ['activity', 'Activity', 'expression', 'log2_enrichment']
            for alt in alt_cols:
                if alt in df.columns:
                    activity_col = alt
                    break
            else:
                raise ValueError(f"Activity column '{dataset_config.activity_col}' not found in {data_path}")

        y = df[activity_col].values

        # Extract physics features
        X, feature_names = self._extract_physics_features(df, include_pwm)
        print(f"  Extracted {len(feature_names)} physics features")

        # Handle missing values
        X, y, valid_mask = self._handle_missing(X, y)
        n_removed = (~valid_mask).sum()
        if n_removed > 0:
            print(f"  Removed {n_removed} samples with missing activity values")
            df = df[valid_mask].reset_index(drop=True)

        return df, X, y, feature_names

    def _find_data_file(self, config: DatasetConfig, split: str) -> Optional[Path]:
        """Find the data file using various patterns."""
        # Try patterns in order of preference
        patterns = []

        # Pattern from config
        cell_type = config.cell_types[0] if config.cell_types else config.name
        patterns.append(self.root / config.data_dir / config.file_pattern.format(
            cell_type=cell_type, split=split
        ))

        # Alternative patterns
        patterns.extend([
            self.root / config.data_dir / f'{cell_type}_{split}_with_features.tsv',
            self.root / config.data_dir / f'{config.name}_{split}_descriptors_with_activity.tsv',
            self.root / config.data_dir / f'{config.name.lower()}_{split}_descriptors_with_activity.tsv',
            self.root / 'physics/output' / f'{config.name}_{split}_descriptors_with_activity.tsv',
            self.root / 'physics/output' / f'{cell_type}_{split}_descriptors_with_activity.tsv',
        ])

        for path in patterns:
            if path.exists():
                return path

        return None

    def _extract_physics_features(
        self,
        df: pd.DataFrame,
        include_pwm: bool
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract physics feature columns from dataframe."""
        physics_prefixes = self.config.get_physics_feature_prefixes()

        feature_cols = []
        for col in df.columns:
            # Check physics families
            for prefix in physics_prefixes:
                if col.startswith(f'{prefix}_'):
                    feature_cols.append(col)
                    break
            else:
                # Check for PWM features
                if include_pwm and col.startswith('pwm_'):
                    feature_cols.append(col)

        # Sort for consistency
        feature_cols = sorted(set(feature_cols))

        # Extract and convert to float
        X = df[feature_cols].astype(float).values

        return X, feature_cols

    def _handle_missing(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Handle missing values in features and activity."""
        # First, remove samples with NaN in activity
        valid_y = ~np.isnan(y)
        X = X[valid_y]
        y = y[valid_y]

        # For features: replace NaN with column mean (or 0 if all NaN)
        X_clean = X.copy()
        for j in range(X_clean.shape[1]):
            col = X_clean[:, j]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                col_mean = np.nanmean(col)
                if np.isnan(col_mean):  # All values are NaN
                    col_mean = 0.0
                col[nan_mask] = col_mean
                X_clean[:, j] = col

        # Create valid mask relative to original data
        # (we've already filtered by valid_y)
        valid_mask = valid_y

        return X_clean, y, valid_mask

    def standardize(
        self,
        X: np.ndarray,
        feature_names: List[str],
        fit: bool = True,
        dataset_name: str = 'default'
    ) -> np.ndarray:
        """
        Standardize features to zero mean and unit variance.

        Args:
            X: Feature matrix
            feature_names: Feature names (for storing stats)
            fit: Whether to fit statistics (True for train, False for test)
            dataset_name: Name for storing/retrieving statistics

        Returns:
            Standardized feature matrix
        """
        if fit:
            # Compute and store statistics
            means = np.nanmean(X, axis=0)
            stds = np.nanstd(X, axis=0)
            stds[stds == 0] = 1.0  # Prevent division by zero

            self._feature_stats[dataset_name] = {
                'means': means,
                'stds': stds,
                'feature_names': feature_names
            }
        else:
            # Use stored statistics
            if dataset_name not in self._feature_stats:
                raise ValueError(f"No statistics stored for {dataset_name}. Call with fit=True first.")
            means = self._feature_stats[dataset_name]['means']
            stds = self._feature_stats[dataset_name]['stds']

        X_std = (X - means) / stds
        return X_std

    def align_features(
        self,
        X_source: np.ndarray,
        features_source: List[str],
        X_target: np.ndarray,
        features_target: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Align features between source and target datasets.

        Takes intersection of features present in both datasets.

        Args:
            X_source: Source feature matrix
            features_source: Source feature names
            X_target: Target feature matrix
            features_target: Target feature names

        Returns:
            Tuple of (aligned_source, aligned_target, common_features)
        """
        # Find common features
        common = sorted(set(features_source) & set(features_target))
        print(f"  Common features: {len(common)} / source:{len(features_source)} / target:{len(features_target)}")

        if len(common) == 0:
            raise ValueError("No common features between source and target!")

        # Get indices
        source_idx = [features_source.index(f) for f in common]
        target_idx = [features_target.index(f) for f in common]

        X_source_aligned = X_source[:, source_idx]
        X_target_aligned = X_target[:, target_idx]

        return X_source_aligned, X_target_aligned, common

    def load_multiple_datasets(
        self,
        dataset_names: List[str],
        split: str = 'train',
        include_pwm: bool = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Load and concatenate multiple datasets.

        Args:
            dataset_names: List of dataset names
            split: Data split
            include_pwm: Whether to include PWM features

        Returns:
            Tuple of (X_combined, y_combined, feature_names, dataset_labels)
        """
        all_X = []
        all_y = []
        all_labels = []
        common_features = None

        for name in dataset_names:
            _, X, y, features = self.load_dataset(name, split, include_pwm)

            if common_features is None:
                common_features = set(features)
            else:
                common_features &= set(features)

            all_X.append((X, features))
            all_y.append(y)
            all_labels.extend([name] * len(y))

        # Align to common features
        common_features = sorted(common_features)
        print(f"Combined datasets: {len(common_features)} common features")

        aligned_X = []
        for X, features in all_X:
            idx = [features.index(f) for f in common_features]
            aligned_X.append(X[:, idx])

        X_combined = np.vstack(aligned_X)
        y_combined = np.concatenate(all_y)

        return X_combined, y_combined, common_features, all_labels

    def get_physics_family_indices(
        self,
        feature_names: List[str]
    ) -> Dict[str, List[int]]:
        """
        Get indices of features belonging to each physics family.

        Args:
            feature_names: List of feature names

        Returns:
            Dict mapping family name to list of feature indices
        """
        families = {}
        for prefix in self.config.physics_families + [self.config.electrostatics_prefix]:
            indices = [i for i, f in enumerate(feature_names) if f.startswith(f'{prefix}_')]
            if indices:
                families[prefix] = indices

        return families

    def subsample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        random_state: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Subsample dataset to n_samples.

        Args:
            X: Feature matrix
            y: Activity values
            n_samples: Number of samples to keep
            random_state: Random seed

        Returns:
            Tuple of (X_sub, y_sub, indices)
        """
        if n_samples >= len(y):
            return X, y, np.arange(len(y))

        rng = np.random.RandomState(random_state or self.config.random_seed)
        indices = rng.choice(len(y), size=n_samples, replace=False)
        indices = np.sort(indices)

        return X[indices], y[indices], indices
