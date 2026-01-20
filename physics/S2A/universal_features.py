"""
Universal Feature Extraction for S2A.

Extracts physics features that are universal across species:
- thermo_*: Thermodynamic properties (DNA chemistry is identical)
- stiff_*: Mechanical stiffness
- bend_*: Bending/curvature
- entropy_*: Sequence complexity
- advanced_*: G4, SIDD, MGW, nucleosome positioning

EXCLUDES:
- pwm_*: Position weight matrix features (species-specific TF binding)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import warnings

from .config import S2AConfig, S2ADatasetConfig, S2A_DATASETS, get_fusemap_root


@dataclass
class UniversalFeatureStats:
    """Statistics for feature normalization."""
    mean: np.ndarray
    std: np.ndarray
    feature_names: List[str]
    n_features: int


class UniversalFeatureExtractor:
    """
    Extract universal physics features from descriptor files.

    Key insight: Physics features are universal because DNA chemistry is
    identical across organisms. PWM features are species-specific because
    transcription factors evolve differently.
    """

    def __init__(self, config: S2AConfig = None):
        """
        Initialize feature extractor.

        Args:
            config: S2AConfig with feature settings
        """
        self.config = config or S2AConfig()
        self.scaler = StandardScaler()
        self._is_fitted = False
        self.feature_names: List[str] = []
        self.stats: Optional[UniversalFeatureStats] = None

    def _find_data_file(
        self,
        dataset_config: S2ADatasetConfig,
        split: str
    ) -> Optional[str]:
        """Find the data file for a dataset and split."""
        root = get_fusemap_root()

        # Try primary pattern
        primary_path = dataset_config.get_data_path(split)
        if primary_path.exists():
            return str(primary_path)

        # Try alternative patterns
        alt_patterns = [
            root / dataset_config.data_dir / f'{dataset_config.name}_{split}_with_features.tsv',
            root / dataset_config.data_dir / f'{dataset_config.name.lower()}_{split}_with_features.tsv',
            root / 'physics/output' / f'{dataset_config.name}_{split}_descriptors_with_activity.tsv',
        ]

        for path in alt_patterns:
            if path.exists():
                return str(path)

        return None

    def _is_universal_feature(self, col_name: str) -> bool:
        """Check if a column is a universal physics feature."""
        # Check if it matches any universal prefix
        is_universal = any(
            col_name.startswith(prefix)
            for prefix in self.config.universal_prefixes
        )

        # Check if it should be excluded
        is_excluded = any(
            col_name.startswith(prefix)
            for prefix in self.config.excluded_prefixes
        )

        return is_universal and not is_excluded

    def extract_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Extract list of universal feature columns from a dataframe.

        Args:
            df: DataFrame with physics features

        Returns:
            List of universal feature column names
        """
        feature_cols = [
            col for col in df.columns
            if self._is_universal_feature(col)
        ]
        return sorted(feature_cols)

    def load_dataset_features(
        self,
        dataset_name: str,
        split: str = 'train',
        return_activity: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Load universal features from a dataset.

        Args:
            dataset_name: Name of dataset (e.g., 'K562', 'arabidopsis_leaf')
            split: Data split ('train', 'val', 'test')
            return_activity: Whether to return activity values

        Returns:
            Tuple of (X_features, y_activity or None, feature_names)
        """
        if dataset_name not in S2A_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {list(S2A_DATASETS.keys())}")

        dataset_config = S2A_DATASETS[dataset_name]

        # Find data file
        data_path = self._find_data_file(dataset_config, split)
        if data_path is None:
            raise FileNotFoundError(
                f"Could not find data file for {dataset_name} {split}"
            )

        print(f"Loading {dataset_name} {split} from {data_path}")

        # Load dataframe
        df = pd.read_csv(data_path, sep='\t')
        print(f"  Loaded {len(df)} sequences, {len(df.columns)} columns")

        # Extract universal features
        feature_cols = self.extract_feature_columns(df)
        print(f"  Found {len(feature_cols)} universal features")

        X = df[feature_cols].astype(np.float32).values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Get activity if requested
        y = None
        if return_activity:
            activity_col = dataset_config.activity_col
            if activity_col in df.columns:
                y = df[activity_col].values.astype(np.float32)

                # Remove samples with NaN activity
                valid_mask = ~np.isnan(y)
                if not valid_mask.all():
                    n_invalid = (~valid_mask).sum()
                    print(f"  Removing {n_invalid} samples with invalid activity")
                    X = X[valid_mask]
                    y = y[valid_mask]
            else:
                warnings.warn(f"Activity column '{activity_col}' not found in {data_path}")

        return X, y, feature_cols

    def find_common_features(
        self,
        datasets: List[str],
        split: str = 'train'
    ) -> List[str]:
        """
        Find features common to all specified datasets.

        Args:
            datasets: List of dataset names
            split: Data split to check

        Returns:
            List of common feature names (sorted)
        """
        common_features: Optional[Set[str]] = None

        for dataset_name in datasets:
            _, _, feature_names = self.load_dataset_features(
                dataset_name, split, return_activity=False
            )

            if common_features is None:
                common_features = set(feature_names)
            else:
                common_features &= set(feature_names)

        if common_features is None:
            return []

        common_list = sorted(common_features)
        print(f"Common features across {len(datasets)} datasets: {len(common_list)}")
        return common_list

    def load_and_align_datasets(
        self,
        datasets: List[str],
        split: str = 'train',
        z_score_per_dataset: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Load multiple datasets and align to common features.

        Args:
            datasets: List of dataset names
            split: Data split
            z_score_per_dataset: Whether to z-score normalize activity per dataset

        Returns:
            Tuple of (X_combined, y_combined, feature_names, dataset_labels)
        """
        # First find common features
        common_features = self.find_common_features(datasets, split)

        if len(common_features) == 0:
            raise ValueError("No common features found across datasets!")

        all_X = []
        all_y = []
        all_labels = []

        for dataset_name in datasets:
            X, y, feature_names = self.load_dataset_features(
                dataset_name, split, return_activity=True
            )

            if y is None:
                warnings.warn(f"Skipping {dataset_name}: no activity values")
                continue

            # Align to common features
            feature_idx = [feature_names.index(f) for f in common_features]
            X_aligned = X[:, feature_idx]

            # Z-score normalize activity per dataset
            if z_score_per_dataset:
                y_mean = np.mean(y)
                y_std = np.std(y)
                if y_std > 1e-8:
                    y = (y - y_mean) / y_std
                else:
                    y = y - y_mean

            all_X.append(X_aligned)
            all_y.append(y)
            all_labels.extend([dataset_name] * len(y))

        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)

        print(f"Combined: {X_combined.shape[0]} samples, "
              f"{X_combined.shape[1]} features")

        return X_combined, y_combined, common_features, all_labels

    def fit(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> 'UniversalFeatureExtractor':
        """
        Fit the feature scaler.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names

        Returns:
            Self for chaining
        """
        self.scaler.fit(X)
        self.feature_names = feature_names
        self._is_fitted = True

        self.stats = UniversalFeatureStats(
            mean=self.scaler.mean_.astype(np.float32),
            std=self.scaler.scale_.astype(np.float32),
            feature_names=feature_names,
            n_features=len(feature_names)
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.

        Args:
            X: Feature matrix

        Returns:
            Scaled feature matrix
        """
        if not self._is_fitted:
            raise ValueError("Extractor not fitted. Call fit() first.")

        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Fit and transform features.

        Args:
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            Scaled feature matrix
        """
        self.fit(X, feature_names)
        return self.transform(X)

    def get_feature_family_indices(self) -> Dict[str, List[int]]:
        """
        Get indices of features by family (thermo, stiff, etc.).

        Returns:
            Dict mapping family prefix to list of feature indices
        """
        if not self._is_fitted:
            raise ValueError("Extractor not fitted. Call fit() first.")

        families = {}
        for prefix in self.config.universal_prefixes:
            prefix_clean = prefix.rstrip('_')
            indices = [
                i for i, name in enumerate(self.feature_names)
                if name.startswith(prefix)
            ]
            if indices:
                families[prefix_clean] = indices

        return families

    def count_features_by_family(self) -> Dict[str, int]:
        """
        Count features by family.

        Returns:
            Dict mapping family name to feature count
        """
        families = self.get_feature_family_indices()
        return {k: len(v) for k, v in families.items()}

    def save_stats(self, filepath: str):
        """Save feature statistics to file."""
        if self.stats is None:
            raise ValueError("No stats to save. Call fit() first.")

        import json
        data = {
            'mean': self.stats.mean.tolist(),
            'std': self.stats.std.tolist(),
            'feature_names': self.stats.feature_names,
            'n_features': self.stats.n_features
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_stats(self, filepath: str):
        """Load feature statistics from file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.stats = UniversalFeatureStats(
            mean=np.array(data['mean'], dtype=np.float32),
            std=np.array(data['std'], dtype=np.float32),
            feature_names=data['feature_names'],
            n_features=data['n_features']
        )

        # Update scaler
        self.scaler.mean_ = self.stats.mean
        self.scaler.scale_ = self.stats.std
        self.scaler.var_ = self.stats.std ** 2
        self.scaler.n_features_in_ = self.stats.n_features
        self.feature_names = self.stats.feature_names
        self._is_fitted = True


def count_universal_vs_total_features(
    dataset_name: str,
    split: str = 'train',
    config: S2AConfig = None
) -> Tuple[int, int, int]:
    """
    Count universal vs total vs PWM features in a dataset.

    Args:
        dataset_name: Dataset name
        split: Data split
        config: S2AConfig

    Returns:
        Tuple of (n_universal, n_total, n_pwm)
    """
    config = config or S2AConfig()
    extractor = UniversalFeatureExtractor(config)

    if dataset_name not in S2A_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_config = S2A_DATASETS[dataset_name]
    data_path = extractor._find_data_file(dataset_config, split)

    if data_path is None:
        raise FileNotFoundError(f"Data file not found for {dataset_name}")

    df = pd.read_csv(data_path, sep='\t')

    # Count universal features
    universal_cols = extractor.extract_feature_columns(df)
    n_universal = len(universal_cols)

    # Count PWM features
    pwm_cols = [col for col in df.columns if col.startswith('pwm_')]
    n_pwm = len(pwm_cols)

    # Count total physics-like features
    all_physics_prefixes = list(config.universal_prefixes) + list(config.excluded_prefixes)
    total_cols = [
        col for col in df.columns
        if any(col.startswith(p) for p in all_physics_prefixes)
    ]
    n_total = len(total_cols)

    return n_universal, n_total, n_pwm
