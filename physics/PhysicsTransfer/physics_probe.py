"""
Physics-to-Activity Probe Models.

Simple probe models that map physics features to expression activity.
These are intentionally simple (linear or shallow MLP) to test whether
physics features capture transferable regulatory information.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from .config import TransferConfig
from .logging_utils import ExperimentLogger, Timer


@dataclass
class ProbeResults:
    """Results from fitting a physics probe."""
    pearson_r: float
    spearman_r: float
    r2: float
    mse: float
    feature_importances: Optional[Dict[str, float]] = None
    cv_predictions: Optional[np.ndarray] = None
    model: Any = None


class PhysicsActivityProbe:
    """
    Probe model mapping physics features to expression activity.

    This is the core component for transfer learning - if physics features
    have predictive power that transfers across species, this probe should
    maintain performance when applied to new species.
    """

    def __init__(self, config: TransferConfig = None):
        """
        Initialize physics probe.

        Args:
            config: TransferConfig with model parameters
        """
        self.config = config or TransferConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._is_fitted = False

    def _create_model(self):
        """Create the probe model based on config."""
        probe_type = self.config.probe_type

        if probe_type == 'elastic_net':
            return ElasticNet(
                alpha=self.config.probe_alpha,
                l1_ratio=self.config.probe_l1_ratio,
                max_iter=10000,
                random_state=self.config.random_seed
            )
        elif probe_type == 'ridge':
            return Ridge(
                alpha=self.config.probe_alpha,
                random_state=self.config.random_seed
            )
        elif probe_type == 'lasso':
            return Lasso(
                alpha=self.config.probe_alpha,
                max_iter=10000,
                random_state=self.config.random_seed
            )
        elif probe_type == 'mlp':
            return MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                alpha=self.config.probe_alpha,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.config.random_seed
            )
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None
    ) -> 'PhysicsActivityProbe':
        """
        Fit the probe model.

        Args:
            X: Physics features (n_samples, n_features)
            y: Activity values (n_samples,)
            feature_names: Optional feature names for importance tracking

        Returns:
            Self for chaining
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_scaled, y)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict activity from physics features.

        Args:
            X: Physics features (n_samples, n_features)

        Returns:
            Predicted activity values
        """
        if not self._is_fitted:
            raise ValueError("Probe not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> ProbeResults:
        """
        Evaluate probe performance.

        Args:
            X: Physics features
            y: True activity values

        Returns:
            ProbeResults with metrics
        """
        y_pred = self.predict(X)

        # Compute metrics
        pearson = pearsonr(y, y_pred)[0]
        spearman = spearmanr(y, y_pred)[0]
        r2 = r2_score(y, y_pred)
        mse = np.mean((y - y_pred) ** 2)

        # Get feature importances if available
        importances = self._get_feature_importances()

        return ProbeResults(
            pearson_r=pearson,
            spearman_r=spearman,
            r2=r2,
            mse=mse,
            feature_importances=importances,
            model=self.model
        )

    def fit_evaluate_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        n_folds: int = None,
        logger: ExperimentLogger = None
    ) -> ProbeResults:
        """
        Fit with cross-validation and return evaluation metrics.

        This performs proper CV evaluation without data leakage,
        with comprehensive per-fold logging.

        Args:
            X: Physics features
            y: Activity values
            feature_names: Optional feature names
            n_folds: Number of CV folds (default from config)
            logger: Optional ExperimentLogger for detailed logging

        Returns:
            ProbeResults with CV metrics
        """
        n_folds = n_folds or self.config.n_folds
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]

        # Standardize (will refit in final model)
        X_scaled = self.scaler.fit_transform(X)

        # Manual CV with per-fold logging
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_seed)

        y_pred_cv = np.zeros_like(y)
        fold_val_pearsons = []
        fold_val_spearmans = []

        total_start = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_scaled)):
            fold_start = time.time()

            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if logger:
                logger.log_fold_start(fold_idx, n_folds, len(train_idx), len(val_idx))

            # Fit model on this fold
            model = self._create_model()
            model.fit(X_train, y_train)

            # Predict on train and val
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Store val predictions for overall CV metrics
            y_pred_cv[val_idx] = y_val_pred

            # Compute fold metrics
            train_pearson = pearsonr(y_train, y_train_pred)[0]
            val_pearson = pearsonr(y_val, y_val_pred)[0]
            train_spearman = spearmanr(y_train, y_train_pred)[0]
            val_spearman = spearmanr(y_val, y_val_pred)[0]
            train_mse = mean_squared_error(y_train, y_train_pred)
            val_mse = mean_squared_error(y_val, y_val_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            fold_time = time.time() - fold_start

            fold_val_pearsons.append(val_pearson)
            fold_val_spearmans.append(val_spearman)

            if logger:
                logger.log_fold_metrics(
                    fold=fold_idx,
                    train_pearson=train_pearson,
                    val_pearson=val_pearson,
                    train_spearman=train_spearman,
                    val_spearman=val_spearman,
                    train_mse=train_mse,
                    val_mse=val_mse,
                    train_r2=train_r2,
                    val_r2=val_r2,
                    train_size=len(train_idx),
                    val_size=len(val_idx),
                    fit_time=fold_time
                )

        total_cv_time = time.time() - total_start

        # Compute overall CV metrics
        pearson = pearsonr(y, y_pred_cv)[0]
        spearman = spearmanr(y, y_pred_cv)[0]
        r2 = r2_score(y, y_pred_cv)
        mse = mean_squared_error(y, y_pred_cv)

        if logger:
            logger.log_cv_summary()
            logger.info(f"Overall CV Pearson r: {pearson:.4f}")
            logger.info(f"CV time: {total_cv_time:.1f}s")

        # Fit final model on all data for feature importances and predictions
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        self._is_fitted = True

        importances = self._get_feature_importances()

        return ProbeResults(
            pearson_r=pearson,
            spearman_r=spearman,
            r2=r2,
            mse=mse,
            feature_importances=importances,
            cv_predictions=y_pred_cv,
            model=self.model
        )

    def _get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Extract feature importances from fitted model."""
        if not self._is_fitted or self.model is None:
            return None

        # Linear models have coef_
        if hasattr(self.model, 'coef_'):
            coefs = np.abs(self.model.coef_)
            return {name: float(coef) for name, coef in zip(self.feature_names, coefs)}

        return None

    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top N most important features.

        Args:
            n: Number of features to return

        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        importances = self._get_feature_importances()
        if importances is None:
            return []

        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def get_physics_family_contributions(
        self,
        family_indices: Dict[str, List[int]]
    ) -> Dict[str, float]:
        """
        Compute contribution of each physics family.

        Args:
            family_indices: Dict mapping family name to feature indices

        Returns:
            Dict mapping family name to total absolute importance
        """
        importances = self._get_feature_importances()
        if importances is None:
            return {}

        contributions = {}
        importance_values = list(importances.values())

        for family, indices in family_indices.items():
            family_importance = sum(importance_values[i] for i in indices if i < len(importance_values))
            contributions[family] = family_importance

        # Normalize to percentages
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total * 100 for k, v in contributions.items()}

        return contributions


class EnsemblePhysicsProbe:
    """
    Ensemble of physics probes for more robust predictions.

    Combines predictions from multiple probe types.
    """

    def __init__(self, config: TransferConfig = None):
        self.config = config or TransferConfig()
        self.probes: Dict[str, PhysicsActivityProbe] = {}
        self.weights: Dict[str, float] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        probe_types: List[str] = None
    ) -> 'EnsemblePhysicsProbe':
        """
        Fit ensemble of probes.

        Args:
            X: Physics features
            y: Activity values
            feature_names: Feature names
            probe_types: Types of probes to include (default: elastic_net, ridge)

        Returns:
            Self for chaining
        """
        probe_types = probe_types or ['elastic_net', 'ridge']

        for probe_type in probe_types:
            config = TransferConfig(
                probe_type=probe_type,
                probe_alpha=self.config.probe_alpha,
                probe_l1_ratio=self.config.probe_l1_ratio,
                random_seed=self.config.random_seed,
                n_folds=self.config.n_folds
            )
            probe = PhysicsActivityProbe(config)
            results = probe.fit_evaluate_cv(X, y, feature_names)

            self.probes[probe_type] = probe
            # Weight by Pearson r
            self.weights[probe_type] = max(0, results.pearson_r)

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        else:
            # Equal weights if all negative correlations
            self.weights = {k: 1.0 / len(self.probes) for k in self.probes}

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using weighted ensemble.

        Args:
            X: Physics features

        Returns:
            Ensemble predictions
        """
        predictions = np.zeros(len(X))

        for probe_type, probe in self.probes.items():
            weight = self.weights[probe_type]
            predictions += weight * probe.predict(X)

        return predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ProbeResults:
        """
        Evaluate ensemble performance.

        Args:
            X: Physics features
            y: True activity values

        Returns:
            ProbeResults with ensemble metrics
        """
        y_pred = self.predict(X)

        pearson = pearsonr(y, y_pred)[0]
        spearman = spearmanr(y, y_pred)[0]
        r2 = r2_score(y, y_pred)
        mse = np.mean((y - y_pred) ** 2)

        return ProbeResults(
            pearson_r=pearson,
            spearman_r=spearman,
            r2=r2,
            mse=mse,
            feature_importances=None,
            model=None
        )
