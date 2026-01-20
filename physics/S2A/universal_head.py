"""
Universal S2A Head Models.

Maps physics features to z-scored activity predictions.
Supports multiple model types: Ridge, ElasticNet, MLP.

The key insight is that while the physics→activity mapping is species-specific,
training on z-scored outputs from multiple species learns a universal
"average" relationship that transfers reasonably well.
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy.stats import spearmanr, pearsonr, rankdata
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from .config import S2AConfig


@dataclass
class HeadPredictionResults:
    """Results from head model prediction."""
    predictions: np.ndarray
    mode: str  # 'zscore', 'ranking'


@dataclass
class HeadEvaluationResults:
    """Results from evaluating a head model."""
    spearman_rho: float
    pearson_r: float
    r2: float
    mse: float
    n_samples: int

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'spearman_rho': self.spearman_rho,
            'pearson_r': self.pearson_r,
            'r2': self.r2,
            'mse': self.mse,
            'n_samples': self.n_samples
        }


class UniversalS2AHead:
    """
    Universal head model for physics→activity prediction.

    Trained on z-scored activity from multiple species.
    Outputs z-scores or rankings for zero-shot inference.
    """

    def __init__(self, config: S2AConfig = None):
        """
        Initialize S2A head.

        Args:
            config: S2AConfig with model parameters
        """
        self.config = config or S2AConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._is_fitted = False

    def _create_model(self) -> Any:
        """Create the underlying regression model."""
        head_type = self.config.head_type

        if head_type == 'ridge':
            return Ridge(
                alpha=self.config.head_alpha,
                random_state=self.config.random_seed
            )
        elif head_type == 'elastic_net':
            return ElasticNet(
                alpha=self.config.head_alpha,
                l1_ratio=self.config.head_l1_ratio,
                max_iter=10000,
                random_state=self.config.random_seed
            )
        elif head_type == 'mlp':
            return MLPRegressor(
                hidden_layer_sizes=tuple(self.config.head_hidden_sizes),
                activation='relu',
                solver='adam',
                alpha=self.config.head_alpha,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.config.random_seed
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def fit(
        self,
        X: np.ndarray,
        y_zscore: np.ndarray,
        feature_names: List[str] = None
    ) -> 'UniversalS2AHead':
        """
        Fit the head model on z-scored activity data.

        Args:
            X: Physics features (n_samples, n_features)
            y_zscore: Z-scored activity values (n_samples,)
            feature_names: Optional feature names

        Returns:
            Self for chaining
        """
        self.feature_names = feature_names or [
            f'feature_{i}' for i in range(X.shape[1])
        ]

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_scaled, y_zscore)

        self._is_fitted = True
        return self

    def predict_zscore(self, X: np.ndarray) -> np.ndarray:
        """
        Predict z-scored activity.

        Args:
            X: Physics features (n_samples, n_features)

        Returns:
            Predicted z-scores (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("Head not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_ranking(self, X: np.ndarray) -> np.ndarray:
        """
        Predict percentile rankings (0-100).

        Rankings are more robust for zero-shot transfer since
        they only depend on relative ordering, not absolute scale.

        Args:
            X: Physics features (n_samples, n_features)

        Returns:
            Percentile ranks (n_samples,), 0-100
        """
        z_scores = self.predict_zscore(X)
        # Convert to percentile ranks
        ranks = rankdata(z_scores, method='average')
        percentiles = (ranks - 1) / (len(ranks) - 1) * 100
        return percentiles

    def predict(
        self,
        X: np.ndarray,
        mode: str = None
    ) -> HeadPredictionResults:
        """
        Predict using specified output mode.

        Args:
            X: Physics features
            mode: Output mode ('zscore', 'ranking'). Uses config default if None.

        Returns:
            HeadPredictionResults with predictions
        """
        mode = mode or self.config.output_mode

        if mode == 'zscore':
            preds = self.predict_zscore(X)
        elif mode == 'ranking':
            preds = self.predict_ranking(X)
        else:
            raise ValueError(f"Unknown output mode: {mode}")

        return HeadPredictionResults(predictions=preds, mode=mode)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> HeadEvaluationResults:
        """
        Evaluate head model performance.

        Args:
            X: Physics features
            y_true: True activity values (can be raw or z-scored)

        Returns:
            HeadEvaluationResults with metrics
        """
        y_pred = self.predict_zscore(X)

        # Compute correlation metrics (invariant to scale)
        spearman = spearmanr(y_true, y_pred)[0]
        pearson = pearsonr(y_true, y_pred)[0]

        # Compute MSE and R2 (sensitive to scale)
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        return HeadEvaluationResults(
            spearman_rho=spearman,
            pearson_r=pearson,
            r2=r2,
            mse=mse,
            n_samples=len(y_true)
        )

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """
        Get feature importances from the model.

        Returns:
            Dict mapping feature names to importance (absolute coefficient)
            or None if not available.
        """
        if not self._is_fitted or self.model is None:
            return None

        # Linear models have coef_
        if hasattr(self.model, 'coef_'):
            coefs = np.abs(self.model.coef_)
            return {
                name: float(coef)
                for name, coef in zip(self.feature_names, coefs)
            }

        return None

    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top N most important features.

        Args:
            n: Number of features to return

        Returns:
            List of (feature_name, importance) tuples, sorted by importance
        """
        importances = self.get_feature_importances()
        if importances is None:
            return []

        sorted_features = sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]

    def get_family_contributions(
        self,
        family_indices: Dict[str, List[int]]
    ) -> Dict[str, float]:
        """
        Compute contribution of each physics family.

        Args:
            family_indices: Dict mapping family name to feature indices

        Returns:
            Dict mapping family name to percentage contribution
        """
        importances = self.get_feature_importances()
        if importances is None:
            return {}

        importance_values = list(importances.values())
        contributions = {}

        for family, indices in family_indices.items():
            family_importance = sum(
                importance_values[i]
                for i in indices
                if i < len(importance_values)
            )
            contributions[family] = family_importance

        # Normalize to percentages
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total * 100 for k, v in contributions.items()}

        return contributions

    def save(self, filepath: str):
        """
        Save the fitted head model.

        Args:
            filepath: Path to save the model
        """
        if not self._is_fitted:
            raise ValueError("Head not fitted. Call fit() first.")

        state = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            '_is_fitted': self._is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str) -> 'UniversalS2AHead':
        """
        Load a saved head model.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded UniversalS2AHead instance
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        head = cls(config=state['config'])
        head.model = state['model']
        head.scaler = state['scaler']
        head.feature_names = state['feature_names']
        head._is_fitted = state['_is_fitted']

        return head


class EnsembleS2AHead:
    """
    Ensemble of S2A heads for more robust predictions.

    Combines predictions from multiple head types (Ridge, ElasticNet, MLP).
    """

    def __init__(self, config: S2AConfig = None):
        """
        Initialize ensemble head.

        Args:
            config: Base S2AConfig (will be modified for each head type)
        """
        self.config = config or S2AConfig()
        self.heads: Dict[str, UniversalS2AHead] = {}
        self.weights: Dict[str, float] = {}
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y_zscore: np.ndarray,
        feature_names: List[str] = None,
        head_types: List[str] = None,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> 'EnsembleS2AHead':
        """
        Fit ensemble of heads.

        Args:
            X: Physics features for training
            y_zscore: Z-scored activity for training
            feature_names: Feature names
            head_types: Types of heads to include (default: ['ridge', 'elastic_net'])
            X_val: Optional validation features for weighting
            y_val: Optional validation activity for weighting

        Returns:
            Self for chaining
        """
        head_types = head_types or ['ridge', 'elastic_net']

        for head_type in head_types:
            print(f"  Training {head_type} head...")

            # Create config for this head type
            head_config = S2AConfig(
                head_type=head_type,
                head_alpha=self.config.head_alpha,
                head_l1_ratio=self.config.head_l1_ratio,
                head_hidden_sizes=self.config.head_hidden_sizes,
                random_seed=self.config.random_seed
            )

            head = UniversalS2AHead(head_config)
            head.fit(X, y_zscore, feature_names)
            self.heads[head_type] = head

            # Compute weight based on validation performance
            if X_val is not None and y_val is not None:
                results = head.evaluate(X_val, y_val)
                weight = max(0, results.spearman_rho)
            else:
                # Use training correlation as weight
                results = head.evaluate(X, y_zscore)
                weight = max(0, results.spearman_rho)

            self.weights[head_type] = weight
            print(f"    {head_type}: Spearman ρ = {results.spearman_rho:.4f}")

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        else:
            # Equal weights if all correlations are negative
            self.weights = {k: 1.0 / len(self.heads) for k in self.heads}

        self._is_fitted = True
        return self

    def predict_zscore(self, X: np.ndarray) -> np.ndarray:
        """
        Predict z-scores using weighted ensemble.

        Args:
            X: Physics features

        Returns:
            Weighted average z-score predictions
        """
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        predictions = np.zeros(len(X))
        for head_type, head in self.heads.items():
            weight = self.weights[head_type]
            predictions += weight * head.predict_zscore(X)

        return predictions

    def predict_ranking(self, X: np.ndarray) -> np.ndarray:
        """
        Predict percentile rankings from ensemble.

        Args:
            X: Physics features

        Returns:
            Percentile ranks (0-100)
        """
        z_scores = self.predict_zscore(X)
        ranks = rankdata(z_scores, method='average')
        percentiles = (ranks - 1) / (len(ranks) - 1) * 100
        return percentiles

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> HeadEvaluationResults:
        """
        Evaluate ensemble performance.

        Args:
            X: Physics features
            y_true: True activity values

        Returns:
            HeadEvaluationResults with ensemble metrics
        """
        y_pred = self.predict_zscore(X)

        spearman = spearmanr(y_true, y_pred)[0]
        pearson = pearsonr(y_true, y_pred)[0]
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        return HeadEvaluationResults(
            spearman_rho=spearman,
            pearson_r=pearson,
            r2=r2,
            mse=mse,
            n_samples=len(y_true)
        )

    def save(self, filepath: str):
        """Save the ensemble."""
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        state = {
            'heads': {k: v for k, v in self.heads.items()},
            'weights': self.weights,
            'config': self.config,
            '_is_fitted': self._is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str) -> 'EnsembleS2AHead':
        """Load a saved ensemble."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        ensemble = cls(config=state['config'])
        ensemble.heads = state['heads']
        ensemble.weights = state['weights']
        ensemble._is_fitted = state['_is_fitted']

        return ensemble
