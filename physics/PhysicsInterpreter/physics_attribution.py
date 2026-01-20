"""
Physics Feature Attribution.

Decomposes model predictions through physics features using trained
linear probes, enabling mechanistic interpretation of predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from .config import InterpreterConfig, PHYSICS_FAMILIES, get_fusemap_root


@dataclass
class AttributionResult:
    """Results from physics feature attribution."""
    # Per-feature contributions
    feature_contributions: Dict[str, float]

    # Family-level contributions (percentages)
    family_contributions: Dict[str, float]

    # Model info
    probe_r2: float
    probe_pearson: float

    # Top features
    top_positive: List[Tuple[str, float]]
    top_negative: List[Tuple[str, float]]

    # Prediction breakdown
    predicted_activity: float
    intercept_contribution: float


@dataclass
class SequenceAttribution:
    """Attribution results for a single sequence."""
    sequence_id: str
    predicted_activity: float

    # Feature values and their contributions
    feature_values: Dict[str, float]
    feature_contributions: Dict[str, float]

    # Family contributions for this sequence
    family_contributions: Dict[str, float]

    # Top contributors
    top_positive_features: List[Tuple[str, float]]
    top_negative_features: List[Tuple[str, float]]


class PhysicsAttributor:
    """
    Physics feature attribution using linear probes.

    Decomposes activity predictions into contributions from physics features,
    enabling mechanistic interpretation.

    Usage:
        attributor = PhysicsAttributor(config)
        attributor.fit(X_physics, y_activity, feature_names)

        # Get overall feature importance
        result = attributor.get_attribution()

        # Get per-sequence attribution
        seq_attr = attributor.attribute_sequence(x_physics, feature_names, seq_id)
    """

    def __init__(self, config: InterpreterConfig = None):
        self.config = config or InterpreterConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.coefficients: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> 'PhysicsAttributor':
        """
        Fit the attribution probe on physics features.

        Args:
            X: Physics features (n_samples, n_features)
            y: Activity values (n_samples,)
            feature_names: Names of features

        Returns:
            Self for chaining
        """
        self.feature_names = feature_names

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Create probe model
        if self.config.attribution_probe_type == 'ridge':
            self.model = Ridge(alpha=self.config.attribution_alpha)
        else:
            self.model = ElasticNet(
                alpha=self.config.attribution_alpha,
                l1_ratio=0.5,
                max_iter=10000
            )

        # Fit model
        self.model.fit(X_scaled, y)

        self.coefficients = self.model.coef_
        self._is_fitted = True

        # Compute fit metrics
        y_pred = self.model.predict(X_scaled)
        self._r2 = self.model.score(X_scaled, y)
        self._pearson = pearsonr(y, y_pred)[0]

        print(f"Attribution probe fitted: R²={self._r2:.4f}, Pearson r={self._pearson:.4f}")

        return self

    def get_attribution(self, top_n: int = 20) -> AttributionResult:
        """
        Get overall feature attribution from fitted probe.

        Args:
            top_n: Number of top features to return

        Returns:
            AttributionResult with feature and family contributions
        """
        if not self._is_fitted:
            raise ValueError("Probe not fitted. Call fit() first.")

        # Per-feature contributions (absolute coefficient magnitude)
        feature_contributions = {
            name: float(coef)
            for name, coef in zip(self.feature_names, self.coefficients)
        }

        # Family contributions
        family_contributions = self._compute_family_contributions(
            self.coefficients, self.feature_names
        )

        # Top positive and negative features
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_positive = [(f, c) for f, c in sorted_features if c > 0][:top_n]
        top_negative = [(f, c) for f, c in sorted_features if c < 0][-top_n:][::-1]

        return AttributionResult(
            feature_contributions=feature_contributions,
            family_contributions=family_contributions,
            probe_r2=self._r2,
            probe_pearson=self._pearson,
            top_positive=top_positive,
            top_negative=top_negative,
            predicted_activity=0.0,  # Overall, not per-sequence
            intercept_contribution=float(self.model.intercept_)
        )

    def attribute_sequence(
        self,
        x: np.ndarray,
        sequence_id: str = None
    ) -> SequenceAttribution:
        """
        Get attribution for a single sequence.

        Args:
            x: Physics features for one sequence (n_features,)
            sequence_id: Optional identifier

        Returns:
            SequenceAttribution with per-feature contributions
        """
        if not self._is_fitted:
            raise ValueError("Probe not fitted. Call fit() first.")

        # Standardize
        x_scaled = self.scaler.transform(x.reshape(1, -1))[0]

        # Compute prediction
        prediction = self.model.predict(x_scaled.reshape(1, -1))[0]

        # Compute per-feature contributions
        # Contribution = coefficient * scaled_feature_value
        contributions = self.coefficients * x_scaled

        feature_values = {
            name: float(val)
            for name, val in zip(self.feature_names, x)
        }

        feature_contributions = {
            name: float(contrib)
            for name, contrib in zip(self.feature_names, contributions)
        }

        # Family contributions for this sequence
        family_contributions = self._compute_family_contributions(
            contributions, self.feature_names
        )

        # Top contributors
        sorted_contrib = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_positive = [(f, c) for f, c in sorted_contrib if c > 0][:10]
        top_negative = [(f, c) for f, c in sorted_contrib if c < 0][:10]

        return SequenceAttribution(
            sequence_id=sequence_id or 'unknown',
            predicted_activity=prediction,
            feature_values=feature_values,
            feature_contributions=feature_contributions,
            family_contributions=family_contributions,
            top_positive_features=top_positive,
            top_negative_features=top_negative
        )

    def _compute_family_contributions(
        self,
        values: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute contributions by physics family."""
        family_sums = {}

        for family, prefixes in PHYSICS_FAMILIES.items():
            family_sum = 0.0
            for i, name in enumerate(feature_names):
                for prefix in prefixes:
                    if name.startswith(prefix):
                        family_sum += abs(values[i])
                        break
            family_sums[family] = family_sum

        # Normalize to percentages
        total = sum(family_sums.values())
        if total > 0:
            family_sums = {k: v / total * 100 for k, v in family_sums.items()}

        return family_sums

    def get_family_breakdown(self) -> pd.DataFrame:
        """Get detailed breakdown by physics family."""
        if not self._is_fitted:
            raise ValueError("Probe not fitted. Call fit() first.")

        rows = []
        for family, prefixes in PHYSICS_FAMILIES.items():
            family_features = []
            for i, name in enumerate(self.feature_names):
                for prefix in prefixes:
                    if name.startswith(prefix):
                        family_features.append({
                            'feature': name,
                            'coefficient': self.coefficients[i],
                            'abs_coefficient': abs(self.coefficients[i])
                        })
                        break

            if family_features:
                df_family = pd.DataFrame(family_features)
                rows.append({
                    'family': family,
                    'n_features': len(family_features),
                    'total_abs_coef': df_family['abs_coefficient'].sum(),
                    'mean_abs_coef': df_family['abs_coefficient'].mean(),
                    'max_abs_coef': df_family['abs_coefficient'].max(),
                    'top_feature': df_family.loc[df_family['abs_coefficient'].idxmax(), 'feature']
                })

        return pd.DataFrame(rows).sort_values('total_abs_coef', ascending=False)

    def save(self, output_path: Path):
        """Save attribution results to JSON."""
        if not self._is_fitted:
            raise ValueError("Probe not fitted. Call fit() first.")

        result = self.get_attribution()

        output = {
            'probe_r2': result.probe_r2,
            'probe_pearson': result.probe_pearson,
            'intercept': result.intercept_contribution,
            'family_contributions': result.family_contributions,
            'top_positive_features': result.top_positive,
            'top_negative_features': result.top_negative,
            'n_features': len(self.feature_names),
            'feature_coefficients': {
                name: float(coef)
                for name, coef in zip(self.feature_names, self.coefficients)
            }
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Attribution saved to {output_path}")


def compute_physics_attribution(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    config: InterpreterConfig = None
) -> AttributionResult:
    """
    Convenience function to compute physics attribution.

    Args:
        X: Physics features
        y: Activity values
        feature_names: Feature names
        config: Optional configuration

    Returns:
        AttributionResult
    """
    attributor = PhysicsAttributor(config)
    attributor.fit(X, y, feature_names)
    return attributor.get_attribution()
