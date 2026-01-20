"""
Physics-Activity Landscape Analysis.

Systematic mapping of how physics features relate to activity,
combining linear correlations, SHAP values, and partial dependence analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from scipy.stats import pearsonr, spearmanr

from .config import InterpreterConfig, PHYSICS_FAMILIES, get_fusemap_root

# Try to import SHAP, but don't fail if unavailable
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")


@dataclass
class FeatureRelationship:
    """Relationship between a physics feature and activity."""
    feature_name: str
    family: str

    # Correlation statistics
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float

    # Linear coefficient (from probe)
    linear_coefficient: float
    linear_coefficient_abs: float

    # SHAP values (if computed)
    mean_shap: Optional[float] = None
    mean_abs_shap: Optional[float] = None

    # Nonlinearity score
    nonlinearity_score: Optional[float] = None


@dataclass
class LandscapeResult:
    """Results from landscape analysis."""
    # Per-feature relationships
    feature_relationships: List[FeatureRelationship]

    # Family-level statistics
    family_correlations: Dict[str, float]
    family_shap_importance: Dict[str, float]

    # Model performance
    linear_r2: float
    nonlinear_r2: float
    shap_available: bool

    # Top features by different criteria
    top_by_correlation: List[Tuple[str, float]]
    top_by_shap: List[Tuple[str, float]]
    top_nonlinear: List[Tuple[str, float]]


class LandscapeAnalyzer:
    """
    Physics-activity landscape analysis.

    Maps how physics features relate to activity through multiple lenses:
    - Linear correlations (Pearson, Spearman)
    - Linear probe coefficients
    - SHAP values for nonlinear importance
    - Nonlinearity detection

    Usage:
        analyzer = LandscapeAnalyzer(config)
        result = analyzer.analyze(X_physics, y_activity, feature_names)
        print(result.top_by_shap)
    """

    def __init__(self, config: InterpreterConfig = None):
        self.config = config or InterpreterConfig()
        self.scaler = StandardScaler()
        self.linear_model = None
        self.nonlinear_model = None
        self.shap_values = None

    def analyze(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        compute_shap: bool = True
    ) -> LandscapeResult:
        """
        Perform landscape analysis.

        Args:
            X: Physics features (n_samples, n_features)
            y: Activity values (n_samples,)
            feature_names: Names of features
            compute_shap: Whether to compute SHAP values

        Returns:
            LandscapeResult with comprehensive analysis
        """
        n_samples, n_features = X.shape

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Split for model fitting
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.config.random_seed
        )

        # Fit linear model
        self.linear_model = ElasticNet(
            alpha=self.config.landscape_elastic_net_alpha,
            l1_ratio=self.config.landscape_elastic_net_l1_ratio,
            max_iter=10000
        )
        self.linear_model.fit(X_train, y_train)
        linear_r2 = self.linear_model.score(X_test, y_test)

        # Fit nonlinear model (gradient boosting)
        self.nonlinear_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.config.random_seed
        )
        self.nonlinear_model.fit(X_train, y_train)
        nonlinear_r2 = self.nonlinear_model.score(X_test, y_test)

        # Compute SHAP values if requested
        shap_values = None
        shap_available = False

        if compute_shap and SHAP_AVAILABLE:
            try:
                # Sample background data for SHAP
                n_background = min(self.config.landscape_shap_samples, len(X_train))
                background_indices = np.random.choice(
                    len(X_train), n_background, replace=False
                )
                background = X_train[background_indices]

                # Create explainer
                explainer = shap.Explainer(
                    self.nonlinear_model.predict,
                    background
                )

                # Compute SHAP values on test set (or sample if large)
                n_explain = min(500, len(X_test))
                explain_indices = np.random.choice(
                    len(X_test), n_explain, replace=False
                )
                shap_values = explainer(X_test[explain_indices])
                shap_available = True

            except Exception as e:
                warnings.warn(f"SHAP computation failed: {e}")
                shap_values = None

        # Compute per-feature relationships
        feature_relationships = self._compute_feature_relationships(
            X, y, feature_names, shap_values, linear_r2, nonlinear_r2
        )

        # Compute family-level statistics
        family_correlations = self._compute_family_correlations(
            feature_relationships
        )
        family_shap = self._compute_family_shap(
            feature_relationships
        ) if shap_available else {}

        # Get top features
        top_by_correlation = sorted(
            [(f.feature_name, abs(f.pearson_r)) for f in feature_relationships],
            key=lambda x: x[1],
            reverse=True
        )[:20]

        top_by_shap = []
        if shap_available:
            top_by_shap = sorted(
                [(f.feature_name, f.mean_abs_shap) for f in feature_relationships
                 if f.mean_abs_shap is not None],
                key=lambda x: x[1],
                reverse=True
            )[:20]

        top_nonlinear = sorted(
            [(f.feature_name, f.nonlinearity_score) for f in feature_relationships
             if f.nonlinearity_score is not None],
            key=lambda x: x[1],
            reverse=True
        )[:20]

        return LandscapeResult(
            feature_relationships=feature_relationships,
            family_correlations=family_correlations,
            family_shap_importance=family_shap,
            linear_r2=linear_r2,
            nonlinear_r2=nonlinear_r2,
            shap_available=shap_available,
            top_by_correlation=top_by_correlation,
            top_by_shap=top_by_shap,
            top_nonlinear=top_nonlinear
        )

    def _compute_feature_relationships(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        shap_values: Optional[Any],
        linear_r2: float,
        nonlinear_r2: float
    ) -> List[FeatureRelationship]:
        """Compute relationship statistics for each feature."""
        relationships = []

        for i, name in enumerate(feature_names):
            # Correlations
            pearson_r, pearson_p = pearsonr(X[:, i], y)
            spearman_r, spearman_p = spearmanr(X[:, i], y)

            # Linear coefficient
            linear_coef = self.linear_model.coef_[i]

            # Determine family
            family = self._get_feature_family(name)

            # SHAP values
            mean_shap = None
            mean_abs_shap = None
            if shap_values is not None:
                feature_shap = shap_values.values[:, i]
                mean_shap = float(np.mean(feature_shap))
                mean_abs_shap = float(np.mean(np.abs(feature_shap)))

            # Nonlinearity score
            # Higher if nonlinear model captures more than linear for this feature
            nonlinearity = max(0, nonlinear_r2 - linear_r2)

            relationships.append(FeatureRelationship(
                feature_name=name,
                family=family,
                pearson_r=float(pearson_r),
                pearson_p=float(pearson_p),
                spearman_r=float(spearman_r),
                spearman_p=float(spearman_p),
                linear_coefficient=float(linear_coef),
                linear_coefficient_abs=float(abs(linear_coef)),
                mean_shap=mean_shap,
                mean_abs_shap=mean_abs_shap,
                nonlinearity_score=nonlinearity
            ))

        return relationships

    def _get_feature_family(self, feature_name: str) -> str:
        """Determine which physics family a feature belongs to."""
        for family, prefixes in PHYSICS_FAMILIES.items():
            for prefix in prefixes:
                if feature_name.startswith(prefix):
                    return family
        return 'unknown'

    def _compute_family_correlations(
        self,
        relationships: List[FeatureRelationship]
    ) -> Dict[str, float]:
        """Compute average correlation by family."""
        family_corrs = {}
        for family in PHYSICS_FAMILIES.keys():
            family_features = [r for r in relationships if r.family == family]
            if family_features:
                avg_corr = np.mean([abs(r.pearson_r) for r in family_features])
                family_corrs[family] = float(avg_corr)
        return family_corrs

    def _compute_family_shap(
        self,
        relationships: List[FeatureRelationship]
    ) -> Dict[str, float]:
        """Compute average SHAP importance by family."""
        family_shap = {}
        for family in PHYSICS_FAMILIES.keys():
            family_features = [r for r in relationships
                             if r.family == family and r.mean_abs_shap is not None]
            if family_features:
                avg_shap = np.mean([r.mean_abs_shap for r in family_features])
                family_shap[family] = float(avg_shap)
        return family_shap

    def get_correlation_matrix(
        self,
        X: np.ndarray,
        feature_names: List[str],
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Compute feature correlation matrix.

        Args:
            X: Physics features
            feature_names: Feature names
            method: 'pearson' or 'spearman'

        Returns:
            DataFrame with correlation matrix
        """
        df = pd.DataFrame(X, columns=feature_names)
        return df.corr(method=method)

    def get_family_summary(
        self,
        result: LandscapeResult
    ) -> pd.DataFrame:
        """
        Get summary by physics family.

        Args:
            result: LandscapeResult from analyze()

        Returns:
            DataFrame with family-level summary
        """
        rows = []
        for family in PHYSICS_FAMILIES.keys():
            family_features = [r for r in result.feature_relationships
                             if r.family == family]

            if not family_features:
                continue

            row = {
                'family': family,
                'n_features': len(family_features),
                'mean_abs_correlation': np.mean([abs(r.pearson_r) for r in family_features]),
                'max_abs_correlation': max([abs(r.pearson_r) for r in family_features]),
                'mean_abs_coefficient': np.mean([r.linear_coefficient_abs for r in family_features]),
            }

            if result.shap_available:
                shap_vals = [r.mean_abs_shap for r in family_features if r.mean_abs_shap is not None]
                if shap_vals:
                    row['mean_shap'] = np.mean(shap_vals)
                    row['max_shap'] = max(shap_vals)

            rows.append(row)

        df = pd.DataFrame(rows)
        return df.sort_values('mean_abs_correlation', ascending=False)

    def get_feature_table(
        self,
        result: LandscapeResult,
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Get detailed feature table.

        Args:
            result: LandscapeResult from analyze()
            top_n: Number of features to include

        Returns:
            DataFrame with feature details
        """
        # Sort by absolute correlation
        sorted_features = sorted(
            result.feature_relationships,
            key=lambda x: abs(x.pearson_r),
            reverse=True
        )[:top_n]

        rows = []
        for f in sorted_features:
            row = {
                'feature': f.feature_name,
                'family': f.family,
                'pearson_r': f.pearson_r,
                'spearman_r': f.spearman_r,
                'linear_coef': f.linear_coefficient,
            }
            if f.mean_abs_shap is not None:
                row['mean_abs_shap'] = f.mean_abs_shap

            rows.append(row)

        return pd.DataFrame(rows)

    def compute_partial_dependence(
        self,
        X: np.ndarray,
        feature_idx: int,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute partial dependence for a feature.

        Args:
            X: Physics features (standardized)
            feature_idx: Feature index
            n_points: Number of grid points

        Returns:
            Tuple of (feature_values, partial_dependence)
        """
        if self.nonlinear_model is None:
            raise ValueError("Model not fitted. Call analyze() first.")

        # Create grid
        feature_min = X[:, feature_idx].min()
        feature_max = X[:, feature_idx].max()
        grid = np.linspace(feature_min, feature_max, n_points)

        # Compute partial dependence
        pd_values = []
        for val in grid:
            X_modified = X.copy()
            X_modified[:, feature_idx] = val
            predictions = self.nonlinear_model.predict(X_modified)
            pd_values.append(predictions.mean())

        return grid, np.array(pd_values)

    def save(self, result: LandscapeResult, output_path: Path):
        """Save landscape results to JSON."""
        output = {
            'linear_r2': result.linear_r2,
            'nonlinear_r2': result.nonlinear_r2,
            'shap_available': result.shap_available,
            'family_correlations': result.family_correlations,
            'family_shap_importance': result.family_shap_importance,
            'top_by_correlation': result.top_by_correlation,
            'top_by_shap': result.top_by_shap,
            'top_nonlinear': result.top_nonlinear,
            'feature_details': [
                {
                    'name': f.feature_name,
                    'family': f.family,
                    'pearson_r': f.pearson_r,
                    'spearman_r': f.spearman_r,
                    'linear_coef': f.linear_coefficient,
                    'mean_abs_shap': f.mean_abs_shap
                }
                for f in result.feature_relationships
            ]
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Landscape results saved to {output_path}")


def analyze_physics_landscape(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    config: InterpreterConfig = None,
    compute_shap: bool = True
) -> LandscapeResult:
    """
    Convenience function for landscape analysis.

    Args:
        X: Physics features
        y: Activity values
        feature_names: Feature names
        config: Optional configuration
        compute_shap: Whether to compute SHAP values

    Returns:
        LandscapeResult
    """
    analyzer = LandscapeAnalyzer(config)
    return analyzer.analyze(X, y, feature_names, compute_shap)
