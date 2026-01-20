"""
Causal Mediation Analysis for Physics Features.

Quantifies how much physics mediates the sequence→activity relationship,
decomposing the total effect into direct and indirect (physics-mediated) effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import scipy.stats as stats

from .config import InterpreterConfig, PHYSICS_FAMILIES, get_fusemap_root


@dataclass
class MediationResult:
    """Results from mediation analysis."""
    # Effect decomposition
    total_effect: float
    direct_effect: float
    indirect_effect: float
    proportion_mediated: float

    # Statistical significance
    indirect_effect_ci: Tuple[float, float]
    indirect_effect_p: float

    # Per-mediator effects
    mediator_effects: Dict[str, float]

    # Model fit statistics
    outcome_r2: float
    mediator_r2: float

    # Sample info
    n_samples: int
    n_mediators: int


@dataclass
class FamilyMediationResult:
    """Mediation results grouped by physics family."""
    family: str
    n_features: int
    indirect_effect: float
    proportion_mediated: float
    ci_lower: float
    ci_upper: float
    significant: bool


class MediationAnalyzer:
    """
    Causal mediation analysis for physics features.

    Implements the Baron & Kenny approach with bootstrap confidence intervals
    to quantify how physics features mediate the sequence→activity relationship.

    Model:
        Sequence Representation → Physics Features → Activity

    The total effect is decomposed into:
        - Direct effect: Sequence → Activity (controlling for physics)
        - Indirect effect: Sequence → Physics → Activity

    Usage:
        analyzer = MediationAnalyzer(config)
        result = analyzer.analyze(X_sequence, X_physics, y_activity)
        print(f"Proportion mediated: {result.proportion_mediated:.2%}")
    """

    def __init__(self, config: InterpreterConfig = None):
        self.config = config or InterpreterConfig()
        self.scaler_seq = StandardScaler()
        self.scaler_phys = StandardScaler()

    def analyze(
        self,
        X_sequence: np.ndarray,
        X_physics: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        sequence_dim_names: List[str] = None
    ) -> MediationResult:
        """
        Perform mediation analysis.

        Args:
            X_sequence: Sequence representations (n_samples, n_seq_features)
                       Can be embeddings, one-hot encodings, etc.
            X_physics: Physics features (n_samples, n_physics_features)
            y: Activity values (n_samples,)
            feature_names: Names of physics features
            sequence_dim_names: Names of sequence dimensions

        Returns:
            MediationResult with effect decomposition
        """
        n_samples = X_sequence.shape[0]
        n_mediators = X_physics.shape[1]

        # Standardize features
        X_seq_scaled = self.scaler_seq.fit_transform(X_sequence)
        X_phys_scaled = self.scaler_phys.fit_transform(X_physics)
        y_centered = y - y.mean()

        # Step 1: Total effect (c path)
        # Sequence → Activity
        model_total = Ridge(alpha=1.0)
        model_total.fit(X_seq_scaled, y_centered)
        total_effect = self._compute_effect(model_total, X_seq_scaled, y_centered)

        # Step 2: Mediator model (a path)
        # Sequence → Physics (for each mediator)
        a_paths = np.zeros(n_mediators)
        mediator_r2_list = []

        for i in range(n_mediators):
            model_med = LinearRegression()
            model_med.fit(X_seq_scaled, X_phys_scaled[:, i])
            a_paths[i] = self._compute_effect(model_med, X_seq_scaled, X_phys_scaled[:, i])
            mediator_r2_list.append(model_med.score(X_seq_scaled, X_phys_scaled[:, i]))

        # Step 3: Outcome model with mediators (b and c' paths)
        # Combined [Sequence, Physics] → Activity
        X_combined = np.hstack([X_seq_scaled, X_phys_scaled])
        model_full = Ridge(alpha=1.0)
        model_full.fit(X_combined, y_centered)

        # Direct effect (c' path) - effect of sequence controlling for physics
        n_seq_features = X_seq_scaled.shape[1]
        direct_effect = np.sum(np.abs(model_full.coef_[:n_seq_features]))

        # b paths (physics → activity controlling for sequence)
        b_paths = model_full.coef_[n_seq_features:]

        # Indirect effect (a*b summed across mediators)
        indirect_effects = a_paths * b_paths
        indirect_effect = np.sum(np.abs(indirect_effects))

        # Proportion mediated
        if abs(total_effect) > 0:
            proportion_mediated = indirect_effect / (direct_effect + indirect_effect)
        else:
            proportion_mediated = 0.0

        # Bootstrap for confidence intervals
        ci_lower, ci_upper, p_value = self._bootstrap_indirect_effect(
            X_seq_scaled, X_phys_scaled, y_centered
        )

        # Per-mediator effects
        mediator_effects = {}
        if feature_names:
            for i, name in enumerate(feature_names):
                mediator_effects[name] = float(abs(indirect_effects[i]))
        else:
            for i in range(n_mediators):
                mediator_effects[f'mediator_{i}'] = float(abs(indirect_effects[i]))

        # Model fit statistics
        outcome_r2 = model_full.score(X_combined, y_centered)
        mediator_r2 = np.mean(mediator_r2_list)

        return MediationResult(
            total_effect=float(total_effect),
            direct_effect=float(direct_effect),
            indirect_effect=float(indirect_effect),
            proportion_mediated=float(proportion_mediated),
            indirect_effect_ci=(ci_lower, ci_upper),
            indirect_effect_p=p_value,
            mediator_effects=mediator_effects,
            outcome_r2=outcome_r2,
            mediator_r2=mediator_r2,
            n_samples=n_samples,
            n_mediators=n_mediators
        )

    def _compute_effect(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Compute effect size as sum of absolute coefficients."""
        return np.sum(np.abs(model.coef_))

    def _bootstrap_indirect_effect(
        self,
        X_seq: np.ndarray,
        X_phys: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float, float]:
        """Bootstrap confidence interval for indirect effect."""
        n_bootstrap = self.config.mediation_n_bootstrap
        confidence = self.config.mediation_confidence

        n_samples = X_seq.shape[0]
        n_mediators = X_phys.shape[1]

        indirect_effects_boot = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_seq_boot = X_seq[indices]
            X_phys_boot = X_phys[indices]
            y_boot = y[indices]

            # Compute a paths
            a_paths = np.zeros(n_mediators)
            for i in range(n_mediators):
                model = LinearRegression()
                model.fit(X_seq_boot, X_phys_boot[:, i])
                a_paths[i] = np.sum(np.abs(model.coef_))

            # Compute b paths
            X_combined = np.hstack([X_seq_boot, X_phys_boot])
            model_full = Ridge(alpha=1.0)
            model_full.fit(X_combined, y_boot)
            b_paths = model_full.coef_[X_seq_boot.shape[1]:]

            # Indirect effect
            indirect = np.sum(np.abs(a_paths * b_paths))
            indirect_effects_boot.append(indirect)

        indirect_effects_boot = np.array(indirect_effects_boot)

        # Confidence interval
        alpha = 1 - confidence
        ci_lower = np.percentile(indirect_effects_boot, alpha / 2 * 100)
        ci_upper = np.percentile(indirect_effects_boot, (1 - alpha / 2) * 100)

        # P-value (two-tailed test against zero)
        p_value = np.mean(indirect_effects_boot <= 0) * 2
        p_value = min(p_value, 1.0)

        return ci_lower, ci_upper, p_value

    def analyze_by_family(
        self,
        X_sequence: np.ndarray,
        X_physics: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[FamilyMediationResult]:
        """
        Perform mediation analysis grouped by physics family.

        Args:
            X_sequence: Sequence representations
            X_physics: Physics features
            y: Activity values
            feature_names: Names of physics features

        Returns:
            List of FamilyMediationResult for each physics family
        """
        results = []

        for family, prefixes in PHYSICS_FAMILIES.items():
            # Find features in this family
            family_indices = []
            for i, name in enumerate(feature_names):
                for prefix in prefixes:
                    if name.startswith(prefix):
                        family_indices.append(i)
                        break

            if not family_indices:
                continue

            # Extract family features
            X_family = X_physics[:, family_indices]
            family_names = [feature_names[i] for i in family_indices]

            # Run mediation analysis
            result = self.analyze(
                X_sequence, X_family, y,
                feature_names=family_names
            )

            # Determine significance
            significant = result.indirect_effect_ci[0] > 0

            results.append(FamilyMediationResult(
                family=family,
                n_features=len(family_indices),
                indirect_effect=result.indirect_effect,
                proportion_mediated=result.proportion_mediated,
                ci_lower=result.indirect_effect_ci[0],
                ci_upper=result.indirect_effect_ci[1],
                significant=significant
            ))

        # Sort by indirect effect
        results.sort(key=lambda x: x.indirect_effect, reverse=True)

        return results

    def get_mediation_summary(
        self,
        result: MediationResult,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get summary of mediation results.

        Args:
            result: MediationResult from analyze()
            top_n: Number of top mediators to show

        Returns:
            DataFrame with summary statistics
        """
        rows = [
            {'metric': 'Total Effect', 'value': result.total_effect},
            {'metric': 'Direct Effect', 'value': result.direct_effect},
            {'metric': 'Indirect Effect', 'value': result.indirect_effect},
            {'metric': 'Proportion Mediated', 'value': result.proportion_mediated},
            {'metric': 'Indirect Effect CI Lower', 'value': result.indirect_effect_ci[0]},
            {'metric': 'Indirect Effect CI Upper', 'value': result.indirect_effect_ci[1]},
            {'metric': 'Indirect Effect P-value', 'value': result.indirect_effect_p},
            {'metric': 'Outcome R²', 'value': result.outcome_r2},
            {'metric': 'Mediator R²', 'value': result.mediator_r2},
            {'metric': 'N Samples', 'value': result.n_samples},
            {'metric': 'N Mediators', 'value': result.n_mediators},
        ]

        return pd.DataFrame(rows)

    def get_top_mediators(
        self,
        result: MediationResult,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get top mediating features.

        Args:
            result: MediationResult from analyze()
            top_n: Number of top mediators

        Returns:
            DataFrame with top mediators
        """
        sorted_mediators = sorted(
            result.mediator_effects.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        rows = []
        total_indirect = sum(result.mediator_effects.values())

        for name, effect in sorted_mediators:
            pct = effect / total_indirect * 100 if total_indirect > 0 else 0
            rows.append({
                'feature': name,
                'indirect_effect': effect,
                'pct_of_total': pct
            })

        return pd.DataFrame(rows)

    def save(self, result: MediationResult, output_path: Path):
        """Save mediation results to JSON."""
        output = {
            'total_effect': result.total_effect,
            'direct_effect': result.direct_effect,
            'indirect_effect': result.indirect_effect,
            'proportion_mediated': result.proportion_mediated,
            'indirect_effect_ci': result.indirect_effect_ci,
            'indirect_effect_p': result.indirect_effect_p,
            'outcome_r2': result.outcome_r2,
            'mediator_r2': result.mediator_r2,
            'n_samples': result.n_samples,
            'n_mediators': result.n_mediators,
            'mediator_effects': result.mediator_effects
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Mediation results saved to {output_path}")


def compute_mediation(
    X_sequence: np.ndarray,
    X_physics: np.ndarray,
    y: np.ndarray,
    feature_names: List[str] = None,
    config: InterpreterConfig = None
) -> MediationResult:
    """
    Convenience function to compute mediation analysis.

    Args:
        X_sequence: Sequence representations
        X_physics: Physics features
        y: Activity values
        feature_names: Optional feature names
        config: Optional configuration

    Returns:
        MediationResult
    """
    analyzer = MediationAnalyzer(config)
    return analyzer.analyze(X_sequence, X_physics, y, feature_names)
