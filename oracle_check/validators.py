"""
Validators for OracleCheck

Implements validation checks for:
- Physics conformity
- Composition hygiene
- Confidence/OOD
- Mahalanobis distance
- Overall validation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.linalg import pinv, LinAlgError

from .config import (
    OracleCheckConfig,
    ValidationThresholds,
    Verdict,
    PHYSICS_FAMILIES,
)
from .reference_panels import ReferencePanel, ReferenceDistribution


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    passed: bool
    score: float
    message: str
    details: Dict = field(default_factory=dict)


@dataclass
class PhysicsValidationResult:
    """Result of physics validation."""
    passed: bool
    overall_score: float
    family_scores: Dict[str, float]  # z-scores per family
    family_passed: Dict[str, bool]
    max_z_score: float
    nll: float
    message: str
    details: Dict = field(default_factory=dict)


@dataclass
class CompositionValidationResult:
    """Result of composition validation."""
    passed: bool
    gc_content: float
    gc_passed: bool
    cpg_oe: float
    cpg_passed: bool
    entropy: float
    entropy_passed: bool
    repeat_fraction: float
    repeat_passed: bool
    max_homopolymer: int
    homopolymer_passed: bool
    message: str


@dataclass
class ConfidenceValidationResult:
    """Result of confidence/OOD validation."""
    passed: bool
    epistemic_std: Optional[float]
    epistemic_passed: bool
    conformal_width: Optional[float]
    conformal_passed: bool
    ood_score: Optional[float]
    ood_passed: bool
    message: str


@dataclass
class MahalanobisValidationResult:
    """Result of Mahalanobis distance validation."""
    passed: bool
    overall_distance: float
    family_distances: Dict[str, float]
    family_passed: Dict[str, bool]
    max_family_distance: float
    percentile: Optional[float]  # Percentile vs reference distribution
    message: str


class MahalanobisValidator:
    """
    Validates sequences using Mahalanobis distance.

    Computes:
    - Per-family Mahalanobis distance for physics features
    - Overall Mahalanobis distance for CADENCE embeddings
    """

    def __init__(
        self,
        reference_panel: ReferencePanel,
        thresholds: ValidationThresholds,
        percentile_threshold: float = 95.0,
    ):
        """
        Initialize Mahalanobis validator.

        Args:
            reference_panel: Reference panel with mean and covariance
            thresholds: Validation thresholds
            percentile_threshold: Percentile threshold for pass/fail
        """
        self.reference = reference_panel
        self.thresholds = thresholds
        self.percentile_threshold = percentile_threshold

        # Pre-compute inverse covariance for CADENCE features
        self._inv_cov = None
        if reference_panel.training_features_cov is not None:
            try:
                self._inv_cov = pinv(reference_panel.training_features_cov)
            except LinAlgError:
                print("Warning: Could not compute inverse covariance matrix")

    def _compute_mahalanobis(
        self,
        x: np.ndarray,
        mean: np.ndarray,
        inv_cov: np.ndarray,
    ) -> float:
        """Compute Mahalanobis distance."""
        diff = x - mean
        return float(np.sqrt(diff @ inv_cov @ diff))

    def validate_physics_features(
        self,
        physics_features: np.ndarray,
        feature_names: List[str],
    ) -> MahalanobisValidationResult:
        """
        Validate physics features using per-family Mahalanobis distance.

        Args:
            physics_features: Physics features [n_features]
            feature_names: Names of features

        Returns:
            MahalanobisValidationResult
        """
        family_distances = {}
        family_passed = {}

        for family, ref_features in self.reference.physics_distributions.items():
            # Collect features for this family
            family_values = []
            family_means = []
            family_stds = []

            for feature_name, ref_dist in ref_features.items():
                if feature_name in feature_names:
                    idx = feature_names.index(feature_name)
                    family_values.append(physics_features[idx])
                    family_means.append(ref_dist.mean)
                    family_stds.append(ref_dist.std + 1e-8)

            if len(family_values) < 2:
                family_distances[family] = 0.0
                family_passed[family] = True
                continue

            # Compute standardized features
            family_values = np.array(family_values)
            family_means = np.array(family_means)
            family_stds = np.array(family_stds)

            # Simple diagonal approximation for Mahalanobis
            # (z-scores averaged in quadrature)
            z_scores = (family_values - family_means) / family_stds
            distance = float(np.sqrt(np.mean(z_scores ** 2)))

            family_distances[family] = distance
            family_passed[family] = distance <= self.thresholds.physics_z_soft

        max_distance = max(family_distances.values()) if family_distances else 0.0
        all_passed = all(family_passed.values())

        if all_passed:
            message = f"Mahalanobis check passed (max family dist={max_distance:.2f})"
        else:
            failing = [f for f, p in family_passed.items() if not p]
            message = f"Mahalanobis check failed in families: {failing}"

        return MahalanobisValidationResult(
            passed=all_passed,
            overall_distance=max_distance,
            family_distances=family_distances,
            family_passed=family_passed,
            max_family_distance=max_distance,
            percentile=None,
            message=message,
        )

    def validate_cadence_features(
        self,
        cadence_features: np.ndarray,
        reference_distances: Optional[np.ndarray] = None,
    ) -> MahalanobisValidationResult:
        """
        Validate CADENCE embedding using Mahalanobis distance.

        Args:
            cadence_features: CADENCE backbone features [n_features]
            reference_distances: Optional reference Mahalanobis distances for percentile

        Returns:
            MahalanobisValidationResult
        """
        if self._inv_cov is None or self.reference.training_features_mean is None:
            return MahalanobisValidationResult(
                passed=True,
                overall_distance=0.0,
                family_distances={},
                family_passed={},
                max_family_distance=0.0,
                percentile=None,
                message="No CADENCE reference available",
            )

        distance = self._compute_mahalanobis(
            cadence_features,
            self.reference.training_features_mean,
            self._inv_cov,
        )

        # Compute percentile if reference available
        percentile = None
        passed = True
        if reference_distances is not None:
            percentile = float(stats.percentileofscore(reference_distances, distance))
            passed = percentile <= self.percentile_threshold

        if passed:
            message = f"CADENCE Mahalanobis passed (d={distance:.2f}"
            if percentile is not None:
                message += f", p{percentile:.0f}"
            message += ")"
        else:
            message = f"CADENCE Mahalanobis failed: d={distance:.2f}, p{percentile:.0f} > {self.percentile_threshold}"

        return MahalanobisValidationResult(
            passed=passed,
            overall_distance=distance,
            family_distances={},
            family_passed={},
            max_family_distance=distance,
            percentile=percentile,
            message=message,
        )


class PhysicsValidator:
    """
    Validates physics conformity of sequences.

    Checks:
    - Per-family z-scores against natural high performers
    - Physics NLL (negative log-likelihood)
    - Mahalanobis distance within each family
    """

    def __init__(
        self,
        reference_panel: ReferencePanel,
        thresholds: ValidationThresholds,
    ):
        self.reference = reference_panel
        self.thresholds = thresholds

    def validate(
        self,
        physics_features: np.ndarray,
        feature_names: List[str],
    ) -> PhysicsValidationResult:
        """
        Validate physics features against reference panel.

        Args:
            physics_features: Physics features [n_features]
            feature_names: Names of features

        Returns:
            PhysicsValidationResult
        """
        family_scores = {}
        family_passed = {}
        nlls = []

        for family, ref_features in self.reference.physics_distributions.items():
            family_z_scores = []

            for feature_name, ref_dist in ref_features.items():
                # Find matching feature
                if feature_name in feature_names:
                    idx = feature_names.index(feature_name)
                    value = physics_features[idx]

                    # Compute z-score
                    z = (value - ref_dist.mean) / (ref_dist.std + 1e-8)
                    family_z_scores.append(abs(z))

                    # Compute NLL contribution
                    nll = 0.5 * z ** 2 + np.log(ref_dist.std + 1e-8) + 0.5 * np.log(2 * np.pi)
                    nlls.append(nll)

            if family_z_scores:
                max_z = max(family_z_scores)
                family_scores[family] = max_z
                family_passed[family] = max_z <= self.thresholds.physics_z_soft
            else:
                family_scores[family] = 0.0
                family_passed[family] = True

        max_z_score = max(family_scores.values()) if family_scores else 0.0
        total_nll = sum(nlls) if nlls else 0.0

        # Determine pass/fail
        all_soft_pass = all(family_passed.values())
        no_hard_fail = max_z_score <= self.thresholds.physics_z_hard

        passed = all_soft_pass and no_hard_fail

        # Generate message
        if passed:
            message = f"Physics check passed (max z={max_z_score:.2f})"
        elif not no_hard_fail:
            message = f"Physics HARD FAIL: max z={max_z_score:.2f} > {self.thresholds.physics_z_hard}"
        else:
            failing = [f for f, p in family_passed.items() if not p]
            message = f"Physics soft fail in families: {failing}"

        return PhysicsValidationResult(
            passed=passed,
            overall_score=max_z_score,
            family_scores=family_scores,
            family_passed=family_passed,
            max_z_score=max_z_score,
            nll=total_nll,
            message=message,
        )


class CompositionValidator:
    """
    Validates composition metrics of sequences.

    Checks:
    - GC content within envelope
    - CpG O/E within natural range
    - Shannon entropy
    - Repeat fraction
    - Maximum homopolymer length
    """

    def __init__(
        self,
        reference_panel: Optional[ReferencePanel],
        thresholds: ValidationThresholds,
    ):
        self.reference = reference_panel
        self.thresholds = thresholds

    def validate(self, sequence: str) -> CompositionValidationResult:
        """
        Validate composition of a sequence.

        Args:
            sequence: DNA sequence

        Returns:
            CompositionValidationResult
        """
        seq = sequence.upper()

        # GC content
        gc = self._compute_gc(seq)
        gc_passed = self.thresholds.gc_min <= gc <= self.thresholds.gc_max

        # CpG O/E
        cpg_oe = self._compute_cpg_oe(seq)
        if self.reference and self.reference.cpg_distribution:
            cpg_low = self.reference.cpg_distribution.percentiles["p5"]
            cpg_high = self.reference.cpg_distribution.percentiles["p95"]
            cpg_passed = cpg_low <= cpg_oe <= cpg_high
        else:
            cpg_passed = 0.3 <= cpg_oe <= 2.0  # Default range

        # Entropy
        entropy = self._compute_entropy(seq)
        if self.reference and self.reference.entropy_distribution:
            entropy_low = self.reference.entropy_distribution.percentiles["p5"]
            entropy_passed = entropy >= entropy_low
        else:
            entropy_passed = entropy >= 1.5  # Reasonable minimum

        # Repeat fraction
        repeat_fraction = self._compute_repeat_fraction(seq)
        repeat_passed = repeat_fraction < self.thresholds.repeat_fraction_hard

        # Max homopolymer
        max_homopolymer = self._compute_max_homopolymer(seq)
        homopolymer_passed = max_homopolymer < self.thresholds.max_homopolymer_hard

        # Overall
        passed = all([gc_passed, cpg_passed, entropy_passed, repeat_passed, homopolymer_passed])

        # Message
        failures = []
        if not gc_passed:
            failures.append(f"GC={gc:.2f}")
        if not cpg_passed:
            failures.append(f"CpG_OE={cpg_oe:.2f}")
        if not entropy_passed:
            failures.append(f"entropy={entropy:.2f}")
        if not repeat_passed:
            failures.append(f"repeats={repeat_fraction:.2f}")
        if not homopolymer_passed:
            failures.append(f"homopolymer={max_homopolymer}")

        if passed:
            message = "Composition check passed"
        else:
            message = f"Composition failures: {', '.join(failures)}"

        return CompositionValidationResult(
            passed=passed,
            gc_content=gc,
            gc_passed=gc_passed,
            cpg_oe=cpg_oe,
            cpg_passed=cpg_passed,
            entropy=entropy,
            entropy_passed=entropy_passed,
            repeat_fraction=repeat_fraction,
            repeat_passed=repeat_passed,
            max_homopolymer=max_homopolymer,
            homopolymer_passed=homopolymer_passed,
            message=message,
        )

    @staticmethod
    def _compute_gc(sequence: str) -> float:
        gc = sum(1 for b in sequence if b in "GC")
        return gc / len(sequence) if len(sequence) > 0 else 0.0

    @staticmethod
    def _compute_cpg_oe(sequence: str) -> float:
        n = len(sequence)
        if n < 2:
            return 1.0
        c_count = sequence.count("C")
        g_count = sequence.count("G")
        cpg_count = sequence.count("CG")
        expected = (c_count * g_count) / n if n > 0 else 0
        return cpg_count / expected if expected > 0 else 1.0

    @staticmethod
    def _compute_entropy(sequence: str) -> float:
        n = len(sequence)
        if n == 0:
            return 0.0
        counts = {b: sequence.count(b) for b in "ACGT"}
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / n
                entropy -= p * np.log2(p)
        return entropy

    @staticmethod
    def _compute_repeat_fraction(sequence: str) -> float:
        """Compute fraction of sequence in simple repeats."""
        n = len(sequence)
        if n < 4:
            return 0.0

        repeat_positions = set()

        # Check for dinucleotide repeats
        for i in range(n - 3):
            dinuc = sequence[i:i+2]
            if sequence[i:i+4] == dinuc * 2:
                repeat_positions.update(range(i, min(i+4, n)))

        # Check for trinucleotide repeats
        for i in range(n - 5):
            trinuc = sequence[i:i+3]
            if sequence[i:i+6] == trinuc * 2:
                repeat_positions.update(range(i, min(i+6, n)))

        return len(repeat_positions) / n

    @staticmethod
    def _compute_max_homopolymer(sequence: str) -> int:
        """Compute maximum homopolymer length."""
        if not sequence:
            return 0
        max_len = 1
        current_len = 1
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 1
        return max_len


class ConfidenceValidator:
    """
    Validates confidence and OOD status of predictions.

    Works with PLACE (Post-hoc Laplace And Conformal Estimation) outputs:
    - Epistemic uncertainty (σ_epi) from Laplace approximation
    - Conformal interval width from conformal prediction
    - OOD score from kNN distance
    """

    def __init__(
        self,
        reference_panel: Optional[ReferencePanel],
        thresholds: ValidationThresholds,
        reference_epistemic_stds: Optional[np.ndarray] = None,
        reference_conformal_widths: Optional[np.ndarray] = None,
    ):
        """
        Initialize confidence validator.

        Args:
            reference_panel: Reference panel for OOD detection
            thresholds: Validation thresholds
            reference_epistemic_stds: Reference σ_epi values for percentile threshold
            reference_conformal_widths: Reference conformal widths for percentile threshold
        """
        self.reference = reference_panel
        self.thresholds = thresholds
        self.reference_epistemic_stds = reference_epistemic_stds
        self.reference_conformal_widths = reference_conformal_widths

        # Pre-compute thresholds if reference data available
        self._epistemic_threshold = None
        self._conformal_threshold = None

        if reference_epistemic_stds is not None:
            self._epistemic_threshold = float(np.percentile(
                reference_epistemic_stds,
                thresholds.epistemic_percentile
            ))

        if reference_conformal_widths is not None:
            self._conformal_threshold = float(np.percentile(
                reference_conformal_widths,
                thresholds.conformal_width_percentile
            ))

    def validate(
        self,
        epistemic_std: Optional[float] = None,
        conformal_width: Optional[float] = None,
        ood_score: Optional[float] = None,
        reference_ood_scores: Optional[np.ndarray] = None,
    ) -> ConfidenceValidationResult:
        """
        Validate confidence metrics from PLACE output.

        Args:
            epistemic_std: Epistemic uncertainty (σ_epi from PLACE Laplace)
            conformal_width: Width of conformal interval (upper - lower)
            ood_score: OOD score (e.g., kNN distance)
            reference_ood_scores: Reference OOD scores for percentile computation

        Returns:
            ConfidenceValidationResult
        """
        epistemic_passed = True
        conformal_passed = True
        ood_passed = True

        # Check epistemic uncertainty against P90 threshold
        if epistemic_std is not None:
            if self._epistemic_threshold is not None:
                epistemic_passed = epistemic_std <= self._epistemic_threshold
            elif self.reference_epistemic_stds is not None:
                # Compute percentile on the fly
                percentile = stats.percentileofscore(
                    self.reference_epistemic_stds, epistemic_std
                )
                epistemic_passed = percentile <= self.thresholds.epistemic_percentile

        # Check conformal width against P95 threshold
        if conformal_width is not None:
            if self._conformal_threshold is not None:
                conformal_passed = conformal_width <= self._conformal_threshold
            elif self.reference_conformal_widths is not None:
                percentile = stats.percentileofscore(
                    self.reference_conformal_widths, conformal_width
                )
                conformal_passed = percentile <= self.thresholds.conformal_width_percentile

        # Check OOD score against P95 threshold
        if ood_score is not None:
            if reference_ood_scores is not None:
                threshold = np.percentile(
                    reference_ood_scores,
                    self.thresholds.ood_percentile
                )
                ood_passed = ood_score <= threshold
            elif self.reference is not None and self.reference.knn_model is not None:
                # Use kNN model to check if OOD
                ood_passed = True  # Default pass if no reference scores

        passed = epistemic_passed and conformal_passed and ood_passed

        # Generate detailed message
        if passed:
            parts = ["Confidence check passed"]
            if epistemic_std is not None:
                parts.append(f"σ_epi={epistemic_std:.3f}")
            if conformal_width is not None:
                parts.append(f"CI_width={conformal_width:.3f}")
            if ood_score is not None:
                parts.append(f"OOD={ood_score:.3f}")
            message = " (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else parts[0]
            message = parts[0] + message
        else:
            failures = []
            if not epistemic_passed:
                failures.append(f"σ_epi={epistemic_std:.3f} > P{self.thresholds.epistemic_percentile:.0f}")
            if not conformal_passed:
                failures.append(f"CI_width={conformal_width:.3f} > P{self.thresholds.conformal_width_percentile:.0f}")
            if not ood_passed:
                failures.append(f"OOD={ood_score:.3f} > P{self.thresholds.ood_percentile:.0f}")
            message = f"Confidence failures: {'; '.join(failures)}"

        return ConfidenceValidationResult(
            passed=passed,
            epistemic_std=epistemic_std,
            epistemic_passed=epistemic_passed,
            conformal_width=conformal_width,
            conformal_passed=conformal_passed,
            ood_score=ood_score,
            ood_passed=ood_passed,
            message=message,
        )


class OracleCheckValidator:
    """
    Main validator that combines all checks.

    Integrates:
    - Physics conformity (z-scores, NLL)
    - Composition hygiene (GC, CpG, entropy, repeats)
    - Confidence validation (PLACE epistemic/conformal, OOD)
    - Mahalanobis distance validation
    """

    def __init__(
        self,
        config: OracleCheckConfig,
        reference_panel: ReferencePanel,
        reference_epistemic_stds: Optional[np.ndarray] = None,
        reference_conformal_widths: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.reference = reference_panel
        self.thresholds = config.thresholds

        # Initialize sub-validators
        self.physics_validator = PhysicsValidator(reference_panel, self.thresholds)
        self.composition_validator = CompositionValidator(reference_panel, self.thresholds)
        self.confidence_validator = ConfidenceValidator(
            reference_panel,
            self.thresholds,
            reference_epistemic_stds=reference_epistemic_stds,
            reference_conformal_widths=reference_conformal_widths,
        )
        self.mahalanobis_validator = MahalanobisValidator(reference_panel, self.thresholds)

    def validate_sequence(
        self,
        sequence: str,
        physics_features: Optional[np.ndarray] = None,
        physics_feature_names: Optional[List[str]] = None,
        prediction_mean: Optional[float] = None,
        epistemic_std: Optional[float] = None,
        conformal_width: Optional[float] = None,
        ood_score: Optional[float] = None,
        cadence_features: Optional[np.ndarray] = None,
        reference_ood_scores: Optional[np.ndarray] = None,
        reference_mahal_distances: Optional[np.ndarray] = None,
    ) -> Tuple[Verdict, Dict]:
        """
        Run all validation checks on a sequence.

        Args:
            sequence: DNA sequence
            physics_features: Optional physics features
            physics_feature_names: Names of physics features
            prediction_mean: Predicted activity
            epistemic_std: Epistemic uncertainty (PLACE σ_epi)
            conformal_width: Conformal interval width
            ood_score: OOD score (kNN distance)
            cadence_features: CADENCE backbone features for Mahalanobis
            reference_ood_scores: Reference OOD scores for percentile
            reference_mahal_distances: Reference Mahalanobis distances for percentile

        Returns:
            Tuple of (Verdict, detailed results dict)
        """
        results = {}
        hard_fails = []
        soft_fails = []

        # Composition check
        comp_result = self.composition_validator.validate(sequence)
        results["composition"] = comp_result

        if not comp_result.repeat_passed or not comp_result.homopolymer_passed:
            hard_fails.append("composition")
        elif not comp_result.passed:
            soft_fails.append("composition")

        # Physics check
        if physics_features is not None and physics_feature_names is not None:
            phys_result = self.physics_validator.validate(physics_features, physics_feature_names)
            results["physics"] = phys_result

            if phys_result.max_z_score > self.thresholds.physics_z_hard:
                hard_fails.append("physics")
            elif not phys_result.passed:
                soft_fails.append("physics")

            # Mahalanobis distance on physics features
            mahal_phys_result = self.mahalanobis_validator.validate_physics_features(
                physics_features, physics_feature_names
            )
            results["mahalanobis_physics"] = mahal_phys_result

            if not mahal_phys_result.passed:
                soft_fails.append("mahalanobis_physics")

        # CADENCE Mahalanobis check
        if cadence_features is not None:
            mahal_cadence_result = self.mahalanobis_validator.validate_cadence_features(
                cadence_features, reference_mahal_distances
            )
            results["mahalanobis_cadence"] = mahal_cadence_result

            if not mahal_cadence_result.passed:
                soft_fails.append("mahalanobis_cadence")

        # Confidence check (PLACE epistemic/conformal + OOD)
        conf_result = self.confidence_validator.validate(
            epistemic_std=epistemic_std,
            conformal_width=conformal_width,
            ood_score=ood_score,
            reference_ood_scores=reference_ood_scores,
        )
        results["confidence"] = conf_result

        if not conf_result.ood_passed:
            hard_fails.append("OOD")
        elif not conf_result.passed:
            soft_fails.append("confidence")

        # Store prediction info
        if prediction_mean is not None:
            results["prediction"] = {
                "mean": prediction_mean,
                "epistemic_std": epistemic_std,
                "conformal_width": conformal_width,
            }

        # Determine verdict
        if hard_fails:
            verdict = Verdict.RED
        elif len(soft_fails) > 1:
            verdict = Verdict.RED
        elif len(soft_fails) == 1:
            verdict = Verdict.YELLOW
        else:
            verdict = Verdict.GREEN

        results["verdict"] = verdict
        results["hard_fails"] = hard_fails
        results["soft_fails"] = soft_fails

        return verdict, results
