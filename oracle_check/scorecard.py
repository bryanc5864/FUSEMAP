"""
Scorecard for OracleCheck

Generates comprehensive scorecards and verdicts for sequence validation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

from .config import Verdict


@dataclass
class SequenceScorecard:
    """
    Complete scorecard for a single sequence.
    """

    sequence: str
    sequence_id: Optional[str] = None

    # Verdict
    verdict: Verdict = Verdict.RED

    # Activity
    activity_mean: Optional[float] = None
    activity_panel_iqr: Optional[float] = None

    # Uncertainty
    aleatoric_std: Optional[float] = None
    epistemic_std: Optional[float] = None
    conformal_width: Optional[float] = None
    ood_score: Optional[float] = None

    # Physics
    physics_max_z: Optional[float] = None
    physics_nll: Optional[float] = None
    physics_family_scores: Dict[str, float] = field(default_factory=dict)
    physics_flags: List[str] = field(default_factory=list)

    # Composition
    gc_content: Optional[float] = None
    cpg_oe: Optional[float] = None
    repeat_fraction: Optional[float] = None
    entropy: Optional[float] = None
    max_homopolymer: Optional[int] = None

    # Detailed check results
    physics_passed: bool = False
    composition_passed: bool = False
    confidence_passed: bool = False
    naturality_passed: bool = False

    # Hard/soft failures
    hard_failures: List[str] = field(default_factory=list)
    soft_failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert scorecard to dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "sequence": self.sequence,
            "verdict": self.verdict.value,
            "activity": {
                "mean": self.activity_mean,
                "panel_iqr": self.activity_panel_iqr,
            },
            "uncertainty": {
                "aleatoric_std": self.aleatoric_std,
                "epistemic_std": self.epistemic_std,
                "conformal_width": self.conformal_width,
                "ood_score": self.ood_score,
            },
            "physics": {
                "max_z": self.physics_max_z,
                "nll": self.physics_nll,
                "family_scores": self.physics_family_scores,
                "flags": self.physics_flags,
                "passed": self.physics_passed,
            },
            "composition": {
                "gc_content": self.gc_content,
                "cpg_oe": self.cpg_oe,
                "repeat_fraction": self.repeat_fraction,
                "entropy": self.entropy,
                "max_homopolymer": self.max_homopolymer,
                "passed": self.composition_passed,
            },
            "confidence_passed": self.confidence_passed,
            "naturality_passed": self.naturality_passed,
            "hard_failures": self.hard_failures,
            "soft_failures": self.soft_failures,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Sequence: {self.sequence_id or self.sequence[:20]}...",
            f"Verdict: {self.verdict.value}",
            f"Activity: {self.activity_mean:.3f}" if self.activity_mean else "Activity: N/A",
        ]

        if self.physics_max_z is not None:
            lines.append(f"Physics max z: {self.physics_max_z:.2f}")

        if self.gc_content is not None:
            lines.append(f"GC: {self.gc_content:.2%}")

        if self.ood_score is not None:
            lines.append(f"OOD score: {self.ood_score:.3f}")

        if self.hard_failures:
            lines.append(f"Hard failures: {', '.join(self.hard_failures)}")
        if self.soft_failures:
            lines.append(f"Soft failures: {', '.join(self.soft_failures)}")

        return "\n".join(lines)


@dataclass
class BatchScorecard:
    """
    Scorecard for a batch of sequences.
    """

    name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Individual scorecards
    scorecards: List[SequenceScorecard] = field(default_factory=list)

    # Aggregate statistics
    n_total: int = 0
    n_green: int = 0
    n_yellow: int = 0
    n_red: int = 0

    # Pass rates
    physics_pass_rate: float = 0.0
    composition_pass_rate: float = 0.0
    confidence_pass_rate: float = 0.0
    overall_pass_rate: float = 0.0

    # Distribution statistics
    activity_mean: float = 0.0
    activity_std: float = 0.0
    physics_z_mean: float = 0.0
    physics_z_max: float = 0.0
    ood_score_mean: float = 0.0
    ood_score_p95: float = 0.0

    def compute_statistics(self):
        """Compute aggregate statistics from individual scorecards."""
        if not self.scorecards:
            return

        self.n_total = len(self.scorecards)
        self.n_green = sum(1 for s in self.scorecards if s.verdict == Verdict.GREEN)
        self.n_yellow = sum(1 for s in self.scorecards if s.verdict == Verdict.YELLOW)
        self.n_red = sum(1 for s in self.scorecards if s.verdict == Verdict.RED)

        # Pass rates
        self.physics_pass_rate = sum(1 for s in self.scorecards if s.physics_passed) / self.n_total
        self.composition_pass_rate = sum(1 for s in self.scorecards if s.composition_passed) / self.n_total
        self.confidence_pass_rate = sum(1 for s in self.scorecards if s.confidence_passed) / self.n_total
        self.overall_pass_rate = self.n_green / self.n_total

        # Activity statistics
        activities = [s.activity_mean for s in self.scorecards if s.activity_mean is not None]
        if activities:
            self.activity_mean = np.mean(activities)
            self.activity_std = np.std(activities)

        # Physics statistics
        physics_z = [s.physics_max_z for s in self.scorecards if s.physics_max_z is not None]
        if physics_z:
            self.physics_z_mean = np.mean(physics_z)
            self.physics_z_max = np.max(physics_z)

        # OOD statistics
        ood_scores = [s.ood_score for s in self.scorecards if s.ood_score is not None]
        if ood_scores:
            self.ood_score_mean = np.mean(ood_scores)
            self.ood_score_p95 = np.percentile(ood_scores, 95)

    def summary(self) -> str:
        """Generate human-readable summary."""
        self.compute_statistics()

        lines = [
            f"=== Batch Scorecard: {self.name} ===",
            f"Timestamp: {self.timestamp}",
            f"",
            f"Verdicts:",
            f"  GREEN:  {self.n_green:4d} ({100*self.n_green/self.n_total:.1f}%)",
            f"  YELLOW: {self.n_yellow:4d} ({100*self.n_yellow/self.n_total:.1f}%)",
            f"  RED:    {self.n_red:4d} ({100*self.n_red/self.n_total:.1f}%)",
            f"",
            f"Pass Rates:",
            f"  Physics:     {100*self.physics_pass_rate:.1f}%",
            f"  Composition: {100*self.composition_pass_rate:.1f}%",
            f"  Confidence:  {100*self.confidence_pass_rate:.1f}%",
            f"  Overall:     {100*self.overall_pass_rate:.1f}%",
            f"",
            f"Activity: {self.activity_mean:.3f} +/- {self.activity_std:.3f}",
            f"Physics z: mean={self.physics_z_mean:.2f}, max={self.physics_z_max:.2f}",
            f"OOD: mean={self.ood_score_mean:.3f}, p95={self.ood_score_p95:.3f}",
        ]

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        self.compute_statistics()

        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "n_total": self.n_total,
            "n_green": self.n_green,
            "n_yellow": self.n_yellow,
            "n_red": self.n_red,
            "pass_rates": {
                "physics": self.physics_pass_rate,
                "composition": self.composition_pass_rate,
                "confidence": self.confidence_pass_rate,
                "overall": self.overall_pass_rate,
            },
            "statistics": {
                "activity_mean": self.activity_mean,
                "activity_std": self.activity_std,
                "physics_z_mean": self.physics_z_mean,
                "physics_z_max": self.physics_z_max,
                "ood_score_mean": self.ood_score_mean,
                "ood_score_p95": self.ood_score_p95,
            },
            "scorecards": [s.to_dict() for s in self.scorecards],
        }


class ScorecardBuilder:
    """
    Builder for creating scorecards from validation results.
    """

    def build_sequence_scorecard(
        self,
        sequence: str,
        sequence_id: Optional[str] = None,
        prediction=None,  # CADENCEPrediction
        physics_result=None,  # PhysicsValidationResult
        composition_result=None,  # CompositionValidationResult
        confidence_result=None,  # ConfidenceValidationResult
        verdict: Verdict = Verdict.RED,
        hard_failures: List[str] = None,
        soft_failures: List[str] = None,
    ) -> SequenceScorecard:
        """
        Build a scorecard from validation results.

        Args:
            sequence: DNA sequence
            sequence_id: Optional identifier
            prediction: CADENCE prediction result
            physics_result: Physics validation result
            composition_result: Composition validation result
            confidence_result: Confidence validation result
            verdict: Final verdict
            hard_failures: List of hard failures
            soft_failures: List of soft failures

        Returns:
            SequenceScorecard
        """
        scorecard = SequenceScorecard(
            sequence=sequence,
            sequence_id=sequence_id,
            verdict=verdict,
            hard_failures=hard_failures or [],
            soft_failures=soft_failures or [],
        )

        # Fill from prediction
        if prediction is not None:
            scorecard.activity_mean = float(prediction.mean[0]) if hasattr(prediction.mean, '__len__') else float(prediction.mean)
            scorecard.aleatoric_std = float(prediction.aleatoric_std[0]) if hasattr(prediction.aleatoric_std, '__len__') else float(prediction.aleatoric_std)
            if prediction.epistemic_std is not None:
                scorecard.epistemic_std = float(prediction.epistemic_std[0]) if hasattr(prediction.epistemic_std, '__len__') else float(prediction.epistemic_std)
            if prediction.conformal_upper is not None and prediction.conformal_lower is not None:
                width = prediction.conformal_upper[0] - prediction.conformal_lower[0]
                scorecard.conformal_width = float(width)
            if prediction.ood_score is not None:
                scorecard.ood_score = float(prediction.ood_score[0]) if hasattr(prediction.ood_score, '__len__') else float(prediction.ood_score)

        # Fill from physics result
        if physics_result is not None:
            scorecard.physics_max_z = physics_result.max_z_score
            scorecard.physics_nll = physics_result.nll
            scorecard.physics_family_scores = physics_result.family_scores
            scorecard.physics_passed = physics_result.passed
            scorecard.physics_flags = [f for f, p in physics_result.family_passed.items() if not p]

        # Fill from composition result
        if composition_result is not None:
            scorecard.gc_content = composition_result.gc_content
            scorecard.cpg_oe = composition_result.cpg_oe
            scorecard.entropy = composition_result.entropy
            scorecard.repeat_fraction = composition_result.repeat_fraction
            scorecard.max_homopolymer = composition_result.max_homopolymer
            scorecard.composition_passed = composition_result.passed

        # Fill from confidence result
        if confidence_result is not None:
            scorecard.confidence_passed = confidence_result.passed
            if confidence_result.ood_score is not None:
                scorecard.ood_score = confidence_result.ood_score

        return scorecard
