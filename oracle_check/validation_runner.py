"""
Validation Protocol Runner for OracleCheck

Orchestrates the complete validation protocol:
1. Generate sequences using various methods
2. Compute physics features
3. Get CADENCE+PLACE predictions
4. Run OracleCheck validation
5. Perform statistical comparisons
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import torch

from .config import OracleCheckConfig, Verdict
from .reference_panels import ReferencePanel, ReferencePanelBuilder, TileFormerInterface
from .validators import OracleCheckValidator
from .motif_validator import MotifValidator, get_motif_validator
from .rc_consistency import FullRCValidator
from .two_sample_tests import BatchComparator


@dataclass
class SequenceResult:
    """Result for a single sequence."""
    sequence: str
    prediction_mean: float
    prediction_std: float
    epistemic_std: Optional[float]
    conformal_lower: Optional[float]
    conformal_upper: Optional[float]
    verdict: Verdict
    validation_results: Dict
    motif_results: Optional[Dict] = None
    rc_results: Optional[Dict] = None
    generation_method: str = ""


@dataclass
class ValidationReport:
    """Complete validation report for a batch."""
    n_sequences: int
    n_green: int
    n_yellow: int
    n_red: int
    green_rate: float
    yellow_rate: float
    red_rate: float
    mean_activity: float
    std_activity: float
    top_10_activity: float
    physics_pass_rate: float
    composition_pass_rate: float
    confidence_pass_rate: float
    rc_consistency_rate: float
    motif_pass_rate: float
    two_sample_results: Optional[Dict] = None
    sequence_results: List[SequenceResult] = field(default_factory=list)
    generation_method: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "n_sequences": self.n_sequences,
            "verdicts": {
                "green": self.n_green,
                "yellow": self.n_yellow,
                "red": self.n_red,
                "green_rate": self.green_rate,
                "yellow_rate": self.yellow_rate,
                "red_rate": self.red_rate,
            },
            "activity": {
                "mean": self.mean_activity,
                "std": self.std_activity,
                "top_10": self.top_10_activity,
            },
            "pass_rates": {
                "physics": self.physics_pass_rate,
                "composition": self.composition_pass_rate,
                "confidence": self.confidence_pass_rate,
                "rc_consistency": self.rc_consistency_rate,
                "motif": self.motif_pass_rate,
            },
            "two_sample_results": self.two_sample_results,
            "generation_method": self.generation_method,
            "timestamp": self.timestamp,
        }


class ValidationProtocolRunner:
    """
    Runs the complete OracleCheck validation protocol.

    Protocol:
    1. Generate/load sequences (from optimization, VAE, or natural)
    2. Compute CADENCE+PLACE predictions
    3. Compute physics features (PhysInformer + TileFormer)
    4. Run OracleCheck validation
    5. Run motif validation
    6. Run RC consistency check
    7. Run two-sample tests vs natural reference
    8. Generate report
    """

    def __init__(
        self,
        config: OracleCheckConfig,
        reference_panel: ReferencePanel,
        cadence_interface=None,
        physinformer_interface=None,
        tileformer_interface: Optional[TileFormerInterface] = None,
        motif_validator: Optional[MotifValidator] = None,
    ):
        """
        Initialize validation runner.

        Args:
            config: OracleCheck configuration
            reference_panel: Reference panel from natural sequences
            cadence_interface: CADENCE+PLACE interface
            physinformer_interface: PhysInformer interface
            tileformer_interface: TileFormer interface
            motif_validator: Motif validator
        """
        self.config = config
        self.reference = reference_panel
        self.cadence = cadence_interface
        self.physinformer = physinformer_interface
        self.tileformer = tileformer_interface
        self.motif_validator = motif_validator

        # Initialize validators
        self.validator = OracleCheckValidator(config, reference_panel)
        self.rc_validator = FullRCValidator(
            delta_threshold=config.thresholds.rc_delta_threshold,
            device=config.device,
        )
        self.batch_comparator = BatchComparator()

        # Reference sequences for two-sample tests
        self.reference_sequences: Optional[List[str]] = None
        self.reference_physics: Optional[np.ndarray] = None

    def set_reference_sequences(
        self,
        sequences: List[str],
        physics_features: Optional[np.ndarray] = None,
    ):
        """Set reference sequences for two-sample tests."""
        self.reference_sequences = sequences
        self.reference_physics = physics_features

    def validate_sequences(
        self,
        sequences: List[str],
        generation_method: str = "unknown",
        run_motif: bool = True,
        run_rc: bool = True,
        run_two_sample: bool = True,
        predict_fn=None,
    ) -> ValidationReport:
        """
        Run complete validation on a batch of sequences.

        Args:
            sequences: List of DNA sequences
            generation_method: How sequences were generated
            run_motif: Whether to run motif validation
            run_rc: Whether to run RC consistency check
            run_two_sample: Whether to run two-sample tests
            predict_fn: Optional custom prediction function

        Returns:
            ValidationReport with all results
        """
        n_sequences = len(sequences)
        sequence_results = []

        # Counters for pass rates
        verdicts = {"GREEN": 0, "YELLOW": 0, "RED": 0}
        physics_passed = 0
        composition_passed = 0
        confidence_passed = 0
        rc_passed = 0
        motif_passed = 0

        activities = []
        physics_features_list = []

        print(f"Validating {n_sequences} sequences ({generation_method})...")

        for i, seq in enumerate(sequences):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{n_sequences}")

            # Get predictions
            pred_mean, pred_std = 0.0, 0.0
            epistemic_std, conf_lower, conf_upper = None, None, None
            physics_features = None
            physics_feature_names = None
            cadence_features = None

            if self.cadence is not None:
                # Get CADENCE+PLACE predictions
                pred_result = self.cadence.predict(seq)
                pred_mean = pred_result.get("mean", 0.0)
                pred_std = pred_result.get("aleatoric_std", 0.0)
                epistemic_std = pred_result.get("epistemic_std")
                conf_lower = pred_result.get("conformal_lower")
                conf_upper = pred_result.get("conformal_upper")
                cadence_features = pred_result.get("features")
            elif predict_fn is not None:
                pred_mean = predict_fn(seq)

            activities.append(pred_mean)

            # Get physics features
            if self.physinformer is not None:
                phys_result = self.physinformer.predict(seq)
                physics_features = phys_result.get("features")
                physics_feature_names = phys_result.get("feature_names")
                if physics_features is not None:
                    physics_features_list.append(physics_features)

            # Run main validation
            conformal_width = None
            if conf_lower is not None and conf_upper is not None:
                conformal_width = conf_upper - conf_lower

            verdict, results = self.validator.validate_sequence(
                sequence=seq,
                physics_features=physics_features,
                physics_feature_names=physics_feature_names,
                prediction_mean=pred_mean,
                epistemic_std=epistemic_std,
                conformal_width=conformal_width,
                cadence_features=cadence_features,
            )

            verdicts[verdict.value] += 1

            if "physics" in results and results["physics"].passed:
                physics_passed += 1
            if "composition" in results and results["composition"].passed:
                composition_passed += 1
            if "confidence" in results and results["confidence"].passed:
                confidence_passed += 1

            # Motif validation
            motif_results = None
            if run_motif and self.motif_validator is not None:
                motif_result = self.motif_validator.validate(seq)
                motif_results = {"passed": motif_result.passed, "message": motif_result.message}
                if motif_result.passed:
                    motif_passed += 1

            # RC consistency
            rc_results = None
            if run_rc:
                rc_result = self.rc_validator.validate(
                    seq,
                    run_ism=False,
                    predict_fn=predict_fn if predict_fn else (
                        lambda s: self.cadence.predict(s)["mean"]
                        if self.cadence else 0.0
                    )
                )
                rc_results = {
                    "passed": rc_result["passed"],
                    "rc_delta": rc_result["rc_result"].delta if rc_result["rc_result"] else None,
                }
                if rc_result["passed"]:
                    rc_passed += 1

            # Store result
            sequence_results.append(SequenceResult(
                sequence=seq,
                prediction_mean=pred_mean,
                prediction_std=pred_std,
                epistemic_std=epistemic_std,
                conformal_lower=conf_lower,
                conformal_upper=conf_upper,
                verdict=verdict,
                validation_results=results,
                motif_results=motif_results,
                rc_results=rc_results,
                generation_method=generation_method,
            ))

        # Two-sample tests
        two_sample_results = None
        if run_two_sample and self.reference_sequences is not None:
            print("Running two-sample tests...")
            if physics_features_list and self.reference_physics is not None:
                designed_physics = np.array(physics_features_list)
                two_sample_results = self.batch_comparator.compare_all(
                    designed_physics,
                    self.reference_physics,
                ).to_dict()
                two_sample_results["sequences"] = {
                    "n_designed": len(sequences),
                    "n_reference": len(self.reference_sequences),
                }

        # Compute summary statistics
        activities = np.array(activities)
        top_10_idx = int(len(activities) * 0.9)
        sorted_activities = np.sort(activities)

        report = ValidationReport(
            n_sequences=n_sequences,
            n_green=verdicts["GREEN"],
            n_yellow=verdicts["YELLOW"],
            n_red=verdicts["RED"],
            green_rate=verdicts["GREEN"] / n_sequences,
            yellow_rate=verdicts["YELLOW"] / n_sequences,
            red_rate=verdicts["RED"] / n_sequences,
            mean_activity=float(np.mean(activities)),
            std_activity=float(np.std(activities)),
            top_10_activity=float(sorted_activities[top_10_idx]) if len(sorted_activities) > top_10_idx else 0.0,
            physics_pass_rate=physics_passed / n_sequences if n_sequences > 0 else 0.0,
            composition_pass_rate=composition_passed / n_sequences if n_sequences > 0 else 0.0,
            confidence_pass_rate=confidence_passed / n_sequences if n_sequences > 0 else 0.0,
            rc_consistency_rate=rc_passed / n_sequences if run_rc and n_sequences > 0 else 1.0,
            motif_pass_rate=motif_passed / n_sequences if run_motif and n_sequences > 0 else 1.0,
            two_sample_results=two_sample_results,
            sequence_results=sequence_results,
            generation_method=generation_method,
            timestamp=datetime.now().isoformat(),
        )

        return report

    def run_protocol(
        self,
        generated_sequences: Dict[str, List[str]],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, ValidationReport]:
        """
        Run the full validation protocol on multiple generation methods.

        Args:
            generated_sequences: Dict mapping method name to sequences
            output_dir: Optional directory to save results

        Returns:
            Dict mapping method name to ValidationReport
        """
        results = {}

        for method, sequences in generated_sequences.items():
            print(f"\n{'='*60}")
            print(f"Validating: {method}")
            print(f"{'='*60}")

            report = self.validate_sequences(
                sequences,
                generation_method=method,
            )
            results[method] = report

            # Print summary
            print(f"\nSummary for {method}:")
            print(f"  Verdicts: GREEN={report.n_green}, YELLOW={report.n_yellow}, RED={report.n_red}")
            print(f"  Green rate: {report.green_rate:.1%}")
            print(f"  Mean activity: {report.mean_activity:.3f}")
            print(f"  Top 10% activity: {report.top_10_activity:.3f}")

        # Save results
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            summary = {
                method: report.to_dict()
                for method, report in results.items()
            }

            with open(output_dir / "validation_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            print(f"\nResults saved to {output_dir}")

        return results

    def compare_methods(
        self,
        reports: Dict[str, ValidationReport],
    ) -> Dict:
        """
        Compare validation results across different generation methods.

        Args:
            reports: Dict mapping method name to ValidationReport

        Returns:
            Comparison results
        """
        comparison = {
            "methods": list(reports.keys()),
            "green_rates": {},
            "mean_activities": {},
            "top_10_activities": {},
            "physics_pass_rates": {},
            "rc_consistency_rates": {},
        }

        for method, report in reports.items():
            comparison["green_rates"][method] = report.green_rate
            comparison["mean_activities"][method] = report.mean_activity
            comparison["top_10_activities"][method] = report.top_10_activity
            comparison["physics_pass_rates"][method] = report.physics_pass_rate
            comparison["rc_consistency_rates"][method] = report.rc_consistency_rate

        # Find best method by green rate
        best_method = max(comparison["green_rates"], key=comparison["green_rates"].get)
        comparison["best_method"] = best_method
        comparison["best_green_rate"] = comparison["green_rates"][best_method]

        return comparison


def create_runner(
    cell_type: str = "K562",
    config: Optional[OracleCheckConfig] = None,
) -> ValidationProtocolRunner:
    """
    Create a validation runner for a specific cell type.

    Args:
        cell_type: Cell type (K562, HepG2, WTC11, S2)
        config: Optional custom config

    Returns:
        ValidationProtocolRunner
    """
    if config is None:
        config = OracleCheckConfig()

    # Build reference panel
    builder = ReferencePanelBuilder(config)

    # Try to load existing panel or build new one
    panel_path = config.reference_panels_dir / cell_type
    reference_panel = None

    if panel_path.exists():
        print(f"Loading reference panel from {panel_path}")
        try:
            reference_panel = ReferencePanel.load(panel_path)
        except Exception as e:
            print(f"Failed to load reference panel: {e}")
            # Delete corrupted panel
            import shutil
            shutil.rmtree(panel_path, ignore_errors=True)

    if reference_panel is None:
        print(f"Building reference panel for {cell_type}...")
        try:
            reference_panel = builder.build_from_lentiMPRA(cell_type)
            panel_path.mkdir(parents=True, exist_ok=True)
            reference_panel.save(panel_path)
            print(f"Reference panel saved to {panel_path}")
        except Exception as e:
            print(f"Could not build reference panel: {e}")
            import traceback
            traceback.print_exc()
            reference_panel = ReferencePanel(cell_type=cell_type)
            print("Using empty reference panel")

    # Get motif validator
    motif_validator = get_motif_validator(cell_type=cell_type)

    # Initialize TileFormer if available
    tileformer = None
    if config.tileformer_checkpoint is not None:
        tileformer = TileFormerInterface(
            checkpoint_path=config.tileformer_checkpoint,
            device=config.device,
        )

    runner = ValidationProtocolRunner(
        config=config,
        reference_panel=reference_panel,
        tileformer_interface=tileformer,
        motif_validator=motif_validator,
    )

    return runner
