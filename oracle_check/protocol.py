"""
Validation Protocol for OracleCheck

Orchestrates the full validation pipeline:
1. Load CADENCE model with PLACE uncertainty
2. Load PhysInformer and TileFormer
3. Build reference panels from natural high performers
4. Run validation on sequences
5. Generate scorecards

Human datasets only: K562, HepG2, WTC11
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json
from datetime import datetime
import logging

from .config import OracleCheckConfig, ValidationThresholds, Verdict
from .cadence_interface import CADENCEInterface, CADENCEPrediction
from .physics_interface import MultiCellTypePhysInformer, PhysicsFeatures
from .tileformer_interface import TileFormerInterface, ElectrostaticsFeatures
from .reference_panels import ReferencePanel, ReferencePanelBuilder
from .validators import (
    OracleCheckValidator,
    PhysicsValidationResult,
    CompositionValidationResult,
    ConfidenceValidationResult,
)
from .scorecard import SequenceScorecard, BatchScorecard, ScorecardBuilder


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Complete validation result for a sequence."""
    sequence: str
    sequence_id: Optional[str]
    verdict: Verdict
    scorecard: SequenceScorecard
    prediction: CADENCEPrediction
    physics_result: Optional[PhysicsValidationResult]
    composition_result: CompositionValidationResult
    confidence_result: ConfidenceValidationResult


class OracleCheckProtocol:
    """
    Main validation protocol for OracleCheck.

    Validates sequences using:
    - CADENCE activity predictions with PLACE uncertainty
    - PhysInformer physics features (cell-type specific)
    - TileFormer electrostatics (universal)
    - Reference panels from natural high performers
    """

    # Supported human cell types
    HUMAN_CELL_TYPES = ["K562", "HepG2", "WTC11"]

    def __init__(
        self,
        config: Optional[OracleCheckConfig] = None,
        cadence_model_name: str = "config2_multi_celltype_v1",
        device: str = "cuda",
    ):
        """
        Initialize OracleCheck protocol.

        Args:
            config: OracleCheck configuration
            cadence_model_name: Name of CADENCE model to use
            device: Device for inference
        """
        self.config = config or OracleCheckConfig()
        self.device = device
        self.cadence_model_name = cadence_model_name

        # Interfaces (lazy loaded)
        self._cadence: Optional[CADENCEInterface] = None
        self._physinformer: Optional[MultiCellTypePhysInformer] = None
        self._tileformer: Optional[TileFormerInterface] = None

        # Reference panels per cell type
        self._reference_panels: Dict[str, ReferencePanel] = {}

        # Validators per cell type
        self._validators: Dict[str, OracleCheckValidator] = {}

        # Scorecard builder
        self.scorecard_builder = ScorecardBuilder()

        logger.info(f"Initialized OracleCheckProtocol with model: {cadence_model_name}")

    @property
    def cadence(self) -> CADENCEInterface:
        """Get CADENCE interface (lazy load)."""
        if self._cadence is None:
            self._cadence = CADENCEInterface(
                model_name=self.cadence_model_name,
                device=self.device,
            )
        return self._cadence

    @property
    def physinformer(self) -> MultiCellTypePhysInformer:
        """Get PhysInformer interface (lazy load)."""
        if self._physinformer is None:
            self._physinformer = MultiCellTypePhysInformer(
                checkpoint_dir=self.config.physinformer_runs_dir,
                cell_types=self.HUMAN_CELL_TYPES,
                device=self.device,
            )
        return self._physinformer

    @property
    def tileformer(self) -> TileFormerInterface:
        """Get TileFormer interface (lazy load)."""
        if self._tileformer is None:
            self._tileformer = TileFormerInterface(device=self.device)
        return self._tileformer

    def load_reference_panel(
        self,
        cell_type: str,
        rebuild: bool = False,
    ) -> ReferencePanel:
        """
        Load or build reference panel for a cell type.

        Args:
            cell_type: Cell type (K562, HepG2, WTC11)
            rebuild: Whether to rebuild even if cached

        Returns:
            ReferencePanel
        """
        if cell_type not in self.HUMAN_CELL_TYPES:
            raise ValueError(f"Unsupported cell type: {cell_type}. Must be one of {self.HUMAN_CELL_TYPES}")

        # Check cache
        if cell_type in self._reference_panels and not rebuild:
            return self._reference_panels[cell_type]

        # Check saved panel
        panel_path = self.config.reference_panels_dir / cell_type
        if panel_path.exists() and not rebuild:
            logger.info(f"Loading reference panel from {panel_path}")
            panel = ReferencePanel.load(panel_path)
            self._reference_panels[cell_type] = panel
            return panel

        # Build new panel
        logger.info(f"Building reference panel for {cell_type}")
        panel = self._build_reference_panel(cell_type)

        # Save
        panel.save(panel_path)
        self._reference_panels[cell_type] = panel

        return panel

    def _build_reference_panel(self, cell_type: str) -> ReferencePanel:
        """Build reference panel from lentiMPRA data."""
        # Load lentiMPRA data - structure is lentiMPRA_data/{cell_type}/{cell_type}_test_with_features.tsv
        data_path = self.config.lentiMPRA_data_dir / cell_type / f"{cell_type}_test_with_features.tsv"
        if not data_path.exists():
            # Try alternative paths
            alt_paths = [
                self.config.lentiMPRA_data_dir / cell_type / f"{cell_type}_calibration_with_features.tsv",
                self.config.lentiMPRA_data_dir / f"lentiMPRA_{cell_type}_test.tsv",
                self.config.lentiMPRA_data_dir / f"{cell_type}_test.tsv",
            ]
            for alt in alt_paths:
                if alt.exists():
                    data_path = alt
                    break

        if not data_path.exists():
            raise FileNotFoundError(f"lentiMPRA data not found for {cell_type}. Tried: {data_path}")

        df = pd.read_csv(data_path, sep="\t")

        # Get sequences and activities
        sequences = df["sequence"].tolist()
        activities = df["activity"].values if "activity" in df.columns else df["mean_activity"].values

        # Get physics features if available
        physics_features = None
        physics_feature_names = None

        try:
            phys_result = self.physinformer.predict(sequences[:100], cell_type)  # Sample for speed
            # For full panel, would need all sequences
        except Exception as e:
            logger.warning(f"Could not get physics features: {e}")

        # Get CADENCE features
        cadence_features = None
        try:
            cadence_features = self.cadence.get_features(
                sequences,
                cell_type=cell_type,
            )
        except Exception as e:
            logger.warning(f"Could not get CADENCE features: {e}")

        # Build panel
        builder = ReferencePanelBuilder(self.config)
        panel = builder.build_from_data(
            sequences=sequences,
            activities=activities,
            cell_type=cell_type,
            cadence_features=cadence_features,
        )

        return panel

    def get_validator(self, cell_type: str) -> OracleCheckValidator:
        """Get validator for a cell type."""
        if cell_type not in self._validators:
            panel = self.load_reference_panel(cell_type)
            self._validators[cell_type] = OracleCheckValidator(
                config=self.config,
                reference_panel=panel,
            )
        return self._validators[cell_type]

    def validate_sequence(
        self,
        sequence: str,
        cell_type: str,
        sequence_id: Optional[str] = None,
        include_physics: bool = True,
    ) -> ValidationResult:
        """
        Validate a single sequence.

        Args:
            sequence: DNA sequence
            cell_type: Target cell type
            sequence_id: Optional identifier
            include_physics: Whether to include physics validation

        Returns:
            ValidationResult
        """
        # Get prediction from CADENCE
        prediction = self.cadence.predict(
            [sequence],
            cell_type=cell_type,
        )[0]  # Single sequence

        # Get physics features if requested
        physics_features = None
        physics_feature_names = None
        if include_physics:
            try:
                phys_result = self.physinformer.predict([sequence], cell_type)
                physics_features = phys_result.features[0]
                physics_feature_names = phys_result.feature_names
            except Exception as e:
                logger.warning(f"Physics prediction failed: {e}")

        # Get validator
        validator = self.get_validator(cell_type)

        # Run validation
        verdict, results = validator.validate_sequence(
            sequence=sequence,
            physics_features=physics_features,
            physics_feature_names=physics_feature_names,
            prediction_mean=prediction.mean,
            epistemic_std=prediction.epistemic_std,
            conformal_width=(
                prediction.conformal_upper - prediction.conformal_lower
                if prediction.conformal_upper is not None
                else None
            ),
            ood_score=prediction.ood_score,
        )

        # Build scorecard
        scorecard = self.scorecard_builder.build_sequence_scorecard(
            sequence=sequence,
            sequence_id=sequence_id,
            prediction=prediction,
            physics_result=results.get("physics"),
            composition_result=results.get("composition"),
            confidence_result=results.get("confidence"),
            verdict=verdict,
            hard_failures=results.get("hard_fails", []),
            soft_failures=results.get("soft_fails", []),
        )

        return ValidationResult(
            sequence=sequence,
            sequence_id=sequence_id,
            verdict=verdict,
            scorecard=scorecard,
            prediction=prediction,
            physics_result=results.get("physics"),
            composition_result=results["composition"],
            confidence_result=results["confidence"],
        )

    def validate_batch(
        self,
        sequences: List[str],
        cell_type: str,
        sequence_ids: Optional[List[str]] = None,
        batch_name: str = "batch",
        include_physics: bool = True,
        batch_size: int = 32,
    ) -> BatchScorecard:
        """
        Validate a batch of sequences.

        Args:
            sequences: List of DNA sequences
            cell_type: Target cell type
            sequence_ids: Optional list of identifiers
            batch_name: Name for the batch
            include_physics: Whether to include physics validation
            batch_size: Batch size for processing

        Returns:
            BatchScorecard with all results
        """
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(sequences))]

        batch_scorecard = BatchScorecard(name=batch_name)

        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            batch_ids = sequence_ids[i:i+batch_size]

            # Get predictions
            predictions = self.cadence.predict(batch_seqs, cell_type=cell_type)

            # Get physics features
            physics_features = None
            physics_feature_names = None
            if include_physics:
                try:
                    phys_result = self.physinformer.predict(batch_seqs, cell_type)
                    physics_features = phys_result.features
                    physics_feature_names = phys_result.feature_names
                except Exception as e:
                    logger.warning(f"Physics prediction failed for batch: {e}")

            # Validate each sequence
            validator = self.get_validator(cell_type)

            for j, (seq, seq_id, pred) in enumerate(zip(batch_seqs, batch_ids, predictions)):
                phys_feat = physics_features[j] if physics_features is not None else None

                verdict, results = validator.validate_sequence(
                    sequence=seq,
                    physics_features=phys_feat,
                    physics_feature_names=physics_feature_names,
                    prediction_mean=pred.mean,
                    epistemic_std=pred.epistemic_std,
                    conformal_width=(
                        pred.conformal_upper - pred.conformal_lower
                        if pred.conformal_upper is not None
                        else None
                    ),
                    ood_score=pred.ood_score,
                )

                scorecard = self.scorecard_builder.build_sequence_scorecard(
                    sequence=seq,
                    sequence_id=seq_id,
                    prediction=pred,
                    physics_result=results.get("physics"),
                    composition_result=results.get("composition"),
                    confidence_result=results.get("confidence"),
                    verdict=verdict,
                    hard_failures=results.get("hard_fails", []),
                    soft_failures=results.get("soft_fails", []),
                )

                batch_scorecard.scorecards.append(scorecard)

            logger.info(f"Processed {min(i+batch_size, len(sequences))}/{len(sequences)} sequences")

        # Compute statistics
        batch_scorecard.compute_statistics()

        return batch_scorecard

    def run_validation_protocol(
        self,
        cell_types: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        n_samples: Optional[int] = None,
    ) -> Dict[str, BatchScorecard]:
        """
        Run full validation protocol on human datasets.

        Args:
            cell_types: Cell types to validate (default: all human)
            output_dir: Directory to save results
            n_samples: Number of samples per cell type (None for all)

        Returns:
            Dict of cell_type -> BatchScorecard
        """
        if cell_types is None:
            cell_types = self.HUMAN_CELL_TYPES

        if output_dir is None:
            output_dir = self.config.output_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for cell_type in cell_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Validating {cell_type}")
            logger.info(f"{'='*50}")

            # Load test data - structure is lentiMPRA_data/{cell_type}/{cell_type}_test_with_features.tsv
            data_path = self.config.lentiMPRA_data_dir / cell_type / f"{cell_type}_test_with_features.tsv"
            if not data_path.exists():
                # Try calibration set
                data_path = self.config.lentiMPRA_data_dir / cell_type / f"{cell_type}_calibration_with_features.tsv"
            if not data_path.exists():
                logger.warning(f"Data not found: {data_path}")
                continue

            df = pd.read_csv(data_path, sep="\t")

            if n_samples is not None:
                df = df.sample(n=min(n_samples, len(df)), random_state=42)

            sequences = df["sequence"].tolist()
            sequence_ids = df["name"].tolist() if "name" in df.columns else None

            # Run validation
            batch_scorecard = self.validate_batch(
                sequences=sequences,
                cell_type=cell_type,
                sequence_ids=sequence_ids,
                batch_name=f"lentiMPRA_{cell_type}",
            )

            results[cell_type] = batch_scorecard

            # Print summary
            print(f"\n{batch_scorecard.summary()}")

            # Save results
            cell_output = output_dir / cell_type
            cell_output.mkdir(exist_ok=True)

            with open(cell_output / "batch_scorecard.json", "w") as f:
                json.dump(batch_scorecard.to_dict(), f, indent=2)

            # Save individual scorecards as TSV
            scorecard_data = []
            for sc in batch_scorecard.scorecards:
                scorecard_data.append({
                    "sequence_id": sc.sequence_id,
                    "verdict": sc.verdict.value,
                    "activity_mean": sc.activity_mean,
                    "physics_max_z": sc.physics_max_z,
                    "gc_content": sc.gc_content,
                    "ood_score": sc.ood_score,
                    "physics_passed": sc.physics_passed,
                    "composition_passed": sc.composition_passed,
                    "confidence_passed": sc.confidence_passed,
                })

            pd.DataFrame(scorecard_data).to_csv(
                cell_output / "scorecards.tsv",
                sep="\t",
                index=False,
            )

        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "cadence_model": self.cadence_model_name,
            "cell_types": cell_types,
            "results": {
                ct: {
                    "n_total": res.n_total,
                    "n_green": res.n_green,
                    "n_yellow": res.n_yellow,
                    "n_red": res.n_red,
                    "overall_pass_rate": res.overall_pass_rate,
                }
                for ct, res in results.items()
            }
        }

        with open(output_dir / "validation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nResults saved to {output_dir}")

        return results


def run_human_validation(
    cadence_model: str = "config2_multi_celltype_v1",
    n_samples: Optional[int] = 100,
    output_dir: Optional[str] = None,
):
    """
    Convenience function to run validation on human datasets.

    Args:
        cadence_model: CADENCE model to use
        n_samples: Number of samples per cell type (None for all)
        output_dir: Output directory
    """
    logging.basicConfig(level=logging.INFO)

    protocol = OracleCheckProtocol(
        cadence_model_name=cadence_model,
        device="cuda",
    )

    results = protocol.run_validation_protocol(
        cell_types=["K562", "HepG2", "WTC11"],
        n_samples=n_samples,
        output_dir=Path(output_dir) if output_dir else None,
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OracleCheck validation protocol")
    parser.add_argument("--model", type=str, default="config2_multi_celltype_v1",
                       help="CADENCE model name")
    parser.add_argument("--n-samples", type=int, default=None,
                       help="Number of samples per cell type")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--cell-types", type=str, nargs="+", default=None,
                       help="Cell types to validate")

    args = parser.parse_args()

    run_human_validation(
        cadence_model=args.model,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
    )
