"""
OracleCheck Configuration

Defines configuration classes for validation thresholds, physics families,
and oracle check parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum


class Verdict(str, Enum):
    """Validation verdict levels."""
    GREEN = "GREEN"   # All checks pass
    YELLOW = "YELLOW"  # Minor drift, at most one soft failure
    RED = "RED"       # Any hard failure


# Physics feature families for grouping
PHYSICS_FAMILIES = {
    "thermodynamics": [
        "thermo_dG_p25", "thermo_dG_p50", "thermo_dG_p75",
        "thermo_dG_mean", "thermo_dG_std", "thermo_dG_range",
        "thermo_Tm_estimate", "thermo_stability_island_count",
        "thermo_low_stability_fraction",
    ],
    "shape_mgw_prot_roll": [
        "mgw_mean", "mgw_std", "mgw_min", "mgw_max",
        "prot_mean", "prot_std", "prot_min", "prot_max",
        "roll_mean", "roll_std", "roll_min", "roll_max",
    ],
    "bending_stiffness": [
        "twist_stiffness_mean", "twist_stiffness_std",
        "tilt_stiffness_mean", "tilt_stiffness_std",
        "roll_stiffness_mean", "roll_stiffness_std",
        "stiffness_anisotropy", "periodicity_10bp_signal",
        "total_bending_energy", "rms_curvature", "curvature_hotspot_count",
    ],
    "stacking": [
        "stacking_energy_mean", "stacking_energy_std",
        "stacking_energy_min", "stacking_energy_max",
    ],
    "sidd_g4": [
        "sidd_destabilization_energy", "sidd_destabilization_count",
        "g4_propensity_score", "g4_quadruplex_count",
    ],
    "entropy": [
        "sequence_entropy", "local_entropy_mean", "local_entropy_std",
    ],
}

# Electrostatics features from TileFormer
ELECTROSTATICS_FEATURES = [
    "electrostatic_potential_mean", "electrostatic_potential_std",
    "electrostatic_potential_min", "electrostatic_potential_max",
    "minor_groove_electrostatics",
]


@dataclass
class ValidationThresholds:
    """Thresholds for validation checks."""

    # Physics conformity
    physics_z_soft: float = 2.5       # Soft threshold for per-family z-scores
    physics_z_hard: float = 4.0       # Hard failure threshold
    physics_nll_percentile: float = 95.0  # NLL must be below this percentile

    # Composition
    gc_min: float = 0.35              # Minimum GC content
    gc_max: float = 0.75              # Maximum GC content
    cpg_percentile_low: float = 5.0   # CpG O/E low percentile
    cpg_percentile_high: float = 95.0 # CpG O/E high percentile
    entropy_percentile_low: float = 5.0  # Entropy low percentile
    repeat_fraction_hard: float = 0.3    # Hard failure for repeat fraction
    max_homopolymer_hard: int = 10       # Hard failure for homopolymer length

    # Confidence / OOD
    epistemic_percentile: float = 90.0  # Epistemic uncertainty threshold
    conformal_width_percentile: float = 95.0  # Conformal width threshold
    ood_percentile: float = 95.0        # OOD score threshold

    # MicroMotif / Syntax
    syntax_violation_hard: int = 5      # Max syntax violations before hard fail
    run_len_max_soft: int = 20          # Soft threshold for run length

    # Ensemble
    panel_iqr_threshold: float = 0.5    # Max IQR across ensemble predictions

    # RC Consistency
    rc_delta_threshold: float = 0.1     # Max prediction difference for RC


@dataclass
class OracleCheckConfig:
    """Main configuration for OracleCheck validation."""

    # Paths to trained models
    cadence_models_dir: Path = field(
        default_factory=lambda: Path("/home/bcheng/sequence_optimization/FUSEMAP/cadence_place")
    )
    physinformer_runs_dir: Path = field(
        default_factory=lambda: Path("/home/bcheng/sequence_optimization/mainproject/PhysiFormer/physpreprocess/PhysInformer/runs")
    )
    tileformer_checkpoint: Optional[Path] = field(
        default_factory=lambda: Path("/home/bcheng/sequence_optimization/mainproject/PhysiFormer/physpreprocess/TileFormer_model/checkpoints")
    )

    # Data paths
    lentiMPRA_data_dir: Path = field(
        default_factory=lambda: Path("/home/bcheng/sequence_optimization/mainproject/PhysiFormer/physpreprocess/data/lentiMPRA_data")
    )

    # Reference panels and output
    reference_panels_dir: Path = field(
        default_factory=lambda: Path("/home/bcheng/sequence_optimization/FUSEMAP/oracle_check/reference_panels")
    )
    output_dir: Path = field(
        default_factory=lambda: Path("/home/bcheng/sequence_optimization/FUSEMAP/oracle_check/results")
    )

    # PhysInformer model mapping (cell_type -> run directory name)
    physinformer_models: Dict[str, str] = field(default_factory=lambda: {
        "K562": "K562_20250829_095741",
        "HepG2": "HepG2_20250829_095749",
        "WTC11": "WTC11_20250829_095738",
    })

    # CADENCE model mapping (dataset -> model name in cadence_place)
    cadence_models: Dict[str, str] = field(default_factory=lambda: {
        "encode4_k562": "config2_multi_celltype_v1",
        "encode4_hepg2": "config2_multi_celltype_v1",
        "encode4_wtc11": "config2_multi_celltype_v1",
    })

    # Validation thresholds
    thresholds: ValidationThresholds = field(default_factory=ValidationThresholds)

    # Reference panel settings
    natural_high_performer_quantile: float = 0.75  # Top 25% for high performers
    background_sample_size: int = 10000  # Number of background samples
    knn_n_neighbors: int = 200  # For OOD detection

    # Device
    device: str = "cuda"

    # Batch sizes for inference
    batch_size: int = 64

    def get_physinformer_checkpoint(self, cell_type: str) -> Path:
        """Get PhysInformer checkpoint path for a cell type."""
        run_name = self.physinformer_models.get(cell_type)
        if run_name is None:
            raise ValueError(f"No PhysInformer model for cell type: {cell_type}")
        return self.physinformer_runs_dir / run_name / "best_model.pt"

    def get_cadence_model_path(self, dataset: str) -> Path:
        """Get CADENCE model path for a dataset."""
        model_name = self.cadence_models.get(dataset)
        if model_name is None:
            raise ValueError(f"No CADENCE model for dataset: {dataset}")
        return self.cadence_models_dir / model_name / "model_with_place.pt"

    def get_lentiMPRA_data_path(self, cell_type: str, split: str = "calibration") -> Path:
        """Get lentiMPRA data path for a cell type and split."""
        return self.lentiMPRA_data_dir / cell_type / f"{cell_type}_{split}_with_features.tsv"


# Conservative protocol weights (alpha values for physics constraints)
CONSERVATIVE_PROTOCOL_WEIGHTS = {
    "thermodynamics": 1.0,
    "shape_mgw_prot_roll": 0.8,
    "bending_stiffness": 0.6,
    "stacking": 0.5,
    "sidd_g4": 0.4,
    "entropy": 0.3,
    "electrostatics": 0.5,
}


# TF classes for MicroMotif validation
TF_CLASSES = [
    "bZIP", "bHLH", "ETS", "CEBP", "NFY", "GATA", "SP1", "AP1",
    "CREB", "E2F", "MYC", "RUNX", "PAX", "SOX", "POU", "HOX",
]
