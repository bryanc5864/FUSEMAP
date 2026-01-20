"""
Configuration for PhysicsInterpreter.

Defines model paths, physics families, and analysis parameters.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


def get_fusemap_root() -> Path:
    """Get FUSEMAP root directory."""
    return Path(__file__).parent.parent.parent


# Model paths organized by type and cell type
MODEL_PATHS = {
    'cadence': {
        'WTC11': 'models/legatoV2/outputs/WTC11_fold8_20250831_212613/checkpoints/best_model.pt',
        'HepG2': 'models/legatoV2/outputs/HepG2_fold8_20250829_070448/checkpoints/best_model.pt',
        'S2': 'models/legatoV2/outputs/s2_cv_fold0_20250904_063226/checkpoints/best_model.pt',
    },
    'physinformer': {
        'K562': 'results/physics_models/models/PhysInformer_K562_best.pt',
        'HepG2': 'results/physics_models/models/PhysInformer_HepG2_best.pt',
        'WTC11': 'results/physics_models/models/PhysInformer_WTC11_best.pt',
        'S2': 'results/physics_models/models/PhysInformer_S2_best.pt',
    },
    'tileformer': 'results/physics_models/models/TileFormer_best.pth',
}

# Physics feature families with their prefixes
PHYSICS_FAMILIES = {
    'thermodynamics': ['thermo_'],
    'mechanics': ['stiff_'],
    'bending': ['bend_'],
    'entropy': ['entropy_'],
    'structural': ['advanced_mgw', 'advanced_stress', 'advanced_melting', 'advanced_stacking'],
    'electrostatics': ['tileformer_', 'elec_'],
    'motif_derived': ['pwm_'],
}


@dataclass
class InterpreterConfig:
    """Configuration for PhysicsInterpreter analyses."""

    # Cell type
    cell_type: str = 'WTC11'

    # Integrated Gradients settings
    ig_steps: int = 50  # Number of interpolation steps
    ig_baseline: str = 'zeros'  # 'zeros', 'shuffle', 'gc_matched'
    ig_batch_size: int = 32

    # Physics attribution settings
    attribution_probe_type: str = 'ridge'  # 'ridge', 'elastic_net'
    attribution_alpha: float = 1.0

    # Mediation analysis settings
    mediation_n_bootstrap: int = 100
    mediation_confidence: float = 0.95

    # Landscape analysis settings
    landscape_correlation_method: str = 'pearson'  # 'pearson', 'spearman'
    landscape_shap_samples: int = 100  # Background samples for SHAP
    landscape_elastic_net_alpha: float = 0.01
    landscape_elastic_net_l1_ratio: float = 0.5

    # Physics families to analyze
    physics_families: List[str] = field(default_factory=lambda: [
        'thermodynamics', 'mechanics', 'bending', 'entropy', 'structural'
    ])

    # Output settings
    output_dir: str = 'physics/PhysicsInterpreter/results'
    random_seed: int = 42

    def get_cadence_path(self) -> Path:
        """Get CADENCE model path for configured cell type."""
        root = get_fusemap_root()
        if self.cell_type not in MODEL_PATHS['cadence']:
            raise ValueError(f"No CADENCE model for {self.cell_type}")
        return root / MODEL_PATHS['cadence'][self.cell_type]

    def get_physinformer_path(self) -> Path:
        """Get PhysInformer model path for configured cell type."""
        root = get_fusemap_root()
        if self.cell_type not in MODEL_PATHS['physinformer']:
            raise ValueError(f"No PhysInformer model for {self.cell_type}")
        return root / MODEL_PATHS['physinformer'][self.cell_type]

    def get_tileformer_path(self) -> Path:
        """Get TileFormer model path."""
        root = get_fusemap_root()
        return root / MODEL_PATHS['tileformer']

    def get_output_dir(self) -> Path:
        """Get output directory path."""
        return get_fusemap_root() / self.output_dir

    def get_physics_prefixes(self) -> List[str]:
        """Get all physics feature prefixes for configured families."""
        prefixes = []
        for family in self.physics_families:
            if family in PHYSICS_FAMILIES:
                prefixes.extend(PHYSICS_FAMILIES[family])
        return prefixes


# Default configuration
DEFAULT_CONFIG = InterpreterConfig()
