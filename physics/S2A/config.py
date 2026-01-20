"""
Configuration for Zero-Shot Sequence-to-Activity (S2A) Prediction System.

Defines S2A configuration, dataset registry, and experiment presets.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from pathlib import Path


def get_fusemap_root() -> Path:
    """Get FUSEMAP root directory."""
    return Path(__file__).parent.parent.parent


@dataclass
class S2AConfig:
    """Configuration for S2A training and inference."""

    # Feature selection - universal physics features only
    universal_prefixes: List[str] = field(default_factory=lambda: [
        'thermo_',    # Thermodynamic features (ΔG, ΔH, ΔS, Tm)
        'stiff_',     # Mechanical stiffness (twist, tilt, roll)
        'bend_',      # Bending and curvature
        'entropy_',   # Sequence complexity and information
        'advanced_',  # G4, SIDD, MGW, nucleosome positioning
    ])

    # Species-specific features to exclude
    excluded_prefixes: List[str] = field(default_factory=lambda: [
        'pwm_',  # Position weight matrix (TF binding - species-specific)
    ])

    # Head model configuration
    head_type: str = 'ridge'  # 'ridge', 'elastic_net', 'mlp'
    head_alpha: float = 1.0   # Regularization strength
    head_l1_ratio: float = 0.5  # For elastic_net only
    head_hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])  # For MLP

    # Output mode
    output_mode: str = 'zscore'  # 'zscore', 'calibrated', 'ranking'

    # Calibration parameters
    calibration_n_samples: int = 50
    calibration_method: str = 'affine'  # 'affine', 'isotonic'

    # Training parameters
    random_seed: int = 42
    n_folds: int = 5  # For CV during training

    # Output directory
    output_dir: str = 'results/s2a'

    def get_full_output_dir(self) -> Path:
        """Get full path to output directory."""
        return get_fusemap_root() / self.output_dir


@dataclass
class S2ADatasetConfig:
    """Configuration for a single S2A dataset."""
    name: str
    species: str
    kingdom: str  # 'animal', 'plant'
    data_dir: str
    file_pattern: str
    activity_col: str
    description: str = ""

    def get_data_path(self, split: str) -> Path:
        """Get full path to data file for a split."""
        root = get_fusemap_root()
        filename = self.file_pattern.format(split=split)
        return root / self.data_dir / filename


# Dataset registry for S2A
S2A_DATASETS: Dict[str, S2ADatasetConfig] = {
    # Human datasets (ENCODE4 lentiMPRA)
    'K562': S2ADatasetConfig(
        name='K562',
        species='human',
        kingdom='animal',
        data_dir='physics/data/lentiMPRA_data/K562',
        file_pattern='K562_{split}_with_features.tsv',
        activity_col='activity',
        description='Human K562 erythroid cells (ENCODE4 lentiMPRA)'
    ),
    'HepG2': S2ADatasetConfig(
        name='HepG2',
        species='human',
        kingdom='animal',
        data_dir='physics/data/lentiMPRA_data/HepG2',
        file_pattern='HepG2_{split}_with_features.tsv',
        activity_col='activity',
        description='Human HepG2 hepatocyte cells (ENCODE4 lentiMPRA)'
    ),
    'WTC11': S2ADatasetConfig(
        name='WTC11',
        species='human',
        kingdom='animal',
        data_dir='physics/data/lentiMPRA_data/WTC11',
        file_pattern='WTC11_{split}_with_features.tsv',
        activity_col='activity',
        description='Human WTC11 iPSC cells (ENCODE4 lentiMPRA)'
    ),

    # Drosophila datasets (DeepSTARR)
    'S2_dev': S2ADatasetConfig(
        name='S2_dev',
        species='drosophila',
        kingdom='animal',
        data_dir='physics/data/drosophila_data/S2',
        file_pattern='S2_{split}_with_features.tsv',
        activity_col='Dev_log2_enrichment',
        description='Drosophila S2 developmental enhancers'
    ),
    'S2_hk': S2ADatasetConfig(
        name='S2_hk',
        species='drosophila',
        kingdom='animal',
        data_dir='physics/data/drosophila_data/S2',
        file_pattern='S2_{split}_with_features.tsv',
        activity_col='Hk_log2_enrichment',
        description='Drosophila S2 housekeeping enhancers'
    ),

    # Plant datasets (Jores et al.)
    'arabidopsis_leaf': S2ADatasetConfig(
        name='arabidopsis_leaf',
        species='arabidopsis',
        kingdom='plant',
        data_dir='physics/output',
        file_pattern='arabidopsis_{split}_descriptors_with_activity.tsv',
        activity_col='enrichment_leaf',
        description='Arabidopsis leaf enrichment (Jores et al.)'
    ),
    'arabidopsis_proto': S2ADatasetConfig(
        name='arabidopsis_proto',
        species='arabidopsis',
        kingdom='plant',
        data_dir='physics/output',
        file_pattern='arabidopsis_{split}_descriptors_with_activity.tsv',
        activity_col='enrichment_proto',
        description='Arabidopsis protoplast enrichment (Jores et al.)'
    ),
    'sorghum_leaf': S2ADatasetConfig(
        name='sorghum_leaf',
        species='sorghum',
        kingdom='plant',
        data_dir='physics/output',
        file_pattern='sorghum_{split}_descriptors_with_activity.tsv',
        activity_col='enrichment_leaf',
        description='Sorghum leaf enrichment (Jores et al.)'
    ),
    'sorghum_proto': S2ADatasetConfig(
        name='sorghum_proto',
        species='sorghum',
        kingdom='plant',
        data_dir='physics/output',
        file_pattern='sorghum_{split}_descriptors_with_activity.tsv',
        activity_col='enrichment_proto',
        description='Sorghum protoplast enrichment (Jores et al.)'
    ),
    'maize_leaf': S2ADatasetConfig(
        name='maize_leaf',
        species='maize',
        kingdom='plant',
        data_dir='physics/output',
        file_pattern='maize_{split}_descriptors_with_activity.tsv',
        activity_col='enrichment_leaf',
        description='Maize leaf enrichment (Jores et al.)'
    ),
    'maize_proto': S2ADatasetConfig(
        name='maize_proto',
        species='maize',
        kingdom='plant',
        data_dir='physics/output',
        file_pattern='maize_{split}_descriptors_with_activity.tsv',
        activity_col='enrichment_proto',
        description='Maize protoplast enrichment (Jores et al.)'
    ),
}

# Predefined dataset groups
S2A_DATASET_GROUPS: Dict[str, Set[str]] = {
    'human': {'K562', 'HepG2', 'WTC11'},
    'drosophila': {'S2_dev', 'S2_hk'},
    'plant': {'arabidopsis_leaf', 'arabidopsis_proto', 'sorghum_leaf',
              'sorghum_proto', 'maize_leaf', 'maize_proto'},
    'plant_leaf': {'arabidopsis_leaf', 'sorghum_leaf', 'maize_leaf'},
    'animal': {'K562', 'HepG2', 'WTC11', 'S2_dev', 'S2_hk'},
    'all': set(S2A_DATASETS.keys()),
}


@dataclass
class S2AExperimentConfig:
    """Configuration for a leave-one-out experiment."""
    name: str
    source_datasets: List[str]
    holdout_dataset: str
    description: str

    def validate(self) -> bool:
        """Validate that all datasets exist."""
        all_datasets = set(S2A_DATASETS.keys())
        for ds in self.source_datasets + [self.holdout_dataset]:
            if ds not in all_datasets:
                return False
        return True


# Predefined experiments
S2A_EXPERIMENTS: Dict[str, S2AExperimentConfig] = {
    # Leave-one-out: holdout S2_dev
    'holdout_s2_dev': S2AExperimentConfig(
        name='holdout_s2_dev',
        source_datasets=['K562', 'HepG2', 'WTC11', 'arabidopsis_leaf',
                        'sorghum_leaf', 'maize_leaf'],
        holdout_dataset='S2_dev',
        description='Train on human + plants, test zero-shot on Drosophila'
    ),

    # Within-plant transfer
    'holdout_maize_leaf': S2AExperimentConfig(
        name='holdout_maize_leaf',
        source_datasets=['arabidopsis_leaf', 'sorghum_leaf'],
        holdout_dataset='maize_leaf',
        description='Train on Arabidopsis + Sorghum, test zero-shot on Maize'
    ),

    # Within-human transfer
    'holdout_wtc11': S2AExperimentConfig(
        name='holdout_wtc11',
        source_datasets=['K562', 'HepG2'],
        holdout_dataset='WTC11',
        description='Train on K562 + HepG2, test zero-shot on WTC11'
    ),

    # Cross-kingdom: animals → plant
    'animals_to_arabidopsis': S2AExperimentConfig(
        name='animals_to_arabidopsis',
        source_datasets=['K562', 'HepG2', 'WTC11', 'S2_dev'],
        holdout_dataset='arabidopsis_leaf',
        description='Train on all animals, test zero-shot on Arabidopsis'
    ),
}


# Default configuration
DEFAULT_S2A_CONFIG = S2AConfig()
