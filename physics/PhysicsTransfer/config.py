"""
Configuration for PhysicsTransfer experiments.

Defines datasets, model paths, and experiment configurations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path


def get_fusemap_root() -> Path:
    """Get FUSEMAP root directory."""
    return Path(__file__).parent.parent.parent


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    species: str
    kingdom: str  # 'animal', 'plant', 'fungi'
    data_dir: str
    file_pattern: str
    activity_col: str
    cell_types: List[str]
    splits: List[str] = field(default_factory=lambda: ['train', 'val', 'test'])
    has_cadence: bool = False
    cadence_checkpoint: Optional[str] = None
    physinformer_checkpoint: Optional[str] = None
    description: str = ""

    def get_data_path(self, cell_type: str, split: str) -> Path:
        """Get full path to data file."""
        root = get_fusemap_root()
        filename = self.file_pattern.format(cell_type=cell_type, split=split)
        return root / self.data_dir / filename


# Dataset configurations
DATASETS: Dict[str, DatasetConfig] = {
    # Human datasets
    'K562': DatasetConfig(
        name='K562',
        species='human',
        kingdom='animal',
        data_dir='physics/data/lentiMPRA_data/K562',
        file_pattern='{cell_type}_{split}_with_features.tsv',
        activity_col='activity',
        cell_types=['K562'],
        has_cadence=False,  # No CADENCE model yet
        physinformer_checkpoint='results/physics_models/models/PhysInformer_K562_best.pt',
        description='Human K562 erythroid cells (ENCODE4 lentiMPRA)'
    ),
    'HepG2': DatasetConfig(
        name='HepG2',
        species='human',
        kingdom='animal',
        data_dir='physics/data/lentiMPRA_data/HepG2',
        file_pattern='{cell_type}_{split}_with_features.tsv',
        activity_col='activity',
        cell_types=['HepG2'],
        has_cadence=False,  # No CADENCE model yet
        physinformer_checkpoint='results/physics_models/models/PhysInformer_HepG2_best.pt',
        description='Human HepG2 hepatocyte cells (ENCODE4 lentiMPRA)'
    ),
    'WTC11': DatasetConfig(
        name='WTC11',
        species='human',
        kingdom='animal',
        data_dir='physics/data/lentiMPRA_data/WTC11',
        file_pattern='{cell_type}_{split}_with_features.tsv',
        activity_col='activity',
        cell_types=['WTC11'],
        has_cadence=True,  # Only WTC11 has CADENCE
        cadence_checkpoint='models/legatoV2/outputs/WTC11_fold8_20250831_212613/checkpoints/best_model.pt',
        physinformer_checkpoint='results/physics_models/models/PhysInformer_WTC11_best.pt',
        description='Human WTC11 iPSC cells (ENCODE4 lentiMPRA)'
    ),
    # Drosophila dataset
    'S2': DatasetConfig(
        name='S2',
        species='drosophila',
        kingdom='animal',
        data_dir='physics/data/drosophila_data/S2',
        file_pattern='{cell_type}_{split}_with_features.tsv',
        activity_col='Dev_log2_enrichment',  # Default to developmental
        cell_types=['S2'],
        has_cadence=False,
        physinformer_checkpoint='results/physics_models/models/PhysInformer_S2_best.pt',
        description='Drosophila S2 cells (DeepSTARR)'
    ),
    'S2_dev': DatasetConfig(
        name='S2_dev',
        species='drosophila',
        kingdom='animal',
        data_dir='physics/data/drosophila_data/S2',
        file_pattern='{cell_type}_{split}_with_features.tsv',
        activity_col='Dev_log2_enrichment',
        cell_types=['S2'],
        has_cadence=False,
        physinformer_checkpoint='results/physics_models/models/PhysInformer_S2_best.pt',
        description='Drosophila S2 developmental enhancers'
    ),
    'S2_hk': DatasetConfig(
        name='S2_hk',
        species='drosophila',
        kingdom='animal',
        data_dir='physics/data/drosophila_data/S2',
        file_pattern='{cell_type}_{split}_with_features.tsv',
        activity_col='Hk_log2_enrichment',
        cell_types=['S2'],
        has_cadence=False,
        physinformer_checkpoint='results/physics_models/models/PhysInformer_S2_best.pt',
        description='Drosophila S2 housekeeping enhancers'
    ),
    # Plant datasets
    'arabidopsis_leaf': DatasetConfig(
        name='arabidopsis_leaf',
        species='arabidopsis',
        kingdom='plant',
        data_dir='physics/output',
        file_pattern='arabidopsis_{split}_descriptors_with_activity.tsv',
        activity_col='enrichment_leaf',
        cell_types=['arabidopsis'],
        has_cadence=False,
        description='Arabidopsis leaf enrichment'
    ),
    'arabidopsis_proto': DatasetConfig(
        name='arabidopsis_proto',
        species='arabidopsis',
        kingdom='plant',
        data_dir='physics/output',
        file_pattern='arabidopsis_{split}_descriptors_with_activity.tsv',
        activity_col='enrichment_proto',
        cell_types=['arabidopsis'],
        has_cadence=False,
        description='Arabidopsis protoplast enrichment'
    ),
    'sorghum_leaf': DatasetConfig(
        name='sorghum_leaf',
        species='sorghum',
        kingdom='plant',
        data_dir='physics/output',
        file_pattern='sorghum_{split}_descriptors_with_activity.tsv',
        activity_col='enrichment_leaf',
        cell_types=['sorghum'],
        has_cadence=False,
        description='Sorghum leaf enrichment'
    ),
    'sorghum_proto': DatasetConfig(
        name='sorghum_proto',
        species='sorghum',
        kingdom='plant',
        data_dir='physics/output',
        file_pattern='sorghum_{split}_descriptors_with_activity.tsv',
        activity_col='enrichment_proto',
        cell_types=['sorghum'],
        has_cadence=False,
        description='Sorghum protoplast enrichment'
    ),
    'maize_leaf': DatasetConfig(
        name='maize_leaf',
        species='maize',
        kingdom='plant',
        data_dir='physics/output',
        file_pattern='maize_{split}_descriptors_with_activity.tsv',
        activity_col='enrichment_leaf',
        cell_types=['maize'],
        has_cadence=False,
        description='Maize leaf enrichment'
    ),
    'maize_proto': DatasetConfig(
        name='maize_proto',
        species='maize',
        kingdom='plant',
        data_dir='physics/output',
        file_pattern='maize_{split}_descriptors_with_activity.tsv',
        activity_col='enrichment_proto',
        cell_types=['maize'],
        has_cadence=False,
        description='Maize protoplast enrichment'
    ),
}


@dataclass
class ExperimentConfig:
    """Configuration for a transfer learning experiment."""
    name: str
    source_datasets: List[str]
    target_dataset: str
    description: str
    fine_tune_sizes: List[int] = field(default_factory=lambda: [1000, 5000, 10000])


# Experiment configurations matching the plan
EXPERIMENTS: Dict[str, ExperimentConfig] = {
    # Experiment 1: Human → Drosophila
    'human_to_drosophila': ExperimentConfig(
        name='human_to_drosophila',
        source_datasets=['K562', 'HepG2', 'WTC11'],
        target_dataset='S2_dev',
        description='Human → Drosophila: Test transfer across 600M years of evolution',
        fine_tune_sizes=[1000, 5000, 10000]
    ),
    'wtc11_to_s2': ExperimentConfig(
        name='wtc11_to_s2',
        source_datasets=['WTC11'],
        target_dataset='S2_dev',
        description='WTC11 → S2: Single source transfer (has CADENCE)',
        fine_tune_sizes=[1000, 5000, 10000]
    ),
    # Experiment 2: Animal → Plant
    'animal_to_arabidopsis': ExperimentConfig(
        name='animal_to_arabidopsis',
        source_datasets=['K562', 'HepG2', 'WTC11', 'S2_dev'],
        target_dataset='arabidopsis_leaf',
        description='Animal → Arabidopsis: Cross-kingdom transfer',
        fine_tune_sizes=[1000, 5000, 10000]
    ),
    'animal_to_maize': ExperimentConfig(
        name='animal_to_maize',
        source_datasets=['K562', 'HepG2', 'WTC11', 'S2_dev'],
        target_dataset='maize_leaf',
        description='Animal → Maize: Cross-kingdom transfer',
        fine_tune_sizes=[1000, 5000, 10000]
    ),
    # Within-kingdom transfers
    'human_cross_celltype': ExperimentConfig(
        name='human_cross_celltype',
        source_datasets=['K562', 'HepG2'],
        target_dataset='WTC11',
        description='K562+HepG2 → WTC11: Within-species cross-cell-type',
        fine_tune_sizes=[1000, 5000, 10000]
    ),
    'plant_cross_species': ExperimentConfig(
        name='plant_cross_species',
        source_datasets=['arabidopsis_leaf', 'sorghum_leaf'],
        target_dataset='maize_leaf',
        description='Arabidopsis+Sorghum → Maize: Within-kingdom cross-species',
        fine_tune_sizes=[1000, 5000, 10000]
    ),
}


@dataclass
class TransferConfig:
    """Master configuration for PhysicsTransfer."""
    # Model paths
    tileformer_checkpoint: str = 'results/physics_models/models/TileFormer_best.pth'

    # Physics feature families
    physics_families: List[str] = field(default_factory=lambda: [
        'thermo',    # Thermodynamic features (ΔG, ΔH, ΔS, Tm)
        'stiff',     # Mechanical stiffness (twist, tilt, roll)
        'bend',      # Bending and curvature
        'entropy',   # Sequence complexity and information
        'advanced',  # G4, SIDD, MGW, nucleosome positioning
    ])

    # PWM/TF features (species-specific, may not transfer)
    include_pwm_in_transfer: bool = False

    # Electrostatics (from TileFormer)
    include_electrostatics: bool = True
    electrostatics_prefix: str = 'tileformer_'

    # Training parameters
    random_seed: int = 42
    n_folds: int = 5

    # Physics probe parameters
    probe_type: str = 'ridge'  # 'ridge', 'elastic_net', 'lasso', 'mlp' (ridge is fastest)
    probe_alpha: float = 1.0  # Regularization strength
    probe_l1_ratio: float = 0.5  # Only used for elastic_net

    # Fine-tuning parameters
    fine_tune_lr: float = 1e-4
    fine_tune_epochs: int = 50
    fine_tune_batch_size: int = 64

    # Output
    output_dir: str = 'physics/PhysicsTransfer/results'
    save_models: bool = True

    def get_physics_feature_prefixes(self) -> List[str]:
        """Get all physics feature prefixes to include."""
        prefixes = list(self.physics_families)
        if self.include_electrostatics:
            prefixes.append(self.electrostatics_prefix)
        return prefixes

    def get_full_output_dir(self) -> Path:
        """Get full path to output directory."""
        return get_fusemap_root() / self.output_dir


# Default configuration
DEFAULT_CONFIG = TransferConfig()
