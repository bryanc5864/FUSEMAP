"""
FUSEMAP Training Module

Multi-species training coordinator with:
- 5 configuration types (single cell, multi-cell, cross-animal, cross-kingdom, universal)
- Temperature-balanced sampling for dataset imbalance
- Activity normalization per dataset
- Multi-head architecture with masking
- Comprehensive logging and validation
- Early stopping and checkpointing

Usage:
    from training import Trainer, get_config
    from training.config import ConfigurationType

    config = get_config(ConfigurationType.CROSS_ANIMAL)
    trainer = Trainer(config)
    results = trainer.train()

CLI Usage:
    python -m training.coordinator --config cross_animal
    python -m training.coordinator --config single_celltype --dataset encode4_k562
"""

from .config import (
    ConfigurationType,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    DatasetInfo,
    DATASET_CATALOG,
    get_config,
    get_config1_single_celltype,
    get_config2_multi_celltype_human,
    get_config3_cross_animal,
    get_config4_cross_kingdom,
    get_config5_universal,
    get_config5_universal_no_yeast,
)

from .datasets import (
    SingleDataset,
    MultiDataset,
    ActivityNormalizer,
    collate_multi_dataset,
    get_validation_loader,
)

from .samplers import (
    TemperatureBalancedSampler,
    GlobalIndexSampler,
    StratifiedMultiDatasetSampler,
    GradNormBalancer,
    DatasetWeightScheduler,
)

from .metrics import (
    MetricResult,
    DatasetMetrics,
    MetricsTracker,
    compute_all_metrics,
    DREAMYeastMetrics,
)

from .models import (
    LegNet,
    MultiOutputLegNet,
    MultiSpeciesCADENCE,
    create_multi_species_model,
    compute_masked_loss,
)

from .trainer import (
    Trainer,
    MultiPhaseTrainer,
)

__all__ = [
    # Config
    'ConfigurationType',
    'ExperimentConfig',
    'ModelConfig',
    'TrainingConfig',
    'DatasetInfo',
    'DATASET_CATALOG',
    'get_config',
    'get_config1_single_celltype',
    'get_config2_multi_celltype_human',
    'get_config3_cross_animal',
    'get_config4_cross_kingdom',
    'get_config5_universal',
    'get_config5_universal_no_yeast',
    # Datasets
    'SingleDataset',
    'MultiDataset',
    'ActivityNormalizer',
    'collate_multi_dataset',
    'get_validation_loader',
    # Samplers
    'TemperatureBalancedSampler',
    'GlobalIndexSampler',
    'StratifiedMultiDatasetSampler',
    'GradNormBalancer',
    'DatasetWeightScheduler',
    # Metrics
    'MetricResult',
    'DatasetMetrics',
    'MetricsTracker',
    'compute_all_metrics',
    'DREAMYeastMetrics',
    # Models
    'LegNet',
    'MultiOutputLegNet',
    'MultiSpeciesCADENCE',
    'create_multi_species_model',
    'compute_masked_loss',
    # Training
    'Trainer',
    'MultiPhaseTrainer',
]

__version__ = '1.0.0'
