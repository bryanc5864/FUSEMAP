"""
PhysicsTransfer: Cross-Species Knowledge Transfer via Biophysical Feature Bridges

This module enables cross-species transfer learning using physics features as a
universal bridge. The key hypothesis is that biophysical constraints are more
conserved across species than sequence motifs.

Transfer Protocols:
    1. Physics-Bridge Zero-Shot: Apply source physics→activity mapping to target
    2. Physics-Anchored Fine-Tuning: Fine-tune with frozen physics encoder
    3. Multi-Species Joint Training: Shared physics encoder, species-specific heads

Experiments:
    - Human → Drosophila (S2)
    - Animal → Plant (Arabidopsis, Sorghum, Maize)
    - Eukaryote → Yeast

Key Question: Does physics-bridge transfer provide >10% improvement in Pearson r
compared to sequence-only transfer?
"""

from .config import TransferConfig, DATASETS, EXPERIMENTS
from .data_loader import PhysicsDataLoader
from .physics_probe import PhysicsActivityProbe
from .protocols import (
    ZeroShotTransfer,
    PhysicsAnchoredFineTuning,
    MultiSpeciesJointTraining
)
from .evaluation import TransferEvaluator
from .logging_utils import ExperimentLogger, Timer

__version__ = '1.0.0'
__all__ = [
    'TransferConfig',
    'DATASETS',
    'EXPERIMENTS',
    'PhysicsDataLoader',
    'PhysicsActivityProbe',
    'ZeroShotTransfer',
    'PhysicsAnchoredFineTuning',
    'MultiSpeciesJointTraining',
    'TransferEvaluator',
    'ExperimentLogger',
    'Timer'
]
