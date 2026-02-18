"""
================================================================================
FUSEMAP TRAINING & APPLICATIONS - Representative Code File 3/3
================================================================================

These 3 representative files contain key excerpts from the FUSEMAP codebase.
They do not contain all FUSEMAP code, as the full implementation is too large
to include here. All code, trained models, and processed datasets are available
under the MIT license at:
    https://github.com/bryanc5864/FUSEMAP

This file contains the training infrastructure and downstream applications:

MODULES INCLUDED:
1. Training Configuration - 5 experiment presets with dataset catalog
   Source: training/config.py
   DatasetInfo, ModelConfig, TrainingConfig, ExperimentConfig
   Config1 (single cell type) through Config5 (universal foundation model)

2. Training Coordinator - Main entry point and seed management
   Source: training/coordinator.py

3. Trainer - Full training loop with multi-dataset support
   Source: training/trainer.py
   Train epoch with AMP + gradient accumulation, validation, checkpointing
   MultiPhaseTrainer for 3-phase Config5 universal training

4. Therapeutic Enhancer Design Pipeline - Cell-type specific design
   Source: applications/therapeutic_enhancer_pipeline.py
   6-step protocol: physics extraction -> VAE generation -> OracleCheck -> ranking

5. Disease Variant Pipeline - Variant effect prediction
   Source: applications/disease_variant_pipeline.py
   VCF loading -> ref/alt prediction -> physics attribution -> ranking

6. Datasets - Activity normalization and multi-dataset handling
   Source: training/datasets.py
   ActivityNormalizer, SingleDataset (with RC + shift augmentation), MultiDataset

7. Samplers - Temperature-balanced and activity-balanced sampling
   Source: training/samplers.py
   TemperatureBalancedSampler, GlobalIndexSampler, BalancedActivitySampler,
   ExtremeAwareSampler (z-score weighted oversampling of distribution tails)

KEY RESULTS (from paper):
- CADENCE: K562 r=0.809, HepG2 r=0.786, WTC11 r=0.698
- DeepSTARR: Dev r=0.909, Hk r=0.920
- Plants: Maize r=0.796, Sorghum r=0.782, Arabidopsis r=0.618
- Yeast: r=0.958
- Therapeutic design: 99% predicted specificity, 96.5% GREEN OracleCheck (HepG2)
- Universal foundation model across 7 species, 7.8M total sequences
================================================================================
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


# =============================================================================
# PART 1: Training Configuration
# Source: training/config.py
# =============================================================================

class ConfigurationType(Enum):
    """Training configuration types corresponding to increasing generalization scope.

    Each level adds more datasets and enables more conditioning embeddings:
      Config1: Single dataset (baseline, no embeddings needed)
      Config2: Multiple human cell types (adds cell-type embedding)
      Config3: Cross-species within Animalia (adds species embedding)
      Config4: Cross-kingdom Animal+Plant (adds kingdom embedding)
      Config5: Universal across all 7 species / 3 kingdoms (all embeddings + phased training)
    """
    SINGLE_CELLTYPE = "config1_single_celltype"
    MULTI_CELLTYPE_HUMAN = "config2_multi_celltype_human"
    CROSS_ANIMAL = "config3_cross_animal"
    CROSS_KINGDOM = "config4_cross_kingdom"
    UNIVERSAL = "config5_universal"


@dataclass
class DatasetInfo:
    """Metadata descriptor for a single dataset in the FUSEMAP catalog.

    Each dataset entry provides the information needed to:
    - Load the data (path, sequence_length)
    - Configure the model head (num_outputs, output_names)
    - Set conditioning embeddings (species, kingdom, cell_type)
    - Choose the validation strategy (kfold vs chromosome holdout vs special)
    """
    name: str                        # Unique identifier used as dict key in DATASET_CATALOG
    path: str                        # Path to processed data directory (contains train/val/test splits)
    sequence_length: int             # Native sequence length in bp (before padding to target_length)
    num_outputs: int                 # Number of prediction heads for this dataset
    output_names: List[str]          # Human-readable names for each output (e.g., ["Dev", "Hk"])
    species: str                     # Species name for species embedding lookup
    kingdom: str                     # Biological kingdom: "animal", "plant", or "fungi"
    cell_type: Optional[str] = None  # Cell type / assay context for cell-type embedding
    element_type: str = "promoter"   # Regulatory element type: "promoter" or "enhancer"

    train_size: int = 0              # Number of training samples (informational, for sampler weighting)
    val_size: int = 0                # Number of validation samples
    test_size: int = 0               # Number of test samples

    validation_scheme: str = "standard"       # "standard" (random split), "kfold", "chromosome_holdout", or "special"
    kfold_splits: int = 10                    # Number of folds for k-fold cross-validation
    holdout_chromosome: Optional[str] = None  # Chromosome held out for validation (e.g., "2R" for DeepSTARR)


# Dataset catalog - all supported datasets
# This catalog defines the 8 datasets spanning 7 species and 3 kingdoms used in FUSEMAP.
# Each entry is keyed by a short name and stores all metadata needed for loading and training.
DATASET_CATALOG: Dict[str, DatasetInfo] = {
    # -------------------------------------------------------------------------
    # ENCODE4 Human lentiMPRA datasets (Agarwal et al., 2023)
    # Assay: Lentiviral Massively Parallel Reporter Assay (lentiMPRA)
    # These measure enhancer activity of 230bp sequences integrated into the genome
    # via lentiviral delivery. Each cell type has a single activity output (log2 RNA/DNA).
    # Validation: 10-fold cross-validation (no held-out chromosome).
    # -------------------------------------------------------------------------
    "encode4_k562": DatasetInfo(
        name="encode4_k562",
        path="data/processed/encode4/k562",
        sequence_length=230,
        num_outputs=1,
        output_names=["activity"],
        species="human",
        kingdom="animal",
        cell_type="K562",
        element_type="enhancer",
        train_size=164307,
        validation_scheme="kfold",
    ),
    "encode4_hepg2": DatasetInfo(
        name="encode4_hepg2",
        path="data/processed/encode4/hepg2",
        sequence_length=230,
        num_outputs=1,
        output_names=["activity"],
        species="human",
        kingdom="animal",
        cell_type="HepG2",
        element_type="enhancer",
        train_size=243780,
        validation_scheme="kfold",
    ),
    # WTC11 = human iPSC (induced pluripotent stem cell) line
    "encode4_wtc11": DatasetInfo(
        name="encode4_wtc11",
        path="data/processed/encode4/wtc11",
        sequence_length=230,
        num_outputs=1,
        output_names=["activity"],
        species="human",
        kingdom="animal",
        cell_type="WTC11",
        element_type="enhancer",
        train_size=75542,
        validation_scheme="kfold",
    ),
    # -------------------------------------------------------------------------
    # DeepSTARR Drosophila dataset (de Almeida et al., 2022)
    # Assay: Self-Transcribing Active Regulatory Region Sequencing (STARR-seq)
    # Measures enhancer activity of 249bp sequences in Drosophila S2 cells.
    # 2 outputs: "Dev" (developmental enhancers) and "Hk" (housekeeping enhancers).
    # Validation: Chromosome 2R held out as test set (matching original DeepSTARR paper).
    # -------------------------------------------------------------------------
    "deepstarr": DatasetInfo(
        name="deepstarr",
        path="data/processed/deepstarr",
        sequence_length=249,
        num_outputs=2,
        output_names=["Dev", "Hk"],
        species="drosophila",
        kingdom="animal",
        cell_type="S2",
        element_type="enhancer",
        train_size=484052,
        validation_scheme="chromosome_holdout",
        holdout_chromosome="2R",
    ),
    # -------------------------------------------------------------------------
    # DREAM Yeast Promoter Activity dataset (Sample et al., 2019 / DREAM Challenge)
    # Assay: Massively parallel promoter activity measurement in S. cerevisiae.
    # 110bp synthetic promoter sequences with 1 output (expression level).
    # Largest dataset: ~6.7M training sequences of random/designed promoters.
    # Validation: Special split defined by the DREAM challenge organizers.
    # -------------------------------------------------------------------------
    "dream_yeast": DatasetInfo(
        name="dream_yeast",
        path="data/processed/dream_yeast",
        sequence_length=110,
        num_outputs=1,
        output_names=["expression"],
        species="yeast",
        kingdom="fungi",
        cell_type="yeast",
        element_type="promoter",
        train_size=6700000,
        validation_scheme="special",
    ),
    # -------------------------------------------------------------------------
    # Jores Plant Promoter datasets (Jores et al., 2021)
    # Assay: Plant STARR-seq measuring promoter activity in tobacco leaf protoplasts.
    # 170bp promoter sequences with 2 outputs: "leaf" (leaf protoplast) and
    # "proto" (root protoplast). Three species: Arabidopsis, Maize, Sorghum.
    # Validation: Standard random train/val/test split.
    # -------------------------------------------------------------------------
    "jores_arabidopsis": DatasetInfo(
        name="jores_arabidopsis",
        path="data/plant_data/jores2021/processed/arabidopsis",
        sequence_length=170,
        num_outputs=2,
        output_names=["leaf", "proto"],
        species="arabidopsis",
        kingdom="plant",
        cell_type="dual_assay",
        element_type="promoter",
        train_size=12000,
    ),
    "jores_maize": DatasetInfo(
        name="jores_maize",
        path="data/plant_data/jores2021/processed/maize",
        sequence_length=170,
        num_outputs=2,
        output_names=["leaf", "proto"],
        species="maize",
        kingdom="plant",
        cell_type="dual_assay",
        element_type="promoter",
        train_size=22000,
    ),
    "jores_sorghum": DatasetInfo(
        name="jores_sorghum",
        path="data/plant_data/jores2021/processed/sorghum",
        sequence_length=170,
        num_outputs=2,
        output_names=["leaf", "proto"],
        species="sorghum",
        kingdom="plant",
        cell_type="dual_assay",
        element_type="promoter",
        train_size=17000,
    ),
}


@dataclass
class ModelConfig:
    """Model architecture configuration for the CADENCE backbone.

    The CADENCE architecture consists of:
    1. A convolutional stem (optionally RC-equivariant) that detects motifs
    2. A stack of inverted-bottleneck blocks (like MobileNetV2) for hierarchical features
    3. Optional conditioning embeddings (species, kingdom, cell type, length)
    4. Per-dataset prediction heads with optional learned uncertainty (Gaussian NLL)
    """
    # --- Stem (initial convolution) ---
    stem_channels: int = 64          # Number of filters in the first conv layer (motif detectors)
    stem_kernel_size: int = 11       # Kernel width in bp; 11bp captures typical TF binding motifs (6-12bp)
    # --- Inverted-bottleneck blocks ---
    block_channels: List[int] = field(default_factory=lambda: [80, 96, 112, 128])  # Channel counts per block (progressive widening)
    block_kernel: int = 9            # Depthwise conv kernel size within each block
    expand_ratio: int = 4            # Expansion ratio for inverted bottleneck (hidden_dim = channels * expand_ratio)

    # --- RC-equivariant stem ---
    # When True, the stem uses reverse-complement equivariant convolutions so that
    # forward and RC strands produce identical features. Essential for multi-species
    # training where strand orientation is arbitrary.
    use_rc_stem: bool = True

    # --- Optional modules (architecture ablations) ---
    use_cluster_space: bool = False     # Cluster-space projection for motif grouping
    use_grammar: bool = False           # Motif grammar / syntax module (pairwise motif interactions)
    use_micromotif: bool = False        # Fine-grained sub-motif resolution module
    use_motif_correlator: bool = False  # Cross-attention between detected motifs

    # --- Conditioning embeddings ---
    # These learned embeddings are concatenated to the pooled features before the head.
    # Only enabled for multi-dataset configs where the model must disambiguate contexts.
    use_species_embedding: bool = False   # Embed species identity (e.g., human=0, drosophila=1, ...)
    species_embed_dim: int = 16           # Dimensionality of species embedding vector
    use_celltype_embedding: bool = False  # Embed cell type / assay context (e.g., K562=0, HepG2=1, ...)
    celltype_embed_dim: int = 32          # Dimensionality of cell-type embedding vector
    use_kingdom_embedding: bool = False   # Embed biological kingdom (animal=0, plant=1, fungi=2)
    kingdom_embed_dim: int = 8            # Dimensionality of kingdom embedding vector
    use_length_embedding: bool = True     # Embed the original sequence length (before padding)
    length_embed_dim: int = 16            # Dimensionality of length embedding vector

    # --- Prediction head ---
    head_hidden: int = 256       # Hidden layer size in the per-dataset MLP head
    use_uncertainty: bool = True  # If True, head outputs (mean, log_var) for Gaussian NLL loss;
                                  # if False, head outputs mean only and uses MSE loss
    dropout: float = 0.3         # Dropout rate in the prediction head (regularization)


@dataclass
class TrainingConfig:
    """Training hyperparameters for FUSEMAP experiments.

    Controls optimizer settings, learning rate scheduling, early stopping,
    mixed-precision training, multi-dataset sampling, and extreme-value weighting.
    """
    # --- Core training ---
    max_epochs: int = 50             # Maximum number of training epochs
    batch_size: int = 128            # Samples per batch (effective batch = batch_size * gradient_accumulation_steps)
    learning_rate: float = 1e-3      # Peak learning rate for AdamW optimizer
    weight_decay: float = 1e-5       # L2 regularization strength for AdamW

    # --- Learning rate scheduler ---
    # Options: "cosine" (CosineAnnealingLR with warmup), "onecycle" (OneCycleLR),
    #          "plateau" (ReduceLROnPlateau). OneCycle is preferred for large MPRA
    #          datasets; cosine for smaller or multi-dataset configs.
    scheduler: str = "cosine"
    warmup_epochs: int = 5           # Linear warmup epochs before cosine decay (cosine scheduler only)
    min_lr: float = 1e-6             # Minimum LR at end of cosine annealing
    onecycle_pct_start: float = 0.3  # Fraction of training spent in LR warmup phase (OneCycleLR)
    onecycle_div_factor: float = 25.0  # initial_lr = max_lr / div_factor (OneCycleLR)

    # --- Early stopping ---
    patience: int = 10               # Stop after this many epochs without improvement
    min_delta: float = 1e-4          # Minimum improvement to reset patience counter
    monitor: str = "val_pearson"     # Metric to monitor: average Pearson r across all datasets

    # --- Gradient management ---
    gradient_clip: float = 1.0                # Max gradient norm (prevents exploding gradients with NLL loss)
    gradient_accumulation_steps: int = 1      # Accumulate gradients over N mini-batches before stepping
    use_amp: bool = True                      # Use Automatic Mixed Precision (FP16 forward, FP32 grads)

    # --- Multi-dataset sampling ---
    sampling_temperature: float = 0.5         # Temperature for TemperatureBalancedSampler (see p_i ~ n_i^tau)
    samples_per_epoch: Optional[int] = None   # If set, caps samples per epoch (useful for huge datasets like yeast)

    # --- Logging ---
    log_every_n_steps: int = 100     # Print training metrics every N optimizer steps
    val_every_n_epochs: int = 1      # Run validation every N epochs

    # --- Extreme value weighting (loss-level) ---
    # Upweights samples at the tails of the activity distribution in the loss function.
    # Formula: w_i = 1 + alpha * |z_i|^beta, where z_i is the within-batch z-score.
    # This helps the model learn rare high/low activity sequences better.
    use_extreme_weights: bool = True
    extreme_alpha: float = 1.0       # Scaling factor (higher = more emphasis on extremes)
    extreme_beta: float = 2.0        # Exponent (2.0 = quadratic growth with z-score distance)

    # --- Balanced activity sampling (sampler-level) ---
    # When enabled, uses BalancedActivitySampler to equalize representation across
    # activity value bins, so extreme-activity sequences appear more often in training.
    use_balanced_sampling: bool = True
    balanced_sampling_bins: int = 10  # Number of equal-width activity bins


@dataclass
class ExperimentConfig:
    """Complete experiment configuration bundling model, training, and data settings.

    This is the top-level config object passed to the Trainer. It specifies which
    datasets to use, the model architecture, training hyperparameters, and optionally
    a multi-phase training schedule (used only by Config5 Universal).
    """
    name: str                              # Experiment name (used for output directory and logging)
    config_type: ConfigurationType         # Which configuration level (Config1-5)
    description: str                       # Human-readable description of this experiment

    datasets: List[str]                    # List of dataset keys from DATASET_CATALOG to include
    target_sequence_length: int = 256      # All sequences padded/cropped to this length before model input

    model: ModelConfig = field(default_factory=ModelConfig)        # Model architecture settings
    training: TrainingConfig = field(default_factory=TrainingConfig)  # Training hyperparameters

    output_dir: str = "results"            # Root output directory (experiment saves to output_dir/name/)
    seed: int = 42                         # Random seed for reproducibility

    # Multi-phase training schedule (Config5 only). Each dict has keys:
    #   "name": phase identifier, "epochs": number of epochs,
    #   "freeze_backbone": whether to freeze the shared backbone,
    #   "lr": learning rate for this phase
    training_phases: Optional[List[dict]] = None


# =============================================================================
# Configuration Presets
# =============================================================================

def get_config1_single_celltype(dataset_name: str, use_mse: bool = False) -> ExperimentConfig:
    """Configuration 1: Single Cell Type Baseline.

    Trains a separate model for each dataset independently (no cross-dataset transfer).
    No conditioning embeddings are used since there is only one data source.

    For human MPRA datasets (ENCODE4): uses aggressive training with large batch (1024),
    high LR (0.01), OneCycleLR scheduler, and strong weight decay (0.1) -- following
    the recipe from the HumanLegNet benchmark. No balanced sampling or extreme weighting
    since the MPRA activity distributions are already well-behaved.

    For other datasets: uses conservative defaults (batch 128, LR 1e-3, cosine scheduler).

    Args:
        dataset_name: Key from DATASET_CATALOG (e.g., "encode4_k562", "deepstarr")
        use_mse: If True, force MSE loss instead of Gaussian NLL
    """
    dataset_info = DATASET_CATALOG[dataset_name]

    # Human MPRA datasets benefit from aggressive optimization (large batch + high LR)
    is_human_mpra = dataset_name.startswith("encode4_")

    if is_human_mpra:
        training_config = TrainingConfig(
            max_epochs=30,
            batch_size=1024,
            learning_rate=0.01,
            weight_decay=0.1,
            scheduler="onecycle",
            patience=30,
            use_balanced_sampling=False,
            use_extreme_weights=False,
        )
    else:
        training_config = TrainingConfig(
            max_epochs=50,
            batch_size=128,
            learning_rate=1e-3,
        )

    return ExperimentConfig(
        name=f"config1_{dataset_name}",
        config_type=ConfigurationType.SINGLE_CELLTYPE,
        description=f"Single cell type baseline for {dataset_name}",
        datasets=[dataset_name],
        target_sequence_length=dataset_info.sequence_length,
        model=ModelConfig(
            use_species_embedding=False,
            use_celltype_embedding=False,
            use_length_embedding=False,
            use_rc_stem=False,
            use_uncertainty=False if is_human_mpra else not use_mse,
        ),
        training=training_config,
    )


def get_config2_multi_celltype_human() -> ExperimentConfig:
    """Configuration 2: Multi-Cell-Type Human.

    Trains a single shared backbone across all 3 human ENCODE4 lentiMPRA cell types
    (K562, HepG2, WTC11). The model uses:
    - RC-equivariant stem (since human enhancers are strand-agnostic)
    - Cell-type embedding (32-dim) to condition predictions on the cell context
    - Length embedding (since sequences are padded from 230bp to 256bp)
    - Separate prediction heads per cell type

    Uses the aggressive MPRA recipe: batch 1024, LR 0.01, OneCycle scheduler.
    Temperature sampling (tau=0.5) balances the 3 datasets by size.
    """
    return ExperimentConfig(
        name="config2_multi_celltype_human",
        config_type=ConfigurationType.MULTI_CELLTYPE_HUMAN,
        description="Multi-cell-type human with shared backbone and cell-type conditioning",
        datasets=["encode4_k562", "encode4_hepg2", "encode4_wtc11"],
        target_sequence_length=256,
        model=ModelConfig(
            use_rc_stem=True,
            use_celltype_embedding=True,
            celltype_embed_dim=32,
            use_length_embedding=True,
        ),
        training=TrainingConfig(
            max_epochs=100,
            batch_size=1024,
            learning_rate=0.01,
            weight_decay=0.1,
            scheduler="onecycle",
            sampling_temperature=0.5,
            patience=15,
        ),
    )


def get_config3_cross_animal() -> ExperimentConfig:
    """Configuration 3: Cross-Animal (Human + Drosophila).

    Extends Config2 by adding the DeepSTARR Drosophila dataset, testing whether
    a shared backbone can learn regulatory grammar across animal species.
    Adds species embedding (16-dim) to distinguish human vs drosophila.
    Uses smaller batch (256) and lower LR (1e-3) since DeepSTARR sequences
    differ substantially from human MPRA.
    """
    return ExperimentConfig(
        name="config3_cross_animal",
        config_type=ConfigurationType.CROSS_ANIMAL,
        description="Cross-animal transfer between human and Drosophila",
        datasets=["encode4_k562", "encode4_hepg2", "encode4_wtc11", "deepstarr"],
        target_sequence_length=256,
        model=ModelConfig(
            use_rc_stem=True,
            use_species_embedding=True,
            species_embed_dim=16,
            use_celltype_embedding=True,
            celltype_embed_dim=32,
            use_length_embedding=True,
        ),
        training=TrainingConfig(
            max_epochs=100,
            batch_size=256,
            learning_rate=1e-3,
            sampling_temperature=0.5,
            patience=15,
            use_balanced_sampling=False,
        ),
    )


def get_config4_cross_kingdom() -> ExperimentConfig:
    """Configuration 4: Cross-Kingdom (Animal + Plant).

    Extends Config3 by adding Arabidopsis and Maize plant promoter datasets,
    testing cross-kingdom transfer between animals and plants. Adds kingdom
    embedding (8-dim) on top of species and cell-type embeddings.
    Uses lower temperature (tau=0.3) to upweight the smaller plant datasets
    relative to the much larger human/drosophila datasets.
    """
    return ExperimentConfig(
        name="config4_cross_kingdom",
        config_type=ConfigurationType.CROSS_KINGDOM,
        description="Cross-kingdom transfer between animals and plants",
        datasets=[
            "encode4_k562", "encode4_hepg2", "encode4_wtc11", "deepstarr",
            "jores_arabidopsis", "jores_maize",
        ],
        target_sequence_length=256,
        model=ModelConfig(
            use_rc_stem=True,
            use_kingdom_embedding=True,
            kingdom_embed_dim=8,
            use_species_embedding=True,
            species_embed_dim=16,
            use_celltype_embedding=True,
            celltype_embed_dim=32,
            use_length_embedding=True,
        ),
        training=TrainingConfig(
            max_epochs=100,
            batch_size=256,
            learning_rate=1e-3,
            scheduler="cosine",
            sampling_temperature=0.3,
            patience=15,
            use_balanced_sampling=False,
        ),
    )


def get_config5_universal() -> ExperimentConfig:
    """Configuration 5: Universal Foundation Model.

    The most ambitious configuration: trains across all 8 datasets, 7 species,
    and 3 kingdoms (animal, plant, fungi) with ~7.8M total sequences.

    Key differences from Config4:
    - Adds Sorghum (plant) and DREAM Yeast (fungi) datasets
    - Uses 3-phase training schedule (MultiPhaseTrainer):
        Phase 1 (50 epochs): Full model training at LR=1e-3 (pre-train backbone)
        Phase 2 (50 epochs): Freeze backbone, train heads only at LR=1e-3
        Phase 3 (50 epochs): Unfreeze all, fine-tune at LR=1e-4 (10x lower)
    - Caps samples_per_epoch=500K to prevent yeast (6.7M) from dominating
    - Uses temperature=0.3 and patience=20 for stable multi-phase convergence
    """
    return ExperimentConfig(
        name="config5_universal",
        config_type=ConfigurationType.UNIVERSAL,
        description="Universal foundation model across all species",
        datasets=[
            "encode4_k562", "encode4_hepg2", "encode4_wtc11", "deepstarr",
            "jores_arabidopsis", "jores_maize", "jores_sorghum",
            "dream_yeast",
        ],
        target_sequence_length=256,
        model=ModelConfig(
            use_rc_stem=True,
            use_kingdom_embedding=True,
            kingdom_embed_dim=8,
            use_species_embedding=True,
            species_embed_dim=16,
            use_celltype_embedding=True,
            celltype_embed_dim=32,
            use_length_embedding=True,
        ),
        training=TrainingConfig(
            max_epochs=150,
            batch_size=256,
            learning_rate=1e-3,
            scheduler="cosine",
            sampling_temperature=0.3,
            samples_per_epoch=500000,
            patience=20,
            use_balanced_sampling=False,
        ),
        # 3-phase universal training protocol:
        # Phase 1: Pre-train the full model (backbone + heads) at standard LR
        # Phase 2: Freeze backbone, train only the per-dataset heads (prevents catastrophic forgetting)
        # Phase 3: Unfreeze all and fine-tune end-to-end at 10x lower LR for final refinement
        training_phases=[
            {"name": "phase1_pretrain", "epochs": 50, "freeze_backbone": False, "lr": 1e-3},
            {"name": "phase2_head_training", "epochs": 50, "freeze_backbone": True, "lr": 1e-3},
            {"name": "phase3_finetune", "epochs": 50, "freeze_backbone": False, "lr": 1e-4},
        ],
    )


# =============================================================================
# PART 2: Training Coordinator
# Source: training/coordinator.py
# =============================================================================

def set_seed(seed: int):
    """Set random seeds for full reproducibility across Python, NumPy, and PyTorch.

    Also sets CUDA deterministic mode (disables cuDNN auto-tuner) which sacrifices
    ~5-10% speed for bitwise-reproducible results across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)          # All GPUs
        torch.backends.cudnn.deterministic = True  # Deterministic convolution algorithms
        torch.backends.cudnn.benchmark = False     # Disable auto-tuner (non-deterministic)


def print_config_summary(config: ExperimentConfig):
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("FUSEMAP Training Coordinator")
    print("=" * 60)
    print(f"\nExperiment: {config.name}")
    print(f"Config Type: {config.config_type.value}")
    print(f"\nDatasets ({len(config.datasets)}):")
    for ds in config.datasets:
        info = DATASET_CATALOG.get(ds)
        if info:
            print(f"  - {ds}: {info.sequence_length}bp, {info.num_outputs} outputs, {info.species}")

    print(f"\nModel:")
    print(f"  - RC stem: {config.model.use_rc_stem}")
    print(f"  - Loss: {'MSE' if not config.model.use_uncertainty else 'Gaussian NLL'}")

    print(f"\nTraining:")
    print(f"  - Max epochs: {config.training.max_epochs}")
    print(f"  - Batch size: {config.training.batch_size}")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Scheduler: {config.training.scheduler}")

    if config.training_phases:
        print(f"\nTraining Phases ({len(config.training_phases)}):")
        for phase in config.training_phases:
            print(f"  - {phase['name']}: {phase['epochs']} epochs, lr={phase['lr']}")

    print(f"\nOutput: {config.output_dir}/{config.name}")
    print("=" * 60 + "\n")


# =============================================================================
# PART 3: Trainer
# Source: training/trainer.py
# Full training loop with multi-dataset support, AMP, and checkpointing
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types (float32, int64, ndarray -> native Python)."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Trainer:
    """
    Main trainer class for FUSEMAP experiments.

    Features:
    - Multi-dataset training with balanced sampling
    - Per-epoch validation with NLL + MSE + correlation tracking
    - Early stopping based on average validation Pearson r
    - Checkpoint management with best model tracking
    - Mixed precision training (AMP) with gradient accumulation
    - Comprehensive logging to file and console
    """

    def __init__(
        self,
        config: ExperimentConfig,
        device: str = "cuda",
        resume_from: Optional[str] = None,
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.resume_from = resume_from

        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config.training.use_amp else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('-inf')
        self.patience_counter = 0

    def _setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.output_dir / "training.log"
        self.logger = logging.getLogger(self.config.name)
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(log_file)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info(f"Experiment: {self.config.name}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def setup_data(self):
        """Setup datasets, samplers, and data loaders.

        Creates MultiDataset for training, validation loaders per dataset,
        and optional test/calibration loaders for held-out evaluation.
        """
        self.logger.info("Setting up datasets...")
        # Full implementation: creates MultiDataset, samplers, loaders
        # See training/datasets.py and training/data_loaders.py for:
        #   - MultiDataset with activity normalization
        #   - TemperatureBalancedSampler or BalancedActivitySampler
        #   - Validation/test/calibration DataLoaders per dataset

    def setup_model(self):
        """Setup model, optimizer, and scheduler.

        Creates MultiSpeciesCADENCE with correct embedding sizes,
        AdamW optimizer, and LR scheduler (cosine/onecycle/plateau).
        """
        self.logger.info("Setting up model...")
        # Full implementation: creates MultiSpeciesCADENCE model
        # See training/models.py for create_multi_species_model()

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with gradient accumulation and AMP.

        Training loop flow:
        1. For each mini-batch, run forward pass under AMP autocast (FP16 compute)
        2. Compute the primary loss (Gaussian NLL or MSE, with optional extreme weighting)
        3. Also compute unweighted MSE for monitoring (not used for backprop)
        4. Scale loss by 1/accumulation_steps for gradient accumulation
        5. Backward pass through GradScaler (scales loss to prevent FP16 underflow)
        6. Every `accumulation_steps` mini-batches:
           a. Unscale gradients back to FP32
           b. Clip gradient norm to prevent explosions (especially with NLL loss)
           c. Optimizer step (via GradScaler which skips if grads contain inf/nan)
           d. Step OneCycleLR scheduler (it steps per batch, not per epoch)
           e. Zero gradients for next accumulation window

        Returns dict with train_loss (NLL), train_mse, train_grad_norm,
        and per-head losses (train_loss_{head_name}).
        """
        self.model.train()

        epoch_losses = []
        epoch_mse_losses = []
        epoch_grad_norms = []
        accumulation_steps = self.config.training.gradient_accumulation_steps
        accumulated_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move all tensors to GPU; dataset_names stays as a list of strings
            sequence = batch['sequence'].to(self.device)
            activity = batch['activity'].to(self.device)
            species_idx = batch['species_idx'].to(self.device)
            kingdom_idx = batch['kingdom_idx'].to(self.device)
            celltype_idx = batch['celltype_idx'].to(self.device)
            original_length = batch['original_length'].to(self.device)
            dataset_names = batch['dataset_names']

            # Forward pass under AMP autocast: convolutions run in FP16 for speed,
            # while accumulations and loss computation stay in FP32 for precision
            with autocast(enabled=self.config.training.use_amp):
                outputs = self.model(
                    sequence=sequence, species_idx=species_idx,
                    kingdom_idx=kingdom_idx, celltype_idx=celltype_idx,
                    original_length=original_length, dataset_names=dataset_names,
                )

                # Primary loss: Gaussian NLL (if uncertainty enabled) with extreme weighting
                loss, head_losses = self._compute_masked_loss(
                    outputs, activity, dataset_names,
                    use_uncertainty=self.config.model.use_uncertainty,
                    use_extreme_weights=self.config.training.use_extreme_weights,
                )

                # Secondary MSE for monitoring only (no uncertainty, no weighting)
                # This provides a comparable metric across loss function choices
                mse_loss, _ = self._compute_masked_loss(
                    outputs, activity, dataset_names,
                    use_uncertainty=False, use_extreme_weights=False,
                )
                # Scale loss for gradient accumulation: gradients from N mini-batches
                # are averaged to simulate a larger effective batch size
                loss = loss / accumulation_steps

            # Backward pass: GradScaler scales the loss to prevent FP16 gradient underflow
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()

            # Optimizer step: only after accumulating gradients from N mini-batches
            if (batch_idx + 1) % accumulation_steps == 0:
                # Unscale gradients from FP16 back to FP32 before clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                # Clip gradient norm to prevent explosive updates (especially important
                # for Gaussian NLL loss where log_var gradients can be large)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip,
                )

                # Step optimizer: GradScaler.step() skips the update if gradients
                # contain inf/nan (which can happen with AMP), then updates the scale
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # OneCycleLR steps per batch (not per epoch) -- this is different from
                # cosine/plateau schedulers which step per epoch in the outer loop
                if self.scheduler and isinstance(
                    self.scheduler, torch.optim.lr_scheduler.OneCycleLR
                ):
                    self.scheduler.step()

                self.optimizer.zero_grad()

                epoch_losses.append(accumulated_loss)
                epoch_mse_losses.append(mse_loss.item())
                epoch_grad_norms.append(
                    grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                )
                accumulated_loss = 0.0
                self.global_step += 1

                # Log periodically
                if self.global_step % self.config.training.log_every_n_steps == 0:
                    self.logger.info(
                        f"Step {self.global_step}: NLL={epoch_losses[-1]:.4f}, "
                        f"MSE={epoch_mse_losses[-1]:.4f}, "
                        f"grad={epoch_grad_norms[-1]:.2f}, "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                    )

        return {
            'train_loss': np.mean(epoch_losses) if epoch_losses else 0.0,
            'train_mse': np.mean(epoch_mse_losses) if epoch_mse_losses else 0.0,
            'train_grad_norm': np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0,
        }

    @staticmethod
    def _compute_masked_loss(
        outputs, targets, dataset_names,
        use_uncertainty=True, use_extreme_weights=False,
        extreme_alpha=1.0, extreme_beta=2.0,
    ):
        """
        Compute masked multi-dataset loss with optional Gaussian NLL and extreme weighting.

        In a multi-dataset batch, different samples belong to different datasets and
        only their corresponding prediction head should contribute to the loss. This
        function handles that routing via dataset-name masking.

        Loss computation per head:
          1. Extract samples belonging to this dataset (mask by dataset_name)
          2. Remove NaN targets (from NaN-padding in collate_multi_dataset)
          3. Compute per-sample loss:
             - If use_uncertainty: Gaussian NLL = 0.5 * (exp(-log_var) * (y - mu)^2 + log_var)
               This lets the model learn heteroscedastic uncertainty: high log_var means
               "I'm not confident about this prediction" and reduces the penalty.
             - If not use_uncertainty: simple MSE = (y - mu)^2
          4. If use_extreme_weights: multiply each sample's loss by
             w_i = 1 + alpha * |z_i|^beta, where z_i is the within-batch z-score
             of the target. This upweights rare extreme-activity samples (tails).
          5. Sum losses across heads, normalize by total number of valid samples.

        Args:
            outputs: Dict of {head_name: {"mean": tensor, "log_var": tensor}} from model
            targets: Tensor [B, max_outputs] with NaN for unused output slots
            dataset_names: List[str] of length B identifying each sample's source dataset
            use_uncertainty: If True, use Gaussian NLL; if False, use MSE
            use_extreme_weights: If True, apply extreme value weighting to loss
            extreme_alpha: Scaling factor for extreme weights (default 1.0)
            extreme_beta: Exponent for extreme weights (default 2.0, quadratic)

        Returns:
            (total_loss, head_losses): Scalar loss and dict of per-head mean losses
        """
        total_loss = 0.0
        head_losses = {}
        n_total = 0

        # Group samples by their source dataset
        unique_datasets = set(dataset_names)

        for ds_name in unique_datasets:
            # Build mask: indices of samples in this batch that belong to ds_name
            mask = [i for i, n in enumerate(dataset_names) if n == ds_name]
            if not mask:
                continue

            mask_t = torch.tensor(mask, dtype=torch.long, device=targets.device)
            ds_targets = targets[mask_t]

            # Match this dataset's samples to the correct prediction head
            for head_name, pred_data in outputs.items():
                if not isinstance(pred_data, dict) or 'mean' not in pred_data:
                    continue
                # Each head is named after its dataset; skip heads for other datasets
                if ds_name not in head_name:
                    continue

                pred_mean = pred_data['mean'][mask_t]
                target = ds_targets[:, 0] if ds_targets.dim() > 1 else ds_targets

                # Remove NaN targets (these come from NaN-padding in collate_multi_dataset
                # when datasets have different numbers of outputs)
                valid = ~torch.isnan(target)
                if valid.sum() == 0:
                    continue

                pred_mean = pred_mean[valid]
                target = target[valid]

                # Compute per-sample loss
                if use_uncertainty and 'log_var' in pred_data:
                    # Gaussian NLL: -log p(y|mu, sigma^2) = 0.5 * (1/sigma^2 * (y-mu)^2 + log(sigma^2))
                    # where log_var = log(sigma^2), so precision = 1/sigma^2 = exp(-log_var)
                    log_var = pred_data['log_var'][mask_t][valid]
                    precision = torch.exp(-log_var)
                    loss = 0.5 * (precision * (target - pred_mean) ** 2 + log_var)
                else:
                    # Simple MSE loss
                    loss = (target - pred_mean) ** 2

                # Extreme value weighting: upweight samples at tails of the distribution
                # w_i = 1 + alpha * |z_i|^beta, where z_i = (y_i - mean) / std
                # With default alpha=1, beta=2: a sample at z=2 gets weight 1+4=5x,
                # while a sample at z=0 (mean) gets weight 1x (no boost).
                if use_extreme_weights:
                    with torch.no_grad():
                        z = (target - target.mean()) / (target.std() + 1e-8)
                        weights = 1.0 + extreme_alpha * torch.abs(z) ** extreme_beta
                    loss = loss * weights

                head_loss = loss.sum()
                total_loss = total_loss + head_loss
                n_total += valid.sum().item()
                head_losses[head_name] = head_loss / valid.sum()

        # Normalize by total number of valid samples across all heads
        if n_total > 0:
            total_loss = total_loss / n_total

        return total_loss, head_losses

    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save training checkpoint with full state."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_metric': self.best_val_metric,
            'config': asdict(self.config),
        }
        torch.save(checkpoint, filepath)

        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, filepath: str):
        """Load training checkpoint and resume state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_val_metric = checkpoint.get('best_val_metric', float('-inf'))
        self.logger.info(f"Resumed from epoch {self.current_epoch}")

    def train(self):
        """Main training loop with validation and early stopping.

        Overall flow:
        1. Setup data (MultiDataset + samplers + DataLoaders)
        2. Setup model (CADENCE backbone + optimizer + scheduler)
        3. For each epoch:
           a. Run train_epoch() (one pass through the sampled training data)
           b. Validate periodically (compute Pearson r per dataset per output)
           c. Track best average Pearson r across all datasets
           d. Save checkpoint (and separately save best model)
           e. Early stopping if no improvement for `patience` epochs
        4. Update LR scheduler:
           - ReduceLROnPlateau: steps based on val metric
           - CosineAnnealingLR: steps per epoch
           - OneCycleLR: already stepped per batch in train_epoch()
        5. Final evaluation on validation set, save results JSON
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Config: {self.config.config_type.value}")

        self.setup_data()
        self.setup_model()

        # Save config for reproducibility
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)

        for epoch in range(self.current_epoch, self.config.training.max_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            train_metrics = self.train_epoch()

            if epoch % self.config.training.val_every_n_epochs == 0:
                val_metrics = self.validate()

                # Compute average Pearson r across ALL datasets and ALL outputs
                # (e.g., for DeepSTARR this averages Dev and Hk correlations)
                val_pearsons = []
                for ds_metrics in val_metrics.values():
                    for out_metrics in ds_metrics.values():
                        if isinstance(out_metrics, dict) and 'pearson' in out_metrics:
                            val_pearsons.append(out_metrics['pearson']['value'])

                avg_val_pearson = np.mean(val_pearsons) if val_pearsons else 0.0

                # Early stopping logic: track best metric and count stagnant epochs
                is_best = avg_val_pearson > self.best_val_metric
                if is_best:
                    self.best_val_metric = avg_val_pearson
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                self.logger.info(
                    f"Epoch {epoch}: loss={train_metrics['train_loss']:.4f}, "
                    f"val_r={avg_val_pearson:.4f} {'(BEST)' if is_best else ''}"
                )

                self.save_checkpoint(
                    self.output_dir / f"checkpoint_epoch{epoch}.pt", is_best=is_best,
                )

                if self.patience_counter >= self.config.training.patience:
                    self.logger.info(f"Early stopping after {epoch} epochs")
                    break

            # Update per-epoch LR schedulers (OneCycleLR is handled per-batch in train_epoch)
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_pearson)  # Needs the metric value
                elif not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()  # CosineAnnealingLR etc.

            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")

        # Final evaluation on best model
        self.logger.info("Training complete. Running final evaluation...")
        final_metrics = self.validate()

        results_path = self.output_dir / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_metrics, f, indent=2, cls=NumpyEncoder)

        return final_metrics


class MultiPhaseTrainer(Trainer):
    """
    Trainer for multi-phase training (Config 5: Universal Foundation Model).

    The 3-phase protocol addresses the challenge of training a single model across
    vastly different datasets (7 species, 3 kingdoms, ~7.8M sequences):

    Phase 1 - Pre-train (50 epochs, LR=1e-3, all params trainable):
        Learn shared regulatory grammar across all species. The backbone discovers
        universal motif patterns (e.g., TATA box, GC-richness effects).

    Phase 2 - Head training (50 epochs, LR=1e-3, backbone frozen):
        Freeze the backbone and train only the per-dataset prediction heads.
        This prevents catastrophic forgetting of the shared features while
        letting each head specialize for its dataset's activity scale and distribution.

    Phase 3 - Fine-tune (50 epochs, LR=1e-4, all params trainable):
        Unfreeze everything and fine-tune end-to-end at 10x lower LR.
        This allows subtle backbone adjustments while preserving the learned structure.

    Each phase has its own early stopping with patience = config.patience // 2
    (more aggressive stopping per phase since total budget is split across 3 phases).
    """

    def train(self):
        """Multi-phase training loop. Falls back to single-phase if no phases defined."""
        self.logger.info("Starting multi-phase training...")
        phases = self.config.training_phases or []

        if not phases:
            return super().train()

        self.setup_data()
        self.setup_model()

        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)

        for phase_idx, phase in enumerate(phases):
            phase_name = phase.get('name', f'phase_{phase_idx}')
            phase_epochs = phase.get('epochs', 50)
            freeze_backbone = phase.get('freeze_backbone', False)
            phase_lr = phase.get('lr', self.config.training.learning_rate)

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"PHASE {phase_idx + 1}: {phase_name}")
            self.logger.info(f"  Epochs: {phase_epochs}, LR: {phase_lr}, "
                           f"Freeze backbone: {freeze_backbone}")
            self.logger.info(f"{'='*60}\n")

            if freeze_backbone:
                self.model.freeze_backbone()
            else:
                self.model.unfreeze_backbone()

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = phase_lr

            self.patience_counter = 0
            phase_start_epoch = self.current_epoch

            for epoch in range(phase_epochs):
                self.current_epoch = phase_start_epoch + epoch
                train_metrics = self.train_epoch()

                val_metrics = self.validate()
                val_pearsons = []
                for ds_m in val_metrics.values():
                    for out_m in ds_m.values():
                        if isinstance(out_m, dict) and 'pearson' in out_m:
                            val_pearsons.append(out_m['pearson']['value'])

                avg_val_pearson = np.mean(val_pearsons) if val_pearsons else 0.0
                is_best = avg_val_pearson > self.best_val_metric

                if is_best:
                    self.best_val_metric = avg_val_pearson
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                self.logger.info(
                    f"[{phase_name}] Epoch {epoch}: loss={train_metrics['train_loss']:.4f}, "
                    f"val_r={avg_val_pearson:.4f} {'(BEST)' if is_best else ''}"
                )

                self.save_checkpoint(
                    self.output_dir / f"checkpoint_{phase_name}_epoch{epoch}.pt",
                    is_best=is_best,
                )

                if self.patience_counter >= self.config.training.patience // 2:
                    self.logger.info(f"Early stopping phase {phase_name}")
                    break

        self.logger.info("Multi-phase training complete.")
        final_metrics = self.validate()
        results_path = self.output_dir / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_metrics, f, indent=2, cls=NumpyEncoder)
        return final_metrics


# =============================================================================
# PART 4: Therapeutic Enhancer Design Pipeline
# Source: applications/therapeutic_enhancer_pipeline.py
# =============================================================================

@dataclass
class MotifConstraintResult:
    """Result of motif constraint checking."""
    passed: bool
    required_found: List[str]
    required_missing: List[str]
    forbidden_found: List[str]
    total_motif_hits: int
    message: str


@dataclass
class OracleCheckResult:
    """Result of OracleCheck sequence composition validation.

    Verdict thresholds:
      GREEN:  No flags (all composition metrics in ideal range)
      YELLOW: Exactly 1 minor flag and homopolymer <= 10bp
      RED:    Multiple flags or severe issues (sequence should be rejected)
    """
    verdict: str          # GREEN (safe), YELLOW (marginal), or RED (reject)
    gc_content: float     # Fraction of G+C bases (ideal: 0.45-0.55)
    max_homopolymer: int  # Length of longest single-nucleotide run (ideal: < 8)
    has_cpg_island: bool  # Whether sequence contains a CpG island (potential silencing risk)
    repeat_fraction: float  # Fraction of sequence in dinucleotide repeats (ideal: < 0.1)
    flags: List[str]      # Human-readable descriptions of each failed check


@dataclass
class EnhancerCandidate:
    """Complete candidate enhancer with all evaluation results from the 6-step pipeline."""
    sequence: str                # The 230bp DNA sequence
    source: str                  # Origin: "vae_generated", "natural", "mutagenesis", etc.

    # CADENCE ensemble predictions (Step 3)
    activities: Dict[str, float] = field(default_factory=dict)       # {cell_type: predicted_activity}
    uncertainties: Dict[str, float] = field(default_factory=dict)    # {cell_type: predicted_uncertainty}

    # Specificity metrics (Step 6)
    target_activity: float = 0.0     # Predicted activity in the target cell type
    max_background: float = 0.0      # Highest activity across background cell types
    specificity_score: float = 0.0   # target - 0.5 * sum(background), higher = more specific

    # Filter results (Steps 4-5)
    oracle_verdict: str = 'UNKNOWN'                           # GREEN, YELLOW, or RED from OracleCheck
    motif_result: Optional[MotifConstraintResult] = None      # Required/forbidden motif check results

    passed_all_filters: bool = False  # True only if OracleCheck is GREEN/YELLOW AND motifs pass


# Cell-type specific motif constraints for therapeutic enhancer design.
# "required" = TF motifs that MUST be present for cell-type-specific activity.
# "forbidden" = TF motifs that should be ABSENT to avoid off-target activation.
# These are derived from known lineage-determining transcription factors:
#   HepG2 (liver): HNF4A, FOXA1/2 (hepatocyte TFs), CEBPA/B (liver enriched)
#   K562 (erythroid): GATA1/2, TAL1, KLF1 (erythroid master regulators)
#   WTC11 (iPSC): POU5F1/NANOG/SOX2 (Yamanaka factors), KLF4 (pluripotency)
CELL_TYPE_MOTIF_CONSTRAINTS = {
    'HepG2': {
        'required': ['HNF4A', 'FOXA1', 'FOXA2', 'CEBPA', 'CEBPB'],
        'forbidden': ['GATA1', 'GATA2', 'GATA3', 'POU5F1', 'NANOG', 'SOX2']
    },
    'K562': {
        'required': ['GATA1', 'GATA2', 'TAL1', 'KLF1', 'RUNX1'],
        'forbidden': ['HNF4A', 'FOXA1', 'FOXA2']
    },
    'WTC11': {
        'required': ['POU5F1', 'NANOG', 'SOX2', 'KLF4'],
        'forbidden': ['GATA1', 'HNF4A']
    },
}


class TherapeuticEnhancerPipeline:
    """
    Complete pipeline for designing cell-type-specific therapeutic enhancers.

    The goal is to produce synthetic enhancer sequences that are:
    - Highly active in the target cell type (e.g., HepG2 for liver gene therapy)
    - Inactive in background cell types (specificity)
    - Synthesizable and safe (no toxic sequence features)
    - Biologically plausible (contain expected TF binding motifs)

    6-Step Protocol:
    1. PHYSICS EXTRACTION: Extract electrostatic/shape profiles from the top
       natural enhancers in the target cell type. These profiles define the
       "physics target" that generated sequences should match.

    2. VAE GENERATION: Feed the target physics profiles into PhysicsVAE to
       generate candidate sequences conditioned on those biophysical properties.
       Produces n_vae_candidates (default 500) diverse candidates.

    3. ACTIVITY PREDICTION: Run each candidate through a CADENCE ensemble to
       predict activity in the target cell type AND all background cell types.
       The ensemble also provides uncertainty estimates.

    4. ORACLECHECK VALIDATION: Filter candidates through OracleCheck, which
       flags sequences with problematic composition (extreme GC, long homopolymer
       runs, CpG islands, repetitive elements). Only GREEN/YELLOW candidates pass.

    5. MOTIF CONSTRAINT CHECKING: Verify that required TF binding motifs are
       present (e.g., HNF4A/FOXA for liver) and forbidden motifs are absent
       (e.g., GATA1 for non-erythroid targets). Uses cell-type-specific constraints.

    6. SPECIFICITY SCORING + DIVERSITY: Rank candidates by specificity score
       (target activity minus penalized background activity), then apply diversity
       filtering to avoid selecting redundant sequences. Returns top n_select.

    Paper results: 99% predicted specificity, 96.5% GREEN OracleCheck (HepG2 target).
    """

    def __init__(
        self,
        target_cell: str = 'HepG2',
        background_cells: List[str] = None,
        device: str = 'cuda',
        required_motifs: List[str] = None,
        forbidden_motifs: List[str] = None,
    ):
        self.target_cell = target_cell
        self.background_cells = background_cells or [
            c for c in ['K562', 'HepG2', 'WTC11'] if c != target_cell
        ]
        self.all_cell_types = [target_cell] + self.background_cells
        self.device = device

        # Motif constraints
        if required_motifs is None and target_cell in CELL_TYPE_MOTIF_CONSTRAINTS:
            required_motifs = CELL_TYPE_MOTIF_CONSTRAINTS[target_cell]['required']
        if forbidden_motifs is None and target_cell in CELL_TYPE_MOTIF_CONSTRAINTS:
            forbidden_motifs = CELL_TYPE_MOTIF_CONSTRAINTS[target_cell]['forbidden']
        self.required_motifs = required_motifs or []
        self.forbidden_motifs = forbidden_motifs or []

        print(f"Initializing Therapeutic Enhancer Pipeline")
        print(f"  Target: {target_cell}, Background: {self.background_cells}")
        print(f"  Required motifs: {self.required_motifs}")
        print(f"  Forbidden motifs: {self.forbidden_motifs}")

        self.candidates: List[EnhancerCandidate] = []

    def check_oracle(self, sequence: str) -> OracleCheckResult:
        """
        OracleCheck validation.

        Checks composition constraints:
        - GC content (45-55% ideal)
        - Homopolymer runs (<8bp)
        - CpG islands
        - Repeat content
        """
        flags = []
        seq_upper = sequence.upper()

        if len(seq_upper) == 0:
            return OracleCheckResult(
                verdict='RED', gc_content=0.0, max_homopolymer=0,
                has_cpg_island=False, repeat_fraction=0.0, flags=['Empty sequence']
            )

        # GC content
        gc = (seq_upper.count('G') + seq_upper.count('C')) / len(seq_upper)
        if gc < 0.35 or gc > 0.65:
            flags.append(f"GC content {gc:.2f} outside range")

        # Homopolymer runs
        max_homo = 1
        current = 1
        for i in range(1, len(seq_upper)):
            if seq_upper[i] == seq_upper[i-1]:
                current += 1
                max_homo = max(max_homo, current)
            else:
                current = 1
        if max_homo > 8:
            flags.append(f"Homopolymer run of {max_homo}")

        # CpG islands
        has_cpg = False
        cpg_count = seq_upper.count('CG')
        expected_cpg = (seq_upper.count('C') * seq_upper.count('G')) / len(seq_upper)
        if expected_cpg > 0 and cpg_count / expected_cpg > 0.6 and gc > 0.5:
            has_cpg = True
            flags.append("Potential CpG island")

        # Repeat content
        repeat_count = 0
        for dinuc in ['AT', 'TA', 'GC', 'CG', 'AC', 'CA', 'GT', 'TG']:
            repeat_count += seq_upper.count(dinuc * 4)
        repeat_frac = repeat_count / len(seq_upper) if len(seq_upper) > 0 else 0
        if repeat_frac > 0.1:
            flags.append(f"High repeat content {repeat_frac:.2f}")

        # Determine verdict
        if len(flags) == 0:
            verdict = 'GREEN'
        elif len(flags) == 1 and max_homo <= 10:
            verdict = 'YELLOW'
        else:
            verdict = 'RED'

        return OracleCheckResult(
            verdict=verdict, gc_content=gc, max_homopolymer=max_homo,
            has_cpg_island=has_cpg, repeat_fraction=repeat_frac, flags=flags
        )

    def compute_specificity(
        self,
        target_activity: float,
        background_activities: List[float]
    ) -> float:
        """Compute cell-type specificity score.

        Formula: specificity = target_activity - 0.5 * sum(background_activities)

        A high specificity score means the enhancer is strongly active in the
        target cell type while having low activity in background cell types.
        The 0.5 penalty factor means background activity is penalized at half
        weight, allowing some background activity as long as target is much higher.
        """
        return target_activity - 0.5 * sum(background_activities)

    def run_full_pipeline(
        self,
        natural_sequences: List[str] = None,
        n_vae_candidates: int = 500,
        n_select: int = 50,
        output_dir: str = None
    ) -> List[EnhancerCandidate]:
        """Run the complete 6-step therapeutic enhancer design pipeline."""
        print("\n" + "="*60)
        print("THERAPEUTIC ENHANCER DESIGN PIPELINE")
        print("="*60)
        print(f"Target cell: {self.target_cell}")
        print(f"Background cells: {self.background_cells}")
        print("="*60)

        # Step 1: Extract physics profile from natural enhancers
        # Step 2: Generate VAE candidates
        # Step 3: Predict activities
        # Step 4: OracleCheck validation
        # Step 5: Motif constraint filtering
        # Step 6: Rank and diversify

        # See full implementation in applications/therapeutic_enhancer_pipeline.py

        return self.candidates


# =============================================================================
# PART 5: Disease Variant Pipeline
# Source: applications/disease_variant_pipeline.py
# =============================================================================

@dataclass
class VariantEffect:
    """Complete variant effect analysis for a single SNP/indel.

    Stores the reference and alternate sequences centered on the variant,
    their predicted regulatory activities, and the computed effect metrics.
    """
    variant_id: str          # Variant identifier (e.g., "rs12345" or "chr1:1000:A>G")
    ref_sequence: str        # Reference allele sequence with flanking context
    alt_sequence: str        # Alternate allele sequence with flanking context

    activity_ref: float              # CADENCE-predicted activity for reference allele
    activity_alt: float              # CADENCE-predicted activity for alternate allele
    delta_activity: float            # activity_alt - activity_ref (positive = gain of function)
    delta_activity_zscore: float     # delta_activity normalized by dataset distribution std

    effect_direction: str  # 'activating' (positive delta), 'repressing' (negative), or 'neutral'
    effect_magnitude: str  # 'strong' (|z| > 2), 'moderate' (1 < |z| < 2), or 'weak' (|z| < 1)

    # Per-physics-feature changes (from PhysInformer), e.g., {"electrostatic": -0.3, "shape": 0.1}
    # These explain WHY the variant affects activity (biophysical mechanism)
    delta_physics: Dict[str, float] = field(default_factory=dict)


class DiseaseVariantPipeline:
    """
    Complete pipeline for interpreting non-coding disease-associated variants.

    Given a set of genetic variants (SNPs/indels) in regulatory regions, this pipeline
    predicts which variants alter enhancer/promoter activity and explains the biophysical
    mechanism behind the effect.

    Pipeline Steps:
    1. LOAD VARIANTS: Parse from VCF file or pandas DataFrame with columns
       (chrom, pos, ref, alt, variant_id)
    2. EXTRACT SEQUENCES: For each variant, extract reference and alternate allele
       sequences with `flank_size` bp of flanking context on each side.
       Total sequence = 2 * flank_size + len(ref/alt) ~ 230bp for ENCODE4 compatibility.
    3. PREDICT ACTIVITY: Run both ref and alt sequences through CADENCE to get
       predicted regulatory activity. Compute delta = alt - ref.
    4. PHYSICS ATTRIBUTION: Run both sequences through PhysInformer to identify
       which biophysical features changed (e.g., electrostatic potential, DNA shape).
       This provides mechanistic interpretability beyond just "activity changed".
    5. SCORE AND RANK: Normalize delta_activity to z-scores using the training
       distribution, classify variants as activating/repressing/neutral with
       strong/moderate/weak magnitude thresholds.
    6. GENERATE REPORT: Output ranked list of variants with predicted effects,
       biophysical explanations, and summary statistics.
    """

    def __init__(
        self,
        reference_genome: str = None,
        cadence_checkpoint: str = None,
        physinformer_checkpoint: str = None,
        cell_type: str = 'K562',
        flank_size: int = 115,
        device: str = 'cuda'
    ):
        self.cell_type = cell_type
        self.flank_size = flank_size    # bp of context on each side of variant (115 + 115 = 230bp total)
        self.device = device

        self.variant_effects: List[VariantEffect] = []  # Accumulated results from analyze_variant()

    def analyze_variant(
        self,
        ref_seq: str,
        alt_seq: str,
        variant_id: str = None
    ) -> VariantEffect:
        """Analyze a single variant."""
        # In full implementation:
        # 1. Predict activity for ref and alt
        # 2. Compute physics features
        # 3. Calculate effect size and direction
        # See applications/disease_variant_pipeline.py

        return VariantEffect(
            variant_id=variant_id or "unknown",
            ref_sequence=ref_seq,
            alt_sequence=alt_seq,
            activity_ref=0.0,
            activity_alt=0.0,
            delta_activity=0.0,
            delta_activity_zscore=0.0,
            effect_direction='neutral',
            effect_magnitude='weak',
        )

    def score_and_rank(
        self,
        min_zscore: float = None,
    ) -> List[VariantEffect]:
        """Score and rank variants by absolute effect magnitude (largest first).

        Args:
            min_zscore: If set, filter out variants with |delta_activity_zscore| below this
                        threshold. Typical values: 1.0 (any effect), 2.0 (significant only).
        """
        effects = sorted(
            self.variant_effects,
            key=lambda x: abs(x.delta_activity_zscore),
            reverse=True
        )

        if min_zscore is not None:
            effects = [e for e in effects if abs(e.delta_activity_zscore) >= min_zscore]

        return effects

    def generate_report(
        self,
        output_path: str = None,
        top_n: int = 50
    ) -> Dict:
        """Generate comprehensive variant interpretation report."""
        report = {
            'pipeline_info': {
                'cell_type': self.cell_type,
                'flank_size': self.flank_size,
                'n_variants_analyzed': len(self.variant_effects),
            },
            'summary_statistics': {
                'n_activating': sum(1 for e in self.variant_effects if e.effect_direction == 'activating'),
                'n_repressing': sum(1 for e in self.variant_effects if e.effect_direction == 'repressing'),
                'n_neutral': sum(1 for e in self.variant_effects if e.effect_direction == 'neutral'),
            },
            'top_variants': []
        }

        sorted_effects = self.score_and_rank()
        for effect in sorted_effects[:top_n]:
            report['top_variants'].append({
                'variant_id': effect.variant_id,
                'delta_activity': effect.delta_activity,
                'zscore': effect.delta_activity_zscore,
                'direction': effect.effect_direction,
                'magnitude': effect.effect_magnitude,
            })

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {output_path}")

        return report


# =============================================================================
# PART 6: Dataset Classes
# Source: training/datasets.py
# =============================================================================

@dataclass
class ActivityNormalizerStats:
    """Statistics for activity normalization."""
    mean: np.ndarray
    std: np.ndarray
    min_val: np.ndarray
    max_val: np.ndarray


class ActivityNormalizer:
    """
    Per-dataset z-score normalization for activity values.

    Different datasets have vastly different activity scales and distributions:
    - ENCODE4 lentiMPRA: log2(RNA/DNA) ~ range [-5, 5]
    - DeepSTARR: log2(enrichment) ~ range [-2, 8]
    - DREAM Yeast: expression level ~ range [0, 15]
    - Jores Plant: log2(activity) ~ range [-3, 3]

    Z-score normalization (x' = (x - mean) / std) puts all datasets on a
    comparable scale so the model can learn shared features without one dataset's
    scale dominating the loss. The normalizer stores per-dataset statistics
    and supports forward (transform) and inverse (inverse_transform) operations.
    """

    def __init__(self):
        self.stats: Dict[str, ActivityNormalizerStats] = {}

    def fit(self, dataset_name: str, activities: np.ndarray):
        """
        Compute mean/std for a dataset.

        Args:
            dataset_name: Name of the dataset
            activities: Array of activities [N] or [N, num_outputs]
        """
        if activities.ndim == 1:
            activities = activities.reshape(-1, 1)

        self.stats[dataset_name] = ActivityNormalizerStats(
            mean=np.nanmean(activities, axis=0),
            std=np.nanstd(activities, axis=0),
            min_val=np.nanmin(activities, axis=0),
            max_val=np.nanmax(activities, axis=0),
        )

        # Prevent division by zero
        self.stats[dataset_name].std = np.maximum(
            self.stats[dataset_name].std, 1e-8
        )

    def transform(
        self,
        dataset_name: str,
        activities: np.ndarray,
    ) -> np.ndarray:
        """Z-score normalize."""
        if dataset_name not in self.stats:
            return activities

        s = self.stats[dataset_name]

        if activities.ndim == 1:
            return (activities - s.mean[0]) / s.std[0]

        return (activities - s.mean) / s.std

    def inverse_transform(
        self,
        dataset_name: str,
        normalized: np.ndarray,
    ) -> np.ndarray:
        """Convert back to original scale."""
        if dataset_name not in self.stats:
            return normalized

        s = self.stats[dataset_name]

        if normalized.ndim == 1:
            return normalized * s.std[0] + s.mean[0]

        return normalized * s.std + s.mean

    def save(self, filepath: str):
        """Save normalizer stats to file."""
        data = {
            name: {
                "mean": stats.mean.tolist(),
                "std": stats.std.tolist(),
                "min": stats.min_val.tolist(),
                "max": stats.max_val.tolist(),
            }
            for name, stats in self.stats.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load normalizer stats from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        for name, vals in data.items():
            self.stats[name] = ActivityNormalizerStats(
                mean=np.array(vals["mean"]),
                std=np.array(vals["std"]),
                min_val=np.array(vals["min"]),
                max_val=np.array(vals["max"]),
            )


def reverse_complement_onehot(seq: np.ndarray) -> np.ndarray:
    """
    Get reverse complement of one-hot encoded sequence.

    For one-hot encoding [A, C, G, T] = [0, 1, 2, 3]:
    - A <-> T means swap channels 0 and 3
    - C <-> G means swap channels 1 and 2
    - Then reverse the sequence order

    Args:
        seq: One-hot encoded sequence [4, L]
    Returns:
        Reverse complement [4, L]
    """
    # Complement: swap A<->T (0<->3) and C<->G (1<->2)
    complement = seq[[3, 2, 1, 0], :]
    # Reverse
    return complement[:, ::-1].copy()


def pad_or_crop_sequence(
    sequence: np.ndarray,
    target_length: int,
    pad_value: float = 0.25,
) -> Tuple[np.ndarray, int]:
    """
    Pad or center-crop a one-hot encoded sequence to a fixed target length.

    Padding uses 0.25 for all 4 channels, representing a uniform distribution
    over nucleotides (i.e., "no information" at this position). This is
    preferable to zero-padding because it does not bias the model toward
    interpreting padding as "not-A, not-C, not-G, not-T".

    Cropping is center-aligned: removes equal flanking regions from both ends.
    Padding is symmetric: adds equal amounts to both ends.

    Args:
        sequence: One-hot encoded sequence [4, L] or [L, 4]
        target_length: Desired length in bp
        pad_value: Value for padding channels (0.25 = uniform over {A,C,G,T})

    Returns:
        (padded_or_cropped_sequence [4, target_length], original_length)
    """
    # Ensure [4, L] format
    if sequence.shape[0] != 4:
        sequence = sequence.T

    original_length = sequence.shape[1]

    if original_length == target_length:
        return sequence, original_length

    if original_length > target_length:
        # Center crop
        start = (original_length - target_length) // 2
        return sequence[:, start:start + target_length], original_length

    # Pad symmetrically
    pad_left = (target_length - original_length) // 2
    pad_right = target_length - original_length - pad_left

    padded = np.full((4, target_length), pad_value, dtype=sequence.dtype)
    padded[:, pad_left:pad_left + original_length] = sequence

    return padded, original_length


class SingleDataset(torch.utils.data.Dataset):
    """
    Dataset for a single data source with augmentation support.

    Handles loading, normalization, and length adjustment for one dataset.

    Data augmentation strategies (applied dynamically in __getitem__, not cached):
    1. SHIFT AUGMENTATION (human MPRA only): Randomly shifts the sequence ±21bp,
       filling exposed positions with uniform 0.25 (no information). This matches
       the HumanLegNet training protocol and teaches the model position invariance.
       Only for ENCODE4 datasets where the 230bp window is a fixed crop of a longer
       regulatory region.

    2. REVERSE COMPLEMENT (RC) AUGMENTATION: Two modes:
       a. double_data_with_rc=True: Dataset is doubled -- first half is original
          sequences, second half is all RC. This is deterministic (same RC every epoch).
       b. use_augmentation=True (default for training): Each sequence is randomly
          RC-flipped with 50% probability. This is stochastic (different each epoch).
       RC augmentation is critical because regulatory activity is strand-independent:
       a motif on either strand should predict the same activity.

    3. PAD/CROP: All sequences are padded (with 0.25 uniform) or center-cropped
       to target_length (typically 256bp) for batching.
    """

    # Shift augmentation parameters (matching HumanLegNet benchmark protocol)
    MAX_SHIFT = 21  # ±21bp random shift range

    def __init__(
        self,
        dataset_info: DatasetInfo,
        split: str = "train",
        target_length: int = 256,
        normalizer: Optional[ActivityNormalizer] = None,
        fold: Optional[int] = None,
        transform: Optional[callable] = None,
        index_mappings: Optional[Dict[str, Dict[str, int]]] = None,
        use_augmentation: bool = True,
        double_data_with_rc: bool = False,
        use_shift: bool = True,
    ):
        self.info = dataset_info
        self.split = split
        self.target_length = target_length
        self.normalizer = normalizer
        self.fold = fold                   # K-fold split index (1-10 for ENCODE4 kfold validation)
        self.transform = transform
        # double_data_with_rc: deterministic RC doubling (only during training)
        # Mutually exclusive with stochastic RC augmentation
        self.double_data_with_rc = double_data_with_rc and (split == "train")
        # use_augmentation: stochastic 50% RC flip (only during training, disabled if double_data)
        self.use_augmentation = use_augmentation and (split == "train") and not self.double_data_with_rc
        # Shift augmentation: only during training, actual application gated by _is_human_mpra
        self._use_shift = use_shift and (split == "train")

        # Determine if this is a human MPRA dataset (for shift augmentation)
        self._is_human_mpra = dataset_info.name.lower().startswith("encode4_")

        # Build or use provided index mappings
        if index_mappings is not None:
            self.species_to_idx = index_mappings.get("species", {dataset_info.species: 0})
            self.kingdom_to_idx = index_mappings.get("kingdom", {dataset_info.kingdom: 0})
            self.celltype_to_idx = index_mappings.get("celltype", {(dataset_info.cell_type or "unknown"): 0})
        else:
            self.species_to_idx = {dataset_info.species: 0}
            self.kingdom_to_idx = {dataset_info.kingdom: 0}
            self.celltype_to_idx = {(dataset_info.cell_type or "unknown"): 0}

        # Load data
        self._load_data()

        # Fit normalizer if training and not already fitted
        if split == "train" and normalizer is not None:
            if dataset_info.name not in normalizer.stats:
                normalizer.fit(dataset_info.name, self.activities)

    def _load_data(self):
        """Load data from files."""
        data_path = Path(self.info.path)

        if (data_path / f"{self.split}.h5").exists():
            self._load_h5(data_path / f"{self.split}.h5")
        elif (data_path / f"{self.split}.npz").exists():
            self._load_npz(data_path / f"{self.split}.npz")
        elif (data_path / f"{self.split}_sequences.npy").exists():
            self._load_npy(data_path, self.split)
        else:
            self._load_from_real_data()

    def _load_h5(self, filepath: Path):
        """Load from HDF5 file."""
        import h5py
        with h5py.File(filepath, 'r') as f:
            self.sequences = np.array(f['sequences'])
            self.activities = np.array(f['activities'])
            self.weights = np.array(f['weights']) if 'weights' in f else None

    def _load_npz(self, filepath: Path):
        """Load from NPZ file."""
        data = np.load(filepath, allow_pickle=True)
        self.sequences = data['sequences']
        self.activities = data['activities']
        self.weights = data.get('weights', None)

    def _load_npy(self, data_path: Path, split: str):
        """Load from separate NPY files."""
        self.sequences = np.load(data_path / f"{split}_sequences.npy")
        self.activities = np.load(data_path / f"{split}_activities.npy")
        weights_path = data_path / f"{split}_weights.npy"
        self.weights = np.load(weights_path) if weights_path.exists() else None

    def _load_from_real_data(self):
        """Load data using real data loaders from data_loaders.py.

        IMPORTANT: We disable augmentation here because data gets cached in self.sequences.
        Augmentation (shift, RC) is applied dynamically in __getitem__() instead.
        """
        from training.data_loaders import (
            LentiMPRADataset, DeepSTARRDataset, DREAMYeastDataset, JoresPlantDataset
        )

        dataset_name = self.info.name.lower()
        print(f"Loading real data for {dataset_name} ({self.split})...")

        if dataset_name.startswith("encode4_"):
            cell_type_map = {
                "k562": "K562", "hepg2": "HepG2", "wtc11": "WTC11", "joint": "K562",
            }
            cell_type_lower = dataset_name.replace("encode4_", "").lower()
            cell_type = cell_type_map.get(cell_type_lower, cell_type_lower.upper())
            real_ds = LentiMPRADataset(
                cell_type=cell_type, split=self.split, fold=self.fold or 1,
                target_length=self.target_length, normalize=False,
                use_augmentation=False, use_shift=False,
            )
            self._is_human_mpra = True
        elif dataset_name == "deepstarr":
            real_ds = DeepSTARRDataset(
                split=self.split, target_length=self.target_length,
                normalize=False, use_augmentation=False,
            )
            self._is_human_mpra = False
        elif dataset_name == "dream_yeast":
            subsample = 100000 if self.split == "train" else None
            real_ds = DREAMYeastDataset(
                split=self.split, target_length=self.target_length,
                normalize=False, subsample=subsample, use_augmentation=False,
            )
            self._is_human_mpra = False
        elif dataset_name.startswith("jores_"):
            species = dataset_name.replace("jores_", "")
            real_ds = JoresPlantDataset(
                species=species, split=self.split, target_length=self.target_length,
                normalize=False, use_augmentation=False,
            )
            self._is_human_mpra = False
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Extract sequences and activities from real dataset
        n_samples = len(real_ds)
        print(f"  Loaded {n_samples} samples")

        sample = real_ds[0]
        seq_shape = sample["sequence"].shape
        act_shape = sample["activity"].shape

        self.sequences = np.zeros((n_samples, *seq_shape), dtype=np.float32)
        self.activities = np.zeros((n_samples, *act_shape), dtype=np.float32)

        for i in range(n_samples):
            item = real_ds[i]
            self.sequences[i] = item["sequence"].numpy()
            self.activities[i] = item["activity"].numpy()

        if self.activities.shape[-1] == 1:
            self.activities = self.activities.squeeze(-1)

        self.weights = None
        print(f"  Sequences shape: {self.sequences.shape}, Activities shape: {self.activities.shape}")

    def __len__(self) -> int:
        base_len = len(self.sequences)
        # When double_data_with_rc is enabled, the dataset reports 2x its actual size.
        # Indices [0, base_len) return original sequences; [base_len, 2*base_len)
        # return the reverse complement of the same sequences.
        if self.double_data_with_rc:
            return base_len * 2
        return base_len

    def _apply_shift_augmentation(self, seq: np.ndarray) -> np.ndarray:
        """
        Apply random shift augmentation (±21bp) to sequence.
        Only used for human MPRA datasets to match HumanLegNet training.
        """
        seq_len = seq.shape[1]
        shift = np.random.randint(-self.MAX_SHIFT, self.MAX_SHIFT + 1)

        if shift == 0:
            return seq

        result = np.full((4, seq_len), 0.25, dtype=np.float32)

        if shift > 0:
            result[:, shift:] = seq[:, :seq_len - shift]
        else:
            shift = abs(shift)
            result[:, :seq_len - shift] = seq[:, shift:]

        return result

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_len = len(self.sequences)

        # Handle doubled dataset: first half original, second half RC
        if self.double_data_with_rc and idx >= base_len:
            real_idx = idx - base_len
            use_rc = True
        else:
            real_idx = idx
            use_rc = False

        seq = self.sequences[real_idx].copy()
        original_length = seq.shape[1]

        # Apply shift augmentation FIRST (only for human MPRA during training)
        if self._use_shift and self._is_human_mpra:
            seq = self._apply_shift_augmentation(seq)

        # Apply reverse complement
        if use_rc:
            seq = reverse_complement_onehot(seq)
        elif self.use_augmentation and np.random.random() > 0.5:
            seq = reverse_complement_onehot(seq)

        # Pad/crop to target length
        seq, _ = pad_or_crop_sequence(seq, self.target_length)

        # Get activity and normalize
        activity = self.activities[real_idx]
        if self.normalizer is not None:
            activity = self.normalizer.transform(self.info.name, activity)

        if self.transform is not None:
            seq = self.transform(seq)

        cell_type = self.info.cell_type or "unknown"

        item = {
            "sequence": torch.from_numpy(seq).float(),
            "activity": torch.tensor(activity).float(),
            "original_length": torch.tensor(original_length),
            "dataset_name": self.info.name,
            "species": self.info.species,
            "kingdom": self.info.kingdom,
            "cell_type": cell_type,
            "species_idx": torch.tensor(self.species_to_idx.get(self.info.species, 0)),
            "kingdom_idx": torch.tensor(self.kingdom_to_idx.get(self.info.kingdom, 0)),
            "celltype_idx": torch.tensor(self.celltype_to_idx.get(cell_type, 0)),
        }

        if self.weights is not None:
            item["weight"] = torch.tensor(self.weights[real_idx]).float()

        return item


class MultiDataset(torch.utils.data.Dataset):
    """
    Combined dataset that concatenates multiple SingleDatasets for joint training.

    Provides a single unified index space across all datasets using cumulative
    size tracking. For example, if K562 has 164K samples and HepG2 has 243K:
      - Indices 0-163,999 map to K562
      - Indices 164,000-407,779 map to HepG2

    Also builds consistent species/kingdom/celltype index mappings shared across
    all constituent datasets, so conditioning embeddings use the same indices
    regardless of which dataset a sample comes from.
    """

    def __init__(
        self,
        dataset_names: List[str],
        split: str = "train",
        target_length: int = 256,
        normalizer: Optional[ActivityNormalizer] = None,
        transform: Optional[callable] = None,
    ):
        self.dataset_names = dataset_names
        self.split = split
        self.target_length = target_length
        self.normalizer = normalizer or ActivityNormalizer()

        # Build species/kingdom/celltype mappings
        self._build_mappings()

        # Load all datasets
        self.datasets: Dict[str, SingleDataset] = {}
        self.cumulative_sizes = [0]

        for name in dataset_names:
            if name not in DATASET_CATALOG:
                print(f"Warning: Dataset {name} not in catalog, skipping")
                continue

            info = DATASET_CATALOG[name]
            dataset = SingleDataset(
                dataset_info=info,
                split=split,
                target_length=target_length,
                normalizer=self.normalizer,
                transform=transform,
            )
            self.datasets[name] = dataset
            self.cumulative_sizes.append(
                self.cumulative_sizes[-1] + len(dataset)
            )

        self.total_size = self.cumulative_sizes[-1]

    def _build_mappings(self):
        """Build consistent integer index mappings for conditioning embeddings.

        Collects all unique species, kingdoms, and cell types across all datasets
        and assigns each a deterministic integer index (sorted alphabetically).
        These indices are used to look up the learned embedding vectors in the model.
        An "unknown" cell type is always appended for datasets without a cell_type.
        """
        species_set = set()
        kingdom_set = set()
        celltype_set = set()

        for name in self.dataset_names:
            if name not in DATASET_CATALOG:
                continue
            info = DATASET_CATALOG[name]
            species_set.add(info.species)
            kingdom_set.add(info.kingdom)
            if info.cell_type:
                celltype_set.add(info.cell_type)

        self.species_to_idx = {s: i for i, s in enumerate(sorted(species_set))}
        self.kingdom_to_idx = {k: i for i, k in enumerate(sorted(kingdom_set))}
        self.celltype_to_idx = {c: i for i, c in enumerate(sorted(celltype_set))}
        self.celltype_to_idx["unknown"] = len(self.celltype_to_idx)

    def get_dataset_sizes(self) -> Dict[str, int]:
        """Get size of each dataset."""
        return {name: len(ds) for name, ds in self.datasets.items()}

    def __len__(self) -> int:
        return self.total_size

    def _find_dataset(self, idx: int) -> Tuple[str, int]:
        """Map a global index to (dataset_name, local_index) using cumulative sizes.

        Uses linear scan through cumulative_sizes to find the dataset boundary.
        Returns the dataset name and the index within that dataset.
        """
        for i, (name, dataset) in enumerate(self.datasets.items()):
            if idx < self.cumulative_sizes[i + 1]:
                local_idx = idx - self.cumulative_sizes[i]
                return name, local_idx
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_name, local_idx = self._find_dataset(idx)
        item = self.datasets[dataset_name][local_idx]

        item["species_idx"] = torch.tensor(self.species_to_idx[item["species"]])
        item["kingdom_idx"] = torch.tensor(self.kingdom_to_idx[item["kingdom"]])
        item["celltype_idx"] = torch.tensor(self.celltype_to_idx[item["cell_type"]])

        return item


def collate_multi_dataset(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for multi-dataset batches.

    The key challenge: different datasets have different numbers of outputs.
    For example, ENCODE4 has 1 output (activity), DeepSTARR has 2 (Dev, Hk),
    and Jores plants have 2 (leaf, proto). When these appear in the same batch,
    we need a uniform tensor shape.

    Solution: Pad the activity tensor to max_outputs columns, filling unused
    slots with NaN. The loss function (_compute_masked_loss) then uses NaN
    masking to ignore these padding values, ensuring only valid targets contribute.

    Example batch with K562 (1 output) and DeepSTARR (2 outputs):
      activities = [[0.5, NaN],   # K562 sample: only column 0 is valid
                    [1.2, 0.8]]   # DeepSTARR sample: both columns valid
    """
    sequences = torch.stack([item["sequence"] for item in batch])
    original_lengths = torch.stack([item["original_length"] for item in batch])
    species_idx = torch.stack([item["species_idx"] for item in batch])
    kingdom_idx = torch.stack([item["kingdom_idx"] for item in batch])
    celltype_idx = torch.stack([item["celltype_idx"] for item in batch])

    # Keep dataset names as a list of strings (not tensorizable)
    dataset_names = [item["dataset_name"] for item in batch]

    # Pad activities to the maximum number of outputs in this batch
    max_outputs = max(item["activity"].numel() for item in batch)
    activities = torch.full((len(batch), max_outputs), float('nan'))  # NaN = "no target here"
    for i, item in enumerate(batch):
        act = item["activity"]
        if act.dim() == 0:
            activities[i, 0] = act       # Scalar activity -> first column
        else:
            activities[i, :len(act)] = act  # Multi-output -> fill first N columns

    if "weight" in batch[0]:
        weights = torch.stack([item.get("weight", torch.tensor(1.0)) for item in batch])
    else:
        weights = None

    result = {
        "sequence": sequences,
        "activity": activities,
        "original_length": original_lengths,
        "species_idx": species_idx,
        "kingdom_idx": kingdom_idx,
        "celltype_idx": celltype_idx,
        "dataset_names": dataset_names,
    }

    if weights is not None:
        result["weight"] = weights

    return result


# =============================================================================
# PART 7: Balanced Sampling Strategies
# Source: training/samplers.py
# =============================================================================

from torch.utils.data import Sampler
import math


class TemperatureBalancedSampler(Sampler):
    """
    Sample datasets with temperature-scaled probability: p_i proportional to n_i^tau.

    This addresses the dataset size imbalance problem in multi-dataset training.
    Without balancing, large datasets (e.g., DREAM Yeast with 6.7M samples) would
    dominate training while small datasets (e.g., Arabidopsis with 12K) are rarely seen.

    Temperature scaling formula:
        p_i = n_i^tau / sum_j(n_j^tau)

    where n_i is the size of dataset i and tau (temperature) controls the balance:
        tau = 1.0: p_i proportional to n_i (natural distribution, large datasets dominate)
        tau = 0.5: p_i proportional to sqrt(n_i) (balanced middle ground, recommended)
        tau = 0.0: p_i = 1/K for K datasets (uniform, each dataset equally likely)

    Example with tau=0.5, sizes [164K, 484K, 12K]:
        probs proportional to [405, 696, 110] -> [0.33, 0.57, 0.09]
    vs tau=1.0:
        probs proportional to [164K, 484K, 12K] -> [0.25, 0.73, 0.02]
    The small dataset goes from 2% to 9% representation.

    Yields (dataset_name, local_index) tuples for use with MultiDataset.
    """

    def __init__(
        self,
        dataset_sizes: Dict[str, int],
        temperature: float = 0.5,
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
    ):
        self.dataset_sizes = dataset_sizes
        self.temperature = temperature
        self.seed = seed
        self.epoch = 0

        sizes = np.array(list(dataset_sizes.values()))
        self.dataset_names = list(dataset_sizes.keys())

        # Temperature scaling: p_i = n_i^tau / sum(n_j^tau)
        weights = sizes ** temperature
        self.probs = weights / weights.sum()

        if samples_per_epoch is None:
            # Default: geometric mean of dataset sizes -- a compromise between
            # the smallest and largest datasets
            self.samples_per_epoch = int(np.exp(np.mean(np.log(sizes))))
        else:
            self.samples_per_epoch = samples_per_epoch

        # Compute cumulative offsets for converting (dataset, local_idx) -> global_idx
        self.cumsum = np.cumsum([0] + list(dataset_sizes.values()))
        self.dataset_to_offset = {
            name: self.cumsum[i]
            for i, name in enumerate(self.dataset_names)
        }

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        for _ in range(self.samples_per_epoch):
            dataset_idx = rng.choice(len(self.dataset_names), p=self.probs)
            dataset_name = self.dataset_names[dataset_idx]
            sample_idx = rng.randint(self.dataset_sizes[dataset_name])
            yield (dataset_name, sample_idx)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def get_effective_samples_per_dataset(self) -> Dict[str, int]:
        """Get expected samples per dataset per epoch."""
        return {
            name: int(self.samples_per_epoch * prob)
            for name, prob in zip(self.dataset_names, self.probs)
        }


class GlobalIndexSampler(Sampler):
    """
    Wrapper that converts TemperatureBalancedSampler's (dataset_name, local_idx) tuples
    into global integer indices compatible with MultiDataset.__getitem__().

    This is needed because PyTorch's DataLoader expects integer indices from a Sampler,
    but TemperatureBalancedSampler yields (name, idx) pairs. GlobalIndexSampler adds
    the dataset's cumulative offset to convert local indices to global ones.
    """

    def __init__(
        self,
        dataset_sizes: Dict[str, int],
        temperature: float = 0.5,
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
    ):
        self.base_sampler = TemperatureBalancedSampler(
            dataset_sizes, temperature, samples_per_epoch, seed
        )

        self.offsets = {}
        current_offset = 0
        for name, size in dataset_sizes.items():
            self.offsets[name] = current_offset
            current_offset += size

    def set_epoch(self, epoch: int):
        self.base_sampler.set_epoch(epoch)

    def __iter__(self):
        for dataset_name, local_idx in self.base_sampler:
            yield self.offsets[dataset_name] + local_idx

    def __len__(self) -> int:
        return len(self.base_sampler)


class BalancedActivitySampler(Sampler):
    """
    Sampler that equalizes representation across activity value ranges.

    Problem: Activity distributions are typically bell-shaped -- most sequences have
    moderate activity, while very high or very low activity sequences are rare.
    Without correction, the model trains mostly on the middle and struggles at extremes.

    Solution: Divide the activity range into N equal-width VALUE bins (not quantile bins),
    then sample from each bin with probability proportional to sqrt(bin_size).

    Why equal-WIDTH bins (not quantiles)?
      Equal-width bins based on the VALUE range mean that extreme activity bins
      naturally contain fewer samples. This is intentional -- those are the rare
      but important samples we want to oversample.

    Why sqrt scaling (not uniform)?
      Uniform bin sampling would massively oversample rare extremes (20-30x),
      causing the model to overfit to a handful of extreme sequences.
      Sqrt scaling is a middle ground:
        - A bin with 100 samples gets weight sqrt(100) = 10
        - A bin with 10000 samples gets weight sqrt(10000) = 100
        - Ratio: 10:1 instead of 1:100 (proportional) or 1:1 (uniform)
      Result: extremes get ~3-5x more representation, not 20-30x.

    Also applies temperature-balanced dataset sampling (same as TemperatureBalancedSampler)
    to handle multi-dataset size imbalance on top of the activity balancing.
    """

    def __init__(
        self,
        activities: np.ndarray,
        dataset_sizes: Dict[str, int],
        n_bins: int = 10,
        temperature: float = 0.5,
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
    ):
        self.n_bins = n_bins
        self.seed = seed
        self.epoch = 0

        self.dataset_sizes = dataset_sizes
        self.dataset_names = list(dataset_sizes.keys())

        # Compute offsets for global indexing
        self.offsets = {}
        current_offset = 0
        for name, size in dataset_sizes.items():
            self.offsets[name] = current_offset
            current_offset += size
        self.total_size = current_offset

        # Bin samples by activity within each dataset
        self._create_bins(activities)

        if samples_per_epoch is None:
            self.samples_per_epoch = self.total_size
        else:
            self.samples_per_epoch = samples_per_epoch

        # Temperature-based dataset probabilities
        sizes = np.array(list(dataset_sizes.values()))
        weights = sizes ** temperature
        self.dataset_probs = weights / weights.sum()

    def _create_bins(self, activities: np.ndarray):
        """Create activity bins using equal-width VALUE ranges.

        Extremes (fewer samples) get repeated more for balanced coverage.
        """
        if activities.ndim > 1:
            activities = np.nanmean(activities, axis=-1)

        self.bins_per_dataset = {}
        self.bin_edges_per_dataset = {}

        offset = 0
        for name, size in self.dataset_sizes.items():
            ds_activities = activities[offset:offset + size]
            valid_activities = ds_activities[~np.isnan(ds_activities)]

            # Equal-width bins based on VALUE RANGE (not quantiles)
            min_val, max_val = valid_activities.min(), valid_activities.max()
            bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf

            bin_indices = np.digitize(ds_activities, bin_edges[1:-1])

            bins = [[] for _ in range(self.n_bins)]
            for i, bin_idx in enumerate(bin_indices):
                bins[min(bin_idx, self.n_bins - 1)].append(i)

            self.bins_per_dataset[name] = [
                np.array(b) if len(b) > 0 else np.array([0]) for b in bins
            ]
            self.bin_edges_per_dataset[name] = bin_edges

            offset += size

        # Log bin statistics
        for name in self.dataset_names:
            bin_sizes = [len(b) for b in self.bins_per_dataset[name]]
            total = sum(bin_sizes)
            print(f"  {name} bin sizes: {bin_sizes}")
            print(f"  {name} bin %: {[f'{100*s/total:.1f}%' for s in bin_sizes]}")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _precompute_indices(self, rng: np.random.RandomState) -> np.ndarray:
        """Precompute all indices for the epoch with sqrt-balanced activity sampling.

        Two-level sampling:
        1. Dataset level: Use temperature-scaled multinomial to allocate samples across datasets
        2. Bin level: Within each dataset, allocate samples across activity bins using
           sqrt(bin_size) weighting, then sample uniformly within each bin (with replacement)
        """
        # Level 1: Allocate total samples across datasets via temperature-scaled probabilities
        dataset_counts = rng.multinomial(self.samples_per_epoch, self.dataset_probs)

        all_indices = []

        for ds_name, ds_count in zip(self.dataset_names, dataset_counts):
            if ds_count == 0:
                continue

            bins = self.bins_per_dataset[ds_name]

            # Level 2: Sqrt scaling for bin probabilities
            # bin_prob_i = sqrt(|bin_i|) / sum_j(sqrt(|bin_j|))
            bin_sizes = np.array([max(len(b), 1) for b in bins])
            bin_weights = np.sqrt(bin_sizes)
            bin_probs = bin_weights / bin_weights.sum()

            bin_sample_counts = rng.multinomial(ds_count, bin_probs)

            ds_indices = []
            for bin_idx, (bin_samples, n_samples) in enumerate(
                zip(bins, bin_sample_counts)
            ):
                if len(bin_samples) == 0 or n_samples == 0:
                    continue
                sampled = rng.choice(bin_samples, size=n_samples, replace=True)
                ds_indices.append(sampled)

            if ds_indices:
                ds_indices = np.concatenate(ds_indices)
                global_indices = self.offsets[ds_name] + ds_indices
                all_indices.append(global_indices)

        all_indices = np.concatenate(all_indices)
        rng.shuffle(all_indices)
        return all_indices

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = self._precompute_indices(rng)
        for idx in indices:
            yield int(idx)

    def __len__(self) -> int:
        return self.samples_per_epoch


class ExtremeAwareSampler(Sampler):
    """
    Sampler that oversamples extreme values (tails of the activity distribution).

    This is an alternative to BalancedActivitySampler that uses continuous
    z-score-based weights rather than discrete bins.

    Weight formula: w_i = 1 + alpha * |z_i|^beta
    where z_i = (activity_i - mean) / std is the z-score of sample i within its dataset.

    Default parameters (alpha=1.0, beta=2.0) give quadratic growth:
        z=0 (mean activity):    w = 1.0   (baseline, no boost)
        z=1 (1 std from mean):  w = 2.0   (2x more likely to be sampled)
        z=2 (2 std from mean):  w = 5.0   (5x more likely)
        z=3 (3 std from mean):  w = 10.0  (10x more likely)
        z=4 (clipped max):      w = 17.0  (17x more likely)

    Z-scores are clipped to [-4, 4] to prevent extreme outliers from receiving
    disproportionate weight. Weights are computed per-dataset to account for
    different activity scales.

    Compared to BalancedActivitySampler:
    - More continuous (no binning artifacts)
    - Directly controllable via alpha/beta parameters
    - Computationally simpler (no bin creation)
    """

    def __init__(
        self,
        activities: np.ndarray,
        dataset_sizes: Dict[str, int],
        extreme_alpha: float = 1.0,    # Scaling factor for extreme weight boost
        extreme_beta: float = 2.0,     # Exponent: 1.0=linear, 2.0=quadratic growth with z-score
        temperature: float = 0.5,
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
    ):
        self.extreme_alpha = extreme_alpha
        self.extreme_beta = extreme_beta
        self.seed = seed
        self.epoch = 0

        self.dataset_sizes = dataset_sizes
        self.dataset_names = list(dataset_sizes.keys())

        self.offsets = {}
        current_offset = 0
        for name, size in dataset_sizes.items():
            self.offsets[name] = current_offset
            current_offset += size
        self.total_size = current_offset

        self._compute_weights(activities)

        if samples_per_epoch is None:
            self.samples_per_epoch = self.total_size
        else:
            self.samples_per_epoch = samples_per_epoch

        sizes = np.array(list(dataset_sizes.values()))
        weights = sizes ** temperature
        self.dataset_probs = weights / weights.sum()

    def _compute_weights(self, activities: np.ndarray):
        """Compute per-sample weights that emphasize extreme activity values.

        For each dataset independently:
        1. Compute z-scores: z_i = |activity_i - mean| / std
        2. Clip to [0, 4] to prevent outlier domination
        3. Apply weight formula: w_i = 1 + alpha * |z_i|^beta
        4. Normalize to probability distribution within each dataset

        The per-dataset normalization ensures that each dataset's internal
        extreme-vs-middle balance is adjusted independently, regardless of
        cross-dataset scale differences.
        """
        # For multi-output datasets, average across outputs for a single activity score
        if activities.ndim > 1:
            activities = np.nanmean(activities, axis=-1)

        weights = np.ones(len(activities), dtype=np.float32)

        offset = 0
        for name, size in self.dataset_sizes.items():
            ds_activities = activities[offset:offset + size]

            # Compute z-scores within this dataset
            mean = np.nanmean(ds_activities)
            std = np.nanstd(ds_activities) + 1e-8  # epsilon prevents division by zero
            z_scores = np.abs((ds_activities - mean) / std)
            z_scores = np.clip(z_scores, 0, 4.0)  # Cap at 4 std to limit max weight to ~17x

            # Apply weight formula: w_i = 1 + alpha * |z_i|^beta
            ds_weights = 1.0 + self.extreme_alpha * (z_scores ** self.extreme_beta)
            ds_weights = np.nan_to_num(ds_weights, nan=1.0)  # NaN activities get baseline weight

            weights[offset:offset + size] = ds_weights
            offset += size

        self.sample_weights = weights

        # Convert absolute weights to per-dataset sampling probabilities
        self.per_dataset_probs = {}
        offset = 0
        for name, size in self.dataset_sizes.items():
            ds_weights = self.sample_weights[offset:offset + size]
            ds_probs = ds_weights / ds_weights.sum()  # Normalize to probability distribution
            self.per_dataset_probs[name] = ds_probs
            offset += size

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _precompute_indices(self, rng: np.random.RandomState) -> np.ndarray:
        """Precompute all indices for the epoch."""
        dataset_counts = rng.multinomial(self.samples_per_epoch, self.dataset_probs)

        all_indices = []
        for i, (name, count) in enumerate(zip(self.dataset_names, dataset_counts)):
            if count == 0:
                continue
            probs = self.per_dataset_probs[name]
            local_indices = rng.choice(len(probs), size=count, p=probs, replace=True)
            global_indices = self.offsets[name] + local_indices
            all_indices.append(global_indices)

        all_indices = np.concatenate(all_indices)
        rng.shuffle(all_indices)
        return all_indices

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = self._precompute_indices(rng)
        for idx in indices:
            yield int(idx)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def get_weight_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about sampling weights per dataset."""
        stats = {}
        offset = 0
        for name, size in self.dataset_sizes.items():
            ds_weights = self.sample_weights[offset:offset + size]
            stats[name] = {
                "min": float(ds_weights.min()),
                "max": float(ds_weights.max()),
                "mean": float(ds_weights.mean()),
                "std": float(ds_weights.std()),
            }
            offset += size
        return stats


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for FUSEMAP training coordinator."""
    parser = argparse.ArgumentParser(
        description="FUSEMAP Training Coordinator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, required=True,
        choices=["single_celltype", "multi_celltype_human", "cross_animal",
                 "cross_kingdom", "universal"],
        help="Configuration type to run",
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (required for single_celltype)")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Get configuration
    if args.config == "single_celltype":
        if args.dataset is None:
            raise ValueError("--dataset required for single_celltype config")
        config = get_config1_single_celltype(args.dataset)
    elif args.config == "multi_celltype_human":
        config = get_config2_multi_celltype_human()
    elif args.config == "cross_animal":
        config = get_config3_cross_animal()
    elif args.config == "cross_kingdom":
        config = get_config4_cross_kingdom()
    elif args.config == "universal":
        config = get_config5_universal()

    config.output_dir = args.output_dir
    config.seed = args.seed

    # Print summary
    print_config_summary(config)

    # Set seed
    set_seed(config.seed)

    # Create and run trainer
    trainer = Trainer(config=config, device=args.device)

    try:
        results = trainer.train()
        print("\nTraining completed successfully!")
        return 0
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
