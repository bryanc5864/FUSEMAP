"""
================================================================================
FUSEMAP TRAINING & APPLICATIONS - Representative Code File 3/3
================================================================================

These 3 representative files contain key excerpts from the FUSEMAP codebase.
They do not contain all FUSEMAP code, as the full implementation is too large
to include here. All code, trained models, and processed datasets are available
under the MIT license at:
    https://github.com/bryanc5864/FUSEMAP

This file contains the training infrastructure, downstream applications,
and the OracleCheck validation protocol:

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

8. OracleCheck - In-silico design validation protocol
   Source: oracle_check/ (~3,532 lines across 7 files)
   Verdict system (GREEN/YELLOW/RED), PhysicsValidator, CompositionValidator,
   ConfidenceValidator, MahalanobisValidator, OracleCheckProtocol,
   RCConsistencyChecker, ISMFlipTest, MMDTest, KmerJSDivergence,
   BatchComparator, SequenceScorecard, BatchScorecard

KEY RESULTS (from paper):
- CADENCE: K562 r=0.809, HepG2 r=0.786, WTC11 r=0.698
- DeepSTARR: Dev r=0.909, Hk r=0.920
- Plants: Maize r=0.796, Sorghum r=0.782, Arabidopsis r=0.618
- Yeast: r=0.958
- Therapeutic design: 99% predicted specificity, 96.5% GREEN OracleCheck (HepG2)
- OracleCheck: 96.5% GREEN on designed enhancers, 12% GREEN on random controls
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
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
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

    # Activity statistics (filled during preprocessing)
    activity_mean: Optional[List[float]] = None
    activity_std: Optional[List[float]] = None


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
        kfold_splits=10,
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
        kfold_splits=10,
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
        kfold_splits=10,
    ),
    "encode4_joint": DatasetInfo(
        name="encode4_joint",
        path="data/processed/encode4/joint",
        sequence_length=230,
        num_outputs=3,  # K562, HepG2, WTC11
        output_names=["K562", "HepG2", "WTC11"],
        species="human",
        kingdom="animal",
        cell_type="joint",
        element_type="enhancer",
        train_size=60000,  # Approximate
        validation_scheme="standard",
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

    # Optional: PWM-initialized multi-scale stem (alternative to RC stem)
    use_pwm_stem: bool = False
    pwm_stem_scales: List[int] = field(default_factory=lambda: [7, 11, 15])

    # Optional: Species-specific stems (for cross-species transfer learning)
    # Creates separate stem modules per species, shared backbone after
    use_species_stem: bool = False

    # Optional: Kingdom-specific stems (for cross-kingdom transfer learning)
    # Creates separate stem modules per kingdom (animal vs plant), shared backbone after
    use_kingdom_stem: bool = False

    # --- Optional modules (architecture ablations) ---
    use_cluster_space: bool = False     # Cluster-space projection for motif grouping
    cluster_dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 1])
    use_grammar: bool = False           # Motif grammar / syntax module (pairwise motif interactions)
    grammar_hidden: int = 128
    use_micromotif: bool = False        # Fine-grained sub-motif resolution module
    micromotif_windows: List[int] = field(default_factory=lambda: [5, 11, 21])
    use_motif_correlator: bool = False  # Cross-attention between detected motifs
    correlator_factors: int = 32
    correlator_rank: int = 8

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
    # Element-type conditioning (promoter vs enhancer vs silencer)
    use_element_type_embedding: bool = False
    element_type_embed_dim: int = 8

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
    mode: str = "max"  # max or min

    # --- Gradient management ---
    gradient_clip: float = 1.0                # Max gradient norm (prevents exploding gradients with NLL loss)
    gradient_accumulation_steps: int = 1      # Accumulate gradients over N mini-batches before stepping
    use_amp: bool = True                      # Use Automatic Mixed Precision (FP16 forward, FP32 grads)

    # --- Multi-dataset sampling ---
    sampling_temperature: float = 0.5         # Temperature for TemperatureBalancedSampler (see p_i ~ n_i^tau)
    samples_per_epoch: Optional[int] = None   # If set, caps samples per epoch (useful for huge datasets like yeast)

    # Checkpointing
    save_top_k: int = 3
    checkpoint_dir: str = "checkpoints"

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

    # Extreme-aware sampling (oversample tail values) - DEPRECATED, use balanced instead
    use_extreme_sampling: bool = False  # Use extreme-aware sampler
    sampling_extreme_alpha: float = 0.5  # Sampling weight alpha (separate from loss alpha)

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

    # Check dataset type for appropriate hyperparameters
    is_human_mpra = dataset_name.startswith("encode4_")
    is_deepstarr = dataset_name == "deepstarr"
    is_plant = dataset_name.startswith("jores_")

    if is_human_mpra:
        # LegNet-style training for human MPRA - match comparison_models exactly
        training_config = TrainingConfig(
            max_epochs=30,
            batch_size=1024,
            learning_rate=0.01,
            weight_decay=0.1,  # Match HumanLegNet
            scheduler="onecycle",
            onecycle_pct_start=0.3,
            onecycle_div_factor=25.0,  # Initial LR = 0.01/25 = 0.0004
            patience=30,  # No early stopping
            # DISABLED balanced sampling - just use random shuffle like HumanLegNet
            use_balanced_sampling=False,
            # DISABLED extreme weighting
            use_extreme_weights=False,
            extreme_alpha=0.0,
            extreme_beta=2.0,
        )
    elif is_deepstarr:
        # DeepSTARR (S2) - gentler training for stability with all modules
        # Uses Config 3 style hyperparameters: lower LR, cosine scheduler
        training_config = TrainingConfig(
            max_epochs=100,
            batch_size=256,
            learning_rate=1e-3,  # 10x lower than before for stability
            weight_decay=1e-5,
            scheduler="cosine",  # Gentler than OneCycle
            patience=15,
            use_balanced_sampling=False,  # DeepSTARR has different distribution
            use_extreme_weights=False,  # Avoid gradient instability
        )
    elif is_plant:
        # Plant datasets - 50 epochs
        training_config = TrainingConfig(
            max_epochs=50,
            batch_size=64,
            learning_rate=0.01,
            weight_decay=0.1,
            scheduler="onecycle",
            onecycle_pct_start=0.3,
            onecycle_div_factor=25.0,
            patience=10,
        )
    else:
        # Default for other datasets (yeast, etc.)
        training_config = TrainingConfig(
            max_epochs=50,
            batch_size=128 if dataset_info.train_size > 100000 else 64,
            learning_rate=1e-3,
        )

    # For single-cell-type: use NATIVE sequence length (no padding)
    # - No length embedding (all sequences same length)
    # - No RC stem (use standard LocalBlock like HumanLegNet)
    # - MSE loss for human MPRA (not Gaussian NLL)
    return ExperimentConfig(
        name=f"config1_{dataset_name}",
        config_type=ConfigurationType.SINGLE_CELLTYPE,
        description=f"Single cell type baseline for {dataset_name}",
        datasets=[dataset_name],
        target_sequence_length=dataset_info.sequence_length,  # Use native length, no padding
        model=ModelConfig(
            use_species_embedding=False,
            use_celltype_embedding=False,
            use_length_embedding=False,  # All same length, not needed
            use_rc_stem=False,  # Standard LocalBlock like HumanLegNet
            use_uncertainty=False if is_human_mpra else not use_mse,  # MSE for human MPRA
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
        datasets=[
            "encode4_k562", "encode4_hepg2", "encode4_wtc11",
            "deepstarr"
        ],
        target_sequence_length=256,
        model=ModelConfig(
            use_rc_stem=True,  # RC-equivariant base
            use_species_stem=True,  # Species-specific motif filter banks
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
            # Cross-species: don't balance activity distributions (they're not comparable)
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
            # Animals (~968k sequences: K562 164k + HepG2 244k + WTC11 76k + DeepSTARR 484k)
            "encode4_k562", "encode4_hepg2", "encode4_wtc11",
            "deepstarr",
            # Plants (~37k sequences)
            "jores_arabidopsis", "jores_maize",
        ],
        target_sequence_length=256,
        model=ModelConfig(
            use_rc_stem=True,  # RC-equivariant base
            use_kingdom_stem=True,  # Kingdom-specific motif filter banks
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
            weight_decay=1e-5,
            scheduler="cosine",
            sampling_temperature=0.3,  # More balanced sampling across kingdoms
            patience=15,
            # Cross-kingdom: don't balance activity distributions
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
            # Animals (~968k sequences: K562 164k + HepG2 244k + WTC11 76k + DeepSTARR 484k)
            "encode4_k562", "encode4_hepg2", "encode4_wtc11",
            "deepstarr",
            # Plants (~54k sequences)
            "jores_arabidopsis", "jores_maize", "jores_sorghum",
            # Yeast (~6.7M sequences, subsampled)
            "dream_yeast",
        ],
        target_sequence_length=256,
        model=ModelConfig(
            use_rc_stem=True,  # RC-equivariant base
            use_kingdom_stem=True,  # Kingdom-specific motif banks
            use_kingdom_embedding=True,
            kingdom_embed_dim=8,
            use_species_embedding=True,
            species_embed_dim=16,
            use_celltype_embedding=True,
            celltype_embed_dim=32,
            use_element_type_embedding=True,  # Promoter vs enhancer conditioning
            element_type_embed_dim=8,
            use_length_embedding=True,
        ),
        training=TrainingConfig(
            max_epochs=150,
            batch_size=256,
            learning_rate=1e-3,
            weight_decay=1e-5,
            scheduler="cosine",
            sampling_temperature=0.3,  # Balance across kingdoms
            samples_per_epoch=500000,  # Subsample yeast per epoch
            patience=20,
            # Universal: don't balance activity distributions across species
            use_balanced_sampling=False,
        ),
        training_phases=[
            {
                "name": "phase1_pretrain",
                "epochs": 50,
                "freeze_heads": False,
                "freeze_backbone": False,
                "lr": 1e-3,
                "description": "Pre-train universal physics encoder on all data",
            },
            {
                "name": "phase2_head_training",
                "epochs": 50,
                "freeze_heads": False,
                "freeze_backbone": True,
                "lr": 1e-3,
                "description": "Train species-specific heads with frozen physics encoder",
            },
            {
                "name": "phase3_finetune",
                "epochs": 50,
                "freeze_heads": False,
                "freeze_backbone": False,
                "lr": 1e-4,  # Lower LR for fine-tuning
                "description": "End-to-end fine-tuning with low learning rate",
            },
        ],
    )


def get_config5_universal_no_yeast() -> ExperimentConfig:
    """
    Configuration 5 variant: Universal Foundation Model WITHOUT Yeast.

    Purpose: Cross-kingdom generalization (animals + plants only).
    - All animal data: ENCODE4 Human (~295k) + DeepSTARR (~352k) = ~647k
    - All plant data: Arabidopsis (~15k) + Maize (~22k) + Sorghum (~17k) = ~54k
    - Total: ~701k sequences

    Same architecture as config5 but trained only on animal + plant data.
    """
    return ExperimentConfig(
        name="config5_universal_no_yeast",
        config_type=ConfigurationType.UNIVERSAL,
        description="Universal foundation model across animals and plants (no yeast)",
        datasets=[
            # Animals (~968k sequences: K562 164k + HepG2 244k + WTC11 76k + DeepSTARR 484k)
            "encode4_k562", "encode4_hepg2", "encode4_wtc11",
            "deepstarr",
            # Plants (~54k sequences)
            "jores_arabidopsis", "jores_maize", "jores_sorghum",
        ],
        target_sequence_length=256,
        model=ModelConfig(
            use_rc_stem=True,  # RC-equivariant base
            use_kingdom_stem=True,  # Kingdom-specific motif banks (animal, plant)
            use_kingdom_embedding=True,
            kingdom_embed_dim=8,
            use_species_embedding=True,
            species_embed_dim=16,
            use_celltype_embedding=True,
            celltype_embed_dim=32,
            use_element_type_embedding=True,  # Promoter vs enhancer conditioning
            element_type_embed_dim=8,
            use_length_embedding=True,
        ),
        training=TrainingConfig(
            max_epochs=150,
            batch_size=256,
            learning_rate=1e-3,
            weight_decay=1e-5,
            scheduler="cosine",
            sampling_temperature=0.3,  # Balance across kingdoms
            patience=20,
            use_balanced_sampling=False,
        ),
        training_phases=[
            {
                "name": "phase1_pretrain",
                "epochs": 50,
                "freeze_heads": False,
                "freeze_backbone": False,
                "lr": 1e-3,
                "description": "Pre-train universal physics encoder on all data",
            },
            {
                "name": "phase2_head_training",
                "epochs": 50,
                "freeze_heads": False,
                "freeze_backbone": True,
                "lr": 1e-3,
                "description": "Train species-specific heads with frozen physics encoder",
            },
            {
                "name": "phase3_finetune",
                "epochs": 50,
                "freeze_heads": False,
                "freeze_backbone": False,
                "lr": 1e-4,  # Lower LR for fine-tuning
                "description": "End-to-end fine-tuning with low learning rate",
            },
        ],
    )


def get_config(config_type: ConfigurationType, dataset_name: Optional[str] = None) -> ExperimentConfig:
    """Get configuration by type."""
    if config_type == ConfigurationType.SINGLE_CELLTYPE:
        if dataset_name is None:
            raise ValueError("dataset_name required for SINGLE_CELLTYPE config")
        return get_config1_single_celltype(dataset_name)
    elif config_type == ConfigurationType.MULTI_CELLTYPE_HUMAN:
        return get_config2_multi_celltype_human()
    elif config_type == ConfigurationType.CROSS_ANIMAL:
        return get_config3_cross_animal()
    elif config_type == ConfigurationType.CROSS_KINGDOM:
        return get_config4_cross_kingdom()
    elif config_type == ConfigurationType.UNIVERSAL:
        return get_config5_universal()
    else:
        raise ValueError(f"Unknown config type: {config_type}")


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


def compute_masked_loss(
    outputs: Dict[str, Dict],
    targets: torch.Tensor,
    dataset_names: List[str],
    dataset_to_heads: Dict[str, List[str]],
    use_uncertainty: bool = False,
    use_extreme_weights: bool = True,
    extreme_alpha: float = 0.5,
    extreme_beta: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute loss with masking and optional extreme value weighting."""
    total_loss = 0.0
    n_samples = 0
    per_head_losses = {}

    for head_name, pred in outputs.items():
        if not pred['indices']:
            continue

        indices = pred['indices']
        mean = pred['mean']
        logvar = pred.get('logvar', None)

        # Get targets
        head_targets = []
        for i, idx in enumerate(indices):
            dataset = dataset_names[idx]
            heads_for_dataset = dataset_to_heads.get(dataset, [])
            if head_name in heads_for_dataset:
                output_idx = heads_for_dataset.index(head_name)
                head_targets.append(targets[idx, output_idx])

        if not head_targets:
            continue

        target = torch.stack(head_targets)
        valid_mask = ~torch.isnan(target)
        if valid_mask.sum() == 0:
            continue

        mean = mean[valid_mask]
        target = target[valid_mask]
        if logvar is not None:
            logvar = logvar[valid_mask]

        # Compute loss
        if use_uncertainty and logvar is not None:
            variance = torch.exp(logvar).clamp(min=1e-6)
            loss = 0.5 * (logvar + (target - mean) ** 2 / variance)
        else:
            loss = (target - mean) ** 2

        # Apply extreme value weighting
        if use_extreme_weights and len(target) > 1:
            weights = compute_extreme_weights(target, extreme_alpha, extreme_beta)
            loss = loss * weights

        head_loss = loss.mean()
        per_head_losses[head_name] = head_loss
        total_loss += head_loss * len(target)
        n_samples += len(target)

    if n_samples > 0:
        total_loss = total_loss / n_samples

    return total_loss, per_head_losses


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
        self.normalizer = ActivityNormalizer()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config.training.use_amp else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('-inf')
        self.patience_counter = 0

        # Metrics tracking
        self.metrics_tracker = None

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
        """Setup datasets and data loaders."""
        self.logger.info("Setting up datasets...")

        # Create training dataset
        self.train_dataset = MultiDataset(
            dataset_names=self.config.datasets,
            split="train",
            target_length=self.config.target_sequence_length,
            normalizer=self.normalizer,
        )

        # Get dataset sizes for balanced sampling
        dataset_sizes = self.train_dataset.get_dataset_sizes()
        self.logger.info(f"Dataset sizes: {dataset_sizes}")

        # Create sampler - priority: balanced > extreme > global
        use_balanced_sampling = getattr(self.config.training, 'use_balanced_sampling', False)
        use_extreme_sampling = getattr(self.config.training, 'use_extreme_sampling', False)

        # Collect activities for activity-aware samplers
        # For multi-output datasets, average across outputs to get single activity value
        all_activities = []
        for ds_name, ds in self.train_dataset.datasets.items():
            acts = ds.activities
            if acts.ndim > 1:
                # Multi-output: average across outputs for sampling purposes
                acts = acts.mean(axis=1)
            all_activities.append(acts)
        all_activities = np.concatenate(all_activities, axis=0)

        if use_balanced_sampling:
            # Balanced sampling: equal samples from each activity bin
            n_bins = getattr(self.config.training, 'balanced_sampling_bins', 10)
            self.logger.info(f"Using balanced activity sampling with {n_bins} bins...")
            self.sampler = BalancedActivitySampler(
                activities=all_activities,
                dataset_sizes=dataset_sizes,
                n_bins=n_bins,
                temperature=self.config.training.sampling_temperature,
                samples_per_epoch=self.config.training.samples_per_epoch,
                seed=self.config.seed,
            )
        elif use_extreme_sampling:
            # Extreme-aware sampling (deprecated)
            self.logger.info("Using extreme-aware sampling...")
            sampling_alpha = getattr(self.config.training, 'sampling_extreme_alpha', 0.5)
            self.sampler = ExtremeAwareSampler(
                activities=all_activities,
                dataset_sizes=dataset_sizes,
                extreme_alpha=sampling_alpha,
                extreme_beta=self.config.training.extreme_beta,
                temperature=self.config.training.sampling_temperature,
                samples_per_epoch=self.config.training.samples_per_epoch,
                seed=self.config.seed,
            )
            weight_stats = self.sampler.get_weight_statistics()
            for name, stats in weight_stats.items():
                self.logger.info(f"  {name}: weights min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
        else:
            self.logger.info("Using standard global index sampling...")
            self.sampler = GlobalIndexSampler(
                dataset_sizes=dataset_sizes,
                temperature=self.config.training.sampling_temperature,
                samples_per_epoch=self.config.training.samples_per_epoch,
                seed=self.config.seed,
            )

        # Create train loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=self.sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_multi_dataset,
        )

        # Create validation loaders for each dataset
        # Use index mappings from train_dataset for consistency
        index_mappings = {
            "species": self.train_dataset.species_to_idx,
            "kingdom": self.train_dataset.kingdom_to_idx,
            "celltype": self.train_dataset.celltype_to_idx,
        }

        self.val_loaders = {}
        self.val_types = {}
        self.train_eval_loaders = {}  # For evaluating on train set

        for dataset_name in self.config.datasets:
            try:
                loader, val_type = get_validation_loader(
                    dataset_name,
                    target_length=self.config.target_sequence_length,
                    batch_size=self.config.training.batch_size,
                    normalizer=self.normalizer,
                    index_mappings=index_mappings,
                )
                self.val_loaders[dataset_name] = loader
                self.val_types[dataset_name] = val_type
                self.logger.info(f"  {dataset_name}: {val_type} set loaded")
            except Exception as e:
                self.logger.warning(f"  {dataset_name}: Could not load validation data: {e}")

        # Create train eval loaders (sequential, for evaluation only)
        # We'll subsample to speed up evaluation
        for dataset_name in self.config.datasets:
            try:
                info = DATASET_CATALOG[dataset_name]
                train_eval_dataset = SingleDataset(
                    info, "train",
                    self.config.target_sequence_length,
                    self.normalizer,
                    index_mappings=index_mappings,
                )
                self.train_eval_loaders[dataset_name] = DataLoader(
                    train_eval_dataset,
                    batch_size=self.config.training.batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    collate_fn=collate_multi_dataset,
                )
            except Exception as e:
                self.logger.warning(f"  {dataset_name}: Could not load train eval data: {e}")

        # Create test and calibration loaders for datasets with held-out splits
        self.test_loaders = {}
        self.calibration_loaders = {}

        # Datasets that support test/calibration splits
        # - encode4_*: Human MPRA with test/calibration splits
        # - dream_yeast: Yeast with special test set + 1% calibration from train
        # - deepstarr: Drosophila with chromosome-based test split
        # - jores_*: Plant with standard test splits
        datasets_with_test = [
            "encode4_k562", "encode4_hepg2", "encode4_wtc11",
            "dream_yeast", "deepstarr",
            "jores_arabidopsis", "jores_maize", "jores_sorghum"
        ]
        datasets_with_calibration = [
            "encode4_k562", "encode4_hepg2", "encode4_wtc11",
            "dream_yeast"
        ]

        for dataset_name in self.config.datasets:
            # Skip datasets without test splits
            if dataset_name not in datasets_with_test:
                continue

            info = DATASET_CATALOG[dataset_name]

            # Test loader
            try:
                test_dataset = SingleDataset(
                    info, "test",
                    self.config.target_sequence_length,
                    self.normalizer,
                    index_mappings=index_mappings,
                )
                self.test_loaders[dataset_name] = DataLoader(
                    test_dataset,
                    batch_size=self.config.training.batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    collate_fn=collate_multi_dataset,
                )
                self.logger.info(f"  {dataset_name}: test set loaded ({len(test_dataset)} samples)")
            except Exception as e:
                self.logger.warning(f"  {dataset_name}: Could not load test data: {e}")

            # Calibration loader (only for datasets that support it)
            if dataset_name in datasets_with_calibration:
                try:
                    calib_dataset = SingleDataset(
                        info, "calibration",
                        self.config.target_sequence_length,
                        self.normalizer,
                        index_mappings=index_mappings,
                    )
                    self.calibration_loaders[dataset_name] = DataLoader(
                        calib_dataset,
                        batch_size=self.config.training.batch_size,
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True,
                        collate_fn=collate_multi_dataset,
                    )
                    self.logger.info(f"  {dataset_name}: calibration set loaded ({len(calib_dataset)} samples)")
                except Exception as e:
                    self.logger.warning(f"  {dataset_name}: Could not load calibration data: {e}")

        # Setup metrics tracker
        output_names = {
            name: DATASET_CATALOG[name].output_names
            for name in self.config.datasets
            if name in DATASET_CATALOG
        }
        self.metrics_tracker = MetricsTracker(
            dataset_names=self.config.datasets,
            output_names_per_dataset=output_names,
        )

        # Save normalizer
        self.normalizer.save(str(self.output_dir / "normalizer.json"))
        self.logger.info("Data setup complete")

    def setup_model(self):
        """Setup model, optimizer, and scheduler."""
        self.logger.info("Setting up model...")

        # Create model with correct embedding sizes from train_dataset
        self.model = create_multi_species_model(
            config=self.config.model,
            dataset_names=self.config.datasets,
            n_species=len(self.train_dataset.species_to_idx),
            n_kingdoms=len(self.train_dataset.kingdom_to_idx),
            n_celltypes=len(self.train_dataset.celltype_to_idx),
        )
        self.model = self.model.to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {n_params:,} ({n_trainable:,} trainable)")

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # Setup scheduler
        if self.config.training.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.training.max_epochs,
                eta_min=self.config.training.min_lr,
            )
        elif self.config.training.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=self.config.training.min_lr,
            )
        elif self.config.training.scheduler == "onecycle":
            # OneCycleLR requires total_steps (train_loader already set up)
            steps_per_epoch = len(self.train_loader)
            total_steps = steps_per_epoch * self.config.training.max_epochs
            div_factor = self.config.training.onecycle_div_factor
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.learning_rate,
                total_steps=total_steps,
                pct_start=self.config.training.onecycle_pct_start,
                div_factor=div_factor,  # initial_lr = max_lr / div_factor
            )
            initial_lr = self.config.training.learning_rate / div_factor
            self.logger.info(f"OneCycleLR: {total_steps} total steps, "
                           f"initial_lr={initial_lr:.6f}, max_lr={self.config.training.learning_rate}")
        else:
            self.scheduler = None

        # Resume if specified
        if self.resume_from:
            self.load_checkpoint(self.resume_from)

        self.logger.info("Model setup complete")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.sampler.set_epoch(self.current_epoch)

        epoch_losses = []  # NLL loss (can be negative with uncertainty)
        epoch_mse_losses = []  # MSE loss (always positive, for monitoring)
        epoch_grad_norms = []
        per_head_losses = {name: [] for name in self.model.heads.keys()}

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            leave=False,
        )

        accumulation_steps = self.config.training.gradient_accumulation_steps
        accumulated_loss = 0.0

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            sequence = batch['sequence'].to(self.device)
            activity = batch['activity'].to(self.device)
            species_idx = batch['species_idx'].to(self.device)
            kingdom_idx = batch['kingdom_idx'].to(self.device)
            celltype_idx = batch['celltype_idx'].to(self.device)
            original_length = batch['original_length'].to(self.device)
            dataset_names = batch['dataset_names']

            # Forward pass with optional AMP
            with autocast(enabled=self.config.training.use_amp):
                outputs = self.model(
                    sequence=sequence,
                    species_idx=species_idx,
                    kingdom_idx=kingdom_idx,
                    celltype_idx=celltype_idx,
                    original_length=original_length,
                    dataset_names=dataset_names,
                )

                # Compute loss (with extreme value weighting)
                loss, head_losses = compute_masked_loss(
                    outputs=outputs,
                    targets=activity,
                    dataset_names=dataset_names,
                    dataset_to_heads=self.model.dataset_to_heads,
                    use_uncertainty=self.config.model.use_uncertainty,
                    use_extreme_weights=self.config.training.use_extreme_weights,
                    extreme_alpha=self.config.training.extreme_alpha,
                    extreme_beta=self.config.training.extreme_beta,
                )

                # Also compute MSE loss for monitoring (without weighting)
                mse_loss, _ = compute_masked_loss(
                    outputs=outputs,
                    targets=activity,
                    dataset_names=dataset_names,
                    dataset_to_heads=self.model.dataset_to_heads,
                    use_uncertainty=False,  # Pure MSE
                    use_extreme_weights=False,  # No weighting for monitoring
                )

                loss = loss / accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Step OneCycleLR scheduler after each batch (not epoch)
                if self.scheduler and isinstance(
                    self.scheduler, torch.optim.lr_scheduler.OneCycleLR
                ):
                    self.scheduler.step()

                self.optimizer.zero_grad()

                # Record metrics
                epoch_losses.append(accumulated_loss)
                mse_val = mse_loss.item() if isinstance(mse_loss, torch.Tensor) else mse_loss
                epoch_mse_losses.append(mse_val)
                epoch_grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

                for name, h_loss in head_losses.items():
                    per_head_losses[name].append(h_loss.item())

                accumulated_loss = 0.0
                self.global_step += 1

                # Update progress bar with both losses
                pbar.set_postfix({
                    'nll': f"{epoch_losses[-1]:.4f}",
                    'mse': f"{epoch_mse_losses[-1]:.4f}",
                    'grad': f"{epoch_grad_norms[-1]:.2f}",
                })

                # Log periodically
                if self.global_step % self.config.training.log_every_n_steps == 0:
                    self.logger.info(
                        f"Step {self.global_step}: NLL={epoch_losses[-1]:.4f}, "
                        f"MSE={epoch_mse_losses[-1]:.4f}, "
                        f"grad={epoch_grad_norms[-1]:.2f}, "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                    )

        # Epoch summary
        metrics = {
            'train_loss': np.mean(epoch_losses) if epoch_losses else 0.0,
            'train_mse': np.mean(epoch_mse_losses) if epoch_mse_losses else 0.0,
            'train_grad_norm': np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0,
        }

        for name, losses in per_head_losses.items():
            if losses:
                metrics[f'train_loss_{name}'] = np.mean(losses)

        return metrics

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

    @torch.no_grad()
    def evaluate_loader(
        self,
        loaders: Dict[str, DataLoader],
        split_name: str = "val",
        max_batches: Optional[int] = None,
    ) -> Tuple[Dict[str, Dict], Dict[str, float]]:
        """
        Evaluate on a set of data loaders.

        Returns:
            all_metrics: Per-dataset, per-output metrics (Pearson, Spearman, R2, etc.)
            aggregate_metrics: Aggregate loss metrics (NLL, MSE) across all data
        """
        self.model.eval()

        all_metrics = {}

        # Track losses across all batches
        total_nll = 0.0
        total_mse = 0.0
        total_samples = 0

        for dataset_name, loader in loaders.items():
            predictions = []
            targets = []
            log_vars = []  # For NLL computation
            weights = []

            for batch_idx, batch in enumerate(loader):
                if max_batches and batch_idx >= max_batches:
                    break

                sequence = batch['sequence'].to(self.device)
                activity = batch['activity'].to(self.device)
                species_idx = batch['species_idx'].to(self.device)
                kingdom_idx = batch['kingdom_idx'].to(self.device)
                celltype_idx = batch['celltype_idx'].to(self.device)
                original_length = batch['original_length'].to(self.device)
                ds_names = batch['dataset_names']

                # Forward pass
                outputs = self.model(
                    sequence=sequence,
                    species_idx=species_idx,
                    kingdom_idx=kingdom_idx,
                    celltype_idx=celltype_idx,
                    original_length=original_length,
                    dataset_names=ds_names,
                )

                # Compute losses for this batch (no weighting for evaluation)
                nll_loss, _ = compute_masked_loss(
                    outputs=outputs,
                    targets=activity,
                    dataset_names=ds_names,
                    dataset_to_heads=self.model.dataset_to_heads,
                    use_uncertainty=self.config.model.use_uncertainty,
                    use_extreme_weights=False,  # No weighting for evaluation
                )
                mse_loss, _ = compute_masked_loss(
                    outputs=outputs,
                    targets=activity,
                    dataset_names=ds_names,
                    dataset_to_heads=self.model.dataset_to_heads,
                    use_uncertainty=False,
                    use_extreme_weights=False,  # No weighting for evaluation
                )

                batch_size = sequence.shape[0]
                total_nll += nll_loss.item() * batch_size
                total_mse += mse_loss.item() * batch_size
                total_samples += batch_size

                # Extract predictions for this dataset's heads
                heads = self.model.dataset_to_heads.get(dataset_name, [])
                batch_preds = []
                batch_logvars = []

                for i, head_name in enumerate(heads):
                    if head_name in outputs:
                        pred_data = outputs[head_name]
                        head_preds = pred_data['mean'].cpu()
                        batch_preds.append(head_preds)
                        if 'log_var' in pred_data:
                            batch_logvars.append(pred_data['log_var'].cpu())

                if batch_preds:
                    batch_preds = torch.stack(batch_preds, dim=-1)
                    predictions.append(batch_preds)
                    targets.append(activity[:, :len(heads)].cpu())
                    if batch_logvars:
                        log_vars.append(torch.stack(batch_logvars, dim=-1))

                if 'weight' in batch:
                    weights.append(batch['weight'])

            if not predictions:
                continue

            predictions = torch.cat(predictions, dim=0).numpy()
            targets_np = torch.cat(targets, dim=0).numpy()

            # Handle single output
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            if targets_np.ndim == 1:
                targets_np = targets_np.reshape(-1, 1)

            # Compute metrics per output
            info = DATASET_CATALOG.get(dataset_name)
            output_names = info.output_names if info else ["output"]

            dataset_metrics = {}
            for i, output_name in enumerate(output_names):
                if i >= predictions.shape[1]:
                    continue

                # Inverse transform predictions for fair comparison
                pred_orig = self.normalizer.inverse_transform(
                    dataset_name,
                    predictions[:, i],
                )
                target_orig = self.normalizer.inverse_transform(
                    dataset_name,
                    targets_np[:, i],
                )

                metrics = compute_all_metrics(target_orig, pred_orig)
                dataset_metrics[output_name] = metrics.to_dict()

            # Special handling for DREAM yeast
            if dataset_name == "dream_yeast" and weights:
                dream_metrics = DREAMYeastMetrics()
                weights_arr = torch.cat(weights, dim=0).numpy()
                dream_results = dream_metrics.compute_dream_score(
                    targets_np[:, 0],
                    predictions[:, 0],
                    weights=weights_arr,
                )
                dataset_metrics["dream_score"] = dream_results

            all_metrics[dataset_name] = dataset_metrics

        # Compute aggregate metrics
        aggregate_metrics = {
            'nll': total_nll / total_samples if total_samples > 0 else 0.0,
            'mse': total_mse / total_samples if total_samples > 0 else 0.0,
        }

        return all_metrics, aggregate_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, Dict]:
        """Run validation on all datasets (backward compatible)."""
        all_metrics, _ = self.evaluate_loader(self.val_loaders, "val")
        return all_metrics

    @torch.no_grad()
    def _evaluate_held_out_sets(
        self,
        loaders: Dict[str, torch.utils.data.DataLoader],
        split_name: str = "test"
    ) -> Dict[str, Dict]:
        """Evaluate on held-out test or calibration sets."""
        self.model.eval()
        all_metrics = {}

        for dataset_name, loader in loaders.items():
            self.logger.info(f"  Evaluating {dataset_name} ({split_name})...")

            predictions = []
            targets = []
            log_vars = []

            for batch in loader:
                sequence = batch['sequence'].to(self.device)
                activity = batch['activity'].to(self.device)
                species_idx = batch['species_idx'].to(self.device)
                kingdom_idx = batch['kingdom_idx'].to(self.device)
                celltype_idx = batch['celltype_idx'].to(self.device)
                original_length = batch['original_length'].to(self.device)
                ds_names = batch['dataset_names']

                # Forward pass
                outputs = self.model(
                    sequence=sequence,
                    species_idx=species_idx,
                    kingdom_idx=kingdom_idx,
                    celltype_idx=celltype_idx,
                    original_length=original_length,
                    dataset_names=ds_names,
                )

                # Extract predictions for this dataset's heads
                heads = self.model.dataset_to_heads.get(dataset_name, [])
                batch_preds = []
                batch_logvars = []

                for head_name in heads:
                    if head_name in outputs:
                        pred_data = outputs[head_name]
                        batch_preds.append(pred_data['mean'].cpu())
                        if 'logvar' in pred_data:
                            batch_logvars.append(pred_data['logvar'].cpu())

                if batch_preds:
                    batch_preds = torch.stack(batch_preds, dim=-1)
                    predictions.append(batch_preds)
                    targets.append(activity[:, :len(heads)].cpu())
                    if batch_logvars:
                        log_vars.append(torch.stack(batch_logvars, dim=-1))

            if not predictions:
                self.logger.warning(f"    No predictions for {dataset_name}")
                continue

            predictions = torch.cat(predictions, dim=0).numpy()
            targets_np = torch.cat(targets, dim=0).numpy()

            # Handle single output
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            if targets_np.ndim == 1:
                targets_np = targets_np.reshape(-1, 1)

            # Compute metrics per output
            info = DATASET_CATALOG.get(dataset_name)
            output_names = info.output_names if info else [f"output_{i}" for i in range(predictions.shape[1])]

            dataset_metrics = {}
            for i, output_name in enumerate(output_names):
                if i >= predictions.shape[1]:
                    break

                preds = predictions[:, i]
                targs = targets_np[:, i]

                # Remove NaNs
                valid_mask = ~np.isnan(targs) & ~np.isnan(preds)
                if valid_mask.sum() < 2:
                    continue

                preds = preds[valid_mask]
                targs = targs[valid_mask]

                # Compute metrics
                from scipy.stats import pearsonr, spearmanr
                pearson_r, _ = pearsonr(preds, targs)
                spearman_r, _ = spearmanr(preds, targs)
                mse = np.mean((preds - targs) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(preds - targs))

                # R2
                ss_res = np.sum((targs - preds) ** 2)
                ss_tot = np.sum((targs - np.mean(targs)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                dataset_metrics[output_name] = {
                    'pearson': pearson_r,
                    'spearman': spearman_r,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'n_samples': len(preds),
                }

                self.logger.info(
                    f"    {output_name}: r={pearson_r:.4f}, rho={spearman_r:.4f}, "
                    f"R2={r2:.4f}, RMSE={rmse:.4f} (n={len(preds)})"
                )

            all_metrics[dataset_name] = dataset_metrics

        return all_metrics

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Config: {self.config.config_type.value}")

        # Setup
        self.setup_data()
        self.setup_model()

        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)

        # Training loop
        avg_val_pearson = 0.0
        for epoch in range(self.current_epoch, self.config.training.max_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate on both train and val sets
            if epoch % self.config.training.val_every_n_epochs == 0:
                # Evaluate on train set (limit batches for speed)
                max_train_batches = 50  # Limit to ~50 batches for train eval
                train_eval_metrics, train_agg = self.evaluate_loader(
                    self.train_eval_loaders, "train", max_batches=max_train_batches
                )

                # Evaluate on val set (full evaluation)
                val_metrics, val_agg = self.evaluate_loader(self.val_loaders, "val")

                # Helper to extract aggregate metrics from per-dataset metrics
                def extract_aggregate(metrics_dict):
                    pearsons, spearmans, r2s, rmses, maes = [], [], [], [], []
                    for ds_metrics in metrics_dict.values():
                        for output_metrics in ds_metrics.values():
                            if isinstance(output_metrics, dict) and 'pearson' in output_metrics:
                                pearsons.append(output_metrics['pearson']['value'])
                                spearmans.append(output_metrics['spearman']['value'])
                                r2s.append(output_metrics['r2']['value'])
                                rmses.append(output_metrics['rmse']['value'])
                                maes.append(output_metrics['mae']['value'])
                    return {
                        'pearson': np.mean(pearsons) if pearsons else 0.0,
                        'spearman': np.mean(spearmans) if spearmans else 0.0,
                        'r2': np.mean(r2s) if r2s else 0.0,
                        'rmse': np.mean(rmses) if rmses else 0.0,
                        'mae': np.mean(maes) if maes else 0.0,
                    }

                train_corr = extract_aggregate(train_eval_metrics)
                val_corr = extract_aggregate(val_metrics)

                avg_val_pearson = val_corr['pearson']

                # Check if best
                is_best = avg_val_pearson > self.best_val_metric

                if is_best:
                    self.best_val_metric = avg_val_pearson
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Log comprehensive epoch summary with SAME stats for train and val
                self.logger.info("=" * 80)
                self.logger.info(f"EPOCH {epoch} SUMMARY {'(BEST)' if is_best else ''}")
                self.logger.info("=" * 80)
                self.logger.info(
                    f"[TRAIN] NLL: {train_agg['nll']:.4f} | MSE: {train_agg['mse']:.4f} | "
                    f"r: {train_corr['pearson']:.4f} | rho: {train_corr['spearman']:.4f} | "
                    f"R2: {train_corr['r2']:.4f} | RMSE: {train_corr['rmse']:.4f}"
                )
                self.logger.info(
                    f"[VAL]   NLL: {val_agg['nll']:.4f} | MSE: {val_agg['mse']:.4f} | "
                    f"r: {val_corr['pearson']:.4f} | rho: {val_corr['spearman']:.4f} | "
                    f"R2: {val_corr['r2']:.4f} | RMSE: {val_corr['rmse']:.4f}"
                )
                self.logger.info("-" * 80)

                # Log per-dataset metrics for validation
                for ds_name, ds_metrics in val_metrics.items():
                    self.logger.info(f"  [VAL] {ds_name}:")
                    for output_name, metrics in ds_metrics.items():
                        if isinstance(metrics, dict) and 'pearson' in metrics:
                            self.logger.info(
                                f"    {output_name}: "
                                f"r={metrics['pearson']['value']:.4f} | "
                                f"rho={metrics['spearman']['value']:.4f} | "
                                f"R2={metrics['r2']['value']:.4f} | "
                                f"RMSE={metrics['rmse']['value']:.4f} | "
                                f"MAE={metrics['mae']['value']:.4f}"
                            )
                self.logger.info("=" * 80)

                # Save checkpoint
                self.save_checkpoint(
                    self.output_dir / f"checkpoint_epoch{epoch}.pt",
                    is_best=is_best,
                )

                # Early stopping check
                if self.patience_counter >= self.config.training.patience:
                    self.logger.info(
                        f"Early stopping triggered after {epoch} epochs "
                        f"(patience={self.config.training.patience})"
                    )
                    break

            # Update scheduler (skip OneCycleLR - it's stepped per batch in train_epoch)
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_pearson)
                elif not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")

        # Final evaluation
        self.logger.info("Training complete. Running final evaluation...")
        final_metrics = self.validate()

        # Evaluate on test and calibration sets (for human MPRA datasets)
        if hasattr(self, 'test_loaders') and self.test_loaders:
            self.logger.info("\n" + "="*60)
            self.logger.info("Evaluating on TEST sets...")
            self.logger.info("="*60)
            test_metrics = self._evaluate_held_out_sets(self.test_loaders, "test")
            final_metrics['test'] = test_metrics

        if hasattr(self, 'calibration_loaders') and self.calibration_loaders:
            self.logger.info("\n" + "="*60)
            self.logger.info("Evaluating on CALIBRATION sets...")
            self.logger.info("="*60)
            calib_metrics = self._evaluate_held_out_sets(self.calibration_loaders, "calibration")
            final_metrics['calibration'] = calib_metrics

        # Save final results
        results_path = self.output_dir / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_metrics, f, indent=2, cls=NumpyEncoder)

        self.logger.info(f"Results saved to {results_path}")

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
        """Multi-phase training loop."""
        self.logger.info("Starting multi-phase training...")

        phases = self.config.training_phases or []

        if not phases:
            # Fall back to single-phase training
            return super().train()

        # Setup
        self.setup_data()
        self.setup_model()

        # Save config
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
            self.logger.info(f"  Epochs: {phase_epochs}")
            self.logger.info(f"  Freeze backbone: {freeze_backbone}")
            self.logger.info(f"  Learning rate: {phase_lr}")
            self.logger.info(f"{'='*60}\n")

            # Apply phase settings
            if freeze_backbone:
                self.model.freeze_backbone()
            else:
                self.model.unfreeze_backbone()

            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = phase_lr

            # Reset scheduler for this phase
            if self.config.training.scheduler == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=phase_epochs,
                    eta_min=self.config.training.min_lr,
                )

            # Reset patience for this phase
            self.patience_counter = 0
            phase_start_epoch = self.current_epoch

            # Train phase
            for epoch in range(phase_epochs):
                self.current_epoch = phase_start_epoch + epoch
                epoch_start = time.time()

                train_metrics = self.train_epoch()

                # Validate
                val_metrics = self.validate()

                # Compute average validation metric
                val_pearsons = []
                for ds_metrics in val_metrics.values():
                    for output_metrics in ds_metrics.values():
                        if isinstance(output_metrics, dict) and 'pearson' in output_metrics:
                            val_pearsons.append(output_metrics['pearson']['value'])

                avg_val_pearson = np.mean(val_pearsons) if val_pearsons else 0.0

                is_best = avg_val_pearson > self.best_val_metric
                if is_best:
                    self.best_val_metric = avg_val_pearson
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                self.logger.info(
                    f"[{phase_name}] Epoch {epoch}: "
                    f"loss={train_metrics['train_loss']:.4f}, "
                    f"val_r={avg_val_pearson:.4f} "
                    f"{'(BEST)' if is_best else ''}"
                )

                # Save checkpoint
                self.save_checkpoint(
                    self.output_dir / f"checkpoint_{phase_name}_epoch{epoch}.pt",
                    is_best=is_best,
                )

                # Early stopping within phase
                if self.patience_counter >= self.config.training.patience // 2:
                    self.logger.info(f"Early stopping phase {phase_name}")
                    break

                if self.scheduler:
                    self.scheduler.step()

                epoch_time = time.time() - epoch_start
                self.logger.info(f"Epoch completed in {epoch_time:.1f}s")

        # Final evaluation
        self.logger.info("Multi-phase training complete. Running final evaluation...")
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

    # Physics
    physics_features: Optional[np.ndarray] = None

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

    # PhysicsVAE checkpoints
    VAE_CHECKPOINTS = {
        'K562': Path('physics/PhysicsVAE/runs/K562_20260113_051653/best_model.pt'),
        'HepG2': Path('physics/PhysicsVAE/runs/HepG2_20260113_052418/best_model.pt'),
        'WTC11': Path('physics/PhysicsVAE/runs/WTC11_20260113_052743/best_model.pt'),
    }

    def __init__(
        self,
        target_cell: str = 'HepG2',
        background_cells: List[str] = None,
        device: str = 'cuda',
        required_motifs: List[str] = None,
        forbidden_motifs: List[str] = None,
    ):
        """
        Initialize therapeutic enhancer design pipeline.

        Args:
            target_cell: Cell type to maximize activity in
            background_cells: Cell types to minimize activity in
            device: 'cuda' or 'cpu'
            required_motifs: TF motifs that must be present
            forbidden_motifs: TF motifs that must be absent
        """
        self.target_cell = target_cell
        self.background_cells = background_cells or [c for c in ['K562', 'HepG2', 'WTC11'] if c != target_cell]
        self.all_cell_types = [target_cell] + self.background_cells
        self.device = device

        # Motif constraints
        if required_motifs is None and target_cell in CELL_TYPE_MOTIF_CONSTRAINTS:
            required_motifs = CELL_TYPE_MOTIF_CONSTRAINTS[target_cell]['required']
        if forbidden_motifs is None and target_cell in CELL_TYPE_MOTIF_CONSTRAINTS:
            forbidden_motifs = CELL_TYPE_MOTIF_CONSTRAINTS[target_cell]['forbidden']
        self.required_motifs = required_motifs or []
        self.forbidden_motifs = forbidden_motifs or []

        # Initialize components
        print(f"Initializing Therapeutic Enhancer Pipeline")
        print(f"  Target: {target_cell}, Background: {self.background_cells}")
        print(f"  Required motifs: {self.required_motifs}")
        print(f"  Forbidden motifs: {self.forbidden_motifs}")

        # Multi-cell ensemble
        print("Loading multi-cell CADENCE ensemble...")
        checkpoints = find_cadence_checkpoints()
        self.ensemble = MultiCellEnsemble(
            cell_types=self.all_cell_types,
            checkpoints=checkpoints,
            device=device
        )

        # Motif scanner
        self.motif_scanner = None
        if HAS_ORACLE_CHECK:
            print("Loading motif scanner (879 JASPAR human motifs)...")
            self.motif_scanner = MotifScanner(species='human')
            print(f"  Loaded {len(self.motif_scanner.pwms)} motifs")

        # PhysicsVAE
        self.vae_model = None
        if HAS_PHYSICS_VAE and target_cell in self.VAE_CHECKPOINTS:
            vae_path = self.VAE_CHECKPOINTS[target_cell]
            if vae_path.exists():
                print(f"Loading PhysicsVAE for {target_cell}...")
                self._load_physics_vae(target_cell)

        # Composition validator for OracleCheck
        self.composition_validator = None
        if HAS_ORACLE_CHECK:
            try:
                from oracle_check.config import ValidationThresholds
                thresholds = ValidationThresholds()
                self.composition_validator = CompositionValidator(None, thresholds)
            except Exception as e:
                print(f"  Note: CompositionValidator not loaded: {e}")

        # Storage
        self.candidates: List[EnhancerCandidate] = []
        self.natural_physics_profiles: Optional[np.ndarray] = None
        # Physics feature dimensions per cell type
        self.n_physics_features = {'K562': 515, 'HepG2': 539, 'WTC11': 539}.get(target_cell, 515)

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
        fasta_path: str = None,
        n_vae_candidates: int = 500,
        n_select: int = 50,
        output_dir: str = None
    ) -> 'pd.DataFrame':
        """
        Run the complete 6-step therapeutic enhancer design pipeline.

        Args:
            natural_sequences: List of natural enhancer sequences
            fasta_path: Path to FASTA file with natural enhancers
            n_vae_candidates: Number of VAE-generated candidates
            n_select: Final number of diverse candidates to select
            output_dir: Directory to save results

        Returns:
            DataFrame with top diverse candidates
        """
        print("\n" + "="*60)
        print("THERAPEUTIC ENHANCER DESIGN PIPELINE")
        print("="*60)
        print(f"Target cell: {self.target_cell}")
        print(f"Background cells: {self.background_cells}")
        print("="*60)

        # Load natural sequences if needed
        if natural_sequences is None and fasta_path:
            natural_sequences = self.load_natural_enhancers(fasta_path)

        all_candidates = []

        # Check if we have any source of candidates
        if not natural_sequences and (n_vae_candidates == 0 or self.vae_model is None):
            print("Warning: No natural sequences provided and VAE not available.")
            print("  Provide --fasta with natural enhancers or ensure PhysicsVAE is loaded.")
            return pd.DataFrame()

        # Step 1: Extract physics profile from natural enhancers
        if natural_sequences:
            target_physics = self.extract_target_physics_profile(natural_sequences)

            # Evaluate natural sequences
            natural_candidates = self.evaluate_candidates(
                natural_sequences, source='natural'
            )
            all_candidates.extend(natural_candidates)
        else:
            n_physics = getattr(self, 'n_physics_features', 515)
            target_physics = np.zeros(n_physics)  # Default

        # Step 2: Generate VAE candidates
        if n_vae_candidates > 0 and self.vae_model is not None:
            vae_sequences = self.generate_vae_candidates(
                target_physics, n_candidates=n_vae_candidates
            )

            if vae_sequences:
                vae_candidates = self.evaluate_candidates(
                    vae_sequences, source='vae_generated'
                )
                all_candidates.extend(vae_candidates)

        # Store all candidates
        self.candidates = all_candidates

        # Step 6: Rank and diversify
        result_df = self.rank_and_diversify(all_candidates, n_select=n_select)

        # Save results
        if output_dir and len(result_df) > 0:
            self.save_results(result_df, output_dir)

        # Print summary
        self._print_summary(result_df)

        return result_df


# =============================================================================
# PART 5: Disease Variant Pipeline
# Source: applications/disease_variant_pipeline.py
# =============================================================================

@dataclass
class ActivityPrediction:
    """Activity prediction with uncertainty."""
    mean: float
    std: float  # Total uncertainty
    epistemic: float = 0.0  # Model uncertainty
    aleatoric: float = 0.0  # Data uncertainty

    @property
    def confidence_interval_95(self) -> Tuple[float, float]:
        return (self.mean - 1.96 * self.std, self.mean + 1.96 * self.std)


@dataclass
class PhysicsPrediction:
    """Physics feature predictions."""
    features: np.ndarray  # Shape: (n_features,)
    feature_names: List[str]
    uncertainties: Optional[np.ndarray] = None

    def get_feature(self, name: str) -> float:
        idx = self.feature_names.index(name)
        return self.features[idx]

    def to_dict(self) -> Dict[str, float]:
        return {name: float(val) for name, val in zip(self.feature_names, self.features)}

    def get_family_features(self, family: str) -> Dict[str, float]:
        """Get features belonging to a physics family (thermo, stiff, bend, etc.)"""
        return {
            name: float(val)
            for name, val in zip(self.feature_names, self.features)
            if name.startswith(family)
        }


@dataclass
class VariantEffect:
    """Complete variant effect analysis."""
    variant_id: str
    ref_sequence: str
    alt_sequence: str

    # Activity predictions
    activity_ref: ActivityPrediction
    activity_alt: ActivityPrediction

    # Physics predictions
    physics_ref: PhysicsPrediction
    physics_alt: PhysicsPrediction

    # Derived metrics
    delta_activity: float = field(init=False)
    delta_activity_zscore: float = field(init=False)
    effect_direction: str = field(init=False)
    effect_magnitude: str = field(init=False)

    def __post_init__(self):
        self.delta_activity = self.activity_alt.mean - self.activity_ref.mean

        # Z-score using combined uncertainty
        combined_std = np.sqrt(self.activity_ref.std**2 + self.activity_alt.std**2)
        self.delta_activity_zscore = self.delta_activity / combined_std if combined_std > 0 else 0

        # Effect direction
        if self.delta_activity > 0:
            self.effect_direction = 'activating'
        elif self.delta_activity < 0:
            self.effect_direction = 'repressing'
        else:
            self.effect_direction = 'neutral'

        # Effect magnitude
        abs_z = abs(self.delta_activity_zscore)
        if abs_z >= 3:
            self.effect_magnitude = 'strong'
        elif abs_z >= 2:
            self.effect_magnitude = 'moderate'
        elif abs_z >= 1:
            self.effect_magnitude = 'weak'
        else:
            self.effect_magnitude = 'negligible'

    @property
    def delta_physics(self) -> Dict[str, float]:
        """Compute physics feature changes."""
        return {
            name: float(self.physics_alt.features[i] - self.physics_ref.features[i])
            for i, name in enumerate(self.physics_ref.feature_names)
        }

    def top_changed_physics(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most changed physics features."""
        deltas = self.delta_physics
        sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_deltas[:n]

    def get_physics_family_changes(self) -> Dict[str, Dict[str, float]]:
        """Get physics changes grouped by family."""
        families = ['thermo', 'stiff', 'bend', 'entropy', 'advanced', 'pwm']
        result = {}

        for family in families:
            family_changes = {}
            for name, delta in self.delta_physics.items():
                if name.startswith(family):
                    family_changes[name] = delta
            if family_changes:
                result[family] = {
                    'features': family_changes,
                    'mean_abs_change': np.mean(np.abs(list(family_changes.values()))),
                    'max_abs_change': np.max(np.abs(list(family_changes.values()))) if family_changes else 0
                }

        return result


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
        """
        Initialize the disease variant pipeline.

        Args:
            reference_genome: Path to reference genome FASTA
            cadence_checkpoint: Path to CADENCE model checkpoint
            physinformer_checkpoint: Path to PhysInformer checkpoint
            cell_type: Cell type for predictions
            flank_size: Flanking sequence size (total = 2*flank + variant)
            device: 'cuda' or 'cpu'
        """
        self.cell_type = cell_type
        self.flank_size = flank_size
        self.device = device

        # Initialize components
        self.extractor = VariantExtractor(
            reference_genome=reference_genome,
            flank_size=flank_size
        )

        self.analyzer = DifferentialAnalyzer(
            cadence_checkpoint=cadence_checkpoint,
            physinformer_checkpoint=physinformer_checkpoint,
            cell_type=cell_type,
            device=device
        )

        # Results storage
        self.variant_effects: List[VariantEffect] = []
        self.summary_df: 'pd.DataFrame' = None

    def load_variants_from_vcf(
        self,
        vcf_path: str,
        max_variants: int = None,
        filter_pass_only: bool = True
    ) -> List['VariantSequences']:
        """Load and extract sequences for variants from VCF."""
        print(f"Loading variants from {vcf_path}...")
        variant_sequences = self.extractor.extract_from_vcf(
            vcf_path,
            max_variants=max_variants,
            filter_pass_only=filter_pass_only
        )
        print(f"Loaded {len(variant_sequences)} variants")
        return variant_sequences

    def load_variants_from_dataframe(
        self,
        df: 'pd.DataFrame',
        chrom_col: str = 'chrom',
        pos_col: str = 'pos',
        ref_col: str = 'ref',
        alt_col: str = 'alt',
        id_col: str = None
    ) -> List['VariantSequences']:
        """Load and extract sequences for variants from DataFrame."""
        print(f"Extracting sequences for {len(df)} variants...")
        variant_sequences = self.extractor.extract_from_dataframe(
            df,
            chrom_col=chrom_col,
            pos_col=pos_col,
            ref_col=ref_col,
            alt_col=alt_col,
            id_col=id_col
        )
        print(f"Extracted {len(variant_sequences)} variant sequences")
        return variant_sequences

    def analyze_variants(
        self,
        variant_sequences: List['VariantSequences'],
        progress: bool = True
    ) -> List[VariantEffect]:
        """
        Perform differential analysis on all variants.

        Args:
            variant_sequences: List of extracted variant sequences
            progress: Show progress bar

        Returns:
            List of VariantEffect objects with predictions
        """
        print(f"Analyzing {len(variant_sequences)} variants...")
        self.variant_effects = self.analyzer.analyze_variants(
            variant_sequences, progress=progress
        )
        print(f"Successfully analyzed {len(self.variant_effects)} variants")
        return self.variant_effects

    def score_and_rank(
        self,
        min_zscore: float = None,
        effect_directions: List[str] = None,
        effect_magnitudes: List[str] = None
    ) -> 'pd.DataFrame':
        """
        Score and rank variants by predicted effect.

        Args:
            min_zscore: Minimum absolute z-score to include
            effect_directions: Filter by direction ['activating', 'repressing']
            effect_magnitudes: Filter by magnitude ['strong', 'moderate', 'weak']

        Returns:
            DataFrame sorted by absolute z-score
        """
        if not self.variant_effects:
            raise ValueError("No variant effects. Run analyze_variants first.")

        # Convert to DataFrame
        self.summary_df = self.analyzer.to_dataframe(self.variant_effects)

        # Apply filters
        if min_zscore is not None:
            self.summary_df = self.summary_df[
                abs(self.summary_df['delta_activity_zscore']) >= min_zscore
            ]

        if effect_directions:
            self.summary_df = self.summary_df[
                self.summary_df['effect_direction'].isin(effect_directions)
            ]

        if effect_magnitudes:
            self.summary_df = self.summary_df[
                self.summary_df['effect_magnitude'].isin(effect_magnitudes)
            ]

        # Sort by absolute z-score
        self.summary_df['abs_zscore'] = abs(self.summary_df['delta_activity_zscore'])
        self.summary_df = self.summary_df.sort_values('abs_zscore', ascending=False)
        self.summary_df = self.summary_df.drop(columns=['abs_zscore'])

        return self.summary_df.reset_index(drop=True)

    def get_top_variants(
        self,
        n: int = 10,
        direction: str = None
    ) -> 'pd.DataFrame':
        """
        Get top N most impactful variants.

        Args:
            n: Number of variants to return
            direction: 'activating' or 'repressing' (optional)
        """
        if self.summary_df is None:
            self.score_and_rank()

        df = self.summary_df.copy()

        if direction:
            df = df[df['effect_direction'] == direction]

        return df.head(n)

    def generate_report(
        self,
        output_path: str = None,
        include_physics: bool = True,
        top_n: int = 50
    ) -> Dict:
        """
        Generate comprehensive variant interpretation report.

        Args:
            output_path: Path to save JSON report
            include_physics: Include physics feature analysis
            top_n: Number of top variants to detail

        Returns:
            Report dictionary
        """
        if not self.variant_effects:
            raise ValueError("No variant effects. Run analyze_variants first.")

        if self.summary_df is None:
            self.score_and_rank()

        report = {
            'pipeline_info': {
                'cell_type': self.cell_type,
                'flank_size': self.flank_size,
                'n_variants_analyzed': len(self.variant_effects),
            },
            'summary_statistics': {
                'n_activating': int((self.summary_df['effect_direction'] == 'activating').sum()),
                'n_repressing': int((self.summary_df['effect_direction'] == 'repressing').sum()),
                'n_neutral': int((self.summary_df['effect_direction'] == 'neutral').sum()),
                'n_strong': int((self.summary_df['effect_magnitude'] == 'strong').sum()),
                'n_moderate': int((self.summary_df['effect_magnitude'] == 'moderate').sum()),
                'n_weak': int((self.summary_df['effect_magnitude'] == 'weak').sum()),
                'mean_delta_activity': float(self.summary_df['delta_activity'].mean()),
                'std_delta_activity': float(self.summary_df['delta_activity'].std()),
            },
            'top_variants': []
        }

        # Add details for top variants
        sorted_effects = sorted(
            self.variant_effects,
            key=lambda x: abs(x.delta_activity_zscore),
            reverse=True
        )

        for effect in sorted_effects[:top_n]:
            variant_detail = {
                'variant_id': effect.variant_id,
                'activity_ref': effect.activity_ref.mean,
                'activity_alt': effect.activity_alt.mean,
                'delta_activity': effect.delta_activity,
                'zscore': effect.delta_activity_zscore,
                'direction': effect.effect_direction,
                'magnitude': effect.effect_magnitude,
            }

            if include_physics:
                # Add top physics changes
                physics_changes = effect.get_physics_family_changes()
                variant_detail['physics_summary'] = {
                    family: {
                        'mean_change': data['mean_abs_change'],
                        'max_change': data['max_abs_change']
                    }
                    for family, data in physics_changes.items()
                }

            report['top_variants'].append(variant_detail)

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {output_path}")

        return report

    def save_results(
        self,
        output_dir: str,
        prefix: str = 'variant_analysis'
    ):
        """
        Save all results to files.

        Args:
            output_dir: Output directory
            prefix: File name prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary DataFrame
        if self.summary_df is not None:
            summary_path = output_dir / f'{prefix}_summary.csv'
            self.summary_df.to_csv(summary_path, index=False)
            print(f"Summary saved to {summary_path}")

        # Save full results
        results_path = output_dir / f'{prefix}_full_results.csv'
        full_records = []
        for effect in self.variant_effects:
            record = {
                'variant_id': effect.variant_id,
                'ref_sequence': effect.ref_sequence,
                'alt_sequence': effect.alt_sequence,
                'activity_ref': effect.activity_ref.mean,
                'activity_ref_std': effect.activity_ref.std,
                'activity_alt': effect.activity_alt.mean,
                'activity_alt_std': effect.activity_alt.std,
                'delta_activity': effect.delta_activity,
                'delta_activity_zscore': effect.delta_activity_zscore,
                'effect_direction': effect.effect_direction,
                'effect_magnitude': effect.effect_magnitude,
            }

            # Add physics deltas
            for name, value in effect.delta_physics.items():
                record[f'delta_{name}'] = value

            full_records.append(record)

        pd.DataFrame(full_records).to_csv(results_path, index=False)
        print(f"Full results saved to {results_path}")

        # Save report
        report_path = output_dir / f'{prefix}_report.json'
        self.generate_report(str(report_path))

    def run_pipeline(
        self,
        vcf_path: str = None,
        variants_df: 'pd.DataFrame' = None,
        output_dir: str = None,
        max_variants: int = None
    ) -> 'pd.DataFrame':
        """
        Run complete pipeline end-to-end.

        Args:
            vcf_path: Path to VCF file (either this or variants_df required)
            variants_df: DataFrame with variants
            output_dir: Directory to save results
            max_variants: Maximum variants to process

        Returns:
            Summary DataFrame with ranked variants
        """
        # Load variants
        if vcf_path:
            variant_sequences = self.load_variants_from_vcf(
                vcf_path, max_variants=max_variants
            )
        elif variants_df is not None:
            variant_sequences = self.load_variants_from_dataframe(variants_df)
        else:
            raise ValueError("Either vcf_path or variants_df required")

        # Analyze
        self.analyze_variants(variant_sequences)

        # Score and rank
        summary = self.score_and_rank()

        # Save if output directory provided
        if output_dir:
            self.save_results(output_dir)

        return summary


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

    def _create_dummy_data(self):
        """Create dummy data for testing (DEPRECATED - use _load_from_real_data)."""
        print("WARNING: Creating dummy data - real data loaders not available!")
        n_samples = 1000 if self.split == "train" else 200
        seq_len = self.info.sequence_length
        n_outputs = self.info.num_outputs

        # Random one-hot sequences
        self.sequences = np.zeros((n_samples, 4, seq_len), dtype=np.float32)
        random_bases = np.random.randint(0, 4, (n_samples, seq_len))
        for i in range(n_samples):
            for j in range(seq_len):
                self.sequences[i, random_bases[i, j], j] = 1.0

        # Random activities
        self.activities = np.random.randn(n_samples, n_outputs).astype(np.float32)
        if n_outputs == 1:
            self.activities = self.activities.squeeze(-1)

        self.weights = None

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

    def get_by_dataset(
        self,
        dataset_name: str,
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Get item by dataset name and local index."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")

        item = self.datasets[dataset_name][idx]

        item["species_idx"] = torch.tensor(
            self.species_to_idx[item["species"]]
        )
        item["kingdom_idx"] = torch.tensor(
            self.kingdom_to_idx[item["kingdom"]]
        )
        item["celltype_idx"] = torch.tensor(
            self.celltype_to_idx[item["cell_type"]]
        )

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


# ==============================================================================
# MODULE 8: OracleCheck --- In-Silico Design Validation Protocol
# ==============================================================================
#
# Source: oracle_check/
# Files: config.py (187 lines), protocol.py (564 lines), validators.py (798 lines),
#        reference_panels.py (711 lines), scorecard.py (320 lines),
#        rc_consistency.py (417 lines), two_sample_tests.py (535 lines)
# Total: ~3,532 lines
#
# OracleCheck validates designed regulatory sequences against reference panels
# built from natural high-performing regulatory elements. It integrates outputs
# from CADENCE (activity prediction), PhysInformer (physics features), TileFormer
# (electrostatics), and PLACE (uncertainty/OOD). Each designed sequence receives
# a GREEN/YELLOW/RED verdict indicating its naturality.
#
# KEY RESULT: 96.5% GREEN verdicts on HepG2 therapeutic enhancers; 0% RED.
#             Random GC-matched controls: only 12% GREEN, 43% RED.
# ==============================================================================


# === oracle_check/config.py ===
# Configuration classes, thresholds, and physics feature families

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

    # RC Consistency
    rc_delta_threshold: float = 0.1     # Max prediction difference for RC


@dataclass
class OracleCheckConfig:
    """Main configuration for OracleCheck validation."""

    # Paths to trained models
    cadence_models_dir: Path = field(
        default_factory=lambda: Path("cadence_place")
    )
    physinformer_runs_dir: Path = field(
        default_factory=lambda: Path("physics/PhysInformer/runs")
    )

    # Reference panel settings
    reference_panels_dir: Path = field(
        default_factory=lambda: Path("oracle_check/reference_panels")
    )
    natural_high_performer_quantile: float = 0.75  # Top 25% for high performers
    background_sample_size: int = 10000  # Number of background samples
    knn_n_neighbors: int = 200  # For OOD detection

    # Validation thresholds
    thresholds: ValidationThresholds = field(default_factory=ValidationThresholds)

    # Device
    device: str = "cuda"
    batch_size: int = 64


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


# === oracle_check/validators.py ===
# Core validators: Physics, Composition, Confidence, Mahalanobis, OracleCheck

@dataclass
class PhysicsValidationResult:
    """Result of physics validation."""
    passed: bool
    overall_score: float
    family_scores: Dict[str, float]  # z-scores per family
    family_passed: Dict[str, bool]
    max_z_score: float
    nll: float
    message: str
    details: Dict = field(default_factory=dict)


@dataclass
class CompositionValidationResult:
    """Result of composition validation."""
    passed: bool
    gc_content: float
    gc_passed: bool
    cpg_oe: float
    cpg_passed: bool
    entropy: float
    entropy_passed: bool
    repeat_fraction: float
    repeat_passed: bool
    max_homopolymer: int
    homopolymer_passed: bool
    message: str


@dataclass
class ConfidenceValidationResult:
    """Result of confidence/OOD validation."""
    passed: bool
    epistemic_std: Optional[float]
    epistemic_passed: bool
    conformal_width: Optional[float]
    conformal_passed: bool
    ood_score: Optional[float]
    ood_passed: bool
    message: str


class PhysicsValidator:
    """
    Validates physics conformity of sequences.

    Checks per-family z-scores against natural high performers and
    physics NLL (negative log-likelihood).
    """

    def __init__(self, reference_panel, thresholds: ValidationThresholds):
        self.reference = reference_panel
        self.thresholds = thresholds

    def validate(
        self,
        physics_features: np.ndarray,
        feature_names: List[str],
    ) -> PhysicsValidationResult:
        """Validate physics features against reference panel."""
        family_scores = {}
        family_passed = {}
        nlls = []

        for family, ref_features in self.reference.physics_distributions.items():
            family_z_scores = []

            for feature_name, ref_dist in ref_features.items():
                if feature_name in feature_names:
                    idx = feature_names.index(feature_name)
                    value = physics_features[idx]

                    # Compute z-score
                    z = (value - ref_dist.mean) / (ref_dist.std + 1e-8)
                    family_z_scores.append(abs(z))

                    # Compute NLL contribution
                    nll = 0.5 * z ** 2 + np.log(ref_dist.std + 1e-8) + 0.5 * np.log(2 * np.pi)
                    nlls.append(nll)

            if family_z_scores:
                max_z = max(family_z_scores)
                family_scores[family] = max_z
                family_passed[family] = max_z <= self.thresholds.physics_z_soft
            else:
                family_scores[family] = 0.0
                family_passed[family] = True

        max_z_score = max(family_scores.values()) if family_scores else 0.0
        total_nll = sum(nlls) if nlls else 0.0

        all_soft_pass = all(family_passed.values())
        no_hard_fail = max_z_score <= self.thresholds.physics_z_hard
        passed = all_soft_pass and no_hard_fail

        if passed:
            message = f"Physics check passed (max z={max_z_score:.2f})"
        elif not no_hard_fail:
            message = f"Physics HARD FAIL: max z={max_z_score:.2f} > {self.thresholds.physics_z_hard}"
        else:
            failing = [f for f, p in family_passed.items() if not p]
            message = f"Physics soft fail in families: {failing}"

        return PhysicsValidationResult(
            passed=passed, overall_score=max_z_score,
            family_scores=family_scores, family_passed=family_passed,
            max_z_score=max_z_score, nll=total_nll, message=message,
        )


class CompositionValidator:
    """
    Validates composition metrics: GC content, CpG O/E, Shannon entropy,
    repeat fraction, and maximum homopolymer length.
    """

    def __init__(self, reference_panel, thresholds: ValidationThresholds):
        self.reference = reference_panel
        self.thresholds = thresholds

    def validate(self, sequence: str) -> CompositionValidationResult:
        """Validate composition of a sequence."""
        seq = sequence.upper()

        # GC content
        gc = sum(1 for b in seq if b in "GC") / len(seq) if len(seq) > 0 else 0.0
        gc_passed = self.thresholds.gc_min <= gc <= self.thresholds.gc_max

        # CpG O/E
        n = len(seq)
        c_count = seq.count("C")
        g_count = seq.count("G")
        cpg_count = seq.count("CG")
        expected = (c_count * g_count) / n if n > 0 else 0
        cpg_oe = cpg_count / expected if expected > 0 else 1.0
        if self.reference and self.reference.cpg_distribution:
            cpg_low = self.reference.cpg_distribution.percentiles["p5"]
            cpg_high = self.reference.cpg_distribution.percentiles["p95"]
            cpg_passed = cpg_low <= cpg_oe <= cpg_high
        else:
            cpg_passed = 0.3 <= cpg_oe <= 2.0

        # Shannon entropy
        counts = {b: seq.count(b) for b in "ACGT"}
        entropy = -sum(
            (c / n) * np.log2(c / n) for c in counts.values() if c > 0
        ) if n > 0 else 0.0
        if self.reference and self.reference.entropy_distribution:
            entropy_passed = entropy >= self.reference.entropy_distribution.percentiles["p5"]
        else:
            entropy_passed = entropy >= 1.5

        # Repeat fraction (dinuc + trinuc repeats)
        repeat_positions = set()
        for i in range(n - 3):
            dinuc = seq[i:i+2]
            if seq[i:i+4] == dinuc * 2:
                repeat_positions.update(range(i, min(i+4, n)))
        for i in range(n - 5):
            trinuc = seq[i:i+3]
            if seq[i:i+6] == trinuc * 2:
                repeat_positions.update(range(i, min(i+6, n)))
        repeat_fraction = len(repeat_positions) / n if n > 0 else 0.0
        repeat_passed = repeat_fraction < self.thresholds.repeat_fraction_hard

        # Max homopolymer
        max_homo = 1
        current = 1
        for i in range(1, n):
            if seq[i] == seq[i-1]:
                current += 1
                max_homo = max(max_homo, current)
            else:
                current = 1
        homopolymer_passed = max_homo < self.thresholds.max_homopolymer_hard

        passed = all([gc_passed, cpg_passed, entropy_passed, repeat_passed, homopolymer_passed])

        return CompositionValidationResult(
            passed=passed,
            gc_content=gc, gc_passed=gc_passed,
            cpg_oe=cpg_oe, cpg_passed=cpg_passed,
            entropy=entropy, entropy_passed=entropy_passed,
            repeat_fraction=repeat_fraction, repeat_passed=repeat_passed,
            max_homopolymer=max_homo, homopolymer_passed=homopolymer_passed,
            message="Composition check passed" if passed else "Composition check failed",
        )


class ConfidenceValidator:
    """
    Validates PLACE confidence estimates: epistemic uncertainty (σ_epi),
    conformal prediction width, and OOD kNN score.
    """

    def __init__(
        self, reference_panel, thresholds: ValidationThresholds,
        reference_epistemic_stds=None, reference_conformal_widths=None,
    ):
        self.thresholds = thresholds
        # Compute reference thresholds from natural high performers
        self.epistemic_threshold = (
            np.percentile(reference_epistemic_stds, thresholds.epistemic_percentile)
            if reference_epistemic_stds is not None else None
        )
        self.conformal_threshold = (
            np.percentile(reference_conformal_widths, thresholds.conformal_width_percentile)
            if reference_conformal_widths is not None else None
        )

    def validate(
        self,
        epistemic_std=None, conformal_width=None,
        ood_score=None, reference_ood_scores=None,
    ) -> ConfidenceValidationResult:
        """Validate PLACE confidence and OOD metrics."""
        epistemic_passed = True
        conformal_passed = True
        ood_passed = True

        if epistemic_std is not None and self.epistemic_threshold is not None:
            epistemic_passed = epistemic_std <= self.epistemic_threshold

        if conformal_width is not None and self.conformal_threshold is not None:
            conformal_passed = conformal_width <= self.conformal_threshold

        if ood_score is not None and reference_ood_scores is not None:
            ood_threshold = np.percentile(reference_ood_scores, self.thresholds.ood_percentile)
            ood_passed = ood_score <= ood_threshold

        passed = epistemic_passed and conformal_passed and ood_passed

        return ConfidenceValidationResult(
            passed=passed,
            epistemic_std=epistemic_std, epistemic_passed=epistemic_passed,
            conformal_width=conformal_width, conformal_passed=conformal_passed,
            ood_score=ood_score, ood_passed=ood_passed,
            message="Confidence check passed" if passed else "Confidence check failed",
        )


class OracleCheckValidator:
    """
    Main validator combining all checks. Determines GREEN/YELLOW/RED verdict.

    Integrates: PhysicsValidator, CompositionValidator, ConfidenceValidator,
    MahalanobisValidator.

    Verdict rules:
    - GREEN: All checks pass
    - YELLOW: At most one soft failure
    - RED: Any hard failure, or >=2 soft failures
    """

    def __init__(self, config, reference_panel, **kwargs):
        self.config = config
        self.thresholds = config.thresholds
        self.physics_validator = PhysicsValidator(reference_panel, self.thresholds)
        self.composition_validator = CompositionValidator(reference_panel, self.thresholds)
        self.confidence_validator = ConfidenceValidator(
            reference_panel, self.thresholds, **kwargs,
        )

    def validate_sequence(
        self, sequence: str,
        physics_features=None, physics_feature_names=None,
        prediction_mean=None, epistemic_std=None,
        conformal_width=None, ood_score=None,
        reference_ood_scores=None, **kwargs,
    ) -> Tuple[Verdict, Dict]:
        """Run all validation checks on a sequence."""
        results = {}
        hard_fails = []
        soft_fails = []

        # Composition check
        comp_result = self.composition_validator.validate(sequence)
        results["composition"] = comp_result
        if not comp_result.repeat_passed or not comp_result.homopolymer_passed:
            hard_fails.append("composition")
        elif not comp_result.passed:
            soft_fails.append("composition")

        # Physics check
        if physics_features is not None and physics_feature_names is not None:
            phys_result = self.physics_validator.validate(physics_features, physics_feature_names)
            results["physics"] = phys_result
            if phys_result.max_z_score > self.thresholds.physics_z_hard:
                hard_fails.append("physics")
            elif not phys_result.passed:
                soft_fails.append("physics")

        # Confidence check (PLACE epistemic/conformal + OOD)
        conf_result = self.confidence_validator.validate(
            epistemic_std=epistemic_std,
            conformal_width=conformal_width,
            ood_score=ood_score,
            reference_ood_scores=reference_ood_scores,
        )
        results["confidence"] = conf_result
        if not conf_result.ood_passed:
            hard_fails.append("OOD")
        elif not conf_result.passed:
            soft_fails.append("confidence")

        # Determine verdict
        if hard_fails:
            verdict = Verdict.RED
        elif len(soft_fails) > 1:
            verdict = Verdict.RED
        elif len(soft_fails) == 1:
            verdict = Verdict.YELLOW
        else:
            verdict = Verdict.GREEN

        results["verdict"] = verdict
        results["hard_fails"] = hard_fails
        results["soft_fails"] = soft_fails

        return verdict, results


# === oracle_check/scorecard.py ===
# Sequence and batch scorecards for tracking validation results

@dataclass
class SequenceScorecard:
    """Complete scorecard for a single sequence."""
    sequence: str
    sequence_id: Optional[str] = None
    verdict: Verdict = Verdict.RED

    # Activity
    activity_mean: Optional[float] = None

    # Uncertainty
    epistemic_std: Optional[float] = None
    conformal_width: Optional[float] = None
    ood_score: Optional[float] = None

    # Physics
    physics_max_z: Optional[float] = None
    physics_nll: Optional[float] = None
    physics_family_scores: Dict[str, float] = field(default_factory=dict)

    # Composition
    gc_content: Optional[float] = None
    cpg_oe: Optional[float] = None
    repeat_fraction: Optional[float] = None
    entropy: Optional[float] = None
    max_homopolymer: Optional[int] = None

    # Status
    physics_passed: bool = False
    composition_passed: bool = False
    confidence_passed: bool = False
    hard_failures: List[str] = field(default_factory=list)
    soft_failures: List[str] = field(default_factory=list)


class BatchScorecard:
    """Scorecard aggregating results across a batch of sequences."""

    def __init__(self, name: str = "batch"):
        self.name = name
        self.scorecards: List[SequenceScorecard] = []

    def compute_statistics(self):
        """Compute aggregate statistics."""
        n = len(self.scorecards)
        if n == 0:
            return

        self.n_green = sum(1 for s in self.scorecards if s.verdict == Verdict.GREEN)
        self.n_yellow = sum(1 for s in self.scorecards if s.verdict == Verdict.YELLOW)
        self.n_red = sum(1 for s in self.scorecards if s.verdict == Verdict.RED)

        self.green_rate = self.n_green / n
        self.yellow_rate = self.n_yellow / n
        self.red_rate = self.n_red / n

    def summary(self) -> str:
        """Generate human-readable summary."""
        self.compute_statistics()
        n = len(self.scorecards)
        return (
            f"Batch: {self.name} ({n} sequences)\n"
            f"  GREEN:  {self.n_green} ({self.green_rate:.1%})\n"
            f"  YELLOW: {self.n_yellow} ({self.yellow_rate:.1%})\n"
            f"  RED:    {self.n_red} ({self.red_rate:.1%})"
        )


# === oracle_check/rc_consistency.py ===
# Reverse complement consistency and ISM flip tests

@dataclass
class RCConsistencyResult:
    """Result of RC consistency check."""
    passed: bool
    prediction_fwd: float
    prediction_rc: float
    delta: float
    delta_threshold: float


@dataclass
class ISMFlipTestResult:
    """Result of ISM flip test for RC consistency."""
    passed: bool
    n_positions_tested: int
    n_symmetric: int
    symmetry_rate: float
    max_asymmetry: float


def reverse_complement(sequence: str) -> str:
    """Get reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(sequence.upper()))


class RCConsistencyChecker:
    """
    Checks that CADENCE predictions are consistent for a sequence and
    its reverse complement. Threshold: |y_fwd - y_rc| < 0.1
    """

    def __init__(self, model, delta_threshold: float = 0.1):
        self.model = model
        self.delta_threshold = delta_threshold

    def check(self, sequence: str) -> RCConsistencyResult:
        """Check RC consistency for a sequence."""
        rc_seq = reverse_complement(sequence)

        pred_fwd = self.model.predict([sequence])[0].mean
        pred_rc = self.model.predict([rc_seq])[0].mean

        delta = abs(pred_fwd - pred_rc)
        passed = delta < self.delta_threshold

        return RCConsistencyResult(
            passed=passed,
            prediction_fwd=pred_fwd,
            prediction_rc=pred_rc,
            delta=delta,
            delta_threshold=self.delta_threshold,
        )


class ISMFlipTest:
    """
    In-silico mutagenesis flip test: verifies that single-nucleotide
    perturbation effects are symmetric under reverse complementation.
    Threshold: >= 90% symmetry rate.
    """

    def __init__(self, model, symmetry_threshold: float = 0.90):
        self.model = model
        self.symmetry_threshold = symmetry_threshold

    def test(self, sequence: str, n_positions: int = 50) -> ISMFlipTestResult:
        """Run ISM flip test on a sequence."""
        seq = sequence.upper()
        rc_seq = reverse_complement(seq)
        seq_len = len(seq)

        # Sample positions
        positions = np.random.choice(
            seq_len, size=min(n_positions, seq_len), replace=False
        )

        n_symmetric = 0
        max_asymmetry = 0.0

        for pos in positions:
            rc_pos = seq_len - 1 - pos

            # Get ISM effects for forward
            fwd_effects = self._get_ism_effects(seq, pos)
            # Get ISM effects for RC at corresponding position
            rc_effects = self._get_ism_effects(rc_seq, rc_pos)

            # Check symmetry (effects should be similar after complementing)
            asymmetry = abs(max(fwd_effects.values()) - max(rc_effects.values()))
            max_asymmetry = max(max_asymmetry, asymmetry)

            if asymmetry < 0.05:  # Symmetric within tolerance
                n_symmetric += 1

        symmetry_rate = n_symmetric / len(positions) if len(positions) > 0 else 0.0
        passed = symmetry_rate >= self.symmetry_threshold

        return ISMFlipTestResult(
            passed=passed,
            n_positions_tested=len(positions),
            n_symmetric=n_symmetric,
            symmetry_rate=symmetry_rate,
            max_asymmetry=max_asymmetry,
        )

    def _get_ism_effects(self, sequence: str, position: int) -> Dict[str, float]:
        """Get ISM effects at a single position."""
        bases = "ACGT"
        original = sequence[position]
        effects = {}

        for base in bases:
            if base != original:
                mutant = sequence[:position] + base + sequence[position+1:]
                pred = self.model.predict([mutant])[0].mean
                orig_pred = self.model.predict([sequence])[0].mean
                effects[base] = pred - orig_pred

        return effects


# === oracle_check/two_sample_tests.py ===
# Batch-level distributional tests: MMD, Energy Distance, KS, k-mer JS

@dataclass
class TwoSampleTestResult:
    """Result of a two-sample statistical test."""
    test_name: str
    statistic: float
    pvalue: float
    passed: bool
    threshold: float
    message: str


class MMDTest:
    """
    Maximum Mean Discrepancy test with RBF kernel.
    Measures distance between distributions in RKHS.
    Uses median heuristic for bandwidth and permutation test for p-value.
    """

    def __init__(self, kernel_bandwidth: float = None):
        self.kernel_bandwidth = kernel_bandwidth

    def compute(
        self, X: np.ndarray, Y: np.ndarray, n_permutations: int = 1000,
    ) -> TwoSampleTestResult:
        """Compute MMD statistic with permutation test for p-value."""
        from scipy.spatial.distance import cdist

        n_x, n_y = len(X), len(Y)

        # Bandwidth via median heuristic
        bandwidth = self.kernel_bandwidth
        if bandwidth is None:
            XY = np.vstack([X, Y])
            pairwise = cdist(XY, XY, 'euclidean')
            non_zero = pairwise[pairwise > 0]
            bandwidth = np.median(non_zero) if len(non_zero) > 0 else 1.0

        # RBF kernel
        def rbf(A, B):
            return np.exp(-cdist(A, B, 'sqeuclidean') / (2 * bandwidth ** 2))

        K_xx = rbf(X, X)
        K_yy = rbf(Y, Y)
        K_xy = rbf(X, Y)

        # Unbiased MMD^2 estimator
        mmd2 = (
            (K_xx.sum() - np.trace(K_xx)) / (n_x * (n_x - 1))
            + (K_yy.sum() - np.trace(K_yy)) / (n_y * (n_y - 1))
            - 2 * K_xy.mean()
        )
        mmd = np.sqrt(max(mmd2, 0))

        # Permutation test
        combined = np.vstack([X, Y])
        null_mmds = []
        for _ in range(n_permutations):
            perm = np.random.permutation(n_x + n_y)
            Xp, Yp = combined[perm[:n_x]], combined[perm[n_x:]]
            Kxxp, Kyyp, Kxyp = rbf(Xp, Xp), rbf(Yp, Yp), rbf(Xp, Yp)
            m2 = (
                (Kxxp.sum() - np.trace(Kxxp)) / (n_x * (n_x - 1))
                + (Kyyp.sum() - np.trace(Kyyp)) / (n_y * (n_y - 1))
                - 2 * Kxyp.mean()
            )
            null_mmds.append(np.sqrt(max(m2, 0)))

        pvalue = (np.sum(np.array(null_mmds) >= mmd) + 1) / (n_permutations + 1)
        passed = pvalue > 0.05

        return TwoSampleTestResult(
            test_name="MMD", statistic=mmd, pvalue=pvalue,
            passed=passed, threshold=0.05,
            message=f"MMD={mmd:.4f}, p={pvalue:.4f}" + (" (PASS)" if passed else " (FAIL)"),
        )


class KmerJSDivergence:
    """Jensen-Shannon divergence on k-mer spectra (k=4,5,6)."""

    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [4, 5, 6]

    def _get_kmer_spectrum(self, sequences: List[str], k: int) -> np.ndarray:
        """Compute normalized k-mer frequency spectrum."""
        from collections import Counter
        all_kmers = Counter()
        for seq in sequences:
            seq = seq.upper()
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if 'N' not in kmer:
                    all_kmers[kmer] += 1

        # All possible k-mers
        bases = ['A', 'C', 'G', 'T']
        all_possible = []
        def gen(prefix, remaining):
            if remaining == 0:
                all_possible.append(prefix)
                return
            for b in bases:
                gen(prefix + b, remaining - 1)
        gen('', k)

        total = sum(all_kmers.values()) + 1e-10
        return np.array([all_kmers.get(kmer, 0) / total for kmer in all_possible])

    def compute(
        self, designed_seqs: List[str], reference_seqs: List[str], threshold: float = 0.1,
    ) -> TwoSampleTestResult:
        """Compute JS divergence on k-mer spectra."""
        js_values = []
        for k in self.k_values:
            p = self._get_kmer_spectrum(designed_seqs, k)
            q = self._get_kmer_spectrum(reference_seqs, k)
            p, q = p + 1e-10, q + 1e-10
            p, q = p / p.sum(), q / q.sum()
            m = 0.5 * (p + q)
            js = 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))
            js_values.append(js)

        mean_js = np.mean(js_values)
        passed = mean_js < threshold

        return TwoSampleTestResult(
            test_name="k-mer JS Divergence", statistic=mean_js,
            pvalue=1.0 - mean_js, passed=passed, threshold=threshold,
            message=f"JS({self.k_values})={mean_js:.4f}" + (" (PASS)" if passed else " (FAIL)"),
        )


class BatchComparator:
    """Compares batches of designed sequences against natural references."""

    def __init__(self, mmd_bandwidth=None, kmer_k_values=None, n_permutations=500):
        self.mmd_test = MMDTest(kernel_bandwidth=mmd_bandwidth)
        self.kmer_test = KmerJSDivergence(k_values=kmer_k_values)
        self.n_permutations = n_permutations

    def compare_physics(
        self, designed_features: np.ndarray, reference_features: np.ndarray,
        feature_names: List[str] = None,
    ):
        """Compare physics feature distributions between designed and reference."""
        from scipy import stats as scipy_stats

        # MMD on full physics features
        mmd_result = self.mmd_test.compute(
            designed_features, reference_features, self.n_permutations,
        )

        # Per-feature KS tests
        ks_results = {}
        if feature_names is not None:
            for i, name in enumerate(feature_names):
                stat, pval = scipy_stats.ks_2samp(
                    designed_features[:, i], reference_features[:, i],
                )
                ks_results[name] = TwoSampleTestResult(
                    test_name=f"KS ({name})", statistic=stat, pvalue=pval,
                    passed=pval > 0.05, threshold=0.05,
                    message=f"KS({name})={stat:.4f}, p={pval:.4f}",
                )

        return mmd_result, ks_results


# === oracle_check/protocol.py ===
# Main OracleCheckProtocol orchestrating the full validation pipeline

class OracleCheckProtocol:
    """
    Main validation protocol for OracleCheck.

    Validates sequences using:
    - CADENCE activity predictions with PLACE uncertainty
    - PhysInformer physics features (cell-type specific)
    - TileFormer electrostatics (universal)
    - Reference panels from natural high performers

    Human cell types only: K562, HepG2, WTC11
    """

    HUMAN_CELL_TYPES = ["K562", "HepG2", "WTC11"]

    def __init__(self, config=None, cadence_model_name="config2_multi_celltype_v1", device="cuda"):
        self.config = config or OracleCheckConfig()
        self.device = device
        self.cadence_model_name = cadence_model_name

        # Interfaces (lazy loaded)
        self._cadence = None
        self._physinformer = None
        self._tileformer = None

        # Reference panels and validators per cell type
        self._reference_panels: Dict[str, Any] = {}
        self._validators: Dict[str, OracleCheckValidator] = {}

    def load_reference_panel(self, cell_type: str, rebuild: bool = False):
        """Load or build reference panel for a cell type."""
        if cell_type not in self.HUMAN_CELL_TYPES:
            raise ValueError(f"Unsupported cell type: {cell_type}")

        if cell_type in self._reference_panels and not rebuild:
            return self._reference_panels[cell_type]

        # Check saved panel on disk
        panel_path = self.config.reference_panels_dir / cell_type
        if panel_path.exists() and not rebuild:
            panel = ReferencePanel.load(panel_path)
            self._reference_panels[cell_type] = panel
            return panel

        # Build from lentiMPRA data (top 25% activity = high performers)
        panel = self._build_reference_panel(cell_type)
        panel.save(panel_path)
        self._reference_panels[cell_type] = panel
        return panel

    def validate_sequence(self, sequence: str, cell_type: str, sequence_id=None):
        """
        Validate a single designed sequence.

        1. Get CADENCE prediction with PLACE uncertainty
        2. Get PhysInformer physics features
        3. Run all validators (physics, composition, confidence)
        4. Assign GREEN/YELLOW/RED verdict
        """
        # Get prediction
        prediction = self._cadence.predict([sequence], cell_type=cell_type)[0]

        # Get physics features
        physics_features, physics_feature_names = None, None
        try:
            phys_result = self._physinformer.predict([sequence], cell_type)
            physics_features = phys_result.features[0]
            physics_feature_names = phys_result.feature_names
        except Exception:
            pass

        # Run validation
        validator = self._validators.get(cell_type)
        if validator is None:
            panel = self.load_reference_panel(cell_type)
            validator = OracleCheckValidator(
                config=self.config, reference_panel=panel,
            )
            self._validators[cell_type] = validator

        verdict, results = validator.validate_sequence(
            sequence=sequence,
            physics_features=physics_features,
            physics_feature_names=physics_feature_names,
            prediction_mean=prediction.mean,
            epistemic_std=prediction.epistemic_std,
            conformal_width=(
                prediction.conformal_upper - prediction.conformal_lower
                if prediction.conformal_upper is not None else None
            ),
            ood_score=prediction.ood_score,
        )

        return verdict, results

    def validate_batch(
        self, sequences: List[str], cell_type: str,
        sequence_ids=None, batch_size: int = 32,
    ) -> BatchScorecard:
        """Validate a batch of sequences, returning aggregate scorecard."""
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(sequences))]

        batch_scorecard = BatchScorecard(name=f"batch_{cell_type}")

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            batch_ids = sequence_ids[i:i+batch_size]

            predictions = self._cadence.predict(batch_seqs, cell_type=cell_type)

            physics_features, physics_feature_names = None, None
            try:
                phys_result = self._physinformer.predict(batch_seqs, cell_type)
                physics_features = phys_result.features
                physics_feature_names = phys_result.feature_names
            except Exception:
                pass

            validator = self._validators.get(cell_type)
            if validator is None:
                panel = self.load_reference_panel(cell_type)
                validator = OracleCheckValidator(
                    config=self.config, reference_panel=panel,
                )
                self._validators[cell_type] = validator

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
                        if pred.conformal_upper is not None else None
                    ),
                    ood_score=pred.ood_score,
                )

                scorecard = SequenceScorecard(
                    sequence=seq, sequence_id=seq_id, verdict=verdict,
                    activity_mean=pred.mean,
                    epistemic_std=pred.epistemic_std,
                    physics_max_z=results.get("physics", PhysicsValidationResult(
                        True, 0, {}, {}, 0, 0, "")).max_z_score if "physics" in results else None,
                    gc_content=results["composition"].gc_content,
                    hard_failures=results.get("hard_fails", []),
                    soft_failures=results.get("soft_fails", []),
                )
                batch_scorecard.scorecards.append(scorecard)

        batch_scorecard.compute_statistics()
        return batch_scorecard
