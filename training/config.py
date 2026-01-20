"""
FUSEMAP Training Configurations

Defines all 5 training configurations:
1. Single Cell Type Baseline
2. Multi-Cell-Type Human
3. Cross-Animal (Human + Drosophila)
4. Cross-Kingdom (Animal + Plant)
5. Universal Foundation Model
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ConfigurationType(Enum):
    SINGLE_CELLTYPE = "config1_single_celltype"
    MULTI_CELLTYPE_HUMAN = "config2_multi_celltype_human"
    CROSS_ANIMAL = "config3_cross_animal"
    CROSS_KINGDOM = "config4_cross_kingdom"
    UNIVERSAL = "config5_universal"


@dataclass
class DatasetInfo:
    """Information about a single dataset."""
    name: str
    path: str
    sequence_length: int
    num_outputs: int
    output_names: List[str]
    species: str
    kingdom: str
    cell_type: Optional[str] = None
    element_type: str = "promoter"  # promoter, enhancer, silencer

    # Sizes
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0

    # Validation scheme
    validation_scheme: str = "standard"  # standard, kfold, chromosome_holdout, special
    kfold_splits: int = 10
    holdout_chromosome: Optional[str] = None

    # Activity statistics (filled during preprocessing)
    activity_mean: Optional[List[float]] = None
    activity_std: Optional[List[float]] = None


# Dataset catalog
DATASET_CATALOG: Dict[str, DatasetInfo] = {
    # ENCODE4 Human datasets (230bp, 1 output)
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

    # DeepSTARR Drosophila (249bp, 2 outputs)
    # Pre-split by original authors (chr 2R test, chr 2L val)
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
        validation_scheme="chromosome_holdout",  # Pre-split train/val/test files
        holdout_chromosome="2R",  # Test chromosome (val is 2L)
    ),

    # DREAM Yeast (110bp, 1 output)
    # Large dataset with standard train/val/test splits
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
        validation_scheme="special",  # Standard splits but with weighted metrics
    ),

    # Plant datasets (170bp for Jores) - Two targets per species:
    # enrichment_leaf (tobacco leaf) and enrichment_proto (maize protoplast)
    "jores_arabidopsis": DatasetInfo(
        name="jores_arabidopsis",
        path="data/plant_data/jores2021/processed/arabidopsis",
        sequence_length=170,
        num_outputs=2,  # Two activity targets: leaf and proto
        output_names=["leaf", "proto"],
        species="arabidopsis",
        kingdom="plant",
        cell_type="dual_assay",
        element_type="promoter",
        train_size=12000,
        validation_scheme="standard",
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
        validation_scheme="standard",
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
        validation_scheme="standard",
    ),
}


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Base architecture - LegNet-style single kernel stem
    stem_channels: int = 64
    stem_kernel_size: int = 11
    # LegNet-style block channels
    block_channels: List[int] = field(default_factory=lambda: [80, 96, 112, 128])
    block_kernel: int = 9  # LegNet uses 9
    expand_ratio: int = 4

    # RC-equivariant stem (strand symmetry)
    use_rc_stem: bool = True  # Use reverse-complement equivariant stem

    # Optional: PWM-initialized multi-scale stem (alternative to RC stem)
    use_pwm_stem: bool = False
    pwm_stem_scales: List[int] = field(default_factory=lambda: [7, 11, 15])

    # Optional: Species-specific stems (for cross-species transfer learning)
    # Creates separate stem modules per species, shared backbone after
    use_species_stem: bool = False

    # Optional: Kingdom-specific stems (for cross-kingdom transfer learning)
    # Creates separate stem modules per kingdom (animal vs plant), shared backbone after
    use_kingdom_stem: bool = False

    # Optional: Cluster space (dilated convolutions for long-range patterns)
    use_cluster_space: bool = False
    cluster_dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 1])

    # Optional: Grammar layer (bidirectional GRU with FiLM)
    use_grammar: bool = False
    grammar_hidden: int = 128

    # Optional: MicroMotif processor (multi-scale motif density)
    use_micromotif: bool = False
    micromotif_windows: List[int] = field(default_factory=lambda: [5, 11, 21])

    # Optional: Motif correlator (low-rank bilinear pooling)
    use_motif_correlator: bool = False
    correlator_factors: int = 32
    correlator_rank: int = 8

    # Conditioning embeddings
    use_species_embedding: bool = False
    species_embed_dim: int = 16
    use_celltype_embedding: bool = False
    celltype_embed_dim: int = 32
    use_kingdom_embedding: bool = False
    kingdom_embed_dim: int = 8
    use_length_embedding: bool = True
    length_embed_dim: int = 16
    # Element-type conditioning (promoter vs enhancer vs silencer)
    use_element_type_embedding: bool = False
    element_type_embed_dim: int = 8

    # Head configuration
    head_hidden: int = 256
    use_uncertainty: bool = True  # False for MSE loss

    # Regularization
    dropout: float = 0.3  # Increased from 0.1 to reduce overfitting


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Basic training
    max_epochs: int = 50  # Reduced from 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Scheduler
    scheduler: str = "cosine"  # cosine, plateau, step, onecycle
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    # OneCycleLR specific
    onecycle_pct_start: float = 0.3  # Fraction of cycle to increase LR
    onecycle_div_factor: float = 25.0  # initial_lr = max_lr / div_factor (LegNet uses 25)

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    monitor: str = "val_pearson"  # metric to monitor
    mode: str = "max"  # max or min

    # Gradient handling
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1

    # Mixed precision
    use_amp: bool = True

    # Sampling
    sampling_temperature: float = 0.5
    samples_per_epoch: Optional[int] = None

    # Checkpointing
    save_top_k: int = 3
    checkpoint_dir: str = "checkpoints"

    # Logging
    log_every_n_steps: int = 100
    val_every_n_epochs: int = 1

    # Extreme value weighting (emphasize tails of distribution)
    use_extreme_weights: bool = True  # Weight extreme values more heavily
    extreme_alpha: float = 1.0  # Scaling factor (higher = more emphasis on extremes)
    extreme_beta: float = 2.0  # Power for z-score (2.0 = quadratic emphasis)

    # Extreme-aware sampling (oversample tail values) - DEPRECATED, use balanced instead
    use_extreme_sampling: bool = False  # Use extreme-aware sampler
    sampling_extreme_alpha: float = 0.5  # Sampling weight alpha (separate from loss alpha)

    # Balanced activity sampling (uniform across activity bins)
    use_balanced_sampling: bool = True  # Sample equally from all activity ranges
    balanced_sampling_bins: int = 10  # Number of activity bins (deciles by default)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Identification
    name: str
    config_type: ConfigurationType
    description: str

    # Data
    datasets: List[str]  # Dataset names from catalog
    target_sequence_length: int = 256  # Pad/crop to this length

    # Model and training
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Output
    output_dir: str = "results"
    seed: int = 42

    # Multi-phase training (for Config 5)
    training_phases: Optional[List[dict]] = None


# ============================================================================
# Configuration Presets
# ============================================================================

def get_config1_single_celltype(dataset_name: str, use_mse: bool = False) -> ExperimentConfig:
    """Configuration 1: Single Cell Type Baseline."""
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
    """
    Configuration 2: Multi-Cell-Type Human (LegNet-style hyperparams).

    Purpose: Learn shared human regulatory grammar with cell-type conditioning.
    - Combines K562, HepG2, WTC11 datasets (~295k sequences)
    - Cell-type embedding (32 dim) concatenated to sequence features
    - Shared convolutional backbone with cell-type-specific output heads
    """
    return ExperimentConfig(
        name="config2_multi_celltype_human",
        config_type=ConfigurationType.MULTI_CELLTYPE_HUMAN,
        description="Multi-cell-type human with shared backbone and cell-type conditioning",
        datasets=["encode4_k562", "encode4_hepg2", "encode4_wtc11"],
        target_sequence_length=256,
        model=ModelConfig(
            use_rc_stem=True,  # RC-equivariant stem
            use_species_embedding=False,
            use_celltype_embedding=True,
            celltype_embed_dim=32,
            use_length_embedding=True,
        ),
        training=TrainingConfig(
            max_epochs=100,
            batch_size=1024,
            learning_rate=0.01,  # LegNet-style
            weight_decay=0.1,    # LegNet-style
            scheduler="onecycle",
            onecycle_pct_start=0.3,
            sampling_temperature=0.5,
            patience=15,
        ),
    )


def get_config3_cross_animal() -> ExperimentConfig:
    """
    Configuration 3: Cross-Animal (Human + Drosophila).

    Purpose: Test whether animal regulatory logic transfers across 600M years of evolution.
    - Human ENCODE4 (~295k sequences) + DeepSTARR Drosophila (~352k sequences)
    - Species-specific stems (separate motif filter banks per organism)
    - Species embedding (16 dim) for conditioning
    - Shared physics-processing backbone (physics is conserved)
    - Species-specific output heads
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
    """
    Configuration 4: Cross-Kingdom (Animal + Plant).

    Purpose: Test universal regulatory principles across kingdoms.
    - ENCODE4 Human (~295k) + DeepSTARR Drosophila (~352k) + Arabidopsis (~15k) + Maize (~22k)
    - Kingdom embedding (8 dim) + species embedding (16 dim)
    - Kingdom-specific motif banks (separate stems per kingdom)
    - Shared physics backbone (maximally conserved)
    - Species-specific prediction heads

    Key hypothesis: Biophysical constraints are kingdom-invariant.
    """
    return ExperimentConfig(
        name="config4_cross_kingdom",
        config_type=ConfigurationType.CROSS_KINGDOM,
        description="Cross-kingdom transfer between animals and plants",
        datasets=[
            # Animals (~647k sequences)
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
    """
    Configuration 5: Universal Foundation Model.

    Purpose: Maximum generalization across all available data.
    - All animal data: ENCODE4 Human (~295k) + DeepSTARR (~352k) = ~647k
    - All plant data: Arabidopsis (~15k) + Maize (~22k) + Sorghum (~17k) = ~54k
    - Yeast data: ~6.7M (subsampled to 500k per epoch for balance)
    - Total: ~1.2M effective sequences per epoch

    Architecture:
    - Full multi-task model with universal physics encoder
    - Hierarchical taxonomy embeddings: Kingdom (8) + Species (16)
    - Element-type conditioning: promoter vs enhancer
    - Cell-type conditioning (32 dim)
    - Task-specific prediction heads per dataset

    Training phases:
    1. Pre-train physics encoder on all data
    2. Train species-specific heads with frozen physics
    3. End-to-end fine-tuning with low learning rate
    """
    return ExperimentConfig(
        name="config5_universal",
        config_type=ConfigurationType.UNIVERSAL,
        description="Universal foundation model across all species",
        datasets=[
            # Animals (~647k sequences)
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
            # Animals (~647k sequences)
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
