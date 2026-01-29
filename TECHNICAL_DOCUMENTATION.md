# FUSEMAP: Comprehensive Technical Documentation

## Table of Contents
1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [PhysInformer](#3-physinformer)
4. [TileFormer](#4-tileformer)
5. [PhysicsVAE](#5-physicsvae)
6. [PhysicsTransfer](#6-physicstransfer)
7. [CADENCE](#7-cadence)
8. [OracleCheck](#8-oraclecheck-validation-protocol)
9. [PhysicsInterpreter](#9-physicsinterpreter-attribution-analysis)
10. [Impact Applications](#10-impact-applications) (incl. Therapeutic Optimization, S2A)
11. [Data Pipeline](#11-data-pipeline)
12. [Training Framework](#12-training-framework)
13. [Physics Feature Specification](#13-physics-feature-specification)
14. [Electrostatics Module](#14-electrostatics-module)

---

## 1. Overview

**FUSEMAP** (Functional Sequence Modeling with Physics-Aware Predictions) is a comprehensive framework for DNA regulatory element analysis that integrates:

- **Physics-informed models** for biophysical property prediction
- **Sequence-to-activity models** for enhancer/promoter activity prediction
- **Generative models** for physics-conditioned sequence design
- **Transfer learning** for cross-species/cross-kingdom generalization

### 1.1 Core Components

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **PhysInformer** | Predict physics from sequence | DNA sequence | 500+ physics features |
| **TileFormer** | Predict electrostatics | DNA sequence | Electrostatic potentials (PSI) |
| **PhysicsVAE** | Generate sequences from physics | Physics features | DNA sequences |
| **PhysicsTransfer** | Cross-species transfer | Source model + target data | Transferred model |
| **CADENCE** | Predict activity from sequence | DNA sequence | Activity score |
| **CADENCE Pro** | Large-scale activity prediction | DNA sequence | Activity + uncertainty |
| **S2A** | Zero-shot universal activity | Physics features | Z-score / calibrated activity |
| **OracleCheck** | Validate designed sequences | Sequence | GREEN/YELLOW/RED verdict |
| **PhysicsInterpreter** | Attribution analysis | Predictions | Physics-mediated explanations |

### 1.2 Directory Structure

```
FUSEMAP/
├── physics/
│   ├── PhysInformer/     # Physics prediction model
│   ├── TileFormer/       # Electrostatics transformer
│   ├── PhysicsVAE/       # Physics-conditioned VAE
│   ├── PhysicsTransfer/  # Transfer learning framework
│   ├── analyses/         # Analysis pipelines
│   └── data/             # Physics datasets
├── models/
│   ├── CADENCE/          # Activity prediction model
│   └── cadence_pro/      # Large-scale transformer (CADENCE Pro)
├── training/             # Multi-species training
├── data/                 # Dataset storage
├── electrostatics/       # APBS integration
└── results/              # Output storage
```

---

## 2. System Architecture

### 2.1 Data Flow

```
                    ┌─────────────────┐
                    │  Raw Sequences  │
                    │  (FASTA/TSV)    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  PhysInformer   │           │   TileFormer    │
    │  (Physics)      │           │ (Electrostatics)│
    └────────┬────────┘           └────────┬────────┘
             │                             │
             └──────────────┬──────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  Physics Feature Vector │
              │  (521+ features)        │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │ PhysicsVAE│    │  CADENCE  │    │ Physics   │
   │ (Generate)│    │ (Activity)│    │ Transfer  │
   └───────────┘    └───────────┘    └───────────┘
```

### 2.2 Model Relationships

```
PhysInformer ──────► Physics Features ──────► PhysicsVAE (Generation)
     │                     │
     │                     ├──────► Activity Prediction (CADENCE auxiliary)
     │                     │
     │                     └──────► S2A Universal Head (Zero-shot transfer)
     │
     └──────► Activity Prediction (with auxiliary heads)

TileFormer ──────► Electrostatic Features ──────► Integrated into Physics

CADENCE ──────► Direct Sequence → Activity (baseline)

CADENCE Pro ──────► Sequence → Activity + Uncertainty (DREAM challenge)

S2A ──────► Physics → Universal Activity (zero-shot cross-species)
```

---

## 3. PhysInformer

### 3.1 Architecture Overview

PhysInformer is a physics-aware convolutional transformer that predicts biophysical properties from DNA sequences.

```
Input: DNA sequence (batch, seq_len) with values {0,1,2,3,4} = {A,C,G,T,N}

┌─────────────────────────────────────────────────────────────────┐
│                        PWMConvStem                               │
│  Embedding(5, 128) → Conv1d(128→192, k=11) → Conv1d(192→256, k=9)│
│  → Conv1d(256→256, k=7) with residual connections               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DualPathFeaturePyramid                        │
│  Local Path: Conv1d(256→192, k=9) → Conv1d(192→192, k=5)        │
│  Global Path: SimplifiedSSMLayer → Linear(256→192)              │
│  Output: Concatenate [local, global] → (batch, seq_len, 384)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PhysicsRouters                              │
│  Property-specific adapters with gating mechanisms:             │
│  - Thermodynamic Router (k=15, RC-aware)                        │
│  - Structural Router (k=11)                                      │
│  - Electrostatic Router (k=21)                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Property-Specific Heads                      │
│  ThermoHead: ΔH, ΔS, ΔG with uncertainty (mean, log_var)        │
│  ElectrostaticHead: Per-window PSI (22 windows × 6 values)      │
│  ScalarPropertyHeads: Individual physics features                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Auxiliary Activity Heads                    │
│  AuxiliaryHeadA: Sequence + Real Features → Activity            │
│  AuxiliaryHeadB: Real Features Only → Activity                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Details

#### 3.2.1 PWMConvStem

```python
class PWMConvStem(nn.Module):
    """PWM-style convolutional stem for motif detection"""

    Parameters:
        vocab_size: int = 5           # {A,C,G,T,N}
        hidden_dims: [128, 192, 256]  # Progressive channel expansion
        kernel_sizes: [11, 9, 7]      # Decreasing receptive fields
        dropout: float = 0.1

    Architecture:
        Embedding(5, 128)
        → Conv1d(128, 128, k=11) + BN + SiLU + Dropout + Residual
        → Conv1d(128, 192, k=9)  + BN + SiLU + Dropout + Residual
        → Conv1d(192, 256, k=7)  + BN + SiLU + Dropout + Residual

    Output: (batch, seq_len, 256)

    # Exact implementation (physics_aware_model.py:16-63):
    def __init__(self, vocab_size=5, hidden_dims=[128, 192, 256],
                 kernel_sizes=[11, 9, 7], dropout=0.1):
        self.embedding = nn.Embedding(vocab_size, hidden_dims[0])
        self.blocks = nn.ModuleList()
        for i, (out_dim, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            block = nn.ModuleDict({
                'conv': nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size//2),
                'activation': nn.SiLU(),
                'dropout': nn.Dropout(dropout),
                'res_proj': nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
            })
            self.blocks.append(block)

    def forward(self, x):
        x = self.embedding(x)           # (batch, seq_len) → (batch, seq_len, 128)
        x = x.transpose(1, 2)           # (batch, 128, seq_len)
        for block in self.blocks:
            residual = block['res_proj'](x)
            x = block['conv'](x)
            x = block['activation'](x)
            x = block['dropout'](x)
            x = x + residual            # Residual connection
        return x.transpose(1, 2)        # (batch, seq_len, 256)
```

#### 3.2.2 SimplifiedSSMLayer (State Space Model)

```python
class SimplifiedSSMLayer(nn.Module):
    """State Space Model for long-range dependencies"""

    Parameters:
        d_model: int = 256
        d_state: int = 16
        dropout: float = 0.1

    State equation:
        h_t = tanh(h_{t-1} @ A^T + B(x_t))
        y_t = C(h_t) + D * x_t

    Gating:
        gate = SiLU(Linear(x))
        output = gate * y + (1 - gate) * x
```

#### 3.2.3 PhysicsRouter

```python
class PhysicsRouter(nn.Module):
    """Routes features through property-specific adapters"""

    Parameters:
        d_input: int
        d_output: int
        kernel_size: int        # Property-specific receptive field
        property_name: str
        use_rc: bool = False    # Reverse-complement averaging

    Operations:
        window_agg: Conv1d with grouped convolutions
        gate: Linear → Sigmoid (sequence-level gating)
        Optional RC: (output + flip(output)) / 2
```

#### 3.2.4 ThermoHead

```python
class ThermoHead(nn.Module):
    """Thermodynamic property prediction (ΔH, ΔS, ΔG)"""

    Parameters:
        d_input: int
        d_hidden: int = 128
        temperature: float = 310.0  # Kelvin

    Outputs (with uncertainty):
        dH_mean, dH_log_var: Enthalpy
        dS_mean, dS_log_var: Entropy
        dG_mean, dG_log_var: Free energy

    Constraint: ΔG = ΔH - TΔS (can be enforced as auxiliary loss)
```

#### 3.2.5 ElectrostaticHead

```python
class ElectrostaticHead(nn.Module):
    """Electrostatic potential prediction"""

    Parameters:
        d_input: int
        d_hidden: int = 128
        n_windows: int = 22     # 20bp windows with stride 10

    Per-window outputs (6 values each):
        STD_MIN, STD_MAX, STD_MEAN: Standard ionic conditions
        ENH_MIN, ENH_MAX, ENH_MEAN: Enhanced ionic conditions

    Global outputs:
        mean_psi, log_var: Sequence-level electrostatic potential
```

#### 3.2.6 Auxiliary Activity Heads

```python
class AuxiliaryHeadA(nn.Module):
    """Activity prediction from Sequence + Real Features"""

    Own sequence encoder (separate from main model):
        Embedding(5, 64)
        → Conv1d(64, 128, k=11) → MaxPool(2)
        → Conv1d(128, 256, k=7) → MaxPool(2)
        → Conv1d(256, hidden_dim, k=5) → GlobalAvgPool

    Feature fusion:
        seq_proj: Linear(hidden_dim → hidden_dim//2)
        feat_proj: Linear(feature_dim → hidden_dim//2)
        gate: Sigmoid(feat_proj) * seq_proj
        fusion: MLP([hidden_dim] → n_activities)

class AuxiliaryHeadB(nn.Module):
    """Activity prediction from Real Features Only"""

    Architecture:
        Linear(feature_dim → 384) → LN → SiLU → Dropout
        → Linear(384 → 192) → LN → SiLU → Dropout
        → Linear(192 → 96) → SiLU
        → Linear(96 → n_activities)
```

### 3.3 Training Configuration

```python
# Default hyperparameters
vocab_size: 5
d_model: 256
d_expanded: 384
seq_len: 230 (K562/HepG2/WTC11) or 249 (S2)
dropout: 0.1
temperature: 310.0 K
n_electrostatic_windows: 22
n_descriptor_features: ~521 (after filtering)

# Optimizer
AdamW, lr=1e-4, weight_decay=0.01
CosineAnnealingLR with warmup

# Loss weights
physics_loss_weight: 1.0
activity_loss_weight: 0.5
thermodynamic_consistency_weight: 0.1
```

### 3.4 File Locations

```
physics/PhysInformer/
├── physics_aware_model.py   # Main architecture (34KB)
├── train.py                 # Training script (45KB)
├── dataset.py               # Data loading (17KB)
├── metrics.py               # Evaluation (27KB)
├── inference_physics.py     # Inference (9KB)
└── runs/                    # Checkpoints per cell type
```

---

## 4. TileFormer

### 4.1 Architecture Overview

TileFormer is a transformer-based model for predicting electrostatic potentials from DNA sequences.

```
Input: DNA sequence (batch, seq_len) with values {0,1,2,3,4}

┌─────────────────────────────────────────────────────────────────┐
│                    NucleotideEmbedding                           │
│  Embedding(5, 256) * sqrt(256)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PositionalEncoding                            │
│  Sinusoidal PE: sin/cos with 10000 base frequency               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TileFormerBlocks × 6                           │
│  MultiheadAttention(256, 8 heads) + LayerNorm                   │
│  FeedForward(256 → 1024 → 256) + GELU + LayerNorm               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GlobalPool                                 │
│  Mean over sequence → Linear(256) → GELU → Dropout              │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────┐           ┌─────────────────────┐
│      PSI Head       │           │  Uncertainty Head   │
│ Linear(256→128→6)   │           │ Linear(256→128→6)   │
│ 6 outputs:          │           │ Softplus activation │
│ STD/ENH × min/max/  │           │                     │
│ mean                │           │                     │
└─────────────────────┘           └─────────────────────┘
```

### 4.2 Component Details

#### 4.2.1 TileFormerBlock

```python
class TileFormerBlock(nn.Module):
    Parameters:
        d_model: int = 256
        n_heads: int = 8
        d_ff: int = 1024
        dropout: float = 0.1

    Architecture:
        x → MultiheadAttention(x, x, x) → Dropout → + x → LayerNorm
        → FeedForward → Dropout → + x → LayerNorm

    FeedForward:
        Linear(256 → 1024) → GELU → Dropout → Linear(1024 → 256)
```

#### 4.2.2 TileFormerWithMetadata

```python
class TileFormerWithMetadata(TileFormer):
    """Extended TileFormer with sequence metadata integration"""

    Additional inputs:
        metadata: (batch, 3)  # gc_content, cpg_density, minor_groove_score

    Metadata processing:
        Linear(3 → 64) → GELU → Linear(64 → 256)

    Feature fusion:
        concat([sequence_features, metadata_features])
        → Linear(512 → 256) → GELU → Dropout
```

### 4.3 Electrostatics Integration

TileFormer predictions are calibrated against APBS (Adaptive Poisson-Boltzmann Solver) calculations:

```
APBS Calculation Pipeline:
1. DNA sequence → 3D structure (NAB/tleap)
2. Structure → PQR file (add charges/radii)
3. PQR → APBS solver → Electrostatic potential grid
4. Grid → Feature extraction (min/max/mean per region)

Calibration:
- GC content series (0-100% in 10% steps)
- CpG context variants (low/med/high)
- Minor groove width variants (narrow/wide)
- Mixed structures for validation
```

### 4.4 Output Specification

```python
outputs = {
    'psi': torch.Tensor,        # (batch, 6) electrostatic potentials
    'uncertainty': torch.Tensor  # (batch, 6) prediction uncertainties
}

# PSI indices:
# 0: STD_MIN  - Standard ionic, minimum potential
# 1: STD_MAX  - Standard ionic, maximum potential
# 2: STD_MEAN - Standard ionic, mean potential
# 3: ENH_MIN  - Enhanced ionic, minimum potential
# 4: ENH_MAX  - Enhanced ionic, maximum potential
# 5: ENH_MEAN - Enhanced ionic, mean potential
```

### 4.5 File Locations

```
physics/TileFormer/
├── models/
│   └── tileformer_architecture.py  # Core model (6KB)
├── electrostatics/
│   ├── gpu_parallel_abps.py        # APBS runner
│   └── tleap_abps_processor.py     # PDB preparation
├── train_orchestrator.py            # Training (22KB)
├── complete_pipeline.py             # Full pipeline (56KB)
└── configs/config.yaml              # Configuration
```

---

## 5. PhysicsVAE

### 5.1 Architecture Overview

PhysicsVAE is a physics-conditioned Variational Autoencoder for generating DNA sequences with target biophysical properties.

```
ENCODER PATH:
                    ┌─────────────────────────────────────────┐
                    │            SequenceEncoder               │
DNA sequence ──────►│ Conv(4→64, k=9) → Conv(64→128, k=7)     │
(batch, 230)        │ → Conv(128→256, k=5) → GlobalAvgPool    │
                    │ → Linear(256→128) for μ                  │
                    │ → Linear(256→128) for log_σ              │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                                    z ~ N(μ, σ²)
                                    (batch, 128)

CONDITIONING PATH:
                    ┌─────────────────────────────────────────┐
Physics features ──►│           PhysicsEncoder                 │
(batch, 521)        │ Linear(521→256) → LN → GELU → Dropout   │
                    │ → Linear(256→128) → LN → GELU → Dropout │
                    │ → Linear(128→64)                         │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                                   z_physics
                                   (batch, 64)

DECODER PATH:
                    ┌─────────────────────────────────────────┐
[z; z_physics] ────►│           SequenceDecoder                │
(batch, 192)        │ Linear(192 → 230×192)                   │
                    │ → Reshape(batch, 230, 192)              │
                    │ → PositionalEncoding                     │
                    │ → TransformerEncoder(192, 4 heads, 4 layers)│
                    │ → Linear(192→4)                          │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                              Nucleotide logits
                              (batch, 230, 4)
```

### 5.2 Component Details

#### 5.2.1 SequenceEncoder

```python
class SequenceEncoder(nn.Module):
    """
    Architecture per spec:
    - Conv stack: 4→64 (k=9), 64→128 (k=7), 128→256 (k=5)
    - Global pooling
    - Outputs: μ_z, log_σ_z with latent_dim=128
    """

    Parameters:
        seq_length: int = 200
        latent_dim: int = 128
        dropout: float = 0.1

    Layers:
        conv1: Conv1d(4, 64, k=9, padding=4) + BN
        conv2: Conv1d(64, 128, k=7, padding=3) + BN
        conv3: Conv1d(128, 256, k=5, padding=2) + BN
        Each followed by: GELU → Dropout → MaxPool(2)

        fc_mu: Linear(256, 128)
        fc_logvar: Linear(256, 128)

    Input: One-hot encoded (batch, 4, seq_len)
    Output: μ (batch, 128), log_σ² (batch, 128)
```

#### 5.2.2 PhysicsEncoder

```python
class PhysicsEncoder(nn.Module):
    """Dense network: n_physics → 256 → 128 → 64"""

    Parameters:
        n_physics_features: int = 521
        hidden_dims: [256, 128]
        output_dim: int = 64
        dropout: float = 0.1

    Architecture:
        Linear(521 → 256) → LayerNorm → GELU → Dropout
        → Linear(256 → 128) → LayerNorm → GELU → Dropout
        → Linear(128 → 64)

    Output: z_physics (batch, 64)
```

#### 5.2.3 SequenceDecoder

```python
class SequenceDecoder(nn.Module):
    """
    Architecture per spec:
    - Concatenate z (128) + z_physics (64) = 192 combined dimension
    - Parallel transformer decoder (dim 192, 4 layers, 4 heads)
    - 1×1 conv (Linear) to nucleotide probabilities
    """

    Parameters:
        seq_length: int = 200
        latent_dim: int = 128
        physics_dim: int = 64
        n_heads: int = 4
        n_layers: int = 4
        dropout: float = 0.1

    hidden_dim = latent_dim + physics_dim = 192

    Architecture:
        latent_to_seq: Linear(192 → seq_length × 192)
        pos_encoding: Sinusoidal positional encoding
        transformer: TransformerEncoder(d_model=192, nhead=4, num_layers=4)
        output_proj: Linear(192 → 4)
```

### 5.3 Loss Function

```python
L = L_recon + β·L_KL + λ·L_physics

Where:
    L_recon = CrossEntropy(logits, target_sequence)  # Per-position
    L_KL = -0.5 * Σ(1 + log_σ² - μ² - σ²)           # KL divergence
    L_physics = MSE(predicted_physics, target_physics)  # Optional

Default weights (per spec):
    β = 0.001  (weak KL regularization)
    λ = 0.1    (physics consistency)
```

### 5.4 Generation Modes

```python
# Mode 1: Generate from target physics
sequences = model.generate(
    physics=target_physics,  # (batch, n_physics)
    n_samples=10,
    temperature=1.0
)

# Mode 2: Interpolate between sequences
sequences = model.interpolate(
    sequence1, sequence2,
    physics1, physics2,
    n_steps=10
)

# Mode 3: Random sampling
sequences = generate_random_physics_samples(
    model, n_samples=100, temperature=1.0
)
```

### 5.5 File Locations

```
physics/PhysicsVAE/
├── models/
│   ├── physics_vae.py    # Main architecture (15KB)
│   └── losses.py         # Loss functions (8KB)
├── data/
│   └── dataset.py        # Data loading (8KB)
├── train.py              # Training script (25KB)
├── generate.py           # Generation script (7KB)
└── runs/                 # Checkpoints per cell type
```

---

## 6. PhysicsTransfer

### 6.1 Overview

PhysicsTransfer implements cross-species and cross-kingdom transfer learning using physics as a universal bridge.

```
Source Species (e.g., Human)          Target Species (e.g., Plant)
        │                                       │
        ▼                                       ▼
┌───────────────┐                     ┌───────────────┐
│ Source Model  │                     │  Few-shot     │
│ (trained)     │                     │  Target Data  │
└───────┬───────┘                     └───────┬───────┘
        │                                     │
        ▼                                     ▼
┌───────────────────────────────────────────────────┐
│              Physics Feature Space                 │
│  (Thermodynamics, Mechanics, Electrostatics)      │
│              [UNIVERSAL BRIDGE]                    │
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────┐
│ Target Model  │
│ (transferred) │
└───────────────┘
```

### 6.2 Transfer Protocols

#### Protocol 1: Zero-Shot Transfer
```python
# Use physics features only (no species-specific motifs)
prediction = physics_probe(source_model.extract_physics(target_sequence))
```

#### Protocol 2: Few-Shot Fine-Tuning
```python
# Freeze physics encoder, fine-tune prediction head
for param in model.physics_encoder.parameters():
    param.requires_grad = False

# Train on target species data (1K-10K samples)
```

#### Protocol 3: Multi-Species Joint Training
```python
# Shared physics encoder, species-specific heads
class MultiSpeciesModel:
    physics_encoder: Shared across all species
    species_heads: {
        'human': ActivityHead,
        'drosophila': ActivityHead,
        'arabidopsis': ActivityHead,
        ...
    }
```

### 6.3 Dataset Configuration

```python
DATASETS = {
    'K562': DatasetConfig(
        path='lentiMPRA_data/K562/splits',
        species='human', kingdom='animal',
        seq_length=230, output_dim=1
    ),
    'S2': DatasetConfig(
        path='S2_data/splits',
        species='drosophila', kingdom='animal',
        seq_length=249, output_dim=2  # Dev, Hk
    ),
    'Arabidopsis': DatasetConfig(
        path='plant_data/arabidopsis',
        species='arabidopsis', kingdom='plant',
        seq_length=170, output_dim=1
    ),
    ...
}
```

### 6.4 Evaluation Metrics

```python
# Cross-species generalization
metrics = {
    'zero_shot_r': pearson_correlation,      # Without fine-tuning
    'few_shot_r_1k': pearson_after_1k,       # With 1K target samples
    'few_shot_r_5k': pearson_after_5k,       # With 5K target samples
    'transfer_gain': (few_shot - baseline) / baseline
}
```

### 6.5 File Locations

```
physics/PhysicsTransfer/
├── run_transfer.py     # Main experiment runner (8KB)
├── config.py           # Dataset configuration (10KB)
├── data_loader.py      # Multi-dataset loading (11KB)
├── protocols.py        # Transfer protocols (19KB)
├── evaluation.py       # Cross-species evaluation (14KB)
└── physics_probe.py    # Physics probing (11KB)
```

---

## 7. CADENCE

### 7.1 Architecture Overview

**CADENCE** is the core sequence-to-expression model based on LegNet with optional advanced modules:

- **RC-equivariant stem** (optional) for strand symmetry
- **ClusterSpace** for dilated convolutions capturing long-range patterns
- **Grammar layer** (BiGRU + FiLM) for sequence syntax
- **MicroMotif** for multi-scale density processing
- **Motif correlator** for low-rank bilinear pooling
- **PLACE** for post-hoc uncertainty estimation (epistemic + aleatoric)

```
Input: One-hot DNA sequence (batch, 4, seq_len)

┌─────────────────────────────────────────────────────────────────┐
│                          Stem                                    │
│  Standard: LocalBlock(4→64, k=11)                               │
│  Optional: RCEquivariantStem(4→64, k=11) for strand symmetry    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Main Blocks × 4                              │
│  For each block:                                                 │
│    ResidualConcat(EffBlock(in_ch, k=9, resize=4))               │
│    → LocalBlock(in_ch×2 → out_ch, k=9)                          │
│    → MaxPool1d(2)                                                │
│                                                                  │
│  Channel progression: 64 → 80 → 96 → 112 → 128                  │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        [ClusterSpace]  [Grammar]     [MicroMotif]
         (optional)    (optional)     (optional)
              │               │               │
              └───────────────┼───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Mapper                                   │
│  BatchNorm1d(128) → Conv1d(128→256, k=1)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Adaptive Pooling + Head                       │
│  AdaptiveAvgPool1d(1) → squeeze                                 │
│  → Linear(256→256) → BatchNorm → SiLU → Linear(256→num_outputs) │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PLACE Uncertainty (Post-hoc)                   │
│  Epistemic: Last-layer Laplace posterior approximation          │
│  Aleatoric: Local adaptive conformal prediction                 │
│  Uses calibration set, doesn't modify training                  │
└─────────────────────────────────────────────────────────────────┘
```

**Implemented modules** (in `models/CADENCE/`):
- `cadence.py`: Main model with LegNet backbone + optional modules
- `cluster_space.py`: Dilated convolutions with residual connections
- `grammar_layer.py`: Bidirectional GRU with FiLM conditioning
- `micro_motif.py`: Multi-scale density processing
- `motif_correlator.py`: Low-rank bilinear pooling
- `place_uncertainty.py`: PLACE post-hoc uncertainty estimation
- `pwm_stem.py`: PWM-initialized stem (planned)
- `dar_readout.py`: Distance-aware readout (planned)

### 7.2 Core Components

#### 7.2.1 SELayer (Squeeze-and-Excitation)

```python
# File: cadence.py:30-46
class SELayer(nn.Module):
    """Channel attention mechanism - global context for channel weighting"""

    def __init__(self, inp, reduction=4):
        self.fc = nn.Sequential(
            nn.Linear(inp, int(inp // reduction)),
            nn.SiLU(),
            nn.Linear(int(inp // reduction), inp),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # Global average pool
        y = self.fc(y).view(b, c, 1)       # Channel weights
        return x * y                        # Scale input
```

#### 7.2.2 EffBlock (EfficientNet-style)

```python
# File: cadence.py:49-75
class EffBlock(nn.Module):
    """Inverted residual block with SE attention"""

    def __init__(self, in_ch, ks, resize_factor, activation, out_ch=None, se_reduction=None):
        self.inner_dim = in_ch * resize_factor
        self.block = nn.Sequential(
            # Expand
            nn.Conv1d(in_ch, self.inner_dim, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(self.inner_dim),
            activation(),
            # Depthwise
            nn.Conv1d(self.inner_dim, self.inner_dim, kernel_size=ks,
                      groups=self.inner_dim, padding='same', bias=False),
            nn.BatchNorm1d(self.inner_dim),
            activation(),
            # Squeeze-and-Excitation
            SELayer(self.inner_dim, reduction=resize_factor),
            # Project back
            nn.Conv1d(self.inner_dim, in_ch, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(in_ch),
            activation(),
        )
```

#### 7.2.3 LocalBlock

```python
# File: cadence.py:78-94
class LocalBlock(nn.Module):
    """Local convolution block with batch normalization"""

    def __init__(self, in_ch, ks, activation, out_ch=None):
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch or in_ch, kernel_size=ks, padding='same', bias=False),
            nn.BatchNorm1d(out_ch or in_ch),
            activation()
        )

    def forward(self, x):
        return self.block(x)
```

#### 7.2.4 ResidualConcat

```python
# File: cadence.py:97-105
class ResidualConcat(nn.Module):
    """Apply function and concatenate with input (doubles channels)"""

    def __init__(self, fn):
        self.fn = fn

    def forward(self, x, **kwargs):
        return torch.concat([self.fn(x, **kwargs), x], dim=1)  # [EffBlock(x), x]
```

#### 7.2.5 RCEquivariantStem

```python
# File: cadence.py:209-245
class RCEquivariantStem(nn.Module):
    """Reverse-complement equivariant stem - encodes DNA strand symmetry"""

    def __init__(self, out_channels=64, kernel_size=11, in_channels=4):
        n_filters = out_channels // 2
        self.conv = nn.Conv1d(in_channels, n_filters, kernel_size, padding='same', bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU()

    def _revcomp_weight(self, weight):
        """Flip kernel and swap A<->T, G<->C for reverse complement"""
        flipped = torch.flip(weight, dims=[2])
        swapped = flipped[:, [3, 2, 1, 0], :]  # A(0)↔T(3), C(1)↔G(2)
        return swapped

    def forward(self, x):
        fwd = self.conv(x)
        rc_weight = self._revcomp_weight(self.conv.weight)
        rev = F.conv1d(x, rc_weight, padding='same')

        sym = fwd + rev    # Strand-symmetric (palindromic motifs)
        asym = fwd - rev   # Strand-asymmetric (directional motifs)

        return self.activation(self.bn(torch.cat([sym, asym], dim=1)))
```

### 7.3 Optional Modules

#### 7.3.1 ClusterSpace

```python
class ClusterSpace(nn.Module):
    """Dilated convolutions for long-range patterns"""

    Parameters:
        dilations: [1, 2, 4, 1]

    Architecture:
        For each dilation d:
            Conv1d(in_ch, in_ch, k=3, dilation=d) → BN → SiLU
        Aggregate with residual connections
```

#### 7.3.2 GrammarLayer

```python
class LightweightGrammarLayer(nn.Module):
    """Bidirectional GRU with FiLM conditioning"""

    Architecture:
        BiGRU(in_ch → hidden) → FiLM modulation → Linear → Residual
```

#### 7.3.3 MicroMotifProcessor

```python
class MicroMotifProcessor(nn.Module):
    """Multi-scale motif density"""

    Parameters:
        window_sizes: [5, 11, 21]

    For each window size:
        Conv1d(in_ch, in_ch//len(windows), k=window)
    Concatenate all scales
```

#### 7.3.4 LowRankMotifCorrelator

```python
class LowRankMotifCorrelator(nn.Module):
    """Low-rank bilinear pooling for pairwise interactions"""

    Parameters:
        n_factors: int = 32
        rank: int = 8

    Architecture:
        Project to factors: Linear(in_ch → n_factors)
        Outer product: factors @ factors.T
        Low-rank approximation: SVD truncation
```

#### 7.3.5 PLACE Uncertainty

```python
# File: place_uncertainty.py
class PLACEUncertainty:
    """Post-hoc Laplace And Conformal Estimation for uncertainty quantification."""

    Parameters:
        feature_dim: int = 256      # Penultimate layer dimension
        lambda_reg: float = 1e-3    # Ridge regularization for Laplace
        n_neighbors: int = 200      # Neighbors for local conformal
        temperature: float = 1.0    # Neighbor weighting temperature
        alpha: float = 0.1          # Significance level (0.1 = 90% coverage)

    # Epistemic Uncertainty (Last-Layer Laplace):
    Posterior covariance: Σ = (λI + σ⁻²ΦᵀΦ)⁻¹
    Variance per input: V_epi(x) = φ(x)ᵀ Σ φ(x)

    # Aleatoric Uncertainty (Local Adaptive Conformal):
    Find K nearest neighbors in feature space
    Weight by distance: w_j ∝ exp(-||f - f_j||²/τ)
    Quantile of weighted residuals: q_{1-α}(x)
    Interval: [μ̂(x) - q_{1-α}(x), μ̂(x) + q_{1-α}(x)]

    Outputs:
        epistemic: σ_epi(x) = √V_epi(x)
        aleatoric: σ_alea(x) ≈ q_{0.84}(x)
        interval: Conformal prediction interval
```

### 7.4 Configuration

```python
@dataclass
class CADENCEConfig:
    # Core LegNet parameters
    in_ch: int = 4
    stem_ch: int = 64
    stem_ks: int = 11
    ef_ks: int = 9
    ef_block_sizes: [80, 96, 112, 128]
    pool_sizes: [2, 2, 2, 2]
    resize_factor: int = 4

    # Optional modules
    use_rc_stem: bool = False
    use_cluster_space: bool = False
    use_grammar: bool = False
    use_micromotif: bool = False
    use_motif_correlator: bool = False

    # Output
    num_outputs: int = 1
```

### 7.5 File Locations

```
models/CADENCE/
├── cadence.py              # Main architecture (24KB)
├── cluster_space.py        # Dilated convolutions (3KB)
├── grammar_layer.py        # BiGRU + FiLM (4KB)
├── micro_motif.py          # Multi-scale density (2KB)
├── motif_correlator.py     # Bilinear pooling (3KB)
├── pwm_stem.py             # PWM-initialized stem (3KB)
├── dar_readout.py          # Distance-aware readout (12KB)
├── place_uncertainty.py    # PLACE post-hoc uncertainty (17KB)
└── ARCHITECTURE_COMPARISON.md  # Documentation (12KB)
```

---

## 8. OracleCheck: Validation Protocol

> **Status: IMPLEMENTED** - Available in `applications/utils/`

OracleCheck is a novel purely in-silico validation program for oracles in sequence design loops, providing comprehensive naturality evaluation.

### 8.1 Reference Panels

```
Natural High-Performers: Top quartile of MPRA activity sequences
- Physics features from PhysInformer (by family)
- MicroMotif statistics (run_len_max, run_sum, center_of_mass, flank_asym)
- Composition metrics (GC, CpG O/E, repeats/entropy)
- Chromatin proxies (ATAC/H3K27ac/CTCF auxiliary heads)

Background Natural: Random genomic tiles matched by GC and length
Training Index: kNN structure in CADENCE features for OOD checks
```

### 8.2 Validation Checks

```python
# Physics Conformity
z_f = (p̂_f - μ_f) / σ_f           # Per-family z-scores
NLL_phys(s) = -Σ log π_nat,family(p̂_family)
Pass: all families |z| ≤ 2.5, NLL_phys < p95 of natural

# Micro-Syntax Validation
# Per TF class: run_len_max, run_sum, skew in adaptive core span
Pass: each class within [p5, p95] or NLL below threshold

# Composition and Repeat Hygiene
# GC within envelope, CpG O/E within natural range
# No repeat explosion, Shannon entropy not in bottom p5

# CADENCE-Specific Checks
# Panel agreement: IQR across seeded oracles < threshold
# Uncertainty: σ_epi < P90, conformal width < threshold
# OOD: kNN/Mahalanobis distance ≤ P95 of training
```

### 8.3 Verdicts

```
GREEN:  All checks pass
YELLOW: Minor drift (≤1 soft failure)
RED:    Any hard failure
        - Physics z-scores > 4.0
        - Repeat fraction > 0.3
        - OOD flag triggered
        - >5 syntax violations
```

---

## 9. PhysicsInterpreter: Attribution Analysis

> **Status: IMPLEMENTED** - Available in `physics/PhysicsInterpreter/`

PhysicsInterpreter decomposes model predictions through the physics pathway for mechanistic interpretability.

### 9.1 Attribution Methods

```python
# Integrated Gradients (sequence attribution)
IG(x)_i = (x_i - x'_i) × ∫₀¹ (∂F(x' + α(x - x'))/∂x_i) dα
# Approximated using 50 interpolation steps
# Output: (length, 4) importance per nucleotide

# Physics Feature Attribution
# Via trained linear probe: physics features → activity
# Contribution = weight × feature value
# Grouped by family (thermo, mechanics, bending, entropy, structural, electro)

# Mediation Analysis
Total effect:    sequence → activity
Indirect effect: sequence → physics → activity (via PhysInformer + probe)
Direct effect:   activity | sequence, controlling for physics
Mediation proportion = indirect_R² / total_R²
```

### 9.2 Physics-Activity Landscape Analysis

```python
# Per-feature correlations with activity
# Multivariate: Elastic Net (α=0.01, L1 ratio=0.5)
# Feature importance rankings and physics→activity R²
# SHAP values for per-sequence explanations
```

---

## 10. Impact Applications

> **Status: IMPLEMENTED** - Available in `applications/`

### 10.1 Disease Variant Interpretation Pipeline

```
Step 1: Extract flanking sequence ±115bp around variant
Step 2: CADENCE prediction (Activity_ref, Activity_alt, ΔActivity)
Step 3: PhysInformer analysis (ΔThermodynamics, ΔMechanics, ΔStructural)
Step 4: PhysicsInterpreter decomposition (direct, physics-mediated, motif effects)
Step 5: Report with mechanistic hypothesis and clinical relevance

Validation: ClinVar (~5K variants), GTEx eQTL (~50K), GWAS (~10K SNPs)
```

### 10.2 Therapeutic Enhancer Design

```
Example: Liver-targeted AAV gene therapy enhancers

Requirements:
- High HepG2 activity, low K562/WTC11
- <250bp for AAV packaging
- No immunogenic sequences
- Stable across integration sites

Protocol:
1. Identify top 100 natural liver-specific enhancers
2. Constrained optimization (physics envelope, required HNF4A/FOXA)
3. Generate candidates via optimization + PhysicsVAE
4. Filter through OracleCheck (GREEN/YELLOW only)
5. Rank by specificity with diversity filter
```

### 10.3 Therapeutic Optimization Methods

Five optimization methods are implemented for cell-type-specific enhancer design:

| Method | Type | Description |
|--------|------|-------------|
| **ISM_target** | Gradient-free | In-silico mutagenesis with cell-type targeting |
| **EMOO** | Evolutionary | Evolutionary multi-objective optimization |
| **HMCPP** | MCMC | Hamiltonian Monte Carlo proxy prediction |
| **PINCSD** | Physics-guided | Physics-informed neural combinatorial design |
| **PVGG** | Generative | Proxy-guided variational generation |

```python
# ISM_target Pipeline (Best Performance)
class ISMTargetOptimizer:
    def __init__(self, cadence_models: Dict[str, CADENCEModel]):
        self.models = cadence_models  # K562, HepG2, WTC11

    def optimize(self, seed_sequence: str, target_cell: str) -> OptimizedSequence:
        # 1. Compute ISM scores for each position
        ism_scores = self._compute_ism(seed_sequence)

        # 2. Apply mutations that increase target activity
        mutated = self._apply_beneficial_mutations(seed_sequence, ism_scores)

        # 3. Verify specificity constraint
        predictions = {cell: model(mutated) for cell, model in self.models.items()}
        specificity = predictions[target_cell] - max(predictions[other] for other in predictions if other != target_cell)

        return OptimizedSequence(sequence=mutated, specificity=specificity)
```

**Location:** `applications/therapeutic_enhancer_pipeline.py`

---

## 10.4 S2A: Universal Sequence-to-Activity Framework

> **Status: IMPLEMENTED** - Available in `physics/S2A/`

S2A (Sequence-to-Activity) enables zero-shot activity prediction across species using universal physics features.

### Architecture

```
DNA Sequence (any species)
         │
         ▼
┌─────────────────────────┐
│     PhysInformer        │  (Seq → 521 physics features)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Universal Feature       │  Extract ~263 universal features
│ Extractor               │  (exclude PWM - species-specific)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   UniversalS2AHead      │  Ridge/ElasticNet/MLP
│   (trained on N-1 spp)  │  Z-score normalized
└───────────┬─────────────┘
            │
      ┌─────┴─────┬─────────────┐
      ▼           ▼             ▼
 ┌────────┐ ┌──────────┐ ┌─────────┐
 │Z-score │ │Calibrated│ │ Ranking │
 └────────┘ └──────────┘ └─────────┘
```

### Core Components

```python
# Configuration
@dataclass
class S2AConfig:
    universal_prefixes = ['thermo_', 'stiff_', 'bend_', 'entropy_', 'advanced_']
    excluded_prefixes = ['pwm_']  # Species-specific, exclude
    head_type: str = 'ridge'       # 'ridge', 'elastic_net', 'mlp'
    output_mode: str = 'zscore'    # 'zscore', 'calibrated', 'ranking'

# Universal Feature Extractor
class UniversalFeatureExtractor:
    """Extract ~263 universal physics features (exclude PWM)"""
    def extract(self, physics_features: np.ndarray) -> np.ndarray:
        universal_mask = self._get_universal_mask()
        return physics_features[:, universal_mask]

# Universal S2A Head
class UniversalS2AHead:
    """Adaptive head trained on multiple species"""
    def __init__(self, head_type='ridge', alpha=1.0):
        self.model = Ridge(alpha=alpha) if head_type == 'ridge' else ...

    def fit(self, X: np.ndarray, y_zscore: np.ndarray):
        """Train on z-scored activities from multiple species"""
        self.model.fit(X, y_zscore)

    def predict_zscore(self, X: np.ndarray) -> np.ndarray:
        """Output relative z-score"""
        return self.model.predict(X)

# Calibration
class AffineCalibrator:
    """Learn y = α * z + β from labeled samples"""
    def calibrate(self, z_pred: np.ndarray, y_true: np.ndarray, n_samples=50):
        # Fit affine transformation
        self.alpha, self.beta = np.polyfit(z_pred[:n_samples], y_true[:n_samples], 1)
```

### Training Protocol

```python
# Leave-one-out evaluation
trainer = UniversalS2ATrainer(config)

for holdout in ['K562', 'HepG2', 'WTC11', 'S2_dev', 'maize_leaf', ...]:
    # Train on all except holdout
    source_datasets = [d for d in all_datasets if d != holdout]
    trainer.train(source_datasets)

    # Evaluate zero-shot
    results = trainer.evaluate(holdout)
    # Results: spearman_rho, pearson_r, calibration_curve
```

### Key Results

| Transfer Type | Zero-Shot Spearman | With 50-sample Calibration |
|---------------|-------------------|---------------------------|
| Plant → Plant | **0.70** | 0.76 |
| Human → Human | 0.26 | 0.45 |
| Cross-kingdom | -0.08 | 0.30 |

**Location:** `physics/S2A/`

---

## 11. Data Pipeline

### 11.1 Dataset Overview

| Dataset | Organism | Cell Type | Sequences | Length | Activity |
|---------|----------|-----------|-----------|--------|----------|
| K562 | Human | Erythroid | 164,307 | 230bp | 1 output |
| HepG2 | Human | Liver | 97,925 | 230bp | 1 output |
| WTC11 | Human | iPSC | 39,201 | 230bp | 1 output |
| S2 | Drosophila | S2 cells | 352,009 | 249bp | 2 outputs (Dev, Hk) |
| Arabidopsis | Plant | Leaf | ~50K | 170bp | 1 output |
| Maize | Plant | Protoplast | ~50K | 170bp | 1 output |
| Yeast | Fungi | - | 6.8M | 110bp | 1 output |

### 11.2 Data Format

```python
# PhysicsVAE Dataset Format (TSV)
columns = [
    'sequence_id',      # Unique identifier
    'sequence',         # DNA sequence string
    'sequence_length',  # Length (for validation)

    # Activity measurements
    'Dev_log2_enrichment',   # Developmental activity
    'Hk_log2_enrichment',    # Housekeeping activity

    # Physics features (521 columns)
    'pwm_*',            # PWM binding scores
    'thermo_*',         # Thermodynamic features
    'stiff_*',          # Stiffness features
    'shape_*',          # Shape features
    'electro_*',        # Electrostatic features
    ...
]

# Data splits
train/val/test split: 80/10/10 (typical)
Calibration set: Additional 10% for uncertainty calibration
```

### 11.3 Physics Feature Categories

```python
PHYSICS_FEATURE_CATEGORIES = {
    'thermodynamics': [
        'thermo_dH_*',      # Enthalpy
        'thermo_dS_*',      # Entropy
        'thermo_dG_*',      # Free energy
        'thermo_Tm_*',      # Melting temperature
    ],
    'structural': [
        'shape_MGW_*',      # Minor groove width
        'shape_ProT_*',     # Propeller twist
        'shape_Roll_*',     # Roll angle
        'shape_HelT_*',     # Helical twist
    ],
    'mechanical': [
        'stiff_bend_*',     # Bending stiffness
        'stiff_twist_*',    # Twist stiffness
        'stiff_stretch_*',  # Stretch stiffness
    ],
    'electrostatic': [
        'electro_psi_*',    # Electrostatic potential
        'electro_charge_*', # Charge distribution
    ],
    'motif': [
        'pwm_*_max_score',      # Maximum binding score
        'pwm_*_mean_score',     # Mean binding score
        'pwm_*_num_sites',      # Number of binding sites
        'pwm_*_entropy',        # Binding entropy
    ],
}
```

### 11.4 Data Loading

```python
# PhysicsVAEDataset
class PhysicsVAEDataset(Dataset):
    def __init__(
        self,
        descriptor_file: str,
        cell_type: str = None,
        normalize_physics: bool = True,
        max_seq_length: int = None
    ):
        # Load TSV
        self.df = pd.read_csv(descriptor_file, sep='\t')

        # Identify physics columns
        exclude_cols = ['sequence_id', 'sequence', ...]
        self.physics_cols = [c for c in df.columns if c not in exclude_cols]

        # Filter constant features
        valid_features = feature_stds > 1e-8

        # Z-score normalization
        self.physics = (physics - mean) / std

    def __getitem__(self, idx):
        return {
            'sequence': torch.tensor(seq_indices, dtype=torch.long),
            'physics': torch.tensor(physics, dtype=torch.float32),
            'idx': idx
        }
```

---

## 12. Training Framework

### 12.1 Configuration Types

```python
class ConfigurationType(Enum):
    SINGLE_CELLTYPE = "single_celltype"         # K562, HepG2, etc.
    MULTI_CELLTYPE_HUMAN = "multi_celltype_human"  # K562 + HepG2 + WTC11
    CROSS_ANIMAL = "cross_animal"               # Human + Drosophila
    CROSS_KINGDOM = "cross_kingdom"             # Animal + Plant + Fungi
    UNIVERSAL = "universal"                     # All datasets
```

### 12.2 Training Protocols

#### Single Cell-Type Training
```python
# Standard training on one dataset
python train.py --cell_type K562 --epochs 100 --batch_size 64
```

#### Multi-Phase Curriculum Learning
```python
class MultiPhaseTrainer:
    """Progressive training across datasets"""

    Phases:
        1. Train on largest dataset (K562)
        2. Add second dataset (HepG2) with balanced sampling
        3. Add third dataset (WTC11)
        4. Fine-tune on target dataset
```

#### Cross-Kingdom Training
```python
# Shared physics encoder, species-specific heads
model = CrossKingdomModel(
    physics_encoder=SharedPhysicsEncoder(),
    species_heads={
        'human': ActivityHead(),
        'drosophila': ActivityHead(),
        'plant': ActivityHead(),
    }
)
```

### 12.3 Loss Functions

```python
# PhysicsVAE Loss
L_vae = L_recon + β·L_KL + λ·L_physics

# PhysInformer Loss
L_physics = Σ_i w_i · MSE(pred_i, target_i)  # Weighted per-feature MSE
L_activity = MSE(pred_activity, target_activity)
L_total = L_physics + α·L_activity + γ·L_thermo_consistency

# CADENCE Loss
L_activity = MSE(pred, target)
L_uncertainty = NLL under predicted distribution (if uncertainty enabled)
```

### 12.4 Metrics

```python
# Per-epoch metrics
metrics = {
    'pearson_r': pearson_correlation(pred, target),
    'spearman_r': spearman_correlation(pred, target),
    'mse': mean_squared_error(pred, target),
    'r2': r_squared(pred, target),

    # For physics models
    'physics_mse': mean_squared_error(pred_physics, target_physics),
    'per_family_mse': {family: mse for family in physics_families},

    # For VAE
    'recon_loss': cross_entropy,
    'kl_loss': kl_divergence,
    'accuracy': nucleotide_accuracy,
    'perplexity': exp(cross_entropy),
}
```

### 14.5 File Locations

```
training/
├── config.py           # Configuration types (18KB)
├── trainer.py          # Main trainer class (44KB)
├── coordinator.py      # CLI entry point (15KB)
├── datasets.py         # Dataset classes (26KB)
├── data_loaders.py     # Data loading (20KB)
├── samplers.py         # Sampling strategies (24KB)
└── metrics.py          # Evaluation metrics (15KB)
```

---

## 13. Physics Feature Specification

### 13.1 Thermodynamic Features

```
Feature                 Description                          Range
─────────────────────────────────────────────────────────────────────
thermo_dH_mean         Mean enthalpy (kcal/mol)             [-50, 0]
thermo_dH_std          Enthalpy std deviation               [0, 20]
thermo_dS_mean         Mean entropy (cal/mol·K)             [-150, 0]
thermo_dS_std          Entropy std deviation                [0, 50]
thermo_dG_mean         Mean free energy at 37°C (kcal/mol)  [-10, 5]
thermo_dG_std          Free energy std deviation            [0, 5]
thermo_Tm_mean         Mean melting temperature (°C)        [40, 90]
thermo_Tm_std          Melting temperature std              [0, 20]
```

### 13.2 Structural Features

```
Feature                 Description                          Range
─────────────────────────────────────────────────────────────────────
shape_MGW_mean         Minor groove width (Å)               [3, 7]
shape_MGW_std          MGW std deviation                    [0, 2]
shape_ProT_mean        Propeller twist (degrees)            [-20, 0]
shape_ProT_std         ProT std deviation                   [0, 10]
shape_Roll_mean        Roll angle (degrees)                 [-10, 10]
shape_Roll_std         Roll std deviation                   [0, 10]
shape_HelT_mean        Helical twist (degrees)              [30, 40]
shape_HelT_std         HelT std deviation                   [0, 5]
```

### 13.3 Mechanical Features

```
Feature                 Description                          Range
─────────────────────────────────────────────────────────────────────
stiff_bend_mean        Bending stiffness                    [0, 100]
stiff_bend_std         Bending stiffness std                [0, 30]
stiff_twist_mean       Twist stiffness                      [0, 100]
stiff_twist_std        Twist stiffness std                  [0, 30]
stiff_stretch_mean     Stretch stiffness                    [0, 100]
stiff_stretch_std      Stretch stiffness std                [0, 30]
```

### 13.4 PWM/Motif Features

```
Feature                 Description                          Range
─────────────────────────────────────────────────────────────────────
pwm_*_max_score        Maximum PWM score for TF             [-20, 20]
pwm_*_mean_score       Mean PWM score across positions      [-10, 10]
pwm_*_delta_g          Binding free energy (kcal/mol)       [-15, 0]
pwm_*_num_high_affinity Number of high-affinity sites       [0, 10]
pwm_*_entropy          Binding site entropy                 [0, 4]
pwm_*_top_k_mean       Mean of top-k binding scores         [-10, 20]
pwm_*_total_weight     Total binding weight                 [0, 100]
pwm_*_var_score        Variance of binding scores           [0, 50]
```

### 13.5 Electrostatic Features

```
Feature                 Description                          Range
─────────────────────────────────────────────────────────────────────
electro_psi_std_min    Min potential (standard ionic)       [-50, 0]
electro_psi_std_max    Max potential (standard ionic)       [0, 50]
electro_psi_std_mean   Mean potential (standard ionic)      [-25, 25]
electro_psi_enh_min    Min potential (enhanced ionic)       [-100, 0]
electro_psi_enh_max    Max potential (enhanced ionic)       [0, 100]
electro_psi_enh_mean   Mean potential (enhanced ionic)      [-50, 50]
```

---

## 14. Electrostatics Module

### 14.1 APBS Integration

The electrostatics module calculates electrostatic potentials using APBS (Adaptive Poisson-Boltzmann Solver).

```
Pipeline:
1. DNA sequence → 3D structure generation (NAB/tleap)
2. Structure → PDB file
3. PDB → PQR file (add atomic charges and radii)
4. PQR → APBS calculation → Electrostatic potential grid
5. Grid → Feature extraction (per-window statistics)
```

### 14.2 APBS Configuration

```yaml
# Standard ionic conditions
ionic_strength: 0.15 M
temperature: 298.15 K
dielectric_solvent: 78.54
dielectric_solute: 2.0

# Enhanced ionic conditions
ionic_strength: 0.50 M
temperature: 310.15 K

# Grid parameters
grid_spacing: 0.5 Å
grid_dimensions: [129, 129, 129]

# Boundary conditions
boundary: sdh (single Debye-Hückel)
```

### 14.3 Calibration Panel

```
Structure           GC%     CpG Density   Purpose
─────────────────────────────────────────────────────
GC00.pdb           0%      Low           Baseline
GC10.pdb           10%     Low           GC gradient
GC20.pdb           20%     Low           GC gradient
...
GC100.pdb          100%    High          Maximum GC
CpG_low.pdb        50%     Low           CpG effect
CpG_med.pdb        50%     Medium        CpG effect
CpG_high.pdb       50%     High          CpG effect
MGW_narrow.pdb     50%     Medium        Shape effect
MGW_wide.pdb       50%     Medium        Shape effect
```

### 14.4 PSI Calibration Model

```python
# Linear calibration from sequence features to APBS values
class PSICalibrationModel:
    features = ['gc_content', 'cpg_density', 'at_skew', 'minor_groove_score']

    # Learned coefficients per PSI output
    coefficients = {
        'psi_std_min': [a1, a2, a3, a4],
        'psi_std_max': [b1, b2, b3, b4],
        'psi_std_mean': [c1, c2, c3, c4],
        ...
    }

    def predict(self, sequence_features):
        return sum(coef * feat for coef, feat in zip(coefficients, features))
```

### 14.5 File Locations

```
electrostatics/
├── psi_calibration_and_annotation_pipeline.py  # Main pipeline (27KB)
├── annotate_all_datasets.py                    # Dataset annotation (14KB)
├── fit_psi_model.py                            # Calibration model (8KB)
├── validate_against_apbs.py                    # Validation (17KB)
├── build_all_dna.py                            # Structure generation (2KB)
├── convert_to_pqr.py                           # PQR conversion (3KB)
├── template_lpbe.in                            # APBS config template
└── pdb_structures/                             # Calibration PDBs
    ├── GC[00-100].pdb
    ├── CpG_[low/med/high].pdb
    └── MGW_[narrow/wide].pdb
```

---

## Appendix A: Model Comparison

| Model | Parameters | Input | Output | Use Case |
|-------|-----------|-------|--------|----------|
| PhysInformer | ~10M | DNA seq | 500+ physics | Physics prediction |
| TileFormer | ~5M | DNA seq | 6 PSI values | Electrostatics surrogate |
| PhysicsVAE | ~10M | Physics | DNA seq | Physics-conditioned generation |
| PhysicsTransfer | - | Source model | Transferred model | Cross-species transfer |
| CADENCE | ~2M | DNA seq | Activity + uncertainty | Activity prediction |
| CADENCE Pro | ~4.5M | DNA seq | Activity + uncertainty | DREAM challenge (r=0.967) |
| S2A | ~1K | Physics | Z-score activity | Zero-shot universal prediction |
| OracleCheck | - | Sequence | Validation verdict | Design validation |
| PhysicsInterpreter | - | Predictions | Attribution | Mechanistic interpretation |

## Appendix B: Command Reference

```bash
# PhysInformer training
python physics/PhysInformer/train.py --cell_type K562 --epochs 100

# TileFormer training
python physics/TileFormer/train_orchestrator.py --config configs/config.yaml

# PhysicsVAE training
python physics/PhysicsVAE/train.py --cell_type K562 --epochs 100 --batch_size 64

# PhysicsTransfer experiment
python physics/PhysicsTransfer/run_transfer.py --source human --target plant

# CADENCE training
python models/CADENCE/train.py --dataset K562 --use_rc_stem

# S2A Leave-One-Out Evaluation
python physics/S2A/run_s2a.py evaluate \
    --datasets K562 HepG2 WTC11 S2_dev arabidopsis_leaf sorghum_leaf maize_leaf \
    --output-dir results/s2a/leave_one_out/

# Therapeutic Enhancer Design
python applications/therapeutic_enhancer_pipeline.py \
    --target-cell HepG2 --n-sequences 200 --method ism_target

# Disease Variant Analysis
python applications/disease_variant_pipeline.py \
    --vcf variants.vcf --cell-type K562 --output results/variants/
```

## Appendix C: Version Information

```
Framework: FUSEMAP v1.0
PyTorch: 2.0+
Python: 3.10+
CUDA: 11.8+

Dependencies:
- numpy, pandas, scipy
- scikit-learn
- matplotlib, seaborn
- biopython
- pyyaml
```

---

*Documentation generated: January 2026*
*Total codebase: 898 Python files, ~1.2M lines of code*
