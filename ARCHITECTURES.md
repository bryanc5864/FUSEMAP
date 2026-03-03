# FUSEMAP Model Architectures

This document provides comprehensive descriptions of the three core neural network architectures in the FUSEMAP framework: **CADENCE**, **TileFormer**, and **PhysInformer**.

---

## 1. CADENCE (LegNet-based CNN)

**Purpose:** Sequence-to-activity prediction — predicts regulatory element expression from DNA sequence.

**Source:** `models/CADENCE/cadence.py`

### Overview

CADENCE is built on the LegNet architecture, an EfficientNet-inspired 1D convolutional neural network. It processes one-hot encoded DNA sequences through a series of inverted residual blocks with squeeze-and-excitation (SE) attention, progressively reducing spatial resolution while increasing channel depth.

### Architecture

```
Input: [batch, 4, seq_len]  (one-hot DNA, typically seq_len=230)

STEM (LocalBlock):
  Conv1d(4 -> 64, kernel_size=11) -> BatchNorm1d(64) -> SiLU

BLOCK 1:
  ResidualConcat(EffBlock(80, ks=9, expand=4)) -> LocalBlock(160 -> 80) -> MaxPool1d(2)

BLOCK 2:
  ResidualConcat(EffBlock(96, ks=9, expand=4)) -> LocalBlock(192 -> 96) -> MaxPool1d(2)

BLOCK 3:
  ResidualConcat(EffBlock(112, ks=9, expand=4)) -> LocalBlock(224 -> 112) -> MaxPool1d(2)

BLOCK 4:
  ResidualConcat(EffBlock(128, ks=9, expand=4)) -> LocalBlock(256 -> 128) -> MaxPool1d(2)

MAPPER:
  BatchNorm1d(128) -> Conv1d(128 -> 256, kernel_size=1)

GLOBAL POOLING:
  AdaptiveAvgPool1d(1)

HEAD:
  Linear(256 -> 256) -> BatchNorm1d(256) -> SiLU -> Linear(256 -> 1)

Output: [batch]  (scalar activity prediction)
```

### Key Building Blocks

**SELayer (Squeeze-and-Excitation):**
Channel attention mechanism that re-weights feature channels based on global context.
```
GlobalAvgPool -> Linear(ch -> ch//reduction) -> SiLU -> Linear(ch//reduction -> ch) -> Sigmoid
```

**EffBlock (EfficientNet-style inverted residual):**
Core processing block with depthwise separable convolutions and channel attention.
```
Conv1d(in -> in*expand, ks=1)       # Pointwise expansion
  -> BatchNorm -> SiLU
Conv1d(in*expand -> in*expand, ks=9, groups=in*expand)  # Depthwise
  -> BatchNorm -> SiLU
SELayer(in*expand)                   # Channel attention
Conv1d(in*expand -> in, ks=1)       # Pointwise projection
  -> BatchNorm -> SiLU
```

**ResidualConcat:**
Applies a function block and concatenates its output with the input, doubling channels (e.g., 80 -> 160).

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input channels | 4 |
| Stem channels / kernel | 64 / 11 |
| Block channels | [80, 96, 112, 128] |
| Block kernel size | 9 |
| Expansion ratio | 4 |
| Pool sizes | [2, 2, 2, 2] |
| Mapper output | 256 |
| Head hidden | 256 |
| Activation | SiLU |
| Parameters | ~2.1M |

### Shape Trace (seq_len=230)

| Stage | Shape |
|-------|-------|
| Input | [B, 4, 230] |
| Stem | [B, 64, 230] |
| Block 1 | [B, 80, 115] |
| Block 2 | [B, 96, 57] |
| Block 3 | [B, 112, 28] |
| Block 4 | [B, 128, 14] |
| Mapper | [B, 256, 14] |
| Global pool | [B, 256] |
| Output | [B] |

### Performance

| Dataset | Pearson r | Spearman rho |
|---------|-----------|-------------|
| K562 | 0.809 | 0.759 |
| HepG2 | 0.786 | 0.770 |
| WTC11 | 0.698 | 0.591 |
| DeepSTARR Dev | 0.909 | 0.889 |
| DeepSTARR Hk | 0.920 | 0.863 |
| Maize (leaf) | 0.796 | 0.799 |

---

## 2. TileFormer

**Purpose:** Electrostatic potential surrogate model — predicts APBS-computed electrostatic properties from 20bp DNA tiles at 10,000x speedup.

**Source:** `physics/TileFormer/models/tileformer_architecture.py` (model definition), `physics/TileFormer/train_orchestrator.py` (training script)

### Overview

TileFormer is a 6-layer transformer encoder that maps 20bp DNA sequence tiles to 6 electrostatic potential summary statistics. It replaces expensive APBS (Adaptive Poisson-Boltzmann Solver) computations with a learned surrogate. Trained on ~52K 20bp sequences with APBS-computed ground truth, it enables rapid electrostatic screening at the tile level. For full-length sequences (e.g., 230bp), TileFormer is applied in a sliding window fashion (20bp windows, stride 10) to produce spatially-resolved electrostatic profiles.

### Architecture

The trained model uses `TileFormerWithMetadata`, which extends the base transformer with a metadata processing branch:

```
Input: [batch, 20]  (integer token indices, 20bp DNA tiles)
Metadata: [batch, 3]  (gc_content, cpg_density, minor_groove_score)

EMBEDDING:
  Embedding(vocab_size=5, d_model=256)  # A=0, T=1, G=2, C=3, N=4
  Scale by sqrt(256)

POSITIONAL ENCODING:
  Sinusoidal positional encoding (max_len=200, only first 20 positions used)
  x = embedding + positional_encoding

TRANSFORMER ENCODER (x6 layers):
  Each layer:
    MultiheadAttention(d_model=256, n_heads=8, dropout=0.1)
    LayerNorm + residual
    FeedForward:
      Linear(256 -> 1024) -> GELU -> Dropout(0.1) -> Linear(1024 -> 256)
    LayerNorm + residual

GLOBAL POOLING:
  Mean pooling over sequence dimension
  Linear(256 -> 256) -> GELU -> Dropout(0.1)

METADATA PROCESSING:
  Linear(3 -> 64) -> GELU -> Linear(64 -> 256)

FEATURE FUSION:
  Concatenate(sequence_features, metadata_features) -> Linear(512 -> 256)

PSI HEAD:
  Linear(256 -> 128) -> GELU -> Dropout(0.1) -> Linear(128 -> 6)

UNCERTAINTY HEAD:
  Linear(256 -> 128) -> GELU -> Dropout(0.1) -> Linear(128 -> 6) -> Softplus

Output: {'psi': [batch, 6], 'uncertainty': [batch, 6]}
```

### Output Targets (6 values)

The 6 output dimensions correspond to electrostatic potential statistics computed via APBS under two ionic configurations:

| Index | Target | Description |
|-------|--------|-------------|
| 0 | STD_PSI_MIN | Standard ionic config - minimum potential |
| 1 | STD_PSI_MAX | Standard ionic config - maximum potential |
| 2 | STD_PSI_MEAN | Standard ionic config - mean potential |
| 3 | ENH_PSI_MIN | Enhanced ionic config - minimum potential |
| 4 | ENH_PSI_MAX | Enhanced ionic config - maximum potential |
| 5 | ENH_PSI_MEAN | Enhanced ionic config - mean potential |

### Training Data

| Split | Sequences | Sequence Length |
|-------|-----------|-----------------|
| Train | 41,582 | 20 bp |
| Val | 5,197 | 20 bp |
| Test | 5,199 | 20 bp |
| **Total** | **51,978** | **20 bp** |

Ground truth: APBS electrostatic simulations via tLEaP (NAB → PDB2PQR → APBS pipeline).

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 5 (A, T, G, C, N) |
| d_model | 256 |
| Attention heads | 8 |
| d_ff (feedforward) | 1024 |
| Transformer layers | 6 |
| Sequence length | 20 bp |
| Metadata features | 3 (GC%, CpG density, minor groove) |
| Dropout | 0.1 |
| Output dimension | 6 + 6 (psi + uncertainty) |
| Activation | GELU |
| Learning rate | 1e-4 |
| Batch size | 32 |
| Epochs | 25 |
| Parameters | ~5.1M |

### Shape Trace

| Stage | Shape |
|-------|-------|
| Input | [B, 20] |
| Embedding | [B, 20, 256] |
| + Positional encoding | [B, 20, 256] |
| Transformer (x6) | [B, 20, 256] |
| Mean pool | [B, 256] |
| Metadata | [B, 3] -> [B, 256] |
| Fused | [B, 512] -> [B, 256] |
| PSI head | [B, 6] |
| Uncertainty head | [B, 6] |

### Performance (Epoch 25, Test Set)

| Target | R^2 | Pearson r | Spearman r | RMSE |
|--------|-----|-----------|------------|------|
| STD_PSI_MIN | 0.962 | 0.982 | 0.981 | 0.005 |
| STD_PSI_MAX | 0.955 | 0.981 | 0.980 | 0.002 |
| STD_PSI_MEAN | 0.961 | 0.984 | 0.983 | 0.003 |
| ENH_PSI_MIN | 0.966 | 0.984 | 0.984 | 0.013 |
| ENH_PSI_MAX | 0.960 | 0.981 | 0.981 | 0.012 |
| ENH_PSI_MEAN | 0.961 | 0.982 | 0.981 | 0.012 |
| **Overall** | **0.961** | **0.982** | **0.981** | **0.009** |

Speedup: **10,000x** compared to APBS computation.

### Downstream Use in PhysInformer

TileFormer predictions are applied in a sliding window fashion over full-length sequences (e.g., 230bp) to produce spatially-resolved electrostatic profiles:
- 22 windows of 20bp with stride 10
- 6 values per window = 132 electrostatic features per sequence
- These pre-computed predictions are stored as NPZ files and used as training targets for PhysInformer's electrostatic head

---

## 3. PhysInformer

**Purpose:** Sequence-to-physics model — predicts hundreds of biophysical descriptors from DNA sequence for downstream transfer learning.

**Source:** `physics/PhysInformer/physics_aware_model.py` (trained model), `physics/PhysInformer/model.py` (earlier prototype, unused)

### Overview

PhysInformer uses a hybrid CNN+SSM (state space model) backbone with physics-informed routing. Rather than a generic shared encoder, it routes features through 7 property-specific adapters with kernel sizes tailored to the spatial scale of each biophysical property. All prediction heads output both mean and log-variance, enabling heteroscedastic uncertainty estimation.

> **Note:** The codebase contains two model files. `model.py` defines an earlier prototype — a standard 8-layer transformer (512d, 8 heads) with independent prediction heads. This was superseded by the `PhysicsAwareModel` in `physics_aware_model.py`, which is the architecture actually used in all training runs (`train.py` imports only `PhysicsAwareModel`). All reported results use the PhysicsAwareModel (~11.4M params) described below.

### Architecture

```
Input: [batch, seq_len]  (seq_len=230)

CONV STEM:
  Embedding(5 -> 128)
  Conv1d(128 -> 128, ks=11) -> SiLU -> Dropout(0.1)
  Conv1d(128 -> 192, ks=9)  -> SiLU -> Dropout(0.1)  + residual projection (128 -> 192)
  Conv1d(192 -> 256, ks=7)  -> SiLU -> Dropout(0.1)  + residual projection (192 -> 256)

SSM LAYERS (x2):
  Each SimplifiedSSMLayer:
    State space model (d_state=16)
    Gating: input -> gate (tanh) -> output
    LayerNorm + residual connection

DUAL-PATH FEATURE PYRAMID:
  Local path:
    Conv1d(256 -> 192, ks=9) -> SiLU
    Conv1d(192 -> 192, ks=5) -> SiLU
  Global path:
    SimplifiedSSMLayer(256)
    Linear(256 -> 192)
  Concatenate: [batch, seq_len, 384]

PHYSICS ROUTERS (7 property-specific adapters):
  Each router: Conv1d(384 -> 128, ks=K) -> SiLU -> gated window aggregation
  thermo:        ks=3
  electrostatic: ks=15
  bend:          ks=11
  stiff:         ks=7
  pwm:           ks=15, output_dim=256
  entropy:       ks=21
  advanced:      ks=13

PROPERTY-SPECIFIC HEADS:

  Thermodynamic Head (via thermo router):
    Shared: Linear(128 -> 128) + SiLU
    dH head: Linear(128 -> 2)  # (mean, log_var)
    dS head: Linear(128 -> 2)
    dG head: Linear(128 -> 2)
    Constraint: dG = dH - T*dS (T=310K)

  Electrostatic Head (via electrostatic router):
    22 window heads (one per 20bp window):
      Linear(128 -> 128) -> SiLU -> Dropout -> Linear(128 -> 6)
    Global head:
      Linear(128 -> 128) -> SiLU -> Linear(128 -> 2)

  Scalar Property Heads (varies by cell type, via appropriate router):
    Each: Linear(router_dim -> 64) -> SiLU -> Dropout -> Linear(64 -> 2)
    Output: (mean, log_var) per property

Output: Dictionary with all property predictions including means and log-variances
```

### Router Design Rationale

Each router uses a different kernel size tailored to the spatial scale of its target property:

| Router | Kernel Size | Rationale |
|--------|-------------|-----------|
| thermo | 3 | Nearest-neighbor base stacking (2-3 bp) |
| electrostatic | 15 | Charge distribution spans ~1.5 helical turns |
| bend | 11 | Bending is ~1 helical turn |
| stiff | 7 | Stiffness is local (~0.5 turns) |
| pwm | 15 | Motif widths typically 6-15 bp |
| entropy | 21 | Shannon entropy requires broader context |
| advanced | 13 | Minor groove width varies over ~1 turn |

### Loss Functions

- **Heteroscedastic loss:** `precision * (mean - target)^2 + log_var` (learns per-prediction uncertainty)
- **Thermodynamic identity loss:** enforces `dG = dH - T*dS`
- **Total variation loss:** smoothness regularization on window predictions
- **Auxiliary loss:** Huber or MSE for activity prediction heads

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 5 (A, T, G, C, N) |
| Embedding dim | 128 |
| Backbone dim (d_model) | 256 |
| Feature pyramid dim | 384 |
| SSM layers | 2 (d_state=16) |
| Conv stem kernels | [11, 9, 7] |
| Dropout | 0.1 |
| Temperature | 310K (thermodynamic constraint) |
| Activation | SiLU |
| Parameters | ~11.4M (K562 config) |

### Number of Descriptor Outputs by Cell Type

| Cell Type | Property Heads | Active (non-zero-variance) |
|-----------|---------------|---------------------------|
| K562 | 515 | ~498 |
| HepG2 | 545 | ~537 |
| WTC11 | 528 | ~522 |

### Within-Cell-Type Performance (Validation)

| Cell Type | Overall Pearson r | Descriptor Mean r | Descriptor Median r |
|-----------|------------------|-------------------|---------------------|
| K562 | 0.918 | 0.896 | 0.987 |
| HepG2 | 0.915 | 0.892 | 0.983 |
| WTC11 | 0.906 | 0.877 | 0.974 |
| S2 (Drosophila) | 0.919 | 0.894 | 0.988 |
| Maize | 0.896 | 0.870 | 0.949 |
| Arabidopsis | 0.838 | 0.799 | 0.930 |
| Sorghum | 0.849 | 0.813 | 0.942 |

### Cross-Species Transfer Performance (Mean Pearson r)

| Source -> Target | Overlap Features | Mean r | Median r |
|-----------------|-----------------|--------|----------|
| K562 -> HepG2 | 411 | 0.847 | 0.968 |
| K562 -> WTC11 | 411 | 0.839 | 0.971 |
| K562 -> S2 | 267 | 0.729 | 0.901 |
| K562 -> Arabidopsis | 267 | 0.656 | 0.722 |

---

## Comparison Summary

| Model | Type | Params | Input | Output | Primary Use |
|-------|------|--------|-------|--------|-------------|
| **CADENCE** | CNN (EfficientNet) | ~2.1M | [B, 4, 230] one-hot | [B] scalar | Activity prediction |
| **TileFormer** | Transformer (6L) | ~5.1M | [B, 20] tokens + [B, 3] metadata | [B, 6] + [B, 6] uncertainty | Electrostatic surrogacy |
| **PhysInformer** | CNN+SSM hybrid | ~11.4M | [B, 230] tokens | Dict (means + log-variances) | Biophysical descriptors |

All models process DNA sequences and share a common vocabulary (A, T, G, C, N). CADENCE takes one-hot encoded input while TileFormer and PhysInformer take integer token indices.
