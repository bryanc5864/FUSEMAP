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
| Parameters | ~1.3M |

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
| **CADENCE** | CNN (EfficientNet) | ~1.3M | [B, 4, 230] one-hot | [B] scalar | Activity prediction |
| **TileFormer** | Transformer (6L) | ~5.1M | [B, 20] tokens + [B, 3] metadata | [B, 6] + [B, 6] uncertainty | Electrostatic surrogacy |
| **PhysInformer** | CNN+SSM hybrid | ~11.4M | [B, 230] tokens | Dict (means + log-variances) | Biophysical descriptors |

All models process DNA sequences and share a common vocabulary (A, T, G, C, N). CADENCE takes one-hot encoded input while TileFormer and PhysInformer take integer token indices.

---

## 4. APBS Electrostatics Pipeline

**Purpose:** Generate ground-truth electrostatic potential features for TileFormer training via numerical solution of the Poisson-Boltzmann equation.

**Source:** `electrostatics/` (modular scripts), `physics/TileFormer/electrostatics/tleap_abps_processor.py` (production pipeline)

### Pipeline Overview

```
DNA Sequence (20 bp)
       │
       ▼
┌──────────────┐
│   1. tLEaP   │  Build canonical B-DNA duplex (BSC1 force field)
│              │  source leaprc.DNA.bsc1
└──────┬───────┘
       │  PDB
       ▼
┌──────────────┐
│ 2. Fix Chains│  Assign chain IDs (A/B), renumber residues, add TER records
└──────┬───────┘
       │  PDB (standardized)
       ▼
┌──────────────┐
│  3. PDB2PQR  │  Add atomic charges and radii (AMBER force field, pH 7.0)
│              │  --ff=AMBER --with-ph=7.0 --titration-state-method=propka
└──────┬───────┘
       │  PQR
       ▼
┌──────────────┐
│   4. APBS    │  Solve linear Poisson-Boltzmann equation on focused grid
│              │  Two ionic configurations (standard + enhanced)
└──────┬───────┘
       │  DX (3D potential grid)
       ▼
┌──────────────┐
│ 5. Shell     │  Extract mean/min/max potential from solvent shell
│  Extraction  │  using cKDTree nearest-neighbor distance queries
└──────┬───────┘
       │
       ▼
6 ψ values (min/max/mean × standard/enhanced)
```

### APBS Parameters

| Parameter | Standard Config | Enhanced Config |
|-----------|----------------|-----------------|
| **Grid dimensions** | 193 × 193 × 193 | 257 × 257 × 257 |
| **Grid spacing** | ~0.21 Å | ~0.16 Å |
| **Coarse grid (cglen)** | 200 × 200 × 200 Å | 200 × 200 × 200 Å |
| **Focused grid (fglen)** | 40 × 40 × 40 Å | 35 × 35 × 35 Å |
| **Ion concentration** | 150 mM (physiological) | 10 mM (reduced screening) |
| **Ion radii (+/-)** | 2.0 Å / 2.0 Å | 2.0 Å / 2.0 Å |
| **Solute dielectric (pdie)** | 2.0 | 2.0 |
| **Solvent dielectric (sdie)** | 78.5 (water, 298 K) | 78.5 |
| **Temperature** | 298.15 K | 298.15 K |
| **Shell inner** | 2.0 Å | 0.5 Å |
| **Shell outer** | 6.0 Å | 2.0 Å |
| **Energy minimization** | Disabled | Enabled (2000 steps, ncyc=500) |
| **Solver** | LPBE | LPBE |
| **Boundary conditions** | Multiple Debye-Hückel (mdh) | Multiple Debye-Hückel (mdh) |
| **Charge discretization** | Cubic B-spline (spl2) | Cubic B-spline (spl2) |
| **Surface** | Molecular surface (smol), srad=1.4 Å | Molecular surface (smol), srad=1.4 Å |

### Standard vs Enhanced Configuration

The **standard** configuration uses physiological ionic strength (150 mM NaCl) and a broad solvent shell (2–6 Å), providing a baseline electrostatic characterization.

The **enhanced** configuration amplifies sequence-dependent signal by 2–3× through three modifications:
1. **Tighter shell** (0.5–2.0 Å): Samples closer to the molecular surface where 1/r field amplification is strongest
2. **Lower ionic strength** (10 mM): Reduces Debye screening, producing ~3× larger potentials
3. **Finer grid** (257³ at 0.16 Å): Better resolves groove-level electrostatic features

The enhanced configuration also enables sander energy minimization (2000 steps) to capture sequence-dependent structural relaxation beyond ideal B-DNA geometry.

### Output: 6 ψ Values

| Index | Target | Description |
|-------|--------|-------------|
| 0 | STD_PSI_MIN | Standard config — minimum potential in shell |
| 1 | STD_PSI_MAX | Standard config — maximum potential in shell |
| 2 | STD_PSI_MEAN | Standard config — mean potential in shell |
| 3 | ENH_PSI_MIN | Enhanced config — minimum potential in shell |
| 4 | ENH_PSI_MAX | Enhanced config — maximum potential in shell |
| 5 | ENH_PSI_MEAN | Enhanced config — mean potential in shell |

All ψ values are in **kT/e** (thermal energy units at 298.15 K). Shell extraction uses `scipy.spatial.cKDTree` for efficient distance queries with artifact filtering (|ψ| ≤ 60 kT/e).

### Key Software Dependencies

- **AmberTools (tLEaP/sander):** BSC1 force field for B-DNA construction and optional minimization
- **PDB2PQR 3.0:** Charge/radius assignment with PROPKA titration at pH 7.0
- **APBS 3.x:** Adaptive Poisson-Boltzmann Solver for electrostatic potential grids
- **gridDataFormats:** OpenDX file parsing for 3D potential grids

---

## 5. Biophysical Descriptors

**Purpose:** Compute sequence-derived biophysical features that serve as ground-truth targets for PhysInformer training.

**Source:** `physics/process_descriptors.py` (3,304 lines — main computation), reference data in `physics/data/`

### Overview

The descriptor pipeline computes ~673 biophysical features per DNA sequence, organized into categories that map to PhysInformer's 7 physics routers. Each category captures a different physical or informational property of DNA at the appropriate spatial scale.

### Feature Categories

| # | Category | Features | Router | Data Source | Key Descriptors |
|---|----------|----------|--------|-------------|-----------------|
| 1 | Thermodynamic | 45 | thermo | SantaLucia NN (1998) | ΔH, ΔS, ΔG, Tm, percentiles, local energy dynamics |
| 2 | Stiffness | ~62 | stiff | Olson et al. (1998) + DNAProperties | Deformation energy (twist/tilt/roll/shift/slide/rise), PCA, z-scores |
| 3 | Bending | 44 | bend | DNAProperties (row 4) | Bending energy, RMS curvature profiles, spectral signatures, hotspots |
| 4 | Entropy & Complexity | 62 | entropy | Computed (Shannon, k-mer, LZ) | Shannon entropy, k-mer entropy (k=1–6), compressibility, Rényi, MI profiles |
| 5 | TF Binding (PWM) | ~405 | pwm | JASPAR 2024 (~50 TFs) | Per-motif: max score, ΔG, occupancy, entropy; aggregate diversity |
| 6 | Advanced | 55 | advanced | SantaLucia NN, DNAProperties, sequence patterns | Melting (14), MGW (5), stacking (15), G4 potential (4), torsional stress (13), fractal (4) |
| 7 | Electrostatic | 132 | electrostatic | APBS pipeline (Section 4) | 22 windows × 6 ψ values (TileFormer sliding-window predictions) |

**Total: ~673 scalar features + 132 electrostatic window features**

### Reference Databases

**SantaLucia Nearest-Neighbor Parameters** (`physics/data/SantaLuciaNN.tsv`):
- 10 unique dinucleotide steps with ΔH (kcal/mol) and ΔS (cal/mol·K)
- Used for: thermodynamic profiles (ΔG = ΔH − TΔS), melting temperature estimation, breathing analysis

**Olson Structural Parameters** (`physics/data/OlsonMatrix.tsv`):
- 10 dinucleotide steps × 6 structural modes (twist, tilt, roll, shift, slide, rise)
- Each mode has mean and standard deviation from crystallographic data
- Used for: deformation energy calculation, stiffness analysis, PCA of structural variation

**DNA Properties Database** (`physics/data/DNAProperties.txt`):
- 125 properties × 16 dinucleotides (all permutations including strand asymmetry)
- Includes: bend parameters (row 4), stacking energies (row 60), minor groove width (empirical pentamer model), stiffness coefficients (rows 67–71)

**JASPAR 2024 PWMs:**
- Top ~50 transcription factor position weight matrices
- 8 features computed per motif: max score, binding ΔG (kcal/mol), mean score, variance, total occupancy weight, high-affinity site count, binding entropy, top-k mean
- 5 aggregate features: max-of-max score, min ΔG, TF diversity, sum top-5 ΔG, best TF index

### Category Details

**Thermodynamic (45 features):** Global sums (ΔH, ΔS, ΔG, Tm), per-dinucleotide statistics (mean, variance, IQR), 7 percentiles each for ΔH/ΔS/ΔG (5th–95th), local energy dynamics over 10 bp windows (min/max/range of extremes), and energy run analysis (consecutive low/high-energy regions at 10th/90th percentile thresholds).

**Stiffness (~62 features):** Total/mean/variance/max/min deformation energy from Olson parameters, per-mode energies (6 modes × 3 stats), PCA projections (PC1/PC2 mean and variance), cross-term interactions, z-score normalized deviations (6 modes × 3 stats), high-energy region counts at 3 thresholds (2.0/5.0/10.0 kcal/mol), energy distribution entropy, nucleotide composition (AT/GC skew, purine ratio, GC content), and composition-stiffness correlations.

**Bending (44 features):** Global bending energy (total, mean, max, variance), RMS curvature at 4 window sizes (5/7/9/11 bp × 2 stats), curvature variance profiles (4 windows × 2 stats), curvature gradient (mean, max), windowed max bend analysis (4 windows × 3 stats), hotspot detection (z-score ≥ 2.0, count + density), spectral signatures at 3 periodicities (1/5, 1/7, 1/10 × 2 stats), and attention bias statistics (mean, min).

**Entropy & Complexity (62 features):** Global Shannon entropy (raw, normalized, GC-binary), k-mer entropies (k=1–6), complexity metrics (gzip compressibility, Lempel-Ziv, conditional entropy, complexity index), Rényi entropies (α=0, 2), sliding window Shannon entropy (3 windows × 4 stats), sliding window GC entropy (3 windows × 4 stats), k-mer entropy profiles (k=2,3 × 2 windows × 3 stats), mutual information at 10 distances, and entropy rate estimate.

**TF Binding (~405 features):** Log-odds scoring against ~50 JASPAR PWMs with Boltzmann-weighted occupancy. Per-motif features: max score, binding free energy, mean/variance of scores, total occupancy weight (capped at exp(10)), high-affinity site count (≥2 bits), binding entropy, top-3 mean score. Aggregate features capture cross-TF binding landscape.

**Advanced (55 features):** Groups six sub-analyses under a single router: melting stability (14 features: ΔG distribution with mean/std/min/max, 7 percentiles, IQR, unstable fraction, soft-min breathing), minor groove width (5 features: mean/std/min/max, narrow groove fraction < 4.5 Å), base stacking energy (15 features: basic stats, percentiles/IQR, entropy, concentration, transition variance, GC/AT-rich patterns), G-quadruplex potential (4 features: G4Hunter-style max/mean score, hotspot count, peak distance), torsional stress (13 features: mean/max/sum stress, opening stretch, opening rate, 7 percentiles, IQR), and fractal dimension (4 features: k-mer self-similarity exponent across k=1–6).

---

## 6. Sequence Datasets

**Purpose:** Training and evaluation data for CADENCE activity prediction models across species and regulatory element types.

**Source:** `training/config.py` (dataset catalog), `training/data_loaders.py` (loading and preprocessing)

### Dataset Summary

| Dataset | Species | Element | Sequences | Seq Length | Outputs | Validation |
|---------|---------|---------|-----------|------------|---------|------------|
| ENCODE4 K562 | Human | Enhancer | 196,664 | 230 bp | 1 (activity) | 10-fold CV |
| ENCODE4 HepG2 | Human | Enhancer | 122,926 | 230 bp | 1 (activity) | 10-fold CV |
| ENCODE4 WTC11 | Human | Enhancer | 46,128 | 230 bp | 1 (activity) | 10-fold CV |
| DeepSTARR | Drosophila | Enhancer | 484,052 | 249 bp | 2 (Dev, Hk) | Chr holdout |
| DREAM Yeast | Yeast | Promoter | ~6.8M | 110 bp | 1 (expression) | Standard |
| Jores Arabidopsis | Arabidopsis | Promoter | 13,462 | 170 bp | 2 (leaf, proto) | Train/Test |
| Jores Maize | Maize | Promoter | 24,604 | 170 bp | 2 (leaf, proto) | Train/Test |
| Jores Sorghum | Sorghum | Promoter | 19,673 | 170 bp | 2 (leaf, proto) | Train/Test |

### ENCODE4 lentiMPRA (Human)

**Source:** ENCODE Project Consortium lentiMPRA data (Nature, 2021)

Three cell types from massively parallel reporter assays measuring enhancer activity as log2(RNA/DNA) ratios:

| Cell Type | Description | Sequences | Folds |
|-----------|-------------|-----------|-------|
| K562 | Chronic myelogenous leukemia | 196,664 | 10 |
| HepG2 | Hepatocellular carcinoma | 122,926 | 10 |
| WTC11 | iPSC-derived fibroblasts | 46,128 | 10 |

- **Sequence length:** 230 bp (native)
- **Output:** 1 scalar — observed log2(RNA/DNA) activity ratio
- **Validation:** 10-fold cross-validation with pre-computed fold splits
- **Data path:** `data/lentiMPRA_data/{K562,HepG2,WTC11}/`
- **Note:** Each fold includes both forward and reverse complement sequences with identical activity labels

A joint multi-cell-type configuration (`encode4_joint`) trains on ~60,000 shared sequences with 3 outputs (one per cell type) using standard train/val/test splits.

### DeepSTARR (Drosophila)

**Source:** DeepSTARR (de Almeida et al.) — *Drosophila melanogaster* S2 embryonic cells

| Split | Sequences |
|-------|-----------|
| Train | 352,009 |
| Val | 40,570 |
| Test | 41,186 |
| Calibration | 50,287 |
| **Total** | **484,052** |

- **Sequence length:** 249 bp (native)
- **Outputs:** 2 — Dev (developmental enhancer activity) and Hk (housekeeping enhancer activity), both as log2 enrichment
- **Validation:** Chromosome holdout (test: chr2R, val: chr2L)
- **Data path:** `data/S2_data/splits/`

### DREAM Yeast

**Source:** DREAM Challenge 2022 — *Saccharomyces cerevisiae* promoter expression prediction

| Split | Sequences |
|-------|-----------|
| Train | 6,638,507 |
| Val | 33,696 |
| Test | 71,103 |
| Calibration | 67,055 |
| **Total** | **~6.8M** |

- **Sequence length:** 110 bp (native)
- **Output:** 1 scalar — MAUDE expression level
- **Validation:** Standard train/val/test splits with weighted metrics for class imbalance
- **Data path:** `data/yeast_data/`
- **Note:** In universal (cross-kingdom) training configurations, subsampled to 500K per epoch for class balance

### Jores Plant Promoters

**Source:** Jores et al. (2021) — synthetic promoter designs from comprehensive plant core promoter analysis

| Species | Train | Test | Total |
|---------|-------|------|-------|
| Arabidopsis (*A. thaliana*) | 12,115 | 1,347 | 13,462 |
| Maize (*Zea mays*) | 22,143 | 2,461 | 24,604 |
| Sorghum (*S. bicolor*) | 17,705 | 1,968 | 19,673 |

- **Sequence length:** 170 bp (native)
- **Outputs:** 2 — leaf (tobacco leaf transient expression) and proto (maize protoplast expression), both as log2 enrichment
- **Validation:** 90/10 train/test split
- **Data path:** `data/plant_data/jores2021/processed/{arabidopsis,maize,sorghum}/`

### Preprocessing Pipeline

All datasets share a common preprocessing pipeline implemented in `training/data_loaders.py`:

1. **One-hot encoding:** DNA → [4, L] array (A→0, C→1, G→2, T→3, N→0); unknown characters get uniform 0.25
2. **Z-score normalization:** Per-output, per-dataset: `(value − μ) / (σ + 1e-8)`, statistics computed per split
3. **Reverse complement augmentation** (training only): 50% probability of flipping to reverse complement
4. **Shift augmentation** (ENCODE4 human datasets only): Random ±21 bp shift with N-padding on the opposite end
5. **Padding/cropping:** Center pad to target length with uniform 0.25 (N); center crop if longer. Native lengths used for single-dataset training; 256 bp for multi-dataset configurations
