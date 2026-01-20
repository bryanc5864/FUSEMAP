# TileFormer Electrostatic Potential Features - Complete Documentation

## Overview

TileFormer predicts 6 electrostatic potential (PSI/ψ) features for DNA sequences using two different physical configurations: **Standard (STD)** and **Enhanced (ENH)**. These represent electrostatic properties computed via Adaptive Poisson-Boltzmann Solver (ABPS).

---

## The 6 Electrostatic Features

For each DNA sequence segment, TileFormer predicts:

| Feature | Description | Configuration |
|---------|-------------|---------------|
| **STD_PSI_MIN** | Minimum electrostatic potential | Standard (physiological) |
| **STD_PSI_MAX** | Maximum electrostatic potential | Standard (physiological) |
| **STD_PSI_MEAN** | Average electrostatic potential | Standard (physiological) |
| **ENH_PSI_MIN** | Minimum electrostatic potential | Enhanced (high sensitivity) |
| **ENH_PSI_MAX** | Maximum electrostatic potential | Enhanced (high sensitivity) |
| **ENH_PSI_MEAN** | Average electrostatic potential | Enhanced (high sensitivity) |

**Units**: kT/e (thermal energy per elementary charge)

---

## Two Physical Scales: STD vs ENH

### Standard (STD) Configuration - Physiological Conditions

**Purpose**: Represents realistic cellular/nuclear electrostatic environment

**Parameters**:
```
Ion concentration:  150 mM (physiological salt)
Ion shell:
  - Inner radius:   2.0 Å
  - Outer radius:   6.0 Å
Grid resolution:    193³ points (~0.21 Å spacing)
Box size:           40 × 40 × 40 Å
Grid tolerance:     1×10⁻⁵
```

**Physical Characteristics**:
- Moderate electrostatic screening due to physiological salt concentration
- Represents how DNA electrostatics behave in typical cellular conditions
- Provides baseline measurements for biological context
- Lower computational sensitivity

**Biological Relevance**:
- Mimics conditions in nucleus (~150 mM ionic strength)
- Relevant for protein-DNA binding in vivo
- Captures biologically functional electrostatic patterns

---

### Enhanced (ENH) Configuration - High Sensitivity

**Purpose**: Amplifies subtle sequence-dependent electrostatic variations

**Parameters**:
```
Ion concentration:  10 mM (low salt - 15× less than physiological)
Ion shell:
  - Inner radius:   0.5 Å (4× tighter than STD)
  - Outer radius:   2.0 Å (3× smaller than STD)
Grid resolution:    257³ points (~0.16 Å spacing - finer)
Box size:           35 × 35 × 35 Å (smaller, focused)
Grid tolerance:     1×10⁻⁶ (10× stricter than STD)
```

**Physical Characteristics**:
- **2-3× amplified signal** from tighter ion shell
- **3× larger potentials** from reduced ionic screening (low salt)
- **Better groove resolution** from finer computational grid
- Can detect >0.2 kT/e differences between single-base substitutions

**Biological Relevance**:
- Reveals intrinsic sequence-dependent electrostatic properties
- Less screening allows detection of subtle structural features
- Useful for discriminating between similar sequences
- Captures minor groove width variations and base-specific effects

---

## Physical Meaning of Electrostatic Potential (PSI)

**What it represents**:
- Electric field around DNA molecule caused by:
  - Negatively charged phosphate backbone
  - Partial charges on nucleotide bases
  - Ion interactions in solution (salt screening)
  - Solvent (water) polarization

**Why it matters**:
- Governs protein-DNA recognition and binding
- Influences DNA-DNA interactions
- Affects chromatin compaction
- Determines accessibility to regulatory factors
- Critical for sequence optimization and design

**Typical values**:
- **STD**: -0.3 to -0.1 kT/e (moderate screening)
- **ENH**: -1.8 to -1.5 kT/e (3× larger due to less screening)

---

## TileFormer Model Performance

**Training data**: 51,981 20bp DNA sequences with ground-truth ABPS-computed PSI values
- Train: 41,583 sequences
- Val: 5,198 sequences
- Test: 5,200 sequences

**Test Set Performance (5,200 20bp sequences)**:

| Feature | Pearson R | R² | RMSE |
|---------|-----------|-----|------|
| STD_PSI_MIN | 0.981 | 0.960 | 0.0051 |
| STD_PSI_MAX | 0.981 | 0.954 | 0.0020 |
| STD_PSI_MEAN | 0.984 | 0.959 | 0.0032 |
| ENH_PSI_MIN | 0.984 | 0.966 | 0.0123 |
| ENH_PSI_MAX | 0.981 | 0.961 | 0.0113 |
| ENH_PSI_MEAN | 0.981 | 0.961 | 0.0121 |

**Overall**: All features achieve **Pearson R > 0.98**, indicating excellent predictive accuracy.

---

## Sliding Window Approach for Longer Sequences

TileFormer is trained on 20bp sequences but applied to longer sequences using a sliding window:

### For 230bp Sequences (Human cell types: HepG2, K562, WTC11)

```
Window size:  20 bp
Stride:       10 bp (50% overlap)
Total windows: 22 windows per sequence

Window positions:
  Window 0:  positions 1-20
  Window 1:  positions 11-30
  Window 2:  positions 21-40
  ...
  Window 21: positions 211-230
```

**Total features per 230bp sequence**: **132 features** (22 windows × 6 PSI values)

### For 249bp Sequences (S2/Drosophila)

```
Window size:  20 bp
Stride:       11 bp (45% overlap)
Total windows: 23 windows per sequence
```

**Total features per 249bp sequence**: **138 features** (23 windows × 6 PSI values)

### For 110bp Sequences (DREAM/Yeast)

```
Window size:  20 bp
Stride:       ~10 bp (50% overlap)
Total windows: ~10 windows per sequence

Calculation: (110 - 20) / 10 + 1 = 10 windows
```

**Total features per 110bp sequence**: **60 features** (10 windows × 6 PSI values)

---

## Feature Naming Convention

Features are named systematically:

```
tileformer_w{window_id}_{configuration}_{statistic}
```

**Examples**:
- `tileformer_w0_STD_PSI_MIN` - Window 0, standard config, minimum potential
- `tileformer_w0_STD_PSI_MAX` - Window 0, standard config, maximum potential
- `tileformer_w0_STD_PSI_MEAN` - Window 0, standard config, average potential
- `tileformer_w0_ENH_PSI_MIN` - Window 0, enhanced config, minimum potential
- `tileformer_w0_ENH_PSI_MAX` - Window 0, enhanced config, maximum potential
- `tileformer_w0_ENH_PSI_MEAN` - Window 0, enhanced config, average potential
- ...
- `tileformer_w21_ENH_PSI_MEAN` - Window 21 (last window for 230bp)

---

## Data Format

### Example for 230bp sequence:

| Column | Description |
|--------|-------------|
| `sequence` | DNA sequence (230bp) |
| `tileformer_w0_STD_PSI_MIN` | Window 0 (bp 1-20), STD minimum |
| `tileformer_w0_STD_PSI_MAX` | Window 0, STD maximum |
| `tileformer_w0_STD_PSI_MEAN` | Window 0, STD mean |
| `tileformer_w0_ENH_PSI_MIN` | Window 0, ENH minimum |
| `tileformer_w0_ENH_PSI_MAX` | Window 0, ENH maximum |
| `tileformer_w0_ENH_PSI_MEAN` | Window 0, ENH mean |
| `tileformer_w1_STD_PSI_MIN` | Window 1 (bp 11-30), STD minimum |
| ... | ... |
| `tileformer_w21_ENH_PSI_MEAN` | Window 21 (bp 211-230), ENH mean |

**Total columns**: 6 metadata + 132 TileFormer features = 138 columns

---

## Current Labeled Datasets

**Location**: `/home/bcheng/sequence_optimization/mainproject/PhysiFormer/physpreprocess/output/`

**Available files**:
- `HepG2_train_tileformer.tsv` (62,453 sequences, 132 features)
- `HepG2_val_tileformer.tsv` (7,807 sequences, 132 features)
- `HepG2_test_tileformer.tsv` (7,807 sequences, 132 features)
- `K562_train_tileformer.tsv` (132 features)
- `K562_val_tileformer.tsv` (132 features)
- `K562_test_tileformer.tsv` (132 features)
- `WTC11_train_tileformer.tsv` (132 features)
- `WTC11_val_tileformer.tsv` (132 features)
- `WTC11_test_tileformer.tsv` (132 features)
- `S2_train_tileformer.tsv` (138 features - 249bp sequences)
- `S2_val_tileformer.tsv` (138 features)
- `S2_test_tileformer.tsv` (138 features)

---

## Why Two Configurations?

### Complementary Information

1. **STD (Physiological)**:
   - Reflects real biological conditions
   - Captures functionally relevant electrostatics
   - Less sensitive to minor sequence variations
   - Relevant for predicting in vivo behavior

2. **ENH (High Sensitivity)**:
   - Amplifies subtle sequence differences
   - Better for sequence discrimination
   - Reveals intrinsic structural properties
   - Useful for optimization algorithms

### Combined Power

Using both configurations provides:
- **Biological relevance** (STD) + **Discriminative power** (ENH)
- Robustness across different ionic conditions
- Richer feature space for machine learning
- Better generalization for sequence design

---

## Relationship to PhysInformer

**Important Note**: PhysInformer models were **NOT** trained on TileFormer features.

**PhysInformer training data**:
- WTC11: 539 biophysical descriptors (PWM, thermodynamics, stiffness, bending, entropy, advanced)
- HepG2: 536 descriptors
- K562: 515 descriptors
- S2: 386 descriptors

**TileFormer features**: Generated separately for potential use in:
- Sequence optimization algorithms
- Downstream analysis
- Multi-modal learning
- Feature ablation studies

---

## Computational Method: ABPS (Adaptive Poisson-Boltzmann Solver)

### Pipeline Overview

1. **Structure Generation**: TLEaP (AmberTools) builds canonical B-DNA from sequence
2. **Charge Assignment**: PDB2PQR assigns atomic charges and radii
3. **Electrostatic Solving**: APBS solves Poisson-Boltzmann equation on 3D grid
4. **Feature Extraction**: Extract min/max/mean from potential grid

### Poisson-Boltzmann Equation

```
∇·[ε(r)∇ψ(r)] - κ²(r)sinh[ψ(r)] = -4πρ(r)

where:
  ψ(r)  = electrostatic potential at position r
  ε(r)  = dielectric constant (2.0 for DNA, 78.5 for water)
  κ²(r) = ionic screening factor (depends on salt concentration)
  ρ(r)  = charge density from DNA atoms
```

### GPU Acceleration

- Uses CUDA + AMGX solver
- ~20× speedup vs CPU
- Enables processing of large datasets

---

## Key Insights

### 1. Sequence-Dependent Electrostatics

Different sequences produce different electrostatic profiles:
- **AT-rich**: Narrower minor groove → more negative potential
- **GC-rich**: Wider minor groove → less negative potential
- **CpG sites**: Enhanced negative potential in minor groove
- **Palindromes**: Symmetric electrostatic patterns

### 2. Multi-Scale Information

- **MIN**: Captures most negative regions (often minor groove)
- **MAX**: Captures least negative regions (often major groove)
- **MEAN**: Overall electrostatic character

### 3. Positional Resolution

With 22 windows and 10bp stride:
- **High spatial resolution**: ~every 10bp along sequence
- **Overlapping context**: Each position covered by 2 windows
- **Full coverage**: All 230bp positions included

---

## Applications

### 1. Sequence Optimization
- Optimize electrostatics for protein binding
- Design sequences with specific potential profiles
- Balance STD (biological) vs ENH (discriminative) objectives

### 2. Regulatory Analysis
- Compare electrostatics of active vs inactive promoters
- Identify electrostatic signatures of regulatory elements
- Correlate with transcription factor binding

### 3. Evolutionary Studies
- Compare electrostatic conservation across orthologs
- Identify electrostatic constraints on evolution
- Detect compensatory mutations maintaining potential

### 4. Structure-Function Relationships
- Link sequence → electrostatics → function
- Predict binding modes from potential patterns
- Design synthetic sequences with desired properties

---

## References

### TileFormer Model
- Architecture: Transformer encoder (256-dim, 8 heads, 6 layers)
- Training: 25 epochs, early stopping at epoch 22
- Best checkpoint: `/home/bcheng/sequence_optimization/mainproject/PhysiFormer/physpreprocess/TileFormer_model/checkpoints/run_20250819_063725/best_model.pth`

### ABPS Method
- Software: APBS 3.0+ with GPU support
- Structure: AmberTools TLEaP (ff14SB forcefield)
- Charge model: PDB2PQR (Amber charges + radii)

### Documentation Files
- Performance: `results/TileFormer/PERFORMANCE_SUMMARY.txt`
- Metrics: `results/TileFormer/ELECTROSTATIC_METRICS.txt`
- Model location: `results/TileFormer/BEST_MODEL_LOCATION.txt`

---

## Summary

**TileFormer provides 6 electrostatic features per 20bp window:**

| Feature Set | Configuration | Purpose |
|-------------|---------------|---------|
| STD_PSI_MIN/MAX/MEAN | Physiological (150 mM salt) | Biological relevance |
| ENH_PSI_MIN/MAX/MEAN | Enhanced (10 mM salt) | High sensitivity |

**For full sequences:**
- 230bp → 132 features (22 windows × 6 values)
- 249bp → 138 features (23 windows × 6 values)
- 110bp → 60 features (10 windows × 6 values)

**Performance:** Pearson R > 0.98 on all 6 features for 20bp test sequences

**Current status:** All human cell type datasets (HepG2, K562, WTC11) and S2 (Drosophila) have been labeled with TileFormer features and saved in `output/` directory.

---

**Document created**: 2025-11-12
**Location**: `/home/bcheng/sequence_optimization/mainproject/PhysiFormer/physpreprocess/TILEFORMER_ELECTROSTATICS_EXPLAINED.md`
