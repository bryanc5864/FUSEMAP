# FUSEMAP: Foundation Universal Sequence-Expression Model for Activity Prediction

A comprehensive framework for predicting regulatory element activity across species using deep learning and biophysical features.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Model Architectures](#model-architectures)
3. [Datasets](#datasets)
4. [Physics Features](#physics-features)
5. [Training Configurations](#training-configurations)
6. [Best Models & Checkpoints](#best-models--checkpoints)
7. [Validation Results](#validation-results)
8. [Key Findings](#key-findings)
9. [File Structure](#file-structure)
10. [Usage Examples](#usage-examples)

---

## Project Overview

FUSEMAP integrates four complementary model families for regulatory sequence analysis:

| Model | Purpose | Key Metric | Best Performance |
|-------|---------|------------|------------------|
| **CADENCE** | Sequence → Activity prediction | Pearson r | 0.81 (K562) |
| **PhysInformer** | Sequence → Physics features | Pearson r | 0.94 (within-cell) |
| **PhysicsVAE** | Physics-conditioned sequence reconstruction | Accuracy | 56% (within-human) |
| **PhysicsTransfer** | Physics → Activity transfer learning | Pearson r | 0.70 (plant zero-shot) |

**Total: ~8.6M sequences across 10+ datasets, 540+ biophysical features**

---

## Model Architectures

### 1. CADENCE (LegNet-based)

Primary sequence-to-activity prediction model.

**Architecture:**
```
Input: [batch, 4, 230] one-hot DNA
    ↓
Stem: Conv1d(4→64, k=11) + BatchNorm + SiLU
    ↓
Block 1: EffBlock(64→80) + MaxPool(2)  [230→115]
Block 2: EffBlock(80→96) + MaxPool(2)  [115→57]
Block 3: EffBlock(96→112) + MaxPool(2) [57→28]
Block 4: EffBlock(112→128) + MaxPool(2) [28→14]
    ↓
Mapper: Conv1d(128→256, k=1)
    ↓
GlobalAvgPool → Linear(256→256) → Linear(256→1)
    ↓
Output: Scalar activity prediction
```

**Parameters:** ~330,000 (base LegNet)

**Optional Modules:**
- RC-Equivariant Stem (strand symmetry)
- Kingdom-specific stems
- Species/cell-type embeddings
- PLACE uncertainty quantification

**Key Files:**
- Architecture: `models/CADENCE/cadence.py`
- Training: `training/trainer.py`
- Config: `training/config.py`

---

### 2. PhysInformer (Transformer)

Predicts 500+ biophysical features from sequence.

**Architecture:**
```
Input: DNA sequence (230-249bp)
    ↓
Nucleotide Embedding (vocab=5, d_model=512)
    ↓
Sinusoidal Positional Encoding
    ↓
8× Transformer Encoder Layers
  - 8 attention heads
  - FFN: 512→2048→512
  - Dropout: 0.1
    ↓
Global Average Pooling
    ↓
Per-Feature Heads (498-537 separate MLPs)
  - Linear(512→256→64→1) per feature
    ↓
Output: Physics feature predictions
```

**Auxiliary Heads:**
- **Head A**: Sequence + Physics → Activity
- **Head B**: Physics Only → Activity (MLP: features→384→192→96→1)

**Key Files:**
- Architecture: `physics/PhysInformer/physics_aware_model.py`
- Training: `physics/PhysInformer/train.py`

---

### 3. PhysicsVAE

Physics-conditioned variational autoencoder for sequence generation.

**Architecture:**
```
Encoder:
  Sequence: Conv stack (4→64→128→256) → μ, log-σ (latent_dim=128)
  Physics: Dense (n_physics→256→128→64)

Decoder:
  Concat: z (128) + physics_cond (64) = 192
  4× Transformer encoder layers (4 heads)
  Output: Nucleotide logits [batch, seq_len, 4]

Loss: L_recon + β·L_KL + γ·L_physics
  - β = 0.001 (weak KL)
  - γ = 0.1 (physics consistency)
```

**Parameters:** 10,767,748

**Key Files:**
- Architecture: `physics/PhysicsVAE/models/physics_vae.py`
- Training: `physics/PhysicsVAE/train.py`

---

### 4. PhysicsTransfer

Transfer learning using physics features as universal representation.

**Probe Models:**
- ElasticNet (α=0.01, l1_ratio=0.5)
- Ridge (α=0.01)
- Lasso (α=0.01)
- MLP (128→64 hidden)

**Transfer Protocols:**
1. **Zero-shot**: Train on source, apply to target
2. **Fine-tuned**: Zero-shot + target data adaptation
3. **Multi-species**: Shared encoder, species-specific heads

**Key Files:**
- Probe: `physics/PhysicsTransfer/physics_probe.py`
- Experiments: `physics/PhysicsTransfer/run_transfer.py`

---

## Datasets

### Summary Table

| Dataset | Organism | Cell Type | Sequences | Length | Source |
|---------|----------|-----------|-----------|--------|--------|
| ENCODE4 K562 | Human | Erythroid | 226,258 | 230bp | ENCODE |
| ENCODE4 HepG2 | Human | Liver | 139,881 | 230bp | ENCODE |
| ENCODE4 WTC11 | Human | iPSC | 55,993 | 230bp | ENCODE |
| DeepSTARR | Drosophila | S2 | 484,056 | 249bp | Stark Lab |
| Jores Tobacco | Plants (3 spp) | Leaf | 72,158 | 170bp | Jores 2021 |
| Jores Maize | Plants (3 spp) | Protoplast | 75,808 | 170bp | Jores 2021 |
| Yeast | S. cerevisiae | - | 6,810,364 | 110bp | GSE163045 |
| Mouse ESC | Mouse | ESC | 27,566 | 230bp | GSE143546 |

**Total: ~8.6M sequences**

### Human Cell Lines (lentiMPRA)

**K562 (Erythroid)**
- Location: `data/lentiMPRA_data/K562/`
- Train: 158,377 | Val: 22,613 | Test: 22,631
- Activity: log2(RNA/DNA)
- ENCODE ID: ENCFF252GNM

**HepG2 (Liver)**
- Location: `data/lentiMPRA_data/HepG2/`
- Train: 97,925 | Val: 13,997 | Test: 13,953
- ENCODE ID: ENCFF755BGY

**WTC11 (iPSC)**
- Location: `data/lentiMPRA_data/WTC11/`
- Train: 39,201 | Val: 5,596 | Test: 5,597
- ENCODE ID: ENCFF949GFZ

### DeepSTARR (Drosophila S2)

- Location: `data/S2_data/`
- Train: 352,009 | Val: 40,570 | Test: 41,186
- Activities: Dev_log2_enrichment, Hk_log2_enrichment
- Source: Avsec et al., Nature Methods 2021

### Plant Promoters (Jores 2021)

**Tobacco Leaf Assay**
- Location: `data/plant_data/processed/tobacco_leaf/`
- Species: Arabidopsis (13,169) + Maize (24,209) + Sorghum (19,502)
- Activity range: [-4.44, 5.47]

**Maize Protoplast Assay**
- Location: `data/plant_data/processed/maize_protoplast/`
- Species: Arabidopsis (13,796) + Maize (25,722) + Sorghum (20,170)
- Activity range: [-5.41, 5.56]

### External Validation

**Mouse ESC STARR-seq (GSE143546)**
- Location: `external_validation/processed/mouse_esc_sequences.csv`
- 27,566 sequences, 230bp
- Conditions: 2iL (ground-state) and SL (metastable)

---

## Physics Features

### Feature Categories (540+ total)

| Category | Count | Description |
|----------|-------|-------------|
| **Thermodynamic** | 42 | ΔH, ΔS, ΔG, melting temperature |
| **Stiffness** | 62 | Deformation energy, twist/tilt/roll |
| **Entropy** | 62 | Shannon, k-mer entropy, complexity |
| **Bending** | 44 | Curvature, hotspots, spectral |
| **Advanced** | 53 | Melting, groove width, G-quadruplex |
| **PWM/TF** | 253-277 | Cell-type specific TF binding |

### Thermodynamic Features (42)
- Global: total ΔH, ΔS, ΔG at 37°C
- Statistics: mean, variance, percentiles (p5-p95)
- Melting: Tm in °C and K
- Source: SantaLucia nearest-neighbor parameters

### Stiffness Features (62)
- Per-mode energies: twist, tilt, roll, shift, slide, rise
- PCA projections and z-scores
- Energy distribution entropy
- Source: Olson matrix parameters

### Entropy Features (62)
- Global: Shannon, Rényi (α=2), k-mer (1-6)
- Windowed: w=10, 30, 50bp
- Complexity: Lempel-Ziv, conditional entropy
- Mutual information at distances 1-10bp

### PWM Features (per TF)
- max_score, delta_g, mean_score, var_score
- total_weight, num_high_affinity, entropy, top_k_mean
- Source: JASPAR 2024 (18 universal + cell-type specific)

### Feature Files
- Location: `physics/output/`
- Format: `{CELLTYPE}_{split}_descriptors.tsv`
- Human: 516-540 features | Plants: 372 features

---

## Training Configurations

### CADENCE Configurations

| Config | Datasets | Key Features | Best Model |
|--------|----------|--------------|------------|
| Config 1 | Single cell type | Pure LegNet | cadence_k562_v2 |
| Config 2 | Multi-human | Cell-type embeddings | - |
| Config 3 | Cross-animal | Species embeddings | - |
| Config 4 | Cross-kingdom | Kingdom stems | config4_cross_kingdom_v1 |
| Config 5 | Universal (7 datasets) | All embeddings + PLACE | config5_universal_no_yeast |

### Training Hyperparameters (Config 5)

```yaml
max_epochs: 150 (3 phases)
batch_size: 256
learning_rate: 0.001
weight_decay: 1e-5
scheduler: cosine
warmup_epochs: 5
gradient_clip: 1.0
use_amp: true
dropout: 0.3
```

### PhysInformer Training

```yaml
epochs: 100
batch_size: 64
learning_rate: 0.0001
scheduler: cosine_annealing
d_model: 512
n_layers: 8
n_heads: 8
```

---

## Best Models & Checkpoints

### CADENCE Single-Species Models (22 total)

**Human Cell Lines:**
| Model | Path | Val Pearson | Test Pearson | Test Spearman |
|-------|------|-------------|--------------|---------------|
| K562 v2 (best) | `results/cadence_k562_v2/best_model.pt` | **0.813** | 0.809 | 0.759 |
| K562 all v2 | `results/cadence_k562_all_v2/best_model.pt` | 0.808 | - | - |
| HepG2 v2 | `results/cadence_hepg2_v2/best_model.pt` | 0.787 | 0.786 | 0.770 |
| HepG2 all v2 | `results/cadence_hepg2_all_v2/best_model.pt` | 0.781 | - | - |
| WTC11 v2 | `results/cadence_wtc11_v2/best_model.pt` | 0.659 | 0.698 | 0.591 |
| WTC11 all v2 | `results/cadence_wtc11_all_v2/best_model.pt` | 0.656 | - | - |

**Drosophila S2 (DeepSTARR):**
| Model | Path | Dev Val r | Dev Test r | Hk Val r | Hk Test r |
|-------|------|-----------|------------|----------|-----------|
| DeepSTARR v2 | `training/results/cadence_deepstarr_v2/best_model.pt` | 0.906 | **0.909** | 0.918 | **0.920** |
| DeepSTARR all v2 | `training/results/cadence_deepstarr_all_v2/best_model.pt` | 0.920 | - | 0.930 | - |

**Plant Species (Jores 2021):**
| Model | Path | Leaf Val r | Leaf Spearman | Proto Val r |
|-------|------|------------|---------------|-------------|
| Arabidopsis v1 | `training/results/cadence_arabidopsis_v1/best_model.pt` | 0.618 | - | 0.508 |
| Maize v1 | `training/results/cadence_maize_v1/best_model.pt` | **0.796** | 0.799 | 0.767 |
| Sorghum v1 | `training/results/cadence_sorghum_v1/best_model.pt` | **0.782** | - | 0.769 |
| Arabidopsis all v1 | `training/results/cadence_arabidopsis_all_v1/best_model.pt` | - | - | - |
| Maize all v1 | `training/results/cadence_maize_all_v1/best_model.pt` | - | - | - |
| Sorghum all v1 | `training/results/cadence_sorghum_all_v1/best_model.pt` | - | - | - |

**Yeast (DREAM Challenge):**
| Model | Path | Val Pearson | Test Pearson | Test Spearman |
|-------|------|-------------|--------------|---------------|
| Yeast v1 | `training/results/cadence_yeast_v1/best_model.pt` | 0.580 | **0.734** | 0.738 |
| Yeast all v1 | `training/results/cadence_yeast_all_v1/best_model.pt` | - | - | - |

### CADENCE Cross-Species Models

| Config | Datasets | Path | Best r |
|--------|----------|------|--------|
| Config 2 | K562+HepG2+WTC11 | `training/results/config2_multi_celltype_v1/` | 0.657 (HepG2) |
| Config 3 | Human+Drosophila | `training/results/config3_cross_animal_v1/` | 0.762 (Hk) |
| Config 4 | Animals+Plants | `results/config4_cross_kingdom_v1/` | 0.711 (K562) |
| Config 5 | Universal (7 datasets) | `results/config5_universal_no_yeast_*/` | 0.786 (Maize) |

### PLACE-Calibrated Models (20 total)

All models with uncertainty quantification in `cadence_place/`:
- Single-species: K562, HepG2, WTC11, DeepSTARR, Arabidopsis, Maize, Sorghum, Yeast
- All-features variants: *_all_v2 for each

### PhysInformer Models

| Model | Path | Overall r |
|-------|------|-----------|
| K562 | `physics/PhysInformer/runs/K562_20250829_095741/best_model.pt` | 0.938 |
| HepG2 | `physics/PhysInformer/runs/HepG2_20250829_095749/best_model.pt` | 0.915 |
| WTC11 | `physics/PhysInformer/runs/WTC11_20250829_095738/best_model.pt` | 0.921 |
| S2 | `physics/PhysInformer/runs/S2_20260114_062536/best_model.pt` | 0.924 |
| Arabidopsis | `physics/PhysInformer/runs/arabidopsis_*/best_model.pt` | 0.90+ |
| Maize | `physics/PhysInformer/runs/maize_*/best_model.pt` | 0.90+ |
| Sorghum | `physics/PhysInformer/runs/sorghum_*/best_model.pt` | 0.90+ |

### PhysicsVAE Models

| Model | Path | Accuracy |
|-------|------|----------|
| K562 | `physics/PhysicsVAE/runs/K562_20260113_051653/best_model.pt` | 55.6% |
| HepG2 | `physics/PhysicsVAE/runs/HepG2_20260113_052418/best_model.pt` | 52.0% |
| WTC11 | `physics/PhysicsVAE/runs/WTC11_20260113_052743/best_model.pt` | 52.0% |
| S2 | `physics/PhysicsVAE/runs/S2_*/best_model.pt` | 51.0% |

### LegatoV2 Models

| Model | Path | Pearson r | R² |
|-------|------|-----------|-----|
| S2 Advanced | `models/legatoV2/runs/s2_advanced_256_2/` | 0.857 | 0.710 |
| S2 Pro | `models/legatoV2/runs/s2_pro_256_raw/` | 0.854 | 0.709 |
| DREAM Pro | `models/legatoV2/outputs/dream_pro_dream_*/` | 0.958 | - |

---

## Benchmark Comparisons

### CADENCE vs LegNet (Human MPRA)

Comparison using identical training data, hyperparameters (batch_size=1024, max_lr=0.01, weight_decay=0.1, 30 epochs):

| Cell Type | CADENCE Test r | LegNet Test r | Difference | Verdict |
|-----------|---------------|---------------|------------|---------|
| K562 | 0.809 | 0.811 | -0.002 | Comparable |
| HepG2 | 0.786 | 0.783 | +0.003 | CADENCE slightly better |
| WTC11 | 0.698 | 0.698 | 0.000 | Identical |

**Conclusion:** CADENCE matches LegNet on human MPRA, suggesting both architectures are at or near the performance ceiling for this task.

### CADENCE vs DREAM-RNN (DeepSTARR)

Comparison on Drosophila S2 STARR-seq data (DREAM-RNN: LSTM architecture, 30 epochs, batch_size=128):

| Output | CADENCE r | DREAM-RNN r | Improvement |
|--------|-----------|-------------|-------------|
| Dev | **0.909** | 0.708 | +0.201 (+28.4%) |
| Hk | **0.920** | 0.779 | +0.141 (+18.1%) |

**Conclusion:** CADENCE significantly outperforms DREAM-RNN, demonstrating the advantage of convolutional (LegNet) architecture over recurrent models on enhancer activity prediction.

### Cross-Organism Performance Summary

| Organism | Model | Best r | Dataset |
|----------|-------|--------|---------|
| Human | CADENCE K562 | 0.813 | K562 MPRA |
| Drosophila | CADENCE DeepSTARR | 0.920 | S2 STARR-seq |
| Maize | CADENCE Maize | 0.796 | Jores leaf |
| Sorghum | CADENCE Sorghum | 0.782 | Jores leaf |
| Arabidopsis | CADENCE Arabidopsis | 0.618 | Jores leaf |
| Yeast | CADENCE Yeast | 0.734 | DREAM test |

---

## Physics Analyses

### PhysicsInterpreter Suite

**Location:** `physics/PhysicsInterpreter/`

| Analysis | Method | Key Finding |
|----------|--------|-------------|
| Attribution | Linear probe decomposition | Physics mediates 30-50% of activity |
| Integrated Gradients | Position-wise importance | Sequence-level attribution |
| Mediation | Baron & Kenny causal | Indirect effects through physics |
| Landscape | SHAP + correlations | Nonlinear effects present |

### Multivariate Analyses (6-part suite)

**Location:** `physics/analyses/`

| Analysis | Purpose | Key Result |
|----------|---------|------------|
| 01_univariate_stability | Feature correlations | 1,140 significant features (p<0.05) |
| 02_multivariate_models | Physics-only prediction | R²=0.178 (physics+PWM) |
| 03_incremental_control | PWM contribution | +131% improvement with all TFs |
| 04_interaction_mapping | Physics×TF interactions | H-statistic identifies key pairs |
| 05_regime_discovery | HDBSCAN clustering | 3-8 physics regimes per cell type |
| 06_cross_celltype | Transfer analysis | Poor physics transfer (R²=0.01-0.07) |

### Key Physics Discoveries

1. **DNA Opening is Top Predictor:** `stress_mean_stress_opening` r=-0.267
2. **Minor Groove Width Critical:** `mgw_mean_mgw` r=+0.267
3. **DNA Bending Facilitates Function:** All bending features r~-0.24
4. **Thermodynamic Instability Drives Expression:** Lower ΔG → higher activity
5. **Sequence Simplicity Over Complexity:** Higher entropy → lower expression

### Feature Importance by Category

| Category | Top Feature | Correlation | Model learns? |
|----------|-------------|-------------|---------------|
| Advanced | stress_mean_stress_opening | -0.267 | r>0.999 |
| Advanced | mgw_mean_mgw | +0.267 | r>0.999 |
| Thermo | total_dH | -0.273 | r>0.99 |
| Bending | curvature_gradient_mean | -0.244 | r>0.99 |
| Entropy | gc_entropy | -0.143 | r>0.98 |
| Stiffness | gc_content_global | +0.253 | r>0.99 |

---

## Validation Results

### CADENCE Transfer Learning (46 experiments)

**Best Results by Target:**

| Source → Target | Data | Strategy | Spearman ρ | Pearson r |
|-----------------|------|----------|------------|-----------|
| K562 → S2 Drosophila | 25% | Full fine-tune | **0.556** | 0.579 |
| HepG2 → S2 Drosophila | 25% | Full fine-tune | 0.524 | 0.543 |
| WTC11 → Mouse ESC | 25% | Full fine-tune | **0.281** | 0.316 |
| K562 → Mouse ESC | 25% | Full fine-tune | 0.216 | 0.250 |

**Key Findings:**
- Full fine-tuning > Frozen backbone (ρ=0.29 vs 0.14)
- More data helps: 1%→25% improves ρ from 0.16 to 0.26
- Drosophila transfer works well (ρ up to 0.56)

### PhysInformer Zero-Shot Transfer (18 experiments)

**Mean Pearson r by Source Model:**

| Source | Mean r | Best Target |
|--------|--------|-------------|
| PhysInformer_K562 | **0.74** | HepG2 (0.85) |
| PhysInformer_WTC11 | 0.52 | K562 (0.83) |
| PhysInformer_HepG2 | 0.51 | K562 (0.66) |

**Feature Category Transferability:**

| Category | Mean r | Interpretation |
|----------|--------|----------------|
| Advanced | **0.79** | Excellent (melting, stacking) |
| Bending | **0.75** | Excellent (DNA curvature) |
| Entropy | 0.47 | Moderate |
| Stiffness | 0.43 | Moderate |
| PWM | 0.27 | Poor (TF-specific) |

### PhysicsVAE Zero-Shot Transfer (18 experiments)

**Reconstruction Accuracy by Target:**

| Target | Mean Accuracy | Notes |
|--------|---------------|-------|
| WTC11 | 53.8% | Best within-human |
| HepG2 | 53.3% | Good |
| K562 | 51.0% | Good |
| S2 | 29.2% | Cross-species degradation |
| Plants | 25-26% | Near random |

**Key Finding:** Within-human transfer works (50-56%), cross-species fails (~25%)

### Physics → Activity Transfer

**Within-Dataset (aux_head_b):**
- Human cells: r = 0.55-0.61
- Plants: r = 0.70-0.80
- S2: r = 0.52

**Zero-Shot Cross-Species:**
- Within-human: r = 0.38 (moderate)
- Human → Drosophila: r ≈ 0 (**no transfer**)

---

## Key Findings

### What Transfers Across Species?

| Property | Transfers? | Evidence |
|----------|------------|----------|
| DNA bending | **Yes** | r = 0.75 |
| DNA melting/stacking | **Yes** | r = 0.79 |
| Sequence entropy | Partial | r = 0.47 |
| DNA stiffness | Partial | r = 0.43 |
| TF binding motifs | **No** | r = 0.27 |
| Physics→Activity mapping | **No** | r ≈ 0 cross-species |

### Model Rankings

| Rank | Model | Score | Best Use Case |
|------|-------|-------|---------------|
| 1 | CADENCE_K562 | r=0.81 | Single cell-type prediction |
| 2 | PhysInformer_K562 | r=0.74 | Physics feature prediction |
| 3 | Config5_Universal | r=0.62 | Multi-species (trades accuracy) |

### Transfer Learning Insights

1. **Fine-tuning is essential**: Full fine-tune achieves 2x improvement over frozen
2. **K562 model transfers best**: Most robust source for transfer
3. **Plants are special**: High within-kingdom transfer (0.70), cross-kingdom fails
4. **Biophysics is universal**: Bending, melting properties conserved
5. **Regulatory logic is not**: TF binding and activity mapping are species-specific

---

## File Structure

```
FUSEMAP/
├── models/
│   ├── CADENCE/              # Main sequence→activity model
│   │   ├── cadence.py        # Architecture (330K params)
│   │   └── place_uncertainty.py
│   └── legatoV2/             # Alternative architecture
│
├── physics/
│   ├── PhysInformer/         # Sequence→physics transformer
│   │   ├── physics_aware_model.py
│   │   └── runs/             # Checkpoints
│   ├── PhysicsVAE/           # Physics-conditioned VAE
│   │   ├── models/physics_vae.py
│   │   └── runs/
│   ├── PhysicsTransfer/      # Transfer learning probes
│   └── output/               # Computed descriptors (540+ features)
│
├── training/
│   ├── config.py             # All CADENCE configurations
│   ├── trainer.py            # Training loop
│   └── coordinator.py        # Multi-phase training
│
├── results/
│   ├── cadence_k562_v2/      # Best single-cell model
│   ├── cadence_hepg2_v2/
│   ├── cadence_wtc11_v2/
│   ├── config4_cross_kingdom_v1/
│   └── config5_universal_*/  # Universal foundation model
│
├── external_validation/
│   ├── processed/            # External test datasets
│   ├── comprehensive_validation/
│   │   ├── run_cadence_transfer.py
│   │   ├── run_physicsvae_transfer.py
│   │   └── run_physinformer_transfer.py
│   └── results/              # 88+ validation experiments
│
├── data/
│   ├── lentiMPRA_data/       # Human ENCODE4 (K562, HepG2, WTC11)
│   ├── S2_data/              # Drosophila DeepSTARR
│   ├── plant_data/           # Jores 2021 (3 species)
│   ├── yeast_data/           # 6.8M yeast promoters
│   └── motifs/               # JASPAR 2024 (2,148 motifs)
│
└── cadence_place/            # PLACE uncertainty calibration
    └── config5_universal_no_yeast/
```

---

## Usage Examples

### Load CADENCE Model
```python
import torch
from models.CADENCE.cadence import CADENCE, CADENCEConfig

config = CADENCEConfig(num_outputs=1)
model = CADENCE(config)
checkpoint = torch.load('results/cadence_k562_v2/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict
seq_onehot = torch.randn(32, 4, 230)  # [batch, channels, length]
activity = model(seq_onehot)
```

### Load PhysInformer
```python
from physics.PhysInformer.physics_aware_model import create_physics_aware_model

model = create_physics_aware_model(
    n_features=498,
    d_model=512,
    n_layers=8,
    n_heads=8
)
checkpoint = torch.load('physics/PhysInformer/runs/K562_*/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Load Physics Features
```python
import pandas as pd

# Load precomputed descriptors
descriptors = pd.read_csv('physics/output/K562_train_descriptors.tsv', sep='\t')
print(f"Features: {len(descriptors.columns)}")  # 516 for K562
```

### Run Transfer Experiment
```python
# CADENCE transfer
python external_validation/comprehensive_validation/run_cadence_transfer.py \
    --source cadence_k562_v2 \
    --target mouse_esc \
    --fraction 0.05 \
    --strategy full_finetune

# PhysInformer zero-shot
python external_validation/comprehensive_validation/run_physinformer_transfer.py \
    --source K562 \
    --target HepG2
```

---

## References

1. **ENCODE4 lentiMPRA**: ENCODE Consortium (2020)
2. **DeepSTARR**: Avsec et al. (2021) Nature Methods
3. **Jores Plant Promoters**: Jores et al. (2021) Nature Plants
4. **Yeast Promoters**: GSE163045
5. **Mouse ESC**: Peng et al. (2020) Genome Biology
6. **LegNet**: Penzar et al. (2023)
7. **JASPAR 2024**: Castro-Mondragon et al. (2024)

---

*Last updated: January 2026*
