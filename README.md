# FUSEMAP: Biophysics-Informed Deep Learning for Regulatory DNA Prediction and Design

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**FUSEMAP** is the first comprehensive framework bridging the gap between deep learning predictions and in vivo performance for cis-regulatory element (CRE) design. By integrating biophysics into regulatory DNA prediction and design, FUSEMAP enables reliable synthetic enhancer and promoter engineering across species.

## Key Results

- **840+ experiments** across 7 species (human, fly, mouse, yeast, Arabidopsis, maize, sorghum)
- **State-of-the-art prediction**: CADENCE achieves r=0.81 on human cell lines, r=0.92 on Drosophila
- **First zero-shot cross-species transfer**: S2A achieves ρ=0.70 plant-to-plant transfer
- **10,000× speedup**: TileFormer predicts electrostatic potentials at R²>0.96
- **99% therapeutic specificity**: HepG2-targeted enhancers with built-in validation

## Framework Components

FUSEMAP comprises six synergistic modules:

| Module | Description | Key Metric |
|--------|-------------|------------|
| **CADENCE** | Sequence-to-activity prediction (LegNet-based CNN) | r=0.81 (K562), r=0.92 (DeepSTARR) |
| **PhysInformer** | Sequence-to-physics transformer (500+ biophysical features) | r=0.92 validation |
| **TileFormer** | Electrostatic potential surrogate model | R²=0.96, 10,000× speedup |
| **S2A** | Zero-shot cross-species activity transfer | ρ=0.70 (plant-to-plant) |
| **PhysicsVAE** | Inverse design with targeted biophysical profiles | 64% reconstruction |
| **PLACE** | Post-hoc uncertainty quantification | Calibrated confidence intervals |

## Repository Structure

```
FUSEMAP/
├── training/           # CADENCE training framework
│   ├── config.py       # Dataset and model configurations
│   ├── trainer.py      # Training loop and evaluation
│   ├── datasets.py     # Data loading utilities
│   └── results/        # Training outputs (not in repo)
├── physics/
│   ├── PhysInformer/   # Sequence-to-physics transformer
│   ├── TileFormer/     # Electrostatic surrogate model
│   ├── PhysicsVAE/     # Generative model for CRE design
│   └── PhysicsTransfer/# Cross-species transfer experiments
├── applications/       # Downstream applications
│   ├── therapeutic_design/    # Cell-type-specific enhancer design
│   ├── variant_interpretation/# ClinVar variant analysis
│   └── enhancer_optimization/ # Gradient-based optimization
├── scripts/            # Experiment scripts
├── FUSEMAP_results/    # Aggregated results (JSON, logs)
└── paper/              # Manuscript and figures
```

## Installation

```bash
# Clone repository
git clone https://github.com/bryanc5864/FUSEMAP.git
cd FUSEMAP

# Create conda environment
conda create -n fusemap python=3.10
conda activate fusemap

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas scipy scikit-learn
pip install matplotlib seaborn
```

## Quick Start

### Training CADENCE on Human MPRA Data

```python
from training.config import get_config1_single_celltype
from training.trainer import Trainer

# Get configuration for K562
config = get_config1_single_celltype("encode4_k562")

# Initialize and train
trainer = Trainer(config)
trainer.train()
```

### Predicting Biophysical Features with PhysInformer

```python
from physics.PhysInformer.model import PhysInformer

model = PhysInformer.load_pretrained("K562")
features = model.predict(sequences)  # Returns 500+ features
```

### Zero-Shot Cross-Species Transfer with S2A

```python
from applications.s2a import S2APredictor

predictor = S2APredictor(source_species=["arabidopsis", "sorghum"])
activity = predictor.predict(maize_sequences)  # Zero-shot prediction
```

## Datasets

FUSEMAP was trained and evaluated on:

| Dataset | Species | Sequences | Sequence Length |
|---------|---------|-----------|-----------------|
| ENCODE4 lentiMPRA | Human (K562, HepG2, WTC11) | 483K | 230 bp |
| DeepSTARR | Drosophila S2 | 485K | 249 bp |
| Jores et al. 2021 | Arabidopsis, Maize, Sorghum | 51K | 170 bp |
| DREAM Challenge | Yeast | 6.7M | 110 bp |

## Results Summary

### CADENCE Activity Prediction

| Cell Type | Pearson r | Spearman ρ | R² |
|-----------|-----------|------------|-----|
| K562 | 0.809 | 0.759 | 0.652 |
| HepG2 | 0.786 | 0.770 | 0.613 |
| WTC11 | 0.698 | 0.591 | 0.472 |
| DeepSTARR Dev | 0.909 | 0.889 | 0.822 |
| DeepSTARR Hk | 0.920 | 0.863 | 0.846 |
| Maize (leaf) | 0.796 | 0.799 | 0.568 |

### PhysInformer Transfer (Mean Pearson r)

| Source → Target | Overlap | Mean r | Median r |
|-----------------|---------|--------|----------|
| K562 → HepG2 | 411 | 0.847 | 0.968 |
| K562 → WTC11 | 411 | 0.839 | 0.971 |
| K562 → S2 | 267 | 0.729 | 0.901 |
| K562 → Arabidopsis | 267 | 0.656 | 0.722 |

### S2A Zero-Shot Transfer

| Scenario | Spearman ρ |
|----------|------------|
| Within-Plant (→ Maize) | **0.700** |
| Within-Human (→ WTC11) | 0.260 |
| Plant → Animal | 0.125 |
| Animal → Plant | -0.321 |

### TileFormer Electrostatic Prediction

| Target | R² | Pearson r | RMSE |
|--------|-----|-----------|------|
| STD_PSI_MIN | 0.960 | 0.981 | 0.005 |
| ENH_PSI_MIN | 0.966 | 0.984 | 0.012 |
| Overall | 0.961 | 0.982 | 0.009 |

## Citation

```bibtex
@article{cheng2026fusemap,
  title={FUSEMAP: Biophysics-Informed Deep Learning for Regulatory DNA Prediction and Design},
  author={Cheng, Bryan},
  journal={},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Bryan Cheng - bryanc5864@github
