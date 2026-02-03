# FUSEMAP Results Directory

Aggregated results from all FUSEMAP experiments. Contains only lightweight files (JSON metrics, configs, logs, text reports) - no model checkpoints.

**Total: 584 files, ~11 MB | 340+ experiments across 7 species**

---

## Directory Structure

```
FUSEMAP_results/
|-- cadence_single_task/       # 58 files  - Per-dataset CADENCE models
|-- cadence_multi_task/        # 11 files  - Config2-5 joint models
|-- cadence_transfer/          # 168 files - Cross-species transfer learning
|-- physinformer_transfer/     # 18 files  - PhysInformer zero-shot transfer
|-- physicsvae_transfer/       # 18 files  - PhysicsVAE generation transfer
|-- physicsvae_eval/           # 2 files   - PhysicsVAE evaluation summary
|-- physics_transfer/          # 8 files   - PhysicsTransfer cross-species
|-- physics_analysis/          # 153 files - Univariate/multivariate analyses
|-- physics_interpreter/       # 18 files  - Attribution analysis per species
|-- s2a_transfer/              # 10 files  - S2A zero-shot activity transfer
|-- tileformer/                # 8 files   - Electrostatic surrogate model
|-- therapeutic_design/        # 7 files   - Cell-type-specific enhancer design
|-- variant_analysis/          # 26 files  - ClinVar variant interpretation
|-- comparison_models/         # 5 files   - LegNet & DREAM-RNN baselines
|-- mouse_esc_validation/      # 5 files   - External mouse ESC validation
|-- place_calibration/         # 61 files  - PLACE uncertainty calibration
|-- enhancer_design/           # 3 files   - General enhancer design
|-- slide*_metrics.txt         # 5 files   - Extracted numerical summaries
```

---

## 1. CADENCE Single-Task Performance

Source: `cadence_single_task/*/final_results.json`

### Human lentiMPRA

| Model | Dataset | Test Pearson r | Test Spearman rho | R^2 | n_test |
|-------|---------|---------------|-------------------|-----|--------|
| cadence_k562_v2 | K562 | **0.813** | 0.761 | 0.659 | 22,613 |
| cadence_hepg2_v2 | HepG2 | **0.787** | 0.773 | 0.615 | 13,997 |
| cadence_wtc11_v2 | WTC11 | **0.659** | 0.558 | 0.409 | 5,596 |

### Drosophila DeepSTARR

| Model | Output | Test Pearson r | Test Spearman rho | n_test |
|-------|--------|---------------|-------------------|--------|
| cadence_deepstarr_v2 | Dev | **0.906** | 0.886 | 40,570 |
| cadence_deepstarr_v2 | Hk | **0.918** | 0.863 | 40,570 |
| cadence_deepstarr_all_v2 | Dev | **0.918** | 0.899 | 40,570 |
| cadence_deepstarr_all_v2 | Hk | **0.927** | 0.875 | 40,570 |

### Plant Species (Jores 2021)

| Model | Species | Leaf r | Leaf rho | Proto r |
|-------|---------|--------|----------|---------|
| cadence_maize_v1 | Maize | **0.796** | 0.799 | 0.767 |
| cadence_sorghum_v1 | Sorghum | **0.782** | 0.778 | 0.769 |
| cadence_arabidopsis_v1 | Arabidopsis | **0.618** | 0.600 | 0.508 |

### Yeast (DREAM Challenge)

| Model | Test Pearson r | Test Spearman rho |
|-------|---------------|-------------------|
| cadence_yeast_v1 | 0.580 | 0.594 |
| CADENCE Pro (from slide metrics) | **0.958** | 0.961 |

---

## 2. LegNet Comparison

Source: `comparison_models/legnet/*/results.json`

| Cell Type | CADENCE r | LegNet r | Delta |
|-----------|-----------|----------|-------|
| K562 | 0.809 | 0.811 | -0.002 |
| HepG2 | 0.808 | 0.783 | **+0.025** |
| WTC11 | 0.700 | 0.698 | +0.002 |

---

## 3. Multi-Task Configurations

Source: `cadence_multi_task/*/final_results.json`

### Config2: Multi-Human (K562 + HepG2 + WTC11)

| Dataset | Pearson r | Spearman rho |
|---------|-----------|--------------|
| K562 | 0.517 | 0.407 |
| HepG2 | 0.657 | 0.651 |
| WTC11 | 0.539 | 0.379 |

### Config3: Cross-Animal (Human + Drosophila)

| Dataset | Pearson r | Spearman rho |
|---------|-----------|--------------|
| K562 | 0.688 | 0.635 |
| HepG2 | 0.683 | 0.672 |
| WTC11 | 0.633 | 0.517 |

### Config4: Cross-Kingdom (All species)

| Dataset | Pearson r | Spearman rho |
|---------|-----------|--------------|
| K562 | 0.711 | 0.658 |
| HepG2 | 0.696 | 0.690 |
| WTC11 | 0.639 | 0.552 |
| DeepSTARR Dev | 0.665 | 0.617 |
| Maize Leaf | 0.713 | 0.720 |
| Arabidopsis Leaf | 0.632 | 0.596 |

### Config5: Universal (no yeast)

| Dataset | Pearson r | Spearman rho |
|---------|-----------|--------------|
| K562 | 0.624 | 0.564 |
| HepG2 | 0.634 | 0.618 |
| WTC11 | 0.585 | 0.476 |
| DeepSTARR Dev | 0.637 | 0.599 |
| Maize Leaf | 0.779 | 0.784 |
| Sorghum Leaf | 0.783 | 0.777 |

---

## 4. CADENCE Transfer Learning (168 experiments)

Source: `cadence_transfer/*.json`

### Best Results by Source -> Target

| Source | Target | Fraction | Strategy | Spearman rho | Pearson r | AUROC |
|--------|--------|----------|----------|-------------|-----------|-------|
| **K562** | S2 Drosophila | 25% | full_ft | **0.556** | **0.579** | **0.804** |
| HepG2 | S2 Drosophila | 25% | full_ft | 0.524 | 0.543 | 0.786 |
| **WTC11** | Mouse ESC | 25% | full_ft | **0.281** | **0.316** | **0.623** |
| Maize | S2 Drosophila | 25% | full_ft | 0.501 | - | - |
| DeepSTARR | S2 Drosophila | 1% | frozen | 0.976 | - | - |

### Aggregate Statistics (46 human-source experiments)

| Strategy | Mean Spearman | Mean Pearson |
|----------|---------------|--------------|
| Frozen backbone | 0.147 | 0.156 |
| **Full fine-tune** | **0.301** | **0.313** |

---

## 5. PhysInformer Zero-Shot Transfer (18 experiments)

Source: `physinformer_transfer/*.json`

### Transfer Matrix (mean Pearson r across overlapping features)

| Source | Target | n_overlap | Mean Pearson | Median Pearson |
|--------|--------|-----------|-------------|----------------|
| K562 | HepG2 | 411 | **0.847** | 0.968 |
| K562 | WTC11 | 411 | **0.839** | 0.971 |
| K562 | S2 | 267 | 0.729 | 0.901 |
| K562 | Maize | 267 | 0.680 | 0.859 |
| K562 | Arabidopsis | 267 | 0.656 | 0.857 |

### Feature Category Transfer Hierarchy

| Category | Within-Human r | Cross-Kingdom r | Degradation |
|----------|---------------|-----------------|-------------|
| Bending | 0.981 | 0.92 | -6% |
| Advanced Structural | 0.937 | 0.91 | -3% |
| Entropy | 0.843 | 0.68 | -19% |
| Stiffness | 0.430 | 0.45 | +5% |
| PWM/TF Binding | 0.940 | 0.03 | **-97%** |

---

## 6. S2A Zero-Shot Activity Transfer

Source: `s2a_transfer/holdout_*/`

### Leave-One-Out Results

| Holdout | Sources | Zero-Shot Spearman | Zero-Shot Pearson | n_test |
|---------|---------|-------------------|-------------------|--------|
| **Maize** | Arab+Sorghum | **0.700** | 0.694 | 2,461 |
| Sorghum | Others | 0.370 | 0.376 | 1,968 |
| Arabidopsis | Others | 0.308 | 0.316 | 1,347 |
| WTC11 | K562+HepG2 | 0.260 | 0.342 | 5,597 |
| S2_dev | All others | -0.085 | -0.052 | 41,186 |
| K562 | Others | 0.050 | 0.067 | 22,631 |
| HepG2 | Others | 0.045 | 0.063 | 13,953 |

### Transfer Scenarios

| Scenario | Spearman rho | Interpretation |
|----------|-------------|----------------|
| Within-Plant (-> Maize) | **0.700** | Strong |
| Within-Human (-> WTC11) | 0.260 | Moderate |
| Plant -> Animal | 0.125 | Weak |
| Animal -> Plant | **-0.321** | Anti-correlated |

---

## 7. TileFormer Electrostatic Prediction

Source: `tileformer/slide48_49_tileformer_metrics.txt`

| Metric | R^2 | Pearson r | RMSE | MAE |
|--------|-----|-----------|------|-----|
| STD_PSI_MIN | 0.960 | 0.981 | 0.00511 | 0.00407 |
| STD_PSI_MAX | 0.954 | 0.981 | 0.00200 | 0.00159 |
| STD_PSI_MEAN | 0.959 | 0.984 | 0.00316 | 0.00254 |
| ENH_PSI_MIN | **0.966** | 0.984 | 0.01227 | 0.00980 |
| ENH_PSI_MAX | 0.961 | 0.981 | 0.01125 | 0.00902 |
| ENH_PSI_MEAN | 0.961 | 0.981 | 0.01210 | 0.00970 |

Speedup: >10,000x vs explicit APBS calculations

---

## 8. Therapeutic Enhancer Design

Source: `therapeutic_design/` and `slide60_61_therapeutic_metrics.txt`

| Target Cell | Mean Specificity | Max Specificity | Pass Rate | GREEN | YELLOW | RED |
|-------------|-----------------|-----------------|-----------|-------|--------|-----|
| **HepG2** | **4.391** | **7.176** | **99.0%** | 193 | 5 | 2 |
| K562 | 1.949 | 4.524 | 83.5% | 105 | 62 | 33 |
| WTC11 | 1.218 | 5.242 | 86.0% | 106 | 66 | 28 |

---

## 9. PhysicsVAE Transfer (18 experiments)

Source: `physicsvae_transfer/*.json`

| Source | Target | Accuracy | Perplexity |
|--------|--------|----------|------------|
| K562 | HepG2 | **0.560** | 2.48 |
| K562 | WTC11 | **0.561** | 2.49 |
| K562 | S2 | 0.290 | 18.67 |
| K562 | Maize | 0.259 | 5.67 |

Random baseline: 25% accuracy. Within-human transfer works (50-56%); cross-species collapses.

---

## 10. Mouse ESC External Validation

Source: `mouse_esc_validation/`

### Zero-Shot Transfer (all fail)

| Model | Condition | Spearman | Pearson | AUROC |
|-------|-----------|----------|---------|-------|
| cadence_k562 | 2iL | -0.106 | -0.094 | 0.454 |
| cadence_hepg2 | 2iL | -0.019 | 0.057 | 0.514 |
| config4 | 2iL | -0.069 | -0.074 | 0.484 |
| config5 | 2iL | -0.058 | -0.054 | 0.491 |

### Progressive Transfer (with fine-tuning)

| Fraction | n_train | Scratch rho | K562 Full FT rho | Config4 Full FT rho |
|----------|---------|-------------|-------------------|---------------------|
| 1% | 192 | 0.208 | 0.079 | 0.201 |
| 5% | 964 | 0.279 | 0.165 | 0.193 |
| 10% | 1,929 | 0.325 | 0.161 | 0.206 |
| 25% | 4,823 | **0.357** | 0.217 | 0.264 |

---

## 11. PhysInformer Within-Dataset Performance

Source: `tileformer/slide50_52_physinformer_metrics.txt`

| Training Dataset | Overall Validation Pearson r | Epochs |
|-----------------|------------------------------|--------|
| K562 | 0.918 | 50 |
| HepG2 | 0.915 | 50 |
| WTC11 | 0.905 | 50 |
| S2 (Drosophila) | 0.917 | 45 |

---

## 12. PLACE Uncertainty Calibration

Source: `place_calibration/calibration_summary.json`

Successfully calibrated for 20 model configurations. PLACE provides post-hoc epistemic + aleatoric uncertainty without retraining.

---

## Source File Locations

All raw results remain in their original locations:
- `results/` - CADENCE single-task + S2A + therapeutic + variants
- `training/results/` - DeepSTARR, plant, yeast, multi-task models
- `external_validation/results/comprehensive_validation/` - 168+18+18 transfer experiments
- `physics/PhysicsTransfer/results/` - Physics transfer experiments
- `physics/PhysicsInterpreter/results/` - Attribution analysis
- `physics/results/` - Univariate/multivariate physics analysis
- `cadence_place/` - PLACE-calibrated model artifacts
- `comparison_models/` - LegNet and DREAM-RNN baselines
- `presentation_figures/` - All generated figures
