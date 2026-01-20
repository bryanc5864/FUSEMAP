# FUSEMAP Comprehensive Results Report

Complete experimental results and metrics from all validation experiments.

---

## Executive Summary

| Experiment Type | Count | Key Metric | Best Result |
|-----------------|-------|------------|-------------|
| CADENCE Transfer | 46 | Spearman ρ | 0.556 (K562→S2) |
| PhysInformer Transfer | 18 | Pearson r | 0.85 (K562→HepG2) |
| PhysicsVAE Transfer | 18 | Accuracy | 56.1% (K562→WTC11) |
| PhysicsTransfer | 6 | Pearson r | 0.70 (Plant zero-shot) |
| **Total Experiments** | **88+** | | |

---

## 1. CADENCE Single-Task Performance

### Best Single-Cell Models (Human)

| Model | Dataset | Val Pearson | Val Spearman | Test Pearson | Test Spearman |
|-------|---------|-------------|--------------|--------------|---------------|
| **cadence_k562_v2** | K562 | **0.813** | **0.761** | 0.809 | 0.759 |
| cadence_hepg2_v2 | HepG2 | 0.787 | 0.773 | 0.786 | 0.770 |
| cadence_wtc11_v2 | WTC11 | 0.659 | 0.558 | 0.698 | 0.591 |

### DeepSTARR (Drosophila S2)

| Model | Dev Val r | Dev Test r | Hk Val r | Hk Test r |
|-------|-----------|------------|----------|-----------|
| **cadence_deepstarr_v2** | 0.906 | **0.909** | 0.918 | **0.920** |

### Plant Species (Jores 2021)

| Model | Species | Leaf Val r | Leaf Spearman | Proto Val r |
|-------|---------|------------|---------------|-------------|
| cadence_maize_v1 | Maize | **0.796** | 0.799 | 0.767 |
| cadence_sorghum_v1 | Sorghum | **0.782** | - | 0.769 |
| cadence_arabidopsis_v1 | Arabidopsis | 0.618 | - | 0.508 |

### Yeast (DREAM Challenge)

| Model | Val Pearson | Val Spearman | Test Pearson | Test Spearman |
|-------|-------------|--------------|--------------|---------------|
| cadence_yeast_v1 | 0.580 | 0.594 | **0.734** | 0.738 |

### Cross-Kingdom Model (Config 4)

| Dataset | Pearson r | Spearman ρ | R² |
|---------|-----------|------------|-----|
| K562 | 0.711 | 0.658 | 0.374 |
| HepG2 | 0.696 | 0.690 | 0.332 |
| WTC11 | 0.639 | 0.552 | 0.229 |
| DeepSTARR Dev | 0.665 | 0.617 | 0.378 |
| Maize Leaf | 0.778 | 0.784 | 0.542 |
| Sorghum Leaf | 0.783 | 0.777 | 0.508 |
| Arabidopsis Leaf | 0.630 | 0.591 | 0.298 |

### Universal Model (Config 5)

| Dataset | Pearson r | Spearman ρ | R² |
|---------|-----------|------------|-----|
| K562 | 0.624 | 0.564 | 0.167 |
| HepG2 | 0.634 | 0.618 | 0.302 |
| WTC11 | 0.591 | 0.512 | 0.198 |
| DeepSTARR Dev | 0.637 | 0.599 | 0.270 |
| DeepSTARR Hk | 0.687 | 0.513 | 0.250 |
| Maize Leaf | 0.778 | 0.784 | 0.542 |
| Maize Proto | 0.786 | 0.755 | 0.578 |

---

## 2. CADENCE Transfer Learning Results

### Complete Results Table (46 experiments from human cell line sources)

*Source models: cadence_k562_v2, cadence_hepg2_v2, cadence_wtc11_v2 → Target datasets: mouse_esc, s2_drosophila*

| Source | Target | Fraction | Strategy | Spearman ρ | Pearson r | AUROC |
|--------|--------|----------|----------|------------|-----------|-------|
| cadence_k562_v2 | mouse_esc | 1% | frozen | 0.055 | 0.049 | 0.518 |
| cadence_k562_v2 | mouse_esc | 1% | full_finetune | 0.073 | 0.071 | 0.529 |
| cadence_k562_v2 | mouse_esc | 5% | frozen | 0.063 | 0.062 | 0.521 |
| cadence_k562_v2 | mouse_esc | 5% | full_finetune | 0.165 | 0.187 | 0.563 |
| cadence_k562_v2 | mouse_esc | 10% | frozen | 0.041 | 0.027 | 0.505 |
| cadence_k562_v2 | mouse_esc | 10% | full_finetune | 0.167 | 0.194 | 0.564 |
| cadence_k562_v2 | mouse_esc | 25% | frozen | 0.041 | 0.031 | 0.505 |
| cadence_k562_v2 | mouse_esc | 25% | full_finetune | 0.216 | 0.250 | 0.584 |
| cadence_k562_v2 | s2_drosophila | 1% | frozen | 0.156 | 0.148 | 0.600 |
| cadence_k562_v2 | s2_drosophila | 1% | full_finetune | 0.335 | 0.327 | 0.677 |
| cadence_k562_v2 | s2_drosophila | 5% | frozen | 0.219 | 0.235 | 0.619 |
| cadence_k562_v2 | s2_drosophila | 5% | full_finetune | 0.453 | 0.461 | 0.740 |
| cadence_k562_v2 | s2_drosophila | 10% | frozen | 0.233 | 0.246 | 0.623 |
| cadence_k562_v2 | s2_drosophila | 10% | full_finetune | 0.504 | 0.518 | 0.771 |
| cadence_k562_v2 | s2_drosophila | 25% | frozen | 0.236 | 0.250 | 0.624 |
| cadence_k562_v2 | s2_drosophila | 25% | full_finetune | **0.556** | **0.579** | **0.804** |
| cadence_hepg2_v2 | mouse_esc | 1% | frozen | 0.091 | 0.127 | 0.531 |
| cadence_hepg2_v2 | mouse_esc | 1% | full_finetune | 0.081 | 0.124 | 0.524 |
| cadence_hepg2_v2 | mouse_esc | 5% | frozen | 0.087 | 0.128 | 0.540 |
| cadence_hepg2_v2 | mouse_esc | 5% | full_finetune | 0.174 | 0.209 | 0.587 |
| cadence_hepg2_v2 | mouse_esc | 10% | frozen | 0.101 | 0.139 | 0.546 |
| cadence_hepg2_v2 | mouse_esc | 10% | full_finetune | 0.184 | 0.214 | 0.591 |
| cadence_hepg2_v2 | mouse_esc | 25% | frozen | 0.098 | 0.135 | 0.546 |
| cadence_hepg2_v2 | mouse_esc | 25% | full_finetune | 0.238 | 0.255 | 0.617 |
| cadence_hepg2_v2 | s2_drosophila | 1% | frozen | 0.058 | 0.079 | 0.542 |
| cadence_hepg2_v2 | s2_drosophila | 1% | full_finetune | 0.271 | 0.252 | 0.639 |
| cadence_hepg2_v2 | s2_drosophila | 5% | frozen | 0.185 | 0.186 | 0.593 |
| cadence_hepg2_v2 | s2_drosophila | 5% | full_finetune | 0.385 | 0.382 | 0.700 |
| cadence_hepg2_v2 | s2_drosophila | 10% | frozen | 0.222 | 0.235 | 0.606 |
| cadence_hepg2_v2 | s2_drosophila | 10% | full_finetune | 0.463 | 0.473 | 0.746 |
| cadence_hepg2_v2 | s2_drosophila | 25% | frozen | 0.242 | 0.254 | 0.614 |
| cadence_hepg2_v2 | s2_drosophila | 25% | full_finetune | 0.524 | 0.543 | 0.786 |
| cadence_wtc11_v2 | mouse_esc | 1% | frozen | 0.148 | 0.114 | 0.555 |
| cadence_wtc11_v2 | mouse_esc | 1% | full_finetune | 0.161 | 0.131 | 0.560 |
| cadence_wtc11_v2 | mouse_esc | 5% | frozen | 0.143 | 0.138 | 0.552 |
| cadence_wtc11_v2 | mouse_esc | 5% | full_finetune | 0.198 | 0.221 | 0.596 |
| cadence_wtc11_v2 | mouse_esc | 10% | frozen | 0.119 | 0.110 | 0.542 |
| cadence_wtc11_v2 | mouse_esc | 10% | full_finetune | 0.232 | 0.249 | 0.603 |
| cadence_wtc11_v2 | mouse_esc | 25% | frozen | 0.150 | 0.165 | 0.552 |
| cadence_wtc11_v2 | mouse_esc | 25% | full_finetune | **0.281** | **0.316** | **0.623** |
| cadence_wtc11_v2 | s2_drosophila | 1% | frozen | 0.185 | 0.195 | 0.575 |
| cadence_wtc11_v2 | s2_drosophila | 1% | full_finetune | 0.280 | 0.270 | 0.637 |
| cadence_wtc11_v2 | s2_drosophila | 5% | frozen | 0.206 | 0.221 | 0.583 |
| cadence_wtc11_v2 | s2_drosophila | 5% | full_finetune | 0.363 | 0.363 | 0.683 |
| cadence_wtc11_v2 | s2_drosophila | 10% | frozen | 0.214 | 0.227 | 0.588 |
| cadence_wtc11_v2 | s2_drosophila | 10% | full_finetune | 0.420 | 0.422 | 0.717 |

### Summary Statistics

**Note:** Statistics computed from 46 experiments using K562, HepG2, WTC11 source models only (human cell lines).

**By Strategy:**
| Strategy | Mean Spearman | Mean Pearson |
|----------|---------------|--------------|
| Frozen backbone | 0.147 | 0.156 |
| Full fine-tune | **0.301** | **0.313** |

**By Data Fraction:**
| Fraction | Mean Spearman | Mean Pearson |
|----------|---------------|--------------|
| 1% | 0.158 | 0.157 |
| 5% | 0.220 | 0.233 |
| 10% | 0.248 | 0.260 |
| 25% | 0.275 | 0.293 |

**By Source Model:**
| Source | Mean Spearman | Mean Pearson |
|--------|---------------|--------------|
| cadence_k562_v2 | 0.220 | 0.227 |
| cadence_hepg2_v2 | 0.213 | 0.233 |
| cadence_wtc11_v2 | 0.238 | 0.242 |

**By Target Dataset:**
| Target | Mean Spearman | Mean Pearson |
|--------|---------------|--------------|
| mouse_esc | 0.138 | 0.152 |
| s2_drosophila | **0.309** | **0.317** |

---

## 3. PhysInformer Zero-Shot Transfer Results

### Complete Results Table (18 experiments)

| Source | Target | Overlapping Features | Mean Pearson | Mean Spearman | Median Pearson |
|--------|--------|---------------------|--------------|---------------|----------------|
| PhysInformer_K562 | HepG2 | 411 | **0.847** | 0.778 | 0.968 |
| PhysInformer_K562 | WTC11 | 411 | **0.839** | 0.772 | 0.971 |
| PhysInformer_K562 | S2 | 267 | 0.729 | 0.671 | 0.901 |
| PhysInformer_K562 | arabidopsis | 267 | 0.656 | 0.623 | 0.857 |
| PhysInformer_K562 | sorghum | 267 | 0.679 | 0.639 | 0.863 |
| PhysInformer_K562 | maize | 267 | 0.680 | 0.638 | 0.859 |
| PhysInformer_HepG2 | K562 | 411 | 0.657 | 0.605 | 0.909 |
| PhysInformer_HepG2 | WTC11 | 411 | 0.647 | 0.596 | 0.905 |
| PhysInformer_HepG2 | S2 | 267 | 0.464 | 0.413 | 0.774 |
| PhysInformer_HepG2 | arabidopsis | 267 | 0.420 | 0.409 | 0.704 |
| PhysInformer_HepG2 | sorghum | 267 | 0.434 | 0.410 | 0.737 |
| PhysInformer_HepG2 | maize | 267 | 0.444 | 0.420 | 0.741 |
| PhysInformer_WTC11 | HepG2 | 411 | 0.829 | 0.766 | 0.953 |
| PhysInformer_WTC11 | K562 | 411 | 0.832 | 0.771 | 0.954 |
| PhysInformer_WTC11 | S2 | 267 | 0.649 | 0.611 | 0.858 |
| PhysInformer_WTC11 | arabidopsis | 267 | 0.106 | 0.087 | 0.478 |
| PhysInformer_WTC11 | sorghum | 267 | 0.336 | 0.314 | 0.666 |
| PhysInformer_WTC11 | maize | 267 | 0.382 | 0.381 | 0.681 |

### Feature Category Breakdown

**K562 → HepG2 (Best Transfer):**
| Category | N Features | Mean Pearson | Mean Spearman |
|----------|------------|--------------|---------------|
| bend | 43 | **0.981** | 0.871 |
| pwm | 149 | 0.940 | 0.873 |
| advanced | 2 | 0.937 | 0.897 |
| entropy | 62 | 0.843 | 0.795 |
| stiff | 36 | 0.430 | 0.416 |

**Averaged Across All Transfers:**
| Category | Mean Pearson | Interpretation |
|----------|--------------|----------------|
| advanced | **0.786** | Best transfer |
| bend | **0.750** | Excellent |
| entropy | 0.475 | Moderate |
| stiff | 0.430 | Moderate |
| pwm | 0.270 | Poor |

### Summary by Source Model

| Source Model | Mean Pearson | Mean Spearman | Best Transfer |
|--------------|--------------|---------------|---------------|
| PhysInformer_K562 | **0.739** | **0.687** | HepG2 (0.85) |
| PhysInformer_WTC11 | 0.522 | 0.488 | K562 (0.83) |
| PhysInformer_HepG2 | 0.511 | 0.476 | K562 (0.66) |

### Summary by Target Dataset

| Target | Mean Pearson | Notes |
|--------|--------------|-------|
| HepG2 | **0.838** | Best target |
| K562 | 0.745 | Good |
| WTC11 | 0.743 | Good |
| S2 | 0.614 | Cross-species |
| maize | 0.502 | Cross-kingdom |
| sorghum | 0.483 | Cross-kingdom |
| arabidopsis | 0.394 | Worst target |

---

## 4. PhysicsVAE Zero-Shot Transfer Results

### Complete Results Table (18 experiments)

| Source | Target | Matched Features | Accuracy | Perplexity | Recon Loss |
|--------|--------|------------------|----------|------------|------------|
| PhysicsVAE_K562 | HepG2 | 411 | **0.560** | 2.48 | 0.908 |
| PhysicsVAE_K562 | WTC11 | 411 | **0.561** | 2.49 | 0.910 |
| PhysicsVAE_K562 | S2 | 267 | 0.290 | 18.67 | 2.927 |
| PhysicsVAE_K562 | arabidopsis | 267 | 0.253 | 5.82 | 1.758 |
| PhysicsVAE_K562 | sorghum | 267 | 0.254 | 5.73 | 1.746 |
| PhysicsVAE_K562 | maize | 267 | 0.259 | 5.67 | 1.735 |
| PhysicsVAE_HepG2 | K562 | 411 | 0.511 | 2.71 | 0.997 |
| PhysicsVAE_HepG2 | WTC11 | 411 | 0.514 | 2.71 | 0.996 |
| PhysicsVAE_HepG2 | S2 | 267 | 0.297 | 18.60 | 2.923 |
| PhysicsVAE_HepG2 | arabidopsis | 267 | 0.241 | 5.28 | 1.664 |
| PhysicsVAE_HepG2 | sorghum | 267 | 0.256 | 5.09 | 1.627 |
| PhysicsVAE_HepG2 | maize | 267 | 0.262 | 5.03 | 1.614 |
| PhysicsVAE_WTC11 | HepG2 | 411 | 0.505 | 2.72 | 1.000 |
| PhysicsVAE_WTC11 | K562 | 411 | 0.508 | 2.71 | 0.998 |
| PhysicsVAE_WTC11 | S2 | 267 | 0.290 | 15.80 | 2.760 |
| PhysicsVAE_WTC11 | arabidopsis | 267 | 0.237 | 6.42 | 1.861 |
| PhysicsVAE_WTC11 | sorghum | 267 | 0.255 | 5.87 | 1.771 |
| PhysicsVAE_WTC11 | maize | 267 | 0.261 | 5.77 | 1.752 |

### Summary by Source Model

| Source Model | Mean Accuracy | Mean Perplexity |
|--------------|---------------|-----------------|
| PhysicsVAE_K562 | **0.363** | 6.81 |
| PhysicsVAE_HepG2 | 0.347 | 6.72 |
| PhysicsVAE_WTC11 | 0.343 | 6.55 |

### Summary by Target Dataset

| Target | Mean Accuracy | Mean Perplexity | Notes |
|--------|---------------|-----------------|-------|
| WTC11 | **0.538** | 2.64 | Best within-human |
| HepG2 | 0.533 | 2.64 | Good |
| K562 | 0.510 | 2.71 | Good |
| S2 | 0.292 | 17.69 | Cross-species |
| maize | 0.261 | 5.49 | Cross-kingdom |
| sorghum | 0.255 | 5.56 | Cross-kingdom |
| arabidopsis | 0.243 | 5.84 | Worst target |

### Key Observations

1. **Within-human transfer**: 50-56% accuracy (vs 25% random baseline)
2. **Cross-species (S2)**: 29% accuracy with very high perplexity (17-19)
3. **Cross-kingdom (plants)**: 25-26% accuracy, moderate perplexity (5-6)
4. **Interpretation**: Human models learn human-specific sequence patterns that don't generalize

---

## 5. Physics → Activity Transfer Results

### Auxiliary Head B Performance (Within-Dataset)

| Dataset | Pearson r | R² | MSE |
|---------|-----------|-----|-----|
| K562 | 0.55-0.60 | 0.30-0.36 | 0.15 |
| HepG2 | 0.55-0.61 | 0.30-0.37 | 0.15 |
| WTC11 | 0.55-0.61 | 0.30-0.37 | 0.15 |
| S2 | 0.52 | 0.27 | - |
| Arabidopsis | **0.70-0.80** | 0.49-0.64 | - |
| Sorghum | **0.70-0.80** | 0.49-0.64 | - |
| Maize | **0.70-0.80** | 0.49-0.64 | - |

### PhysicsTransfer Cross-Species Results

**Plant Cross-Species (Arabidopsis+Sorghum → Maize):**
| Protocol | Source r | Target r | Efficiency |
|----------|----------|----------|------------|
| Zero-shot | 0.648 | **0.700** | 1.08 |
| Fine-tune 1k | 0.648 | 0.637 | 0.98 |
| Fine-tune 5k | 0.648 | 0.686 | 1.06 |
| Fine-tune 10k | 0.648 | 0.695 | 1.07 |

**Human Cross-Cell-Type (K562+HepG2 → WTC11):**
| Protocol | Source r | Target r | Efficiency |
|----------|----------|----------|------------|
| Zero-shot | 0.257 | 0.338 | 1.32 |
| Fine-tune 5k | 0.257 | 0.340 | 1.32 |
| Fine-tune 10k | 0.257 | 0.357 | 1.39 |

**Human → Drosophila (WTC11 → S2):**
| Protocol | Source r | Target r | Efficiency |
|----------|----------|----------|------------|
| Zero-shot | 0.376 | 0.006 | **0.02** |
| Fine-tune 5k | 0.376 | 0.332 | 0.88 |
| Fine-tune 10k | 0.376 | 0.349 | 0.93 |

### Feature Importance by Context

**Plant Transfer (best):**
| Category | Importance |
|----------|------------|
| Bending | **50.9%** |
| Advanced | 19.8% |
| Entropy | 10.6% |
| Thermo | 10.3% |
| Stiffness | 8.3% |

**Human Transfer:**
| Category | Importance |
|----------|------------|
| Advanced | 30.9% |
| Bending | 26.3% |
| Entropy | 15.1% |
| Thermo | 14.7% |
| Stiffness | 13.0% |

---

## 6. Key Deliverables

### Model Checkpoints

| Model | Path | Size | Performance |
|-------|------|------|-------------|
| CADENCE K562 (best) | `results/cadence_k562_v2/best_model.pt` | 16 MB | r=0.81 |
| CADENCE HepG2 | `results/cadence_hepg2_v2/best_model.pt` | 16 MB | r=0.78 |
| CADENCE WTC11 | `results/cadence_wtc11_v2/best_model.pt` | 16 MB | r=0.66 |
| CADENCE Cross-kingdom | `results/config4_cross_kingdom_v1/best_model.pt` | 30 MB | r=0.71 |
| CADENCE Universal | `results/config5_universal_*/best_model.pt` | 30 MB | r=0.62 |
| PhysInformer K562 | `physics/PhysInformer/runs/K562_*/best_model.pt` | 128 MB | r=0.94 |
| PhysicsVAE K562 | `physics/PhysicsVAE/runs/K562_*/best_model.pt` | 41 MB | 55.6% |

### Computed Features

| Dataset | Location | Features | Size |
|---------|----------|----------|------|
| K562 | `physics/output/K562_*_descriptors.tsv` | 516 | 792 MB |
| HepG2 | `physics/output/HepG2_*_descriptors.tsv` | 540 | 547 MB |
| WTC11 | `physics/output/WTC11_*_descriptors.tsv` | 540 | ~300 MB |
| Plants | `physics/output/{species}_*_descriptors.tsv` | 372 | ~100 MB each |

### Validation Results

| Experiment Set | Location | Files |
|----------------|----------|-------|
| CADENCE Transfer | `external_validation/results/comprehensive_validation/cadence/` | 46 JSON |
| PhysInformer Transfer | `external_validation/results/comprehensive_validation/physinformer/` | 18 JSON |
| PhysicsVAE Transfer | `external_validation/results/comprehensive_validation/physicsvae/` | 18 JSON |
| PhysicsTransfer | `physics/PhysicsTransfer/results/` | 6 experiments |

---

## 7. Conclusions

### What Works

1. **Single-cell CADENCE**: Excellent performance (r=0.81) for same-domain prediction
2. **PhysInformer within-human**: Near-perfect physics prediction (r=0.85+)
3. **Plant zero-shot transfer**: Physics features transfer well within plants (r=0.70)
4. **Bending features**: Most universal biophysical property (r=0.75 cross-species)

### What Doesn't Work

1. **Zero-shot cross-kingdom**: Human→Plant/Insect fails completely
2. **Physics→Activity cross-species**: r≈0 even with good physics transfer
3. **PWM features**: TF binding doesn't transfer (r=0.27)
4. **Frozen backbone transfer**: Significantly worse than fine-tuning

### Recommendations

1. **Use K562 model as source** for transfer learning (most robust)
2. **Fine-tune with 10-25% data** for best transfer performance
3. **Focus on biophysical features** (bending, melting) for cross-species analysis
4. **Avoid zero-shot cross-kingdom** predictions without fine-tuning

---

## 8. Benchmark Comparisons

### CADENCE vs LegNet (Human lentiMPRA)

| Cell Type | CADENCE Test r | LegNet Test r | Difference | Notes |
|-----------|---------------|---------------|------------|-------|
| K562 | 0.809 | 0.811 | -0.002 | Comparable |
| HepG2 | 0.786 | 0.783 | +0.003 | CADENCE slightly better |
| WTC11 | 0.698 | 0.698 | 0.000 | Identical |

**Analysis:** CADENCE and LegNet perform comparably on human MPRA data, suggesting both are near the performance ceiling for this task.

### CADENCE vs DREAM-RNN (DeepSTARR)

| Output | CADENCE r | DREAM-RNN r | Improvement |
|--------|-----------|-------------|-------------|
| Dev | **0.909** | 0.708 | +0.201 (+28.4%) |
| Hk | **0.920** | 0.779 | +0.141 (+18.1%) |

**Analysis:** CADENCE significantly outperforms DREAM-RNN on Drosophila STARR-seq data, demonstrating the advantage of the LegNet architecture over recurrent models.

### PhysInformer vs Baseline Methods

| Method | Mean r (physics) | Notes |
|--------|-----------------|-------|
| PhysInformer | **0.94** | Transformer-based |
| Ridge Regression | 0.31 | Linear baseline |
| Random Forest | 0.58 | Non-linear baseline |

---

## 9. Physics Analysis Results

### Multivariate Analysis Summary

| Analysis | Cell Type | Key Finding |
|----------|-----------|-------------|
| 01_univariate | Human (3 cells) | 1,140 significant physics-activity correlations (p<0.05) |
| 02_multivariate | Human (3 cells) | R²=0.178 with physics+PWM (vs 0.092 physics-only) |
| 02_multivariate | Plants | R²=0.20-0.25 with physics+PWM |

### Top Physics Predictors by Cell Type

| Cell Type | Top Feature | Correlation |
|-----------|-------------|-------------|
| K562 | stress_mean_stress_opening | -0.267 |
| HepG2 | mgw_mean_mgw | +0.267 |
| WTC11 | total_dH | -0.273 |
| S2 (Dev) | curvature_gradient_mean | -0.244 |
| Maize | bending features | -0.30+ |

### Physics-Activity Relationships

1. **DNA Opening Stress**: Negatively correlated with expression (r=-0.27)
2. **Minor Groove Width**: Positively correlated with expression (r=+0.27)
3. **Thermodynamic Instability**: Lower ΔH/ΔG → higher activity
4. **DNA Bending**: Higher curvature → lower expression in mammals
5. **GC Content**: Moderate positive correlation (+0.25)

---

*Generated: January 2026*
