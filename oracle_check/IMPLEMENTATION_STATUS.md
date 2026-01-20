# OracleCheck Implementation Status

## Overview

This document tracks the implementation status of OracleCheck against the full specification.

**Last Updated:** 2026-01-14

---

## Reference Panels

| Component | Status | Notes |
|-----------|--------|-------|
| Natural High-Performers (top quartile) | ✅ Done | `ReferencePanelBuilder` in `reference_panels.py` |
| Physics distributions (PhysInformer) | ✅ Done | 6 families implemented |
| Physics distributions (TileFormer) | ✅ Done | Electrostatics integrated into reference panels |
| MicroMotif statistics | ✅ Replaced | Using JASPAR PWM screening instead |
| Composition metrics | ✅ Done | GC, CpG O/E, entropy, repeats |
| Chromatin proxies (ATAC/H3K27ac/CTCF) | ❌ Skipped | Not necessary per user request |
| Background Natural (random genomic tiles) | ⚠️ Optional | Can be added if needed |
| Training Index (kNN for OOD) | ✅ Done | kNN in CADENCE penultimate features |

---

## Validation Checks

### Physics Conformity

| Check | Status | Notes |
|-------|--------|-------|
| Per-family z-scores | ✅ Done | `z_f = (p̂_f - μ_f) / σ_f` |
| Physics NLL | ✅ Done | `NLL_phys(s) = -Σ log π_nat,family(p̂_family)` |
| Mahalanobis distance (per family) | ✅ Done | `MahalanobisValidator` in `validators.py` |
| Mahalanobis distance (CADENCE embedding) | ✅ Done | Full covariance with pseudo-inverse |
| Pass rule: all families |z| ≤ 2.5 | ✅ Done | Soft threshold |
| Pass rule: NLL below 95th percentile | ✅ Done | Hard threshold |

### Motif/Syntax Validation (JASPAR-based)

| Check | Status | Notes |
|-------|--------|-------|
| JASPAR PWM scanning | ✅ Done | `MotifScanner` in `motif_validator.py` |
| Species-specific motifs | ✅ Done | Human/vertebrate, fly/insect, plant, yeast/fungi |
| Binding load validation | ✅ Done | Against reference [p5, p95] |
| Motif diversity check | ✅ Done | Unique motifs vs total |
| Per-TF scoring | ✅ Done | Log-odds scoring |

### Composition and Repeat Hygiene

| Check | Status | Notes |
|-------|--------|-------|
| GC content envelope | ✅ Done | 35-75% or cell-type prior |
| CpG O/E within natural [p5, p95] | ✅ Done | |
| Shannon entropy not in bottom p5 | ✅ Done | |
| Repeat fraction | ✅ Done | Hard fail if > 0.3 |
| Max homopolymer | ✅ Done | Hard fail if > 10 |

### CADENCE-Specific Checks

| Check | Status | Notes |
|-------|--------|-------|
| Panel agreement (ensemble IQR) | ✅ Done | In protocol logic |
| Epistemic σ_epi below P90 | ✅ Done | `ConfidenceValidator` with PLACE thresholds |
| Conformal width below P95 | ✅ Done | `ConfidenceValidator` with PLACE thresholds |
| OOD kNN/Mahalanobis ≤ P95 | ✅ Done | Both kNN and Mahalanobis implemented |

### Two-Sample Distribution Tests (Batch-Level)

| Check | Status | Notes |
|-------|--------|-------|
| MMD on physics vectors | ✅ Done | `MMDTest` in `two_sample_tests.py` |
| Energy distance | ✅ Done | `EnergyDistanceTest` with permutation test |
| KS-tests per scalar feature | ✅ Done | `KSTest` class |
| k-mer spectrum JS-divergence | ✅ Done | `KmerJSDivergence` for k=4,5,6 |

### RC Consistency

| Check | Status | Notes |
|-------|--------|-------|
| Predict both s and RC(s) | ✅ Done | `RCConsistencyChecker` in `rc_consistency.py` |
| ISM flip test | ✅ Done | `ISMFlipTest` class |
| Delta threshold check | ✅ Done | Configurable threshold (default 0.1) |

---

## Scorecards

| Field | Status | Notes |
|-------|--------|-------|
| Activity (panel mean, IQR) | ✅ Done | |
| Uncertainty (σ_epi, conf_width, OOD) | ✅ Done | |
| Physics (family max |z|, NLL, Mahalanobis) | ✅ Done | |
| Composition (GC, CpG O/E, repeats, entropy) | ✅ Done | |
| Motif validation (binding load, diversity) | ✅ Done | |
| Verdicts (GREEN/YELLOW/RED) | ✅ Done | |

### Verdict Rules

| Rule | Status | Notes |
|------|--------|-------|
| GREEN: all checks pass | ✅ Done | |
| YELLOW: ≤1 soft failure | ✅ Done | |
| RED: any hard failure | ✅ Done | |
| Hard fail: physics z > 4.0 | ✅ Done | |
| Hard fail: repeat fraction > 0.3 | ✅ Done | |
| Hard fail: OOD flag | ✅ Done | |
| Hard fail: >5 syntax violations | ✅ Done | Via motif validation |

---

## Validation Protocol

### Generation Sets Support

| Generation Set | Status | Notes |
|----------------|--------|-------|
| Set 1: Unconstrained Optimization | ✅ Ready | `ISMOptimizer` available |
| Set 2: Physics-Constrained Optimization | ✅ Ready | `PINCSDOptimizer` available |
| Set 3: PhysicsVAE Generation | ✅ Ready | Models in `physics/PhysicsVAE/runs/` |
| Set 4: Transfer-Guided Generation | ✅ Done | `physics/PhysicsTransfer/` with 3 protocols |
| Protocol orchestrator | ✅ Done | `ValidationProtocolRunner` in `validation_runner.py` |

### Analysis Protocol

| Analysis | Status | Notes |
|----------|--------|-------|
| Activity predictions (μ, σ_aleatoric, σ_epistemic, CI) | ✅ Done | Via CADENCE+PLACE |
| Physics analysis (500+ features) | ✅ Done | PhysInformer + TileFormer |
| Motif analysis | ✅ Done | JASPAR PWM scanning |
| Naturality scores | ✅ Done | GMM, kNN, Mahalanobis |
| Composition analysis | ✅ Done | |
| OracleCheck results aggregation | ✅ Done | `ValidationReport` dataclass |

### Statistical Comparisons

| Comparison | Status | Notes |
|------------|--------|-------|
| Unconstrained vs Physics-Constrained | ✅ Done | `StatisticalComparator` |
| PhysicsVAE vs Optimization | ✅ Done | Paired t-test, chi-squared |
| Designed vs Natural High-Activity | ✅ Done | MMD, KS, JS divergence |
| Cross-Species Transfer | ✅ Done | `PhysicsTransfer` with zero-shot, fine-tuning, joint training |

---

## Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `validators.py` | Modified | Added Mahalanobis validator, PLACE thresholds |
| `motif_validator.py` | Created | JASPAR PWM-based motif validation |
| `two_sample_tests.py` | Created | MMD, KS, energy distance, k-mer JS |
| `rc_consistency.py` | Created | RC consistency check, ISM flip test |
| `reference_panels.py` | Modified | Added TileFormer integration |
| `validation_runner.py` | Created | Protocol orchestrator |
| `statistical_comparisons.py` | Created | Comparison framework |
| `__init__.py` | Modified | Export all new modules |

---

## Current Capabilities

With the current implementation, you CAN:

1. ✅ Validate sequences for physics conformity (z-scores, NLL, Mahalanobis)
2. ✅ Check composition hygiene (GC, CpG, entropy, repeats)
3. ✅ Detect OOD sequences via kNN and Mahalanobis distance
4. ✅ Validate motif content with JASPAR PWMs (species-specific)
5. ✅ Run RC consistency checks with ISM flip test
6. ✅ Perform two-sample tests (MMD, KS, JS) vs natural reference
7. ✅ Generate GREEN/YELLOW/RED verdicts
8. ✅ Run batch validation with detailed reports
9. ✅ Perform statistical comparisons between generation methods
10. ✅ Generate comprehensive comparison reports

## Usage Example

```python
from oracle_check import (
    create_runner,
    generate_comparison_report,
    MotifValidator,
    RCConsistencyChecker,
)

# Create validation runner for K562
runner = create_runner(cell_type="K562")

# Validate a batch of designed sequences
report = runner.validate_sequences(
    sequences=designed_sequences,
    generation_method="ISM_optimization",
    run_motif=True,
    run_rc=True,
    run_two_sample=True,
)

# Print results
print(f"Green rate: {report.green_rate:.1%}")
print(f"Mean activity: {report.mean_activity:.3f}")
print(f"Physics pass rate: {report.physics_pass_rate:.1%}")

# Compare multiple methods
reports = {
    "unconstrained": unconstrained_report,
    "physics_constrained": constrained_report,
    "physics_vae": vae_report,
}
comparison = generate_comparison_report(reports, output_path="comparison.json")
```

---

## Priority Items (If Needed)

### Already Completed
- ✅ TileFormer integration
- ✅ Mahalanobis distance
- ✅ JASPAR motif validation
- ✅ PLACE epistemic/conformal thresholds
- ✅ Two-sample tests
- ✅ RC consistency
- ✅ Validation protocol runner
- ✅ Statistical comparisons

### Optional Enhancements
1. Background Natural panel (GC/length-matched random tiles)
2. ~~Cross-species transfer evaluation~~ ✅ Done via `PhysicsTransfer`
3. Ensemble panel agreement (multiple seeded oracles)
4. Chromatin proxy checks (if auxiliary heads available)
