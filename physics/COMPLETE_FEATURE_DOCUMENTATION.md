# Complete Biophysical Feature Documentation
## PhysiFormer Preprocessing Pipeline - Full Feature Report

### Executive Summary
This document provides comprehensive documentation of all 540/516/540 biophysical features extracted from 230bp DNA sequences for HepG2, K562, and WTC11 cell types respectively. The pipeline combines universal DNA biophysical properties with cell-type specific transcription factor binding predictions using JASPAR 2024 PWMs.

---

## 📊 Feature Statistics by Cell Type

| Cell Type | Total Features | Biophysical Features | PWM Features | Unique TFs |
|-----------|---------------|---------------------|--------------|------------|
| **HepG2** | 540 | 263 | 277 | 34 TFs |
| **K562**  | 516 | 263 | 253 | 31 TFs |
| **WTC11** | 540 | 263 | 277 | 34 TFs |

---

## 🧬 Section 1: Thermodynamic Features (42 features)
**Based on SantaLucia Nearest-Neighbor Parameters**

### Global Thermodynamic Properties (3 features)
1. `thermo_total_dH` - Total enthalpy change (ΔH) across all dinucleotide steps
2. `thermo_total_dS` - Total entropy change (ΔS) across all dinucleotide steps  
3. `thermo_total_dG` - Total Gibbs free energy change (ΔG) at 37°C

### Statistical Thermodynamic Descriptors (6 features)
4. `thermo_mean_dH` - Average enthalpy per dinucleotide step
5. `thermo_mean_dS` - Average entropy per dinucleotide step
6. `thermo_mean_dG` - Average free energy per step at 37°C
7. `thermo_var_dH` - Variance of enthalpy distribution
8. `thermo_var_dS` - Variance of entropy distribution
9. `thermo_var_dG` - Variance of free energy distribution

### Thermodynamic Extremes (6 features)
10. `thermo_min_dH` - Minimum enthalpy (strongest interaction)
11. `thermo_max_dH` - Maximum enthalpy (weakest interaction)
12. `thermo_min_dS` - Minimum entropy (most ordering)
13. `thermo_max_dS` - Maximum entropy (least ordering)
14. `thermo_min_dG` - Minimum free energy (most stable)
15. `thermo_max_dG` - Maximum free energy (least stable)

### Thermodynamic Percentiles (21 features)
16-22. `thermo_dH_p5`, `thermo_dH_p10`, `thermo_dH_p25`, `thermo_dH_p50`, `thermo_dH_p75`, `thermo_dH_p90`, `thermo_dH_p95`
23-29. `thermo_dS_p5`, `thermo_dS_p10`, `thermo_dS_p25`, `thermo_dS_p50`, `thermo_dS_p75`, `thermo_dS_p90`, `thermo_dS_p95`
30-36. `thermo_dG_p5`, `thermo_dG_p10`, `thermo_dG_p25`, `thermo_dG_p50`, `thermo_dG_p75`, `thermo_dG_p90`, `thermo_dG_p95`

### Thermodynamic Range Metrics (3 features)
37. `thermo_dH_iqr` - Interquartile range of enthalpy
38. `thermo_dS_iqr` - Interquartile range of entropy
39. `thermo_dG_iqr` - Interquartile range of free energy

### Melting Temperature (2 features)
40. `thermo_estimated_Tm_C` - Predicted melting temperature in Celsius
41. `thermo_estimated_Tm_K` - Melting temperature in Kelvin

### Stability Metric (1 feature)
42. `thermo_stability_ratio` - Fraction of steps with ΔG < 0

---

## 🔬 Section 2: DNA Stiffness/Mechanical Features (62 features)
**Based on Olson Matrix and Crystal Structure Parameters**

### Overall Deformation Energy (5 features)
43. `stiff_total_relative_energy` - Sum of elastic deformation energies
44. `stiff_mean_relative_energy` - Average deformation energy per step
45. `stiff_var_relative_energy` - Variance of deformation energies
46. `stiff_max_relative_energy` - Maximum deformation energy
47. `stiff_min_relative_energy` - Minimum deformation energy

### Component-Specific Energies (18 features)
48-50. `stiff_twist_total_energy`, `stiff_twist_mean_energy`, `stiff_twist_max_energy`
51-53. `stiff_tilt_total_energy`, `stiff_tilt_mean_energy`, `stiff_tilt_max_energy`
54-56. `stiff_roll_total_energy`, `stiff_roll_mean_energy`, `stiff_roll_max_energy`
57-59. `stiff_shift_total_energy`, `stiff_shift_mean_energy`, `stiff_shift_max_energy`
60-62. `stiff_slide_total_energy`, `stiff_slide_mean_energy`, `stiff_slide_max_energy`
63-65. `stiff_rise_total_energy`, `stiff_rise_mean_energy`, `stiff_rise_max_energy`

### Principal Component Analysis (7 features)
66. `stiff_avg_pc1` - Average projection on PC1 (main bending mode)
67. `stiff_avg_pc2` - Average projection on PC2 (stretch-twist mode)
68. `stiff_pc1_variance` - Variance of PC1 projections
69. `stiff_pc2_variance` - Variance of PC2 projections
70-72. `stiff_cross_terms_total`, `stiff_cross_terms_mean`, `stiff_cross_terms_max`

### Z-Score Normalized Deformations (18 features)
73-75. `stiff_twist_zscore_mean`, `stiff_twist_zscore_var`, `stiff_twist_zscore_max`
76-78. `stiff_tilt_zscore_mean`, `stiff_tilt_zscore_var`, `stiff_tilt_zscore_max`
79-81. `stiff_roll_zscore_mean`, `stiff_roll_zscore_var`, `stiff_roll_zscore_max`
82-84. `stiff_shift_zscore_mean`, `stiff_shift_zscore_var`, `stiff_shift_zscore_max`
85-87. `stiff_slide_zscore_mean`, `stiff_slide_zscore_var`, `stiff_slide_zscore_max`
88-90. `stiff_rise_zscore_mean`, `stiff_rise_zscore_var`, `stiff_rise_zscore_max`

### High-Energy Threshold Analysis (6 features)
91-92. `stiff_high_energy_count_t2.0`, `stiff_high_energy_fraction_t2.0`
93-94. `stiff_high_energy_count_t5.0`, `stiff_high_energy_fraction_t5.0`
95-96. `stiff_high_energy_count_t10.0`, `stiff_high_energy_fraction_t10.0`

### Energy Distribution (2 features)
97. `stiff_energy_distribution_entropy_norm` - Normalized Shannon entropy
98. `stiff_energy_distribution_entropy_raw` - Raw Boltzmann entropy

### Sequence-Structure Correlations (6 features)
99. `stiff_gc_stiffness_correlation` - GC content vs stiffness correlation
100. `stiff_purine_stiffness_correlation` - Purine content vs stiffness
101. `stiff_at_skew` - AT-skew: (A-T)/(A+T)
102. `stiff_gc_skew` - GC-skew: (G-C)/(G+C)
103. `stiff_purine_pyrimidine_ratio` - (A+G)/(C+T) ratio
104. `stiff_gc_content_global` - Overall GC fraction

---

## 📈 Section 3: Information Theory/Entropy Features (62 features)

### Global Entropy Measures (10 features)
105. `entropy_global_shannon_entropy` - Shannon entropy of nucleotide frequencies
106. `entropy_normalized_shannon_entropy` - Normalized Shannon entropy [0,1]
107. `entropy_global_gc_entropy` - Binary entropy (GC vs AT)
108. `entropy_global_kmer1_entropy` - Mononucleotide entropy
109. `entropy_global_kmer2_entropy` - Dinucleotide entropy
110. `entropy_global_kmer3_entropy` - Trinucleotide entropy
111. `entropy_global_kmer4_entropy` - Tetranucleotide entropy
112. `entropy_global_kmer5_entropy` - Pentanucleotide entropy
113. `entropy_global_kmer6_entropy` - Hexanucleotide entropy
114. `entropy_sequence_compressibility` - Compression ratio (gzip)

### Complexity Measures (4 features)
115. `entropy_lempel_ziv_complexity` - Normalized Lempel-Ziv complexity
116. `entropy_conditional_entropy` - First-order conditional entropy
117. `entropy_renyi_entropy_alpha0.0` - Rényi entropy (α=0, Hartley)
118. `entropy_renyi_entropy_alpha2.0` - Rényi entropy (α=2, collision)

### Windowed Shannon Entropy (12 features)
119-122. `entropy_shannon_w10_mean`, `entropy_shannon_w10_var`, `entropy_shannon_w10_max`, `entropy_shannon_w10_min`
123-126. `entropy_shannon_w30_mean`, `entropy_shannon_w30_var`, `entropy_shannon_w30_max`, `entropy_shannon_w30_min`
127-130. `entropy_shannon_w50_mean`, `entropy_shannon_w50_var`, `entropy_shannon_w50_max`, `entropy_shannon_w50_min`

### Windowed GC Entropy (12 features)
131-134. `entropy_gc_entropy_w10_mean`, `entropy_gc_entropy_w10_var`, `entropy_gc_entropy_w10_max`, `entropy_gc_entropy_w10_min`
135-138. `entropy_gc_entropy_w30_mean`, `entropy_gc_entropy_w30_var`, `entropy_gc_entropy_w30_max`, `entropy_gc_entropy_w30_min`
139-142. `entropy_gc_entropy_w50_mean`, `entropy_gc_entropy_w50_var`, `entropy_gc_entropy_w50_max`, `entropy_gc_entropy_w50_min`

### Windowed K-mer Entropy (12 features)
143-145. `entropy_kmer2_entropy_w30_mean`, `entropy_kmer2_entropy_w30_var`, `entropy_kmer2_entropy_w30_max`
146-148. `entropy_kmer2_entropy_w50_mean`, `entropy_kmer2_entropy_w50_var`, `entropy_kmer2_entropy_w50_max`
149-151. `entropy_kmer3_entropy_w30_mean`, `entropy_kmer3_entropy_w30_var`, `entropy_kmer3_entropy_w30_max`
152-154. `entropy_kmer3_entropy_w50_mean`, `entropy_kmer3_entropy_w50_var`, `entropy_kmer3_entropy_w50_max`

### Mutual Information (10 features)
155-164. `entropy_mi_d1` through `entropy_mi_d10` - Mutual information at distances 1-10bp

### Entropy Rate and Complexity (2 features)
165. `entropy_entropy_rate_estimate` - Estimated entropy rate
166. `entropy_complexity_index` - Combined complexity metric

---

## 🧪 Section 4: Advanced Biophysics Features (53 features)

### Fractal Analysis (4 features)
167. `advanced_fractal_exponent` - Fractal dimension exponent
168. `advanced_fractal_mean_correlation` - Mean correlation across scales
169. `advanced_fractal_std_correlation` - Std of correlations
170. `advanced_fractal_r2` - R² of fractal fit

### Melting Energetics (14 features)
171. `advanced_melting_mean` - Mean melting energy
172. `advanced_melting_std` - Std of melting energies
173. `advanced_melting_min` - Minimum melting energy
174. `advanced_melting_max` - Maximum melting energy
175. `advanced_melting_unstable_fraction` - Fraction of unstable regions
176-182. `advanced_melting_p5` through `advanced_melting_p95` - Percentiles
183. `advanced_melting_iqr` - Interquartile range
184. `advanced_melting_soft_minimum` - Soft minimum value

### Minor Groove Width (5 features)
185. `advanced_mgw_mean` - Mean minor groove width
186. `advanced_mgw_std` - Std of groove widths
187. `advanced_mgw_narrow_fraction` - Fraction < 4.5 Å
188. `advanced_mgw_min` - Minimum width
189. `advanced_mgw_max` - Maximum width

### Base Stacking Energies (13 features)
190. `advanced_stacking_mean` - Mean stacking energy
191. `advanced_stacking_std` - Std of stacking energies
192. `advanced_stacking_skewness` - Skewness of distribution
193. `advanced_stacking_min` - Minimum stacking energy
194. `advanced_stacking_max` - Maximum stacking energy
195-201. `advanced_stacking_p5` through `advanced_stacking_p95` - Percentiles
202. `advanced_stacking_iqr` - Interquartile range

### G-Quadruplex Potential (4 features)
203. `advanced_g4_max_score` - Maximum G4 formation score
204. `advanced_g4_hotspot_count` - Number of G4 hotspots
205. `advanced_g4_mean_score` - Mean G4 score
206. `advanced_g4_peak_distance` - Distance between G4 peaks

### Stress-Induced Opening (13 features)
207. `advanced_opening_mean` - Mean opening probability
208. `advanced_opening_max` - Maximum opening probability
209. `advanced_opening_sum` - Total opening probability
210. `advanced_opening_max_stretch` - Longest open stretch
211. `advanced_opening_local_rate` - Local opening rate
212-218. `advanced_opening_p5` through `advanced_opening_p95` - Percentiles
219. `advanced_opening_iqr` - Interquartile range

---

## 🌊 Section 5: DNA Bending/Curvature Features (44 features)

### Overall Bending (4 features)
220. `bend_total_energy` - Total bending energy
221. `bend_mean_cost` - Mean bending cost
222. `bend_max_cost` - Maximum bending cost
223. `bend_variance` - Variance of bending

### RMS Curvature Windows (8 features)
224-225. `bend_rms_5bp_mean`, `bend_rms_5bp_max`
226-227. `bend_rms_7bp_mean`, `bend_rms_7bp_max`
228-229. `bend_rms_9bp_mean`, `bend_rms_9bp_max`
230-231. `bend_rms_11bp_mean`, `bend_rms_11bp_max`

### Curvature Variance Windows (8 features)
232-233. `bend_var_5bp_mean`, `bend_var_5bp_max`
234-235. `bend_var_7bp_mean`, `bend_var_7bp_max`
236-237. `bend_var_9bp_mean`, `bend_var_9bp_max`
238-239. `bend_var_11bp_mean`, `bend_var_11bp_max`

### Curvature Gradients (2 features)
240. `bend_gradient_mean` - Mean curvature gradient
241. `bend_gradient_max_abs` - Maximum absolute gradient

### Maximum Bends (12 features)
242-244. `bend_max_5bp_mean`, `bend_max_5bp_global_max`, `bend_max_5bp_fraction`
245-247. `bend_max_7bp_mean`, `bend_max_7bp_global_max`, `bend_max_7bp_fraction`
248-250. `bend_max_9bp_mean`, `bend_max_9bp_global_max`, `bend_max_9bp_fraction`
251-253. `bend_max_11bp_mean`, `bend_max_11bp_global_max`, `bend_max_11bp_fraction`

### Bend Hotspots (2 features)
254. `bend_hotspot_count` - Number of bend hotspots (z > 2.0)
255. `bend_hotspot_density` - Density of bend hotspots

### Spectral Analysis (6 features)
256-257. `bend_spectral_power_1_5`, `bend_spectral_phase_1_5` - 5bp periodicity
258-259. `bend_spectral_power_1_7`, `bend_spectral_phase_1_7` - 7bp periodicity
260-261. `bend_spectral_power_1_10`, `bend_spectral_phase_1_10` - 10bp periodicity

### Attention Bias (2 features)
262. `bend_attention_mean` - Mean attention bias
263. `bend_attention_min_span` - Minimum span-wise energy

---

## 🎯 Section 6: Transcription Factor Binding Features (Variable by Cell Type)

### Universal/Housekeeping TFs (18 TFs - All Cell Types)
These TFs regulate core cellular processes and are included for all cell types:

1. **MA0079.5 (SP1)** - GC-rich core promoter regulation
2. **MA0060.4 (NFYA)** - CCAAT box binding
3. **MA0506.3 (NRF1)** - Mitochondrial/housekeeping genes
4. **MA0095.4 (YY1)** - Core promoter/insulator, 3D chromatin
5. **MA0139.2 (CTCF)** - Architectural protein, TAD boundaries
6. **MA0088.2 (ZNF143)** - Promoter-enhancer bridge
7. **MA0024.3 (E2F1)** - Cell cycle, S-phase genes
8. **MA0028.3 (ELK1)** - Serum response, ETS family
9. **MA0083.3 (SRF)** - Serum response factor
10. **MA0093.4 (USF1)** - E-box binding, metabolism
11. **MA0526.5 (USF2)** - E-box binding partner
12. **MA0058.4 (MAX)** - MYC/MAX network
13. **MA0018.5 (CREB1)** - cAMP/PKA responsive
14. **MA0099.4 (FOS::JUN)** - AP-1, stress response
15. **MA0090.4 (TEAD1)** - Hippo pathway
16. **MA0809.3 (TEAD4)** - Hippo pathway
17. **MA0108.3 (TBP)** - TATA box binding
18. **MA0527.2 (ZBTB33)** - Methylation-sensitive

### Cell-Type Specific TFs

#### HepG2 (Liver) - 17 Additional TFs
- **Hepatocyte Nuclear Factors**: MA0114.5 (HNF4A), MA0484.3 (HNF4G), MA0046.3 (HNF1A), MA0153.2 (HNF1B)
- **Forkhead Box**: MA0148.5 (FOXA1), MA0047.4 (FOXA2), MA1683.2 (FOXA3)
- **CCAAT/Enhancer-Binding**: MA0102.5 (CEBPA), MA0466.4 (CEBPB), MA0836.3 (CEBPD)
- **Nuclear Receptors**: MA0512.2 (RXRA), MA2338.1 (PPARA), MA1148.2 (PPARA::RXRA), MA2337.1 (NR1H3), MA0494.2 (NR1H3::RXRA)
- **Other**: MA0679.3 (ONECUT1)

#### K562 (Erythroid) - 15 Additional TFs
- **GATA Family**: MA0035.5 (GATA1), MA0036.4 (GATA2), MA0140.3 (GATA1::TAL1)
- **Erythroid TFs**: MA0091.2 (TAL1::TCF3), MA0493.3 (KLF1), MA1516.2 (KLF3)
- **NFE2 Complex**: MA0841.2 (NFE2), MA0501.2 (MAF::NFE2), MA0496.4 (MAFK), MA0495.4 (MAFF)
- **Other**: MA1633.2 (BACH1), MA0502.3 (NFYB), MA0150.3 (NFE2L2)

#### WTC11 (iPSC) - 16 Additional TFs
- **Pluripotency Core**: MA0142.1 (POU5F1::SOX2), MA0143.5 (SOX2), MA2339.1 (NANOG), MA0039.5 (KLF4)
- **SOX Family**: MA0870.1 (SOX1), MA0078.3 (SOX17), MA0866.1 (SOX21), MA0515.1 (SOX6), MA0514.3 (SOX3), MA0077.2 (SOX9), MA0084.2 (SRY)
- **POU Family**: MA0785.2 (POU2F1), MA0787.1 (POU3F2)
- **Other**: MA0141.4 (ESRRB), MA0059.2 (MAX::MYC), MA1723.2 (PRDM9)

### PWM Feature Types (8 per TF)
For each transcription factor, we compute:

1. **`{TF_ID}_max_score`** - Maximum log-odds binding score
2. **`{TF_ID}_delta_g`** - Binding free energy ΔG = -kT×ln(Z)
3. **`{TF_ID}_mean_score`** - Mean binding score across sequence
4. **`{TF_ID}_var_score`** - Variance of binding scores
5. **`{TF_ID}_total_weight`** - Total statistical weight (partition function)
6. **`{TF_ID}_num_high_affinity`** - Number of high-affinity sites (score > 2)
7. **`{TF_ID}_entropy`** - Binding site position entropy
8. **`{TF_ID}_top_k_mean`** - Mean of top-3 binding scores

### Aggregate PWM Features (5 features)
- **`pwm_max_of_max_score`** - Maximum score across all TFs
- **`pwm_min_delta_g`** - Most favorable binding energy
- **`pwm_tf_binding_diversity`** - Number of TFs with strong binding
- **`pwm_sum_top5_delta_g`** - Sum of top-5 binding energies
- **`pwm_best_tf_index`** - Index of best-binding TF

---

## 📊 Total Feature Counts

### HepG2: 540 Features
- **Thermodynamic**: 42 features
- **Stiffness/Mechanical**: 62 features
- **Entropy/Information**: 62 features
- **Advanced Biophysics**: 53 features
- **Bending/Curvature**: 44 features
- **PWM/TF Binding**: 277 features (34 TFs × 8 + 5 aggregate)

### K562: 516 Features
- **Thermodynamic**: 42 features
- **Stiffness/Mechanical**: 62 features
- **Entropy/Information**: 62 features
- **Advanced Biophysics**: 53 features
- **Bending/Curvature**: 44 features
- **PWM/TF Binding**: 253 features (31 TFs × 8 + 5 aggregate)

### WTC11: 540 Features
- **Thermodynamic**: 42 features
- **Stiffness/Mechanical**: 62 features
- **Entropy/Information**: 62 features
- **Advanced Biophysics**: 53 features
- **Bending/Curvature**: 44 features
- **PWM/TF Binding**: 277 features (34 TFs × 8 + 5 aggregate)

---

## 🔧 Technical Implementation

### GPU Optimization
- **Hardware**: NVIDIA A100 80GB PCIe
- **Framework**: PyTorch with CUDA acceleration
- **Batch Size**: 1,000 sequences
- **Processing Speed**: 8-10 sequences/second
- **Memory Usage**: ~500MB GPU memory per batch

### Data Processing
- **Input**: 230bp DNA sequences with expression values
- **Output**: TSV files with 540/516/540 features per sequence
- **Total Processed**: 248,793 sequences across 3 cell types

### Quality Control
- All features verified for proper variance (std > 0)
- Cell-type specific PWMs confirmed via JASPAR 2024
- Universal TFs present in all cell types
- Thermodynamic parameters validated against literature

---

## 📚 References

1. **SantaLucia Jr, J.** (1998). A unified view of polymer, dumbbell, and oligonucleotide DNA nearest-neighbor thermodynamics. PNAS 95(4), 1460-1465.

2. **Olson, W.K., et al.** (2001). A standard reference frame for the description of nucleic acid base-pair geometry. JMB 313(1), 229-237.

3. **Castro-Mondragon, J.A., et al.** (2024). JASPAR 2024: 20th anniversary of the open-access database of transcription factor binding profiles. NAR 52(D1), D174-D182.

4. **Rohs, R., et al.** (2009). The role of DNA shape in protein-DNA recognition. Nature 461(7268), 1248-1253.

5. **Zhou, T., et al.** (2013). DNAshape: a method for the high-throughput prediction of DNA structural features. NAR 41(W1), W56-W62.

---

*Generated by PhysiFormer Preprocessing Pipeline v2.0*
*Last Updated: August 2024*