# PWMs (Position Weight Matrices) - Transcription Factor Binding Motifs

## What are PWMs?

**PWMs ARE transcription factor binding motifs** - they're the same thing!

- **Motif**: The DNA sequence pattern that a transcription factor recognizes and binds to
- **PWM**: The mathematical representation (matrix) of that motif

**In other words**: PWM is the **data structure** that encodes the TF binding motif.

---

## Relationship Between TFs, Motifs, and PWMs

```
Transcription Factor (protein)
    ↓ (binds to DNA)
Binding Motif (sequence pattern)
    ↓ (represented mathematically as)
Position Weight Matrix (PWM)
    ↓ (stored in database)
JASPAR ID (e.g., MA0079.5)
```

**Example**:
- **Protein**: SP1 (Specificity Protein 1)
- **Motif**: GGGCGGGGC (GC-rich sequence)
- **PWM**: MA0079.5 (the matrix encoding this motif)
- **Source**: JASPAR 2024 database

---

## What Does a PWM Look Like?

### Example: SP1 (MA0079.5)

```
Position:     1      2      3      4      5      6      7      8      9
A:         0.001  0.001  0.001  0.001  0.001  0.001  0.001  0.001  0.001
C:         0.001  0.001  0.001  0.001  0.950  0.001  0.001  0.001  0.001
G:         0.950  0.997  0.997  0.997  0.001  0.997  0.997  0.997  0.997
T:         0.047  0.001  0.001  0.001  0.047  0.001  0.001  0.001  0.001
```

**Interpretation**:
- Position 1: 95% G, 4.7% T → Strong preference for G
- Position 2-4: 99.7% G → Must be G
- Position 5: 95% C → Strong preference for C
- Position 6-9: 99.7% G → Must be G

**Consensus sequence**: `GGGG-C-GGGG` = GC-rich motif for SP1 binding

---

### Example: GATA1 (MA0035.5) - Blood Cell TF

```
Position:     1      2      3      4      5      6      7
A:         0.055  0.018  0.692  0.940  0.045  0.968  0.065
C:         0.869  0.008  0.010  0.018  0.015  0.010  0.862
G:         0.038  0.009  0.012  0.014  0.934  0.012  0.035
T:         0.038  0.964  0.286  0.028  0.007  0.010  0.038
```

**Consensus sequence**: `C-T-A-A-G-A-C` = GATA motif (with flanking bases)

**Core**: `GATA` (positions 3-6) → Why it's called "GATA1"

---

## PWMs in This Project

### JASPAR Database

All PWMs come from **JASPAR 2024** - the gold standard TF binding motif database:
- Experimentally validated binding sites
- Compiled from ChIP-seq, SELEX, and other methods
- Each entry has unique ID (e.g., MA0079.5)

**Files used**:
- Human TFs: `data/JASPAR2024_CORE_non-redundant_pfms_meme.txt`
- Yeast TFs: `data/DREAM_data/scer_core2024_pfms.meme`

---

## Cell-Type Specific PWM Sets

### Why Different PWMs for Different Cell Types?

**Not all TFs are expressed in all cells!**

Different cell types have different **transcriptional programs**:

### 1. Universal PWMs (18 TFs) - Present in ALL human cell types

```python
'MA0079.5'  # SP1 - GC-rich core promoter
'MA0060.4'  # NFYA - CCAAT module
'MA0139.2'  # CTCF - Architectural/insulator
'MA0108.3'  # TBP - TATA box core promoter
'MA0099.4'  # FOS::JUN (AP-1) - Stress responsive
# ... (13 more housekeeping TFs)
```

**Function**: Core cellular processes, present everywhere

---

### 2. HepG2 PWMs (17 TFs) - Liver-Specific

```python
'MA0114.5'  # HNF4A - Master hepatocyte regulator
'MA0148.5'  # FOXA1 - Liver development
'MA0102.5'  # CEBPA - Hepatocyte differentiation
'MA0512.2'  # RXRA - Nuclear receptor
'MA2338.1'  # PPARA - Lipid metabolism
# ... (12 more liver TFs)
```

**Function**: Liver-specific genes (metabolism, detoxification)
**Total for HepG2**: 18 universal + 17 liver = **35 TFs**

---

### 3. K562 PWMs (15 TFs) - Blood Cell-Specific

```python
'MA0035.5'  # GATA1 - Erythroid master regulator
'MA0036.4'  # GATA2 - Hematopoietic stem cells
'MA0140.3'  # GATA1::TAL1 - Erythroid complex
'MA0493.3'  # KLF1 - Globin gene activation
'MA0841.2'  # NFE2 - Hemoglobin regulation
# ... (10 more blood TFs)
```

**Function**: Blood cell differentiation, hemoglobin production
**Total for K562**: 18 universal + 15 blood = **33 TFs**

---

### 4. WTC11 PWMs (17 TFs) - Stem Cell/Pluripotency

```python
'MA0142.1'  # POU5F1::SOX2 - Pluripotency master complex
'MA0143.5'  # SOX2 - Stem cell maintenance
'MA2339.1'  # NANOG - Self-renewal
'MA0039.5'  # KLF4 - Yamanaka factor
'MA0141.4'  # ESRRB - Pluripotency support
# ... (12 more pluripotency TFs)
```

**Function**: Maintain stem cell state, prevent differentiation
**Total for WTC11**: 18 universal + 17 pluripotency = **35 TFs**

---

### 5. DREAM PWMs (170 TFs) - Yeast-Specific

```python
'MA0265.3'  # Yeast TF 1
'MA0266.2'  # Yeast TF 2
# ... (168 more yeast TFs)
```

**Function**: All S. cerevisiae transcription factors
**Total for DREAM**: **170 yeast TFs** (no universal, different organism)

---

## How PWMs Are Used for Feature Extraction

### Scanning Process

For each DNA sequence:

1. **Slide PWM along sequence** (both strands)
2. **Calculate binding score** at each position
3. **Extract features**:
   - Maximum score (best match)
   - Mean score (overall affinity)
   - Number of strong binding sites
   - Position of best match

### Example: Scanning SP1 motif (MA0079.5)

```
Sequence: ATGGGCGGGGCTTA...

Position 3-11: GGGCGGGGC
  Score = log-odds of this matching SP1 motif
  Score = 12.5 (very high → strong SP1 binding site)

Feature generated:
  pwm_MA0079.5_max_score = 12.5
  pwm_MA0079.5_mean_score = 4.2
  pwm_MA0079.5_num_sites = 3
```

---

## Total PWM Features Per Cell Type

### For 230bp sequences:

| Cell Type | # TFs | # PWM Features |
|-----------|-------|----------------|
| WTC11 | 35 | ~140 features |
| HepG2 | 35 | ~140 features |
| K562 | 33 | ~132 features |
| DREAM | 170 | ~680 features |

**Feature types per TF** (~4 features):
- Max binding score
- Mean binding score
- Number of binding sites
- Position of best site

**Total biophysical features**:
- WTC11: 539 total (140 PWM + 399 others)
- HepG2: 536 total (140 PWM + 396 others)
- K562: 515 total (132 PWM + 383 others)
- S2: 386 total

---

## Real Examples from Your Data

### Example 1: TATA Box Detection

```
MOTIF: MA0108.3 TBP (TATA-binding protein)
Consensus: TATAWAW (W = A or T)

Sequence: ...ATATAAAA... (promoter region)
  ↓
pwm_MA0108.3_max_score = 11.2 (strong TATA box)
  → Likely active core promoter

Sequence: ...CGCGATGC... (no TATA)
  ↓
pwm_MA0108.3_max_score = 2.1 (no TATA box)
  → TATA-less promoter
```

---

### Example 2: Liver-Specific Enhancer

```
HepG2 sequence with liver TF binding sites:

pwm_MA0114.5_max_score = 13.5  (HNF4A - strong)
pwm_MA0148.5_max_score = 10.2  (FOXA1 - moderate)
pwm_MA0102.5_max_score = 12.8  (CEBPA - strong)
  → Likely active liver enhancer

K562 sequence (same enhancer in blood cells):
pwm_MA0114.5_max_score = 3.2   (no HNF4A)
pwm_MA0148.5_max_score = 2.8   (no FOXA1)
pwm_MA0102.5_max_score = 3.5   (no CEBPA)
  → Inactive in blood cells (no liver TFs)
```

---

### Example 3: Blood Cell Enhancer

```
K562 sequence with GATA motifs:

pwm_MA0035.5_max_score = 14.2  (GATA1 - very strong)
pwm_MA0140.3_max_score = 11.5  (GATA1::TAL1 complex)
pwm_MA0493.3_max_score = 9.8   (KLF1)
  → Active erythroid enhancer

HepG2 sequence (same region in liver):
pwm_MA0035.5_max_score = 4.1   (no GATA1)
pwm_MA0140.3_max_score = 3.2   (no complex)
pwm_MA0493.3_max_score = 3.8   (no KLF1)
  → Inactive in liver (no blood TFs)
```

---

## Key Concepts

### 1. PWM = Motif = TF Binding Site

They're all the same thing, just different terminology:
- **Biologists** say: "TF binding motif"
- **Computational biologists** say: "PWM"
- **Database** says: "MA0079.5"

### 2. Cell-Type Specificity

Using the right PWMs for the right cell type is critical:
- HepG2 sequences → Use liver TF PWMs
- K562 sequences → Use blood TF PWMs
- Don't scan for TFs that aren't expressed!

### 3. PWM Features Capture Regulatory Logic

High PWM scores → TF can bind → Gene can be regulated
- Multiple strong PWM scores → Cooperative regulation
- Cell-specific PWMs → Cell-specific regulation

### 4. Complementary to Other Features

PWMs tell you about **TF binding potential**, combined with:
- **Thermodynamics**: Will DNA be accessible?
- **Electrostatics**: Will TF be attracted electrostatically?
- **Shape**: Will TF physically fit in groove?
- **Bending**: Will DNA bend to facilitate binding?

Together = complete picture of regulatory potential

---

## Summary

| Term | What It Is |
|------|------------|
| **Transcription Factor (TF)** | Protein that binds DNA to regulate genes |
| **Binding Motif** | DNA sequence pattern the TF recognizes |
| **PWM (Position Weight Matrix)** | Mathematical matrix encoding the motif |
| **JASPAR ID** | Database identifier (e.g., MA0079.5) |

**In this project**:
- PWMs from JASPAR 2024 database
- Cell-type specific TF sets (35-170 TFs)
- ~140-680 PWM features per sequence
- Combined with 250-400 other biophysical features
- Used to train PhysInformer models

**Bottom line**: PWMs and motifs are the same thing - PWM is just the mathematical way of representing a TF's DNA binding preference!

---

**Document created**: 2025-11-12
**Location**: `/home/bcheng/sequence_optimization/mainproject/PhysiFormer/physpreprocess/PWM_EXPLANATION.md`
