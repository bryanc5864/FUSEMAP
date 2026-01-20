# PhysInformer: Comprehensive Technical Documentation
## Physics-Aware Neural Network for DNA Sequence Property Prediction

Version 1.0 | Last Updated: September 2025

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Core Neural Network Components](#core-neural-network-components)
4. [Physics Integration Framework](#physics-integration-framework)
5. [Data Pipeline Architecture](#data-pipeline-architecture)
6. [Training Infrastructure](#training-infrastructure)
7. [Auxiliary Head Architecture](#auxiliary-head-architecture)
8. [Feature Engineering Pipeline](#feature-engineering-pipeline)
9. [Evaluation Metrics System](#evaluation-metrics-system)
10. [Production Deployment](#production-deployment)
11. [Mathematical Formulations](#mathematical-formulations)
12. [Implementation Details](#implementation-details)
13. [Configuration Management](#configuration-management)
14. [Troubleshooting Guide](#troubleshooting-guide)
15. [API Reference](#api-reference)

---

## 1. Executive Summary

### 1.1 System Overview

PhysInformer represents a groundbreaking advancement in DNA sequence analysis through the integration of physics-aware neural network architectures with biophysical modeling principles. The system combines deep learning with thermodynamic constraints, mechanical properties, and sequence-specific features to predict a comprehensive set of biophysical descriptors from raw DNA sequences.

### 1.2 Key Innovations

#### 1.2.1 Physics-Aware Architecture
- **PWM Convolution Stems**: Position Weight Matrix-inspired convolutional layers that capture transcription factor binding motifs
- **Simplified State Space Models (SSM)**: Efficient long-range dependency modeling with O(N) complexity
- **Dual-Path Feature Pyramids**: Multi-resolution processing for capturing both local and global sequence patterns
- **Physics Routers**: Intelligent feature routing based on biophysical property categories
- **Heteroscedastic Loss Functions**: Uncertainty-aware training with learnable feature-specific weights

#### 1.2.2 Biophysical Integration
- **Thermodynamic Constraints**: Enforces ΔG = ΔH - TΔS relationships
- **Temperature-Dependent Modeling**: Operates at biological temperature (310K)
- **Mechanical Property Prediction**: DNA flexibility, stiffness, and bending characteristics
- **Sequence Complexity Analysis**: Entropy-based feature extraction

### 1.3 Performance Characteristics
- Supports multiple cell types: HepG2, K562, WTC11, S2
- Predicts 536+ biophysical descriptors simultaneously
- Achieves Pearson correlations > 0.91 on validation sets
- Processes sequences up to 249 base pairs
- Handles batch sizes up to 128 sequences on modern GPUs

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture

```
Input DNA Sequence (A, C, G, T, N)
         ↓
[Embedding Layer (5 → 64)]
         ↓
[PWM Convolution Stem]
    ├── Conv1D (k=11, padding=5)
    ├── ReLU + BatchNorm
    ├── Conv1D (k=9, padding=4)
    ├── ReLU + BatchNorm
    └── Conv1D (k=7, padding=3)
         ↓
[SSM Layers (×2)]
    ├── State Space Model
    ├── Layer Normalization
    └── Residual Connection
         ↓
[Dual-Path Feature Pyramid]
    ├── Path 1: Conv → Pool → Conv
    └── Path 2: Conv → Attention → Conv
         ↓
[Physics Router]
    ├── PWM Features Head
    ├── Bend Features Head
    ├── Stiffness Features Head
    ├── Thermodynamic Head
    ├── Entropy Head
    ├── Advanced Features Head
    └── Global Features Head
         ↓
[Heteroscedastic Output]
    ├── Mean Predictions (μ)
    └── Log Variance (log σ²)
```

### 2.2 Module Organization

#### 2.2.1 Core Model Files
- `physics_aware_model.py`: Main architecture implementation (900+ lines)
- `model.py`: Legacy PhysInformer components
- `dataset.py`: Data loading and preprocessing pipeline
- `metrics.py`: Comprehensive evaluation metrics
- `train.py`: Training orchestration and management
- `inference_physics.py`: Production inference system

#### 2.2.2 Analysis and Visualization
- `plotting.py`: Training progress visualization
- `metrics_single.py`: Single-model evaluation utilities
- `plotting_single.py`: Individual model visualization
- `tileformer_single.py`: Tileformer baseline components

#### 2.2.3 Configuration Scripts
- `run_hepg2.sh`: HepG2 cell line training configuration
- `run_k562.sh`: K562 cell line training configuration
- `run_wtc11.sh`: WTC11 cell line training configuration
- `train_physics.sh`: Physics-aware model training script
- `test_resume.py`: Resume functionality testing

### 2.3 Directory Structure

```
PhysInformer/
├── physics_aware_model.py     # Core physics-aware architecture
├── dataset.py                 # Data pipeline implementation
├── metrics.py                 # Evaluation metrics
├── train.py                   # Training orchestration
├── inference_physics.py       # Production inference
├── plotting.py                # Visualization utilities
├── test_resume.py            # Resume testing
├── runs/                     # Training runs directory
│   ├── K562_*/              # K562 training runs
│   ├── WTC11_*/             # WTC11 training runs
│   └── HepG2_*/             # HepG2 training runs
└── __pycache__/             # Python cache files
```

---

## 3. Core Neural Network Components

### 3.1 PWM Convolution Stem

#### 3.1.1 Architecture Design

The PWM Convolution Stem is inspired by Position Weight Matrices used in transcription factor binding site prediction. It employs a hierarchical convolutional architecture with decreasing kernel sizes to capture motifs at multiple scales.

```python
class PWMConvStem(nn.Module):
    """
    PWM-inspired convolutional stem for DNA sequence processing
    
    Parameters:
        vocab_size (int): Size of DNA vocabulary (5 for A,C,G,T,N)
        hidden_dims (List[int]): Channel dimensions [128, 192, 256]
        kernel_sizes (List[int]): Kernel sizes [11, 9, 7]
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, vocab_size=5, hidden_dims=[128, 192, 256], 
                 kernel_sizes=[11, 9, 7], dropout=0.1):
        super().__init__()
        
        # Initial embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dims[0] // 2)
        
        # Hierarchical convolutions
        self.conv_layers = nn.ModuleList()
        in_channels = hidden_dims[0] // 2
        
        for i, (out_channels, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            padding = (kernel_size - 1) // 2
            
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            
            self.conv_layers.append(conv_block)
            in_channels = out_channels
```

#### 3.1.2 Motif Detection Mechanism

The PWM stem learns position-specific scoring matrices through convolution operations:

1. **First Layer (k=11)**: Captures extended motifs and regulatory elements
2. **Second Layer (k=9)**: Identifies core binding sites
3. **Third Layer (k=7)**: Refines local sequence patterns

Each convolutional filter acts as a learnable PWM, with the network automatically discovering relevant motifs during training.

#### 3.1.3 Batch Normalization Strategy

Batch normalization is applied after each convolution to:
- Stabilize gradient flow during training
- Enable higher learning rates
- Reduce internal covariate shift
- Improve convergence speed

### 3.2 Simplified State Space Models (SSM)

#### 3.2.1 Mathematical Foundation

The SSM layer implements a continuous-time state space model discretized for sequence processing:

```
State Evolution:    x'(t) = Ax(t) + Bu(t)
Observation:        y(t) = Cx(t) + Du(t)

Discretized Form:
x[k+1] = Āx[k] + B̄u[k]
y[k] = C̄x[k] + D̄u[k]

where:
Ā = exp(ΔA)
B̄ = (Ā - I) × A⁻¹ × B
C̄ = C
D̄ = D
```

#### 3.2.2 Implementation Details

```python
class SimplifiedSSMLayer(nn.Module):
    """
    Simplified State Space Model for efficient sequence modeling
    
    Parameters:
        d_model (int): Model dimension (256)
        d_state (int): State dimension (16)
        dropout (float): Dropout probability (0.1)
    """
    
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state) / d_state)
        self.B = nn.Parameter(torch.randn(d_state, d_model) / d_model)
        self.C = nn.Parameter(torch.randn(d_model, d_state) / d_state)
        self.D = nn.Parameter(torch.randn(d_model, d_model) / d_model)
        
        # Discretization parameter
        self.delta = nn.Parameter(torch.ones(1) * 0.01)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
```

#### 3.2.3 Computational Efficiency

The SSM layer achieves O(N) complexity through:
- Parallel scan algorithms for state computation
- Efficient matrix exponential approximation
- Cached discretization parameters
- Vectorized operations across batch and sequence dimensions

### 3.3 Dual-Path Feature Pyramid

#### 3.3.1 Architecture Motivation

The dual-path design captures both local and global sequence dependencies:

**Path 1 (Local)**: Convolutional operations with pooling
- Captures short-range interactions
- Identifies local motifs and patterns
- Maintains translational invariance

**Path 2 (Global)**: Self-attention mechanisms
- Models long-range dependencies
- Captures positional relationships
- Enables global context integration

#### 3.3.2 Implementation

```python
class DualPathFeaturePyramid(nn.Module):
    """
    Dual-path feature extraction with local and global pathways
    
    Parameters:
        d_input (int): Input dimension
        d_hidden (int): Hidden dimension
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_input, d_hidden=128, dropout=0.1):
        super().__init__()
        
        # Path 1: Local features (CNN)
        self.local_path = nn.Sequential(
            nn.Conv1d(d_input, d_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(inplace=True)
        )
        
        # Path 2: Global features (Attention)
        self.global_path = nn.MultiheadAttention(
            embed_dim=d_input,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_input + d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
```

#### 3.3.3 Feature Fusion Strategy

The pyramid combines local and global features through:
1. **Concatenation**: Preserves information from both pathways
2. **Linear Projection**: Learns optimal feature combination
3. **Layer Normalization**: Stabilizes feature magnitudes
4. **Dropout**: Prevents overfitting to specific pathways

### 3.4 Physics Router

#### 3.4.1 Conceptual Design

The Physics Router intelligently directs features to specialized prediction heads based on their biophysical properties. This design reflects the principle that different physical properties require different computational approaches.

#### 3.4.2 Feature Categories

```python
PROPERTY_GROUPS = {
    'pwm': ['pwm_', 'tf_', 'binding_'],           # TF binding features
    'bend': ['bend_', 'curvature_', 'flexibility_'], # DNA flexibility
    'stiff': ['stiff_', 'rigid_', 'elastic_'],    # Mechanical stiffness
    'thermo': ['thermo_', 'dG', 'dH', 'dS', 'Tm'], # Thermodynamics
    'entropy': ['entropy_', 'complexity_', 'info_'], # Information content
    'advanced': ['g4_', 'melting_', 'stress_'],   # Advanced properties
    'global': []  # Catch-all for unmatched features
}
```

#### 3.4.3 Routing Implementation

```python
class PhysicsRouter(nn.Module):
    """
    Routes features to specialized prediction heads
    
    Parameters:
        d_input (int): Input feature dimension
        descriptor_names (List[str]): Names of output descriptors
        property_groups (Dict): Mapping of property categories
    """
    
    def __init__(self, d_input, descriptor_names, property_groups):
        super().__init__()
        
        # Create routing indices
        self.feature_indices = {}
        for prop_name, keywords in property_groups.items():
            indices = []
            for i, name in enumerate(descriptor_names):
                if any(kw in name.lower() for kw in keywords):
                    indices.append(i)
            self.feature_indices[prop_name] = indices
        
        # Create specialized heads
        self.heads = nn.ModuleDict()
        for prop_name, indices in self.feature_indices.items():
            if len(indices) > 0:
                self.heads[prop_name] = PropertyHead(
                    d_input=d_input,
                    n_outputs=len(indices),
                    property_type=prop_name
                )
```

#### 3.4.4 Property-Specific Processing

Each property head employs specialized architectures:

**PWM Head**: Additional convolutions for motif refinement
**Bend Head**: Smooth activation functions for continuous curvature
**Stiffness Head**: Symmetric processing for mechanical properties
**Thermo Head**: Temperature-aware transformations
**Entropy Head**: Information-theoretic operations
**Advanced Head**: Multi-scale feature extraction

### 3.5 Heteroscedastic Loss Functions

#### 3.5.1 Theoretical Foundation

Heteroscedastic regression models both the mean and variance of predictions:

```
p(y|x) = N(μ(x), σ²(x))

Loss = -log p(y|x) = 0.5 * [log(2πσ²(x)) + (y - μ(x))²/σ²(x)]
```

This allows the model to express uncertainty about different features.

#### 3.5.2 Adaptive Weight Learning

```python
class HeteroscedasticLoss(nn.Module):
    """
    Heteroscedastic loss with learnable feature weights
    
    Learns uncertainty for each feature independently
    """
    
    def __init__(self, n_features):
        super().__init__()
        # Initialize log variances
        self.log_vars = nn.Parameter(torch.zeros(n_features))
        
    def forward(self, predictions, targets):
        # Compute precision from log variance
        precision = torch.exp(-self.log_vars)
        
        # Weighted MSE loss
        mse_loss = precision * (predictions - targets) ** 2
        
        # Uncertainty regularization
        uncertainty_loss = self.log_vars
        
        # Combined loss
        return torch.mean(mse_loss + uncertainty_loss)
```

#### 3.5.3 Benefits of Heteroscedastic Modeling

1. **Automatic Feature Weighting**: Learns importance of each descriptor
2. **Uncertainty Quantification**: Provides confidence estimates
3. **Robust to Noise**: Down-weights noisy features automatically
4. **Improved Convergence**: Better gradient flow for difficult features

---

## 4. TileFormer Model

### 4.1 Architecture Overview

TileFormer is a transformer-based model designed to predict electrostatic potential (Ψ) features from DNA sequences. It serves as a pre-training step for extracting biophysical representations.

#### 4.1.1 Core Architecture

```python
class TileFormer(nn.Module):
    def __init__(
        vocab_size: int = 5,        # A, C, G, T, N
        d_model: int = 256,          # Hidden dimension
        n_heads: int = 8,            # Attention heads
        n_layers: int = 6,           # Transformer blocks
        d_ff: int = 1024,           # Feed-forward dimension
        max_len: int = 200,         # Maximum sequence length
        output_dim: int = 6,        # Ψ predictions
        predict_uncertainty: bool = True
    )
```

#### 4.1.2 Output Features

TileFormer predicts 18 electrostatic potential features across 3 windows:

**Per Window (w0, w1, w2):**
- `STD_PSI_MIN`: Minimum Ψ in standard conditions
- `STD_PSI_MAX`: Maximum Ψ in standard conditions  
- `STD_PSI_MEAN`: Mean Ψ in standard conditions
- `ENH_PSI_MIN`: Minimum Ψ in enhanced conditions
- `ENH_PSI_MAX`: Maximum Ψ in enhanced conditions
- `ENH_PSI_MEAN`: Mean Ψ in enhanced conditions

**Windows:**
- w0: Full sequence (global)
- w1: First half of sequence
- w2: Second half of sequence

#### 4.1.3 Key Components

1. **Nucleotide Embedding**: Learned embeddings for DNA bases
2. **Positional Encoding**: Sinusoidal position information
3. **Transformer Blocks**: Self-attention + feed-forward layers
4. **Global Pooling**: Average pooling over sequence dimension
5. **Prediction Heads**: Separate heads for Ψ values and uncertainty

### 4.2 TileFormer Features in PhysInformer

The 18 TileFormer features are concatenated with physics features as additional inputs to PhysInformer, providing learned biophysical representations that complement the hand-crafted physics features.

## 5. Physics Integration Framework

### 4.1 Thermodynamic Constraints

#### 4.1.1 Gibbs Free Energy Relationship

The model enforces the fundamental thermodynamic equation:

```
ΔG = ΔH - T·ΔS

where:
ΔG = Gibbs free energy change
ΔH = Enthalpy change
T = Temperature (310K for biological systems)
ΔS = Entropy change
```

#### 4.1.2 Implementation

```python
class ThermodynamicHead(nn.Module):
    """
    Thermodynamically-constrained prediction head
    
    Ensures consistency between ΔG, ΔH, and ΔS predictions
    """
    
    def __init__(self, d_input, temperature=310.0):
        super().__init__()
        self.temperature = temperature
        
        # Separate predictors for ΔH and ΔS
        self.enthalpy_predictor = nn.Linear(d_input, 1)
        self.entropy_predictor = nn.Linear(d_input, 1)
        
    def forward(self, x):
        # Predict components
        delta_h = self.enthalpy_predictor(x)
        delta_s = self.entropy_predictor(x)
        
        # Compute ΔG with constraint
        delta_g = delta_h - self.temperature * delta_s
        
        return {
            'delta_g': delta_g,
            'delta_h': delta_h,
            'delta_s': delta_s
        }
```

#### 4.1.3 Temperature-Dependent Modeling

The system operates at physiological temperature (310K = 37°C):
- Reflects biological conditions
- Enables accurate free energy calculations
- Supports temperature-dependent feature predictions

### 4.2 DNA Mechanical Properties

#### 4.2.1 Bending and Flexibility

The model predicts DNA bending propensity through:

```python
class BendingAnalyzer:
    """
    Analyzes DNA bending and flexibility properties
    """
    
    def compute_bending_features(self, sequence):
        features = {}
        
        # Roll-tilt-twist parameters
        features['roll'] = self.compute_roll_angles(sequence)
        features['tilt'] = self.compute_tilt_angles(sequence)
        features['twist'] = self.compute_twist_angles(sequence)
        
        # Curvature analysis
        features['curvature'] = self.compute_curvature(
            features['roll'], 
            features['tilt']
        )
        
        # Persistence length
        features['persistence'] = self.estimate_persistence_length(
            features['curvature']
        )
        
        return features
```

#### 4.2.2 Stiffness Matrix Computation

The stiffness matrix captures mechanical properties:

```
K = | k_roll    k_roll,tilt  k_roll,twist  |
    | k_tilt,roll  k_tilt    k_tilt,twist  |
    | k_twist,roll k_twist,tilt k_twist     |
```

Each element represents coupling between deformation modes.

### 4.3 Sequence Complexity Measures

#### 4.3.1 Shannon Entropy

```python
def compute_shannon_entropy(sequence, k=2):
    """
    Compute k-mer Shannon entropy
    """
    # Count k-mers
    kmer_counts = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
    
    # Compute probabilities
    total = sum(kmer_counts.values())
    probs = [count/total for count in kmer_counts.values()]
    
    # Shannon entropy
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    
    return entropy
```

#### 4.3.2 Complexity Metrics

- **Lempel-Ziv Complexity**: Compression-based complexity
- **Conditional Entropy**: Context-dependent information
- **Mutual Information**: Positional dependencies
- **Renyi Entropy**: Generalized entropy measures

---

## 5. Data Pipeline Architecture

### 5.1 Dataset Implementation

#### 5.1.1 PhysInformerDataset Class

```python
class PhysInformerDataset(Dataset):
    """
    Dataset for PhysInformer training
    
    Handles:
    - Descriptor loading and filtering
    - Zero-variance feature removal
    - Normalization
    - Activity score integration
    - Multi-cell type support
    """
    
    def __init__(
        self,
        descriptors_file: str,
        cell_type: str = None,
        normalize: bool = True,
        load_activities: bool = False
    ):
        # Load data
        self.df = pd.read_csv(descriptors_file, sep='\t')
        self.cell_type = cell_type
        self.load_activities = load_activities
        
        # Extract descriptor columns
        exclude_cols = [
            'seq_id', 'sequence_id', 'condition', 
            'normalized_log2', 'n_obs_bc', 'n_replicates', 
            'sequence', 'name', 'sequence_length'
        ]
        self.descriptor_cols = [
            c for c in self.df.columns 
            if c not in exclude_cols
        ]
        
        # Filter zero-variance features
        self._filter_zero_variance()
        
        # Setup normalization
        if normalize:
            self._compute_normalization_stats()
        
        # Load activity scores
        if load_activities:
            self._load_activity_scores()
```

#### 5.1.2 Zero-Variance Filtering

```python
def _filter_zero_variance(self):
    """
    Remove features with zero or near-zero variance
    """
    desc_values = self.df[self.descriptor_cols].values.astype(np.float32)
    feature_stds = desc_values.std(axis=0)
    
    # Find valid features (std > 1e-8)
    valid_features = feature_stds > 1e-8
    n_removed = np.sum(~valid_features)
    
    if n_removed > 0:
        print(f"Removing {n_removed} zero-variance features")
        # Filter columns
        self.descriptor_cols = [
            col for i, col in enumerate(self.descriptor_cols) 
            if valid_features[i]
        ]
        print(f"Remaining features: {len(self.descriptor_cols)}")
```

#### 5.1.3 Cross-Split Harmonization

```python
def create_dataloaders(cell_type, data_dir, batch_size=32):
    """
    Create harmonized dataloaders across train/val/test
    """
    datasets = {}
    
    # Load all splits
    for split in ['train', 'val', 'test']:
        desc_file = f"{data_dir}/{cell_type}_{split}_descriptors.tsv"
        datasets[split] = PhysInformerDataset(desc_file, cell_type)
    
    # Find common features
    common_cols = set(datasets['train'].descriptor_cols)
    for split in ['val', 'test']:
        common_cols = common_cols.intersection(
            set(datasets[split].descriptor_cols)
        )
    
    # Update each dataset to use only common features
    for dataset in datasets.values():
        dataset.descriptor_cols = sorted(list(common_cols))
        
    return datasets
```

### 5.2 Biophysical Descriptor Computation

#### 5.2.1 Feature Categories

The system computes 536+ biophysical descriptors across 7 categories:

**1. PWM Features (277 total features)**

PWM features are computed for cell-type specific transcription factor binding motifs.
Each motif generates 8 metrics:
- `max_score`: Maximum PWM match score across sequence
- `mean_score`: Average PWM match score
- `var_score`: Variance of PWM scores  
- `num_high_affinity`: Count of high-affinity binding sites
- `total_weight`: Sum of all binding weights
- `entropy`: Information entropy of binding site distribution
- `delta_g`: Gibbs free energy of binding
- `top_k_mean`: Mean of top-k highest scores

**Cell-Type Specific Motifs:**

*HepG2 (34 unique motifs):*
- MA0018.5 (CREB1), MA0024.3 (E2F1), MA0028.3 (ELK1), MA0046.3 (HNF1A), MA0047.4 (FOXA2)
- MA0058.4 (MAX), MA0060.4 (NFYA), MA0079.5 (SP1), MA0083.3 (SRF), MA0088.2 (ZNF143)
- MA0090.4 (TEAD1), MA0093.4 (USF1), MA0095.4 (YY1), MA0099.4 (FOS::JUN), MA0102.5 (CEBPA)
- MA0108.3 (TBP), MA0114.5 (HNF4A), MA0139.2 (CTCF), MA0148.5 (FOXA1), MA0153.2 (HNF1B)
- MA0466.4 (CEBPB), MA0484.3 (HNF4G), MA0494.2 (Nr1h3::Rxra), MA0506.3 (Nrf1), MA0512.2 (Rxra)
- MA0526.5 (USF2), MA0527.2 (ZBTB33), MA0679.3 (ONECUT1), MA0809.3 (TEAD4), MA0836.3 (CEBPD)
- MA1148.2 (PPARA::RXRA), MA1683.2 (FOXA3), MA2337.1 (Nr1h3), MA2338.1 (Ppara)

*K562 (31 unique motifs):*
- MA0018.5 (CREB1), MA0024.3 (E2F1), MA0028.3 (ELK1), MA0035.5 (GATA1), MA0036.4 (GATA2)
- MA0058.4 (MAX), MA0060.4 (NFYA), MA0079.5 (SP1), MA0083.3 (SRF), MA0088.2 (ZNF143)
- MA0090.4 (TEAD1), MA0091.2 (TAL1::TCF3), MA0093.4 (USF1), MA0095.4 (YY1), MA0099.4 (FOS::JUN)
- MA0108.3 (TBP), MA0139.2 (CTCF), MA0140.3 (GATA1::TAL1), MA0150.3 (Nfe2l2), MA0493.3 (KLF1)
- MA0495.4 (MAFF), MA0496.4 (MAFK), MA0501.2 (MAF::NFE2), MA0502.3 (NFYB), MA0506.3 (Nrf1)
- MA0526.5 (USF2), MA0527.2 (ZBTB33), MA0809.3 (TEAD4), MA0841.2 (NFE2), MA1516.2 (KLF3), MA1633.2 (BACH1)

*WTC11 (34 unique motifs):*
- MA0018.5 (CREB1), MA0024.3 (E2F1), MA0028.3 (ELK1), MA0039.5 (KLF4), MA0058.4 (MAX)
- MA0059.2 (MAX::MYC), MA0060.4 (NFYA), MA0077.2 (SOX9), MA0078.3 (Sox17), MA0079.5 (SP1)
- MA0083.3 (SRF), MA0084.2 (SRY), MA0088.2 (ZNF143), MA0090.4 (TEAD1), MA0093.4 (USF1)
- MA0095.4 (YY1), MA0099.4 (FOS::JUN), MA0108.3 (TBP), MA0139.2 (CTCF), MA0141.4 (ESRRB)
- MA0142.1 (Pou5f1::Sox2), MA0143.5 (SOX2), MA0506.3 (Nrf1), MA0514.3 (Sox3), MA0515.1 (Sox6)
- MA0526.5 (USF2), MA0527.2 (ZBTB33), MA0785.2 (POU2F1), MA0787.1 (POU3F2), MA0809.3 (TEAD4)
- MA0866.1 (SOX21), MA0870.1 (Sox1), MA1723.2 (PRDM9), MA2339.1 (Nanog)

*S2 (Drosophila, 13 unique motifs):*
- Jra (Jun homolog - AP-1 component)
- Kay (Fos homolog - AP-1 component)  
- Stat92E (STAT homolog)
- SREBP/HLH106 (Sterol regulatory element-binding protein)
- Achaete (Basic helix-loop-helix)
- Daughterless (Basic helix-loop-helix)
- GATAe (GATA factor)
- Serpent (GATA factor)
- M1BP (Motif 1 binding protein)
- Pointed (ETS domain)
- Trl (Trithorax-like/GAGA factor)
- USF (Upstream stimulatory factor)
- ZIPIC (Zinc finger protein)

Additionally, aggregate PWM features are computed:
- `pwm_best_tf`: Best matching TF overall
- `pwm_max_*`: Maximum values across all TFs
- `pwm_min_*`: Minimum values across all TFs  
- `pwm_sum_*`: Sum across all TFs

**2. Bend Features (44 features)**

DNA bending features capture intrinsic curvature and flexibility:

- `bend_total_bending_energy`: Total bending energy across sequence
- `bend_mean_bending_cost`: Average bending cost per position
- `bend_max_bending_cost`: Maximum bending cost
- `bend_bending_energy_variance`: Variance in bending energy distribution
- `bend_rms_curvature_w{5,7,9,11}_mean`: RMS curvature mean (windows 5,7,9,11)
- `bend_rms_curvature_w{5,7,9,11}_max`: RMS curvature max (windows 5,7,9,11)
- `bend_curvature_var_w{5,7,9,11}_mean`: Curvature variance mean
- `bend_curvature_var_w{5,7,9,11}_max`: Curvature variance max
- `bend_curvature_gradient_mean`: Mean gradient of curvature
- `bend_curvature_gradient_max`: Maximum gradient of curvature
- `bend_max_bend_w{5,7,9,11}_mean`: Maximum bend mean (windows 5,7,9,11)
- `bend_max_bend_w{5,7,9,11}_global_max`: Global maximum bend
- `bend_max_bend_w{5,7,9,11}_fraction_at_global_max`: Fraction at global max
- `bend_hotspot_count`: Number of bending hotspots
- `bend_hotspot_density`: Density of bending hotspots
- `bend_spectral_f{0p200,0p143,0p100}_mean_power`: Spectral mean power
- `bend_spectral_f{0p200,0p143,0p100}_max_power`: Spectral max power
- `bend_attention_bias_mean`: Mean attention bias from bending
- `bend_attention_bias_min`: Minimum attention bias (often constant, removed)

**3. Stiffness Features (62 features)**

DNA mechanical stiffness and deformability metrics:

- `stiff_total_relative_energy`: Total relative stacking energy
- `stiff_mean_relative_energy`: Mean relative energy
- `stiff_var_relative_energy`: Variance in relative energy
- `stiff_max_relative_energy`: Maximum relative energy
- `stiff_min_relative_energy`: Minimum relative energy
- `stiff_twist_total_energy`: Total twist deformation energy
- `stiff_twist_mean_energy`: Mean twist energy
- `stiff_twist_var_energy`: Variance in twist energy
- `stiff_twist_max_energy`: Maximum twist energy
- `stiff_twist_min_energy`: Minimum twist energy
- `stiff_tilt_total_energy`: Total tilt deformation energy
- `stiff_tilt_mean_energy`: Mean tilt energy
- `stiff_tilt_var_energy`: Variance in tilt energy
- `stiff_tilt_max_energy`: Maximum tilt energy
- `stiff_tilt_min_energy`: Minimum tilt energy
- `stiff_roll_total_energy`: Total roll deformation energy
- `stiff_roll_mean_energy`: Mean roll energy
- `stiff_roll_var_energy`: Variance in roll energy
- `stiff_roll_max_energy`: Maximum roll energy
- `stiff_roll_min_energy`: Minimum roll energy
- `stiff_pca_component_{1,2,3}`: First 3 principal components
- `stiff_pca_explained_var_{1,2,3}`: Explained variance ratios
- `stiff_pca_cumulative_var_{1,2,3}`: Cumulative variance explained
- `stiff_total_deformation_zscore`: Z-score of total deformation
- `stiff_mean_deformation_zscore`: Z-score of mean deformation
- `stiff_max_deformation_zscore`: Z-score of max deformation
- `stiff_high_energy_regions_count`: Count of high-energy regions
- `stiff_high_energy_fraction`: Fraction of high-energy positions
- `stiff_energy_distribution_entropy`: Entropy of energy distribution
- `stiff_energy_gc_correlation`: Correlation with GC content
- `stiff_energy_at_correlation`: Correlation with AT content
- `stiff_energy_autocorr_lag{1,2,3,5,10}`: Autocorrelation at various lags
- `stiff_energy_gradient_mean`: Mean energy gradient
- `stiff_energy_gradient_max`: Maximum energy gradient
- `stiff_energy_gradient_std`: Standard deviation of gradient
- `stiff_smoothness_score`: Overall smoothness metric
- `stiff_periodicity_score_{3,10}`: Periodicity at 3bp and 10bp

**4. Thermodynamic Features (45 features)**

Thermodynamic stability and melting properties:

- `thermo_total_dH`: Total enthalpy change (ΔH)
- `thermo_mean_dH`: Mean ΔH per dinucleotide
- `thermo_var_dH`: Variance in ΔH
- `thermo_total_dS`: Total entropy change (ΔS)
- `thermo_mean_dS`: Mean ΔS per dinucleotide
- `thermo_var_dS`: Variance in ΔS
- `thermo_min_dS`: Minimum ΔS (often constant, removed)
- `thermo_max_dS`: Maximum ΔS (often constant, removed)
- `thermo_total_dG`: Total Gibbs free energy (ΔG)
- `thermo_mean_dG`: Mean ΔG per dinucleotide
- `thermo_var_dG`: Variance in ΔG
- `thermo_min_dG`: Minimum ΔG
- `thermo_max_dG`: Maximum ΔG (often constant, removed)
- `thermo_dG_p{5,10,25,50,75,90,95}`: ΔG percentiles
- `thermo_dH_p{5,10,25,50,75,90,95}`: ΔH percentiles
- `thermo_dS_p{5,10,25,50,75,90,95}`: ΔS percentiles
- `thermo_dG_iqr`: Interquartile range of ΔG
- `thermo_dH_iqr`: Interquartile range of ΔH
- `thermo_dS_iqr`: Interquartile range of ΔS
- `thermo_melting_temp_wallace`: Wallace rule melting temperature
- `thermo_melting_temp_gc_adjusted`: GC-adjusted melting temperature
- `thermo_local_min_count`: Number of local minima
- `thermo_local_max_count`: Number of local maxima
- `thermo_low_energy_runs`: Count of low-energy runs
- `thermo_high_energy_runs`: Count of high-energy runs

**5. Entropy Features (62 features)**

Information content and sequence complexity:

- `entropy_shannon_global`: Global Shannon entropy
- `entropy_shannon_w{5,10,15,20}_mean`: Mean Shannon entropy (windows)
- `entropy_shannon_w{5,10,15,20}_max`: Max Shannon entropy (windows)
- `entropy_shannon_w{5,10,15,20}_min`: Min Shannon entropy (windows)
- `entropy_shannon_w{5,10,15,20}_std`: Std Shannon entropy (windows)
- `entropy_kmer_{1,2,3,4,5,6}mer`: K-mer entropy (k=1 to 6)
- `entropy_conditional_order{1,2,3}`: Conditional entropy (orders 1-3)
- `entropy_renyi_entropy_alpha{0.0,0.5,2.0,inf}`: Rényi entropy (α values)
- `entropy_lempel_ziv_complexity`: Lempel-Ziv complexity measure
- `entropy_mutual_info_dist{1,2,3,5,10}`: Mutual information at distances
- `entropy_gc_entropy_w{5,10}_mean`: GC entropy mean (windows 5,10)
- `entropy_gc_entropy_w{5,10}_max`: GC entropy max (windows 5,10)
- `entropy_linguistic_complexity`: Linguistic complexity measure
- `entropy_compression_ratio`: Theoretical compression ratio
- `entropy_run_length_entropy`: Entropy of run lengths
- `entropy_palindrome_density`: Density of palindromic sequences
- `entropy_repeat_density`: Density of repeat sequences
- `entropy_tandem_repeat_fraction`: Fraction of tandem repeats

**6. Advanced Features (55 features)**

Complex structural and biophysical properties:

- `advanced_gquad_score_mean`: Mean G-quadruplex formation score
- `advanced_gquad_score_max`: Maximum G-quadruplex score
- `advanced_gquad_count`: Number of potential G-quadruplexes
- `advanced_melting_mean_melting_temp`: Mean local melting temperature
- `advanced_melting_var_melting_temp`: Variance in melting temperature
- `advanced_melting_min_melting_temp`: Minimum melting temperature
- `advanced_melting_max_melting_temp`: Maximum melting temperature
- `advanced_melting_mean_melting_dH`: Mean melting enthalpy
- `advanced_melting_var_melting_dH`: Variance in melting enthalpy
- `advanced_melting_mean_melting_dS`: Mean melting entropy
- `advanced_melting_var_melting_dS`: Variance in melting entropy
- `advanced_melting_mean_melting_dG`: Mean melting free energy
- `advanced_melting_var_melting_dG`: Variance in melting free energy
- `advanced_melting_min_melting_dG`: Minimum melting free energy
- `advanced_melting_max_melting_dG`: Maximum melting free energy (often constant)
- `advanced_sidd_mean_opening_prob`: Mean stress-induced denaturation probability
- `advanced_sidd_max_opening_prob`: Maximum opening probability
- `advanced_sidd_total_opening_energy`: Total opening energy
- `advanced_sidd_opening_hotspots`: Number of opening hotspots
- `advanced_mgw_mean_width`: Mean minor groove width
- `advanced_mgw_var_width`: Variance in groove width
- `advanced_mgw_min_width`: Minimum groove width
- `advanced_mgw_max_width`: Maximum groove width
- `advanced_mgw_narrow_regions`: Count of narrow regions
- `advanced_mgw_wide_regions`: Count of wide regions
- `advanced_stacking_mean_stacking_energy`: Mean stacking energy
- `advanced_stacking_var_stacking_energy`: Variance in stacking energy
- `advanced_stacking_min_stacking_energy`: Minimum stacking energy (often constant)
- `advanced_stacking_max_stacking_energy`: Maximum stacking energy
- `advanced_stacking_total_stacking_energy`: Total stacking energy
- `advanced_nucleosome_dyad_score`: Nucleosome dyad positioning score
- `advanced_nucleosome_occupancy`: Predicted nucleosome occupancy
- `advanced_nucleosome_positioning_signal`: Positioning signal strength
- `advanced_dnase_sensitivity_score`: Predicted DNase sensitivity
- `advanced_z_dna_formation_potential`: Z-DNA formation potential
- `advanced_a_tract_count`: Number of A-tracts
- `advanced_a_tract_max_length`: Maximum A-tract length
- `advanced_cpg_island_score`: CpG island score
- `advanced_cpg_observed_expected`: CpG observed/expected ratio
- `advanced_methylation_potential`: Predicted methylation potential
- Additional advanced structural features...

#### 5.2.2 Physics Feature Calculation Methods

**Reference Datasets and Sources:**

1. **SantaLucia Nearest-Neighbor Parameters** (`SantaLuciaNN.tsv`)
   - Source: SantaLucia Jr., J. (1998) PNAS 95, 1460-1465
   - Unified thermodynamic parameters for DNA duplex formation
   - Contains ΔH (enthalpy) and ΔS (entropy) for all 10 unique dinucleotide steps
   - Temperature: 37°C (310K) physiological conditions

2. **Olson Structural Parameters** (`OlsonMatrix.tsv`)
   - Source: Olson et al. (1998) J. Mol. Biol. 313, 229-237
   - Mean and standard deviation of helical parameters from crystal structures
   - Parameters: Twist, Tilt, Roll, Shift, Slide, Rise
   - Based on analysis of 86 B-DNA crystal structures

3. **DNA Properties Database** (`DNAProperties.txt`)
   - Comprehensive collection of 79 dinucleotide properties
   - **Specific rows used in PhysiFormer:**
     
     **Bending Features:**
     - Row 4: Bend parameters - Goodsell & Dickerson (1994) NAR 22, 5497-5503
       * Nucleosome positioning preferences
       * Values range: 2.16 (CC/GG) to 6.74 (TA)
     
     **Stacking Energy:**
     - Row 60: Šponer et al. (2004) Chem. Eur. J. 12, 2854-2865
       * MP2/CCSD(T) quantum mechanical calculations
       * Values: -14.7 to -19.5 kcal/mol
     
     **Structural Parameters (Rows 61-66):**
     - Row 61: Twist - Olson et al. (1998) J. Mol. Biol. 313, 229-237
       * Mean: 28° (AG) to 43° (CA/TG/WTC11)
     - Row 62: Tilt - Olson et al. (1998)
       * Mean: -1.4° (TA) to 0.3° (AT)
     - Row 63: Roll - Olson et al. (1998)
       * Mean: -6.8° (GC) to 6.2° (CG)
     - Row 64: Shift - Olson et al. (1998)
       * Mean: -0.3 Å (GC) to 0.12 Å (AT)
     - Row 65: Slide - Olson et al. (1998)
       * Mean: -0.57 Å (AT) to 1.88 Å (CA/TG)
     - Row 66: Rise - Olson et al. (1998)
       * Mean: 3.23 Å (AC) to 3.57 Å (GC)
     
     **Stiffness Coefficients (Rows 67-71):**
     - Row 67: Slide stiffness - Lankas et al. (2003) J. Mol. Biol. 299, 695-709
       * Values: 1.2 (TA) to 3.83 (AT) pN
     - Row 68: Shift stiffness - Lankas et al. (2003)
       * Values: 0.72 (TA) to 1.69 (AA/TT) pN
     - Row 69: Roll stiffness - Lankas et al. (2003)
       * Values: 0.016 (CG/TA) to 0.026 (GC) pN·nm²
     - Row 70: Tilt stiffness - Lankas et al. (2003)
       * Values: 0.018 (TA) to 0.042 (CC/GG) pN·nm²
     - Row 71: Twist stiffness - Lankas et al. (2003)
       * Values: 0.014 (CG) to 0.036 (AC/GT) pN·nm²
     
     **Additional Parameters (not currently used):**
     - Rows 39-59: Full stiffness matrix elements - Gorin et al. (1995)
     - Rows 72-75: Free energy profiles - Sivolob & Khrapunov (1995)
     - Row 22-23: Alternative enthalpy/entropy - Sugimoto et al. (1996)

**Calculation Algorithms:**

**1. Thermodynamic Features (ThermodynamicProcessor)**

Gibbs free energy calculation at physiological temperature (310K):
```python
ΔG = ΔH - T·ΔS
where:
  ΔH: enthalpy change (kcal/mol) from SantaLucia parameters
  ΔS: entropy change (cal/mol·K) from SantaLucia parameters
  T: temperature = 310K (37°C)
  
For each dinucleotide step:
  dG[i] = nn_params[dinuc]['dH'] - 310 * (nn_params[dinuc]['dS'] / 1000)
```

Melting temperature calculations:
- Wallace rule: Tm = 2*(A+T) + 4*(G+C)
- GC-adjusted: Tm = 81.5 + 16.6*log10([Na+]) + 0.41*(%GC) - 675/length

**2. Stiffness Features (StiffnessProcessor)**

Deformation energy using harmonic approximation:
```python
E_deformation = 0.5 * Σ k_i * (θ_i - θ_0i)²
where:
  k_i: stiffness constant for parameter i (from rows 67-71)
  θ_i: observed parameter value
  θ_0i: ideal B-DNA value (canonical)
  
Ideal B-DNA values:
  Twist: 36.0°, Tilt: 0.0°, Roll: 0.0°
  Shift: 0.0 Å, Slide: 0.0 Å, Rise: 3.4 Å
```

Principal Component Analysis:
- Applied to 6D parameter space (twist, tilt, roll, shift, slide, rise)
- Captures major modes of deformation
- PC1 typically represents twist-slide coupling
- PC2 often captures roll-slide anticorrelation

**3. Bending Features (BendingEnergyProcessor)**

Bending energy from nucleosome positioning data:
```python
E_bend = Σ bend_params[dinuc] * kappa0
where:
  bend_params: from DNAProperties.txt row 4
  kappa0: global stiffness constant (default 1.0)
  
Curvature calculation:
  curvature[i] = bend_cost[i] / kappa0
  
RMS curvature over window w:
  RMS_w = sqrt(mean(curvature[i:i+w]²))
```

Spectral analysis:
- FFT of curvature profile
- Power at helical repeat frequencies (0.1, 0.143, 0.2 cycles/bp)
- Identifies periodic bending patterns

**4. Entropy Features (EntropyProcessor)**

Shannon entropy:
```python
H = -Σ p_i * log2(p_i)
where p_i is frequency of nucleotide/k-mer i
```

K-mer entropy (k=1 to 6):
- Counts k-mer frequencies in sequence
- Applies Shannon formula to k-mer distribution
- Higher k captures longer-range patterns

Conditional entropy of order n:
```python
H(X|X_{n-1}...X_1) = H(X_n, X_{n-1}...X_1) - H(X_{n-1}...X_1)
```

Rényi entropy with parameter α:
```python
H_α = (1/(1-α)) * log2(Σ p_i^α)
Special cases:
  α=0: Max entropy (log2 of alphabet size)
  α=1: Shannon entropy (limit)
  α=2: Collision entropy
  α=∞: Min entropy
```

Lempel-Ziv complexity:
- Measures sequence compressibility
- Counts number of unique substrings in left-to-right parse
- Normalized by sequence length

**5. Advanced Features (AdvancedBiophysicsProcessor)**

G-quadruplex scoring:
```python
Pattern: G{3,}[ATGC]{1,7}G{3,}[ATGC]{1,7}G{3,}[ATGC]{1,7}G{3,}
Score = Σ (n_G_runs * loop_stability * cation_factor)
```

Minor groove width:
- From DNAProperties.txt row 11
- Identifies A-tracts (narrow minor groove)
- Important for protein-DNA recognition

Stress-induced DNA denaturation (SIDD):
```python
Opening probability = exp(-ΔG_opening / RT)
ΔG_opening = ΔH_opening - T*ΔS_opening + σ*ΔL
where σ is superhelical stress
```

Z-DNA formation potential:
- Identifies alternating purine-pyrimidine stretches
- Scores based on dinucleotide Z-DNA propensity
- Enhanced by negative supercoiling

**6. PWM Features (PWMProcessor)**

Position Weight Matrix scoring for transcription factor binding:
```python
# PWM score calculation
score(seq, pos) = Σ PWM[base_i][i] for i in motif_length
where:
  PWM[base][pos]: log-odds score for base at position
  base_i: nucleotide at position i in sequence

# Binding affinity to Gibbs free energy
ΔG = -RT * ln(K_d)
where:
  R: gas constant
  T: temperature (310K)
  K_d: dissociation constant from PWM score

# Feature calculations per TF:
max_score: max(scores) across all positions
mean_score: mean(scores) 
var_score: variance(scores)
num_high_affinity: count(scores > threshold)
total_weight: sum(exp(scores/RT))
entropy: -Σ p_i * log(p_i) where p_i = exp(score_i)/Z
top_k_mean: mean(top k scores)
```

PWM sources and quality control:
- JASPAR 2024 CORE non-redundant collection
- HOCOMOCO v11 human motifs (quality grades A, B)
- Minimum information content: 1.0 bits
- Minimum sites for PWM derivation: 500

#### 5.2.3 Complete Motif Database

The system uses a curated database of 216 transcription factor binding motifs from HOCOMOCO and JASPAR databases, representing 195 unique transcription factors. The full motif database is stored in:
`/home/bcheng/sequence_optimization/mainproject/PhysiFormer/optimization_methods/motif_processing/filtered_motifs/filtered_motifs.json`

**Key TF Families Represented:**
- **Zinc Finger Proteins**: ZNF143, ZNF24, ZNF281, ZNF341, ZNF384, ZNF410, etc. (50+ members)
- **Forkhead Box**: FOXA2, FOXE1, FOXI1, FOXJ3 
- **Homeobox**: HOXC9, HOXC11, HOXD12
- **STAT Family**: STAT1, STAT2, STAT3, STAT5A, STAT5B
- **E2F Family**: E2F3, E2F6
- **SOX Family**: SOX3, SOX4, SOX8, SOX11, SOX12, SOX14, SOX21
- **Nuclear Receptors**: ESR1, NR1D1, NR1D2, NR1I3, NR2F6, PPARD, RARA, RARG
- **Basic Leucine Zipper**: ATF3, ATF4, ATF6, CEBPB, CEBPG, FOS, JUN, JUND
- **GATA Family**: GATA4, GATA5, GATA6
- **KLF Family**: KLF1, KLF4, KLF9
- **POU Domain**: POU1F1, POU2F3, POU4F2, POU6F2

**Motif Selection Criteria:**
- Quality score threshold: > 4.0
- Information content: > 1.0 bits
- Minimum sites: > 500 sequences
- Cell-type specificity based on expression data

#### 5.2.3 Zero-Variance Feature Removal

During preprocessing, features with zero variance across training samples are automatically removed to prevent numerical instability. These constant features provide no discriminative value and can cause NaN values in correlation calculations.

**Features Commonly Removed (HepG2 example):**
1. `bend_attention_bias_min`: Always ~0 (1.1095e-304)
2. `thermo_min_dS`: Constant at -0.0244
3. `thermo_max_dS`: Constant at -0.0199  
4. `thermo_max_dG`: Constant at -0.597
5. `entropy_renyi_entropy_alpha0.0`: Always 2.0
6. `entropy_gc_entropy_w10_max`: Always 1.0
7. `advanced_melting_max_melting_dG`: Constant at -0.597
8. `advanced_stacking_min_stacking_energy`: Constant minimum

**Impact on Feature Counts:**
- HepG2: 539 → 529 features (10 removed)
- K562: Similar removal pattern
- WTC11: Similar removal pattern
- S2: Different pattern due to distinct sequences

### 5.3 Activity Score Integration

#### 5.3.1 Cell-Type Specific Activities

Different cell types have different activity measurements:

```python
ACTIVITY_MAPPINGS = {
    'HepG2': ['normalized_log2'],           # Single activity
    'K562': ['normalized_log2'],            # Single activity
    'WTC11': ['normalized_log2'],           # Single activity
    'S2': ['Dev_log2_enrichment',           # Dual activities
           'Hk_log2_enrichment']
}
```

#### 5.3.2 Activity Loading Strategy

```python
def _load_activity_scores(self):
    """
    Load activity scores from various sources
    """
    if self.cell_type == 'S2':
        # S2 has activities in main file
        activity_cols = ['Dev_log2_enrichment', 'Hk_log2_enrichment']
        if all(col in self.df.columns for col in activity_cols):
            self.activity_cols = activity_cols
            self.has_activities = True
            
    elif self.cell_type in ['HepG2', 'K562', 'WTC11']:
        # Check tileformer file
        tileformer_file = self.descriptors_file.replace(
            '_descriptors', '_tileformer'
        )
        if os.path.exists(tileformer_file):
            df_tile = pd.read_csv(tileformer_file, sep='\t')
            if 'normalized_log2' in df_tile.columns:
                # Map activities by sequence name
                activity_map = dict(
                    zip(df_tile['name'], df_tile['normalized_log2'])
                )
                self.df['normalized_log2'] = self.df['name'].map(activity_map)
                self.activity_cols = ['normalized_log2']
                self.has_activities = True
```

---

## 6. Training Infrastructure

### 6.1 Training Pipeline

#### 6.1.1 Main Training Loop

```python
def train_epoch(model, dataloader, optimizer, scheduler, device, 
                epoch, metrics_calculator, feature_stats, 
                batch_log_file, aux_optimizer=None, aux_loss_fn=None):
    """
    Train one epoch with comprehensive logging
    """
    model.train()
    epoch_loss = 0.0
    all_predictions = []
    all_targets = []
    all_seq_ids = []
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} Training')
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        sequences = batch['sequence'].to(device)
        descriptors = batch['descriptors'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences, descriptors)
        
        # Compute main loss
        predictions = reconstruct_predictions(outputs, feature_indices)
        main_loss = compute_adaptive_loss(
            predictions, descriptors, feature_stats
        )
        
        # Auxiliary task loss (if enabled)
        aux_loss = 0.0
        if aux_optimizer and 'activities' in batch:
            activities = batch['activities'].to(device)
            
            # Head A: Sequence + Features
            if 'aux_seq_feat_predictions' in outputs:
                loss_a = aux_loss_fn(
                    outputs['aux_seq_feat_predictions'], 
                    activities
                )
                
            # Head B: Features only
            if 'aux_feat_only_predictions' in outputs:
                loss_b = aux_loss_fn(
                    outputs['aux_feat_only_predictions'], 
                    activities
                )
                
            aux_loss = loss_a + loss_b
        
        # Backward pass
        main_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer steps
        optimizer.step()
        scheduler.step()
        
        # Auxiliary optimizer step
        if aux_optimizer:
            aux_optimizer.zero_grad()
            aux_loss.backward()
            aux_optimizer.step()
        
        # Logging
        if batch_idx % 10 == 0:
            log_batch_metrics(
                batch_log_file, epoch, batch_idx, 
                main_loss, aux_loss, optimizer
            )
```

#### 6.1.2 Validation Loop

```python
def validate_epoch(model, dataloader, device, epoch, 
                   metrics_calculator, feature_stats):
    """
    Validate model performance
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Epoch {epoch} Validation'):
            sequences = batch['sequence'].to(device)
            descriptors = batch['descriptors'].to(device)
            
            # Forward pass
            outputs = model(sequences, descriptors)
            predictions = reconstruct_predictions(outputs, feature_indices)
            
            # Compute loss
            loss = compute_adaptive_loss(
                predictions, descriptors, feature_stats
            )
            
            total_loss += loss.item() * sequences.size(0)
            all_predictions.append(predictions.cpu())
            all_targets.append(descriptors.cpu())
    
    # Compute metrics
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    
    metrics = metrics_calculator.compute_all_metrics(
        predictions, targets, prefix='val'
    )
    
    return total_loss / len(dataloader.dataset), metrics
```

### 6.2 OneCycleLR Scheduling

#### 6.2.1 Configuration

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.learning_rate * 10,     # Peak LR = 10x base
    total_steps=total_steps,
    pct_start=0.1,                      # 10% warmup
    anneal_strategy='cos',              # Cosine annealing
    cycle_momentum=False,               # AdamW doesn't use momentum
    div_factor=25,                      # Initial LR = max_lr/25
    final_div_factor=10000              # Final LR = max_lr/10000
)
```

#### 6.2.2 Learning Rate Evolution

```
Phase 1 (0-10%): Linear warmup from lr/25 to 10*lr
Phase 2 (10-90%): Cosine annealing from 10*lr to lr/400
Phase 3 (90-100%): Continued annealing to lr/10000
```

### 6.3 Gradient Management

#### 6.3.1 Gradient Clipping

```python
# Clip gradients to prevent explosion
torch.nn.utils.clip_grad_norm_(
    model.parameters(), 
    max_norm=1.0
)
```

#### 6.3.2 Gradient Isolation for Auxiliary Tasks

```python
# Main model gradient computation
main_loss.backward(retain_graph=True)

# Detach features for auxiliary heads
detached_features = features.detach()

# Auxiliary gradient computation (isolated)
aux_predictions = aux_head(detached_features)
aux_loss = criterion(aux_predictions, targets)
aux_loss.backward()
```

### 6.4 Checkpoint Management

#### 6.4.1 Checkpoint Structure

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'metrics_history': metrics_history,
    'feature_weights': model.get_feature_weights(),
    'training_args': args,
    'random_state': {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'random': random.getstate()
    }
}
```

#### 6.4.2 Resume Functionality

```python
def resume_from_checkpoint(checkpoint_path):
    """
    Resume training from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore scheduler state
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Restore training state
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    
    # Restore random states
    torch.set_rng_state(checkpoint['random_state']['torch'])
    np.random.set_state(checkpoint['random_state']['numpy'])
    random.setstate(checkpoint['random_state']['random'])
    
    return start_epoch, best_val_loss
```

---

## 7. Auxiliary Head Architecture

### 7.1 Design Philosophy

The auxiliary heads serve multiple purposes:
1. **Multi-task Learning**: Improve feature representations
2. **Diagnostic Tool**: Assess information content
3. **Baseline Comparison**: Features-only vs. sequence+features

### 7.2 Head A: Sequence + Features

#### 7.2.1 Architecture

```python
class AuxiliaryHeadA(nn.Module):
    """
    Auxiliary head combining sequence and real features
    
    Tests whether sequence adds value beyond computed features
    """
    
    def __init__(self, vocab_size=5, seq_len=230, feature_dim=536,
                 hidden_dim=256, n_activities=1, dropout=0.1):
        super().__init__()
        
        # Independent sequence encoder
        self.sequence_encoder = nn.Sequential(
            nn.Embedding(vocab_size, 64),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Feature processor
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined predictor
        self.predictor = nn.Sequential(
            nn.Linear(256 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_activities)
        )
```

#### 7.2.2 Information Flow

```
Raw Sequence → Embedding → CNN Encoder ─┐
                                         ├→ Concatenate → Predictor → Activity
Real Features → Feature Processor ──────┘
```

### 7.3 Head B: Features Only

#### 7.3.1 Architecture

```python
class AuxiliaryHeadB(nn.Module):
    """
    Baseline auxiliary head using only real features
    
    Establishes baseline performance without sequence information
    """
    
    def __init__(self, feature_dim=536, hidden_dim=256,
                 n_activities=1, dropout=0.1):
        super().__init__()
        
        self.feature_stack = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, n_activities)
        )
```

### 7.4 Gradient Isolation

#### 7.4.1 Implementation

```python
def forward(self, sequences, real_features):
    # Main model forward pass
    main_output = self.main_model(sequences)
    
    # Detach features for auxiliary heads
    detached_features = real_features.detach()
    detached_sequences = sequences.detach()
    
    # Auxiliary predictions (no gradient to main model)
    aux_a = self.aux_head_a(detached_sequences, detached_features)
    aux_b = self.aux_head_b(detached_features)
    
    return {
        'main': main_output,
        'aux_a': aux_a,
        'aux_b': aux_b
    }
```

#### 7.4.2 Separate Optimizers

```python
# Main model optimizer
main_optimizer = optim.AdamW(
    model.get_main_parameters(),
    lr=learning_rate,
    weight_decay=0.01
)

# Auxiliary heads optimizer
aux_optimizer = optim.AdamW(
    model.get_auxiliary_parameters(),
    lr=learning_rate * 0.1,  # Lower LR for auxiliary
    weight_decay=0.01
)
```

---

## 8. Feature Engineering Pipeline

### 8.1 PWM Feature Extraction

#### 8.1.1 Transcription Factor Binding

```python
class PWMFeatureExtractor:
    """
    Extract PWM-based features for TF binding prediction
    """
    
    def __init__(self, pwm_database):
        self.pwms = self.load_pwm_database(pwm_database)
        
    def extract_features(self, sequence):
        features = {}
        
        for tf_name, pwm in self.pwms.items():
            # Scan sequence with PWM
            scores = self.scan_sequence(sequence, pwm)
            
            # Compute statistics
            features[f'{tf_name}_max_score'] = np.max(scores)
            features[f'{tf_name}_mean_score'] = np.mean(scores)
            features[f'{tf_name}_var_score'] = np.var(scores)
            
            # Binding affinity
            features[f'{tf_name}_delta_g'] = self.score_to_delta_g(
                scores, temperature=310
            )
            
            # High-affinity sites
            threshold = np.percentile(scores, 90)
            features[f'{tf_name}_num_high_affinity'] = np.sum(
                scores > threshold
            )
            
            # Information content
            features[f'{tf_name}_entropy'] = self.compute_entropy(scores)
            
        return features
```

### 8.2 DNA Shape Features

#### 8.2.1 Minor Groove Width

```python
def compute_mgw(sequence):
    """
    Compute minor groove width profile
    """
    mgw_params = {
        'AA': 5.0, 'AT': 5.5, 'AG': 5.3, 'AC': 5.2,
        'TA': 6.0, 'TT': 5.0, 'TG': 5.4, 'TC': 5.3,
        'GA': 5.3, 'GT': 5.4, 'GG': 5.0, 'GC': 4.5,
        'CA': 5.2, 'CT': 5.3, 'CG': 4.5, 'CC': 5.0
    }
    
    mgw_profile = []
    for i in range(len(sequence) - 1):
        dinuc = sequence[i:i+2]
        mgw_profile.append(mgw_params.get(dinuc, 5.0))
    
    return np.array(mgw_profile)
```

### 8.3 G-Quadruplex Features

#### 8.3.1 G4 Score Computation

```python
def compute_g4_scores(sequence):
    """
    Compute G-quadruplex formation potential
    """
    # G4 pattern: G{3,}[ATCG]{1,7}G{3,}[ATCG]{1,7}G{3,}[ATCG]{1,7}G{3,}
    g4_pattern = r'G{3,}\w{1,7}G{3,}\w{1,7}G{3,}\w{1,7}G{3,}'
    
    features = {}
    
    # Find all G4 motifs
    g4_matches = re.finditer(g4_pattern, sequence)
    g4_positions = [(m.start(), m.end()) for m in g4_matches]
    
    # G4 count
    features['g4_count'] = len(g4_positions)
    
    # G4 density
    features['g4_density'] = len(g4_positions) / len(sequence)
    
    # G4 stability scores
    if g4_positions:
        stabilities = []
        for start, end in g4_positions:
            motif = sequence[start:end]
            stability = compute_g4_stability(motif)
            stabilities.append(stability)
        
        features['g4_max_stability'] = np.max(stabilities)
        features['g4_mean_stability'] = np.mean(stabilities)
    
    return features
```

---

## 9. Evaluation Metrics System

### 9.1 Comprehensive Metrics

#### 9.1.1 MetricsCalculator Implementation

```python
class MetricsCalculator:
    """
    Comprehensive metrics computation for multi-output regression
    """
    
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def compute_all_metrics(self, predictions, targets, prefix=''):
        """
        Compute all evaluation metrics
        """
        metrics = {}
        
        # Overall metrics (across all features)
        metrics[f'{prefix}_overall_pearson'] = self.pearson_correlation(
            predictions.flatten(), targets.flatten()
        )
        metrics[f'{prefix}_overall_spearman'] = self.spearman_correlation(
            predictions.flatten(), targets.flatten()
        )
        metrics[f'{prefix}_overall_mse'] = F.mse_loss(
            predictions, targets
        ).item()
        metrics[f'{prefix}_overall_mae'] = F.l1_loss(
            predictions, targets
        ).item()
        metrics[f'{prefix}_overall_r2'] = self.r2_score(
            predictions.flatten(), targets.flatten()
        )
        
        # Per-feature metrics
        feature_metrics = self.compute_per_feature_metrics(
            predictions, targets
        )
        
        # Aggregate statistics
        for metric_type in ['pearson', 'spearman', 'mse', 'mae', 'r2']:
            values = feature_metrics[metric_type]
            metrics[f'{prefix}_{metric_type}_mean'] = np.mean(values)
            metrics[f'{prefix}_{metric_type}_std'] = np.std(values)
            metrics[f'{prefix}_{metric_type}_min'] = np.min(values)
            metrics[f'{prefix}_{metric_type}_max'] = np.max(values)
            metrics[f'{prefix}_{metric_type}_median'] = np.median(values)
        
        # Top/Bottom performing features
        pearson_scores = feature_metrics['pearson']
        sorted_indices = np.argsort(pearson_scores)
        
        metrics[f'{prefix}_worst_features'] = [
            (self.feature_names[i], pearson_scores[i]) 
            for i in sorted_indices[:5]
        ]
        metrics[f'{prefix}_best_features'] = [
            (self.feature_names[i], pearson_scores[i]) 
            for i in sorted_indices[-5:]
        ]
        
        return metrics
```

### 9.2 Statistical Measures

#### 9.2.1 Correlation Metrics

```python
def pearson_correlation(pred, target):
    """
    Compute Pearson correlation coefficient
    """
    pred_mean = pred.mean()
    target_mean = target.mean()
    
    numerator = ((pred - pred_mean) * (target - target_mean)).sum()
    denominator = torch.sqrt(
        ((pred - pred_mean) ** 2).sum() * 
        ((target - target_mean) ** 2).sum()
    )
    
    return (numerator / (denominator + 1e-8)).item()

def spearman_correlation(pred, target):
    """
    Compute Spearman rank correlation
    """
    pred_ranks = rankdata(pred.cpu().numpy())
    target_ranks = rankdata(target.cpu().numpy())
    
    return pearsonr(pred_ranks, target_ranks)[0]
```

#### 9.2.2 R² Score

```python
def r2_score(pred, target):
    """
    Compute coefficient of determination
    """
    target_mean = target.mean()
    ss_tot = ((target - target_mean) ** 2).sum()
    ss_res = ((target - pred) ** 2).sum()
    
    return 1 - (ss_res / ss_tot).item()
```

### 9.3 Distribution Analysis

#### 9.3.1 Pearson Distribution Tracking

```python
def analyze_pearson_distribution(feature_scores):
    """
    Analyze distribution of Pearson correlations
    """
    scores = np.array([score for _, score in feature_scores])
    
    analysis = {
        'mean': np.mean(scores),
        'median': np.median(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'q1': np.percentile(scores, 25),
        'q2': np.percentile(scores, 50),
        'q3': np.percentile(scores, 75),
        'iqr': np.percentile(scores, 75) - np.percentile(scores, 25),
        'range': np.max(scores) - np.min(scores)
    }
    
    return analysis
```

---

## 10. Production Deployment

### 10.1 Inference Pipeline

#### 10.1.1 InferenceEngine Class

```python
class InferenceEngine:
    """
    Production inference system for PhysInformer
    """
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = self._build_model(checkpoint['config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load normalization stats
        self.norm_stats = checkpoint['normalization_stats']
        
        # Feature metadata
        self.feature_names = checkpoint['feature_names']
        self.feature_indices = checkpoint['feature_indices']
        
    @torch.no_grad()
    def predict(self, sequences):
        """
        Predict biophysical descriptors for sequences
        """
        # Preprocess sequences
        encoded = self.encode_sequences(sequences)
        
        # Model inference
        outputs = self.model(encoded.to(self.device))
        
        # Reconstruct predictions
        predictions = self.reconstruct_predictions(outputs)
        
        # Denormalize
        predictions = self.denormalize(predictions)
        
        return predictions
```

### 10.2 Batch Processing

#### 10.2.1 Efficient Batching

```python
class BatchProcessor:
    """
    Efficient batch processing for large-scale inference
    """
    
    def __init__(self, model, batch_size=128):
        self.model = model
        self.batch_size = batch_size
        
    def process_file(self, input_file, output_file):
        """
        Process sequences from file
        """
        # Load sequences
        sequences = self.load_sequences(input_file)
        
        # Process in batches
        all_predictions = []
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i:i + self.batch_size]
            predictions = self.model.predict(batch)
            all_predictions.append(predictions)
        
        # Combine results
        results = np.concatenate(all_predictions)
        
        # Save predictions
        self.save_predictions(results, output_file)
```

### 10.3 API Integration

#### 10.3.1 REST API Endpoint

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
model = InferenceEngine('model.pt')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for sequence prediction
    """
    try:
        # Parse request
        data = request.json
        sequences = data['sequences']
        
        # Validate input
        if not validate_sequences(sequences):
            return jsonify({'error': 'Invalid sequences'}), 400
        
        # Model prediction
        predictions = model.predict(sequences)
        
        # Format response
        response = {
            'predictions': predictions.tolist(),
            'feature_names': model.feature_names
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

---

## 11. Mathematical Formulations

### 11.1 Loss Functions

#### 11.1.1 Adaptive Feature Loss

```
L_adaptive = Σᵢ wᵢ · (yᵢ - ŷᵢ)² / σᵢ²

where:
wᵢ = EMA(|∇L/∇ŷᵢ|) = α·wᵢ₋₁ + (1-α)·|∇L/∇ŷᵢ|
σᵢ² = learned variance for feature i
```

#### 11.1.2 Heteroscedastic Loss

```
L_hetero = 0.5 · Σᵢ [log(2πσᵢ²) + (yᵢ - μᵢ)²/σᵢ²]

where:
μᵢ = E[yᵢ|x] (predicted mean)
σᵢ² = Var[yᵢ|x] (predicted variance)
```

### 11.2 State Space Formulation

#### 11.2.1 Continuous System

```
dx/dt = Ax + Bu
y = Cx + Du

where:
x ∈ ℝⁿ: hidden state
u ∈ ℝᵐ: input
y ∈ ℝᵖ: output
A, B, C, D: learnable matrices
```

#### 11.2.2 Discretization

```
x[k+1] = exp(ΔA)·x[k] + (exp(ΔA) - I)·A⁻¹·B·u[k]
y[k] = C·x[k] + D·u[k]

Δ: discretization step size
```

### 11.3 Attention Mechanisms

#### 11.3.1 Multi-Head Self-Attention

```
Attention(Q, K, V) = softmax(QK^T/√d_k)V

MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O
where headᵢ = Attention(QWᵢᵍ, KWᵢᴷ, VWᵢⱽ)
```

---

## 12. Implementation Details

### 12.1 Memory Optimization

#### 12.1.1 Gradient Checkpointing

```python
class CheckpointedLayer(nn.Module):
    """
    Memory-efficient layer with gradient checkpointing
    """
    
    def forward(self, x):
        if self.training:
            # Use checkpointing during training
            return checkpoint(self._forward_impl, x)
        else:
            # Normal forward during inference
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        # Actual forward computation
        x = self.layer1(x)
        x = self.layer2(x)
        return x
```

#### 12.1.2 Mixed Precision Training

```python
# Automatic mixed precision
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 12.2 Numerical Stability

#### 12.2.1 Log-Sum-Exp Trick

```python
def stable_log_sum_exp(x):
    """
    Numerically stable log-sum-exp
    """
    max_x = x.max(dim=-1, keepdim=True)[0]
    return max_x + torch.log(torch.exp(x - max_x).sum(dim=-1, keepdim=True))
```

#### 12.2.2 Variance Regularization

```python
def compute_stable_variance(log_var):
    """
    Compute variance with bounds for stability
    """
    # Clamp log variance to reasonable range
    log_var = torch.clamp(log_var, min=-10, max=10)
    
    # Convert to variance
    variance = torch.exp(log_var)
    
    # Additional safety bounds
    variance = torch.clamp(variance, min=1e-6, max=1e6)
    
    return variance
```

### 12.3 Debugging Utilities

#### 12.3.1 Gradient Flow Analysis

```python
def analyze_gradients(model):
    """
    Analyze gradient flow through model
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm={grad_norm:.6f}")
            
            if grad_norm < 1e-7:
                print(f"  WARNING: Vanishing gradient")
            elif grad_norm > 100:
                print(f"  WARNING: Exploding gradient")
```

---

## 13. Configuration Management

### 13.1 Training Configuration

#### 13.1.1 Default Parameters

```yaml
# config/default.yaml
model:
  architecture: physics_aware
  vocab_size: 5
  d_model: 256
  d_expanded: 384
  dropout: 0.1
  temperature: 310.0

training:
  batch_size: 32
  learning_rate: 0.0001
  epochs: 50
  gradient_clip: 1.0
  weight_decay: 0.01
  
scheduler:
  type: OneCycleLR
  max_lr_multiplier: 10
  pct_start: 0.1
  anneal_strategy: cos
  
data:
  max_seq_length: 249
  num_workers: 4
  normalize: true
  filter_zero_variance: true
```

### 13.2 Cell-Type Specific Configurations

#### 13.2.1 HepG2 Configuration

```bash
#!/bin/bash
# run_hepg2.sh

python train.py \
    --cell_type HepG2 \
    --data_dir ../output \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --epochs 50 \
    --device cuda \
    --num_workers 4 \
    --seed 42
```

#### 13.2.2 S2 Configuration

```bash
#!/bin/bash
# run_s2.sh

python train.py \
    --cell_type S2 \
    --data_dir ../output \
    --batch_size 128 \  # Larger batch for more data
    --learning_rate 0.0005 \  # Higher LR for S2
    --epochs 50 \
    --device cuda \
    --num_workers 8
```

---

## 14. Troubleshooting Guide

### 14.1 Common Issues

#### 14.1.1 Out of Memory Errors

**Problem**: CUDA out of memory during training

**Solutions**:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision training
4. Reduce model dimensions

```python
# Memory-efficient configuration
config = {
    'batch_size': 16,  # Reduced from 32
    'gradient_checkpointing': True,
    'mixed_precision': True,
    'd_model': 192  # Reduced from 256
}
```

#### 14.1.2 Vanishing/Exploding Gradients

**Problem**: Gradients become too small or too large

**Solutions**:
1. Adjust learning rate
2. Use gradient clipping
3. Check initialization
4. Add batch normalization

```python
# Gradient management
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 14.1.3 Poor Convergence

**Problem**: Model doesn't improve during training

**Solutions**:
1. Check data normalization
2. Verify loss function
3. Adjust learning rate schedule
4. Increase model capacity

### 14.2 Performance Optimization

#### 14.2.1 Training Speed

```python
# Optimizations for faster training
optimizations = {
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 2,
    'torch.backends.cudnn.benchmark': True
}
```

#### 14.2.2 Inference Speed

```python
# Optimizations for faster inference
@torch.jit.script
def optimized_forward(x):
    # JIT-compiled forward pass
    return model(x)

# ONNX export for deployment
torch.onnx.export(model, dummy_input, "model.onnx")
```

---

## 15. API Reference

### 15.1 Model Classes

#### 15.1.1 PhysicsAwareModel

```python
class PhysicsAwareModel(nn.Module):
    """
    Main physics-aware model
    
    Args:
        vocab_size (int): Vocabulary size (default: 5)
        d_model (int): Model dimension (default: 256)
        d_expanded (int): Expanded dimension (default: 384)
        seq_len (int): Maximum sequence length (default: 230)
        dropout (float): Dropout probability (default: 0.1)
        temperature (float): Temperature for thermodynamic calculations (default: 310.0)
        n_descriptor_features (int): Number of descriptor features
        descriptor_names (List[str]): Names of descriptor features
        property_groups (Dict): Property groupings for routing
    
    Methods:
        forward(sequences, real_features=None): Forward pass
        enable_auxiliary_heads(n_real_features, n_activities): Enable auxiliary tasks
        get_main_parameters(): Get main model parameters
        get_auxiliary_parameters(): Get auxiliary head parameters
    """
```

#### 15.1.2 MetricsCalculator

```python
class MetricsCalculator:
    """
    Metrics computation utility
    
    Args:
        feature_names (List[str]): Names of features
    
    Methods:
        compute_all_metrics(predictions, targets, prefix=''): Compute all metrics
        compute_per_feature_metrics(predictions, targets): Per-feature metrics
        compute_pearson_distribution(predictions, targets): Pearson distribution analysis
    """
```

### 15.2 Training Functions

#### 15.2.1 train_epoch

```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    epoch: int,
    metrics_calculator: MetricsCalculator,
    feature_stats: Dict,
    batch_log_file: Path,
    aux_optimizer: Optional[Optimizer] = None,
    aux_loss_fn: Optional[nn.Module] = None
) -> Dict[str, float]:
    """
    Train one epoch
    
    Returns:
        Dictionary of training metrics
    """
```

### 15.3 Data Pipeline

#### 15.3.1 PhysInformerDataset

```python
class PhysInformerDataset(Dataset):
    """
    Dataset for PhysInformer
    
    Args:
        descriptors_file (str): Path to descriptors TSV
        cell_type (str): Cell type identifier
        normalize (bool): Whether to normalize features
        load_activities (bool): Whether to load activity scores
    
    Methods:
        __len__(): Dataset size
        __getitem__(idx): Get item by index
        get_feature_names(): Get feature names
        get_normalization_stats(): Get normalization statistics
    """
```

---

## Conclusion

PhysInformer represents a comprehensive physics-aware approach to DNA sequence analysis, combining deep learning with biophysical principles. The system's modular architecture, extensive feature engineering, and sophisticated training infrastructure enable accurate prediction of hundreds of biophysical descriptors from raw sequences.

Key strengths include:
- Physics-guided neural architecture design
- Comprehensive biophysical feature coverage
- Robust training and evaluation infrastructure
- Production-ready deployment capabilities
- Extensive diagnostic and debugging tools

The system continues to evolve with ongoing research into improved architectures, additional biophysical constraints, and expanded feature sets.

---

## Appendices

### A. Feature Name Mappings

Complete listing of all 536+ biophysical descriptor names and their categories...

### B. Hyperparameter Tuning Guide

Recommended hyperparameter ranges for different scenarios...

### C. Performance Benchmarks

Detailed performance metrics across cell types and feature categories...

### D. Code Examples

Complete working examples for common use cases...

---

*End of Technical Documentation*

Total Lines: 1,876
Last Updated: September 2025
Version: 1.0.0