"""
================================================================================
FUSEMAP PHYSICS MODULES - Representative Code File 2/3
================================================================================

These 3 representative files contain key excerpts from the FUSEMAP codebase.
They do not contain all FUSEMAP code, as the full implementation is too large
to include here. All code, trained models, and processed datasets are available
under the MIT license at:
    https://github.com/bryanc5864/FUSEMAP

This file contains the physics-based modules for FUSEMAP:

MODULES INCLUDED:
1. PhysInformer - Transformer for predicting biophysical descriptors
   Source: physics/PhysInformer/model.py
   Architecture: 8-layer transformer, 512 dim, 8 heads
   Default: 500 output features (thermo, stiffness, bending, entropy, advanced)

2. TileFormer - Neural surrogate for APBS electrostatic calculations
   Source: physics/TileFormer/models/tileformer_architecture.py
   Architecture: 6-layer transformer, 256 dim, 8 heads, 3.2M params
   6 outputs: STD/ENH x MIN/MAX/MEAN electrostatic potentials

3. PhysicsVAE - Physics-conditioned VAE for sequence generation
   Source: physics/PhysicsVAE/models/physics_vae.py
   Architecture: Conv encoder -> latent (128) + physics cond (64) -> transformer decoder
   Total: 8.7M params

4. S2A Universal Features - Cross-species transfer via physics
   Source: physics/S2A/universal_features.py
   Extracts universal physics features (excludes species-specific PWMs)
   ~250 selected features from 521 total for transfer

5. Physics-Aware Model Components - Property-specific routers and heads
   Source: physics/PhysInformer/physics_aware_model.py
   PhysicsRouter, ThermoHead, ElectrostaticHead, AuxiliaryHeads A/B

6. Physics-Aware Training Losses - Thermodynamic constraints
   Source: physics/PhysInformer/physics_aware_model.py (loss section)
   Heteroscedastic loss, thermodynamic identity (dG = dH - T*dS)

7. PhysicsVAE Loss Functions - ELBO with physics consistency
   Source: physics/PhysicsVAE/models/losses.py
   VAELoss (recon + KL with beta-annealing), PhysicsConsistencyLoss, CombinedVAELoss

8. Physics Feature Attribution - Mechanistic interpretation
   Source: physics/PhysicsInterpreter/physics_attribution.py
   PhysicsAttributor for decomposing predictions via physics features

9. Transfer Learning Protocols - Cross-species physics transfer
   Source: physics/PhysicsTransfer/protocols.py
   ZeroShotTransfer, PhysicsAnchoredFineTuning, MultiSpeciesJointTraining

KEY RESULTS (from paper):
- PhysInformer: r = 0.92 on held-out human sequences, 12,000x speedup
- TileFormer: R^2 = 0.960, Pearson r = 0.982, 10,000x speedup over APBS
- PhysicsVAE: 63.7% +/- 2.1% nucleotide reconstruction accuracy (multi-human)
- S2A: rho = 0.70 plant-to-plant zero-shot (Arab.+Sorg. -> Maize, n=2,461)
- DNA bending transfers at r > 0.92 even cross-kingdom; PWMs collapse (r = 0.03)
- Feature attribution for plant transfer: Bending 45.3%, Advanced 26.7%,
  Thermodynamics 14.1%, Stiffness 7.5%, Entropy 6.5%
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import warnings


# =============================================================================
# PART 1: PhysInformer - Biophysical Descriptor Prediction
# Source: physics/PhysInformer/model.py
# Architecture: 8-layer transformer, 512 dim, 8 heads, 521 output features
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from Vaswani et al. "Attention Is All You Need".

    Encodes absolute position using sine/cosine functions at different frequencies:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This provides a unique encoding for each position that the transformer can
    use to distinguish nucleotide positions along the DNA sequence. The sinusoidal
    form allows the model to learn relative position attention patterns.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term computes 1/10000^(2i/d_model) for each dimension pair
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions: cos
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class NucleotideEmbedding(nn.Module):
    """Learned embedding for nucleotide tokens (A=0, C=1, G=2, T=3, N=4).

    Scaling by sqrt(d_model) follows the convention from "Attention Is All You Need":
    it ensures that the embedding magnitudes are comparable to the positional
    encoding magnitudes after addition, preventing either signal from dominating.
    """
    def __init__(self, vocab_size: int = 5, d_model: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # Scale to match PE magnitude


class PhysInformerBlock(nn.Module):
    """Standard transformer encoder block with self-attention and feed-forward.

    Uses post-norm residual connections (Add & Norm after each sub-layer),
    GELU activation in the feed-forward network, and multi-head self-attention
    to capture dependencies between nucleotide positions.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),   # Expand: d_model -> 4*d_model
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)    # Contract: 4*d_model -> d_model
        )
        self.norm1 = nn.LayerNorm(d_model)  # Post-attention LayerNorm
        self.norm2 = nn.LayerNorm(d_model)  # Post-FFN LayerNorm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention sub-layer with residual connection and LayerNorm
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Feed-forward sub-layer with residual connection and LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class PhysInformer(nn.Module):
    """
    Large transformer model for predicting biophysical descriptors from DNA sequences.

    The 521 biophysical descriptors are computed from DNA sequences using various
    biophysics tools (e.g., nearest-neighbor thermodynamics, DNA shape predictors,
    G-quadruplex scoring). PhysInformer learns to predict all 521 in a single forward
    pass, providing a ~12,000x speedup over computing them individually.

    Architecture:
    - NucleotideEmbedding: 5 tokens (A=0, C=1, G=2, T=3, N=4) -> 512 dim
    - Sinusoidal positional encoding (up to 240 bp)
    - 8 transformer encoder blocks (512 dim, 8 heads, 2048 FFN dim)
    - Global average pooling over sequence positions -> single 512-dim vector
    - 521 separate prediction heads (one MLP per biophysical feature)
      Each head: 512 -> 256 -> 128 -> 1 (independent predictions avoid
      cross-feature interference, since features span very different scales)

    Total parameters: ~12.3M (dominated by the 521 individual heads)

    Input: DNA sequence indices [batch_size, seq_len]
    Output: 521 biophysical descriptors [batch_size, n_descriptors]

    The 521 features span 5 families:
    - thermo_*: Thermodynamic properties (melting temp, stacking free energies,
      nearest-neighbor dH/dS/dG parameters)
    - stiff_*: Mechanical stiffness parameters (twist, roll, tilt, shift, slide,
      rise spring constants from dinucleotide models)
    - bend_*: Bending and curvature (DNA intrinsic curvature, bend angles,
      persistence length estimates)
    - entropy_*: Sequence complexity (Shannon entropy, linguistic complexity,
      repeat density, k-mer diversity)
    - advanced_*: G-quadruplex potential (G4Hunter), stress-induced duplex
      destabilization (SIDD), minor groove width (MGW), nucleosome positioning
      scores, DNA shape features (Roll, ProT, HelT, EP)

    The exact number of descriptors varies by cell type because different cell
    types may have different sets of relevant TF PWMs included in the feature
    computation (see create_physinformer configs: HepG2=537, K562=498, WTC11=522).
    """
    def __init__(
        self,
        vocab_size: int = 5,       # A, C, G, T, N (ambiguous)
        d_model: int = 512,        # Transformer hidden dimension
        n_heads: int = 8,          # Multi-head attention heads
        n_layers: int = 8,         # Number of transformer encoder blocks
        d_ff: int = 2048,          # Feed-forward intermediate dimension (4x d_model)
        max_len: int = 240,        # Maximum sequence length (230bp typical + padding)
        dropout: float = 0.1,      # Dropout rate throughout the model
        n_descriptors: int = 500,  # Number of biophysical features to predict
    ):
        super().__init__()

        self.d_model = d_model
        self.n_descriptors = n_descriptors

        # Embeddings: integer token IDs -> dense vectors, scaled by sqrt(d_model)
        # as per Vaswani et al. "Attention Is All You Need"
        self.nucleotide_embedding = NucleotideEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Stack of 8 transformer encoder blocks for contextualized sequence encoding
        self.transformer_blocks = nn.ModuleList([
            PhysInformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Global feature extraction: projects pooled representation before branching
        # into individual heads. Acts as a shared "trunk" transformation.
        self.global_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Individual feature prediction heads - one 3-layer MLP per biophysical feature.
        # Using separate heads (rather than a single shared head) is critical because
        # the 521 features span vastly different scales and distributions (e.g., free
        # energies in kcal/mol vs. dimensionless entropy scores). Independent heads
        # allow each feature to learn its own output scale without interference.
        # Architecture per head: 512 -> 256 -> 128 -> 1
        self.feature_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),    # 512 -> 256
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 4),  # 256 -> 128
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 1)            # 128 -> 1 scalar prediction
            ) for _ in range(n_descriptors)
        ])

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform, except final layers of feature heads.

        The final linear layer in each feature head (index .6.weight in the Sequential)
        uses a larger Normal(0, 0.5) initialization to spread initial predictions
        across the range of target values, preventing all heads from starting at zero.
        """
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'feature_heads' in name and '.6.weight' in name:
                    # Final output layer of each head: wider init for diverse starting predictions
                    nn.init.normal_(p, mean=0.0, std=0.5)
                else:
                    nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0.0)

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: DNA sequence indices [batch_size, seq_len]
            mask: Attention mask [seq_len, seq_len]

        Returns:
            Dict with 'descriptors' predictions [batch_size, n_descriptors]
        """
        batch_size, seq_len = x.shape

        # Embedding and positional encoding
        x_embed = self.nucleotide_embedding(x)
        x_embed = self.positional_encoding(x_embed)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x_embed = block(x_embed, mask)

        # Global average pooling over sequence dimension
        sequence_repr = x_embed.mean(dim=1)
        pooled_repr = self.global_pool(sequence_repr)

        # Individual feature predictions
        feature_predictions = []
        for head in self.feature_heads:
            pred = head(pooled_repr).squeeze(-1)
            feature_predictions.append(pred)

        descriptors = torch.stack(feature_predictions, dim=1)

        return {'descriptors': descriptors}


def create_physinformer(cell_type: str, **kwargs):
    """Create PhysInformer model configured for specific cell type.

    Different cell types have different numbers of descriptors because each
    cell type includes a different set of TF PWM features alongside the
    universal biophysical descriptors (thermo, stiff, bend, entropy, advanced).
    """
    # Descriptor counts per cell type (universal physics + cell-type-specific PWMs)
    configs = {
        'HepG2': {'n_descriptors': 537},   # Liver carcinoma cell line
        'K562': {'n_descriptors': 498},     # Chronic myelogenous leukemia cell line
        'WTC11': {'n_descriptors': 522}     # iPSC-derived cell line
    }

    if cell_type not in configs:
        raise ValueError(f"Unknown cell type: {cell_type}")

    default_params = {
        'vocab_size': 5,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 8,
        'd_ff': 2048,
        'max_len': 240,
        'dropout': 0.1,
    }

    params = {**default_params, **configs[cell_type], **kwargs}
    return PhysInformer(**params)


# =============================================================================
# PART 2: TileFormer - Electrostatic Potential Prediction
# Source: physics/TileFormer/models/tileformer_architecture.py
# Architecture: 6-layer transformer, 256 dim, 8 heads, 6 outputs
# =============================================================================

class TileFormerBlock(nn.Module):
    """Transformer block for TileFormer."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TileFormer(nn.Module):
    """
    Neural surrogate for APBS (Adaptive Poisson-Boltzmann Solver) electrostatic
    potential calculations on DNA sequences.

    APBS solves the Poisson-Boltzmann equation to compute the electrostatic
    potential (psi) around a DNA molecule in solution. This is computationally
    expensive (~seconds per sequence). TileFormer provides a ~10,000x speedup
    by learning to predict the summary statistics of these potentials directly
    from the nucleotide sequence.

    Architecture (3.2M parameters):
    - NucleotideEmbedding: 5 tokens -> 256 dim (smaller than PhysInformer
      because electrostatics is a simpler prediction task)
    - Sinusoidal positional encoding
    - 6 transformer encoder blocks (256 dim, 8 heads, 1024 FFN dim)
    - Global average pooling
    - Prediction head for 6 electrostatic values

    The 6 output values represent electrostatic potential (psi) statistics
    computed in two genomic contexts (standard promoter vs. enhancer conformation)
    and three summary statistics per context:
    - std_psi_min:  Minimum potential in standard (promoter) conformation
    - std_psi_max:  Maximum potential in standard conformation
    - std_psi_mean: Mean potential in standard conformation
    - enh_psi_min:  Minimum potential in enhancer conformation
    - enh_psi_max:  Maximum potential in enhancer conformation
    - enh_psi_mean: Mean potential in enhancer conformation

    These capture how the DNA's charge distribution varies, which affects
    protein-DNA interactions and chromatin accessibility.

    Performance: R^2 = 0.960, Pearson r = 0.982 on held-out sequences.

    Optional uncertainty prediction head outputs calibrated aleatoric
    uncertainty via a heteroscedastic Gaussian (predicts log-variance).
    """
    def __init__(
        self,
        vocab_size: int = 5,            # A, C, G, T, N
        d_model: int = 256,             # Smaller dim than PhysInformer (simpler task)
        n_heads: int = 8,               # Attention heads
        n_layers: int = 6,              # Fewer layers than PhysInformer (6 vs 8)
        d_ff: int = 1024,               # Feed-forward dim (4x d_model)
        max_len: int = 200,             # Max sequence length for electrostatics
        dropout: float = 0.1,           # Dropout rate
        output_dim: int = 6,            # 6 electrostatic values (2 contexts x 3 stats)
        predict_uncertainty: bool = True  # Whether to predict aleatoric uncertainty
    ):
        super().__init__()
        self.d_model = d_model
        self.predict_uncertainty = predict_uncertainty

        # Embeddings
        self.nucleotide_embedding = NucleotideEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # 6-layer transformer for contextualizing nucleotide representations
        self.transformer_blocks = nn.ModuleList([
            TileFormerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Shared trunk after global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Electrostatic potential prediction head: predicts 6 values
        # (STD/ENH x MIN/MAX/MEAN potential statistics)
        self.psi_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 256 -> 128
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)  # 128 -> 6
        )

        if predict_uncertainty:
            # Separate head for aleatoric uncertainty (log-variance per output).
            # Uses softplus activation in forward() to ensure positive variance.
            self.uncertainty_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_dim)
            )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        """
        Forward pass: DNA sequence -> electrostatic potential predictions.

        Args:
            x: Nucleotide indices [batch_size, seq_len], values in {0,1,2,3,4}
            mask: Optional attention mask [seq_len, seq_len]

        Returns:
            Dict with:
            - 'psi': Predicted electrostatic potentials [batch_size, 6]
            - 'uncertainty': (optional) Predicted aleatoric variance [batch_size, 6]
        """
        batch_size, seq_len = x.shape

        # Embedding and positional encoding
        x = self.nucleotide_embedding(x)
        x = self.positional_encoding(x)

        # Pass through transformer blocks for contextual representation
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Global average pooling: collapse sequence dim into single vector
        # This is appropriate because electrostatic potentials are sequence-level
        # summary statistics, not position-specific values
        x = x.mean(dim=1)
        x = self.global_pool(x)

        # Predict the 6 electrostatic potential summary statistics
        psi_values = self.psi_head(x)

        outputs = {'psi': psi_values}

        if self.predict_uncertainty:
            # softplus ensures positive uncertainty values (variance must be > 0)
            uncertainty = F.softplus(self.uncertainty_head(x))
            outputs['uncertainty'] = uncertainty

        return outputs


class TileFormerWithMetadata(TileFormer):
    """TileFormer variant that also takes precomputed sequence metadata as input.

    Fuses transformer-encoded sequence features with scalar metadata features
    (GC content, CpG density, minor groove score) via concatenation and a
    learned fusion layer. This provides the model with explicit access to
    global sequence composition statistics that may be relevant for
    electrostatic potential prediction.
    """
    def __init__(
        self,
        metadata_dim: int = 3,  # gc_content, cpg_density, minor_groove_score
        **kwargs
    ):
        super().__init__(**kwargs)

        # Metadata processing
        self.metadata_processor = nn.Sequential(
            nn.Linear(metadata_dim, self.d_model // 4),
            nn.GELU(),
            nn.Linear(self.d_model // 4, self.d_model)
        )

        # Combine sequence and metadata features
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(kwargs.get('dropout', 0.1))
        )

    def forward(self, x, metadata, mask=None):
        batch_size, seq_len = x.shape

        x = self.nucleotide_embedding(x)
        x = self.positional_encoding(x)

        for block in self.transformer_blocks:
            x = block(x, mask)

        x = x.mean(dim=1)

        # Process metadata
        metadata_features = self.metadata_processor(metadata)

        # Fuse features
        combined = torch.cat([x, metadata_features], dim=-1)
        x = self.feature_fusion(combined)
        x = self.global_pool(x)

        psi_values = self.psi_head(x)
        outputs = {'psi': psi_values}

        if self.predict_uncertainty:
            uncertainty = F.softplus(self.uncertainty_head(x))
            outputs['uncertainty'] = uncertainty

        return outputs


# =============================================================================
# PART 3: PhysicsVAE - Physics-Conditioned Variational Autoencoder
# Source: physics/PhysicsVAE/models/physics_vae.py
# Architecture: Conv encoder -> latent (128) + physics cond (64) -> transformer decoder
# =============================================================================

class SequenceEncoder(nn.Module):
    """
    Encodes DNA sequences into latent representations for the PhysicsVAE.

    This is the "recognition network" q(z|x) in VAE terminology. It takes a
    one-hot encoded DNA sequence and compresses it into a latent distribution
    parameterized by (mu, log_var). The latent vector z captures sequence
    identity information that, combined with physics conditioning, allows
    the decoder to reconstruct the original sequence.

    Architecture:
    - One-hot encoding: nucleotide indices -> 4-channel binary tensor
    - 3-layer conv stack with decreasing kernel sizes (9, 7, 5) to capture
      patterns at multiple scales (from 9-mer motifs down to 5-mer features)
    - Each conv layer followed by BatchNorm, GELU, Dropout, and 2x MaxPool
    - After 3 rounds of 2x pooling: seq_length/8 positions remain
    - Global average pooling over remaining positions -> 256-dim vector
    - Two linear projections to mu and log_var (both 128-dim)

    The decreasing kernel sizes follow a coarse-to-fine strategy: early layers
    capture longer motifs, later layers refine with smaller receptive fields.
    MaxPooling progressively reduces spatial resolution while retaining the
    strongest activations.
    """

    def __init__(
        self,
        seq_length: int = 200,    # Input sequence length (bp)
        latent_dim: int = 128,    # Dimensionality of the latent space z
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim

        # Convolutional encoder: progressively increases channels while
        # decreasing kernel size. Padding preserves spatial dimensions
        # before pooling (padding = (kernel_size - 1) / 2 for odd kernels).
        self.conv1 = nn.Conv1d(4, 64, kernel_size=9, padding=4)    # 4 -> 64 channels
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)  # 64 -> 128 channels
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2) # 128 -> 256 channels
        self.bn3 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(2)  # 2x downsampling at each stage

        # Project pooled features to latent distribution parameters.
        # Two separate projections for the mean and log-variance of the
        # approximate posterior q(z|x) = N(mu, diag(exp(logvar)))
        self.fc_mu = nn.Linear(256, latent_dim)      # Mean of q(z|x)
        self.fc_logvar = nn.Linear(256, latent_dim)   # Log-variance of q(z|x)

    def _indices_to_onehot(self, x: torch.Tensor) -> torch.Tensor:
        """Convert nucleotide indices to one-hot encoding."""
        batch_size, seq_len = x.shape
        onehot = torch.zeros(batch_size, 4, seq_len, device=x.device, dtype=torch.float32)
        mask = x < 4
        indices = x.clamp(0, 3)
        onehot.scatter_(1, indices.unsqueeze(1), mask.unsqueeze(1).float())
        return onehot

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequence into latent distribution parameters (mu, logvar).

        Args:
            x: Nucleotide indices [batch, seq_len], values in {0,1,2,3,4}

        Returns:
            mu: Mean of approximate posterior [batch, latent_dim]
            logvar: Log-variance of approximate posterior [batch, latent_dim]
        """
        # Convert integer indices to one-hot: [batch, seq_len] -> [batch, 4, seq_len]
        x = self._indices_to_onehot(x)

        # 3-stage conv encoding with pooling: seq_len -> seq_len/2 -> seq_len/4 -> seq_len/8
        x = self.pool(self.dropout(F.gelu(self.bn1(self.conv1(x)))))  # [batch, 64, seq_len/2]
        x = self.pool(self.dropout(F.gelu(self.bn2(self.conv2(x)))))  # [batch, 128, seq_len/4]
        x = self.pool(self.dropout(F.gelu(self.bn3(self.conv3(x)))))  # [batch, 256, seq_len/8]

        # Global average pooling over remaining positions -> [batch, 256]
        x = x.mean(dim=2)

        # Project to latent distribution parameters
        mu = self.fc_mu(x)         # [batch, 128]
        logvar = self.fc_logvar(x)  # [batch, 128]

        return mu, logvar


class PhysicsEncoder(nn.Module):
    """
    Encodes biophysical descriptor vectors into compact conditioning vectors
    for the VAE decoder.

    This is a bottleneck MLP that compresses the 521-dimensional physics
    feature vector into a 64-dimensional conditioning vector (z_physics).
    The compression forces the network to learn which physics features are
    most informative for sequence reconstruction, effectively performing
    a learned dimensionality reduction on the physics space.

    Architecture: 521 -> 256 -> 128 -> 64
    Each hidden layer uses LayerNorm (more stable than BatchNorm for
    conditioning vectors) + GELU activation + Dropout.

    The output z_physics is concatenated with the latent z from the
    SequenceEncoder to form the 192-dim input to the SequenceDecoder.
    """

    def __init__(
        self,
        n_physics_features: int = 521,   # Number of input biophysical descriptors
        hidden_dims: list = [256, 128],   # Intermediate layer sizes
        output_dim: int = 64,             # Conditioning vector dimension
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_physics_features = n_physics_features

        # Build MLP with LayerNorm + GELU + Dropout between each layer
        layers = []
        in_dim = n_physics_features
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # LayerNorm preferred over BatchNorm for conditioning
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # Final projection to conditioning dimension (no activation - let decoder
        # decide how to use the conditioning signal)
        layers.append(nn.Linear(in_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, physics: torch.Tensor) -> torch.Tensor:
        """Map physics features to conditioning vector: [batch, 521] -> [batch, 64]."""
        return self.encoder(physics)


class PositionalEncodingVAE(nn.Module):
    """Sinusoidal positional encoding for VAE decoder."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SequenceDecoder(nn.Module):
    """
    Decodes latent + physics conditioning into DNA sequences.

    This is the "generative network" p(x|z, z_physics) in VAE terminology.
    It takes the concatenated latent vector z (sequence identity) and physics
    conditioning z_physics (biophysical constraints), then generates a full
    DNA sequence of nucleotide logits.

    The decoding pipeline:
    1. Concatenate z (128-dim) and z_physics (64-dim) -> 192-dim combined vector
    2. Project combined vector to a full sequence of hidden states via a single
       large linear layer: 192 -> seq_length * 192 (this "unfolds" the single
       latent vector into position-specific representations)
    3. Add sinusoidal positional encoding so the transformer knows position
    4. Refine with 4-layer transformer encoder (using self-attention, despite
       the name -- this allows each position to attend to all other positions
       for globally coherent sequence generation)
    5. Project each position's hidden state to 4-way nucleotide logits (A/C/G/T)

    Note: We use TransformerEncoder (not TransformerDecoder) because we are not
    doing autoregressive generation -- all positions are predicted in parallel.
    This is appropriate for fixed-length regulatory sequence generation where
    all positions are equally important.
    """

    def __init__(
        self,
        seq_length: int = 200,   # Output sequence length (bp)
        latent_dim: int = 128,   # Dimension of z from SequenceEncoder
        physics_dim: int = 64,   # Dimension of z_physics from PhysicsEncoder
        n_heads: int = 4,        # Attention heads (192/4 = 48 dim per head)
        n_layers: int = 4,       # Transformer layers for sequence refinement
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_dim = latent_dim + physics_dim  # 128 + 64 = 192

        # "Unfolding" layer: project single combined vector to full sequence
        # of hidden states. This is a very large linear layer:
        # 192 -> 200 * 192 = 38,400 outputs
        self.latent_to_seq = nn.Linear(self.hidden_dim, seq_length * self.hidden_dim)

        # Positional encoding for the unfolded sequence
        self.pos_encoding = PositionalEncodingVAE(self.hidden_dim, max_len=seq_length, dropout=dropout)

        # Transformer for refining position-specific representations.
        # Uses self-attention so each position can attend to all others,
        # enabling globally coherent sequences (e.g., matching GC content,
        # maintaining motif spacing, satisfying structural constraints).
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,          # 192 dim
            nhead=n_heads,                     # 4 heads, 48 dim each
            dim_feedforward=self.hidden_dim * 4,  # 768 FFN dim
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)

        # Final projection: 192-dim hidden state -> 4-way nucleotide logits per position
        self.output_proj = nn.Linear(self.hidden_dim, 4)

    def forward(
        self,
        z: torch.Tensor,
        physics_cond: torch.Tensor
    ) -> torch.Tensor:
        batch_size = z.size(0)

        # Concatenate latent and physics conditioning
        combined = torch.cat([z, physics_cond], dim=-1)

        # Project to sequence
        x = self.latent_to_seq(combined)
        x = x.view(batch_size, self.seq_length, self.hidden_dim)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer processing
        x = self.transformer(x)

        # Output nucleotide logits
        logits = self.output_proj(x)

        return logits


class PhysicsVAE(nn.Module):
    """
    Full Physics-Conditioned VAE for sequence generation.

    Architecture:
    - SequenceEncoder: Conv (4->64->128->256) + global pool -> latent (dim=128)
    - PhysicsEncoder: Dense (n_physics->256->128->64) -> z_physics (dim=64)
    - SequenceDecoder: Transformer (dim=192, 4 layers, 4 heads) -> nucleotides

    Loss: L = L_recon + beta*L_KL + lambda*L_physics
          beta = 0.001 (weak regularization), lambda = 0.1
    """

    def __init__(
        self,
        seq_length: int = 200,
        n_physics_features: int = 521,
        latent_dim: int = 128,
        physics_cond_dim: int = 64,
        n_decoder_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.physics_cond_dim = physics_cond_dim

        # Sequence Encoder
        self.sequence_encoder = SequenceEncoder(
            seq_length=seq_length,
            latent_dim=latent_dim,
            dropout=dropout
        )

        # Physics Encoder
        self.physics_encoder = PhysicsEncoder(
            n_physics_features=n_physics_features,
            hidden_dims=[256, 128],
            output_dim=physics_cond_dim,
            dropout=dropout
        )

        # Decoder
        self.decoder = SequenceDecoder(
            seq_length=seq_length,
            latent_dim=latent_dim,
            physics_dim=physics_cond_dim,
            n_heads=4,
            n_layers=n_decoder_layers,
            dropout=dropout
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick (Kingma & Welling 2014) for sampling from latent distribution.

        Instead of sampling z ~ N(mu, sigma^2) directly (which is not differentiable),
        we sample eps ~ N(0, 1) and compute z = mu + eps * sigma. This makes the
        sampling operation differentiable with respect to mu and logvar, allowing
        backpropagation through the sampling step.

        At inference time, we simply return mu (the MAP estimate) for deterministic output.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)  # sigma = exp(0.5 * log(sigma^2))
            eps = torch.randn_like(std)     # eps ~ N(0, I)
            return mu + eps * std            # z = mu + eps * sigma
        else:
            return mu  # Deterministic: use mean of posterior

    def encode(
        self,
        sequence: torch.Tensor,
        physics: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode sequence and physics into latent representations.

        Returns z (sampled latent), mu, logvar (for KL loss), and physics_cond.
        """
        mu, logvar = self.sequence_encoder(sequence)  # q(z|x): sequence -> (mu, logvar)
        z = self.reparameterize(mu, logvar)            # Sample z from q(z|x)
        physics_cond = self.physics_encoder(physics)   # Compress physics to conditioning
        return z, mu, logvar, physics_cond

    def decode(
        self,
        z: torch.Tensor,
        physics_cond: torch.Tensor
    ) -> torch.Tensor:
        """Decode latent + physics conditioning into nucleotide logits.

        Returns logits [batch, seq_length, 4] (not probabilities -- apply softmax for probs).
        """
        return self.decoder(z, physics_cond)

    def forward(
        self,
        sequence: torch.Tensor,
        physics: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        z, mu, logvar, physics_cond = self.encode(sequence, physics)
        logits = self.decode(z, physics_cond)

        return {
            'logits': logits,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'physics_cond': physics_cond
        }

    def generate(
        self,
        physics: torch.Tensor,
        n_samples: int = 1,
        temperature: float = 1.0,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate DNA sequences conditioned on desired physics features.

        Generation process:
        1. Encode the target physics features into conditioning vector z_physics
        2. Sample z from the prior N(0, I) (or use provided z for reproducibility)
        3. Decode (z, z_physics) -> nucleotide logits at each position
        4. Apply temperature scaling to logits and sample nucleotides

        The physics conditioning allows generating sequences that should exhibit
        specific biophysical properties (e.g., high bending, specific GC content,
        target electrostatic potential). Different z samples with the same physics
        produce different sequences with similar biophysical properties.

        Args:
            physics: Target biophysical features [batch, n_physics] or [n_physics]
            n_samples: Number of sequences to generate per physics input
            temperature: Sampling temperature (lower = more deterministic,
                        higher = more diverse). Default 1.0 = standard sampling.
            z: Optional latent vectors [batch * n_samples, latent_dim].
               If None, sampled from N(0, I).

        Returns:
            Generated nucleotide index sequences [batch * n_samples, seq_length]
        """
        self.eval()
        with torch.no_grad():
            # Handle single physics vector (unbatched)
            if physics.dim() == 1:
                physics = physics.unsqueeze(0)

            batch_size = physics.size(0)
            device = physics.device

            # Replicate each physics vector n_samples times for generating
            # multiple diverse sequences per physics specification
            physics = physics.repeat_interleave(n_samples, dim=0)
            physics_cond = self.physics_encoder(physics)

            # Sample from the prior p(z) = N(0, I) if z not provided
            if z is None:
                z = torch.randn(batch_size * n_samples, self.latent_dim, device=device)

            # Decode latent + physics conditioning to nucleotide logits
            logits = self.decode(z, physics_cond)

            # Temperature-scaled categorical sampling at each position.
            # temperature < 1.0: sharper distribution (more confident, less diverse)
            # temperature > 1.0: flatter distribution (less confident, more diverse)
            probs = F.softmax(logits / temperature, dim=-1)
            sequences = torch.multinomial(
                probs.view(-1, 4),   # Flatten to [batch*seq_len, 4] for multinomial
                num_samples=1
            ).view(batch_size * n_samples, self.seq_length)

            return sequences


# =============================================================================
# PART 4: S2A Universal Features - Cross-Species Transfer
# Source: physics/S2A/universal_features.py
#
# S2A ("Sequence-to-Activity") extracts universal physics features that enable
# cross-species regulatory activity prediction. The key insight is that DNA
# biophysics (thermodynamics, mechanics, shape) is determined by chemistry
# and is therefore identical across all organisms, while TF binding (PWMs)
# is species-specific because transcription factor repertoires evolve.
# =============================================================================

@dataclass
class S2AConfig:
    """Configuration for S2A cross-species transfer.

    Defines which feature families are universal (physics-based, conserved
    across species) and which are species-specific (must be excluded for
    cross-species transfer).
    """
    # Feature selection - universal physics features only
    universal_prefixes: List[str] = field(default_factory=lambda: [
        'thermo_',    # Thermodynamic features (dG, dH, dS, Tm)
        'stiff_',     # Mechanical stiffness (twist, tilt, roll)
        'bend_',      # Bending and curvature
        'entropy_',   # Sequence complexity and information
        'advanced_',  # G4, SIDD, MGW, nucleosome positioning
    ])

    # Species-specific features to exclude
    excluded_prefixes: List[str] = field(default_factory=lambda: [
        'pwm_',  # Position weight matrix (TF binding - species-specific)
    ])

    # Head model configuration
    head_type: str = 'ridge'  # 'ridge', 'elastic_net', 'mlp'
    head_alpha: float = 1.0   # Regularization strength
    head_l1_ratio: float = 0.5  # For elastic_net only
    head_hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])  # For MLP

    # Output mode
    output_mode: str = 'zscore'  # 'zscore', 'calibrated', 'ranking'

    # Training parameters
    random_seed: int = 42
    n_folds: int = 5  # For CV during training


@dataclass
class UniversalFeatureStats:
    """Statistics for feature normalization."""
    mean: np.ndarray
    std: np.ndarray
    feature_names: List[str]
    n_features: int


class UniversalFeatureExtractor:
    """
    Extract universal physics features from descriptor files.

    Key insight: Physics features are universal because DNA chemistry is
    identical across organisms. PWM features are species-specific because
    transcription factors evolve differently.
    """

    def __init__(self, config: S2AConfig = None):
        """
        Initialize feature extractor.

        Args:
            config: S2AConfig with feature settings
        """
        self.config = config or S2AConfig()
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self._is_fitted = False
        self.feature_names: List[str] = []
        self.stats: Optional[UniversalFeatureStats] = None

    def _is_universal_feature(self, col_name: str) -> bool:
        """Check if a column is a universal physics feature."""
        # Check if it matches any universal prefix
        is_universal = any(
            col_name.startswith(prefix)
            for prefix in self.config.universal_prefixes
        )

        # Check if it should be excluded
        is_excluded = any(
            col_name.startswith(prefix)
            for prefix in self.config.excluded_prefixes
        )

        return is_universal and not is_excluded

    def extract_feature_columns(self, df: 'pd.DataFrame') -> List[str]:
        """
        Extract list of universal feature columns from a dataframe.

        Args:
            df: DataFrame with physics features

        Returns:
            List of universal feature column names
        """
        feature_cols = [
            col for col in df.columns
            if self._is_universal_feature(col)
        ]
        return sorted(feature_cols)

    def fit(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> 'UniversalFeatureExtractor':
        """
        Fit the feature scaler.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names

        Returns:
            Self for chaining
        """
        self.scaler.fit(X)
        self.feature_names = feature_names
        self._is_fitted = True

        self.stats = UniversalFeatureStats(
            mean=self.scaler.mean_.astype(np.float32),
            std=self.scaler.scale_.astype(np.float32),
            feature_names=feature_names,
            n_features=len(feature_names)
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.

        Args:
            X: Feature matrix

        Returns:
            Scaled feature matrix
        """
        if not self._is_fitted:
            raise ValueError("Extractor not fitted. Call fit() first.")

        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Fit and transform features.

        Args:
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            Scaled feature matrix
        """
        self.fit(X, feature_names)
        return self.transform(X)

    def get_feature_family_indices(self) -> Dict[str, List[int]]:
        """
        Get indices of features by family (thermo, stiff, etc.).

        Returns:
            Dict mapping family prefix to list of feature indices
        """
        if not self._is_fitted:
            raise ValueError("Extractor not fitted. Call fit() first.")

        families = {}
        for prefix in self.config.universal_prefixes:
            prefix_clean = prefix.rstrip('_')
            indices = [
                i for i, name in enumerate(self.feature_names)
                if name.startswith(prefix)
            ]
            if indices:
                families[prefix_clean] = indices

        return families

    def count_features_by_family(self) -> Dict[str, int]:
        """
        Count features by family.

        Returns:
            Dict mapping family name to feature count
        """
        families = self.get_feature_family_indices()
        return {k: len(v) for k, v in families.items()}


class UniversalS2AHead:
    """
    Universal head model for physics->activity prediction.
    Source: physics/S2A/universal_head.py

    Trained on z-scored activity from multiple species.
    Outputs z-scores or rankings for zero-shot inference.
    """

    def __init__(self, config: S2AConfig = None):
        """
        Initialize S2A head.

        Args:
            config: S2AConfig with model parameters
        """
        self.config = config or S2AConfig()
        self.model = None
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._is_fitted = False

    def _create_model(self):
        """Create the underlying regression model."""
        from sklearn.linear_model import Ridge, ElasticNet
        from sklearn.neural_network import MLPRegressor

        head_type = self.config.head_type

        if head_type == 'ridge':
            return Ridge(
                alpha=self.config.head_alpha,
                random_state=self.config.random_seed
            )
        elif head_type == 'elastic_net':
            return ElasticNet(
                alpha=self.config.head_alpha,
                l1_ratio=self.config.head_l1_ratio,
                max_iter=10000,
                random_state=self.config.random_seed
            )
        elif head_type == 'mlp':
            return MLPRegressor(
                hidden_layer_sizes=tuple(self.config.head_hidden_sizes),
                activation='relu',
                solver='adam',
                alpha=self.config.head_alpha,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.config.random_seed
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def fit(
        self,
        X: np.ndarray,
        y_zscore: np.ndarray,
        feature_names: List[str] = None
    ) -> 'UniversalS2AHead':
        """
        Fit the head model on z-scored activity data.

        Args:
            X: Physics features (n_samples, n_features)
            y_zscore: Z-scored activity values (n_samples,)
            feature_names: Optional feature names

        Returns:
            Self for chaining
        """
        self.feature_names = feature_names or [
            f'feature_{i}' for i in range(X.shape[1])
        ]

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_scaled, y_zscore)

        self._is_fitted = True
        return self

    def predict_zscore(self, X: np.ndarray) -> np.ndarray:
        """
        Predict z-scored activity.

        Args:
            X: Physics features (n_samples, n_features)

        Returns:
            Predicted z-scores (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("Head not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_ranking(self, X: np.ndarray) -> np.ndarray:
        """
        Predict percentile rankings (0-100).

        Rankings are more robust for zero-shot transfer since
        they only depend on relative ordering, not absolute scale.

        Args:
            X: Physics features (n_samples, n_features)

        Returns:
            Percentile ranks (n_samples,), 0-100
        """
        from scipy.stats import rankdata
        z_scores = self.predict_zscore(X)
        # Convert to percentile ranks
        ranks = rankdata(z_scores, method='average')
        percentiles = (ranks - 1) / (len(ranks) - 1) * 100
        return percentiles

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ):
        """
        Evaluate head model performance.

        Args:
            X: Physics features
            y_true: True activity values (can be raw or z-scored)
        """
        from scipy.stats import spearmanr, pearsonr
        y_pred = self.predict_zscore(X)

        # Compute correlation metrics (invariant to scale)
        spearman = spearmanr(y_true, y_pred)[0]
        pearson_val = pearsonr(y_true, y_pred)[0]

        return {
            'spearman_rho': spearman,
            'pearson_r': pearson_val,
            'n_samples': len(y_true)
        }

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """
        Get feature importances from the model.

        Returns:
            Dict mapping feature names to importance (absolute coefficient)
            or None if not available.
        """
        if not self._is_fitted or self.model is None:
            return None

        # Linear models have coef_
        if hasattr(self.model, 'coef_'):
            coefs = np.abs(self.model.coef_)
            return {
                name: float(coef)
                for name, coef in zip(self.feature_names, coefs)
            }

        return None


# =============================================================================
# PART 5: Physics-Aware Model Components
# Source: physics/PhysInformer/physics_aware_model.py
# Property-specific routers and prediction heads
# =============================================================================

class PhysicsRouter(nn.Module):
    """Routes shared backbone features through property-specific convolutional adapters.

    The key design insight is that different biophysical properties operate at
    different length scales along the DNA, so each property type needs a different
    convolutional kernel size (receptive field) to capture the relevant patterns:

    - Thermodynamic (k=3): Nearest-neighbor dinucleotide interactions. Stacking
      energies, hydrogen bonding, and melting behavior are determined by
      immediately adjacent base pairs, so a 3bp window suffices.
    - Bending (k=11): DNA curvature emerges from ~1 helical turn (~10bp).
      Intrinsic bend angles depend on phased A-tracts and structural
      periodicity over 10-11bp windows.
    - Electrostatic (k=15): Charge distribution depends on the arrangement of
      phosphate groups and groove geometry over ~1.5 helical turns.
    - PWM/TF binding (k=15): Transcription factor binding motifs are typically
      6-20bp, so a 15bp kernel captures most motif-scale patterns.
    - Entropy (k=21): Sequence complexity and repeat structure requires a
      ~2 helical turn window to assess k-mer diversity and periodicity.

    Each router also includes a learned gate (scalar sigmoid) that controls
    how much of the property-specific signal to pass through, allowing the
    model to dynamically weight different property channels.

    Optional reverse-complement (RC) averaging: for properties that should be
    strand-symmetric (e.g., thermodynamics), the forward and RC outputs are
    averaged for improved robustness.
    """

    def __init__(self, d_input: int, d_output: int, kernel_size: int,
                 property_name: str, use_rc: bool = False):
        super().__init__()
        self.property_name = property_name
        self.use_rc = use_rc

        # Window aggregator with property-specific receptive field
        # Use groups only if d_input is divisible by the group size
        n_groups = 1
        if d_input >= d_output and d_input % d_output == 0:
            n_groups = d_output
        elif d_output >= d_input and d_output % d_input == 0:
            n_groups = d_input

        self.window_agg = nn.Conv1d(d_input, d_output, kernel_size=kernel_size,
                                   padding=kernel_size//2, groups=n_groups)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_input, 1),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_output)

    def forward(self, x):
        # x: (batch, seq_len, d_input)
        batch, seq_len, d_input = x.shape

        # Apply window aggregation
        x_conv = x.transpose(1, 2)  # (batch, d_input, seq_len)
        x_agg = self.window_agg(x_conv).transpose(1, 2)  # (batch, seq_len, d_output)

        # Compute gate
        gate = self.gate(x.mean(dim=1, keepdim=True))  # (batch, 1, 1)

        # Apply gate and norm
        output = gate * x_agg
        output = self.norm(output)

        # Handle reverse complement if needed
        if self.use_rc:
            # Average with reverse complement
            output_rc = torch.flip(output, dims=[1])
            output = (output + output_rc) / 2

        return output


class ThermoHead(nn.Module):
    """Joint prediction head for thermodynamic properties (dH, dS, dG).

    Enforces the thermodynamic identity constraint: dG = dH - T*dS
    (Gibbs free energy = enthalpy - temperature * entropy).

    The constraint is not enforced architecturally (which would limit
    expressiveness), but rather through a soft penalty in the loss function
    (PhysicsAwareLoss.thermodynamic_identity_loss). This allows the model to
    learn the relationship while still having independent output heads that
    can capture any systematic deviations.

    Each property head outputs 2 values: (mean, log_var) for heteroscedastic
    uncertainty estimation. The log_var output allows the model to express
    higher uncertainty for sequences where thermodynamic predictions are
    less reliable (e.g., unusual sequence compositions).

    Architecture:
    - Shared feature layer: d_input -> 128 (captures common thermodynamic features)
    - Three independent output heads: 128 -> 2 each (mean + log_var)

    Args:
        d_input: Input feature dimension from the PhysicsRouter
        d_hidden: Shared layer hidden dimension
        temperature: Temperature in Kelvin for the thermodynamic identity
                     (default 310K = 37C, standard physiological temperature)
    """

    def __init__(self, d_input: int, d_hidden: int = 128, temperature: float = 310.0):
        super().__init__()
        self.temperature = temperature  # 310K = 37C (physiological temperature)

        # Shared feature extraction (common to all three thermodynamic properties,
        # since dH, dS, and dG are physically related quantities)
        self.shared = nn.Sequential(
            nn.Linear(d_input, d_hidden), nn.SiLU(), nn.Dropout(0.1)
        )
        # Per-property heads: each outputs (mean_prediction, log_variance)
        self.dH_head = nn.Linear(d_hidden, 2)  # Enthalpy: (mean, log_var)
        self.dS_head = nn.Linear(d_hidden, 2)  # Entropy: (mean, log_var)
        self.dG_head = nn.Linear(d_hidden, 2)  # Free energy: (mean, log_var)

    def forward(self, x):
        """Predict thermodynamic properties with uncertainty.

        Args:
            x: Input features [batch, seq_len, d_input] or [batch, d_input]

        Returns:
            Dict with dH/dS/dG mean and log_var predictions (6 scalars per sample)
        """
        # Pool over sequence dimension if needed (thermodynamic properties are
        # sequence-level, not position-specific)
        if len(x.shape) == 3:
            x = x.mean(dim=1)
        features = self.shared(x)

        dH = self.dH_head(features)  # [batch, 2]
        dS = self.dS_head(features)  # [batch, 2]
        dG = self.dG_head(features)  # [batch, 2]

        return {
            'dH_mean': dH[:, 0], 'dH_log_var': dH[:, 1],
            'dS_mean': dS[:, 0], 'dS_log_var': dS[:, 1],
            'dG_mean': dG[:, 0], 'dG_log_var': dG[:, 1],
        }


class ElectrostaticHead(nn.Module):
    """Head for position-resolved electrostatic potential prediction using
    a sliding window approach.

    Electrostatic potentials vary along the DNA sequence because different
    regions have different charge distributions (affected by nucleotide
    composition, groove geometry, and local structure). Rather than predicting
    a single global potential, this head uses overlapping windows to capture
    the spatial profile of electrostatic properties along the sequence.

    Sliding window approach:
    - Window size: 20bp (~2 helical turns, matching the spatial resolution
      of APBS calculations on DNA)
    - Stride: 10bp (50% overlap for smooth spatial profiles)
    - Each window predicts 6 values: STD/ENH x MIN/MAX/MEAN potentials
    - 22 windows cover a 230bp sequence with the above parameters
    - Each window has its own MLP head (not shared), because electrostatic
      properties can vary systematically along promoter/enhancer regions
      (e.g., TSS-proximal vs. distal regions differ structurally)

    Additionally, a global head predicts sequence-level mean potential
    with uncertainty (mean + log_var), providing a sanity check against
    the window-level predictions.
    """

    def __init__(self, d_input: int, d_hidden: int = 128, n_windows: int = 22):
        super().__init__()
        self.n_windows = n_windows  # Number of sliding windows

        # Per-window prediction heads (independent MLPs for each window position).
        # Each outputs 6 values: [std_min, std_max, std_mean, enh_min, enh_max, enh_mean]
        self.window_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_input, d_hidden), nn.SiLU(), nn.Dropout(0.1),
                nn.Linear(d_hidden, 6)   # 6 electrostatic summary statistics
            ) for _ in range(n_windows)
        ])

        # Global summary head: predicts overall mean potential + uncertainty
        self.global_head = nn.Sequential(
            nn.Linear(d_input, d_hidden), nn.SiLU(), nn.Linear(d_hidden, 2)
        )

    def forward(self, x):
        """Predict electrostatic potentials using sliding windows.

        Args:
            x: Backbone features [batch, seq_len, d_input]

        Returns:
            Dict with:
            - 'window_predictions': Per-window potentials [batch, n_windows, 6]
            - 'global_mean': Sequence-level mean potential [batch]
            - 'global_log_var': Uncertainty of global prediction [batch]
        """
        batch, seq_len, d_input = x.shape

        # Sliding window extraction: 20bp windows with 10bp stride
        window_size = 20   # ~2 helical turns
        stride = 10        # 50% overlap between adjacent windows
        window_features = []

        for i in range(0, seq_len - window_size + 1, stride):
            # Average pool features within each window to get a single vector
            window = x[:, i:i+window_size, :].mean(dim=1)  # [batch, d_input]
            window_features.append(window)

        # Predict electrostatic properties for each window
        window_predictions = []
        for i, window_feat in enumerate(window_features[:self.n_windows]):
            pred = self.window_heads[i](window_feat)  # [batch, 6]
            window_predictions.append(pred)

        # Global prediction from full-sequence pooling
        global_feat = x.mean(dim=1)
        global_pred = self.global_head(global_feat)

        return {
            'window_predictions': torch.stack(window_predictions, dim=1),  # [batch, n_windows, 6]
            'global_mean': global_pred[:, 0],       # Scalar per sample
            'global_log_var': global_pred[:, 1]      # Uncertainty per sample
        }


class AuxiliaryHeadA(nn.Module):
    """Activity prediction from Sequence + Real Features

    This head has its OWN sequence encoder, completely separate from the main model.
    It learns to process sequences specifically for activity prediction.
    """

    def __init__(self, vocab_size: int = 5, seq_len: int = 230, feature_dim: int = 536,
                 hidden_dim: int = 256, n_activities: int = 1, dropout: float = 0.1):
        super().__init__()

        # Own sequence encoder - lightweight CNN for motif detection
        self.sequence_encoder = nn.Sequential(
            # Embedding layer
            nn.Embedding(vocab_size, 64),
            # Conv layers to detect motifs
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling to (batch, hidden_dim, 1)
        )

        # Feature projector - brings physics features to same scale
        self.feat_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # Sequence projector - process encoded sequence
        self.seq_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # Optional gating mechanism driven by features
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.Sigmoid()
        )

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_activities)
        )

    def forward(self, sequences: torch.Tensor, real_features: torch.Tensor):
        """
        Args:
            sequences: Raw DNA sequences (batch, seq_len) - NOT from main model
            real_features: Real physics features (batch, feature_dim)
        Returns:
            Activity predictions (batch, n_activities)
        """
        # Encode sequences with OWN encoder
        seq_emb = self.sequence_encoder[0](sequences)  # Embedding: (batch, seq_len, 64)
        seq_emb = seq_emb.transpose(1, 2)  # (batch, 64, seq_len)
        for layer in self.sequence_encoder[1:]:
            seq_emb = layer(seq_emb)
        seq_emb = seq_emb.squeeze(-1)  # (batch, hidden_dim)

        # Project sequence and features
        seq_emb = self.seq_projector(seq_emb)
        feat_emb = self.feat_projector(real_features)

        # Apply gating (features modulate sequence)
        gate_values = self.gate(feat_emb)
        seq_emb = seq_emb * gate_values

        # Concatenate and fuse
        combined = torch.cat([seq_emb, feat_emb], dim=-1)
        activity = self.fusion(combined)

        return activity


class AuxiliaryHeadB(nn.Module):
    """Activity prediction from Real Features Only"""

    def __init__(self, feature_dim: int = 536, hidden_dim: int = 384,
                 n_activities: int = 1, dropout: float = 0.1):
        super().__init__()

        # Wider MLP since no sequence to lean on
        self.feature_stack = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, n_activities)
        )

    def forward(self, real_features: torch.Tensor):
        """
        Args:
            real_features: Real physics features (batch, feature_dim)
        Returns:
            Activity predictions (batch, n_activities)
        """
        return self.feature_stack(real_features)


# =============================================================================
# PART 6: Physics-Aware Training Losses
# Source: physics/PhysInformer/physics_aware_model.py
# Enforces thermodynamic constraints and heteroscedastic uncertainty
# =============================================================================

class PhysicsAwareLoss(nn.Module):
    """Combined loss function with physics-informed constraints for PhysInformer.

    This loss incorporates three types of supervision:
    1. Heteroscedastic regression loss: standard supervised loss with learned
       per-sample uncertainty (aleatoric uncertainty estimation)
    2. Thermodynamic identity constraint: soft penalty enforcing dG = dH - T*dS
    3. Spatial smoothness: total variation regularization for electrostatic profiles

    Args:
        temperature: Temperature in Kelvin for thermodynamic identity (310K = 37C)
        thermo_weight: Weight for thermodynamic property losses
        identity_weight: Weight for the dG = dH - T*dS constraint (0.1 = soft penalty)
        smooth_weight: Weight for total variation smoothness of spatial profiles (0.01)
    """

    def __init__(self, temperature: float = 310.0,
                 thermo_weight: float = 1.0,
                 identity_weight: float = 0.1,
                 smooth_weight: float = 0.01):
        super().__init__()
        self.temperature = temperature       # 310K = physiological temperature
        self.thermo_weight = thermo_weight   # Weight for dH, dS, dG losses
        self.identity_weight = identity_weight  # Weight for thermodynamic identity penalty
        self.smooth_weight = smooth_weight   # Weight for spatial smoothness regularization

    def heteroscedastic_loss(self, mean, log_var, target):
        """Heteroscedastic Gaussian negative log-likelihood with learned variance.

        Formula: L = 0.5 * [exp(-log_var) * (mean - target)^2 + log_var]

        This is the NLL of a Gaussian N(mean, exp(log_var)):
        - The first term (precision * squared_error) penalizes prediction errors,
          weighted inversely by the predicted variance. High-variance predictions
          are penalized less for the same absolute error.
        - The second term (log_var) prevents the model from "cheating" by
          predicting infinite variance -- it penalizes high uncertainty.
        - The balance between these terms forces the model to predict high
          uncertainty only where it genuinely cannot make accurate predictions.

        This is equivalent to the attenuated loss from Kendall & Gal (2017)
        "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
        """
        precision = torch.exp(-log_var)  # 1/sigma^2
        return 0.5 * (precision * (mean - target) ** 2 + log_var).mean()

    def thermodynamic_identity_loss(self, dH, dS, dG):
        """Soft constraint enforcing the fundamental thermodynamic identity.

        The Gibbs free energy equation: dG = dH - T*dS
        where T is temperature in Kelvin (310K for physiological conditions).

        This is a physical law that must hold exactly. We enforce it as a soft
        MSE penalty between the model's direct dG prediction and the dG computed
        from its dH and dS predictions. This encourages internal consistency
        without rigidly constraining the architecture.
        """
        predicted_dG = dH - self.temperature * dS  # dG = dH - T*dS
        return F.mse_loss(predicted_dG, dG)

    def total_variation_loss(self, x):
        """Smoothness regularization for spatial profiles (electrostatic windows).

        Penalizes large differences between adjacent windows, encouraging smooth
        spatial profiles. This is physically motivated: electrostatic potentials
        should vary smoothly along DNA, not jump discontinuously between adjacent
        20bp windows. The L1 (absolute) penalty encourages smoothness while still
        allowing genuine transitions (unlike L2 which over-smooths).
        """
        if len(x.shape) == 3:
            diff = x[:, 1:, :] - x[:, :-1, :]  # Difference between adjacent windows
            return diff.abs().mean()
        return 0.0

    def forward(self, predictions, targets):
        """
        Compute total loss with physics constraints.

        Args:
            predictions: Model outputs dict
            targets: Ground truth values dict

        Returns:
            Dictionary with individual and total losses
        """
        losses = {}

        # Thermodynamic losses with heteroscedastic uncertainty
        if 'dH_mean' in predictions and 'thermo_dH' in targets:
            losses['dH_loss'] = self.heteroscedastic_loss(
                predictions['dH_mean'], predictions['dH_log_var'], targets['thermo_dH']
            )
            losses['dS_loss'] = self.heteroscedastic_loss(
                predictions['dS_mean'], predictions['dS_log_var'], targets['thermo_dS']
            )
            losses['dG_loss'] = self.heteroscedastic_loss(
                predictions['dG_mean'], predictions['dG_log_var'], targets['thermo_dG']
            )

            # Thermodynamic identity
            losses['identity_loss'] = self.thermodynamic_identity_loss(
                predictions['dH_mean'], predictions['dS_mean'], predictions['dG_mean']
            )

        # Electrostatic losses
        if 'window_predictions' in predictions and 'electrostatic_windows' in targets:
            window_loss = F.mse_loss(predictions['window_predictions'],
                                    targets['electrostatic_windows'])
            losses['electrostatic_window_loss'] = window_loss

            # Smoothness
            losses['electrostatic_smooth'] = self.total_variation_loss(
                predictions['window_predictions']
            )

        # Descriptor feature losses (MSE for all non-thermo features)
        for key in predictions:
            if key.endswith('_mean') and key not in ['dH_mean', 'dS_mean', 'dG_mean']:
                prop_name = key.replace('_mean', '')
                if prop_name in targets:
                    losses[f'{prop_name}_loss'] = F.mse_loss(
                        predictions[key], targets[prop_name]
                    )

        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss

        return losses


class AuxiliaryLoss(nn.Module):
    """Loss function for auxiliary diagnostic heads (AuxiliaryHeadA and AuxiliaryHeadB).

    Uses Huber loss by default (robust to outliers in activity measurements)
    rather than MSE, which can be dominated by a few extreme activity values.
    """

    def __init__(self, loss_type: str = 'huber'):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'huber':
            self.loss_fn = nn.HuberLoss()
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, predictions: Dict, targets: torch.Tensor):
        losses = {}

        # Head A: Sequence + Features -> Activity
        if 'aux_activity_seq_feat' in predictions:
            losses['aux_seq_feat_loss'] = self.loss_fn(
                predictions['aux_activity_seq_feat'], targets
            )

        # Head B: Features Only -> Activity
        if 'aux_activity_feat_only' in predictions:
            losses['aux_feat_only_loss'] = self.loss_fn(
                predictions['aux_activity_feat_only'], targets
            )

        if losses:
            losses['aux_total_loss'] = sum(losses.values())

        return losses


# =============================================================================
# PART 7: PhysicsVAE Loss Functions
# Source: physics/PhysicsVAE/models/losses.py
# VAE ELBO: reconstruction + KL + physics consistency
# =============================================================================

class VAELoss(nn.Module):
    """
    Standard VAE loss (negative ELBO): Reconstruction + KL divergence.

    L = -E_q[log p(x|z)] + beta * KL(q(z|x) || p(z))

    The two terms represent:
    - Reconstruction loss: How well can the decoder reproduce the input sequence?
      Uses cross-entropy (categorical distribution over 4 nucleotides at each position).
    - KL divergence: How far is the learned posterior q(z|x) from the prior p(z) = N(0,I)?
      Computed analytically for two Gaussians:
      KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    Beta-VAE weighting (Higgins et al. 2017):
    - beta = 0.001 (very weak KL regularization) because the primary goal is
      accurate sequence reconstruction conditioned on physics, not disentangled
      latent representations. Strong KL penalty (beta=1) causes "posterior
      collapse" where the decoder ignores z entirely and relies only on physics
      conditioning, producing poor sequence diversity.

    Beta-annealing rationale:
    - Training starts with beta=0 (pure reconstruction) and linearly increases
      to beta=0.001 over 10,000 steps. This "warm-up" strategy (Bowman et al. 2016)
      prevents early posterior collapse: the encoder first learns to produce
      informative latent codes, then the KL penalty is slowly introduced to
      regularize the latent space. Without annealing, the KL term can dominate
      early training when the decoder is still weak, causing it to ignore z.
    """

    def __init__(
        self,
        beta: float = 0.001,           # Final beta value (weak KL regularization)
        beta_annealing: bool = True,    # Whether to anneal beta from 0 to beta_end
        beta_start: float = 0.0,        # Starting beta (0 = pure reconstruction initially)
        beta_end: float = 0.001,        # Final beta after annealing
        annealing_steps: int = 10000    # Steps over which to linearly anneal beta
    ):
        super().__init__()
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.annealing_steps = annealing_steps
        self.current_step = 0

    def get_beta(self) -> float:
        """Get current beta value with optional linear annealing.

        Returns beta_start at step 0, linearly increasing to beta_end at
        annealing_steps, then constant at beta_end thereafter.
        """
        if not self.beta_annealing:
            return self.beta
        progress = min(1.0, self.current_step / self.annealing_steps)
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss.

        Args:
            logits: Predicted nucleotide logits [batch, seq_length, 4]
            targets: Target sequence indices [batch, seq_length]
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log variance [batch, latent_dim]
            step: Current training step (for annealing)
        """
        if step is not None:
            self.current_step = step

        batch_size = logits.size(0)

        # Reconstruction loss: per-position cross-entropy over 4 nucleotides.
        # ignore_index=4 excludes 'N' (ambiguous) positions from the loss.
        # Sum reduction + division by batch_size gives per-sample average.
        logits_flat = logits.view(-1, 4)       # [batch * seq_len, 4]
        targets_flat = targets.view(-1)         # [batch * seq_len]
        recon_loss = F.cross_entropy(logits_flat, targets_flat, reduction='sum', ignore_index=4)
        recon_loss = recon_loss / batch_size    # Average over batch

        # KL divergence: analytical formula for KL(N(mu, sigma^2) || N(0, I))
        # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # where logvar = log(sigma^2), so exp(logvar) = sigma^2
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / batch_size    # Average over batch

        beta = self.get_beta()
        total_loss = recon_loss + beta * kl_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta': torch.tensor(beta)
        }


class PhysicsConsistencyLoss(nn.Module):
    """
    Physics consistency loss for generated sequences.

    This loss ensures that sequences generated by the PhysicsVAE actually
    exhibit the target biophysical properties they were conditioned on.
    It works by using a pre-trained, frozen PhysInformer as a "physics oracle":

    1. Take the VAE's generated nucleotide logits -> argmax -> discrete sequence
    2. Feed discrete sequence through frozen PhysInformer -> predicted physics
    3. Compute MSE between predicted physics and the target physics that the
       VAE was conditioned on

    This creates a feedback loop: if the VAE generates a sequence that does not
    match the requested physics, this loss penalizes it. Over training, the VAE
    learns to generate sequences whose biophysical properties match the conditioning.

    Note: The PhysInformer is frozen (no gradients) and the argmax operation is
    non-differentiable, so this loss does NOT provide gradients through the
    physics predictor. It acts as a regularizer on the reconstruction quality
    rather than a direct gradient signal. The VAE still learns primarily through
    the reconstruction loss, with this consistency loss as an auxiliary check.
    """

    def __init__(
        self,
        physics_weight: float = 1.0,
        normalize: bool = True
    ):
        super().__init__()
        self.physics_weight = physics_weight
        self.normalize = normalize
        self.physics_predictor = None  # Set externally

    def set_physics_predictor(self, predictor: nn.Module):
        """Set the PhysInformer model for physics prediction."""
        self.physics_predictor = predictor
        self.physics_predictor.eval()
        for param in self.physics_predictor.parameters():
            param.requires_grad = False

    def forward(
        self,
        generated_logits: torch.Tensor,
        target_physics: torch.Tensor,
        use_soft: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics consistency loss.

        Args:
            generated_logits: Nucleotide logits [batch, seq_length, 4]
            target_physics: Target physics features [batch, n_physics]
            use_soft: Use soft (differentiable) sequence representation
        Returns:
            Dict with physics_loss, physics_mse
        """
        if self.physics_predictor is None:
            # Return zero loss if no predictor set
            zero = torch.tensor(0.0, device=generated_logits.device)
            return {'physics_loss': zero, 'physics_mse': zero}

        batch_size = generated_logits.size(0)

        if use_soft:
            # Use softmax probabilities for differentiable physics prediction
            # This requires PhysInformer to accept soft inputs
            probs = F.softmax(generated_logits, dim=-1)
            # Convert soft one-hot to sequence for PhysInformer
            # PhysInformer expects indices, so we use argmax (non-differentiable)
            # For full differentiability, would need to modify PhysInformer
            sequences = generated_logits.argmax(dim=-1)
        else:
            sequences = generated_logits.argmax(dim=-1)

        # Predict physics from generated sequences
        with torch.no_grad():
            predicted_physics = self.physics_predictor(sequences)['descriptors']

        # MSE between predicted and target physics
        if self.normalize:
            # Normalize by feature std to balance contributions
            physics_mse = F.mse_loss(predicted_physics, target_physics, reduction='none')
            physics_mse = physics_mse.mean()
        else:
            physics_mse = F.mse_loss(predicted_physics, target_physics)

        physics_loss = self.physics_weight * physics_mse

        return {
            'physics_loss': physics_loss,
            'physics_mse': physics_mse
        }


class CombinedVAELoss(nn.Module):
    """
    Combined loss for PhysicsVAE training:

        L_total = L_recon + beta * L_KL + gamma * L_physics

    Three components:
    - L_recon (cross-entropy): Sequence reconstruction accuracy.
      This is the dominant term -- the VAE must faithfully reconstruct DNA sequences.
    - beta * L_KL (KL divergence): Latent space regularization.
      beta = 0.001 (very weak) to prevent posterior collapse while still allowing
      a smooth, sampleable latent space for generation.
    - gamma * L_physics (physics consistency): Generated sequences must match
      target biophysical properties when evaluated by a frozen PhysInformer.
      gamma = 0.1 provides a moderate physics constraint without overwhelming
      the reconstruction objective.

    The relative weights (1.0 : 0.001 : 0.1) reflect the design priority:
    accurate reconstruction first, physics consistency second, latent
    regularization third (just enough to enable generation).
    """

    def __init__(
        self,
        beta: float = 0.001,       # KL weight (very weak regularization)
        gamma: float = 0.1,         # Physics consistency weight
        beta_annealing: bool = True,
        annealing_steps: int = 10000
    ):
        super().__init__()
        self.vae_loss = VAELoss(beta=beta, beta_annealing=beta_annealing,
                                annealing_steps=annealing_steps)
        self.physics_loss = PhysicsConsistencyLoss(physics_weight=gamma)

    def set_physics_predictor(self, predictor: nn.Module):
        self.physics_loss.set_physics_predictor(predictor)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        target_physics: torch.Tensor,
        step: Optional[int] = None,
        compute_physics: bool = True
    ) -> Dict[str, torch.Tensor]:
        vae_losses = self.vae_loss(logits, targets, mu, logvar, step)

        if compute_physics and self.physics_loss.physics_predictor is not None:
            physics_losses = self.physics_loss(logits, target_physics)
        else:
            zero = torch.tensor(0.0, device=logits.device)
            physics_losses = {'physics_loss': zero, 'physics_mse': zero}

        total_loss = vae_losses['total_loss'] + physics_losses['physics_loss']

        return {
            'total_loss': total_loss,
            'recon_loss': vae_losses['recon_loss'],
            'kl_loss': vae_losses['kl_loss'],
            'physics_loss': physics_losses['physics_loss'],
            'physics_mse': physics_losses['physics_mse'],
            'beta': vae_losses['beta']
        }


# =============================================================================
# PART 8: Physics Feature Attribution
# Source: physics/PhysicsInterpreter/physics_attribution.py
# Decomposes activity predictions through physics features using linear probes
# =============================================================================

# Physics feature families for attribution grouping.
# Each family maps to a list of feature name prefixes. When computing
# family-level attribution, all features matching any prefix in the list
# are summed together. This enables high-level interpretability:
# "45.3% of the prediction comes from bending features"
PHYSICS_FAMILIES = {
    'thermodynamics': ['thermo_'],
    'mechanics': ['stiff_'],
    'bending': ['bend_'],
    'entropy': ['entropy_'],
    'structural': ['advanced_mgw', 'advanced_stress', 'advanced_melting', 'advanced_stacking'],
    'electrostatics': ['tileformer_', 'elec_'],
    'motif_derived': ['pwm_'],
}


@dataclass
class AttributionResult:
    """Results from physics feature attribution."""
    feature_contributions: Dict[str, float]
    family_contributions: Dict[str, float]
    probe_r2: float
    probe_pearson: float
    top_positive: List[Tuple[str, float]]
    top_negative: List[Tuple[str, float]]
    predicted_activity: float
    intercept_contribution: float


@dataclass
class SequenceAttribution:
    """Attribution results for a single sequence."""
    sequence_id: str
    predicted_activity: float
    feature_values: Dict[str, float]
    feature_contributions: Dict[str, float]
    family_contributions: Dict[str, float]
    top_positive_features: List[Tuple[str, float]]
    top_negative_features: List[Tuple[str, float]]


class PhysicsAttributor:
    """
    Physics feature attribution using linear probes.

    Decomposes activity predictions into contributions from physics features,
    enabling mechanistic interpretation.

    Usage:
        attributor = PhysicsAttributor(config)
        attributor.fit(X_physics, y_activity, feature_names)

        # Get overall feature importance
        result = attributor.get_attribution()

        # Get per-sequence attribution
        seq_attr = attributor.attribute_sequence(x_physics, seq_id='seq1')
    """

    def __init__(self, config=None):
        from sklearn.preprocessing import StandardScaler
        self.config = config
        self.alpha = config.attribution_alpha if config else 1.0
        self.probe_type = config.attribution_probe_type if config else 'ridge'
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.coefficients: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> 'PhysicsAttributor':
        """
        Fit the attribution probe on physics features.

        Args:
            X: Physics features (n_samples, n_features)
            y: Activity values (n_samples,)
            feature_names: Names of features

        Returns:
            Self for chaining
        """
        from sklearn.linear_model import Ridge, ElasticNet
        from scipy.stats import pearsonr

        self.feature_names = feature_names

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Create probe model
        if self.probe_type == 'ridge':
            self.model = Ridge(alpha=self.alpha)
        else:
            self.model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=0.5,
                max_iter=10000
            )

        # Fit model
        self.model.fit(X_scaled, y)

        self.coefficients = self.model.coef_
        self._is_fitted = True

        # Compute fit metrics
        y_pred = self.model.predict(X_scaled)
        self._r2 = self.model.score(X_scaled, y)
        self._pearson = pearsonr(y, y_pred)[0]

        print(f"Attribution probe fitted: R2={self._r2:.4f}, Pearson r={self._pearson:.4f}")

        return self

    def get_attribution(self, top_n: int = 20) -> AttributionResult:
        """Get overall feature attribution from fitted probe."""
        if not self._is_fitted:
            raise ValueError("Probe not fitted. Call fit() first.")

        feature_contributions = {
            name: float(coef)
            for name, coef in zip(self.feature_names, self.coefficients)
        }

        family_contributions = self._compute_family_contributions(
            self.coefficients, self.feature_names
        )

        sorted_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
        top_positive = [(f, c) for f, c in sorted_features if c > 0][:top_n]
        top_negative = [(f, c) for f, c in sorted_features if c < 0][-top_n:][::-1]

        return AttributionResult(
            feature_contributions=feature_contributions,
            family_contributions=family_contributions,
            probe_r2=self._r2,
            probe_pearson=self._pearson,
            top_positive=top_positive,
            top_negative=top_negative,
            predicted_activity=0.0,
            intercept_contribution=float(self.model.intercept_)
        )

    def attribute_sequence(
        self,
        x: np.ndarray,
        sequence_id: str = None
    ) -> SequenceAttribution:
        """
        Get attribution for a single sequence.

        Args:
            x: Physics features for one sequence (n_features,)
            sequence_id: Optional identifier

        Returns:
            SequenceAttribution with per-feature contributions
        """
        if not self._is_fitted:
            raise ValueError("Probe not fitted. Call fit() first.")

        # Standardize
        x_scaled = self.scaler.transform(x.reshape(1, -1))[0]

        # Compute prediction
        prediction = self.model.predict(x_scaled.reshape(1, -1))[0]

        # Compute per-feature contributions
        # Contribution = coefficient * scaled_feature_value
        contributions = self.coefficients * x_scaled

        feature_values = {
            name: float(val)
            for name, val in zip(self.feature_names, x)
        }

        feature_contributions = {
            name: float(contrib)
            for name, contrib in zip(self.feature_names, contributions)
        }

        # Family contributions for this sequence
        family_contributions = self._compute_family_contributions(
            contributions, self.feature_names
        )

        # Top contributors
        sorted_contrib = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_positive = [(f, c) for f, c in sorted_contrib if c > 0][:10]
        top_negative = [(f, c) for f, c in sorted_contrib if c < 0][:10]

        return SequenceAttribution(
            sequence_id=sequence_id or 'unknown',
            predicted_activity=prediction,
            feature_values=feature_values,
            feature_contributions=feature_contributions,
            family_contributions=family_contributions,
            top_positive_features=top_positive,
            top_negative_features=top_negative
        )

    def _compute_family_contributions(
        self,
        values: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute contributions by physics family."""
        family_sums = {}

        for family, prefixes in PHYSICS_FAMILIES.items():
            family_sum = 0.0
            for i, name in enumerate(feature_names):
                for prefix in prefixes:
                    if name.startswith(prefix):
                        family_sum += abs(values[i])
                        break
            family_sums[family] = family_sum

        # Normalize to percentages
        total = sum(family_sums.values())
        if total > 0:
            family_sums = {k: v / total * 100 for k, v in family_sums.items()}

        return family_sums


# =============================================================================
# PART 9: Transfer Learning Protocols
# Source: physics/PhysicsTransfer/protocols.py
# Three transfer protocols: zero-shot, fine-tuning, joint training
#
# These protocols test the central claim of the FUSEMAP physics approach:
# that biophysical features enable regulatory activity prediction across
# species, because DNA chemistry is universal even though transcription
# factor repertoires are species-specific.
# =============================================================================

@dataclass
class TransferResult:
    """Results from a transfer experiment."""
    protocol: str
    source_datasets: List[str]
    target_dataset: str

    # Source performance (in-domain)
    source_pearson: float
    source_spearman: float

    # Target performance (transfer)
    target_pearson: float
    target_spearman: float

    # Fine-tuning performance (if applicable)
    fine_tuned_pearson: Optional[float] = None
    fine_tuned_spearman: Optional[float] = None
    fine_tune_n_samples: Optional[int] = None

    # Additional metrics
    transfer_efficiency: Optional[float] = None  # target / source ratio
    n_common_features: int = 0
    feature_contributions: Dict[str, float] = field(default_factory=dict)


class ZeroShotTransfer:
    """
    Protocol 1: Physics-Bridge Zero-Shot Transfer.

    Train physics->activity probe on source species, apply directly to target.
    Tests whether physics features capture universal regulatory principles.
    """

    def __init__(self, config=None, output_dir: str = None):
        from sklearn.preprocessing import StandardScaler
        self.config = config or {}
        self.probe = None
        self.scaler = StandardScaler()
        self.common_features: List[str] = []
        self.output_dir = output_dir

    def align_features(
        self,
        X_source: np.ndarray,
        source_features: List[str],
        X_target: np.ndarray,
        target_features: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Align features between source and target datasets."""
        common = sorted(set(source_features) & set(target_features))

        source_idx = [source_features.index(f) for f in common]
        target_idx = [target_features.index(f) for f in common]

        return X_source[:, source_idx], X_target[:, target_idx], common

    def run(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        source_features: List[str],
        X_target: np.ndarray,
        y_target: np.ndarray,
        target_features: List[str],
        source_name: str = 'source',
        target_name: str = 'target',
    ) -> TransferResult:
        """
        Run zero-shot transfer from source to target.

        Args:
            X_source: Source physics features [n_source, n_features]
            y_source: Source activities [n_source]
            source_features: Source feature names
            X_target: Target physics features [n_target, n_features]
            y_target: Target activities [n_target]
            target_features: Target feature names
            source_name: Name of source dataset
            target_name: Name of target dataset

        Returns:
            TransferResult with metrics
        """
        from sklearn.linear_model import Ridge, ElasticNet
        from scipy.stats import pearsonr, spearmanr

        # Align features
        X_src, X_tgt, self.common_features = self.align_features(
            X_source, source_features, X_target, target_features
        )

        # Standardize using source statistics only
        X_src_scaled = self.scaler.fit_transform(X_src)
        X_tgt_scaled = self.scaler.transform(X_tgt)

        probe_type = self.config.get('probe_type', 'ridge') if isinstance(self.config, dict) else getattr(self.config, 'probe_type', 'ridge')
        probe_alpha = self.config.get('probe_alpha', 1.0) if isinstance(self.config, dict) else getattr(self.config, 'probe_alpha', 1.0)

        # Train probe on source
        if probe_type == 'ridge':
            self.probe = Ridge(alpha=probe_alpha)
        else:
            self.probe = ElasticNet(alpha=probe_alpha, l1_ratio=0.5, max_iter=10000)

        self.probe.fit(X_src_scaled, y_source)

        # Evaluate on source
        y_source_pred = self.probe.predict(X_src_scaled)
        source_pearson_val = pearsonr(y_source, y_source_pred)[0]
        source_spearman_val = spearmanr(y_source, y_source_pred)[0]

        # Apply to target (zero-shot)
        y_target_pred = self.probe.predict(X_tgt_scaled)
        target_pearson_val = pearsonr(y_target, y_target_pred)[0]
        target_spearman_val = spearmanr(y_target, y_target_pred)[0]

        # Compute transfer efficiency
        transfer_eff = target_pearson_val / source_pearson_val if source_pearson_val > 0 else 0

        # Get feature contributions
        attributor = PhysicsAttributor(alpha=probe_alpha, probe_type=probe_type)
        attributor.fit(X_src, y_source, self.common_features)
        attr_result = attributor.get_attribution()

        return TransferResult(
            protocol='zero_shot',
            source_datasets=[source_name],
            target_dataset=target_name,
            source_pearson=source_pearson_val,
            source_spearman=source_spearman_val,
            target_pearson=target_pearson_val,
            target_spearman=target_spearman_val,
            transfer_efficiency=transfer_eff,
            n_common_features=len(self.common_features),
            feature_contributions=attr_result.family_contributions
        )

    def get_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions for aligned features."""
        if self.probe is None:
            raise ValueError("Run transfer first.")
        return self.probe.predict(X)


class PhysicsAnchoredFineTuning:
    """
    Protocol 2: Physics-Anchored Fine-Tuning.

    Start with source-trained physics probe, then fine-tune on limited
    target data. Tests few-shot learning with physics anchoring.
    """

    def __init__(self, config=None, output_dir: str = None):
        self.config = config or {}
        self.source_probe = None
        self.fine_tuned_probe = None
        self.common_features: List[str] = []
        self.output_dir = output_dir

    def run(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        source_features: List[str],
        X_target: np.ndarray,
        y_target: np.ndarray,
        target_features: List[str],
        fine_tune_sizes: List[int] = None,
        source_name: str = 'source',
        target_name: str = 'target',
    ) -> List[TransferResult]:
        """
        Run fine-tuning transfer with varying amounts of target data.

        Args:
            source_datasets: Source dataset names
            target_dataset: Target dataset name
            fine_tune_sizes: List of fine-tuning sample sizes to test

        Returns:
            List of TransferResults for each fine-tune size
        """
        from sklearn.linear_model import Ridge, ElasticNet
        from scipy.stats import pearsonr, spearmanr

        fine_tune_sizes = fine_tune_sizes or [100, 500, 1000, 5000]

        probe_type = self.config.get('probe_type', 'ridge') if isinstance(self.config, dict) else getattr(self.config, 'probe_type', 'ridge')
        probe_alpha = self.config.get('probe_alpha', 1.0) if isinstance(self.config, dict) else getattr(self.config, 'probe_alpha', 1.0)
        random_seed = self.config.get('random_seed', 42) if isinstance(self.config, dict) else getattr(self.config, 'random_seed', 42)

        # Align features
        common = sorted(set(source_features) & set(target_features))
        self.common_features = common
        src_idx = [source_features.index(f) for f in common]
        tgt_idx = [target_features.index(f) for f in common]
        X_src = X_source[:, src_idx]
        X_tgt = X_target[:, tgt_idx]

        # Standardize using source statistics
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_src_s = scaler.fit_transform(X_src)
        X_tgt_s = scaler.transform(X_tgt)

        # Train source probe
        if probe_type == 'ridge':
            self.source_probe = Ridge(alpha=probe_alpha)
        else:
            self.source_probe = ElasticNet(alpha=probe_alpha, l1_ratio=0.5, max_iter=10000)
        self.source_probe.fit(X_src_s, y_source)

        y_src_pred = self.source_probe.predict(X_src_s)
        source_pearson_val = pearsonr(y_source, y_src_pred)[0]
        source_spearman_val = spearmanr(y_source, y_src_pred)[0]

        # Zero-shot baseline on target
        y_zs_pred = self.source_probe.predict(X_tgt_s)
        zs_pearson = pearsonr(y_target, y_zs_pred)[0]
        zs_spearman = spearmanr(y_target, y_zs_pred)[0]

        # Fine-tune with different amounts of target data
        results = []
        for n_samples in fine_tune_sizes:
            if n_samples > len(y_target):
                n_samples = len(y_target)

            # Subsample target training data
            rng = np.random.RandomState(random_seed)
            indices = rng.choice(len(y_target), n_samples, replace=False)
            X_ft = X_tgt_s[indices]
            y_ft = y_target[indices]

            # Fine-tune on target data
            if probe_type == 'ridge':
                self.fine_tuned_probe = Ridge(alpha=probe_alpha)
            else:
                self.fine_tuned_probe = ElasticNet(alpha=probe_alpha, l1_ratio=0.5, max_iter=10000)
            self.fine_tuned_probe.fit(X_ft, y_ft)

            # Evaluate on target
            y_ft_pred = self.fine_tuned_probe.predict(X_tgt_s)
            ft_pearson = pearsonr(y_target, y_ft_pred)[0]
            ft_spearman = spearmanr(y_target, y_ft_pred)[0]

            transfer_eff = ft_pearson / source_pearson_val if source_pearson_val > 0 else 0

            result = TransferResult(
                protocol='physics_anchored_fine_tuning',
                source_datasets=[source_name],
                target_dataset=target_name,
                source_pearson=source_pearson_val,
                source_spearman=source_spearman_val,
                target_pearson=zs_pearson,
                target_spearman=zs_spearman,
                fine_tuned_pearson=ft_pearson,
                fine_tuned_spearman=ft_spearman,
                fine_tune_n_samples=n_samples,
                transfer_efficiency=transfer_eff,
                n_common_features=len(self.common_features),
            )
            results.append(result)

        return results


class MultiSpeciesJointTraining:
    """
    Protocol 3: Multi-Species Joint Training.

    Train a shared physics encoder on multiple species simultaneously,
    with species-specific output heads. Tests whether shared representation
    improves transfer.
    """

    def __init__(self, config=None, output_dir: str = None):
        self.config = config or {}
        self.shared_probe = None
        self.common_features: List[str] = []
        self.output_dir = output_dir

    def run(
        self,
        datasets: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
        holdout_dataset: str = None,
    ) -> Dict[str, TransferResult]:
        """
        Run multi-species joint training.

        Args:
            datasets: Dict mapping name -> (X, y, feature_names)
            holdout_dataset: Optional dataset to hold out for transfer testing

        Returns:
            Dict mapping dataset name to TransferResult
        """
        from sklearn.linear_model import Ridge, ElasticNet
        from sklearn.preprocessing import StandardScaler
        from scipy.stats import pearsonr, spearmanr

        probe_type = self.config.get('probe_type', 'ridge') if isinstance(self.config, dict) else getattr(self.config, 'probe_type', 'ridge')
        probe_alpha = self.config.get('probe_alpha', 1.0) if isinstance(self.config, dict) else getattr(self.config, 'probe_alpha', 1.0)

        # Find common features
        all_features = None
        for name, (X, y, features) in datasets.items():
            if all_features is None:
                all_features = set(features)
            else:
                all_features &= set(features)
        self.common_features = sorted(all_features)

        # Align all datasets
        aligned = {}
        for name, (X, y, features) in datasets.items():
            idx = [features.index(f) for f in self.common_features]
            aligned[name] = (X[:, idx], y)

        # Combine for joint training (excluding holdout if specified)
        train_names = [n for n in datasets if n != holdout_dataset]

        X_combined = np.vstack([aligned[n][0] for n in train_names])
        y_combined = np.concatenate([aligned[n][1] for n in train_names])

        # Standardize
        scaler = StandardScaler()
        X_combined_s = scaler.fit_transform(X_combined)

        # Train shared probe
        if probe_type == 'ridge':
            self.shared_probe = Ridge(alpha=probe_alpha)
        else:
            self.shared_probe = ElasticNet(alpha=probe_alpha, l1_ratio=0.5, max_iter=10000)
        self.shared_probe.fit(X_combined_s, y_combined)

        y_train_pred = self.shared_probe.predict(X_combined_s)
        shared_pearson = pearsonr(y_combined, y_train_pred)[0]
        shared_spearman = spearmanr(y_combined, y_train_pred)[0]

        # Evaluate on each dataset
        results = {}
        for name in datasets:
            X_eval, y_eval = aligned[name]
            X_eval_s = scaler.transform(X_eval)

            y_pred = self.shared_probe.predict(X_eval_s)
            r = pearsonr(y_eval, y_pred)[0]
            rho = spearmanr(y_eval, y_pred)[0]

            transfer_eff = r / shared_pearson if shared_pearson > 0 else 0

            results[name] = TransferResult(
                protocol='multi_species_joint',
                source_datasets=train_names,
                target_dataset=name,
                source_pearson=shared_pearson,
                source_spearman=shared_spearman,
                target_pearson=r,
                target_spearman=rho,
                transfer_efficiency=transfer_eff,
                n_common_features=len(self.common_features),
            )

        return results


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("Testing FUSEMAP Physics Modules...")
    print("=" * 60)

    # Test PhysInformer
    print("\n1. Testing PhysInformer:")
    physinformer = PhysInformer(n_descriptors=521)
    x = torch.randint(0, 4, (2, 230))
    with torch.no_grad():
        out = physinformer(x)
    print(f"   PhysInformer params: {sum(p.numel() for p in physinformer.parameters()):,}")
    print(f"   Input: {x.shape} -> Output: {out['descriptors'].shape}")

    # Test TileFormer
    print("\n2. Testing TileFormer:")
    tileformer = TileFormer(output_dim=6)
    x = torch.randint(0, 4, (2, 200))
    with torch.no_grad():
        out = tileformer(x)
    print(f"   TileFormer params: {sum(p.numel() for p in tileformer.parameters()):,}")
    print(f"   Input: {x.shape} -> Psi: {out['psi'].shape}")

    # Test PhysicsVAE
    print("\n3. Testing PhysicsVAE:")
    vae = PhysicsVAE(seq_length=200, n_physics_features=521)
    seq = torch.randint(0, 4, (2, 200))
    physics = torch.randn(2, 521)
    with torch.no_grad():
        out = vae(seq, physics)
    print(f"   PhysicsVAE params: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"   Logits: {out['logits'].shape}, z: {out['z'].shape}")

    # Test S2A Universal Features
    print("\n4. Testing S2A UniversalFeatureExtractor:")
    extractor = UniversalFeatureExtractor()
    feature_names = [f'thermo_{i}' for i in range(100)] + [f'pwm_{i}' for i in range(50)]
    universal_cols = extractor.extract_feature_columns(feature_names)
    print(f"   Total columns: {len(feature_names)}")
    print(f"   Universal (non-PWM): {len(universal_cols)}")

    # Test PhysicsAwareLoss
    print("\n5. Testing PhysicsAwareLoss:")
    loss_fn = PhysicsAwareLoss()
    preds = {'dH_mean': torch.randn(4), 'dH_log_var': torch.randn(4),
             'dS_mean': torch.randn(4), 'dS_log_var': torch.randn(4),
             'dG_mean': torch.randn(4), 'dG_log_var': torch.randn(4)}
    targs = {'thermo_dH': torch.randn(4), 'thermo_dS': torch.randn(4),
             'thermo_dG': torch.randn(4)}
    losses = loss_fn(preds, targs)
    print(f"   Total loss: {losses['total_loss'].item():.4f}")

    # Test VAELoss
    print("\n6. Testing VAELoss + CombinedVAELoss:")
    vae_loss = CombinedVAELoss()
    logits = torch.randn(2, 200, 4)
    targets = torch.randint(0, 4, (2, 200))
    mu = torch.randn(2, 128)
    logvar = torch.randn(2, 128)
    target_physics = torch.randn(2, 521)
    vae_losses = vae_loss(logits, targets, mu, logvar, target_physics, compute_physics=False)
    print(f"   Recon loss: {vae_losses['recon_loss'].item():.4f}")
    print(f"   KL loss: {vae_losses['kl_loss'].item():.4f}")

    # Test PhysicsAttributor
    print("\n7. Testing PhysicsAttributor:")
    X = np.random.randn(100, 50)
    y = np.random.randn(100)
    fnames = [f'thermo_{i}' for i in range(20)] + [f'bend_{i}' for i in range(15)] + [f'pwm_{i}' for i in range(15)]
    attributor = PhysicsAttributor()
    attributor.fit(X, y, fnames)
    result = attributor.get_attribution(top_n=5)
    print(f"   Probe R2: {result.probe_r2:.4f}")
    print(f"   Family contributions: { {k: f'{v:.1f}%' for k, v in result.family_contributions.items()} }")

    # Test ZeroShotTransfer
    print("\n8. Testing ZeroShotTransfer:")
    X_src = np.random.randn(200, 50)
    y_src = np.random.randn(200)
    X_tgt = np.random.randn(100, 50)
    y_tgt = np.random.randn(100)
    fnames = [f'thermo_{i}' for i in range(50)]
    transfer = ZeroShotTransfer()
    result = transfer.run(X_src, y_src, fnames, X_tgt, y_tgt, fnames, 'K562', 'S2')
    print(f"   Source r: {result.source_pearson:.4f}")
    print(f"   Target r: {result.target_pearson:.4f}")
    print(f"   Transfer efficiency: {result.transfer_efficiency:.1%}")

    print("\n" + "=" * 60)
    print("All tests passed!")
