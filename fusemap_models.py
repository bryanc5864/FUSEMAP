#!/usr/bin/env python3
"""
================================================================================
FUSEMAP CORE MODELS - Representative Code File 1/3
================================================================================

These 3 representative files contain key excerpts from the FUSEMAP codebase.
They do not contain all FUSEMAP code, as the full implementation is too large
to include here. All code, trained models, and processed datasets are available
under the MIT license at:
    https://github.com/bryanc5864/FUSEMAP

This file contains the core model architectures for FUSEMAP:

MODULES INCLUDED:
1. CADENCE - Core Activity Prediction Model (LegNet-based)
   Source: models/CADENCE/cadence.py
   - SELayer: Squeeze-and-Excitation for channel attention (reduction=16)
   - EffBlock: EfficientNet-style inverted residual blocks (expand_ratio=4)
   - LocalBlock: Local convolution blocks
   - ResidualConcat: Skip connections with concatenation
   - MapperBlock: Channel mapping layers
   - LegNet: Base architecture (literal copy from HumanMPRALegNet)
   - RCEquivariantStem: Reverse-complement equivariant stem (k=15, c=256)
   - CADENCE: Full model with optional modules (1.45M params)
   - MultiHeadCADENCE: Multi-output variant for multi-dataset training

2. ClusterSpace - Dilated Convolutions for Long-Range Patterns
   Source: models/CADENCE/cadence.py (ClusterSpace section)
   - DilatedEffBlock: EfficientNet block with dilation
   - ClusterSpace: Stack of dilated blocks

3. GrammarLayer - Bidirectional GRU with FiLM Conditioning
   Source: models/CADENCE/cadence.py (GrammarLayer section)
   - LightweightGrammarLayer: Single BiGRU with FiLM modulation

4. PLACE - Post-hoc Uncertainty Quantification
   Source: models/place_uncertainty.py
   - PLACEUncertainty: Last-layer Laplace + local adaptive conformal prediction
   - Calibrated 90% coverage intervals, k=200 neighbors

5. LegatoV2 - Compact High-Performance Model
   Source: models/legatoV2/legato_v2.py
   - RCEMotifStem: RC-equivariant motif extraction
   - GDCStack: Grouped dilated separable convolutions
   - LegatoV2: Full architecture

KEY RESULTS (from paper, Table 3):
- K562: Pearson r = 0.809, Spearman rho = 0.759
- HepG2: Pearson r = 0.786, Spearman rho = 0.770
- WTC11: Pearson r = 0.698, Spearman rho = 0.591
- DeepSTARR Dev: Pearson r = 0.909, Hk: Pearson r = 0.920
- Maize: Pearson r = 0.796, Sorghum: r = 0.782
- Yeast: Pearson r = 0.958

================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# =============================================================================
# PART 1: CADENCE - CORE ACTIVITY PREDICTION MODEL
# Source: models/CADENCE/cadence.py
# =============================================================================
# =============================================================================
"""
CADENCE - Convolutional Activity preDiction for ENhancer Characterization and Evaluation

This is the core sequence-to-activity prediction module of FUSEMAP. The architecture
is based on LegNet (de Almeida et al.) with optional advanced modules for improved
performance across species.

ARCHITECTURE OVERVIEW:
- Input: One-hot encoded DNA sequence [batch, 4, seq_len]
- Stem: LocalBlock or RCEquivariantStem (4 -> 64 channels)
- Main: 4x [ResidualConcat(EffBlock) -> LocalBlock -> MaxPool2]
  - Channel progression: 64 -> 80 -> 96 -> 112 -> 128
- Optional: ClusterSpace (dilated convolutions)
- Optional: GrammarLayer (BiGRU with FiLM)
- Mapper: BN -> Conv1d (128 -> 256)
- Pool: AdaptiveAvgPool1d
- Head: Linear -> BN -> SiLU -> Linear (256 -> 1)

The code below is a LITERAL COPY from the original implementation to ensure
reproducibility of published results.

Source: models/CADENCE/cadence.py
"""


# -----------------------------------------------------------------------------
# Squeeze-and-Excitation Layer
# -----------------------------------------------------------------------------
# The SE layer provides channel-wise attention by computing a global descriptor
# of each channel and using it to reweight channel responses. This allows the
# network to model channel interdependencies and focus on informative features.

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation layer for channel attention.

    Implements the SE module from Hu et al. (2018) "Squeeze-and-Excitation Networks".
    The SE block adaptively recalibrates channel-wise feature responses by
    explicitly modeling interdependencies between channels.

    Architecture:
        1. Squeeze: Global average pooling across spatial dimension
        2. Excitation: Two FC layers with reduction ratio (bottleneck)
        3. Scale: Channel-wise multiplication with input

    Args:
        inp: Number of input channels
        reduction: Reduction ratio for bottleneck (default 4)

    Example:
        >>> se = SELayer(64, reduction=4)
        >>> x = torch.randn(2, 64, 100)  # [batch, channels, length]
        >>> out = se(x)  # [batch, 64, 100] - same shape, recalibrated
    """

    def __init__(self, inp: int, reduction: int = 4):
        super(SELayer, self).__init__()
        # Two-layer MLP for channel attention
        # inp -> inp/reduction -> inp
        self.fc = nn.Sequential(
            nn.Linear(inp, int(inp // reduction)),
            nn.SiLU(),  # Swish activation (smooth, non-monotonic)
            nn.Linear(int(inp // reduction), inp),
            nn.Sigmoid()  # Output attention weights in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply squeeze-and-excitation attention.

        Args:
            x: Input tensor [batch, channels, length]

        Returns:
            Recalibrated tensor [batch, channels, length]
        """
        b, c, _ = x.size()
        # Squeeze: global average pooling
        y = x.view(b, c, -1).mean(dim=2)  # [batch, channels]
        # Excitation: learn channel weights
        y = self.fc(y).view(b, c, 1)  # [batch, channels, 1]
        # Scale: multiply input by attention weights
        return x * y


# -----------------------------------------------------------------------------
# EfficientNet-style Inverted Residual Block
# -----------------------------------------------------------------------------
# The EffBlock is the core building block of LegNet, inspired by MobileNetV2
# and EfficientNet architectures. It uses depthwise separable convolutions
# for parameter efficiency and SE attention for channel recalibration.

class EffBlock(nn.Module):
    """
    EfficientNet-style inverted residual block for 1D sequences.

    Implements the MBConv (Mobile Inverted Bottleneck Convolution) from
    EfficientNet with SE attention. The "inverted" design expands channels
    before depthwise convolution (opposite of traditional bottleneck).

    Architecture:
        1. Expand: 1x1 conv to expand channels by resize_factor
        2. Depthwise: k×k depthwise conv (each channel independently)
        3. SE: Squeeze-and-Excitation attention
        4. Project: 1x1 conv to restore original channels

    This design captures local patterns efficiently while the SE module
    enables channel interdependency modeling.

    Args:
        in_ch: Number of input channels
        ks: Kernel size for depthwise convolution
        resize_factor: Channel expansion factor (typically 4)
        activation: Activation function class
        out_ch: Output channels (default: same as in_ch)
        se_reduction: SE reduction ratio (default: resize_factor)

    Example:
        >>> block = EffBlock(64, ks=9, resize_factor=4, activation=nn.SiLU)
        >>> x = torch.randn(2, 64, 230)
        >>> out = block(x)  # [batch, 64, 230]
    """

    def __init__(
        self,
        in_ch: int,
        ks: int,
        resize_factor: int,
        activation,
        out_ch: int = None,
        se_reduction: int = None
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.resize_factor = resize_factor
        self.se_reduction = resize_factor if se_reduction is None else se_reduction
        self.ks = ks
        self.inner_dim = self.in_ch * self.resize_factor  # Expanded dimension

        # Inverted residual block with SE attention
        self.block = nn.Sequential(
            # 1. Expand: 1x1 convolution to expand channels
            nn.Conv1d(
                in_channels=self.in_ch,
                out_channels=self.inner_dim,
                kernel_size=1,
                padding='same',
                bias=False  # No bias before BatchNorm
            ),
            nn.BatchNorm1d(self.inner_dim),
            activation(),

            # 2. Depthwise: k×k convolution with groups=channels
            # Each channel is convolved independently (very parameter efficient)
            nn.Conv1d(
                in_channels=self.inner_dim,
                out_channels=self.inner_dim,
                kernel_size=ks,
                groups=self.inner_dim,  # Depthwise: one filter per channel
                padding='same',
                bias=False
            ),
            nn.BatchNorm1d(self.inner_dim),
            activation(),

            # 3. SE attention: recalibrate channels based on global info
            SELayer(self.inner_dim, reduction=self.se_reduction),

            # 4. Project: 1x1 convolution to restore channels
            nn.Conv1d(
                in_channels=self.inner_dim,
                out_channels=self.in_ch,
                kernel_size=1,
                padding='same',
                bias=False
            ),
            nn.BatchNorm1d(self.in_ch),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through inverted residual block."""
        return self.block(x)


# -----------------------------------------------------------------------------
# Local Convolution Block
# -----------------------------------------------------------------------------
# A simple convolutional block used in the stem and after residual connections.
# This provides local feature extraction with BatchNorm and activation.

class LocalBlock(nn.Module):
    """
    Local convolution block for sequence feature extraction.

    Simple but effective block: Conv1d -> BatchNorm -> Activation.
    Used in the stem layer and after residual concatenations.

    Args:
        in_ch: Number of input channels
        ks: Kernel size
        activation: Activation function class
        out_ch: Output channels (default: same as in_ch)

    Example:
        >>> block = LocalBlock(4, ks=11, activation=nn.SiLU, out_ch=64)
        >>> x = torch.randn(2, 4, 230)  # One-hot DNA
        >>> out = block(x)  # [batch, 64, 230]
    """

    def __init__(
        self,
        in_ch: int,
        ks: int,
        activation,
        out_ch: int = None
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.ks = ks

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_ch,
                out_channels=self.out_ch,
                kernel_size=self.ks,
                padding='same',
                bias=False
            ),
            nn.BatchNorm1d(self.out_ch),
            activation()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through local convolution."""
        return self.block(x)


# -----------------------------------------------------------------------------
# Residual Concatenation
# -----------------------------------------------------------------------------
# Instead of additive skip connections, this concatenates the processed
# features with the original input. This preserves all information and
# lets the network learn to combine them.

class ResidualConcat(nn.Module):
    """
    Apply function and concatenate output with input.

    Unlike standard residual connections (addition), this module concatenates
    the output with the input along the channel dimension. This doubles the
    channels but preserves all information from both branches.

    This is followed by a LocalBlock to reduce channels back down.

    Args:
        fn: Module to apply to input

    Example:
        >>> fn = EffBlock(64, ks=9, resize_factor=4, activation=nn.SiLU)
        >>> residual = ResidualConcat(fn)
        >>> x = torch.randn(2, 64, 230)
        >>> out = residual(x)  # [batch, 128, 230] - doubled channels
    """

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply function and concatenate with input."""
        return torch.concat([self.fn(x, **kwargs), x], dim=1)


# -----------------------------------------------------------------------------
# Channel Mapping Block
# -----------------------------------------------------------------------------
# Used to project features to a different channel dimension, typically
# for the final feature mapping before the prediction head.

class MapperBlock(nn.Module):
    """
    Channel mapping block for feature projection.

    Projects features to a different channel dimension using 1x1 convolution.
    Includes BatchNorm for normalization.

    Args:
        in_features: Number of input channels
        out_features: Number of output channels
        activation: Activation function class (not used, kept for API)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation=nn.SiLU
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Conv1d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input to different channel dimension."""
        return self.block(x)


# -----------------------------------------------------------------------------
# LegNet Architecture
# -----------------------------------------------------------------------------
# This is a LITERAL COPY of the LegNet architecture from HumanMPRALegNet.
# It achieves state-of-the-art performance on MPRA prediction tasks.

class LegNet(nn.Module):
    """
    LegNet: Deep learning model for MPRA regulatory activity prediction.

    Architecture combines EfficientNet-style blocks with sequence-specific
    components for DNA regulatory element analysis. This is a LITERAL COPY
    from the original HumanMPRALegNet implementation.

    Architecture Overview:
        1. Stem: LocalBlock (4 -> stem_ch, kernel=11)
        2. Main: 4x [ResidualConcat(EffBlock) -> LocalBlock -> MaxPool2]
           - Channel progression: stem_ch -> 80 -> 96 -> 112 -> 128
        3. Mapper: BN -> Conv1d(128 -> 256)
        4. Pool: AdaptiveAvgPool1d(1)
        5. Head: Linear(256->256) -> BN -> SiLU -> Linear(256->1)

    Default hyperparameters are tuned for 230bp ENCODE4 sequences but
    work well for other lengths (110-250bp).

    Args:
        in_ch: Input channels (4 for DNA one-hot)
        stem_ch: Stem output channels (64)
        stem_ks: Stem kernel size (11)
        ef_ks: EffBlock kernel size (9)
        ef_block_sizes: Channel sizes per block [80, 96, 112, 128]
        pool_sizes: Pooling sizes per block [2, 2, 2, 2]
        resize_factor: EffBlock expansion factor (4)
        activation: Activation function class

    Example:
        >>> model = LegNet(**LEGNET_DEFAULTS)
        >>> x = torch.randn(32, 4, 230)
        >>> y = model(x)  # [32] - predicted activities
    """

    def __init__(
        self,
        in_ch: int,
        stem_ch: int,
        stem_ks: int,
        ef_ks: int,
        ef_block_sizes: List[int],
        pool_sizes: List[int],
        resize_factor: int,
        activation=nn.SiLU
    ):
        super().__init__()
        assert len(pool_sizes) == len(ef_block_sizes), \
            "pool_sizes and ef_block_sizes must have same length"

        self.in_ch = in_ch

        # Stem: initial feature extraction
        self.stem = LocalBlock(
            in_ch=in_ch,
            out_ch=stem_ch,
            ks=stem_ks,
            activation=activation
        )

        # Main blocks: progressive channel expansion with pooling
        blocks = []
        in_ch = stem_ch
        out_ch = stem_ch

        for pool_sz, out_ch in zip(pool_sizes, ef_block_sizes):
            blc = nn.Sequential(
                # Residual + EffBlock (doubles channels via concat)
                ResidualConcat(
                    EffBlock(
                        in_ch=in_ch,
                        out_ch=in_ch,
                        ks=ef_ks,
                        resize_factor=resize_factor,
                        activation=activation
                    )
                ),
                # LocalBlock to project concatenated channels
                LocalBlock(
                    in_ch=in_ch * 2,  # Doubled from concat
                    out_ch=out_ch,
                    ks=ef_ks,
                    activation=activation
                ),
                # Pooling (reduces sequence length by factor of pool_sz)
                nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity()
            )
            in_ch = out_ch
            blocks.append(blc)

        self.main = nn.Sequential(*blocks)

        # Mapper: project to head dimension
        self.mapper = MapperBlock(
            in_features=out_ch,
            out_features=out_ch * 2
        )

        # Head: final prediction layers
        self.head = nn.Sequential(
            nn.Linear(out_ch * 2, out_ch * 2),
            nn.BatchNorm1d(out_ch * 2),
            activation(),
            nn.Linear(out_ch * 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LegNet.

        Args:
            x: Input tensor [batch_size, 4, sequence_length]
               One-hot encoded DNA sequence

        Returns:
            Predicted activity scores [batch_size]
        """
        # Feature extraction
        x = self.stem(x)       # [batch, 64, L]
        x = self.main(x)       # [batch, 128, L/16]
        x = self.mapper(x)     # [batch, 256, L/16]

        # Global pooling
        x = F.adaptive_avg_pool1d(x, 1)  # [batch, 256, 1]
        x = x.squeeze(-1)      # [batch, 256]

        # Prediction
        x = self.head(x)       # [batch, 1]
        x = x.squeeze(-1)      # [batch]
        return x


# Default LegNet hyperparameters (from HumanMPRALegNet, de Almeida et al.)
# These achieve r=0.81 on K562, r=0.79 on HepG2, r=0.77 on WTC11
# Total parameters: ~1.45M (compact enough for MPRA-scale datasets ~100K sequences)
LEGNET_DEFAULTS = {
    'in_ch': 4,           # DNA one-hot encoding (A=0, C=1, G=2, T=3)
    'stem_ch': 64,        # Initial channels (captures ~64 motif detectors)
    'stem_ks': 11,        # Stem kernel (captures ~1 turn of DNA helix = 10.5bp)
    'ef_ks': 9,           # EffBlock kernel (slightly smaller for efficiency)
    'ef_block_sizes': [80, 96, 112, 128],  # Progressive channel widening
    'pool_sizes': [2, 2, 2, 2],            # 16x total downsampling (230bp -> ~14)
    'resize_factor': 4,   # EffBlock channel expansion (in_ch -> 4*in_ch -> in_ch)
}


def create_legnet(**kwargs) -> LegNet:
    """
    Create a LegNet model with default HumanMPRALegNet hyperparameters.

    Args:
        **kwargs: Override any default parameters

    Returns:
        LegNet model instance
    """
    config = LEGNET_DEFAULTS.copy()
    config.update(kwargs)
    return LegNet(**config)


# -----------------------------------------------------------------------------
# Reverse-Complement Equivariant Stem
# -----------------------------------------------------------------------------
# DNA is double-stranded, and the same regulatory function can be encoded on
# either strand. The RC-equivariant stem processes both orientations and
# separates strand-symmetric (same on both strands) from strand-asymmetric
# (different on each strand) features.

class RCEquivariantStem(nn.Module):
    """
    Reverse-complement equivariant stem for strand-aware feature extraction.

    DNA regulatory elements can function on either strand. This stem processes
    both forward and reverse-complement orientations simultaneously, outputting:
    - Symmetric channels: fwd + rev (strand-invariant patterns)
    - Asymmetric channels: fwd - rev (strand-specific patterns)

    This enables the network to learn both types of patterns while maintaining
    biological plausibility. The RC is computed via weight sharing (not data
    augmentation), making it deterministic and more efficient.

    Architecture:
        1. Forward conv: standard convolution
        2. RC conv: same weights, flipped and base-complemented
        3. Sum: fwd + rev (strand-symmetric)
        4. Diff: fwd - rev (strand-asymmetric)
        5. Concat -> BatchNorm -> Activation

    Args:
        out_channels: Total output channels (split evenly between sum/diff)
        kernel_size: Convolution kernel size
        in_channels: Input channels (4 for DNA)
        activation: Activation function class

    Example:
        >>> stem = RCEquivariantStem(out_channels=64, kernel_size=11)
        >>> x = torch.randn(2, 4, 230)
        >>> out = stem(x)  # [batch, 64, 230]
    """

    def __init__(
        self,
        out_channels: int = 64,
        kernel_size: int = 11,
        in_channels: int = 4,
        activation=nn.SiLU,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        # Half the filters for forward, other half for RC
        n_filters = out_channels // 2

        self.conv = nn.Conv1d(
            in_channels,
            n_filters,
            kernel_size,
            padding='same',
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation()
        self.out_channels = out_channels

    def _revcomp_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Create reverse complement of convolutional weight.

        For DNA with encoding [A, C, G, T] = channels [0, 1, 2, 3]:
        - A <-> T means swap channels 0 and 3
        - C <-> G means swap channels 1 and 2
        - Then reverse the kernel (5' to 3' becomes 3' to 5')
        """
        # Flip kernel spatially (reverse the sequence direction)
        flipped = torch.flip(weight, dims=[2])
        # Swap complementary bases: A<->T, C<->G
        swapped = flipped[:, [3, 2, 1, 0], :]
        return swapped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequence through RC-equivariant convolution.

        Args:
            x: One-hot DNA [batch, 4, length]

        Returns:
            RC-equivariant features [batch, out_channels, length]
        """
        # Forward pass with learned weights
        fwd = self.conv(x)

        # Reverse complement pass with transformed weights
        rc_weight = self._revcomp_weight(self.conv.weight)
        rev = F.conv1d(x, rc_weight, padding='same')

        # Strand-symmetric features (same pattern on both strands)
        sym = fwd + rev
        # Strand-asymmetric features (different on each strand)
        asym = fwd - rev

        # Concatenate and normalize
        out = torch.cat([sym, asym], dim=1)
        return self.activation(self.bn(out))


# -----------------------------------------------------------------------------
# CADENCE Configuration
# -----------------------------------------------------------------------------

@dataclass
class CADENCEConfig:
    """
    Configuration dataclass for CADENCE model.

    Default values match HumanLegNet exactly for reproducibility.
    Optional modules can be enabled for enhanced performance.

    Core Parameters (from LegNet):
        in_ch: Input channels (4 for DNA one-hot)
        stem_ch: Stem output channels
        stem_ks: Stem kernel size
        ef_ks: EffBlock kernel size
        ef_block_sizes: Channels per main block
        pool_sizes: Pooling per block
        resize_factor: EffBlock expansion

    Optional Modules:
        use_rc_stem: Use RC-equivariant stem instead of standard
        use_cluster_space: Add dilated convolutions for long-range
        use_grammar: Add BiGRU with FiLM for motif grammar
        use_micromotif: Add multi-scale motif density
        use_motif_correlator: Add pairwise motif interaction

    Output:
        num_outputs: Number of output values (1 for single task)
    """
    # LegNet core parameters (tuned on ENCODE4 230bp sequences)
    in_ch: int = 4              # DNA one-hot: A, C, G, T
    stem_ch: int = 64           # Initial feature channels
    stem_ks: int = 11           # ~1 helical turn (10.5bp), captures core motifs
    ef_ks: int = 9              # EffBlock kernel, similar reasoning
    ef_block_sizes: List[int] = field(default_factory=lambda: [80, 96, 112, 128])
    # Gradual channel increase (64->80->96->112->128) prevents information
    # bottleneck while keeping parameter count moderate (~1.45M total)
    pool_sizes: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    # 4x MaxPool2 reduces 230bp -> ~14 positions, compressing spatial dimension
    resize_factor: int = 4      # EffBlock expansion: 4x channels in bottleneck

    # Optional: RC-equivariant stem (replaces standard LocalBlock stem)
    # Recommended for organisms where strand orientation is unknown or mixed
    use_rc_stem: bool = False

    # Optional: ClusterSpace dilated convolutions for long-range dependencies
    # Adds ~250K params. Dilation pattern [1,2,4,1] gives exponentially
    # growing receptive field at the pooled resolution.
    use_cluster_space: bool = False
    cluster_dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 1])

    # Optional: GrammarLayer (BiGRU + FiLM) for motif ordering/spacing
    # Adds ~100K params. Captures sequential dependencies CNNs miss.
    use_grammar: bool = False
    grammar_hidden: int = 128   # GRU hidden dim (bidirectional -> 256 output)

    # Optional: MicroMotif for multi-scale motif density processing
    use_micromotif: bool = False
    micromotif_windows: List[int] = field(default_factory=lambda: [5, 11, 21])

    # Optional: Motif correlator for pairwise motif interaction modeling
    # Uses low-rank bilinear pooling to capture motif co-occurrence
    use_motif_correlator: bool = False
    correlator_factors: int = 32    # Number of bilinear factors
    correlator_rank: int = 8        # Rank of factorization

    # Output dimension (1 for single-task, 2 for DeepSTARR Dev+Hk)
    num_outputs: int = 1


# -----------------------------------------------------------------------------
# CADENCE Model
# -----------------------------------------------------------------------------

class CADENCE(nn.Module):
    """
    CADENCE: LegNet architecture with optional advanced modules.

    Core architecture is IDENTICAL to LegNet when no optional modules enabled:
    - Stem: LocalBlock(4 -> 64, ks=11)
    - Main: 4x [ResidualConcat(EffBlock) -> LocalBlock -> MaxPool2]
    - Mapper: BN -> Conv1d(128 -> 256)
    - Pool: AdaptiveAvgPool1d
    - Head: Linear(256->256) -> BN -> SiLU -> Linear(256->1)

    Optional modules (inserted after main, before mapper):
    - RC-equivariant stem: Strand-symmetric/asymmetric features
    - Cluster space: Dilated convolutions for long-range patterns
    - Grammar layer: Bidirectional GRU with FiLM conditioning
    - MicroMotif: Multi-scale motif density processing
    - Motif correlator: Low-rank bilinear pooling

    Args:
        config: CADENCEConfig with model parameters
        **kwargs: Alternative to config, passed to CADENCEConfig

    Example:
        >>> # Standard LegNet-equivalent
        >>> model = CADENCE()
        >>> x = torch.randn(32, 4, 230)
        >>> y = model(x)  # [32]

        >>> # With optional modules
        >>> config = CADENCEConfig(use_rc_stem=True, use_grammar=True)
        >>> model = CADENCE(config)
    """

    def __init__(self, config: CADENCEConfig = None, **kwargs):
        super().__init__()

        if config is None:
            config = CADENCEConfig(**kwargs)
        self.config = config

        activation = nn.SiLU

        # Stem: either standard LegNet or RC-equivariant
        if config.use_rc_stem:
            self.stem = RCEquivariantStem(
                out_channels=config.stem_ch,
                kernel_size=config.stem_ks,
                in_channels=config.in_ch,
                activation=activation,
            )
        else:
            # EXACT LegNet stem
            self.stem = LocalBlock(
                in_ch=config.in_ch,
                out_ch=config.stem_ch,
                ks=config.stem_ks,
                activation=activation
            )

        # Main blocks: EXACT LegNet architecture
        blocks = []
        in_ch = config.stem_ch
        out_ch = config.stem_ch

        for pool_sz, out_ch in zip(config.pool_sizes, config.ef_block_sizes):
            blc = nn.Sequential(
                ResidualConcat(
                    EffBlock(
                        in_ch=in_ch,
                        out_ch=in_ch,
                        ks=config.ef_ks,
                        resize_factor=config.resize_factor,
                        activation=activation
                    )
                ),
                LocalBlock(
                    in_ch=in_ch * 2,
                    out_ch=out_ch,
                    ks=config.ef_ks,
                    activation=activation
                ),
                nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity()
            )
            in_ch = out_ch
            blocks.append(blc)

        self.main = nn.Sequential(*blocks)

        final_ch = config.ef_block_sizes[-1]

        # Optional Cluster Space
        self.cluster_space = None
        if config.use_cluster_space:
            self.cluster_space = ClusterSpace(
                in_channels=final_ch,
                block_configs=[(final_ch, final_ch, d) for d in config.cluster_dilations],
            )

        # Optional Grammar Layer
        self.grammar = None
        if config.use_grammar:
            self.grammar = LightweightGrammarLayer(
                in_channels=final_ch,
                hidden=config.grammar_hidden,
            )

        # Optional MicroMotif (not shown - see full code)
        self.micromotif = None

        # Optional Motif Correlator (not shown - see full code)
        self.correlator = None

        # Mapper: EXACT LegNet
        self.mapper = MapperBlock(
            in_features=final_ch,
            out_features=final_ch * 2
        )

        # Head: EXACT LegNet (but with configurable num_outputs)
        head_dim = final_ch * 2
        self.head = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.BatchNorm1d(head_dim),
            activation(),
            nn.Linear(head_dim, config.num_outputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: matches LegNet exactly when no optional modules enabled.

        The pipeline is:
            stem -> main -> [optional plugins] -> mapper -> pool -> head

        Optional plugins are inserted between the main convolution stack and
        the mapper. They all preserve the channel dimension (final_ch=128)
        and spatial dimension, so they compose cleanly in any combination.
        The order matters: ClusterSpace expands receptive field first, then
        GrammarLayer captures sequential dependencies in the expanded context,
        then MicroMotif/Correlator add higher-order motif interactions.

        Args:
            x: One-hot encoded sequence [batch, 4, seq_len]

        Returns:
            Predicted activity [batch, num_outputs] or [batch] if num_outputs=1
        """
        x = self.stem(x)       # [batch, 4, L] -> [batch, stem_ch, L]
        x = self.main(x)       # [batch, stem_ch, L] -> [batch, 128, L/16]

        # Optional modules (inserted before mapper, all preserve shape)
        # Each module is designed to be independently toggleable:
        if self.cluster_space is not None:
            x = self.cluster_space(x)   # Dilated convs for long-range patterns
        if self.grammar is not None:
            x = self.grammar(x)         # BiGRU + FiLM for motif ordering
        if self.micromotif is not None:
            x = self.micromotif(x)      # Multi-scale motif density
        if self.correlator is not None:
            x = self.correlator(x)      # Pairwise motif interactions

        x = self.mapper(x)             # [batch, 128, L/16] -> [batch, 256, L/16]
        x = F.adaptive_avg_pool1d(x, 1)  # [batch, 256, 1] global avg pool
        x = x.squeeze(-1)              # [batch, 256]
        x = self.head(x)               # [batch, num_outputs]
        x = x.squeeze(-1)              # [batch] if num_outputs=1
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get features before head (for PLACE uncertainty or analysis).

        Returns the 256-dimensional representation before the prediction head.
        This is the "penultimate layer" feature vector used by:
        - PLACE: for Laplace approximation (epistemic) and KNN (aleatoric)
        - Visualization: for t-SNE/UMAP of learned sequence representations
        - Transfer learning: as a fixed feature extractor for new tasks
        """
        x = self.stem(x)
        x = self.main(x)

        if self.cluster_space is not None:
            x = self.cluster_space(x)
        if self.grammar is not None:
            x = self.grammar(x)
        if self.micromotif is not None:
            x = self.micromotif(x)
        if self.correlator is not None:
            x = self.correlator(x)

        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        return x

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count by component for analysis."""
        counts = {}
        counts['stem'] = sum(p.numel() for p in self.stem.parameters())
        counts['main'] = sum(p.numel() for p in self.main.parameters())

        if self.cluster_space is not None:
            counts['cluster_space'] = sum(p.numel() for p in self.cluster_space.parameters())
        if self.grammar is not None:
            counts['grammar'] = sum(p.numel() for p in self.grammar.parameters())
        if self.micromotif is not None:
            counts['micromotif'] = sum(p.numel() for p in self.micromotif.parameters())
        if self.correlator is not None:
            counts['correlator'] = sum(p.numel() for p in self.correlator.parameters())

        counts['mapper'] = sum(p.numel() for p in self.mapper.parameters())
        counts['head'] = sum(p.numel() for p in self.head.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


# -----------------------------------------------------------------------------
# Multi-Head CADENCE
# -----------------------------------------------------------------------------

class MultiHeadCADENCE(nn.Module):
    """
    CADENCE with multiple output heads for multi-dataset training.

    Shared backbone (LegNet), separate heads per dataset. This enables
    joint training across cell types while learning cell-type-specific
    prediction functions. The shared backbone learns universal sequence
    features (motifs, grammar), while each head learns a cell-type-specific
    mapping from features to activity.

    Training paradigm:
        During each training step, a batch from ONE dataset is sampled.
        The shared backbone processes the batch, and only the corresponding
        head is updated. Gradients from all heads flow back through the
        shared backbone, creating an implicit multi-task regularization
        that improves generalization (especially for small datasets).

    The head_names parameter in forward() controls which heads are
    evaluated. During training, only the active dataset's head is used.
    During evaluation, all heads can be run simultaneously.

    Args:
        config: CADENCEConfig for backbone
        head_configs: Dict mapping head name to num_outputs
                      e.g., {'K562': 1, 'HepG2': 1, 'DeepSTARR': 2}
                      DeepSTARR has 2 outputs (Dev and Hk enhancer activities)

    Example:
        >>> config = CADENCEConfig()
        >>> heads = {'K562': 1, 'HepG2': 1, 'DeepSTARR': 2}
        >>> model = MultiHeadCADENCE(config, heads)
        >>> x = torch.randn(32, 4, 230)
        >>> outputs = model(x, head_names=['K562'])
        >>> # outputs['K562'] shape: [32]
    """

    def __init__(
        self,
        config: CADENCEConfig,
        head_configs: Dict[str, int],  # {head_name: num_outputs}
    ):
        super().__init__()
        self.config = config
        self.head_to_outputs = head_configs

        activation = nn.SiLU

        # Stem
        if config.use_rc_stem:
            self.stem = RCEquivariantStem(
                out_channels=config.stem_ch,
                kernel_size=config.stem_ks,
                in_channels=config.in_ch,
                activation=activation,
            )
        else:
            self.stem = LocalBlock(
                in_ch=config.in_ch,
                out_ch=config.stem_ch,
                ks=config.stem_ks,
                activation=activation
            )

        # Main blocks: EXACT LegNet
        blocks = []
        in_ch = config.stem_ch
        out_ch = config.stem_ch

        for pool_sz, out_ch in zip(config.pool_sizes, config.ef_block_sizes):
            blc = nn.Sequential(
                ResidualConcat(
                    EffBlock(
                        in_ch=in_ch,
                        out_ch=in_ch,
                        ks=config.ef_ks,
                        resize_factor=config.resize_factor,
                        activation=activation
                    )
                ),
                LocalBlock(
                    in_ch=in_ch * 2,
                    out_ch=out_ch,
                    ks=config.ef_ks,
                    activation=activation
                ),
                nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity()
            )
            in_ch = out_ch
            blocks.append(blc)

        self.main = nn.Sequential(*blocks)

        final_ch = config.ef_block_sizes[-1]

        # Optional modules
        self.cluster_space = None
        if config.use_cluster_space:
            self.cluster_space = ClusterSpace(in_channels=final_ch)

        self.grammar = None
        if config.use_grammar:
            self.grammar = LightweightGrammarLayer(
                in_channels=final_ch,
                hidden=config.grammar_hidden
            )

        self.micromotif = None
        self.correlator = None

        # Mapper
        self.mapper = MapperBlock(
            in_features=final_ch,
            out_features=final_ch * 2
        )

        # Multiple heads: each head is an independent LegNet-style prediction
        # network (Linear -> BN -> SiLU -> Linear). Using nn.ModuleDict so
        # heads are registered as submodules and appear in state_dict with
        # their dataset name as key (e.g., "heads.K562.0.weight").
        head_dim = final_ch * 2
        self.heads = nn.ModuleDict()

        for head_name, num_outputs in head_configs.items():
            self.heads[head_name] = nn.Sequential(
                nn.Linear(head_dim, head_dim),
                nn.BatchNorm1d(head_dim),
                activation(),
                nn.Linear(head_dim, num_outputs)
            )

    def forward(
        self,
        x: torch.Tensor,
        head_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional head selection.

        Args:
            x: Input sequences [batch, 4, length]
            head_names: Which heads to use (default: all)

        Returns:
            Dict mapping head name to predictions
        """
        x = self.stem(x)
        x = self.main(x)

        if self.cluster_space is not None:
            x = self.cluster_space(x)
        if self.grammar is not None:
            x = self.grammar(x)
        if self.micromotif is not None:
            x = self.micromotif(x)
        if self.correlator is not None:
            x = self.correlator(x)

        x = self.mapper(x)
        features = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        if head_names is None:
            head_names = list(self.heads.keys())

        outputs = {}
        for name in head_names:
            if name in self.heads:
                out = self.heads[name](features)
                if out.shape[-1] == 1:
                    out = out.squeeze(-1)
                outputs[name] = out

        return outputs

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features before heads (for PLACE)."""
        x = self.stem(x)
        x = self.main(x)

        if self.cluster_space is not None:
            x = self.cluster_space(x)
        if self.grammar is not None:
            x = self.grammar(x)
        if self.micromotif is not None:
            x = self.micromotif(x)
        if self.correlator is not None:
            x = self.correlator(x)

        x = self.mapper(x)
        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)


# =============================================================================
# =============================================================================
# PART 2: CLUSTER SPACE - DILATED CONVOLUTIONS FOR LONG-RANGE PATTERNS
# Source: models/CADENCE/cluster_space.py
# =============================================================================
# =============================================================================
"""
ClusterSpace Module

LegNet's 4x MaxPool operations reduce a 230bp sequence to ~14 positions.
While this compression is necessary for the prediction head, it limits the
model's ability to capture patterns that span the full regulatory element.
ClusterSpace adds dilated convolutions to capture long-range dependencies
at this reduced resolution WITHOUT additional pooling (which would lose
too much spatial information at ~14 positions).

Why dilated convolutions?
- Standard convolutions with kernel k have receptive field = k positions
- Dilated convolutions with dilation d have receptive field = d*(k-1) + 1
- This gives exponentially growing receptive fields without pooling

The dilation pattern [1, 2, 4, 1] expands the effective receptive field:
- Dilation 1: kernel=7 -> 7bp effective (local motif refinement)
- Dilation 2: kernel=7 -> 13bp effective (~full pooled sequence)
- Dilation 4: kernel=7 -> 25bp effective (captures cross-position patterns)
- Dilation 1: kernel=7 -> 7bp effective (consolidation/cleanup)

At the pooled resolution, each position represents ~16bp of the original
sequence, so dilation 4 effectively spans ~400bp of original sequence,
enabling modeling of:
- Motif spacing constraints (e.g., GATA-to-Ebox distance)
- Distal regulatory interactions (activator-repressor interplay)
- Higher-order chromatin structure signals

~250K additional parameters.

Source: models/CADENCE/cluster_space.py
"""


class DilatedEffBlock(nn.Module):
    """
    Combines LegNet's EffBlock with dilated convolutions.

    Captures both local and long-range patterns efficiently using
    dilated depthwise separable convolutions with SE attention.

    Architecture:
        1. Expand: 1x1 conv (in_ch -> in_ch * expand)
        2. Dilated depthwise: k×k with dilation
        3. SE attention
        4. Project: 1x1 conv (hidden -> out_ch)
        5. Skip connection

    Args:
        in_ch: Input channels
        out_ch: Output channels
        expand: Channel expansion factor (default 4)
        kernel: Kernel size (default 7)
        dilation: Dilation rate (default 1)

    Example:
        >>> block = DilatedEffBlock(128, 128, dilation=4)
        >>> x = torch.randn(2, 128, 14)  # After LegNet pooling
        >>> out = block(x)  # [2, 128, 14]
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand: int = 4,
        kernel: int = 7,
        dilation: int = 1,
    ):
        super().__init__()
        hidden = in_ch * expand

        # Inverted bottleneck with dilation
        self.expand = nn.Conv1d(in_ch, hidden, 1)
        self.bn1 = nn.BatchNorm1d(hidden)

        # Dilated depthwise convolution
        # Padding ensures output size matches input
        padding = (kernel // 2) * dilation
        self.depthwise = nn.Conv1d(
            hidden, hidden, kernel,
            padding=padding,
            dilation=dilation,
            groups=hidden  # Depthwise: one filter per channel
        )
        self.bn2 = nn.BatchNorm1d(hidden)

        # Squeeze-and-Excitation (implemented with 1x1 convolutions instead
        # of Linear layers for convenience - functionally identical when the
        # spatial dimension is pooled to 1 by AdaptiveAvgPool1d)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),            # Squeeze: [B, C, L] -> [B, C, 1]
            nn.Conv1d(hidden, hidden // 4, 1),  # Reduce: C -> C/4
            nn.SiLU(),
            nn.Conv1d(hidden // 4, hidden, 1),  # Expand: C/4 -> C
            nn.Sigmoid()                        # Attention weights in [0, 1]
        )

        # Project back to output dimension
        self.project = nn.Conv1d(hidden, out_ch, 1)
        self.bn3 = nn.BatchNorm1d(out_ch)

        # Skip connection (1x1 conv if dimensions differ)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process through dilated inverted residual block.

        The pre-activation residual pattern (BN before activation in main path,
        skip added before final activation) ensures gradient flow even with
        deep stacks of dilated blocks.
        """
        skip = self.skip(x)  # Identity or 1x1 conv for channel matching

        # Main path: expand -> dilated depthwise -> SE -> project
        out = F.silu(self.bn1(self.expand(x)))      # 1x1 expand channels
        out = F.silu(self.bn2(self.depthwise(out)))  # Dilated depthwise conv
        out = out * self.se(out)                     # Channel recalibration
        out = self.bn3(self.project(out))            # 1x1 project back

        return F.silu(out + skip)  # Residual connection + activation


class ClusterSpace(nn.Module):
    """
    Stack of dilated EffBlocks for multi-scale receptive fields.

    NO additional pooling: operates on already-pooled features from LegNet.
    After LegNet's 4 MaxPool2 operations, 230bp -> ~14bp, so we preserve
    spatial dimension and use dilation for increased receptive field.

    Default configuration:
    - Block 1: dilation=1 (local, 7bp receptive field)
    - Block 2: dilation=2 (medium, 14bp receptive field)
    - Block 3: dilation=4 (long-range, 28bp receptive field)
    - Block 4: dilation=1 (consolidation)

    Args:
        in_channels: Number of input channels (default 128 from LegNet)
        block_configs: List of (in_ch, out_ch, dilation) tuples

    Example:
        >>> cluster = ClusterSpace(in_channels=128)
        >>> x = torch.randn(2, 128, 14)  # Pooled features
        >>> out = cluster(x)  # [2, 128, 14]
    """

    def __init__(
        self,
        in_channels: int = 128,
        block_configs: Optional[List[tuple]] = None,
    ):
        super().__init__()

        # Default: (in_ch, out_ch, dilation)
        # Maintain channel dimension, vary dilation for multi-scale
        if block_configs is None:
            block_configs = [
                (in_channels, in_channels, 1),  # Local
                (in_channels, in_channels, 2),  # Medium range
                (in_channels, in_channels, 4),  # Long range
                (in_channels, in_channels, 1),  # Consolidate
            ]

        self.blocks = nn.ModuleList([
            DilatedEffBlock(in_ch, out_ch, dilation=dil)
            for in_ch, out_ch, dil in block_configs
        ])

        self.out_channels = block_configs[-1][1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process through dilated block stack.

        Args:
            x: Features [batch, in_channels, L] (L is ~14 after LegNet)

        Returns:
            Processed features [batch, out_channels, L] (same spatial dim)
        """
        for block in self.blocks:
            x = block(x)
        return x


# =============================================================================
# =============================================================================
# PART 3: GRAMMAR LAYER - BIDIRECTIONAL GRU WITH FiLM CONDITIONING
# Source: models/CADENCE/grammar_layer.py
# =============================================================================
# =============================================================================
"""
LightweightGrammarLayer

Regulatory elements have "grammar": the ordering and spacing of motifs matters.
For example, an enhancer might require:
- A GATA motif followed by an E-box within 10-20bp
- No intervening repressor sites

CNNs capture local patterns well but struggle with long-range dependencies.
The GrammarLayer adds a bidirectional GRU to model sequential relationships
between detected motifs.

Key design choices:
1. Single bidirectional GRU (not stacked) - parameter efficient
2. FiLM conditioning - use global context to modulate local features
3. Gated combination - learn how to blend CNN and RNN features

FiLM (Feature-wise Linear Modulation):
- Extract global context from GRU final states
- Generate per-channel scale (gamma) and shift (beta)
- Apply: x_modulated = gamma * x + beta

~100K additional parameters.

Source: models/CADENCE/grammar_layer.py
"""


class LightweightGrammarLayer(nn.Module):
    """
    Lightweight grammar layer with bidirectional GRU and FiLM conditioning.

    Captures motif ordering and spacing patterns that CNNs miss.
    Uses FiLM (Feature-wise Linear Modulation) for efficient conditioning.

    Architecture:
        1. Bidirectional GRU: processes sequence both directions
        2. Global context: concatenate final forward/backward states
        3. FiLM: generate gamma, beta from global context
        4. Modulate: gamma * input + beta
        5. Project: GRU output -> input channels
        6. Gate: sigmoid(proj) controls blend of modulated and projected

    FiLM is initialized for near-identity (gamma=1, beta=0) so the layer
    starts as a skip connection and learns to contribute gradually.

    Args:
        in_channels: Number of input channels
        hidden: GRU hidden dimension (output is 2x for bidirectional)
        dropout: Dropout rate (default 0.1)

    Example:
        >>> grammar = LightweightGrammarLayer(128, hidden=128)
        >>> x = torch.randn(2, 128, 14)
        >>> out = grammar(x)  # [2, 128, 14]
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden = hidden

        # Single bidirectional GRU
        # Input: [batch, length, channels]
        # Output: [batch, length, 2*hidden]
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden,
            bidirectional=True,
            batch_first=True
        )

        # Output dimension from bidirectional GRU
        self.gru_out_dim = hidden * 2

        # FiLM: generate scale (gamma) and shift (beta) from global context
        self.film_gamma = nn.Linear(self.gru_out_dim, in_channels)
        self.film_beta = nn.Linear(self.gru_out_dim, in_channels)

        # Initialize FiLM for near-identity at start
        # gamma = 1 + 0*context, beta = 0 + 0*context
        # So initially: x_modulated = 1*x + 0 = x
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # Output projection to match input channels
        self.out_proj = nn.Conv1d(self.gru_out_dim, in_channels, 1)
        self.bn = nn.BatchNorm1d(in_channels)
        self.dropout = nn.Dropout(dropout)

        self.out_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process through grammar layer with FiLM conditioning.

        Args:
            x: CNN features [batch, in_channels, length]

        Returns:
            Grammar-aware features [batch, in_channels, length]
        """
        batch, channels, length = x.shape

        # Transpose for GRU: [batch, L, channels]
        x_t = x.transpose(1, 2)

        # Process through bidirectional GRU
        gru_out, h_n = self.gru(x_t)
        # gru_out: [batch, L, 2*hidden]
        # h_n: [2, batch, hidden] (forward and backward final states)

        # Global context from final hidden states
        h_fwd = h_n[0]  # [batch, hidden]
        h_bwd = h_n[1]  # [batch, hidden]
        global_context = torch.cat([h_fwd, h_bwd], dim=1)  # [batch, 2*hidden]

        # FiLM: generate per-channel scale and shift
        gamma = self.film_gamma(global_context)  # [batch, in_channels]
        beta = self.film_beta(global_context)    # [batch, in_channels]

        # Apply FiLM (Feature-wise Linear Modulation) to original CNN features.
        # FiLM allows the global sequence context (from GRU final states) to
        # modulate each channel of the local CNN features. This is a form of
        # conditional normalization: the network can amplify channels that are
        # globally relevant and suppress irrelevant ones.
        # Formula: x_modulated = gamma * x + beta (per-channel affine transform)
        # Due to initialization (gamma=1, beta=0), this starts as identity.
        gamma = gamma.unsqueeze(-1)  # [batch, in_channels, 1] - broadcast over length
        beta = beta.unsqueeze(-1)    # [batch, in_channels, 1]
        x_modulated = gamma * x + beta  # [batch, in_channels, length]

        # Project GRU output back to input channels
        gru_features = gru_out.transpose(1, 2)  # [batch, 2*hidden, L]
        gru_proj = self.out_proj(gru_features)  # [batch, in_channels, L]
        gru_proj = self.bn(gru_proj)
        gru_proj = self.dropout(gru_proj)

        # Gated combination: the sigmoid gate controls how much each source
        # contributes at each position and channel. When gate -> 1, the output
        # is dominated by FiLM-modulated CNN features (local patterns with
        # global context). When gate -> 0, the GRU's sequential features dominate.
        # This soft gating lets the network learn position-specific blending:
        # e.g., use CNN features where local motifs are strong, GRU features
        # where long-range dependencies matter.
        gate = torch.sigmoid(gru_proj)          # [batch, in_channels, L] in (0,1)
        out = x_modulated * gate + gru_proj * (1 - gate)

        return F.silu(out)


# Legacy aliases for backward compatibility with older config files and
# checkpoint state_dicts that reference these class names. All three
# names now point to the same LightweightGrammarLayer implementation.
GrammarCrossGate = LightweightGrammarLayer
GrammarCrossGateV2 = LightweightGrammarLayer
GrammarLayer = LightweightGrammarLayer


# =============================================================================
# =============================================================================
# PART 4: PLACE - POST-HOC UNCERTAINTY QUANTIFICATION
# Source: models/CADENCE/place_uncertainty.py
# =============================================================================
# =============================================================================
"""
PLACE: Post-hoc Laplace And Conformal Estimation

Neural networks typically output point predictions without uncertainty.
PLACE adds calibrated uncertainty estimates POST-HOC (after training)
without modifying the model or retraining.

Two types of uncertainty:
1. Epistemic (model uncertainty): What the model doesn't know
   - High for out-of-distribution inputs
   - Computed via last-layer Laplace approximation

2. Aleatoric (data uncertainty): Irreducible noise
   - High for inherently variable sequences
   - Computed via local adaptive conformal prediction

Key advantages:
- No retraining required
- Calibrated coverage (90% intervals contain 90% of true values)
- Distinguishes epistemic from aleatoric uncertainty
- Supports Mondrian splits for regime-specific calibration

Algorithm:
1. Fit Laplace approximation to last-layer posterior
2. Build KNN model on calibration features
3. For new predictions:
   a. Epistemic: quadratic form with posterior covariance
   b. Aleatoric: weighted quantile of neighbor residuals
   c. Prediction intervals from conformal quantiles

Source: models/CADENCE/place_uncertainty.py
"""


class PLACEUncertainty:
    """
    Post-hoc uncertainty quantification using PLACE method.

    Provides calibrated epistemic and aleatoric uncertainty estimates
    without modifying the trained model.

    Epistemic Uncertainty (model doesn't know):
        - Last-layer Laplace approximation
        - V_epi(x) = φ(x)^T Σ φ(x) where Σ is posterior covariance

    Aleatoric Uncertainty (data is noisy):
        - Local adaptive conformal prediction
        - Find K nearest neighbors in feature space
        - Weight residuals by distance
        - Compute weighted quantiles for intervals

    Args:
        feature_dim: Dimension of feature space (penultimate layer)
        lambda_reg: Ridge regularization for Laplace (prevents singular cov)
        n_neighbors: Number of neighbors for local conformal
        temperature: Temperature for neighbor distance weighting
        alpha: Significance level (0.1 for 90% coverage)
        min_neighbors: Minimum neighbors before global fallback
        use_mondrian: Whether to use regime-specific calibration
        device: Computation device

    Example:
        >>> place = PLACEUncertainty(feature_dim=256)
        >>> place.fit(model, calibration_loader)
        >>> results = place.predict_uncertainty(model, test_sequences)
        >>> print(f"Epistemic std: {results['epistemic_std']}")
        >>> print(f"90% interval: {results['intervals_90']}")
    """

    def __init__(
        self,
        feature_dim: int = 256,
        lambda_reg: float = 1e-3,
        n_neighbors: int = 200,
        temperature: float = 1.0,
        alpha: float = 0.1,  # For 90% coverage intervals
        min_neighbors: int = 50,
        use_mondrian: bool = False,
        device: str = 'cuda'
    ):
        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg      # Ridge regularization: prevents singular
                                          # posterior covariance. 1e-3 is standard.
        self.n_neighbors = n_neighbors    # K=200 balances locality vs stability:
                                          # too few -> noisy estimates, too many ->
                                          # over-smoothed (loses heteroscedasticity)
        self.temperature = temperature    # Controls Gaussian kernel bandwidth for
                                          # neighbor weighting. T=1.0 is default.
        self.alpha = alpha                # Significance level: alpha=0.1 -> 90% coverage
        self.min_neighbors = min_neighbors  # Fallback threshold for Mondrian splits
        self.use_mondrian = use_mondrian  # Mondrian: regime-specific calibration
                                          # (e.g., separate intervals for HK vs Dev)
        self.device = device

        # Calibration data storage (populated during fit())
        self.calibration_features = None    # Penultimate layer activations [N, d]
        self.calibration_residuals = None   # |y_true - y_pred| for each cal. sample
        self.calibration_regimes = None     # Optional regime labels for Mondrian splits

        # Laplace approximation components (computed during _fit_laplace())
        self.posterior_cov = None   # Sigma: [d, d] posterior covariance matrix
        self.noise_var = None       # sigma^2: scalar noise variance estimate

        # KNN model for local conformal prediction (fitted during _fit_conformal())
        self.knn_model = None       # sklearn NearestNeighbors on normalized features

        self.is_fitted = False

    def fit(
        self,
        model: nn.Module,
        calibration_loader,
        regime_labels: Optional[np.ndarray] = None
    ):
        """
        Fit PLACE on calibration data.

        Args:
            model: Trained model (must have method to get features)
            calibration_loader: DataLoader for calibration set
            regime_labels: Optional labels for Mondrian splits (e.g., HK vs Dev)
        """
        print("Fitting PLACE uncertainty estimator...")

        model.eval()

        # Collect features, predictions, and targets
        all_features = []
        all_predictions = []
        all_targets = []
        all_regimes = [] if regime_labels is not None else None

        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_loader):
                if isinstance(batch, dict):
                    sequences = batch['sequence'].to(self.device)
                    targets = batch['target'].to(self.device)
                else:
                    sequences, targets = batch
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)

                # Get features from penultimate layer
                features = self._extract_features(model, sequences)

                # Get predictions
                outputs = model(sequences)
                if isinstance(outputs, dict):
                    if 'predictions' in outputs:
                        predictions = outputs['predictions']
                    else:
                        first_head = list(outputs.keys())[0]
                        predictions = outputs[first_head]['mean']
                else:
                    predictions = outputs

                all_features.append(features.cpu())
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

                if regime_labels is not None:
                    batch_start = batch_idx * calibration_loader.batch_size
                    batch_end = batch_start + sequences.size(0)
                    all_regimes.append(regime_labels[batch_start:batch_end])

        # Concatenate all data
        self.calibration_features = torch.cat(all_features, dim=0)
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        if all_regimes:
            self.calibration_regimes = np.concatenate(all_regimes)

        # Step 1: Fit Laplace approximation for epistemic uncertainty
        self._fit_laplace(self.calibration_features, predictions, targets)

        # Step 2: Prepare conformal prediction for aleatoric uncertainty
        self._fit_conformal(predictions, targets)

        self.is_fitted = True
        print(f"PLACE fitted on {len(self.calibration_features)} calibration samples")

    def _extract_features(
        self,
        model: nn.Module,
        sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract features from penultimate layer using forward hook.

        This captures the activations just before the final linear layer,
        which represent the model's learned representation of the sequence.
        We use PyTorch forward hooks (register_forward_hook) to intercept
        the INPUT to the head/task_head module during the forward pass,
        rather than requiring the model to expose an explicit feature method.
        This makes PLACE compatible with any model architecture.
        """
        features = []

        def hook_fn(module, input, output):
            # Capture the input to the head layer (= penultimate features)
            features.append(input[0])

        # Register hook on final layer (capture input, not output).
        # Try known attribute names first, then fall back to finding
        # the last Linear layer in the module tree.
        if hasattr(model, 'task_head'):
            handle = model.task_head.register_forward_hook(hook_fn)
        elif hasattr(model, 'head'):
            handle = model.head.register_forward_hook(hook_fn)
        else:
            # Find last linear layer as fallback
            last_linear = None
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
            handle = last_linear.register_forward_hook(hook_fn)

        # Forward pass to trigger hook
        with torch.no_grad():
            _ = model(sequences)

        # Remove hook
        handle.remove()

        return features[0] if features else None

    def _fit_laplace(
        self,
        features: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Fit last-layer Laplace approximation for epistemic uncertainty.

        The Laplace approximation treats the last layer as Bayesian linear
        regression with a Gaussian posterior over weights.

        Posterior covariance: Σ = (λI + (1/σ²)Φ^T Φ)^{-1}
        where:
            λ = regularization (prior precision)
            σ² = noise variance (estimated from residuals)
            Φ = feature matrix [N, d]
        """
        N, d = features.shape

        # Estimate noise variance using robust MAD (Median Absolute Deviation)
        # estimator. MAD is more robust to outliers than standard deviation,
        # which is important because calibration data may contain outlier sequences.
        residuals = targets - predictions
        mad = torch.median(torch.abs(residuals - torch.median(residuals)))
        # The constant 1.4826 converts MAD to standard deviation for a Gaussian:
        #   For X ~ N(mu, sigma), MAD = 0.6745 * sigma, so sigma = MAD / 0.6745
        #   1/0.6745 = 1.4826. Squaring gives variance: sigma^2 = (1.4826 * MAD)^2
        self.noise_var = (1.4826 * mad) ** 2  # Convert MAD to variance

        # Compute posterior covariance for Bayesian linear regression on last layer.
        # Under a Gaussian likelihood and Gaussian prior, the posterior over the
        # last-layer weights w is:
        #   p(w | D) = N(w_MAP, Sigma)
        #   Sigma^{-1} = lambda * I + (1/sigma^2) * Phi^T Phi
        # where Phi is the feature matrix, sigma^2 is noise variance,
        # and lambda is the prior precision (ridge regularization).
        Phi = features  # [N, d] - penultimate layer activations

        # Precision matrix (inverse covariance): (1/sigma^2) * Phi^T Phi
        # This is the Fisher information matrix of the last layer.
        gram = torch.matmul(Phi.T, Phi) / self.noise_var  # [d, d]

        # Add prior precision (regularization) to ensure invertibility
        # and encode prior belief that weights are small
        gram_reg = gram + self.lambda_reg * torch.eye(d)

        # Invert to get posterior covariance. Cholesky is preferred because
        # gram_reg is symmetric positive-definite, making Cholesky O(d^3/3)
        # and numerically stable.
        try:
            L = torch.linalg.cholesky(gram_reg)
            self.posterior_cov = torch.cholesky_inverse(L)
        except:
            # Fallback to pseudo-inverse if numerical issues prevent
            # Cholesky (e.g., near-singular due to collinear features)
            print("Warning: Using pseudo-inverse for posterior covariance")
            self.posterior_cov = torch.linalg.pinv(gram_reg)

        print(f"Laplace approximation fitted: noise_var={self.noise_var:.4f}")

    def _fit_conformal(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Prepare conformal prediction components for aleatoric uncertainty.

        Stores calibration residuals and builds KNN model for finding
        similar sequences during prediction.
        """
        # Ensure 1D tensors for residual computation
        preds_1d = predictions.squeeze()
        targets_1d = targets.squeeze()

        # Store absolute residuals for conformal quantiles
        self.calibration_residuals = torch.abs(targets_1d - preds_1d)
        assert self.calibration_residuals.dim() == 1, \
            f"Residuals should be 1D, got shape {self.calibration_residuals.shape}"

        # L2-normalize features so cosine distance = 1 - dot product.
        # Cosine distance is preferred over Euclidean because neural network
        # feature magnitudes can vary widely, and we care about the direction
        # of the feature vector (what patterns are present) rather than the
        # magnitude (how strongly they are activated).
        self.calibration_features_norm = F.normalize(
            self.calibration_features, p=2, dim=1
        )

        # Fit KNN model for finding similar calibration samples.
        # At prediction time, we find the K closest calibration sequences
        # and use their residuals to estimate local noise level.
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self.calibration_features)),
            metric='cosine'  # Cosine distance in normalized feature space
        )
        self.knn_model.fit(self.calibration_features_norm.numpy())

        print(f"Conformal calibration: {len(self.calibration_residuals)} residuals")

    def predict_uncertainty(
        self,
        model: nn.Module,
        sequences: torch.Tensor,
        regime: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty estimates.

        Args:
            model: Trained model
            sequences: Input sequences [batch_size, 4, length]
            regime: Optional regime indicator for Mondrian calibration

        Returns:
            Dictionary with:
                - predictions: Point predictions
                - epistemic_std: Model uncertainty (high = unfamiliar input)
                - aleatoric_std: Data uncertainty (high = inherently noisy)
                - total_std: Combined uncertainty
                - intervals_90: 90% prediction intervals
                - epistemic_var: Variance version
                - aleatoric_var: Variance version
        """
        if not self.is_fitted:
            raise ValueError("PLACE must be fitted before prediction")

        model.eval()

        with torch.no_grad():
            # Get point predictions
            outputs = model(sequences)
            if isinstance(outputs, dict):
                if 'predictions' in outputs:
                    predictions = outputs['predictions']
                else:
                    first_head = list(outputs.keys())[0]
                    predictions = outputs[first_head]['mean']
            else:
                predictions = outputs

            # Extract features
            features = self._extract_features(model, sequences)

            # Compute epistemic uncertainty via last-layer Laplace approximation.
            # Returns variance: V_epi(x) = phi(x)^T Sigma phi(x)
            epistemic_var = self._compute_epistemic(features)

            # Compute aleatoric uncertainty via local adaptive conformal prediction.
            # Returns std dev (from 84th percentile of neighbor residuals)
            # and 90% conformal prediction intervals.
            aleatoric_std, intervals = self._compute_aleatoric(
                features, predictions, regime
            )

            # Total uncertainty: independent sources add in variance space.
            # sigma_total = sqrt(sigma^2_aleatoric + sigma^2_epistemic)
            # This follows from Var[Y] = Var_w[E[Y|w]] + E_w[Var[Y|w]]
            # (law of total variance), where the first term is epistemic
            # and the second is aleatoric.
            total_std = torch.sqrt(aleatoric_std**2 + epistemic_var)

        return {
            'predictions': predictions,
            'epistemic_std': torch.sqrt(epistemic_var),
            'aleatoric_std': aleatoric_std,
            'total_std': total_std,
            'intervals_90': intervals,
            'epistemic_var': epistemic_var,
            'aleatoric_var': aleatoric_std**2
        }

    def _compute_epistemic(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute epistemic variance using Laplace approximation.

        V_epi(x) = phi(x)^T * Sigma * phi(x)

        where Sigma is the posterior covariance matrix over last-layer weights
        and phi(x) is the feature vector (penultimate layer activations).

        Intuition: This quadratic form measures how uncertain the model's
        weight posterior is in the direction of phi(x). For inputs similar
        to training data, phi(x) aligns with well-constrained directions
        (small variance). For out-of-distribution inputs, phi(x) points
        into poorly constrained directions (large variance).

        This is the predictive variance from Bayesian linear regression,
        which decomposes as: Var[y|x] = sigma^2 + phi(x)^T Sigma phi(x)
        where sigma^2 is noise (aleatoric) and the second term is epistemic.
        """
        batch_size = features.size(0)
        epistemic_vars = []

        for i in range(batch_size):
            phi = features[i:i+1].T  # [d, 1] - column vector
            # Quadratic form: phi^T Sigma phi gives scalar variance
            var = torch.matmul(
                torch.matmul(phi.T, self.posterior_cov),
                phi
            )
            epistemic_vars.append(var.squeeze())

        return torch.stack(epistemic_vars)

    def _compute_aleatoric(
        self,
        features: torch.Tensor,
        predictions: torch.Tensor,
        regime: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute aleatoric uncertainty using local adaptive conformal prediction.

        Standard conformal prediction uses a single global quantile of residuals.
        Local adaptive conformal prediction instead uses NEARBY calibration points,
        adapting the interval width to the local noise level. This captures
        heteroscedastic noise (e.g., higher variance in promoter vs. enhancer regions).

        Algorithm for each test point x:
        1. Find K=200 nearest neighbors in feature space (cosine distance)
        2. Weight neighbors by Gaussian kernel: w_i = exp(-d_i^2 / temperature)
           (closer neighbors contribute more to the local noise estimate)
        3. Compute weighted quantiles of their absolute residuals |y_i - f(x_i)|
        4. 84th percentile approximates the standard deviation
           (for a half-normal distribution, P(|Z| < sigma) = 0.84)
        5. (1 - alpha/2) quantile gives the conformal prediction interval half-width

        If Mondrian splits are enabled, neighbors are filtered by regime (e.g.,
        housekeeping vs developmental enhancers) for regime-specific calibration.
        """
        batch_size = features.size(0)
        aleatoric_stds = []
        intervals = []

        # Normalize features and convert to numpy
        features_norm = F.normalize(features.cpu(), p=2, dim=1).numpy()

        # Get calibration residuals as numpy
        calib_residuals_np = self.calibration_residuals.numpy()

        for i in range(batch_size):
            # Find K nearest neighbors
            distances, neighbor_indices = self.knn_model.kneighbors(
                features_norm[i:i+1],
                n_neighbors=self.n_neighbors
            )

            # Flatten to 1D arrays
            neighbor_indices = neighbor_indices.flatten()
            distances = distances.flatten()

            # Handle Mondrian (regime-specific calibration) if enabled
            if self.use_mondrian and regime is not None and self.calibration_regimes is not None:
                regime_mask = self.calibration_regimes[neighbor_indices] == regime
                neighbor_indices = neighbor_indices[regime_mask]
                distances = distances[regime_mask]

                # Fall back to global if too few regime-specific neighbors
                if len(neighbor_indices) < self.min_neighbors:
                    neighbor_indices = np.where(self.calibration_regimes == regime)[0]
                    if len(neighbor_indices) < self.min_neighbors:
                        neighbor_indices = np.arange(len(calib_residuals_np))
                    distances = np.ones(len(neighbor_indices))

            # Compute distance-based weights using Gaussian kernel.
            # w_i = exp(-d_i^2 / T) where T is temperature.
            # Small T -> sharp weighting (only very close neighbors matter)
            # Large T -> uniform weighting (approaches global conformal)
            # Default T=1.0 provides moderate locality.
            weights = np.exp(-distances**2 / self.temperature)
            weights = weights / (weights.sum() + 1e-8)  # Normalize to sum to 1

            # Get residuals for neighbors
            neighbor_residuals = calib_residuals_np[neighbor_indices]

            # Compute weighted quantiles of neighbor residuals
            # q_high: the (1-alpha/2) quantile, used for conformal intervals
            #   For alpha=0.1, this is the 95th percentile of |residuals|
            q_high = self._weighted_quantile(
                neighbor_residuals, weights, 1 - self.alpha/2
            )
            # q_84: the 84th percentile, used as standard deviation estimate
            #   For |Z| where Z ~ N(0, sigma), P(|Z| < sigma) ≈ 0.8413
            #   So the 84th percentile of |residuals| ≈ sigma (aleatoric std)
            q_84 = self._weighted_quantile(
                neighbor_residuals, weights, 0.84
            )

            aleatoric_stds.append(float(q_84))

            # Conformal prediction interval: [pred - q_high, pred + q_high]
            # This interval has guaranteed (1-alpha) coverage asymptotically
            # when the local neighborhood assumption holds.
            pred = predictions[i].item()
            intervals.append([pred - q_high, pred + q_high])

        aleatoric_stds = torch.tensor(aleatoric_stds, device=features.device)
        intervals = torch.tensor(intervals, device=features.device)

        return aleatoric_stds, intervals

    def _weighted_quantile(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        quantile: float
    ) -> float:
        """
        Compute weighted quantile using the cumulative distribution approach.

        This generalizes np.quantile to weighted samples. The weighted CDF is:
            F(v) = sum(w_i for v_i <= v) / sum(w_i)
        and the weighted quantile Q(p) is the smallest v such that F(v) >= p.

        Args:
            values: Sample values (e.g., neighbor residuals)
            weights: Non-negative weights (e.g., distance-based)
            quantile: Desired quantile in [0, 1]

        Returns:
            The weighted quantile value
        """
        values = np.asarray(values).flatten()
        weights = np.asarray(weights).flatten()

        # Sort values and corresponding weights together
        sort_idx = np.argsort(values)
        sorted_values = values[sort_idx]
        sorted_weights = weights[sort_idx]

        # Build weighted CDF: cumulative sum of normalized weights
        cum_weights = np.cumsum(sorted_weights)
        cum_weights = cum_weights / (cum_weights[-1] + 1e-8)  # Normalize to [0, 1]

        # Find the smallest index where the CDF exceeds the desired quantile
        idx = np.searchsorted(cum_weights, quantile)
        idx = min(idx, len(sorted_values) - 1)

        return float(sorted_values[idx])

    def calibrate_total_uncertainty(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epistemic_vars: torch.Tensor,
        aleatoric_vars: torch.Tensor,
        coverage_target: float = 0.9
    ) -> float:
        """
        Calibrate kappa for combining epistemic and aleatoric uncertainties.

        The total uncertainty combines both sources:
            sigma_total = sqrt(sigma^2_alea + kappa * sigma^2_epi)

        The mixing coefficient kappa controls how much epistemic uncertainty
        contributes to the total. We find kappa by grid search over
        log-spaced values [0.01, 10] such that the resulting prediction
        intervals achieve the desired coverage on held-out calibration data.

        The prediction interval is: [pred - z * sigma_total, pred + z * sigma_total]
        where z = Phi^{-1}((1 + coverage) / 2) is the Gaussian quantile.

        Returns:
            Optimal kappa value that achieves closest to target coverage
        """
        # Search over log-spaced kappa values from 0.01 to 10
        # Log spacing is used because kappa can vary over orders of magnitude
        kappa_values = np.logspace(-2, 1, 50)
        best_kappa = 1.0
        best_diff = float('inf')

        for kappa in kappa_values:
            # Total std with this kappa: sigma = sqrt(sigma^2_alea + kappa * sigma^2_epi)
            total_std = torch.sqrt(aleatoric_vars + kappa * epistemic_vars)

            # Gaussian z-score for desired coverage level:
            # For 90% coverage, z = Phi^{-1}(0.95) = 1.645
            z_score = stats.norm.ppf((1 + coverage_target) / 2)

            # Compute prediction interval and empirical coverage
            lower = predictions - z_score * total_std
            upper = predictions + z_score * total_std
            coverage = ((targets >= lower) & (targets <= upper)).float().mean().item()

            # Track kappa that gives coverage closest to target
            diff = abs(coverage - coverage_target)
            if diff < best_diff:
                best_diff = diff
                best_kappa = kappa

        print(f"Calibrated κ={best_kappa:.3f} for {coverage_target*100:.0f}% coverage")
        return best_kappa


# =============================================================================
# =============================================================================
# PART 5: RC-EQUIVARIANT MOTIF STEM (LegatoV2)
# Source: models/legatoV2/rce_stem.py
# NOTE: The full LegatoV2 model (including GDCStack - Grouped Dilated
# separable Convolutions) is in models/legatoV2/legato_v2.py but is not
# excerpted here. Only the RCEMotifStem component is included. The GDCStack
# uses grouped depthwise separable convolutions with exponentially increasing
# dilation rates (1, 2, 4, 8, ...) to build multi-scale receptive fields
# while keeping parameter counts low via channel grouping.
# =============================================================================
# =============================================================================
"""
RCEMotifStem - RC-Equivariant Motif Stem (from LegatoV2)

Alternative to the simple RCEquivariantStem with multi-scale kernels.
Uses multiple kernel sizes to capture motifs at different scales:
- Small (3, 5): Core binding sites
- Medium (7, 11): TF binding sites
- Large (15, 19): Complex regulatory elements

The RC weight sharing ensures that the same motif on either strand
produces related (sum/diff) outputs.

Source: models/legatoV2/rce_stem.py
"""


def reverse_complement_filter(weight: torch.Tensor) -> torch.Tensor:
    """
    Create reverse complement of a convolutional filter.

    For DNA with encoding [A, C, G, T] = channels [0, 1, 2, 3]:
    - A <-> T means swap channels 0 and 3
    - C <-> G means swap channels 1 and 2
    - Then reverse the kernel (5' to 3' flip)

    Args:
        weight: Convolutional filter [out_channels, in_channels, kernel_size]
               in_channels should be 4 (A, C, G, T) or 5 (+ reverse flag)

    Returns:
        RC version with base complementation and kernel reversal
    """
    # Clone to avoid in-place operations
    rc_weight = weight.clone()
    in_channels = weight.shape[1]

    if in_channels == 4:
        # Standard DNA: Swap A<->T (0<->3) and C<->G (1<->2)
        rc_weight[:, [0, 1, 2, 3], :] = rc_weight[:, [3, 2, 1, 0], :]
    elif in_channels == 5:
        # DNA + reverse flag: Swap DNA channels, keep flag unchanged
        rc_weight[:, [0, 1, 2, 3], :] = rc_weight[:, [3, 2, 1, 0], :]
    else:
        raise ValueError(f"Unsupported number of input channels: {in_channels}")

    # Reverse the kernel dimension (5' to 3' flip)
    rc_weight = rc_weight.flip(dims=[2])

    return rc_weight


class RCEquivariantConv(nn.Module):
    """
    RC-equivariant convolution that outputs (sum, diff) channels.

    Processes sequence with shared forward/RC weights and outputs:
    - Sum: fwd + rc (strand-invariant)
    - Diff: fwd - rc (strand-specific)

    Args:
        kernel_size: Convolution kernel size
        num_filters: Number of filters per direction
    """

    def __init__(self, kernel_size: int, num_filters: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters

        # Only store forward filters, RC computed via weight sharing
        self.weight = nn.Parameter(
            torch.randn(num_filters, 4, kernel_size) * 0.01
        )

        # Separate biases for sum and diff
        self.bias_sum = nn.Parameter(torch.zeros(num_filters))
        self.bias_diff = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RC-equivariant convolution.

        Args:
            x: One-hot DNA [batch, 4, length]

        Returns:
            (sum_features, diff_features) each [batch, num_filters, length]
        """
        # Forward convolution
        padding = (self.kernel_size - 1) // 2
        conv_fwd = F.conv1d(x, self.weight, padding=padding)

        # RC convolution (tied weights)
        rc_weight = reverse_complement_filter(self.weight)
        conv_rc = F.conv1d(x, rc_weight, padding=padding)

        # Sum (RC-invariant) and Diff (RC-anti-invariant)
        conv_sum = conv_fwd + conv_rc
        conv_diff = conv_fwd - conv_rc

        # Add biases
        conv_sum = conv_sum + self.bias_sum.view(1, -1, 1)
        conv_diff = conv_diff + self.bias_diff.view(1, -1, 1)

        return conv_sum, conv_diff


class RCEMotifStem(nn.Module):
    """
    RC-Equivariant Motif Stem with multi-scale convolutions.

    Uses multiple kernel sizes to capture motifs at different scales,
    all with RC weight sharing. Output is concatenation of sum/diff
    channels across all kernel sizes.

    Default kernel sizes: [7, 11, 15] capture:
    - 7bp: Core binding sites (E-boxes, TATA)
    - 11bp: Standard TF motifs (GATA, HNF4A)
    - 15bp: Extended motifs with context

    Args:
        num_filters: Filters per kernel size (default 48)
        kernel_sizes: List of kernel sizes (default [7, 11, 15])
        dropout: Dropout rate
        num_groups: Groups for GroupNorm

    Example:
        >>> stem = RCEMotifStem(num_filters=48, kernel_sizes=[7, 11, 15])
        >>> x = torch.randn(2, 4, 230)
        >>> out = stem(x)  # [2, 288, 230] = 2 * 3 * 48
    """

    def __init__(
        self,
        num_filters: int = 48,
        kernel_sizes: List[int] = None,
        dropout: float = 0.05,
        num_groups: int = 8
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [7, 11, 15]

        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes

        # Create RC-equivariant convolutions for each kernel size
        self.rc_convs = nn.ModuleList([
            RCEquivariantConv(k, num_filters)
            for k in kernel_sizes
        ])

        # Output channels: 2 (sum/diff) × num_kernels × num_filters
        self.out_channels = 2 * len(kernel_sizes) * num_filters

        # Normalization and activation
        self.norm = nn.GroupNorm(num_groups, self.out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process through multi-scale RC-equivariant convolutions.

        Args:
            x: One-hot DNA [batch, 4, length]

        Returns:
            Multi-scale RC-aware features [batch, out_channels, length]
        """
        features = []

        # Apply each kernel size
        for rc_conv in self.rc_convs:
            sum_feat, diff_feat = rc_conv(x)
            features.append(sum_feat)
            features.append(diff_feat)

        # Concatenate all features
        x = torch.cat(features, dim=1)  # [batch, 2*K*F, length]

        # Apply normalization, activation, dropout
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class RCEMotifStemV2(nn.Module):
    """
    Alternative RCEMotifStem with learnable sum/diff projections.

    Like RCEMotifStem but adds 1x1 convolutions after sum/diff
    for additional channel mixing within each kernel scale.

    Args:
        num_filters: Filters per kernel size
        kernel_sizes: List of kernel sizes
        dropout: Dropout rate
        num_groups: Groups for GroupNorm
        init_gain: Initialization gain for stability
        in_channels: Input channels (4 or 5)
    """

    def __init__(
        self,
        num_filters: int = 48,
        kernel_sizes: List[int] = None,
        dropout: float = 0.05,
        num_groups: int = 8,
        init_gain: float = 0.1,
        in_channels: int = 4
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [7, 11, 15]

        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = 2 * len(kernel_sizes) * num_filters

        # Store base filters for each kernel size
        self.base_filters = nn.ParameterList()

        for k in kernel_sizes:
            # Xavier-like initialization scaled by init_gain for training stability.
            # The factor 1/sqrt(in_channels * k) normalizes by fan-in, and
            # init_gain=0.1 further reduces initial magnitudes so the sum/diff
            # operations don't produce large activations at initialization.
            weight = torch.randn(num_filters, in_channels, k) * init_gain / (in_channels * k) ** 0.5
            self.base_filters.append(nn.Parameter(weight))

        # Learnable 1x1 projections for channel mixing within each kernel scale.
        # Using groups=len(kernel_sizes) means each kernel size's filters are
        # mixed independently - this prevents cross-scale interference while
        # allowing within-scale channel interaction. Cross-scale mixing happens
        # later in the downstream network.
        self.sum_proj = nn.Conv1d(
            len(kernel_sizes) * num_filters,
            len(kernel_sizes) * num_filters,
            kernel_size=1,
            groups=len(kernel_sizes)  # Grouped: no cross-kernel mixing
        )

        self.diff_proj = nn.Conv1d(
            len(kernel_sizes) * num_filters,
            len(kernel_sizes) * num_filters,
            kernel_size=1,
            groups=len(kernel_sizes)  # Grouped: no cross-kernel mixing
        )

        # Post-processing
        self.norm = nn.GroupNorm(num_groups, self.out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process sequence through RC-equivariant convolutions."""
        sum_features = []
        diff_features = []

        for weight, kernel_size in zip(self.base_filters, self.kernel_sizes):
            padding = (kernel_size - 1) // 2

            # Forward conv
            conv_fwd = F.conv1d(x, weight, padding=padding)

            # RC conv (tied weights)
            rc_weight = reverse_complement_filter(weight)
            conv_rc = F.conv1d(x, rc_weight, padding=padding)

            # Collect sum and diff
            sum_features.append(conv_fwd + conv_rc)
            diff_features.append(conv_fwd - conv_rc)

        # Concatenate and project
        sum_concat = torch.cat(sum_features, dim=1)
        diff_concat = torch.cat(diff_features, dim=1)

        sum_out = self.sum_proj(sum_concat)
        diff_out = self.diff_proj(diff_concat)

        # Interleave sum and diff channels for better downstream mixing.
        # Without interleaving, all sum channels come first, then all diff.
        # Interleaving places each kernel's sum/diff channels adjacent, so
        # downstream grouped convolutions can jointly process related features.
        # Layout: [k1_sum, k1_diff, k2_sum, k2_diff, k3_sum, k3_diff] per filter
        batch_size, _, length = sum_out.shape
        num_kernels = len(self.kernel_sizes)

        # Reshape to separate kernel groups: [batch, kernels, filters, length]
        sum_out = sum_out.view(batch_size, num_kernels, self.num_filters, length)
        diff_out = diff_out.view(batch_size, num_kernels, self.num_filters, length)

        # Stack sum/diff as a new dimension and flatten:
        # [batch, kernels, 2, filters, length] -> [batch, 2*kernels*filters, length]
        x = torch.stack([sum_out, diff_out], dim=2)
        x = x.reshape(batch_size, -1, length)

        # Apply normalization and activation
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


# =============================================================================
# =============================================================================
# FACTORY FUNCTIONS
# Source: models/CADENCE/cadence.py (factory section)
# =============================================================================
# =============================================================================

def create_cadence(
    num_outputs: int = 1,
    use_rc_stem: bool = False,
    use_cluster_space: bool = False,
    use_grammar: bool = False,
    use_micromotif: bool = False,
    use_motif_correlator: bool = False,
    **kwargs
) -> CADENCE:
    """
    Create CADENCE model with specified options.

    Args:
        num_outputs: Number of prediction outputs
        use_rc_stem: Use RC-equivariant stem
        use_cluster_space: Add dilated convolutions
        use_grammar: Add BiGRU with FiLM
        use_micromotif: Add multi-scale motif density
        use_motif_correlator: Add pairwise motif interaction
        **kwargs: Additional CADENCEConfig parameters

    Returns:
        Configured CADENCE model
    """
    config = CADENCEConfig(
        num_outputs=num_outputs,
        use_rc_stem=use_rc_stem,
        use_cluster_space=use_cluster_space,
        use_grammar=use_grammar,
        use_micromotif=use_micromotif,
        use_motif_correlator=use_motif_correlator,
        **kwargs
    )
    return CADENCE(config)


def create_cadence_for_dataset(dataset: str, **kwargs) -> CADENCE:
    """
    Create CADENCE configured for a specific dataset.

    Args:
        dataset: Dataset name (encode4_k562, deepstarr, etc.)
        **kwargs: Override parameters

    Returns:
        CADENCE model configured for dataset
    """
    dataset_configs = {
        'encode4_k562': {'num_outputs': 1},
        'encode4_hepg2': {'num_outputs': 1},
        'encode4_wtc11': {'num_outputs': 1},
        'deepstarr': {'num_outputs': 2},  # Dev and Hk
        'dream_yeast': {'num_outputs': 1},
        'jores_arabidopsis': {'num_outputs': 1},
        'jores_maize': {'num_outputs': 1},
        'jores_sorghum': {'num_outputs': 1},
    }

    base_config = dataset_configs.get(dataset.lower(), {'num_outputs': 1})
    base_config.update(kwargs)
    return create_cadence(**base_config)


# =============================================================================
# =============================================================================
# TEST CODE
# =============================================================================
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FUSEMAP Core Models - Test Suite")
    print("=" * 70)
    print("\nThis file is part of the FUSEMAP representative code collection.")
    print("Full code available at: https://github.com/bryanc5864/FUSEMAP")
    print("=" * 70)

    # Test 1: LegNet
    print("\n1. Testing pure LegNet:")
    legnet = create_legnet()
    x = torch.randn(2, 4, 230)
    with torch.no_grad():
        out = legnet(x)
    print(f"   Parameters: {sum(p.numel() for p in legnet.parameters()):,}")
    print(f"   Input: {x.shape} -> Output: {out.shape}")

    # Test 2: CADENCE (basic)
    print("\n2. Testing CADENCE (no optional modules):")
    cadence = create_cadence(num_outputs=1)
    with torch.no_grad():
        out = cadence(x)
    print(f"   Parameters: {sum(p.numel() for p in cadence.parameters()):,}")
    print(f"   Input: {x.shape} -> Output: {out.shape}")

    # Test 3: CADENCE with all modules
    print("\n3. Testing CADENCE (all optional modules):")
    cadence_full = create_cadence(
        num_outputs=1,
        use_rc_stem=True,
        use_cluster_space=True,
        use_grammar=True,
    )
    counts = cadence_full.get_parameter_count()
    print(f"   Component breakdown:")
    for k, v in counts.items():
        print(f"     {k}: {v:,}")

    with torch.no_grad():
        out = cadence_full(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")

    # Test 4: Multi-head CADENCE
    print("\n4. Testing MultiHeadCADENCE:")
    config = CADENCEConfig()
    heads = {'K562': 1, 'HepG2': 1, 'DeepSTARR': 2}
    multi_head = MultiHeadCADENCE(config, heads)
    with torch.no_grad():
        outputs = multi_head(x)
    print(f"   Heads: {list(outputs.keys())}")
    for name, out in outputs.items():
        print(f"     {name}: {out.shape}")

    # Test 5: RC-equivariant stem
    print("\n5. Testing RCEMotifStemV2:")
    stem = RCEMotifStemV2(num_filters=48, kernel_sizes=[7, 11, 15])
    with torch.no_grad():
        out = stem(x)
    print(f"   Parameters: {sum(p.numel() for p in stem.parameters()):,}")
    print(f"   Input: {x.shape} -> Output: {out.shape}")

    # Test 6: ClusterSpace
    print("\n6. Testing ClusterSpace:")
    cluster = ClusterSpace(in_channels=128)
    x_pooled = torch.randn(2, 128, 14)
    with torch.no_grad():
        out = cluster(x_pooled)
    print(f"   Parameters: {sum(p.numel() for p in cluster.parameters()):,}")
    print(f"   Input: {x_pooled.shape} -> Output: {out.shape}")

    # Test 7: GrammarLayer
    print("\n7. Testing LightweightGrammarLayer:")
    grammar = LightweightGrammarLayer(in_channels=128, hidden=128)
    with torch.no_grad():
        out = grammar(x_pooled)
    print(f"   Parameters: {sum(p.numel() for p in grammar.parameters()):,}")
    print(f"   Input: {x_pooled.shape} -> Output: {out.shape}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
