"""
FUSEMAP Models - Uses CADENCE (LegNet-based) backbone

This module imports the CADENCE backbone from models/CADENCE/cadence.py
and wraps it for multi-species training with conditioning embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .config import ModelConfig, DATASET_CATALOG

# Import LegNet building blocks from CADENCE module
from models.CADENCE.cadence import (
    SELayer,
    EffBlock,
    LocalBlock,
    ResidualConcat,
    MapperBlock,
    LegNet,
    CADENCE,
    CADENCEConfig,
    RCEquivariantStem,  # For RC-equivariant stem
    create_cadence,
    create_legnet,
    LEGNET_DEFAULTS,
)

# Import optional CADENCE modules
from models.CADENCE.cluster_space import ClusterSpace
from models.CADENCE.grammar_layer import LightweightGrammarLayer
from models.CADENCE.micro_motif import MicroMotifProcessor
from models.CADENCE.motif_correlator import LowRankMotifCorrelator
from models.CADENCE.pwm_stem import PWMMultiScaleStem
from models.CADENCE.place_uncertainty import PLACEUncertainty


# ============================================================================
# Weight Initialization (matching LegNet)
# ============================================================================

def initialize_weights(m):
    """LegNet weight initialization - critical for preventing mean collapse."""
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# ============================================================================
# Multi-Output LegNet (for datasets with multiple outputs like DeepSTARR)
# ============================================================================

class MultiOutputLegNet(nn.Module):
    """
    LegNet with multiple output heads.
    Shared backbone, separate heads per output.
    """

    def __init__(
        self,
        in_ch: int = 4,
        stem_ch: int = 64,
        stem_ks: int = 11,
        ef_ks: int = 9,
        ef_block_sizes: List[int] = None,
        pool_sizes: List[int] = None,
        resize_factor: int = 4,
        num_outputs: int = 1,
        activation=nn.SiLU,
    ):
        super().__init__()

        if ef_block_sizes is None:
            ef_block_sizes = [80, 96, 112, 128]
        if pool_sizes is None:
            pool_sizes = [2, 2, 2, 2]

        # Shared backbone using LegNet blocks
        self.stem = LocalBlock(in_ch=in_ch, out_ch=stem_ch, ks=stem_ks, activation=activation)

        blocks = []
        in_ch_block = stem_ch
        for pool_sz, out_ch in zip(pool_sizes, ef_block_sizes):
            blc = nn.Sequential(
                ResidualConcat(
                    EffBlock(in_ch=in_ch_block, ks=ef_ks, resize_factor=resize_factor, activation=activation)
                ),
                LocalBlock(in_ch=in_ch_block * 2, out_ch=out_ch, ks=ef_ks, activation=activation),
                nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity()
            )
            in_ch_block = out_ch
            blocks.append(blc)
        self.main = nn.Sequential(*blocks)

        final_ch = ef_block_sizes[-1]
        self.mapper = MapperBlock(in_features=final_ch, out_features=final_ch * 2)

        # Separate heads per output
        head_dim = final_ch * 2
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_dim, head_dim),
                nn.BatchNorm1d(head_dim),
                activation(),
                nn.Linear(head_dim, 1)
            )
            for _ in range(num_outputs)
        ])

        self.apply(initialize_weights)

    def forward(self, x):
        x = self.stem(x)
        x = self.main(x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)

        outputs = [head(x).squeeze(-1) for head in self.heads]
        return torch.stack(outputs, dim=1)


# ============================================================================
# Multi-Species CADENCE (wrapper for multi-dataset training)
# ============================================================================

class MultiSpeciesCADENCE(nn.Module):
    """
    Multi-species model using LegNet/CADENCE backbone with optional advanced modules.

    Uses the exact LegNet architecture with:
    - RC-equivariant or PWM-initialized stem
    - Optional ClusterSpace (dilated convolutions for long-range patterns)
    - Optional Grammar layer (bidirectional GRU with FiLM)
    - Optional MicroMotif processor (multi-scale density)
    - Optional Motif correlator (low-rank bilinear pooling)
    - Separate prediction heads per dataset/output
    - Optional conditioning embeddings (species, kingdom, celltype, length)
    - PLACE uncertainty estimation (post-hoc)
    """

    def __init__(
        self,
        config: ModelConfig,
        dataset_names: List[str],
        n_species: int = 1,
        n_kingdoms: int = 1,
        n_celltypes: int = 1,
        max_length: int = 320,
    ):
        super().__init__()

        self.config = config
        self.dataset_names = dataset_names
        activation = nn.SiLU

        # ====================================================================
        # STEM: Choose between RC-equivariant, PWM multi-scale, or standard
        # Optionally create species-specific stems for cross-species transfer
        # ====================================================================
        use_pwm_stem = getattr(config, 'use_pwm_stem', False)
        use_rc_stem = getattr(config, 'use_rc_stem', True)
        use_species_stem = getattr(config, 'use_species_stem', False)
        use_kingdom_stem = getattr(config, 'use_kingdom_stem', False)

        self.use_species_stem = use_species_stem
        self.use_kingdom_stem = use_kingdom_stem
        self.n_species = n_species
        self.n_kingdoms = n_kingdoms

        def _create_stem():
            """Helper to create a single stem module."""
            if use_pwm_stem:
                pwm_scales = getattr(config, 'pwm_stem_scales', [7, 11, 15, 19])
                stem = PWMMultiScaleStem(
                    n_motifs_per_scale=24,
                    kernel_sizes=pwm_scales,
                    in_channels=4,
                )
                stem_out_ch = stem.out_channels
                adapter = nn.Sequential(
                    nn.Conv1d(stem_out_ch, config.stem_channels, 1),
                    nn.BatchNorm1d(config.stem_channels),
                    nn.SiLU(),
                )
                return stem, adapter
            elif use_rc_stem:
                stem = RCEquivariantStem(
                    out_channels=config.stem_channels,
                    kernel_size=config.stem_kernel_size,
                    in_channels=4,
                    activation=activation,
                )
                return stem, None
            else:
                stem = LocalBlock(
                    in_ch=4,
                    out_ch=config.stem_channels,
                    ks=config.stem_kernel_size,
                    activation=activation
                )
                return stem, None

        if use_species_stem and n_species > 1:
            # Create separate stems per species for cross-species transfer
            self.species_stems = nn.ModuleList()
            self.species_stem_adapters = nn.ModuleList()
            for _ in range(n_species):
                stem, adapter = _create_stem()
                self.species_stems.append(stem)
                self.species_stem_adapters.append(adapter if adapter else nn.Identity())
            self.stem = None
            self.stem_adapter = None
            self.kingdom_stems = None
            self.kingdom_stem_adapters = None
        elif use_kingdom_stem and n_kingdoms > 1:
            # Create separate stems per kingdom for cross-kingdom transfer
            self.kingdom_stems = nn.ModuleList()
            self.kingdom_stem_adapters = nn.ModuleList()
            for _ in range(n_kingdoms):
                stem, adapter = _create_stem()
                self.kingdom_stems.append(stem)
                self.kingdom_stem_adapters.append(adapter if adapter else nn.Identity())
            self.stem = None
            self.stem_adapter = None
            self.species_stems = None
            self.species_stem_adapters = None
        else:
            # Single shared stem (default for single-cell-type training)
            self.species_stems = None
            self.species_stem_adapters = None
            self.kingdom_stems = None
            self.kingdom_stem_adapters = None
            self.stem, self.stem_adapter = _create_stem()

        # ====================================================================
        # MAIN BLOCKS: LegNet architecture
        # ====================================================================
        blocks = []
        in_ch = config.stem_channels
        pool_sizes = getattr(config, 'pool_sizes', [2, 2, 2, 2])

        for pool_sz, out_ch in zip(pool_sizes, config.block_channels):
            blc = nn.Sequential(
                ResidualConcat(
                    EffBlock(
                        in_ch=in_ch,
                        out_ch=in_ch,
                        ks=config.block_kernel,
                        resize_factor=config.expand_ratio,
                        activation=activation
                    )
                ),
                LocalBlock(in_ch=in_ch * 2, out_ch=out_ch, ks=config.block_kernel, activation=activation),
                nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity()
            )
            in_ch = out_ch
            blocks.append(blc)
        self.main = nn.Sequential(*blocks)

        final_ch = config.block_channels[-1]

        # ====================================================================
        # OPTIONAL MODULES (inserted after main, before mapper)
        # ====================================================================

        # ClusterSpace: dilated convolutions for long-range patterns
        self.cluster_space = None
        if getattr(config, 'use_cluster_space', False):
            cluster_dilations = getattr(config, 'cluster_dilations', [1, 2, 4, 1])
            self.cluster_space = ClusterSpace(
                in_channels=final_ch,
                block_configs=[(final_ch, final_ch, d) for d in cluster_dilations],
            )

        # Grammar layer: bidirectional GRU with FiLM
        self.grammar = None
        if getattr(config, 'use_grammar', False):
            grammar_hidden = getattr(config, 'grammar_hidden', 128)
            self.grammar = LightweightGrammarLayer(
                in_channels=final_ch,
                hidden=grammar_hidden,
            )

        # MicroMotif processor: multi-scale density
        self.micromotif = None
        if getattr(config, 'use_micromotif', False):
            micromotif_windows = getattr(config, 'micromotif_windows', [5, 11, 21])
            self.micromotif = MicroMotifProcessor(
                in_channels=final_ch,
                window_sizes=micromotif_windows,
            )

        # Motif correlator: low-rank bilinear pooling
        self.correlator = None
        if getattr(config, 'use_motif_correlator', False):
            correlator_factors = getattr(config, 'correlator_factors', 32)
            correlator_rank = getattr(config, 'correlator_rank', 8)
            self.correlator = LowRankMotifCorrelator(
                in_channels=final_ch,
                n_factors=correlator_factors,
                rank=correlator_rank,
            )

        # ====================================================================
        # MAPPER AND HEAD
        # ====================================================================
        self.mapper = MapperBlock(in_features=final_ch, out_features=final_ch * 2)
        backbone_out = final_ch * 2

        # Optional conditioning embeddings
        conditioning_dim = 0
        if config.use_species_embedding:
            self.species_embed = nn.Embedding(n_species, config.species_embed_dim)
            conditioning_dim += config.species_embed_dim
        if config.use_kingdom_embedding:
            self.kingdom_embed = nn.Embedding(n_kingdoms, config.kingdom_embed_dim)
            conditioning_dim += config.kingdom_embed_dim
        if config.use_celltype_embedding:
            self.celltype_embed = nn.Embedding(n_celltypes, config.celltype_embed_dim)
            conditioning_dim += config.celltype_embed_dim
        if config.use_length_embedding:
            self.length_embed = nn.Embedding(max_length // 10 + 1, config.length_embed_dim)
            conditioning_dim += config.length_embed_dim
        # Element-type embedding (promoter vs enhancer vs silencer)
        use_element_type_embedding = getattr(config, 'use_element_type_embedding', False)
        if use_element_type_embedding:
            element_type_embed_dim = getattr(config, 'element_type_embed_dim', 8)
            # Count unique element types from datasets
            element_types = set()
            for name in dataset_names:
                if name in DATASET_CATALOG:
                    element_types.add(DATASET_CATALOG[name].element_type)
            element_types.add("unknown")
            self.element_type_to_idx = {et: i for i, et in enumerate(sorted(element_types))}
            self.element_type_embed = nn.Embedding(len(element_types), element_type_embed_dim)
            conditioning_dim += element_type_embed_dim
        else:
            self.element_type_to_idx = None
            self.element_type_embed = None

        # Heads - one per dataset/output combination (LegNet style)
        head_input = backbone_out + conditioning_dim
        self.head_input_dim = head_input
        self.heads = nn.ModuleDict()
        self.dataset_to_heads = {}

        for dataset_name in dataset_names:
            if dataset_name not in DATASET_CATALOG:
                continue
            info = DATASET_CATALOG[dataset_name]
            head_names = []
            for output_name in info.output_names:
                head_name = f"{dataset_name}_{output_name}"
                if head_name not in self.heads:
                    self.heads[head_name] = nn.Sequential(
                        nn.Linear(head_input, head_input),
                        nn.BatchNorm1d(head_input),
                        nn.SiLU(),
                        nn.Linear(head_input, 1)
                    )
                head_names.append(head_name)
            self.dataset_to_heads[dataset_name] = head_names

        # PLACE uncertainty estimator (post-hoc, initialized after training)
        self.place_uncertainty = None

        self.apply(initialize_weights)

    def _backbone_forward(
        self,
        sequence: torch.Tensor,
        species_idx: Optional[torch.Tensor] = None,
        kingdom_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward through backbone only (for feature extraction).

        Args:
            sequence: Input sequences [B, 4, L]
            species_idx: Species indices [B] for species-specific stems
            kingdom_idx: Kingdom indices [B] for kingdom-specific stems
        """
        # Apply stem(s) - priority: species > kingdom > shared
        if self.use_species_stem and self.species_stems is not None and species_idx is not None:
            # Species-specific stems: route each sample through its species' stem
            batch_size = sequence.shape[0]

            # Process each species separately and combine
            outputs = []
            for i in range(batch_size):
                sp_idx = species_idx[i].item()
                sp_idx = min(sp_idx, len(self.species_stems) - 1)  # Clamp to valid range
                x_i = self.species_stems[sp_idx](sequence[i:i+1])
                x_i = self.species_stem_adapters[sp_idx](x_i)
                outputs.append(x_i)
            x = torch.cat(outputs, dim=0)
        elif self.use_kingdom_stem and self.kingdom_stems is not None and kingdom_idx is not None:
            # Kingdom-specific stems: route each sample through its kingdom's stem
            batch_size = sequence.shape[0]

            outputs = []
            for i in range(batch_size):
                k_idx = kingdom_idx[i].item()
                k_idx = min(k_idx, len(self.kingdom_stems) - 1)  # Clamp to valid range
                x_i = self.kingdom_stems[k_idx](sequence[i:i+1])
                x_i = self.kingdom_stem_adapters[k_idx](x_i)
                outputs.append(x_i)
            x = torch.cat(outputs, dim=0)
        else:
            # Single shared stem (default)
            x = self.stem(sequence)
            if self.stem_adapter is not None:
                x = self.stem_adapter(x)

        x = self.main(x)

        # Apply optional modules
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
        return x.squeeze(-1)

    def get_features(self, sequence: torch.Tensor) -> torch.Tensor:
        """Get features before heads (for PLACE uncertainty)."""
        return self._backbone_forward(sequence)

    def forward(
        self,
        sequence: torch.Tensor,
        species_idx: Optional[torch.Tensor] = None,
        kingdom_idx: Optional[torch.Tensor] = None,
        celltype_idx: Optional[torch.Tensor] = None,
        original_length: Optional[torch.Tensor] = None,
        dataset_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Forward pass with optional conditioning."""
        batch_size = sequence.shape[0]
        device = sequence.device

        # Backbone forward (pass species_idx/kingdom_idx for specific stems)
        features = self._backbone_forward(sequence, species_idx=species_idx, kingdom_idx=kingdom_idx)

        # Conditioning embeddings
        conditioning = []
        if self.config.use_species_embedding:
            if species_idx is not None:
                conditioning.append(self.species_embed(species_idx))
            else:
                conditioning.append(torch.zeros(batch_size, self.config.species_embed_dim, device=device))
        if self.config.use_kingdom_embedding:
            if kingdom_idx is not None:
                conditioning.append(self.kingdom_embed(kingdom_idx))
            else:
                conditioning.append(torch.zeros(batch_size, self.config.kingdom_embed_dim, device=device))
        if self.config.use_celltype_embedding:
            if celltype_idx is not None:
                conditioning.append(self.celltype_embed(celltype_idx))
            else:
                conditioning.append(torch.zeros(batch_size, self.config.celltype_embed_dim, device=device))
        if self.config.use_length_embedding:
            if original_length is not None:
                length_buckets = (original_length // 10).clamp(0, 31)
                conditioning.append(self.length_embed(length_buckets))
            else:
                conditioning.append(torch.zeros(batch_size, self.config.length_embed_dim, device=device))
        # Element-type embedding based on dataset
        if self.element_type_embed is not None and dataset_names is not None:
            element_type_indices = []
            for ds_name in dataset_names:
                if ds_name in DATASET_CATALOG:
                    et = DATASET_CATALOG[ds_name].element_type
                else:
                    et = "unknown"
                et_idx = self.element_type_to_idx.get(et, self.element_type_to_idx.get("unknown", 0))
                element_type_indices.append(et_idx)
            et_idx_tensor = torch.tensor(element_type_indices, device=device)
            conditioning.append(self.element_type_embed(et_idx_tensor))

        if conditioning:
            features = torch.cat([features] + conditioning, dim=1)

        # Compute outputs per head
        if dataset_names is None:
            results = {}
            for head_name, head in self.heads.items():
                pred = head(features).squeeze(-1)
                results[head_name] = {'mean': pred, 'indices': list(range(len(pred)))}
            return results

        # Group by dataset
        outputs = {head_name: {'indices': [], 'features': []} for head_name in self.heads.keys()}

        for i, dataset in enumerate(dataset_names):
            if dataset not in self.dataset_to_heads:
                continue
            for head_name in self.dataset_to_heads[dataset]:
                outputs[head_name]['indices'].append(i)
                outputs[head_name]['features'].append(features[i])

        results = {}
        for head_name, data in outputs.items():
            if data['features']:
                stacked = torch.stack(data['features'])
                pred = self.heads[head_name](stacked).squeeze(-1)
                results[head_name] = {
                    'mean': pred,
                    'indices': data['indices'],
                }

        return results

    def init_place_uncertainty(
        self,
        feature_dim: Optional[int] = None,
        n_neighbors: int = 200,
        alpha: float = 0.1,
    ) -> PLACEUncertainty:
        """Initialize PLACE uncertainty estimator (call after training)."""
        if feature_dim is None:
            feature_dim = self.head_input_dim
        self.place_uncertainty = PLACEUncertainty(
            feature_dim=feature_dim,
            n_neighbors=n_neighbors,
            alpha=alpha,
        )
        return self.place_uncertainty

    def get_active_modules(self) -> Dict[str, bool]:
        """Get which optional modules are active."""
        return {
            'rc_stem': isinstance(self.stem, RCEquivariantStem),
            'pwm_stem': isinstance(self.stem, PWMMultiScaleStem),
            'cluster_space': self.cluster_space is not None,
            'grammar': self.grammar is not None,
            'micromotif': self.micromotif is not None,
            'correlator': self.correlator is not None,
            'place_uncertainty': self.place_uncertainty is not None,
        }

    def get_parameter_counts(self) -> Dict[str, int]:
        """Get parameter count by component."""
        counts = {}

        # Handle different stem configurations
        if self.stem is not None:
            counts['stem'] = sum(p.numel() for p in self.stem.parameters())
        elif self.species_stems is not None:
            counts['stem'] = sum(p.numel() for stems in self.species_stems for p in stems.parameters())
        elif self.kingdom_stems is not None:
            counts['stem'] = sum(p.numel() for stems in self.kingdom_stems for p in stems.parameters())
        else:
            counts['stem'] = 0

        counts['main'] = sum(p.numel() for p in self.main.parameters())
        counts['mapper'] = sum(p.numel() for p in self.mapper.parameters())
        counts['heads'] = sum(p.numel() for p in self.heads.parameters())

        if self.stem_adapter is not None:
            counts['stem_adapter'] = sum(p.numel() for p in self.stem_adapter.parameters())
        if self.cluster_space is not None:
            counts['cluster_space'] = sum(p.numel() for p in self.cluster_space.parameters())
        if self.grammar is not None:
            counts['grammar'] = sum(p.numel() for p in self.grammar.parameters())
        if self.micromotif is not None:
            counts['micromotif'] = sum(p.numel() for p in self.micromotif.parameters())
        if self.correlator is not None:
            counts['correlator'] = sum(p.numel() for p in self.correlator.parameters())

        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts

    def freeze_backbone(self):
        """Freeze backbone parameters (stems, main, optional modules, mapper)."""
        # Freeze stems
        if self.stem is not None:
            for param in self.stem.parameters():
                param.requires_grad = False
        if self.species_stems is not None:
            for stem in self.species_stems:
                for param in stem.parameters():
                    param.requires_grad = False
        if self.kingdom_stems is not None:
            for stem in self.kingdom_stems:
                for param in stem.parameters():
                    param.requires_grad = False
        if self.stem_adapter is not None:
            for param in self.stem_adapter.parameters():
                param.requires_grad = False
        if self.species_stem_adapters is not None:
            for adapter in self.species_stem_adapters:
                for param in adapter.parameters():
                    param.requires_grad = False
        if self.kingdom_stem_adapters is not None:
            for adapter in self.kingdom_stem_adapters:
                for param in adapter.parameters():
                    param.requires_grad = False

        # Freeze main blocks
        for param in self.main.parameters():
            param.requires_grad = False

        # Freeze optional modules
        if self.cluster_space is not None:
            for param in self.cluster_space.parameters():
                param.requires_grad = False
        if self.grammar is not None:
            for param in self.grammar.parameters():
                param.requires_grad = False
        if self.micromotif is not None:
            for param in self.micromotif.parameters():
                param.requires_grad = False
        if self.correlator is not None:
            for param in self.correlator.parameters():
                param.requires_grad = False

        # Freeze mapper
        for param in self.mapper.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        # Unfreeze stems
        if self.stem is not None:
            for param in self.stem.parameters():
                param.requires_grad = True
        if self.species_stems is not None:
            for stem in self.species_stems:
                for param in stem.parameters():
                    param.requires_grad = True
        if self.kingdom_stems is not None:
            for stem in self.kingdom_stems:
                for param in stem.parameters():
                    param.requires_grad = True
        if self.stem_adapter is not None:
            for param in self.stem_adapter.parameters():
                param.requires_grad = True
        if self.species_stem_adapters is not None:
            for adapter in self.species_stem_adapters:
                for param in adapter.parameters():
                    param.requires_grad = True
        if self.kingdom_stem_adapters is not None:
            for adapter in self.kingdom_stem_adapters:
                for param in adapter.parameters():
                    param.requires_grad = True

        # Unfreeze main blocks
        for param in self.main.parameters():
            param.requires_grad = True

        # Unfreeze optional modules
        if self.cluster_space is not None:
            for param in self.cluster_space.parameters():
                param.requires_grad = True
        if self.grammar is not None:
            for param in self.grammar.parameters():
                param.requires_grad = True
        if self.micromotif is not None:
            for param in self.micromotif.parameters():
                param.requires_grad = True
        if self.correlator is not None:
            for param in self.correlator.parameters():
                param.requires_grad = True

        # Unfreeze mapper
        for param in self.mapper.parameters():
            param.requires_grad = True


# ============================================================================
# Loss Functions
# ============================================================================

def compute_extreme_weights(
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> torch.Tensor:
    """
    Compute sample weights that emphasize extreme values.

    Weight formula: 1 + alpha * |z|^beta
    where z = (target - mean) / std
    """
    mean = targets.mean()
    std = targets.std() + 1e-8
    z_scores = torch.abs((targets - mean) / std)
    z_scores = z_scores.clamp(max=4.0)
    weights = 1.0 + alpha * (z_scores ** beta)
    weights = weights / weights.mean()
    return weights


def compute_masked_loss(
    outputs: Dict[str, Dict],
    targets: torch.Tensor,
    dataset_names: List[str],
    dataset_to_heads: Dict[str, List[str]],
    use_uncertainty: bool = False,
    use_extreme_weights: bool = True,
    extreme_alpha: float = 0.5,
    extreme_beta: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute loss with masking and optional extreme value weighting."""
    total_loss = 0.0
    n_samples = 0
    per_head_losses = {}

    for head_name, pred in outputs.items():
        if not pred['indices']:
            continue

        indices = pred['indices']
        mean = pred['mean']
        logvar = pred.get('logvar', None)

        # Get targets
        head_targets = []
        for i, idx in enumerate(indices):
            dataset = dataset_names[idx]
            heads_for_dataset = dataset_to_heads.get(dataset, [])
            if head_name in heads_for_dataset:
                output_idx = heads_for_dataset.index(head_name)
                head_targets.append(targets[idx, output_idx])

        if not head_targets:
            continue

        target = torch.stack(head_targets)
        valid_mask = ~torch.isnan(target)
        if valid_mask.sum() == 0:
            continue

        mean = mean[valid_mask]
        target = target[valid_mask]
        if logvar is not None:
            logvar = logvar[valid_mask]

        # Compute loss
        if use_uncertainty and logvar is not None:
            variance = torch.exp(logvar).clamp(min=1e-6)
            loss = 0.5 * (logvar + (target - mean) ** 2 / variance)
        else:
            loss = (target - mean) ** 2

        # Apply extreme value weighting
        if use_extreme_weights and len(target) > 1:
            weights = compute_extreme_weights(target, extreme_alpha, extreme_beta)
            loss = loss * weights

        head_loss = loss.mean()
        per_head_losses[head_name] = head_loss
        total_loss += head_loss * len(target)
        n_samples += len(target)

    if n_samples > 0:
        total_loss = total_loss / n_samples

    return total_loss, per_head_losses


# ============================================================================
# Factory Function
# ============================================================================

def create_multi_species_model(
    config: ModelConfig,
    dataset_names: List[str],
    n_species: int = 1,
    n_kingdoms: int = 1,
    n_celltypes: int = 1,
) -> MultiSpeciesCADENCE:
    """Create multi-species CADENCE model."""
    if n_species is None or n_kingdoms is None or n_celltypes is None:
        species_set, kingdom_set, celltype_set = set(), set(), {"unknown"}
        for name in dataset_names:
            if name in DATASET_CATALOG:
                info = DATASET_CATALOG[name]
                species_set.add(info.species)
                kingdom_set.add(info.kingdom)
                if info.cell_type:
                    celltype_set.add(info.cell_type)
        n_species = max(len(species_set), 1)
        n_kingdoms = max(len(kingdom_set), 1)
        n_celltypes = max(len(celltype_set), 1)

    return MultiSpeciesCADENCE(
        config=config,
        dataset_names=dataset_names,
        n_species=n_species,
        n_kingdoms=n_kingdoms,
        n_celltypes=n_celltypes,
    )
