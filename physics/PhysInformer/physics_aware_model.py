"""
Physics-Aware Architecture for DNA Sequence Property Prediction
Integrates thermodynamics, electrostatics, structural mechanics, and sequence features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List
import numpy as np

# ============================================================================
# CORE BUILDING BLOCKS
# ============================================================================

class PWMConvStem(nn.Module):
    """PWM-style convolutional stem for motif detection"""
    
    def __init__(self, vocab_size: int = 5, hidden_dims: List[int] = [128, 192, 256], 
                 kernel_sizes: List[int] = [11, 9, 7], dropout: float = 0.1):
        super().__init__()
        
        # Initial embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dims[0])
        
        # Simpler conv blocks without depthwise separation for now
        self.blocks = nn.ModuleList()
        in_dim = hidden_dims[0]
        
        for i, (out_dim, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            # Add residual projection if dimensions change
            if in_dim != out_dim:
                res_proj = nn.Conv1d(in_dim, out_dim, 1)
            else:
                res_proj = nn.Identity()
                
            block = nn.ModuleDict({
                'conv': nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, 
                                 padding=kernel_size//2),
                'activation': nn.SiLU(),
                'dropout': nn.Dropout(dropout),
                'res_proj': res_proj
            })
            self.blocks.append(block)
            in_dim = out_dim
    
    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, hidden_dim)
        x = self.embedding(x)
        x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        
        for block in self.blocks:
            # Apply block with residual
            residual = x
            x = block['conv'](x)
            x = block['activation'](x)
            x = block['dropout'](x)
            
            # Residual connection
            residual = block['res_proj'](residual)
            x = x + residual
            
        return x.transpose(1, 2)  # Back to (batch, seq_len, hidden_dim)


class SimplifiedSSMLayer(nn.Module):
    """Simplified State Space Model for long-range dependencies"""
    
    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Linear(d_model, d_state)
        self.C = nn.Linear(d_state, d_model)
        self.D = nn.Parameter(torch.ones(d_model) * 0.01)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU()
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        
        # Compute SSM
        state = torch.zeros(batch, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            state = torch.tanh(state @ self.A.T + self.B(x_t))
            y_t = self.C(state) + self.D * x_t
            outputs.append(y_t)
            
        output = torch.stack(outputs, dim=1)
        
        # Apply gating
        gate = self.gate(x)
        output = gate * output + (1 - gate) * x
        
        return self.norm(output + x)  # Residual connection


class DualPathFeaturePyramid(nn.Module):
    """Dual-path feature pyramid for multi-scale processing"""
    
    def __init__(self, d_model: int = 256, d_expanded: int = 384, dropout: float = 0.1):
        super().__init__()
        
        # Local path - sharp motif edges (simplified)
        self.local_conv1 = nn.Conv1d(d_model, d_expanded//2, kernel_size=9, padding=4)
        self.local_conv2 = nn.Conv1d(d_expanded//2, d_expanded//2, kernel_size=5, padding=2)
        self.local_act = nn.SiLU()
        self.local_dropout = nn.Dropout(dropout)
        
        # Global path - smooth aggregation
        self.global_path = SimplifiedSSMLayer(d_model, dropout=dropout)
        self.global_proj = nn.Linear(d_model, d_expanded//2)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        
        # Local path
        x_local = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x_local = self.local_conv1(x_local)
        x_local = self.local_act(x_local)
        x_local = self.local_conv2(x_local)
        x_local = self.local_act(x_local)
        x_local = self.local_dropout(x_local)
        x_local = x_local.transpose(1, 2)  # (batch, seq_len, d_expanded//2)
        
        # Global path
        x_global = self.global_path(x)
        x_global = self.global_proj(x_global)  # (batch, seq_len, d_expanded//2)
        
        # Concatenate
        return torch.cat([x_local, x_global], dim=-1)  # (batch, seq_len, d_expanded)


# ============================================================================
# PHYSICS ROUTERS (Property-Specific Adapters)
# ============================================================================

class PhysicsRouter(nn.Module):
    """Routes features through property-specific adapters"""
    
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


# ============================================================================
# PROPERTY-SPECIFIC HEADS
# ============================================================================

class ThermoHead(nn.Module):
    """Joint head for thermodynamic properties (ΔH, ΔS, ΔG)"""
    
    def __init__(self, d_input: int, d_hidden: int = 128, temperature: float = 310.0):
        super().__init__()
        self.temperature = temperature
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        # Individual predictions with uncertainty
        self.dH_head = nn.Linear(d_hidden, 2)  # (mean, log_var)
        self.dS_head = nn.Linear(d_hidden, 2)
        self.dG_head = nn.Linear(d_hidden, 2)
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize to reasonable ranges for thermodynamics
        nn.init.normal_(self.dH_head.weight, 0, 0.1)
        nn.init.normal_(self.dS_head.weight, 0, 0.1)
        nn.init.normal_(self.dG_head.weight, 0, 0.1)
        
    def forward(self, x):
        # Pool over sequence
        if len(x.shape) == 3:
            x = x.mean(dim=1)  # Global average pooling
            
        features = self.shared(x)
        
        # Predict with uncertainty
        dH = self.dH_head(features)
        dS = self.dS_head(features)
        dG = self.dG_head(features)
        
        outputs = {
            'dH_mean': dH[:, 0],
            'dH_log_var': dH[:, 1],
            'dS_mean': dS[:, 0],
            'dS_log_var': dS[:, 1],
            'dG_mean': dG[:, 0],
            'dG_log_var': dG[:, 1],
        }
        
        return outputs


class ElectrostaticHead(nn.Module):
    """Head for electrostatic potential prediction"""
    
    def __init__(self, d_input: int, d_hidden: int = 128, n_windows: int = 22):
        super().__init__()
        self.n_windows = n_windows
        
        # Per-window predictors
        self.window_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_input, d_hidden),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(d_hidden, 6)  # STD_MIN/MAX/MEAN, ENH_MIN/MAX/MEAN
            ) for _ in range(n_windows)
        ])
        
        # Global aggregator
        self.global_head = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 2)  # mean_psi, log_var
        )
        
    def forward(self, x):
        batch, seq_len, d_input = x.shape
        
        # Extract windows (20bp with stride 10)
        window_size = 20
        stride = 10
        window_features = []
        
        for i in range(0, seq_len - window_size + 1, stride):
            window = x[:, i:i+window_size, :].mean(dim=1)  # Pool over window
            window_features.append(window)
            
        # Predict for each window
        window_predictions = []
        for i, window_feat in enumerate(window_features[:self.n_windows]):
            pred = self.window_heads[i](window_feat)
            window_predictions.append(pred)
            
        # Global prediction
        global_feat = x.mean(dim=1)  # Global pool
        global_pred = self.global_head(global_feat)
        
        outputs = {
            'window_predictions': torch.stack(window_predictions, dim=1),  # (batch, n_windows, 6)
            'global_mean': global_pred[:, 0],
            'global_log_var': global_pred[:, 1]
        }
        
        return outputs


class ScalarPropertyHead(nn.Module):
    """Generic head for scalar property prediction"""
    
    def __init__(self, d_input: int, property_name: str, d_hidden: int = 64):
        super().__init__()
        self.property_name = property_name
        
        self.head = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 2)  # (mean, log_var)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize final layer with small std
        nn.init.normal_(self.head[-1].weight, 0, 0.1)
        nn.init.zeros_(self.head[-1].bias)
        
    def forward(self, x):
        # Pool if needed
        if len(x.shape) == 3:
            x = x.mean(dim=1)
            
        output = self.head(x)
        return {
            f'{self.property_name}_mean': output[:, 0],
            f'{self.property_name}_log_var': output[:, 1]
        }


# ============================================================================
# AUXILIARY HEADS FOR ACTIVITY PREDICTION
# ============================================================================

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


# ============================================================================
# MAIN PHYSICS-AWARE MODEL
# ============================================================================

class PhysicsAwareModel(nn.Module):
    """
    Complete physics-aware model for DNA sequence property prediction
    """
    
    def __init__(
        self,
        vocab_size: int = 5,
        d_model: int = 256,
        d_expanded: int = 384,
        seq_len: int = 230,
        dropout: float = 0.1,
        temperature: float = 310.0,
        n_electrostatic_windows: int = 22,
        n_descriptor_features: int = 536,  # Actual number after filtering
        descriptor_names: List[str] = None,  # Actual feature names for proper routing
        property_groups: Optional[Dict] = None
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.temperature = temperature
        
        # ============ Backbone ============
        # PWM-style conv stem
        self.conv_stem = PWMConvStem(vocab_size=vocab_size, 
                                     hidden_dims=[128, 192, d_model],
                                     kernel_sizes=[11, 9, 7],
                                     dropout=dropout)
        
        # Long-range mixer (2 SSM layers)
        self.ssm_layers = nn.ModuleList([
            SimplifiedSSMLayer(d_model, dropout=dropout) for _ in range(2)
        ])
        
        # Dual-path feature pyramid
        self.feature_pyramid = DualPathFeaturePyramid(d_model, d_expanded, dropout)
        
        # ============ Physics Routers ============
        # Define routers for different property types
        self.routers = nn.ModuleDict({
            'thermo': PhysicsRouter(d_expanded, 128, kernel_size=3, property_name='thermo'),
            'electrostatic': PhysicsRouter(d_expanded, 128, kernel_size=15, property_name='electrostatic'),
            'bend': PhysicsRouter(d_expanded, 128, kernel_size=11, property_name='bend'),
            'stiff': PhysicsRouter(d_expanded, 128, kernel_size=7, property_name='stiff'),
            'pwm': PhysicsRouter(d_expanded, 256, kernel_size=15, property_name='pwm'),
            'entropy': PhysicsRouter(d_expanded, 128, kernel_size=21, property_name='entropy'),
            'advanced': PhysicsRouter(d_expanded, 128, kernel_size=13, property_name='advanced')
        })
        
        # ============ Property Heads ============
        # Thermodynamic head (joint for ΔH, ΔS, ΔG)
        self.thermo_head = ThermoHead(128, temperature=temperature)
        
        # Electrostatic head
        self.electrostatic_head = ElectrostaticHead(128, n_windows=n_electrostatic_windows)
        
        # Individual scalar heads for descriptor features
        self.property_heads = nn.ModuleDict()
        
        # Create heads for all descriptor features 
        # The descriptor features are all the biophysical descriptors (PWM, bend, etc.)
        # No need to subtract thermodynamic features since those are handled separately
        
        # Create property heads for all descriptor features
        # Route based on actual feature names if provided
        self.feature_routing = {}  # Map feature index to router name
        
        for i in range(n_descriptor_features):
            head_name = f'feature_{i}'
            
            # Determine router based on feature name if available
            if descriptor_names and i < len(descriptor_names):
                feat_name = descriptor_names[i]
                if feat_name.startswith('pwm_'):
                    router_name = 'pwm'
                    router_features = self.routers['pwm'].window_agg.out_channels  # 256
                elif feat_name.startswith('bend_') or 'bend' in feat_name:
                    router_name = 'bend'
                    router_features = self.routers['bend'].window_agg.out_channels  # 128
                elif feat_name.startswith('stiff_') or 'stiff' in feat_name:
                    router_name = 'stiff'
                    router_features = self.routers['stiff'].window_agg.out_channels  # 128
                elif feat_name.startswith('thermo_') or 'entropy' in feat_name:
                    router_name = 'entropy'
                    router_features = self.routers['entropy'].window_agg.out_channels  # 128
                elif feat_name.startswith('advanced_') or any(x in feat_name for x in ['mgw', 'melting', 'stress', 'stacking']):
                    router_name = 'advanced'
                    router_features = self.routers['advanced'].window_agg.out_channels  # 128
                else:
                    # Default to PWM for unknown features
                    router_name = 'pwm'
                    router_features = self.routers['pwm'].window_agg.out_channels  # 256
            else:
                # Fallback to index-based routing if no names provided
                if i < 80:  # PWM-like features (first ~80 features)
                    router_name = 'pwm'
                    router_features = self.routers['pwm'].window_agg.out_channels  # 256
                elif i < 160:  # Bend-like features  
                    router_name = 'bend'
                    router_features = self.routers['bend'].window_agg.out_channels  # 128
                elif i < 240:  # Stiffness-like features
                    router_name = 'stiff'
                    router_features = self.routers['stiff'].window_agg.out_channels  # 128
                elif i < 400:  # Advanced structural features
                    router_name = 'advanced'
                    router_features = self.routers['advanced'].window_agg.out_channels  # 128
                else:  # Entropy and other features
                    router_name = 'entropy'
                    router_features = self.routers['entropy'].window_agg.out_channels  # 128
            
            self.feature_routing[i] = router_name
            self.property_heads[head_name] = ScalarPropertyHead(
                router_features, head_name
            )
        
        # ============ Auxiliary Heads (Optional) ============
        # These are diagnostic heads for activity prediction
        # They use detached features and have separate optimizers
        self.use_auxiliary = False  # Will be enabled when real features are provided
        self.n_activities = 1  # Default for human cell types, S2 has 2
        
        # Initialize auxiliary heads (will be created when needed)
        self.aux_head_a = None  # Sequence + Real Features → Activity
        self.aux_head_b = None  # Real Features Only → Activity
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with physics-aware priors"""
        # Already handled in individual components
        pass
        
    def forward(self, x, real_features: Optional[torch.Tensor] = None, 
                real_activities: Optional[torch.Tensor] = None,
                return_features: bool = False):
        """
        Forward pass
        
        Args:
            x: Input DNA sequences (batch, seq_len)
            real_features: Optional real physics features for auxiliary heads (batch, n_features)
            real_activities: Optional real activity labels for auxiliary training
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary of predictions for all properties
        """
        # ============ Backbone Processing ============
        # PWM stem
        h = self.conv_stem(x)  # (batch, seq_len, d_model)
        
        # SSM layers for long-range
        for ssm in self.ssm_layers:
            h = ssm(h)
            
        # Feature pyramid
        h = self.feature_pyramid(h)  # (batch, seq_len, d_expanded)
        
        # ============ Route Through Property Adapters ============
        routed_features = {}
        for name, router in self.routers.items():
            routed_features[name] = router(h)
            
        # ============ Property Predictions ============
        outputs = {}
        
        # Thermodynamics
        thermo_out = self.thermo_head(routed_features['thermo'])
        outputs.update(thermo_out)
        
        # Electrostatics
        elec_out = self.electrostatic_head(routed_features['electrostatic'])
        outputs.update(elec_out)
        
        # Descriptor features - use the proper routing determined during initialization
        for i, (head_name, head) in enumerate(self.property_heads.items()):
            # Use the router mapping determined during initialization
            router_name = self.feature_routing.get(i, 'pwm')  # Default to PWM if not found
            routed_feat = routed_features[router_name]
            
            prop_out = head(routed_feat)
            outputs.update(prop_out)
                
        # ============ Auxiliary Predictions (Optional) ============
        if real_features is not None and self.use_auxiliary:
            # Auxiliary Head A: Raw Sequence + Real Features → Activity
            # This head has its OWN sequence encoder, separate from main model
            if self.aux_head_a is not None:
                outputs['aux_activity_seq_feat'] = self.aux_head_a(x, real_features)
            
            # Auxiliary Head B: Real Features Only → Activity
            if self.aux_head_b is not None:
                outputs['aux_activity_feat_only'] = self.aux_head_b(real_features)
        
        if return_features:
            outputs['features'] = h
            outputs['routed_features'] = routed_features
            
        return outputs
    
    def enable_auxiliary_heads(self, n_real_features: int, n_activities: int = 1):
        """
        Enable and initialize auxiliary heads for activity prediction
        
        Args:
            n_real_features: Number of real physics features
            n_activities: Number of activity scores to predict (1 for human, 2 for S2)
        """
        self.use_auxiliary = True
        self.n_activities = n_activities
        
        # Initialize auxiliary heads if not already created
        if self.aux_head_a is None:
            # Head A now has its own sequence encoder
            self.aux_head_a = AuxiliaryHeadA(
                vocab_size=5,
                seq_len=self.seq_len,
                feature_dim=n_real_features,
                hidden_dim=256,
                n_activities=n_activities,
                dropout=0.1
            )
            # Move to same device as model
            self.aux_head_a = self.aux_head_a.to(next(self.parameters()).device)
        
        if self.aux_head_b is None:
            self.aux_head_b = AuxiliaryHeadB(
                feature_dim=n_real_features,
                n_activities=n_activities,
                dropout=0.1
            )
            # Move to same device as model
            self.aux_head_b = self.aux_head_b.to(next(self.parameters()).device)
    
    def get_auxiliary_parameters(self):
        """Get parameters of auxiliary heads only (for separate optimizer)"""
        aux_params = []
        if self.aux_head_a is not None:
            aux_params.extend(self.aux_head_a.parameters())
        if self.aux_head_b is not None:
            aux_params.extend(self.aux_head_b.parameters())
        return aux_params
    
    def get_main_parameters(self):
        """Get parameters of main model only (excluding auxiliary heads)"""
        main_params = []
        for name, param in self.named_parameters():
            if 'aux_head' not in name:
                main_params.append(param)
        return main_params


# ============================================================================
# PHYSICS-AWARE LOSSES
# ============================================================================

class PhysicsAwareLoss(nn.Module):
    """Combined loss function with physics constraints"""
    
    def __init__(self, temperature: float = 310.0, 
                 thermo_weight: float = 1.0,
                 identity_weight: float = 0.1,
                 smooth_weight: float = 0.01):
        super().__init__()
        self.temperature = temperature
        self.thermo_weight = thermo_weight
        self.identity_weight = identity_weight
        self.smooth_weight = smooth_weight
        
    def heteroscedastic_loss(self, mean, log_var, target):
        """Heteroscedastic regression loss"""
        precision = torch.exp(-log_var)
        return 0.5 * (precision * (mean - target) ** 2 + log_var).mean()
        
    def thermodynamic_identity_loss(self, dH, dS, dG):
        """Enforce ΔG = ΔH - T*ΔS"""
        predicted_dG = dH - self.temperature * dS
        return F.mse_loss(predicted_dG, dG)
        
    def total_variation_loss(self, x):
        """Smoothness regularization for profiles"""
        if len(x.shape) == 3:
            # (batch, seq_len, features)
            diff = x[:, 1:, :] - x[:, :-1, :]
            return diff.abs().mean()
        return 0.0
        
    def forward(self, predictions, targets):
        """
        Compute total loss
        
        Args:
            predictions: Model outputs
            targets: Ground truth values
            
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Thermodynamic losses
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
            
        # Other property losses (simple MSE for descriptor features)
        for key in predictions:
            if key.endswith('_mean') and key not in ['dH_mean', 'dS_mean', 'dG_mean']:
                prop_name = key.replace('_mean', '')
                if prop_name in targets:
                    # Use simple MSE loss for descriptor features
                    losses[f'{prop_name}_loss'] = F.mse_loss(
                        predictions[key], targets[prop_name]
                    )
                        
        # Combine losses
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses


class AuxiliaryLoss(nn.Module):
    """Loss function for auxiliary activity prediction heads"""
    
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
        """
        Compute auxiliary losses
        
        Args:
            predictions: Model outputs containing auxiliary predictions
            targets: Real activity values (batch, n_activities)
            
        Returns:
            Dictionary with individual auxiliary losses
        """
        losses = {}
        
        # Loss for Head A: Sequence + Features
        if 'aux_activity_seq_feat' in predictions:
            losses['aux_seq_feat_loss'] = self.loss_fn(
                predictions['aux_activity_seq_feat'], targets
            )
        
        # Loss for Head B: Features Only
        if 'aux_activity_feat_only' in predictions:
            losses['aux_feat_only_loss'] = self.loss_fn(
                predictions['aux_activity_feat_only'], targets
            )
        
        # Total auxiliary loss
        if losses:
            losses['aux_total_loss'] = sum(losses.values())
        
        return losses


# ============================================================================
# MODEL CREATION FUNCTIONS
# ============================================================================

def create_physics_aware_model(cell_type: str = 'HepG2', n_descriptor_features: int = None, descriptor_names: List[str] = None, **kwargs):
    """
    Create physics-aware model for specific cell type
    
    Args:
        cell_type: One of 'HepG2', 'K562', 'WTC11', 'S2'
        n_descriptor_features: Actual number of descriptor features after filtering
        **kwargs: Additional model parameters
    """
    # Default number of features per cell type (will be updated based on actual data)
    default_features = {
        'HepG2': 536,  # After filtering 1 zero-variance feature
        'K562': 515,   # After filtering 6 zero-variance features
        'WTC11': 539,  # After filtering 6 zero-variance features  
        'S2': 537      # Needs data cleaning (has string values)
    }
    
    if n_descriptor_features is None:
        n_descriptor_features = default_features.get(cell_type, 536)
    
    config = {
        'n_descriptor_features': n_descriptor_features,
        'n_electrostatic_windows': 22
    }
    config.update(kwargs)
    
    # Add descriptor names if provided
    if descriptor_names is not None:
        config['descriptor_names'] = descriptor_names
    
    return PhysicsAwareModel(**config)