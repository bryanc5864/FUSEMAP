import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class NucleotideEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 5, d_model: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PhysInformerBlock(nn.Module):
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

class PhysInformer(nn.Module):
    """
    Large transformer model for predicting biophysical descriptors from DNA sequences.
    """
    def __init__(
        self,
        vocab_size: int = 5,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        max_len: int = 240,
        dropout: float = 0.1,
        n_descriptors: int = 500,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_descriptors = n_descriptors
        
        # Embeddings
        self.nucleotide_embedding = NucleotideEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            PhysInformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Global feature extraction
        self.global_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Individual feature prediction heads - one head per feature
        # This creates 529 separate prediction heads, each predicting one feature
        self.feature_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 1)  # Single value per feature
            ) for _ in range(n_descriptors)
        ])
        
        
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # Special initialization for final layers of feature heads
                if 'feature_heads' in name and '.6.weight' in name:
                    # Final layer of each feature head - proper scale for clipped range [-2.5, 2.5]
                    # Need larger initialization to prevent collapse to mean
                    nn.init.normal_(p, mean=0.0, std=0.5)  # Increased to match clipped range
                else:
                    # All other layers use xavier initialization
                    nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                # Initialize all biases to zero
                nn.init.constant_(p, 0.0)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: DNA sequence indices [batch_size, seq_len]
            mask: Attention mask [seq_len, seq_len]
            
        Returns:
            Dict with 'descriptors' predictions
        """
        batch_size, seq_len = x.shape
        
        # Embedding and positional encoding
        x_embed = self.nucleotide_embedding(x)
        x_embed = self.positional_encoding(x_embed)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x_embed = block(x_embed, mask)
        
        # Global average pooling over sequence dimension
        sequence_repr = x_embed.mean(dim=1)  # (batch_size, d_model)
        pooled_repr = self.global_pool(sequence_repr)
        
        # Individual feature predictions
        # Each feature head predicts one value from the same global representation
        feature_predictions = []
        for i, head in enumerate(self.feature_heads):
            pred = head(pooled_repr).squeeze(-1)  # [batch_size] - single value per feature
            feature_predictions.append(pred)
        
        # Stack all predictions into [batch_size, n_descriptors]
        descriptors = torch.stack(feature_predictions, dim=1)  # (batch_size, n_descriptors)
        
        outputs = {
            'descriptors': descriptors,
        }
        
        return outputs

def create_physinformer(cell_type: str, **kwargs):
    """
    Create PhysInformer model configured for specific cell type
    
    Args:
        cell_type: 'HepG2', 'K562', or 'WTC11'
        **kwargs: Additional model parameters
    """
    # Cell-type specific configurations
    configs = {
        'HepG2': {'n_descriptors': 537},  # Updated with new features (545 total - 8 zero-variance)
        'K562': {'n_descriptors': 498}, 
        'WTC11': {'n_descriptors': 522}
    }
    
    if cell_type not in configs:
        raise ValueError(f"Unknown cell type: {cell_type}. Choose from {list(configs.keys())}")
    
    # Default parameters (balanced model size)
    default_params = {
        'vocab_size': 5,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 8,
        'd_ff': 2048,
        'max_len': 240,
        'dropout': 0.1,
    }
    
    # Update with cell-type specific and user parameters
    params = {**default_params, **configs[cell_type], **kwargs}
    
    return PhysInformer(**params)