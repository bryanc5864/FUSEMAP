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
    def __init__(self, vocab_size: int = 5, d_model: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class TileFormerBlock(nn.Module):
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

class TileFormerSingle(nn.Module):
    """
    Transformer architecture to predict a single global feature (thermo_dG_p25) from DNA sequence.
    This model processes the entire sequence to predict sequence-level properties.
    """
    def __init__(
        self,
        vocab_size: int = 5,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_len: int = 240,
        dropout: float = 0.1,
        target_feature: str = 'thermo_dG_p25'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.target_feature = target_feature
        
        # Embeddings
        self.nucleotide_embedding = NucleotideEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks for sequence processing
        self.transformer_blocks = nn.ModuleList([
            TileFormerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Global pooling for sequence-level representation
        self.global_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Single global feature prediction head
        # This operates on the global sequence representation
        self.feature_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)  # Single value prediction
        )
        
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # Use normal initialization for final layers
                if 'feature_head.6.weight' in name:
                    # After clipping outliers at 2.5 std, range is [-2.5, 2.5]
                    # Use moderate std to cover this range
                    nn.init.normal_(p, mean=0.0, std=0.3)  # Smaller std for tighter range
                else:
                    nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                if 'feature_head' in name:
                    nn.init.constant_(p, 0.0)  # Zero bias
                else:
                    nn.init.constant_(p, 0.0)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: DNA sequence indices [batch_size, seq_len]
            mask: Attention mask [seq_len, seq_len]
            
        Returns:
            Dict with global feature prediction and auxiliary score
        """
        batch_size, seq_len = x.shape
        
        # Embedding and positional encoding
        x_embed = self.nucleotide_embedding(x)
        x_embed = self.positional_encoding(x_embed)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x_embed = block(x_embed, mask)
        
        # Global sequence representation
        sequence_repr = x_embed.mean(dim=1)  # Global average pooling over entire sequence
        pooled_repr = self.global_pool(sequence_repr)
        
        # Predict target feature from global sequence representation
        target_feature_pred = self.feature_head(pooled_repr).squeeze(-1)  # [batch_size]
        
        outputs = {
            'target_feature': target_feature_pred,  # [batch_size] - global feature prediction
        }
        
        return outputs

def create_tileformer_single(target_feature: str = 'thermo_dG_p25', **kwargs):
    """
    Create TileFormer model for single global feature prediction
    
    Args:
        target_feature: Name of the target feature to predict
        **kwargs: Additional model parameters
    """
    # Default parameters (smaller than PhysInformer for single feature)
    default_params = {
        'vocab_size': 5,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'max_len': 240,
        'dropout': 0.1,
        'target_feature': target_feature
    }
    
    # Update with user parameters
    params = {**default_params, **kwargs}
    
    return TileFormerSingle(**params)