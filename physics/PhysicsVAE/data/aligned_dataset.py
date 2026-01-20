"""
Aligned Dataset for PhysicsVAE evaluation.
Ensures consistent physics features between train/test/val splits.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class AlignedPhysicsVAEDataset(Dataset):
    """
    Dataset that aligns physics features with a reference feature list.
    Use this for evaluation when train/test/val have different columns.
    """

    def __init__(
        self,
        descriptor_file: str,
        reference_features: List[str],
        reference_mean: np.ndarray,
        reference_std: np.ndarray,
        max_seq_length: Optional[int] = None
    ):
        """
        Args:
            descriptor_file: Path to descriptor TSV file
            reference_features: List of physics feature names (from training)
            reference_mean: Mean values for normalization (from training)
            reference_std: Std values for normalization (from training)
            max_seq_length: Maximum sequence length
        """
        self.reference_features = reference_features
        self.reference_mean = reference_mean
        self.reference_std = reference_std

        # Load data
        print(f"Loading {descriptor_file}...")
        self.df = pd.read_csv(descriptor_file, sep='\t')

        # Get sequences
        self.sequences = self.df['sequence'].values

        # Determine sequence length
        seq_lengths = [len(s) for s in self.sequences]
        self.native_seq_length = max(seq_lengths)
        self.seq_length = max_seq_length or self.native_seq_length

        # Extract and align physics features
        self._align_features()

        print(f"Loaded {len(self.df)} sequences")
        print(f"Sequence length: {self.seq_length}")
        print(f"Physics features: {self.n_physics_features} (aligned to reference)")

    def _align_features(self):
        """Align physics features to reference list."""
        n_samples = len(self.df)
        n_features = len(self.reference_features)

        # Initialize with zeros (for missing features)
        self.physics = np.zeros((n_samples, n_features), dtype=np.float32)

        available_cols = set(self.df.columns)
        matched = 0
        missing = []

        for i, feat in enumerate(self.reference_features):
            if feat in available_cols:
                values = self.df[feat].values.astype(np.float32)
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                # Normalize using reference stats
                self.physics[:, i] = (values - self.reference_mean[i]) / (self.reference_std[i] + 1e-8)
                matched += 1
            else:
                missing.append(feat)
                # Leave as 0 (which is the mean in normalized space)

        self.n_physics_features = n_features
        print(f"  Matched {matched}/{n_features} features")
        if missing:
            print(f"  Missing features (set to 0): {missing[:5]}{'...' if len(missing) > 5 else ''}")

    def __len__(self) -> int:
        return len(self.df)

    def _sequence_to_indices(self, sequence: str) -> np.ndarray:
        """Convert DNA sequence to nucleotide indices."""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        indices = np.zeros(self.seq_length, dtype=np.int64)

        for i, base in enumerate(sequence[:self.seq_length]):
            indices[i] = mapping.get(base.upper(), 4)

        return indices

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        physics = self.physics[idx]

        seq_indices = self._sequence_to_indices(sequence)

        return {
            'sequence': torch.tensor(seq_indices, dtype=torch.long),
            'physics': torch.tensor(physics, dtype=torch.float32),
            'idx': idx
        }


def get_training_feature_info(train_file: str) -> Dict:
    """
    Extract feature list and normalization stats from training data.

    Returns dict with:
        - features: list of feature names
        - mean: numpy array of means
        - std: numpy array of stds
    """
    # Must match PhysicsVAEDataset exclude_cols exactly
    exclude_cols = ['name', 'seq_id', 'sequence_id', 'sequence', 'condition',
                    'sequence_length', 'normalized_log2', 'n_obs_bc', 'n_replicates',
                    'cell_type', 'activity', 'is_reverse',
                    # Drosophila activity columns
                    'Dev_log2_enrichment', 'Hk_log2_enrichment',
                    'Dev_log2_enrichment_scaled', 'Hk_log2_enrichment_scaled',
                    'Dev_log2_enrichment_quantile_normalized', 'Hk_log2_enrichment_quantile_normalized',
                    # Plant activity columns
                    'enrichment_leaf', 'enrichment_proto']

    print(f"Loading training data for feature reference: {train_file}")
    df = pd.read_csv(train_file, sep='\t')

    physics_cols = [c for c in df.columns if c not in exclude_cols]
    physics = df[physics_cols].values.astype(np.float32)
    physics = np.nan_to_num(physics, nan=0.0, posinf=0.0, neginf=0.0)

    # Filter constant features
    stds = physics.std(axis=0)
    valid = stds > 1e-8

    valid_cols = [c for i, c in enumerate(physics_cols) if valid[i]]
    valid_physics = physics[:, valid]

    # Compute normalization stats
    mean = valid_physics.mean(axis=0)
    std = valid_physics.std(axis=0) + 1e-8

    print(f"  {len(valid_cols)} non-constant features")

    return {
        'features': valid_cols,
        'mean': mean,
        'std': std
    }
