"""
Dataset for PhysicsVAE training.

Loads sequences and their corresponding physics features from PhysInformer outputs.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class PhysicsVAEDataset(Dataset):
    """
    Dataset for PhysicsVAE: sequences + physics features.

    Loads from descriptor TSV files containing:
    - name: sequence identifier
    - sequence: DNA sequence
    - physics features (pwm_*, thermo_*, stiff_*, etc.)
    """

    def __init__(
        self,
        descriptor_file: str,
        cell_type: str = None,
        normalize_physics: bool = True,
        max_seq_length: Optional[int] = None
    ):
        """
        Args:
            descriptor_file: Path to descriptor TSV file
            cell_type: Cell type identifier
            normalize_physics: Whether to z-score normalize physics features
            max_seq_length: Maximum sequence length (truncate/pad if needed)
        """
        self.cell_type = cell_type
        self.normalize_physics = normalize_physics

        # Load data
        print(f"Loading {descriptor_file}...")
        self.df = pd.read_csv(descriptor_file, sep='\t')

        # Identify physics columns (everything except name, sequence, and metadata)
        exclude_cols = ['name', 'seq_id', 'sequence_id', 'sequence', 'condition',
                        'sequence_length', 'normalized_log2', 'n_obs_bc', 'n_replicates',
                        'cell_type', 'activity', 'is_reverse',
                        # Drosophila activity columns
                        'Dev_log2_enrichment', 'Hk_log2_enrichment',
                        'Dev_log2_enrichment_scaled', 'Hk_log2_enrichment_scaled',
                        'Dev_log2_enrichment_quantile_normalized', 'Hk_log2_enrichment_quantile_normalized',
                        # Plant activity columns
                        'enrichment_leaf', 'enrichment_proto']
        self.physics_cols = [c for c in self.df.columns if c not in exclude_cols]

        # Get sequences
        self.sequences = self.df['sequence'].values

        # Determine sequence length
        seq_lengths = [len(s) for s in self.sequences]
        self.native_seq_length = max(seq_lengths)
        self.seq_length = max_seq_length or self.native_seq_length

        # Extract physics features
        self.physics = self.df[self.physics_cols].values.astype(np.float32)

        # Handle NaN/Inf values
        self.physics = np.nan_to_num(self.physics, nan=0.0, posinf=0.0, neginf=0.0)

        # Filter out constant features
        feature_stds = self.physics.std(axis=0)
        valid_features = feature_stds > 1e-8
        n_removed = np.sum(~valid_features)
        if n_removed > 0:
            print(f"Removing {n_removed} constant features")
            self.physics_cols = [c for i, c in enumerate(self.physics_cols) if valid_features[i]]
            self.physics = self.physics[:, valid_features]

        self.n_physics_features = len(self.physics_cols)

        # Normalize physics features
        if self.normalize_physics:
            self.physics_mean = self.physics.mean(axis=0)
            self.physics_std = self.physics.std(axis=0) + 1e-8
            self.physics = (self.physics - self.physics_mean) / self.physics_std

        print(f"Loaded {len(self.df)} sequences")
        print(f"Sequence length: {self.seq_length}")
        print(f"Physics features: {self.n_physics_features}")

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

        # Convert sequence to indices
        seq_indices = self._sequence_to_indices(sequence)

        return {
            'sequence': torch.tensor(seq_indices, dtype=torch.long),
            'physics': torch.tensor(physics, dtype=torch.float32),
            'idx': idx
        }

    def get_physics_stats(self) -> Dict[str, np.ndarray]:
        """Return normalization statistics."""
        if self.normalize_physics:
            return {
                'mean': self.physics_mean,
                'std': self.physics_std
            }
        return None

    def denormalize_physics(self, physics: np.ndarray) -> np.ndarray:
        """Convert normalized physics back to original scale."""
        if self.normalize_physics:
            return physics * self.physics_std + self.physics_mean
        return physics


def create_dataloaders(
    cell_type: str,
    data_dir: str = '../output',
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.1
) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders for PhysicsVAE.

    Args:
        cell_type: One of 'K562', 'HepG2', 'WTC11', 'S2', 'arabidopsis', 'sorghum', 'maize'
        data_dir: Directory containing descriptor files
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_split: Fraction of training data for validation

    Returns:
        Dict with 'train', 'val', 'test' dataloaders
    """
    from .aligned_dataset import AlignedPhysicsVAEDataset

    data_dir = Path(data_dir)
    dataloaders = {}

    # Plant datasets use _with_activity suffix
    plant_types = ['arabidopsis', 'sorghum', 'maize']
    is_plant = cell_type.lower() in plant_types

    if is_plant:
        # Plant file pattern
        train_file = data_dir / f"{cell_type}_train_descriptors_with_activity.tsv"
        val_file = data_dir / f"{cell_type}_val_descriptors_with_activity.tsv"
        test_file = data_dir / f"{cell_type}_test_descriptors_with_activity.tsv"
    else:
        # Animal file pattern
        train_file = data_dir / f"{cell_type}_train_descriptors.tsv"
        val_file = data_dir / f"{cell_type}_val_descriptors.tsv"
        test_file = data_dir / f"{cell_type}_test_descriptors.tsv"

    # Check which files exist
    has_val = val_file.exists()
    has_test = test_file.exists()

    # Load training data
    print(f"\nLoading training data for {cell_type}...")
    train_dataset = PhysicsVAEDataset(str(train_file), cell_type=cell_type)

    # Get reference feature info from training data for alignment
    train_physics_cols = train_dataset.physics_cols
    train_physics_mean = train_dataset.physics_mean if train_dataset.normalize_physics else np.zeros(len(train_physics_cols))
    train_physics_std = train_dataset.physics_std if train_dataset.normalize_physics else np.ones(len(train_physics_cols))
    train_seq_length = train_dataset.seq_length

    # Create validation split if no separate val file
    if has_val:
        print(f"\nLoading validation data (aligned to training features)...")
        val_dataset = AlignedPhysicsVAEDataset(
            str(val_file),
            reference_features=train_physics_cols,
            reference_mean=train_physics_mean,
            reference_std=train_physics_std,
            max_seq_length=train_seq_length
        )
    else:
        # Split training data
        n_train = len(train_dataset)
        n_val = int(n_train * val_split)
        n_train_new = n_train - n_val

        indices = np.random.permutation(n_train)
        train_indices = indices[:n_train_new]
        val_indices = indices[n_train_new:]

        # Use Subset for val (same features as train since same file)
        full_train_dataset = train_dataset
        train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)
        print(f"Split training data: {n_train_new} train, {n_val} val")

    # Load test data with aligned features
    if has_test:
        print(f"\nLoading test data (aligned to training features)...")
        test_dataset = AlignedPhysicsVAEDataset(
            str(test_file),
            reference_features=train_physics_cols,
            reference_mean=train_physics_mean,
            reference_std=train_physics_std,
            max_seq_length=train_seq_length
        )
    else:
        test_dataset = None

    # Create dataloaders
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    dataloaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    if test_dataset:
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return dataloaders


def get_physics_feature_count(cell_type: str, data_dir: str = '../output') -> int:
    """Get number of physics features for a cell type."""
    data_dir = Path(data_dir)

    # Plant datasets use _with_activity suffix
    plant_types = ['arabidopsis', 'sorghum', 'maize']
    is_plant = cell_type.lower() in plant_types

    if is_plant:
        train_file = data_dir / f"{cell_type}_train_descriptors_with_activity.tsv"
    else:
        train_file = data_dir / f"{cell_type}_train_descriptors.tsv"

    # Read just header to count columns
    df = pd.read_csv(train_file, sep='\t', nrows=0)
    exclude_cols = ['name', 'seq_id', 'sequence_id', 'sequence', 'condition',
                    'sequence_length', 'normalized_log2', 'n_obs_bc', 'n_replicates',
                    'cell_type', 'activity', 'is_reverse',
                    'Dev_log2_enrichment', 'Hk_log2_enrichment',
                    'Dev_log2_enrichment_scaled', 'Hk_log2_enrichment_scaled',
                    'Dev_log2_enrichment_quantile_normalized', 'Hk_log2_enrichment_quantile_normalized',
                    'enrichment_leaf', 'enrichment_proto']
    physics_cols = [c for c in df.columns if c not in exclude_cols]
    return len(physics_cols)
