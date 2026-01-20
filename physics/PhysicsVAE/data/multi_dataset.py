"""
Multi-dataset loader for PhysicsVAE.

Combines multiple cell types with aligned physics features and padded sequences.
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class AlignedMultiDataset(Dataset):
    """
    Dataset that loads from multiple cell types with aligned physics features.

    Handles:
    - Different sequence lengths (pads to max_length)
    - Different physics features (uses intersection)
    - Consistent normalization across all data
    """

    def __init__(
        self,
        data_files: List[str],
        cell_types: List[str],
        max_seq_length: int,
        physics_prefixes: List[str] = None,
        exclude_prefixes: List[str] = None,
        normalize_physics: bool = True,
        pad_mode: str = 'center'  # 'center', 'right', 'left'
    ):
        """
        Args:
            data_files: List of TSV file paths
            cell_types: List of cell type names (same order as data_files)
            max_seq_length: Maximum sequence length (pad shorter seqs)
            physics_prefixes: Only include features with these prefixes
            exclude_prefixes: Exclude features with these prefixes
            normalize_physics: Whether to z-score normalize
            pad_mode: How to pad shorter sequences
        """
        self.max_seq_length = max_seq_length
        self.physics_prefixes = physics_prefixes or ['thermo', 'stiff', 'bend', 'entropy', 'advanced']
        self.exclude_prefixes = exclude_prefixes or ['pwm']
        self.normalize_physics = normalize_physics
        self.pad_mode = pad_mode

        # Load all dataframes
        print(f"Loading {len(data_files)} datasets...")
        dfs = []
        for f, ct in zip(data_files, cell_types):
            print(f"  Loading {ct} from {Path(f).name}...")
            df = pd.read_csv(f, sep='\t')
            df['_cell_type'] = ct
            dfs.append(df)

        # Find common physics features
        self.physics_cols = self._find_common_physics_cols(dfs)
        print(f"  Common physics features: {len(self.physics_cols)}")

        # Combine all data
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"  Total samples: {len(self.df)}")

        # Extract sequences
        self.sequences = self.df['sequence'].values
        self.cell_types = self.df['_cell_type'].values

        # Extract and normalize physics features
        self.physics = self.df[self.physics_cols].values.astype(np.float32)
        self.physics = np.nan_to_num(self.physics, nan=0.0, posinf=0.0, neginf=0.0)

        # Remove constant features
        stds = self.physics.std(axis=0)
        valid = stds > 1e-8
        n_removed = np.sum(~valid)
        if n_removed > 0:
            print(f"  Removing {n_removed} constant features")
            self.physics_cols = [c for i, c in enumerate(self.physics_cols) if valid[i]]
            self.physics = self.physics[:, valid]

        self.n_physics_features = len(self.physics_cols)

        if self.normalize_physics:
            self.physics_mean = self.physics.mean(axis=0)
            self.physics_std = self.physics.std(axis=0) + 1e-8
            self.physics = (self.physics - self.physics_mean) / self.physics_std

        print(f"  Final physics features: {self.n_physics_features}")

    def _find_common_physics_cols(self, dfs: List[pd.DataFrame]) -> List[str]:
        """Find physics columns common to all dataframes."""
        # Get physics cols from first df
        common_cols = None

        for df in dfs:
            # Filter by prefix
            physics_cols = []
            for col in df.columns:
                # Check if should include
                include = any(col.startswith(p) for p in self.physics_prefixes)
                exclude = any(col.startswith(p) for p in self.exclude_prefixes)

                if include and not exclude:
                    physics_cols.append(col)

            if common_cols is None:
                common_cols = set(physics_cols)
            else:
                common_cols = common_cols.intersection(physics_cols)

        return sorted(list(common_cols))

    def _sequence_to_indices(self, sequence: str) -> np.ndarray:
        """Convert DNA sequence to indices with padding."""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

        seq_len = len(sequence)
        indices = np.full(self.max_seq_length, 4, dtype=np.int64)  # Fill with N

        if seq_len >= self.max_seq_length:
            # Truncate (center)
            start = (seq_len - self.max_seq_length) // 2
            sequence = sequence[start:start + self.max_seq_length]
            for i, base in enumerate(sequence):
                indices[i] = mapping.get(base.upper(), 4)
        else:
            # Pad
            if self.pad_mode == 'center':
                start = (self.max_seq_length - seq_len) // 2
            elif self.pad_mode == 'left':
                start = self.max_seq_length - seq_len
            else:  # right
                start = 0

            for i, base in enumerate(sequence):
                indices[start + i] = mapping.get(base.upper(), 4)

        return indices

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        physics = self.physics[idx]

        seq_indices = self._sequence_to_indices(sequence)

        return {
            'sequence': torch.tensor(seq_indices, dtype=torch.long),
            'physics': torch.tensor(physics, dtype=torch.float32),
            'cell_type': self.cell_types[idx],
            'idx': idx
        }

    def get_physics_feature_names(self) -> List[str]:
        """Return list of physics feature names."""
        return self.physics_cols


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_multi_dataloaders(
    config_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.1
) -> Tuple[Dict[str, DataLoader], Dict]:
    """
    Create dataloaders from a multi-dataset config.

    Args:
        config_path: Path to YAML config file
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_split: Validation split ratio

    Returns:
        Tuple of (dataloaders dict, config dict)
    """
    config = load_config(config_path)
    config_dir = Path(config_path).parent.parent  # PhysicsVAE directory

    # Collect data files for each split
    train_files = []
    val_files = []
    test_files = []
    cell_types = []

    for ds in config['datasets']:
        data_dir = config_dir / ds['data_dir']
        cell_type = ds['cell_type']
        pattern = ds['file_pattern']

        train_file = data_dir / pattern.format(cell_type=cell_type, split='train')
        val_file = data_dir / pattern.format(cell_type=cell_type, split='val')
        test_file = data_dir / pattern.format(cell_type=cell_type, split='test')

        if train_file.exists():
            train_files.append(str(train_file))
            cell_types.append(ds['name'])

        if val_file.exists():
            val_files.append(str(val_file))

        if test_file.exists():
            test_files.append(str(test_file))

    # Get config values
    max_seq_length = config['sequence']['max_length']
    physics_prefixes = config['physics'].get('include_prefixes', ['thermo', 'stiff', 'bend', 'entropy', 'advanced'])
    exclude_prefixes = config['physics'].get('exclude_prefixes', ['pwm'])

    dataloaders = {}

    # Training dataset
    print("\n=== Loading Training Data ===")
    train_dataset = AlignedMultiDataset(
        train_files, cell_types, max_seq_length,
        physics_prefixes=physics_prefixes,
        exclude_prefixes=exclude_prefixes
    )

    # Create val split if no separate val files
    if len(val_files) == len(train_files):
        print("\n=== Loading Validation Data ===")
        val_cell_types = [ds['name'] for ds in config['datasets'] if (config_dir / ds['data_dir'] / ds['file_pattern'].format(cell_type=ds['cell_type'], split='val')).exists()]
        val_dataset = AlignedMultiDataset(
            val_files, val_cell_types, max_seq_length,
            physics_prefixes=physics_prefixes,
            exclude_prefixes=exclude_prefixes
        )
    else:
        # Split training data
        n_train = len(train_dataset)
        n_val = int(n_train * val_split)
        indices = np.random.permutation(n_train)
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        print(f"\n  Split: {len(train_indices)} train, {len(val_indices)} val")

    # Test dataset
    if len(test_files) == len(train_files):
        print("\n=== Loading Test Data ===")
        test_cell_types = [ds['name'] for ds in config['datasets']]
        test_dataset = AlignedMultiDataset(
            test_files, test_cell_types, max_seq_length,
            physics_prefixes=physics_prefixes,
            exclude_prefixes=exclude_prefixes
        )
    else:
        test_dataset = None

    # Create dataloaders
    dataloaders['train'] = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    dataloaders['val'] = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    if test_dataset:
        dataloaders['test'] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    # Return config info
    if hasattr(train_dataset, 'n_physics_features'):
        n_physics = train_dataset.n_physics_features
    else:
        n_physics = train_dataset.dataset.n_physics_features

    config['_computed'] = {
        'n_physics_features': n_physics,
        'max_seq_length': max_seq_length,
        'n_datasets': len(cell_types),
        'cell_types': cell_types
    }

    return dataloaders, config
