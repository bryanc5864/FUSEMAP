import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

class PhysInformerDataset(Dataset):
    """Dataset for PhysInformer training"""
    
    def __init__(
        self,
        descriptors_file: str,
        cell_type: str = None,
        normalize: bool = True,
        load_activities: bool = False
    ):
        """
        Args:
            descriptors_file: Path to descriptors TSV file
            cell_type: Cell type for validation
            normalize: Whether to normalize features
            load_activities: Whether to load activity scores
        """
        # Load descriptors
        self.df = pd.read_csv(descriptors_file, sep='\t')
        self.cell_type = cell_type
        self.load_activities = load_activities
        
        
        # Extract target features - handle different column naming conventions
        # S2 data uses different column names
        exclude_cols = ['seq_id', 'sequence_id', 'condition', 'normalized_log2', 'n_obs_bc',
                       'n_replicates', 'sequence', 'name', 'sequence_length',
                       'Dev_log2_enrichment', 'Hk_log2_enrichment',  # S2 specific activity columns
                       'Dev_log2_enrichment_scaled', 'Hk_log2_enrichment_scaled',
                       'Dev_log2_enrichment_quantile_normalized', 'Hk_log2_enrichment_quantile_normalized',
                       'enrichment_leaf', 'enrichment_proto']  # Plant specific activity columns
        self.descriptor_cols = [c for c in self.df.columns if c not in exclude_cols]
        
        
        # Filter out zero-variance features before normalization
        desc_values = self.df[self.descriptor_cols].values.astype(np.float32)
        feature_stds = desc_values.std(axis=0)
        
        # Find non-zero variance features
        valid_features = feature_stds > 1e-8
        n_removed = np.sum(~valid_features)
        
        if n_removed > 0:
            print(f"Removing {n_removed} zero-variance features")
            # Filter descriptor columns
            self.descriptor_cols = [col for i, col in enumerate(self.descriptor_cols) if valid_features[i]]
            # Update desc_values
            desc_values = desc_values[:, valid_features]
            print(f"Remaining features: {len(self.descriptor_cols)}")
        
        # Compute normalization statistics if needed
        self.normalize = normalize
        if self.normalize:
            self.desc_mean = desc_values.mean(axis=0)
            self.desc_std = desc_values.std(axis=0) + 1e-8  # Avoid division by zero
            
        
        # Load activity scores if requested
        self.activity_cols = []
        self.has_activities = False
        
        if self.load_activities:
            if cell_type == 'S2':
                # S2 has Dev and Hk scores
                activity_cols = ['Dev_log2_enrichment', 'Hk_log2_enrichment']
                if all(col in self.df.columns for col in activity_cols):
                    self.activity_cols = activity_cols
                    self.has_activities = True
            elif cell_type in ['arabidopsis', 'sorghum', 'maize']:
                # Plant species have enrichment_leaf and enrichment_proto
                activity_cols = ['enrichment_leaf', 'enrichment_proto']
                if all(col in self.df.columns for col in activity_cols):
                    self.activity_cols = activity_cols
                    self.has_activities = True
                    print(f"Loaded plant activity columns: {activity_cols}")
            elif cell_type in ['HepG2', 'K562', 'WTC11']:
                # Human cell types have normalized_log2
                if 'normalized_log2' in self.df.columns:
                    self.activity_cols = ['normalized_log2']
                    self.has_activities = True
                else:
                    # Try loading from tileformer file (K562 and WTC11 have it there)
                    import os
                    tileformer_file = descriptors_file.replace('_descriptors', '_tileformer')
                    if os.path.exists(tileformer_file):
                        df_tile = pd.read_csv(tileformer_file, sep='\t')
                        if 'normalized_log2' in df_tile.columns:
                            # Match by name column
                            if 'name' in self.df.columns and 'name' in df_tile.columns:
                                # Create a mapping
                                activity_map = dict(zip(df_tile['name'], df_tile['normalized_log2']))
                                self.df['normalized_log2'] = self.df['name'].map(activity_map)
                                self.activity_cols = ['normalized_log2']
                                self.has_activities = True
                            else:
                                # If no name column, assume same order
                                if len(df_tile) == len(self.df):
                                    self.df['normalized_log2'] = df_tile['normalized_log2'].values
                                    self.activity_cols = ['normalized_log2']
                                    self.has_activities = True
                    
                    # Fallback to activity file for HepG2
                    if not self.has_activities and cell_type == 'HepG2':
                        # Check both filtered and regular activity files
                        activity_file_filtered = descriptors_file.replace('_descriptors_filtered', '_descriptors_with_activity')
                        activity_file_regular = descriptors_file.replace('_descriptors', '_descriptors_with_activity')
                        
                        for activity_file in [activity_file_filtered, activity_file_regular]:
                            if os.path.exists(activity_file):
                                df_activity = pd.read_csv(activity_file, sep='\t')
                                if 'normalized_log2' in df_activity.columns:
                                    # Match by name column if available
                                    if 'name' in self.df.columns and 'name' in df_activity.columns:
                                        activity_map = dict(zip(df_activity['name'], df_activity['normalized_log2']))
                                        self.df['normalized_log2'] = self.df['name'].map(activity_map)
                                    else:
                                        # Assume same order
                                        if len(df_activity) == len(self.df):
                                            self.df['normalized_log2'] = df_activity['normalized_log2'].values
                                    self.activity_cols = ['normalized_log2']
                                    self.has_activities = True
                                    break
        
        print(f"Loaded dataset: {len(self.df)} sequences")
        print(f"Descriptor features: {len(self.descriptor_cols)}")
        if self.normalize:
            print(f"Normalization enabled - descriptors std range: [{self.desc_std.min():.4f}, {self.desc_std.max():.4f}]")
        if self.has_activities:
            print(f"Activity columns loaded: {self.activity_cols}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Convert sequence to indices
        sequence = row['sequence'].upper()
        sequence_indices = self._sequence_to_indices(sequence)
        
        # Get descriptor targets
        descriptors = row[self.descriptor_cols].values.astype(np.float32)
        
        
        # Apply normalization if enabled
        if self.normalize:
            descriptors = (descriptors - self.desc_mean) / self.desc_std
            # Clip extreme outliers at 2.5 standard deviations
            descriptors = np.clip(descriptors, -2.5, 2.5)
        
        result = {
            'sequence': torch.tensor(sequence_indices, dtype=torch.long),
            'descriptors': torch.tensor(descriptors, dtype=torch.float32),
            'seq_id': row.get('seq_id', idx)
        }
        
        # Add activity scores if available
        if self.has_activities:
            activities = row[self.activity_cols].values.astype(np.float32)
            result['activities'] = torch.tensor(activities, dtype=torch.float32)
        
        return result
    
    def _sequence_to_indices(self, sequence: str) -> np.ndarray:
        """Convert DNA sequence to indices for embedding"""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        indices = np.zeros(len(sequence), dtype=np.int64)
        for i, base in enumerate(sequence):
            indices[i] = mapping.get(base, 4)  # Default to N (index 4)
        return indices
    
    def get_feature_names(self):
        """Return feature names for logging"""
        return {
            'descriptors': self.descriptor_cols
        }
    
    def get_normalization_stats(self):
        """Return normalization statistics for denormalization during inference"""
        if not self.normalize:
            return None
        return {
            'desc_mean': torch.tensor(self.desc_mean, dtype=torch.float32),
            'desc_std': torch.tensor(self.desc_std, dtype=torch.float32),
        }

def create_dataloaders(
    cell_type: str,
    data_dir: str = 'output',
    batch_size: int = 32,
    num_workers: int = 4,
    load_activities: bool = True
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders for a cell type
    
    Args:
        cell_type: 'HepG2', 'K562', or 'WTC11'
        data_dir: Directory containing the data files
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        load_activities: Whether to load activity scores for auxiliary training
        
    Returns:
        Dict with 'train', 'val', 'test' dataloaders
    """
    dataloaders = {}
    datasets = {}
    
    # First load all datasets to find common features
    for split in ['train', 'val', 'test']:
        # Check for filtered descriptor file first (with constant features removed)
        desc_file_filtered = f"{data_dir}/{cell_type}_{split}_descriptors_filtered.tsv"
        desc_file_regular = f"{data_dir}/{cell_type}_{split}_descriptors.tsv"
        
        import os
        if os.path.exists(desc_file_filtered):
            desc_file = desc_file_filtered
            print(f"Using filtered descriptor file: {desc_file_filtered}")
        else:
            desc_file = desc_file_regular
            print(f"Using regular descriptor file: {desc_file_regular}")
        
        datasets[split] = PhysInformerDataset(desc_file, cell_type, load_activities=load_activities)
    
    # Find common descriptor columns across all splits
    common_cols = set(datasets['train'].descriptor_cols)
    for split in ['val', 'test']:
        common_cols = common_cols.intersection(set(datasets[split].descriptor_cols))
    
    common_cols = sorted(list(common_cols))  # Keep consistent ordering
    
    # Check if we need to filter
    if len(common_cols) < len(datasets['train'].descriptor_cols):
        print(f"Harmonizing features across splits: using {len(common_cols)} common features")
        
        # Update each dataset to use only common columns
        for split in ['train', 'val', 'test']:
            dataset = datasets[split]
            # Find indices of common columns
            common_indices = [dataset.descriptor_cols.index(col) for col in common_cols]
            
            # Update dataset attributes
            dataset.descriptor_cols = common_cols
            
            # Update normalization stats if present
            if dataset.normalize:
                dataset.desc_mean = dataset.desc_mean[common_indices]
                dataset.desc_std = dataset.desc_std[common_indices]
    
    # Create dataloaders
    for split in ['train', 'val', 'test']:
        dataset = datasets[split]
        
        # Use shuffle only for training
        shuffle = (split == 'train')
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        dataloaders[split] = dataloader

    return dataloaders


def create_plant_dataloaders(
    cell_type: str,
    data_dir: str = 'output',
    batch_size: int = 32,
    num_workers: int = 4,
    load_activities: bool = True,
    val_split: float = 0.1
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders for plant data.

    Plant data only has train/test splits, so we create val from train.

    Args:
        cell_type: 'arabidopsis', 'sorghum', or 'maize'
        data_dir: Directory containing the data files
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        load_activities: Whether to load activity scores for auxiliary training
        val_split: Fraction of training data to use for validation

    Returns:
        Dict with 'train', 'val', 'test' dataloaders
    """
    import os
    from torch.utils.data import Subset

    dataloaders = {}
    datasets = {}

    # Load train and test datasets
    for split in ['train', 'test']:
        # Look for _with_activity files first
        desc_file_activity = f"{data_dir}/{cell_type}_{split}_descriptors_with_activity.tsv"
        desc_file_regular = f"{data_dir}/{cell_type}_{split}_descriptors.tsv"

        if os.path.exists(desc_file_activity):
            desc_file = desc_file_activity
            print(f"Using descriptor file with activities: {desc_file_activity}")
        else:
            desc_file = desc_file_regular
            print(f"Using regular descriptor file: {desc_file_regular}")

        datasets[split] = PhysInformerDataset(desc_file, cell_type, load_activities=load_activities)

    # Find common descriptor columns across train and test
    common_cols = set(datasets['train'].descriptor_cols)
    common_cols = common_cols.intersection(set(datasets['test'].descriptor_cols))
    common_cols = sorted(list(common_cols))

    if len(common_cols) < len(datasets['train'].descriptor_cols):
        print(f"Harmonizing features across splits: using {len(common_cols)} common features")

        for split in ['train', 'test']:
            dataset = datasets[split]
            common_indices = [dataset.descriptor_cols.index(col) for col in common_cols]
            dataset.descriptor_cols = common_cols

            if dataset.normalize:
                dataset.desc_mean = dataset.desc_mean[common_indices]
                dataset.desc_std = dataset.desc_std[common_indices]

    # Split training data into train/val
    train_dataset = datasets['train']
    n_train = len(train_dataset)
    n_val = int(n_train * val_split)
    n_train_new = n_train - n_val

    # Create random indices for splitting
    indices = np.random.permutation(n_train)
    train_indices = indices[:n_train_new]
    val_indices = indices[n_train_new:]

    print(f"Splitting training data: {n_train_new} train, {n_val} val")

    # Create subsets - note: Subset doesn't preserve dataset attributes
    # So we create a wrapper that keeps the original dataset's attributes
    class SubsetWithAttributes(Subset):
        def __init__(self, dataset, indices):
            super().__init__(dataset, indices)
            # Copy attributes from original dataset
            self.descriptor_cols = dataset.descriptor_cols
            self.has_activities = dataset.has_activities
            self.activity_cols = dataset.activity_cols
            self.normalize = dataset.normalize
            if hasattr(dataset, 'desc_mean'):
                self.desc_mean = dataset.desc_mean
                self.desc_std = dataset.desc_std

        def get_feature_names(self):
            return self.dataset.get_feature_names()

        def get_normalization_stats(self):
            return self.dataset.get_normalization_stats()

    train_subset = SubsetWithAttributes(train_dataset, train_indices)
    val_subset = SubsetWithAttributes(train_dataset, val_indices)

    # Create dataloaders
    dataloaders['train'] = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    dataloaders['val'] = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    dataloaders['test'] = torch.utils.data.DataLoader(
        datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return dataloaders