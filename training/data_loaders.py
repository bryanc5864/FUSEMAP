"""
FUSEMAP Real Data Loaders

Loaders for actual data formats in the FUSEMAP data directory:
- lentiMPRA (K562, HepG2, WTC11): TSV with fold splits
- DeepSTARR (S2): TSV with Dev and Hk activities
- DREAM Yeast: TSV with expression levels
- Jores Plant: TSV with species and activities

Includes shift augmentation (±21bp) matching HumanLegNet implementation.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import gzip


# Default shift augmentation parameters (matching HumanLegNet)
DEFAULT_MAX_SHIFT = 21


# Base data directory
DATA_ROOT = Path("/home/bcheng/sequence_optimization/FUSEMAP/data")


def one_hot_encode(sequence: str) -> np.ndarray:
    """Convert DNA sequence to one-hot encoding [4, L]."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    seq_len = len(sequence)
    one_hot = np.zeros((4, seq_len), dtype=np.float32)

    for i, base in enumerate(sequence.upper()):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
        else:
            # Unknown base - uniform
            one_hot[:, i] = 0.25

    return one_hot


def reverse_complement_onehot(seq: np.ndarray) -> np.ndarray:
    """
    Get reverse complement of one-hot encoded sequence.

    For one-hot encoding [A, C, G, T] = [0, 1, 2, 3]:
    - A <-> T means swap channels 0 and 3
    - C <-> G means swap channels 1 and 2
    - Then reverse the sequence order

    Args:
        seq: One-hot encoded sequence [4, L]
    Returns:
        Reverse complement [4, L]
    """
    # Complement: swap A<->T (0<->3) and C<->G (1<->2)
    complement = seq[[3, 2, 1, 0], :]
    # Reverse
    return complement[:, ::-1].copy()


def pad_or_crop(seq: np.ndarray, target_length: int) -> Tuple[np.ndarray, int]:
    """Pad or crop sequence to target length."""
    original_length = seq.shape[1]

    if original_length == target_length:
        return seq, original_length

    if original_length > target_length:
        start = (original_length - target_length) // 2
        return seq[:, start:start + target_length], original_length

    pad_left = (target_length - original_length) // 2
    pad_right = target_length - original_length - pad_left
    padded = np.full((4, target_length), 0.25, dtype=np.float32)
    padded[:, pad_left:pad_left + original_length] = seq

    return padded, original_length


def apply_shift_augmentation(
    seq: np.ndarray,
    max_shift: int = DEFAULT_MAX_SHIFT,
    target_length: Optional[int] = None
) -> np.ndarray:
    """
    Apply random shift augmentation to sequence.

    Shifts the sequence randomly by ±max_shift positions, padding with
    N bases (uniform 0.25) on the opposite side. This simulates viewing
    different windows of the regulatory element, matching HumanLegNet.

    Args:
        seq: One-hot encoded sequence [4, L]
        max_shift: Maximum shift in either direction (default 21bp)
        target_length: Target length after shift (default: same as input)

    Returns:
        Shifted sequence [4, L] or [4, target_length]
    """
    seq_len = seq.shape[1]
    if target_length is None:
        target_length = seq_len

    # Random shift in range [-max_shift, +max_shift]
    shift = np.random.randint(-max_shift, max_shift + 1)

    if shift == 0:
        if seq_len == target_length:
            return seq
        return pad_or_crop(seq, target_length)[0]

    # Create output array with N padding (uniform 0.25)
    result = np.full((4, seq_len), 0.25, dtype=np.float32)

    if shift > 0:
        # Shift right: take sequence from start, pad end with N
        # This views upstream context more
        result[:, shift:] = seq[:, :seq_len - shift]
    else:
        # Shift left: take sequence from end, pad start with N
        # This views downstream context more
        shift = abs(shift)
        result[:, :seq_len - shift] = seq[:, shift:]

    # Crop/pad to target length if different
    if seq_len != target_length:
        result = pad_or_crop(result, target_length)[0]

    return result


class LentiMPRADataset(Dataset):
    """
    ENCODE4 lentiMPRA dataset (K562, HepG2, WTC11).

    Uses pre-defined train/val/test/calibration splits from PhysiFormer preprocessing.
    Supports shift augmentation (±21bp) and reverse complement augmentation.
    """

    # Path to pre-split data files
    PRESPLIT_DATA_ROOT = Path("/home/bcheng/sequence_optimization/mainproject/PhysiFormer/physpreprocess/data/lentiMPRA_data")

    def __init__(
        self,
        cell_type: str,  # K562, HepG2, WTC11
        split: str = "train",  # train, val, test, calibration
        fold: int = 1,  # Ignored - kept for backwards compatibility
        target_length: int = 256,
        normalize: bool = True,
        use_augmentation: bool = True,  # Reverse complement augmentation
        use_shift: bool = True,  # Shift augmentation (±21bp)
        max_shift: int = DEFAULT_MAX_SHIFT,  # Maximum shift amount
    ):
        self.cell_type = cell_type
        self.split = split
        self.fold = fold
        self.target_length = target_length
        self.max_shift = max_shift
        # Only use augmentation during training
        self.use_augmentation = use_augmentation and (split == "train")
        self.use_shift = use_shift and (split == "train")

        # Load data from pre-split files
        data_path = self.PRESPLIT_DATA_ROOT / cell_type
        split_file = data_path / f"{cell_type}_{split}_with_features.tsv"

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.data = pd.read_csv(split_file, sep='\t')

        # Column names: 'sequence' and 'activity' in pre-split files
        self.seq_col = 'sequence'
        self.activity_col = 'activity'

        # Normalize
        self.mean = 0.0
        self.std = 1.0
        if normalize:
            self.mean = self.data[self.activity_col].mean()
            self.std = self.data[self.activity_col].std()

        aug_info = []
        if self.use_shift:
            aug_info.append(f"shift±{max_shift}")
        if self.use_augmentation:
            aug_info.append("RC")
        aug_str = f" ({', '.join(aug_info)})" if aug_info else ""
        print(f"LentiMPRA {cell_type} {split}: {len(self.data)} samples{aug_str}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]

        # Get sequence
        seq_str = row[self.seq_col]
        seq = one_hot_encode(seq_str)
        orig_len = seq.shape[1]

        # Apply shift augmentation first (before RC)
        if self.use_shift:
            seq = apply_shift_augmentation(seq, max_shift=self.max_shift)

        # Apply reverse complement augmentation with 50% probability
        if self.use_augmentation and np.random.random() > 0.5:
            seq = reverse_complement_onehot(seq)

        # Pad/crop to target length
        seq, _ = pad_or_crop(seq, self.target_length)

        # Get activity
        activity = (row[self.activity_col] - self.mean) / (self.std + 1e-8)

        return {
            "sequence": torch.from_numpy(seq).float(),
            "activity": torch.tensor([activity]).float(),
            "original_length": torch.tensor(orig_len),
            "dataset_name": f"encode4_{self.cell_type.lower()}",
            "species": "human",
            "kingdom": "animal",
            "cell_type": self.cell_type,
        }


class DeepSTARRDataset(Dataset):
    """
    DeepSTARR Drosophila S2 dataset.

    Data format: TSV with columns: sequence, Dev_log2_enrichment, Hk_log2_enrichment
    Has two outputs: Dev and Hk enhancers.
    No shift augmentation (only for human datasets).
    """

    def __init__(
        self,
        split: str = "train",  # train, val, test, calib
        target_length: int = 256,
        normalize: bool = True,
        use_augmentation: bool = True,  # Reverse complement augmentation
    ):
        self.split = split
        self.target_length = target_length
        # Only use augmentation during training
        self.use_augmentation = use_augmentation and (split == "train")

        # Use TSV files in splits directory
        data_path = DATA_ROOT / "S2_data" / "splits"

        # Map split names to file names
        split_map = {
            "train": "train",
            "val": "val",
            "test": "test",
            "calibration": "val",  # Use val as calibration if no separate file
        }
        file_name = split_map.get(split, split)

        # Load data from TSV
        tsv_file = data_path / f"{file_name}.tsv"
        self.data = pd.read_csv(tsv_file, sep='\t')

        # Get sequences
        self.sequences = self.data['sequence'].tolist()

        # Normalize
        self.dev_mean, self.dev_std = 0.0, 1.0
        self.hk_mean, self.hk_std = 0.0, 1.0

        if normalize:
            self.dev_mean = self.data['Dev_log2_enrichment'].mean()
            self.dev_std = self.data['Dev_log2_enrichment'].std()
            self.hk_mean = self.data['Hk_log2_enrichment'].mean()
            self.hk_std = self.data['Hk_log2_enrichment'].std()

        aug_str = " (RC)" if self.use_augmentation else ""
        print(f"DeepSTARR {split}: {len(self.sequences)} samples{aug_str}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get sequence
        seq_str = self.sequences[idx]
        seq = one_hot_encode(seq_str)
        orig_len = seq.shape[1]

        # Apply reverse complement augmentation with 50% probability
        if self.use_augmentation and np.random.random() > 0.5:
            seq = reverse_complement_onehot(seq)

        seq, _ = pad_or_crop(seq, self.target_length)

        # Get activities (2 outputs)
        row = self.data.iloc[idx]
        dev = (row['Dev_log2_enrichment'] - self.dev_mean) / (self.dev_std + 1e-8)
        hk = (row['Hk_log2_enrichment'] - self.hk_mean) / (self.hk_std + 1e-8)

        return {
            "sequence": torch.from_numpy(seq).float(),
            "activity": torch.tensor([dev, hk]).float(),
            "original_length": torch.tensor(orig_len),
            "dataset_name": "deepstarr",
            "species": "drosophila",
            "kingdom": "animal",
            "cell_type": "S2",
        }


class DREAMYeastDataset(Dataset):
    """
    DREAM Yeast promoter dataset.

    Very large dataset (6.7M), supports subsampling.
    No shift augmentation (only for human datasets).

    Splits:
    - train: 99% of yeast_train.txt (~6.64M samples)
    - val: yeast_val.txt (~34k samples)
    - test: yeast_test.txt (~71k samples) - SPECIAL, do not touch
    - calibration: 1% of yeast_train.txt (~67k samples)
    """

    # Calibration fraction (1% of training data)
    CALIB_FRACTION = 0.01

    def __init__(
        self,
        split: str = "train",
        target_length: int = 110,
        normalize: bool = True,
        subsample: Optional[int] = None,  # Subsample to this many
        use_augmentation: bool = True,  # Reverse complement augmentation
    ):
        self.split = split
        self.target_length = target_length
        self.subsample = subsample
        # Only use augmentation during training
        self.use_augmentation = use_augmentation and (split == "train")

        data_path = DATA_ROOT / "yeast_data"

        # Handle calibration split (carved from training data)
        if split == "calibration":
            # Load training data and take 1%
            self.data = pd.read_csv(data_path / "yeast_train.txt", sep='\t')
            if 'label' in self.data.columns:
                self.data = self.data.rename(columns={'label': 'activity'})
            # Use fixed seed for reproducibility - take 1% as calibration
            n_calib = int(len(self.data) * self.CALIB_FRACTION)
            np.random.seed(42)
            calib_indices = np.random.choice(len(self.data), n_calib, replace=False)
            self.data = self.data.iloc[calib_indices].reset_index(drop=True)
        elif split == "train":
            # Load training data and exclude calibration indices
            self.data = pd.read_csv(data_path / "yeast_train.txt", sep='\t')
            if 'label' in self.data.columns:
                self.data = self.data.rename(columns={'label': 'activity'})
            # Use same seed to get same calibration indices, then exclude them
            n_calib = int(len(self.data) * self.CALIB_FRACTION)
            np.random.seed(42)
            calib_indices = set(np.random.choice(len(self.data), n_calib, replace=False))
            train_indices = [i for i in range(len(self.data)) if i not in calib_indices]
            self.data = self.data.iloc[train_indices].reset_index(drop=True)
        else:
            # val or test - load directly
            file_name = f"yeast_{split}.txt"
            self.data = pd.read_csv(data_path / file_name, sep='\t')
            # Handle different column names
            if 'label' in self.data.columns:
                self.data = self.data.rename(columns={'label': 'activity'})
            elif 'maude_expression' in self.data.columns:
                # Test set uses different column name
                self.data = self.data.rename(columns={'maude_expression': 'activity'})

        # Subsample if requested
        if subsample and len(self.data) > subsample:
            self.data = self.data.sample(n=subsample, random_state=42).reset_index(drop=True)

        # Normalize
        self.mean, self.std = 0.0, 1.0
        if normalize:
            self.mean = self.data['activity'].mean()
            self.std = self.data['activity'].std()

        aug_str = " (RC)" if self.use_augmentation else ""
        print(f"DREAM Yeast {split}: {len(self.data)} samples{aug_str}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]

        # Get sequence
        seq_str = row['sequence']
        seq = one_hot_encode(seq_str)

        # Apply reverse complement augmentation with 50% probability
        if self.use_augmentation and np.random.random() > 0.5:
            seq = reverse_complement_onehot(seq)

        seq, orig_len = pad_or_crop(seq, self.target_length)

        # Get activity
        activity = (row['activity'] - self.mean) / (self.std + 1e-8)

        return {
            "sequence": torch.from_numpy(seq).float(),
            "activity": torch.tensor([activity]).float(),
            "original_length": torch.tensor(orig_len),
            "dataset_name": "dream_yeast",
            "species": "yeast",
            "kingdom": "fungi",
            "cell_type": "yeast",
        }


class JoresPlantDataset(Dataset):
    """
    Jores 2021 Plant promoter dataset.

    Has Arabidopsis, Maize, Sorghum promoters tested in BOTH
    tobacco leaf (enrichment_leaf) and maize protoplast (enrichment_proto) assays.
    Returns 2 activity targets per sequence.
    """

    def __init__(
        self,
        species: str,  # arabidopsis, maize, or sorghum
        split: str = "train",
        target_length: int = 170,
        normalize: bool = True,
        use_augmentation: bool = True,  # Reverse complement augmentation
    ):
        self.species = species.lower()
        self.split = split
        self.target_length = target_length
        # Only use augmentation during training
        self.use_augmentation = use_augmentation and (split == "train")

        # Data is in jores2021/processed/{species}/{species}_{split}.tsv
        data_path = DATA_ROOT / "plant_data" / "jores2021" / "processed" / self.species
        file_name = f"{self.species}_{split}.tsv"

        # Load data
        self.data = pd.read_csv(data_path / file_name, sep='\t')

        # Two activity columns: enrichment_leaf and enrichment_proto
        self.activity_cols = ['enrichment_leaf', 'enrichment_proto']

        # Normalize each activity separately
        self.means = {}
        self.stds = {}
        if normalize:
            for col in self.activity_cols:
                self.means[col] = self.data[col].mean()
                self.stds[col] = self.data[col].std()
        else:
            for col in self.activity_cols:
                self.means[col] = 0.0
                self.stds[col] = 1.0

        print(f"Jores {self.species} {split}: {len(self.data)} samples, 2 targets (leaf, proto)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]

        # Get sequence
        seq_str = row['sequence']
        seq = one_hot_encode(seq_str)

        # Apply reverse complement augmentation with 50% probability
        if self.use_augmentation and np.random.random() > 0.5:
            seq = reverse_complement_onehot(seq)

        seq, orig_len = pad_or_crop(seq, self.target_length)

        # Get both activities (normalized)
        activities = []
        for col in self.activity_cols:
            val = (row[col] - self.means[col]) / (self.stds[col] + 1e-8)
            activities.append(val)

        return {
            "sequence": torch.from_numpy(seq).float(),
            "activity": torch.tensor(activities).float(),  # Shape: [2]
            "original_length": torch.tensor(orig_len),
            "dataset_name": f"jores_{self.species}",
            "species": self.species,
            "kingdom": "plant",
            "cell_type": "dual_assay",
        }


def get_dataset(
    dataset_name: str,
    split: str = "train",
    target_length: int = 256,
    **kwargs,
) -> Dataset:
    """
    Factory function to get a dataset by name.

    Args:
        dataset_name: One of encode4_k562, encode4_hepg2, encode4_wtc11,
                      deepstarr, dream_yeast, jores_tobacco_*, jores_maize_*
        split: train, val, or test
        target_length: Target sequence length
        **kwargs: Additional arguments for specific datasets

    Returns:
        Dataset instance
    """
    name_lower = dataset_name.lower()

    if name_lower.startswith("encode4_"):
        cell_type = name_lower.replace("encode4_", "").upper()
        return LentiMPRADataset(
            cell_type=cell_type,
            split=split,
            target_length=target_length,
            **kwargs,
        )

    elif name_lower == "deepstarr":
        return DeepSTARRDataset(
            split=split,
            target_length=target_length,
            **kwargs,
        )

    elif name_lower == "dream_yeast":
        return DREAMYeastDataset(
            split=split,
            target_length=target_length,
            **kwargs,
        )

    elif name_lower.startswith("jores_"):
        # Format: jores_{species} where species is arabidopsis, maize, or sorghum
        # Each species has 2 targets: enrichment_leaf and enrichment_proto
        species = name_lower.replace("jores_", "")

        return JoresPlantDataset(
            species=species,
            split=split,
            target_length=target_length,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def test_data_loaders():
    """Test all data loaders."""
    print("Testing data loaders...\n")

    # Test LentiMPRA
    try:
        ds = LentiMPRADataset("K562", split="train", fold=1, target_length=256)
        sample = ds[0]
        print(f"  K562 sample: seq={sample['sequence'].shape}, "
              f"activity={sample['activity']}, "
              f"len={sample['original_length']}")
    except Exception as e:
        print(f"  K562 failed: {e}")

    # Test DeepSTARR
    try:
        ds = DeepSTARRDataset(split="train", target_length=256)
        sample = ds[0]
        print(f"  DeepSTARR sample: seq={sample['sequence'].shape}, "
              f"activity={sample['activity']}, "
              f"len={sample['original_length']}")
    except Exception as e:
        print(f"  DeepSTARR failed: {e}")

    # Test DREAM Yeast
    try:
        ds = DREAMYeastDataset(split="train", target_length=128, subsample=10000)
        sample = ds[0]
        print(f"  DREAM Yeast sample: seq={sample['sequence'].shape}, "
              f"activity={sample['activity']}, "
              f"len={sample['original_length']}")
    except Exception as e:
        print(f"  DREAM Yeast failed: {e}")

    # Test Jores Plant
    try:
        ds = JoresPlantDataset(assay="tobacco_leaf", split="train", target_length=256)
        sample = ds[0]
        print(f"  Jores Tobacco sample: seq={sample['sequence'].shape}, "
              f"activity={sample['activity']}, "
              f"species={sample['species']}")
    except Exception as e:
        print(f"  Jores Tobacco failed: {e}")

    print("\nData loader tests complete!")


if __name__ == "__main__":
    test_data_loaders()
