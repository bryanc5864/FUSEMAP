"""
FUSEMAP Dataset Classes

Handles multiple datasets with:
- Length padding/cropping to fixed size
- Activity normalization (per-dataset z-score)
- Multi-output handling
- Metadata for conditioning (species, cell type, etc.)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import h5py
import json
from dataclasses import dataclass

from .config import DatasetInfo, DATASET_CATALOG


@dataclass
class ActivityNormalizerStats:
    """Statistics for activity normalization."""
    mean: np.ndarray
    std: np.ndarray
    min_val: np.ndarray
    max_val: np.ndarray


class ActivityNormalizer:
    """
    Normalize activity values per dataset to standard scale.
    """

    def __init__(self):
        self.stats: Dict[str, ActivityNormalizerStats] = {}

    def fit(self, dataset_name: str, activities: np.ndarray):
        """
        Compute mean/std for a dataset.

        Args:
            dataset_name: Name of the dataset
            activities: Array of activities [N] or [N, num_outputs]
        """
        if activities.ndim == 1:
            activities = activities.reshape(-1, 1)

        self.stats[dataset_name] = ActivityNormalizerStats(
            mean=np.nanmean(activities, axis=0),
            std=np.nanstd(activities, axis=0),
            min_val=np.nanmin(activities, axis=0),
            max_val=np.nanmax(activities, axis=0),
        )

        # Prevent division by zero
        self.stats[dataset_name].std = np.maximum(
            self.stats[dataset_name].std, 1e-8
        )

    def transform(
        self,
        dataset_name: str,
        activities: np.ndarray,
    ) -> np.ndarray:
        """Z-score normalize."""
        if dataset_name not in self.stats:
            return activities

        s = self.stats[dataset_name]

        if activities.ndim == 1:
            return (activities - s.mean[0]) / s.std[0]

        return (activities - s.mean) / s.std

    def inverse_transform(
        self,
        dataset_name: str,
        normalized: np.ndarray,
    ) -> np.ndarray:
        """Convert back to original scale."""
        if dataset_name not in self.stats:
            return normalized

        s = self.stats[dataset_name]

        if normalized.ndim == 1:
            return normalized * s.std[0] + s.mean[0]

        return normalized * s.std + s.mean

    def save(self, filepath: str):
        """Save normalizer stats to file."""
        data = {
            name: {
                "mean": stats.mean.tolist(),
                "std": stats.std.tolist(),
                "min": stats.min_val.tolist(),
                "max": stats.max_val.tolist(),
            }
            for name, stats in self.stats.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load normalizer stats from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        for name, vals in data.items():
            self.stats[name] = ActivityNormalizerStats(
                mean=np.array(vals["mean"]),
                std=np.array(vals["std"]),
                min_val=np.array(vals["min"]),
                max_val=np.array(vals["max"]),
            )


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


def pad_or_crop_sequence(
    sequence: np.ndarray,
    target_length: int,
    pad_value: float = 0.25,
) -> Tuple[np.ndarray, int]:
    """
    Pad or crop sequence to target length.

    Args:
        sequence: One-hot encoded sequence [4, L] or [L, 4]
        target_length: Desired length
        pad_value: Value for padding (0.25 = uniform)

    Returns:
        (padded_sequence, original_length)
    """
    # Ensure [4, L] format
    if sequence.shape[0] != 4:
        sequence = sequence.T

    original_length = sequence.shape[1]

    if original_length == target_length:
        return sequence, original_length

    if original_length > target_length:
        # Center crop
        start = (original_length - target_length) // 2
        return sequence[:, start:start + target_length], original_length

    # Pad symmetrically
    pad_left = (target_length - original_length) // 2
    pad_right = target_length - original_length - pad_left

    padded = np.full((4, target_length), pad_value, dtype=sequence.dtype)
    padded[:, pad_left:pad_left + original_length] = sequence

    return padded, original_length


class SingleDataset(Dataset):
    """
    Dataset for a single data source.

    Handles loading, normalization, and length adjustment.
    Applies shift augmentation (±21bp) only for human MPRA datasets.
    """

    # Shift augmentation parameters (matching HumanLegNet)
    MAX_SHIFT = 21  # ±21bp shift

    def __init__(
        self,
        dataset_info: DatasetInfo,
        split: str = "train",
        target_length: int = 256,
        normalizer: Optional[ActivityNormalizer] = None,
        fold: Optional[int] = None,  # For k-fold CV
        transform: Optional[callable] = None,
        index_mappings: Optional[Dict[str, Dict[str, int]]] = None,  # Pre-built mappings
        use_augmentation: bool = True,  # Reverse complement augmentation (random 50% flip)
        double_data_with_rc: bool = False,  # False = random 50% RC (like HumanLegNet), True = deterministic doubling
        use_shift: bool = True,  # Shift augmentation (only for human MPRA)
    ):
        self.info = dataset_info
        self.split = split
        self.target_length = target_length
        self.normalizer = normalizer
        self.fold = fold
        self.transform = transform
        # Double dataset by including both original and RC for each sample
        self.double_data_with_rc = double_data_with_rc and (split == "train")
        # Only use RANDOM augmentation during training AND when NOT doubling
        # (If doubling, we deterministically have original in first half and RC in second half)
        self.use_augmentation = use_augmentation and (split == "train") and not self.double_data_with_rc
        # Shift augmentation: enabled for training, will be applied only for human MPRA
        self._use_shift = use_shift and (split == "train")

        # Determine if this is a human MPRA dataset (for shift augmentation)
        # Will be updated in _load_from_real_data() if needed
        self._is_human_mpra = dataset_info.name.lower().startswith("encode4_")

        # Build or use provided index mappings
        if index_mappings is not None:
            self.species_to_idx = index_mappings.get("species", {dataset_info.species: 0})
            self.kingdom_to_idx = index_mappings.get("kingdom", {dataset_info.kingdom: 0})
            self.celltype_to_idx = index_mappings.get("celltype", {(dataset_info.cell_type or "unknown"): 0})
        else:
            # Default single-dataset mappings
            self.species_to_idx = {dataset_info.species: 0}
            self.kingdom_to_idx = {dataset_info.kingdom: 0}
            self.celltype_to_idx = {(dataset_info.cell_type or "unknown"): 0}

        # Load data
        self._load_data()

        # Fit normalizer if training and not already fitted
        if split == "train" and normalizer is not None:
            if dataset_info.name not in normalizer.stats:
                normalizer.fit(dataset_info.name, self.activities)

    def _load_data(self):
        """Load data from files or use real data loaders."""
        data_path = Path(self.info.path)

        # Try different file formats
        if (data_path / f"{self.split}.h5").exists():
            self._load_h5(data_path / f"{self.split}.h5")
        elif (data_path / f"{self.split}.npz").exists():
            self._load_npz(data_path / f"{self.split}.npz")
        elif (data_path / f"{self.split}_sequences.npy").exists():
            self._load_npy(data_path, self.split)
        else:
            # Use real data loaders from data_loaders.py
            self._load_from_real_data()

    def _load_h5(self, filepath: Path):
        """Load from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            self.sequences = np.array(f['sequences'])
            self.activities = np.array(f['activities'])

            if 'weights' in f:
                self.weights = np.array(f['weights'])
            else:
                self.weights = None

    def _load_npz(self, filepath: Path):
        """Load from NPZ file."""
        data = np.load(filepath, allow_pickle=True)
        self.sequences = data['sequences']
        self.activities = data['activities']
        self.weights = data.get('weights', None)

    def _load_npy(self, data_path: Path, split: str):
        """Load from separate NPY files."""
        self.sequences = np.load(data_path / f"{split}_sequences.npy")
        self.activities = np.load(data_path / f"{split}_activities.npy")

        weights_path = data_path / f"{split}_weights.npy"
        self.weights = np.load(weights_path) if weights_path.exists() else None

    def _load_from_real_data(self):
        """Load data using real data loaders from data_loaders.py.

        IMPORTANT: We disable augmentation here because data gets cached in self.sequences.
        Augmentation (shift, RC) is applied dynamically in __getitem__() instead.
        """
        from .data_loaders import (
            LentiMPRADataset, DeepSTARRDataset, DREAMYeastDataset, JoresPlantDataset
        )

        dataset_name = self.info.name.lower()
        print(f"Loading real data for {dataset_name} ({self.split})...")

        # Map dataset names to loaders
        # DISABLE augmentation since we cache data and apply augmentation in __getitem__
        if dataset_name.startswith("encode4_"):
            # Map to correct directory names (case-sensitive)
            cell_type_map = {
                "k562": "K562",
                "hepg2": "HepG2",
                "wtc11": "WTC11",
                "joint": "K562",  # Joint uses K562 data
            }
            cell_type_lower = dataset_name.replace("encode4_", "").lower()
            cell_type = cell_type_map.get(cell_type_lower, cell_type_lower.upper())
            real_ds = LentiMPRADataset(
                cell_type=cell_type,
                split=self.split,
                fold=self.fold or 1,
                target_length=self.target_length,
                normalize=False,  # We'll normalize ourselves
                use_augmentation=False,  # Disable RC - applied in __getitem__
                use_shift=False,  # Disable shift - applied in __getitem__
            )
            # Track that this is a human MPRA dataset for shift augmentation
            self._is_human_mpra = True
        elif dataset_name == "deepstarr":
            real_ds = DeepSTARRDataset(
                split=self.split,
                target_length=self.target_length,
                normalize=False,
                use_augmentation=False,  # Disable RC - applied in __getitem__
            )
            self._is_human_mpra = False
        elif dataset_name == "dream_yeast":
            # Subsample yeast for memory
            subsample = 100000 if self.split == "train" else None
            # DREAM yeast has train/val/test splits (no separate "eval" file)
            # The "special" validation scheme uses the val set during training
            real_ds = DREAMYeastDataset(
                split=self.split,  # Use split directly: train/val/test
                target_length=self.target_length,
                normalize=False,
                subsample=subsample,
                use_augmentation=False,  # Disable RC - applied in __getitem__
            )
            self._is_human_mpra = False
        elif dataset_name.startswith("jores_"):
            # Determine species from dataset name (jores_arabidopsis, jores_maize, jores_sorghum)
            # Each species has 2 targets: enrichment_leaf and enrichment_proto
            species = dataset_name.replace("jores_", "")

            real_ds = JoresPlantDataset(
                species=species,
                split=self.split,  # Jores has train/val/test splits
                target_length=self.target_length,
                normalize=False,
                use_augmentation=False,  # Disable RC - applied in __getitem__
            )
            self._is_human_mpra = False
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Extract sequences and activities from real dataset
        n_samples = len(real_ds)
        print(f"  Loaded {n_samples} samples")

        # Pre-allocate arrays
        sample = real_ds[0]
        seq_shape = sample["sequence"].shape
        act_shape = sample["activity"].shape

        self.sequences = np.zeros((n_samples, *seq_shape), dtype=np.float32)
        self.activities = np.zeros((n_samples, *act_shape), dtype=np.float32)

        # Load all data
        for i in range(n_samples):
            item = real_ds[i]
            self.sequences[i] = item["sequence"].numpy()
            self.activities[i] = item["activity"].numpy()

        # Squeeze activities if single output
        if self.activities.shape[-1] == 1:
            self.activities = self.activities.squeeze(-1)

        self.weights = None
        print(f"  Sequences shape: {self.sequences.shape}, Activities shape: {self.activities.shape}")

    def _create_dummy_data(self):
        """Create dummy data for testing (DEPRECATED - use _load_from_real_data)."""
        print("WARNING: Creating dummy data - real data loaders not available!")
        n_samples = 1000 if self.split == "train" else 200
        seq_len = self.info.sequence_length
        n_outputs = self.info.num_outputs

        # Random one-hot sequences
        self.sequences = np.zeros((n_samples, 4, seq_len), dtype=np.float32)
        random_bases = np.random.randint(0, 4, (n_samples, seq_len))
        for i in range(n_samples):
            for j in range(seq_len):
                self.sequences[i, random_bases[i, j], j] = 1.0

        # Random activities
        self.activities = np.random.randn(n_samples, n_outputs).astype(np.float32)
        if n_outputs == 1:
            self.activities = self.activities.squeeze(-1)

        self.weights = None

    def __len__(self) -> int:
        base_len = len(self.sequences)
        # Double the dataset if using RC doubling
        if self.double_data_with_rc:
            return base_len * 2
        return base_len

    def _apply_shift_augmentation(self, seq: np.ndarray) -> np.ndarray:
        """
        Apply random shift augmentation (±21bp) to sequence.

        Only used for human MPRA datasets to match HumanLegNet training.
        Shifts the sequence randomly, padding with N bases (uniform 0.25).
        """
        seq_len = seq.shape[1]
        shift = np.random.randint(-self.MAX_SHIFT, self.MAX_SHIFT + 1)

        if shift == 0:
            return seq

        result = np.full((4, seq_len), 0.25, dtype=np.float32)

        if shift > 0:
            # Shift right: pad start with N, take from beginning
            result[:, shift:] = seq[:, :seq_len - shift]
        else:
            # Shift left: pad end with N, take from end
            shift = abs(shift)
            result[:, :seq_len - shift] = seq[:, shift:]

        return result

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_len = len(self.sequences)

        # Handle doubled dataset: first half is original, second half is RC
        if self.double_data_with_rc and idx >= base_len:
            real_idx = idx - base_len
            use_rc = True
        else:
            real_idx = idx
            use_rc = False

        # Get sequence
        seq = self.sequences[real_idx].copy()  # Copy to avoid modifying original
        original_length = seq.shape[1]  # Track original length before any augmentation

        # Apply shift augmentation FIRST (only for human MPRA datasets during training)
        if self._use_shift and self._is_human_mpra:
            seq = self._apply_shift_augmentation(seq)

        # Apply reverse complement
        if use_rc:
            # Deterministic RC for second half of doubled dataset
            seq = reverse_complement_onehot(seq)
        elif self.use_augmentation and np.random.random() > 0.5:
            # Random RC augmentation (only if not already doubled)
            seq = reverse_complement_onehot(seq)

        # Pad/crop to target length
        seq, _ = pad_or_crop_sequence(seq, self.target_length)

        # Get activity and normalize
        activity = self.activities[real_idx]
        if self.normalizer is not None:
            activity = self.normalizer.transform(self.info.name, activity)

        # Apply transform if any
        if self.transform is not None:
            seq = self.transform(seq)

        # Get cell type
        cell_type = self.info.cell_type or "unknown"

        # Build output with index mappings
        item = {
            "sequence": torch.from_numpy(seq).float(),
            "activity": torch.tensor(activity).float(),
            "original_length": torch.tensor(original_length),
            "dataset_name": self.info.name,
            "species": self.info.species,
            "kingdom": self.info.kingdom,
            "cell_type": cell_type,
            # Add index mappings for conditioning
            "species_idx": torch.tensor(self.species_to_idx.get(self.info.species, 0)),
            "kingdom_idx": torch.tensor(self.kingdom_to_idx.get(self.info.kingdom, 0)),
            "celltype_idx": torch.tensor(self.celltype_to_idx.get(cell_type, 0)),
        }

        if self.weights is not None:
            item["weight"] = torch.tensor(self.weights[real_idx]).float()

        return item


class MultiDataset(Dataset):
    """
    Combined dataset from multiple sources.

    Handles indexing across datasets and provides metadata for conditioning.
    """

    def __init__(
        self,
        dataset_names: List[str],
        split: str = "train",
        target_length: int = 256,
        normalizer: Optional[ActivityNormalizer] = None,
        transform: Optional[callable] = None,
    ):
        self.dataset_names = dataset_names
        self.split = split
        self.target_length = target_length
        self.normalizer = normalizer or ActivityNormalizer()

        # Build species/kingdom/celltype mappings
        self._build_mappings()

        # Load all datasets
        self.datasets: Dict[str, SingleDataset] = {}
        self.cumulative_sizes = [0]

        for name in dataset_names:
            if name not in DATASET_CATALOG:
                print(f"Warning: Dataset {name} not in catalog, skipping")
                continue

            info = DATASET_CATALOG[name]
            dataset = SingleDataset(
                dataset_info=info,
                split=split,
                target_length=target_length,
                normalizer=self.normalizer,
                transform=transform,
            )
            self.datasets[name] = dataset
            self.cumulative_sizes.append(
                self.cumulative_sizes[-1] + len(dataset)
            )

        self.total_size = self.cumulative_sizes[-1]

    def _build_mappings(self):
        """Build mappings for species, kingdom, cell type."""
        species_set = set()
        kingdom_set = set()
        celltype_set = set()

        for name in self.dataset_names:
            if name not in DATASET_CATALOG:
                continue
            info = DATASET_CATALOG[name]
            species_set.add(info.species)
            kingdom_set.add(info.kingdom)
            if info.cell_type:
                celltype_set.add(info.cell_type)

        self.species_to_idx = {s: i for i, s in enumerate(sorted(species_set))}
        self.kingdom_to_idx = {k: i for i, k in enumerate(sorted(kingdom_set))}
        self.celltype_to_idx = {c: i for i, c in enumerate(sorted(celltype_set))}

        # Add unknown
        self.celltype_to_idx["unknown"] = len(self.celltype_to_idx)

    def get_dataset_sizes(self) -> Dict[str, int]:
        """Get size of each dataset."""
        return {name: len(ds) for name, ds in self.datasets.items()}

    def __len__(self) -> int:
        return self.total_size

    def _find_dataset(self, idx: int) -> Tuple[str, int]:
        """Find which dataset an index belongs to."""
        for i, (name, dataset) in enumerate(self.datasets.items()):
            if idx < self.cumulative_sizes[i + 1]:
                local_idx = idx - self.cumulative_sizes[i]
                return name, local_idx
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_name, local_idx = self._find_dataset(idx)
        item = self.datasets[dataset_name][local_idx]

        # Add index mappings
        item["species_idx"] = torch.tensor(
            self.species_to_idx[item["species"]]
        )
        item["kingdom_idx"] = torch.tensor(
            self.kingdom_to_idx[item["kingdom"]]
        )
        item["celltype_idx"] = torch.tensor(
            self.celltype_to_idx[item["cell_type"]]
        )

        return item

    def get_by_dataset(
        self,
        dataset_name: str,
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Get item by dataset name and local index."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")

        item = self.datasets[dataset_name][idx]

        item["species_idx"] = torch.tensor(
            self.species_to_idx[item["species"]]
        )
        item["kingdom_idx"] = torch.tensor(
            self.kingdom_to_idx[item["kingdom"]]
        )
        item["celltype_idx"] = torch.tensor(
            self.celltype_to_idx[item["cell_type"]]
        )

        return item


def collate_multi_dataset(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for multi-dataset batches.

    Handles variable-sized outputs by padding with NaN.
    """
    # Stack simple tensors
    sequences = torch.stack([item["sequence"] for item in batch])
    original_lengths = torch.stack([item["original_length"] for item in batch])
    species_idx = torch.stack([item["species_idx"] for item in batch])
    kingdom_idx = torch.stack([item["kingdom_idx"] for item in batch])
    celltype_idx = torch.stack([item["celltype_idx"] for item in batch])

    # Collect dataset names
    dataset_names = [item["dataset_name"] for item in batch]

    # Handle activities with different output sizes
    max_outputs = max(
        item["activity"].numel() for item in batch
    )

    activities = torch.full((len(batch), max_outputs), float('nan'))
    for i, item in enumerate(batch):
        act = item["activity"]
        if act.dim() == 0:
            activities[i, 0] = act
        else:
            activities[i, :len(act)] = act

    # Handle optional weights
    if "weight" in batch[0]:
        weights = torch.stack([item.get("weight", torch.tensor(1.0)) for item in batch])
    else:
        weights = None

    result = {
        "sequence": sequences,
        "activity": activities,
        "original_length": original_lengths,
        "species_idx": species_idx,
        "kingdom_idx": kingdom_idx,
        "celltype_idx": celltype_idx,
        "dataset_names": dataset_names,
    }

    if weights is not None:
        result["weight"] = weights

    return result


def get_validation_loader(
    dataset_name: str,
    target_length: int = 256,
    batch_size: int = 128,
    normalizer: Optional[ActivityNormalizer] = None,
    index_mappings: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[torch.utils.data.DataLoader, str]:
    """
    Get validation data loader based on dataset's validation scheme.

    Args:
        dataset_name: Name of the dataset
        target_length: Target sequence length
        batch_size: Batch size for loader
        normalizer: Activity normalizer (optional)
        index_mappings: Pre-built mappings for species, kingdom, celltype indices

    Returns:
        (DataLoader, validation_type) where validation_type is
        'val', 'test', or 'kfold'
    """
    if dataset_name not in DATASET_CATALOG:
        raise ValueError(f"Dataset {dataset_name} not found")

    info = DATASET_CATALOG[dataset_name]
    scheme = info.validation_scheme

    if scheme == "standard":
        # Use validation set if exists, else test
        try:
            dataset = SingleDataset(
                info, "val", target_length, normalizer,
                index_mappings=index_mappings
            )
            val_type = "val"
        except:
            dataset = SingleDataset(
                info, "test", target_length, normalizer,
                index_mappings=index_mappings
            )
            val_type = "test"

    elif scheme == "kfold":
        # For k-fold, use val split (held-out fold) during training
        # The trainer calls this for validation during training epochs
        dataset = SingleDataset(
            info, "val", target_length, normalizer,
            index_mappings=index_mappings
        )
        val_type = "val"

    elif scheme == "chromosome_holdout":
        # DeepSTARR: use val set (pre-split by original authors)
        dataset = SingleDataset(
            info, "val", target_length, normalizer,
            index_mappings=index_mappings
        )
        val_type = "val"

    elif scheme == "special":
        # DREAM yeast - use the val set (not a separate "eval" file)
        dataset = SingleDataset(
            info, "val", target_length, normalizer,
            index_mappings=index_mappings
        )
        val_type = "val"

    else:
        dataset = SingleDataset(
            info, "test", target_length, normalizer,
            index_mappings=index_mappings
        )
        val_type = "test"

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda x: collate_multi_dataset(x),
    )

    return loader, val_type
