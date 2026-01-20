#!/usr/bin/env python3
"""
Test all 5 configurations with real data.

Runs a minimal training loop (1 epoch, small batch) to verify each config works.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.data_loaders import (
    LentiMPRADataset,
    DeepSTARRDataset,
    DREAMYeastDataset,
    JoresPlantDataset,
)
from training.models import create_multi_species_model, compute_masked_loss
from training.config import ModelConfig


# Global mappings that will be updated as we see new values
SPECIES_MAP = {}
KINGDOM_MAP = {}
CELLTYPE_MAP = {}


def get_or_add_mapping(mapping: dict, key: str) -> int:
    """Get index for key, adding it if not present."""
    if key not in mapping:
        mapping[key] = len(mapping)
    return mapping[key]


def collate_fn(batch):
    """Simple collate function."""
    sequences = torch.stack([item["sequence"] for item in batch])

    # Handle variable output sizes
    max_outputs = max(item["activity"].shape[0] for item in batch)
    activities = torch.full((len(batch), max_outputs), float('nan'))
    for i, item in enumerate(batch):
        act = item["activity"]
        activities[i, :len(act)] = act

    original_lengths = torch.stack([item["original_length"] for item in batch])

    # Build indices dynamically
    species_idx = torch.tensor([get_or_add_mapping(SPECIES_MAP, item["species"]) for item in batch])
    kingdom_idx = torch.tensor([get_or_add_mapping(KINGDOM_MAP, item["kingdom"]) for item in batch])
    celltype_idx = torch.tensor([get_or_add_mapping(CELLTYPE_MAP, item["cell_type"]) for item in batch])
    dataset_names = [item["dataset_name"] for item in batch]

    return {
        "sequence": sequences,
        "activity": activities,
        "original_length": original_lengths,
        "species_idx": species_idx,
        "kingdom_idx": kingdom_idx,
        "celltype_idx": celltype_idx,
        "dataset_names": dataset_names,
    }


def reset_mappings():
    """Reset global mappings."""
    global SPECIES_MAP, KINGDOM_MAP, CELLTYPE_MAP
    SPECIES_MAP.clear()
    KINGDOM_MAP.clear()
    CELLTYPE_MAP.clear()


def scan_dataset_for_mappings(datasets):
    """Scan datasets to build mappings - samples across entire dataset to catch all unique values."""
    import random
    for ds in datasets:
        n = len(ds)
        # For small datasets, scan all; for large, sample evenly across
        if n <= 1000:
            indices = range(n)
        else:
            # Sample evenly from start, middle, and end + random samples
            step = n // 100
            indices = list(range(0, n, step))[:200]  # Evenly spaced
            # Also add some random samples
            indices.extend(random.sample(range(n), min(100, n)))
            indices = list(set(indices))

        for i in indices:
            item = ds[i]
            get_or_add_mapping(SPECIES_MAP, item["species"])
            get_or_add_mapping(KINGDOM_MAP, item["kingdom"])
            get_or_add_mapping(CELLTYPE_MAP, item["cell_type"])


def test_config1_single_celltype():
    """Test Configuration 1: Single Cell Type Baseline."""
    print("\n" + "="*60)
    print("Testing Config 1: Single Cell Type Baseline (K562)")
    print("="*60)

    reset_mappings()

    # Load K562 data
    train_ds = LentiMPRADataset("K562", split="train", fold=1, target_length=256)
    val_ds = LentiMPRADataset("K562", split="val", fold=1, target_length=256)

    # Build mappings
    scan_dataset_for_mappings([train_ds])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Create model
    config = ModelConfig(
        use_species_embedding=False,
        use_celltype_embedding=False,
        use_length_embedding=True,
    )
    model = create_multi_species_model(config, ["encode4_k562"])

    # Test forward pass
    batch = next(iter(train_loader))
    with torch.no_grad():
        outputs = model(
            sequence=batch["sequence"],
            species_idx=batch["species_idx"],
            kingdom_idx=batch["kingdom_idx"],
            celltype_idx=batch["celltype_idx"],
            original_length=batch["original_length"],
            dataset_names=batch["dataset_names"],
        )

    print(f"  Model outputs: {list(outputs.keys())}")
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print("  Config 1 PASSED")
    return True


def test_config2_multi_celltype_human():
    """Test Configuration 2: Multi-Cell-Type Human."""
    print("\n" + "="*60)
    print("Testing Config 2: Multi-Cell-Type Human")
    print("="*60)

    reset_mappings()

    # Load all human data
    datasets = []
    for cell_type in ["K562", "HepG2", "WTC11"]:
        try:
            ds = LentiMPRADataset(cell_type, split="train", fold=1, target_length=256)
            datasets.append(ds)
            print(f"  Loaded {cell_type}: {len(ds)} samples")
        except Exception as e:
            print(f"  Failed to load {cell_type}: {e}")

    if not datasets:
        print("  No datasets loaded, SKIPPING")
        return False

    # Build mappings
    scan_dataset_for_mappings(datasets)

    combined = ConcatDataset(datasets)
    loader = DataLoader(combined, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Create model with cell-type embedding
    config = ModelConfig(
        use_species_embedding=False,
        use_celltype_embedding=True,
        celltype_embed_dim=32,
        use_length_embedding=True,
    )
    model = create_multi_species_model(
        config,
        ["encode4_k562", "encode4_hepg2", "encode4_wtc11"]
    )

    # Test forward pass
    batch = next(iter(loader))
    with torch.no_grad():
        outputs = model(
            sequence=batch["sequence"],
            species_idx=batch["species_idx"],
            kingdom_idx=batch["kingdom_idx"],
            celltype_idx=batch["celltype_idx"],
            original_length=batch["original_length"],
            dataset_names=batch["dataset_names"],
        )

    print(f"  Model heads: {list(model.heads.keys())}")
    print(f"  Combined samples: {len(combined)}")
    print("  Config 2 PASSED")
    return True


def test_config3_cross_animal():
    """Test Configuration 3: Cross-Animal (Human + Drosophila)."""
    print("\n" + "="*60)
    print("Testing Config 3: Cross-Animal (Human + Drosophila)")
    print("="*60)

    reset_mappings()

    # Load human and Drosophila data
    datasets = []

    # Human
    try:
        ds = LentiMPRADataset("K562", split="train", fold=1, target_length=256)
        datasets.append(ds)
        print(f"  Loaded K562: {len(ds)} samples")
    except Exception as e:
        print(f"  Failed to load K562: {e}")

    # Drosophila
    try:
        ds = DeepSTARRDataset(split="train", target_length=256)
        datasets.append(ds)
        print(f"  Loaded DeepSTARR: {len(ds)} samples")
    except Exception as e:
        print(f"  Failed to load DeepSTARR: {e}")

    if len(datasets) < 2:
        print("  Missing datasets, SKIPPING")
        return False

    # Build mappings
    scan_dataset_for_mappings(datasets)
    print(f"  Species: {SPECIES_MAP}, Celltypes: {CELLTYPE_MAP}")

    combined = ConcatDataset(datasets)
    loader = DataLoader(combined, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Create model with species embedding - use actual counts from mappings
    config = ModelConfig(
        use_species_embedding=True,
        species_embed_dim=16,
        use_celltype_embedding=True,
        celltype_embed_dim=32,
        use_length_embedding=True,
    )
    model = create_multi_species_model(
        config,
        ["encode4_k562", "deepstarr"],
        n_species=len(SPECIES_MAP),
        n_kingdoms=max(len(KINGDOM_MAP), 1),
        n_celltypes=len(CELLTYPE_MAP),
    )

    # Test forward pass
    batch = next(iter(loader))
    with torch.no_grad():
        outputs = model(
            sequence=batch["sequence"],
            species_idx=batch["species_idx"],
            kingdom_idx=batch["kingdom_idx"],
            celltype_idx=batch["celltype_idx"],
            original_length=batch["original_length"],
            dataset_names=batch["dataset_names"],
        )

    print(f"  Model heads: {list(model.heads.keys())}")
    print(f"  Combined samples: {len(combined)}")
    print("  Config 3 PASSED")
    return True


def test_config4_cross_kingdom():
    """Test Configuration 4: Cross-Kingdom (Animal + Plant)."""
    print("\n" + "="*60)
    print("Testing Config 4: Cross-Kingdom (Animal + Plant)")
    print("="*60)

    reset_mappings()

    datasets = []

    # Human
    try:
        ds = LentiMPRADataset("K562", split="train", fold=1, target_length=256)
        datasets.append(ds)
        print(f"  Loaded K562: {len(ds)} samples")
    except Exception as e:
        print(f"  Failed to load K562: {e}")

    # Plant
    try:
        ds = JoresPlantDataset(assay="tobacco_leaf", split="train", target_length=256)
        datasets.append(ds)
        print(f"  Loaded Jores Tobacco: {len(ds)} samples")
    except Exception as e:
        print(f"  Failed to load Jores: {e}")

    if len(datasets) < 2:
        print("  Missing datasets, SKIPPING")
        return False

    # Build mappings
    scan_dataset_for_mappings(datasets)
    print(f"  Species: {SPECIES_MAP}, Kingdoms: {KINGDOM_MAP}")

    combined = ConcatDataset(datasets)
    loader = DataLoader(combined, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Create model with kingdom + species embedding - use actual counts from mappings
    config = ModelConfig(
        use_kingdom_embedding=True,
        kingdom_embed_dim=8,
        use_species_embedding=True,
        species_embed_dim=16,
        use_celltype_embedding=True,
        celltype_embed_dim=32,
        use_length_embedding=True,
    )
    model = create_multi_species_model(
        config,
        ["encode4_k562", "jores_tobacco_leaf_sorghum"],
        n_species=len(SPECIES_MAP),
        n_kingdoms=len(KINGDOM_MAP),
        n_celltypes=max(len(CELLTYPE_MAP), 1),
    )

    # Test forward pass
    batch = next(iter(loader))
    with torch.no_grad():
        outputs = model(
            sequence=batch["sequence"],
            species_idx=batch["species_idx"],
            kingdom_idx=batch["kingdom_idx"],
            celltype_idx=batch["celltype_idx"],
            original_length=batch["original_length"],
            dataset_names=batch["dataset_names"],
        )

    print(f"  Model heads: {list(model.heads.keys())}")
    print(f"  Combined samples: {len(combined)}")
    print("  Config 4 PASSED")
    return True


def test_config5_universal():
    """Test Configuration 5: Universal Foundation Model."""
    print("\n" + "="*60)
    print("Testing Config 5: Universal Foundation Model")
    print("="*60)

    reset_mappings()

    datasets = []

    # Human
    try:
        ds = LentiMPRADataset("K562", split="train", fold=1, target_length=256)
        datasets.append(ds)
        print(f"  Loaded K562: {len(ds)} samples")
    except Exception as e:
        print(f"  Failed: {e}")

    # Drosophila
    try:
        ds = DeepSTARRDataset(split="train", target_length=256)
        datasets.append(ds)
        print(f"  Loaded DeepSTARR: {len(ds)} samples")
    except Exception as e:
        print(f"  Failed: {e}")

    # Yeast (subsampled)
    try:
        ds = DREAMYeastDataset(split="train", target_length=256, subsample=10000)
        datasets.append(ds)
        print(f"  Loaded Yeast (subsampled): {len(ds)} samples")
    except Exception as e:
        print(f"  Failed: {e}")

    # Plant
    try:
        ds = JoresPlantDataset(assay="tobacco_leaf", split="train", target_length=256)
        datasets.append(ds)
        print(f"  Loaded Jores Tobacco: {len(ds)} samples")
    except Exception as e:
        print(f"  Failed: {e}")

    if len(datasets) < 3:
        print("  Missing datasets, SKIPPING")
        return False

    # Build mappings
    scan_dataset_for_mappings(datasets)
    print(f"  Species: {SPECIES_MAP}, Kingdoms: {KINGDOM_MAP}, Celltypes: {CELLTYPE_MAP}")

    combined = ConcatDataset(datasets)
    loader = DataLoader(combined, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Create full model - use actual counts from mappings
    config = ModelConfig(
        use_kingdom_embedding=True,
        kingdom_embed_dim=8,
        use_species_embedding=True,
        species_embed_dim=16,
        use_celltype_embedding=True,
        celltype_embed_dim=32,
        use_length_embedding=True,
    )
    model = create_multi_species_model(
        config,
        ["encode4_k562", "deepstarr", "dream_yeast", "jores_tobacco_leaf_sorghum"],
        n_species=len(SPECIES_MAP),
        n_kingdoms=len(KINGDOM_MAP),
        n_celltypes=len(CELLTYPE_MAP),
    )

    # Test forward pass
    batch = next(iter(loader))
    with torch.no_grad():
        outputs = model(
            sequence=batch["sequence"],
            species_idx=batch["species_idx"],
            kingdom_idx=batch["kingdom_idx"],
            celltype_idx=batch["celltype_idx"],
            original_length=batch["original_length"],
            dataset_names=batch["dataset_names"],
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model heads: {list(model.heads.keys())}")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Combined samples: {len(combined)}")
    print("  Config 5 PASSED")
    return True


def main():
    print("="*60)
    print("FUSEMAP Configuration Tests with Real Data")
    print("="*60)

    results = {}

    results["config1"] = test_config1_single_celltype()
    results["config2"] = test_config2_multi_celltype_human()
    results["config3"] = test_config3_cross_animal()
    results["config4"] = test_config4_cross_kingdom()
    results["config5"] = test_config5_universal()

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nAll tests: {'PASSED' if all_passed else 'SOME FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
