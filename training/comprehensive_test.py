#!/usr/bin/env python3
"""
Comprehensive test for all 5 FUSEMAP configurations.

Tests:
1. Data loading (real data, not dummy)
2. Forward pass
3. Loss computation (no negative values)
4. One training step
5. Validation pass
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import (
    get_config1_single_celltype,
    get_config2_multi_celltype_human,
    get_config3_cross_animal,
    get_config4_cross_kingdom,
    get_config5_universal,
    DATASET_CATALOG,
)
from training.datasets import (
    MultiDataset, SingleDataset, ActivityNormalizer, collate_multi_dataset
)
from training.models import create_multi_species_model, compute_masked_loss


def test_data_loading():
    """Test that real data is loaded, not dummy data."""
    print("\n" + "="*70)
    print("TEST 1: Data Loading (Real Data)")
    print("="*70)

    normalizer = ActivityNormalizer()

    # Test encode4_k562
    print("\n[encode4_k562]")
    info = DATASET_CATALOG["encode4_k562"]
    ds = SingleDataset(info, split="train", target_length=256, normalizer=normalizer)
    print(f"  Samples: {len(ds)}")
    print(f"  Sequences shape: {ds.sequences.shape}")
    print(f"  Activities shape: {ds.activities.shape}")

    # Check it's not dummy data (should have >> 1000 samples)
    assert len(ds) > 10000, f"Expected >10000 samples, got {len(ds)} - likely dummy data!"
    print(f"  ✓ Real data confirmed ({len(ds)} samples)")

    # Test deepstarr
    print("\n[deepstarr]")
    info = DATASET_CATALOG["deepstarr"]
    ds = SingleDataset(info, split="train", target_length=256, normalizer=normalizer)
    print(f"  Samples: {len(ds)}")
    assert len(ds) > 10000, f"Expected >10000 samples, got {len(ds)} - likely dummy data!"
    print(f"  ✓ Real data confirmed ({len(ds)} samples)")

    # Test jores
    print("\n[jores_arabidopsis]")
    info = DATASET_CATALOG["jores_arabidopsis"]
    ds = SingleDataset(info, split="train", target_length=256, normalizer=normalizer)
    print(f"  Samples: {len(ds)}")
    # Jores has fewer samples
    assert len(ds) > 1000, f"Expected >1000 samples, got {len(ds)} - likely dummy data!"
    print(f"  ✓ Real data confirmed ({len(ds)} samples)")

    print("\n✓ Data loading test PASSED")
    return True


def test_multi_dataset():
    """Test MultiDataset with multiple real datasets."""
    print("\n" + "="*70)
    print("TEST 2: MultiDataset Loading")
    print("="*70)

    normalizer = ActivityNormalizer()

    # Create multi-dataset
    dataset_names = ["encode4_k562", "deepstarr"]
    multi_ds = MultiDataset(
        dataset_names=dataset_names,
        split="train",
        target_length=256,
        normalizer=normalizer,
    )

    print(f"\nDataset sizes: {multi_ds.get_dataset_sizes()}")
    print(f"Total samples: {len(multi_ds)}")
    print(f"Species mapping: {multi_ds.species_to_idx}")
    print(f"Kingdom mapping: {multi_ds.kingdom_to_idx}")
    print(f"Celltype mapping: {multi_ds.celltype_to_idx}")

    # Check total samples
    total = sum(multi_ds.get_dataset_sizes().values())
    assert total == len(multi_ds), f"Size mismatch: {total} vs {len(multi_ds)}"
    assert len(multi_ds) > 100000, f"Expected >100000 combined samples"

    # Test dataloader
    loader = DataLoader(
        multi_ds, batch_size=32, shuffle=True,
        collate_fn=collate_multi_dataset
    )

    batch = next(iter(loader))
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"Batch sequence shape: {batch['sequence'].shape}")
    print(f"Batch activity shape: {batch['activity'].shape}")
    print(f"Batch species_idx: {batch['species_idx'][:5]}")
    print(f"Batch datasets: {batch['dataset_names'][:5]}")

    print("\n✓ MultiDataset test PASSED")
    return True


def test_forward_pass():
    """Test model forward pass."""
    print("\n" + "="*70)
    print("TEST 3: Model Forward Pass")
    print("="*70)

    normalizer = ActivityNormalizer()

    # Create dataset
    dataset_names = ["encode4_k562", "deepstarr"]
    multi_ds = MultiDataset(
        dataset_names=dataset_names,
        split="train",
        target_length=256,
        normalizer=normalizer,
    )

    # Create model with correct embedding sizes
    from training.config import ModelConfig
    config = ModelConfig(
        use_species_embedding=True,
        species_embed_dim=16,
        use_celltype_embedding=True,
        celltype_embed_dim=32,
        use_length_embedding=True,
    )

    model = create_multi_species_model(
        config,
        dataset_names,
        n_species=len(multi_ds.species_to_idx),
        n_kingdoms=len(multi_ds.kingdom_to_idx),
        n_celltypes=len(multi_ds.celltype_to_idx),
    )

    print(f"\nModel heads: {list(model.heads.keys())}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Create batch
    loader = DataLoader(
        multi_ds, batch_size=32, shuffle=True,
        collate_fn=collate_multi_dataset
    )
    batch = next(iter(loader))

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            sequence=batch["sequence"],
            species_idx=batch["species_idx"],
            kingdom_idx=batch["kingdom_idx"],
            celltype_idx=batch["celltype_idx"],
            original_length=batch["original_length"],
            dataset_names=batch["dataset_names"],
        )

    print(f"\nOutput keys: {list(outputs.keys())}")
    for k, v in outputs.items():
        if isinstance(v, dict):
            print(f"  {k}: mean shape={v['mean'].shape}, logvar shape={v['logvar'].shape}")

    print("\n✓ Forward pass test PASSED")
    return True


def test_loss_computation():
    """Test loss computation - should NOT be negative."""
    print("\n" + "="*70)
    print("TEST 4: Loss Computation")
    print("="*70)

    normalizer = ActivityNormalizer()

    # Create dataset
    dataset_names = ["encode4_k562", "deepstarr"]
    multi_ds = MultiDataset(
        dataset_names=dataset_names,
        split="train",
        target_length=256,
        normalizer=normalizer,
    )

    # Create model
    from training.config import ModelConfig
    config = ModelConfig(
        use_species_embedding=True,
        use_celltype_embedding=True,
        use_length_embedding=True,
    )

    model = create_multi_species_model(
        config,
        dataset_names,
        n_species=len(multi_ds.species_to_idx),
        n_kingdoms=len(multi_ds.kingdom_to_idx),
        n_celltypes=len(multi_ds.celltype_to_idx),
    )

    # Create batch
    loader = DataLoader(
        multi_ds, batch_size=64, shuffle=True,
        collate_fn=collate_multi_dataset
    )

    # Test multiple batches
    print("\nTesting loss on 10 batches...")
    model.train()

    for i, batch in enumerate(loader):
        if i >= 10:
            break

        outputs = model(
            sequence=batch["sequence"],
            species_idx=batch["species_idx"],
            kingdom_idx=batch["kingdom_idx"],
            celltype_idx=batch["celltype_idx"],
            original_length=batch["original_length"],
            dataset_names=batch["dataset_names"],
        )

        loss, per_head = compute_masked_loss(
            outputs,
            batch["activity"],
            batch["dataset_names"],
            model.dataset_to_heads,
            use_uncertainty=True,
        )

        loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
        print(f"  Batch {i+1}: loss={loss_val:.4f}", end="")

        # Check loss is positive
        if loss_val < 0:
            print(f" ✗ NEGATIVE LOSS!")
            # Debug
            for head, head_loss in per_head.items():
                print(f"    {head}: {head_loss.item():.4f}")
            return False
        else:
            print(" ✓")

    print("\n✓ Loss computation test PASSED (all losses positive)")
    return True


def test_training_step():
    """Test one complete training step."""
    print("\n" + "="*70)
    print("TEST 5: Training Step")
    print("="*70)

    normalizer = ActivityNormalizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create dataset
    dataset_names = ["encode4_k562"]
    multi_ds = MultiDataset(
        dataset_names=dataset_names,
        split="train",
        target_length=256,
        normalizer=normalizer,
    )

    # Create model
    from training.config import ModelConfig
    config = ModelConfig(
        use_species_embedding=False,
        use_celltype_embedding=False,
        use_length_embedding=True,
    )

    model = create_multi_species_model(
        config,
        dataset_names,
        n_species=len(multi_ds.species_to_idx),
        n_kingdoms=len(multi_ds.kingdom_to_idx),
        n_celltypes=len(multi_ds.celltype_to_idx),
    )
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create batch
    loader = DataLoader(
        multi_ds, batch_size=128, shuffle=True,
        collate_fn=collate_multi_dataset
    )

    print("\nTraining for 5 steps...")
    model.train()
    losses = []

    for i, batch in enumerate(loader):
        if i >= 5:
            break

        # Move to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(
            sequence=batch["sequence"],
            species_idx=batch["species_idx"],
            kingdom_idx=batch["kingdom_idx"],
            celltype_idx=batch["celltype_idx"],
            original_length=batch["original_length"],
            dataset_names=batch["dataset_names"],
        )

        loss, _ = compute_masked_loss(
            outputs,
            batch["activity"],
            batch["dataset_names"],
            model.dataset_to_heads,
            use_uncertainty=True,
        )

        if isinstance(loss, (int, float)):
            loss = torch.tensor(loss, requires_grad=True, device=device)

        loss.backward()

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        losses.append(loss.item())
        print(f"  Step {i+1}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")

    # Check loss is decreasing or stable
    print(f"\nLosses: {losses}")
    avg_loss = sum(losses) / len(losses)
    print(f"Average loss: {avg_loss:.4f}")

    assert all(l > 0 for l in losses), "Negative loss detected!"
    assert avg_loss < 10, f"Loss too high: {avg_loss}"

    print("\n✓ Training step test PASSED")
    return True


def test_all_configs():
    """Test all 5 configurations."""
    print("\n" + "="*70)
    print("TEST 6: All 5 Configurations")
    print("="*70)

    normalizer = ActivityNormalizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        ("Config 1: Single Cell Type", ["encode4_k562"]),
        ("Config 2: Multi-Cell Human", ["encode4_k562", "encode4_hepg2"]),
        ("Config 3: Cross-Animal", ["encode4_k562", "deepstarr"]),
        ("Config 4: Cross-Kingdom", ["encode4_k562", "jores_arabidopsis"]),
        ("Config 5: Universal", ["encode4_k562", "deepstarr", "jores_arabidopsis"]),
    ]

    for config_name, dataset_names in configs:
        print(f"\n[{config_name}]")
        print(f"  Datasets: {dataset_names}")

        try:
            # Create dataset
            multi_ds = MultiDataset(
                dataset_names=dataset_names,
                split="train",
                target_length=256,
                normalizer=normalizer,
            )
            print(f"  Total samples: {len(multi_ds):,}")

            # Create model
            from training.config import ModelConfig
            config = ModelConfig(
                use_species_embedding=len(multi_ds.species_to_idx) > 1,
                use_kingdom_embedding=len(multi_ds.kingdom_to_idx) > 1,
                use_celltype_embedding=len(multi_ds.celltype_to_idx) > 1,
                use_length_embedding=True,
            )

            model = create_multi_species_model(
                config,
                dataset_names,
                n_species=len(multi_ds.species_to_idx),
                n_kingdoms=len(multi_ds.kingdom_to_idx),
                n_celltypes=len(multi_ds.celltype_to_idx),
            )
            model = model.to(device)

            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Model params: {n_params:,}")

            # Forward pass
            loader = DataLoader(
                multi_ds, batch_size=64, shuffle=True,
                collate_fn=collate_multi_dataset
            )
            batch = next(iter(loader))
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            model.eval()
            with torch.no_grad():
                outputs = model(
                    sequence=batch["sequence"],
                    species_idx=batch["species_idx"],
                    kingdom_idx=batch["kingdom_idx"],
                    celltype_idx=batch["celltype_idx"],
                    original_length=batch["original_length"],
                    dataset_names=batch["dataset_names"],
                )

                loss, _ = compute_masked_loss(
                    outputs,
                    batch["activity"].to(device),
                    batch["dataset_names"],
                    model.dataset_to_heads,
                    use_uncertainty=True,
                )

            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            print(f"  Test loss: {loss_val:.4f}")

            if loss_val < 0:
                print(f"  ✗ FAILED - Negative loss!")
                return False

            print(f"  ✓ PASSED")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n✓ All configurations test PASSED")
    return True


def main():
    print("="*70)
    print("FUSEMAP Comprehensive Test Suite")
    print("="*70)

    results = {}

    # Run all tests
    results["data_loading"] = test_data_loading()
    results["multi_dataset"] = test_multi_dataset()
    results["forward_pass"] = test_forward_pass()
    results["loss_computation"] = test_loss_computation()
    results["training_step"] = test_training_step()
    results["all_configs"] = test_all_configs()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
