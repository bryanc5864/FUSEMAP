#!/usr/bin/env python3
"""
Thorough test of FUSEMAP training pipeline.

Checks:
1. Real data loading (no dummy data)
2. Activity normalization
3. Model embedding sizes match data
4. Loss computation (positive values only)
5. Gradient flow
6. Learning (loss decreases)
7. Validation metrics are reasonable
8. All 5 configurations work
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import (
    DATASET_CATALOG,
    ModelConfig,
)
from training.datasets import (
    MultiDataset, SingleDataset, ActivityNormalizer, collate_multi_dataset
)
from training.models import create_multi_species_model, compute_masked_loss


def check_real_data():
    """Verify real data is loaded, not dummy data."""
    print("\n" + "="*70)
    print("CHECK 1: Real Data Loading")
    print("="*70)

    errors = []

    datasets_to_check = [
        ("encode4_k562", 170000, 200000),  # Expected range
        ("encode4_hepg2", 100000, 130000),
        ("encode4_wtc11", 35000, 50000),
        ("deepstarr", 300000, 400000),
        ("jores_arabidopsis", 10000, 20000),
    ]

    normalizer = ActivityNormalizer()

    for name, min_samples, max_samples in datasets_to_check:
        print(f"\n  [{name}]")
        try:
            info = DATASET_CATALOG[name]
            ds = SingleDataset(info, split="train", target_length=256, normalizer=normalizer)
            n = len(ds)
            print(f"    Samples: {n:,}")

            if n < min_samples:
                errors.append(f"{name}: Only {n} samples, expected >{min_samples} (likely dummy data)")
            elif n > max_samples:
                errors.append(f"{name}: {n} samples, expected <{max_samples}")
            else:
                print(f"    ✓ Sample count OK")

            # Check sequence shape
            sample = ds[0]
            seq_shape = sample["sequence"].shape
            if seq_shape != (4, 256):
                errors.append(f"{name}: Unexpected sequence shape {seq_shape}")
            else:
                print(f"    ✓ Sequence shape OK: {seq_shape}")

            # Check activity is not all zeros or NaN
            acts = ds.activities
            if np.all(acts == 0):
                errors.append(f"{name}: All activities are zero")
            elif np.any(np.isnan(acts)):
                errors.append(f"{name}: Activities contain NaN")
            else:
                print(f"    ✓ Activities OK: mean={np.mean(acts):.3f}, std={np.std(acts):.3f}")

        except Exception as e:
            errors.append(f"{name}: Failed to load - {e}")

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    ✗ {e}")
        return False

    print("\n  ✓ All real data checks PASSED")
    return True


def check_normalization():
    """Verify activity normalization works correctly."""
    print("\n" + "="*70)
    print("CHECK 2: Activity Normalization")
    print("="*70)

    errors = []
    normalizer = ActivityNormalizer()

    # Fit on encode4_k562
    info = DATASET_CATALOG["encode4_k562"]
    ds = SingleDataset(info, split="train", target_length=256, normalizer=normalizer)

    print(f"\n  [encode4_k562]")
    if "encode4_k562" in normalizer.stats:
        stats = normalizer.stats["encode4_k562"]
        print(f"    Mean: {stats.mean}")
        print(f"    Std: {stats.std}")

        # Check normalized values
        normalized = normalizer.transform("encode4_k562", ds.activities)
        norm_mean = np.mean(normalized)
        norm_std = np.std(normalized)
        print(f"    Normalized mean: {norm_mean:.4f} (should be ~0)")
        print(f"    Normalized std: {norm_std:.4f} (should be ~1)")

        if abs(norm_mean) > 0.1:
            errors.append(f"Normalized mean too far from 0: {norm_mean}")
        if abs(norm_std - 1.0) > 0.1:
            errors.append(f"Normalized std too far from 1: {norm_std}")
    else:
        errors.append("Normalizer not fitted for encode4_k562")

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    ✗ {e}")
        return False

    print("\n  ✓ Normalization checks PASSED")
    return True


def check_model_embeddings():
    """Verify model embedding sizes match data."""
    print("\n" + "="*70)
    print("CHECK 3: Model Embedding Sizes")
    print("="*70)

    errors = []
    normalizer = ActivityNormalizer()

    # Create multi-dataset
    dataset_names = ["encode4_k562", "deepstarr", "jores_arabidopsis"]
    multi_ds = MultiDataset(
        dataset_names=dataset_names,
        split="train",
        target_length=256,
        normalizer=normalizer,
    )

    print(f"\n  Dataset mappings:")
    print(f"    Species: {multi_ds.species_to_idx}")
    print(f"    Kingdoms: {multi_ds.kingdom_to_idx}")
    print(f"    Celltypes: {multi_ds.celltype_to_idx}")

    n_species = len(multi_ds.species_to_idx)
    n_kingdoms = len(multi_ds.kingdom_to_idx)
    n_celltypes = len(multi_ds.celltype_to_idx)

    # Create model
    config = ModelConfig(
        use_species_embedding=True,
        use_kingdom_embedding=True,
        use_celltype_embedding=True,
        use_length_embedding=True,
    )

    model = create_multi_species_model(
        config, dataset_names,
        n_species=n_species,
        n_kingdoms=n_kingdoms,
        n_celltypes=n_celltypes,
    )

    print(f"\n  Model embedding sizes:")
    if hasattr(model, 'species_embed'):
        print(f"    Species: {model.species_embed.num_embeddings} (expected {n_species})")
        if model.species_embed.num_embeddings != n_species:
            errors.append(f"Species embedding mismatch: {model.species_embed.num_embeddings} vs {n_species}")

    if hasattr(model, 'kingdom_embed'):
        print(f"    Kingdom: {model.kingdom_embed.num_embeddings} (expected {n_kingdoms})")
        if model.kingdom_embed.num_embeddings != n_kingdoms:
            errors.append(f"Kingdom embedding mismatch")

    if hasattr(model, 'celltype_embed'):
        print(f"    Celltype: {model.celltype_embed.num_embeddings} (expected {n_celltypes})")
        if model.celltype_embed.num_embeddings != n_celltypes:
            errors.append(f"Celltype embedding mismatch")

    # Test forward pass with actual batch
    loader = DataLoader(multi_ds, batch_size=32, shuffle=True, collate_fn=collate_multi_dataset)
    batch = next(iter(loader))

    print(f"\n  Testing forward pass:")
    print(f"    Batch species_idx range: [{batch['species_idx'].min()}, {batch['species_idx'].max()}]")
    print(f"    Batch kingdom_idx range: [{batch['kingdom_idx'].min()}, {batch['kingdom_idx'].max()}]")
    print(f"    Batch celltype_idx range: [{batch['celltype_idx'].min()}, {batch['celltype_idx'].max()}]")

    try:
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
        print(f"    ✓ Forward pass succeeded")
    except IndexError as e:
        errors.append(f"Embedding index error: {e}")
    except Exception as e:
        errors.append(f"Forward pass failed: {e}")

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    ✗ {e}")
        return False

    print("\n  ✓ Model embedding checks PASSED")
    return True


def check_loss_computation():
    """Verify loss computation produces positive values."""
    print("\n" + "="*70)
    print("CHECK 4: Loss Computation")
    print("="*70)

    errors = []
    normalizer = ActivityNormalizer()

    dataset_names = ["encode4_k562", "deepstarr"]
    multi_ds = MultiDataset(
        dataset_names=dataset_names,
        split="train",
        target_length=256,
        normalizer=normalizer,
    )

    config = ModelConfig(
        use_species_embedding=True,
        use_celltype_embedding=True,
        use_length_embedding=True,
    )

    model = create_multi_species_model(
        config, dataset_names,
        n_species=len(multi_ds.species_to_idx),
        n_kingdoms=len(multi_ds.kingdom_to_idx),
        n_celltypes=len(multi_ds.celltype_to_idx),
    )

    loader = DataLoader(multi_ds, batch_size=64, shuffle=True, collate_fn=collate_multi_dataset)

    print(f"\n  Testing loss on 20 batches:")
    model.train()
    losses = []

    for i, batch in enumerate(loader):
        if i >= 20:
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
            outputs, batch["activity"], batch["dataset_names"],
            model.dataset_to_heads, use_uncertainty=True,
        )

        loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
        losses.append(loss_val)

        if loss_val < 0:
            errors.append(f"Batch {i}: Negative loss {loss_val}")
        if loss_val > 100:
            errors.append(f"Batch {i}: Unreasonably high loss {loss_val}")

    print(f"    Loss range: [{min(losses):.4f}, {max(losses):.4f}]")
    print(f"    Loss mean: {np.mean(losses):.4f}")
    print(f"    Loss std: {np.std(losses):.4f}")

    if min(losses) < 0:
        errors.append(f"Negative losses detected!")

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    ✗ {e}")
        return False

    print("\n  ✓ Loss computation checks PASSED")
    return True


def check_gradient_flow():
    """Verify gradients flow through all parameters."""
    print("\n" + "="*70)
    print("CHECK 5: Gradient Flow")
    print("="*70)

    errors = []
    normalizer = ActivityNormalizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_names = ["encode4_k562"]
    multi_ds = MultiDataset(
        dataset_names=dataset_names,
        split="train",
        target_length=256,
        normalizer=normalizer,
    )

    config = ModelConfig(use_length_embedding=True)
    model = create_multi_species_model(
        config, dataset_names,
        n_species=len(multi_ds.species_to_idx),
        n_kingdoms=len(multi_ds.kingdom_to_idx),
        n_celltypes=len(multi_ds.celltype_to_idx),
    )
    model = model.to(device)

    loader = DataLoader(multi_ds, batch_size=64, shuffle=True, collate_fn=collate_multi_dataset)
    batch = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    model.train()
    model.zero_grad()

    outputs = model(
        sequence=batch["sequence"],
        species_idx=batch["species_idx"],
        kingdom_idx=batch["kingdom_idx"],
        celltype_idx=batch["celltype_idx"],
        original_length=batch["original_length"],
        dataset_names=batch["dataset_names"],
    )

    loss, _ = compute_masked_loss(
        outputs, batch["activity"], batch["dataset_names"],
        model.dataset_to_heads, use_uncertainty=True,
    )

    if isinstance(loss, (int, float)):
        loss = torch.tensor(loss, requires_grad=True, device=device)

    loss.backward()

    # Check gradients
    total_params = 0
    params_with_grad = 0
    params_without_grad = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None and param.grad.abs().sum() > 0:
                params_with_grad += 1
            else:
                params_without_grad.append(name)

    print(f"\n  Parameters with gradients: {params_with_grad}/{total_params}")

    if params_without_grad:
        print(f"  Parameters without gradients:")
        for name in params_without_grad[:5]:
            print(f"    - {name}")
        if len(params_without_grad) > 5:
            print(f"    ... and {len(params_without_grad) - 5} more")

    # Allow some params without grad (e.g., unused heads)
    if params_with_grad < total_params * 0.8:
        errors.append(f"Too few parameters have gradients: {params_with_grad}/{total_params}")

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    ✗ {e}")
        return False

    print("\n  ✓ Gradient flow checks PASSED")
    return True


def check_learning():
    """Verify model actually learns (loss decreases)."""
    print("\n" + "="*70)
    print("CHECK 6: Learning (Loss Decreases)")
    print("="*70)

    errors = []
    normalizer = ActivityNormalizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_names = ["encode4_k562"]
    multi_ds = MultiDataset(
        dataset_names=dataset_names,
        split="train",
        target_length=256,
        normalizer=normalizer,
    )

    config = ModelConfig(use_length_embedding=True)
    model = create_multi_species_model(
        config, dataset_names,
        n_species=len(multi_ds.species_to_idx),
        n_kingdoms=len(multi_ds.kingdom_to_idx),
        n_celltypes=len(multi_ds.celltype_to_idx),
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(multi_ds, batch_size=128, shuffle=True, collate_fn=collate_multi_dataset)

    print(f"\n  Training for 50 steps on {device}...")
    model.train()

    losses = []
    for i, batch in enumerate(loader):
        if i >= 50:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

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
            outputs, batch["activity"], batch["dataset_names"],
            model.dataset_to_heads, use_uncertainty=True,
        )

        if isinstance(loss, (int, float)):
            loss = torch.tensor(loss, requires_grad=True, device=device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % 10 == 0:
            print(f"    Step {i+1}: loss={loss.item():.4f}")

    # Check if loss decreased
    first_10 = np.mean(losses[:10])
    last_10 = np.mean(losses[-10:])

    print(f"\n  First 10 steps avg: {first_10:.4f}")
    print(f"  Last 10 steps avg: {last_10:.4f}")
    print(f"  Improvement: {(first_10 - last_10) / first_10 * 100:.1f}%")

    if last_10 >= first_10:
        errors.append(f"Loss did not decrease: {first_10:.4f} -> {last_10:.4f}")

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    ✗ {e}")
        return False

    print("\n  ✓ Learning checks PASSED")
    return True


def check_validation():
    """Verify validation metrics are computed correctly."""
    print("\n" + "="*70)
    print("CHECK 7: Validation Metrics")
    print("="*70)

    errors = []
    normalizer = ActivityNormalizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train briefly then validate
    dataset_names = ["encode4_k562"]
    train_ds = MultiDataset(dataset_names, split="train", target_length=256, normalizer=normalizer)

    config = ModelConfig(use_length_embedding=True)
    model = create_multi_species_model(
        config, dataset_names,
        n_species=len(train_ds.species_to_idx),
        n_kingdoms=len(train_ds.kingdom_to_idx),
        n_celltypes=len(train_ds.celltype_to_idx),
    )
    model = model.to(device)

    # Quick training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_multi_dataset)

    model.train()
    for i, batch in enumerate(train_loader):
        if i >= 20:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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
            outputs, batch["activity"], batch["dataset_names"],
            model.dataset_to_heads, use_uncertainty=True,
        )
        if isinstance(loss, (int, float)):
            loss = torch.tensor(loss, requires_grad=True, device=device)
        loss.backward()
        optimizer.step()

    # Validation
    print(f"\n  Running validation...")

    # Load validation data with same mappings
    index_mappings = {
        "species": train_ds.species_to_idx,
        "kingdom": train_ds.kingdom_to_idx,
        "celltype": train_ds.celltype_to_idx,
    }

    info = DATASET_CATALOG["encode4_k562"]
    val_ds = SingleDataset(
        info, split="val", target_length=256,
        normalizer=normalizer, index_mappings=index_mappings
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_multi_dataset)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(
                sequence=batch["sequence"],
                species_idx=batch["species_idx"],
                kingdom_idx=batch["kingdom_idx"],
                celltype_idx=batch["celltype_idx"],
                original_length=batch["original_length"],
                dataset_names=batch["dataset_names"],
            )

            # Get predictions for the k562 head
            head_name = "encode4_k562_activity"
            if head_name in outputs:
                pred = outputs[head_name]
                if isinstance(pred, dict):
                    mean = pred['mean'].cpu().numpy()
                else:
                    mean = pred[0].cpu().numpy()
                all_preds.extend(mean.flatten())

            targets = batch["activity"][:, 0].cpu().numpy()
            all_targets.extend(targets)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Remove NaN
    valid = ~(np.isnan(all_preds) | np.isnan(all_targets))
    all_preds = all_preds[valid]
    all_targets = all_targets[valid]

    if len(all_preds) > 0:
        corr, _ = pearsonr(all_preds, all_targets)
        mse = np.mean((all_preds - all_targets) ** 2)

        print(f"    Validation samples: {len(all_preds)}")
        print(f"    Pearson correlation: {corr:.4f}")
        print(f"    MSE: {mse:.4f}")

        # After only 20 steps, expect correlation to be low but not terrible
        if np.isnan(corr):
            errors.append("Correlation is NaN")
        elif corr < -0.5:
            errors.append(f"Correlation too negative: {corr}")
    else:
        errors.append("No valid predictions")

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    ✗ {e}")
        return False

    print("\n  ✓ Validation checks PASSED")
    return True


def check_all_configs():
    """Test all 5 configurations with real training."""
    print("\n" + "="*70)
    print("CHECK 8: All 5 Configurations")
    print("="*70)

    errors = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        ("Config 1: Single Cell Type", ["encode4_k562"], False, False),
        ("Config 2: Multi-Cell Human", ["encode4_k562", "encode4_hepg2"], False, True),
        ("Config 3: Cross-Animal", ["encode4_k562", "deepstarr"], True, True),
        ("Config 4: Cross-Kingdom", ["encode4_k562", "jores_arabidopsis"], True, True),
        ("Config 5: Universal", ["encode4_k562", "deepstarr", "jores_arabidopsis"], True, True),
    ]

    for config_name, dataset_names, use_species, use_celltype in configs:
        print(f"\n  [{config_name}]")
        print(f"    Datasets: {dataset_names}")

        try:
            normalizer = ActivityNormalizer()
            multi_ds = MultiDataset(
                dataset_names=dataset_names,
                split="train",
                target_length=256,
                normalizer=normalizer,
            )
            print(f"    Total samples: {len(multi_ds):,}")

            config = ModelConfig(
                use_species_embedding=use_species,
                use_kingdom_embedding=use_species,
                use_celltype_embedding=use_celltype,
                use_length_embedding=True,
            )

            model = create_multi_species_model(
                config, dataset_names,
                n_species=len(multi_ds.species_to_idx),
                n_kingdoms=len(multi_ds.kingdom_to_idx),
                n_celltypes=len(multi_ds.celltype_to_idx),
            )
            model = model.to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loader = DataLoader(multi_ds, batch_size=64, shuffle=True, collate_fn=collate_multi_dataset)

            # Train 10 steps
            model.train()
            losses = []
            for i, batch in enumerate(loader):
                if i >= 10:
                    break

                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

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
                    outputs, batch["activity"], batch["dataset_names"],
                    model.dataset_to_heads, use_uncertainty=True,
                )

                if isinstance(loss, (int, float)):
                    loss = torch.tensor(loss, requires_grad=True, device=device)

                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            avg_loss = np.mean(losses)
            print(f"    Avg loss (10 steps): {avg_loss:.4f}")

            if avg_loss < 0:
                errors.append(f"{config_name}: Negative loss")
            elif avg_loss > 50:
                errors.append(f"{config_name}: Loss too high: {avg_loss}")
            else:
                print(f"    ✓ PASSED")

        except Exception as e:
            errors.append(f"{config_name}: {e}")
            import traceback
            traceback.print_exc()

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    ✗ {e}")
        return False

    print("\n  ✓ All configuration checks PASSED")
    return True


def main():
    print("="*70)
    print("FUSEMAP Thorough Test Suite")
    print("="*70)

    results = {}

    results["real_data"] = check_real_data()
    results["normalization"] = check_normalization()
    results["embeddings"] = check_model_embeddings()
    results["loss"] = check_loss_computation()
    results["gradients"] = check_gradient_flow()
    results["learning"] = check_learning()
    results["validation"] = check_validation()
    results["all_configs"] = check_all_configs()

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
        print("ALL CHECKS PASSED! Training pipeline is ready.")
    else:
        print("SOME CHECKS FAILED! Please fix issues before training.")
    print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
