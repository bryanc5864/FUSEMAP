"""
Final evaluation script for PhysicsVAE models.
Handles PyTorch 2.6 weights_only issues.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import sys

# Fix PyTorch 2.6 numpy scalar issue
import numpy._core.multiarray
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])

sys.path.insert(0, str(Path(__file__).parent))

from models.physics_vae import PhysicsVAE
from data.dataset import PhysicsVAEDataset
from data.aligned_dataset import AlignedPhysicsVAEDataset, get_training_feature_info
from torch.utils.data import DataLoader


def indices_to_sequence(indices):
    """Convert nucleotide indices to DNA sequence string."""
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
    return ''.join([mapping[i] for i in indices])


def compute_metrics(model, dataloader, device):
    """Compute evaluation metrics."""
    model.eval()

    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_correct = 0
    total_positions = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequence'].to(device)
            physics = batch['physics'].to(device)

            # Forward pass
            output = model(sequences, physics)
            logits = output['logits']
            mu = output['mu']
            logvar = output['logvar']

            # Reconstruction loss
            logits_flat = logits.view(-1, 4)
            targets_flat = sequences.view(-1)
            recon_loss = F.cross_entropy(logits_flat, targets_flat, reduction='sum', ignore_index=4)

            # KL loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            batch_size = sequences.size(0)
            total_recon += recon_loss.item() / batch_size
            total_kl += kl_loss.item() / batch_size
            total_loss += (recon_loss.item() + 0.001 * kl_loss.item()) / batch_size

            # Accuracy
            predictions = logits.argmax(dim=-1)
            mask = sequences < 4  # Exclude N positions
            correct = ((predictions == sequences) & mask).sum().item()
            total_correct += correct
            total_positions += mask.sum().item()

            n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'kl_loss': total_kl / n_batches,
        'accuracy': total_correct / total_positions if total_positions > 0 else 0
    }


def evaluate_model(run_dir: Path, device: torch.device):
    """Run final evaluation on a trained model."""
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION: {run_dir.name}")
    print(f"{'='*60}")

    best_model_path = run_dir / 'best_model.pt'
    if not best_model_path.exists():
        print(f"  ERROR: No best_model.pt found")
        return None

    # Load checkpoint
    try:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"  ERROR loading checkpoint: {e}")
        return None

    # Get config
    config = checkpoint.get('config', {})
    model_info = config.get('model_info', {})
    training_args = config.get('training_args', {})

    cell_type = training_args.get('cell_type', run_dir.name.split('_')[0])
    config_path = training_args.get('config', None)
    seq_length = model_info.get('seq_length', 200)
    n_physics = model_info.get('n_physics', 521)
    latent_dim = model_info.get('latent_dim', 128)

    # Skip multi-dataset configs (handled separately)
    if cell_type is None or config_path is not None:
        print(f"  Skipping multi-dataset config (use separate multi-eval script)")
        return None

    print(f"  Cell type: {cell_type}")
    print(f"  Seq length: {seq_length}, Physics: {n_physics}, Latent: {latent_dim}")
    print(f"  Best epoch: {checkpoint.get('epoch', 'N/A')}")

    # Create model
    model = PhysicsVAE(
        seq_length=seq_length,
        n_physics_features=n_physics,
        latent_dim=latent_dim,
        physics_cond_dim=64,
        n_decoder_layers=4,
        dropout=0.1
    )

    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load test data
    data_dir = Path(training_args.get('data_dir', '../output'))

    # Plant datasets use different file pattern
    plant_types = ['arabidopsis', 'sorghum', 'maize']
    is_plant = cell_type.lower() in plant_types

    if is_plant:
        test_file = data_dir / f"{cell_type}_test_descriptors_with_activity.tsv"
        val_file = data_dir / f"{cell_type}_val_descriptors_with_activity.tsv"
        train_file = data_dir / f"{cell_type}_train_descriptors_with_activity.tsv"
    else:
        test_file = data_dir / f"{cell_type}_test_descriptors.tsv"
        val_file = data_dir / f"{cell_type}_val_descriptors.tsv"
        train_file = data_dir / f"{cell_type}_train_descriptors.tsv"

    results = {'cell_type': cell_type, 'best_epoch': checkpoint.get('epoch', 'N/A')}

    # Get training feature info for alignment if needed
    train_feature_info = None

    # Evaluate on test set
    if test_file.exists():
        print(f"\n  Evaluating on TEST set...")
        test_dataset = PhysicsVAEDataset(str(test_file), cell_type=cell_type, max_seq_length=seq_length)

        # Check for physics feature mismatch
        if test_dataset.n_physics_features != n_physics:
            print(f"  Physics feature mismatch (model: {n_physics}, data: {test_dataset.n_physics_features})")
            print(f"  Using aligned dataset...")

            # Load training feature info if not already loaded
            if train_feature_info is None and train_file.exists():
                train_feature_info = get_training_feature_info(str(train_file))

            if train_feature_info and len(train_feature_info['features']) == n_physics:
                test_dataset = AlignedPhysicsVAEDataset(
                    str(test_file),
                    reference_features=train_feature_info['features'],
                    reference_mean=train_feature_info['mean'],
                    reference_std=train_feature_info['std'],
                    max_seq_length=seq_length
                )
            else:
                print(f"  Could not align features, skipping test evaluation")
                test_dataset = None

        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

            test_metrics = compute_metrics(model, test_loader, device)
            results['test'] = test_metrics

            print(f"  TEST Results:")
            print(f"    Loss: {test_metrics['loss']:.4f}")
            print(f"    Recon Loss: {test_metrics['recon_loss']:.4f}")
            print(f"    KL Loss: {test_metrics['kl_loss']:.4f}")
            print(f"    Accuracy: {test_metrics['accuracy']*100:.2f}%")
    else:
        print(f"  No test file found at {test_file}")

    # Evaluate on val set
    if val_file.exists():
        print(f"\n  Evaluating on VAL set...")
        val_dataset = PhysicsVAEDataset(str(val_file), cell_type=cell_type, max_seq_length=seq_length)

        # Check for physics feature mismatch
        if val_dataset.n_physics_features != n_physics:
            print(f"  Physics feature mismatch (model: {n_physics}, data: {val_dataset.n_physics_features})")
            print(f"  Using aligned dataset...")

            # Load training feature info if not already loaded
            if train_feature_info is None and train_file.exists():
                train_feature_info = get_training_feature_info(str(train_file))

            if train_feature_info and len(train_feature_info['features']) == n_physics:
                val_dataset = AlignedPhysicsVAEDataset(
                    str(val_file),
                    reference_features=train_feature_info['features'],
                    reference_mean=train_feature_info['mean'],
                    reference_std=train_feature_info['std'],
                    max_seq_length=seq_length
                )
            else:
                print(f"  Could not align features, skipping val evaluation")
                val_dataset = None

        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

            val_metrics = compute_metrics(model, val_loader, device)
            results['val'] = val_metrics

            print(f"  VAL Results:")
            print(f"    Loss: {val_metrics['loss']:.4f}")
            print(f"    Recon Loss: {val_metrics['recon_loss']:.4f}")
            print(f"    KL Loss: {val_metrics['kl_loss']:.4f}")
            print(f"    Accuracy: {val_metrics['accuracy']*100:.2f}%")

    # Generation test
    print(f"\n  Generation test...")
    with torch.no_grad():
        # Use random physics vector with correct dimensions
        sample_physics = np.random.randn(1, n_physics).astype(np.float32)

        physics_tensor = torch.tensor(sample_physics, device=device)
        generated = model.generate(physics_tensor, n_samples=3, temperature=0.8)

        print(f"    Generated {generated.shape[0]} sequences of length {generated.shape[1]}")

        # Show sample
        for i in range(min(3, generated.shape[0])):
            seq = indices_to_sequence(generated[i].cpu().numpy())
            print(f"    Sample {i+1}: {seq[:50]}...")

    results['generation_test'] = 'PASSED'

    return results


def main():
    """Run final evaluation on all trained models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    runs_dir = Path(__file__).parent / 'runs'

    # Find runs with best_model.pt
    run_dirs = []
    for d in sorted(runs_dir.iterdir()):
        if d.is_dir() and (d / 'best_model.pt').exists():
            run_dirs.append(d)

    print(f"\nFound {len(run_dirs)} trained models")

    all_results = []
    for run_dir in run_dirs:
        result = evaluate_model(run_dir, device)
        if result:
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("FINAL EVALUATION SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Cell Type':<10} {'Best Ep':<8} {'Test Acc':<10} {'Val Acc':<10} {'Test Loss':<10}")
    print("-" * 50)

    for r in all_results:
        cell = r.get('cell_type', 'N/A')
        epoch = r.get('best_epoch', 'N/A')
        test_acc = r.get('test', {}).get('accuracy', 0) * 100
        val_acc = r.get('val', {}).get('accuracy', 0) * 100
        test_loss = r.get('test', {}).get('loss', 0)

        print(f"{cell:<10} {epoch:<8} {test_acc:<10.2f}% {val_acc:<10.2f}% {test_loss:<10.4f}")

    # Save results
    output_file = runs_dir / 'final_eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
