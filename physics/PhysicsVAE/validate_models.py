"""
Validate trained PhysicsVAE models.
Loads best checkpoints and runs evaluation metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.physics_vae import PhysicsVAE
from data.dataset import PhysicsVAEDataset


def validate_model(run_dir: Path, device: torch.device):
    """Validate a single trained model."""
    print(f"\n{'='*60}")
    print(f"Validating: {run_dir.name}")
    print(f"{'='*60}")

    # Check for best model
    best_model_path = run_dir / 'best_model.pt'
    if not best_model_path.exists():
        print(f"  ERROR: No best_model.pt found")
        return None

    # Load checkpoint
    try:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        print(f"  Checkpoint loaded successfully")
    except Exception as e:
        print(f"  ERROR loading checkpoint: {e}")
        return None

    # Extract info from checkpoint
    results = {
        'run_dir': str(run_dir),
        'cell_type': run_dir.name.split('_')[0],
    }

    # Check checkpoint contents
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")

    if 'epoch' in checkpoint:
        results['epoch'] = checkpoint['epoch']
        print(f"  Best epoch: {checkpoint['epoch']}")

    if 'best_val_loss' in checkpoint:
        results['best_val_loss'] = checkpoint['best_val_loss']
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")

    # Check model architecture from state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Infer dimensions from state dict keys
    encoder_keys = [k for k in state_dict.keys() if 'encoder' in k]
    decoder_keys = [k for k in state_dict.keys() if 'decoder' in k]

    print(f"  Encoder layers: {len(encoder_keys)}")
    print(f"  Decoder layers: {len(decoder_keys)}")

    # Get latent dim from mu layer
    if 'seq_encoder.fc_mu.weight' in state_dict:
        latent_dim = state_dict['seq_encoder.fc_mu.weight'].shape[0]
        results['latent_dim'] = latent_dim
        print(f"  Latent dim: {latent_dim}")

    # Get physics dim from physics encoder
    if 'physics_encoder.fc.0.weight' in state_dict:
        physics_dim = state_dict['physics_encoder.fc.0.weight'].shape[1]
        results['physics_dim'] = physics_dim
        print(f"  Physics dim: {physics_dim}")

    # Try to instantiate model and load weights
    try:
        # Get config from checkpoint
        config = checkpoint.get('config', {})
        print(f"  Model config: {config}")

        # Extract model parameters from config (stored with model_info)
        model_info = config.get('model_info', {})
        training_args = config.get('training_args', {})

        seq_length = model_info.get('seq_length', 200)
        n_physics_features = model_info.get('n_physics', 521)
        latent_dim = model_info.get('latent_dim', 128)
        physics_cond_dim = 64  # Default
        n_decoder_layers = 4  # Default
        dropout = 0.1  # Default

        # Override from state dict if needed
        if 'physics_encoder.encoder.0.weight' in state_dict:
            n_physics_features = state_dict['physics_encoder.encoder.0.weight'].shape[1]
        if 'sequence_encoder.fc_mu.weight' in state_dict:
            latent_dim = state_dict['sequence_encoder.fc_mu.weight'].shape[0]

        print(f"  Using: seq_length={seq_length}, n_physics={n_physics_features}, latent_dim={latent_dim}")

        results['n_physics_features'] = n_physics_features
        results['latent_dim'] = latent_dim

        model = PhysicsVAE(
            seq_length=seq_length,
            n_physics_features=n_physics_features,
            latent_dim=latent_dim,
            physics_cond_dim=physics_cond_dim,
            n_decoder_layers=n_decoder_layers,
            dropout=dropout
        )

        # Try loading state dict
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        results['model_loaded'] = True
        print(f"  Model instantiated and weights loaded successfully!")

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results['n_params'] = n_params
        print(f"  Total parameters: {n_params:,}")

        # Quick sanity check: generate a random sequence
        with torch.no_grad():
            # Random physics conditioning
            random_physics = torch.randn(1, n_physics_features, device=device)
            generated = model.generate(random_physics)
            print(f"  Generation test: shape {generated.shape}")

            # Check output is valid nucleotide indices
            if generated.max() <= 3 and generated.min() >= 0:
                print(f"  Generation test PASSED (valid nucleotide indices)")
            else:
                print(f"  Generation test WARNING: values outside 0-3 range")

    except Exception as e:
        results['model_loaded'] = False
        print(f"  Could not instantiate model: {e}")

    # Check training logs
    batch_log = run_dir / 'batch_log.csv'
    if batch_log.exists():
        import pandas as pd
        try:
            df = pd.read_csv(batch_log)
            results['final_epoch'] = df['epoch'].max()
            results['total_batches'] = len(df)

            # Get final metrics
            final_rows = df[df['epoch'] == df['epoch'].max()]
            results['final_recon_loss'] = final_rows['recon_loss'].mean()
            results['final_accuracy'] = final_rows['accuracy'].mean()

            print(f"  Training epochs completed: {results['final_epoch'] + 1}")
            print(f"  Final reconstruction loss: {results['final_recon_loss']:.4f}")
            print(f"  Final accuracy: {results['final_accuracy']:.4f}")

        except Exception as e:
            print(f"  Could not parse batch_log: {e}")

    return results


def main():
    """Validate all trained PhysicsVAE models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    runs_dir = Path(__file__).parent / 'runs'

    if not runs_dir.exists():
        print(f"No runs directory found at {runs_dir}")
        return

    # Find all run directories
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(run_dirs)} run directories")

    # Validate each
    all_results = []
    for run_dir in run_dirs:
        result = validate_model(run_dir, device)
        if result:
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    for result in all_results:
        cell_type = result.get('cell_type', 'Unknown')
        loaded = "✓" if result.get('model_loaded', False) else "✗"
        epochs = result.get('final_epoch', 'N/A')
        val_loss = result.get('best_val_loss', float('inf'))
        accuracy = result.get('final_accuracy', 0)

        if isinstance(epochs, int):
            epochs = epochs + 1  # Convert from 0-indexed

        if isinstance(val_loss, float) and val_loss != float('inf'):
            print(f"  {cell_type:8s}: {loaded} loaded | epochs: {epochs:3} | val_loss: {val_loss:.2f} | accuracy: {accuracy:.3f}")
        else:
            print(f"  {cell_type:8s}: {loaded} loaded | epochs: {epochs:3}")

    # Save results
    output_file = runs_dir / 'validation_results.json'
    with open(output_file, 'w') as f:
        # Convert any numpy/torch types
        clean_results = []
        for r in all_results:
            clean = {}
            for k, v in r.items():
                if isinstance(v, (np.floating, np.integer)):
                    clean[k] = float(v)
                elif isinstance(v, torch.Tensor):
                    clean[k] = v.item() if v.numel() == 1 else v.tolist()
                else:
                    clean[k] = v
            clean_results.append(clean)
        json.dump(clean_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
