"""
Generation script for PhysicsVAE.

Generate sequences conditioned on target physics features.

Usage:
    python generate.py --checkpoint runs/K562_*/best_model.pt --n_samples 100
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import json
import sys

# Handle imports whether running from PhysicsVAE directory or imported from elsewhere
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from models.physics_vae import PhysicsVAE
from data.dataset import PhysicsVAEDataset


def indices_to_sequence(indices: torch.Tensor) -> str:
    """Convert nucleotide indices to DNA sequence string."""
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
    return ''.join([mapping[i.item()] for i in indices])


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained PhysicsVAE model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint first, then fallback to config file
    args = {}
    if 'config' in checkpoint:
        model_info = checkpoint['config'].get('model_info', {})
        args = {
            'seq_length': model_info.get('seq_length'),
            'n_physics': model_info.get('n_physics'),
            'latent_dim': model_info.get('latent_dim'),
        }

    # Fallback to config file if not in checkpoint
    if not args.get('seq_length'):
        run_dir = Path(checkpoint_path).parent
        config_file = run_dir / 'results.json'
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                args = config.get('args', {})

    # Default model config (per spec architecture)
    model = PhysicsVAE(
        seq_length=args.get('seq_length', 230),
        n_physics_features=args.get('n_physics', 515),
        latent_dim=args.get('latent_dim', 128),
        physics_cond_dim=64,  # Spec: z_physics dim = 64
        n_decoder_layers=4,   # Spec: 4 transformer layers
        dropout=0.1
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def generate_from_physics(
    model: PhysicsVAE,
    physics: torch.Tensor,
    n_samples: int = 1,
    temperature: float = 1.0,
    device: torch.device = None
) -> list:
    """
    Generate sequences from physics features.

    Args:
        model: Trained PhysicsVAE
        physics: Target physics features [n_targets, n_physics]
        n_samples: Samples per physics target
        temperature: Sampling temperature
        device: Torch device

    Returns:
        List of generated sequence strings
    """
    if device is None:
        device = next(model.parameters()).device

    physics = physics.to(device)

    with torch.no_grad():
        sequences = model.generate(
            physics,
            n_samples=n_samples,
            temperature=temperature
        )

    # Convert to strings
    seq_strings = [indices_to_sequence(seq) for seq in sequences]

    return seq_strings


def generate_random_physics_samples(
    model: PhysicsVAE,
    n_samples: int = 100,
    temperature: float = 1.0,
    device: torch.device = None
) -> list:
    """
    Generate sequences from random latent samples with zero physics conditioning.

    Useful for exploring the learned latent space.
    """
    if device is None:
        device = next(model.parameters()).device

    # Sample from prior
    z = torch.randn(n_samples, model.latent_dim, device=device)

    # Use zero physics conditioning (or could sample from training distribution)
    physics_cond = torch.zeros(n_samples, 64, device=device)

    with torch.no_grad():
        logits = model.decoder(z, physics_cond)
        probs = torch.softmax(logits / temperature, dim=-1)
        sequences = torch.multinomial(
            probs.view(-1, 4),
            num_samples=1
        ).view(n_samples, -1)

    seq_strings = [indices_to_sequence(seq) for seq in sequences]

    return seq_strings


def interpolate_sequences(
    model: PhysicsVAE,
    dataset: PhysicsVAEDataset,
    idx1: int,
    idx2: int,
    n_steps: int = 10,
    device: torch.device = None
) -> list:
    """
    Interpolate between two sequences in latent space.
    """
    if device is None:
        device = next(model.parameters()).device

    # Get source sequences and physics
    sample1 = dataset[idx1]
    sample2 = dataset[idx2]

    seq1 = sample1['sequence'].unsqueeze(0).to(device)
    seq2 = sample2['sequence'].unsqueeze(0).to(device)
    phys1 = sample1['physics'].unsqueeze(0).to(device)
    phys2 = sample2['physics'].unsqueeze(0).to(device)

    # Interpolate
    with torch.no_grad():
        sequences = model.interpolate(
            seq1.squeeze(0), seq2.squeeze(0),
            phys1.squeeze(0), phys2.squeeze(0),
            n_steps=n_steps
        )

    seq_strings = [indices_to_sequence(seq) for seq in sequences]

    return seq_strings


def main():
    parser = argparse.ArgumentParser(description='Generate sequences with PhysicsVAE')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to descriptor file for physics targets')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--output', type=str, default='generated_sequences.txt',
                        help='Output file')
    parser.add_argument('--mode', type=str, default='random',
                        choices=['random', 'from_data', 'interpolate'],
                        help='Generation mode')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully")

    # Generate sequences
    if args.mode == 'random':
        print(f"Generating {args.n_samples} sequences from random latent samples...")
        sequences = generate_random_physics_samples(
            model, args.n_samples, args.temperature, device
        )

    elif args.mode == 'from_data':
        if args.data_file is None:
            raise ValueError("--data_file required for 'from_data' mode")

        print(f"Loading physics targets from {args.data_file}...")
        dataset = PhysicsVAEDataset(args.data_file)

        # Sample random physics targets
        indices = np.random.choice(len(dataset), min(args.n_samples, len(dataset)), replace=False)
        physics = torch.stack([dataset[i]['physics'] for i in indices])

        print(f"Generating sequences for {len(indices)} physics targets...")
        sequences = generate_from_physics(model, physics, n_samples=1, temperature=args.temperature, device=device)

    elif args.mode == 'interpolate':
        if args.data_file is None:
            raise ValueError("--data_file required for 'interpolate' mode")

        dataset = PhysicsVAEDataset(args.data_file)

        # Random pair
        idx1, idx2 = np.random.choice(len(dataset), 2, replace=False)
        print(f"Interpolating between sequences {idx1} and {idx2}...")

        sequences = interpolate_sequences(model, dataset, idx1, idx2, n_steps=args.n_samples, device=device)

    # Save sequences
    print(f"Saving {len(sequences)} sequences to {args.output}...")
    with open(args.output, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">generated_{i}\n{seq}\n")

    print("Done!")

    # Print sample
    print("\nSample generated sequences:")
    for i, seq in enumerate(sequences[:5]):
        print(f"  {i+1}: {seq[:50]}...")


if __name__ == '__main__':
    main()
