#!/usr/bin/env python3
"""
Add TileFormer electrostatic features to existing physics data files.

This script runs TileFormer inference on sequences and merges the electrostatic
features back into the *_with_features.tsv files.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'TileFormer'))

from physics.TileFormer.models.tileformer_architecture import TileFormerWithMetadata


def calculate_metadata(sequence: str) -> np.ndarray:
    """Calculate sequence metadata features."""
    sequence = sequence.upper()
    seq_len = len(sequence)

    gc_count = sequence.count('G') + sequence.count('C')
    gc_content = gc_count / seq_len if seq_len > 0 else 0.0

    cpg_count = sequence.count('CG')
    cpg_density = cpg_count / (seq_len - 1) if seq_len > 1 else 0.0

    at_count = sequence.count('A') + sequence.count('T')
    minor_groove_score = 1.0 - (at_count / seq_len) if seq_len > 0 else 0.5

    return np.array([gc_content, cpg_density, minor_groove_score], dtype=np.float32)


def sequence_to_indices(sequence: str) -> np.ndarray:
    """Convert DNA sequence to indices."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    sequence = sequence.upper()
    indices = np.array([mapping.get(base, 4) for base in sequence], dtype=np.int64)
    return indices


def extract_windows(sequence: str, window_size: int = 20, stride: int = 10) -> list:
    """Extract sliding windows from sequence."""
    windows = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        windows.append(sequence[i:i + window_size])
    return windows


def load_tileformer_model(checkpoint_path: str, device: str = 'cuda'):
    """Load TileFormer model."""
    model = TileFormerWithMetadata(
        vocab_size=5,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_len=200,
        dropout=0.1,
        output_dim=6,
        predict_uncertainty=True,
        metadata_dim=3
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def compute_tileformer_summary_features(predictions: np.ndarray) -> dict:
    """
    Compute summary statistics from TileFormer window predictions.

    Args:
        predictions: Shape (n_windows, 6) - 6 electrostatic outputs per window

    Returns:
        Dictionary of summary features
    """
    output_names = ['STD_PSI_MIN', 'STD_PSI_MAX', 'STD_PSI_MEAN',
                    'ENH_PSI_MIN', 'ENH_PSI_MAX', 'ENH_PSI_MEAN']

    features = {}

    for i, name in enumerate(output_names):
        col = predictions[:, i]
        features[f'tileformer_{name}_mean'] = np.mean(col)
        features[f'tileformer_{name}_std'] = np.std(col)
        features[f'tileformer_{name}_min'] = np.min(col)
        features[f'tileformer_{name}_max'] = np.max(col)

    # Also add overall electrostatic summary
    features['tileformer_overall_mean'] = np.mean(predictions)
    features['tileformer_overall_std'] = np.std(predictions)
    features['tileformer_std_psi_range'] = np.mean(predictions[:, 1] - predictions[:, 0])  # MAX - MIN
    features['tileformer_enh_psi_range'] = np.mean(predictions[:, 4] - predictions[:, 3])  # MAX - MIN

    return features


def predict_batch_sequences(model, sequences: list, window_size: int = 20, stride: int = 10,
                           batch_size: int = 512, device: str = 'cuda'):
    """Predict TileFormer features for multiple sequences."""
    all_windows = []
    window_counts = []

    for sequence in sequences:
        windows = extract_windows(sequence, window_size, stride)
        if len(windows) == 0 and len(sequence) < window_size:
            windows = [sequence + 'N' * (window_size - len(sequence))]
        all_windows.extend(windows)
        window_counts.append(len(windows))

    if len(all_windows) == 0:
        return [np.zeros((1, 6)) for _ in sequences]

    encoded_windows = np.array([sequence_to_indices(w) for w in all_windows], dtype=np.int64)
    metadata_array = np.array([calculate_metadata(w) for w in all_windows], dtype=np.float32)

    all_predictions = []

    with torch.no_grad():
        for i in range(0, len(encoded_windows), batch_size):
            batch = encoded_windows[i:i + batch_size]
            batch_metadata = metadata_array[i:i + batch_size]
            batch_tensor = torch.from_numpy(batch).long().to(device)
            metadata_tensor = torch.from_numpy(batch_metadata).float().to(device)

            output = model(batch_tensor, metadata_tensor)
            predictions = output['psi']
            all_predictions.append(predictions.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)

    # Split by sequence
    sequence_predictions = []
    start_idx = 0
    for count in window_counts:
        sequence_predictions.append(all_predictions[start_idx:start_idx + count])
        start_idx += count

    return sequence_predictions


def process_file(input_file: str, output_file: str, model, device: str = 'cuda',
                batch_size: int = 512, stride: int = 10):
    """Process a data file and add TileFormer features."""
    print(f"Processing: {input_file}")

    df = pd.read_csv(input_file, sep='\t')
    print(f"  Loaded {len(df)} sequences")

    if 'sequence' not in df.columns:
        raise ValueError(f"No 'sequence' column in {input_file}")

    # Process in batches
    process_batch_size = 200
    all_features = []

    for i in tqdm(range(0, len(df), process_batch_size), desc="  Computing TileFormer"):
        batch_df = df.iloc[i:i + process_batch_size]
        sequences = batch_df['sequence'].tolist()

        batch_predictions = predict_batch_sequences(
            model, sequences, stride=stride, batch_size=batch_size, device=device
        )

        for pred in batch_predictions:
            features = compute_tileformer_summary_features(pred)
            all_features.append(features)

    # Add features to dataframe
    features_df = pd.DataFrame(all_features)
    print(f"  Added {len(features_df.columns)} TileFormer features")

    df = pd.concat([df, features_df], axis=1)

    # Save
    df.to_csv(output_file, sep='\t', index=False)
    print(f"  Saved to {output_file}")

    return len(features_df.columns)


def main():
    parser = argparse.ArgumentParser(description='Add TileFormer features to physics data')
    parser.add_argument('--cell-type', type=str, default='WTC11',
                       help='Cell type to process (WTC11, HepG2, K562, S2)')
    parser.add_argument('--model-path', type=str,
                       default='results/physics_models/models/TileFormer_best.pth',
                       help='Path to TileFormer checkpoint')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       help='Data splits to process')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--in-place', action='store_true',
                       help='Overwrite original files (default: create new files with _tileformer suffix)')

    args = parser.parse_args()

    # Setup paths
    root = Path(__file__).parent.parent.parent
    model_path = root / args.model_path

    if not model_path.exists():
        raise FileNotFoundError(f"TileFormer model not found: {model_path}")

    # Determine data directory
    data_dir_map = {
        'WTC11': root / 'physics/data/lentiMPRA_data/WTC11',
        'HepG2': root / 'physics/data/lentiMPRA_data/HepG2',
        'K562': root / 'physics/data/lentiMPRA_data/K562',
        'S2': root / 'physics/data/drosophila_data/S2',
    }

    if args.cell_type not in data_dir_map:
        raise ValueError(f"Unknown cell type: {args.cell_type}")

    data_dir = data_dir_map[args.cell_type]

    # Determine stride based on cell type
    stride = 11 if args.cell_type == 'S2' else 10

    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Device: {device}")
    print(f"Cell type: {args.cell_type}")
    print(f"Model: {model_path}")

    # Load model
    print("Loading TileFormer model...")
    model = load_tileformer_model(str(model_path), device)

    # Process each split
    for split in args.splits:
        input_file = data_dir / f'{args.cell_type}_{split}_with_features.tsv'

        if not input_file.exists():
            print(f"Skipping {split}: file not found")
            continue

        if args.in_place:
            output_file = input_file
        else:
            output_file = data_dir / f'{args.cell_type}_{split}_with_features_tileformer.tsv'

        process_file(
            str(input_file), str(output_file), model,
            device=device, batch_size=args.batch_size, stride=stride
        )

    print("\nDone! TileFormer features added.")


if __name__ == '__main__':
    main()
