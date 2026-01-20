#!/usr/bin/env python3
"""
Generic TileFormer labeling for any sequence length.

Usage:
    python label_generic_tileformer.py --input data.tsv --output labeled.tsv --seq_length 170
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Add TileFormer path
sys.path.append('/home/bcheng/sequence_optimization/mainproject/PhysiFormer/physpreprocess/TileFormer_model')
from models.tileformer_architecture import TileFormerWithMetadata

# Default model path
DEFAULT_MODEL = '/home/bcheng/sequence_optimization/FUSEMAP/physics/TileFormer/checkpoints/run_20250819_063725/best_model.pth'

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
    return np.array([mapping.get(b, 4) for b in sequence], dtype=np.int64)

def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load TileFormer model."""
    model = TileFormerWithMetadata(
        vocab_size=5, d_model=256, n_heads=8, n_layers=6,
        d_ff=1024, max_len=200, dropout=0.1, output_dim=6,
        predict_uncertainty=True, metadata_dim=3
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

def extract_windows(sequence: str, window_size: int = 20, stride: int = 10) -> list:
    """Extract sliding windows from sequence."""
    windows = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        windows.append(sequence[i:i + window_size])
    return windows

def predict_batch(model, sequences: list, stride: int, batch_size: int, device: str):
    """Predict for a batch of sequences."""
    all_windows = []
    window_counts = []

    for seq in sequences:
        windows = extract_windows(seq, window_size=20, stride=stride)
        if len(windows) == 0:
            padded = seq + 'N' * (20 - len(seq))
            windows = [padded]
        all_windows.extend(windows)
        window_counts.append(len(windows))

    encoded = np.array([sequence_to_indices(w) for w in all_windows], dtype=np.int64)
    metadata = np.array([calculate_metadata(w) for w in all_windows], dtype=np.float32)

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(encoded), batch_size):
            batch = torch.from_numpy(encoded[i:i+batch_size]).long().to(device)
            meta = torch.from_numpy(metadata[i:i+batch_size]).float().to(device)
            out = model(batch, meta)
            all_preds.append(out['psi'].cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # Split back to sequences
    results = []
    idx = 0
    for count in window_counts:
        results.append(all_preds[idx:idx+count])
        idx += count
    return results

def process_file(input_file: str, output_file: str, model, seq_length: int,
                 batch_size: int, device: str, seq_col: str = 'sequence'):
    """Process a single file."""
    print(f"\nProcessing: {input_file}")

    # Calculate stride to get reasonable number of windows
    stride = max(1, (seq_length - 20) // 15)  # ~15 windows
    n_windows = (seq_length - 20) // stride + 1
    print(f"  Sequence length: {seq_length}bp, stride: {stride}bp, windows: {n_windows}")

    df = pd.read_csv(input_file, sep='\t')
    print(f"  Loaded {len(df)} sequences")

    all_predictions = []
    process_batch = 100

    for i in tqdm(range(0, len(df), process_batch), desc="  Predicting"):
        batch_seqs = df[seq_col].iloc[i:i+process_batch].tolist()
        preds = predict_batch(model, batch_seqs, stride, batch_size, device)
        for p in preds:
            all_predictions.append(p.flatten())

    all_predictions = np.array(all_predictions)
    print(f"  Predictions shape: {all_predictions.shape}")

    # Create column names
    output_names = ['STD_PSI_MIN', 'STD_PSI_MAX', 'STD_PSI_MEAN',
                    'ENH_PSI_MIN', 'ENH_PSI_MAX', 'ENH_PSI_MEAN']
    col_names = [f'tileformer_w{w}_{n}' for w in range(n_windows) for n in output_names]

    # Add to dataframe
    for i, col in enumerate(col_names):
        if i < all_predictions.shape[1]:
            df[col] = all_predictions[:, i]

    df.to_csv(output_file, sep='\t', index=False)
    print(f"  Saved: {output_file}")

    # Save numpy version
    npz_file = output_file.replace('.tsv', '_tileformer.npz')
    np.savez_compressed(npz_file, predictions=all_predictions, columns=col_names)
    print(f"  Saved: {npz_file}")

def main():
    parser = argparse.ArgumentParser(description='Generic TileFormer labeling')
    parser.add_argument('--input', type=str, required=True, help='Input TSV file')
    parser.add_argument('--output', type=str, required=True, help='Output TSV file')
    parser.add_argument('--seq_length', type=int, required=True, help='Sequence length (e.g., 170, 110)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Model checkpoint')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--seq_col', type=str, default='sequence', help='Sequence column name')

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Loading model: {args.model}")
    model = load_model(args.model, device)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    process_file(args.input, args.output, model, args.seq_length,
                 args.batch_size, device, args.seq_col)

if __name__ == '__main__':
    main()
