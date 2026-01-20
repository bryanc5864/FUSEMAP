#!/usr/bin/env python3
"""
Label DNA sequences with TileFormer electrostatic predictions using sliding window approach.

For 230bp sequences: Use stride of 10bp (gives 22 windows)
For 249bp sequences: Use stride of 11bp (gives 23 windows)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import os

# Add TileFormer path
sys.path.append('/home/bcheng/sequence_optimization/mainproject/PhysiFormer/physpreprocess/TileFormer_model')
from models.tileformer_architecture import TileFormerWithMetadata

def calculate_metadata(sequence: str) -> np.ndarray:
    """Calculate sequence metadata features.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Array of [gc_content, cpg_density, minor_groove_width]
    """
    sequence = sequence.upper()
    seq_len = len(sequence)
    
    # GC content
    gc_count = sequence.count('G') + sequence.count('C')
    gc_content = gc_count / seq_len if seq_len > 0 else 0.0
    
    # CpG density
    cpg_count = sequence.count('CG')
    cpg_density = cpg_count / (seq_len - 1) if seq_len > 1 else 0.0
    
    # Simple minor groove width approximation (AT-rich regions have narrower grooves)
    at_count = sequence.count('A') + sequence.count('T')
    minor_groove_score = 1.0 - (at_count / seq_len) if seq_len > 0 else 0.5
    
    return np.array([gc_content, cpg_density, minor_groove_score], dtype=np.float32)

def sequence_to_indices(sequence: str) -> np.ndarray:
    """Convert DNA sequence to indices for embedding.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Index array of shape (len(sequence),)
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}  # N gets its own index
    sequence = sequence.upper()
    
    indices = np.zeros(len(sequence), dtype=np.int64)
    for i, base in enumerate(sequence):
        indices[i] = mapping.get(base, 4)  # Default to N (index 4) for unknown
    
    return indices

def extract_windows(sequence: str, window_size: int = 20, stride: int = 10) -> list:
    """Extract sliding windows from a sequence.
    
    Args:
        sequence: DNA sequence string
        window_size: Size of each window (default 20bp for TileFormer)
        stride: Step size between windows
        
    Returns:
        List of window sequences
    """
    windows = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        windows.append(sequence[i:i + window_size])
    
    # For 249bp sequences with stride 11, add final window to cover the end
    # This ensures we cover positions 240-249 that would otherwise be missed
    if len(sequence) == 249 and stride == 11:
        # Add window at position 229-248 (last 20bp)
        windows.append(sequence[229:249])
    
    return windows

def load_tileformer_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained TileFormer model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    # Initialize model with same architecture as training (with metadata)
    model = TileFormerWithMetadata(
        vocab_size=5,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_len=200,
        dropout=0.1,
        output_dim=6,  # 6 electrostatic outputs
        predict_uncertainty=True,
        metadata_dim=3  # GC content, CpG density, minor groove
    )
    
    # Load checkpoint (weights_only=False needed for numpy scalars in checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

def predict_sequence(model, sequence: str, window_size: int = 20, stride: int = 10, 
                     batch_size: int = 128, device: str = 'cuda'):
    """Predict electrostatic values for a sequence using sliding windows.
    
    Args:
        model: TileFormer model
        sequence: DNA sequence string
        window_size: Size of each window
        stride: Step size between windows
        batch_size: Batch size for inference
        device: Device for computation
        
    Returns:
        Array of predictions shape (n_windows, 6)
    """
    # Extract windows
    windows = extract_windows(sequence, window_size, stride)
    
    if len(windows) == 0:
        # Sequence too short, pad or handle specially
        if len(sequence) < window_size:
            # Pad sequence to window_size
            padded_seq = sequence + 'N' * (window_size - len(sequence))
            windows = [padded_seq]
    
    # Encode windows as indices and calculate metadata
    encoded_windows = np.array([sequence_to_indices(w) for w in windows])
    metadata_array = np.array([calculate_metadata(w) for w in windows])
    
    # Predict in batches
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(encoded_windows), batch_size):
            batch = encoded_windows[i:i + batch_size]
            batch_metadata = metadata_array[i:i + batch_size]
            batch_tensor = torch.from_numpy(batch).long().to(device)
            metadata_tensor = torch.from_numpy(batch_metadata).float().to(device)
            
            # Model prediction
            output = model(batch_tensor, metadata_tensor)
            predictions = output['psi']
            all_predictions.append(predictions.cpu().numpy())
    
    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=0)
    
    return predictions

def predict_batch_sequences(model, sequences: list, window_size: int = 20, stride: int = 10,
                           batch_size: int = 512, device: str = 'cuda'):
    """GPU-optimized batch prediction for multiple sequences.
    
    Args:
        model: TileFormer model
        sequences: List of DNA sequence strings
        window_size: Size of each window
        stride: Step size between windows
        batch_size: Batch size for inference
        device: Device for computation
        
    Returns:
        List of prediction arrays
    """
    # Collect all windows from all sequences
    all_windows = []
    sequence_indices = []
    window_counts = []
    
    for seq_idx, sequence in enumerate(sequences):
        windows = extract_windows(sequence, window_size, stride)
        
        if len(windows) == 0:
            # Handle short sequences
            if len(sequence) < window_size:
                padded_seq = sequence + 'N' * (window_size - len(sequence))
                windows = [padded_seq]
        
        all_windows.extend(windows)
        sequence_indices.extend([seq_idx] * len(windows))
        window_counts.append(len(windows))
    
    # Encode all windows at once as indices and metadata
    encoded_windows = np.array([sequence_to_indices(w) for w in all_windows], dtype=np.int64)
    metadata_array = np.array([calculate_metadata(w) for w in all_windows], dtype=np.float32)
    
    # Predict in large batches for GPU efficiency
    all_predictions = []
    
    with torch.no_grad():
        # Use larger batch size for GPU efficiency
        for i in range(0, len(encoded_windows), batch_size):
            batch = encoded_windows[i:i + batch_size]
            batch_metadata = metadata_array[i:i + batch_size]
            batch_tensor = torch.from_numpy(batch).long().to(device)
            metadata_tensor = torch.from_numpy(batch_metadata).float().to(device)
            
            # Model prediction
            output = model(batch_tensor, metadata_tensor)
            predictions = output['psi']
            all_predictions.append(predictions.cpu().numpy())
    
    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Split predictions back to individual sequences
    sequence_predictions = []
    start_idx = 0
    for count in window_counts:
        sequence_predictions.append(all_predictions[start_idx:start_idx + count])
        start_idx += count
    
    return sequence_predictions

def process_dataset(input_file: str, output_file: str, model_path: str, 
                   sequence_length: int = 230, is_s2: bool = False,
                   batch_size: int = 512, device: str = 'cuda'):
    """Process a dataset file and add TileFormer predictions with GPU optimization.
    
    Args:
        input_file: Input TSV file with sequences
        output_file: Output file with added predictions
        model_path: Path to TileFormer checkpoint
        sequence_length: Expected sequence length (230 or 249)
        is_s2: Whether this is S2 data (249bp)
        batch_size: Batch size for inference (increased for GPU efficiency)
        device: Device for computation
    """
    print(f"\nProcessing {input_file}")
    print(f"Sequence length: {sequence_length}bp")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    
    # Check GPU memory if using CUDA
    if device == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Determine stride based on sequence length
    if sequence_length == 230:
        stride = 10
        n_windows = 22  # Full coverage with stride 10
    elif sequence_length == 249:
        stride = 11
        n_windows = 22  # 21 regular + 1 final window for full coverage
    else:
        # General formula: try to get ~22 windows
        stride = max(1, (sequence_length - 20) // 21)
        n_windows = (sequence_length - 20) // stride + 1
    
    print(f"Using stride {stride}bp, expecting {n_windows} windows per sequence")
    
    # Load model
    print(f"Loading model from {model_path}")
    model = load_tileformer_model(model_path, device)
    
    # Load data
    df = pd.read_csv(input_file, sep='\t')
    print(f"Loaded {len(df)} sequences")
    
    # Process sequences in larger batches for GPU efficiency
    all_predictions = []
    process_batch_size = 100  # Process 100 sequences at a time
    
    for i in tqdm(range(0, len(df), process_batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i + process_batch_size]
        sequences = batch_df['sequence'].tolist()
        
        # Get batch predictions using GPU-optimized function
        batch_predictions = predict_batch_sequences(model, sequences, stride=stride,
                                                   batch_size=batch_size, device=device)
        
        # Flatten each prediction
        for pred in batch_predictions:
            flat_predictions = pred.flatten()
            all_predictions.append(flat_predictions)
    
    # Convert to array
    all_predictions = np.array(all_predictions)
    print(f"Predictions shape: {all_predictions.shape}")
    
    # Add predictions to dataframe as separate columns
    # Create column names for each window and output
    output_names = ['STD_PSI_MIN', 'STD_PSI_MAX', 'STD_PSI_MEAN', 
                    'ENH_PSI_MIN', 'ENH_PSI_MAX', 'ENH_PSI_MEAN']
    
    col_names = []
    for w in range(n_windows):
        for out_name in output_names:
            col_names.append(f'tileformer_w{w}_{out_name}')
    
    # Add columns to dataframe
    for i, col_name in enumerate(col_names):
        df[col_name] = all_predictions[:, i]
    
    # Save output
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Saved {len(df)} sequences with {len(col_names)} TileFormer features to {output_file}")
    
    # Also save as numpy array for easier loading
    npz_file = output_file.replace('.tsv', '_tileformer_vectors.npz')
    np.savez_compressed(npz_file, 
                       predictions=all_predictions,
                       column_names=col_names,
                       stride=stride,
                       n_windows=n_windows)
    print(f"Also saved predictions as vectors to {npz_file}")

def main():
    parser = argparse.ArgumentParser(description='Label sequences with TileFormer predictions')
    parser.add_argument('--model', type=str, 
                       default='TileFormer_model/checkpoints/run_20250819_063725/best_model.pth',
                       help='Path to TileFormer checkpoint (default: best_model.pth from epoch 24)')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for labeled files')
    parser.add_argument('--s2', action='store_true',
                       help='Process S2 data (249bp) instead of ENCODE4 (230bp)')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for inference (default: 512 for GPU efficiency)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for computation (cuda/cpu)')
    parser.add_argument('--datasets', nargs='+', default=['train', 'val', 'test'],
                       help='Datasets to process (default: train val test)')
    parser.add_argument('--cell_types', nargs='+', default=['HepG2', 'K562', 'WTC11'],
                       help='Cell types to process for ENCODE4 (default: HepG2 K562 WTC11)')
    parser.add_argument('--gpu', type=int, default=1,
                       help='GPU device ID to use (default: 1)')
    
    args = parser.parse_args()
    
    # Set GPU device
    if args.device == 'cuda' and torch.cuda.is_available():
        if args.gpu < torch.cuda.device_count():
            torch.cuda.set_device(args.gpu)
            args.device = f'cuda:{args.gpu}'
            print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        else:
            print(f"GPU {args.gpu} not available, using default GPU 0")
            args.device = 'cuda:0'
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process datasets
    processed_files = []
    
    if args.s2:
        print("\n=== Processing S2 Data (249bp sequences) ===")
        print(f"Model: {args.model}")
        print(f"Output directory: {args.output_dir}")
        print(f"Datasets to process: {args.datasets}")
        sequence_length = 249
        
        for dataset in args.datasets:
            input_file = Path(args.data_dir) / 'S2_data' / 'splits' / f'{dataset}.tsv'
            output_file = Path(args.output_dir) / f'S2_{dataset}_tileformer.tsv'
            
            if input_file.exists():
                process_dataset(str(input_file), str(output_file), args.model,
                              sequence_length=sequence_length, is_s2=True,
                              batch_size=args.batch_size, device=args.device)
                processed_files.append(str(output_file))
            else:
                print(f"Warning: {input_file} not found, skipping...")
    else:
        print("\n=== Processing ENCODE4 Data (230bp sequences) ===")
        print(f"Model: {args.model}")
        print(f"Output directory: {args.output_dir}")
        print(f"Datasets to process: {args.datasets}")
        sequence_length = 230
        
        # Process each cell type
        cell_types = args.cell_types
        
        for cell_type in cell_types:
            print(f"\n--- Processing {cell_type} ---")
            
            for dataset in args.datasets:
                # Check different possible paths
                possible_paths = [
                    Path(args.data_dir) / f'{cell_type}_data' / 'splits' / f'{dataset}.tsv',
                    Path(args.data_dir) / 'lentiMPRA_data' / cell_type / 'splits' / f'{dataset}.tsv',
                ]
                
                input_file = None
                for path in possible_paths:
                    if path.exists():
                        input_file = path
                        break
                
                if input_file:
                    output_file = Path(args.output_dir) / f'{cell_type}_{dataset}_tileformer.tsv'
                    print(f"Processing {cell_type} {dataset}: {input_file}")
                    process_dataset(str(input_file), str(output_file), args.model,
                                  sequence_length=sequence_length, is_s2=False,
                                  batch_size=args.batch_size, device=args.device)
                    processed_files.append(str(output_file))
                else:
                    print(f"Warning: No file found for {cell_type} {dataset}, skipping...")
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Processed {len(processed_files)} datasets:")
    for f in processed_files:
        print(f"  - {f}")
    print("\nTileFormer features added:")
    print(f"  - {'23 windows × 6 outputs = 138 features' if args.s2 else '22 windows × 6 outputs = 132 features'}")
    print("  - Outputs: STD_PSI_MIN, STD_PSI_MAX, STD_PSI_MEAN, ENH_PSI_MIN, ENH_PSI_MAX, ENH_PSI_MEAN")
    print(f"  - Stride: {'11bp' if args.s2 else '10bp'}")

if __name__ == '__main__':
    main()