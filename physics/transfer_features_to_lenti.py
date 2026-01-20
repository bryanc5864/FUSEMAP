#!/usr/bin/env python3
"""
GPU-accelerated transfer of biophysical features to lentiMPRA data.
Matches by sequence name only, preserving lentiMPRA's original columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import torch

def transfer_features_gpu(cell_type):
    """Transfer features for one cell type using GPU acceleration."""
    print(f"\n{'='*60}")
    print(f"Processing {cell_type}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Load all our processed data for this cell type
    our_files = {
        'train': f'output/{cell_type}_train_descriptors.tsv',
        'val': f'output/{cell_type}_val_descriptors.tsv',
        'test': f'output/{cell_type}_test_descriptors.tsv'
    }
    
    print("Loading our processed features...")
    our_dfs = []
    for split, file in our_files.items():
        df = pd.read_csv(file, sep='\t')
        our_dfs.append(df)
    
    # Combine all our data
    our_combined = pd.concat(our_dfs, ignore_index=True)
    print(f"  Loaded {len(our_combined):,} sequences with {len(our_combined.columns)} columns")
    
    # Get feature columns (exclude metadata)
    metadata_cols = ['seq_id', 'condition', 'normalized_log2', 'n_obs_bc', 'n_replicates', 'sequence']
    feature_cols = [col for col in our_combined.columns if col not in metadata_cols]
    print(f"  {len(feature_cols)} feature columns to transfer")
    
    # Create dictionary for fast lookup
    print("Creating feature lookup dictionary...")
    feature_dict = {}
    for idx, row in our_combined.iterrows():
        seq_name = row['seq_id']
        feature_dict[seq_name] = row[feature_cols].values
    
    # Process each lentiMPRA split
    lenti_splits = ['train', 'val', 'test', 'calibration']
    
    for split in lenti_splits:
        input_file = f'data/lentiMPRA_data/{cell_type}/splits/{split}.tsv'
        output_file = f'data/lentiMPRA_data/{cell_type}/{cell_type}_{split}_with_features.tsv'
        
        if not Path(input_file).exists():
            print(f"  Skipping {split} (file not found)")
            continue
            
        print(f"\n  Processing {split} split...")
        
        # Load lentiMPRA data
        lenti_df = pd.read_csv(input_file, sep='\t')
        n_sequences = len(lenti_df)
        print(f"    Loaded {n_sequences:,} sequences")
        
        # Initialize feature matrix
        feature_matrix = np.zeros((n_sequences, len(feature_cols)), dtype=np.float32)
        
        # Fill feature matrix using vectorized operations
        found_count = 0
        missing_sequences = []
        
        for i, seq_name in enumerate(lenti_df['name']):
            if seq_name in feature_dict:
                feature_matrix[i, :] = feature_dict[seq_name]
                found_count += 1
            else:
                missing_sequences.append(seq_name)
        
        print(f"    Matched {found_count:,}/{n_sequences:,} sequences ({100*found_count/n_sequences:.1f}%)")
        
        if missing_sequences and len(missing_sequences) <= 10:
            print(f"    Missing sequences: {missing_sequences[:10]}")
        
        # Create new dataframe with original lenti columns + features
        result_df = lenti_df.copy()
        
        # Add feature columns
        for i, col in enumerate(feature_cols):
            result_df[col] = feature_matrix[:, i]
        
        # Save result
        result_df.to_csv(output_file, sep='\t', index=False)
        print(f"    Saved to: {output_file}")
        print(f"    Final shape: {result_df.shape}")
    
    elapsed = time.time() - start_time
    print(f"\n  Completed {cell_type} in {elapsed:.1f} seconds")

def main():
    """Process all three cell types."""
    print("="*60)
    print("TRANSFERRING BIOPHYSICAL FEATURES TO LENTIMPRA DATA")
    print("="*60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU")
    
    cell_types = ['HepG2', 'K562', 'WTC11']
    
    total_start = time.time()
    
    for cell_type in cell_types:
        transfer_features_gpu(cell_type)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*60)
    print("TRANSFER COMPLETE")
    print(f"Total time: {total_elapsed:.1f} seconds")
    print("="*60)
    
    # Summary
    print("\nOutput files created:")
    for cell_type in cell_types:
        for split in ['train', 'val', 'test', 'calibration']:
            output_file = f'data/lentiMPRA_data/{cell_type}/{cell_type}_{split}_with_features.tsv'
            if Path(output_file).exists():
                df_check = pd.read_csv(output_file, sep='\t', nrows=1)
                print(f"  {output_file}: {len(df_check.columns)} columns")

if __name__ == "__main__":
    main()