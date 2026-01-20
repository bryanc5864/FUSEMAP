#!/usr/bin/env python3
"""
Fix NaN values in the dataset splits by calculating GC content from sequences
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_gc_content(sequence):
    """Calculate GC content percentage from a DNA sequence."""
    seq = sequence.upper()
    gc_count = seq.count('G') + seq.count('C')
    return (gc_count / len(seq)) * 100 if len(seq) > 0 else 0

def calculate_minor_groove_score(sequence):
    """Calculate minor groove score using the original corpus generator method."""
    seq = sequence.upper()
    at_score = 0
    for i in range(len(seq) - 2):
        triplet = seq[i:i+3]
        if triplet in ['AAA', 'TTT', 'AAT', 'ATT']:
            at_score += 1
    
    return at_score / max(1, len(seq) - 2)

def fix_dataset_nans(input_path, output_path):
    """Fix NaN values in a dataset."""
    print(f"\nProcessing {input_path.name}...")
    
    # Load data
    data = pd.read_csv(input_path, sep='\t')
    initial_rows = len(data)
    
    # Check initial NaN counts
    gc_nan_count = data['gc_content'].isna().sum()
    mg_nan_count = data['minor_groove_score'].isna().sum()
    
    print(f"  Initial NaN counts:")
    print(f"    - gc_content: {gc_nan_count:,} ({(gc_nan_count/len(data))*100:.1f}%)")
    print(f"    - minor_groove_score: {mg_nan_count:,} ({(mg_nan_count/len(data))*100:.1f}%)")
    
    # Fix GC content NaN values by calculating from sequence
    if gc_nan_count > 0:
        print(f"  Calculating GC content for {gc_nan_count} sequences...")
        gc_nan_mask = data['gc_content'].isna()
        data.loc[gc_nan_mask, 'gc_content'] = data.loc[gc_nan_mask, 'sequence'].apply(calculate_gc_content)
    
    # Fix minor_groove_score NaN values by calculating from sequence
    if mg_nan_count > 0:
        print(f"  Calculating minor_groove_score for {mg_nan_count} sequences...")
        mg_nan_mask = data['minor_groove_score'].isna()
        data.loc[mg_nan_mask, 'minor_groove_score'] = data.loc[mg_nan_mask, 'sequence'].apply(calculate_minor_groove_score)
    
    # Verify no NaN values remain in critical columns
    critical_cols = ['gc_content', 'cpg_density', 'minor_groove_score',
                    'std_psi_min', 'std_psi_max', 'std_psi_mean',
                    'enh_psi_min', 'enh_psi_max', 'enh_psi_mean']
    
    remaining_nans = {}
    for col in critical_cols:
        if col in data.columns:
            nan_count = data[col].isna().sum()
            if nan_count > 0:
                remaining_nans[col] = nan_count
    
    if remaining_nans:
        print(f"  WARNING: Remaining NaN values: {remaining_nans}")
    else:
        print(f"  ✓ All critical columns have no NaN values")
    
    # Save fixed data
    data.to_csv(output_path, sep='\t', index=False)
    print(f"  Saved to: {output_path}")
    
    # Verify saved file
    assert len(data) == initial_rows, f"Row count changed: {initial_rows} -> {len(data)}"
    
    return data

def main():
    """Fix NaN values in all dataset splits."""
    
    splits_dir = Path('/home/bcheng/TileFormer/data/splits')
    
    print("Fixing NaN values in dataset splits")
    print("=" * 60)
    
    for split_name in ['train', 'val', 'test']:
        input_file = splits_dir / f'{split_name}.tsv'
        output_file = splits_dir / f'{split_name}_fixed.tsv'
        
        if not input_file.exists():
            print(f"ERROR: {input_file} not found!")
            continue
            
        fix_dataset_nans(input_file, output_file)
    
    print("\n" + "=" * 60)
    print("All datasets processed successfully!")
    print("\nTo use the fixed datasets, update your scripts to use:")
    print("  - train_fixed.tsv")
    print("  - val_fixed.tsv") 
    print("  - test_fixed.tsv")

if __name__ == "__main__":
    main()