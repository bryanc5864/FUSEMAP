#!/usr/bin/env python3
"""
Remove constant/zero-variance features from processed descriptor files.
These features have NaN correlations and provide no information for ML.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Features identified as constant (zero variance)
CONSTANT_FEATURES = [
    'thermo_min_dH',
    'thermo_max_dH', 
    'thermo_min_dS',
    'thermo_max_dS',
    'thermo_min_dG',
    'thermo_max_dG',
    'entropy_renyi_entropy_alpha0.0',
    'entropy_shannon_w10_max',
    'entropy_gc_entropy_w10_max',
    'advanced_melting_min_melting_dG',
    'advanced_melting_max_melting_dG',
    'advanced_stacking_min_stacking_energy',
    'advanced_stacking_max_stacking_energy',
    'advanced_stacking_stacking_p95',
    'advanced_g4_g4_hotspot_count',
    'advanced_stress_max_stress_opening',
    'advanced_stress_max_opening_stretch'
]

def remove_constant_features(input_file, output_file):
    """Remove constant features from a descriptor file."""
    print(f"Processing {input_file}...")
    
    # Load data
    df = pd.read_csv(input_file, sep='\t')
    original_cols = len(df.columns)
    
    # Find which constant features exist in this dataset
    features_to_remove = [col for col in CONSTANT_FEATURES if col in df.columns]
    
    if features_to_remove:
        print(f"  Removing {len(features_to_remove)} constant features")
        df = df.drop(columns=features_to_remove)
    
    # Save cleaned data
    df.to_csv(output_file, sep='\t', index=False)
    
    final_cols = len(df.columns)
    print(f"  Original: {original_cols} columns")
    print(f"  Final: {final_cols} columns")
    print(f"  Removed: {original_cols - final_cols} columns")
    
    return original_cols - final_cols

def main():
    """Process all descriptor files."""
    
    # Create cleaned output directory
    output_dir = Path('output_cleaned')
    output_dir.mkdir(exist_ok=True)
    
    # Process all descriptor files
    files_to_process = [
        ('output/HepG2_train_descriptors.tsv', 'output_cleaned/HepG2_train_descriptors.tsv'),
        ('output/HepG2_val_descriptors.tsv', 'output_cleaned/HepG2_val_descriptors.tsv'),
        ('output/HepG2_test_descriptors.tsv', 'output_cleaned/HepG2_test_descriptors.tsv'),
        ('output/K562_train_descriptors.tsv', 'output_cleaned/K562_train_descriptors.tsv'),
        ('output/K562_val_descriptors.tsv', 'output_cleaned/K562_val_descriptors.tsv'),
        ('output/K562_test_descriptors.tsv', 'output_cleaned/K562_test_descriptors.tsv'),
        ('output/WTC11_train_descriptors.tsv', 'output_cleaned/WTC11_train_descriptors.tsv'),
        ('output/WTC11_val_descriptors.tsv', 'output_cleaned/WTC11_val_descriptors.tsv'),
        ('output/WTC11_test_descriptors.tsv', 'output_cleaned/WTC11_test_descriptors.tsv'),
    ]
    
    print("="*60)
    print("REMOVING CONSTANT FEATURES FROM DESCRIPTOR FILES")
    print("="*60)
    print(f"\nRemoving {len(CONSTANT_FEATURES)} constant features:")
    for feat in CONSTANT_FEATURES:
        print(f"  - {feat}")
    
    print("\n" + "="*60)
    print("PROCESSING FILES")
    print("="*60)
    
    total_removed = 0
    for input_file, output_file in files_to_process:
        if Path(input_file).exists():
            removed = remove_constant_features(input_file, output_file)
            total_removed += removed
        else:
            print(f"Skipping {input_file} (not found)")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total features removed: {total_removed}")
    print(f"Cleaned files saved in: output_cleaned/")
    
    # Final feature counts
    print("\nFinal feature counts:")
    for cell_type in ['HepG2', 'K562', 'WTC11']:
        train_file = f'output_cleaned/{cell_type}_train_descriptors.tsv'
        if Path(train_file).exists():
            df = pd.read_csv(train_file, sep='\t', nrows=1)
            # Subtract metadata columns
            metadata_cols = ['seq_id', 'condition', 'normalized_log2', 
                           'n_obs_bc', 'n_replicates', 'sequence']
            n_features = len(df.columns) - len(metadata_cols)
            print(f"  {cell_type}: {n_features} features")

if __name__ == "__main__":
    main()