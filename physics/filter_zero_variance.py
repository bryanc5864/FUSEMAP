#!/usr/bin/env python3
"""
Filter zero-variance features from HepG2 descriptor files
"""
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

def filter_zero_variance(input_file, output_file, zero_var_features):
    """Filter out zero-variance features from descriptor file"""
    print(f"Processing {input_file}...")
    
    # Read with chunks for memory efficiency
    df = pd.read_csv(input_file, sep='\t')
    print(f"  Original shape: {df.shape}")
    
    # Metadata columns to keep
    metadata_cols = ['name', 'sequence']
    
    # Get feature columns (excluding metadata)
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Remove zero-variance features
    features_to_keep = [col for col in feature_cols if col not in zero_var_features]
    cols_to_keep = metadata_cols + features_to_keep
    
    # Filter dataframe
    df_filtered = df[cols_to_keep]
    
    # Save filtered version
    df_filtered.to_csv(output_file, sep='\t', index=False)
    print(f"  Filtered shape: {df_filtered.shape}")
    print(f"  Removed {len(zero_var_features)} zero-variance features")
    
    return df_filtered.shape

def main():
    # Zero-variance features identified from train set
    zero_var_features = [
        'entropy_renyi_entropy_alpha0.0',
        'entropy_shannon_w10_max', 
        'entropy_gc_entropy_w10_max',
        'advanced_melting_min_melting_dG',
        'advanced_melting_max_melting_dG',
        'advanced_g4_g4_hotspot_count',
        'advanced_stress_max_stress_opening',
        'advanced_stress_max_opening_stretch'
    ]
    
    # Also remove bend_attention_bias_min if it has very low variance
    # It had 716 unique values but might still be problematic
    # zero_var_features.append('bend_attention_bias_min')
    
    print(f"Filtering out {len(zero_var_features)} zero-variance features...")
    
    # Process each split
    for split in ['train', 'val', 'test']:
        input_file = f'output/HepG2_{split}_descriptors.tsv'
        output_file = f'output/HepG2_{split}_descriptors_filtered.tsv'
        
        if Path(input_file).exists():
            shape = filter_zero_variance(input_file, output_file, zero_var_features)
        else:
            print(f"  {input_file} not found, skipping...")
    
    print("\n✅ Filtering complete!")
    print(f"Final feature count: {shape[1] - 2} features + 2 metadata columns")

if __name__ == "__main__":
    main()