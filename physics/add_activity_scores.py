#!/usr/bin/env python3
"""
Add normalized_log2 activity scores from original data to descriptor files
"""
import pandas as pd
import sys

def add_activity_scores(cell_type='HepG2'):
    """Add activity scores to descriptor files"""
    
    for split in ['train', 'val', 'test']:
        print(f"Processing {split}...")
        
        # Load original data with activity scores
        orig_file = f'data/{cell_type}_data/splits/{split}.tsv'
        orig_df = pd.read_csv(orig_file, sep='\t')
        print(f"  Original data: {orig_df.shape}")
        
        # Load processed descriptors
        desc_file = f'output/{cell_type}_{split}_descriptors.tsv'
        desc_df = pd.read_csv(desc_file, sep='\t')
        print(f"  Descriptor data: {desc_df.shape}")
        
        # Merge on name column to add normalized_log2
        # Keep all descriptor columns and add normalized_log2
        merged = desc_df.merge(
            orig_df[['name', 'normalized_log2']], 
            on='name', 
            how='left'
        )
        
        # Check merge success
        missing = merged['normalized_log2'].isna().sum()
        if missing > 0:
            print(f"  WARNING: {missing} sequences missing activity scores!")
        
        # Reorder to put normalized_log2 after name and sequence
        cols = merged.columns.tolist()
        cols.remove('normalized_log2')
        # Insert after sequence (position 2)
        cols.insert(2, 'normalized_log2')
        merged = merged[cols]
        
        # Save updated file
        output_file = f'output/{cell_type}_{split}_descriptors_with_activity.tsv'
        merged.to_csv(output_file, sep='\t', index=False)
        print(f"  Saved to {output_file}")
        print(f"  Activity score stats: mean={merged['normalized_log2'].mean():.3f}, std={merged['normalized_log2'].std():.3f}")
        print()

if __name__ == "__main__":
    add_activity_scores('HepG2')