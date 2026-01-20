#!/usr/bin/env python3
"""
Fast removal of constant features using column selection.
"""

import pandas as pd
import sys
from pathlib import Path

# Features to remove
CONSTANT_FEATURES = [
    'thermo_min_dH', 'thermo_max_dH', 'thermo_min_dS', 'thermo_max_dS',
    'thermo_min_dG', 'thermo_max_dG', 'entropy_renyi_entropy_alpha0.0',
    'entropy_shannon_w10_max', 'entropy_gc_entropy_w10_max',
    'advanced_melting_min_melting_dG', 'advanced_melting_max_melting_dG',
    'advanced_stacking_min_stacking_energy', 'advanced_stacking_max_stacking_energy',
    'advanced_stacking_stacking_p95', 'advanced_g4_g4_hotspot_count',
    'advanced_stress_max_stress_opening', 'advanced_stress_max_opening_stretch'
]

def process_file(input_file, output_file):
    """Process a single file."""
    print(f"Processing {input_file}...")
    
    # Read header to get columns
    with open(input_file, 'r') as f:
        header = f.readline().strip().split('\t')
    
    # Find columns to keep (not in constant features)
    cols_to_keep = [col for col in header if col not in CONSTANT_FEATURES]
    removed_count = len(header) - len(cols_to_keep)
    
    print(f"  Removing {removed_count} columns...")
    print(f"  Original: {len(header)} columns")
    print(f"  Final: {len(cols_to_keep)} columns")
    
    # Use pandas with specific columns for efficiency
    df = pd.read_csv(input_file, sep='\t', usecols=cols_to_keep)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"  Saved to {output_file}")
    
    return removed_count

def main():
    if len(sys.argv) > 1:
        # Process specific file
        cell_type = sys.argv[1]
        split = sys.argv[2] if len(sys.argv) > 2 else 'train'
        input_file = f'output/{cell_type}_{split}_descriptors.tsv'
        output_file = f'output_cleaned/{cell_type}_{split}_descriptors.tsv'
        
        Path('output_cleaned').mkdir(exist_ok=True)
        process_file(input_file, output_file)
    else:
        # Process all files
        Path('output_cleaned').mkdir(exist_ok=True)
        
        for cell_type in ['HepG2', 'K562', 'WTC11']:
            for split in ['train', 'val', 'test']:
                input_file = f'output/{cell_type}_{split}_descriptors.tsv'
                output_file = f'output_cleaned/{cell_type}_{split}_descriptors.tsv'
                
                if Path(input_file).exists():
                    process_file(input_file, output_file)
                    print()

if __name__ == "__main__":
    main()