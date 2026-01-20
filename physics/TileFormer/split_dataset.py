#!/usr/bin/env python3
"""
Split the ABPS-labeled corpus into train/val/test sets (80/10/10)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def split_dataset(input_path: str, output_dir: str, seed: int = 42):
    """Split dataset into train/val/test sets."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {input_path}")
    data = pd.read_csv(input_path, sep='\t')
    logger.info(f"Total sequences: {len(data)}")
    
    # Remove rows with missing ABPS values
    target_cols = ['std_psi_min', 'std_psi_max', 'std_psi_mean',
                  'enh_psi_min', 'enh_psi_max', 'enh_psi_mean']
    
    initial_count = len(data)
    data = data.dropna(subset=target_cols)
    removed_count = initial_count - len(data)
    logger.info(f"Removed {removed_count} sequences with missing ABPS values")
    logger.info(f"Sequences with complete ABPS values: {len(data)}")
    
    # Set random seed and shuffle
    np.random.seed(seed)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info("Data shuffled randomly")
    
    # Calculate split sizes
    n_total = len(data)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val  # Ensure all samples are used
    
    # Split data
    train_data = data.iloc[:n_train]
    val_data = data.iloc[n_train:n_train+n_val]
    test_data = data.iloc[n_train+n_val:]
    
    # Save splits
    train_path = output_dir / 'train.tsv'
    val_path = output_dir / 'val.tsv'
    test_path = output_dir / 'test.tsv'
    
    train_data.to_csv(train_path, sep='\t', index=False)
    val_data.to_csv(val_path, sep='\t', index=False)
    test_data.to_csv(test_path, sep='\t', index=False)
    
    logger.info(f"\nDataset split complete:")
    logger.info(f"  Train: {len(train_data):,} sequences -> {train_path}")
    logger.info(f"  Val:   {len(val_data):,} sequences -> {val_path}")
    logger.info(f"  Test:  {len(test_data):,} sequences -> {test_path}")
    
    # Print statistics for each split
    for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        logger.info(f"\n{split_name} set statistics:")
        logger.info(f"  GC content: {split_data['gc_content'].mean():.3f} ± {split_data['gc_content'].std():.3f}")
        logger.info(f"  CpG density: {split_data['cpg_density'].mean():.3f} ± {split_data['cpg_density'].std():.3f}")
        if 'category' in split_data.columns:
            category_counts = split_data['category'].value_counts()
            logger.info(f"  Categories: {dict(category_counts.head(5))}")
    
    # Save split information
    split_info = {
        'seed': seed,
        'total_sequences': n_total,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'train_fraction': len(train_data) / n_total,
        'val_fraction': len(val_data) / n_total,
        'test_fraction': len(test_data) / n_total
    }
    
    info_path = output_dir / 'split_info.txt'
    with open(info_path, 'w') as f:
        f.write("Dataset Split Information\n")
        f.write("=" * 50 + "\n")
        for key, value in split_info.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"\nSplit information saved to {info_path}")
    return train_data, val_data, test_data

if __name__ == "__main__":
    input_file = "/home/bcheng/TileFormer/data/corpus_50k_with_abps_optimized.tsv"
    output_directory = "/home/bcheng/TileFormer/data/splits"
    
    split_dataset(input_file, output_directory, seed=42)