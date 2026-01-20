#!/usr/bin/env python3
"""
Merge plant physics descriptor files with original activity files.

This script joins the physics descriptors (PWM features, physical properties)
with the original plant activity labels (enrichment_leaf, enrichment_proto).
"""

import pandas as pd
import os
from pathlib import Path

# Define paths
ORIGINAL_DATA_DIR = Path("/home/bcheng/sequence_optimization/FUSEMAP/data/plant_data/jores2021/processed")
PHYSICS_OUTPUT_DIR = Path("/home/bcheng/sequence_optimization/FUSEMAP/physics/output")

# Species and splits to process
SPECIES = ['arabidopsis', 'sorghum', 'maize']
SPLITS = ['train', 'test']

def merge_files(species: str, split: str) -> None:
    """Merge physics descriptors with original activity labels."""

    # File paths
    original_file = ORIGINAL_DATA_DIR / species / f"{species}_{split}.tsv"
    physics_file = PHYSICS_OUTPUT_DIR / f"{species}_{split}_descriptors.tsv"
    output_file = PHYSICS_OUTPUT_DIR / f"{species}_{split}_descriptors_with_activity.tsv"

    print(f"\nProcessing {species} {split}...")
    print(f"  Original: {original_file}")
    print(f"  Physics:  {physics_file}")

    # Check files exist
    if not original_file.exists():
        print(f"  ERROR: Original file not found!")
        return
    if not physics_file.exists():
        print(f"  ERROR: Physics file not found!")
        return

    # Read files
    print(f"  Reading original file...")
    original_df = pd.read_csv(original_file, sep='\t')
    print(f"    Rows: {len(original_df)}, Columns: {list(original_df.columns)}")

    print(f"  Reading physics file...")
    physics_df = pd.read_csv(physics_file, sep='\t')
    print(f"    Rows: {len(physics_df)}, Columns: {len(physics_df.columns)}")

    # Verify row counts match
    if len(original_df) != len(physics_df):
        print(f"  ERROR: Row count mismatch! Original={len(original_df)}, Physics={len(physics_df)}")
        return

    # Check if sequences match (they should be in same order)
    seq_match = (original_df['sequence'].values == physics_df['sequence'].values).all()
    if seq_match:
        print(f"  Sequences match in order - using direct merge")
        # Direct column merge (faster, preserves order)
        merged_df = physics_df.copy()
        merged_df['name'] = original_df['name'].values
        merged_df['enrichment_leaf'] = original_df['enrichment_leaf'].values
        merged_df['enrichment_proto'] = original_df['enrichment_proto'].values
    else:
        print(f"  Sequences not in same order - using merge on sequence column")
        # Merge on sequence
        activity_cols = original_df[['sequence', 'name', 'enrichment_leaf', 'enrichment_proto']]
        merged_df = physics_df.merge(activity_cols, on='sequence', how='left')

        # Check for missing merges
        missing = merged_df['enrichment_leaf'].isna().sum()
        if missing > 0:
            print(f"  WARNING: {missing} sequences didn't match!")

    # Reorder columns - put identifiers and activities first
    id_cols = ['seq_id', 'name', 'sequence', 'condition', 'enrichment_leaf', 'enrichment_proto']
    feature_cols = [c for c in merged_df.columns if c not in id_cols]
    merged_df = merged_df[id_cols + feature_cols]

    # Save merged file
    print(f"  Saving to: {output_file}")
    merged_df.to_csv(output_file, sep='\t', index=False)
    print(f"  Output: {len(merged_df)} rows, {len(merged_df.columns)} columns")

    # Summary stats
    print(f"  Activity stats:")
    print(f"    enrichment_leaf: mean={merged_df['enrichment_leaf'].mean():.3f}, std={merged_df['enrichment_leaf'].std():.3f}")
    print(f"    enrichment_proto: mean={merged_df['enrichment_proto'].mean():.3f}, std={merged_df['enrichment_proto'].std():.3f}")

def main():
    print("=" * 60)
    print("Merging plant physics descriptors with activity labels")
    print("=" * 60)

    for species in SPECIES:
        for split in SPLITS:
            merge_files(species, split)

    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)

    # List output files
    print("\nOutput files:")
    for species in SPECIES:
        for split in SPLITS:
            output_file = PHYSICS_OUTPUT_DIR / f"{species}_{split}_descriptors_with_activity.tsv"
            if output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"  {output_file.name}: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
