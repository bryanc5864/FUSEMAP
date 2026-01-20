#!/usr/bin/env python3
"""
Merge TileFormer features into plant physics data files.

Combines:
- physics/output/{species}_{split}_descriptors_with_activity.tsv (physics features)
- data/plant_data/jores2021/processed/{species}/{species}_{split}_tileformer.tsv (tileformer features)

Output:
- physics/output/{species}_{split}_all_features.tsv (unified)
"""

import pandas as pd
from pathlib import Path
import argparse


def merge_plant_data(species: str, split: str, root: Path):
    """Merge physics and TileFormer features for a plant dataset."""

    # File paths
    physics_file = root / f'physics/output/{species}_{split}_descriptors_with_activity.tsv'
    tileformer_file = root / f'data/plant_data/jores2021/processed/{species}/{species}_{split}_tileformer.tsv'
    output_file = root / f'physics/output/{species}_{split}_all_features.tsv'

    # Check files exist
    if not physics_file.exists():
        print(f"  Physics file not found: {physics_file}")
        return False
    if not tileformer_file.exists():
        print(f"  TileFormer file not found: {tileformer_file}")
        return False

    print(f"  Loading physics: {physics_file}")
    df_physics = pd.read_csv(physics_file, sep='\t')

    print(f"  Loading tileformer: {tileformer_file}")
    df_tileformer = pd.read_csv(tileformer_file, sep='\t')

    # Get only tileformer columns (exclude duplicates like name, sequence, enrichment)
    tileformer_cols = [c for c in df_tileformer.columns if c.startswith('tileformer_')]
    print(f"  TileFormer columns: {len(tileformer_cols)}")

    # Merge on 'name' column
    if 'name' in df_physics.columns and 'name' in df_tileformer.columns:
        df_tileformer_subset = df_tileformer[['name'] + tileformer_cols]
        df_merged = df_physics.merge(df_tileformer_subset, on='name', how='left')
    else:
        # If no name column, assume same order and concat
        print("  Warning: No 'name' column, concatenating by index")
        df_merged = pd.concat([df_physics, df_tileformer[tileformer_cols]], axis=1)

    print(f"  Merged: {len(df_merged)} rows, {len(df_merged.columns)} columns")

    # Save
    df_merged.to_csv(output_file, sep='\t', index=False)
    print(f"  Saved: {output_file}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Merge plant physics and TileFormer data')
    parser.add_argument('--species', nargs='+', default=['arabidopsis', 'maize', 'sorghum'],
                       help='Species to process')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'],
                       help='Data splits to process')

    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent

    print("Merging plant physics + TileFormer data")
    print("=" * 50)

    for species in args.species:
        print(f"\n{species.upper()}")
        for split in args.splits:
            print(f"  Processing {split}...")
            success = merge_plant_data(species, split, root)
            if not success:
                print(f"  SKIPPED {species} {split}")

    print("\n" + "=" * 50)
    print("Done! Merged files in physics/output/")


if __name__ == '__main__':
    main()
