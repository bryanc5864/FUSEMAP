#!/usr/bin/env python3
"""
Run Therapeutic Enhancer Design for all cell types.

Generates cell-type specific enhancers for:
- HepG2 (liver) vs K562, WTC11
- WTC11 (stem) vs K562, HepG2
- K562 (blood) vs HepG2, WTC11

Each run produces 50 diverse candidates optimized for specificity.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Configuration
CELL_TYPES = ['HepG2', 'WTC11', 'K562']
N_VAE_CANDIDATES = 1000  # Generate 1000, select top 100
N_SELECT = 100
OUTPUT_BASE = Path(__file__).parent.parent / 'results' / 'therapeutic_all_celltypes'


def run_pipeline(target_cell: str, device: str = 'cuda'):
    """Run therapeutic pipeline for a single target cell type."""

    # Background cells are the other two
    background = [c for c in CELL_TYPES if c != target_cell]

    output_dir = OUTPUT_BASE / f'therapeutic_{target_cell.lower()}'

    print(f"\n{'='*60}")
    print(f"Running: Target={target_cell}, Background={background}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'therapeutic_enhancer_pipeline.py'),
        '--target-cell', target_cell,
        '--background-cells', *background,
        '--n-vae', str(N_VAE_CANDIDATES),
        '--n-select', str(N_SELECT),
        '--output-dir', str(output_dir),
        '--device', device,
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"WARNING: Pipeline for {target_cell} exited with code {result.returncode}")

    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run therapeutic pipeline for all cell types')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--targets', type=str, nargs='+', default=CELL_TYPES,
                        choices=CELL_TYPES, help='Target cell types to run')
    args = parser.parse_args()

    # Create output directory
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"Therapeutic Enhancer Design - All Cell Types")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Targets: {args.targets}")
    print(f"N candidates per target: {N_SELECT}")

    results = {}

    for target in args.targets:
        success = run_pipeline(target, device=args.device)
        results[target] = 'SUCCESS' if success else 'FAILED'

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for target, status in results.items():
        print(f"  {target}: {status}")

    # Check outputs
    print(f"\nOutput files:")
    for target in args.targets:
        output_dir = OUTPUT_BASE / f'therapeutic_{target.lower()}'
        csv_file = output_dir / 'therapeutic_enhancers.csv'
        if csv_file.exists():
            import pandas as pd
            df = pd.read_csv(csv_file)
            print(f"  {target}: {len(df)} candidates, max specificity={df['specificity_score'].max():.3f}")
        else:
            print(f"  {target}: No output found")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
