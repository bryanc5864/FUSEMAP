#!/usr/bin/env python3
"""Quick test of all analyses with minimal data (10 sequences)."""

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import pandas as pd

ANALYSES = [
    ('01', '01_univariate_stability.py'),
    ('02', '02_multivariate_models.py'),
    ('03', '03_incremental_control.py'),
    ('04', '04_interaction_mapping.py'),
    ('05', '05_regime_discovery.py'),
    ('06', '06_cross_celltype_generalization.py'),
]

def create_test_data(n_samples=10):
    """Create minimal test data from existing lentiMPRA data."""
    script_dir = Path(__file__).parent.resolve()
    fusemap_root = script_dir.parent

    # Source data
    source_dir = fusemap_root / 'data' / 'lentiMPRA_data'

    # Create temp directory structure
    temp_dir = Path(tempfile.mkdtemp(prefix='test_physics_'))

    for cell_type in ['K562', 'HepG2', 'WTC11']:
        source_file = source_dir / cell_type / f'{cell_type}_train_with_features.tsv'
        if source_file.exists():
            # Read and subset
            df = pd.read_csv(source_file, sep='\t', nrows=n_samples)

            # Create output dir
            out_dir = temp_dir / cell_type
            out_dir.mkdir(parents=True, exist_ok=True)

            # Save subset
            df.to_csv(out_dir / f'{cell_type}_train_with_features.tsv', sep='\t', index=False)
            print(f"  Created test data for {cell_type}: {len(df)} rows")

    return temp_dir

def test_analysis(script_name, data_dir, output_dir, cell_types, min_celltypes=1):
    """Test a single analysis."""
    script_path = Path(__file__).parent / script_name

    if len(cell_types) < min_celltypes:
        return 'skipped', f'requires {min_celltypes}+ cell types'

    cmd = [
        sys.executable,
        str(script_path),
        '--data_dir', str(data_dir),
        '--output_dir', str(output_dir),
        '--cell_types'] + cell_types + [
        '--split', 'train',
        '--n_jobs', '4'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            return 'success', None
        else:
            return 'failed', result.stderr[-500:] if result.stderr else result.stdout[-500:]
    except subprocess.TimeoutExpired:
        return 'timeout', 'Exceeded 120s'
    except Exception as e:
        return 'error', str(e)

def main():
    print("=" * 60)
    print("PHYSICS ANALYSES TEST (10 sequences)")
    print("=" * 60)

    # Create test data
    print("\nCreating minimal test data...")
    test_data_dir = create_test_data(n_samples=10)
    print(f"Test data directory: {test_data_dir}")

    # Create test output directory
    test_output_dir = Path(tempfile.mkdtemp(prefix='test_output_'))
    print(f"Test output directory: {test_output_dir}")

    cell_types = ['K562', 'HepG2', 'WTC11']

    results = {}

    for analysis_id, script_name in ANALYSES:
        print(f"\n{'='*60}")
        print(f"Testing [{analysis_id}] {script_name}")
        print('='*60)

        output_dir = test_output_dir / f'{analysis_id}_test'
        output_dir.mkdir(exist_ok=True)

        min_cts = 2 if analysis_id == '06' else 1

        status, error = test_analysis(
            script_name=script_name,
            data_dir=test_data_dir,
            output_dir=output_dir,
            cell_types=cell_types,
            min_celltypes=min_cts
        )

        results[analysis_id] = {'status': status, 'error': error}

        if status == 'success':
            print(f"  [OK] {script_name}")
        else:
            print(f"  [FAIL] {script_name}: {status}")
            if error:
                print(f"  Error: {error}")

    # Summary
    print(f"\n\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)

    all_passed = True
    for analysis_id, result in results.items():
        symbol = '[OK]' if result['status'] == 'success' else '[FAIL]'
        print(f"  {symbol} {analysis_id}: {result['status']}")
        if result['status'] != 'success':
            all_passed = False

    # Cleanup
    print(f"\nCleaning up test directories...")
    shutil.rmtree(test_data_dir, ignore_errors=True)
    shutil.rmtree(test_output_dir, ignore_errors=True)

    if all_passed:
        print("\n[SUCCESS] All analyses passed!")
        return 0
    else:
        print("\n[FAILED] Some analyses failed - check errors above")
        return 1

if __name__ == '__main__':
    sys.exit(main())
