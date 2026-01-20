#!/usr/bin/env python3
"""
run_analyses.py - Physics Analyses Coordinator

Coordinator script to run all physics analyses in sequence.
Results are saved to /results subfolder with organized structure.

Supports GPU acceleration (cuML) and parallel CPU processing.

Usage:
    python run_analyses.py --human              # Run on human lentiMPRA data
    python run_analyses.py --human --drosophila # Run on both
    python run_analyses.py --all                # Run on all datasets
    python run_analyses.py --list               # List available datasets

Analyses (run chronologically):
    01. Univariate Stability Analysis
    02. Physics-Only Multivariate Models
    03. Incremental Control Analysis
    04. Physics x TF Interaction Mapping
    05. Regime Discovery (HDBSCAN + UMAP)
    06. Cross-Cell-Type Generalization
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import os
import multiprocessing
import shutil

# Analysis definitions
ANALYSES = [
    {
        'id': '01',
        'name': 'Univariate Stability',
        'script': '01_univariate_stability.py',
        'description': 'Correlations, partial correlations, CV stability',
        'outputs': ['univariate_*.csv', 'figures/', 'report.txt']
    },
    {
        'id': '02',
        'name': 'Multivariate Models',
        'script': '02_multivariate_models.py',
        'description': 'Elastic Net, Ridge, GAM on physics features',
        'outputs': ['multivariate_*.json', 'figures/', 'report.txt']
    },
    {
        'id': '03',
        'name': 'Incremental Control',
        'script': '03_incremental_control.py',
        'description': 'Physics -> +PWM aggregates -> +all PWM, coefficient attenuation',
        'outputs': ['incremental_*.json', 'figures/', 'report.txt']
    },
    {
        'id': '04',
        'name': 'Interaction Mapping',
        'script': '04_interaction_mapping.py',
        'description': 'Physics x TF interaction H-statistics',
        'outputs': ['interactions_*.json', 'figures/', 'report.txt']
    },
    {
        'id': '05',
        'name': 'Regime Discovery',
        'script': '05_regime_discovery.py',
        'description': 'HDBSCAN clustering + UMAP embedding',
        'outputs': ['regimes_*.json', 'embedding_*.npz', 'figures/', 'report.txt']
    },
    {
        'id': '06',
        'name': 'Cross-Cell-Type',
        'script': '06_cross_celltype_generalization.py',
        'description': 'Train/test transfer across cell types',
        'outputs': ['cross_celltype_results.json', 'figures/', 'report.txt'],
        'min_celltypes': 2
    }
]

# Dataset configurations
DATASETS = {
    'human': {
        'data_dir': 'data/lentiMPRA_data',
        'cell_types': ['K562', 'HepG2', 'WTC11'],
        'activity_col': 'activity',
        'description': 'Human lentiMPRA (K562, HepG2, WTC11)'
    },
    'drosophila_dev': {
        'data_dir': 'data/drosophila_data',
        'cell_types': ['S2'],
        'activity_col': 'Dev_log2_enrichment',
        'description': 'Drosophila S2 cells (Developmental enhancers)'
    },
    'drosophila_hk': {
        'data_dir': 'data/drosophila_data',
        'cell_types': ['S2'],
        'activity_col': 'Hk_log2_enrichment',
        'description': 'Drosophila S2 cells (Housekeeping enhancers)'
    },
    'yeast': {
        'data_dir': 'data/DREAM_data',
        'cell_types': ['yeast'],
        'activity_col': 'activity',
        'description': 'DREAM yeast data'
    },
    # Arabidopsis (two activity readouts)
    'arabidopsis_leaf': {
        'data_dir': 'output',
        'cell_types': ['arabidopsis'],
        'activity_col': 'enrichment_leaf',
        'file_pattern': '{cell_type}_{split}_descriptors_with_activity.tsv',
        'description': 'Arabidopsis (leaf enrichment)'
    },
    'arabidopsis_proto': {
        'data_dir': 'output',
        'cell_types': ['arabidopsis'],
        'activity_col': 'enrichment_proto',
        'file_pattern': '{cell_type}_{split}_descriptors_with_activity.tsv',
        'description': 'Arabidopsis (protoplast enrichment)'
    },
    # Sorghum (two activity readouts)
    'sorghum_leaf': {
        'data_dir': 'output',
        'cell_types': ['sorghum'],
        'activity_col': 'enrichment_leaf',
        'file_pattern': '{cell_type}_{split}_descriptors_with_activity.tsv',
        'description': 'Sorghum (leaf enrichment)'
    },
    'sorghum_proto': {
        'data_dir': 'output',
        'cell_types': ['sorghum'],
        'activity_col': 'enrichment_proto',
        'file_pattern': '{cell_type}_{split}_descriptors_with_activity.tsv',
        'description': 'Sorghum (protoplast enrichment)'
    },
    # Maize (two activity readouts)
    'maize_leaf': {
        'data_dir': 'output',
        'cell_types': ['maize'],
        'activity_col': 'enrichment_leaf',
        'file_pattern': '{cell_type}_{split}_descriptors_with_activity.tsv',
        'description': 'Maize (leaf enrichment)'
    },
    'maize_proto': {
        'data_dir': 'output',
        'cell_types': ['maize'],
        'activity_col': 'enrichment_proto',
        'file_pattern': '{cell_type}_{split}_descriptors_with_activity.tsv',
        'description': 'Maize (protoplast enrichment)'
    }
}


def get_script_path() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).parent.resolve()


def get_fusemap_root() -> Path:
    """Get the FUSEMAP root directory."""
    return get_script_path().parent


def get_results_dir() -> Path:
    """Get the results directory in FUSEMAP root."""
    return get_fusemap_root() / 'results'


def check_dataset_exists(dataset_key: str) -> bool:
    """Check if a dataset has labeled data."""
    config = DATASETS[dataset_key]
    data_dir = get_fusemap_root() / config['data_dir']

    for cell_type in config['cell_types']:
        # Check if using custom file pattern
        if 'file_pattern' in config:
            pattern = config['file_pattern']
            train_file = data_dir / pattern.format(cell_type=cell_type, split='train')
        else:
            train_file = data_dir / cell_type / f'{cell_type}_train_with_features.tsv'
            if not train_file.exists():
                train_file = data_dir / f'{cell_type}_train_with_features.tsv'

        if not train_file.exists():
            return False
    return True


def run_analysis(
    analysis: dict,
    data_dir: Path,
    output_dir: Path,
    cell_types: list,
    split: str = 'train',
    n_jobs: int = -1,
    activity_col: str = 'activity'
) -> dict:
    """Run a single analysis script."""
    script_path = get_script_path() / analysis['script']

    if not script_path.exists():
        return {'status': 'error', 'message': f"Script not found: {script_path}"}

    # Create output directory
    analysis_output = output_dir / f"{analysis['id']}_{analysis['name'].lower().replace(' ', '_').replace('-', '_')}"
    analysis_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        '--data_dir', str(data_dir),
        '--output_dir', str(analysis_output),
        '--cell_types'] + cell_types + [
        '--split', split,
        '--n_jobs', str(n_jobs),
        '--activity_col', activity_col
    ]

    print(f"\n{'='*70}")
    print(f"[{analysis['id']}] {analysis['name']}")
    print(f"    {analysis['description']}")
    print(f"    Output: {analysis_output}")
    print('='*70)

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            cwd=str(get_fusemap_root())
        )

        return {
            'status': 'success' if result.returncode == 0 else 'failed',
            'returncode': result.returncode,
            'output_dir': str(analysis_output)
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def consolidate_reports(output_dir: Path, dataset_key: str) -> Path:
    """Consolidate all text reports into a single master report."""
    master_report = output_dir / f'MASTER_REPORT_{dataset_key}.txt'

    with open(master_report, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"PHYSICS ANALYSIS MASTER REPORT - {dataset_key.upper()}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for analysis in ANALYSES:
            analysis_dir = output_dir / f"{analysis['id']}_{analysis['name'].lower().replace(' ', '_').replace('-', '_')}"
            report_file = analysis_dir / 'report.txt'

            if report_file.exists():
                f.write("\n" + "#" * 80 + "\n")
                f.write(f"# {analysis['id']}. {analysis['name'].upper()}\n")
                f.write("#" * 80 + "\n\n")

                with open(report_file, 'r') as rf:
                    f.write(rf.read())
                f.write("\n")

    print(f"\nMaster report saved to: {master_report}")
    return master_report


def run_all_analyses(
    datasets_to_run: list,
    analyses_to_run: list = None,
    split: str = 'train',
    n_jobs: int = -1
) -> dict:
    """Run all analyses for specified datasets."""

    results_base = get_results_dir()
    results_base.mkdir(parents=True, exist_ok=True)

    if analyses_to_run is None:
        analyses_to_run = ANALYSES

    results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Check system resources
    n_cpus = multiprocessing.cpu_count()
    if n_jobs == -1:
        n_jobs = n_cpus

    print(f"\n{'#'*70}")
    print(f"# PHYSICS ANALYSES COORDINATOR")
    print(f"# CPUs: {n_cpus}, Using: {n_jobs} jobs")
    print(f"# Results directory: {results_base}")
    print(f"{'#'*70}")

    for dataset_key in datasets_to_run:
        config = DATASETS[dataset_key]

        print(f"\n\n{'#'*70}")
        print(f"# DATASET: {config['description']}")
        print(f"{'#'*70}")

        # Check dataset exists
        if not check_dataset_exists(dataset_key):
            print(f"WARNING: Dataset '{dataset_key}' not found or not labeled.")
            print(f"         Run process_descriptors.py first to label this dataset.")
            results[dataset_key] = {'status': 'skipped', 'reason': 'data not found'}
            continue

        data_dir = get_fusemap_root() / config['data_dir']
        output_dir = results_base / dataset_key
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_results = {}

        for analysis in analyses_to_run:
            # Skip cross-cell-type for single cell type datasets
            min_cts = analysis.get('min_celltypes', 1)
            if len(config['cell_types']) < min_cts:
                print(f"\nSkipping {analysis['name']} (requires {min_cts}+ cell types)")
                dataset_results[analysis['id']] = {
                    'status': 'skipped',
                    'reason': f'requires {min_cts}+ cell types'
                }
                continue

            result = run_analysis(
                analysis=analysis,
                data_dir=data_dir,
                output_dir=output_dir,
                cell_types=config['cell_types'],
                split=split,
                n_jobs=n_jobs,
                activity_col=config.get('activity_col', 'activity')
            )
            dataset_results[analysis['id']] = result

        results[dataset_key] = dataset_results

        # Consolidate reports for this dataset
        consolidate_reports(output_dir, dataset_key)

    # Print summary
    print(f"\n\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print('='*70)

    for dataset_key, dataset_results in results.items():
        print(f"\n{dataset_key.upper()}:")
        if isinstance(dataset_results, dict) and 'status' in dataset_results:
            print(f"  {dataset_results['status']}: {dataset_results.get('reason', '')}")
        else:
            for analysis_id, result in dataset_results.items():
                status = result.get('status', 'unknown')
                symbol = {
                    'success': '[OK]',
                    'failed': '[FAIL]',
                    'skipped': '[SKIP]',
                    'error': '[ERR]'
                }.get(status, '[?]')
                print(f"  {symbol} {analysis_id}: {status}")

    # Save run log
    log_path = results_base / f'run_log_{timestamp}.json'
    with open(log_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'datasets': datasets_to_run,
            'n_jobs': n_jobs,
            'results': results
        }, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Run log: {log_path}")
    print(f"Results: {results_base}")
    print('='*70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Physics Analyses Coordinator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_analyses.py --human
    python run_analyses.py --human --drosophila
    python run_analyses.py --all
    python run_analyses.py --human --analyses 01 02 03
    python run_analyses.py --list
        """
    )

    # Dataset flags
    parser.add_argument('--human', action='store_true',
                       help='Run on human lentiMPRA data (K562, HepG2, WTC11)')
    parser.add_argument('--drosophila', action='store_true',
                       help='Run on Drosophila S2 data (both Dev and Hk activities)')
    parser.add_argument('--drosophila_dev', action='store_true',
                       help='Run on Drosophila S2 data (Developmental enhancers only)')
    parser.add_argument('--drosophila_hk', action='store_true',
                       help='Run on Drosophila S2 data (Housekeeping enhancers only)')
    parser.add_argument('--yeast', action='store_true',
                       help='Run on DREAM yeast data')
    # Plant datasets (each with leaf and proto activity readouts)
    parser.add_argument('--arabidopsis', action='store_true',
                       help='Run on Arabidopsis (both leaf and proto)')
    parser.add_argument('--arabidopsis_leaf', action='store_true',
                       help='Run on Arabidopsis (leaf enrichment only)')
    parser.add_argument('--arabidopsis_proto', action='store_true',
                       help='Run on Arabidopsis (protoplast enrichment only)')
    parser.add_argument('--sorghum', action='store_true',
                       help='Run on Sorghum (both leaf and proto)')
    parser.add_argument('--sorghum_leaf', action='store_true',
                       help='Run on Sorghum (leaf enrichment only)')
    parser.add_argument('--sorghum_proto', action='store_true',
                       help='Run on Sorghum (protoplast enrichment only)')
    parser.add_argument('--maize', action='store_true',
                       help='Run on Maize (both leaf and proto)')
    parser.add_argument('--maize_leaf', action='store_true',
                       help='Run on Maize (leaf enrichment only)')
    parser.add_argument('--maize_proto', action='store_true',
                       help='Run on Maize (protoplast enrichment only)')
    parser.add_argument('--plants', action='store_true',
                       help='Run on all plant datasets (arabidopsis, sorghum, maize)')
    parser.add_argument('--all', action='store_true',
                       help='Run on all available datasets')

    # Analysis selection
    parser.add_argument('--analyses', type=str, nargs='+',
                       choices=['01', '02', '03', '04', '05', '06'],
                       help='Specific analyses to run (default: all)')

    # Options
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Data split to use')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all CPUs)')
    parser.add_argument('--list', action='store_true',
                       help='List available analyses and datasets')

    args = parser.parse_args()

    # List mode
    if args.list:
        print("\n" + "=" * 60)
        print("AVAILABLE ANALYSES")
        print("=" * 60)
        for analysis in ANALYSES:
            print(f"\n  {analysis['id']}. {analysis['name']}")
            print(f"      {analysis['description']}")

        print("\n" + "=" * 60)
        print("AVAILABLE DATASETS")
        print("=" * 60)
        for key, config in DATASETS.items():
            exists = "[OK]" if check_dataset_exists(key) else "[--]"
            print(f"  {exists} --{key:15s} : {config['description']}")
        print(f"\n  [OK] = Data available, [--] = Needs labeling first")
        return

    # Determine datasets
    datasets_to_run = []
    if args.all:
        datasets_to_run = list(DATASETS.keys())
    else:
        if args.human:
            datasets_to_run.append('human')
        if args.drosophila:
            # Run both Dev and Hk
            datasets_to_run.append('drosophila_dev')
            datasets_to_run.append('drosophila_hk')
        if args.drosophila_dev:
            datasets_to_run.append('drosophila_dev')
        if args.drosophila_hk:
            datasets_to_run.append('drosophila_hk')
        if args.yeast:
            datasets_to_run.append('yeast')
        # Plant datasets
        if args.plants:
            datasets_to_run.extend(['arabidopsis_leaf', 'arabidopsis_proto',
                                   'sorghum_leaf', 'sorghum_proto',
                                   'maize_leaf', 'maize_proto'])
        if args.arabidopsis:
            datasets_to_run.extend(['arabidopsis_leaf', 'arabidopsis_proto'])
        if args.arabidopsis_leaf:
            datasets_to_run.append('arabidopsis_leaf')
        if args.arabidopsis_proto:
            datasets_to_run.append('arabidopsis_proto')
        if args.sorghum:
            datasets_to_run.extend(['sorghum_leaf', 'sorghum_proto'])
        if args.sorghum_leaf:
            datasets_to_run.append('sorghum_leaf')
        if args.sorghum_proto:
            datasets_to_run.append('sorghum_proto')
        if args.maize:
            datasets_to_run.extend(['maize_leaf', 'maize_proto'])
        if args.maize_leaf:
            datasets_to_run.append('maize_leaf')
        if args.maize_proto:
            datasets_to_run.append('maize_proto')

    # Remove duplicates while preserving order
    datasets_to_run = list(dict.fromkeys(datasets_to_run))

    if not datasets_to_run:
        print("Error: No dataset specified.")
        print("       Use --human, --drosophila, --yeast, --plants, --arabidopsis, --sorghum, --maize, or --all")
        print("       Use --list to see available datasets")
        sys.exit(1)

    # Determine analyses
    analyses_to_run = ANALYSES
    if args.analyses:
        analyses_to_run = [a for a in ANALYSES if a['id'] in args.analyses]

    # Run
    run_all_analyses(
        datasets_to_run=datasets_to_run,
        analyses_to_run=analyses_to_run,
        split=args.split,
        n_jobs=args.n_jobs
    )


if __name__ == '__main__':
    main()
