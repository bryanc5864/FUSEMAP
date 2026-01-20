#!/usr/bin/env python
"""
CLI Runner for PhysicsInterpreter.

Run attribution and mechanistic analysis through the physics pathway.

Usage:
    # Physics feature attribution
    python run_interpreter.py attribution --cell-type WTC11

    # Integrated gradients for sequences
    python run_interpreter.py ig --sequence ATGC... --cell-type WTC11

    # Mediation analysis
    python run_interpreter.py mediation --cell-type WTC11

    # Landscape analysis
    python run_interpreter.py landscape --cell-type WTC11 --compute-shap

    # Full analysis pipeline
    python run_interpreter.py full --cell-type WTC11
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import warnings

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from physics.PhysicsInterpreter.config import (
    InterpreterConfig, MODEL_PATHS, PHYSICS_FAMILIES, get_fusemap_root
)
from physics.PhysicsInterpreter.integrated_gradients import IntegratedGradients
from physics.PhysicsInterpreter.physics_attribution import PhysicsAttributor
from physics.PhysicsInterpreter.mediation_analysis import MediationAnalyzer
from physics.PhysicsInterpreter.landscape_analysis import LandscapeAnalyzer


def setup_output_dir(config: InterpreterConfig, analysis_type: str) -> Path:
    """Create output directory for analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.get_output_dir() / f"{analysis_type}_{config.cell_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_physics_data(cell_type: str, split: str = 'train', max_samples: int = None, max_features: int = None) -> tuple:
    """Load physics features and activity data for a cell type."""
    root = get_fusemap_root()

    # Map cell types to their data directories
    data_dir_map = {
        'WTC11': 'physics/data/lentiMPRA_data/WTC11',
        'HepG2': 'physics/data/lentiMPRA_data/HepG2',
        'K562': 'physics/data/lentiMPRA_data/K562',
        'S2': 'physics/data/drosophila_data/S2',
    }

    # Plant datasets (stored in physics/output/)
    plant_datasets = ['arabidopsis', 'maize', 'sorghum']

    # Try different data locations
    data_paths = []

    # Plant data - unified files in physics/output/
    if cell_type.lower() in plant_datasets:
        species = cell_type.lower()
        data_paths.extend([
            root / 'physics' / 'output' / f'{species}_{split}_all_features.tsv',
            root / 'physics' / 'output' / f'{species}_train_all_features.tsv',  # Fallback
            root / 'physics' / 'output' / f'{species}_{split}_descriptors_with_activity.tsv',
        ])

    # Standard lentiMPRA format (human/animal)
    elif cell_type in data_dir_map:
        data_dir = root / data_dir_map[cell_type]
        data_paths.extend([
            data_dir / f'{cell_type}_{split}_with_features.tsv',
            data_dir / f'{cell_type}_train_with_features.tsv',  # Fallback to train
        ])

    # Also try direct paths
    data_paths.extend([
        root / 'physics' / 'data' / f'{cell_type}_physics_features.csv',
        root / 'physics' / 'output' / f'{cell_type}_physics.csv',
    ])

    df = None
    for path in data_paths:
        if path.exists():
            print(f"Loading data from {path}")
            sep = '\t' if str(path).endswith('.tsv') else ','
            df = pd.read_csv(path, sep=sep)
            break

    if df is None:
        raise FileNotFoundError(
            f"Could not find physics data for {cell_type}. "
            f"Tried: {[str(p) for p in data_paths]}"
        )

    # Identify physics feature columns
    physics_prefixes = []
    for family, prefixes in PHYSICS_FAMILIES.items():
        physics_prefixes.extend(prefixes)

    feature_cols = []
    for col in df.columns:
        for prefix in physics_prefixes:
            if col.startswith(prefix):
                feature_cols.append(col)
                break

    # Find activity column
    activity_cols = [
        'activity', 'log_activity', 'expression', 'log_expression', 'y',
        'enrichment_leaf', 'enrichment_proto',  # Plant datasets
        'Dev_log2_enrichment', 'Hk_log2_enrichment',  # S2/Drosophila
    ]
    activity_col = None
    for col in activity_cols:
        if col in df.columns:
            activity_col = col
            break

    if activity_col is None:
        raise ValueError(f"Could not find activity column. Available: {df.columns.tolist()}")

    X = df[feature_cols].values
    y = df[activity_col].values
    feature_names = feature_cols

    # Handle NaN - impute with column mean instead of dropping rows
    for i in range(X.shape[1]):
        col_mean = np.nanmean(X[:, i])
        if np.isnan(col_mean):
            col_mean = 0.0
        X[np.isnan(X[:, i]), i] = col_mean

    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    # Limit samples if requested
    if max_samples is not None and len(y) > max_samples:
        indices = np.random.choice(len(y), max_samples, replace=False)
        X = X[indices]
        y = y[indices]

    # Limit features if requested (select by variance)
    if max_features is not None and len(feature_names) > max_features:
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-max_features:]
        X = X[:, top_indices]
        feature_names = [feature_names[i] for i in top_indices]

    print(f"Loaded {len(y)} samples with {len(feature_names)} physics features")

    return X, y, feature_names


def run_attribution(args):
    """Run physics feature attribution analysis."""
    print(f"\n{'='*60}")
    print(f"Physics Feature Attribution - {args.cell_type}")
    print(f"{'='*60}\n")

    config = InterpreterConfig(
        cell_type=args.cell_type,
        attribution_probe_type=args.probe_type,
        attribution_alpha=args.alpha
    )

    output_dir = setup_output_dir(config, 'attribution')
    print(f"Output directory: {output_dir}")

    # Load data
    max_samples = getattr(args, 'max_samples', None)
    max_features = getattr(args, 'max_features', None)
    X, y, feature_names = load_physics_data(args.cell_type, max_samples=max_samples, max_features=max_features)

    # Run attribution
    attributor = PhysicsAttributor(config)
    attributor.fit(X, y, feature_names)
    result = attributor.get_attribution(top_n=args.top_n)

    # Print results
    print(f"\nProbe Performance:")
    print(f"  R² = {result.probe_r2:.4f}")
    print(f"  Pearson r = {result.probe_pearson:.4f}")

    print(f"\nFamily Contributions:")
    for family, pct in sorted(result.family_contributions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {family}: {pct:.1f}%")

    print(f"\nTop Positive Features:")
    for name, coef in result.top_positive[:10]:
        print(f"  {name}: {coef:.4f}")

    print(f"\nTop Negative Features:")
    for name, coef in result.top_negative[:10]:
        print(f"  {name}: {coef:.4f}")

    # Save results
    attributor.save(output_dir / 'attribution_results.json')

    # Save family breakdown
    family_df = attributor.get_family_breakdown()
    family_df.to_csv(output_dir / 'family_breakdown.csv', index=False)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({
            'cell_type': args.cell_type,
            'probe_type': args.probe_type,
            'alpha': args.alpha,
            'n_samples': len(y),
            'n_features': len(feature_names)
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return result


def run_integrated_gradients(args):
    """Run integrated gradients for sequence attribution."""
    print(f"\n{'='*60}")
    print(f"Integrated Gradients - {args.cell_type}")
    print(f"{'='*60}\n")

    config = InterpreterConfig(
        cell_type=args.cell_type,
        ig_steps=args.steps,
        ig_baseline=args.baseline
    )

    # Load model
    ig = IntegratedGradients(config)
    ig.load_model()

    # Process sequence(s)
    if args.sequence:
        sequences = [args.sequence]
    elif args.sequence_file:
        with open(args.sequence_file) as f:
            sequences = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Must provide --sequence or --sequence-file")

    output_dir = setup_output_dir(config, 'ig')
    print(f"Output directory: {output_dir}")

    results = []
    for i, seq in enumerate(sequences):
        print(f"\nProcessing sequence {i+1}/{len(sequences)} (length {len(seq)})")

        result = ig.attribute(seq, n_steps=args.steps)

        print(f"  Prediction: {result.prediction:.4f}")
        print(f"  Baseline prediction: {result.baseline_prediction:.4f}")
        print(f"  Convergence delta: {result.convergence_delta:.6f}")

        # Get position importance
        importance = ig.get_nucleotide_importance(result)
        top_positions = np.argsort(importance)[-10:][::-1]
        print(f"  Top positions: {top_positions.tolist()}")

        results.append({
            'sequence_idx': i,
            'sequence_length': len(seq),
            'prediction': result.prediction,
            'baseline_prediction': result.baseline_prediction,
            'convergence_delta': result.convergence_delta,
            'top_positions': top_positions.tolist()
        })

        # Save attributions
        np.save(output_dir / f'attributions_{i}.npy', result.attributions)

    # Save summary
    with open(output_dir / 'ig_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return results


def run_mediation(args):
    """Run mediation analysis."""
    print(f"\n{'='*60}")
    print(f"Mediation Analysis - {args.cell_type}")
    print(f"{'='*60}\n")

    config = InterpreterConfig(
        cell_type=args.cell_type,
        mediation_n_bootstrap=args.n_bootstrap,
        mediation_confidence=args.confidence
    )

    output_dir = setup_output_dir(config, 'mediation')
    print(f"Output directory: {output_dir}")

    # Load physics data
    max_samples = getattr(args, 'max_samples', None)
    max_features = getattr(args, 'max_features', None)
    X_physics, y, feature_names = load_physics_data(args.cell_type, max_samples=max_samples, max_features=max_features)

    # For mediation, we need sequence representations
    # Use physics features as a proxy for sequence if not available
    # In practice, you'd load embeddings from CADENCE or similar
    print("Note: Using PCA of physics features as sequence proxy")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(50, X_physics.shape[1]))
    X_sequence = pca.fit_transform(X_physics)

    # Run mediation
    analyzer = MediationAnalyzer(config)
    result = analyzer.analyze(X_sequence, X_physics, y, feature_names)

    # Print results
    print(f"\nEffect Decomposition:")
    print(f"  Total Effect: {result.total_effect:.4f}")
    print(f"  Direct Effect: {result.direct_effect:.4f}")
    print(f"  Indirect Effect: {result.indirect_effect:.4f}")
    print(f"  Proportion Mediated: {result.proportion_mediated:.1%}")

    print(f"\nStatistical Significance:")
    print(f"  Indirect Effect 95% CI: [{result.indirect_effect_ci[0]:.4f}, {result.indirect_effect_ci[1]:.4f}]")
    print(f"  P-value: {result.indirect_effect_p:.4f}")

    print(f"\nModel Fit:")
    print(f"  Outcome R²: {result.outcome_r2:.4f}")
    print(f"  Mediator R²: {result.mediator_r2:.4f}")

    # Family-level analysis
    family_results = analyzer.analyze_by_family(X_sequence, X_physics, y, feature_names)

    print(f"\nFamily-Level Mediation:")
    for fr in family_results:
        sig = "***" if fr.significant else ""
        print(f"  {fr.family}: {fr.proportion_mediated:.1%} mediated {sig}")

    # Save results
    analyzer.save(result, output_dir / 'mediation_results.json')

    # Save top mediators
    top_mediators = analyzer.get_top_mediators(result, top_n=30)
    top_mediators.to_csv(output_dir / 'top_mediators.csv', index=False)

    # Save summary
    summary = analyzer.get_mediation_summary(result)
    summary.to_csv(output_dir / 'mediation_summary.csv', index=False)

    print(f"\nResults saved to {output_dir}")
    return result


def run_landscape(args):
    """Run landscape analysis."""
    print(f"\n{'='*60}")
    print(f"Physics-Activity Landscape - {args.cell_type}")
    print(f"{'='*60}\n")

    config = InterpreterConfig(
        cell_type=args.cell_type,
        landscape_shap_samples=args.shap_samples,
        landscape_elastic_net_alpha=args.alpha,
        landscape_elastic_net_l1_ratio=args.l1_ratio
    )

    output_dir = setup_output_dir(config, 'landscape')
    print(f"Output directory: {output_dir}")

    # Load data
    max_samples = getattr(args, 'max_samples', None)
    max_features = getattr(args, 'max_features', None)
    X, y, feature_names = load_physics_data(args.cell_type, max_samples=max_samples, max_features=max_features)

    # Run landscape analysis
    analyzer = LandscapeAnalyzer(config)
    result = analyzer.analyze(X, y, feature_names, compute_shap=args.compute_shap)

    # Print results
    print(f"\nModel Performance:")
    print(f"  Linear R²: {result.linear_r2:.4f}")
    print(f"  Nonlinear R²: {result.nonlinear_r2:.4f}")
    print(f"  SHAP computed: {result.shap_available}")

    print(f"\nFamily Correlations:")
    for family, corr in sorted(result.family_correlations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {family}: {corr:.4f}")

    print(f"\nTop Features by Correlation:")
    for name, corr in result.top_by_correlation[:10]:
        print(f"  {name}: {corr:.4f}")

    if result.shap_available:
        print(f"\nTop Features by SHAP:")
        for name, shap_val in result.top_by_shap[:10]:
            print(f"  {name}: {shap_val:.4f}")

    # Save results
    analyzer.save(result, output_dir / 'landscape_results.json')

    # Save family summary
    family_df = analyzer.get_family_summary(result)
    family_df.to_csv(output_dir / 'family_summary.csv', index=False)

    # Save feature table
    feature_df = analyzer.get_feature_table(result, top_n=100)
    feature_df.to_csv(output_dir / 'feature_table.csv', index=False)

    print(f"\nResults saved to {output_dir}")
    return result


def run_full_analysis(args):
    """Run full analysis pipeline."""
    print(f"\n{'='*60}")
    print(f"Full PhysicsInterpreter Analysis - {args.cell_type}")
    print(f"{'='*60}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = get_fusemap_root() / 'physics' / 'PhysicsInterpreter' / 'results'
    output_dir = output_base / f"full_{args.cell_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    results = {}

    # 1. Attribution Analysis
    print("\n" + "="*40)
    print("Step 1: Physics Feature Attribution")
    print("="*40)
    try:
        args.top_n = 20
        args.probe_type = 'ridge'
        args.alpha = 1.0
        attribution_result = run_attribution(args)
        results['attribution'] = 'success'
    except Exception as e:
        print(f"Attribution failed: {e}")
        results['attribution'] = f'failed: {str(e)}'

    # 2. Mediation Analysis
    print("\n" + "="*40)
    print("Step 2: Mediation Analysis")
    print("="*40)
    try:
        args.n_bootstrap = 100
        args.confidence = 0.95
        mediation_result = run_mediation(args)
        results['mediation'] = 'success'
    except Exception as e:
        print(f"Mediation failed: {e}")
        results['mediation'] = f'failed: {str(e)}'

    # 3. Landscape Analysis
    print("\n" + "="*40)
    print("Step 3: Landscape Analysis")
    print("="*40)
    try:
        args.shap_samples = 100
        args.alpha = 0.01
        args.l1_ratio = 0.5
        args.compute_shap = True
        landscape_result = run_landscape(args)
        results['landscape'] = 'success'
    except Exception as e:
        print(f"Landscape failed: {e}")
        results['landscape'] = f'failed: {str(e)}'

    # Save summary
    with open(output_dir / 'pipeline_summary.json', 'w') as f:
        json.dump({
            'cell_type': args.cell_type,
            'timestamp': timestamp,
            'results': results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("Pipeline Complete")
    print(f"{'='*60}")
    print(f"Results saved to {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='PhysicsInterpreter: Attribution and Mechanistic Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Analysis type')

    # Common arguments for all parsers
    def add_common_args(p):
        p.add_argument('--max-samples', type=int, default=None,
                      help='Max samples to use (for faster testing)')
        p.add_argument('--max-features', type=int, default=None,
                      help='Max features to use (for faster testing)')

    # Attribution subparser
    attr_parser = subparsers.add_parser('attribution', help='Physics feature attribution')
    attr_parser.add_argument('--cell-type', type=str, default='WTC11',
                            help='Cell type (WTC11, HepG2, K562, S2)')
    attr_parser.add_argument('--probe-type', type=str, default='ridge',
                            choices=['ridge', 'elastic_net'])
    attr_parser.add_argument('--alpha', type=float, default=1.0)
    attr_parser.add_argument('--top-n', type=int, default=20)
    add_common_args(attr_parser)

    # IG subparser
    ig_parser = subparsers.add_parser('ig', help='Integrated gradients')
    ig_parser.add_argument('--cell-type', type=str, default='WTC11')
    ig_parser.add_argument('--sequence', type=str, help='DNA sequence')
    ig_parser.add_argument('--sequence-file', type=str, help='File with sequences')
    ig_parser.add_argument('--steps', type=int, default=50)
    ig_parser.add_argument('--baseline', type=str, default='zeros',
                          choices=['zeros', 'shuffle', 'gc_matched'])

    # Mediation subparser
    med_parser = subparsers.add_parser('mediation', help='Mediation analysis')
    med_parser.add_argument('--cell-type', type=str, default='WTC11')
    med_parser.add_argument('--n-bootstrap', type=int, default=100)
    med_parser.add_argument('--confidence', type=float, default=0.95)
    add_common_args(med_parser)

    # Landscape subparser
    land_parser = subparsers.add_parser('landscape', help='Landscape analysis')
    land_parser.add_argument('--cell-type', type=str, default='WTC11')
    land_parser.add_argument('--compute-shap', action='store_true')
    land_parser.add_argument('--shap-samples', type=int, default=100)
    land_parser.add_argument('--alpha', type=float, default=0.01)
    land_parser.add_argument('--l1-ratio', type=float, default=0.5)
    add_common_args(land_parser)

    # Full analysis subparser
    full_parser = subparsers.add_parser('full', help='Full analysis pipeline')
    full_parser.add_argument('--cell-type', type=str, default='WTC11')
    add_common_args(full_parser)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Run appropriate analysis
    if args.command == 'attribution':
        run_attribution(args)
    elif args.command == 'ig':
        run_integrated_gradients(args)
    elif args.command == 'mediation':
        run_mediation(args)
    elif args.command == 'landscape':
        run_landscape(args)
    elif args.command == 'full':
        run_full_analysis(args)


if __name__ == '__main__':
    main()
