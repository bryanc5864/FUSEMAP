#!/usr/bin/env python3
"""
06_cross_celltype_generalization.py - Cross-Cell-Type Physics Generalization

Test whether physics-based models generalize across cell types:
- Train on cell type A, test on cell types B, C
- Compare physics-only vs physics+TF models for transfer
- Identify universally predictive physics features

Optimized for parallel training across cell type pairs.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import argparse
import json
from itertools import permutations
import multiprocessing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

N_CPUS = multiprocessing.cpu_count()


def load_dataset(data_dir: Path, cell_type: str, split: str = 'train', file_pattern: str = None) -> pd.DataFrame:
    """Load a labeled dataset."""
    # Try multiple file patterns
    patterns_to_try = []
    if file_pattern:
        patterns_to_try.append(data_dir / file_pattern.format(cell_type=cell_type, split=split))
    patterns_to_try.extend([
        data_dir / cell_type / f'{cell_type}_{split}_with_features.tsv',
        data_dir / f'{cell_type}_{split}_with_features.tsv',
        data_dir / f'{cell_type}_{split}_descriptors_with_activity.tsv',
        data_dir / f'{cell_type}_{split}_descriptors.tsv',
    ])

    file_path = None
    for pattern in patterns_to_try:
        if pattern.exists():
            file_path = pattern
            break

    if file_path is None:
        raise FileNotFoundError(f"Dataset not found. Tried: {patterns_to_try[0]}")

    df = pd.read_csv(file_path, sep='\t')
    print(f"Loaded {len(df)} sequences from {cell_type} {split} ({file_path.name})")
    return df


def get_feature_groups(columns: list) -> dict:
    """Separate features into groups."""
    physics_prefixes = ['thermo_', 'stiff_', 'bend_', 'entropy_', 'advanced_']
    pure_physics = [c for c in columns if any(c.startswith(p) for p in physics_prefixes)]
    pwm_features = [c for c in columns if c.startswith('pwm_')]

    return {
        'physics': pure_physics,
        'pwm': pwm_features,
        'all': pure_physics + pwm_features
    }


def prepare_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Prepare feature matrix, handling NaN and zero-variance."""
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    variances = np.var(X, axis=0)
    valid_mask = variances > 1e-10
    X = X[:, valid_mask]
    valid_features = [feature_cols[i] for i in range(len(feature_cols)) if valid_mask[i]]

    return X, valid_features


def get_common_features(datasets: dict, feature_group: str) -> list:
    """Get features common across all datasets."""
    common = None
    for df in datasets.values():
        groups = get_feature_groups(df.columns.tolist())
        features = set(groups[feature_group])
        if common is None:
            common = features
        else:
            common = common.intersection(features)
    return list(common)


def fit_and_evaluate(X_train, y_train, X_test, y_test, n_jobs: int = -1) -> dict:
    """Fit model on training data and evaluate on test data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = ElasticNetCV(
        l1_ratio=[0.5, 0.9],
        alphas=np.logspace(-4, 1, 30),
        cv=5,
        max_iter=5000,
        random_state=42,
        n_jobs=n_jobs
    )
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    return {
        'r2_train': float(r2_score(y_train, y_pred_train)),
        'r2_test': float(r2_score(y_test, y_pred_test)),
        'rmse_train': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'rmse_test': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        'n_nonzero': int(np.sum(model.coef_ != 0)),
        'coefficients': model.coef_,
        'model': model,
        'scaler': scaler
    }


def train_test_pair(train_ct, test_ct, prepared, n_jobs):
    """Train on one cell type, test on another."""
    X_train = prepared[train_ct]['X']
    y_train = prepared[train_ct]['y']
    X_test = prepared[test_ct]['X']
    y_test = prepared[test_ct]['y']

    result = fit_and_evaluate(X_train, y_train, X_test, y_test, n_jobs=1)
    return train_ct, test_ct, {
        'r2': result['r2_test'],
        'rmse': result['rmse_test'],
        'is_same_celltype': train_ct == test_ct
    }


def cross_celltype_transfer(datasets: dict, feature_type: str = 'physics', n_jobs: int = -1) -> dict:
    """Evaluate cross-cell-type transfer for a given feature type."""
    cell_types = list(datasets.keys())
    common_features = get_common_features(datasets, feature_type)

    print(f"  Found {len(common_features)} common {feature_type} features")

    # Find features valid across ALL cell types (non-zero variance in all)
    valid_in_all = set(common_features)
    for cell_type, df in datasets.items():
        X = df[common_features].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        variances = np.var(X, axis=0)
        valid_here = {common_features[i] for i in range(len(common_features)) if variances[i] > 1e-10}
        valid_in_all = valid_in_all.intersection(valid_here)

    valid_features = sorted(list(valid_in_all))
    print(f"  Using {len(valid_features)} features valid across all cell types")

    prepared = {}
    for cell_type, df in datasets.items():
        X = df[valid_features].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = df['activity'].values
        prepared[cell_type] = {'X': X, 'y': y, 'features': valid_features}

    # Parallel training across all pairs
    pairs = [(train_ct, test_ct, prepared, n_jobs)
             for train_ct in cell_types for test_ct in cell_types]

    results_list = Parallel(n_jobs=min(n_jobs, len(pairs)))(
        delayed(train_test_pair)(train_ct, test_ct, prepared, n_jobs)
        for train_ct, test_ct, prepared, _ in pairs
    )

    results = {}
    for train_ct, test_ct, res in results_list:
        if train_ct not in results:
            results[train_ct] = {'train_on': train_ct, 'test_results': {}}
        results[train_ct]['test_results'][test_ct] = res

    return results, common_features


def analyze_feature_transfer(datasets: dict, n_top: int = 20, n_jobs: int = -1) -> dict:
    """Identify features that transfer best across cell types."""
    cell_types = list(datasets.keys())
    common_physics = get_common_features(datasets, 'physics')

    all_coefs = {}
    for cell_type, df in datasets.items():
        X, valid_features = prepare_features(df, common_physics)
        y = df['activity'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = ElasticNetCV(l1_ratio=0.5, cv=5, max_iter=5000, random_state=42, n_jobs=n_jobs)
        model.fit(X_scaled, y)

        all_coefs[cell_type] = {valid_features[i]: model.coef_[i] for i in range(len(valid_features))}

    feature_consistency = []
    for feature in common_physics:
        coefs = []
        for cell_type in cell_types:
            if feature in all_coefs[cell_type]:
                coefs.append(all_coefs[cell_type][feature])

        if len(coefs) == len(cell_types):
            signs = [1 if c > 0 else -1 if c < 0 else 0 for c in coefs]
            sign_consistent = len(set([s for s in signs if s != 0])) <= 1
            mean_coef = np.mean(coefs)
            std_coef = np.std(coefs)

            feature_consistency.append({
                'feature': feature,
                'mean_coefficient': float(mean_coef),
                'std_coefficient': float(std_coef),
                'sign_consistent': sign_consistent,
                'cv': float(std_coef / abs(mean_coef)) if abs(mean_coef) > 1e-10 else float('inf'),
                'coefficients': {ct: float(all_coefs[ct].get(feature, 0)) for ct in cell_types}
            })

    feature_consistency = sorted(feature_consistency, key=lambda x: x['cv'] if x['sign_consistent'] else float('inf'))

    return {
        'most_consistent': feature_consistency[:n_top],
        'least_consistent': feature_consistency[-n_top:] if len(feature_consistency) > n_top else [],
        'n_sign_consistent': sum(1 for f in feature_consistency if f['sign_consistent'])
    }


def create_visualizations(all_results: dict, output_dir: Path, cell_types: list):
    """Create visualization plots."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Transfer matrices heatmaps
    for feat_type in ['physics', 'pwm', 'all']:
        if f'{feat_type}_transfer' not in all_results:
            continue

        transfer_data = all_results[f'{feat_type}_transfer']
        matrix = np.zeros((len(cell_types), len(cell_types)))

        for i, train_ct in enumerate(cell_types):
            for j, test_ct in enumerate(cell_types):
                if train_ct in transfer_data and test_ct in transfer_data[train_ct]['test_results']:
                    matrix[i, j] = transfer_data[train_ct]['test_results'][test_ct]['r2']

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=max(0.5, matrix.max()))

        ax.set_xticks(range(len(cell_types)))
        ax.set_xticklabels(cell_types)
        ax.set_yticks(range(len(cell_types)))
        ax.set_yticklabels(cell_types)
        ax.set_xlabel('Test Cell Type')
        ax.set_ylabel('Train Cell Type')

        for i in range(len(cell_types)):
            for j in range(len(cell_types)):
                ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', fontsize=10)

        plt.colorbar(im, label='R²')
        ax.set_title(f'Cross-Cell-Type Transfer ({feat_type.upper()} features)')
        plt.tight_layout()
        plt.savefig(fig_dir / f'transfer_matrix_{feat_type}.png', dpi=150)
        plt.close()

    # 2. Physics vs PWM transfer comparison
    if 'summary' in all_results:
        summary = all_results['summary']

        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Same Cell Type', 'Cross Cell Type', 'Transfer Gap']
        physics_vals = [summary['physics_same_celltype_r2'], summary['physics_cross_celltype_r2'], summary['physics_transfer_gap']]
        pwm_vals = [summary['pwm_same_celltype_r2'], summary['pwm_cross_celltype_r2'], summary['pwm_transfer_gap']]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax.bar(x - width/2, physics_vals, width, label='Physics', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, pwm_vals, width, label='PWM', color='forestgreen', alpha=0.8)

        ax.set_ylabel('R²')
        ax.set_title('Physics vs PWM Transfer Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(fig_dir / 'physics_vs_pwm_transfer.png', dpi=150)
        plt.close()

    # 3. Consistent features across cell types
    if 'feature_transfer' in all_results:
        feat_transfer = all_results['feature_transfer']
        consistent = feat_transfer['most_consistent'][:20]

        if consistent:
            fig, ax = plt.subplots(figsize=(12, 8))
            features = [f['feature'] for f in consistent]
            cvs = [f['cv'] for f in consistent]

            colors = ['green' if f['sign_consistent'] else 'red' for f in consistent]
            ax.barh(range(len(features)), cvs, color=colors, alpha=0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=8)
            ax.set_xlabel('Coefficient of Variation (lower = more consistent)')
            ax.set_title('Most Consistent Features Across Cell Types')
            ax.axvline(x=1, color='red', linestyle='--', linewidth=1, label='CV=1')
            ax.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / 'consistent_features.png', dpi=150)
            plt.close()

    print(f"  Saved visualizations to {fig_dir}")


def generate_text_report(all_results: dict, output_dir: Path, cell_types: list) -> str:
    """Generate text summary report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CROSS-CELL-TYPE PHYSICS GENERALIZATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("\nThis analysis tests how well physics features transfer across cell types.")

    if 'summary' in all_results:
        summary = all_results['summary']
        report_lines.append(f"\n--- Summary ---")
        report_lines.append(f"Physics features: same-cell R² = {summary['physics_same_celltype_r2']:.4f}")
        report_lines.append(f"Physics features: cross-cell R² = {summary['physics_cross_celltype_r2']:.4f}")
        report_lines.append(f"Physics transfer gap: {summary['physics_transfer_gap']:.4f}")
        report_lines.append(f"\nPWM features: same-cell R² = {summary['pwm_same_celltype_r2']:.4f}")
        report_lines.append(f"PWM features: cross-cell R² = {summary['pwm_cross_celltype_r2']:.4f}")
        report_lines.append(f"PWM transfer gap: {summary['pwm_transfer_gap']:.4f}")

        if summary['physics_transfer_gap'] < summary['pwm_transfer_gap']:
            report_lines.append("\n>>> Physics features TRANSFER BETTER than PWM features")
        else:
            report_lines.append("\n>>> PWM features transfer better than physics features")

    # Transfer matrices
    for feat_type in ['physics', 'pwm', 'all']:
        if f'{feat_type}_transfer' not in all_results:
            continue

        report_lines.append(f"\n--- {feat_type.upper()} Transfer Matrix ---")
        transfer_data = all_results[f'{feat_type}_transfer']

        header = "         " + "  ".join(f"{ct:>8}" for ct in cell_types)
        report_lines.append(header)

        for train_ct in cell_types:
            if train_ct in transfer_data:
                row_values = []
                for test_ct in cell_types:
                    if test_ct in transfer_data[train_ct]['test_results']:
                        row_values.append(f"{transfer_data[train_ct]['test_results'][test_ct]['r2']:8.4f}")
                    else:
                        row_values.append("     N/A")
                report_lines.append(f"  {train_ct:>6} " + "  ".join(row_values))

    # Feature transfer analysis
    if 'feature_transfer' in all_results:
        feat_transfer = all_results['feature_transfer']
        report_lines.append(f"\n--- Feature Transfer Analysis ---")
        report_lines.append(f"Sign-consistent features: {feat_transfer['n_sign_consistent']}")

        report_lines.append("\nMost consistent features:")
        for feat in feat_transfer['most_consistent'][:15]:
            sign = 'Y' if feat['sign_consistent'] else 'N'
            report_lines.append(f"  {feat['feature']:40s}: CV={feat['cv']:.3f}, sign={sign}")

    report_text = '\n'.join(report_lines)
    report_path = output_dir / 'report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"  Saved text report to {report_path}")
    return report_text


def run_analysis(
    data_dir: Path,
    output_dir: Path,
    cell_types: list,
    split: str = 'train',
    n_jobs: int = -1,
    activity_col: str = 'activity'
) -> dict:
    """Run full cross-cell-type generalization analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if n_jobs == -1:
        n_jobs = N_CPUS

    print(f"Using {n_jobs} CPU cores")

    datasets = {}
    for cell_type in cell_types:
        try:
            datasets[cell_type] = load_dataset(data_dir, cell_type, split)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if len(datasets) < 2:
        print("Need at least 2 cell types for cross-cell-type analysis")
        return {}

    print(f"\n{'='*50}")
    print("Cross-Cell-Type Transfer Analysis")
    print('='*50)

    all_results = {}
    cts = list(datasets.keys())

    # Physics-only transfer
    print("\n1. Physics-only features:")
    physics_results, physics_features = cross_celltype_transfer(datasets, 'physics', n_jobs=n_jobs)
    all_results['physics_transfer'] = physics_results

    transfer_matrix_physics = np.zeros((len(cts), len(cts)))
    for i, train_ct in enumerate(cts):
        for j, test_ct in enumerate(cts):
            if train_ct in physics_results and test_ct in physics_results[train_ct]['test_results']:
                transfer_matrix_physics[i, j] = physics_results[train_ct]['test_results'][test_ct]['r2']

    print("  Transfer R² matrix (physics):")
    print("        " + "  ".join(f"{ct:>8}" for ct in cts))
    for i, train_ct in enumerate(cts):
        row = "  ".join(f"{transfer_matrix_physics[i,j]:8.3f}" for j in range(len(cts)))
        print(f"  {train_ct:>6} {row}")

    # PWM transfer
    print("\n2. PWM features:")
    pwm_results, pwm_features = cross_celltype_transfer(datasets, 'pwm', n_jobs=n_jobs)
    all_results['pwm_transfer'] = pwm_results

    transfer_matrix_pwm = np.zeros((len(cts), len(cts)))
    for i, train_ct in enumerate(cts):
        for j, test_ct in enumerate(cts):
            if train_ct in pwm_results and test_ct in pwm_results[train_ct]['test_results']:
                transfer_matrix_pwm[i, j] = pwm_results[train_ct]['test_results'][test_ct]['r2']

    print("  Transfer R² matrix (PWM):")
    print("        " + "  ".join(f"{ct:>8}" for ct in cts))
    for i, train_ct in enumerate(cts):
        row = "  ".join(f"{transfer_matrix_pwm[i,j]:8.3f}" for j in range(len(cts)))
        print(f"  {train_ct:>6} {row}")

    # Combined transfer
    print("\n3. All features (physics + PWM):")
    all_feat_results, all_features = cross_celltype_transfer(datasets, 'all', n_jobs=n_jobs)
    all_results['all_transfer'] = all_feat_results

    transfer_matrix_all = np.zeros((len(cts), len(cts)))
    for i, train_ct in enumerate(cts):
        for j, test_ct in enumerate(cts):
            if train_ct in all_feat_results and test_ct in all_feat_results[train_ct]['test_results']:
                transfer_matrix_all[i, j] = all_feat_results[train_ct]['test_results'][test_ct]['r2']

    # Feature transfer analysis
    print("\n4. Feature Transfer Analysis:")
    feature_transfer = analyze_feature_transfer(datasets, n_jobs=n_jobs)
    all_results['feature_transfer'] = feature_transfer

    print(f"  {feature_transfer['n_sign_consistent']} features have consistent signs")

    # Summary statistics
    off_diag_mask = ~np.eye(len(cts), dtype=bool)

    summary = {
        'cell_types': cts,
        'n_physics_features': len(physics_features),
        'n_pwm_features': len(pwm_features),
        'physics_same_celltype_r2': float(np.mean(np.diag(transfer_matrix_physics))),
        'physics_cross_celltype_r2': float(np.mean(transfer_matrix_physics[off_diag_mask])),
        'physics_transfer_gap': float(np.mean(np.diag(transfer_matrix_physics)) - np.mean(transfer_matrix_physics[off_diag_mask])),
        'pwm_same_celltype_r2': float(np.mean(np.diag(transfer_matrix_pwm))),
        'pwm_cross_celltype_r2': float(np.mean(transfer_matrix_pwm[off_diag_mask])),
        'pwm_transfer_gap': float(np.mean(np.diag(transfer_matrix_pwm)) - np.mean(transfer_matrix_pwm[off_diag_mask])),
        'all_same_celltype_r2': float(np.mean(np.diag(transfer_matrix_all))),
        'all_cross_celltype_r2': float(np.mean(transfer_matrix_all[off_diag_mask])),
        'all_transfer_gap': float(np.mean(np.diag(transfer_matrix_all)) - np.mean(transfer_matrix_all[off_diag_mask]))
    }

    all_results['summary'] = summary
    all_results['transfer_matrices'] = {
        'physics': transfer_matrix_physics.tolist(),
        'pwm': transfer_matrix_pwm.tolist(),
        'all': transfer_matrix_all.tolist()
    }

    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    print(f"Physics: same-cell R²={summary['physics_same_celltype_r2']:.3f}, cross-cell R²={summary['physics_cross_celltype_r2']:.3f}, gap={summary['physics_transfer_gap']:.3f}")
    print(f"PWM:     same-cell R²={summary['pwm_same_celltype_r2']:.3f}, cross-cell R²={summary['pwm_cross_celltype_r2']:.3f}, gap={summary['pwm_transfer_gap']:.3f}")

    if summary['physics_transfer_gap'] < summary['pwm_transfer_gap']:
        print("\n>>> Physics features transfer BETTER than PWM features <<<")
    else:
        print("\n>>> PWM features transfer better than physics features <<<")

    # Save results
    with open(output_dir / 'cross_celltype_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    # Create visualizations
    create_visualizations(all_results, output_dir, cts)

    # Generate text report
    generate_text_report(all_results, output_dir, cts)

    print(f"\nResults saved to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Cross-Cell-Type Physics Generalization')
    parser.add_argument('--data_dir', type=str, default='data/lentiMPRA_data')
    parser.add_argument('--output_dir', type=str, default='analyses/results/06_cross_celltype')
    parser.add_argument('--cell_types', type=str, nargs='+', default=['K562', 'HepG2', 'WTC11'])
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--activity_col', type=str, default='activity', help='Column name for activity values')

    args = parser.parse_args()

    run_analysis(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        cell_types=args.cell_types,
        split=args.split,
        n_jobs=args.n_jobs,
        activity_col=args.activity_col
    )


if __name__ == '__main__':
    main()
