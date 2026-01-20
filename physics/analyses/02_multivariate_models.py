#!/usr/bin/env python3
"""
02_multivariate_models.py - Physics-Only Multivariate Models

Fit sparse models using physics features to predict activity:
- Elastic Net regression with cross-validation
- Generalized Additive Models (GAMs) with spline terms
- Feature importance rankings
- Partial dependence plots for top features

Optimized for GPU (cuML) with CPU fallback, parallel processing.
"""

# Patch numpy for TensorFlow compatibility with NumPy 2.0
import numpy as np
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64

import pandas as pd
from pathlib import Path
import argparse
import json
import multiprocessing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Try GPU-accelerated libraries
try:
    import cuml
    from cuml.linear_model import ElasticNet as cuElasticNet
    from cuml.linear_model import Ridge as cuRidge
    GPU_AVAILABLE = True
    print("GPU acceleration available (cuML)")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU not available, using CPU (sklearn)")

from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import partial_dependence

try:
    from pygam import LinearGAM, s
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False

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


def get_physics_features(df: pd.DataFrame, exclude_pwm: bool = False) -> tuple:
    """Extract physics feature matrix and feature names."""
    prefixes = ['thermo_', 'stiff_', 'bend_', 'entropy_', 'advanced_']
    if not exclude_pwm:
        prefixes.append('pwm_')

    feature_cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove zero-variance features
    variances = np.var(X, axis=0)
    valid_mask = variances > 1e-10
    X = X[:, valid_mask]
    feature_names = [feature_cols[i] for i in range(len(feature_cols)) if valid_mask[i]]

    return X, feature_names


def fit_elastic_net_gpu(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, l1_ratio: float = 0.5):
    """Fit Elastic Net using GPU if available."""
    if GPU_AVAILABLE:
        try:
            import cupy as cp
            X_gpu = cp.asarray(X)
            y_gpu = cp.asarray(y)
            model = cuElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            model.fit(X_gpu, y_gpu)
            return model, True
        except Exception as e:
            print(f"  GPU failed, falling back to CPU: {e}")

    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X, y)
    return model, False


def fit_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_folds: int = 5,
    n_jobs: int = -1
) -> dict:
    """Fit Elastic Net with cross-validation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
        alphas=np.logspace(-4, 1, 50),
        cv=n_folds,
        max_iter=10000,
        random_state=42,
        n_jobs=n_jobs
    )
    model.fit(X_scaled, y)

    cv_scores = cross_val_score(model, X_scaled, y, cv=n_folds, scoring='r2', n_jobs=n_jobs)

    coefs = model.coef_
    importance_order = np.argsort(np.abs(coefs))[::-1]
    nonzero_mask = coefs != 0
    n_nonzero = np.sum(nonzero_mask)

    results = {
        'model_type': 'ElasticNet',
        'best_alpha': float(model.alpha_),
        'best_l1_ratio': float(model.l1_ratio_),
        'r2_train': float(r2_score(y, model.predict(X_scaled))),
        'r2_cv_mean': float(np.mean(cv_scores)),
        'r2_cv_std': float(np.std(cv_scores)),
        'rmse_train': float(np.sqrt(mean_squared_error(y, model.predict(X_scaled)))),
        'n_features_total': len(feature_names),
        'n_features_nonzero': int(n_nonzero),
        'sparsity': float(1 - n_nonzero / len(feature_names)),
        'top_features': [
            {'feature': feature_names[i], 'coefficient': float(coefs[i]), 'rank': int(rank + 1)}
            for rank, i in enumerate(importance_order[:50]) if coefs[i] != 0
        ],
        'feature_importances': {
            feature_names[i]: float(coefs[i]) for i in range(len(feature_names)) if coefs[i] != 0
        }
    }

    return results, model, scaler


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_folds: int = 5,
    n_jobs: int = -1
) -> dict:
    """Fit Ridge regression with cross-validation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=n_folds)
    model.fit(X_scaled, y)

    cv_scores = cross_val_score(model, X_scaled, y, cv=n_folds, scoring='r2', n_jobs=n_jobs)

    coefs = model.coef_
    importance_order = np.argsort(np.abs(coefs))[::-1]

    results = {
        'model_type': 'Ridge',
        'best_alpha': float(model.alpha_),
        'r2_train': float(r2_score(y, model.predict(X_scaled))),
        'r2_cv_mean': float(np.mean(cv_scores)),
        'r2_cv_std': float(np.std(cv_scores)),
        'rmse_train': float(np.sqrt(mean_squared_error(y, model.predict(X_scaled)))),
        'n_features_total': len(feature_names),
        'top_features': [
            {'feature': feature_names[i], 'coefficient': float(coefs[i]), 'rank': int(rank + 1)}
            for rank, i in enumerate(importance_order[:50])
        ]
    }

    return results, model, scaler


def fit_gam(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_top_features: int = 20
) -> dict:
    """Fit Generalized Additive Model with spline terms."""
    if not PYGAM_AVAILABLE:
        return {'error': 'pygam not available'}, None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # First fit elastic net to select top features
    enet = ElasticNetCV(l1_ratio=0.5, cv=5, max_iter=5000, random_state=42)
    enet.fit(X_scaled, y)

    importance_order = np.argsort(np.abs(enet.coef_))[::-1]
    top_indices = importance_order[:n_top_features]

    X_top = X_scaled[:, top_indices]
    top_feature_names = [feature_names[i] for i in top_indices]

    try:
        gam = LinearGAM(sum([s(i, n_splines=10) for i in range(n_top_features)]))
        gam.gridsearch(X_top, y, progress=False)
        y_pred = gam.predict(X_top)

        results = {
            'model_type': 'GAM',
            'n_features': n_top_features,
            'r2_train': float(r2_score(y, y_pred)),
            'rmse_train': float(np.sqrt(mean_squared_error(y, y_pred))),
            'feature_names': top_feature_names,
            'aic': float(gam.statistics_['AIC']),
            'edof': float(gam.statistics_['edof'])
        }

        return results, gam, scaler, top_indices
    except Exception as e:
        return {'error': str(e)}, None, None, None


def compare_physics_vs_full(df: pd.DataFrame, activity_col: str = 'activity', n_jobs: int = -1) -> dict:
    """Compare physics-only model vs physics+PWM model."""
    y = df[activity_col].values

    X_physics, names_physics = get_physics_features(df, exclude_pwm=True)
    X_full, names_full = get_physics_features(df, exclude_pwm=False)

    print(f"  Physics-only features: {len(names_physics)}")
    print(f"  Full features (with PWM): {len(names_full)}")

    results_physics, _, _ = fit_elastic_net(X_physics, y, names_physics, n_jobs=n_jobs)
    results_full, _, _ = fit_elastic_net(X_full, y, names_full, n_jobs=n_jobs)

    comparison = {
        'physics_only': {
            'n_features': len(names_physics),
            'r2_cv': results_physics['r2_cv_mean'],
            'n_nonzero': results_physics['n_features_nonzero']
        },
        'with_pwm': {
            'n_features': len(names_full),
            'r2_cv': results_full['r2_cv_mean'],
            'n_nonzero': results_full['n_features_nonzero']
        },
        'r2_improvement_from_pwm': results_full['r2_cv_mean'] - results_physics['r2_cv_mean']
    }

    return comparison


def create_visualizations(
    enet_results: dict,
    ridge_results: dict,
    comparison: dict,
    output_dir: Path,
    cell_type: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    model,
    scaler
):
    """Create visualization plots."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Top features coefficient plot
    if enet_results['top_features']:
        top_feats = enet_results['top_features'][:30]
        fig, ax = plt.subplots(figsize=(12, 10))
        colors = ['green' if f['coefficient'] > 0 else 'red' for f in top_feats]
        ax.barh(range(len(top_feats)), [f['coefficient'] for f in top_feats], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels([f['feature'] for f in top_feats], fontsize=8)
        ax.set_xlabel('Elastic Net Coefficient')
        ax.set_title(f'Top Features (Elastic Net) - {cell_type}')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(fig_dir / f'enet_coefficients_{cell_type}.png', dpi=150)
        plt.close()

    # 2. Physics vs PWM comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ['Physics Only', 'Physics + PWM']
    r2_values = [comparison['physics_only']['r2_cv'], comparison['with_pwm']['r2_cv']]
    colors = ['steelblue', 'forestgreen']
    bars = ax.bar(categories, r2_values, color=colors, alpha=0.8)
    ax.set_ylabel('Cross-Validation R²')
    ax.set_title(f'Model Performance Comparison - {cell_type}')
    ax.set_ylim(0, max(r2_values) * 1.2)

    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12)

    improvement = comparison['r2_improvement_from_pwm']
    ax.text(0.5, 0.95, f'PWM improvement: {improvement:+.3f}',
            transform=ax.transAxes, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(fig_dir / f'physics_vs_pwm_{cell_type}.png', dpi=150)
    plt.close()

    # 3. Predicted vs Actual scatter
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y, y_pred, alpha=0.3, s=10)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    ax.set_xlabel('Actual Activity')
    ax.set_ylabel('Predicted Activity')
    ax.set_title(f'Elastic Net Predictions - {cell_type}\nR² = {enet_results["r2_train"]:.3f}')
    plt.tight_layout()
    plt.savefig(fig_dir / f'pred_vs_actual_{cell_type}.png', dpi=150)
    plt.close()

    # 4. Feature importance by category
    if enet_results['feature_importances']:
        cat_importance = {'thermo': 0, 'stiff': 0, 'bend': 0, 'entropy': 0, 'advanced': 0, 'pwm': 0}
        cat_counts = {'thermo': 0, 'stiff': 0, 'bend': 0, 'entropy': 0, 'advanced': 0, 'pwm': 0}

        for feat, coef in enet_results['feature_importances'].items():
            for cat in cat_importance.keys():
                if feat.startswith(f'{cat}_'):
                    cat_importance[cat] += abs(coef)
                    cat_counts[cat] += 1
                    break

        fig, ax = plt.subplots(figsize=(10, 6))
        cats = list(cat_importance.keys())
        values = [cat_importance[c] for c in cats]
        counts = [cat_counts[c] for c in cats]

        bars = ax.bar(cats, values, color='steelblue', alpha=0.8)
        ax.set_ylabel('Sum of |Coefficients|')
        ax.set_title(f'Feature Importance by Category - {cell_type}')

        for bar, val, cnt in zip(bars, values, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'n={cnt}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(fig_dir / f'category_importance_{cell_type}.png', dpi=150)
        plt.close()

    print(f"  Saved visualizations to {fig_dir}")


def generate_text_report(
    all_results: dict,
    output_dir: Path
) -> str:
    """Generate text summary report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MULTIVARIATE MODEL ANALYSIS REPORT")
    report_lines.append("=" * 80)

    for cell_type, results in all_results.items():
        report_lines.append(f"\n{'='*40}")
        report_lines.append(f"Cell Type: {cell_type}")
        report_lines.append('='*40)

        enet = results['elastic_net']
        ridge = results['ridge']
        comp = results['physics_vs_full']

        report_lines.append("\n--- Elastic Net ---")
        report_lines.append(f"R² (CV): {enet['r2_cv_mean']:.4f} +/- {enet['r2_cv_std']:.4f}")
        report_lines.append(f"R² (Train): {enet['r2_train']:.4f}")
        report_lines.append(f"RMSE (Train): {enet['rmse_train']:.4f}")
        report_lines.append(f"Features: {enet['n_features_nonzero']}/{enet['n_features_total']} non-zero")
        report_lines.append(f"Best alpha: {enet['best_alpha']:.4e}, L1 ratio: {enet['best_l1_ratio']:.2f}")

        report_lines.append("\n--- Ridge ---")
        report_lines.append(f"R² (CV): {ridge['r2_cv_mean']:.4f} +/- {ridge['r2_cv_std']:.4f}")
        report_lines.append(f"R² (Train): {ridge['r2_train']:.4f}")

        report_lines.append("\n--- Physics vs Physics+PWM ---")
        report_lines.append(f"Physics-only R²: {comp['physics_only']['r2_cv']:.4f}")
        report_lines.append(f"With PWM R²: {comp['with_pwm']['r2_cv']:.4f}")
        report_lines.append(f"Improvement from PWM: {comp['r2_improvement_from_pwm']:+.4f}")

        report_lines.append("\n--- Top 20 Features (Elastic Net) ---")
        for feat in enet['top_features'][:20]:
            report_lines.append(f"  {feat['rank']:2d}. {feat['feature']:40s}: {feat['coefficient']:+.4f}")

        if results.get('gam') and 'r2_train' in results['gam']:
            gam = results['gam']
            report_lines.append(f"\n--- GAM ---")
            report_lines.append(f"R² (Train): {gam['r2_train']:.4f}")
            report_lines.append(f"AIC: {gam['aic']:.2f}")

    # Summary across cell types
    report_lines.append(f"\n{'='*40}")
    report_lines.append("SUMMARY ACROSS CELL TYPES")
    report_lines.append('='*40)

    mean_r2 = np.mean([r['elastic_net']['r2_cv_mean'] for r in all_results.values()])
    mean_physics = np.mean([r['physics_vs_full']['physics_only']['r2_cv'] for r in all_results.values()])
    mean_improvement = np.mean([r['physics_vs_full']['r2_improvement_from_pwm'] for r in all_results.values()])

    report_lines.append(f"\nMean R² (CV): {mean_r2:.4f}")
    report_lines.append(f"Mean physics-only R²: {mean_physics:.4f}")
    report_lines.append(f"Mean PWM improvement: {mean_improvement:+.4f}")

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
    """Run full multivariate model analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if n_jobs == -1:
        n_jobs = N_CPUS

    print(f"Using {n_jobs} CPU cores, GPU: {'Yes' if GPU_AVAILABLE else 'No'}")

    all_results = {}

    for cell_type in cell_types:
        print(f"\n{'='*50}")
        print(f"Processing {cell_type}")
        print('='*50)

        try:
            df = load_dataset(data_dir, cell_type, split)
            y = df['activity'].values

            X, feature_names = get_physics_features(df, exclude_pwm=False)
            print(f"Using {len(feature_names)} features, {len(y)} samples")

            # Fit Elastic Net
            print("\nFitting Elastic Net...")
            enet_results, enet_model, enet_scaler = fit_elastic_net(X, y, feature_names, n_jobs=n_jobs)
            print(f"  R² (CV): {enet_results['r2_cv_mean']:.4f} +/- {enet_results['r2_cv_std']:.4f}")
            print(f"  Non-zero features: {enet_results['n_features_nonzero']}")

            # Fit Ridge
            print("\nFitting Ridge...")
            ridge_results, ridge_model, ridge_scaler = fit_ridge(X, y, feature_names, n_jobs=n_jobs)
            print(f"  R² (CV): {ridge_results['r2_cv_mean']:.4f} +/- {ridge_results['r2_cv_std']:.4f}")

            # Compare physics-only vs full
            print("\nComparing physics-only vs with PWM...")
            comparison = compare_physics_vs_full(df, activity_col=activity_col, n_jobs=n_jobs)
            print(f"  Physics-only R²: {comparison['physics_only']['r2_cv']:.4f}")
            print(f"  With PWM R²: {comparison['with_pwm']['r2_cv']:.4f}")
            print(f"  Improvement from PWM: {comparison['r2_improvement_from_pwm']:.4f}")

            # Fit GAM if available
            gam_results = None
            if PYGAM_AVAILABLE:
                print("\nFitting GAM...")
                gam_output = fit_gam(X, y, feature_names)
                if isinstance(gam_output, tuple):
                    gam_results, gam_model, gam_scaler, gam_indices = gam_output
                    if gam_results and 'r2_train' in gam_results:
                        print(f"  R² (train): {gam_results['r2_train']:.4f}")

            cell_results = {
                'elastic_net': enet_results,
                'ridge': ridge_results,
                'physics_vs_full': comparison,
                'gam': gam_results
            }

            all_results[cell_type] = cell_results

            # Save per-cell-type results
            with open(output_dir / f'multivariate_{cell_type}.json', 'w') as f:
                json.dump(cell_results, f, indent=2)

            # Create visualizations
            create_visualizations(
                enet_results, ridge_results, comparison,
                output_dir, cell_type, X, y, feature_names,
                enet_model, enet_scaler
            )

            # Print top features
            print(f"\nTop 10 features (Elastic Net) for {cell_type}:")
            for feat in enet_results['top_features'][:10]:
                print(f"  {feat['rank']:2d}. {feat['feature']}: {feat['coefficient']:.4f}")

        except Exception as e:
            print(f"Error processing {cell_type}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined results
    with open(output_dir / 'multivariate_all.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate text report
    generate_text_report(all_results, output_dir)

    # Summary
    summary = {
        'cell_types': cell_types,
        'mean_r2_cv': float(np.mean([r['elastic_net']['r2_cv_mean'] for r in all_results.values()])),
        'physics_only_mean_r2': float(np.mean([r['physics_vs_full']['physics_only']['r2_cv'] for r in all_results.values()])),
        'pwm_improvement_mean': float(np.mean([r['physics_vs_full']['r2_improvement_from_pwm'] for r in all_results.values()]))
    }

    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    print(f"Mean R² (CV): {summary['mean_r2_cv']:.4f}")
    print(f"Mean physics-only R²: {summary['physics_only_mean_r2']:.4f}")
    print(f"Mean improvement from PWM: {summary['pwm_improvement_mean']:.4f}")

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Physics-Only Multivariate Models')
    parser.add_argument('--data_dir', type=str, default='data/lentiMPRA_data')
    parser.add_argument('--output_dir', type=str, default='analyses/results/02_multivariate')
    parser.add_argument('--cell_types', type=str, nargs='+', default=['K562', 'HepG2', 'WTC11'])
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs (-1 for all CPUs)')
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
