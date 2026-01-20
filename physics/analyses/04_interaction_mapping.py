#!/usr/bin/env python3
"""
04_interaction_mapping.py - Physics×TF Interaction Mapping

Identify non-additive interactions between physics features and TF binding:
- H-statistic for feature interactions
- Interaction strength quantification
- Visualization of top interactions

Optimized for GPU (cuML) with parallel H-statistic computation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse
import json
from itertools import product
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
    from cuml.ensemble import RandomForestRegressor as cuRF
    GPU_AVAILABLE = True
    print("GPU acceleration available (cuML)")
except ImportError:
    GPU_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

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


def get_feature_groups(df: pd.DataFrame) -> dict:
    """Separate features into physics and PWM groups."""
    columns = df.columns.tolist()

    physics_prefixes = ['thermo_', 'stiff_', 'bend_', 'entropy_', 'advanced_']
    physics_features = [c for c in columns if any(c.startswith(p) for p in physics_prefixes)]

    pwm_agg_patterns = ['pwm_max_of_max', 'pwm_min_delta_g', 'pwm_tf_binding_diversity',
                        'pwm_sum_top5', 'pwm_best_tf']
    pwm_per_tf = [c for c in columns if c.startswith('pwm_') and
                  not any(p in c for p in pwm_agg_patterns)]

    return {
        'physics': physics_features,
        'pwm_per_tf': pwm_per_tf
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


def compute_h_statistic_single(
    model,
    X: np.ndarray,
    feature_i: int,
    feature_j: int,
    n_samples: int = 300,
    n_grid: int = 15
) -> float:
    """Compute Friedman's H-statistic for interaction between two features."""
    if X.shape[0] > n_samples:
        idx = np.random.choice(X.shape[0], n_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    vals_i = np.percentile(X_sample[:, feature_i], np.linspace(0, 100, n_grid))
    vals_j = np.percentile(X_sample[:, feature_j], np.linspace(0, 100, n_grid))

    # Partial dependence for feature i
    pd_i = np.zeros(len(vals_i))
    for k, val in enumerate(vals_i):
        X_temp = X_sample.copy()
        X_temp[:, feature_i] = val
        pd_i[k] = np.mean(model.predict(X_temp))

    # Partial dependence for feature j
    pd_j = np.zeros(len(vals_j))
    for k, val in enumerate(vals_j):
        X_temp = X_sample.copy()
        X_temp[:, feature_j] = val
        pd_j[k] = np.mean(model.predict(X_temp))

    # 2D partial dependence
    pd_ij = np.zeros((len(vals_i), len(vals_j)))
    for ki, val_i in enumerate(vals_i):
        for kj, val_j in enumerate(vals_j):
            X_temp = X_sample.copy()
            X_temp[:, feature_i] = val_i
            X_temp[:, feature_j] = val_j
            pd_ij[ki, kj] = np.mean(model.predict(X_temp))

    mean_pred = np.mean(pd_ij)
    interaction_effect = np.zeros((len(vals_i), len(vals_j)))

    for ki in range(len(vals_i)):
        for kj in range(len(vals_j)):
            interaction_effect[ki, kj] = (pd_ij[ki, kj] - pd_i[ki] - pd_j[kj] + mean_pred)

    var_interaction = np.var(interaction_effect)
    var_joint = np.var(pd_ij)

    if var_joint < 1e-10:
        return 0.0

    h_squared = var_interaction / var_joint
    return np.sqrt(max(0, h_squared))


def compute_h_pair(args):
    """Wrapper for parallel H-statistic computation."""
    model, X_scaled, phys_feat, phys_idx, pwm_feat, pwm_idx = args
    h_stat = compute_h_statistic_single(model, X_scaled, phys_idx, pwm_idx)
    return {'physics_feature': phys_feat, 'pwm_feature': pwm_feat, 'h_statistic': float(h_stat)}


def select_top_features(X: np.ndarray, y: np.ndarray, feature_names: list, n_top: int = 20, n_jobs: int = -1) -> tuple:
    """Select top features using Random Forest importance."""
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=n_jobs)
    rf.fit(X, y)

    importance_order = np.argsort(rf.feature_importances_)[::-1]
    top_indices = importance_order[:n_top]

    return top_indices, rf.feature_importances_


def compute_interaction_matrix(
    df: pd.DataFrame,
    n_top_physics: int = 15,
    n_top_pwm: int = 15,
    activity_col: str = 'activity',
    n_jobs: int = -1
) -> dict:
    """Compute H-statistics for physics×PWM interactions."""
    y = df[activity_col].values
    groups = get_feature_groups(df)

    all_features = groups['physics'] + groups['pwm_per_tf']
    X_all, valid_features = prepare_features(df, all_features)

    physics_in_valid = [f for f in valid_features if any(f.startswith(p) for p in
                        ['thermo_', 'stiff_', 'bend_', 'entropy_', 'advanced_'])]
    pwm_in_valid = [f for f in valid_features if f.startswith('pwm_')]

    print(f"  Valid physics features: {len(physics_in_valid)}")
    print(f"  Valid PWM features: {len(pwm_in_valid)}")

    physics_indices = [valid_features.index(f) for f in physics_in_valid]
    pwm_indices = [valid_features.index(f) for f in pwm_in_valid]

    top_physics_idx, physics_importance = select_top_features(
        X_all[:, physics_indices], y, physics_in_valid, n_top_physics, n_jobs=n_jobs
    )
    top_pwm_idx, pwm_importance = select_top_features(
        X_all[:, pwm_indices], y, pwm_in_valid, n_top_pwm, n_jobs=n_jobs
    )

    top_physics = [physics_in_valid[i] for i in top_physics_idx]
    top_pwm = [pwm_in_valid[i] for i in top_pwm_idx]

    print(f"  Top {n_top_physics} physics features selected")
    print(f"  Top {n_top_pwm} PWM features selected")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    print("  Training Gradient Boosting model...")
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
    model.fit(X_scaled, y)

    print(f"  Computing H-statistics for {len(top_physics) * len(top_pwm)} pairs...")

    # Prepare arguments for parallel computation
    pairs = []
    for phys_feat in top_physics:
        phys_idx = valid_features.index(phys_feat)
        for pwm_feat in top_pwm:
            pwm_idx = valid_features.index(pwm_feat)
            pairs.append((model, X_scaled, phys_feat, phys_idx, pwm_feat, pwm_idx))

    # Parallel H-statistic computation
    if n_jobs == -1:
        n_jobs = N_CPUS

    interactions = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(compute_h_pair)(args) for args in pairs
    )

    interactions = sorted(interactions, key=lambda x: x['h_statistic'], reverse=True)

    # Build matrix
    h_matrix = np.zeros((len(top_physics), len(top_pwm)))
    for inter in interactions:
        i = top_physics.index(inter['physics_feature'])
        j = top_pwm.index(inter['pwm_feature'])
        h_matrix[i, j] = inter['h_statistic']

    results = {
        'top_physics_features': top_physics,
        'top_pwm_features': top_pwm,
        'h_matrix': h_matrix.tolist(),
        'top_interactions': interactions[:50],
        'mean_h_statistic': float(np.mean(h_matrix)),
        'max_h_statistic': float(np.max(h_matrix)),
        'n_strong_interactions': int(np.sum(h_matrix > 0.1)),
        '_h_matrix_np': h_matrix
    }

    return results


def create_visualizations(results: dict, output_dir: Path, cell_type: str):
    """Create visualization plots."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    h_matrix = results.get('_h_matrix_np', np.array(results['h_matrix']))
    top_physics = results['top_physics_features']
    top_pwm = results['top_pwm_features']

    # 1. Interaction heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(h_matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(top_pwm)))
    ax.set_xticklabels([f.replace('pwm_', '').replace('_max_score', '') for f in top_pwm],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(top_physics)))
    ax.set_yticklabels(top_physics, fontsize=8)

    plt.colorbar(im, label='H-statistic')
    ax.set_xlabel('TF PWM Features')
    ax.set_ylabel('Physics Features')
    ax.set_title(f'Physics × TF Interaction Strength - {cell_type}')
    plt.tight_layout()
    plt.savefig(fig_dir / f'interaction_heatmap_{cell_type}.png', dpi=150)
    plt.close()

    # 2. Top interactions bar plot
    top_inters = results['top_interactions'][:20]
    fig, ax = plt.subplots(figsize=(12, 8))

    labels = [f"{i['physics_feature'][:20]} × {i['pwm_feature'].replace('pwm_', '')[:15]}"
              for i in top_inters]
    values = [i['h_statistic'] for i in top_inters]

    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(values)))
    ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('H-statistic')
    ax.set_title(f'Top 20 Physics × TF Interactions - {cell_type}')
    ax.axvline(x=0.1, color='red', linestyle='--', linewidth=1, label='Strong (0.1)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f'top_interactions_{cell_type}.png', dpi=150)
    plt.close()

    # 3. Physics feature interaction profile
    physics_avg = np.mean(h_matrix, axis=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    order = np.argsort(physics_avg)[::-1]
    ax.barh(range(len(physics_avg)), physics_avg[order], color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(physics_avg)))
    ax.set_yticklabels([top_physics[i] for i in order], fontsize=8)
    ax.set_xlabel('Mean H-statistic')
    ax.set_title(f'Physics Features by Average Interaction Strength - {cell_type}')
    plt.tight_layout()
    plt.savefig(fig_dir / f'physics_interaction_profile_{cell_type}.png', dpi=150)
    plt.close()

    # 4. H-statistic distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(h_matrix.flatten(), bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='Strong (>0.1)')
    ax.axvline(x=0.05, color='orange', linestyle='--', linewidth=2, label='Moderate (>0.05)')
    ax.set_xlabel('H-statistic')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Interaction Strengths - {cell_type}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f'h_distribution_{cell_type}.png', dpi=150)
    plt.close()

    print(f"  Saved visualizations to {fig_dir}")


def generate_text_report(all_results: dict, output_dir: Path) -> str:
    """Generate text summary report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PHYSICS × TF INTERACTION MAPPING REPORT")
    report_lines.append("=" * 80)
    report_lines.append("\nH-statistic measures non-additive interaction strength between features.")
    report_lines.append("Values > 0.1 indicate strong interactions; > 0.05 moderate interactions.")

    for cell_type, results in all_results.items():
        report_lines.append(f"\n{'='*40}")
        report_lines.append(f"Cell Type: {cell_type}")
        report_lines.append('='*40)

        report_lines.append(f"\nMean H-statistic: {results['mean_h_statistic']:.4f}")
        report_lines.append(f"Max H-statistic: {results['max_h_statistic']:.4f}")
        report_lines.append(f"Strong interactions (H > 0.1): {results['n_strong_interactions']}")

        report_lines.append("\n--- Top 15 Interactions ---")
        for i, inter in enumerate(results['top_interactions'][:15], 1):
            report_lines.append(
                f"  {i:2d}. {inter['physics_feature']:30s} × {inter['pwm_feature']:25s}: H = {inter['h_statistic']:.4f}"
            )

        report_lines.append("\n--- Physics Features Most Interactive ---")
        h_matrix = np.array(results['h_matrix'])
        physics_avg = np.mean(h_matrix, axis=1)
        order = np.argsort(physics_avg)[::-1]
        for i in order[:10]:
            report_lines.append(f"  {results['top_physics_features'][i]:40s}: mean H = {physics_avg[i]:.4f}")

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
    """Run full interaction mapping analysis."""

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
            results = compute_interaction_matrix(df, activity_col=activity_col, n_jobs=n_jobs)

            h_matrix_np = results.pop('_h_matrix_np', None)
            all_results[cell_type] = results

            with open(output_dir / f'interactions_{cell_type}.json', 'w') as f:
                json.dump(results, f, indent=2)

            results['_h_matrix_np'] = h_matrix_np
            create_visualizations(results, output_dir, cell_type)
            results.pop('_h_matrix_np', None)

            print(f"\n  Top 10 Physics×PWM Interactions:")
            for inter in results['top_interactions'][:10]:
                print(f"    {inter['physics_feature']} × {inter['pwm_feature']}: H={inter['h_statistic']:.3f}")

        except Exception as e:
            print(f"Error processing {cell_type}: {e}")
            import traceback
            traceback.print_exc()

    with open(output_dir / 'interactions_all.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    generate_text_report(all_results, output_dir)

    print(f"\nResults saved to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Physics×TF Interaction Mapping')
    parser.add_argument('--data_dir', type=str, default='data/lentiMPRA_data')
    parser.add_argument('--output_dir', type=str, default='analyses/results/04_interactions')
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
