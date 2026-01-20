#!/usr/bin/env python3
"""
05_regime_discovery.py - Regime Discovery Analysis

Discover distinct "physics regimes" in sequence space using clustering:
- HDBSCAN clustering on physics features (GPU-accelerated if available)
- UMAP dimensionality reduction for visualization (GPU-accelerated if available)
- Characterize each regime by dominant physics properties
- Analyze activity distributions per regime

Optimized for GPU (cuML UMAP/HDBSCAN) with CPU fallback.
"""

# Patch numpy for TensorFlow compatibility with NumPy 2.0
import numpy as np
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'string_'):
    np.string_ = np.bytes_
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_
if not hasattr(np, 'object_'):
    np.object_ = np.object_

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
GPU_UMAP = False
GPU_HDBSCAN = False

try:
    from cuml.manifold import UMAP as cuUMAP
    import cupy as cp
    GPU_UMAP = True
    print("GPU UMAP available (cuML)")
except (ImportError, AttributeError, Exception) as e:
    # Catch AttributeError for NumPy 2.0 compatibility issues
    pass

try:
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    GPU_HDBSCAN = True
    print("GPU HDBSCAN available (cuML)")
except (ImportError, AttributeError, Exception) as e:
    pass

# CPU fallbacks
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score

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


def get_physics_features(df: pd.DataFrame) -> list:
    """Get pure physics feature columns (no PWM)."""
    physics_prefixes = ['thermo_', 'stiff_', 'bend_', 'entropy_', 'advanced_']
    return [c for c in df.columns if any(c.startswith(p) for p in physics_prefixes)]


def prepare_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Prepare feature matrix, handling NaN and zero-variance."""
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    variances = np.var(X, axis=0)
    valid_mask = variances > 1e-10
    X = X[:, valid_mask]
    valid_features = [feature_cols[i] for i in range(len(feature_cols)) if valid_mask[i]]

    return X, valid_features


def compute_umap_embedding(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute UMAP embedding with GPU acceleration if available."""
    if GPU_UMAP:
        try:
            import cupy as cp
            reducer = cuUMAP(n_components=n_components, n_neighbors=30, min_dist=0.1, random_state=42)
            embedding = reducer.fit_transform(X)
            return cp.asnumpy(embedding) if hasattr(embedding, 'get') else np.array(embedding)
        except Exception as e:
            print(f"  GPU UMAP failed, falling back to CPU: {e}")

    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=n_components, n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42)
        return reducer.fit_transform(X)

    print("  Using PCA (UMAP not available)")
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def cluster_physics_space(X: np.ndarray, min_cluster_size: int = 50) -> np.ndarray:
    """Cluster sequences in physics feature space with GPU acceleration if available."""
    if GPU_HDBSCAN:
        try:
            clusterer = cuHDBSCAN(min_cluster_size=min_cluster_size, min_samples=10)
            labels = clusterer.fit_predict(X)
            return np.array(labels)
        except Exception as e:
            print(f"  GPU HDBSCAN failed, falling back to CPU: {e}")

    if HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10, metric='euclidean', cluster_selection_method='eom')
        return clusterer.fit_predict(X)

    print("  Using K-means (HDBSCAN not available)")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    return kmeans.fit_predict(X)


def characterize_clusters(X: np.ndarray, labels: np.ndarray, feature_names: list, activity: np.ndarray) -> dict:
    """Characterize each cluster by its physics profile."""
    unique_labels = np.unique(labels)
    cluster_profiles = {}

    global_means = np.mean(X, axis=0)
    global_stds = np.std(X, axis=0)

    for label in unique_labels:
        mask = labels == label
        n_samples = mask.sum()

        if n_samples < 10:
            continue

        cluster_X = X[mask]
        cluster_activity = activity[mask]

        cluster_means = np.mean(cluster_X, axis=0)

        z_scores = np.zeros(len(feature_names))
        for i in range(len(feature_names)):
            if global_stds[i] > 1e-10:
                z_scores[i] = (cluster_means[i] - global_means[i]) / global_stds[i]

        feature_z = list(zip(feature_names, z_scores))
        feature_z_sorted = sorted(feature_z, key=lambda x: abs(x[1]), reverse=True)

        top_positive = [(f, z) for f, z in feature_z_sorted if z > 0][:5]
        top_negative = [(f, z) for f, z in feature_z_sorted if z < 0][:5]

        cluster_profiles[int(label)] = {
            'n_samples': int(n_samples),
            'fraction': float(n_samples / len(labels)),
            'activity_mean': float(np.mean(cluster_activity)),
            'activity_std': float(np.std(cluster_activity)),
            'activity_median': float(np.median(cluster_activity)),
            'activity_q25': float(np.percentile(cluster_activity, 25)),
            'activity_q75': float(np.percentile(cluster_activity, 75)),
            'top_positive_features': [{'feature': f, 'z_score': float(z)} for f, z in top_positive],
            'top_negative_features': [{'feature': f, 'z_score': float(z)} for f, z in top_negative],
            'top_distinguishing': [{'feature': f, 'z_score': float(z)} for f, z in feature_z_sorted[:10]]
        }

    return cluster_profiles


def analyze_regime_activity_models(df: pd.DataFrame, labels: np.ndarray, feature_cols: list, n_jobs: int = -1) -> dict:
    """Fit separate models per regime."""
    unique_labels = [l for l in np.unique(labels) if l >= 0]
    regime_models = {}

    X, valid_features = prepare_features(df, feature_cols)
    y = df['activity'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for label in unique_labels:
        mask = labels == label
        if mask.sum() < 100:
            continue

        X_regime = X_scaled[mask]
        y_regime = y[mask]

        model = ElasticNetCV(l1_ratio=[0.5, 0.9], cv=5, max_iter=5000, random_state=42, n_jobs=n_jobs)
        model.fit(X_regime, y_regime)

        coefs = model.coef_
        importance_order = np.argsort(np.abs(coefs))[::-1]
        top_features = [
            {'feature': valid_features[i], 'coefficient': float(coefs[i])}
            for i in importance_order[:10] if coefs[i] != 0
        ]

        regime_models[int(label)] = {
            'n_samples': int(mask.sum()),
            'r2': float(r2_score(y_regime, model.predict(X_regime))),
            'n_nonzero_features': int(np.sum(coefs != 0)),
            'top_features': top_features
        }

    return regime_models


def run_regime_discovery(df: pd.DataFrame, activity_col: str = 'activity', n_jobs: int = -1) -> dict:
    """Run full regime discovery analysis."""
    physics_cols = get_physics_features(df)
    X, valid_features = prepare_features(df, physics_cols)
    y = df[activity_col].values

    print(f"  Using {len(valid_features)} physics features")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("  Computing UMAP embedding...")
    embedding = compute_umap_embedding(X_scaled)

    print("  Clustering physics space...")
    labels = cluster_physics_space(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  Found {n_clusters} clusters")

    print("  Characterizing clusters...")
    cluster_profiles = characterize_clusters(X_scaled, labels, valid_features, y)

    print("  Fitting per-regime models...")
    regime_models = analyze_regime_activity_models(df, labels, physics_cols, n_jobs=n_jobs)

    noise_fraction = np.mean(labels == -1) if -1 in labels else 0

    results = {
        'n_clusters': n_clusters,
        'noise_fraction': float(noise_fraction),
        'cluster_profiles': cluster_profiles,
        'regime_models': regime_models,
        '_embedding': embedding,
        '_labels': labels
    }

    return results


def create_visualizations(results: dict, output_dir: Path, cell_type: str):
    """Create visualization plots."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    embedding = results.get('_embedding')
    labels = results.get('_labels')
    cluster_profiles = results['cluster_profiles']

    if embedding is not None and labels is not None:
        # 1. UMAP scatter colored by cluster
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', alpha=0.5, s=5)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'Physics Space Clustering - {cell_type}')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.savefig(fig_dir / f'umap_clusters_{cell_type}.png', dpi=150)
        plt.close()

    # 2. Activity distribution per cluster
    if cluster_profiles:
        fig, ax = plt.subplots(figsize=(12, 6))
        cluster_ids = sorted([k for k in cluster_profiles.keys() if k >= 0])
        means = [cluster_profiles[k]['activity_mean'] for k in cluster_ids]
        stds = [cluster_profiles[k]['activity_std'] for k in cluster_ids]
        sizes = [cluster_profiles[k]['n_samples'] for k in cluster_ids]

        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
        bars = ax.bar(range(len(cluster_ids)), means, yerr=stds, color=colors, alpha=0.8, capsize=5)
        ax.set_xticks(range(len(cluster_ids)))
        ax.set_xticklabels([f'Cluster {k}\n(n={sizes[i]})' for i, k in enumerate(cluster_ids)])
        ax.set_ylabel('Mean Activity')
        ax.set_title(f'Activity by Physics Regime - {cell_type}')
        plt.tight_layout()
        plt.savefig(fig_dir / f'regime_activity_{cell_type}.png', dpi=150)
        plt.close()

    # 3. Cluster characterization heatmap
    if cluster_profiles:
        cluster_ids = sorted([k for k in cluster_profiles.keys() if k >= 0])
        all_features = set()
        for k in cluster_ids:
            for feat in cluster_profiles[k]['top_distinguishing'][:5]:
                all_features.add(feat['feature'])
        all_features = list(all_features)[:20]

        if all_features:
            z_matrix = np.zeros((len(cluster_ids), len(all_features)))
            for i, k in enumerate(cluster_ids):
                feat_dict = {f['feature']: f['z_score'] for f in cluster_profiles[k]['top_distinguishing']}
                for j, feat in enumerate(all_features):
                    z_matrix[i, j] = feat_dict.get(feat, 0)

            fig, ax = plt.subplots(figsize=(14, 8))
            im = ax.imshow(z_matrix, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
            ax.set_xticks(range(len(all_features)))
            ax.set_xticklabels(all_features, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(cluster_ids)))
            ax.set_yticklabels([f'Cluster {k}' for k in cluster_ids])
            plt.colorbar(im, label='Z-score')
            ax.set_title(f'Cluster Physics Profiles - {cell_type}')
            plt.tight_layout()
            plt.savefig(fig_dir / f'cluster_profiles_{cell_type}.png', dpi=150)
            plt.close()

    # 4. Per-regime model performance
    if results['regime_models']:
        regime_models = results['regime_models']
        fig, ax = plt.subplots(figsize=(10, 6))
        regime_ids = sorted(regime_models.keys())
        r2_values = [regime_models[k]['r2'] for k in regime_ids]
        sizes = [regime_models[k]['n_samples'] for k in regime_ids]

        colors = plt.cm.tab10(np.linspace(0, 1, len(regime_ids)))
        bars = ax.bar(range(len(regime_ids)), r2_values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(regime_ids)))
        ax.set_xticklabels([f'Regime {k}\n(n={sizes[i]})' for i, k in enumerate(regime_ids)])
        ax.set_ylabel('R² (within-regime)')
        ax.set_title(f'Per-Regime Model Performance - {cell_type}')

        for bar, val in zip(bars, r2_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(fig_dir / f'regime_model_r2_{cell_type}.png', dpi=150)
        plt.close()

    print(f"  Saved visualizations to {fig_dir}")


def generate_text_report(all_results: dict, output_dir: Path) -> str:
    """Generate text summary report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("REGIME DISCOVERY ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("\nThis analysis identifies distinct physics 'regimes' with different activity profiles.")

    for cell_type, results in all_results.items():
        report_lines.append(f"\n{'='*40}")
        report_lines.append(f"Cell Type: {cell_type}")
        report_lines.append('='*40)

        report_lines.append(f"\nNumber of regimes: {results['n_clusters']}")
        report_lines.append(f"Noise fraction: {results['noise_fraction']:.2%}")

        report_lines.append("\n--- Regime Profiles ---")
        for label, profile in sorted(results['cluster_profiles'].items()):
            if label < 0:
                continue
            report_lines.append(f"\nRegime {label}:")
            report_lines.append(f"  Samples: {profile['n_samples']} ({profile['fraction']:.1%})")
            report_lines.append(f"  Activity: {profile['activity_mean']:.2f} +/- {profile['activity_std']:.2f}")
            report_lines.append(f"  Top distinguishing features:")
            for feat in profile['top_distinguishing'][:5]:
                report_lines.append(f"    {feat['feature']:35s}: z = {feat['z_score']:+.2f}")

        if results['regime_models']:
            report_lines.append("\n--- Per-Regime Model Performance ---")
            for label, model in sorted(results['regime_models'].items()):
                report_lines.append(f"  Regime {label}: R² = {model['r2']:.3f} (n={model['n_samples']})")

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
    """Run full regime discovery analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if n_jobs == -1:
        n_jobs = N_CPUS

    print(f"Using {n_jobs} CPU cores, GPU UMAP: {GPU_UMAP}, GPU HDBSCAN: {GPU_HDBSCAN}")

    all_results = {}

    for cell_type in cell_types:
        print(f"\n{'='*50}")
        print(f"Processing {cell_type}")
        print('='*50)

        try:
            df = load_dataset(data_dir, cell_type, split)
            results = run_regime_discovery(df, activity_col=activity_col, n_jobs=n_jobs)

            embedding = results.pop('_embedding', None)
            labels = results.pop('_labels', None)

            all_results[cell_type] = results

            results_summary = {k: v for k, v in results.items()}
            with open(output_dir / f'regimes_{cell_type}.json', 'w') as f:
                json.dump(results_summary, f, indent=2, default=float)

            if embedding is not None:
                np.savez(output_dir / f'embedding_{cell_type}.npz', embedding=embedding, labels=labels)

            results['_embedding'] = embedding
            results['_labels'] = labels
            create_visualizations(results, output_dir, cell_type)

            print(f"\n  Regime Summary:")
            print(f"    Number of regimes: {results['n_clusters']}")
            for label, profile in sorted(results['cluster_profiles'].items()):
                if label >= 0:
                    print(f"    Regime {label}: n={profile['n_samples']}, activity={profile['activity_mean']:.2f}")

        except Exception as e:
            print(f"Error processing {cell_type}: {e}")
            import traceback
            traceback.print_exc()

    generate_text_report(all_results, output_dir)

    print(f"\nResults saved to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Regime Discovery Analysis')
    parser.add_argument('--data_dir', type=str, default='data/lentiMPRA_data')
    parser.add_argument('--output_dir', type=str, default='analyses/results/05_regimes')
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
