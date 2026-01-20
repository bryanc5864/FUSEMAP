#!/usr/bin/env python3
"""
01_univariate_stability.py - Univariate Stability Analysis

For each physics feature, compute:
- Pearson and Spearman correlations with activity
- 10-fold cross-validation resampling stability
- Partial correlations controlling for GC content
- Sign-consistency across folds and cell types

Optimized for parallel CPU processing.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Get number of CPUs
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


def get_feature_categories(columns: List[str]) -> Dict[str, List[str]]:
    """Categorize features by type."""
    categories = {
        'pwm': [], 'thermo': [], 'stiff': [],
        'bend': [], 'entropy': [], 'advanced': []
    }

    for col in columns:
        for prefix in categories.keys():
            if col.startswith(f'{prefix}_'):
                categories[prefix].append(col)
                break

    return categories


def compute_gc_content(sequence: str) -> float:
    """Compute GC content of a sequence."""
    if not sequence or len(sequence) == 0:
        return 0.0
    gc_count = sequence.upper().count('G') + sequence.upper().count('C')
    return gc_count / len(sequence)


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """Compute partial correlation between x and y, controlling for z."""
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[mask], y[mask], z[mask]

    if len(x) < 10:
        return np.nan, np.nan

    try:
        slope_xz = np.polyfit(z, x, 1)
        x_resid = x - np.polyval(slope_xz, z)
        slope_yz = np.polyfit(z, y, 1)
        y_resid = y - np.polyval(slope_yz, z)
        r, p = pearsonr(x_resid, y_resid)
        return r, p
    except:
        return np.nan, np.nan


def compute_single_feature_correlation(
    feature: str,
    feature_values: np.ndarray,
    activity: np.ndarray,
    gc: np.ndarray,
    category: str,
    n_folds: int = 10
) -> dict:
    """Compute correlations for a single feature (parallelizable)."""
    x = feature_values

    # Skip if all NaN or zero variance
    if np.all(np.isnan(x)) or np.nanstd(x) == 0:
        return None

    mask = ~(np.isnan(x) | np.isnan(activity))
    if mask.sum() < 10:
        return None

    x_clean = x[mask]
    y_clean = activity[mask]
    gc_clean = gc[mask]

    try:
        pearson_r, pearson_p = pearsonr(x_clean, y_clean)
        spearman_r, spearman_p = spearmanr(x_clean, y_clean)
        partial_r, partial_p = partial_correlation(x_clean, y_clean, gc_clean)
    except:
        return None

    # Cross-validation stability
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_correlations = []
    for train_idx, val_idx in kf.split(x_clean):
        try:
            r, _ = pearsonr(x_clean[val_idx], y_clean[val_idx])
            fold_correlations.append(r)
        except:
            pass

    if len(fold_correlations) >= 5:
        fold_std = np.std(fold_correlations)
        sign_consistency = np.mean([1 if r * pearson_r > 0 else 0 for r in fold_correlations])
    else:
        fold_std = np.nan
        sign_consistency = np.nan

    return {
        'feature': feature,
        'category': category,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'partial_r_gc': partial_r,
        'partial_p_gc': partial_p,
        'fold_std': fold_std,
        'sign_consistency': sign_consistency,
        'n_samples': len(x_clean)
    }


def compute_univariate_correlations(
    df: pd.DataFrame,
    activity_col: str = 'activity',
    n_folds: int = 10,
    n_jobs: int = -1
) -> pd.DataFrame:
    """Compute univariate correlations with parallel processing."""

    if n_jobs == -1:
        n_jobs = N_CPUS

    feature_cols = [c for c in df.columns if any(c.startswith(p) for p in
                   ['pwm_', 'thermo_', 'stiff_', 'bend_', 'entropy_', 'advanced_'])]

    categories = get_feature_categories(feature_cols)
    feature_to_category = {}
    for cat, cols in categories.items():
        for col in cols:
            feature_to_category[col] = cat

    if 'gc_content' not in df.columns:
        df['gc_content'] = df['sequence'].apply(compute_gc_content)

    activity = df[activity_col].values
    gc = df['gc_content'].values

    print(f"Computing correlations for {len(feature_cols)} features using {n_jobs} CPUs...")

    # Parallel computation
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(compute_single_feature_correlation)(
            feature=feature,
            feature_values=df[feature].values,
            activity=activity,
            gc=gc,
            category=feature_to_category.get(feature, 'unknown'),
            n_folds=n_folds
        )
        for feature in feature_cols
    )

    # Filter None results
    results = [r for r in results if r is not None]
    return pd.DataFrame(results)


def analyze_cross_celltype_consistency(
    results_by_celltype: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Analyze which features have consistent effects across cell types."""
    all_features = set()
    for df in results_by_celltype.values():
        all_features.update(df['feature'].tolist())

    consistency_results = []
    for feature in all_features:
        correlations = {}
        for cell_type, df in results_by_celltype.items():
            row = df[df['feature'] == feature]
            if len(row) > 0:
                correlations[cell_type] = row['pearson_r'].values[0]

        if len(correlations) >= 2:
            values = list(correlations.values())
            signs = [1 if v > 0 else -1 for v in values]

            consistency_results.append({
                'feature': feature,
                'n_celltypes': len(correlations),
                'mean_r': np.mean(values),
                'std_r': np.std(values),
                'sign_agreement': len(set(signs)) == 1,
                **{f'{ct}_r': r for ct, r in correlations.items()}
            })

    return pd.DataFrame(consistency_results)


def create_visualizations(
    results: pd.DataFrame,
    output_dir: Path,
    cell_type: str
):
    """Create visualization plots."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Correlation distribution by category
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    categories = ['thermo', 'stiff', 'bend', 'entropy', 'advanced', 'pwm']

    for ax, cat in zip(axes.flatten(), categories):
        cat_data = results[results['category'] == cat]['pearson_r'].dropna()
        if len(cat_data) > 0:
            ax.hist(cat_data, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Pearson r')
            ax.set_ylabel('Count')
            ax.set_title(f'{cat.upper()} (n={len(cat_data)})')
            ax.text(0.05, 0.95, f'mean={cat_data.mean():.3f}',
                   transform=ax.transAxes, verticalalignment='top')

    plt.suptitle(f'Correlation Distribution by Feature Category - {cell_type}', fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / f'correlation_distribution_{cell_type}.png', dpi=150)
    plt.close()

    # 2. Top features bar plot
    top_n = 30
    top_features = results.reindex(results['pearson_r'].abs().nlargest(top_n).index)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['green' if r > 0 else 'red' for r in top_features['pearson_r']]
    bars = ax.barh(range(len(top_features)), top_features['pearson_r'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=8)
    ax.set_xlabel('Pearson Correlation with Activity')
    ax.set_title(f'Top {top_n} Physics Features by |Correlation| - {cell_type}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / f'top_features_{cell_type}.png', dpi=150)
    plt.close()

    # 3. Partial vs Raw correlation scatter
    fig, ax = plt.subplots(figsize=(10, 10))
    physics_only = results[results['category'] != 'pwm']
    ax.scatter(physics_only['pearson_r'], physics_only['partial_r_gc'],
               alpha=0.5, c='blue', s=20)
    ax.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='y=x')
    ax.set_xlabel('Raw Pearson r')
    ax.set_ylabel('Partial r (controlling GC)')
    ax.set_title(f'Effect of GC Content Control - {cell_type}')
    ax.legend()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(fig_dir / f'partial_vs_raw_{cell_type}.png', dpi=150)
    plt.close()

    # 4. Stability heatmap (fold_std vs correlation magnitude)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        results['pearson_r'].abs(),
        results['fold_std'],
        c=results['sign_consistency'],
        cmap='RdYlGn',
        alpha=0.6,
        s=30
    )
    ax.set_xlabel('|Pearson r|')
    ax.set_ylabel('Fold Std (instability)')
    ax.set_title(f'Correlation Stability - {cell_type}')
    plt.colorbar(scatter, label='Sign Consistency')
    plt.tight_layout()
    plt.savefig(fig_dir / f'stability_{cell_type}.png', dpi=150)
    plt.close()

    print(f"  Saved visualizations to {fig_dir}")


def create_cross_celltype_visualizations(
    consistency: pd.DataFrame,
    results_by_celltype: Dict[str, pd.DataFrame],
    output_dir: Path,
    cell_types: List[str]
):
    """Create cross-cell-type visualization plots."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Correlation heatmap across cell types
    if len(cell_types) >= 2:
        # Get top universal features (by absolute mean_r)
        consistent = consistency[consistency['sign_agreement'] == True].copy()
        consistent['abs_mean_r'] = consistent['mean_r'].abs()
        universal = consistent.nlargest(50, 'abs_mean_r').drop(columns=['abs_mean_r'])

        if len(universal) > 0:
            # Build correlation matrix
            corr_matrix = np.zeros((len(universal), len(cell_types)))
            for i, (_, row) in enumerate(universal.iterrows()):
                for j, ct in enumerate(cell_types):
                    col = f'{ct}_r'
                    if col in row:
                        corr_matrix[i, j] = row[col]

            fig, ax = plt.subplots(figsize=(8, 14))
            im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
            ax.set_xticks(range(len(cell_types)))
            ax.set_xticklabels(cell_types)
            ax.set_yticks(range(len(universal)))
            ax.set_yticklabels(universal['feature'], fontsize=7)
            plt.colorbar(im, label='Pearson r')
            ax.set_title('Top Universal Features Across Cell Types')
            plt.tight_layout()
            plt.savefig(fig_dir / 'cross_celltype_heatmap.png', dpi=150)
            plt.close()

    # 2. Cell type correlation comparison
    if len(cell_types) >= 2:
        fig, axes = plt.subplots(1, len(cell_types)-1, figsize=(5*(len(cell_types)-1), 5))
        if len(cell_types) == 2:
            axes = [axes]

        ref_ct = cell_types[0]
        for ax, other_ct in zip(axes, cell_types[1:]):
            ref_col = f'{ref_ct}_r'
            other_col = f'{other_ct}_r'

            if ref_col in consistency.columns and other_col in consistency.columns:
                ax.scatter(consistency[ref_col], consistency[other_col], alpha=0.3, s=10)
                ax.plot([-1, 1], [-1, 1], 'r--', linewidth=2)
                ax.set_xlabel(f'{ref_ct} Pearson r')
                ax.set_ylabel(f'{other_ct} Pearson r')
                ax.set_title(f'{ref_ct} vs {other_ct}')
                ax.set_xlim(-0.6, 0.6)
                ax.set_ylim(-0.6, 0.6)

        plt.tight_layout()
        plt.savefig(fig_dir / 'celltype_correlation_comparison.png', dpi=150)
        plt.close()

    print(f"  Saved cross-cell-type visualizations to {fig_dir}")


def generate_text_report(
    results_by_celltype: Dict[str, pd.DataFrame],
    consistency: pd.DataFrame,
    output_dir: Path,
    cell_types: List[str]
) -> str:
    """Generate text summary report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("UNIVARIATE STABILITY ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    for cell_type, results in results_by_celltype.items():
        report_lines.append(f"\n{'='*40}")
        report_lines.append(f"Cell Type: {cell_type}")
        report_lines.append('='*40)

        report_lines.append(f"\nTotal features analyzed: {len(results)}")
        report_lines.append(f"Significant (p < 0.05): {(results['pearson_p'] < 0.05).sum()}")
        report_lines.append(f"Highly significant (p < 0.01): {(results['pearson_p'] < 0.01).sum()}")

        report_lines.append("\n--- Top 15 Positive Correlations ---")
        top_pos = results.nlargest(15, 'pearson_r')[['feature', 'category', 'pearson_r', 'partial_r_gc', 'sign_consistency']]
        report_lines.append(top_pos.to_string(index=False))

        report_lines.append("\n--- Top 15 Negative Correlations ---")
        top_neg = results.nsmallest(15, 'pearson_r')[['feature', 'category', 'pearson_r', 'partial_r_gc', 'sign_consistency']]
        report_lines.append(top_neg.to_string(index=False))

        report_lines.append("\n--- By Category Summary ---")
        for cat in ['thermo', 'stiff', 'bend', 'entropy', 'advanced']:
            cat_data = results[results['category'] == cat]
            if len(cat_data) > 0:
                mean_r = cat_data['pearson_r'].abs().mean()
                max_r = cat_data['pearson_r'].abs().max()
                report_lines.append(f"  {cat:10s}: n={len(cat_data):3d}, mean|r|={mean_r:.3f}, max|r|={max_r:.3f}")

    # Cross-cell-type summary
    if len(cell_types) > 1 and len(consistency) > 0:
        report_lines.append(f"\n{'='*40}")
        report_lines.append("CROSS-CELL-TYPE CONSISTENCY")
        report_lines.append('='*40)

        n_consistent = consistency['sign_agreement'].sum()
        report_lines.append(f"\nFeatures with consistent sign across all cell types: {n_consistent}/{len(consistency)}")

        report_lines.append("\n--- Top 20 Universal Features ---")
        consistent_df = consistency[consistency['sign_agreement'] == True].copy()
        consistent_df['abs_mean_r'] = consistent_df['mean_r'].abs()
        universal = consistent_df.nlargest(20, 'abs_mean_r').drop(columns=['abs_mean_r'])
        if len(universal) > 0:
            display_cols = ['feature', 'mean_r', 'std_r'] + [f'{ct}_r' for ct in cell_types if f'{ct}_r' in universal.columns]
            report_lines.append(universal[display_cols].to_string(index=False))

    report_text = '\n'.join(report_lines)

    # Save report
    report_path = output_dir / 'report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"  Saved text report to {report_path}")
    return report_text


def run_analysis(
    data_dir: Path,
    output_dir: Path,
    cell_types: List[str],
    split: str = 'train',
    n_jobs: int = -1,
    activity_col: str = 'activity'
) -> Dict:
    """Run full univariate stability analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_celltype = {}

    for cell_type in cell_types:
        print(f"\n{'='*50}")
        print(f"Processing {cell_type}")
        print('='*50)

        try:
            df = load_dataset(data_dir, cell_type, split)
            results = compute_univariate_correlations(df, activity_col=activity_col, n_jobs=n_jobs)
            results['cell_type'] = cell_type
            results_by_celltype[cell_type] = results

            # Save per-cell-type results
            results.to_csv(output_dir / f'univariate_{cell_type}.csv', index=False)

            # Create visualizations
            create_visualizations(results, output_dir, cell_type)

            # Print top features
            print(f"\nTop 10 features by |Pearson r| for {cell_type}:")
            top = results.reindex(results['pearson_r'].abs().nlargest(10).index)[['feature', 'category', 'pearson_r', 'partial_r_gc']]
            print(top.to_string(index=False))

        except Exception as e:
            print(f"Error processing {cell_type}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-cell-type consistency
    consistency = pd.DataFrame()
    if len(results_by_celltype) > 1:
        print(f"\n{'='*50}")
        print("Cross-Cell-Type Consistency Analysis")
        print('='*50)

        consistency = analyze_cross_celltype_consistency(results_by_celltype)
        consistency.to_csv(output_dir / 'cross_celltype_consistency.csv', index=False)

        # Create cross-cell-type visualizations
        create_cross_celltype_visualizations(consistency, results_by_celltype, output_dir, cell_types)

        # Universal features
        universal_candidates = consistency[
            (consistency['sign_agreement'] == True) &
            (consistency['n_celltypes'] == len(cell_types))
        ].copy()
        universal_candidates['abs_mean_r'] = universal_candidates['mean_r'].abs()
        universal = universal_candidates.nlargest(20, 'abs_mean_r').drop(columns=['abs_mean_r'])

        print(f"\nTop 20 universal physics features:")
        print(universal[['feature', 'mean_r', 'std_r']].to_string(index=False))

    # Combine all results
    all_results = pd.concat(results_by_celltype.values(), ignore_index=True)
    all_results.to_csv(output_dir / 'univariate_all.csv', index=False)

    # Generate text report
    generate_text_report(results_by_celltype, consistency, output_dir, cell_types)

    # Summary JSON
    summary = {
        'cell_types': cell_types,
        'n_features': len(all_results['feature'].unique()),
        'n_significant_p05': int((all_results['pearson_p'] < 0.05).sum()),
        'n_significant_p01': int((all_results['pearson_p'] < 0.01).sum()),
        'top_features_by_category': {}
    }

    for category in ['thermo', 'stiff', 'bend', 'entropy', 'advanced']:
        cat_results = all_results[all_results['category'] == category]
        if len(cat_results) > 0:
            top = cat_results.reindex(cat_results['pearson_r'].abs().nlargest(5).index)['feature'].tolist()
            summary['top_features_by_category'][category] = top

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='Univariate Stability Analysis')
    parser.add_argument('--data_dir', type=str, default='data/lentiMPRA_data')
    parser.add_argument('--output_dir', type=str, default='analyses/results/01_univariate')
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
