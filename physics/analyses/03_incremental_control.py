#!/usr/bin/env python3
"""
03_incremental_control.py - Incremental Control Analysis

Add PWM aggregates as controls and quantify the attenuation of physics coefficients:
- Step 1: Physics-only model
- Step 2: Physics + PWM aggregates
- Step 3: Physics + all TF features
- Measure Delta R² and coefficient attenuation (indicates mediation)

Optimized for parallel CPU processing with visualizations.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
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
    """Separate features into groups."""
    columns = df.columns.tolist()

    physics_prefixes = ['thermo_', 'stiff_', 'bend_', 'entropy_', 'advanced_']
    pure_physics = [c for c in columns if any(c.startswith(p) for p in physics_prefixes)]

    pwm_agg_patterns = ['pwm_max_of_max', 'pwm_min_delta_g', 'pwm_tf_binding_diversity',
                        'pwm_sum_top5', 'pwm_best_tf']
    pwm_aggregates = [c for c in columns if any(p in c for p in pwm_agg_patterns)]

    pwm_per_tf = [c for c in columns if c.startswith('pwm_') and c not in pwm_aggregates]

    return {
        'pure_physics': pure_physics,
        'pwm_aggregates': pwm_aggregates,
        'pwm_per_tf': pwm_per_tf
    }


def prepare_feature_matrix(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Prepare feature matrix, handling NaN and zero-variance."""
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    variances = np.var(X, axis=0)
    valid_mask = variances > 1e-10
    X = X[:, valid_mask]
    valid_features = [feature_cols[i] for i in range(len(feature_cols)) if valid_mask[i]]

    return X, valid_features


def fit_model(X: np.ndarray, y: np.ndarray, n_folds: int = 5, n_jobs: int = -1) -> dict:
    """Fit Elastic Net and return metrics."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=np.logspace(-4, 1, 30),
        cv=n_folds,
        max_iter=5000,
        random_state=42,
        n_jobs=n_jobs
    )
    model.fit(X_scaled, y)

    cv_scores = cross_val_score(model, X_scaled, y, cv=n_folds, scoring='r2', n_jobs=n_jobs)

    return {
        'r2_train': r2_score(y, model.predict(X_scaled)),
        'r2_cv_mean': np.mean(cv_scores),
        'r2_cv_std': np.std(cv_scores),
        'n_features': X.shape[1],
        'n_nonzero': np.sum(model.coef_ != 0),
        'coefficients': model.coef_,
        'model': model,
        'scaler': scaler
    }


def compute_coefficient_attenuation(
    coefs_before: np.ndarray,
    coefs_after: np.ndarray,
    feature_names: list
) -> pd.DataFrame:
    """Compute attenuation of coefficients when adding controls."""
    results = []

    for i, name in enumerate(feature_names):
        before = abs(coefs_before[i])
        after = abs(coefs_after[i]) if i < len(coefs_after) else 0

        if before > 1e-10:
            attenuation = (before - after) / before
        else:
            attenuation = 0.0

        results.append({
            'feature': name,
            'coef_before': coefs_before[i],
            'coef_after': coefs_after[i] if i < len(coefs_after) else 0,
            'abs_before': before,
            'abs_after': after,
            'attenuation': attenuation,
            'sign_change': (coefs_before[i] * (coefs_after[i] if i < len(coefs_after) else 0)) < 0
        })

    return pd.DataFrame(results)


def run_incremental_analysis(
    df: pd.DataFrame,
    activity_col: str = 'activity',
    n_jobs: int = -1
) -> dict:
    """Run incremental control analysis."""
    y = df[activity_col].values
    groups = get_feature_groups(df)

    results = {}

    # Step 1: Physics only
    print("  Step 1: Physics only...")
    X_physics, physics_features = prepare_feature_matrix(df, groups['pure_physics'])
    step1 = fit_model(X_physics, y, n_jobs=n_jobs)
    results['step1_physics_only'] = {
        'r2_cv': step1['r2_cv_mean'],
        'r2_cv_std': step1['r2_cv_std'],
        'n_features': step1['n_features'],
        'n_nonzero': step1['n_nonzero']
    }
    print(f"    R² (CV): {step1['r2_cv_mean']:.4f}")

    # Step 2: Physics + PWM aggregates
    print("  Step 2: Physics + PWM aggregates...")
    combined_agg = groups['pure_physics'] + groups['pwm_aggregates']
    X_agg, agg_features = prepare_feature_matrix(df, combined_agg)
    step2 = fit_model(X_agg, y, n_jobs=n_jobs)
    results['step2_with_pwm_agg'] = {
        'r2_cv': step2['r2_cv_mean'],
        'r2_cv_std': step2['r2_cv_std'],
        'n_features': step2['n_features'],
        'n_nonzero': step2['n_nonzero'],
        'delta_r2_from_step1': step2['r2_cv_mean'] - step1['r2_cv_mean']
    }
    print(f"    R² (CV): {step2['r2_cv_mean']:.4f} (Delta R² = {step2['r2_cv_mean'] - step1['r2_cv_mean']:+.4f})")

    # Step 3: Physics + all PWM
    print("  Step 3: Physics + all PWM features...")
    combined_all = groups['pure_physics'] + groups['pwm_aggregates'] + groups['pwm_per_tf']
    X_all, all_features = prepare_feature_matrix(df, combined_all)
    step3 = fit_model(X_all, y, n_jobs=n_jobs)
    results['step3_with_all_pwm'] = {
        'r2_cv': step3['r2_cv_mean'],
        'r2_cv_std': step3['r2_cv_std'],
        'n_features': step3['n_features'],
        'n_nonzero': step3['n_nonzero'],
        'delta_r2_from_step1': step3['r2_cv_mean'] - step1['r2_cv_mean'],
        'delta_r2_from_step2': step3['r2_cv_mean'] - step2['r2_cv_mean']
    }
    print(f"    R² (CV): {step3['r2_cv_mean']:.4f} (Delta R² from step1 = {step3['r2_cv_mean'] - step1['r2_cv_mean']:+.4f})")

    # Compute coefficient attenuation for physics features
    n_physics = len(physics_features)
    physics_coefs_step1 = step1['coefficients']

    # Get physics coefficients from step3
    physics_idx_in_all = [all_features.index(f) for f in physics_features if f in all_features]
    physics_coefs_step3 = step3['coefficients'][physics_idx_in_all] if len(physics_idx_in_all) == len(physics_features) else np.zeros(len(physics_features))

    attenuation_df = compute_coefficient_attenuation(
        physics_coefs_step1,
        physics_coefs_step3,
        physics_features
    )

    # Summarize attenuation
    results['attenuation_summary'] = {
        'mean_attenuation': float(attenuation_df['attenuation'].mean()),
        'median_attenuation': float(attenuation_df['attenuation'].median()),
        'n_fully_attenuated': int((attenuation_df['attenuation'] > 0.9).sum()),
        'n_sign_changes': int(attenuation_df['sign_change'].sum()),
        'most_attenuated': attenuation_df.nlargest(10, 'attenuation')[['feature', 'attenuation']].to_dict('records'),
        'least_attenuated': attenuation_df.nsmallest(10, 'attenuation')[['feature', 'attenuation']].to_dict('records')
    }

    results['attenuation_details'] = attenuation_df.to_dict('records')
    results['_attenuation_df'] = attenuation_df  # For visualization

    return results


def create_visualizations(
    results: dict,
    output_dir: Path,
    cell_type: str
):
    """Create visualization plots."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Incremental R² bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = ['Physics Only', 'Physics + PWM Agg', 'Physics + All PWM']
    r2_values = [
        results['step1_physics_only']['r2_cv'],
        results['step2_with_pwm_agg']['r2_cv'],
        results['step3_with_all_pwm']['r2_cv']
    ]
    colors = ['steelblue', 'forestgreen', 'darkorange']

    bars = ax.bar(steps, r2_values, color=colors, alpha=0.8)
    ax.set_ylabel('Cross-Validation R²')
    ax.set_title(f'Incremental Model Performance - {cell_type}')
    ax.set_ylim(0, max(r2_values) * 1.2)

    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11)

    # Add delta annotations
    delta1 = results['step2_with_pwm_agg']['delta_r2_from_step1']
    delta2 = results['step3_with_all_pwm']['delta_r2_from_step2']
    ax.annotate('', xy=(1, r2_values[1]), xytext=(0, r2_values[0]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(2, r2_values[2]), xytext=(1, r2_values[1]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    plt.tight_layout()
    plt.savefig(fig_dir / f'incremental_r2_{cell_type}.png', dpi=150)
    plt.close()

    # 2. Attenuation distribution
    if '_attenuation_df' in results:
        attenuation_df = results['_attenuation_df']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of attenuation
        ax = axes[0]
        ax.hist(attenuation_df['attenuation'], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No attenuation')
        ax.axvline(x=1, color='green', linestyle='--', linewidth=2, label='Full attenuation')
        ax.set_xlabel('Coefficient Attenuation')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of Physics Coefficient Attenuation - {cell_type}')
        ax.legend()

        # Top attenuated features
        ax = axes[1]
        top_att = attenuation_df.nlargest(15, 'attenuation')
        colors = ['green' if a > 0.5 else 'orange' if a > 0 else 'red' for a in top_att['attenuation']]
        ax.barh(range(len(top_att)), top_att['attenuation'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_att)))
        ax.set_yticklabels(top_att['feature'], fontsize=8)
        ax.set_xlabel('Attenuation')
        ax.set_title(f'Most Attenuated Physics Features - {cell_type}')
        ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1)

        plt.tight_layout()
        plt.savefig(fig_dir / f'attenuation_distribution_{cell_type}.png', dpi=150)
        plt.close()

    # 3. Before vs After coefficient scatter
    if '_attenuation_df' in results:
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(attenuation_df['coef_before'], attenuation_df['coef_after'],
                  alpha=0.5, s=30, c='steelblue')
        ax.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='No change')
        ax.plot([-1, 1], [0, 0], 'g--', linewidth=1, alpha=0.5, label='Full attenuation')
        ax.set_xlabel('Coefficient (Physics Only)')
        ax.set_ylabel('Coefficient (Physics + All PWM)')
        ax.set_title(f'Coefficient Attenuation - {cell_type}')
        ax.legend()
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)

        plt.tight_layout()
        plt.savefig(fig_dir / f'coef_before_after_{cell_type}.png', dpi=150)
        plt.close()

    # 4. Category-wise attenuation
    if '_attenuation_df' in results:
        cat_attenuation = {'thermo': [], 'stiff': [], 'bend': [], 'entropy': [], 'advanced': []}

        for _, row in attenuation_df.iterrows():
            for cat in cat_attenuation.keys():
                if row['feature'].startswith(f'{cat}_'):
                    cat_attenuation[cat].append(row['attenuation'])
                    break

        fig, ax = plt.subplots(figsize=(10, 6))
        cats = list(cat_attenuation.keys())
        means = [np.mean(cat_attenuation[c]) if cat_attenuation[c] else 0 for c in cats]
        stds = [np.std(cat_attenuation[c]) if cat_attenuation[c] else 0 for c in cats]

        bars = ax.bar(cats, means, yerr=stds, color='steelblue', alpha=0.8, capsize=5)
        ax.set_ylabel('Mean Attenuation')
        ax.set_title(f'Attenuation by Physics Category - {cell_type}')
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='50% threshold')
        ax.legend()

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(fig_dir / f'category_attenuation_{cell_type}.png', dpi=150)
        plt.close()

    print(f"  Saved visualizations to {fig_dir}")


def generate_text_report(
    all_results: dict,
    output_dir: Path
) -> str:
    """Generate text summary report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("INCREMENTAL CONTROL ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("\nThis analysis tests whether TF binding mediates physics effects on activity.")
    report_lines.append("If physics coefficients attenuate when adding PWM features, it suggests mediation.")

    for cell_type, results in all_results.items():
        report_lines.append(f"\n{'='*40}")
        report_lines.append(f"Cell Type: {cell_type}")
        report_lines.append('='*40)

        s1 = results['step1_physics_only']
        s2 = results['step2_with_pwm_agg']
        s3 = results['step3_with_all_pwm']
        att = results['attenuation_summary']

        report_lines.append("\n--- Incremental Model Performance ---")
        report_lines.append(f"Step 1 (Physics only):       R² = {s1['r2_cv']:.4f} +/- {s1['r2_cv_std']:.4f}")
        report_lines.append(f"Step 2 (+ PWM aggregates):   R² = {s2['r2_cv']:.4f} +/- {s2['r2_cv_std']:.4f}  (Delta = {s2['delta_r2_from_step1']:+.4f})")
        report_lines.append(f"Step 3 (+ all PWM):          R² = {s3['r2_cv']:.4f} +/- {s3['r2_cv_std']:.4f}  (Delta = {s3['delta_r2_from_step1']:+.4f})")

        report_lines.append("\n--- Coefficient Attenuation Summary ---")
        report_lines.append(f"Mean attenuation: {att['mean_attenuation']:.2%}")
        report_lines.append(f"Median attenuation: {att['median_attenuation']:.2%}")
        report_lines.append(f"Fully attenuated (>90%): {att['n_fully_attenuated']} features")
        report_lines.append(f"Sign changes: {att['n_sign_changes']} features")

        report_lines.append("\n--- Most Attenuated Features (likely mediated by TF binding) ---")
        for feat in att['most_attenuated'][:10]:
            report_lines.append(f"  {feat['feature']:40s}: {feat['attenuation']:.2%}")

        report_lines.append("\n--- Least Attenuated Features (independent of TF binding) ---")
        for feat in att['least_attenuated'][:10]:
            report_lines.append(f"  {feat['feature']:40s}: {feat['attenuation']:.2%}")

    # Summary interpretation
    report_lines.append(f"\n{'='*40}")
    report_lines.append("INTERPRETATION")
    report_lines.append('='*40)

    mean_att = np.mean([r['attenuation_summary']['mean_attenuation'] for r in all_results.values()])
    mean_delta = np.mean([r['step3_with_all_pwm']['delta_r2_from_step1'] for r in all_results.values()])

    report_lines.append(f"\nMean coefficient attenuation: {mean_att:.1%}")
    report_lines.append(f"Mean R² improvement from PWM: {mean_delta:+.4f}")

    if mean_att > 0.5:
        report_lines.append("\n>>> High attenuation suggests TF binding MEDIATES many physics effects")
    elif mean_att > 0.2:
        report_lines.append("\n>>> Moderate attenuation suggests PARTIAL mediation by TF binding")
    else:
        report_lines.append("\n>>> Low attenuation suggests physics effects are INDEPENDENT of TF binding")

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
    """Run full incremental control analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if n_jobs == -1:
        n_jobs = N_CPUS

    print(f"Using {n_jobs} CPU cores")

    all_results = {}

    for cell_type in cell_types:
        print(f"\n{'='*50}")
        print(f"Processing {cell_type}")
        print('='*50)

        try:
            df = load_dataset(data_dir, cell_type, split)
            results = run_incremental_analysis(df, activity_col=activity_col, n_jobs=n_jobs)

            # Store attenuation_df separately
            attenuation_df = results.pop('_attenuation_df', None)

            all_results[cell_type] = results

            # Save per-cell-type results
            with open(output_dir / f'incremental_{cell_type}.json', 'w') as f:
                json.dump(results, f, indent=2, default=float)

            # Create visualizations (need to add back attenuation_df)
            results['_attenuation_df'] = attenuation_df
            create_visualizations(results, output_dir, cell_type)
            results.pop('_attenuation_df', None)

            # Print summary
            print(f"\n  Attenuation Summary:")
            print(f"    Mean attenuation: {results['attenuation_summary']['mean_attenuation']:.2%}")
            print(f"    Fully attenuated (>90%): {results['attenuation_summary']['n_fully_attenuated']}")

        except Exception as e:
            print(f"Error processing {cell_type}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined results
    with open(output_dir / 'incremental_all.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    # Generate text report
    generate_text_report(all_results, output_dir)

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)

    for cell_type, results in all_results.items():
        s1 = results['step1_physics_only']['r2_cv']
        s3 = results['step3_with_all_pwm']['r2_cv']
        att = results['attenuation_summary']['mean_attenuation']
        print(f"{cell_type}: Physics R²={s1:.3f}, +PWM R²={s3:.3f}, Mean attenuation={att:.1%}")

    print(f"\nResults saved to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Incremental Control Analysis')
    parser.add_argument('--data_dir', type=str, default='data/lentiMPRA_data')
    parser.add_argument('--output_dir', type=str, default='analyses/results/03_incremental')
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
