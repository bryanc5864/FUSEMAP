#!/usr/bin/env python3
"""
Comprehensive feature analysis for each cell type:
1. Generate histograms of features in 5x5 grids
2. Calculate correlations with scores
3. Calculate feature statistics (variance, mean, std)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from pathlib import Path

def load_data(cell_type):
    """Load train data and features for a cell type."""
    # Load features (which already include scores)
    feature_file = f'output/{cell_type}_train_descriptors.tsv'
    df = pd.read_csv(feature_file, sep='\t')
    
    print(f"  Loaded {len(df)} sequences with {len(df.columns)} columns")
    
    return df

def generate_feature_histograms(df, cell_type, output_dir):
    """Generate 5x5 grid histograms for all features."""
    # Get feature columns (exclude metadata columns)
    metadata_cols = ['seq_id', 'condition', 'normalized_log2', 
                     'n_obs_bc', 'n_replicates', 'sequence']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Create output directory
    hist_dir = Path(output_dir) / f'{cell_type}_histograms'
    hist_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate histograms in batches of 25
    n_features = len(feature_cols)
    n_pages = (n_features + 24) // 25  # Ceiling division
    
    print(f"\nGenerating histograms for {cell_type}: {n_features} features in {n_pages} pages")
    
    for page in range(n_pages):
        start_idx = page * 25
        end_idx = min(start_idx + 25, n_features)
        page_features = feature_cols[start_idx:end_idx]
        
        # Create 5x5 grid
        fig, axes = plt.subplots(5, 5, figsize=(20, 16))
        fig.suptitle(f'{cell_type} Features - Page {page+1}/{n_pages} (Features {start_idx+1}-{end_idx})', 
                     fontsize=16)
        axes = axes.flatten()
        
        for i, feature in enumerate(page_features):
            ax = axes[i]
            values = df[feature].dropna()
            
            # Plot histogram
            ax.hist(values, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title(feature[:30], fontsize=8)  # Truncate long names
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = values.mean()
            std_val = values.std()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
            
            # Add text with stats
            stats_text = f'μ={mean_val:.2e}\nσ={std_val:.2e}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=6, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for i in range(len(page_features), 25):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        output_file = hist_dir / f'{cell_type}_features_page_{page+1:02d}.png'
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved page {page+1}: {output_file}")

def calculate_correlations(df, cell_type, output_dir):
    """Calculate correlations between features and scores."""
    # Get feature columns
    metadata_cols = ['seq_id', 'condition', 'normalized_log2', 
                     'n_obs_bc', 'n_replicates', 'sequence']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    correlations = []
    scores = df['normalized_log2'].values
    
    print(f"\nCalculating correlations for {cell_type}...")
    
    for feature in feature_cols:
        values = df[feature].values
        # Handle NaN values
        mask = ~(np.isnan(values) | np.isnan(scores))
        if mask.sum() > 1:
            corr, p_value = stats.pearsonr(values[mask], scores[mask])
            spearman_corr, spearman_p = stats.spearmanr(values[mask], scores[mask])
        else:
            corr, p_value = np.nan, np.nan
            spearman_corr, spearman_p = np.nan, np.nan
        
        correlations.append({
            'feature': feature,
            'pearson_correlation': corr,
            'pearson_p_value': p_value,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'n_valid': mask.sum() if 'mask' in locals() else 0
        })
    
    # Save to file
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('pearson_correlation', key=abs, ascending=False)
    
    output_file = Path(output_dir) / f'{cell_type}_feature_correlations.txt'
    with open(output_file, 'w') as f:
        f.write(f"Feature Correlations with Score for {cell_type}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Feature':<50} {'Pearson r':>12} {'P-value':>12} {'Spearman ρ':>12} {'P-value':>12} {'N':>8}\n")
        f.write("-" * 120 + "\n")
        
        for _, row in corr_df.iterrows():
            f.write(f"{row['feature']:<50} {row['pearson_correlation']:>12.6f} "
                   f"{row['pearson_p_value']:>12.2e} {row['spearman_correlation']:>12.6f} "
                   f"{row['spearman_p_value']:>12.2e} {row['n_valid']:>8d}\n")
    
    print(f"  Saved correlations: {output_file}")
    
    # Also save top correlations summary
    print(f"\n  Top 10 positive correlations:")
    for _, row in corr_df.head(10).iterrows():
        print(f"    {row['feature']:<40} r={row['pearson_correlation']:>7.4f}")
    
    print(f"\n  Top 10 negative correlations:")
    for _, row in corr_df.tail(10).iterrows():
        print(f"    {row['feature']:<40} r={row['pearson_correlation']:>7.4f}")

def calculate_statistics(df, cell_type, output_dir):
    """Calculate variance, mean, and standard deviation for all features."""
    # Get feature columns
    metadata_cols = ['seq_id', 'condition', 'normalized_log2', 
                     'n_obs_bc', 'n_replicates', 'sequence']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    statistics = []
    
    print(f"\nCalculating statistics for {cell_type}...")
    
    for feature in feature_cols:
        values = df[feature].dropna()
        
        statistics.append({
            'feature': feature,
            'mean': values.mean(),
            'std': values.std(),
            'variance': values.var(),
            'min': values.min(),
            'max': values.max(),
            'q25': values.quantile(0.25),
            'median': values.quantile(0.50),
            'q75': values.quantile(0.75),
            'n_valid': len(values),
            'n_missing': df[feature].isna().sum()
        })
    
    # Save to file
    stats_df = pd.DataFrame(statistics)
    
    output_file = Path(output_dir) / f'{cell_type}_feature_statistics.txt'
    with open(output_file, 'w') as f:
        f.write(f"Feature Statistics for {cell_type}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Feature':<50} {'Mean':>12} {'Std':>12} {'Variance':>12} {'Min':>10} {'Max':>10} {'N_valid':>8}\n")
        f.write("-" * 120 + "\n")
        
        for _, row in stats_df.iterrows():
            f.write(f"{row['feature']:<50} {row['mean']:>12.4e} {row['std']:>12.4e} "
                   f"{row['variance']:>12.4e} {row['min']:>10.2f} {row['max']:>10.2f} "
                   f"{row['n_valid']:>8d}\n")
    
    print(f"  Saved statistics: {output_file}")
    
    # Check for problematic features
    zero_var_features = stats_df[stats_df['variance'] < 1e-10]
    if len(zero_var_features) > 0:
        print(f"\n  WARNING: {len(zero_var_features)} features with near-zero variance:")
        for _, row in zero_var_features.iterrows():
            print(f"    {row['feature']}: var={row['variance']:.2e}")

def main():
    """Main analysis function."""
    cell_types = ['HepG2', 'K562', 'WTC11']
    output_dir = 'feature_analysis'
    Path(output_dir).mkdir(exist_ok=True)
    
    for cell_type in cell_types:
        print(f"\n{'='*60}")
        print(f"Processing {cell_type}")
        print(f"{'='*60}")
        
        # Load data
        print(f"Loading data for {cell_type}...")
        df = load_data(cell_type)
        
        # Generate histograms
        generate_feature_histograms(df, cell_type, output_dir)
        
        # Calculate correlations
        calculate_correlations(df, cell_type, output_dir)
        
        # Calculate statistics
        calculate_statistics(df, cell_type, output_dir)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved in: {output_dir}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()