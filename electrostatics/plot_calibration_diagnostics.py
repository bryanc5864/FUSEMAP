#!/usr/bin/env python3
"""
Diagnostic plots for electrostatic potential calibration data.
Examines feature distributions, correlations, and relationships.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_calibration_data(filename="psi_calibration_expanded_corrected.tsv"):
    """Load the calibration dataset."""
    df = pd.read_csv(filename, sep='\t')
    df = df.dropna(subset=['psi_APBS'])
    
    print(f"Loaded {len(df)} sequences")
    print(f"Potential range: {df['psi_APBS'].min():.3f} to {df['psi_APBS'].max():.3f}")
    
    return df

def plot_feature_distributions(df):
    """Plot histograms of all features."""
    
    features = ['GC_frac', 'CpG_density', 'run_frac', 'mgw_avg']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, feat in zip(axes, features):
        data = df[feat].dropna()
        ax.hist(data, bins=15, edgecolor='k', alpha=0.7, color='steelblue')
        ax.set_title(f'{feat} Distribution')
        ax.set_xlabel(feat)
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = data.mean()
        std_val = data.std()
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                   label=f'Mean: {mean_val:.3f}')
        ax.text(0.05, 0.95, f'μ = {mean_val:.3f}\nσ = {std_val:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Feature Distributions in Calibration Dataset', fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_psi_vs_features(df):
    """Plot ψ_APBS vs each feature to examine relationships."""
    
    features = ['GC_frac', 'CpG_density', 'run_frac', 'mgw_avg']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, feat in zip(axes, features):
        # Scatter plot
        ax.scatter(df[feat], df['psi_APBS'], alpha=0.7, s=50, color='steelblue')
        ax.set_xlabel(feat)
        ax.set_ylabel('ψ_APBS (kT/e)')
        ax.set_title(f'ψ_APBS vs {feat}')
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = df[feat].corr(df['psi_APBS'])
        ax.text(0.05, 0.95, f'r = {correlation:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add trend line
        if not df[feat].var() == 0:  # Avoid division by zero
            z = np.polyfit(df[feat], df['psi_APBS'], 1)
            p = np.poly1d(z)
            ax.plot(df[feat], p(df[feat]), "r--", alpha=0.8, linewidth=2)
    
    plt.suptitle('Electrostatic Potential vs Sequence Features', fontsize=14)
    plt.tight_layout()
    plt.savefig('psi_vs_features.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_matrix(df):
    """Plot correlation heatmap of all features."""
    
    # Select features for correlation analysis
    corr_features = ['GC_frac', 'CpG_density', 'run_frac', 'mgw_avg', 'psi_APBS']
    corr_data = df[corr_features]
    corr = corr_data.corr()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create heatmap
    im = ax.matshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient')
    
    # Set ticks and labels
    ticks = np.arange(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=45, ha='left')
    ax.set_yticklabels(corr.columns)
    
    # Add correlation values as text
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f'{corr.iloc[i, j]:.3f}', 
                   ha='center', va='center', color='black', fontweight='bold')
    
    plt.title('Feature Correlation Matrix', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr

def plot_psi_distribution(df):
    """Plot distribution of ψ_APBS values."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(df['psi_APBS'], bins=20, edgecolor='k', alpha=0.7, color='darkgreen')
    ax1.set_xlabel('ψ_APBS (kT/e)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Electrostatic Potentials')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_psi = df['psi_APBS'].mean()
    std_psi = df['psi_APBS'].std()
    ax1.axvline(mean_psi, color='red', linestyle='--', alpha=0.8, 
               label=f'Mean: {mean_psi:.3f}')
    ax1.text(0.05, 0.95, f'μ = {mean_psi:.3f}\nσ = {std_psi:.3f}', 
            transform=ax1.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Box plot by sequence type
    seq_types = []
    psi_by_type = {}
    
    for _, row in df.iterrows():
        seq_id = row['id']
        if seq_id.startswith('GC'):
            seq_type = 'GC_ladder'
        elif seq_id.startswith('CpG'):
            seq_type = 'CpG_variants'
        elif seq_id.startswith('RUN'):
            seq_type = 'Homopolymer_runs'
        elif seq_id.startswith('MIX'):
            seq_type = 'Mixed'
        else:
            seq_type = 'Other'
        
        if seq_type not in psi_by_type:
            psi_by_type[seq_type] = []
        psi_by_type[seq_type].append(row['psi_APBS'])
    
    # Create box plot
    box_data = [psi_by_type[key] for key in sorted(psi_by_type.keys())]
    box_labels = sorted(psi_by_type.keys())
    
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('ψ_APBS (kT/e)')
    ax2.set_title('Potential Distribution by Sequence Type')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('psi_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_sequence_features(df):
    """Analyze sequence composition patterns."""
    
    print("\n" + "="*60)
    print("SEQUENCE FEATURE ANALYSIS")
    print("="*60)
    
    features = ['GC_frac', 'CpG_density', 'run_frac', 'psi_APBS']
    
    for feat in features:
        data = df[feat]
        print(f"\n{feat}:")
        print(f"  Range: {data.min():.3f} to {data.max():.3f}")
        print(f"  Mean ± Std: {data.mean():.3f} ± {data.std():.3f}")
        print(f"  Median: {data.median():.3f}")
    
    # Show extreme sequences
    print(f"\nSequences with most negative potential:")
    most_negative = df.nsmallest(5, 'psi_APBS')[['id', 'seq', 'psi_APBS', 'GC_frac', 'CpG_density']]
    print(most_negative.to_string(index=False))
    
    print(f"\nSequences with least negative potential:")
    least_negative = df.nlargest(5, 'psi_APBS')[['id', 'seq', 'psi_APBS', 'GC_frac', 'CpG_density']]
    print(least_negative.to_string(index=False))
    
    # Correlation analysis
    print(f"\nCorrelations with ψ_APBS:")
    for feat in ['GC_frac', 'CpG_density', 'run_frac']:
        corr = df[feat].corr(df['psi_APBS'])
        print(f"  {feat}: {corr:.3f}")

def main():
    """Main function to run all diagnostic analyses."""
    
    print("Electrostatic Potential Calibration - Diagnostic Analysis")
    print("="*60)
    
    # Load data
    df = load_calibration_data()
    
    # Generate diagnostic plots
    print("\nGenerating diagnostic plots...")
    
    plot_feature_distributions(df)
    plot_psi_vs_features(df)
    plot_psi_distribution(df)
    
    print("\nComputing correlation matrix...")
    corr_matrix = plot_correlation_matrix(df)
    
    # Text analysis
    analyze_sequence_features(df)
    
    print(f"\nDiagnostic analysis complete!")
    print(f"Generated plots saved as PNG files")
    print(f"Ready for comprehensive model evaluation")

if __name__ == "__main__":
    main() 