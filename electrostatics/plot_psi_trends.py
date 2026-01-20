#!/usr/bin/env python3
"""
Plot ψ_APBS vs sequence features to visualize expected physical trends.

Expected: Higher GC content → less negative potential (closer to 0)
Expected: Higher CpG density → less negative potential
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def main():
    # Read calibration data
    df = pd.read_csv('psi-calibration.tsv', sep='\t')
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: ψ_APBS vs GC_frac
    ax1 = axes[0]
    scatter1 = ax1.scatter(df['GC_frac'], df['psi_APBS'], 
                          alpha=0.7, s=60, c='blue', edgecolor='black')
    
    # Add trend line
    linreg1 = stats.linregress(df['GC_frac'], df['psi_APBS'])
    slope1, intercept1, r1, p1, se1 = (linreg1.slope, linreg1.intercept, 
                                       linreg1.rvalue, linreg1.pvalue, 
                                       linreg1.stderr)
    x_trend = np.linspace(df['GC_frac'].min(), df['GC_frac'].max(), 100)
    y_trend1 = slope1 * x_trend + intercept1
    ax1.plot(x_trend, y_trend1, 'r--', alpha=0.8, 
             label=f'R² = {r1**2:.3f}, p = {p1:.3f}')
    
    ax1.set_xlabel('GC Fraction')
    ax1.set_ylabel('ψ_APBS (kT/e)')
    ax1.set_title('Electrostatic Potential vs GC Content')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Annotate some key points
    for idx, row in df.iterrows():
        if row['id'] in ['GC00', 'GC100', 'CpG_high', 'CpG_low']:
            ax1.annotate(row['id'], (row['GC_frac'], row['psi_APBS']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    # Plot 2: ψ_APBS vs CpG_density
    ax2 = axes[1]
    scatter2 = ax2.scatter(df['CpG_density'], df['psi_APBS'], 
                          alpha=0.7, s=60, c='green', edgecolor='black')
    
    # Add trend line
    linreg2 = stats.linregress(df['CpG_density'], df['psi_APBS'])
    slope2, intercept2, r2, p2, se2 = (linreg2.slope, linreg2.intercept,
                                       linreg2.rvalue, linreg2.pvalue,
                                       linreg2.stderr)
    x_trend2 = np.linspace(df['CpG_density'].min(), 
                          df['CpG_density'].max(), 100)
    y_trend2 = slope2 * x_trend2 + intercept2
    ax2.plot(x_trend2, y_trend2, 'r--', alpha=0.8, 
             label=f'R² = {r2**2:.3f}, p = {p2:.3f}')
    
    ax2.set_xlabel('CpG Density')
    ax2.set_ylabel('ψ_APBS (kT/e)')
    ax2.set_title('Electrostatic Potential vs CpG Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Annotate CpG variants
    for idx, row in df.iterrows():
        if 'CpG' in row['id']:
            ax2.annotate(row['id'], (row['CpG_density'], row['psi_APBS']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('psi_trends_plot.png', dpi=150, bbox_inches='tight')
    print("Plot saved as psi_trends_plot.png")
    
    # Print summary statistics
    print("=== ψ_APBS Trend Analysis ===")
    print(f"Data range: {df['psi_APBS'].min():.3f} to "
          f"{df['psi_APBS'].max():.3f} kT/e")
    print(f"Mean ± SD: {df['psi_APBS'].mean():.3f} ± "
          f"{df['psi_APBS'].std():.3f} kT/e")
    print()
    
    print("GC_frac vs ψ_APBS:")
    print(f"  Slope: {slope1:.3f} kT/e per GC unit")
    print(f"  R² = {r1**2:.3f}, p = {p1:.3f}")
    print(f"  Expected: positive slope (higher GC → less negative)")
    print(f"  Observed: {'✓' if slope1 > 0 else '✗'}")
    print()
    
    print("CpG_density vs ψ_APBS:")
    print(f"  Slope: {slope2:.3f} kT/e per CpG unit")
    print(f"  R² = {r2**2:.3f}, p = {p2:.3f}")
    print(f"  Expected: positive slope (higher CpG → less negative)")
    print(f"  Observed: {'✓' if slope2 > 0 else '✗'}")
    print()
    
    # Show extreme cases
    print("Extreme cases:")
    min_idx = df['psi_APBS'].idxmin()
    max_idx = df['psi_APBS'].idxmax()
    print(f"  Most negative: {df.loc[min_idx, 'id']} "
          f"(ψ = {df.loc[min_idx, 'psi_APBS']:.3f}, "
          f"GC = {df.loc[min_idx, 'GC_frac']:.3f})")
    print(f"  Least negative: {df.loc[max_idx, 'id']} "
          f"(ψ = {df.loc[max_idx, 'psi_APBS']:.3f}, "
          f"GC = {df.loc[max_idx, 'GC_frac']:.3f})")

if __name__ == "__main__":
    main() 