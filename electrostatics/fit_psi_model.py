#!/usr/bin/env python3
"""
Fit regression model to predict electrostatic potential from sequence features.
Uses calibration panel data to train a physics-based predictor.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import statsmodels.formula.api as smf

def load_calibration_data(features_file="psi_calibration_features.tsv"):
    """Load calibration panel features and potentials."""
    try:
        df = pd.read_csv(features_file, sep='\t')
        return df
    except FileNotFoundError:
        print(f"Error: {features_file} not found")
        print("Run extract_potential_features.py first")
        return None

def fit_linear_model(df):
    """Fit linear regression model with quadratic GC term."""
    
    # Remove NaN values
    df_clean = df.dropna(subset=['psi_APBS'])
    
    if len(df_clean) < 3:
        print("Error: Need at least 3 valid data points for fitting")
        return None, None
    
    print(f"Fitting model with {len(df_clean)} data points...")
    
    # Add quadratic GC term
    df_clean = df_clean.copy()
    df_clean['GC2'] = df_clean['GC_frac'] ** 2
    
    # Fit using statsmodels for detailed statistics
    formula = "psi_APBS ~ GC_frac + GC2 + CpG_density + run_frac + mgw_avg"
    
    try:
        model = smf.ols(formula, data=df_clean).fit()
        
        print("\nModel Summary:")
        print("=" * 50)
        print(f"R-squared: {model.rsquared:.3f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.3f}")
        print(f"AIC: {model.aic:.1f}")
        print(f"P-values:")
        for param, pval in model.pvalues.items():
            significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {param}: {pval:.3f} {significance}")
        
        return model, df_clean
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        return None, None

def save_model_parameters(model, output_file="psi_model_params.json"):
    """Save model parameters to JSON file."""
    
    params = {
        'model_type': 'linear_quadratic_gc',
        'parameters': model.params.to_dict(),
        'r_squared': float(model.rsquared),
        'r_squared_adj': float(model.rsquared_adj),
        'aic': float(model.aic),
        'n_observations': int(model.nobs),
        'formula': 'psi_APBS ~ GC_frac + GC2 + CpG_density + run_frac + mgw_avg',
        'version': '1.0',
        'units': 'kT_per_e'
    }
    
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"\nModel parameters saved to: {output_file}")
    return params

def predict_psi(sequence, model_params):
    """
    Predict electrostatic potential for a sequence using fitted model.
    
    Parameters:
    - sequence: DNA sequence string
    - model_params: dict with model parameters
    
    Returns:
    - predicted psi value (kT/e)
    """
    
    # Compute sequence features (same as in extract_potential_features.py)
    seq = sequence.upper()
    L = len(seq)
    
    gc_count = seq.count('G') + seq.count('C')
    gc_frac = gc_count / L
    
    cpg_count = sum(1 for i in range(L-1) if seq[i:i+2] == 'CG')
    cpg_density = cpg_count / (L - 1) if L > 1 else 0
    
    # Run fraction
    max_run = 0
    current_run = 1
    for i in range(1, L):
        if seq[i] == seq[i-1]:
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 1
    max_run = max(max_run, current_run)
    run_frac = max_run / L
    
    mgw_avg = 5.5  # Default
    
    # Predict using model parameters
    params = model_params['parameters']
    
    psi_pred = (params['Intercept'] +
                params['GC_frac'] * gc_frac +
                params['GC2'] * (gc_frac ** 2) +
                params['CpG_density'] * cpg_density +
                params['run_frac'] * run_frac +
                params['mgw_avg'] * mgw_avg)
    
    return float(psi_pred)

def plot_model_diagnostics(model, df_clean, output_dir="."):
    """Create diagnostic plots for the fitted model."""
    
    predictions = model.predict()
    residuals = model.resid
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Observed vs Predicted
    axes[0,0].scatter(df_clean['psi_APBS'], predictions, alpha=0.7)
    axes[0,0].plot([df_clean['psi_APBS'].min(), df_clean['psi_APBS'].max()],
                   [df_clean['psi_APBS'].min(), df_clean['psi_APBS'].max()], 
                   'r--', alpha=0.8)
    axes[0,0].set_xlabel('Observed ψ (kT/e)')
    axes[0,0].set_ylabel('Predicted ψ (kT/e)')
    axes[0,0].set_title(f'Observed vs Predicted (R² = {model.rsquared:.3f})')
    
    # 2. Residuals vs Predicted
    axes[0,1].scatter(predictions, residuals, alpha=0.7)
    axes[0,1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[0,1].set_xlabel('Predicted ψ (kT/e)')
    axes[0,1].set_ylabel('Residuals')
    axes[0,1].set_title('Residuals vs Predicted')
    
    # 3. GC content vs Potential
    axes[1,0].scatter(df_clean['GC_frac'], df_clean['psi_APBS'], alpha=0.7, label='Observed')
    gc_range = np.linspace(0, 1, 100)
    # Simple prediction for visualization (holding other variables at mean)
    mean_cpg = df_clean['CpG_density'].mean()
    mean_run = df_clean['run_frac'].mean()
    mean_mgw = df_clean['mgw_avg'].mean()
    
    psi_curve = []
    params = model.params
    for gc in gc_range:
        psi = (params['Intercept'] + params['GC_frac']*gc + 
               params['GC2']*(gc**2) + params['CpG_density']*mean_cpg + 
               params['run_frac']*mean_run + params['mgw_avg']*mean_mgw)
        psi_curve.append(psi)
    
    axes[1,0].plot(gc_range, psi_curve, 'r-', alpha=0.8, label='Model')
    axes[1,0].set_xlabel('GC Fraction')
    axes[1,0].set_ylabel('ψ (kT/e)')
    axes[1,0].set_title('GC Content vs Electrostatic Potential')
    axes[1,0].legend()
    
    # 4. Feature importance (coefficients)
    feature_names = ['GC_frac', 'GC2', 'CpG_density', 'run_frac', 'mgw_avg']
    coefficients = [model.params[name] for name in feature_names]
    
    axes[1,1].barh(feature_names, coefficients)
    axes[1,1].set_xlabel('Coefficient Value')
    axes[1,1].set_title('Feature Importance')
    axes[1,1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/psi_model_diagnostics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Diagnostic plots saved to: {output_dir}/psi_model_diagnostics.png")

def main():
    print("Fitting electrostatic potential prediction model...")
    
    # Load calibration data
    df = load_calibration_data()
    if df is None:
        return
    
    print(f"Loaded {len(df)} calibration sequences")
    
    # Check for valid potentials
    valid_psi = df['psi_APBS'].dropna()
    print(f"Valid potentials: {len(valid_psi)}/{len(df)}")
    
    if len(valid_psi) < 3:
        print("ERROR: Insufficient valid potential data for model fitting")
        print("Need to resolve APBS calculation issues first")
        return
    
    # Fit model
    model, df_clean = fit_linear_model(df)
    if model is None:
        return
    
    # Save model parameters
    params = save_model_parameters(model)
    
    # Create diagnostic plots
    plot_model_diagnostics(model, df_clean)
    
    # Test prediction on a few sequences
    print("\nTesting model predictions:")
    test_sequences = [
        ("AT-rich", "ATATATATATATATATATATATATATA"),
        ("GC-rich", "GCGCGCGCGCGCGCGCGCGCGCGCGC"),
        ("Balanced", "ACGTACGTACGTACGTACGTACGTAC")
    ]
    
    for name, seq in test_sequences:
        psi_pred = predict_psi(seq, params)
        print(f"  {name}: ψ = {psi_pred:.3f} kT/e")
    
    print(f"\nModel ready for annotation pipeline!")
    print(f"Use psi_model_params.json for sequence annotation")

if __name__ == "__main__":
    main() 