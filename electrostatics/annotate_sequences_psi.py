#!/usr/bin/env python3
"""
Annotate 400bp sequences with electrostatic potential predictions.
Complete pipeline for physics-based electrostatic features.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path

def load_model_parameters(params_file="psi_model_params.json"):
    """Load fitted model parameters."""
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        return params
    except FileNotFoundError:
        print(f"Error: {params_file} not found")
        print("Run fit_psi_model.py first to train the model")
        return None

def compute_sequence_features(seq):
    """
    Compute sequence-based features for electrostatic prediction.
    Must match exactly the features used in model training.
    """
    
    seq = seq.upper()
    L = len(seq)
    
    if L == 0:
        return {
            'GC_frac': 0, 'CpG_density': 0, 'run_frac': 0, 'mgw_avg': 5.5,
            'N_frac': 1.0, 'psi_quality': 'low'
        }
    
    # Count valid bases
    valid_bases = sum(1 for base in seq if base in 'ATGC')
    n_frac = 1.0 - (valid_bases / L)
    
    # Quality flag
    psi_quality = 'low' if n_frac > 0.25 else 'good'
    
    # For sequences with too many Ns, use GC approximation
    if n_frac > 0.5:
        # Use sequence without Ns for calculation
        clean_seq = ''.join(base for base in seq if base in 'ATGC')
        if len(clean_seq) == 0:
            return {
                'GC_frac': 0.5, 'CpG_density': 0, 'run_frac': 0, 'mgw_avg': 5.5,
                'N_frac': n_frac, 'psi_quality': 'low'
            }
        seq = clean_seq
        L = len(seq)
    
    # GC fraction
    gc_count = seq.count('G') + seq.count('C')
    gc_frac = gc_count / L
    
    # CpG density (CG dinucleotides per bp)
    cpg_count = sum(1 for i in range(L-1) if seq[i:i+2] == 'CG')
    cpg_density = cpg_count / (L - 1) if L > 1 else 0
    
    # Run fraction (longest homopolymer run)
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
    
    # Minor groove width (placeholder)
    mgw_avg = 5.5  # Default B-DNA value
    
    return {
        'GC_frac': gc_frac,
        'CpG_density': cpg_density,
        'run_frac': run_frac,
        'mgw_avg': mgw_avg,
        'N_frac': n_frac,
        'psi_quality': psi_quality
    }

def predict_psi(sequence, model_params):
    """
    Predict electrostatic potential for a sequence using fitted model.
    
    Parameters:
    - sequence: DNA sequence string
    - model_params: dict with model parameters
    
    Returns:
    - predicted psi value (kT/e)
    """
    
    # Compute sequence features
    features = compute_sequence_features(sequence)
    
    # Handle low quality sequences
    if features['psi_quality'] == 'low':
        if features['N_frac'] > 0.5:
            return np.nan  # Too many Ns to predict reliably
    
    # Predict using model parameters
    params = model_params['parameters']
    
    gc_frac = features['GC_frac']
    
    psi_pred = (params['Intercept'] +
                params['GC_frac'] * gc_frac +
                params['GC2'] * (gc_frac ** 2) +
                params['CpG_density'] * features['CpG_density'] +
                params['run_frac'] * features['run_frac'] +
                params['mgw_avg'] * features['mgw_avg'])
    
    return float(psi_pred)

def compute_reference_scaling(training_positives, model_params):
    """
    Compute reference distribution statistics from training positives.
    Used for z-score normalization.
    """
    
    print("Computing reference scaling from training positives...")
    
    psi_values = []
    for _, row in training_positives.iterrows():
        sequence = row['seq'] if 'seq' in row else row['sequence']
        psi = predict_psi(sequence, model_params)
        if not np.isnan(psi):
            psi_values.append(psi)
    
    psi_values = np.array(psi_values)
    
    if len(psi_values) == 0:
        print("Warning: No valid positive sequences for reference scaling")
        return {'mu_pos': 0.0, 'sigma_pos': 1.0}
    
    mu_pos = float(np.mean(psi_values))
    sigma_pos = float(np.std(psi_values))
    
    if sigma_pos == 0:
        sigma_pos = 1.0  # Avoid division by zero
    
    print(f"Reference statistics: μ = {mu_pos:.3f}, σ = {sigma_pos:.3f}")
    print(f"Based on {len(psi_values)} valid positive sequences")
    
    return {'mu_pos': mu_pos, 'sigma_pos': sigma_pos}

def annotate_sequences(sequences_df, model_params, reference_stats):
    """
    Annotate sequences with electrostatic potential features.
    
    Parameters:
    - sequences_df: DataFrame with sequences
    - model_params: fitted model parameters
    - reference_stats: dict with mu_pos, sigma_pos for z-scoring
    
    Returns:
    - DataFrame with added electrostatic features
    """
    
    print(f"Annotating {len(sequences_df)} sequences...")
    
    results = []
    
    for idx, row in sequences_df.iterrows():
        # Get sequence
        sequence = row['seq'] if 'seq' in row else row['sequence']
        
        # Compute features
        features = compute_sequence_features(sequence)
        
        # Predict raw potential
        psi_raw = predict_psi(sequence, model_params)
        
        # Compute z-score relative to positives
        if not np.isnan(psi_raw):
            psi_z_pos = (psi_raw - reference_stats['mu_pos']) / reference_stats['sigma_pos']
            # Clip for numerical stability
            psi_clip = np.clip(psi_z_pos, -5.0, 5.0)
        else:
            psi_z_pos = np.nan
            psi_clip = np.nan
        
        # Add electrostatic features to row
        result = row.copy()
        result['psi_raw'] = psi_raw
        result['psi_z_pos'] = psi_z_pos
        result['psi_clip'] = psi_clip
        result['GC_frac'] = features['GC_frac']
        result['CpG_density'] = features['CpG_density']
        result['run_frac'] = features['run_frac']
        result['N_frac'] = features['N_frac']
        result['psi_quality'] = features['psi_quality']
        result['psi_model_version'] = model_params['version']
        
        results.append(result)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(sequences_df)} sequences")
    
    return pd.DataFrame(results)

def main():
    """Main annotation pipeline."""
    
    if len(sys.argv) < 3:
        print("Usage: python annotate_sequences_psi.py <sequences.tsv> <training_positives.tsv> [output.tsv]")
        print("\nExample:")
        print("  python annotate_sequences_psi.py all_sequences.tsv training_positives.tsv sequences_with_psi.tsv")
        return
    
    sequences_file = sys.argv[1]
    training_positives_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "sequences_annotated_psi.tsv"
    
    print("Electrostatic Potential Annotation Pipeline")
    print("=" * 50)
    
    # Load model parameters
    model_params = load_model_parameters()
    if model_params is None:
        return
    
    print(f"Loaded model: {model_params['model_type']}")
    print(f"Model R²: {model_params['r_squared']:.3f}")
    print(f"Training samples: {model_params['n_observations']}")
    
    # Load sequences to annotate
    try:
        sequences_df = pd.read_csv(sequences_file, sep='\t')
        print(f"Loaded {len(sequences_df)} sequences from {sequences_file}")
    except FileNotFoundError:
        print(f"Error: {sequences_file} not found")
        return
    
    # Load training positives for reference scaling
    try:
        training_positives = pd.read_csv(training_positives_file, sep='\t')
        print(f"Loaded {len(training_positives)} training positives from {training_positives_file}")
    except FileNotFoundError:
        print(f"Error: {training_positives_file} not found")
        return
    
    # Compute reference scaling
    reference_stats = compute_reference_scaling(training_positives, model_params)
    
    # Annotate sequences
    annotated_df = annotate_sequences(sequences_df, model_params, reference_stats)
    
    # Save results
    annotated_df.to_csv(output_file, sep='\t', index=False)
    
    print(f"\nAnnotation complete!")
    print(f"Results saved to: {output_file}")
    
    # Summary statistics
    valid_psi = annotated_df['psi_raw'].dropna()
    print(f"\nSummary:")
    print(f"  Total sequences: {len(annotated_df)}")
    print(f"  Valid predictions: {len(valid_psi)}")
    print(f"  ψ range: {valid_psi.min():.3f} to {valid_psi.max():.3f} kT/e")
    print(f"  Mean ψ: {valid_psi.mean():.3f} ± {valid_psi.std():.3f}")
    
    # Quality breakdown
    quality_counts = annotated_df['psi_quality'].value_counts()
    print(f"  Quality: {dict(quality_counts)}")

if __name__ == "__main__":
    main() 