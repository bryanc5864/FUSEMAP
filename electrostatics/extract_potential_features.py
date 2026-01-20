#!/usr/bin/env python3
"""
Extract scalar electrostatic potentials from APBS DX files.
Computes shell-averaged potentials and sequence-based features.
"""

import numpy as np
import pandas as pd
import os
import glob
from scipy.spatial import cKDTree

try:
    import gridData
except ImportError:
    print("Warning: gridData not found. Install with: pip install gridDataFormats")
    gridData = None

def load_dx(dxfile):
    """Load OpenDX file using gridDataFormats."""
    if gridData is None:
        raise ImportError("gridDataFormats required for DX file loading")
    
    g = gridData.Grid(dxfile)
    origin = np.array(g.origin)
    Δ = np.array(g.delta)
    
    # Pick off the three grid spacings from the diagonal
    if Δ.ndim == 2:
        delta = np.array([Δ[0,0], Δ[1,1], Δ[2,2]])
    else:
        delta = Δ
    
    return g.grid, origin, delta

def load_atoms_from_pqr(pqrfile):
    """Extract atomic coordinates from PQR file."""
    coords = []
    with open(pqrfile, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                except (ValueError, IndexError):
                    continue
    return np.array(coords)

def shell_mean_potential(dxfile, pqrfile, inner=2.0, outer=3.0):
    """
    Compute mean electrostatic potential in a narrow solvent shell.
    
    Parameters:
    - dxfile: APBS potential DX file
    - pqrfile: corresponding PQR structure file
    - inner: inner shell radius (Å) - default 2.0
    - outer: outer shell radius (Å) - default 3.0 (narrower shell)
    
    Returns:
    - mean_potential: average potential in shell (kT/e)
    - std_potential: standard deviation
    """
    
    try:
        # Load potential grid with corrected delta extraction
        grid, origin, delta = load_dx(dxfile)
        
        # Load atomic coordinates
        atoms = load_atoms_from_pqr(pqrfile)
        
        if len(atoms) == 0:
            return np.nan, np.nan
        
        # Build KDTree for efficient distance queries
        tree = cKDTree(atoms)
        
        # Generate grid coordinate array
        idx = np.indices(grid.shape).reshape(3, -1).T  # (Nvox, 3)
        xyz = origin + idx * delta  # Now delta is 1x3, broadcast works correctly
        
        # Find distance to nearest atom for each grid point
        dmin, _ = tree.query(xyz, k=1)
        
        # Select points in the narrow solvent shell (2-3 Å by default)
        mask = (dmin >= inner) & (dmin <= outer)
        
        if not np.any(mask):
            return np.nan, np.nan
        
        # Extract potential values in the shell
        vals = grid.ravel()[mask]
        
        # Remove any NaN or infinite values
        vals = vals[np.isfinite(vals)]
        
        if len(vals) == 0:
            return np.nan, np.nan
        
        return float(np.mean(vals)), float(np.std(vals))
        
    except Exception as e:
        print(f"Error processing {dxfile}: {e}")
        return np.nan, np.nan

def compute_sequence_features(seq):
    """
    Compute sequence-based features for electrostatic prediction.
    
    Parameters:
    - seq: DNA sequence string
    
    Returns:
    - dict with features: GC_frac, CpG_density, run_frac, mgw_avg
    """
    
    seq = seq.upper()
    L = len(seq)
    
    if L == 0:
        return {'GC_frac': 0, 'CpG_density': 0, 'run_frac': 0, 'mgw_avg': 5.5}
    
    # GC fraction
    gc_count = seq.count('G') + seq.count('C')
    gc_frac = gc_count / L
    
    # CpG density (CG dinucleotides per bp)
    cpg_count = 0
    for i in range(L - 1):
        if seq[i:i+2] == 'CG':
            cpg_count += 1
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
    
    # Minor groove width (placeholder - would need DNAshape predictor)
    mgw_avg = 5.5  # Default B-DNA value
    
    return {
        'GC_frac': gc_frac,
        'CpG_density': cpg_density,
        'run_frac': run_frac,
        'mgw_avg': mgw_avg
    }

def process_calibration_panel(calibration_dir="psi_calibration",
                              panel_file="calibration_panel_psi.tsv"):
    """
    Process all calibration panel results to extract potentials and features.
    
    Returns:
    - DataFrame with calibration results
    """
    
    # Read panel sequences
    try:
        panel_df = pd.read_csv(panel_file, sep='\t')
    except FileNotFoundError:
        print(f"Error: {panel_file} not found")
        return None
    
    results = []
    
    for _, row in panel_df.iterrows():
        seq_id = row['id']
        sequence = row['seq']
        
        print(f"Processing {seq_id}...")
        
        # File paths
        dx_file = f"{calibration_dir}/{seq_id}_pot.dx"
        pqr_file = f"{calibration_dir}/{seq_id}.pqr"
        
        # Extract potential
        if os.path.exists(dx_file) and os.path.exists(pqr_file):
            psi_mean, psi_std = shell_mean_potential(dx_file, pqr_file)
            print(f"  ✓ Potential: {psi_mean:.3f} ± {psi_std:.3f}")
        else:
            psi_mean, psi_std = np.nan, np.nan
            print(f"  ✗ Missing files for {seq_id}")
        
        # Compute sequence features
        features = compute_sequence_features(sequence)
        
        # Combine results
        result = {
            'id': seq_id,
            'seq': sequence,
            'psi_APBS': psi_mean,
            'psi_sd': psi_std,
            **features
        }
        results.append(result)
    
    return pd.DataFrame(results)

def main():
    print("Extracting electrostatic potential features from calibration panel...")
    
    # Process calibration results
    df = process_calibration_panel()
    
    if df is None:
        return
    
    # Save results
    output_file = "psi_calibration_features.tsv"
    df.to_csv(output_file, sep='\t', index=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Processed {len(df)} sequences")
    
    # Show summary statistics
    valid_psi = df['psi_APBS'].dropna()
    if len(valid_psi) > 0:
        print(f"Valid potentials: {len(valid_psi)}/{len(df)}")
        print(f"Potential range: {valid_psi.min():.3f} to {valid_psi.max():.3f}")
        print(f"Mean potential: {valid_psi.mean():.3f} ± {valid_psi.std():.3f}")
    else:
        print("No valid potential values found - check APBS calculations")

if __name__ == "__main__":
    main() 