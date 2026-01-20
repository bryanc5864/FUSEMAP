#!/bin/bash

# Complete APBS Electrostatic Potential Analysis Script
# Adapted for DNA calibration panel

set -e  # Exit on any error

echo "=== APBS Electrostatic Potential Analysis ==="

# Navigate to the fixed PDB directory
cd pdb_structures_fixed

# Check if APBS is installed
if ! command -v apbs &> /dev/null; then
    echo "Error: APBS is not installed or not in PATH"
    echo "Please install APBS and make sure it's in your conda environment"
    exit 1
fi

# Create APBS input template with corrected syntax
cat > template_lpbe.in << 'EOF'
read
    mol pqr %PQRFILE%
end

elec name solvated
    mg-auto
    dime 97 97 97
    cglen 1.5 1.5 1.5
    fglen 20 20 20
    cgcent mol 1
    fgcent mol 1
    mol 1
    lpbe
    bcfl sdh
    pdie 2.0
    sdie 78.5
    ion charge 1 conc 0.15 radius 2.0
    ion charge -1 conc 0.15 radius 2.0
    srfm smol
    chgm spl2
    sdens 10.0
    srad 1.4
    swin 0.30
    temp 298.15
    calcenergy total
    calcforce no
    write pot dx %OUTPREFIX%
end

quit
EOF

# List of DNA panel IDs based on actual calibration panel
panel_ids=("GC00" "GC10" "GC20" "GC30" "GC40" "GC50" "GC60" "GC70" "GC80" "GC90" "GC100" 
           "CpG_low" "CpG_med" "CpG_high" "MGW_narrow" "MGW_wide" "RUN4_A" "RUN4_C" "MIXED1" "MIXED2")

echo "Step 1: Running APBS calculations..."

# Generate input files and run APBS for each panel
for id in "${panel_ids[@]}"; do
    echo "Processing $id..."
    
    # Check if PQR file exists
    if [[ ! -f "${id}.pqr" ]]; then
        echo "Warning: ${id}.pqr not found, skipping..."
        continue
    fi
    
    # Generate APBS input file
    sed "s|%PQRFILE%|${id}.pqr|; s|%OUTPREFIX%|${id}_pot|" template_lpbe.in > "${id}.in"
    
    # Run APBS
    echo "  Running APBS for $id..."
    if apbs "${id}.in" > "${id}.log" 2>&1; then
        echo "  ✓ APBS calculation completed for $id"
    else
        echo "  ✗ APBS calculation failed for $id (check ${id}.log)"
        continue
    fi
    
    # Check if output file was created
    if [[ ! -f "${id}_pot.dx" ]]; then
        echo "  ✗ Warning: Expected output file ${id}_pot.dx not found"
    fi
done

echo ""
echo "Step 2: Analyzing electrostatic potential maps..."

# Create Python analysis script with actual sequences
cat > analyze_potentials.py << 'EOF'
#!/usr/bin/env python3
"""
Analyze APBS electrostatic potential maps and compute sequence features
"""

import numpy as np
import os
import sys
from typing import Tuple, List, Dict, Optional

def install_griddata():
    """Install gridData if not available"""
    try:
        import gridData
        return gridData
    except ImportError:
        print("Installing gridData library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "GridDataFormats"])
        import gridData
        return gridData

def load_dx(dxfile: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load OpenDX file using gridData library
    Returns: (grid_data, origin, delta)
    """
    gridData = install_griddata()
    
    try:
        g = gridData.Grid(dxfile)
        return g.grid, np.array(g.origin), np.array(g.delta)
    except Exception as e:
        print(f"Error loading {dxfile}: {e}")
        raise

def load_atoms_from_pqr(pqrfile: str) -> np.ndarray:
    """
    Load atomic coordinates from PQR file
    Returns: array of (x, y, z) coordinates
    """
    coords = []
    
    if not os.path.exists(pqrfile):
        raise FileNotFoundError(f"PQR file not found: {pqrfile}")
    
    with open(pqrfile, 'r') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(("ATOM", "HETATM")):
                try:
                    # PQR format: columns 31-38, 39-46, 47-54 for x, y, z
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip()) 
                    z = float(line[46:54].strip())
                    coords.append((x, y, z))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line in {pqrfile}: {line[:50]}...")
                    continue
    
    if not coords:
        raise ValueError(f"No valid ATOM/HETATM records found in {pqrfile}")
    
    return np.array(coords)

def shell_mean(dxfile: str, pqrfile: str, inner: float = 2.0, outer: float = 6.0) -> Tuple[float, float]:
    """
    Calculate mean potential in a solvent shell around the molecule
    
    Args:
        dxfile: Path to OpenDX potential file
        pqrfile: Path to PQR structure file  
        inner: Inner shell radius (Å)
        outer: Outer shell radius (Å)
        
    Returns:
        (mean_potential, std_potential)
    """
    try:
        # Load grid data and atomic coordinates
        grid, origin, delta = load_dx(dxfile)
        atoms = load_atoms_from_pqr(pqrfile)
        
        print(f"  Grid shape: {grid.shape}")
        print(f"  Origin: {origin}")
        print(f"  Delta: {delta}")
        print(f"  Number of atoms: {len(atoms)}")
        
        # Create coordinate arrays for all grid points
        # Handle both 1D and 3D delta arrays
        if delta.ndim == 1:
            dx, dy, dz = delta[0], delta[1], delta[2]
        else:
            dx, dy, dz = delta[0,0], delta[1,1], delta[2,2]
            
        # Generate grid coordinates
        nx, ny, nz = grid.shape
        x_coords = origin[0] + np.arange(nx) * dx
        y_coords = origin[1] + np.arange(ny) * dy  
        z_coords = origin[2] + np.arange(nz) * dz
        
        # Create meshgrid
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        # Calculate minimum distance from each grid point to any atom
        print(f"  Calculating distances for {len(grid_points)} grid points...")
        
        # Compute distances efficiently using broadcasting
        # Shape: (n_grid_points, n_atoms)
        distances = np.sqrt(np.sum((grid_points[:, np.newaxis, :] - atoms[np.newaxis, :, :]) ** 2, axis=2))
        min_distances = np.min(distances, axis=1)
        
        # Apply shell mask
        shell_mask = (min_distances >= inner) & (min_distances <= outer)
        shell_points = np.sum(shell_mask)
        
        print(f"  Shell points ({inner}-{outer} Å): {shell_points} / {len(grid_points)}")
        
        if shell_points == 0:
            print(f"  Warning: No grid points found in shell {inner}-{outer} Å")
            return float('nan'), float('nan')
        
        # Extract potential values in the shell
        shell_potentials = grid.ravel()[shell_mask]
        
        # Calculate statistics
        mean_pot = float(np.nanmean(shell_potentials))
        std_pot = float(np.nanstd(shell_potentials))
        
        print(f"  Shell potential: {mean_pot:.3f} ± {std_pot:.3f} kT/e")
        
        return mean_pot, std_pot
        
    except Exception as e:
        print(f"  Error in shell_mean for {dxfile}: {e}")
        return float('nan'), float('nan')

def compute_sequence_features(sequence: str) -> Dict[str, float]:
    """
    Compute basic sequence features
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Dictionary with computed features
    """
    sequence = sequence.upper().replace(' ', '')
    L = len(sequence)
    
    if L == 0:
        return {'GC_frac': 0.0, 'CpG_density': 0.0, 'run_frac': 0.0}
    
    # GC fraction
    gc_count = sequence.count('G') + sequence.count('C')
    gc_frac = gc_count / L
    
    # CpG density (CG dinucleotides)
    cg_count = sequence.count('CG')
    cpg_density = cg_count / (L - 1) if L > 1 else 0.0
    
    # Longest homopolymer run
    max_run = 1
    current_run = 1
    
    for i in range(1, L):
        if sequence[i] == sequence[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    
    run_frac = max_run / L
    
    return {
        'GC_frac': gc_frac,
        'CpG_density': cpg_density,
        'run_frac': run_frac
    }

def get_sequence_for_panel(panel_id: str) -> str:
    """
    Get DNA sequence for a given panel ID from the actual calibration panel
    """
    # Actual sequences from calibration_panel_psi.tsv
    sequences = {
        'GC00': 'ATATATATATATATATATATATAT',
        'GC10': 'ATATACATATACATATACATATAT',
        'GC20': 'ATGATGATGATGATGATGATGATG',
        'GC30': 'ACGTACGTACGTACGTACGTACGT',
        'GC40': 'ACGCGTACGCGTACGCGTACGCGT',
        'GC50': 'GCGCGCGCGCGCGCGCGCGCGCGC',
        'GC60': 'GCGGCGGCGGCGGCGGCGGCGGCG',
        'GC70': 'GGGCGGGCGGGCGGGCGGGCGGGC',
        'GC80': 'GGGGAGGGGAGGGGAGGGGAGGGG',
        'GC90': 'GGGGGGGGGGGGGGGGGGGGGGGG',
        'GC100': 'GGGGGGGGGGGGGGGGGGGGGGGG',
        'CpG_low': 'ATATCGATATATCGATATATCGAT',
        'CpG_med': 'ACGCGTACGCGTACGCGTACGCGT',
        'CpG_high': 'CGCGCGCGCGCGCGCGCGCGCGCG',
        'MGW_narrow': 'AAATTTAAATTTAAATTTAAATTT',
        'MGW_wide': 'ATGCTAGCTAGCTAGCTAGCTAGT',
        'RUN4_A': 'AAAAACCCCCGGGGTTTTAAAAAA',
        'RUN4_C': 'CCCCCAAAATTTTGGGGCCCCCCC',
        'MIXED1': 'ACGTACGATCGATGCTAGCATGCA',
        'MIXED2': 'TGCATGCACTGACTGACTGACTGA'
    }
    
    return sequences.get(panel_id, 'ATCGATCGATCGATCGATCGATCG')

def main():
    """Main analysis function"""
    panel_ids = ['GC00', 'GC10', 'GC20', 'GC30', 'GC40', 'GC50', 'GC60', 'GC70', 'GC80', 'GC90', 'GC100',
                 'CpG_low', 'CpG_med', 'CpG_high', 'MGW_narrow', 'MGW_wide', 'RUN4_A', 'RUN4_C', 'MIXED1', 'MIXED2']
    
    results = []
    
    print("Analyzing electrostatic potentials...")
    
    for panel_id in panel_ids:
        dx_file = f"{panel_id}_pot.dx"
        pqr_file = f"{panel_id}.pqr"
        
        print(f"\nProcessing {panel_id}...")
        
        # Check files exist
        if not os.path.exists(dx_file):
            print(f"  Warning: {dx_file} not found, skipping...")
            continue
            
        if not os.path.exists(pqr_file):
            print(f"  Warning: {pqr_file} not found, skipping...")
            continue
        
        # Calculate shell potential
        try:
            psi_mean, psi_std = shell_mean(dx_file, pqr_file, inner=2.0, outer=6.0)
        except Exception as e:
            print(f"  Error calculating potential: {e}")
            psi_mean, psi_std = float('nan'), float('nan')
        
        # Get sequence and compute features
        sequence = get_sequence_for_panel(panel_id)
        seq_features = compute_sequence_features(sequence)
        
        # Store results
        result = {
            'id': panel_id,
            'psi_APBS': psi_mean,
            'psi_sd': psi_std,
            **seq_features
        }
        
        results.append(result)
        
        print(f"  Results: ψ = {psi_mean:.3f} ± {psi_std:.3f}, "
              f"GC = {seq_features['GC_frac']:.3f}, "
              f"CpG = {seq_features['CpG_density']:.3f}")
    
    # Write results to file
    output_file = 'psi-calibration.tsv'
    
    if results:
        print(f"\nWriting results to {output_file}...")
        
        with open(output_file, 'w') as f:
            # Write header
            f.write("id\tGC_frac\tCpG_density\trun_frac\tpsi_APBS\tpsi_sd\n")
            
            # Write data
            for result in results:
                f.write(f"{result['id']}\t"
                       f"{result['GC_frac']:.4f}\t"
                       f"{result['CpG_density']:.4f}\t"
                       f"{result['run_frac']:.4f}\t"
                       f"{result['psi_APBS']:.4f}\t"
                       f"{result['psi_sd']:.4f}\n")
        
        print(f"✓ Results written to {output_file}")
        
        # Print summary
        print("\nSummary:")
        print("=" * 70)
        print(f"{'ID':<12} {'GC_frac':<8} {'CpG_dens':<8} {'run_frac':<8} {'ψ_APBS':<8} {'ψ_sd':<8}")
        print("-" * 70)
        
        for result in results:
            if not np.isnan(result['psi_APBS']):
                print(f"{result['id']:<12} "
                      f"{result['GC_frac']:<8.3f} "
                      f"{result['CpG_density']:<8.3f} "
                      f"{result['run_frac']:<8.3f} "
                      f"{result['psi_APBS']:<8.3f} "
                      f"{result['psi_sd']:<8.3f}")
    else:
        print("No results to write.")

if __name__ == "__main__":
    main()
EOF

# Make Python script executable
chmod +x analyze_potentials.py

echo "Running Python analysis..."
python3 analyze_potentials.py

echo ""
echo "=== Analysis Complete ==="
echo ""
echo "Output files:"
echo "  - Individual APBS input files: {id}.in"
echo "  - APBS log files: {id}.log" 
echo "  - Electrostatic potential maps: {id}_pot.dx"
echo "  - Analysis results: psi-calibration.tsv"
echo ""
echo "To visualize results, you can load the .dx files in:"
echo "  - PyMOL: pymol {id}_pot.dx"
echo "  - VMD: vmd -e load_dx_script.tcl"
echo "  - ChimeraX: open {id}_pot.dx" 