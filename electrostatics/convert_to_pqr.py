#!/usr/bin/env python3
"""
Convert cleaned DNA PDB structures to PQR format for APBS calculations.
Uses pdb2pqr30 with AMBER force field and proper flags.
"""

import pandas as pd
import subprocess
import os
import glob

def convert_pdb_to_pqr(clean_pdb_file, output_dir="psi_calibration"):
    """Convert a single cleaned PDB file to PQR format."""
    
    base_name = os.path.splitext(os.path.basename(clean_pdb_file))[0]
    
    pqr_file = f"{output_dir}/{base_name}.pqr"
    
    cmd = [
        'pdb2pqr30',
        '--ff=AMBER',
        '--noopt',  # Skip optimization to avoid hangs
        '--with-ph=7.0',
        '--keep-chain',  # Preserve chain IDs
        '--titration-state-method=propka',
        clean_pdb_file,
        pqr_file
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        success = result.returncode == 0
        if success:
            print(f"  ✓ {base_name}.pqr created")
        else:
            print(f"  ✗ Failed to convert {base_name}")
            print(f"    Error: {result.stderr}")
            # Try with minimal flags if first attempt fails
            print(f"  → Trying with minimal flags...")
            cmd_minimal = [
                'pdb2pqr30',
                '--ff=AMBER',
                '--with-ph=7.0',
                clean_pdb_file,
                pqr_file
            ]
            result2 = subprocess.run(cmd_minimal, capture_output=True, text=True)
            if result2.returncode == 0:
                print(f"  ✓ {base_name}.pqr created (minimal flags)")
                success = True
            else:
                print(f"    Minimal flags also failed: {result2.stderr}")
        
        return success
        
    except Exception as e:
        print(f"  ✗ Exception converting {base_name}: {e}")
        return False

def main():
    input_dir = "pdb_structures_fixed"
    output_dir = "apbs_calculations"
    
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} directory not found")
        print("Run fix_pdb_chains.py first")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all fixed PDB files
    clean_pdb_files = glob.glob(f"{input_dir}/*.pdb")
    
    if not clean_pdb_files:
        print(f"No PDB files found in {input_dir}")
        print("Run build_panel_pdbs.py and fix_pdb_chains.py first")
        return
    
    print("Converting cleaned PDB structures to PQR format...")
    
    success_count = 0
    for clean_pdb_file in sorted(clean_pdb_files):
        base_name = os.path.splitext(os.path.basename(clean_pdb_file))[0]
        display_name = base_name
            
        print(f"Converting {display_name}")
        
        if convert_pdb_to_pqr(clean_pdb_file, output_dir):
            success_count += 1
    
    print(f"\nConverted {success_count}/{len(clean_pdb_files)} files successfully")
    
    if success_count == len(clean_pdb_files):
        print("Ready for APBS calculations!")
    else:
        print("Some conversions failed. Check errors above.")

if __name__ == "__main__":
    main() 