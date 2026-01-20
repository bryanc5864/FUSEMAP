#!/usr/bin/env python3
"""
Clean PDB files for pdb2pqr compatibility.
Fixes residue names, adds chain IDs, and ensures proper formatting.
"""

import os
import glob
import re

def clean_pdb_file(input_pdb, output_pdb):
    """Clean a single PDB file for pdb2pqr compatibility."""
    
    with open(input_pdb, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    atom_count = 1
    
    for line in lines:
        # Only keep ATOM and HETATM lines
        if line.startswith(('ATOM', 'HETATM')):
            # Parse the line
            record = line[:6].strip()
            atom_num = line[6:11].strip()
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21:22].strip()
            res_num = line[22:26].strip()
            x = line[30:38].strip()
            y = line[38:46].strip() 
            z = line[46:54].strip()
            occupancy = line[54:60].strip() if len(line) > 54 else "1.00"
            temp_factor = line[60:66].strip() if len(line) > 60 else "0.00"
            
            # Fix residue names: DA->A, DT->T, DG->G, DC->C
            res_mapping = {'DA': 'A', 'DT': 'T', 'DG': 'G', 'DC': 'C'}
            if res_name in res_mapping:
                res_name = res_mapping[res_name]
            
            # Add chain ID if missing
            if not chain_id:
                chain_id = 'A'
            
            # Ensure proper formatting
            try:
                x_val = float(x)
                y_val = float(y)
                z_val = float(z)
                occ_val = float(occupancy) if occupancy else 1.00
                temp_val = float(temp_factor) if temp_factor else 0.00
            except ValueError:
                continue  # Skip malformed lines
            
            # Rebuild the line with proper formatting
            cleaned_line = f"{record:<6}{atom_count:>5} {atom_name:<4}{res_name:>3} {chain_id:>1}{int(res_num):>4}    {x_val:>8.3f}{y_val:>8.3f}{z_val:>8.3f}{occ_val:>6.2f}{temp_val:>6.2f}\n"
            cleaned_lines.append(cleaned_line)
            atom_count += 1
    
    # Write cleaned PDB
    with open(output_pdb, 'w') as f:
        f.writelines(cleaned_lines)
    
    return len(cleaned_lines)

def main():
    calibration_dir = "psi_calibration"
    
    if not os.path.exists(calibration_dir):
        print(f"Error: {calibration_dir} directory not found")
        return
    
    # Find all PDB files
    pdb_files = glob.glob(f"{calibration_dir}/*.pdb")
    
    if not pdb_files:
        print(f"No PDB files found in {calibration_dir}")
        return
    
    print("Cleaning PDB files for pdb2pqr compatibility...")
    
    success_count = 0
    for pdb_file in sorted(pdb_files):
        base_name = os.path.splitext(os.path.basename(pdb_file))[0]
        clean_pdb = f"{calibration_dir}/{base_name}_clean.pdb"
        
        print(f"Cleaning {base_name}.pdb...")
        
        try:
            atom_count = clean_pdb_file(pdb_file, clean_pdb)
            print(f"  ✓ {base_name}_clean.pdb created ({atom_count} atoms)")
            success_count += 1
        except Exception as e:
            print(f"  ✗ Failed to clean {base_name}: {e}")
    
    print(f"\nCleaned {success_count}/{len(pdb_files)} files successfully")
    
    if success_count == len(pdb_files):
        print("Ready for pdb2pqr conversion!")
    else:
        print("Some files failed to clean. Check errors above.")

if __name__ == "__main__":
    main() 