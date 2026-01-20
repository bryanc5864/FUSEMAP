#!/usr/bin/env python3
"""
Fix hydrogen naming conventions in PDB files for Amber compatibility.
H5' -> H5T, H5'' -> H5T (second occurrence), etc.
"""

import os
import glob
import sys

def fix_hydrogen_names(input_pdb, output_pdb):
    """
    Fix hydrogen naming for Amber compatibility.
    """
    
    with open(input_pdb, 'r') as f:
        lines = f.readlines()
    
    output_lines = []
    
    for line in lines:
        if line.startswith('ATOM'):
            # Extract atom name (columns 13-16, but trim spaces)
            atom_name = line[12:16].strip()
            
            # Fix hydrogen naming for Amber
            if atom_name == "H5'":
                # Replace H5' with H5T (5' terminal hydrogen)
                new_line = line[:12] + " H5T" + line[16:]
                output_lines.append(new_line)
            elif atom_name == 'H5"' or atom_name == "H5''":
                # Replace H5'' with H5T (but this should be second occurrence)
                new_line = line[:12] + " H5T" + line[16:]
                output_lines.append(new_line)
            else:
                # Keep other lines unchanged
                output_lines.append(line)
        else:
            # Keep non-ATOM lines unchanged
            output_lines.append(line)
    
    # Write output
    with open(output_pdb, 'w') as f:
        f.writelines(output_lines)

def main():
    """Fix hydrogen names in all PDB files."""
    
    if len(sys.argv) > 1:
        pdb_dir = sys.argv[1]
    else:
        pdb_dir = "pdb_structures_fixed"
    
    if not os.path.exists(pdb_dir):
        print(f"Error: Directory {pdb_dir} not found")
        sys.exit(1)
    
    # Find all PDB files
    pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
    
    if not pdb_files:
        print(f"No PDB files found in {pdb_dir}")
        sys.exit(1)
    
    print(f"Fixing hydrogen names in {len(pdb_files)} PDB files...")
    
    success = 0
    for pdb_file in sorted(pdb_files):
        filename = os.path.basename(pdb_file)
        
        try:
            print(f"→ {filename}")
            fix_hydrogen_names(pdb_file, pdb_file)  # Overwrite in place
            print(f"  ✓ Fixed hydrogen names")
            success += 1
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"\nProcessed {success}/{len(pdb_files)} files successfully")

if __name__ == "__main__":
    main() 