#!/usr/bin/env python3
"""
Fix PDB chain IDs and residue numbering for duplex DNA structures.
Assigns chain A to first strand, chain B to second strand.
"""

import os
import sys
import glob

def fix_pdb_chains(input_pdb, output_pdb):
    """
    Fix chain IDs and residue numbering for a duplex DNA PDB.
    Assumes first half of residues = strand A, second half = strand B.
    """
    
    with open(input_pdb, 'r') as f:
        lines = f.readlines()
    
    # Find all ATOM lines and count residues
    atom_lines = [line for line in lines if line.startswith('ATOM')]
    if not atom_lines:
        raise ValueError(f"No ATOM records found in {input_pdb}")
    
    # Get unique residue numbers to determine strand break
    residue_numbers = []
    for line in atom_lines:
        resnum = int(line[22:26].strip())
        if resnum not in residue_numbers:
            residue_numbers.append(resnum)
    
    residue_numbers.sort()
    n_residues = len(residue_numbers)
    
    if n_residues % 2 != 0:
        print(f"Warning: {input_pdb} has {n_residues} residues (not even number)")
    
    # Split residues into two strands
    strand_a_residues = residue_numbers[:n_residues//2]
    strand_b_residues = residue_numbers[n_residues//2:]
    
    print(f"  Strand A: residues {strand_a_residues[0]}-{strand_a_residues[-1]} → 1-{len(strand_a_residues)}")
    print(f"  Strand B: residues {strand_b_residues[0]}-{strand_b_residues[-1]} → 1-{len(strand_b_residues)}")
    
    # Create mapping for residue renumbering
    resnum_map = {}
    for i, resnum in enumerate(strand_a_residues):
        resnum_map[resnum] = (i + 1, 'A')
    for i, resnum in enumerate(strand_b_residues):
        resnum_map[resnum] = (i + 1, 'B')
    
    # Process all lines
    output_lines = []
    ter_added_a = False
    
    for line in lines:
        if line.startswith('ATOM'):
            # Extract current residue number
            old_resnum = int(line[22:26].strip())
            new_resnum, chain_id = resnum_map[old_resnum]
            
            # Rebuild the line with proper chain ID and residue number
            new_line = (
                line[:21] +           # ATOM, atom number, atom name, residue name
                chain_id +            # Chain ID (column 22)
                f"{new_resnum:4d}" +  # Residue number (columns 23-26)
                line[26:]             # Rest of line
            )
            output_lines.append(new_line)
            
            # Add TER after last atom of strand A
            if chain_id == 'A' and old_resnum == strand_a_residues[-1] and not ter_added_a:
                # Check if this is the last atom of this residue
                next_atom_same_res = False
                for future_line in lines[lines.index(line)+1:]:
                    if future_line.startswith('ATOM'):
                        future_resnum = int(future_line[22:26].strip())
                        if future_resnum == old_resnum:
                            next_atom_same_res = True
                            break
                        else:
                            break
                
                if not next_atom_same_res:
                    # Get the atom number from current line for proper TER format
                    atom_number = int(line[6:11].strip())
                    resname = line[17:20]
                    ter_line = f"TER   {atom_number:5d}      {resname} {chain_id} {new_resnum:3d}\n"
                    output_lines.append(ter_line)
                    ter_added_a = True
        
        elif line.startswith('TER'):
            # Skip original TER lines, we'll add our own
            continue
        
        elif line.startswith('END'):
            # Add final TER for strand B before END
            last_b_resnum = len(strand_b_residues)
            last_b_resname = "DG"  # Could be DC or DG, get from last atom
            ter_line_b = f"TER   {last_b_resnum:4d}      {last_b_resname} B\n"
            output_lines.append(ter_line_b)
            output_lines.append(line)
        
        else:
            # Keep other lines (HEADER, REMARK, etc.)
            output_lines.append(line)
    
    # Write output
    with open(output_pdb, 'w') as f:
        f.writelines(output_lines)

def main():
    """Fix all PDB files in the pdb_structures directory."""
    
    if len(sys.argv) > 1:
        pdb_dir = sys.argv[1]
    else:
        pdb_dir = "pdb_structures"
    
    if not os.path.exists(pdb_dir):
        print(f"Error: Directory {pdb_dir} not found")
        sys.exit(1)
    
    # Create output directory
    output_dir = f"{pdb_dir}_fixed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDB files
    pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
    
    if not pdb_files:
        print(f"No PDB files found in {pdb_dir}")
        sys.exit(1)
    
    print(f"Fixing chain IDs for {len(pdb_files)} PDB files...")
    print(f"Output directory: {output_dir}")
    
    success = 0
    for pdb_file in sorted(pdb_files):
        filename = os.path.basename(pdb_file)
        output_file = os.path.join(output_dir, filename)
        
        try:
            print(f"\n→ {filename}")
            fix_pdb_chains(pdb_file, output_file)
            print(f"  ✓ Fixed → {filename}")
            success += 1
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"\nProcessed {success}/{len(pdb_files)} files successfully")
    print(f"Fixed PDBs saved in: {output_dir}/")

if __name__ == "__main__":
    main() 