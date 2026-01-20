#!/usr/bin/env python3
"""
Build DNA structures for all calibration sequences using tleap.
This script generates PDB files for electrostatic calculations.
"""

import pandas as pd
import subprocess
import os

def sequence_to_dna_residues(seq):
    """Convert sequence string to tleap DNA residue format."""
    mapping = {'A': 'DA', 'T': 'DT', 'G': 'DG', 'C': 'DC'}
    return ' '.join(mapping[base] for base in seq.upper())

def create_tleap_script(seq_id, dna_residues, output_dir):
    """Create a tleap script for building DNA structure."""
    script_content = f"""source leaprc.DNA.bsc1
mol = sequence {{ {dna_residues} }}
savePdb mol {seq_id}.pdb
quit"""
    
    script_path = f"{output_dir}/build_{seq_id}.in"
    with open(script_path, 'w') as f:
        f.write(script_content)
    return script_path

def build_dna_structure(script_path, output_dir):
    """Run tleap to build DNA structure."""
    try:
        result = subprocess.run(
            ['tleap', '-f', os.path.basename(script_path)],
            cwd=output_dir,
            capture_output=True,
            text=True
        )
        return result.returncode == 0, result.stderr
    except Exception as e:
        return False, str(e)

def main():
    # Read calibration sequences
    df = pd.read_csv('calibration_panel/calibration_sequences.tsv', 
                     sep='\t')
    
    # Create output directory
    output_dir = 'dna_structures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Building DNA structures for calibration sequences...")
    
    for _, row in df.iterrows():
        seq_id = row['id']
        sequence = row['seq']
        
        print(f"Building {seq_id}: {sequence}")
        
        # Convert sequence to DNA residues
        dna_residues = sequence_to_dna_residues(sequence)
        
        # Create tleap script
        script_path = create_tleap_script(seq_id, dna_residues, 
                                          output_dir)
        
        # Build structure
        success, error = build_dna_structure(script_path, output_dir)
        
        if success:
            print(f"  ✓ {seq_id}.pdb created")
            # Clean up script file
            os.remove(script_path)
        else:
            print(f"  ✗ Failed to build {seq_id}: {error}")
    
    print(f"\nDNA structures saved in: {output_dir}/")
    print("Ready for pdb2pqr and APBS calculations!")

if __name__ == "__main__":
    main() 