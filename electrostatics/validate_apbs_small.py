#!/usr/bin/env python3
"""
Simplified APBS validation with 10 sequences for debugging.
"""

import os
import pandas as pd
import numpy as np
import random
import subprocess

def main():
    print("Small APBS Validation Test (10 sequences)")
    print("="*50)
    
    # Sample 10 sequences
    dataset_path = "annotated_datasets/train_data_with_sequences_psi_annotated.tsv"
    df = pd.read_csv(dataset_path, sep='\t')
    
    # Get 10 random sequences
    random.seed(42)
    sample_indices = random.sample(range(len(df)), 10)
    
    # Create validation panel
    os.makedirs("validation_small", exist_ok=True)
    os.makedirs("validation_small/calibration_panel", exist_ok=True)
    
    panel_file = "validation_small/calibration_panel/calibration_panel_psi.tsv"
    
    with open(panel_file, 'w') as f:
        f.write("id\tseq\n")
        
        for i, idx in enumerate(sample_indices):
            row = df.iloc[idx]
            sequence = str(row['sequence']).upper().strip()
            seq_id = f"test_{i:02d}"
            
            f.write(f"{seq_id}\t{sequence}\n")
    
    print(f"Created panel with 10 sequences: {panel_file}")
    
    # Copy scripts
    scripts = ['build_panel_pdbs.py', 'convert_to_pqr.py', 'run_apbs_calibration.py', 
               'extract_potential_features.py', 'template_lpbe.in']
    
    for script in scripts:
        if os.path.exists(script):
            subprocess.run(['cp', script, 'validation_small/'])
    
    # Test each step manually
    print("\nStep 1: Building PDB structures...")
    result = subprocess.run(['python', 'build_panel_pdbs.py'], 
                           cwd='validation_small', capture_output=True, text=True)
    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"STDOUT: {result.stdout[:500]}")
    if result.stderr:
        print(f"STDERR: {result.stderr[:500]}")
    
    if result.returncode == 0:
        print("✓ PDB building successful")
        
        # Check what was created
        pdb_files = subprocess.run(['find', '.', '-name', '*.pdb'], 
                                  cwd='validation_small', capture_output=True, text=True)
        print(f"PDB files created: {len(pdb_files.stdout.strip().split()) if pdb_files.stdout.strip() else 0}")
        
        print("\nStep 2: Converting to PQR...")
        result = subprocess.run(['python', 'convert_to_pqr.py'], 
                               cwd='validation_small', capture_output=True, text=True)
        print(f"Return code: {result.returncode}")
        if result.stderr:
            print(f"STDERR: {result.stderr[:500]}")
            
        if result.returncode == 0:
            print("✓ PQR conversion successful")
            
            # Check PQR files
            pqr_files = subprocess.run(['find', '.', '-name', '*.pqr'], 
                                      cwd='validation_small', capture_output=True, text=True)
            print(f"PQR files created: {len(pqr_files.stdout.strip().split()) if pqr_files.stdout.strip() else 0}")
            
            print("\nStep 3: Running APBS...")
            result = subprocess.run(['python', 'run_apbs_calibration.py'], 
                                   cwd='validation_small', capture_output=True, text=True)
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print(f"STDERR: {result.stderr[:500]}")
                
            if result.returncode == 0:
                print("✓ APBS successful")
                
                # Check DX files
                dx_files = subprocess.run(['find', '.', '-name', '*.dx'], 
                                         cwd='validation_small', capture_output=True, text=True)
                print(f"DX files created: {len(dx_files.stdout.strip().split()) if dx_files.stdout.strip() else 0}")
                
                print("\nStep 4: Extracting features...")
                result = subprocess.run(['python', 'extract_potential_features.py'], 
                                       cwd='validation_small', capture_output=True, text=True)
                print(f"Return code: {result.returncode}")
                if result.stderr:
                    print(f"STDERR: {result.stderr[:500]}")
                    
                if result.returncode == 0:
                    print("✓ Feature extraction successful")
                    
                    # Check output files
                    output_files = subprocess.run(['find', '.', '-name', '*.tsv'], 
                                                 cwd='validation_small', capture_output=True, text=True)
                    print(f"Output TSV files: {output_files.stdout}")
                    
                else:
                    print("✗ Feature extraction failed")
            else:
                print("✗ APBS failed")
        else:
            print("✗ PQR conversion failed")
    else:
        print("✗ PDB building failed")
    
    print(f"\nContents of validation_small:")
    subprocess.run(['ls', '-la', 'validation_small/'])

if __name__ == "__main__":
    main() 