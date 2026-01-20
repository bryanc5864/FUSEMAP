#!/usr/bin/env python3
"""
Run APBS calculations for electrostatic potential calibration panel.
Generates input files and runs APBS for each PQR file.
"""

import os
import glob
import subprocess
import pandas as pd

def create_apbs_input(pqr_file, template_file="template_lpbe.in", 
                      output_dir="psi_calibration"):
    """Create APBS input file from template for a specific PQR file."""
    
    base_name = os.path.splitext(os.path.basename(pqr_file))[0]
    
    # Read template
    with open(template_file, 'r') as f:
        template = f.read()
    
    # Substitute placeholders
    apbs_input = template.replace('%PQRFILE%', f'{base_name}.pqr')
    apbs_input = apbs_input.replace('%OUTPREFIX%', f'{base_name}_pot')
    
    # Write APBS input file
    input_file = f"{output_dir}/{base_name}.in"
    with open(input_file, 'w') as f:
        f.write(apbs_input)
    
    return input_file

def run_apbs(input_file, output_dir="psi_calibration"):
    """Run APBS calculation for a specific input file."""
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    log_file = f"{output_dir}/{base_name}.log"
    
    try:
        with open(log_file, 'w') as log:
            bn = os.path.basename(input_file)  # Just the filename
            result = subprocess.run(
                ['apbs', bn],
                cwd=output_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        success = result.returncode == 0
        if success:
            # Check if output DX file was created
            dx_file = f"{output_dir}/{base_name}_pot.dx"
            if os.path.exists(dx_file):
                print(f"  ✓ {base_name}_pot.dx created")
            else:
                print(f"  ✗ APBS ran but no DX file found for {base_name}")
                success = False
        else:
            print(f"  ✗ APBS failed for {base_name}")
            # Show last few lines of log for debugging
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"    Last error: {lines[-1].strip()}")
            except:
                pass
        
        return success
        
    except Exception as e:
        print(f"  ✗ Exception running APBS for {base_name}: {e}")
        return False

def main():
    calibration_dir = "apbs_calculations"
    
    if not os.path.exists(calibration_dir):
        print(f"Error: {calibration_dir} directory not found")
        return
    
    # Find all PQR files
    pqr_files = glob.glob(f"{calibration_dir}/*.pqr")
    
    if not pqr_files:
        print(f"No PQR files found in {calibration_dir}")
        print("Run convert_to_pqr.py first")
        return
    
    print(f"Running APBS calculations for {len(pqr_files)} structures...")
    
    # Check if APBS is available
    try:
        subprocess.run(['apbs', '--version'], capture_output=True)
    except FileNotFoundError:
        print("Error: APBS not found. Make sure it's installed and in PATH")
        return
    
    success_count = 0
    for pqr_file in sorted(pqr_files):
        base_name = os.path.splitext(os.path.basename(pqr_file))[0]
        print(f"Processing {base_name}...")
        
        # Create APBS input file
        try:
            input_file = create_apbs_input(pqr_file, 
                                           output_dir=calibration_dir)
            print(f"  → Created {base_name}.in")
        except Exception as e:
            print(f"  ✗ Failed to create input for {base_name}: {e}")
            continue
        
        # Run APBS
        if run_apbs(input_file, output_dir=calibration_dir):
            success_count += 1
    
    print(f"\nAPBS calculations completed: {success_count}/{len(pqr_files)} successful")
    
    if success_count == len(pqr_files):
        print("Ready for potential extraction!")
    else:
        print("Some calculations failed. Check log files for details.")

if __name__ == "__main__":
    main() 