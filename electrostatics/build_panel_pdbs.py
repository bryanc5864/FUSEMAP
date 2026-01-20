#!/usr/bin/env python3
"""
Build canonical B-DNA duplex PDBs for a calibration panel using tleap.
"""

import os
import subprocess
import sys
import pandas as pd

PANEL_TSV = "calibration_panel/calibration_panel_psi.tsv"    # your TSV with columns: id, seq, …
OUT_DIR   = "pdb_structures"               # where PDBs will go

# Map A/C/G/T to Amber residue names
RES_MAP = {"A":"DA","T":"DT","G":"DG","C":"DC"}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def run(cmd, cwd=None):
    """Run a subprocess, raise on error."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed:\n{result.stderr}")
    return result

def write_tleap_input(seq_id, seq, out_dir):
    """Create a tleap input script that builds a duplex from seq."""
    # convert to 'DA DT DG DC ...'
    residues = " ".join(RES_MAP[b] for b in seq.upper())
    tleap_in = os.path.join(out_dir, f"build_{seq_id}.in")
    pdb_out   = f"{seq_id}.pdb"  # relative path only
    with open(tleap_in, "w") as f:
        f.write(f"""\
source leaprc.DNA.bsc1

# Build duplex from single‐strand code
mol = sequence {{ {residues} }}
savePdb mol {pdb_out}
quit
""")
    return tleap_in, pdb_out

def build_pdb(seq_id, seq, out_dir):
    """Run tleap to build the PDB for one panel member."""
    tleap_in, pdb_out = write_tleap_input(seq_id, seq, out_dir)
    # call tleap from the output directory
    tleap_input_file = os.path.basename(tleap_in)
    run(["tleap", "-f", tleap_input_file], cwd=out_dir)
    os.remove(tleap_in)
    full_pdb_path = os.path.join(out_dir, pdb_out)
    if not os.path.exists(full_pdb_path):
        raise RuntimeError(f"PDB not created: {full_pdb_path}")
    print(f"  ✓ {seq_id}.pdb")
    return full_pdb_path

def main():
    if not os.path.exists(PANEL_TSV):
        print(f"Error: '{PANEL_TSV}' not found", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(PANEL_TSV, sep="\t")[["id","seq"]]
    ensure_dir(OUT_DIR)
    print(f"Building {len(df)} duplex PDBs in '{OUT_DIR}/' …")
    success = 0
    for _, row in df.iterrows():
        seq_id, seq = row["id"], row["seq"]
        try:
            print(f"→ {seq_id}: {seq}")
            build_pdb(seq_id, seq, OUT_DIR)
            success += 1
        except Exception as e:
            print(f"  ✗ {seq_id} failed: {e}", file=sys.stderr)
    print(f"\nBuilt {success}/{len(df)} structures.")
    if success != len(df):
        sys.exit(2)

if __name__ == "__main__":
    main() 