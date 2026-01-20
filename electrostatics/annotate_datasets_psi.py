#!/usr/bin/env python3
"""
Annotate datasets with electrostatic potential predictions and sequence features.
Uses our calibrated ψ prediction model: ψ = -2.879 - 0.088 × GC_frac
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Tuple

class ElectrostaticPotentialAnnotator:
    """Annotates DNA sequences with predicted electrostatic potential and features."""
    
    def __init__(self):
        """Initialize with our calibrated model."""
        # Our calibrated model: ψ = -2.879 - 0.088 × GC_frac
        self.psi_intercept = -2.879
        self.psi_slope = -0.088
        
        # Model performance metrics
        self.model_r2 = 0.425
        self.model_mae = 0.020  # kT/e
        
        print("Electrostatic Potential Annotator initialized")
        print(f"Model: ψ = {self.psi_intercept:.3f} + {self.psi_slope:.3f} × GC_frac")
        print(f"Performance: R² = {self.model_r2:.3f}, MAE = {self.model_mae:.3f} kT/e")
    
    def compute_sequence_features(self, sequence: str) -> Dict[str, float]:
        """Compute sequence-based features for a DNA sequence."""
        
        sequence = sequence.upper().strip()
        L = len(sequence)
        
        if L == 0:
            return {
                'GC_frac': 0.0,
                'CpG_density': 0.0, 
                'run_frac': 0.0,
                'length': 0
            }
        
        # GC fraction
        gc_count = sequence.count('G') + sequence.count('C')
        gc_frac = gc_count / L
        
        # CpG density (CG dinucleotides per bp)
        cpg_count = 0
        for i in range(L - 1):
            if sequence[i:i+2] == 'CG':
                cpg_count += 1
        cpg_density = cpg_count / (L - 1) if L > 1 else 0.0
        
        # Run fraction (longest homopolymer run)
        max_run = 1
        current_run = 1
        for i in range(1, L):
            if sequence[i] == sequence[i-1]:
                current_run += 1
            else:
                max_run = max(max_run, current_run)
                current_run = 1
        max_run = max(max_run, current_run)
        run_frac = max_run / L
        
        return {
            'GC_frac': gc_frac,
            'CpG_density': cpg_density,
            'run_frac': run_frac,
            'length': L
        }
    
    def predict_electrostatic_potential(self, gc_frac: float) -> float:
        """Predict electrostatic potential using our calibrated model."""
        return self.psi_intercept + self.psi_slope * gc_frac
    
    def annotate_sequence(self, sequence: str) -> Dict[str, float]:
        """Annotate a single sequence with all features and predictions."""
        
        # Compute sequence features
        features = self.compute_sequence_features(sequence)
        
        # Predict electrostatic potential
        psi_predicted = self.predict_electrostatic_potential(features['GC_frac'])
        
        # Combine results
        return {
            'psi_predicted': psi_predicted,
            'GC_frac': features['GC_frac'],
            'CpG_density': features['CpG_density'],
            'run_frac': features['run_frac'],
            'seq_length': features['length']
        }

def load_bed_file(bed_path: str) -> pd.DataFrame:
    """Load a BED format file with proper column handling."""
    
    print(f"Loading BED file: {bed_path}")
    
    # Try different possible BED formats
    try:
        # Standard BED format: chr, start, end, [name], [score], [strand], ...
        df = pd.read_csv(bed_path, sep='\t', header=None)
        
        # Determine number of columns and assign names
        if df.shape[1] >= 3:
            base_cols = ['chr', 'start', 'end']
            if df.shape[1] >= 4:
                base_cols.append('name')
            if df.shape[1] >= 5:
                base_cols.append('score')
            if df.shape[1] >= 6:
                base_cols.append('strand')
            
            # Add any remaining columns as extra_col_N
            remaining_cols = [f'extra_col_{i}' for i in range(len(base_cols), df.shape[1])]
            df.columns = base_cols + remaining_cols
        
        print(f"  Loaded {len(df)} entries with {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error loading BED file {bed_path}: {e}")
        return None

def load_sequence_file(seq_path: str) -> List[str]:
    """Load sequences from a text file (one per line)."""
    
    print(f"Loading sequences: {seq_path}")
    
    try:
        with open(seq_path, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
        
        print(f"  Loaded {len(sequences)} sequences")
        if sequences:
            print(f"  Example length: {len(sequences[0])} bp")
        
        return sequences
        
    except Exception as e:
        print(f"Error loading sequences {seq_path}: {e}")
        return []

def annotate_bed_dataset(bed_path: str, seq_path: str, output_path: str, 
                        annotator: ElectrostaticPotentialAnnotator) -> bool:
    """Annotate a BED dataset with sequences."""
    
    print(f"\nAnnotating BED dataset: {os.path.basename(bed_path)}")
    print("="*50)
    
    # Load BED file and sequences
    bed_df = load_bed_file(bed_path)
    sequences = load_sequence_file(seq_path)
    
    if bed_df is None or not sequences:
        print(f"Failed to load data for {bed_path}")
        return False
    
    if len(bed_df) != len(sequences):
        print(f"Mismatch: {len(bed_df)} BED entries vs {len(sequences)} sequences")
        return False
    
    # Annotate each sequence
    print("Computing electrostatic potential annotations...")
    annotations = []
    
    start_time = time.time()
    for i, sequence in enumerate(sequences):
        if i % 10000 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (len(sequences) - i) / rate / 60
            print(f"  Processed {i:,}/{len(sequences):,} ({i/len(sequences)*100:.1f}%) - ETA: {eta:.1f} min")
        
        annotation = annotator.annotate_sequence(sequence)
        annotations.append(annotation)
    
    # Create annotated dataframe
    annotation_df = pd.DataFrame(annotations)
    
    # Combine BED data with annotations
    annotated_df = pd.concat([bed_df.reset_index(drop=True), 
                             annotation_df.reset_index(drop=True)], axis=1)
    
    # Add the actual sequence
    annotated_df['sequence'] = sequences
    
    # Save annotated dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    annotated_df.to_csv(output_path, sep='\t', index=False)
    
    # Summary statistics
    print(f"\nAnnotation Summary:")
    print(f"  Total sequences: {len(annotated_df):,}")
    print(f"  ψ range: {annotation_df['psi_predicted'].min():.3f} to {annotation_df['psi_predicted'].max():.3f} kT/e")
    print(f"  GC range: {annotation_df['GC_frac'].min():.3f} to {annotation_df['GC_frac'].max():.3f}")
    print(f"  CpG range: {annotation_df['CpG_density'].min():.3f} to {annotation_df['CpG_density'].max():.3f}")
    print(f"  Saved to: {output_path}")
    
    return True

def check_deepstarr_data(deepstarr_path: str) -> List[str]:
    """Check what DeepSTARR data files are available."""
    
    if not os.path.exists(deepstarr_path):
        return []
    
    files = []
    for root, dirs, filenames in os.walk(deepstarr_path):
        for filename in filenames:
            if filename.endswith(('.h5', '.tsv', '.txt', '.bed', '.fa', '.fasta')):
                files.append(os.path.join(root, filename))
    
    return files

def main():
    """Main function to annotate all datasets."""
    
    print("Electrostatic Potential Dataset Annotation")
    print("="*60)
    
    # Initialize annotator
    annotator = ElectrostaticPotentialAnnotator()
    
    # Define paths
    data_dir = "../data"
    bed_dir = os.path.join(data_dir, "bed_files")
    deepstarr_dir = os.path.join(data_dir, "DeepSTARR")
    output_dir = "annotated_datasets"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Annotate BED datasets
    bed_datasets = [
        ("train_data.bed", "train_sequences.txt", "train_data_psi_annotated.tsv"),
        ("test_data.bed", "test_sequences.txt", "test_data_psi_annotated.tsv"),
        ("valid_data.bed", "valid_sequences.txt", "valid_data_psi_annotated.tsv")
    ]
    
    successful_annotations = 0
    total_sequences = 0
    
    for bed_file, seq_file, output_file in bed_datasets:
        bed_path = os.path.join(bed_dir, bed_file)
        seq_path = os.path.join(bed_dir, seq_file)
        output_path = os.path.join(output_dir, output_file)
        
        if os.path.exists(bed_path) and os.path.exists(seq_path):
            success = annotate_bed_dataset(bed_path, seq_path, output_path, annotator)
            if success:
                successful_annotations += 1
                # Count sequences
                with open(seq_path, 'r') as f:
                    total_sequences += sum(1 for line in f if line.strip())
        else:
            print(f"Missing files for {bed_file}: BED={os.path.exists(bed_path)}, SEQ={os.path.exists(seq_path)}")
    
    # Check DeepSTARR data
    print(f"\nChecking DeepSTARR data...")
    deepstarr_files = check_deepstarr_data(deepstarr_dir)
    if deepstarr_files:
        print(f"Found {len(deepstarr_files)} DeepSTARR files:")
        for f in deepstarr_files[:10]:  # Show first 10
            print(f"  {f}")
        if len(deepstarr_files) > 10:
            print(f"  ... and {len(deepstarr_files)-10} more")
    else:
        print("No DeepSTARR data files found")
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"ANNOTATION COMPLETE")
    print(f"="*60)
    print(f"Successfully annotated: {successful_annotations}/3 BED datasets")
    print(f"Total sequences annotated: {total_sequences:,}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"Model used: ψ = {annotator.psi_intercept:.3f} + {annotator.psi_slope:.3f} × GC_frac")
    print(f"Model performance: R² = {annotator.model_r2:.3f}, MAE = {annotator.model_mae:.3f} kT/e")
    
    print(f"\nAnnotated files contain:")
    print(f"  - Original BED coordinates (chr, start, end, etc.)")
    print(f"  - DNA sequence")
    print(f"  - psi_predicted: Electrostatic potential (kT/e)")
    print(f"  - GC_frac: GC content fraction")
    print(f"  - CpG_density: CpG dinucleotide density")
    print(f"  - run_frac: Longest homopolymer run fraction")
    print(f"  - seq_length: Sequence length")

if __name__ == "__main__":
    main() 