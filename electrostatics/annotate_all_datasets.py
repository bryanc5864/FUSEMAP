#!/usr/bin/env python3
"""
Annotate all datasets with electrostatic potential predictions.
Processes 6 files separately: 3 DeepSTARR + 3 BED datasets.
No merging - keeps original file structures.
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

def extract_sequence_from_bed_name(bed_file: str) -> List[str]:
    """Extract sequences from BED file names when they contain sequence info."""
    sequences = []
    
    try:
        df = pd.read_csv(bed_file, sep='\t', header=None)
        
        # Check if any column contains sequence-like data
        for col in df.columns:
            sample_values = df[col].dropna().head(10)
            # Check if values look like DNA sequences
            for val in sample_values:
                if isinstance(val, str) and len(val) > 50 and all(c in 'ATCGN' for c in val.upper()):
                    print(f"Found sequences in column {col}")
                    return df[col].dropna().tolist()
        
        print("No sequences found in BED file columns")
        return []
        
    except Exception as e:
        print(f"Error extracting sequences from BED: {e}")
        return []

def annotate_deepstarr_bed(bed_path: str, output_path: str, 
                          annotator: ElectrostaticPotentialAnnotator) -> bool:
    """Annotate a DeepSTARR BED file that may contain sequences."""
    
    print(f"\nAnnotating DeepSTARR BED: {os.path.basename(bed_path)}")
    print("="*50)
    
    try:
        # Load BED file
        df = pd.read_csv(bed_path, sep='\t', header=None)
        print(f"Loaded {len(df)} entries with {df.shape[1]} columns")
        
        # Try to find sequence column
        sequences = []
        seq_col = None
        
        for col in df.columns:
            sample_vals = df[col].dropna().head(5)
            for val in sample_vals:
                if isinstance(val, str) and len(val) > 100 and all(c in 'ATCGN' for c in val.upper()):
                    sequences = df[col].tolist()
                    seq_col = col
                    print(f"Found sequences in column {col}")
                    break
            if sequences:
                break
        
        if not sequences:
            print("No sequences found in BED file - cannot annotate")
            return False
        
        # Annotate sequences
        print("Computing electrostatic potential annotations...")
        annotations = []
        
        start_time = time.time()
        for i, sequence in enumerate(sequences):
            if pd.isna(sequence):
                # Handle missing sequences
                annotations.append({
                    'psi_predicted': np.nan,
                    'GC_frac': np.nan,
                    'CpG_density': np.nan,
                    'run_frac': np.nan,
                    'seq_length': np.nan
                })
            else:
                if i % 10000 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    eta = (len(sequences) - i) / rate / 60
                    print(f"  Processed {i:,}/{len(sequences):,} ({i/len(sequences)*100:.1f}%) - ETA: {eta:.1f} min")
                
                annotation = annotator.annotate_sequence(str(sequence))
                annotations.append(annotation)
        
        # Add annotations as new columns
        annotation_df = pd.DataFrame(annotations)
        
        # Assign standard BED column names for first few columns
        bed_cols = ['chr', 'start', 'end', 'name', 'score', 'strand']
        orig_cols = []
        for i in range(min(len(bed_cols), df.shape[1])):
            orig_cols.append(bed_cols[i])
        
        # Add remaining columns as extra
        for i in range(len(orig_cols), df.shape[1]):
            orig_cols.append(f'col_{i}')
        
        df.columns = orig_cols
        
        # Combine original data with annotations
        annotated_df = pd.concat([df, annotation_df], axis=1)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        annotated_df.to_csv(output_path, sep='\t', index=False)
        
        # Summary
        valid_psi = annotation_df['psi_predicted'].dropna()
        print(f"\nAnnotation Summary:")
        print(f"  Total entries: {len(annotated_df):,}")
        print(f"  Valid annotations: {len(valid_psi):,}")
        if len(valid_psi) > 0:
            print(f"  ψ range: {valid_psi.min():.3f} to {valid_psi.max():.3f} kT/e")
            print(f"  GC range: {annotation_df['GC_frac'].min():.3f} to {annotation_df['GC_frac'].max():.3f}")
        print(f"  Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {bed_path}: {e}")
        return False

def annotate_bed_with_sequences(bed_path: str, seq_path: str, output_path: str,
                               annotator: ElectrostaticPotentialAnnotator) -> bool:
    """Annotate BED file using separate sequence file."""
    
    print(f"\nAnnotating BED + sequences: {os.path.basename(bed_path)}")
    print("="*50)
    
    try:
        # Load BED file
        bed_df = pd.read_csv(bed_path, sep='\t', header=None)
        
        # Load sequences
        seq_df = pd.read_csv(seq_path, sep='\t')
        sequences = seq_df['sequence'].tolist()
        
        print(f"Loaded {len(bed_df)} BED entries and {len(sequences)} sequences")
        
        if len(bed_df) != len(sequences):
            print(f"WARNING: Mismatch in counts - using min length")
            min_len = min(len(bed_df), len(sequences))
            bed_df = bed_df.iloc[:min_len]
            sequences = sequences[:min_len]
        
        # Annotate sequences
        print("Computing electrostatic potential annotations...")
        annotations = []
        
        start_time = time.time()
        for i, sequence in enumerate(sequences):
            if i % 10000 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(sequences) - i) / rate / 60
                print(f"  Processed {i:,}/{len(sequences):,} ({i/len(sequences)*100:.1f}%) - ETA: {eta:.1f} min")
            
            annotation = annotator.annotate_sequence(str(sequence))
            annotations.append(annotation)
        
        # Assign BED column names
        bed_cols = ['chr', 'start', 'end', 'name', 'score', 'strand']
        orig_cols = []
        for i in range(min(len(bed_cols), bed_df.shape[1])):
            orig_cols.append(bed_cols[i])
        for i in range(len(orig_cols), bed_df.shape[1]):
            orig_cols.append(f'col_{i}')
        
        bed_df.columns = orig_cols
        
        # Add annotations and sequences
        annotation_df = pd.DataFrame(annotations)
        bed_df['sequence'] = sequences
        
        # Combine all data
        annotated_df = pd.concat([bed_df, annotation_df], axis=1)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        annotated_df.to_csv(output_path, sep='\t', index=False)
        
        # Summary
        print(f"\nAnnotation Summary:")
        print(f"  Total entries: {len(annotated_df):,}")
        print(f"  ψ range: {annotation_df['psi_predicted'].min():.3f} to {annotation_df['psi_predicted'].max():.3f} kT/e")
        print(f"  GC range: {annotation_df['GC_frac'].min():.3f} to {annotation_df['GC_frac'].max():.3f}")
        print(f"  Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {bed_path} + {seq_path}: {e}")
        return False

def main():
    """Main function to annotate all 6 datasets separately."""
    
    print("Electrostatic Potential Dataset Annotation - All 6 Files")
    print("="*60)
    
    # Initialize annotator
    annotator = ElectrostaticPotentialAnnotator()
    
    # Define paths
    output_dir = "annotated_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Track results
    results = []
    
    # 1. DeepSTARR comprehensive BED files (3 files)
    deepstarr_files = [
        ("../processed_data/DeepSTARR_data/DeepSTARR_train_comprehensive.bed", 
         "DeepSTARR_train_comprehensive_psi_annotated.bed"),
        ("../processed_data/DeepSTARR_data/DeepSTARR_test_comprehensive.bed", 
         "DeepSTARR_test_comprehensive_psi_annotated.bed"),
        ("../processed_data/DeepSTARR_data/DeepSTARR_val_comprehensive.bed", 
         "DeepSTARR_val_comprehensive_psi_annotated.bed")
    ]
    
    for bed_path, output_file in deepstarr_files:
        output_path = os.path.join(output_dir, output_file)
        if os.path.exists(bed_path):
            success = annotate_deepstarr_bed(bed_path, output_path, annotator)
            results.append((output_file, success))
        else:
            print(f"Missing DeepSTARR file: {bed_path}")
            results.append((output_file, False))
    
    # 2. BED files with separate sequence files (3 files)
    bed_seq_files = [
        ("../data/bed_files/train_data.bed", "../data/bed_files/train_sequences.txt", 
         "train_data_with_sequences_psi_annotated.tsv"),
        ("../data/bed_files/test_data.bed", "../data/bed_files/test_sequences.txt", 
         "test_data_with_sequences_psi_annotated.tsv"),
        ("../data/bed_files/valid_data.bed", "../data/bed_files/valid_sequences.txt", 
         "valid_data_with_sequences_psi_annotated.tsv")
    ]
    
    for bed_path, seq_path, output_file in bed_seq_files:
        output_path = os.path.join(output_dir, output_file)
        if os.path.exists(bed_path) and os.path.exists(seq_path):
            success = annotate_bed_with_sequences(bed_path, seq_path, output_path, annotator)
            results.append((output_file, success))
        else:
            print(f"Missing files: BED={os.path.exists(bed_path)}, SEQ={os.path.exists(seq_path)}")
            results.append((output_file, False))
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"ANNOTATION COMPLETE - 6 SEPARATE FILES")
    print(f"="*60)
    
    successful = sum(1 for _, success in results if success)
    print(f"Successfully annotated: {successful}/6 files")
    
    print(f"\nResults:")
    for filename, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {filename}")
    
    print(f"\nOutput directory: {os.path.abspath(output_dir)}")
    print(f"Model: ψ = {annotator.psi_intercept:.3f} + {annotator.psi_slope:.3f} × GC_frac")
    print(f"Performance: R² = {annotator.model_r2:.3f}, MAE = {annotator.model_mae:.3f} kT/e")
    
    print(f"\nEach annotated file contains:")
    print(f"  - Original data structure (BED format preserved)")
    print(f"  - psi_predicted: Electrostatic potential (kT/e)")
    print(f"  - GC_frac: GC content fraction")
    print(f"  - CpG_density: CpG dinucleotide density")
    print(f"  - run_frac: Longest homopolymer run fraction")
    print(f"  - seq_length: Sequence length")

if __name__ == "__main__":
    main() 