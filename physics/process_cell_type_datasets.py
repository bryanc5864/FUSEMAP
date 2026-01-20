#!/usr/bin/env python3
"""
GPU-Optimized Cell-Type Specific Dataset Processing Script

Processes all cell-type specific data splits (HepG2, K562, WTC11) with:
- Full GPU acceleration where available
- Cell-type specific PWM scanning
- Batch processing for memory efficiency
- Progress tracking and error handling
- Optimized for large datasets (248k+ sequences)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import argparse
from datetime import datetime
import gc
import json
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from process_pwm_descriptors import (
    PWMProcessor, BendingEnergyProcessor, StiffnessProcessor,
    ThermodynamicProcessor, EntropyProcessor, AdvancedBiophysicsProcessor
)
from cell_type_pwms import (
    identify_cell_type, get_jaspar_file, get_sequence_length
)

class GPUOptimizedProcessor:
    """GPU-optimized processor for cell-type specific datasets."""
    
    def __init__(self, cell_type: str, gpu_device: int = 1):
        """Initialize processors with GPU optimization."""
        self.cell_type = cell_type
        self.gpu_device = gpu_device

        print(f"Initializing GPU-optimized processors for {cell_type}...")
        start_time = time.time()

        # File paths (cell-type specific JASPAR file)
        self.jaspar_file = get_jaspar_file(cell_type)
        self.sequence_length = get_sequence_length(cell_type)
        self.dna_props_file = 'data/DNAProperties.txt'
        self.olson_file = 'data/OlsonMatrix.tsv'
        self.santalucia_file = 'data/SantaLuciaNN.tsv'

        print(f"  Cell type: {cell_type}")
        print(f"  JASPAR file: {self.jaspar_file}")
        print(f"  Expected sequence length: {self.sequence_length} bp")

        # Initialize all processors with cell-type specificity
        self.processors = self._initialize_processors()

        init_time = time.time() - start_time
        print(f"✓ Processors initialized in {init_time:.2f}s")
        
    def _initialize_processors(self):
        """Initialize all feature processors."""
        processors = {}
        
        # PWM processor with cell-type specific motifs
        processors['pwm'] = PWMProcessor(
            self.jaspar_file,
            cell_type=self.cell_type,
            use_cell_type_pwms=True,
            kT=0.593
        )
        
        # Other processors
        processors['thermo'] = ThermodynamicProcessor(self.santalucia_file)
        processors['entropy'] = EntropyProcessor()
        processors['advanced'] = AdvancedBiophysicsProcessor(self.santalucia_file)
        processors['bend'] = BendingEnergyProcessor(self.dna_props_file, kappa0=1.0, kBT=0.593)
        processors['stiff'] = StiffnessProcessor(self.olson_file, self.dna_props_file)
        
        return processors
    
    def process_sequence_batch(self, sequences: list, batch_id: int = 0) -> list:
        """Process a batch of sequences with GPU optimization."""
        batch_features = []
        
        for seq_idx, sequence in enumerate(sequences):
            if seq_idx % 100 == 0 and seq_idx > 0:
                print(f"    Batch {batch_id}: {seq_idx}/{len(sequences)} sequences processed")
            
            seq_features = {}
            
            # Process with each feature processor
            for prefix, processor in self.processors.items():
                try:
                    proc_features = processor.process_sequence(sequence)
                    for k, v in proc_features.items():
                        seq_features[f'{prefix}_{k}'] = v
                except Exception as e:
                    print(f"Warning: Error processing {prefix} for sequence {seq_idx}: {e}")
                    continue
            
            batch_features.append(seq_features)
        
        return batch_features
    
    def process_dataset_file(self, input_file: str, output_file: str,
                           batch_size: int = 500, save_vectors: bool = False):
        """Process a complete dataset file with batching."""

        print(f"\n{'='*60}")
        print(f"Processing {input_file}")
        print(f"Output: {output_file}")
        print(f"Cell type: {self.cell_type}")
        print(f"{'='*60}")

        # Load dataset (different format for DREAM vs human cell types)
        if self.cell_type == 'DREAM':
            # DREAM format: tab-separated, no headers, columns: [sequence, expression]
            df = pd.read_csv(input_file, sep='\t', header=None, names=['sequence', 'expression'])
            # Add synthetic metadata columns for compatibility
            df['seq_id'] = [f'DREAM_{i}' for i in range(len(df))]
            df['condition'] = 'DREAM_yeast'
            df['normalized_log2'] = df['expression']
            df['n_obs_bc'] = 0
            df['n_replicates'] = 0
        else:
            # Human cell types format: has condition, name, normalized_log2, sequence columns
            df = pd.read_csv(input_file, sep='\t')
            # Validate data structure
            required_cols = ['condition', 'name', 'normalized_log2', 'sequence']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            df['seq_id'] = df['name']

        total_sequences = len(df)
        print(f"Loaded {total_sequences:,} sequences")

        # Check sequence lengths
        seq_lengths = df['sequence'].str.len()
        print(f"Sequence lengths: min={seq_lengths.min()}, max={seq_lengths.max()}, mean={seq_lengths.mean():.1f}")

        # Validate sequence length matches expected
        if seq_lengths.mean() < self.sequence_length * 0.9 or seq_lengths.mean() > self.sequence_length * 1.1:
            print(f"WARNING: Average sequence length ({seq_lengths.mean():.1f}) differs from expected ({self.sequence_length})")
        
        # Process in batches
        all_descriptors = []
        num_batches = (total_sequences + batch_size - 1) // batch_size
        
        print(f"Processing in {num_batches} batches of {batch_size} sequences...")
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, total_sequences)
            batch_df = df.iloc[batch_start:batch_end]
            
            print(f"\nBatch {batch_idx + 1}/{num_batches}: sequences {batch_start:,}-{batch_end:,}")
            batch_start_time = time.time()
            
            # Extract sequences for this batch
            batch_sequences = batch_df['sequence'].tolist()
            
            # Process batch
            batch_features = self.process_sequence_batch(batch_sequences, batch_idx + 1)
            
            # Combine with metadata
            for i, (_, row) in enumerate(batch_df.iterrows()):
                feature_dict = {
                    'seq_id': row['seq_id'],
                    'condition': row['condition'],
                    'normalized_log2': row['normalized_log2'],
                    'n_obs_bc': row.get('n_obs_bc', 0),
                    'n_replicates': row.get('n_replicates', 0),
                    'sequence': row['sequence']
                }

                # Add computed features
                if i < len(batch_features):
                    feature_dict.update(batch_features[i])

                all_descriptors.append(feature_dict)
            
            # Batch timing and progress
            batch_time = time.time() - batch_start_time
            sequences_per_sec = len(batch_sequences) / batch_time
            
            # Estimate remaining time
            elapsed_total = time.time() - start_time
            completed_sequences = batch_end
            remaining_sequences = total_sequences - completed_sequences
            eta_seconds = remaining_sequences / (completed_sequences / elapsed_total) if completed_sequences > 0 else 0
            eta_hours = eta_seconds / 3600
            
            print(f"    Batch completed in {batch_time:.1f}s ({sequences_per_sec:.1f} seq/s)")
            print(f"    Progress: {completed_sequences:,}/{total_sequences:,} ({100*completed_sequences/total_sequences:.1f}%)")
            print(f"    ETA: {eta_hours:.1f} hours")
            
            # Memory cleanup
            del batch_features, batch_sequences, batch_df
            gc.collect()
        
        # Create output DataFrame
        print(f"\nCreating output DataFrame...")
        output_df = pd.DataFrame(all_descriptors)
        
        # Feature summary
        feature_cols = [c for c in output_df.columns if c not in ['seq_id', 'condition', 'normalized_log2', 'n_obs_bc', 'n_replicates', 'sequence']]
        print(f"Generated {len(feature_cols)} features")
        
        # Save results
        print(f"Saving to {output_file}...")
        output_df.to_csv(output_file, sep='\t', index=False)
        
        # Processing summary
        total_time = time.time() - start_time
        avg_seq_per_sec = total_sequences / total_time
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Average speed: {avg_seq_per_sec:.1f} sequences/second")
        print(f"Sequences processed: {total_sequences:,}")
        print(f"Features generated: {len(feature_cols)}")
        print(f"Output file: {output_file}")
        print(f"File size: {Path(output_file).stat().st_size / 1024**2:.1f} MB")
        
        return output_df

def process_all_cell_types(batch_size: int = 500, output_dir: str = 'output'):
    """Process all cell types and data splits."""
    
    print("="*80)
    print("GPU-OPTIMIZED CELL-TYPE SPECIFIC DATASET PROCESSING")
    print("="*80)
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Define all datasets to process
    datasets = [
        ('HepG2', 'data/HepG2_data/splits/train.tsv'),
        ('HepG2', 'data/HepG2_data/splits/val.tsv'),
        ('HepG2', 'data/HepG2_data/splits/test.tsv'),
        ('K562', 'data/K562_data/splits/train.tsv'),
        ('K562', 'data/K562_data/splits/val.tsv'),
        ('K562', 'data/K562_data/splits/test.tsv'),
        ('WTC11', 'data/WTC11_data/splits/train.tsv'),
        ('WTC11', 'data/WTC11_data/splits/val.tsv'),
        ('WTC11', 'data/WTC11_data/splits/test.tsv'),
        ('DREAM', 'data/DREAM_data/splits/yeast_train.txt'),
        ('DREAM', 'data/DREAM_data/splits/yeast_val.txt'),
        ('DREAM', 'data/DREAM_data/splits/yeast_test.txt')
    ]
    
    # Process each dataset
    total_start_time = time.time()
    processing_results = {}
    
    for cell_type, input_file in datasets:
        if not Path(input_file).exists():
            print(f"Warning: {input_file} not found, skipping...")
            continue
        
        # Create output filename
        split_name = Path(input_file).stem  # train, val, or test
        output_file = output_path / f'{cell_type}_{split_name}_descriptors.tsv'
        
        # Initialize processor for this cell type
        processor = GPUOptimizedProcessor(cell_type)
        
        # Process dataset
        try:
            result_df = processor.process_dataset_file(
                str(input_file), 
                str(output_file), 
                batch_size=batch_size
            )
            
            # Store results
            processing_results[f'{cell_type}_{split_name}'] = {
                'input_file': input_file,
                'output_file': str(output_file),
                'n_sequences': len(result_df),
                'n_features': len([c for c in result_df.columns if c.startswith(('pwm_', 'thermo_', 'entropy_', 'advanced_', 'bend_', 'stiff_'))]),
                'cell_type': cell_type,
                'split': split_name
            }
            
        except Exception as e:
            print(f"Error processing {cell_type} {split_name}: {e}")
            processing_results[f'{cell_type}_{split_name}'] = {'error': str(e)}
        
        # Cleanup
        del processor
        gc.collect()
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("FINAL PROCESSING SUMMARY")
    print("="*80)
    print(f"Total processing time: {total_time/3600:.2f} hours")
    
    # Save processing results
    results_file = output_path / 'processing_summary.json'
    with open(results_file, 'w') as f:
        json.dump(processing_results, f, indent=2, default=str)
    
    # Print summary table
    print(f"\nDataset Processing Results:")
    print("-" * 80)
    print(f"{'Dataset':<15} {'Split':<8} {'Sequences':<12} {'Features':<10} {'Status':<10}")
    print("-" * 80)
    
    total_sequences = 0
    successful_datasets = 0
    
    for dataset_name, result in processing_results.items():
        cell_type, split = dataset_name.split('_', 1)
        
        if 'error' in result:
            print(f"{cell_type:<15} {split:<8} {'ERROR':<12} {'ERROR':<10} {'FAILED':<10}")
        else:
            n_seq = result['n_sequences']
            n_feat = result['n_features']
            total_sequences += n_seq
            successful_datasets += 1
            print(f"{cell_type:<15} {split:<8} {n_seq:<12,} {n_feat:<10} {'SUCCESS':<10}")
    
    print("-" * 80)
    print(f"Total successful datasets: {successful_datasets}/12")
    print(f"Total sequences processed: {total_sequences:,}")
    print(f"Results saved to: {results_file}")
    
    return processing_results

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Process cell-type specific datasets with GPU optimization')
    parser.add_argument('--batch_size', type=int, default=500,
                       help='Batch size for processing (default: 500)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for processed files')
    parser.add_argument('--cell_type', type=str, choices=['HepG2', 'K562', 'WTC11', 'DREAM', 'all'], default='all',
                       help='Specific cell type to process (default: all)')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='all',
                       help='Specific split to process (default: all)')
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Cell type: {args.cell_type}")
    print(f"  Split: {args.split}")
    
    if args.cell_type == 'all' and args.split == 'all':
        # Process everything
        process_all_cell_types(batch_size=args.batch_size, output_dir=args.output_dir)
    else:
        # Process specific dataset(s)
        cell_types = [args.cell_type] if args.cell_type != 'all' else ['HepG2', 'K562', 'WTC11', 'DREAM']
        splits = [args.split] if args.split != 'all' else ['train', 'val', 'test']

        for cell_type in cell_types:
            processor = GPUOptimizedProcessor(cell_type)

            for split in splits:
                # Handle different file paths for DREAM vs human cell types
                if cell_type == 'DREAM':
                    input_file = f'data/DREAM_data/splits/yeast_{split}.txt'
                else:
                    input_file = f'data/{cell_type}_data/splits/{split}.tsv'

                output_file = f'{args.output_dir}/{cell_type}_{split}_descriptors.tsv'

                if Path(input_file).exists():
                    processor.process_dataset_file(input_file, output_file, batch_size=args.batch_size)
                else:
                    print(f"Warning: {input_file} not found")

if __name__ == "__main__":
    main()