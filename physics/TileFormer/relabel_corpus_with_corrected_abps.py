#!/usr/bin/env python3
"""
Re-label 50k DNA corpus with corrected ABPS electrostatic potential values.

This script uses the fixed ABPS methodology that:
1. Uses sequence-dependent geometry via sander minimization
2. Employs mg-auto focusing with proper grid parameters
3. Extracts potential from 2-6Å solvent shell (not just phosphates)
4. Captures real sequence-dependent electrostatic differences

The corrected methodology fixes the issues identified:
- Canonical geometry → sequence-dependent via minimization
- Phosphate-only extraction → 2-6Å shell averaging
- Wrong grid spacing → 193x193x193 with 0.21Å resolution
- Salt screening effects → proper boundary conditions
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add electrostatics module to path
sys.path.append(str(Path(__file__).parent))

from electrostatics.tleap_abps_processor import TLEaPABPSProcessor, ABPSConfig
from electrostatics.parallel_abps_runner import ParallelABPSRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('relabel_corpus.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CorpusRelabeler:
    """Re-label corpus with corrected ABPS methodology."""
    
    def __init__(
        self,
        input_file: str = "data/corpus_50k_complete.tsv",
        output_file: str = "data/corpus_50k_with_corrected_abps.tsv",
        work_dir: str = "/shared/data06/TileFormer_ABPS_Corrected",
        n_processes: int = None,
        batch_size: int = 1000,
        compression_batch_size: int = 500,
        test_mode: bool = False
    ):
        """
        Initialize corpus relabeler.
        
        Args:
            input_file: Path to input corpus file (without ABPS values)
            output_file: Path to output file (with corrected ABPS values)
            work_dir: Working directory for ABPS calculations (in /shared/data06)
            n_processes: Number of parallel processes
            batch_size: Batch size for parallel processing
            compression_batch_size: Compress results every N sequences (default: 500)
            test_mode: If True, only process first 100 sequences for testing
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.work_dir = Path(work_dir)
        self.n_processes = n_processes or max(1, multiprocessing.cpu_count() - 2)
        self.batch_size = batch_size
        self.compression_batch_size = compression_batch_size
        self.test_mode = test_mode
        
        # Create work directory
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure corrected ABPS methodology
        self.abps_config = ABPSConfig(
            # mg-auto focusing with proper resolution
            dime=(193, 193, 193),           # ≈0.21 Å spacing in inner box
            cglen=(200.0, 200.0, 200.0),   # outer boundary ≥4 Debye lengths
            fglen=(40.0, 40.0, 40.0),      # duplex + 12 Å solvent margin
            
            # Ion settings
            ion_conc=0.150,                # 150 mM salt
            pos_ion_radius=2.0,
            neg_ion_radius=2.0,
            
            # Solvent shell extraction (key fix!)
            shell_inner=2.0,               # 2 Å inner cutoff
            shell_outer=6.0,               # 6 Å outer cutoff
            
            # Enable sequence-dependent geometry
            enable_minimization=True,       # Critical for sequence dependence!
            min_steps=2000,                # 500 SD + 1500 CG steps
            ncyc=500
        )
        
        logger.info("🔧 Configured corrected ABPS methodology:")
        logger.info(f"  Grid: {self.abps_config.dime} (mg-auto focusing)")
        logger.info(f"  Shell extraction: {self.abps_config.shell_inner}-{self.abps_config.shell_outer} Å")
        logger.info(f"  Minimization: {'Enabled' if self.abps_config.enable_minimization else 'Disabled'}")
        logger.info(f"  Parallel processes: {self.n_processes}")
    
    def load_corpus(self) -> pd.DataFrame:
        """Load the corpus file."""
        logger.info(f"📂 Loading corpus from {self.input_file}")
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        df = pd.read_csv(self.input_file, sep='\t')
        logger.info(f"📊 Loaded {len(df)} sequences")
        
        # Validate required columns
        required_cols = ['seq_id', 'sequence']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter to 20bp sequences
        df = df[df['sequence'].str.len() == 20].copy()
        logger.info(f"📊 Filtered to {len(df)} sequences of length 20bp")
        
        # Test mode: only first 100 sequences
        if self.test_mode:
            df = df.head(100).copy()
            logger.info(f"🧪 Test mode: using first {len(df)} sequences")
        
        return df
    
    def validate_corrected_methodology(self) -> bool:
        """
        Quick validation with GC-extreme sequences to ensure sequence dependence.
        """
        logger.info("🧪 Validating corrected methodology with GC-extreme sequences...")
        
        # Test sequences: all-A vs all-G
        test_sequences = {
            "all_A": "A" * 20,
            "all_G": "G" * 20,
            "AT_rich": "ATATATATATATATATATAT",
            "GC_rich": "GCGCGCGCGCGCGCGCGCGC"
        }
        
        processor = TLEaPABPSProcessor(
            work_dir=self.work_dir / "validation",
            config=self.abps_config,
            cleanup=True
        )
        
        results = {}
        for name, seq in test_sequences.items():
            logger.info(f"  Testing {name}: {seq}")
            psi = processor.process_sequence(seq, f"test_{name}")
            results[name] = psi
            logger.info(f"    ψ = {psi:.6f} kT/e" if psi else "    Failed")
        
        # Check for sequence dependence
        valid_results = {k: v for k, v in results.items() if v is not None}
        if len(valid_results) < 2:
            logger.error("❌ Validation failed: too few successful calculations")
            return False
        
        psi_values = list(valid_results.values())
        psi_range = max(psi_values) - min(psi_values)
        
        logger.info(f"🔬 Validation results:")
        for name, psi in valid_results.items():
            logger.info(f"  {name}: {psi:.6f} kT/e")
        logger.info(f"  Range: {psi_range:.6f} kT/e")
        
        # Expect at least 1-3 kT/e difference for sequence dependence
        if psi_range < 0.5:
            logger.warning(f"⚠️ Small range ({psi_range:.6f} kT/e) - may still be sequence-insensitive")
            return False
        else:
            logger.info(f"✅ Good range ({psi_range:.6f} kT/e) - sequence dependence detected!")
            return True
    
    def process_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of sequences."""
        batch_results = batch_df.copy()
        batch_results['electrostatic_potential'] = np.nan
        
        # Use parallel runner for batch processing with compression
        runner = ParallelABPSRunner(
            work_dir=self.work_dir / f"batch_{time.time():.0f}",
            config=self.abps_config,
            n_processes=min(self.n_processes, len(batch_df)),
            cleanup=True,
            batch_compression_size=self.compression_batch_size,
            compression_level=22  # Ultra compression for /shared/data06
        )
        
        sequences = batch_df['sequence'].tolist()
        seq_ids = batch_df['seq_id'].tolist()
        
        logger.info(f"🔬 Processing batch of {len(sequences)} sequences...")
        
        try:
            psi_results = runner.process_batch(sequences, seq_ids)
            
            # Map results back to dataframe
            for i, seq_id in enumerate(seq_ids):
                if seq_id in psi_results and psi_results[seq_id] is not None:
                    batch_results.loc[batch_df.index[i], 'electrostatic_potential'] = psi_results[seq_id]
            
            success_count = batch_results['electrostatic_potential'].notna().sum()
            logger.info(f"✅ Batch completed: {success_count}/{len(batch_df)} successful")
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
        
        return batch_results
    
    def relabel_corpus(self) -> None:
        """Re-label the entire corpus with corrected ABPS values."""
        logger.info("🚀 Starting corpus re-labeling with corrected ABPS methodology...")
        
        # Load corpus
        df = self.load_corpus()
        
        # Validate methodology first
        if not self.validate_corrected_methodology():
            logger.error("❌ Validation failed - methodology may still be sequence-insensitive")
            if not self.test_mode:
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    return
        
        # Initialize results dataframe
        results_df = df.copy()
        results_df['electrostatic_potential'] = np.nan
        
        # Process in batches
        n_batches = (len(df) + self.batch_size - 1) // self.batch_size
        logger.info(f"📦 Processing {len(df)} sequences in {n_batches} batches of {self.batch_size}")
        
        successful = 0
        failed = 0
        
        with tqdm(total=len(df), desc="Relabeling corpus") as pbar:
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(df))
                
                batch_df = df.iloc[start_idx:end_idx].copy()
                logger.info(f"🔄 Processing batch {batch_idx + 1}/{n_batches} (rows {start_idx}-{end_idx})")
                
                batch_results = self.process_batch(batch_df)
                
                # Update results
                results_df.iloc[start_idx:end_idx] = batch_results
                
                # Update statistics
                batch_successful = batch_results['electrostatic_potential'].notna().sum()
                batch_failed = len(batch_results) - batch_successful
                successful += batch_successful
                failed += batch_failed
                
                pbar.update(len(batch_df))
                pbar.set_postfix({
                    'Success': f"{successful}/{successful+failed}",
                    'Rate': f"{successful/(successful+failed)*100:.1f}%"
                })
                
                # Save intermediate results every 5 batches
                if (batch_idx + 1) % 5 == 0:
                    temp_file = self.output_file.with_suffix('.tmp.tsv')
                    results_df.to_csv(temp_file, sep='\t', index=False)
                    logger.info(f"💾 Saved intermediate results to {temp_file}")
        
        # Final statistics
        total_processed = successful + failed
        success_rate = successful / total_processed * 100 if total_processed > 0 else 0
        
        logger.info(f"📊 Final results:")
        logger.info(f"  Total sequences: {total_processed}")
        logger.info(f"  Successful: {successful} ({success_rate:.1f}%)")
        logger.info(f"  Failed: {failed} ({100-success_rate:.1f}%)")
        
        # Analyze results
        valid_psi = results_df['electrostatic_potential'].dropna()
        if len(valid_psi) > 0:
            logger.info(f"  ABPS ψ statistics:")
            logger.info(f"    Mean: {valid_psi.mean():.6f} kT/e")
            logger.info(f"    Std:  {valid_psi.std():.6f} kT/e")
            logger.info(f"    Min:  {valid_psi.min():.6f} kT/e")
            logger.info(f"    Max:  {valid_psi.max():.6f} kT/e")
            logger.info(f"    Range: {valid_psi.max() - valid_psi.min():.6f} kT/e")
        
        # Save final results
        results_df.to_csv(self.output_file, sep='\t', index=False)
        logger.info(f"💾 Saved final results to {self.output_file}")
        
        # Remove temporary file
        temp_file = self.output_file.with_suffix('.tmp.tsv')
        if temp_file.exists():
            temp_file.unlink()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Re-label DNA corpus with corrected ABPS values")
    
    parser.add_argument(
        '--input', '-i',
        default='data/corpus_50k_complete.tsv',
        help='Input corpus file (default: data/corpus_50k_complete.tsv)'
    )
    parser.add_argument(
        '--output', '-o', 
        default='data/corpus_50k_with_corrected_abps.tsv',
        help='Output file with corrected ABPS values (default: data/corpus_50k_with_corrected_abps.tsv)'
    )
    parser.add_argument(
        '--work-dir', '-w',
        default='/shared/data06/TileFormer_ABPS_Corrected',
        help='Working directory for ABPS calculations'
    )
    parser.add_argument(
        '--processes', '-p',
        type=int,
        default=None,
        help='Number of parallel processes (default: CPU count - 2)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1000,
        help='Batch size for processing (default: 1000)'
    )
    parser.add_argument(
        '--compression-batch-size', '-c',
        type=int,
        default=500,
        help='Compress results every N sequences to save disk space (default: 500)'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test mode: only process first 100 sequences'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🔬 TileFormer Corpus Re-labeling with Corrected ABPS Methodology")
    logger.info("="*80)
    logger.info("This addresses the sequence-insensitive ABPS issues:")
    logger.info("  ✅ Sequence-dependent geometry via sander minimization")
    logger.info("  ✅ 2-6Å solvent shell extraction (not phosphate-only)")
    logger.info("  ✅ mg-auto focusing with proper grid resolution")
    logger.info("  ✅ Proper boundary conditions to eliminate artifacts")
    logger.info("="*80)
    
    # Create relabeler and run
    relabeler = CorpusRelabeler(
        input_file=args.input,
        output_file=args.output,
        work_dir=args.work_dir,
        n_processes=args.processes,
        batch_size=args.batch_size,
        compression_batch_size=args.compression_batch_size,
        test_mode=args.test
    )
    
    try:
        relabeler.relabel_corpus()
        logger.info("🎉 Corpus re-labeling completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Process interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error during re-labeling: {e}")
        raise

if __name__ == "__main__":
    main()