#!/usr/bin/env python3
"""
Recover ABPS results from existing .dx files in worker directories.
"""

import pandas as pd
import logging
import os
import sys
import time
import numpy as np
import shutil
from pathlib import Path
from electrostatics.tleap_abps_processor import DualConfigABPSProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/recover_abps_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_psi_from_dx_file(dx_file):
    """Extract psi values from a .dx file using the processor."""
    try:
        # Create a temporary processor to use its extraction method
        temp_processor = DualConfigABPSProcessor(work_dir="/tmp/temp_recovery")
        dx_path = Path(dx_file)
        
        # Use the private method to extract psi
        psi_data = temp_processor._extract_psi_from_dx(dx_path.parent)
        
        if psi_data is not None:
            return {
                'min': psi_data['min'],
                'max': psi_data['max'], 
                'mean': psi_data['mean']
            }
    except Exception as e:
        logger.warning(f"Failed to extract psi from {dx_file}: {e}")
    return None

def recover_results_from_workers():
    """Recover results from existing worker directories."""
    logger.info("🔍 Recovering ABPS results from worker directories...")
    
    results = []
    worker_dirs = list(Path("/tmp").glob("fast_abps_worker_*"))
    
    logger.info(f"Found {len(worker_dirs)} worker directories")
    
    for worker_dir in worker_dirs:
        if not worker_dir.is_dir():
            continue
            
        # Look for sequence directories
        for seq_dir in worker_dir.iterdir():
            if not seq_dir.is_dir() or not seq_dir.name.startswith("tile_"):
                continue
                
            seq_id = seq_dir.name.replace("tile_", "")
            
            # Check for psi.txt file first (already extracted)
            psi_file = seq_dir / "psi.txt"
            if psi_file.exists():
                try:
                    with open(psi_file, 'r') as f:
                        lines = f.readlines()
                    
                    values = {}
                    for line in lines:
                        if ':' in line:
                            key, value = line.strip().split(':')
                            try:
                                values[key.strip()] = float(value.strip())
                            except:
                                continue
                    
                    if 'min' in values and 'max' in values and 'mean' in values:
                        results.append({
                            'seq_id': seq_id,
                            'std_psi_min': values['min'],
                            'std_psi_max': values['max'],
                            'std_psi_mean': values['mean'],
                            'enh_psi_min': None,  # Single config only
                            'enh_psi_max': None,
                            'enh_psi_mean': None,
                            'success': True,
                            'config_type': 'single'
                        })
                        logger.info(f"✅ Recovered single config results for {seq_id}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse psi.txt for {seq_id}: {e}")
            
            # Check for dual config (.dx files)
            std_dx = seq_dir / "pot_apbs_std.dx"
            enh_dx = seq_dir / "pot_apbs_enh.dx"
            
            if std_dx.exists() and enh_dx.exists():
                # Extract from both .dx files
                std_psi = extract_psi_from_dx_file(std_dx)
                enh_psi = extract_psi_from_dx_file(enh_dx)
                
                if std_psi and enh_psi:
                    results.append({
                        'seq_id': seq_id,
                        'std_psi_min': std_psi['min'],
                        'std_psi_max': std_psi['max'],
                        'std_psi_mean': std_psi['mean'],
                        'enh_psi_min': enh_psi['min'],
                        'enh_psi_max': enh_psi['max'],
                        'enh_psi_mean': enh_psi['mean'],
                        'success': True,
                        'config_type': 'dual'
                    })
                    logger.info(f"✅ Recovered dual config results for {seq_id}")
                elif std_psi:
                    results.append({
                        'seq_id': seq_id,
                        'std_psi_min': std_psi['min'],
                        'std_psi_max': std_psi['max'],
                        'std_psi_mean': std_psi['mean'],
                        'enh_psi_min': None,
                        'enh_psi_max': None,
                        'enh_psi_mean': None,
                        'success': True,
                        'config_type': 'std_only'
                    })
                    logger.info(f"✅ Recovered std-only results for {seq_id}")
                else:
                    logger.warning(f"⚠️ Failed to extract psi from .dx files for {seq_id}")
            elif std_dx.exists():
                # Only standard config
                std_psi = extract_psi_from_dx_file(std_dx)
                if std_psi:
                    results.append({
                        'seq_id': seq_id,
                        'std_psi_min': std_psi['min'],
                        'std_psi_max': std_psi['max'],
                        'std_psi_mean': std_psi['mean'],
                        'enh_psi_min': None,
                        'enh_psi_max': None,
                        'enh_psi_mean': None,
                        'success': True,
                        'config_type': 'std_only'
                    })
                    logger.info(f"✅ Recovered std-only results for {seq_id}")
    
    logger.info(f"📊 Recovered {len(results)} results")
    return results

def get_remaining_sequences(recovered_results, corpus_file):
    """Get list of sequences that still need processing."""
    logger.info("📋 Determining remaining sequences to process...")
    
    # Load original corpus
    df = pd.read_csv(corpus_file, sep='\t')
    
    # Get processed sequence IDs
    processed_ids = set(result['seq_id'] for result in recovered_results)
    
    # Find remaining sequences
    remaining_sequences = []
    for _, row in df.iterrows():
        seq_id = str(row.get('seq_id', row.name))
        if seq_id not in processed_ids:
            remaining_sequences.append((seq_id, row['sequence']))
    
    logger.info(f"📊 Total sequences: {len(df)}")
    logger.info(f"📊 Already processed: {len(processed_ids)}")
    logger.info(f"📊 Remaining to process: {len(remaining_sequences)}")
    
    return remaining_sequences

def main():
    """Main function to recover and resume ABPS processing."""
    logger.info("🔄 Starting ABPS results recovery...")
    
    # Input/output files
    corpus_file = "data/corpus_50k_complete.tsv"
    output_file = "/shared/data06/TileFormer_ABPS/corpus_50k_with_abps_optimized.tsv.gz"
    
    if not Path(corpus_file).exists():
        logger.error(f"❌ Corpus file not found: {corpus_file}")
        return False
    
    # Recover existing results
    recovered_results = recover_results_from_workers()
    
    if not recovered_results:
        logger.warning("⚠️ No existing results found. Starting fresh processing...")
        # Run the original script
        os.system("python run_fast_abps.py")
        return True
    
    # Get remaining sequences
    remaining_sequences = get_remaining_sequences(recovered_results, corpus_file)
    
    if not remaining_sequences:
        logger.info("✅ All sequences already processed!")
        # Just save the recovered results
        df_results = pd.DataFrame(recovered_results)
        df_corpus = pd.read_csv(corpus_file, sep='\t')
        df_merged = df_corpus.merge(df_results, on='seq_id', how='left')
        
        logger.info(f"💾 Saving recovered results to {output_file}")
        df_merged.to_csv(output_file, sep='\t', index=False, compression='gzip')
        
        successful = len(df_results[df_results['success'] == True])
        total = len(df_results)
        
        logger.info(f"✅ Recovery completed!")
        logger.info(f"   Total sequences: {total}")
        logger.info(f"   Successful: {successful} ({successful/total*100:.1f}%)")
        logger.info(f"   Output: {output_file}")
        return True
    
    # Process remaining sequences
    logger.info(f"🚀 Processing {len(remaining_sequences)} remaining sequences...")
    
    # For now, just run the original script on remaining sequences
    # Create a temporary corpus with only remaining sequences
    temp_corpus = "data/corpus_remaining.tsv"
    remaining_df = pd.DataFrame([
        {'seq_id': seq_id, 'sequence': sequence} 
        for seq_id, sequence in remaining_sequences
    ])
    remaining_df.to_csv(temp_corpus, sep='\t', index=False)
    
    # Run processing on remaining sequences
    os.system(f"python run_fast_abps.py --input {temp_corpus} --output /tmp/remaining_results.tsv.gz")
    
    # Load remaining results and combine
    if Path("/tmp/remaining_results.tsv.gz").exists():
        remaining_results_df = pd.read_csv("/tmp/remaining_results.tsv.gz", sep='\t')
        
        # Combine all results
        all_results = recovered_results + remaining_results_df.to_dict('records')
        
        # Create final dataframe
        df_results = pd.DataFrame(all_results)
        df_corpus = pd.read_csv(corpus_file, sep='\t')
        df_merged = df_corpus.merge(df_results, on='seq_id', how='left')
        
        # Save results
        logger.info(f"💾 Saving combined results to {output_file}")
        df_merged.to_csv(output_file, sep='\t', index=False, compression='gzip')
        
        # Cleanup
        os.remove(temp_corpus)
        os.remove("/tmp/remaining_results.tsv.gz")
        
        successful = len(df_results[df_results['success'] == True])
        total = len(df_results)
        
        logger.info(f"✅ Processing completed!")
        logger.info(f"   Total sequences: {total}")
        logger.info(f"   Successful: {successful} ({successful/total*100:.1f}%)")
        logger.info(f"   Output: {output_file}")
        
        return True
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
