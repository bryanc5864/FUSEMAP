#!/usr/bin/env python3
"""
Optimized ABPS processor with worker reuse, chunking, and compression.
"""

import pandas as pd
import logging
from pathlib import Path
import sys
import os
from multiprocessing import Pool
import time
import numpy as np
import shutil
from electrostatics.tleap_abps_processor import TLEaPABPSProcessor, ABPSConfig, DualConfigABPSProcessor

# Global processor instance for worker reuse
processor = None

def init_worker():
    """Initialize dual config processor once per worker process for true reuse."""
    global processor
    worker_id = os.getpid()
    work_dir = f"/tmp/fast_abps_worker_{worker_id}"
    
    # Use dual config processor for both standard and enhanced configurations
    processor = DualConfigABPSProcessor(
        work_dir=work_dir,
        cleanup=True  # Clean up each sequence directory after processing
    )
    print(f"🔧 Worker {worker_id} initialized with dual config processor: {work_dir}")

def process_sequence_batch(args_list):
    """Process a batch of sequences using the shared processor (true reuse)."""
    global processor
    worker_id = os.getpid()
    results = []
    
    # Use single shared log file for all workers
    log_file = "logs/abps_continuous_results.txt"
    os.makedirs("logs", exist_ok=True)
    
    for sequence, seq_id in args_list:
        try:
            # Process sequence using dual config processor (gets all 6 values)
            result = processor.process_sequence_dual_config(sequence, seq_id)
            
            if result is not None:
                # Extract all 6 values from dual config result
                std_psi_min = result.get('psi_standard_min')
                std_psi_max = result.get('psi_standard_max')
                std_psi_mean = result.get('psi_standard_mean')
                enh_psi_min = result.get('psi_enhanced_min')
                enh_psi_max = result.get('psi_enhanced_max')
                enh_psi_mean = result.get('psi_enhanced_mean')
                
                # Log to continuous file immediately (thread-safe append)
                with open(log_file, 'a') as f:
                    f.write(f"{seq_id}\t{std_psi_min}\t{std_psi_max}\t{std_psi_mean}\t{enh_psi_min}\t{enh_psi_max}\t{enh_psi_mean}\tSUCCESS\n")
                    f.flush()  # Ensure immediate write
                
                results.append({
                    'seq_id': seq_id,
                    'std_psi_min': std_psi_min,
                    'std_psi_max': std_psi_max, 
                    'std_psi_mean': std_psi_mean,
                    'enh_psi_min': enh_psi_min,
                    'enh_psi_max': enh_psi_max,
                    'enh_psi_mean': enh_psi_mean,
                    'success': True
                })
                print(f"✅ Worker {worker_id}: {seq_id} complete (6 values extracted)")
            else:
                # Log failure
                with open(log_file, 'a') as f:
                    f.write(f"{seq_id}\tNone\tNone\tNone\tNone\tNone\tNone\tFAILED\n")
                    f.flush()  # Ensure immediate write
                
                results.append({
                    'seq_id': seq_id,
                    'std_psi_min': None, 'std_psi_max': None, 'std_psi_mean': None,
                    'enh_psi_min': None, 'enh_psi_max': None, 'enh_psi_mean': None,
                    'success': False
                })
                print(f"⚠️ Worker {worker_id}: {seq_id} failed")
                
        except Exception as e:
            # Log error
            with open(log_file, 'a') as f:
                f.write(f"{seq_id}\tNone\tNone\tNone\tNone\tNone\tNone\tERROR:{str(e)}\n")
                f.flush()  # Ensure immediate write
            
            results.append({
                'seq_id': seq_id,
                'std_psi_min': None, 'std_psi_max': None, 'std_psi_mean': None,
                'enh_psi_min': None, 'enh_psi_max': None, 'enh_psi_mean': None,
                'success': False
            })
            print(f"❌ Worker {worker_id}: {seq_id} error - {e}")
    
    return results

def cleanup_worker_dirs():
    """Clean up worker directories to save disk space."""
    base_dir = Path("/shared/data06/TileFormer_ABPS")
    worker_dirs = list(base_dir.glob("worker_*"))
    
    total_size = 0
    for worker_dir in worker_dirs:
        if worker_dir.is_dir():
            # Calculate size before deletion
            size = sum(f.stat().st_size for f in worker_dir.rglob('*') if f.is_file())
            total_size += size
            # Remove the worker directory
            shutil.rmtree(worker_dir, ignore_errors=True)
    
    print(f"🧹 Cleaned up {len(worker_dirs)} worker directories, freed {total_size/(1024**3):.2f} GB")
    return total_size

def save_summary_stats(df_merged, output_file, processing_time, success_count, total_count):
    """Save compact summary statistics to logs."""
    summary_file = f"logs/abps_summary_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Calculate statistics for successful results
    successful_df = df_merged[df_merged['success'] == True]
    
    summary = f"""ABPS Processing Summary - {time.strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
Input: {total_count} sequences
Success: {success_count} sequences ({100*success_count/total_count:.1f}%)
Processing time: {processing_time/3600:.2f} hours
Rate: {total_count/processing_time:.1f} sequences/sec
Output: {output_file}
Output size: {Path(output_file).stat().st_size/(1024**2):.1f} MB

Statistics for successful sequences:
Standard Config:
  ψ min: {successful_df['std_psi_min'].describe() if len(successful_df) > 0 else 'N/A'}
  ψ max: {successful_df['std_psi_max'].describe() if len(successful_df) > 0 else 'N/A'}  
  ψ mean: {successful_df['std_psi_mean'].describe() if len(successful_df) > 0 else 'N/A'}

Enhanced Config:
  ψ min: {successful_df['enh_psi_min'].describe() if len(successful_df) > 0 else 'N/A'}
  ψ max: {successful_df['enh_psi_max'].describe() if len(successful_df) > 0 else 'N/A'}
  ψ mean: {successful_df['enh_psi_mean'].describe() if len(successful_df) > 0 else 'N/A'}
"""
    
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(f"📊 Summary saved to {summary_file}")
    return summary_file

def log_system_status():
    """Log current system resource usage using basic system tools."""
    try:
        # Use basic system monitoring (no external dependencies)
        import subprocess
        
        # Get memory info
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        memory_line = result.stdout.split('\n')[1]  # Mem: line
        mem_info = memory_line.split()
        
        # Get disk info  
        result = subprocess.run(['df', '-h', '/shared/data06'], capture_output=True, text=True)
        disk_line = result.stdout.split('\n')[1]  # data line
        disk_info = disk_line.split()
        
        print(f"📊 System Status:")
        print(f"   RAM: {mem_info[2]} used / {mem_info[1]} total")
        print(f"   Disk: {disk_info[2]} used / {disk_info[1]} total ({disk_info[4]} used)")
        
    except Exception as e:
        print(f"📊 System monitoring unavailable: {e}")

# Setup logging  
os.makedirs('logs', exist_ok=True)
log_file = f'logs/fast_abps_{time.strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise from ABPS internal logging
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep main script logging at INFO

def main():
    """Optimized main processing function."""
    # Input/output files
    input_file = "data/corpus_50k_complete.tsv"
    output_file = "/shared/data06/TileFormer_ABPS/corpus_50k_with_abps_optimized.tsv.gz"
    
    # Check input file exists
    if not Path(input_file).exists():
        logger.error(f"❌ Input file not found: {input_file}")
        return False
    
    # Load sequences
    logger.info(f"📂 Loading sequences from {input_file}")
    df = pd.read_csv(input_file, sep='\t')
    logger.info(f"📊 Found {len(df)} sequences to process")
    
    # Optimal process count for CPU-intensive ABPS calculations
    # Use physical cores (24) + some hyperthreads for I/O = 32 workers
    num_workers = 32  # Optimized for 48-thread server (24 physical cores)
    chunk_size = 8   # Smaller chunks for better load balancing with more workers
    
    logger.info(f"🔧 Configuration:")
    logger.info(f"   Workers: {num_workers}")
    logger.info(f"   Chunk size: {chunk_size}")
    logger.info(f"   Total cores: {os.cpu_count()}")
    
    # Prepare sequence batches for chunked processing
    sequences = df['sequence'].tolist()
    seq_ids = df['seq_id'].tolist()
    sequence_pairs = list(zip(sequences, seq_ids))
    
    # Split into chunks for batch processing
    chunks = [sequence_pairs[i:i + chunk_size] for i in range(0, len(sequence_pairs), chunk_size)]
    
    logger.info(f"🚀 Starting optimized ABPS processing...")
    logger.info(f"   Server: 48-core Intel Xeon Silver 4214, 503GB RAM")
    logger.info(f"   Workers: {num_workers} (using ~2/3 of available cores)")
    logger.info(f"   {len(chunks)} chunks of {chunk_size} sequences")
    logger.info(f"   Worker reuse: ✅ (true processor reuse)")
    logger.info(f"   Fast config: ✅ (no minimization)")
    
    # Log initial system status
    log_system_status()
    
    start_time = time.time()
    all_results = []
    processed_count = 0
    success_count = 0
    
    # Process chunks in parallel with worker reuse
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        for chunk_results in pool.imap_unordered(process_sequence_batch, chunks, chunksize=1):
            # Flatten chunk results
            for result in chunk_results:
                all_results.append(result)
                processed_count += 1
                if result['success']:
                    success_count += 1
            
            # Progress reporting
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            eta = (len(sequences) - processed_count) / rate if rate > 0 else 0
            success_rate = 100 * success_count / processed_count if processed_count > 0 else 0
            
            logger.info(f"📈 Progress: {processed_count}/{len(sequences)} ({100*processed_count/len(sequences):.1f}%) "
                       f"- {success_count} successful ({success_rate:.1f}%) "
                       f"- Rate: {rate:.1f}/sec - ETA: {eta/3600:.1f}h")
    
    # Convert results to dataframe format
    results_df = pd.DataFrame(all_results)
    
    # Merge with original dataframe
    df_merged = df.merge(results_df, on='seq_id', how='left')
    
    # Save compressed results
    logger.info(f"💾 Saving compressed results to {output_file}")
    df_merged.to_csv(output_file, sep='\t', index=False, compression='gzip' if output_file.endswith('.gz') else None)
    
    # Clean up worker directories to save space
    logger.info("🧹 Cleaning up worker directories...")
    freed_space = cleanup_worker_dirs()
    
    # Save summary statistics
    total_time = time.time() - start_time
    logger.info("📊 Saving summary statistics...")
    summary_file = save_summary_stats(df_merged, output_file, total_time, success_count, len(sequences))
    
    # Final statistics
    logger.info(f"🎉 Processing complete!")
    logger.info(f"   Total time: {total_time/3600:.2f} hours")
    logger.info(f"   Success rate: {success_count}/{len(sequences)} ({100*success_count/len(sequences):.1f}%)")
    logger.info(f"   Average rate: {processed_count/total_time:.1f} sequences/sec")
    logger.info(f"   6 values per sequence: std_psi_min/max/mean + enh_psi_min/max/mean")
    logger.info(f"   Output: {output_file} (compressed)")
    logger.info(f"   Summary: {summary_file}")
    logger.info(f"   Freed space: {freed_space/(1024**3):.2f} GB")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)