#!/usr/bin/env python3
"""
Combine all continuous ABPS log files into a single result file.
"""

import pandas as pd
import os
from pathlib import Path
import glob

def combine_continuous_logs():
    """Combine continuous log file into a single result file."""
    print("🔍 Processing continuous ABPS log file...")
    
    # Single continuous log file
    log_file = "logs/abps_continuous_results.txt"
    
    if not Path(log_file).exists():
        print(f"❌ Continuous log file not found: {log_file}")
        return False
    
    print(f"📂 Processing {log_file}...")
    
    all_results = []
    
    with open(log_file, 'r') as f:
        for line in f:
                line = line.strip()
                if not line or line.startswith('seq_id'):
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 8:
                    seq_id = parts[0]
                    std_psi_min = float(parts[1]) if parts[1] != 'None' else None
                    std_psi_max = float(parts[2]) if parts[2] != 'None' else None
                    std_psi_mean = float(parts[3]) if parts[3] != 'None' else None
                    enh_psi_min = float(parts[4]) if parts[4] != 'None' else None
                    enh_psi_max = float(parts[5]) if parts[5] != 'None' else None
                    enh_psi_mean = float(parts[6]) if parts[6] != 'None' else None
                    status = parts[7]
                    
                    success = status == 'SUCCESS'
                    
                    all_results.append({
                        'seq_id': seq_id,
                        'std_psi_min': std_psi_min,
                        'std_psi_max': std_psi_max,
                        'std_psi_mean': std_psi_mean,
                        'enh_psi_min': enh_psi_min,
                        'enh_psi_max': enh_psi_max,
                        'enh_psi_mean': enh_psi_mean,
                        'success': success,
                        'status': status
                    })
    
    if not all_results:
        print("❌ No results found in log files")
        return False
    
    # Create dataframe
    df_results = pd.DataFrame(all_results)
    
    # Remove duplicates (keep first occurrence)
    df_results = df_results.drop_duplicates(subset=['seq_id'], keep='first')
    
    # Load original corpus and merge
    corpus_file = "data/corpus_50k_complete.tsv"
    if Path(corpus_file).exists():
        df_corpus = pd.read_csv(corpus_file, sep='\t')
        df_merged = df_corpus.merge(df_results, on='seq_id', how='left')
    else:
        df_merged = df_results
    
    # Save combined results
    output_file = "/shared/data06/TileFormer_ABPS/corpus_50k_with_abps_optimized.tsv.gz"
    df_merged.to_csv(output_file, sep='\t', index=False, compression='gzip')
    
    # Statistics
    successful = len(df_results[df_results['success'] == True])
    total = len(df_results)
    
    print(f"✅ Processed continuous log file")
    print(f"📊 Total sequences: {total}")
    print(f"📊 Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"💾 Output: {output_file}")
    
    return True

if __name__ == "__main__":
    combine_continuous_logs()
