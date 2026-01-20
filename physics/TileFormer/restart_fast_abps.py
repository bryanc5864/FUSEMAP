#!/usr/bin/env python3
"""
Restart ABPS processing with fast configuration (no minimization)
"""

import subprocess
import sys
import time
import pandas as pd
from pathlib import Path

def kill_existing_processes():
    """Kill any existing ABPS-related processes."""
    print("🛑 Stopping existing ABPS processes...")
    
    # Kill any sander, tleap, apbs, or python processes related to ABPS
    processes_to_kill = ['sander', 'tleap', 'apbs', 'pdb2pqr']
    
    for process in processes_to_kill:
        try:
            result = subprocess.run(['pkill', '-f', process], capture_output=True)
            if result.returncode == 0:
                print(f"   ✅ Stopped {process} processes")
            else:
                print(f"   ℹ️  No {process} processes found")
        except Exception as e:
            print(f"   ⚠️  Could not kill {process}: {e}")
    
    # Kill any python processes with ABPS in the command line
    try:
        result = subprocess.run(['pkill', '-f', 'abps'], capture_output=True)
        print("   ✅ Stopped ABPS-related Python processes")
    except:
        print("   ℹ️  No ABPS Python processes found")
    
    print("   ⏱️  Waiting 5 seconds for processes to terminate...")
    time.sleep(5)

def start_fast_abps():
    """Start ABPS processing with fast configuration."""
    print("🚀 Starting fast ABPS processing...")
    
    # Check if we have the corpus file
    corpus_file = "data/corpus_50k_complete.tsv"
    if not Path(corpus_file).exists():
        print(f"❌ Corpus file not found: {corpus_file}")
        return False
    
    # Load corpus to see how many sequences we have
    df = pd.read_csv(corpus_file, sep='\t')
    print(f"📊 Found {len(df)} sequences to process")
    
    # Create fast ABPS command
    cmd = [
        "python", "electrostatics/run_optimized_abps_batch.py",
        "--input", corpus_file,
        "--output", "data/corpus_50k_with_abps_fast.tsv", 
        "--batch-size", "500",
        "--processes", "16",
        "--config", "fast"  # Use fast config with no minimization
    ]
    
    print(f"🏃 Running command: {' '.join(cmd)}")
    
    # Start the process
    try:
        subprocess.Popen(cmd)
        print("✅ Fast ABPS processing started!")
        print("📋 Progress will be logged to terminal")
        print("⏱️  Expected completion: ~2-3 hours (vs 51+ hours with minimization)")
        return True
    except Exception as e:
        print(f"❌ Failed to start fast ABPS: {e}")
        return False

def main():
    """Main function."""
    print("🔄 Restarting ABPS with fast configuration...")
    print("=" * 60)
    
    # Kill existing processes
    kill_existing_processes()
    
    # Start fast processing
    success = start_fast_abps()
    
    if success:
        print("\n🎉 Fast ABPS processing has been started!")
        print("📈 Benefits of fast config:")
        print("   - No sander minimization = no 'standard' errors")
        print("   - Uses ideal B-DNA geometry = consistent results")  
        print("   - ~20x faster processing = 2-3 hours vs 51+ hours")
        print("   - More reliable = fewer failed sequences")
        print("\n📊 Monitor progress with:")
        print("   tail -f logs/abps_processing_*.log")
    else:
        print("\n❌ Failed to start fast ABPS processing")
        print("🔧 Manual command:")
        print("   python electrostatics/run_optimized_abps_batch.py --input data/corpus_50k_complete.tsv --output data/corpus_50k_with_abps_fast.tsv --config fast")

if __name__ == "__main__":
    main()