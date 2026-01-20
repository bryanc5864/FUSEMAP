#!/usr/bin/env python3
"""
Debug version to see exactly what's happening in dual-config processing.
"""

import tempfile
import sys
from pathlib import Path

# Add electrostatics to path
sys.path.append('electrostatics')

from tleap_abps_processor import DualConfigABPSProcessor

def debug_dual_config():
    """Debug the dual-config processing step by step."""
    print("🧪 Debug dual-config processing...")
    
    # Simple test sequence
    sequence = "ATCGATCGATCGATCGATCG"
    seq_id = "debug_test"
    
    # Use a local temp directory we can inspect
    temp_dir = "/tmp/debug_dual_config"
    Path(temp_dir).mkdir(exist_ok=True)
    
    processor = DualConfigABPSProcessor(work_dir=temp_dir, cleanup=False)  # Don't cleanup for debug
    
    try:
        print(f"🔬 Processing sequence: {sequence}")
        print(f"📁 Work dir: {temp_dir}")
        
        result = processor.process_sequence_dual_config(sequence, seq_id)
        
        if result:
            print("✅ Success!")
            print(f"   Standard: {result['psi_standard_mean']:.6f}")
            print(f"   Enhanced: {result['psi_enhanced_mean']:.6f}")
        else:
            print("❌ Processing failed")
            
            # List files in work directory for debugging
            work_path = Path(temp_dir) / f"tile_{seq_id}"
            if work_path.exists():
                print(f"\n📂 Files in {work_path}:")
                for file in work_path.iterdir():
                    print(f"   {file.name}")
                    
                # Show any error files
                for error_file in ["min.out", "build.out"]:
                    error_path = work_path / error_file
                    if error_path.exists():
                        print(f"\n📄 Contents of {error_file}:")
                        with open(error_path) as f:
                            content = f.read()
                            print(content[-1000:])  # Last 1000 chars
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dual_config()