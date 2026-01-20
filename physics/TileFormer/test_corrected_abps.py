#!/usr/bin/env python3
"""
Test script to validate the corrected ABPS methodology.

This script tests the key fixes:
1. Sequence-dependent geometry via minimization
2. 2-6Å solvent shell extraction  
3. Proper APBS parameters
4. Sequence sensitivity validation
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add electrostatics module to path
sys.path.append(str(Path(__file__).parent))

from electrostatics.tleap_abps_processor import TLEaPABPSProcessor, ABPSConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_corrected_abps():
    """Test the corrected ABPS methodology."""
    
    logger.info("🧪 Testing Corrected ABPS Methodology")
    logger.info("="*50)
    
    # Configure corrected methodology
    config = ABPSConfig(
        # Proper focusing parameters
        dime=(193, 193, 193),
        cglen=(200.0, 200.0, 200.0), 
        fglen=(40.0, 40.0, 40.0),
        
        # Shell extraction (key fix!)
        shell_inner=2.0,
        shell_outer=6.0,
        
        # Enable sequence-dependent geometry
        enable_minimization=True,
        min_steps=2000,
        ncyc=500
    )
    
    processor = TLEaPABPSProcessor(
        work_dir="test_corrected_abps",
        config=config,
        cleanup=True
    )
    
    # Test sequences designed to show sequence dependence
    test_cases = {
        # GC content extremes
        "AT_rich": "ATATATATATATATATATAT",   # 0% GC
        "GC_rich": "GCGCGCGCGCGCGCGCGCGC",   # 100% GC
        
        # Purine/pyrimidine bias
        "purine_rich": "AGAGAGAGAGAGAGAGAGAG",  # All purines
        "pyrimidine_rich": "CTCTCTCTCTCTCTCTCTCT", # All pyrimidines
        
        # Real sequences
        "random_1": "ACGTACGTACGTACGTACGT",
        "random_2": "TGCATGCATGCATGCATGCA",
        
        # Palindromic vs non-palindromic
        "palindrome": "AATTCCGGCCGGAATTCCGG",
        "non_palindrome": "ACGTACGTACGTACGTACGT"
    }
    
    logger.info("🔬 Processing test sequences...")
    results = {}
    
    for name, sequence in test_cases.items():
        logger.info(f"  Testing {name}: {sequence}")
        
        try:
            psi = processor.process_sequence(sequence, f"test_{name}")
            results[name] = psi
            
            if psi is not None:
                logger.info(f"    ✅ ψ = {psi:.6f} kT/e")
            else:
                logger.error(f"    ❌ Failed")
                
        except Exception as e:
            logger.error(f"    ❌ Error: {e}")
            results[name] = None
    
    # Analyze results
    logger.info("\n📊 Results Analysis:")
    logger.info("-" * 30)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) < 2:
        logger.error("❌ Too few successful calculations to assess sequence dependence")
        return False
    
    psi_values = np.array(list(valid_results.values()))
    psi_mean = np.mean(psi_values)
    psi_std = np.std(psi_values)
    psi_range = np.max(psi_values) - np.min(psi_values)
    
    logger.info(f"Valid calculations: {len(valid_results)}/{len(test_cases)}")
    logger.info(f"Mean ψ: {psi_mean:.6f} ± {psi_std:.6f} kT/e")
    logger.info(f"Range: {psi_range:.6f} kT/e")
    logger.info(f"Min: {np.min(psi_values):.6f} kT/e")
    logger.info(f"Max: {np.max(psi_values):.6f} kT/e")
    
    # Print individual results
    logger.info("\nIndividual results:")
    for name, psi in sorted(valid_results.items()):
        logger.info(f"  {name:15s}: {psi:8.6f} kT/e")
    
    # Assessment
    logger.info("\n🎯 Assessment:")
    
    if psi_range < 0.5:
        logger.warning(f"⚠️  Small range ({psi_range:.6f} kT/e) - potential sequence insensitivity")
        logger.warning("   This suggests the methodology may still be averaging out sequence differences")
        return False
    elif psi_range < 2.0:
        logger.info(f"✅ Moderate range ({psi_range:.6f} kT/e) - some sequence dependence detected")
        return True
    else:
        logger.info(f"✅ Large range ({psi_range:.6f} kT/e) - strong sequence dependence!")
        return True

def test_methodology_comparison():
    """Compare old vs new methodology on same sequences."""
    
    logger.info("\n🔄 Methodology Comparison Test")
    logger.info("="*40)
    
    test_sequences = ["ATATATATATATATATATAT", "GCGCGCGCGCGCGCGCGCGC"]
    
    # Old methodology (sequence-insensitive)
    old_config = ABPSConfig(
        dime=(129, 129, 129),           # Coarse grid
        cglen=(200.0, 200.0, 200.0),
        fglen=(200.0, 200.0, 200.0),   # No focusing
        enable_minimization=False       # Ideal geometry
        # Would use phosphate extraction (not implemented here)
    )
    
    # New methodology (sequence-sensitive)
    new_config = ABPSConfig(
        dime=(193, 193, 193),           # Fine grid
        cglen=(200.0, 200.0, 200.0),
        fglen=(40.0, 40.0, 40.0),      # Proper focusing
        shell_inner=2.0,                # Shell extraction
        shell_outer=6.0,
        enable_minimization=True        # Sequence-dependent geometry
    )
    
    for method_name, config in [("Old", old_config), ("New", new_config)]:
        logger.info(f"\n{method_name} methodology:")
        
        processor = TLEaPABPSProcessor(
            work_dir=f"test_{method_name.lower()}_method",
            config=config,
            cleanup=True
        )
        
        results = []
        for i, seq in enumerate(test_sequences):
            psi = processor.process_sequence(seq, f"{method_name.lower()}_{i}")
            results.append(psi)
            logger.info(f"  {seq}: {psi:.6f} kT/e" if psi else f"  {seq}: Failed")
        
        if all(r is not None for r in results):
            diff = abs(results[1] - results[0])
            logger.info(f"  Difference: {diff:.6f} kT/e")
            
            if diff < 0.1:
                logger.info(f"  → Sequence insensitive (expected for old method)")
            else:
                logger.info(f"  → Sequence dependent!")

if __name__ == "__main__":
    logger.info("🚀 Starting ABPS Methodology Validation")
    
    # Run main test
    success = test_corrected_abps()
    
    # Run comparison test
    # test_methodology_comparison()  # Uncomment to compare methodologies
    
    if success:
        logger.info("\n🎉 Validation PASSED - Corrected methodology shows sequence dependence!")
        logger.info("✅ Ready to re-label the full corpus")
    else:
        logger.error("\n❌ Validation FAILED - Methodology may still be sequence-insensitive")
        logger.error("🔧 Further debugging needed before full corpus re-labeling")