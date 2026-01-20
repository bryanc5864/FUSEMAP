#!/usr/bin/env python3
"""
Test script to verify resume functionality without actually running training
"""

import torch
from pathlib import Path
import sys
import argparse

def test_resume_checkpoint(cell_type, run_dir):
    """Test loading a checkpoint and verify all components"""
    print(f"\n{'='*60}")
    print(f"Testing resume for {cell_type}")
    print(f"{'='*60}")
    
    # Check run directory
    if not run_dir.exists():
        print(f"ERROR: Run directory does not exist: {run_dir}")
        return False
    print(f"✓ Run directory exists: {run_dir}")
    
    # Find latest checkpoint
    checkpoint_files = sorted(run_dir.glob('checkpoint_epoch_*.pt'), 
                            key=lambda x: int(x.stem.split('_')[-1]))
    if not checkpoint_files:
        print("ERROR: No checkpoint files found")
        return False
    
    latest_checkpoint = checkpoint_files[-1]
    epoch_num = int(latest_checkpoint.stem.split('_')[-1])
    print(f"✓ Found {len(checkpoint_files)} checkpoints")
    print(f"✓ Latest checkpoint: {latest_checkpoint.name} (epoch {epoch_num})")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return False
    
    # Verify checkpoint contents
    required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict']
    for key in required_keys:
        if key not in checkpoint:
            print(f"ERROR: Missing required key in checkpoint: {key}")
            return False
    print("✓ All required keys present in checkpoint")
    
    # Check auxiliary optimizer if present
    if 'aux_optimizer_state_dict' in checkpoint:
        print("✓ Auxiliary optimizer state present")
    
    # Verify epoch number
    if checkpoint['epoch'] != epoch_num:
        print(f"WARNING: Checkpoint epoch ({checkpoint['epoch']}) != filename epoch ({epoch_num})")
    else:
        print(f"✓ Checkpoint epoch matches: {epoch_num}")
    
    # Check best_val_loss
    if 'best_val_loss' in checkpoint:
        print(f"✓ Best validation loss in checkpoint: {checkpoint['best_val_loss']:.6f}")
    else:
        # Try to get from training log
        training_log = run_dir / 'training.log'
        if training_log.exists():
            with open(training_log, 'r') as f:
                log_content = f.read()
            import re
            matches = re.findall(r'New best validation loss: ([\d.]+)', log_content)
            if matches:
                best_val_loss = float(matches[-1])
                print(f"✓ Best validation loss from log: {best_val_loss:.6f}")
                print("  (Note: will use this value when resuming)")
    
    # Check batch log
    batch_log = run_dir / 'batch_log.txt'
    if not batch_log.exists():
        print("WARNING: No batch log file found")
    else:
        with open(batch_log, 'r') as f:
            lines = f.readlines()
        data_lines = [l for l in lines[1:] if l.strip()]
        if data_lines:
            last_parts = data_lines[-1].split(',')
            last_epoch = int(last_parts[0])
            last_batch = int(last_parts[1])
            print(f"✓ Batch log exists: {len(data_lines)} entries")
            print(f"  Last entry: epoch {last_epoch}, batch {last_batch}")
            
            if last_epoch == epoch_num:
                print(f"✓ Batch log is clean (ends at checkpoint epoch)")
            else:
                print(f"WARNING: Batch log last epoch ({last_epoch}) != checkpoint epoch ({epoch_num})")
    
    # Check epoch results
    epoch_results = run_dir / 'epoch_results.txt'
    if not epoch_results.exists():
        print("WARNING: No epoch results file found")
    else:
        with open(epoch_results, 'r') as f:
            content = f.read()
        import re
        epochs_found = re.findall(r'^EPOCH (\d+):', content, re.MULTILINE)
        if epochs_found:
            max_epoch = max(int(e) for e in epochs_found)
            print(f"✓ Epoch results file exists: epochs 1-{max_epoch}")
            if max_epoch == epoch_num:
                print(f"✓ Epoch results is clean (ends at checkpoint epoch)")
    
    # Summary
    start_epoch = epoch_num + 1
    print(f"\n✓ Ready to resume from epoch {start_epoch}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test resume functionality')
    parser.add_argument('--cell_type', type=str, choices=['K562', 'WTC11', 'HepG2', 'all'],
                      default='all', help='Cell type to test')
    args = parser.parse_args()
    
    # Define runs
    runs = {
        'K562': Path('runs/K562_20250829_095741'),
        'WTC11': Path('runs/WTC11_20250829_095738'),
        'HepG2': Path('runs/HepG2_20250829_095749')
    }
    
    if args.cell_type == 'all':
        cell_types = ['K562', 'WTC11', 'HepG2']
    else:
        cell_types = [args.cell_type]
    
    success = True
    for cell_type in cell_types:
        if cell_type in runs:
            result = test_resume_checkpoint(cell_type, runs[cell_type])
            success = success and result
        else:
            print(f"Unknown cell type: {cell_type}")
            success = False
    
    if success:
        print(f"\n{'='*60}")
        print("✓ All tests passed! Resume functionality is ready.")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("⚠ Some tests failed. Please check the errors above.")
        print(f"{'='*60}")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())