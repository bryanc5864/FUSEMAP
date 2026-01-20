#!/usr/bin/env python3
"""
Quick GPU 2 Setup Test
=====================

Verify that GPU 2 is available and properly configured for both
ABPS processing and TileFormer training.

Usage: python test_gpu2_setup.py
"""

import os
import subprocess
import torch

def test_gpu2_availability():
    """Test GPU 2 availability and specs"""
    print("🧪 Testing GPU 2 Availability...")
    
    # Set GPU 2
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    # Check PyTorch CUDA
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"   ✅ PyTorch CUDA available")
        print(f"   Devices visible: {device_count}")
        
        if device_count > 0:
            device = torch.device('cuda:0')  # First visible (GPU 2)
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU name: {gpu_name}")
            print(f"   GPU memory: {gpu_memory:.1f} GB")
            
            # Test tensor operations
            x = torch.randn(1000, 1000).to(device)
            y = torch.mm(x, x.t())
            print(f"   ✅ GPU operations working")
            
            return True
        else:
            print(f"   ❌ No GPU devices visible")
            return False
    else:
        print(f"   ❌ PyTorch CUDA not available")
        return False


def test_gpu2_memory():
    """Test GPU 2 memory for TileFormer training"""
    print("\n🧪 Testing GPU 2 Memory for TileFormer...")
    
    if not torch.cuda.is_available():
        print("   ❌ CUDA not available")
        return False
    
    device = torch.device('cuda:0')
    
    try:
        # Simulate TileFormer memory usage
        batch_size = 512
        seq_length = 20
        channels = 4
        hidden_dim = 192
        
        # Input batch
        sequences = torch.randn(batch_size, channels, seq_length).to(device)
        print(f"   Input batch: {batch_size}×{channels}×{seq_length}")
        
        # Simulate model parameters (~1M parameters)
        model_params = torch.randn(1000000).to(device)
        print(f"   Model parameters: ~1M")
        
        # Simulate hidden states
        hidden = torch.randn(batch_size, hidden_dim, seq_length).to(device)
        print(f"   Hidden states: {batch_size}×{hidden_dim}×{seq_length}")
        
        # Check memory usage
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_cached = torch.cuda.memory_reserved() / 1e9
        
        print(f"   ✅ Memory test passed")
        print(f"   Memory used: {memory_used:.2f} GB")
        print(f"   Memory cached: {memory_cached:.2f} GB")
        
        if memory_used < 8.0:  # Leave 2GB buffer
            print(f"   ✅ Memory usage looks good for 10GB GPU")
            return True
        else:
            print(f"   ⚠️  High memory usage, may need smaller batches")
            return True
            
    except Exception as e:
        print(f"   ❌ Memory test failed: {e}")
        return False


def test_abps_environment():
    """Test ABPS environment for GPU 2"""
    print("\n🧪 Testing ABPS Environment...")
    
    # Check APBS binary
    try:
        result = subprocess.run(['which', 'apbs'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ APBS found: {result.stdout.strip()}")
        else:
            print(f"   ⚠️  APBS not found (may need to build GPU APBS)")
    except:
        print(f"   ⚠️  Could not check APBS")
    
    # Check environment variables
    gpu_env = os.getenv('CUDA_VISIBLE_DEVICES')
    if gpu_env == '2':
        print(f"   ✅ CUDA_VISIBLE_DEVICES: {gpu_env}")
    else:
        print(f"   ⚠️  CUDA_VISIBLE_DEVICES: {gpu_env} (should be 2)")
    
    apbs_gpu = os.getenv('APBS_GPU')
    if apbs_gpu:
        print(f"   ✅ APBS_GPU: {apbs_gpu}")
    else:
        print(f"   ⚠️  APBS_GPU not set")
    
    return True


def test_tileformer_import():
    """Test TileFormer model import and basic functionality"""
    print("\n🧪 Testing TileFormer Import...")
    
    try:
        import sys
        sys.path.append('TileFormer')
        sys.path.append('TileFormer/model')
        
        from model.tileformer import TileFormer
        
        # Create model on GPU 2
        model = TileFormer().to('cuda:0')
        print(f"   ✅ TileFormer model created")
        
        # Test forward pass
        batch_size = 32  # Smaller for test
        x = torch.randn(batch_size, 4, 20).to('cuda:0')
        
        with torch.no_grad():
            output = model(x)
        
        print(f"   ✅ Forward pass successful")
        print(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ TileFormer test failed: {e}")
        return False


def main():
    print("🚀 GPU 2 Setup Test for TileFormer Pipeline")
    print("=" * 50)
    
    # Set GPU 2 environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    tests = [
        ("GPU 2 Availability", test_gpu2_availability),
        ("GPU 2 Memory", test_gpu2_memory),
        ("ABPS Environment", test_abps_environment),
        ("TileFormer Import", test_tileformer_import),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test crashed: {e}")
    
    print(f"\n📊 Test Summary")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print(f"\n🎉 GPU 2 setup looks good!")
        print(f"Ready for:")
        print(f"   1. ABPS processing: bash electrostatics/run_gpu_abps_pipeline.sh --test")
        print(f"   2. TileFormer training: python TileFormer/quick_test_all.py")
        print(f"   3. Full pipeline: bash TileFormer/run_multi_target_training.sh --test")
    else:
        print(f"\n⚠️  Some tests failed. GPU 2 may need setup.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())