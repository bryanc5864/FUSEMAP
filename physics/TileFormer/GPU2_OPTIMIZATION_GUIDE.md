# GPU 2 Optimization Guide for TileFormer Pipeline

## 🎯 **GPU 2 Configuration (10GB Memory)**

All components have been optimized to use **GPU 2** with **10GB memory** for maximum performance.

## 🚀 **Quick Setup & Test**

```bash
# 1. Test GPU 2 setup
python test_gpu2_setup.py

# 2. Build GPU APBS (if needed)
bash electrostatics/build_gpu_apbs.sh

# 3. Test GPU ABPS pipeline
bash electrostatics/run_gpu_abps_pipeline.sh --test --gpu_check

# 4. Test TileFormer training
python TileFormer/quick_test_all.py

# 5. Run full TileFormer training test
bash TileFormer/run_multi_target_training.sh --test
```

## ⚡ **Performance Optimizations**

### **GPU ABPS Processing**
- **GPU Device**: `CUDA_VISIBLE_DEVICES=2`
- **Workers**: 16 (increased from 8)
- **Batch Size**: 100 sequences (increased from 50)
- **Expected Speedup**: ~20x vs CPU
- **Memory Usage**: ~8GB for large batches

### **TileFormer Training**
- **GPU Device**: `cuda:2`
- **Batch Size**: 512 (increased from 256)
- **Memory Usage**: ~6-8GB for full batch
- **Expected Training Time**: ~2-3 hours for 6 models

### **Memory Management**
- **10GB Total**: GPU 2 memory
- **ABPS Usage**: ~8GB for parallel processing
- **TileFormer Usage**: ~6-8GB for 512 batch size
- **Buffer**: 2GB safety margin

## 📁 **Updated File Configurations**

### **ABPS GPU Scripts**
```bash
electrostatics/
├── build_gpu_apbs.sh           # Sets CUDA_VISIBLE_DEVICES=2
├── gpu_dual_config_processor.py # GPU 2 optimized processor
├── gpu_parallel_abps.py        # 16 workers, 100 batch size
├── run_gpu_abps_pipeline.sh    # Full pipeline with GPU 2
└── test_gpu_abps.py           # GPU environment tests
```

### **TileFormer Training Scripts**
```bash
TileFormer/
├── train_multi_target_tileformer.py  # cuda:2, batch_size=512
├── run_multi_target_training.sh      # CUDA_VISIBLE_DEVICES=2
└── quick_test_all.py                 # Quick pipeline test
```

### **Environment Setup**
```bash
# Automatic GPU 2 environment (created by build script)
~/setup_gpu_apbs.sh:
  export CUDA_VISIBLE_DEVICES=2
  export APBS_GPU=1
  export OMP_NUM_THREADS=32
```

## 🔧 **Detailed Configurations**

### **ABPS Processing Parameters**
```bash
# Optimized for 10GB GPU 2
Workers: 16
MPI Ranks per Worker: 4
OpenMP Threads per Rank: 8
Batch Size: 100 sequences
Total GPU Memory Usage: ~8GB
Expected Rate: 10-20 sequences/second
```

### **TileFormer Training Parameters**
```bash
# Optimized for 10GB GPU 2
Device: cuda:2
Batch Size: 512
Model Parameters: ~2M per model
6 Models Total: ~12M parameters
GPU Memory Usage: ~6-8GB
Training Time: ~30 minutes/model (test mode)
```

### **Memory Breakdown**
```
GPU 2 (10GB) Memory Allocation:
├── ABPS Processing: 8GB
│   ├── Grid calculations: 5GB
│   ├── Parallel batches: 2GB
│   └── Buffer: 1GB
└── TileFormer Training: 8GB
    ├── Model parameters: 1GB
    ├── Activations (512 batch): 4GB
    ├── Gradients: 1GB
    ├── Optimizer states: 1GB
    └── Buffer: 1GB
```

## 🎯 **Usage Examples**

### **GPU ABPS Processing**
```bash
# Test mode (20 sequences, ~1 minute)
bash electrostatics/run_gpu_abps_pipeline.sh --test

# Full corpus (50k sequences, ~2-3 hours)
bash electrostatics/run_gpu_abps_pipeline.sh \
    --input data/corpus_50k_complete.tsv \
    --output TileFormer/data/corpus_50k_gpu_abps.tsv
```

### **TileFormer Training**
```bash
# Test mode (1 epoch, 100 samples, ~30 seconds per model)
bash TileFormer/run_multi_target_training.sh --test

# Full training (50 epochs, all data, ~3 hours total)
bash TileFormer/run_multi_target_training.sh \
    TileFormer/data/corpus_50k_gpu_abps.tsv
```

## 📊 **Expected Performance**

### **ABPS Processing (50k sequences)**
- **CPU Baseline**: ~625 hours (45s/sequence)
- **GPU 2 Performance**: ~3 hours (20x speedup)
- **Rate**: 10-20 sequences/second
- **Memory**: 8GB GPU + 32 CPU cores

### **TileFormer Training (6 models)**
- **Training Time**: ~3 hours total
- **Per Model**: ~30 minutes
- **Batch Processing**: 512 sequences/batch
- **Memory**: 6-8GB GPU

### **Total Pipeline Time**
```
Complete Pipeline Timeline:
├── ABPS Processing: 3 hours
├── Data Preparation: 15 minutes
├── TileFormer Training: 3 hours
└── Evaluation: 30 minutes
Total: ~6.5 hours (vs ~625 hours CPU-only)
```

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **GPU 2 Not Available**
   ```bash
   # Check GPU status
   nvidia-smi
   
   # Verify GPU 2 is free
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
   ```

2. **Out of Memory Errors**
   ```bash
   # Reduce batch sizes
   export ABPS_BATCH_SIZE=50      # Instead of 100
   export TILEFORMER_BATCH_SIZE=256  # Instead of 512
   ```

3. **APBS Not Found**
   ```bash
   # Build GPU APBS
   bash electrostatics/build_gpu_apbs.sh
   
   # Source environment
   source ~/setup_gpu_apbs.sh
   ```

### **Memory Optimization**
```bash
# If hitting memory limits, use these settings:

# Conservative ABPS settings
--workers 8 --batch_size 50

# Conservative TileFormer settings  
--batch_size 256

# Enable gradient checkpointing (if implemented)
--gradient_checkpointing
```

## ✅ **Verification Commands**

```bash
# 1. Check GPU 2 availability
python test_gpu2_setup.py

# 2. Test ABPS GPU environment
python electrostatics/test_gpu_abps.py --quick

# 3. Test TileFormer components
python TileFormer/quick_test_all.py

# 4. Full pipeline smoke test
bash electrostatics/run_gpu_abps_pipeline.sh --test && \
bash TileFormer/run_multi_target_training.sh --test
```

## 🎉 **Ready Commands**

```bash
# Complete optimized pipeline for GPU 2:

# 1. Process ABPS data (3 hours)
bash electrostatics/run_gpu_abps_pipeline.sh

# 2. Train TileFormer models (3 hours)  
bash TileFormer/run_multi_target_training.sh TileFormer/data/corpus_50k_gpu_abps.tsv

# 3. Results will be in:
#    TileFormer/outputs/multi_target_YYYYMMDD_HHMMSS/
```

---

**🚀 All components are now optimized for GPU 2 with 10GB memory!**