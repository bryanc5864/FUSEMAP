#!/bin/bash

# Training script for Physics-Aware Model with OneCycleLR

# Default parameters
CELL_TYPE="${1:-HepG2}"
EPOCHS="${2:-50}"
BATCH_SIZE="${3:-32}"
MAX_LR="${4:-1e-3}"  # Peak learning rate for OneCycleLR

echo "=================================="
echo "Training Physics-Aware Model"
echo "=================================="
echo "Cell Type: $CELL_TYPE"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Max LR: $MAX_LR"
echo "=================================="

# Run training with OneCycleLR (10% warmup to max LR)
python train.py \
    --cell_type $CELL_TYPE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $MAX_LR \
    --device cuda \
    --data_dir ../output \
    --num_workers 4

# Usage examples:
# ./train_physics.sh HepG2 50 32 1e-3
# ./train_physics.sh K562 50 32 1e-3
# ./train_physics.sh WTC11 50 32 1e-3