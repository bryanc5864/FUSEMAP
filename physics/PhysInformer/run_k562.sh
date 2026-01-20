#!/bin/bash

# PhysInformer training script for K562
# Run with: bash run_k562.sh

echo "Starting PhysInformer training for K562..."

python train.py \
    --cell_type K562 \
    --data_dir ../output \
    --output_dir ./runs \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 25 \
    --device cuda \
    --num_workers 4

echo "K562 training completed!"