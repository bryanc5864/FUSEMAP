#!/bin/bash

# PhysInformer training script for WTC11
# Run with: bash run_wtc11.sh

echo "Starting PhysInformer training for WTC11..."

python train.py \
    --cell_type WTC11 \
    --data_dir ../output \
    --output_dir ./runs \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 25 \
    --device cuda \
    --num_workers 4

echo "WTC11 training completed!"