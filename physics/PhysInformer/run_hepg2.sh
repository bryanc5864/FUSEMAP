#!/bin/bash

# PhysInformer training script for HepG2
# Run with: bash run_hepg2.sh

echo "Starting PhysInformer training for HepG2..."

python train.py \
    --cell_type HepG2 \
    --data_dir ../output \
    --output_dir ./runs \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 25 \
    --device cuda \
    --num_workers 4

echo "HepG2 training completed!"