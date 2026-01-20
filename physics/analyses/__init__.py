"""
Physics Analyses Package

This package contains analysis scripts for understanding physics features:

01_univariate_stability.py   - Correlations, partial correlations, CV stability
02_multivariate_models.py    - Elastic Net, Ridge, GAM on physics features
03_incremental_control.py    - Physics → +PWM, coefficient attenuation
04_interaction_mapping.py    - Physics×TF interaction H-statistics
05_regime_discovery.py       - HDBSCAN clustering in physics space
06_cross_celltype_generalization.py - Train/test transfer across cell types

run_analyses.py              - Coordinator script to run all analyses

Usage:
    python run_analyses.py --human    # Run on human lentiMPRA data
    python run_analyses.py --list     # List available datasets
"""
