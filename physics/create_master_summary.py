#!/usr/bin/env python3
"""
Create master summary document for all results.
"""

import os
import shutil
import json
from pathlib import Path

def copy_analysis_pngs():
    """Copy analysis PNGs to results/Analyses folder."""
    source_dir = 'feature_analysis'
    dest_dir = 'results/Analyses/feature_histograms'

    os.makedirs(dest_dir, exist_ok=True)

    png_count = 0
    for cell_type in ['HepG2', 'K562', 'WTC11']:
        cell_source = os.path.join(source_dir, f'{cell_type}_histograms')
        if os.path.exists(cell_source):
            cell_dest = os.path.join(dest_dir, cell_type)
            os.makedirs(cell_dest, exist_ok=True)

            # Copy all PNGs
            for png_file in os.listdir(cell_source):
                if png_file.endswith('.png'):
                    src = os.path.join(cell_source, png_file)
                    dst = os.path.join(cell_dest, png_file)
                    shutil.copy2(src, dst)
                    png_count += 1

    print(f"Copied {png_count} analysis PNG files")

    # Copy feature statistics and correlations
    for cell_type in ['HepG2', 'K562', 'WTC11']:
        for file_type in ['feature_statistics', 'feature_correlations']:
            src = os.path.join(source_dir, f'{cell_type}_{file_type}.txt')
            if os.path.exists(src):
                shutil.copy2(src, 'results/Analyses/')
                print(f"Copied: {cell_type}_{file_type}.txt")

def create_master_analysis_document():
    """Create comprehensive master analysis document."""
    output_path = 'results/Analyses/MASTER_ANALYSIS_SUMMARY.txt'

    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(" " * 30 + "PHYSIFORMER PROJECT\n")
        f.write(" " * 25 + "MASTER ANALYSIS SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        f.write("This document provides a comprehensive overview of all analyses, results, and\n")
        f.write("trained models for the PhysInformer project.\n\n")

        # Table of Contents
        f.write("TABLE OF CONTENTS:\n")
        f.write("-" * 100 + "\n")
        f.write("  I.   Project Overview\n")
        f.write("  II.  Data & Feature Analysis\n")
        f.write("  III. TileFormer Model (DNA Sequence Embeddings)\n")
        f.write("  IV.  PhysInformer Models (4 Cell Types)\n")
        f.write("        A. WTC11 (Human Cardiomyocytes)\n")
        f.write("        B. HepG2 (Human Liver Cells)\n")
        f.write("        C. K562 (Human Leukemia Cells)\n")
        f.write("        D. S2 (Drosophila Cells)\n")
        f.write("  V.   Cross-Model Comparison\n")
        f.write("  VI.  Key Findings & Conclusions\n\n")

        # Section I: Project Overview
        f.write("\n" + "=" * 100 + "\n")
        f.write("I. PROJECT OVERVIEW\n")
        f.write("=" * 100 + "\n\n")

        f.write("Project Goal:\n")
        f.write("-" * 100 + "\n")
        f.write("Develop physics-aware deep learning models (PhysInformer) that predict biophysical\n")
        f.write("descriptor values for DNA sequences while simultaneously predicting regulatory activity.\n")
        f.write("The models integrate learned sequence representations (from TileFormer) with explicit\n")
        f.write("biophysical features to achieve interpretable and accurate predictions.\n\n")

        f.write("Architecture Components:\n")
        f.write("-" * 100 + "\n")
        f.write("1. TileFormer: Transformer-based model for learning DNA sequence embeddings (20bp)\n")
        f.write("2. PhysInformer: Physics-aware model with:\n")
        f.write("   - Main branch: Predicts 386-539 biophysical descriptors from sequence embeddings\n")
        f.write("   - Auxiliary Head A: Predicts activity from sequence + predicted descriptors\n")
        f.write("   - Auxiliary Head B: Predicts activity from predicted descriptors only\n\n")

        f.write("Biophysical Descriptor Categories:\n")
        f.write("-" * 100 + "\n")
        f.write("  • Thermodynamics: Free energy (dG), melting temperature (Tm), entropy (dS)\n")
        f.write("  • DNA Mechanics: Stiffness, twist, tilt, roll, slide, shift, rise parameters\n")
        f.write("  • DNA Shape: Minor groove width, bending propensity, propeller twist\n")
        f.write("  • Structural: Melting properties, stress/opening, stacking, G-quadruplex potential\n")
        f.write("  • Sequence Complexity: Shannon entropy, mutual information\n")
        f.write("  • Transcription Factors: 386-515 PWM binding predictions (cell-type specific)\n\n")

        # Section II: Data & Feature Analysis
        f.write("\n" + "=" * 100 + "\n")
        f.write("II. DATA & FEATURE ANALYSIS\n")
        f.write("=" * 100 + "\n\n")

        f.write("Datasets:\n")
        f.write("-" * 100 + "\n")
        f.write("  Cell Type   | Source       | # Features | Activity Measure          | Data Size\n")
        f.write("  " + "-" * 94 + "\n")
        f.write("  WTC11       | lentiMPRA    | 539        | normalized_log2           | 424 MB\n")
        f.write("  HepG2       | lentiMPRA    | 536        | normalized_log2           | 590 MB\n")
        f.write("  K562        | lentiMPRA    | 515        | normalized_log2           | 830 MB\n")
        f.write("  S2 (Dros.)  | S2 MPRA      | 386        | Dev + Hk log2_enrichment  | 2.5 GB\n\n")

        f.write("Feature Analysis Results:\n")
        f.write("-" * 100 + "\n")
        f.write("  • Feature statistics computed for all cell types (mean, std, min, max, percentiles)\n")
        f.write("  • Correlation matrices generated to identify redundant features\n")
        f.write("  • Distribution histograms created for all features (results/Analyses/feature_histograms/)\n")
        f.write("  • Zero-variance features removed prior to training\n\n")

        f.write("Key Statistics (from feature_statistics.txt files):\n")
        f.write("-" * 100 + "\n")

        # Read and summarize feature statistics if available
        for cell_type in ['WTC11', 'HepG2', 'K562']:
            stats_file = f'results/Analyses/{cell_type}_feature_statistics.txt'
            if os.path.exists(stats_file):
                f.write(f"  {cell_type}: See {cell_type}_feature_statistics.txt for complete statistics\n")

        f.write("\n")

        # Section III: TileFormer
        f.write("\n" + "=" * 100 + "\n")
        f.write("III. TILEFORMER MODEL - DNA SEQUENCE EMBEDDINGS\n")
        f.write("=" * 100 + "\n\n")

        # Load TileFormer metrics
        tileformer_json = 'results/TileFormer/parsed_epochs.json'
        if os.path.exists(tileformer_json):
            with open(tileformer_json, 'r') as tf_file:
                tileformer_epochs = json.load(tf_file)
                last_epoch = tileformer_epochs[-1]

                f.write("Model Architecture:\n")
                f.write("-" * 100 + "\n")
                f.write("  • Type: Transformer encoder\n")
                f.write("  • Sequence Length: 20 bp\n")
                f.write("  • Model Dimension: 192\n")
                f.write("  • Attention Heads: 4\n")
                f.write("  • Layers: 2\n")
                f.write("  • Feed-forward Dimension: 256\n")
                f.write("  • Dropout: 0.1\n")
                f.write("  • Total Parameters: ~5M\n\n")

                f.write("Training Configuration:\n")
                f.write("-" * 100 + "\n")
                f.write("  • Epochs: 25\n")
                f.write("  • Batch Size: 64\n")
                f.write("  • Learning Rate: 0.0003\n")
                f.write("  • Weight Decay: 1e-5\n")
                f.write("  • Mixed Precision: Enabled\n\n")

                f.write("Final Performance (Epoch 25):\n")
                f.write("-" * 100 + "\n")
                f.write(f"  • MSE: {last_epoch.get('mse', 'N/A'):.6f}\n")
                f.write(f"  • RMSE: {last_epoch.get('rmse', 'N/A'):.6f}\n")
                f.write(f"  • MAE: {last_epoch.get('mae', 'N/A'):.6f}\n")
                f.write(f"  • Pearson R: {last_epoch.get('pearson_r', 'N/A'):.6f}\n")
                f.write(f"  • Spearman R: {last_epoch.get('spearman_r', 'N/A'):.6f}\n")
                f.write(f"  • R²: {last_epoch.get('r2', 'N/A'):.6f}\n\n")

        f.write("Electrostatic Potential Predictions:\n")
        f.write("-" * 100 + "\n")
        f.write("  TileFormer predicts 6 electrostatic potential (PSI) features:\n")
        f.write("  • STD_PSI_MIN, STD_PSI_MAX, STD_PSI_MEAN (standard electrostatics)\n")
        f.write("  • ENH_PSI_MIN, ENH_PSI_MAX, ENH_PSI_MEAN (enhanced electrostatics)\n")
        f.write("  All features achieve Pearson R > 0.98 (see ELECTROSTATIC_METRICS.txt)\n\n")

        f.write("Files & Visualizations:\n")
        f.write("-" * 100 + "\n")
        f.write("  • Location: results/TileFormer/\n")
        f.write("  • Performance Summary: PERFORMANCE_SUMMARY.txt\n")
        f.write("  • Electrostatic Metrics: ELECTROSTATIC_METRICS.txt\n")
        f.write("  • Training Curves: comprehensive_metrics.png\n")
        f.write("  • Additional Plots: training_plots/ directory\n\n")

        # Section IV: PhysInformer Models
        f.write("\n" + "=" * 100 + "\n")
        f.write("IV. PHYSINFORMER MODELS - BIOPHYSICAL DESCRIPTOR PREDICTION\n")
        f.write("=" * 100 + "\n\n")

        # Load and summarize each PhysInformer model
        models = {
            'WTC11': {
                'name': 'WTC11 (Human Cardiomyocytes)',
                'cell_line': 'Human induced pluripotent stem cell-derived cardiomyocytes',
                'application': 'Cardiac regulatory sequence analysis'
            },
            'HepG2': {
                'name': 'HepG2 (Human Liver Cells)',
                'cell_line': 'Human hepatocellular carcinoma cells',
                'application': 'Hepatic regulatory analysis'
            },
            'K562': {
                'name': 'K562 (Human Leukemia Cells)',
                'cell_line': 'Human chronic myelogenous leukemia cells',
                'application': 'Hematopoietic regulatory analysis'
            },
            'S2': {
                'name': 'S2 (Drosophila)',
                'cell_line': 'Drosophila melanogaster Schneider 2 cells',
                'application': 'Cross-species regulatory analysis (predicts 2 activities)'
            }
        }

        for cell_type, info in models.items():
            f.write(f"\n{'-' * 100}\n")
            f.write(f"MODEL: {info['name']}\n")
            f.write(f"{'-' * 100}\n\n")

            f.write(f"Cell Line: {info['cell_line']}\n")
            f.write(f"Application: {info['application']}\n\n")

            # Load performance data
            perf_json = f'results/PhysInformer_{cell_type}/parsed_epochs.json'
            if os.path.exists(perf_json):
                with open(perf_json, 'r') as pf:
                    epochs = json.load(pf)
                    first_epoch = epochs[0]
                    last_epoch = epochs[-1]
                    best_epoch = max(epochs, key=lambda x: x.get('val_pearson', -999))

                    f.write("Training Summary:\n")
                    f.write("  " + "-" * 96 + "\n")
                    f.write(f"  • Total Epochs: {len(epochs)}\n")
                    f.write(f"  • Biophysical Features: {536 if cell_type == 'HepG2' else 539 if cell_type == 'WTC11' else 515 if cell_type == 'K562' else 386}\n")
                    f.write(f"  • Batch Size: 32\n")
                    f.write(f"  • Learning Rate: 0.0001\n\n")

                    f.write("Performance Evolution:\n")
                    f.write("  " + "-" * 96 + "\n")
                    f.write(f"  Metric              | Epoch 1    | Final ({last_epoch['epoch']})  | Best (Epoch {best_epoch['epoch']})\n")
                    f.write("  " + "-" * 96 + "\n")

                    # Overall Pearson
                    e1_p = first_epoch.get('val_pearson')
                    last_p = last_epoch.get('val_pearson')
                    best_p = best_epoch.get('val_pearson')
                    f.write(f"  Val Pearson         | {e1_p:.4f}    | {last_p:.4f}        | {best_p:.4f}\n")

                    # Descriptor Mean
                    e1_dm = first_epoch.get('val_desc_mean')
                    last_dm = last_epoch.get('val_desc_mean')
                    best_dm = best_epoch.get('val_desc_mean')
                    f.write(f"  Val Desc Mean       | {e1_dm:.4f}    | {last_dm:.4f}        | {best_dm:.4f}\n")

                    # Descriptor Median
                    e1_dmed = first_epoch.get('val_desc_median')
                    last_dmed = last_epoch.get('val_desc_median')
                    best_dmed = best_epoch.get('val_desc_median')
                    f.write(f"  Val Desc Median     | {e1_dmed:.4f}    | {last_dmed:.4f}        | {best_dmed:.4f}\n")

                    # Loss
                    e1_l = first_epoch.get('val_total_loss')
                    last_l = last_epoch.get('val_total_loss')
                    best_l = best_epoch.get('val_total_loss')
                    f.write(f"  Val Loss            | {e1_l:.2f}    | {last_l:.2f}        | {best_l:.2f}\n\n")

                    # Auxiliary heads
                    f.write("  Auxiliary Head Performance (Final Epoch):\n")
                    f.write("  " + "-" * 96 + "\n")
                    aux_a_p = last_epoch.get('aux_a_pearson')
                    aux_b_p = last_epoch.get('aux_b_pearson')
                    f.write(f"    Head A (Seq+Feat): Pearson = {aux_a_p:.4f}\n")
                    f.write(f"    Head B (Feat Only): Pearson = {aux_b_p:.4f}\n\n")

            # Top/Bottom descriptors
            desc_file = f'results/PhysInformer_{cell_type}/final_descriptor_scores.txt'
            if os.path.exists(desc_file):
                with open(desc_file, 'r') as df:
                    lines = df.readlines()
                    f.write("  Top 5 Best Predicted Descriptors:\n")
                    f.write("  " + "-" * 96 + "\n")
                    in_top_section = False
                    count = 0
                    for line in lines:
                        if 'TOP' in line and 'BEST' in line:
                            in_top_section = True
                            continue
                        if in_top_section and line.strip() and not line.startswith('=') and not line.startswith('-') and not line.startswith('Rank'):
                            if count < 5:
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    desc_name = ' '.join(parts[1:-1])
                                    score = parts[-1]
                                    f.write(f"    {count+1}. {desc_name}: {score}\n")
                                    count += 1
                        if 'BOTTOM' in line:
                            break
                    f.write("\n")

            f.write("  Files & Visualizations:\n")
            f.write("  " + "-" * 96 + "\n")
            f.write(f"    • Location: results/PhysInformer_{cell_type}/\n")
            f.write(f"    • Complete Performance: PERFORMANCE_SUMMARY.txt\n")
            f.write(f"    • Descriptor Scores: final_descriptor_scores.txt\n")
            f.write(f"    • Loss Curves: loss_curves.png\n")
            f.write(f"    • Pearson Evolution: pearson_evolution.png\n")
            f.write(f"    • Training Plots: training_plots/ directory\n\n")

        # Section V: Cross-Model Comparison
        f.write("\n" + "=" * 100 + "\n")
        f.write("V. CROSS-MODEL COMPARISON\n")
        f.write("=" * 100 + "\n\n")

        f.write("Final Validation Performance Comparison:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Model':<15} {'Val Pearson':<15} {'Desc Mean':<15} {'Desc Median':<15} {'# Features':<12}\n")
        f.write("-" * 100 + "\n")

        for cell_type in ['S2', 'WTC11', 'HepG2', 'K562']:
            perf_json = f'results/PhysInformer_{cell_type}/parsed_epochs.json'
            if os.path.exists(perf_json):
                with open(perf_json, 'r') as pf:
                    epochs = json.load(pf)
                    last_epoch = epochs[-1]
                    val_p = last_epoch.get('val_pearson', 0)
                    desc_mean = last_epoch.get('val_desc_mean', 0)
                    desc_median = last_epoch.get('val_desc_median', 0)
                    n_feat = 386 if cell_type == 'S2' else 515 if cell_type == 'K562' else 536 if cell_type == 'HepG2' else 539
                    f.write(f"{cell_type:<15} {val_p:.4f}{'':10} {desc_mean:.4f}{'':10} {desc_median:.4f}{'':10} {n_feat:<12}\n")

        f.write("\n")
        f.write("Key Observations:\n")
        f.write("-" * 100 + "\n")
        f.write("  • S2 (Drosophila) achieves highest validation Pearson despite having fewest features\n")
        f.write("  • All models achieve descriptor-level mean Pearson > 0.83\n")
        f.write("  • Descriptor-level median Pearson > 0.97 across all models (highly accurate predictions)\n")
        f.write("  • Human cell line models (WTC11, HepG2, K562) show similar performance profiles\n\n")

        # Section VI: Conclusions
        f.write("\n" + "=" * 100 + "\n")
        f.write("VI. KEY FINDINGS & CONCLUSIONS\n")
        f.write("=" * 100 + "\n\n")

        f.write("1. MODEL PERFORMANCE\n")
        f.write("-" * 100 + "\n")
        f.write("  • PhysInformer successfully predicts 386-539 biophysical descriptors simultaneously\n")
        f.write("  • Median descriptor-level Pearson > 0.97 indicates highly accurate predictions\n")
        f.write("  • Overall validation Pearson > 0.88 for all cell types\n")
        f.write("  • TileFormer provides effective sequence embeddings (Pearson R > 0.98 for PSI features)\n\n")

        f.write("2. INTERPRETABILITY\n")
        f.write("-" * 100 + "\n")
        f.write("  • Top predicted features: Thermodynamic properties (dG, Tm), PWM scores, G4 potential\n")
        f.write("  • Challenging features: DNA stiffness max energy terms, relative energy metrics\n")
        f.write("  • Auxiliary heads enable direct comparison of sequence-only vs feature-based predictions\n\n")

        f.write("3. CROSS-SPECIES APPLICABILITY\n")
        f.write("-" * 100 + "\n")
        f.write("  • S2 Drosophila model demonstrates framework applicability across species\n")
        f.write("  • Successfully predicts dual activities (developmental + housekeeping enrichment)\n")
        f.write("  • Achieves strong performance despite different experimental design\n\n")

        f.write("4. PRACTICAL APPLICATIONS\n")
        f.write("-" * 100 + "\n")
        f.write("  • Cell-type specific regulatory sequence design and optimization\n")
        f.write("  • Mechanistic understanding through biophysical descriptor analysis\n")
        f.write("  • Variant effect prediction for functional genomics\n")
        f.write("  • Transfer learning potential across cell types and species\n\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF MASTER ANALYSIS SUMMARY\n")
        f.write("=" * 100 + "\n")
        f.write(f"\nGenerated: {os.popen('date').read().strip()}\n")
        f.write(f"Location: {os.path.abspath(output_path)}\n")

    print(f"\nCreated master analysis summary: {output_path}")

def main():
    """Main function."""
    print("\n" + "="*80)
    print("Creating Master Analysis Summary")
    print("="*80)

    print("\n1. Copying analysis PNGs...")
    copy_analysis_pngs()

    print("\n2. Creating master analysis document...")
    create_master_analysis_document()

    print("\n" + "="*80)
    print("MASTER ANALYSIS SUMMARY COMPLETED!")
    print("="*80)
    print("\nAll results organized in: results/")
    print("  • PhysInformer_S2/")
    print("  • PhysInformer_WTC11/")
    print("  • PhysInformer_HepG2/")
    print("  • PhysInformer_K562/")
    print("  • TileFormer/")
    print("  • Analyses/")
    print("    - MASTER_ANALYSIS_SUMMARY.txt")
    print("    - feature_histograms/")
    print("    - feature statistics & correlations\n")

if __name__ == '__main__':
    main()
