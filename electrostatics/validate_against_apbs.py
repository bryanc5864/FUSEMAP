#!/usr/bin/env python3
"""
Proper validation: Compare predicted electrostatic potential against actual APBS calculations.
Sample 100 random sequences, run full APBS pipeline, compare predicted vs APBS-calculated ψ.
"""

import os
import pandas as pd
import numpy as np
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class APBSValidator:
    """Validates electrostatic potential predictions against APBS calculations."""
    
    def __init__(self):
        """Initialize validator."""
        # Our calibrated model: ψ = -2.879 - 0.088 × GC_frac
        self.psi_intercept = -2.879
        self.psi_slope = -0.088
        
        # Expected performance from calibration
        self.expected_r2 = 0.425
        self.expected_mae = 0.020  # kT/e
        
        # Setup validation directory
        self.validation_dir = "validation_apbs"
        os.makedirs(self.validation_dir, exist_ok=True)
        
        print("APBS Validator initialized")
        print(f"Model: ψ = {self.psi_intercept:.3f} + {self.psi_slope:.3f} × GC_frac")
        print(f"Expected: R² = {self.expected_r2:.3f}, MAE = {self.expected_mae:.3f} kT/e")
    
    def compute_gc_fraction(self, sequence: str) -> float:
        """Compute GC fraction for prediction."""
        sequence = sequence.upper().strip()
        if len(sequence) == 0:
            return 0.0
        
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)
    
    def predict_psi(self, gc_frac: float) -> float:
        """Predict electrostatic potential using our model."""
        return self.psi_intercept + self.psi_slope * gc_frac
    
    def sample_sequences(self, dataset_paths: List[str], n_samples: int = 100) -> List[Dict]:
        """Sample random sequences from annotated datasets."""
        
        all_sequences = []
        
        for dataset_path in dataset_paths:
            if not os.path.exists(dataset_path):
                continue
                
            try:
                df = pd.read_csv(dataset_path, sep='\t')
                print(f"Loading from {os.path.basename(dataset_path)}: {len(df)} entries")
                
                # Find sequence column
                seq_col = None
                for col in df.columns:
                    if 'sequence' in col.lower():
                        seq_col = col
                        break
                
                if seq_col is None:
                    continue
                
                # Extract sequences
                for idx, row in df.iterrows():
                    if pd.notna(row[seq_col]) and len(str(row[seq_col])) > 100:
                        all_sequences.append({
                            'id': f"{os.path.basename(dataset_path).split('_')[0]}_{idx}",
                            'sequence': str(row[seq_col]).upper().strip(),
                            'dataset': os.path.basename(dataset_path),
                            'original_psi': row.get('psi_predicted', np.nan)
                        })
                        
                        if len(all_sequences) >= n_samples * 3:  # Get extra for sampling
                            break
                    
            except Exception as e:
                print(f"Error loading {dataset_path}: {e}")
        
        # Random sample
        if len(all_sequences) < n_samples:
            print(f"Warning: Only {len(all_sequences)} sequences available")
            return all_sequences
        
        sampled = random.sample(all_sequences, n_samples)
        print(f"Sampled {len(sampled)} sequences for APBS validation")
        
        return sampled
    
    def create_validation_panel(self, sequences: List[Dict]) -> str:
        """Create validation panel TSV file."""
        
        # Create calibration_panel directory structure that scripts expect
        panel_dir = os.path.join(self.validation_dir, "calibration_panel")
        os.makedirs(panel_dir, exist_ok=True)
        
        panel_file = os.path.join(panel_dir, "calibration_panel_psi.tsv")
        
        with open(panel_file, 'w') as f:
            f.write("id\tseq\tpredicted_psi\tgc_frac\n")
            
            for seq_data in sequences:
                seq_id = seq_data['id']
                sequence = seq_data['sequence']
                gc_frac = self.compute_gc_fraction(sequence)
                predicted_psi = self.predict_psi(gc_frac)
                
                f.write(f"{seq_id}\t{sequence}\t{predicted_psi:.6f}\t{gc_frac:.6f}\n")
        
        print(f"Validation panel created: {panel_file}")
        return panel_file
    
    def run_apbs_pipeline(self, panel_file: str) -> bool:
        """Run full APBS pipeline on validation sequences."""
        
        print("\nRunning APBS pipeline for validation...")
        
        # 1. Build PDB structures
        print("Step 1: Building PDB structures...")
        try:
            result = subprocess.run([
                'python', 'build_panel_pdbs.py'
            ], cwd=self.validation_dir, capture_output=True, text=True, timeout=1800)
            
            if result.returncode != 0:
                print(f"PDB building failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error building PDBs: {e}")
            return False
        
        # 2. Convert to PQR
        print("Step 2: Converting to PQR...")
        try:
            result = subprocess.run([
                'python', '../convert_to_pqr.py'
            ], cwd=self.validation_dir, capture_output=True, text=True, timeout=1800)
            
            if result.returncode != 0:
                print(f"PQR conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error converting to PQR: {e}")
            return False
        
        # 3. Run APBS calculations
        print("Step 3: Running APBS calculations...")
        try:
            result = subprocess.run([
                'python', '../run_apbs_calibration.py'
            ], cwd=self.validation_dir, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                print(f"APBS calculations failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error running APBS: {e}")
            return False
        
        # 4. Extract potential features
        print("Step 4: Extracting potential features...")
        try:
            result = subprocess.run([
                'python', '../extract_potential_features.py'
            ], cwd=self.validation_dir, capture_output=True, text=True, timeout=1800)
            
            if result.returncode != 0:
                print(f"Feature extraction failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error extracting features: {e}")
            return False
        
        print("APBS pipeline completed successfully!")
        return True
    
    def compare_predictions(self) -> Dict[str, float]:
        """Compare predicted vs APBS-calculated electrostatic potentials."""
        
        # Load APBS results
        apbs_results_file = os.path.join(self.validation_dir, "psi_calibration_expanded_corrected.tsv")
        
        if not os.path.exists(apbs_results_file):
            print(f"APBS results file not found: {apbs_results_file}")
            return {}
        
        try:
            apbs_df = pd.read_csv(apbs_results_file, sep='\t')
            print(f"Loaded APBS results for {len(apbs_df)} sequences")
            
        except Exception as e:
            print(f"Error loading APBS results: {e}")
            return {}
        
        # Load validation panel with predictions
        panel_file = os.path.join(self.validation_dir, "calibration_panel", "calibration_panel_psi.tsv")
        try:
            panel_df = pd.read_csv(panel_file, sep='\t')
            
        except Exception as e:
            print(f"Error loading validation panel: {e}")
            return {}
        
        # Merge data
        merged_df = pd.merge(panel_df, apbs_df, on='id', how='inner')
        
        if len(merged_df) == 0:
            print("No matching sequences between predictions and APBS results")
            return {}
        
        print(f"Successfully matched {len(merged_df)} sequences")
        
        # Calculate metrics
        predicted_psi = merged_df['predicted_psi'].values
        apbs_psi = merged_df['psi_APBS'].values
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(predicted_psi) | np.isnan(apbs_psi))
        predicted_psi = predicted_psi[valid_mask]
        apbs_psi = apbs_psi[valid_mask]
        
        if len(predicted_psi) == 0:
            print("No valid predictions for comparison")
            return {}
        
        # Calculate metrics
        mae = np.mean(np.abs(predicted_psi - apbs_psi))
        rmse = np.sqrt(np.mean((predicted_psi - apbs_psi) ** 2))
        
        if len(predicted_psi) > 1:
            correlation = np.corrcoef(predicted_psi, apbs_psi)[0, 1]
            r_squared = correlation ** 2
        else:
            correlation = np.nan
            r_squared = np.nan
        
        # Ranges and statistics
        pred_range = (predicted_psi.min(), predicted_psi.max())
        apbs_range = (apbs_psi.min(), apbs_psi.max())
        
        metrics = {
            'n_sequences': len(predicted_psi),
            'mae': mae,
            'rmse': rmse,
            'r_squared': r_squared,
            'correlation': correlation,
            'predicted_range': pred_range,
            'apbs_range': apbs_range,
            'predicted_mean': predicted_psi.mean(),
            'apbs_mean': apbs_psi.mean(),
            'predicted_std': predicted_psi.std(),
            'apbs_std': apbs_psi.std()
        }
        
        # Save comparison data
        comparison_df = merged_df[['id', 'sequence', 'predicted_psi', 'psi_APBS', 'gc_frac', 'GC_frac']].copy()
        comparison_df['error'] = comparison_df['predicted_psi'] - comparison_df['psi_APBS']
        comparison_df['abs_error'] = np.abs(comparison_df['error'])
        
        comparison_file = os.path.join(self.validation_dir, "prediction_vs_apbs_comparison.tsv")
        comparison_df.to_csv(comparison_file, sep='\t', index=False)
        print(f"Detailed comparison saved to: {comparison_file}")
        
        return metrics, comparison_df
    
    def plot_validation_results(self, comparison_df: pd.DataFrame, metrics: Dict[str, float]):
        """Plot validation results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Predicted vs APBS ψ
        axes[0, 0].scatter(comparison_df['predicted_psi'], comparison_df['psi_APBS'], alpha=0.7)
        min_psi = min(comparison_df['predicted_psi'].min(), comparison_df['psi_APBS'].min())
        max_psi = max(comparison_df['predicted_psi'].max(), comparison_df['psi_APBS'].max())
        axes[0, 0].plot([min_psi, max_psi], [min_psi, max_psi], 'r--', label='Perfect agreement')
        axes[0, 0].set_xlabel('Predicted ψ (kT/e)')
        axes[0, 0].set_ylabel('APBS ψ (kT/e)')
        axes[0, 0].set_title(f'Predicted vs APBS\nMAE = {metrics["mae"]:.4f} kT/e, R² = {metrics["r_squared"]:.3f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error distribution
        axes[0, 1].hist(comparison_df['error'], bins=15, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', label='Zero error')
        axes[0, 1].set_xlabel('Prediction Error (kT/e)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title(f'Error Distribution\nMean = {comparison_df["error"].mean():.4f} kT/e')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. GC fraction vs both predictions
        gc_frac = comparison_df['gc_frac']
        axes[1, 0].scatter(gc_frac, comparison_df['predicted_psi'], alpha=0.7, label='Predicted')
        axes[1, 0].scatter(gc_frac, comparison_df['psi_APBS'], alpha=0.7, label='APBS')
        # Add model line
        gc_range = np.linspace(gc_frac.min(), gc_frac.max(), 100)
        psi_model = self.psi_intercept + self.psi_slope * gc_range
        axes[1, 0].plot(gc_range, psi_model, 'k-', label=f'Model: ψ = {self.psi_intercept:.3f} + {self.psi_slope:.3f} × GC')
        axes[1, 0].set_xlabel('GC Fraction')
        axes[1, 0].set_ylabel('ψ (kT/e)')
        axes[1, 0].set_title('Model vs APBS vs GC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Absolute error vs GC fraction
        axes[1, 1].scatter(gc_frac, comparison_df['abs_error'], alpha=0.7)
        axes[1, 1].set_xlabel('GC Fraction')
        axes[1, 1].set_ylabel('Absolute Error (kT/e)')
        axes[1, 1].set_title('Error vs GC Content')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.validation_dir, "validation_against_apbs.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Validation plots saved to: {plot_file}")
        
        return fig

def main():
    """Main validation function."""
    
    print("Electrostatic Potential Model Validation Against APBS")
    print("="*60)
    
    # Initialize validator
    validator = APBSValidator()
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Define dataset paths (only TSV files have sequences easily accessible)
    dataset_paths = [
        "annotated_datasets/train_data_with_sequences_psi_annotated.tsv",
        "annotated_datasets/test_data_with_sequences_psi_annotated.tsv",
        "annotated_datasets/valid_data_with_sequences_psi_annotated.tsv"
    ]
    
    # Sample sequences
    print("\nSampling 100 random sequences for APBS validation...")
    sequences = validator.sample_sequences(dataset_paths, n_samples=100)
    
    if len(sequences) == 0:
        print("No sequences found for validation")
        return
    
    # Create validation panel
    panel_file = validator.create_validation_panel(sequences)
    
    # Copy necessary scripts to validation directory
    print("\nCopying required scripts...")
    scripts_to_copy = [
        'build_panel_pdbs.py',
        'convert_to_pqr.py', 
        'run_apbs_calibration.py',
        'extract_potential_features.py',
        'template_lpbe.in'
    ]
    
    for script in scripts_to_copy:
        if os.path.exists(script):
            subprocess.run(['cp', script, validator.validation_dir])
    
    # Run APBS pipeline
    print(f"\nRunning full APBS pipeline on {len(sequences)} sequences...")
    print("This may take 30-60 minutes depending on sequence complexity...")
    
    start_time = time.time()
    success = validator.run_apbs_pipeline(panel_file)
    elapsed_time = time.time() - start_time
    
    if not success:
        print("APBS pipeline failed. Cannot complete validation.")
        return
    
    print(f"APBS pipeline completed in {elapsed_time/60:.1f} minutes")
    
    # Compare predictions
    print("\nComparing predictions against APBS results...")
    result = validator.compare_predictions()
    
    if not result:
        print("Comparison failed")
        return
    
    metrics, comparison_df = result
    
    # Print results
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    print(f"Number of sequences: {metrics['n_sequences']}")
    print(f"MAE: {metrics['mae']:.4f} kT/e")
    print(f"RMSE: {metrics['rmse']:.4f} kT/e")
    print(f"R² correlation: {metrics['r_squared']:.4f}")
    print(f"Pearson correlation: {metrics['correlation']:.4f}")
    print(f"Predicted range: {metrics['predicted_range'][0]:.4f} to {metrics['predicted_range'][1]:.4f} kT/e")
    print(f"APBS range: {metrics['apbs_range'][0]:.4f} to {metrics['apbs_range'][1]:.4f} kT/e")
    print(f"Predicted mean ± std: {metrics['predicted_mean']:.4f} ± {metrics['predicted_std']:.4f} kT/e")
    print(f"APBS mean ± std: {metrics['apbs_mean']:.4f} ± {metrics['apbs_std']:.4f} kT/e")
    
    # Compare to expected performance
    print(f"\nComparison to Expected Performance:")
    print(f"Expected MAE: {validator.expected_mae:.3f} kT/e")
    print(f"Observed MAE: {metrics['mae']:.4f} kT/e")
    print(f"Expected R²: {validator.expected_r2:.3f}")
    print(f"Observed R²: {metrics['r_squared']:.4f}")
    
    # Performance assessment
    mae_ratio = metrics['mae'] / validator.expected_mae
    if mae_ratio <= 1.0:
        mae_assessment = "GOOD (at or better than expected)"
    elif mae_ratio <= 1.5:
        mae_assessment = "ACCEPTABLE (somewhat worse than expected)"
    else:
        mae_assessment = "POOR (much worse than expected)"
    
    print(f"MAE Assessment: {mae_assessment} (ratio: {mae_ratio:.2f})")
    
    # Generate plots
    validator.plot_validation_results(comparison_df, metrics)
    
    print(f"\n🎯 CONCLUSION: Model shows {mae_assessment.lower()}")
    print(f"Validation completed in {elapsed_time/60:.1f} minutes")

if __name__ == "__main__":
    main() 