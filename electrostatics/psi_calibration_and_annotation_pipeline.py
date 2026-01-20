#!/usr/bin/env python3
"""
Complete ψ_APBS Calibration and Annotation Pipeline

This script:
1. Builds regression models from APBS calibration data
2. Evaluates models and selects the best one
3. Applies the model to annotate large sequence datasets
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class PSIRegressionBuilder:
    """Builds and evaluates regression models from APBS calibration data"""
    
    def __init__(self, calibration_file: str):
        self.df = pd.read_csv(calibration_file, sep='\t')
        self.models = {}
        self.evaluation_results = {}
        self.best_model_name = None
        print(f"Loaded calibration data: {len(self.df)} sequences")
        print(f"Features: {list(self.df.columns)}")
    
    def compute_aic(self, y_true: np.ndarray, y_pred: np.ndarray, 
                   n_params: int) -> float:
        """Compute Akaike Information Criterion"""
        n = len(y_true)
        mse = mean_squared_error(y_true, y_pred)
        # AIC = n * ln(MSE) + 2 * k
        aic = n * np.log(mse) + 2 * n_params
        return aic
    
    def compute_adjusted_r2(self, r2: float, n: int, p: int) -> float:
        """Compute adjusted R-squared"""
        if n <= p + 1:
            return r2
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    def fit_linear_gc_model(self):
        """Fit simple linear model: ψ = a + b*GC_frac"""
        X = self.df[['GC_frac']]
        y = self.df['psi_APBS']
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        r2 = r2_score(y, y_pred)
        adj_r2 = self.compute_adjusted_r2(r2, len(y), 1)
        aic = self.compute_aic(y, y_pred, 2)  # intercept + 1 coef
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=min(5, len(y)), 
                                   scoring='r2')
        
        self.models['linear_gc'] = {
            'model': model,
            'feature_names': ['GC_frac'],
            'coefficients': model.coef_.tolist(),
            'intercept': float(model.intercept_),
            'scaler': None,
            'scaler_type': None
        }
        
        self.evaluation_results['linear_gc'] = {
            'r2': r2,
            'adj_r2': adj_r2,
            'aic': aic,
            'mse': mean_squared_error(y, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_params': 2
        }
        
        print(f"Linear GC model: R²={r2:.3f}, Adj-R²={adj_r2:.3f}, "
              f"AIC={aic:.1f}")
    
    def fit_quadratic_gc_model(self):
        """Fit quadratic model: ψ = a + b*GC + c*GC²"""
        X = self.df[['GC_frac']]
        y = self.df['psi_APBS']
        
        # Create polynomial features manually for better control
        X_poly = np.column_stack([X['GC_frac'], X['GC_frac']**2])
        
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        r2 = r2_score(y, y_pred)
        adj_r2 = self.compute_adjusted_r2(r2, len(y), 2)
        aic = self.compute_aic(y, y_pred, 3)  # intercept + 2 coef
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_poly, y, cv=min(5, len(y)), 
                                   scoring='r2')
        
        self.models['quadratic_gc'] = {
            'model': model,
            'feature_names': ['GC_frac', 'GC_frac^2'],
            'coefficients': model.coef_.tolist(),
            'intercept': float(model.intercept_),
            'scaler': None,
            'scaler_type': None
        }
        
        self.evaluation_results['quadratic_gc'] = {
            'r2': r2,
            'adj_r2': adj_r2,
            'aic': aic,
            'mse': mean_squared_error(y, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_params': 3
        }
        
        print(f"Quadratic GC model: R²={r2:.3f}, Adj-R²={adj_r2:.3f}, "
              f"AIC={aic:.1f}")
    
    def fit_gc_cpg_model(self):
        """Fit model: ψ = a + b*GC + c*CpG + d*GC²"""
        X_base = self.df[['GC_frac', 'CpG_density']]
        y = self.df['psi_APBS']
        
        # Add quadratic GC term
        X = np.column_stack([X_base['GC_frac'], X_base['CpG_density'], 
                            X_base['GC_frac']**2])
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        r2 = r2_score(y, y_pred)
        adj_r2 = self.compute_adjusted_r2(r2, len(y), 3)
        aic = self.compute_aic(y, y_pred, 4)  # intercept + 3 coef
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=min(5, len(y)), 
                                   scoring='r2')
        
        self.models['gc_cpg'] = {
            'model': model,
            'feature_names': ['GC_frac', 'CpG_density', 'GC_frac^2'],
            'coefficients': model.coef_.tolist(),
            'intercept': float(model.intercept_),
            'scaler': None,
            'scaler_type': None
        }
        
        self.evaluation_results['gc_cpg'] = {
            'r2': r2,
            'adj_r2': adj_r2,
            'aic': aic,
            'mse': mean_squared_error(y, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_params': 4
        }
        
        print(f"GC+CpG model: R²={r2:.3f}, Adj-R²={adj_r2:.3f}, "
              f"AIC={aic:.1f}")
    
    def fit_multifeature_model(self):
        """Fit model with all features + scaling"""
        feature_cols = ['GC_frac', 'CpG_density', 'run_frac']
        X_base = self.df[feature_cols]
        y = self.df['psi_APBS']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_base)
        
        # Add quadratic GC term (using scaled GC)
        gc_scaled = X_scaled[:, 0]  # First column is GC_frac
        X = np.column_stack([X_scaled, gc_scaled**2])
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        r2 = r2_score(y, y_pred)
        adj_r2 = self.compute_adjusted_r2(r2, len(y), 4)
        aic = self.compute_aic(y, y_pred, 5)  # intercept + 4 coef
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=min(5, len(y)), 
                                   scoring='r2')
        
        self.models['gc_cpg_run'] = {
            'model': model,
            'feature_names': ['GC_frac', 'CpG_density', 'run_frac', 
                             'GC_frac^2'],
            'coefficients': model.coef_.tolist(),
            'intercept': float(model.intercept_),
            'scaler': scaler,
            'scaler_type': 'standard'
        }
        
        self.evaluation_results['gc_cpg_run'] = {
            'r2': r2,
            'adj_r2': adj_r2,
            'aic': aic,
            'mse': mean_squared_error(y, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_params': 5
        }
        
        print(f"Multi-feature model: R²={r2:.3f}, Adj-R²={adj_r2:.3f}, "
              f"AIC={aic:.1f}")
    
    def select_best_model(self) -> str:
        """Select best model based on adjusted R² and AIC"""
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        results_df = pd.DataFrame(self.evaluation_results).T
        results_df = results_df.round(3)
        print(results_df)
        
        # Penalize models with too many parameters for small dataset
        n_samples = len(self.df)
        for model_name, results in self.evaluation_results.items():
            if results['n_params'] > n_samples / 3:  # Rule of thumb
                print(f"Warning: {model_name} may be overfit "
                      f"({results['n_params']} params, {n_samples} samples)")
        
        # Select based on adjusted R² (higher is better)
        # Could also use AIC (lower is better) as tie-breaker
        best_adj_r2 = max(results_df['adj_r2'])
        best_models = results_df[results_df['adj_r2'] == best_adj_r2]
        
        if len(best_models) > 1:
            # Use AIC as tie-breaker (lower is better)
            self.best_model_name = best_models['aic'].idxmin()
        else:
            self.best_model_name = best_models.index[0]
        
        print(f"\nSelected model: {self.best_model_name}")
        print(f"Adj-R²: {best_adj_r2:.3f}")
        print(f"AIC: {results_df.loc[self.best_model_name, 'aic']:.1f}")
        
        return self.best_model_name
    
    def save_best_model(self, output_file: str):
        """Save the best model to JSON file"""
        if self.best_model_name is None:
            raise ValueError("No model selected. Run select_best_model() first.")
        
        model_info = self.models[self.best_model_name].copy()
        
        # Remove sklearn objects, convert to JSON-serializable format
        del model_info['model']
        if model_info['scaler'] is not None:
            scaler = model_info['scaler']
            model_info['scaler_mean'] = scaler.mean_.tolist()
            model_info['scaler_scale'] = scaler.scale_.tolist()
            del model_info['scaler']
        
        # Add metadata
        model_data = {
            'model_name': self.best_model_name,
            'model_version': '1.0.0',
            'created_date': datetime.now().isoformat(),
            'calibration_data': {
                'n_sequences': len(self.df),
                'sequence_length': '24bp',
                'apbs_method': 'shell_averaging_2-6A'
            },
            'evaluation': self.evaluation_results[self.best_model_name],
            **model_info
        }
        
        with open(output_file, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Best model saved to {output_file}")
        return model_data
    
    def create_diagnostic_plots(self, output_dir: str):
        """Create diagnostic plots for the best model"""
        if self.best_model_name is None:
            raise ValueError("No model selected.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get predictions from best model
        best_model_info = self.models[self.best_model_name]
        model = best_model_info['model']
        
        # Prepare features based on model type
        if self.best_model_name == 'linear_gc':
            X = self.df[['GC_frac']]
        elif self.best_model_name == 'quadratic_gc':
            X = np.column_stack([self.df['GC_frac'], 
                               self.df['GC_frac']**2])
        elif self.best_model_name == 'gc_cpg':
            X = np.column_stack([self.df['GC_frac'], self.df['CpG_density'],
                               self.df['GC_frac']**2])
        elif self.best_model_name == 'gc_cpg_run':
            scaler = best_model_info['scaler']
            X_base = self.df[['GC_frac', 'CpG_density', 'run_frac']]
            X_scaled = scaler.transform(X_base)
            gc_scaled = X_scaled[:, 0]
            X = np.column_stack([X_scaled, gc_scaled**2])
        
        y_true = self.df['psi_APBS']
        y_pred = model.predict(X)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Predicted vs Actual
        ax1 = axes[0, 0]
        ax1.scatter(y_true, y_pred, alpha=0.7, s=60)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax1.set_xlabel('Actual ψ_APBS')
        ax1.set_ylabel('Predicted ψ_APBS')
        ax1.set_title(f'Predicted vs Actual\n'
                     f'R² = {self.evaluation_results[self.best_model_name]["r2"]:.3f}')
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals vs Predicted
        ax2 = axes[0, 1]
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.7, s=60)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Predicted ψ_APBS')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Predicted')
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature importance (coefficients)
        ax3 = axes[1, 0]
        feature_names = best_model_info['feature_names']
        coefficients = best_model_info['coefficients']
        bars = ax3.bar(range(len(feature_names)), coefficients)
        ax3.set_xticks(range(len(feature_names)))
        ax3.set_xticklabels(feature_names, rotation=45, ha='right')
        ax3.set_ylabel('Coefficient Value')
        ax3.set_title('Feature Coefficients')
        ax3.grid(True, alpha=0.3)
        
        # 4. Model comparison
        ax4 = axes[1, 1]
        model_names = list(self.evaluation_results.keys())
        adj_r2_values = [self.evaluation_results[name]['adj_r2'] 
                        for name in model_names]
        bars = ax4.bar(range(len(model_names)), adj_r2_values)
        
        # Highlight best model
        best_idx = model_names.index(self.best_model_name)
        bars[best_idx].set_color('red')
        
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.set_ylabel('Adjusted R²')
        ax4.set_title('Model Comparison')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'model_diagnostics.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Diagnostic plots saved to {plot_file}")
        
        return fig

class PSIAnnotator:
    """Applies fitted ψ_APBS regression model to sequences"""
    
    def __init__(self, model_file: str):
        self.model_data = self.load_model(model_file)
        self.model_name = self.model_data['model_name']
        self.feature_names = self.model_data['feature_names']
        self.coefficients = np.array(self.model_data['coefficients'])
        self.intercept = self.model_data['intercept']
        
        # Load scaling parameters if needed
        self.scaler_type = self.model_data.get('scaler_type', None)
        if self.scaler_type == 'standard':
            self.scaler_mean = np.array(self.model_data['scaler_mean'])
            self.scaler_scale = np.array(self.model_data['scaler_scale'])
        
        print(f"Loaded model: {self.model_name}")
        print(f"Features: {self.feature_names}")
        print(f"Model version: {self.model_data.get('model_version', 'unknown')}")
    
    def load_model(self, model_file: str) -> Dict:
        """Load the saved regression model"""
        with open(model_file, 'r') as f:
            model_data = json.load(f)
        return model_data
    
    def compute_sequence_features(self, sequence: str) -> Dict[str, float]:
        """Compute sequence features for a single sequence"""
        sequence = sequence.upper().replace('N', '').replace(' ', '')
        L = len(sequence)
        
        if L == 0:
            return {'GC_frac': 0.0, 'CpG_density': 0.0, 'run_frac': 0.0}
        
        # GC fraction
        gc_count = sequence.count('G') + sequence.count('C')
        gc_frac = gc_count / L
        
        # CpG density
        cg_count = sequence.count('CG')
        cpg_density = cg_count / (L - 1) if L > 1 else 0.0
        
        # Longest homopolymer run
        max_run = 1
        current_run = 1
        for i in range(1, L):
            if sequence[i] == sequence[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        run_frac = max_run / L
        
        return {
            'GC_frac': gc_frac,
            'CpG_density': cpg_density,
            'run_frac': run_frac
        }
    
    def prepare_features(self, features_dict: Dict[str, float]) -> np.ndarray:
        """Prepare features for model prediction based on model type"""
        if self.model_name == 'linear_gc':
            return np.array([features_dict['GC_frac']])
            
        elif self.model_name == 'quadratic_gc':
            gc = features_dict['GC_frac']
            return np.array([gc, gc**2])
            
        elif self.model_name == 'gc_cpg':
            gc = features_dict['GC_frac']
            cpg = features_dict['CpG_density']
            return np.array([gc, cpg, gc**2])
            
        elif self.model_name == 'gc_cpg_run':
            # Standard scaling + quadratic GC
            base_features = np.array([
                features_dict['GC_frac'],
                features_dict['CpG_density'], 
                features_dict['run_frac']
            ])
            scaled_features = (base_features - self.scaler_mean) / self.scaler_scale
            gc_scaled = scaled_features[0]
            return np.array([scaled_features[0], scaled_features[1], 
                           scaled_features[2], gc_scaled**2])
        
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")
    
    def predict_single(self, sequence: str) -> Tuple[float, Dict[str, float]]:
        """Predict ψ_APBS for a single sequence"""
        features_dict = self.compute_sequence_features(sequence)
        features_array = self.prepare_features(features_dict)
        
        # Linear prediction
        psi_raw = self.intercept + np.dot(self.coefficients, features_array)
        
        return float(psi_raw), features_dict
    
    def predict_batch(self, sequences: List[str], 
                     sequence_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Predict ψ_APBS for a batch of sequences"""
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(sequences))]
        
        results = []
        for i, (seq_id, sequence) in enumerate(zip(sequence_ids, sequences)):
            try:
                psi_raw, features = self.predict_single(sequence)
                
                result = {
                    'sequence_id': seq_id,
                    'sequence_length': len(sequence.replace('N', '')),
                    'psi_raw': psi_raw,
                    **features
                }
                results.append(result)
                
                if (i + 1) % 5000 == 0:
                    print(f"Processed {i + 1:,} sequences...")
                    
            except Exception as e:
                print(f"Error processing sequence {seq_id}: {e}")
                print(f"  Sequence preview: {sequence[:50]}...")
                result = {
                    'sequence_id': seq_id,
                    'sequence_length': len(sequence),
                    'psi_raw': np.nan,
                    'GC_frac': np.nan,
                    'CpG_density': np.nan,
                    'run_frac': np.nan
                }
                results.append(result)
        
        return pd.DataFrame(results)

def load_fasta_sequences(fasta_file: str) -> Tuple[List[str], List[str]]:
    """Load sequences from FASTA file"""
    sequences = []
    sequence_ids = []
    
    current_seq = ""
    current_id = ""
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                    sequence_ids.append(current_id)
                current_id = line[1:].split()[0]
                current_seq = ""
            else:
                current_seq += line.upper()
    
    # Add last sequence
    if current_seq:
        sequences.append(current_seq)
        sequence_ids.append(current_id)
    
    print(f"Loaded {len(sequences):,} sequences from {fasta_file}")
    return sequences, sequence_ids

def load_bed_sequences(bed_file: str, seq_column: int = 6) -> Tuple[List[str], List[str]]:
    """Load sequences from BED file (0-indexed column)"""
    df = pd.read_csv(bed_file, sep='\t')
    
    # Get sequences from specified column
    sequences = df.iloc[:, seq_column].astype(str).tolist()
    
    # Generate IDs from name column if available, otherwise use row index
    if 'name' in df.columns:
        sequence_ids = df['name'].astype(str).tolist()
    elif df.shape[1] > 3:  # Standard BED has name in column 3
        sequence_ids = df.iloc[:, 3].astype(str).tolist()
    else:
        sequence_ids = [f"seq_{i}" for i in range(len(sequences))]
    
    print(f"Loaded {len(sequences):,} sequences from {bed_file}")
    return sequences, sequence_ids

def main():
    parser = argparse.ArgumentParser(
        description="Complete ψ_APBS calibration and annotation pipeline"
    )
    parser.add_argument('--calibration-file', 
                       default='apbs_calculations/psi-calibration.tsv',
                       help='APBS calibration data file')
    parser.add_argument('--output-dir', default='../annotated_data',
                       help='Output directory for models and annotations')
    parser.add_argument('--data-dir', default='../processed_data',
                       help='Directory containing sequence datasets to annotate')
    parser.add_argument('--skip-modeling', action='store_true',
                       help='Skip model building, use existing model')
    parser.add_argument('--model-file', 
                       help='Existing model file (if --skip-modeling)')
    
    args = parser.parse_args()
    
    print("ψ_APBS Complete Calibration and Annotation Pipeline")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Build regression models (unless skipping)
    if not args.skip_modeling:
        print("\nStep 1: Building regression models...")
        print("-" * 40)
        
        builder = PSIRegressionBuilder(args.calibration_file)
        
        # Fit all models
        builder.fit_linear_gc_model()
        builder.fit_quadratic_gc_model()
        builder.fit_gc_cpg_model()
        builder.fit_multifeature_model()
        
        # Select best model
        best_model_name = builder.select_best_model()
        
        # Save best model
        model_file = os.path.join(args.output_dir, 'best_psi_model.json')
        model_data = builder.save_best_model(model_file)
        
        # Create diagnostic plots
        builder.create_diagnostic_plots(args.output_dir)
        
    else:
        if not args.model_file:
            raise ValueError("Must provide --model-file when using --skip-modeling")
        model_file = args.model_file
        print(f"Using existing model: {model_file}")
    
    # Step 2: Annotate datasets
    print("\nStep 2: Annotating sequence datasets...")
    print("-" * 40)
    
    annotator = PSIAnnotator(model_file)
    
    # Define datasets to annotate
    datasets = [
        # DeepSTARR data (FASTA format, 249bp)
        {
            'name': 'DeepSTARR_train',
            'file': os.path.join(args.data_dir, 'DeepSTARR_data/Sequences_Train.fa'),
            'format': 'fasta',
            'output': os.path.join(args.output_dir, 'DeepSTARR_train_psi_annotations.tsv.gz')
        },
        {
            'name': 'DeepSTARR_test',
            'file': os.path.join(args.data_dir, 'DeepSTARR_data/Sequences_Test.fa'),
            'format': 'fasta',
            'output': os.path.join(args.output_dir, 'DeepSTARR_test_psi_annotations.tsv.gz')
        },
        {
            'name': 'DeepSTARR_val',
            'file': os.path.join(args.data_dir, 'DeepSTARR_data/Sequences_Val.fa'),
            'format': 'fasta',
            'output': os.path.join(args.output_dir, 'DeepSTARR_val_psi_annotations.tsv.gz')
        },
        # BED data (BED format, 230bp, sequence in column 7)
        {
            'name': 'train_data',
            'file': os.path.join(args.data_dir, 'train_data_with_sequences.bed'),
            'format': 'bed',
            'output': os.path.join(args.output_dir, 'train_data_psi_annotations.tsv.gz')
        },
        {
            'name': 'test_data',
            'file': os.path.join(args.data_dir, 'test_data_with_sequences.bed'),
            'format': 'bed',
            'output': os.path.join(args.output_dir, 'test_data_psi_annotations.tsv.gz')
        },
        {
            'name': 'valid_data',
            'file': os.path.join(args.data_dir, 'valid_data_with_sequences.bed'),
            'format': 'bed',
            'output': os.path.join(args.output_dir, 'valid_data_psi_annotations.tsv.gz')
        }
    ]
    
    # Process each dataset
    for dataset in datasets:
        print(f"\nProcessing {dataset['name']}...")
        
        if not os.path.exists(dataset['file']):
            print(f"  Warning: {dataset['file']} not found, skipping...")
            continue
        
        try:
            # Load sequences
            if dataset['format'] == 'fasta':
                sequences, sequence_ids = load_fasta_sequences(dataset['file'])
            elif dataset['format'] == 'bed':
                sequences, sequence_ids = load_bed_sequences(dataset['file'])
            else:
                print(f"  Unknown format: {dataset['format']}, skipping...")
                continue
            
            # Annotate sequences
            results_df = annotator.predict_batch(sequences, sequence_ids)
            
            # Add dataset metadata
            results_df['dataset'] = dataset['name']
            results_df['psi_model_version'] = annotator.model_data.get('model_version', '1.0.0')
            
            # Reorder columns
            column_order = [
                'sequence_id', 'dataset', 'sequence_length', 
                'GC_frac', 'CpG_density', 'run_frac', 'psi_raw',
                'psi_model_version'
            ]
            results_df = results_df[[col for col in column_order if col in results_df.columns]]
            
            # Save results
            results_df.to_csv(dataset['output'], sep='\t', index=False, compression='gzip')
            print(f"  Saved {len(results_df):,} annotations to {dataset['output']}")
            
            # Print summary stats
            psi_stats = results_df['psi_raw'].describe()
            print(f"  ψ_raw range: [{psi_stats['min']:.3f}, {psi_stats['max']:.3f}]")
            print(f"  ψ_raw mean ± std: {psi_stats['mean']:.3f} ± {psi_stats['std']:.3f}")
            
        except Exception as e:
            print(f"  Error processing {dataset['name']}: {e}")
            continue
    
    print(f"\nPipeline complete! Results saved to {args.output_dir}/")
    print("\nFiles created:")
    print(f"  - best_psi_model.json (fitted regression model)")
    print(f"  - model_diagnostics.png (evaluation plots)")
    print(f"  - *_psi_annotations.tsv.gz (sequence annotations)")

if __name__ == "__main__":
    main() 