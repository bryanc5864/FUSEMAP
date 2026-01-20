#!/usr/bin/env python3
"""
Comprehensive model fitting for electrostatic potential prediction.
Tests multiple models and feature combinations on expanded calibration dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV, LeaveOneOut
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_calibration_data(filename="psi_calibration_expanded_corrected.tsv"):
    """Load the calibration dataset."""
    df = pd.read_csv(filename, sep='\t')
    
    # Remove any sequences with NaN potentials
    df = df.dropna(subset=['psi_APBS'])
    
    print(f"Loaded {len(df)} sequences")
    print(f"Potential range: {df['psi_APBS'].min():.3f} to {df['psi_APBS'].max():.3f}")
    
    return df

def create_feature_sets(df):
    """Create different feature combinations for testing."""
    
    feature_sets = {
        'GC_only': ['GC_frac'],
        'CpG_only': ['CpG_density'], 
        'run_only': ['run_frac'],
        'mgw_only': ['mgw_avg'],
        'GC_CpG': ['GC_frac', 'CpG_density'],
        'GC_run': ['GC_frac', 'run_frac'],
        'all_features': ['GC_frac', 'CpG_density', 'run_frac', 'mgw_avg']
    }
    
    return feature_sets

def evaluate_model(model, X, y, use_loocv=False):
    """Evaluate model performance using cross-validation."""
    
    # Choose CV strategy
    if use_loocv and len(X) <= 100:
        cv = LeaveOneOut()
        cv_folds = LeaveOneOut().get_n_splits(X)
    else:
        cv = 5
        cv_folds = 5
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    cv_mae = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    
    # Fit on full data for additional metrics
    model.fit(X, y)
    y_pred = model.predict(X)
    
    train_r2 = r2_score(y, y_pred)
    train_mae = mean_absolute_error(y, y_pred)
    train_rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    return {
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'cv_mae_mean': cv_mae.mean(),
        'cv_mae_std': cv_mae.std(),
        'train_r2': train_r2,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'cv_folds': cv_folds
    }

def fit_linear_models(X, y):
    """Fit basic linear regression models with consistent scaling."""
    
    models = {}
    
    # Simple linear regression with scaling
    models['Linear'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    # Ridge regression with CV and scaling
    models['Ridge'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5))
    ])
    
    # Elastic Net with CV and scaling
    models['ElasticNet'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNetCV(alphas=np.logspace(-3, 1, 10), 
                              l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                              cv=5, max_iter=2000, random_state=42))
    ])
    
    return models

def fit_polynomial_ridge(X, y, degree=2):
    """Fit ridge-penalized polynomial regression."""
    
    # Create polynomial features pipeline with scaling
    poly_ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5))
    ])
    
    return {'PolyRidge': poly_ridge}

def fit_svr_models(X, y):
    """Fit Support Vector Regression with RBF kernel."""
    
    param_grid = {
        'model__C': [1, 10, 100, 1000],
        'model__gamma': [0.01, 0.1, 1.0],
        'model__epsilon': [0.01, 0.1, 0.2]
    }
    
    svr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='rbf'))
    ])
    
    svr = GridSearchCV(svr_pipeline, param_grid, cv=5, scoring='r2')
    
    return {'SVR_RBF': svr}

def fit_random_forest(X, y):
    """Fit Random Forest with shallow trees."""
    
    # Random Forest doesn't need scaling, but use pipeline for consistency
    rf_pipeline = Pipeline([
        ('model', RandomForestRegressor(
            n_estimators=200, 
            max_depth=3, 
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        ))
    ])
    
    return {'RandomForest': rf_pipeline}

def fit_gaussian_process(X, y):
    """Fit Gaussian Process Regression."""
    
    # Define kernel
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    
    gp_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GaussianProcessRegressor(
            kernel=kernel,
            random_state=42,
            alpha=1e-10,
            normalize_y=True
        ))
    ])
    
    return {'GaussianProcess': gp_pipeline}

def run_comprehensive_evaluation(df, use_loocv=False):
    """Run comprehensive model evaluation."""
    
    feature_sets = create_feature_sets(df)
    y = df['psi_APBS'].values
    
    results = []
    
    for feature_name, features in feature_sets.items():
        print(f"\n{'='*60}")
        print(f"Testing feature set: {feature_name}")
        print(f"Features: {features}")
        print(f"{'='*60}")
        
        X = df[features].values
        
        # Test different model types
        all_models = {}
        
        # Linear models (now with consistent scaling)
        all_models.update(fit_linear_models(X, y))
        
        # Polynomial Ridge
        all_models.update(fit_polynomial_ridge(X, y))
        
        # SVR
        all_models.update(fit_svr_models(X, y))
        
        # Random Forest
        all_models.update(fit_random_forest(X, y))
        
        # Gaussian Process (skip if too many features to avoid O(N³) scaling)
        if len(features) <= 2:  # Only for simple feature sets
            all_models.update(fit_gaussian_process(X, y))
        
        # Evaluate each model
        for model_name, model in all_models.items():
            print(f"\nEvaluating {model_name}...")
            
            try:
                metrics = evaluate_model(model, X, y, use_loocv=use_loocv)
                
                result = {
                    'feature_set': feature_name,
                    'model': model_name,
                    'n_features': len(features),
                    **metrics
                }
                results.append(result)
                
                print(f"  CV R²: {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")
                print(f"  CV MAE: {metrics['cv_mae_mean']:.3f} ± {metrics['cv_mae_std']:.3f}")
                print(f"  Train R²: {metrics['train_r2']:.3f}")
                print(f"  CV folds: {metrics['cv_folds']}")
                
                # Show best parameters for GridSearch models
                if hasattr(model, 'best_params_'):
                    print(f"  Best params: {model.best_params_}")
                    print(f"  Best CV score: {model.best_score_:.3f}")
                
                # Show feature importance for RF
                if model_name == 'RandomForest':
                    rf_model = model.named_steps['model']
                    importances = rf_model.feature_importances_
                    for feat, imp in zip(features, importances):
                        print(f"    {feat}: {imp:.3f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    return pd.DataFrame(results)

def plot_results(results_df):
    """Plot model comparison results."""
    
    # Import seaborn only when needed
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn not available, creating basic matplotlib plots")
        sns = None
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    if sns is not None:
        # Plot 1: CV R² by model type
        ax1 = axes[0, 0]
        sns.boxplot(data=results_df, x='model', y='cv_r2_mean', ax=ax1)
        ax1.set_title('Cross-Validation R² by Model Type')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Plot 2: CV R² by feature set
        ax2 = axes[0, 1]
        sns.boxplot(data=results_df, x='feature_set', y='cv_r2_mean', ax=ax2)
        ax2.set_title('Cross-Validation R² by Feature Set')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # Plot 3: CV MAE by model type
        ax3 = axes[1, 0]
        sns.boxplot(data=results_df, x='model', y='cv_mae_mean', ax=ax3)
        ax3.set_title('Cross-Validation MAE by Model Type')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    else:
        # Fallback to basic matplotlib plots
        ax1 = axes[0, 0]
        results_df.boxplot(column='cv_r2_mean', by='model', ax=ax1)
        ax1.set_title('CV R² by Model')
        
        ax2 = axes[0, 1]
        results_df.boxplot(column='cv_r2_mean', by='feature_set', ax=ax2)
        ax2.set_title('CV R² by Features')
        
        ax3 = axes[1, 0]
        results_df.boxplot(column='cv_mae_mean', by='model', ax=ax3)
        ax3.set_title('CV MAE by Model')
    
    # Plot 4: R² vs MAE trade-off
    ax4 = axes[1, 1]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        ax4.scatter(model_data['cv_mae_mean'], model_data['cv_r2_mean'], 
                   label=model, alpha=0.7, s=60)
    ax4.set_xlabel('CV MAE')
    ax4.set_ylabel('CV R²')
    ax4.set_title('R² vs MAE Trade-off')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function."""
    
    print("Comprehensive Electrostatic Potential Model Evaluation")
    print("="*60)
    
    # Load data
    df = load_calibration_data()
    
    # Decide on CV strategy based on dataset size
    use_loocv = len(df) <= 100
    cv_strategy = "Leave-One-Out" if use_loocv else "5-fold"
    print(f"Using {cv_strategy} cross-validation")
    
    # Run comprehensive evaluation
    results_df = run_comprehensive_evaluation(df, use_loocv=use_loocv)
    
    # Save results
    results_df.to_csv('model_evaluation_results.tsv', sep='\t', index=False)
    print(f"\nResults saved to: model_evaluation_results.tsv")
    
    # Show top performers
    print("\nTop 10 Models by CV R²:")
    top_models = results_df.nlargest(10, 'cv_r2_mean')[
        ['feature_set', 'model', 'cv_r2_mean', 'cv_r2_std', 'cv_mae_mean']
    ]
    print(top_models.to_string(index=False))
    
    # Plot results
    plot_results(results_df)
    
    print(f"\nEvaluation complete! Tested {len(results_df)} model/feature combinations.")

if __name__ == "__main__":
    main() 