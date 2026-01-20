import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import logging

class SingleFeatureMetricsCalculator:
    """Calculate metrics for single feature prediction (like thermo_dG_p25)"""
    
    def __init__(self, target_feature: str):
        self.target_feature = target_feature
        
    def calculate_metrics(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for single feature prediction
        
        Args:
            predictions: Dict with 'target_feature' predictions
            targets: Dict with target feature values
            
        Returns:
            Dict with all metrics
        """
        metrics = {}
        
        # Convert to numpy for metric calculations
        target_preds = predictions['target_feature'].cpu().numpy()  # [batch_size]
        target_values = targets[self.target_feature].cpu().numpy()  # [batch_size]
        
        # Calculate main task metrics (direct prediction vs target)
        main_metrics = self._calculate_regression_metrics(
            target_preds, target_values, prefix='main'
        )
        metrics.update(main_metrics)
        
        
        return metrics
    
    def _calculate_regression_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        prefix: str = 'main'
    ) -> Dict[str, float]:
        """Calculate standard regression metrics"""
        metrics = {}
        
        # Basic regression metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        
        # Correlation metrics (if valid)
        if np.std(predictions) > 1e-8 and np.std(targets) > 1e-8:
            try:
                pearson_r, pearson_p = pearsonr(predictions, targets)
                spearman_r, spearman_p = spearmanr(predictions, targets)
                r2 = r2_score(targets, predictions)
                
                # Check for NaN values
                if np.isnan(pearson_r):
                    pearson_r = 0.0
                if np.isnan(spearman_r):
                    spearman_r = 0.0
                if np.isnan(r2):
                    r2 = 0.0
                    
            except Exception:
                pearson_r = 0.0
                spearman_r = 0.0
                r2 = 0.0
        else:
            pearson_r = 0.0
            spearman_r = 0.0
            r2 = 0.0
        
        metrics.update({
            f'{prefix}_mse': mse,
            f'{prefix}_mae': mae,
            f'{prefix}_rmse': rmse,
            f'{prefix}_pearson': pearson_r,
            f'{prefix}_spearman': spearman_r,
            f'{prefix}_r2': r2
        })
        
        return metrics
    
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int, split: str, logger):
        """Log metrics in a structured way"""
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch} - {split.upper()} METRICS - {self.target_feature}")
        logger.info(f"{'='*60}")
        
        # Main task metrics (window aggregation)
        logger.info(f"\nMain Task - {self.target_feature} Prediction:")
        logger.info(f"  MSE: {metrics['main_mse']:.6f}")
        logger.info(f"  MAE: {metrics['main_mae']:.6f}")
        logger.info(f"  RMSE: {metrics['main_rmse']:.6f}")
        logger.info(f"  Pearson: {metrics['main_pearson']:.4f}")
        logger.info(f"  Spearman: {metrics['main_spearman']:.4f}")
        logger.info(f"  R²: {metrics['main_r2']:.4f}")
        
        
        
        logger.info(f"\n{'='*60}")

def calculate_single_feature_loss(
    predictions: Dict[str, torch.Tensor], 
    targets: Dict[str, torch.Tensor],
    target_feature: str = 'entropy_mi_d5'
) -> Dict[str, torch.Tensor]:
    """
    Calculate loss for single feature prediction
    
    Args:
        predictions: Dict with 'target_feature' predictions
        targets: Dict with target feature values
        target_feature: Name of the target feature
        
    Returns:
        Dict with individual losses and total loss
    """
    # Main task: direct prediction of target feature
    target_preds = predictions['target_feature']  # [batch_size]
    target_values = targets[target_feature]  # [batch_size]
    
    # Main task loss: MSE between prediction and target
    main_loss = torch.nn.functional.mse_loss(target_preds, target_values)
    
    # No variance penalty needed for direct prediction
    variance_penalty = torch.tensor(0.0, device=target_preds.device)
    
    # Total loss is just main loss
    total_loss = main_loss
    
    return {
        'total_loss': total_loss,
        'main_loss': main_loss,
    }