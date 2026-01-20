import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import logging

class MetricsCalculator:
    """Calculate comprehensive metrics for PhysInformer predictions"""
    
    def __init__(self, feature_names: Dict[str, List[str]]):
        self.feature_names = feature_names
        self.descriptor_names = feature_names['descriptors']
        
    def calculate_metrics(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics
        
        Args:
            predictions: Dict with 'descriptors' predictions
            targets: Dict with 'descriptors' targets
            
        Returns:
            Dict with all metrics
        """
        metrics = {}
        
        # Convert to numpy for metric calculations
        pred_desc = predictions['descriptors'].cpu().numpy()
        target_desc = targets['descriptors'].cpu().numpy()
        
        # Calculate metrics for descriptors
        desc_metrics = self._calculate_feature_metrics(pred_desc, target_desc, 'descriptors')
        metrics.update(desc_metrics)
        
        # Calculate overall metrics (only descriptors)
        overall_metrics = self._calculate_overall_metrics(pred_desc, target_desc)
        metrics.update(overall_metrics)
        
        # Calculate per-feature losses if available
        if 'per_feature_losses' in predictions:
            loss_metrics = self._calculate_loss_metrics(predictions['per_feature_losses'])
            metrics.update(loss_metrics)
        
        return metrics
    
    def _calculate_feature_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray, 
        prefix: str
    ) -> Dict[str, float]:
        """Calculate metrics for descriptors"""
        metrics = {}
        
        # Per-feature metrics
        n_features = predictions.shape[1]
        pearson_scores = []
        spearman_scores = []
        r2_scores = []
        mse_scores = []
        mae_scores = []
        
        for i in range(n_features):
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            
            # Skip constant features for correlation metrics
            target_std = np.std(target_i)
            pred_std = np.std(pred_i)
            
            # Only calculate correlation-based metrics if both have variance
            if target_std > 1e-8 and pred_std > 1e-8:
                try:
                    pearson_i, _ = pearsonr(pred_i, target_i)
                    spearman_i, _ = spearmanr(pred_i, target_i)
                    
                    # R² can be very negative if predictions are bad
                    # Clip to reasonable range for averaging
                    r2_i = r2_score(target_i, pred_i)
                    r2_i = max(r2_i, -1.0)  # Clip at -1 for bad predictions
                    
                    # Only add if not NaN
                    if not np.isnan(pearson_i):
                        pearson_scores.append(pearson_i)
                    if not np.isnan(spearman_i):
                        spearman_scores.append(spearman_i)
                    if not np.isnan(r2_i):
                        r2_scores.append(r2_i)
                except:
                    # Skip this feature if correlation fails
                    pass
            
            # MSE and MAE can always be calculated
            mse_i = mean_squared_error(target_i, pred_i)
            mae_i = mean_absolute_error(target_i, pred_i)
            mse_scores.append(mse_i)
            mae_scores.append(mae_i)
        
        # Average metrics
        metrics[f'{prefix}_pearson_mean'] = np.mean(pearson_scores) if pearson_scores else 0.0
        metrics[f'{prefix}_pearson_std'] = np.std(pearson_scores) if pearson_scores else 0.0
        metrics[f'{prefix}_pearson_min'] = np.min(pearson_scores) if pearson_scores else 0.0
        metrics[f'{prefix}_pearson_max'] = np.max(pearson_scores) if pearson_scores else 0.0
        metrics[f'{prefix}_pearson_median'] = np.median(pearson_scores) if pearson_scores else 0.0
        
        metrics[f'{prefix}_spearman_mean'] = np.mean(spearman_scores) if spearman_scores else 0.0
        metrics[f'{prefix}_spearman_std'] = np.std(spearman_scores) if spearman_scores else 0.0
        metrics[f'{prefix}_r2_mean'] = np.mean(r2_scores) if r2_scores else 0.0
        metrics[f'{prefix}_r2_std'] = np.std(r2_scores) if r2_scores else 0.0
        metrics[f'{prefix}_mse_mean'] = np.mean(mse_scores)
        metrics[f'{prefix}_mse_std'] = np.std(mse_scores)
        metrics[f'{prefix}_mae_mean'] = np.mean(mae_scores)
        metrics[f'{prefix}_mae_std'] = np.std(mae_scores)
        
        # Store feature-level scores with names for detailed logging
        if prefix == 'descriptors' and self.feature_names:
            feature_scores = []
            for i, name in enumerate(self.feature_names.get('descriptors', [])):
                if i < len(pearson_scores):
                    feature_scores.append((name, pearson_scores[i]))
            metrics[f'{prefix}_feature_scores'] = feature_scores
        
        # Top 10 and bottom 10 features for descriptors
        if pearson_scores:
            pearson_array = np.array(pearson_scores)
            spearman_array = np.array(spearman_scores)  
            r2_array = np.array(r2_scores)
            mse_array = np.array(mse_scores)
            
            # Top/bottom by Pearson
            pearson_top_indices = np.argsort(pearson_array)[-10:][::-1]
            pearson_bottom_indices = np.argsort(pearson_array)[:10]
            
            # Top/bottom by MSE (lowest MSE = best)
            mse_top_indices = np.argsort(mse_array)[:10]  # Best = lowest MSE
            mse_bottom_indices = np.argsort(mse_array)[-10:][::-1]  # Worst = highest MSE
            
            # Store for logging
            metrics[f'{prefix}_top10_pearson_values'] = pearson_array[pearson_top_indices].tolist()
            metrics[f'{prefix}_bottom10_pearson_values'] = pearson_array[pearson_bottom_indices].tolist()
            metrics[f'{prefix}_top10_pearson_indices'] = pearson_top_indices.tolist()
            metrics[f'{prefix}_bottom10_pearson_indices'] = pearson_bottom_indices.tolist()
            
            metrics[f'{prefix}_top10_mse_values'] = mse_array[mse_top_indices].tolist()
            metrics[f'{prefix}_bottom10_mse_values'] = mse_array[mse_bottom_indices].tolist()
            metrics[f'{prefix}_top10_mse_indices'] = mse_top_indices.tolist()
            metrics[f'{prefix}_bottom10_mse_indices'] = mse_bottom_indices.tolist()
        
        return metrics
    
    def _calculate_loss_metrics(self, per_feature_losses: torch.Tensor) -> Dict[str, float]:
        """Calculate metrics for per-feature losses"""
        losses_np = per_feature_losses.detach().cpu().numpy()
        
        # Ensure we have a 1D array
        if losses_np.ndim == 0:
            losses_np = np.array([losses_np])
        
        # Basic statistics
        loss_mean = np.mean(losses_np)
        loss_std = np.std(losses_np) if len(losses_np) > 1 else 0.0
        
        # Only calculate top/bottom if we have enough features
        if len(losses_np) >= 10:
            # Top/bottom by loss (lowest loss = best, highest loss = worst)
            loss_top_indices = np.argsort(losses_np)[:10]  # Best = lowest loss
            loss_bottom_indices = np.argsort(losses_np)[-10:][::-1]  # Worst = highest loss
            
            return {
                'per_feature_loss_mean': loss_mean,
                'per_feature_loss_std': loss_std,
                'top10_loss_values': losses_np[loss_top_indices].tolist(),
                'bottom10_loss_values': losses_np[loss_bottom_indices].tolist(),
                'top10_loss_indices': loss_top_indices.tolist(),
                'bottom10_loss_indices': loss_bottom_indices.tolist()
            }
        else:
            return {
                'per_feature_loss_mean': loss_mean,
                'per_feature_loss_std': loss_std
            }
    
    def _calculate_overall_metrics(
        self, 
        pred_desc: np.ndarray, 
        target_desc: np.ndarray
    ) -> Dict[str, float]:
        """Calculate overall metrics across all descriptor features"""
        # Flatten all predictions and targets (only descriptors now)
        all_pred = pred_desc.flatten()
        all_target = target_desc.flatten()
        
        # Overall metrics
        overall_mse = mean_squared_error(all_target, all_pred)
        overall_mae = mean_absolute_error(all_target, all_pred)
        
        if np.std(all_pred) > 1e-8 and np.std(all_target) > 1e-8:
            overall_pearson, _ = pearsonr(all_pred, all_target)
            overall_spearman, _ = spearmanr(all_pred, all_target)
            overall_r2 = r2_score(all_target, all_pred)
        else:
            overall_pearson = 0.0
            overall_spearman = 0.0
            overall_r2 = 0.0
        
        return {
            'overall_mse': overall_mse,
            'overall_mae': overall_mae,
            'overall_pearson': overall_pearson,
            'overall_spearman': overall_spearman,
            'overall_r2': overall_r2,
            'overall_rmse': np.sqrt(overall_mse)
        }
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int, split: str, logger):
        """Log metrics in a structured way"""
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch} - {split.upper()} METRICS")
        logger.info(f"{'='*60}")
        
        # Overall metrics
        logger.info(f"\nOverall Performance:")
        logger.info(f"  MSE: {metrics['overall_mse']:.6f}")
        logger.info(f"  MAE: {metrics['overall_mae']:.6f}")
        logger.info(f"  RMSE: {metrics['overall_rmse']:.6f}")
        logger.info(f"  Pearson: {metrics['overall_pearson']:.4f}")
        logger.info(f"  Spearman: {metrics['overall_spearman']:.4f}")
        logger.info(f"  R²: {metrics['overall_r2']:.4f}")
        
        # Auxiliary head metrics if available
        if 'aux_seq_feat_pearson' in metrics:
            logger.info(f"\n=== AUXILIARY HEAD A (Sequence + Features) ===")
            logger.info(f"  Pearson: {metrics['aux_seq_feat_pearson']:.4f}")
            logger.info(f"  Spearman: {metrics['aux_seq_feat_spearman']:.4f}")
            logger.info(f"  R²: {metrics['aux_seq_feat_r2']:.4f}")
            logger.info(f"  MSE: {metrics['aux_seq_feat_mse']:.6f}")
            logger.info(f"  MAE: {metrics['aux_seq_feat_mae']:.6f}")
            logger.info(f"  RMSE: {metrics['aux_seq_feat_rmse']:.6f}")
            
        if 'aux_feat_only_pearson' in metrics:
            logger.info(f"\n=== AUXILIARY HEAD B (Features Only) ===")
            logger.info(f"  Pearson: {metrics['aux_feat_only_pearson']:.4f}")
            logger.info(f"  Spearman: {metrics['aux_feat_only_spearman']:.4f}")
            logger.info(f"  R²: {metrics['aux_feat_only_r2']:.4f}")
            logger.info(f"  MSE: {metrics['aux_feat_only_mse']:.6f}")
            logger.info(f"  MAE: {metrics['aux_feat_only_mae']:.6f}")
            logger.info(f"  RMSE: {metrics['aux_feat_only_rmse']:.6f}")
        
        # Descriptor metrics with range
        logger.info(f"\nBiophysical Descriptors (across {len(self.feature_names.get('descriptors', []))} features):")
        logger.info(f"  Pearson: mean={metrics['descriptors_pearson_mean']:.4f}, "
                   f"median={metrics.get('descriptors_pearson_median', 0):.4f}, "
                   f"range=[{metrics.get('descriptors_pearson_min', 0):.4f}, {metrics.get('descriptors_pearson_max', 0):.4f}]")
        logger.info(f"  Spearman: mean={metrics['descriptors_spearman_mean']:.4f}, std={metrics['descriptors_spearman_std']:.4f}")
        logger.info(f"  R²: mean={metrics['descriptors_r2_mean']:.4f}, std={metrics['descriptors_r2_std']:.4f}")
        logger.info(f"  MSE: mean={metrics['descriptors_mse_mean']:.6f}, std={metrics['descriptors_mse_std']:.6f}")
        logger.info(f"  MAE: mean={metrics['descriptors_mae_mean']:.6f}, std={metrics['descriptors_mae_std']:.6f}")
        
        # Log top and bottom features if available
        if 'descriptors_feature_scores' in metrics:
            feature_scores = metrics['descriptors_feature_scores']
            if feature_scores:
                # Sort by score
                sorted_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
                
                # Top 10
                logger.info(f"\nTop 10 Best Predicted Descriptors (by Pearson):")
                for i, (name, score) in enumerate(sorted_scores[:10], 1):
                    logger.info(f"   {i:2d}. {name}: {score:.4f}")
                
                # Bottom 10
                logger.info(f"\nBottom 10 Worst Predicted Descriptors (by Pearson):")
                for i, (name, score) in enumerate(sorted_scores[-10:], 1):
                    logger.info(f"   {i:2d}. {name}: {score:.4f}")
        
        # Per-feature loss statistics
        if 'per_feature_loss_mean' in metrics:
            logger.info(f"\nPer-Feature Loss Statistics:")
            logger.info(f"  Average Loss: {metrics['per_feature_loss_mean']:.6f} ± {metrics['per_feature_loss_std']:.6f}")
        
        # Top and bottom performing DESCRIPTOR features
        if 'descriptors_top10_pearson_values' in metrics:
            logger.info(f"\n=== DESCRIPTOR FEATURES (Top/Bottom 10) ===")
            logger.info(f"\nTop 10 Best Predicted Descriptors (by Pearson):")
            for i, (idx, score) in enumerate(zip(metrics['descriptors_top10_pearson_indices'], metrics['descriptors_top10_pearson_values'])):
                feat_name = self.descriptor_names[idx] if idx < len(self.descriptor_names) else f"desc_{idx}"
                logger.info(f"  {i+1:2d}. {feat_name}: {score:.4f}")
                
            logger.info(f"\nBottom 10 Worst Predicted Descriptors (by Pearson):")
            for i, (idx, score) in enumerate(zip(metrics['descriptors_bottom10_pearson_indices'], metrics['descriptors_bottom10_pearson_values'])):
                feat_name = self.descriptor_names[idx] if idx < len(self.descriptor_names) else f"desc_{idx}"
                logger.info(f"  {i+1:2d}. {feat_name}: {score:.4f}")
                
            logger.info(f"\nTop 10 Best Predicted Descriptors (by MSE - Lowest):")
            for i, (idx, score) in enumerate(zip(metrics['descriptors_top10_mse_indices'], metrics['descriptors_top10_mse_values'])):
                feat_name = self.descriptor_names[idx] if idx < len(self.descriptor_names) else f"desc_{idx}"
                logger.info(f"  {i+1:2d}. {feat_name}: {score:.6f}")
                
            logger.info(f"\nBottom 10 Worst Predicted Descriptors (by MSE - Highest):")
            for i, (idx, score) in enumerate(zip(metrics['descriptors_bottom10_mse_indices'], metrics['descriptors_bottom10_mse_values'])):
                feat_name = self.descriptor_names[idx] if idx < len(self.descriptor_names) else f"desc_{idx}"
                logger.info(f"  {i+1:2d}. {feat_name}: {score:.6f}")
                
        # Per-feature loss top/bottom if available
        if 'top10_loss_values' in metrics:
            logger.info(f"\n=== PER-FEATURE LOSS ANALYSIS ===")
            logger.info(f"\nTop 10 Features (Lowest Loss):")
            for i, (idx, loss) in enumerate(zip(metrics['top10_loss_indices'], metrics['top10_loss_values'])):
                feat_name = self.descriptor_names[idx] if idx < len(self.descriptor_names) else f"desc_{idx}"
                logger.info(f"  {i+1:2d}. {feat_name}: {loss:.6f}")
                
            logger.info(f"\nBottom 10 Features (Highest Loss):")
            for i, (idx, loss) in enumerate(zip(metrics['bottom10_loss_indices'], metrics['bottom10_loss_values'])):
                feat_name = self.descriptor_names[idx] if idx < len(self.descriptor_names) else f"desc_{idx}"
                logger.info(f"  {i+1:2d}. {feat_name}: {loss:.6f}")
        
        logger.info(f"\n{'='*60}")

def compute_auxiliary_metrics(predictions: Dict[str, torch.Tensor], 
                             targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for auxiliary activity predictions
    
    Args:
        predictions: Dict with 'aux_activity_seq_feat' and/or 'aux_activity_feat_only'
        targets: Activity targets (batch, n_activities)
        
    Returns:
        Dict with metrics for each auxiliary head
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {}
    
    # Convert targets to numpy
    if torch.is_tensor(targets):
        targets_np = targets.cpu().numpy()
    else:
        targets_np = targets
    
    # Flatten if multiple activities (for overall correlation)
    targets_flat = targets_np.flatten()
    
    # Metrics for Head A: Sequence + Features (supports both naming conventions)
    if 'aux_head_a' in predictions or 'aux_activity_seq_feat' in predictions:
        pred_a = predictions.get('aux_head_a', predictions.get('aux_activity_seq_feat'))
        if torch.is_tensor(pred_a):
            pred_a = pred_a.cpu().numpy()
        pred_a_flat = pred_a.flatten()
        
        # Compute metrics
        if len(pred_a_flat) > 1 and np.std(pred_a_flat) > 1e-8 and np.std(targets_flat) > 1e-8:
            pearson_a, _ = pearsonr(pred_a_flat, targets_flat)
            spearman_a, _ = spearmanr(pred_a_flat, targets_flat)
            r2_a = r2_score(targets_flat, pred_a_flat)
        else:
            pearson_a = spearman_a = r2_a = 0.0
            
        mse_a = mean_squared_error(targets_flat, pred_a_flat)
        mae_a = mean_absolute_error(targets_flat, pred_a_flat)
        rmse_a = np.sqrt(mse_a)
        
        metrics.update({
            'aux_seq_feat_pearson': pearson_a,
            'aux_seq_feat_spearman': spearman_a,
            'aux_seq_feat_r2': r2_a,
            'aux_seq_feat_mse': mse_a,
            'aux_seq_feat_mae': mae_a,
            'aux_seq_feat_rmse': rmse_a
        })
    
    # Metrics for Head B: Features Only (supports both naming conventions)
    if 'aux_head_b' in predictions or 'aux_activity_feat_only' in predictions:
        pred_b = predictions.get('aux_head_b', predictions.get('aux_activity_feat_only'))
        if torch.is_tensor(pred_b):
            pred_b = pred_b.cpu().numpy()
        pred_b_flat = pred_b.flatten()
        
        # Compute metrics
        if len(pred_b_flat) > 1 and np.std(pred_b_flat) > 1e-8 and np.std(targets_flat) > 1e-8:
            pearson_b, _ = pearsonr(pred_b_flat, targets_flat)
            spearman_b, _ = spearmanr(pred_b_flat, targets_flat)
            r2_b = r2_score(targets_flat, pred_b_flat)
        else:
            pearson_b = spearman_b = r2_b = 0.0
            
        mse_b = mean_squared_error(targets_flat, pred_b_flat)
        mae_b = mean_absolute_error(targets_flat, pred_b_flat)
        rmse_b = np.sqrt(mse_b)
        
        metrics.update({
            'aux_feat_only_pearson': pearson_b,
            'aux_feat_only_spearman': spearman_b,
            'aux_feat_only_r2': r2_b,
            'aux_feat_only_mse': mse_b,
            'aux_feat_only_mae': mae_b,
            'aux_feat_only_rmse': rmse_b
        })
    
    return metrics


def compute_feature_statistics(dataloader, device):
    """
    Compute feature statistics (mean and std) from training data for normalization
    
    Args:
        dataloader: Training dataloader
        device: Device to compute on
        
    Returns:
        Dict with 'desc_mean', 'desc_std'
    """
    desc_sum = None
    desc_sq_sum = None
    n_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            desc = batch['descriptors'].to(device)
            batch_size = desc.shape[0]
            
            if desc_sum is None:
                desc_sum = torch.zeros(desc.shape[1], device=device)
                desc_sq_sum = torch.zeros(desc.shape[1], device=device)
            
            desc_sum += desc.sum(dim=0)
            desc_sq_sum += (desc ** 2).sum(dim=0)
            n_samples += batch_size
    
    # Compute mean and std
    desc_mean = desc_sum / n_samples
    desc_var = (desc_sq_sum / n_samples) - (desc_mean ** 2)
    desc_std = torch.sqrt(torch.clamp(desc_var, min=1e-8))
    
    return {
        'desc_mean': desc_mean,
        'desc_std': desc_std
    }

class AdaptiveLossWeights:
    """Maintains adaptive per-feature loss weights with exponential moving average"""
    
    def __init__(self, n_features: int = 529, device: str = 'cuda', alpha: float = 0.99):
        """
        Args:
            n_features: Number of features
            device: Device for tensors
            alpha: EMA decay rate (0.99 = slow adaptation, 0.9 = fast adaptation)
        """
        self.n_features = n_features
        self.device = device
        self.alpha = alpha  # EMA decay
        
        # Initialize running averages of losses (start with ones)
        self.running_losses = torch.ones(n_features, device=device)
        self.initialized = False
        
    def update_and_get_weights(self, current_losses: torch.Tensor) -> torch.Tensor:
        """
        Update running average and return balanced weights
        
        Args:
            current_losses: Per-feature losses for current batch [n_features]
            
        Returns:
            weights: Per-feature weights that balance contributions [n_features]
        """
        with torch.no_grad():
            if not self.initialized:
                # First batch: initialize with current losses
                self.running_losses = current_losses.detach().clone()
                self.initialized = True
            else:
                # Update exponential moving average
                self.running_losses = (self.alpha * self.running_losses + 
                                      (1 - self.alpha) * current_losses.detach())
            
            # Calculate weights to aggressively balance features
            # With normalized data, losses are in narrow range [0, ~1]
            # Need sensitive weighting to make a difference
            
            # Direct proportional weighting to loss
            # Features with loss near 1 (struggling) need much higher weight
            # Features with loss near 0 (easy) need much lower weight
            
            # Add small epsilon to prevent division by zero
            eps = 1e-6
            
            # Use power scaling to amplify differences
            # Power > 1 makes high losses get even higher weights
            power = 2.0  # Square the losses to amplify differences
            
            # Calculate weights proportional to loss^power
            weights = torch.pow(self.running_losses + eps, power)
            
            # Normalize by mean to get relative weights
            mean_weight = weights.mean()
            weights = weights / (mean_weight + eps)
            
            # Ensure minimum weight so no feature is ignored
            min_weight = 0.01
            weights = torch.maximum(weights, torch.tensor(min_weight, device=weights.device))
            
            # Cap maximum weight to prevent single feature domination
            max_weight = 100.0
            weights = torch.minimum(weights, torch.tensor(max_weight, device=weights.device))
            
            # Final normalization to maintain gradient scale
            weights = weights * (self.n_features / weights.sum())
            
        return weights

# Global instance to maintain state across batches
_adaptive_weights = None

def calculate_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], 
                   feature_stats: Optional[Dict] = None, epoch: int = 1,
                   use_adaptive_weights: bool = True) -> Dict[str, torch.Tensor]:
    """
    Calculate BALANCED per-feature loss with adaptive weighting
    
    Features that are struggling (high loss) get prioritized with higher weights.
    Features that are easy (low loss) get lower weights.
    This ensures all features learn at similar rates.
    
    Args:
        predictions: Dict with 'descriptors' predictions
        targets: Dict with 'descriptors' targets
        feature_stats: Optional - not used when data is already normalized
        epoch: Current epoch for adaptive penalty weight
        use_adaptive_weights: Whether to use adaptive balancing (default True)
        
    Returns:
        Dict with individual losses and total loss
    """
    global _adaptive_weights
    
    # Calculate PER-FEATURE losses for descriptors (each of 529 features)
    # Shape: [batch_size, n_features] -> [n_features] (loss per feature)
    desc_per_feature_loss = torch.nn.functional.mse_loss(
        predictions['descriptors'], targets['descriptors'], 
        reduction='none'
    ).mean(dim=0)  # Average over batch, keep features separate -> [n_features]
    
    if use_adaptive_weights:
        # Initialize adaptive weights if needed
        if _adaptive_weights is None:
            device = predictions['descriptors'].device
            n_features = predictions['descriptors'].shape[1]
            _adaptive_weights = AdaptiveLossWeights(n_features, device)
        
        # Get balanced weights based on running average of losses
        weights = _adaptive_weights.update_and_get_weights(desc_per_feature_loss)
        
        # Apply weights to losses
        # High-loss features get boosted, low-loss features get reduced
        weighted_losses = desc_per_feature_loss * weights
        
        # Sum weighted losses - this is what gets optimized
        total_loss = weighted_losses.sum()
        
        # For logging: also return the weights for monitoring
        weight_info = {
            'min_weight': weights.min().item(),
            'max_weight': weights.max().item(),
            'mean_weight': weights.mean().item(),
            'all_weights': weights.detach().cpu().numpy(),  # All 529 weights
        }
    else:
        # Simple sum (original approach)
        total_loss = desc_per_feature_loss.sum()
        weight_info = {}
    
    # For logging purposes
    desc_loss_sum = desc_per_feature_loss.sum()  # Original sum (not optimized)
    desc_loss_mean = desc_per_feature_loss.mean()  # Average for logging
    
    result = {
        'total_loss': total_loss,  # ADAPTIVE weighted sum (what gets optimized)
        'desc_loss': desc_loss_sum,  # Original sum for backward compatibility  
        'desc_loss_mean': desc_loss_mean,  # AVERAGE loss per descriptor (logging only)
        'per_feature_losses': desc_per_feature_loss  # [n_features] - individual loss per feature
    }
    result.update(weight_info)
    
    return result