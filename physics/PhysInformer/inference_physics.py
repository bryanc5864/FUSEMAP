"""
Inference utilities for PhysicsAware model with proper denormalization
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Union
import json

class PhysicsAwarePredictor:
    """Wrapper for PhysicsAware model inference with denormalization"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model
        from physics_aware_model import create_physics_aware_model
        self.cell_type = checkpoint.get('cell_type', 'HepG2')
        
        # Get feature count from checkpoint or use default
        if 'model_state_dict' in checkpoint:
            # Infer feature count from model state dict
            feature_keys = [k for k in checkpoint['model_state_dict'].keys() 
                          if 'property_heads.feature_' in k]
            n_features = len(set(k.split('.')[1] for k in feature_keys))
        else:
            n_features = None
            
        self.model = create_physics_aware_model(self.cell_type, n_descriptor_features=n_features)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load normalization stats
        if 'normalization_stats' in checkpoint:
            self.norm_stats = checkpoint['normalization_stats']
            # Move stats to device
            for key in ['desc_mean', 'desc_std']:
                if key in self.norm_stats:
                    self.norm_stats[key] = self.norm_stats[key].to(self.device)
        else:
            print("Warning: No normalization stats found in checkpoint. Predictions will be in normalized space.")
            self.norm_stats = None
    
    def predict_from_sequence(self, sequence: Union[str, List[str]], return_denormalized: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict descriptors from DNA sequence(s)
        
        Args:
            sequence: DNA sequence string or list of sequences
            return_denormalized: Whether to denormalize predictions to original scale
            
        Returns:
            Dict with 'descriptors' predictions and raw model outputs
        """
        # Handle single sequence or batch
        if isinstance(sequence, str):
            sequences = [sequence]
            single_seq = True
        else:
            sequences = sequence
            single_seq = False
            
        # Convert sequences to indices
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        indices_list = []
        for seq in sequences:
            seq = seq.upper()
            indices = torch.tensor([mapping.get(base, 4) for base in seq], dtype=torch.long)
            indices_list.append(indices)
        
        # Stack into batch
        indices_batch = torch.stack(indices_list).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(indices_batch)
        
        # Reconstruct descriptor tensor from individual feature predictions
        n_features = len([k for k in outputs.keys() if k.startswith('feature_') and k.endswith('_mean')])
        batch_size = indices_batch.shape[0]
        
        desc_pred = torch.zeros(batch_size, n_features, device=self.device)
        for i in range(n_features):
            key = f'feature_{i}_mean'
            if key in outputs:
                desc_pred[:, i] = outputs[key]
        
        desc_pred = desc_pred.cpu()
        
        # Denormalize if requested and stats available
        if return_denormalized and self.norm_stats is not None:
            desc_pred = desc_pred * self.norm_stats['desc_std'].cpu() + self.norm_stats['desc_mean'].cpu()
        
        # Return single or batch
        if single_seq:
            return {
                'descriptors': desc_pred.numpy().squeeze(),
                'raw_outputs': {k: v.cpu().numpy() for k, v in outputs.items()}
            }
        else:
            return {
                'descriptors': desc_pred.numpy(),
                'raw_outputs': {k: v.cpu().numpy() for k, v in outputs.items()}
            }
    
    def get_feature_names(self, data_dir: str = '../output') -> List[str]:
        """
        Get feature names from dataset
        
        Args:
            data_dir: Directory containing data files
            
        Returns:
            List of feature names
        """
        from dataset import create_dataloaders
        
        # Load dataset to get feature names
        dataloaders = create_dataloaders(self.cell_type, data_dir, batch_size=1, num_workers=1)
        return dataloaders['train'].dataset.descriptor_cols
    
    def evaluate_predictions(self, predictions: torch.Tensor, targets: torch.Tensor, 
                            denormalize: bool = True) -> Dict:
        """
        Evaluate predictions against targets with optional denormalization
        
        Args:
            predictions: Model predictions (normalized)
            targets: Ground truth (normalized)
            denormalize: Whether to denormalize before computing metrics
            
        Returns:
            Dict with evaluation metrics
        """
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Denormalize if requested
        if denormalize and self.norm_stats is not None:
            predictions = predictions * self.norm_stats['desc_std'] + self.norm_stats['desc_mean']
            targets = targets * self.norm_stats['desc_std'] + self.norm_stats['desc_mean']
        
        # Convert to numpy
        pred_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        target_np = targets.cpu().numpy() if torch.is_tensor(targets) else targets
        
        # Compute metrics
        mse = mean_squared_error(target_np.flatten(), pred_np.flatten())
        mae = mean_absolute_error(target_np.flatten(), pred_np.flatten())
        
        # Per-feature correlations
        n_features = pred_np.shape[1] if len(pred_np.shape) > 1 else 1
        correlations = []
        
        for i in range(n_features):
            if len(pred_np.shape) > 1:
                pred_i = pred_np[:, i]
                target_i = target_np[:, i]
            else:
                pred_i = pred_np
                target_i = target_np
                
            if np.std(pred_i) > 1e-8 and np.std(target_i) > 1e-8:
                corr, _ = pearsonr(pred_i, target_i)
                correlations.append(corr)
        
        mean_corr = np.mean(correlations) if correlations else 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'mean_pearson': mean_corr,
            'per_feature_correlations': correlations
        }


def denormalize_tensor(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Utility function to denormalize a tensor
    
    Args:
        tensor: Normalized tensor
        mean: Mean used for normalization
        std: Standard deviation used for normalization
        
    Returns:
        Denormalized tensor
    """
    return tensor * std + mean


def normalize_tensor(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Utility function to normalize a tensor
    
    Args:
        tensor: Raw tensor
        mean: Mean for normalization
        std: Standard deviation for normalization
        
    Returns:
        Normalized tensor
    """
    return (tensor - mean) / std


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict descriptors for DNA sequences')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--sequences', type=str, nargs='+', help='DNA sequences to predict')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--normalized', action='store_true', help='Return normalized values')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = PhysicsAwarePredictor(args.checkpoint, args.device)
    
    # Predict
    results = predictor.predict_from_sequence(
        args.sequences,
        return_denormalized=not args.normalized
    )
    
    print(f"Predicted descriptors shape: {results['descriptors'].shape}")
    print(f"Sample values (first 5 features):")
    for i in range(min(5, results['descriptors'].shape[-1])):
        if len(results['descriptors'].shape) > 1:
            print(f"  Feature {i}: {results['descriptors'][0, i]:.4f}")
        else:
            print(f"  Feature {i}: {results['descriptors'][i]:.4f}")