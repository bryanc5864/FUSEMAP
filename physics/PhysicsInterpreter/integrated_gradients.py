"""
Integrated Gradients for CADENCE sequence attribution.

Implements the Integrated Gradients method (Sundararajan et al., 2017)
for computing sequence-level attributions through the CADENCE model.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import warnings

from .config import InterpreterConfig, get_fusemap_root


@dataclass
class IGResult:
    """Results from Integrated Gradients computation."""
    attributions: np.ndarray  # Shape: (length, 4) or (batch, length, 4)
    convergence_delta: float  # Completeness check
    sequence: str
    prediction: float
    baseline_prediction: float


class IntegratedGradients:
    """
    Integrated Gradients for CADENCE model attribution.

    Computes sequence attributions by integrating gradients along a path
    from a baseline to the input sequence.

    Usage:
        ig = IntegratedGradients(config)
        ig.load_model()
        result = ig.attribute(sequence)
        print(result.attributions)  # (length, 4) importance scores
    """

    def __init__(self, config: InterpreterConfig = None):
        self.config = config or InterpreterConfig()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model_path: str = None):
        """
        Load CADENCE model for attribution.

        Args:
            model_path: Optional path override
        """
        if model_path is None:
            model_path = self.config.get_cadence_path()

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"CADENCE model not found: {model_path}")

        print(f"Loading CADENCE model from {model_path}")

        # Load the model - we need to import CADENCE architecture
        try:
            import sys
            fusemap_root = get_fusemap_root()
            sys.path.insert(0, str(fusemap_root))

            from models.CADENCE.model import CADENCE

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Get model config from checkpoint
            if 'config' in checkpoint:
                model_config = checkpoint['config']
            else:
                # Default config
                model_config = {
                    'seq_len': 200,
                    'd_model': 256,
                    'n_heads': 8,
                    'n_layers': 6,
                    'dropout': 0.1
                }

            # Initialize model
            self.model = CADENCE(**model_config)

            # Load weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"Model loaded successfully on {self.device}")

        except ImportError as e:
            warnings.warn(f"Could not import CADENCE model: {e}")
            raise

    def _sequence_to_onehot(self, sequence: str) -> torch.Tensor:
        """Convert DNA sequence to one-hot encoding."""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
        indices = [mapping.get(nt.upper(), 0) for nt in sequence]
        onehot = torch.zeros(len(sequence), 4)
        for i, idx in enumerate(indices):
            onehot[i, idx] = 1.0
        return onehot

    def _get_baseline(self, sequence: str) -> torch.Tensor:
        """
        Create baseline for integrated gradients.

        Args:
            sequence: Input DNA sequence

        Returns:
            Baseline tensor of same shape as one-hot encoded sequence
        """
        seq_len = len(sequence)

        if self.config.ig_baseline == 'zeros':
            # All zeros baseline
            return torch.zeros(seq_len, 4)

        elif self.config.ig_baseline == 'shuffle':
            # Shuffled sequence baseline
            import random
            shuffled = list(sequence)
            random.shuffle(shuffled)
            return self._sequence_to_onehot(''.join(shuffled))

        elif self.config.ig_baseline == 'gc_matched':
            # GC-content matched random sequence
            gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
            random_seq = []
            for _ in range(seq_len):
                if np.random.random() < gc_content:
                    random_seq.append(np.random.choice(['G', 'C']))
                else:
                    random_seq.append(np.random.choice(['A', 'T']))
            return self._sequence_to_onehot(''.join(random_seq))

        else:
            return torch.zeros(seq_len, 4)

    def attribute(
        self,
        sequence: str,
        n_steps: int = None,
        return_convergence: bool = True
    ) -> IGResult:
        """
        Compute integrated gradients for a single sequence.

        Args:
            sequence: DNA sequence
            n_steps: Number of interpolation steps (default from config)
            return_convergence: Whether to compute convergence delta

        Returns:
            IGResult with attributions and metadata
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        n_steps = n_steps or self.config.ig_steps

        # Convert sequence to one-hot
        input_tensor = self._sequence_to_onehot(sequence)
        baseline = self._get_baseline(sequence)

        input_tensor = input_tensor.to(self.device)
        baseline = baseline.to(self.device)

        # Compute interpolated inputs
        alphas = torch.linspace(0, 1, n_steps + 1, device=self.device)
        interpolated = []
        for alpha in alphas:
            interp = baseline + alpha * (input_tensor - baseline)
            interpolated.append(interp)

        interpolated = torch.stack(interpolated)  # (n_steps+1, seq_len, 4)

        # Enable gradients
        interpolated.requires_grad_(True)

        # Forward pass through all interpolated inputs
        # Process in batches if needed
        batch_size = self.config.ig_batch_size
        all_outputs = []

        for i in range(0, len(interpolated), batch_size):
            batch = interpolated[i:i + batch_size]
            # Add batch dimension if needed
            if batch.dim() == 3:
                batch = batch.unsqueeze(0) if batch.size(0) == 1 else batch

            with torch.enable_grad():
                outputs = self.model(batch)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                all_outputs.append(outputs)

        outputs = torch.cat(all_outputs, dim=0)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=outputs.sum(),
            inputs=interpolated,
            create_graph=False
        )[0]

        # Average gradients (Riemann approximation)
        avg_gradients = gradients.mean(dim=0)

        # Compute attributions: (input - baseline) * avg_gradients
        attributions = (input_tensor - baseline) * avg_gradients

        # Get predictions
        with torch.no_grad():
            input_batch = input_tensor.unsqueeze(0)
            baseline_batch = baseline.unsqueeze(0)

            pred = self.model(input_batch)
            baseline_pred = self.model(baseline_batch)

            if isinstance(pred, tuple):
                pred = pred[0]
            if isinstance(baseline_pred, tuple):
                baseline_pred = baseline_pred[0]

        prediction = pred.item()
        baseline_prediction = baseline_pred.item()

        # Convergence check: sum of attributions should equal pred - baseline_pred
        attr_sum = attributions.sum().item()
        expected = prediction - baseline_prediction
        convergence_delta = abs(attr_sum - expected)

        return IGResult(
            attributions=attributions.detach().cpu().numpy(),
            convergence_delta=convergence_delta,
            sequence=sequence,
            prediction=prediction,
            baseline_prediction=baseline_prediction
        )

    def attribute_batch(
        self,
        sequences: List[str],
        n_steps: int = None
    ) -> List[IGResult]:
        """
        Compute integrated gradients for multiple sequences.

        Args:
            sequences: List of DNA sequences
            n_steps: Number of interpolation steps

        Returns:
            List of IGResult objects
        """
        results = []
        for seq in sequences:
            result = self.attribute(seq, n_steps)
            results.append(result)
        return results

    def get_nucleotide_importance(
        self,
        ig_result: IGResult,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Get per-position importance scores (summed across nucleotides).

        Args:
            ig_result: IGResult from attribute()
            normalize: Whether to normalize to sum to 1

        Returns:
            Array of shape (length,) with importance per position
        """
        # Sum absolute attributions across nucleotides
        importance = np.abs(ig_result.attributions).sum(axis=1)

        if normalize:
            importance = importance / importance.sum()

        return importance

    def get_motif_importance(
        self,
        ig_result: IGResult,
        motif_start: int,
        motif_end: int
    ) -> float:
        """
        Get importance of a specific motif region.

        Args:
            ig_result: IGResult from attribute()
            motif_start: Start position of motif
            motif_end: End position of motif

        Returns:
            Total importance of the motif region
        """
        importance = self.get_nucleotide_importance(ig_result, normalize=False)
        return importance[motif_start:motif_end].sum()


def compute_ig_attribution(
    sequence: str,
    model_path: str = None,
    cell_type: str = 'WTC11',
    n_steps: int = 50
) -> IGResult:
    """
    Convenience function to compute IG attribution for a sequence.

    Args:
        sequence: DNA sequence
        model_path: Optional CADENCE model path
        cell_type: Cell type for model selection
        n_steps: Number of interpolation steps

    Returns:
        IGResult with attributions
    """
    config = InterpreterConfig(cell_type=cell_type, ig_steps=n_steps)
    ig = IntegratedGradients(config)
    ig.load_model(model_path)
    return ig.attribute(sequence)
