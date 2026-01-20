"""
FUSEMAP Training Metrics

Comprehensive metrics for model evaluation including:
- Pearson correlation
- Spearman correlation
- R-squared
- MSE, RMSE, MAE
- Per-dataset and per-output metrics
- Special validation schemes (DREAM yeast weighted)
"""

import numpy as np
import torch
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import json


@dataclass
class MetricResult:
    """Container for metric results."""
    value: float
    stderr: Optional[float] = None
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "stderr": self.stderr,
            "n_samples": self.n_samples,
        }


@dataclass
class DatasetMetrics:
    """Metrics for a single dataset/output."""
    pearson: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    spearman: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    r2: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    mse: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    rmse: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    mae: MetricResult = field(default_factory=lambda: MetricResult(0.0))

    def to_dict(self) -> dict:
        return {
            "pearson": self.pearson.to_dict(),
            "spearman": self.spearman.to_dict(),
            "r2": self.r2.to_dict(),
            "mse": self.mse.to_dict(),
            "rmse": self.rmse.to_dict(),
            "mae": self.mae.to_dict(),
        }


def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute Pearson correlation with p-value."""
    if len(y_true) < 3:
        return 0.0, 1.0
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 3:
        return 0.0, 1.0
    r, p = stats.pearsonr(y_true[mask], y_pred[mask])
    return float(r), float(p)


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute Spearman correlation with p-value."""
    if len(y_true) < 3:
        return 0.0, 1.0
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 3:
        return 0.0, 1.0
    rho, p = stats.spearmanr(y_true[mask], y_pred[mask])
    return float(rho), float(p)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared (coefficient of determination)."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return 0.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return float('inf')
    return float(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return float('inf')
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> DatasetMetrics:
    """Compute all metrics for a single output."""
    n_samples = len(y_true)

    # Handle tensor inputs
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Weighted metrics for special validation (DREAM yeast)
    if weights is not None:
        weights = np.asarray(weights).flatten()
        # Weighted versions would go here
        # For now, use unweighted but with proper masking

    pearson_r, pearson_p = compute_pearson(y_true, y_pred)
    spearman_rho, spearman_p = compute_spearman(y_true, y_pred)
    r2 = compute_r2(y_true, y_pred)
    mse = compute_mse(y_true, y_pred)
    rmse = np.sqrt(mse) if mse != float('inf') else float('inf')
    mae = compute_mae(y_true, y_pred)

    return DatasetMetrics(
        pearson=MetricResult(pearson_r, n_samples=n_samples),
        spearman=MetricResult(spearman_rho, n_samples=n_samples),
        r2=MetricResult(r2, n_samples=n_samples),
        mse=MetricResult(mse, n_samples=n_samples),
        rmse=MetricResult(rmse, n_samples=n_samples),
        mae=MetricResult(mae, n_samples=n_samples),
    )


class MetricsTracker:
    """
    Track metrics across training for logging and early stopping.
    """

    def __init__(
        self,
        dataset_names: List[str],
        output_names_per_dataset: Dict[str, List[str]],
    ):
        self.dataset_names = dataset_names
        self.output_names = output_names_per_dataset

        # History storage
        self.train_history: Dict[str, List[DatasetMetrics]] = {
            ds: [] for ds in dataset_names
        }
        self.val_history: Dict[str, List[DatasetMetrics]] = {
            ds: [] for ds in dataset_names
        }
        self.test_results: Dict[str, DatasetMetrics] = {}

        # Loss tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        # Gradient tracking
        self.gradient_norms: List[float] = []

        # Best metrics for early stopping
        self.best_epoch = 0
        self.best_val_metric = float('-inf')
        self.patience_counter = 0

    def update_train_loss(self, loss: float, gradient_norm: Optional[float] = None):
        """Update training loss and gradient norm."""
        self.train_losses.append(loss)
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)

    def update_val_metrics(
        self,
        epoch: int,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, DatasetMetrics]:
        """
        Update validation metrics for all datasets.

        Args:
            epoch: Current epoch
            predictions: Dict of dataset_name -> predictions array
            targets: Dict of dataset_name -> targets array
            weights: Optional weights for weighted evaluation

        Returns:
            Dict of computed metrics per dataset
        """
        epoch_metrics = {}

        for dataset_name in self.dataset_names:
            if dataset_name not in predictions:
                continue

            pred = predictions[dataset_name]
            target = targets[dataset_name]
            w = weights.get(dataset_name) if weights else None

            # Handle multi-output datasets
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            if target.ndim == 1:
                target = target.reshape(-1, 1)

            # Compute metrics per output
            output_names = self.output_names.get(dataset_name, ["output"])
            dataset_metrics_list = []

            for i, output_name in enumerate(output_names):
                if i >= pred.shape[1]:
                    continue
                metrics = compute_all_metrics(
                    target[:, i] if target.ndim > 1 else target,
                    pred[:, i] if pred.ndim > 1 else pred,
                    weights=w,
                )
                dataset_metrics_list.append(metrics)

            # Average across outputs for summary
            avg_metrics = self._average_metrics(dataset_metrics_list)
            epoch_metrics[dataset_name] = avg_metrics
            self.val_history[dataset_name].append(avg_metrics)

        return epoch_metrics

    def _average_metrics(self, metrics_list: List[DatasetMetrics]) -> DatasetMetrics:
        """Average metrics across multiple outputs."""
        if not metrics_list:
            return DatasetMetrics()

        n = len(metrics_list)
        return DatasetMetrics(
            pearson=MetricResult(
                np.mean([m.pearson.value for m in metrics_list]),
                n_samples=sum(m.pearson.n_samples for m in metrics_list),
            ),
            spearman=MetricResult(
                np.mean([m.spearman.value for m in metrics_list]),
                n_samples=sum(m.spearman.n_samples for m in metrics_list),
            ),
            r2=MetricResult(
                np.mean([m.r2.value for m in metrics_list]),
                n_samples=sum(m.r2.n_samples for m in metrics_list),
            ),
            mse=MetricResult(
                np.mean([m.mse.value for m in metrics_list]),
                n_samples=sum(m.mse.n_samples for m in metrics_list),
            ),
            rmse=MetricResult(
                np.mean([m.rmse.value for m in metrics_list]),
                n_samples=sum(m.rmse.n_samples for m in metrics_list),
            ),
            mae=MetricResult(
                np.mean([m.mae.value for m in metrics_list]),
                n_samples=sum(m.mae.n_samples for m in metrics_list),
            ),
        )

    def check_early_stopping(
        self,
        epoch: int,
        val_metrics: Dict[str, DatasetMetrics],
        metric_name: str = "pearson",
        patience: int = 10,
        min_delta: float = 1e-4,
    ) -> Tuple[bool, bool]:
        """
        Check if early stopping criteria is met.

        Args:
            epoch: Current epoch
            val_metrics: Validation metrics dict
            metric_name: Metric to monitor (pearson, r2, mse, etc.)
            patience: Number of epochs to wait
            min_delta: Minimum improvement threshold

        Returns:
            (should_stop, is_best)
        """
        # Compute average metric across datasets
        values = []
        for dataset_name, metrics in val_metrics.items():
            metric_obj = getattr(metrics, metric_name, None)
            if metric_obj is not None:
                values.append(metric_obj.value)

        if not values:
            return False, False

        current_metric = np.mean(values)

        # Check if best
        is_best = current_metric > self.best_val_metric + min_delta

        if is_best:
            self.best_val_metric = current_metric
            self.best_epoch = epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        should_stop = self.patience_counter >= patience

        return should_stop, is_best

    def get_summary(self, epoch: int) -> Dict:
        """Get summary of current training state."""
        summary = {
            "epoch": epoch,
            "train_loss": self.train_losses[-1] if self.train_losses else None,
            "gradient_norm": self.gradient_norms[-1] if self.gradient_norms else None,
            "best_epoch": self.best_epoch,
            "best_val_metric": self.best_val_metric,
            "patience_counter": self.patience_counter,
        }

        # Add per-dataset val metrics
        for dataset_name in self.dataset_names:
            if self.val_history[dataset_name]:
                latest = self.val_history[dataset_name][-1]
                summary[f"{dataset_name}_val_pearson"] = latest.pearson.value
                summary[f"{dataset_name}_val_r2"] = latest.r2.value

        return summary

    def save_results(self, filepath: str):
        """Save all results to JSON file."""
        results = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "gradient_norms": self.gradient_norms,
            "best_epoch": self.best_epoch,
            "best_val_metric": self.best_val_metric,
            "val_history": {
                ds: [m.to_dict() for m in metrics]
                for ds, metrics in self.val_history.items()
            },
            "test_results": {
                ds: m.to_dict() for ds, m in self.test_results.items()
            },
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)


class DREAMYeastMetrics:
    """
    Special metrics computation for DREAM Yeast challenge.

    The DREAM challenge uses:
    - 1% evaluation set with specific weighting
    - Weighted Pearson correlation
    - Expression bin-specific evaluation
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def compute_weighted_pearson(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Compute weighted Pearson correlation."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(weights))
        if mask.sum() < 3:
            return 0.0

        y_true = y_true[mask]
        y_pred = y_pred[mask]
        weights = weights[mask]

        # Weighted means
        w_sum = weights.sum()
        mean_true = np.sum(weights * y_true) / w_sum
        mean_pred = np.sum(weights * y_pred) / w_sum

        # Weighted covariance and standard deviations
        cov = np.sum(weights * (y_true - mean_true) * (y_pred - mean_pred)) / w_sum
        std_true = np.sqrt(np.sum(weights * (y_true - mean_true) ** 2) / w_sum)
        std_pred = np.sqrt(np.sum(weights * (y_pred - mean_pred) ** 2) / w_sum)

        if std_true == 0 or std_pred == 0:
            return 0.0

        return float(cov / (std_true * std_pred))

    def compute_binned_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute metrics per expression bin."""
        # Bin by true expression level
        bins = np.percentile(y_true, np.linspace(0, 100, self.n_bins + 1))
        bin_indices = np.digitize(y_true, bins[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        binned_metrics = {}
        for i in range(self.n_bins):
            mask = bin_indices == i
            if mask.sum() >= 3:
                r, _ = compute_pearson(y_true[mask], y_pred[mask])
                binned_metrics[f"bin_{i}_pearson"] = r
                binned_metrics[f"bin_{i}_n"] = int(mask.sum())

        return binned_metrics

    def compute_dream_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute full DREAM-style evaluation.

        Returns various metrics used in the DREAM challenge.
        """
        results = {}

        # Standard Pearson
        r, _ = compute_pearson(y_true, y_pred)
        results["pearson"] = r

        # Spearman
        rho, _ = compute_spearman(y_true, y_pred)
        results["spearman"] = rho

        # Weighted Pearson if weights provided
        if weights is not None:
            results["weighted_pearson"] = self.compute_weighted_pearson(
                y_true, y_pred, weights
            )

        # Binned metrics
        binned = self.compute_binned_metrics(y_true, y_pred)
        results.update(binned)

        # DREAM challenge final score (example formula)
        results["dream_score"] = (r + rho) / 2

        return results
