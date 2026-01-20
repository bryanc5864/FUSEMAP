"""
Calibration for S2A Predictions.

Transform z-score predictions to target species scale using a small
number of labeled samples (20-50 examples).

Methods:
- Affine calibration: y = α * z + β (simple, robust)
- Isotonic calibration: Non-parametric monotonic mapping

The key insight is that z-scores remove dataset-specific scale, but
a small calibration step can recover the target scale accurately.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import spearmanr, pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


@dataclass
class CalibrationStats:
    """Statistics from calibration fitting."""
    alpha: float  # Scale factor
    beta: float   # Offset
    r2_fit: float  # R² on calibration set
    n_samples: int


@dataclass
class CalibrationEvaluation:
    """Evaluation results after calibration."""
    spearman_rho: float
    pearson_r: float
    r2: float
    mse: float
    n_samples: int

    # Comparison to uncalibrated
    spearman_improvement: float
    pearson_improvement: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'spearman_rho': self.spearman_rho,
            'pearson_r': self.pearson_r,
            'r2': self.r2,
            'mse': self.mse,
            'n_samples': self.n_samples,
            'spearman_improvement': self.spearman_improvement,
            'pearson_improvement': self.pearson_improvement
        }


class AffineCalibrator:
    """
    Affine calibration: y = α * z + β

    Learns scale (α) and offset (β) from a small number of labeled samples
    to transform z-score predictions to the target species' activity scale.

    This is robust because:
    1. Only 2 parameters to learn (low variance)
    2. Preserves relative ranking (Spearman stays same)
    3. Can significantly improve Pearson r and MSE
    """

    def __init__(self):
        """Initialize affine calibrator."""
        self.alpha: float = 1.0
        self.beta: float = 0.0
        self._is_fitted = False
        self.stats: Optional[CalibrationStats] = None

    def fit(
        self,
        z_pred: np.ndarray,
        y_true: np.ndarray
    ) -> 'AffineCalibrator':
        """
        Fit affine calibration parameters.

        Args:
            z_pred: Predicted z-scores from S2A head
            y_true: True activity values (in target scale)

        Returns:
            Self for chaining
        """
        # Simple linear regression: y = α * z + β
        z_pred = np.asarray(z_pred).reshape(-1, 1)
        y_true = np.asarray(y_true).ravel()

        reg = LinearRegression()
        reg.fit(z_pred, y_true)

        self.alpha = float(reg.coef_[0])
        self.beta = float(reg.intercept_)

        # Compute fit statistics
        y_calibrated = self.transform(z_pred.ravel())
        r2 = r2_score(y_true, y_calibrated)

        self.stats = CalibrationStats(
            alpha=self.alpha,
            beta=self.beta,
            r2_fit=r2,
            n_samples=len(y_true)
        )

        self._is_fitted = True
        return self

    def transform(self, z_pred: np.ndarray) -> np.ndarray:
        """
        Apply affine calibration.

        Args:
            z_pred: Predicted z-scores

        Returns:
            Calibrated predictions in target scale
        """
        return self.alpha * np.asarray(z_pred) + self.beta

    def fit_transform(
        self,
        z_pred: np.ndarray,
        y_true: np.ndarray
    ) -> np.ndarray:
        """Fit and transform."""
        self.fit(z_pred, y_true)
        return self.transform(z_pred)

    def get_params(self) -> Tuple[float, float]:
        """Get (alpha, beta) parameters."""
        return self.alpha, self.beta


class IsotonicCalibrator:
    """
    Isotonic calibration: non-parametric monotonic mapping.

    More flexible than affine, but requires more samples (50+)
    and may overfit with few samples.
    """

    def __init__(self, out_of_bounds: str = 'clip'):
        """
        Initialize isotonic calibrator.

        Args:
            out_of_bounds: How to handle predictions outside training range.
                'clip' (default), 'nan', or 'raise'
        """
        self.model = IsotonicRegression(out_of_bounds=out_of_bounds)
        self._is_fitted = False

    def fit(
        self,
        z_pred: np.ndarray,
        y_true: np.ndarray
    ) -> 'IsotonicCalibrator':
        """
        Fit isotonic calibration.

        Args:
            z_pred: Predicted z-scores
            y_true: True activity values

        Returns:
            Self for chaining
        """
        self.model.fit(z_pred, y_true)
        self._is_fitted = True
        return self

    def transform(self, z_pred: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.

        Args:
            z_pred: Predicted z-scores

        Returns:
            Calibrated predictions
        """
        if not self._is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        return self.model.predict(z_pred)

    def fit_transform(
        self,
        z_pred: np.ndarray,
        y_true: np.ndarray
    ) -> np.ndarray:
        """Fit and transform."""
        self.fit(z_pred, y_true)
        return self.transform(z_pred)


def evaluate_calibration(
    z_pred: np.ndarray,
    y_true: np.ndarray,
    calibrator: AffineCalibrator,
    z_pred_uncal: np.ndarray = None
) -> CalibrationEvaluation:
    """
    Evaluate calibration performance.

    Args:
        z_pred: Z-score predictions (for calibration samples)
        y_true: True activity values
        calibrator: Fitted calibrator
        z_pred_uncal: Optional uncalibrated predictions for comparison

    Returns:
        CalibrationEvaluation with metrics
    """
    y_calibrated = calibrator.transform(z_pred)

    spearman = spearmanr(y_true, y_calibrated)[0]
    pearson = pearsonr(y_true, y_calibrated)[0]
    r2 = r2_score(y_true, y_calibrated)
    mse = mean_squared_error(y_true, y_calibrated)

    # Compute improvement over uncalibrated
    if z_pred_uncal is not None:
        spearman_uncal = spearmanr(y_true, z_pred_uncal)[0]
        pearson_uncal = pearsonr(y_true, z_pred_uncal)[0]
    else:
        spearman_uncal = spearmanr(y_true, z_pred)[0]
        pearson_uncal = pearsonr(y_true, z_pred)[0]

    return CalibrationEvaluation(
        spearman_rho=spearman,
        pearson_r=pearson,
        r2=r2,
        mse=mse,
        n_samples=len(y_true),
        spearman_improvement=spearman - spearman_uncal,
        pearson_improvement=pearson - pearson_uncal
    )


def calibration_curve_analysis(
    z_pred: np.ndarray,
    y_true: np.ndarray,
    sample_sizes: list = None,
    n_repeats: int = 10,
    random_seed: int = 42
) -> Dict[int, Dict[str, float]]:
    """
    Analyze calibration performance vs. number of calibration samples.

    Args:
        z_pred: All z-score predictions
        y_true: All true activity values
        sample_sizes: List of calibration sample sizes to test
        n_repeats: Number of random repeats per sample size
        random_seed: Random seed for reproducibility

    Returns:
        Dict mapping sample_size to average metrics
    """
    sample_sizes = sample_sizes or [10, 20, 50, 100, 200]
    rng = np.random.RandomState(random_seed)

    n_total = len(y_true)
    results = {}

    for n_samples in sample_sizes:
        if n_samples >= n_total:
            continue

        metrics_list = {
            'spearman': [],
            'pearson': [],
            'r2': [],
            'mse': []
        }

        for _ in range(n_repeats):
            # Random split
            indices = rng.permutation(n_total)
            cal_idx = indices[:n_samples]
            test_idx = indices[n_samples:]

            z_cal = z_pred[cal_idx]
            y_cal = y_true[cal_idx]
            z_test = z_pred[test_idx]
            y_test = y_true[test_idx]

            # Fit calibrator on calibration set
            calibrator = AffineCalibrator()
            calibrator.fit(z_cal, y_cal)

            # Evaluate on test set
            y_calibrated = calibrator.transform(z_test)

            metrics_list['spearman'].append(spearmanr(y_test, y_calibrated)[0])
            metrics_list['pearson'].append(pearsonr(y_test, y_calibrated)[0])
            metrics_list['r2'].append(r2_score(y_test, y_calibrated))
            metrics_list['mse'].append(mean_squared_error(y_test, y_calibrated))

        # Average over repeats
        results[n_samples] = {
            'spearman_mean': np.mean(metrics_list['spearman']),
            'spearman_std': np.std(metrics_list['spearman']),
            'pearson_mean': np.mean(metrics_list['pearson']),
            'pearson_std': np.std(metrics_list['pearson']),
            'r2_mean': np.mean(metrics_list['r2']),
            'r2_std': np.std(metrics_list['r2']),
            'mse_mean': np.mean(metrics_list['mse']),
            'mse_std': np.std(metrics_list['mse']),
        }

    return results


def select_calibration_samples(
    z_pred: np.ndarray,
    n_samples: int,
    method: str = 'stratified',
    random_seed: int = 42
) -> np.ndarray:
    """
    Select samples for calibration.

    Args:
        z_pred: Z-score predictions (used for stratification)
        n_samples: Number of samples to select
        method: Selection method
            'random': Random sampling
            'stratified': Stratified by z-score percentiles (recommended)
            'extremes': Select from extremes + middle
        random_seed: Random seed

    Returns:
        Indices of selected samples
    """
    rng = np.random.RandomState(random_seed)
    n_total = len(z_pred)

    if n_samples >= n_total:
        return np.arange(n_total)

    if method == 'random':
        return rng.choice(n_total, size=n_samples, replace=False)

    elif method == 'stratified':
        # Divide into percentile bins and sample from each
        n_bins = min(n_samples, 10)
        samples_per_bin = n_samples // n_bins

        indices = []
        percentiles = np.percentile(z_pred, np.linspace(0, 100, n_bins + 1))

        for i in range(n_bins):
            low, high = percentiles[i], percentiles[i + 1]
            bin_indices = np.where(
                (z_pred >= low) & (z_pred < high if i < n_bins - 1 else z_pred <= high)
            )[0]

            if len(bin_indices) > 0:
                n_select = min(samples_per_bin, len(bin_indices))
                selected = rng.choice(bin_indices, size=n_select, replace=False)
                indices.extend(selected)

        # Fill remaining with random
        remaining = n_samples - len(indices)
        if remaining > 0:
            available = np.setdiff1d(np.arange(n_total), indices)
            extra = rng.choice(available, size=min(remaining, len(available)), replace=False)
            indices.extend(extra)

        return np.array(indices[:n_samples])

    elif method == 'extremes':
        # Select from high/low extremes plus middle
        sorted_idx = np.argsort(z_pred)

        n_extreme = n_samples // 3
        n_middle = n_samples - 2 * n_extreme

        low_idx = sorted_idx[:n_extreme]
        high_idx = sorted_idx[-n_extreme:]

        middle_start = len(sorted_idx) // 2 - n_middle // 2
        middle_idx = sorted_idx[middle_start:middle_start + n_middle]

        return np.concatenate([low_idx, middle_idx, high_idx])

    else:
        raise ValueError(f"Unknown calibration method: {method}")
