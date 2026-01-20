"""
Inference module for S2A predictions.

Supports multiple output modes:
- zscore: Raw z-score predictions (no calibration)
- calibrated: Affine-calibrated predictions using small labeled set
- ranking: Percentile rankings (robust for zero-shot)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from .config import S2AConfig, S2A_DATASETS
from .universal_features import UniversalFeatureExtractor
from .universal_head import UniversalS2AHead, EnsembleS2AHead
from .calibration import AffineCalibrator, select_calibration_samples
from .training import UniversalS2ATrainer


@dataclass
class S2APrediction:
    """Container for S2A predictions."""
    predictions: np.ndarray
    mode: str  # 'zscore', 'calibrated', 'ranking'
    calibrator: Optional[AffineCalibrator] = None
    feature_names: Optional[List[str]] = None
    n_samples: int = 0

    def to_dataframe(self, sequence_ids: List[str] = None) -> pd.DataFrame:
        """
        Convert predictions to DataFrame.

        Args:
            sequence_ids: Optional list of sequence identifiers

        Returns:
            DataFrame with predictions
        """
        df = pd.DataFrame({
            f's2a_{self.mode}': self.predictions
        })

        if sequence_ids is not None:
            df.insert(0, 'sequence_id', sequence_ids)

        return df


class S2APredictor:
    """
    Predictor for zero-shot and calibrated S2A inference.

    Usage:
        # Load trained model
        predictor = S2APredictor.from_checkpoint('results/s2a/')

        # Zero-shot prediction
        preds = predictor.predict_zscore(X_features)

        # Calibrated prediction (with 50 labeled samples)
        preds = predictor.predict_calibrated(X_features, X_cal, y_cal)

        # Ranking prediction
        preds = predictor.predict_ranking(X_features)
    """

    def __init__(
        self,
        head: Union[UniversalS2AHead, EnsembleS2AHead],
        feature_extractor: UniversalFeatureExtractor,
        config: S2AConfig = None
    ):
        """
        Initialize predictor.

        Args:
            head: Fitted S2A head (single or ensemble)
            feature_extractor: Fitted feature extractor
            config: S2AConfig
        """
        self.head = head
        self.feature_extractor = feature_extractor
        self.config = config or S2AConfig()
        self._calibrator: Optional[AffineCalibrator] = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        name: str = 'universal_s2a',
        config: S2AConfig = None
    ) -> 'S2APredictor':
        """
        Load predictor from checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint
            name: Checkpoint name prefix
            config: Optional config override

        Returns:
            Loaded S2APredictor
        """
        head, extractor = UniversalS2ATrainer.load_checkpoint(
            checkpoint_dir, name, config
        )
        return cls(head, extractor, config)

    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare features for prediction.

        Assumes X is already aligned to expected feature order.
        Applies scaling.
        """
        return self.feature_extractor.transform(X)

    def predict_zscore(self, X: np.ndarray) -> S2APrediction:
        """
        Predict z-scores (zero-shot, no calibration).

        Args:
            X: Physics features (n_samples, n_features)

        Returns:
            S2APrediction with z-score predictions
        """
        X_scaled = self._prepare_features(X)
        predictions = self.head.predict_zscore(X_scaled)

        return S2APrediction(
            predictions=predictions,
            mode='zscore',
            feature_names=self.feature_extractor.feature_names,
            n_samples=len(predictions)
        )

    def predict_ranking(self, X: np.ndarray) -> S2APrediction:
        """
        Predict percentile rankings.

        Rankings are robust for zero-shot transfer since they
        only depend on relative ordering.

        Args:
            X: Physics features

        Returns:
            S2APrediction with ranking predictions (0-100)
        """
        X_scaled = self._prepare_features(X)
        predictions = self.head.predict_ranking(X_scaled)

        return S2APrediction(
            predictions=predictions,
            mode='ranking',
            feature_names=self.feature_extractor.feature_names,
            n_samples=len(predictions)
        )

    def fit_calibrator(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> AffineCalibrator:
        """
        Fit a calibrator using labeled samples.

        Args:
            X_cal: Calibration features
            y_cal: Calibration activity values (in target scale)

        Returns:
            Fitted AffineCalibrator
        """
        X_scaled = self._prepare_features(X_cal)
        z_cal = self.head.predict_zscore(X_scaled)

        calibrator = AffineCalibrator()
        calibrator.fit(z_cal, y_cal)

        self._calibrator = calibrator
        return calibrator

    def predict_calibrated(
        self,
        X: np.ndarray,
        X_cal: np.ndarray = None,
        y_cal: np.ndarray = None,
        calibrator: AffineCalibrator = None
    ) -> S2APrediction:
        """
        Predict with calibration to target scale.

        Either provide (X_cal, y_cal) to fit a new calibrator,
        or provide a pre-fitted calibrator.

        Args:
            X: Physics features for prediction
            X_cal: Optional calibration features
            y_cal: Optional calibration activity values
            calibrator: Optional pre-fitted calibrator

        Returns:
            S2APrediction with calibrated predictions
        """
        # Fit calibrator if needed
        if calibrator is not None:
            self._calibrator = calibrator
        elif X_cal is not None and y_cal is not None:
            self.fit_calibrator(X_cal, y_cal)
        elif self._calibrator is None:
            raise ValueError(
                "No calibrator available. Provide (X_cal, y_cal) or a calibrator."
            )

        # Predict
        X_scaled = self._prepare_features(X)
        z_pred = self.head.predict_zscore(X_scaled)
        predictions = self._calibrator.transform(z_pred)

        return S2APrediction(
            predictions=predictions,
            mode='calibrated',
            calibrator=self._calibrator,
            feature_names=self.feature_extractor.feature_names,
            n_samples=len(predictions)
        )

    def predict(
        self,
        X: np.ndarray,
        mode: str = None,
        X_cal: np.ndarray = None,
        y_cal: np.ndarray = None
    ) -> S2APrediction:
        """
        Predict using specified mode.

        Args:
            X: Physics features
            mode: Output mode ('zscore', 'calibrated', 'ranking')
            X_cal: Calibration features (required for 'calibrated' mode)
            y_cal: Calibration activity (required for 'calibrated' mode)

        Returns:
            S2APrediction
        """
        mode = mode or self.config.output_mode

        if mode == 'zscore':
            return self.predict_zscore(X)
        elif mode == 'ranking':
            return self.predict_ranking(X)
        elif mode == 'calibrated':
            return self.predict_calibrated(X, X_cal, y_cal)
        else:
            raise ValueError(f"Unknown output mode: {mode}")


def predict_from_descriptors_file(
    predictor: S2APredictor,
    input_file: str,
    output_file: str = None,
    mode: str = 'zscore',
    calibration_file: str = None,
    calibration_activity_col: str = None,
    calibration_n_samples: int = 50
) -> pd.DataFrame:
    """
    Predict from a descriptors TSV file.

    Args:
        predictor: Fitted S2APredictor
        input_file: Path to input descriptors TSV
        output_file: Optional path to save predictions
        mode: Output mode
        calibration_file: Optional file with calibration data
        calibration_activity_col: Activity column name for calibration
        calibration_n_samples: Number of calibration samples to use

    Returns:
        DataFrame with predictions
    """
    # Load input
    df = pd.read_csv(input_file, sep='\t')
    print(f"Loaded {len(df)} sequences from {input_file}")

    # Extract features
    feature_names = predictor.feature_extractor.feature_names
    available_features = [f for f in feature_names if f in df.columns]

    if len(available_features) != len(feature_names):
        missing = set(feature_names) - set(available_features)
        raise ValueError(f"Missing features in input file: {missing}")

    X = df[feature_names].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Handle calibration
    X_cal, y_cal = None, None
    if mode == 'calibrated':
        if calibration_file is None:
            raise ValueError("Calibration file required for calibrated mode")

        cal_df = pd.read_csv(calibration_file, sep='\t')

        if calibration_activity_col not in cal_df.columns:
            raise ValueError(f"Activity column '{calibration_activity_col}' not in calibration file")

        X_cal_full = cal_df[feature_names].values.astype(np.float32)
        X_cal_full = np.nan_to_num(X_cal_full, nan=0.0, posinf=0.0, neginf=0.0)
        y_cal_full = cal_df[calibration_activity_col].values

        # Select calibration samples
        if calibration_n_samples < len(y_cal_full):
            # Get z-scores for stratified selection
            pred_temp = predictor.predict_zscore(X_cal_full)
            cal_idx = select_calibration_samples(
                pred_temp.predictions,
                calibration_n_samples,
                method='stratified'
            )
            X_cal = X_cal_full[cal_idx]
            y_cal = y_cal_full[cal_idx]
        else:
            X_cal = X_cal_full
            y_cal = y_cal_full

        print(f"Using {len(y_cal)} calibration samples")

    # Predict
    result = predictor.predict(X, mode=mode, X_cal=X_cal, y_cal=y_cal)

    # Build output dataframe
    output_df = pd.DataFrame()

    # Add sequence ID if available
    if 'sequence' in df.columns:
        output_df['sequence'] = df['sequence']
    elif 'seq_id' in df.columns:
        output_df['seq_id'] = df['seq_id']

    output_df[f's2a_{mode}'] = result.predictions

    # Save if output file specified
    if output_file is not None:
        output_df.to_csv(output_file, sep='\t', index=False)
        print(f"Saved predictions to {output_file}")

    return output_df


def predict_for_dataset(
    predictor: S2APredictor,
    dataset_name: str,
    split: str = 'test',
    mode: str = 'zscore',
    calibration_split: str = 'train',
    calibration_n_samples: int = 50
) -> Tuple[S2APrediction, Optional[np.ndarray]]:
    """
    Predict for a registered dataset.

    Args:
        predictor: Fitted S2APredictor
        dataset_name: Dataset name from registry
        split: Split to predict on
        mode: Output mode
        calibration_split: Split to use for calibration samples
        calibration_n_samples: Number of calibration samples

    Returns:
        Tuple of (predictions, true_activity or None)
    """
    if dataset_name not in S2A_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load test data
    extractor = predictor.feature_extractor
    X_test, y_test, test_features = extractor.load_dataset_features(
        dataset_name, split, return_activity=True
    )

    # Align features
    feature_names = extractor.feature_names
    test_feature_idx = [test_features.index(f) for f in feature_names]
    X_test_aligned = X_test[:, test_feature_idx]

    # Handle calibration
    X_cal, y_cal = None, None
    if mode == 'calibrated':
        X_cal_full, y_cal_full, cal_features = extractor.load_dataset_features(
            dataset_name, calibration_split, return_activity=True
        )

        cal_feature_idx = [cal_features.index(f) for f in feature_names]
        X_cal_full_aligned = X_cal_full[:, cal_feature_idx]

        # Select calibration samples
        if calibration_n_samples < len(y_cal_full):
            pred_temp = predictor.predict_zscore(X_cal_full_aligned)
            cal_idx = select_calibration_samples(
                pred_temp.predictions,
                calibration_n_samples,
                method='stratified'
            )
            X_cal = X_cal_full_aligned[cal_idx]
            y_cal = y_cal_full[cal_idx]
        else:
            X_cal = X_cal_full_aligned
            y_cal = y_cal_full

        print(f"Using {len(y_cal)} calibration samples from {calibration_split}")

    # Predict
    result = predictor.predict(X_test_aligned, mode=mode, X_cal=X_cal, y_cal=y_cal)

    return result, y_test
