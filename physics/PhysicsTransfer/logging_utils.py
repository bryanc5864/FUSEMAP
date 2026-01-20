"""
Comprehensive logging utilities for PhysicsTransfer experiments.

Provides:
- Structured experiment logging with timestamps
- Hyperparameter JSON dumps
- Per-fold/epoch metrics tracking
- CSV metric exports
- Timing utilities
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import numpy as np


@dataclass
class FoldMetrics:
    """Metrics from a single CV fold."""
    fold: int
    train_size: int
    val_size: int
    train_pearson: float
    val_pearson: float
    train_spearman: float
    val_spearman: float
    train_mse: float
    val_mse: float
    train_r2: float
    val_r2: float
    fit_time_sec: float


@dataclass
class EpochMetrics:
    """Metrics from a single training epoch (for neural models)."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_pearson: Optional[float] = None
    val_pearson: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    time_sec: float = 0.0


@dataclass
class ExperimentMetrics:
    """Full experiment metrics."""
    experiment_name: str
    timestamp: str
    config: Dict[str, Any]

    # Data info
    n_train_samples: int = 0
    n_val_samples: int = 0
    n_test_samples: int = 0
    n_features: int = 0
    feature_names: List[str] = field(default_factory=list)

    # CV metrics
    fold_metrics: List[FoldMetrics] = field(default_factory=list)
    cv_pearson_mean: float = 0.0
    cv_pearson_std: float = 0.0
    cv_spearman_mean: float = 0.0
    cv_spearman_std: float = 0.0

    # Final metrics
    final_train_pearson: float = 0.0
    final_val_pearson: float = 0.0
    final_test_pearson: float = 0.0

    # Timing
    total_time_sec: float = 0.0
    data_load_time_sec: float = 0.0
    train_time_sec: float = 0.0
    eval_time_sec: float = 0.0


class ExperimentLogger:
    """
    Comprehensive experiment logger.

    Usage:
        logger = ExperimentLogger('my_experiment', output_dir='results/')
        logger.log_hyperparams(config_dict)
        logger.log_data_info(n_train=1000, n_features=245)

        for fold in range(5):
            # ... training ...
            logger.log_fold_metrics(fold, train_metrics, val_metrics, time)

        logger.log_final_metrics(train_r, val_r, test_r)
        logger.save()
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()

        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir) / f"{experiment_name}_{self.timestamp}"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

        # Initialize metrics storage
        self.metrics = ExperimentMetrics(
            experiment_name=experiment_name,
            timestamp=self.timestamp,
            config={}
        )

        # Epoch metrics for MLP training
        self.epoch_metrics: List[EpochMetrics] = []

        # Setup Python logger
        self.logger = self._setup_logger(console_level, file_level)

        self.logger.info(f"{'='*60}")
        self.logger.info(f"Experiment: {experiment_name}")
        self.logger.info(f"Timestamp: {self.timestamp}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info(f"{'='*60}")

    def _setup_logger(self, console_level: int, file_level: int) -> logging.Logger:
        """Setup Python logger with console and file handlers."""
        logger = logging.getLogger(f"PhysicsTransfer.{self.experiment_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []  # Clear existing handlers

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File handler
        if self.output_dir:
            file_handler = logging.FileHandler(
                self.output_dir / 'experiment.log'
            )
            file_handler.setLevel(file_level)
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)

        return logger

    def log_hyperparams(self, config: Dict[str, Any]):
        """Log hyperparameters and save to JSON."""
        self.metrics.config = config

        self.logger.info("Hyperparameters:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

        if self.output_dir:
            with open(self.output_dir / 'hyperparams.json', 'w') as f:
                json.dump(config, f, indent=2, default=str)
            self.logger.debug(f"Saved hyperparams to {self.output_dir / 'hyperparams.json'}")

    def log_data_info(
        self,
        n_train: int,
        n_val: int = 0,
        n_test: int = 0,
        n_features: int = 0,
        feature_names: List[str] = None,
        source_datasets: List[str] = None,
        target_dataset: str = None
    ):
        """Log data information."""
        self.metrics.n_train_samples = n_train
        self.metrics.n_val_samples = n_val
        self.metrics.n_test_samples = n_test
        self.metrics.n_features = n_features
        self.metrics.feature_names = feature_names or []

        self.logger.info("Data Info:")
        self.logger.info(f"  Train samples: {n_train:,}")
        if n_val > 0:
            self.logger.info(f"  Val samples: {n_val:,}")
        if n_test > 0:
            self.logger.info(f"  Test samples: {n_test:,}")
        self.logger.info(f"  Features: {n_features}")

        if source_datasets:
            self.logger.info(f"  Source datasets: {source_datasets}")
        if target_dataset:
            self.logger.info(f"  Target dataset: {target_dataset}")

    def log_fold_start(self, fold: int, n_folds: int, train_size: int, val_size: int):
        """Log start of a CV fold."""
        self.logger.info(f"Fold {fold+1}/{n_folds}: train={train_size:,}, val={val_size:,}")

    def log_fold_metrics(
        self,
        fold: int,
        train_pearson: float,
        val_pearson: float,
        train_spearman: float = 0.0,
        val_spearman: float = 0.0,
        train_mse: float = 0.0,
        val_mse: float = 0.0,
        train_r2: float = 0.0,
        val_r2: float = 0.0,
        train_size: int = 0,
        val_size: int = 0,
        fit_time: float = 0.0
    ):
        """Log metrics from a CV fold."""
        fold_metrics = FoldMetrics(
            fold=fold,
            train_size=train_size,
            val_size=val_size,
            train_pearson=train_pearson,
            val_pearson=val_pearson,
            train_spearman=train_spearman,
            val_spearman=val_spearman,
            train_mse=train_mse,
            val_mse=val_mse,
            train_r2=train_r2,
            val_r2=val_r2,
            fit_time_sec=fit_time
        )
        self.metrics.fold_metrics.append(fold_metrics)

        self.logger.info(
            f"  Fold {fold+1}: train_r={train_pearson:.4f}, val_r={val_pearson:.4f}, "
            f"time={fit_time:.1f}s"
        )

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float = None,
        train_pearson: float = None,
        val_pearson: float = None,
        learning_rate: float = None,
        gradient_norm: float = None,
        time_sec: float = 0.0
    ):
        """Log metrics from a training epoch (for neural models)."""
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_pearson=train_pearson,
            val_pearson=val_pearson,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            time_sec=time_sec
        )
        self.epoch_metrics.append(epoch_metrics)

        msg = f"Epoch {epoch:3d}: train_loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f", val_loss={val_loss:.4f}"
        if val_pearson is not None:
            msg += f", val_r={val_pearson:.4f}"
        if gradient_norm is not None:
            msg += f", grad_norm={gradient_norm:.4f}"
        msg += f", time={time_sec:.1f}s"

        self.logger.debug(msg)

    def log_cv_summary(self):
        """Log summary of CV results."""
        if not self.metrics.fold_metrics:
            return

        val_pearsons = [f.val_pearson for f in self.metrics.fold_metrics]
        val_spearmans = [f.val_spearman for f in self.metrics.fold_metrics]

        self.metrics.cv_pearson_mean = np.mean(val_pearsons)
        self.metrics.cv_pearson_std = np.std(val_pearsons)
        self.metrics.cv_spearman_mean = np.mean(val_spearmans)
        self.metrics.cv_spearman_std = np.std(val_spearmans)

        total_time = sum(f.fit_time_sec for f in self.metrics.fold_metrics)

        self.logger.info(f"CV Summary:")
        self.logger.info(f"  Pearson r: {self.metrics.cv_pearson_mean:.4f} +/- {self.metrics.cv_pearson_std:.4f}")
        self.logger.info(f"  Spearman r: {self.metrics.cv_spearman_mean:.4f} +/- {self.metrics.cv_spearman_std:.4f}")
        self.logger.info(f"  Total CV time: {total_time:.1f}s")

    def log_final_metrics(
        self,
        train_pearson: float = 0.0,
        val_pearson: float = 0.0,
        test_pearson: float = 0.0,
        additional: Dict[str, float] = None
    ):
        """Log final evaluation metrics."""
        self.metrics.final_train_pearson = train_pearson
        self.metrics.final_val_pearson = val_pearson
        self.metrics.final_test_pearson = test_pearson

        self.logger.info("Final Metrics:")
        if train_pearson != 0:
            self.logger.info(f"  Train Pearson r: {train_pearson:.4f}")
        if val_pearson != 0:
            self.logger.info(f"  Val Pearson r: {val_pearson:.4f}")
        if test_pearson != 0:
            self.logger.info(f"  Test Pearson r: {test_pearson:.4f}")

        if additional:
            for key, value in additional.items():
                self.logger.info(f"  {key}: {value:.4f}")

    def log_transfer_result(
        self,
        source_pearson: float,
        target_pearson: float,
        transfer_efficiency: float
    ):
        """Log transfer learning results."""
        self.logger.info("Transfer Results:")
        self.logger.info(f"  Source Pearson r: {source_pearson:.4f}")
        self.logger.info(f"  Target Pearson r: {target_pearson:.4f}")
        self.logger.info(f"  Transfer efficiency: {transfer_efficiency:.1%}")

    def log_feature_importance(self, importances: Dict[str, float], top_n: int = 10):
        """Log top feature importances."""
        sorted_imp = sorted(importances.items(), key=lambda x: -x[1])[:top_n]

        self.logger.info(f"Top {top_n} Features:")
        for i, (name, imp) in enumerate(sorted_imp, 1):
            self.logger.info(f"  {i}. {name}: {imp:.4f}")

    def log_timing(self, stage: str, time_sec: float):
        """Log timing for a specific stage."""
        if stage == 'data_load':
            self.metrics.data_load_time_sec = time_sec
        elif stage == 'train':
            self.metrics.train_time_sec = time_sec
        elif stage == 'eval':
            self.metrics.eval_time_sec = time_sec

        self.logger.info(f"Timing - {stage}: {time_sec:.1f}s")

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def save(self) -> Dict[str, Path]:
        """Save all logs and metrics to files."""
        if not self.output_dir:
            self.logger.warning("No output directory specified, skipping save")
            return {}

        self.metrics.total_time_sec = time.time() - self.start_time

        files = {}

        # Save full metrics as JSON
        metrics_path = self.output_dir / 'metrics.json'
        metrics_dict = asdict(self.metrics)
        # Convert fold metrics
        metrics_dict['fold_metrics'] = [asdict(f) for f in self.metrics.fold_metrics]
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        files['metrics'] = metrics_path

        # Save fold metrics as CSV
        if self.metrics.fold_metrics:
            fold_csv_path = self.output_dir / 'fold_metrics.csv'
            import csv
            with open(fold_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.metrics.fold_metrics[0]).keys())
                writer.writeheader()
                for fold in self.metrics.fold_metrics:
                    writer.writerow(asdict(fold))
            files['fold_csv'] = fold_csv_path

        # Save epoch metrics as CSV (for neural models)
        if self.epoch_metrics:
            epoch_csv_path = self.output_dir / 'epoch_metrics.csv'
            import csv
            with open(epoch_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.epoch_metrics[0]).keys())
                writer.writeheader()
                for epoch in self.epoch_metrics:
                    writer.writerow(asdict(epoch))
            files['epoch_csv'] = epoch_csv_path

        self.logger.info(f"{'='*60}")
        self.logger.info(f"Experiment Complete")
        self.logger.info(f"Total time: {self.metrics.total_time_sec:.1f}s")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"{'='*60}")

        return files


class Timer:
    """Simple timer context manager."""

    def __init__(self, name: str = None, logger: ExperimentLogger = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.logger and self.name:
            self.logger.log_timing(self.name, self.elapsed)


def get_config_dict(config) -> Dict[str, Any]:
    """Extract config as dictionary for logging."""
    if hasattr(config, '__dict__'):
        return {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    return dict(config) if config else {}
