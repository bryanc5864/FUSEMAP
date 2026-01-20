"""
Evaluation module for S2A.

Implements:
- Leave-one-species-out evaluation
- Calibration curve analysis
- Cross-validation within species
- Performance comparison between transfer scenarios
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error
import time

from .config import S2AConfig, S2A_DATASETS, S2A_DATASET_GROUPS
from .universal_features import UniversalFeatureExtractor
from .universal_head import UniversalS2AHead
from .calibration import AffineCalibrator, calibration_curve_analysis, select_calibration_samples
from .training import UniversalS2ATrainer, LeaveOneOutResults


@dataclass
class DatasetEvaluation:
    """Evaluation results for a single dataset."""
    dataset_name: str
    species: str
    kingdom: str

    # Zero-shot metrics
    zeroshot_spearman: float
    zeroshot_pearson: float
    zeroshot_r2: float
    zeroshot_mse: float

    # Calibrated metrics (50 samples)
    calibrated_spearman: float
    calibrated_pearson: float
    calibrated_r2: float
    calibrated_mse: float

    # Sample counts
    n_test_samples: int
    n_calibration_samples: int

    def to_dict(self) -> Dict:
        """Convert to dictionary with JSON-serializable types."""
        d = asdict(self)
        # Convert numpy types to Python types
        for k, v in d.items():
            if hasattr(v, 'item'):  # numpy scalar
                d[k] = v.item()
        return d


@dataclass
class FullEvaluationResults:
    """Results from full leave-one-out evaluation."""
    datasets_evaluated: List[str]
    per_dataset_results: Dict[str, DatasetEvaluation]
    summary_statistics: Dict[str, Dict[str, float]]
    config: Dict

    def to_dataframe(self) -> pd.DataFrame:
        """Convert per-dataset results to DataFrame."""
        records = [r.to_dict() for r in self.per_dataset_results.values()]
        return pd.DataFrame.from_records(records)

    def save(self, output_dir: str):
        """Save results to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save per-dataset results
        df = self.to_dataframe()
        df.to_csv(output_dir / 'per_dataset_results.csv', index=False)

        # Convert numpy types in summary_statistics
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            return obj

        # Save summary
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump({
                'datasets_evaluated': self.datasets_evaluated,
                'summary_statistics': convert_numpy(self.summary_statistics),
                'config': convert_numpy(self.config)
            }, f, indent=2)

        print(f"Saved evaluation results to {output_dir}")


class S2AEvaluator:
    """
    Evaluator for S2A zero-shot transfer.

    Implements leave-one-out evaluation across all datasets.
    """

    def __init__(self, config: S2AConfig = None):
        """
        Initialize evaluator.

        Args:
            config: S2AConfig
        """
        self.config = config or S2AConfig()
        self.trainer = UniversalS2ATrainer(config)

    def evaluate_single_holdout(
        self,
        all_datasets: List[str],
        holdout: str,
        train_split: str = 'train',
        test_split: str = 'test',
        calibration_n_samples: int = 50,
        verbose: bool = True
    ) -> DatasetEvaluation:
        """
        Evaluate zero-shot and calibrated performance on a single holdout.

        Args:
            all_datasets: All dataset names
            holdout: Dataset to hold out
            train_split: Split for training sources
            test_split: Split for testing holdout
            calibration_n_samples: Samples for calibration
            verbose: Print progress

        Returns:
            DatasetEvaluation for the holdout dataset
        """
        if holdout not in S2A_DATASETS:
            raise ValueError(f"Unknown holdout dataset: {holdout}")

        dataset_config = S2A_DATASETS[holdout]

        # Train on all except holdout
        source_datasets = [d for d in all_datasets if d != holdout]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating holdout: {holdout}")
            print(f"Sources: {source_datasets}")
            print(f"{'='*60}")

        head, extractor, train_results = self.trainer.train(
            source_datasets, train_split, verbose=verbose
        )

        # Load holdout test data
        X_test, y_test, test_features = extractor.load_dataset_features(
            holdout, test_split, return_activity=True
        )

        if y_test is None:
            raise ValueError(f"No activity values for {holdout} {test_split}")

        # Align features
        common_features = extractor.feature_names
        test_feature_idx = [test_features.index(f) for f in common_features]
        X_test_aligned = X_test[:, test_feature_idx]
        X_test_scaled = extractor.transform(X_test_aligned)

        # Zero-shot evaluation
        z_pred = head.predict_zscore(X_test_scaled)

        zs_spearman = spearmanr(y_test, z_pred)[0]
        zs_pearson = pearsonr(y_test, z_pred)[0]
        zs_r2 = r2_score(y_test, z_pred)
        zs_mse = mean_squared_error(y_test, z_pred)

        if verbose:
            print(f"\nZero-Shot Results:")
            print(f"  Spearman ρ: {zs_spearman:.4f}")
            print(f"  Pearson r: {zs_pearson:.4f}")

        # Calibrated evaluation (using subset of test data)
        n_cal = min(calibration_n_samples, len(y_test) // 3)

        if n_cal >= 10:
            # Select calibration samples
            cal_idx = select_calibration_samples(
                z_pred, n_cal, method='stratified',
                random_seed=self.config.random_seed
            )
            test_idx = np.setdiff1d(np.arange(len(y_test)), cal_idx)

            X_cal = X_test_scaled[cal_idx]
            y_cal = y_test[cal_idx]
            z_cal = z_pred[cal_idx]

            X_eval = X_test_scaled[test_idx]
            y_eval = y_test[test_idx]
            z_eval = z_pred[test_idx]

            # Fit calibrator
            calibrator = AffineCalibrator()
            calibrator.fit(z_cal, y_cal)

            # Calibrated predictions
            y_calibrated = calibrator.transform(z_eval)

            cal_spearman = spearmanr(y_eval, y_calibrated)[0]
            cal_pearson = pearsonr(y_eval, y_calibrated)[0]
            cal_r2 = r2_score(y_eval, y_calibrated)
            cal_mse = mean_squared_error(y_eval, y_calibrated)

            if verbose:
                print(f"\nCalibrated Results ({n_cal} samples):")
                print(f"  Spearman ρ: {cal_spearman:.4f}")
                print(f"  Pearson r: {cal_pearson:.4f}")
                print(f"  Calibration α={calibrator.alpha:.3f}, β={calibrator.beta:.3f}")
        else:
            # Not enough samples for calibration
            cal_spearman = np.nan
            cal_pearson = np.nan
            cal_r2 = np.nan
            cal_mse = np.nan
            n_cal = 0

        return DatasetEvaluation(
            dataset_name=holdout,
            species=dataset_config.species,
            kingdom=dataset_config.kingdom,
            zeroshot_spearman=zs_spearman,
            zeroshot_pearson=zs_pearson,
            zeroshot_r2=zs_r2,
            zeroshot_mse=zs_mse,
            calibrated_spearman=cal_spearman,
            calibrated_pearson=cal_pearson,
            calibrated_r2=cal_r2,
            calibrated_mse=cal_mse,
            n_test_samples=len(y_test),
            n_calibration_samples=n_cal
        )

    def run_full_evaluation(
        self,
        datasets: List[str] = None,
        train_split: str = 'train',
        test_split: str = 'test',
        calibration_n_samples: int = 50,
        verbose: bool = True
    ) -> FullEvaluationResults:
        """
        Run full leave-one-out evaluation on all datasets.

        Args:
            datasets: List of datasets (uses all if None)
            train_split: Split for training
            test_split: Split for testing
            calibration_n_samples: Samples for calibration
            verbose: Print progress

        Returns:
            FullEvaluationResults with all metrics
        """
        datasets = datasets or list(S2A_DATASETS.keys())

        print(f"\n{'#'*60}")
        print(f"# Full Leave-One-Out Evaluation")
        print(f"# Datasets: {len(datasets)}")
        print(f"{'#'*60}")

        per_dataset_results = {}
        start_time = time.time()

        for holdout in datasets:
            try:
                result = self.evaluate_single_holdout(
                    datasets, holdout,
                    train_split, test_split,
                    calibration_n_samples, verbose
                )
                per_dataset_results[holdout] = result

            except Exception as e:
                print(f"Error evaluating {holdout}: {e}")
                continue

        # Compute summary statistics
        summary = self._compute_summary_statistics(per_dataset_results)

        total_time = time.time() - start_time

        if verbose:
            print(f"\n{'='*60}")
            print(f"SUMMARY")
            print(f"{'='*60}")
            print(f"Total time: {total_time/60:.1f} minutes")
            print(f"\nZero-Shot Performance:")
            print(f"  Mean Spearman ρ: {summary['zeroshot']['spearman_mean']:.4f} "
                  f"± {summary['zeroshot']['spearman_std']:.4f}")
            print(f"  Mean Pearson r: {summary['zeroshot']['pearson_mean']:.4f} "
                  f"± {summary['zeroshot']['pearson_std']:.4f}")

            if 'calibrated' in summary:
                print(f"\nCalibrated Performance:")
                print(f"  Mean Spearman ρ: {summary['calibrated']['spearman_mean']:.4f} "
                      f"± {summary['calibrated']['spearman_std']:.4f}")
                print(f"  Mean Pearson r: {summary['calibrated']['pearson_mean']:.4f} "
                      f"± {summary['calibrated']['pearson_std']:.4f}")

        return FullEvaluationResults(
            datasets_evaluated=datasets,
            per_dataset_results=per_dataset_results,
            summary_statistics=summary,
            config=asdict(self.config)
        )

    def _compute_summary_statistics(
        self,
        results: Dict[str, DatasetEvaluation]
    ) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics from per-dataset results."""
        zs_spearman = [r.zeroshot_spearman for r in results.values()]
        zs_pearson = [r.zeroshot_pearson for r in results.values()]

        summary = {
            'zeroshot': {
                'spearman_mean': np.nanmean(zs_spearman),
                'spearman_std': np.nanstd(zs_spearman),
                'spearman_median': np.nanmedian(zs_spearman),
                'pearson_mean': np.nanmean(zs_pearson),
                'pearson_std': np.nanstd(zs_pearson),
                'pearson_median': np.nanmedian(zs_pearson),
            }
        }

        # Calibrated summary (if available)
        cal_spearman = [r.calibrated_spearman for r in results.values()
                       if not np.isnan(r.calibrated_spearman)]
        cal_pearson = [r.calibrated_pearson for r in results.values()
                      if not np.isnan(r.calibrated_pearson)]

        if cal_spearman:
            summary['calibrated'] = {
                'spearman_mean': np.mean(cal_spearman),
                'spearman_std': np.std(cal_spearman),
                'pearson_mean': np.mean(cal_pearson),
                'pearson_std': np.std(cal_pearson),
            }

        # By kingdom
        for kingdom in ['animal', 'plant']:
            kingdom_results = [r for r in results.values() if r.kingdom == kingdom]
            if kingdom_results:
                summary[f'zeroshot_{kingdom}'] = {
                    'spearman_mean': np.nanmean([r.zeroshot_spearman for r in kingdom_results]),
                    'spearman_std': np.nanstd([r.zeroshot_spearman for r in kingdom_results]),
                    'pearson_mean': np.nanmean([r.zeroshot_pearson for r in kingdom_results]),
                    'pearson_std': np.nanstd([r.zeroshot_pearson for r in kingdom_results]),
                    'n_datasets': len(kingdom_results),
                }

        return summary

    def evaluate_calibration_curve(
        self,
        all_datasets: List[str],
        holdout: str,
        sample_sizes: List[int] = None,
        n_repeats: int = 10,
        verbose: bool = True
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate calibration performance vs. number of calibration samples.

        Args:
            all_datasets: All dataset names
            holdout: Dataset to hold out
            sample_sizes: List of sample sizes to test
            n_repeats: Number of random repeats per size
            verbose: Print progress

        Returns:
            Dict mapping sample_size to metrics
        """
        sample_sizes = sample_sizes or [10, 20, 50, 100, 200]

        # Train model
        source_datasets = [d for d in all_datasets if d != holdout]
        head, extractor, _ = self.trainer.train(
            source_datasets, 'train', verbose=verbose
        )

        # Load holdout test data
        X_test, y_test, test_features = extractor.load_dataset_features(
            holdout, 'test', return_activity=True
        )

        # Align and scale
        common_features = extractor.feature_names
        test_feature_idx = [test_features.index(f) for f in common_features]
        X_test_aligned = X_test[:, test_feature_idx]
        X_test_scaled = extractor.transform(X_test_aligned)

        # Get predictions
        z_pred = head.predict_zscore(X_test_scaled)

        # Run calibration curve analysis
        results = calibration_curve_analysis(
            z_pred, y_test,
            sample_sizes=sample_sizes,
            n_repeats=n_repeats,
            random_seed=self.config.random_seed
        )

        if verbose:
            print(f"\nCalibration Curve for {holdout}:")
            print(f"{'N':>6} {'Spearman':>10} {'Pearson':>10} {'R²':>10}")
            print("-" * 40)
            for n, metrics in sorted(results.items()):
                print(f"{n:>6} {metrics['spearman_mean']:>10.4f} "
                      f"{metrics['pearson_mean']:>10.4f} "
                      f"{metrics['r2_mean']:>10.4f}")

        return results


def compare_transfer_scenarios(
    evaluator: S2AEvaluator,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare different transfer scenarios.

    Scenarios:
    1. Within-human: K562/HepG2 → WTC11
    2. Within-plant: Arabidopsis/Sorghum → Maize
    3. Human → Drosophila
    4. Animal → Plant
    5. Plant → Animal

    Returns:
        DataFrame with comparison results
    """
    scenarios = [
        {
            'name': 'within_human',
            'sources': ['K562', 'HepG2'],
            'holdout': 'WTC11',
            'description': 'Within-human cross-cell-type'
        },
        {
            'name': 'within_plant',
            'sources': ['arabidopsis_leaf', 'sorghum_leaf'],
            'holdout': 'maize_leaf',
            'description': 'Within-plant cross-species'
        },
        {
            'name': 'human_to_fly',
            'sources': ['K562', 'HepG2', 'WTC11'],
            'holdout': 'S2_dev',
            'description': 'Human → Drosophila'
        },
        {
            'name': 'animal_to_plant',
            'sources': ['K562', 'HepG2', 'WTC11', 'S2_dev'],
            'holdout': 'arabidopsis_leaf',
            'description': 'All animals → Arabidopsis'
        },
        {
            'name': 'plant_to_animal',
            'sources': ['arabidopsis_leaf', 'sorghum_leaf', 'maize_leaf'],
            'holdout': 'S2_dev',
            'description': 'All plants → Drosophila'
        },
    ]

    results = []

    for scenario in scenarios:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Scenario: {scenario['description']}")
            print(f"{'='*50}")

        try:
            all_datasets = scenario['sources'] + [scenario['holdout']]
            eval_result = evaluator.evaluate_single_holdout(
                all_datasets,
                scenario['holdout'],
                verbose=verbose
            )

            results.append({
                'scenario': scenario['name'],
                'description': scenario['description'],
                'sources': ', '.join(scenario['sources']),
                'holdout': scenario['holdout'],
                'zeroshot_spearman': eval_result.zeroshot_spearman,
                'zeroshot_pearson': eval_result.zeroshot_pearson,
                'calibrated_spearman': eval_result.calibrated_spearman,
                'calibrated_pearson': eval_result.calibrated_pearson,
                'n_test': eval_result.n_test_samples
            })

        except Exception as e:
            print(f"Error in scenario {scenario['name']}: {e}")
            continue

    df = pd.DataFrame(results)

    if verbose:
        print(f"\n{'='*60}")
        print("TRANSFER SCENARIO COMPARISON")
        print(f"{'='*60}")
        print(df.to_string(index=False))

    return df
