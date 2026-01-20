"""
Evaluation framework for PhysicsTransfer experiments.

Provides metrics, comparisons, and visualizations for transfer learning results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .config import TransferConfig, DATASETS, EXPERIMENTS, get_fusemap_root
from .protocols import TransferResult


@dataclass
class ComparisonMetrics:
    """Metrics comparing transfer to baseline."""
    baseline_pearson: float
    transfer_pearson: float
    improvement_absolute: float
    improvement_relative: float  # percentage improvement
    p_value: Optional[float] = None
    significant: bool = False


class TransferEvaluator:
    """
    Evaluate and compare transfer learning results.

    Provides:
    - Comparison between protocols
    - Statistical significance testing
    - Feature importance analysis
    - Result summarization and export
    """

    def __init__(self, config: TransferConfig = None):
        self.config = config or TransferConfig()
        self.results: Dict[str, Any] = {}

    def add_result(self, name: str, result: Any):
        """Add a result for tracking."""
        self.results[name] = result

    def compare_to_baseline(
        self,
        transfer_result: TransferResult,
        baseline_pearson: float
    ) -> ComparisonMetrics:
        """
        Compare transfer result to a baseline.

        Args:
            transfer_result: Transfer learning result
            baseline_pearson: Baseline Pearson r (e.g., train-from-scratch)

        Returns:
            ComparisonMetrics with comparison statistics
        """
        transfer_pearson = transfer_result.target_pearson
        if transfer_result.fine_tuned_pearson is not None:
            transfer_pearson = transfer_result.fine_tuned_pearson

        improvement_abs = transfer_pearson - baseline_pearson
        improvement_rel = (improvement_abs / abs(baseline_pearson) * 100) if baseline_pearson != 0 else 0

        return ComparisonMetrics(
            baseline_pearson=baseline_pearson,
            transfer_pearson=transfer_pearson,
            improvement_absolute=improvement_abs,
            improvement_relative=improvement_rel,
            significant=improvement_abs > 0.05  # Simple threshold
        )

    def compute_transfer_efficiency(
        self,
        source_pearson: float,
        target_pearson: float
    ) -> float:
        """
        Compute transfer efficiency (how well performance transfers).

        Returns ratio of target to source performance.
        Values close to 1.0 indicate good transfer.
        """
        if source_pearson <= 0:
            return 0.0
        return target_pearson / source_pearson

    def summarize_experiment(
        self,
        experiment_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Create summary DataFrame from experiment results.

        Args:
            experiment_results: Results dict from run_experiment

        Returns:
            DataFrame with summary statistics
        """
        rows = []

        # Zero-shot result
        if 'zero_shot' in experiment_results.get('protocols', {}):
            zs = experiment_results['protocols']['zero_shot']
            rows.append({
                'protocol': 'Zero-Shot',
                'source_pearson': zs.source_pearson,
                'target_pearson': zs.target_pearson,
                'transfer_efficiency': zs.transfer_efficiency,
                'n_features': zs.n_common_features,
                'fine_tune_samples': None
            })

        # Fine-tuning results
        if 'fine_tuning' in experiment_results.get('protocols', {}):
            for ft in experiment_results['protocols']['fine_tuning']:
                rows.append({
                    'protocol': 'Fine-Tuning',
                    'source_pearson': ft.source_pearson,
                    'target_pearson': ft.target_pearson,
                    'fine_tuned_pearson': ft.fine_tuned_pearson,
                    'transfer_efficiency': ft.transfer_efficiency,
                    'n_features': ft.n_common_features,
                    'fine_tune_samples': ft.fine_tune_n_samples
                })

        # Joint training results
        if 'joint_training' in experiment_results.get('protocols', {}):
            for name, jt in experiment_results['protocols']['joint_training'].items():
                rows.append({
                    'protocol': f'Joint-{name}',
                    'source_pearson': jt.source_pearson,
                    'target_pearson': jt.target_pearson,
                    'transfer_efficiency': jt.transfer_efficiency,
                    'n_features': jt.n_common_features,
                    'fine_tune_samples': None
                })

        return pd.DataFrame(rows)

    def compare_feature_contributions(
        self,
        results_list: List[TransferResult]
    ) -> pd.DataFrame:
        """
        Compare feature family contributions across experiments.

        Args:
            results_list: List of TransferResults

        Returns:
            DataFrame with feature contributions per experiment
        """
        rows = []
        for result in results_list:
            row = {
                'source': ','.join(result.source_datasets),
                'target': result.target_dataset,
                'protocol': result.protocol
            }
            row.update(result.feature_contributions)
            rows.append(row)

        return pd.DataFrame(rows)

    def generate_report(
        self,
        experiment_results: Dict[str, Any],
        output_path: Path = None
    ) -> str:
        """
        Generate a text report of experiment results.

        Args:
            experiment_results: Results from run_experiment
            output_path: Optional path to save report

        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"PHYSICS TRANSFER EXPERIMENT REPORT")
        lines.append(f"Experiment: {experiment_results.get('experiment', 'Unknown')}")
        lines.append(f"Description: {experiment_results.get('description', '')}")
        lines.append("=" * 70)

        # Zero-shot results
        if 'zero_shot' in experiment_results.get('protocols', {}):
            zs = experiment_results['protocols']['zero_shot']
            lines.append("\n## PROTOCOL 1: Zero-Shot Transfer")
            lines.append(f"Source datasets: {', '.join(zs.source_datasets)}")
            lines.append(f"Target dataset: {zs.target_dataset}")
            lines.append(f"Common features: {zs.n_common_features}")
            lines.append(f"\nPerformance:")
            lines.append(f"  Source Pearson r:  {zs.source_pearson:.4f}")
            lines.append(f"  Target Pearson r:  {zs.target_pearson:.4f}")
            lines.append(f"  Transfer efficiency: {zs.transfer_efficiency:.2%}")
            lines.append(f"\nFeature Contributions:")
            for family, contrib in sorted(zs.feature_contributions.items(), key=lambda x: -x[1]):
                lines.append(f"  {family}: {contrib:.1f}%")

        # Fine-tuning results
        if 'fine_tuning' in experiment_results.get('protocols', {}):
            lines.append("\n## PROTOCOL 2: Physics-Anchored Fine-Tuning")
            ft_results = experiment_results['protocols']['fine_tuning']
            lines.append("\n| N Samples | Zero-Shot | Fine-Tuned | Improvement |")
            lines.append("|-----------|-----------|------------|-------------|")
            for ft in ft_results:
                improvement = ft.fine_tuned_pearson - ft.target_pearson
                lines.append(f"| {ft.fine_tune_n_samples:9d} | {ft.target_pearson:.4f}    | {ft.fine_tuned_pearson:.4f}     | {improvement:+.4f}      |")

        # Joint training results
        if 'joint_training' in experiment_results.get('protocols', {}):
            lines.append("\n## PROTOCOL 3: Multi-Species Joint Training")
            jt_results = experiment_results['protocols']['joint_training']
            lines.append("\n| Dataset | Pearson r | Transfer Eff |")
            lines.append("|---------|-----------|--------------|")
            for name, jt in jt_results.items():
                lines.append(f"| {name:7s} | {jt.target_pearson:.4f}    | {jt.transfer_efficiency:.2%}       |")

        # Key findings
        lines.append("\n" + "=" * 70)
        lines.append("KEY FINDINGS")
        lines.append("=" * 70)

        if 'zero_shot' in experiment_results.get('protocols', {}):
            zs = experiment_results['protocols']['zero_shot']
            if zs.transfer_efficiency > 0.8:
                lines.append("✓ Excellent transfer: >80% of source performance maintained")
            elif zs.transfer_efficiency > 0.5:
                lines.append("○ Moderate transfer: 50-80% of source performance maintained")
            else:
                lines.append("✗ Limited transfer: <50% of source performance maintained")

        report = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")

        return report

    def export_results(
        self,
        experiment_results: Dict[str, Any],
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Export results to files.

        Args:
            experiment_results: Results from run_experiment
            output_dir: Output directory

        Returns:
            Dict mapping result type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = {}

        # Summary CSV
        summary_df = self.summarize_experiment(experiment_results)
        summary_path = output_dir / 'summary.csv'
        summary_df.to_csv(summary_path, index=False)
        files['summary'] = summary_path

        # Report
        report_path = output_dir / 'report.txt'
        self.generate_report(experiment_results, report_path)
        files['report'] = report_path

        # JSON with full results
        json_path = output_dir / 'results.json'
        json_results = self._results_to_json(experiment_results)
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        files['json'] = json_path

        return files

    def _results_to_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert results to JSON-serializable format."""
        json_results = {
            'experiment': results.get('experiment'),
            'description': results.get('description'),
            'protocols': {}
        }

        if 'zero_shot' in results.get('protocols', {}):
            zs = results['protocols']['zero_shot']
            json_results['protocols']['zero_shot'] = asdict(zs)

        if 'fine_tuning' in results.get('protocols', {}):
            json_results['protocols']['fine_tuning'] = [
                asdict(ft) for ft in results['protocols']['fine_tuning']
            ]

        if 'joint_training' in results.get('protocols', {}):
            json_results['protocols']['joint_training'] = {
                name: asdict(jt) for name, jt in results['protocols']['joint_training'].items()
            }

        return json_results


class BaselineComparator:
    """
    Compare transfer learning to baseline approaches.

    Baselines:
    1. Train from scratch on target (no transfer)
    2. Random features (sanity check)
    3. Sequence-only transfer (if applicable)
    """

    def __init__(self, config: TransferConfig = None):
        self.config = config or TransferConfig()

    def compute_scratch_baseline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = None
    ) -> float:
        """
        Compute baseline by training from scratch on target.

        Args:
            X: Target features
            y: Target activity
            n_samples: Number of samples to use (default: all)

        Returns:
            Pearson r from cross-validation
        """
        from .physics_probe import PhysicsActivityProbe

        if n_samples and n_samples < len(y):
            rng = np.random.RandomState(self.config.random_seed)
            idx = rng.choice(len(y), n_samples, replace=False)
            X, y = X[idx], y[idx]

        probe = PhysicsActivityProbe(self.config)
        results = probe.fit_evaluate_cv(X, y)
        return results.pearson_r

    def compute_random_baseline(
        self,
        y: np.ndarray,
        n_permutations: int = 100
    ) -> Tuple[float, float]:
        """
        Compute random permutation baseline.

        Args:
            y: Activity values
            n_permutations: Number of permutations

        Returns:
            Tuple of (mean correlation, std correlation) from random permutations
        """
        rng = np.random.RandomState(self.config.random_seed)
        correlations = []

        for _ in range(n_permutations):
            y_shuffled = rng.permutation(y)
            r = pearsonr(y, y_shuffled)[0]
            correlations.append(r)

        return np.mean(correlations), np.std(correlations)

    def compare_transfer_to_scratch(
        self,
        transfer_pearson: float,
        scratch_pearson: float
    ) -> Dict[str, Any]:
        """
        Compare transfer performance to training from scratch.

        Args:
            transfer_pearson: Transfer learning Pearson r
            scratch_pearson: From-scratch Pearson r

        Returns:
            Dict with comparison metrics
        """
        improvement = transfer_pearson - scratch_pearson
        relative_improvement = (improvement / abs(scratch_pearson) * 100) if scratch_pearson != 0 else 0

        return {
            'transfer_pearson': transfer_pearson,
            'scratch_pearson': scratch_pearson,
            'improvement': improvement,
            'relative_improvement_pct': relative_improvement,
            'transfer_better': transfer_pearson > scratch_pearson
        }


def quick_evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Quick evaluation of predictions.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dict with metrics
    """
    return {
        'pearson_r': pearsonr(y_true, y_pred)[0],
        'spearman_r': spearmanr(y_true, y_pred)[0],
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred)
    }
