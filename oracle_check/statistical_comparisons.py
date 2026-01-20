"""
Statistical Comparison Framework for OracleCheck

Implements statistical comparisons between:
- Unconstrained vs Physics-Constrained optimization
- PhysicsVAE vs Optimization approaches
- Designed vs Natural high-activity sequences
- Cross-species transfer evaluation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
import json

from .config import OracleCheckConfig, Verdict
from .validation_runner import ValidationReport
from .two_sample_tests import BatchComparator, MMDTest, KSTest, KmerJSDivergence


@dataclass
class PairedComparisonResult:
    """Result of a paired comparison between two methods."""
    method_a: str
    method_b: str
    metric: str
    value_a: float
    value_b: float
    difference: float
    pvalue: float
    significant: bool
    effect_size: float
    winner: str
    message: str


@dataclass
class MethodComparisonSuite:
    """Complete comparison suite for two methods."""
    method_a: str
    method_b: str
    n_sequences_a: int
    n_sequences_b: int
    comparisons: List[PairedComparisonResult]
    summary: Dict
    overall_winner: str


class StatisticalComparator:
    """
    Performs statistical comparisons between sequence generation methods.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize comparator.

        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha
        self.batch_comparator = BatchComparator()

    def paired_ttest(
        self,
        values_a: np.ndarray,
        values_b: np.ndarray,
        metric_name: str,
        method_a: str = "A",
        method_b: str = "B",
    ) -> PairedComparisonResult:
        """
        Perform paired t-test comparison.

        Args:
            values_a: Values from method A
            values_b: Values from method B
            metric_name: Name of the metric
            method_a: Name of method A
            method_b: Name of method B

        Returns:
            PairedComparisonResult
        """
        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))
        diff = mean_a - mean_b

        # Paired t-test if same length, otherwise independent
        if len(values_a) == len(values_b):
            t_stat, pvalue = stats.ttest_rel(values_a, values_b)
        else:
            t_stat, pvalue = stats.ttest_ind(values_a, values_b)

        # Cohen's d effect size
        pooled_std = np.sqrt((np.std(values_a)**2 + np.std(values_b)**2) / 2)
        effect_size = diff / pooled_std if pooled_std > 0 else 0.0

        significant = pvalue < self.alpha
        winner = method_a if diff > 0 else method_b if diff < 0 else "tie"

        if significant:
            message = f"{metric_name}: {winner} significantly better (p={pvalue:.4f}, d={effect_size:.2f})"
        else:
            message = f"{metric_name}: No significant difference (p={pvalue:.4f})"

        return PairedComparisonResult(
            method_a=method_a,
            method_b=method_b,
            metric=metric_name,
            value_a=mean_a,
            value_b=mean_b,
            difference=diff,
            pvalue=float(pvalue),
            significant=significant,
            effect_size=float(effect_size),
            winner=winner,
            message=message,
        )

    def chi_squared_test(
        self,
        counts_a: Dict[str, int],
        counts_b: Dict[str, int],
        metric_name: str,
        method_a: str = "A",
        method_b: str = "B",
    ) -> PairedComparisonResult:
        """
        Perform chi-squared test for categorical data.

        Args:
            counts_a: Category counts for method A
            counts_b: Category counts for method B
            metric_name: Name of the metric
            method_a: Name of method A
            method_b: Name of method B

        Returns:
            PairedComparisonResult
        """
        categories = set(counts_a.keys()) | set(counts_b.keys())
        observed = np.array([[counts_a.get(c, 0), counts_b.get(c, 0)] for c in categories])

        chi2, pvalue, dof, expected = stats.chi2_contingency(observed)

        # Cramér's V effect size
        n = observed.sum()
        min_dim = min(observed.shape) - 1
        effect_size = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0

        significant = pvalue < self.alpha

        # Determine winner (higher green rate is better)
        green_rate_a = counts_a.get("GREEN", 0) / sum(counts_a.values()) if counts_a else 0
        green_rate_b = counts_b.get("GREEN", 0) / sum(counts_b.values()) if counts_b else 0

        if significant:
            winner = method_a if green_rate_a > green_rate_b else method_b
            message = f"{metric_name}: Distribution differs significantly (p={pvalue:.4f}, V={effect_size:.2f})"
        else:
            winner = "tie"
            message = f"{metric_name}: No significant difference in distribution (p={pvalue:.4f})"

        return PairedComparisonResult(
            method_a=method_a,
            method_b=method_b,
            metric=metric_name,
            value_a=green_rate_a,
            value_b=green_rate_b,
            difference=green_rate_a - green_rate_b,
            pvalue=float(pvalue),
            significant=significant,
            effect_size=float(effect_size),
            winner=winner,
            message=message,
        )

    def compare_reports(
        self,
        report_a: ValidationReport,
        report_b: ValidationReport,
    ) -> MethodComparisonSuite:
        """
        Compare two validation reports.

        Args:
            report_a: First report
            report_b: Second report

        Returns:
            MethodComparisonSuite with all comparisons
        """
        method_a = report_a.generation_method
        method_b = report_b.generation_method

        comparisons = []

        # Compare verdict distributions (chi-squared)
        verdict_counts_a = {"GREEN": report_a.n_green, "YELLOW": report_a.n_yellow, "RED": report_a.n_red}
        verdict_counts_b = {"GREEN": report_b.n_green, "YELLOW": report_b.n_yellow, "RED": report_b.n_red}

        verdict_comparison = self.chi_squared_test(
            verdict_counts_a, verdict_counts_b,
            "Verdict Distribution", method_a, method_b
        )
        comparisons.append(verdict_comparison)

        # Compare activity values
        if report_a.sequence_results and report_b.sequence_results:
            activities_a = np.array([r.prediction_mean for r in report_a.sequence_results])
            activities_b = np.array([r.prediction_mean for r in report_b.sequence_results])

            activity_comparison = self.paired_ttest(
                activities_a, activities_b,
                "Activity", method_a, method_b
            )
            comparisons.append(activity_comparison)

            # Compare top 10% activities
            top_a = np.percentile(activities_a, 90)
            top_b = np.percentile(activities_b, 90)

            # Bootstrap for top 10% comparison
            n_bootstrap = 1000
            top_diffs = []
            for _ in range(n_bootstrap):
                sample_a = np.random.choice(activities_a, len(activities_a), replace=True)
                sample_b = np.random.choice(activities_b, len(activities_b), replace=True)
                top_diffs.append(np.percentile(sample_a, 90) - np.percentile(sample_b, 90))

            pvalue = np.mean([d <= 0 if top_a > top_b else d >= 0 for d in top_diffs])
            pvalue = min(pvalue * 2, 1.0)  # Two-tailed

            comparisons.append(PairedComparisonResult(
                method_a=method_a,
                method_b=method_b,
                metric="Top 10% Activity",
                value_a=float(top_a),
                value_b=float(top_b),
                difference=float(top_a - top_b),
                pvalue=float(pvalue),
                significant=pvalue < self.alpha,
                effect_size=float((top_a - top_b) / np.std(top_diffs)) if np.std(top_diffs) > 0 else 0.0,
                winner=method_a if top_a > top_b else method_b,
                message=f"Top 10% Activity: {method_a if top_a > top_b else method_b} better (p={pvalue:.4f})",
            ))

        # Compare pass rates using proportion test
        pass_metrics = [
            ("Physics Pass Rate", report_a.physics_pass_rate, report_b.physics_pass_rate),
            ("Composition Pass Rate", report_a.composition_pass_rate, report_b.composition_pass_rate),
            ("RC Consistency Rate", report_a.rc_consistency_rate, report_b.rc_consistency_rate),
            ("Motif Pass Rate", report_a.motif_pass_rate, report_b.motif_pass_rate),
        ]

        for metric_name, rate_a, rate_b in pass_metrics:
            n_a = report_a.n_sequences
            n_b = report_b.n_sequences

            # Proportion test
            successes = [int(rate_a * n_a), int(rate_b * n_b)]
            totals = [n_a, n_b]

            try:
                z_stat, pvalue = stats.proportions_ztest(successes, totals)
            except:
                pvalue = 1.0
                z_stat = 0.0

            diff = rate_a - rate_b
            significant = pvalue < self.alpha
            winner = method_a if diff > 0 else method_b if diff < 0 else "tie"

            comparisons.append(PairedComparisonResult(
                method_a=method_a,
                method_b=method_b,
                metric=metric_name,
                value_a=rate_a,
                value_b=rate_b,
                difference=diff,
                pvalue=float(pvalue),
                significant=significant,
                effect_size=float(z_stat),
                winner=winner if significant else "tie",
                message=f"{metric_name}: {winner if significant else 'No difference'} (p={pvalue:.4f})",
            ))

        # Determine overall winner
        wins_a = sum(1 for c in comparisons if c.significant and c.winner == method_a)
        wins_b = sum(1 for c in comparisons if c.significant and c.winner == method_b)

        if wins_a > wins_b:
            overall_winner = method_a
        elif wins_b > wins_a:
            overall_winner = method_b
        else:
            overall_winner = "tie"

        summary = {
            "wins": {method_a: wins_a, method_b: wins_b},
            "significant_comparisons": sum(1 for c in comparisons if c.significant),
            "total_comparisons": len(comparisons),
            "green_rate_diff": report_a.green_rate - report_b.green_rate,
            "activity_diff": report_a.mean_activity - report_b.mean_activity,
        }

        return MethodComparisonSuite(
            method_a=method_a,
            method_b=method_b,
            n_sequences_a=report_a.n_sequences,
            n_sequences_b=report_b.n_sequences,
            comparisons=comparisons,
            summary=summary,
            overall_winner=overall_winner,
        )

    def compare_to_natural(
        self,
        designed_report: ValidationReport,
        natural_sequences: List[str],
        natural_activities: np.ndarray,
        designed_physics: Optional[np.ndarray] = None,
        natural_physics: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compare designed sequences to natural high-performers.

        Args:
            designed_report: Validation report for designed sequences
            natural_sequences: Natural high-activity sequences
            natural_activities: Activity values for natural sequences
            designed_physics: Physics features for designed sequences
            natural_physics: Physics features for natural sequences

        Returns:
            Comparison results
        """
        results = {
            "method": designed_report.generation_method,
            "n_designed": designed_report.n_sequences,
            "n_natural": len(natural_sequences),
        }

        # Activity comparison
        designed_activities = np.array([r.prediction_mean for r in designed_report.sequence_results])

        ks_stat, ks_pvalue = stats.ks_2samp(designed_activities, natural_activities)
        results["activity_ks"] = {
            "statistic": float(ks_stat),
            "pvalue": float(ks_pvalue),
            "significant": ks_pvalue < self.alpha,
        }

        t_stat, t_pvalue = stats.ttest_ind(designed_activities, natural_activities)
        results["activity_ttest"] = {
            "statistic": float(t_stat),
            "pvalue": float(t_pvalue),
            "designed_mean": float(np.mean(designed_activities)),
            "natural_mean": float(np.mean(natural_activities)),
            "difference": float(np.mean(designed_activities) - np.mean(natural_activities)),
        }

        # Physics comparison via two-sample tests
        if designed_physics is not None and natural_physics is not None:
            batch_results = self.batch_comparator.compare_all(designed_physics, natural_physics)
            results["physics_comparison"] = batch_results.to_dict()

        # K-mer comparison
        kmer_js = KmerJSDivergence()

        # Extract designed sequences
        designed_seqs = [r.sequence for r in designed_report.sequence_results]

        for k in [4, 5, 6]:
            js_result = kmer_js.compute(designed_seqs, natural_sequences, k=k)
            results[f"kmer_{k}_js"] = {
                "divergence": float(js_result.js_divergence),
                "pvalue": float(js_result.pvalue),
                "passed": js_result.passed,
            }

        return results


def generate_comparison_report(
    reports: Dict[str, ValidationReport],
    output_path: Optional[Path] = None,
) -> Dict:
    """
    Generate a comprehensive comparison report across all methods.

    Args:
        reports: Dict mapping method name to ValidationReport
        output_path: Optional path to save report

    Returns:
        Comprehensive comparison results
    """
    comparator = StatisticalComparator()

    methods = list(reports.keys())
    n_methods = len(methods)

    results = {
        "methods": methods,
        "n_methods": n_methods,
        "summaries": {},
        "pairwise_comparisons": [],
        "rankings": {},
    }

    # Summary for each method
    for method, report in reports.items():
        results["summaries"][method] = report.to_dict()

    # Pairwise comparisons
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            method_a = methods[i]
            method_b = methods[j]

            comparison = comparator.compare_reports(reports[method_a], reports[method_b])

            results["pairwise_comparisons"].append({
                "method_a": method_a,
                "method_b": method_b,
                "overall_winner": comparison.overall_winner,
                "summary": comparison.summary,
                "comparisons": [
                    {
                        "metric": c.metric,
                        "value_a": c.value_a,
                        "value_b": c.value_b,
                        "pvalue": c.pvalue,
                        "significant": c.significant,
                        "winner": c.winner,
                    }
                    for c in comparison.comparisons
                ],
            })

    # Overall rankings
    # Rank by green rate
    green_rates = {m: r.green_rate for m, r in reports.items()}
    results["rankings"]["green_rate"] = sorted(green_rates.items(), key=lambda x: -x[1])

    # Rank by mean activity
    mean_activities = {m: r.mean_activity for m, r in reports.items()}
    results["rankings"]["mean_activity"] = sorted(mean_activities.items(), key=lambda x: -x[1])

    # Rank by top 10% activity
    top_activities = {m: r.top_10_activity for m, r in reports.items()}
    results["rankings"]["top_10_activity"] = sorted(top_activities.items(), key=lambda x: -x[1])

    # Rank by physics pass rate
    physics_rates = {m: r.physics_pass_rate for m, r in reports.items()}
    results["rankings"]["physics_pass_rate"] = sorted(physics_rates.items(), key=lambda x: -x[1])

    # Save if path provided
    if output_path is not None:
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Comparison report saved to {output_path}")

    return results
