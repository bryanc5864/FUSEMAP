"""
Two-Sample Distribution Tests for OracleCheck

Implements batch-level statistical tests to compare designed sequences
against natural reference distributions.

Tests include:
- Maximum Mean Discrepancy (MMD)
- Energy Distance
- Kolmogorov-Smirnov (KS) tests
- Jensen-Shannon Divergence for k-mer spectra
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter


@dataclass
class TwoSampleTestResult:
    """Result of a two-sample statistical test."""
    test_name: str
    statistic: float
    pvalue: float
    passed: bool
    threshold: float
    message: str


@dataclass
class BatchComparisonResult:
    """Result of batch-level comparison between designed and natural sequences."""
    passed: bool
    n_designed: int
    n_reference: int

    # Individual test results
    mmd_result: Optional[TwoSampleTestResult] = None
    energy_result: Optional[TwoSampleTestResult] = None
    ks_results: Dict[str, TwoSampleTestResult] = None
    kmer_js_result: Optional[TwoSampleTestResult] = None

    # Summary
    n_tests_passed: int = 0
    n_tests_total: int = 0
    flags: List[str] = None
    message: str = ""

    def __post_init__(self):
        if self.flags is None:
            self.flags = []
        if self.ks_results is None:
            self.ks_results = {}


class MMDTest:
    """
    Maximum Mean Discrepancy test with RBF kernel.

    MMD measures the distance between two distributions in a reproducing
    kernel Hilbert space (RKHS).
    """

    def __init__(self, kernel_bandwidth: float = None):
        """
        Initialize MMD test.

        Args:
            kernel_bandwidth: RBF kernel bandwidth (None for median heuristic)
        """
        self.kernel_bandwidth = kernel_bandwidth

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute RBF kernel matrix."""
        sq_dist = cdist(X, Y, 'sqeuclidean')
        return np.exp(-sq_dist / (2 * bandwidth ** 2))

    def _median_heuristic(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute kernel bandwidth using median heuristic."""
        XY = np.vstack([X, Y])
        pairwise_dists = cdist(XY, XY, 'euclidean')
        # Use median of non-zero distances
        non_zero = pairwise_dists[pairwise_dists > 0]
        if len(non_zero) > 0:
            return np.median(non_zero)
        return 1.0

    def compute(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_permutations: int = 1000,
    ) -> TwoSampleTestResult:
        """
        Compute MMD statistic with permutation test for p-value.

        Args:
            X: First sample [n_x, d]
            Y: Second sample [n_y, d]
            n_permutations: Number of permutations for p-value

        Returns:
            TwoSampleTestResult
        """
        n_x, n_y = len(X), len(Y)

        # Bandwidth
        bandwidth = self.kernel_bandwidth
        if bandwidth is None:
            bandwidth = self._median_heuristic(X, Y)

        # Compute kernel matrices
        K_xx = self._rbf_kernel(X, X, bandwidth)
        K_yy = self._rbf_kernel(Y, Y, bandwidth)
        K_xy = self._rbf_kernel(X, Y, bandwidth)

        # Unbiased MMD^2 estimator
        mmd2 = (
            (K_xx.sum() - np.trace(K_xx)) / (n_x * (n_x - 1))
            + (K_yy.sum() - np.trace(K_yy)) / (n_y * (n_y - 1))
            - 2 * K_xy.mean()
        )

        mmd = np.sqrt(max(mmd2, 0))

        # Permutation test for p-value
        combined = np.vstack([X, Y])
        null_mmds = []

        for _ in range(n_permutations):
            perm = np.random.permutation(n_x + n_y)
            X_perm = combined[perm[:n_x]]
            Y_perm = combined[perm[n_x:]]

            K_xx_p = self._rbf_kernel(X_perm, X_perm, bandwidth)
            K_yy_p = self._rbf_kernel(Y_perm, Y_perm, bandwidth)
            K_xy_p = self._rbf_kernel(X_perm, Y_perm, bandwidth)

            mmd2_p = (
                (K_xx_p.sum() - np.trace(K_xx_p)) / (n_x * (n_x - 1))
                + (K_yy_p.sum() - np.trace(K_yy_p)) / (n_y * (n_y - 1))
                - 2 * K_xy_p.mean()
            )
            null_mmds.append(np.sqrt(max(mmd2_p, 0)))

        pvalue = (np.sum(np.array(null_mmds) >= mmd) + 1) / (n_permutations + 1)

        # Threshold at p=0.05
        threshold = 0.05
        passed = pvalue > threshold

        return TwoSampleTestResult(
            test_name="MMD",
            statistic=mmd,
            pvalue=pvalue,
            passed=passed,
            threshold=threshold,
            message=f"MMD={mmd:.4f}, p={pvalue:.4f}" + (" (PASS)" if passed else " (FAIL)")
        )


class EnergyDistanceTest:
    """
    Energy Distance test.

    Energy distance is a metric that measures the distance between
    probability distributions based on pairwise distances.
    """

    def compute(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_permutations: int = 1000,
    ) -> TwoSampleTestResult:
        """
        Compute energy distance with permutation test.

        Args:
            X: First sample [n_x, d]
            Y: Second sample [n_y, d]
            n_permutations: Number of permutations for p-value

        Returns:
            TwoSampleTestResult
        """
        n_x, n_y = len(X), len(Y)

        # Compute energy distance
        # E = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
        d_xy = cdist(X, Y, 'euclidean').mean()
        d_xx = cdist(X, X, 'euclidean').mean()
        d_yy = cdist(Y, Y, 'euclidean').mean()

        energy = 2 * d_xy - d_xx - d_yy

        # Permutation test
        combined = np.vstack([X, Y])
        null_energies = []

        for _ in range(n_permutations):
            perm = np.random.permutation(n_x + n_y)
            X_perm = combined[perm[:n_x]]
            Y_perm = combined[perm[n_x:]]

            d_xy_p = cdist(X_perm, Y_perm, 'euclidean').mean()
            d_xx_p = cdist(X_perm, X_perm, 'euclidean').mean()
            d_yy_p = cdist(Y_perm, Y_perm, 'euclidean').mean()

            null_energies.append(2 * d_xy_p - d_xx_p - d_yy_p)

        pvalue = (np.sum(np.array(null_energies) >= energy) + 1) / (n_permutations + 1)

        threshold = 0.05
        passed = pvalue > threshold

        return TwoSampleTestResult(
            test_name="Energy Distance",
            statistic=energy,
            pvalue=pvalue,
            passed=passed,
            threshold=threshold,
            message=f"Energy={energy:.4f}, p={pvalue:.4f}" + (" (PASS)" if passed else " (FAIL)")
        )


class KSTest:
    """
    Kolmogorov-Smirnov test for univariate distributions.
    """

    def compute(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_name: str = "feature",
    ) -> TwoSampleTestResult:
        """
        Compute two-sample KS test.

        Args:
            x: First sample [n_x]
            y: Second sample [n_y]
            feature_name: Name of the feature being tested

        Returns:
            TwoSampleTestResult
        """
        statistic, pvalue = stats.ks_2samp(x, y)

        threshold = 0.05
        passed = pvalue > threshold

        return TwoSampleTestResult(
            test_name=f"KS ({feature_name})",
            statistic=statistic,
            pvalue=pvalue,
            passed=passed,
            threshold=threshold,
            message=f"KS({feature_name})={statistic:.4f}, p={pvalue:.4f}" + (" (PASS)" if passed else " (FAIL)")
        )


class KmerJSDivergence:
    """
    Jensen-Shannon divergence on k-mer spectra.
    """

    def __init__(self, k_values: List[int] = None):
        """
        Initialize k-mer JS divergence test.

        Args:
            k_values: List of k values to use (default: [4, 5, 6])
        """
        self.k_values = k_values or [4, 5, 6]

    def _get_kmer_spectrum(self, sequences: List[str], k: int) -> np.ndarray:
        """Compute normalized k-mer frequency spectrum."""
        all_kmers = Counter()

        for seq in sequences:
            seq = seq.upper()
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if 'N' not in kmer:
                    all_kmers[kmer] += 1

        # Get all possible k-mers
        bases = ['A', 'C', 'G', 'T']
        all_possible = []

        def generate_kmers(prefix, remaining):
            if remaining == 0:
                all_possible.append(prefix)
                return
            for b in bases:
                generate_kmers(prefix + b, remaining - 1)

        generate_kmers('', k)

        # Create frequency vector
        total = sum(all_kmers.values()) + 1e-10
        spectrum = np.array([all_kmers.get(kmer, 0) / total for kmer in all_possible])

        return spectrum

    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence."""
        # Add small epsilon for numerical stability
        p = p + 1e-10
        q = q + 1e-10
        p = p / p.sum()
        q = q / q.sum()

        m = 0.5 * (p + q)

        # KL divergences
        kl_pm = np.sum(p * np.log2(p / m))
        kl_qm = np.sum(q * np.log2(q / m))

        return 0.5 * (kl_pm + kl_qm)

    def compute(
        self,
        designed_seqs: List[str],
        reference_seqs: List[str],
        threshold: float = 0.1,
    ) -> TwoSampleTestResult:
        """
        Compute JS divergence on k-mer spectra.

        Args:
            designed_seqs: List of designed sequences
            reference_seqs: List of reference sequences
            threshold: Maximum allowable JS divergence

        Returns:
            TwoSampleTestResult
        """
        js_values = []

        for k in self.k_values:
            designed_spectrum = self._get_kmer_spectrum(designed_seqs, k)
            reference_spectrum = self._get_kmer_spectrum(reference_seqs, k)

            js = self._js_divergence(designed_spectrum, reference_spectrum)
            js_values.append(js)

        # Use mean JS across all k values
        mean_js = np.mean(js_values)

        passed = mean_js < threshold

        return TwoSampleTestResult(
            test_name="k-mer JS Divergence",
            statistic=mean_js,
            pvalue=1.0 - mean_js,  # Not a true p-value
            passed=passed,
            threshold=threshold,
            message=f"JS({self.k_values})={mean_js:.4f}" + (" (PASS)" if passed else " (FAIL)")
        )


class BatchComparator:
    """
    Compares batches of designed sequences against natural references.
    """

    def __init__(
        self,
        mmd_bandwidth: float = None,
        kmer_k_values: List[int] = None,
        n_permutations: int = 500,
    ):
        """
        Initialize batch comparator.

        Args:
            mmd_bandwidth: MMD kernel bandwidth (None for median heuristic)
            kmer_k_values: k values for k-mer spectrum (default: [4, 5, 6])
            n_permutations: Number of permutations for MMD/energy tests
        """
        self.mmd_test = MMDTest(kernel_bandwidth=mmd_bandwidth)
        self.energy_test = EnergyDistanceTest()
        self.ks_test = KSTest()
        self.kmer_test = KmerJSDivergence(k_values=kmer_k_values)
        self.n_permutations = n_permutations

    def compare_physics(
        self,
        designed_features: np.ndarray,
        reference_features: np.ndarray,
        feature_names: List[str] = None,
    ) -> BatchComparisonResult:
        """
        Compare physics features between designed and reference sequences.

        Args:
            designed_features: Physics features for designed sequences [n_designed, n_features]
            reference_features: Physics features for reference sequences [n_ref, n_features]
            feature_names: Names of physics features

        Returns:
            BatchComparisonResult
        """
        n_designed = len(designed_features)
        n_reference = len(reference_features)

        result = BatchComparisonResult(
            passed=True,
            n_designed=n_designed,
            n_reference=n_reference,
            flags=[],
        )

        # MMD test on full feature vectors
        result.mmd_result = self.mmd_test.compute(
            designed_features,
            reference_features,
            n_permutations=self.n_permutations,
        )
        if not result.mmd_result.passed:
            result.passed = False
            result.flags.append(result.mmd_result.message)

        # Energy distance test
        result.energy_result = self.energy_test.compute(
            designed_features,
            reference_features,
            n_permutations=self.n_permutations,
        )
        if not result.energy_result.passed:
            result.passed = False
            result.flags.append(result.energy_result.message)

        # KS tests per feature
        result.ks_results = {}
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(designed_features.shape[1])]

        # Only test a subset of features to avoid multiple testing issues
        n_features_to_test = min(20, len(feature_names))
        indices = np.linspace(0, len(feature_names) - 1, n_features_to_test, dtype=int)

        for i in indices:
            name = feature_names[i]
            ks_result = self.ks_test.compute(
                designed_features[:, i],
                reference_features[:, i],
                feature_name=name,
            )
            result.ks_results[name] = ks_result

        # Count tests
        result.n_tests_total = 2 + len(result.ks_results)  # MMD + Energy + KS tests
        result.n_tests_passed = (
            int(result.mmd_result.passed)
            + int(result.energy_result.passed)
            + sum(1 for r in result.ks_results.values() if r.passed)
        )

        # Summary message
        if result.passed:
            result.message = f"Physics comparison passed ({result.n_tests_passed}/{result.n_tests_total} tests)"
        else:
            result.message = f"Physics comparison failed: {'; '.join(result.flags)}"

        return result

    def compare_sequences(
        self,
        designed_seqs: List[str],
        reference_seqs: List[str],
    ) -> TwoSampleTestResult:
        """
        Compare sequences using k-mer JS divergence.

        Args:
            designed_seqs: List of designed sequences
            reference_seqs: List of reference sequences

        Returns:
            TwoSampleTestResult for k-mer comparison
        """
        return self.kmer_test.compute(designed_seqs, reference_seqs)

    def full_comparison(
        self,
        designed_seqs: List[str],
        designed_features: np.ndarray,
        reference_seqs: List[str],
        reference_features: np.ndarray,
        feature_names: List[str] = None,
    ) -> BatchComparisonResult:
        """
        Run full comparison between designed and reference batches.

        Args:
            designed_seqs: Designed sequences
            designed_features: Physics features for designed sequences
            reference_seqs: Reference sequences
            reference_features: Physics features for reference sequences
            feature_names: Names of physics features

        Returns:
            BatchComparisonResult
        """
        # Physics comparison
        result = self.compare_physics(
            designed_features,
            reference_features,
            feature_names,
        )

        # k-mer comparison
        result.kmer_js_result = self.compare_sequences(designed_seqs, reference_seqs)
        if not result.kmer_js_result.passed:
            result.passed = False
            result.flags.append(result.kmer_js_result.message)

        # Update counts
        result.n_tests_total += 1
        result.n_tests_passed += int(result.kmer_js_result.passed)

        # Update message
        if result.passed:
            result.message = f"Full comparison passed ({result.n_tests_passed}/{result.n_tests_total} tests)"
        else:
            result.message = f"Full comparison failed: {'; '.join(result.flags)}"

        return result
