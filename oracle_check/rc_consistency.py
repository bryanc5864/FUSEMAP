"""
Reverse Complement Consistency Check for OracleCheck

Validates that predictions are consistent for a sequence and its reverse complement.
Also implements ISM flip test to verify symmetric behavior under perturbations.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RCConsistencyResult:
    """Result of RC consistency check."""
    passed: bool
    sequence: str
    rc_sequence: str
    prediction_fwd: float
    prediction_rc: float
    delta: float
    delta_threshold: float
    message: str


@dataclass
class ISMFlipTestResult:
    """Result of ISM flip test for RC consistency."""
    passed: bool
    n_positions_tested: int
    n_symmetric: int
    n_asymmetric: int
    symmetry_rate: float
    max_asymmetry: float
    mean_asymmetry: float
    asymmetric_positions: List[int]
    message: str


def reverse_complement(sequence: str) -> str:
    """
    Get reverse complement of a DNA sequence.

    Args:
        sequence: DNA sequence

    Returns:
        Reverse complement sequence
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(sequence.upper()))


def encode_sequence(sequence: str) -> torch.Tensor:
    """
    One-hot encode a DNA sequence.

    Args:
        sequence: DNA sequence

    Returns:
        Tensor of shape [4, len(sequence)]
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq = sequence.upper()

    encoded = torch.zeros(4, len(seq))
    for i, base in enumerate(seq):
        if base in mapping:
            encoded[mapping[base], i] = 1.0
        else:
            # N or other: uniform distribution
            encoded[:, i] = 0.25

    return encoded


class RCConsistencyChecker:
    """
    Checks prediction consistency between a sequence and its reverse complement.
    """

    def __init__(
        self,
        model: nn.Module = None,
        delta_threshold: float = 0.1,
        device: str = "cuda",
    ):
        """
        Initialize RC consistency checker.

        Args:
            model: CADENCE model for predictions
            delta_threshold: Maximum allowed difference between fwd and RC predictions
            device: Device for inference
        """
        self.model = model
        self.delta_threshold = delta_threshold
        self.device = device

    def check_sequence(
        self,
        sequence: str,
        predict_fn=None,
    ) -> RCConsistencyResult:
        """
        Check RC consistency for a single sequence.

        Args:
            sequence: DNA sequence
            predict_fn: Optional prediction function (sequence -> prediction)

        Returns:
            RCConsistencyResult
        """
        rc_seq = reverse_complement(sequence)

        if predict_fn is not None:
            pred_fwd = predict_fn(sequence)
            pred_rc = predict_fn(rc_seq)
        elif self.model is not None:
            pred_fwd = self._predict(sequence)
            pred_rc = self._predict(rc_seq)
        else:
            raise ValueError("Either model or predict_fn must be provided")

        delta = abs(pred_fwd - pred_rc)
        passed = delta <= self.delta_threshold

        if passed:
            message = f"RC consistency passed (Δ={delta:.4f})"
        else:
            message = f"RC consistency failed: Δ={delta:.4f} > {self.delta_threshold}"

        return RCConsistencyResult(
            passed=passed,
            sequence=sequence,
            rc_sequence=rc_seq,
            prediction_fwd=pred_fwd,
            prediction_rc=pred_rc,
            delta=delta,
            delta_threshold=self.delta_threshold,
            message=message,
        )

    def _predict(self, sequence: str) -> float:
        """Get model prediction for a sequence."""
        if self.model is None:
            raise ValueError("Model not set")

        self.model.eval()
        with torch.no_grad():
            encoded = encode_sequence(sequence).unsqueeze(0).to(self.device)
            output = self.model(encoded)

            if isinstance(output, dict):
                # Handle MultiSpeciesCADENCE output format
                first_key = list(output.keys())[0]
                if isinstance(output[first_key], dict):
                    pred = output[first_key]['mean']
                else:
                    pred = output[first_key]
            else:
                pred = output

            return float(pred.squeeze().cpu())

    def check_batch(
        self,
        sequences: List[str],
        predict_fn=None,
    ) -> Tuple[List[RCConsistencyResult], float]:
        """
        Check RC consistency for a batch of sequences.

        Args:
            sequences: List of DNA sequences
            predict_fn: Optional prediction function

        Returns:
            Tuple of (list of results, overall pass rate)
        """
        results = []
        for seq in sequences:
            result = self.check_sequence(seq, predict_fn)
            results.append(result)

        pass_rate = sum(1 for r in results if r.passed) / len(results)
        return results, pass_rate


class ISMFlipTest:
    """
    In-Silico Mutagenesis flip test for RC consistency.

    Tests whether single-base perturbations yield symmetric deltas
    under reverse complement transformation.
    """

    def __init__(
        self,
        model: nn.Module = None,
        asymmetry_threshold: float = 0.05,
        device: str = "cuda",
    ):
        """
        Initialize ISM flip test.

        Args:
            model: CADENCE model for predictions
            asymmetry_threshold: Maximum allowed asymmetry
            device: Device for inference
        """
        self.model = model
        self.asymmetry_threshold = asymmetry_threshold
        self.device = device

    def run_test(
        self,
        sequence: str,
        positions: List[int] = None,
        predict_fn=None,
    ) -> ISMFlipTestResult:
        """
        Run ISM flip test on a sequence.

        For each position i, we compute:
        - Δ_fwd = prediction(mutant_fwd) - prediction(original)
        - Δ_rc = prediction(RC(mutant)) - prediction(RC(original))

        They should be approximately equal for RC-symmetric models.

        Args:
            sequence: DNA sequence
            positions: Positions to test (default: all)
            predict_fn: Optional prediction function

        Returns:
            ISMFlipTestResult
        """
        seq_len = len(sequence)
        rc_seq = reverse_complement(sequence)

        if positions is None:
            # Sample positions if sequence is long
            if seq_len > 100:
                positions = list(np.linspace(10, seq_len - 10, 50, dtype=int))
            else:
                positions = list(range(seq_len))

        bases = ['A', 'C', 'G', 'T']
        asymmetries = []
        asymmetric_positions = []

        # Get baseline predictions
        if predict_fn is not None:
            pred_fwd = predict_fn(sequence)
            pred_rc = predict_fn(rc_seq)
        elif self.model is not None:
            pred_fwd = self._predict(sequence)
            pred_rc = self._predict(rc_seq)
        else:
            raise ValueError("Either model or predict_fn must be provided")

        for pos in positions:
            original_base = sequence[pos].upper()
            rc_pos = seq_len - 1 - pos

            for mut_base in bases:
                if mut_base == original_base:
                    continue

                # Forward mutant
                mutant_fwd = sequence[:pos] + mut_base + sequence[pos+1:]
                if predict_fn is not None:
                    pred_mut_fwd = predict_fn(mutant_fwd)
                else:
                    pred_mut_fwd = self._predict(mutant_fwd)

                delta_fwd = pred_mut_fwd - pred_fwd

                # RC mutant (equivalent mutation in RC space)
                rc_mut_base = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}[mut_base]
                mutant_rc_seq = rc_seq[:rc_pos] + rc_mut_base + rc_seq[rc_pos+1:]

                if predict_fn is not None:
                    pred_mut_rc = predict_fn(mutant_rc_seq)
                else:
                    pred_mut_rc = self._predict(mutant_rc_seq)

                delta_rc = pred_mut_rc - pred_rc

                # Asymmetry: difference in effect sizes
                asymmetry = abs(delta_fwd - delta_rc)
                asymmetries.append(asymmetry)

                if asymmetry > self.asymmetry_threshold:
                    if pos not in asymmetric_positions:
                        asymmetric_positions.append(pos)

        n_symmetric = sum(1 for a in asymmetries if a <= self.asymmetry_threshold)
        n_asymmetric = len(asymmetries) - n_symmetric
        symmetry_rate = n_symmetric / len(asymmetries) if asymmetries else 1.0

        # Pass if symmetry rate is above 90%
        passed = symmetry_rate >= 0.9

        if passed:
            message = f"ISM flip test passed (symmetry rate: {symmetry_rate:.1%})"
        else:
            message = f"ISM flip test failed: symmetry rate {symmetry_rate:.1%} < 90%"

        return ISMFlipTestResult(
            passed=passed,
            n_positions_tested=len(positions),
            n_symmetric=n_symmetric,
            n_asymmetric=n_asymmetric,
            symmetry_rate=symmetry_rate,
            max_asymmetry=max(asymmetries) if asymmetries else 0.0,
            mean_asymmetry=np.mean(asymmetries) if asymmetries else 0.0,
            asymmetric_positions=asymmetric_positions,
            message=message,
        )

    def _predict(self, sequence: str) -> float:
        """Get model prediction for a sequence."""
        if self.model is None:
            raise ValueError("Model not set")

        self.model.eval()
        with torch.no_grad():
            encoded = encode_sequence(sequence).unsqueeze(0).to(self.device)
            output = self.model(encoded)

            if isinstance(output, dict):
                first_key = list(output.keys())[0]
                if isinstance(output[first_key], dict):
                    pred = output[first_key]['mean']
                else:
                    pred = output[first_key]
            else:
                pred = output

            return float(pred.squeeze().cpu())


class FullRCValidator:
    """
    Combined RC consistency validator.
    """

    def __init__(
        self,
        model: nn.Module = None,
        delta_threshold: float = 0.1,
        asymmetry_threshold: float = 0.05,
        device: str = "cuda",
    ):
        """
        Initialize full RC validator.

        Args:
            model: CADENCE model
            delta_threshold: Threshold for direct RC comparison
            asymmetry_threshold: Threshold for ISM flip test
            device: Device for inference
        """
        self.rc_checker = RCConsistencyChecker(
            model=model,
            delta_threshold=delta_threshold,
            device=device,
        )
        self.ism_test = ISMFlipTest(
            model=model,
            asymmetry_threshold=asymmetry_threshold,
            device=device,
        )

    def validate(
        self,
        sequence: str,
        run_ism: bool = True,
        ism_positions: List[int] = None,
        predict_fn=None,
    ) -> Dict:
        """
        Run full RC validation.

        Args:
            sequence: DNA sequence
            run_ism: Whether to run ISM flip test
            ism_positions: Positions for ISM test
            predict_fn: Optional prediction function

        Returns:
            Dictionary with RC consistency and ISM results
        """
        results = {
            'passed': True,
            'rc_result': None,
            'ism_result': None,
        }

        # RC consistency check
        results['rc_result'] = self.rc_checker.check_sequence(sequence, predict_fn)
        if not results['rc_result'].passed:
            results['passed'] = False

        # ISM flip test
        if run_ism:
            results['ism_result'] = self.ism_test.run_test(sequence, ism_positions, predict_fn)
            if not results['ism_result'].passed:
                results['passed'] = False

        return results
