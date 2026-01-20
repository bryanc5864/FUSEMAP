#!/usr/bin/env python3
"""
Therapeutic Enhancer Design - Method Comparison.

Compares 4 sequence optimization methods for cell-type-specific enhancers:
1. PINCSD - Physics-Informed Naturality-Constrained Sequence Design (gradient-guided)
2. EMOO - Evolutionary Multi-Objective Optimization (NSGA-II)
3. ISM - In-Silico Mutagenesis baseline (greedy single mutations, optimizes specificity)
4. ISM-Target - ISM baseline optimizing target activity ONLY (no specificity penalty)

Objective: Design sequences with:
- High activity in target cell type (e.g., HepG2 liver)
- Low activity in off-target cell types (e.g., K562 blood, WTC11 pluripotent)

Specificity Score: S(seq) = Activity_target - λ₁·Activity_off1 - λ₂·Activity_off2
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import sys
from dataclasses import dataclass, field
from tqdm import tqdm
import random
from datetime import datetime
from collections import defaultdict

# Add paths
FUSEMAP_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FUSEMAP_ROOT))
sys.path.insert(0, str(FUSEMAP_ROOT / 'applications/utils'))

# Core imports
try:
    from applications.utils.multicell_ensemble import (
        MultiCellEnsemble, MultiCellPrediction,
        find_cadence_checkpoints
    )
except ImportError:
    from utils.multicell_ensemble import (
        MultiCellEnsemble, MultiCellPrediction,
        find_cadence_checkpoints
    )

# Physics imports
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# PhysicsVAE imports
try:
    from physics.PhysicsVAE import PhysicsVAE
    HAS_PHYSICS_VAE = True
except ImportError:
    HAS_PHYSICS_VAE = False

# PhysInformer imports
try:
    from physics.PhysInformer import PhysInformer
    HAS_PHYSINFORMER = True
except ImportError:
    HAS_PHYSINFORMER = False

# OracleCheck imports
try:
    from oracle_check.oraclecheck import OracleCheck
    HAS_ORACLE_CHECK = True
except ImportError:
    HAS_ORACLE_CHECK = False


# =============================================================================
# CONSTANTS
# =============================================================================

NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
IDX_TO_NUC = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

# Physics envelope bounds (from natural enhancers - approximate ranges)
DEFAULT_PHYSICS_ENVELOPE = {
    'gc_content': (0.35, 0.65),
    'max_homopolymer': (1, 6),
    'complexity': (0.7, 1.0),
}


@dataclass
class OptimizationResult:
    """Result from one optimization method."""
    method: str
    cell_type: str
    sequences: List[str]
    target_activities: List[float]
    background_activities: Dict[str, List[float]]
    specificities: List[float]
    uncertainties: List[float]
    oracle_verdicts: List[str]
    optimization_history: List[Dict] = field(default_factory=list)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def seq_to_onehot(seq: str) -> np.ndarray:
    """Convert sequence string to one-hot encoding [L, 4]."""
    onehot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nuc in enumerate(seq):
        if nuc in NUC_TO_IDX:
            onehot[i, NUC_TO_IDX[nuc]] = 1.0
    return onehot


def onehot_to_seq(onehot: np.ndarray) -> str:
    """Convert one-hot encoding to sequence string."""
    indices = np.argmax(onehot, axis=-1)
    return ''.join(IDX_TO_NUC[i] for i in indices)


def sequence_identity(seq1: str, seq2: str) -> float:
    """Compute sequence identity between two sequences."""
    if len(seq1) != len(seq2):
        return 0.0
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)


def softmax(x: List[float], temperature: float = 1.0) -> np.ndarray:
    """Compute softmax with temperature."""
    x = np.array(x) / temperature
    x = x - np.max(x)  # Stability
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-10)


def simple_oracle_check(sequence: str) -> str:
    """Simple OracleCheck for GC content and homopolymers."""
    seq = sequence.upper()
    if len(seq) == 0:
        return 'RED'

    # GC content
    gc = (seq.count('G') + seq.count('C')) / len(seq)

    # Homopolymer runs
    max_homo = 1
    current = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            current += 1
            max_homo = max(max_homo, current)
        else:
            current = 1

    # Complexity (simple k-mer diversity)
    kmers = set()
    for i in range(len(seq) - 3):
        kmers.add(seq[i:i+4])
    complexity = len(kmers) / max(len(seq) - 3, 1)

    flags = []
    if gc < 0.35 or gc > 0.65:
        flags.append('gc')
    if max_homo > 8:
        flags.append('homo')
    if complexity < 0.5:
        flags.append('complexity')

    if len(flags) == 0:
        return 'GREEN'
    elif len(flags) == 1 and max_homo <= 10:
        return 'YELLOW'
    else:
        return 'RED'


# =============================================================================
# MAIN COMPARISON CLASS
# =============================================================================

class TherapeuticMethodComparison:
    """Compare 5 different therapeutic enhancer optimization methods."""

    VAE_CHECKPOINTS = {
        'K562': FUSEMAP_ROOT / 'physics/PhysicsVAE/runs/K562_20260113_051653/best_model.pt',
        'HepG2': FUSEMAP_ROOT / 'physics/PhysicsVAE/runs/HepG2_20260113_052418/best_model.pt',
        'WTC11': FUSEMAP_ROOT / 'physics/PhysicsVAE/runs/WTC11_20260113_052743/best_model.pt',
    }

    PHYSINFORMER_CHECKPOINT = FUSEMAP_ROOT / 'physics/PhysInformer/checkpoints/best_model.pt'

    # Correct data paths - lentiMPRA data with sequences
    DATA_PATHS = {
        'K562': FUSEMAP_ROOT / 'data/lentiMPRA_data/K562/fold_splits_with_seq',
        'HepG2': FUSEMAP_ROOT / 'data/lentiMPRA_data/HepG2/fold_splits_with_seq',
        'WTC11': FUSEMAP_ROOT / 'data/lentiMPRA_data/WTC11/fold_splits_with_seq',
    }

    def __init__(
        self,
        device: str = 'cuda',
        lambda1: float = 0.5,
        lambda2: float = 0.5,
    ):
        """
        Initialize comparison framework.

        Args:
            device: CUDA or CPU
            lambda1: Penalty weight for first background cell
            lambda2: Penalty weight for second background cell
        """
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.all_cell_types = ['K562', 'HepG2', 'WTC11']

        # Load multi-cell ensemble
        print("Loading multi-cell CADENCE ensemble...")
        checkpoints = find_cadence_checkpoints()
        self.ensemble = MultiCellEnsemble(
            cell_types=self.all_cell_types,
            checkpoints=checkpoints,
            device=device
        )

        # VAE models (load on demand)
        self.vae_models = {}

        # PhysInformer (load on demand)
        self.physinformer = None

        # Natural sequences cache
        self.natural_sequences = {}

        # Physics reference (for EMOO GMM)
        self.physics_reference = None

    def _load_vae(self, cell_type: str):
        """Load PhysicsVAE for a cell type."""
        if cell_type in self.vae_models:
            return self.vae_models[cell_type]

        if not HAS_PHYSICS_VAE or not HAS_TORCH:
            return None

        vae_path = self.VAE_CHECKPOINTS.get(cell_type)
        if not vae_path or not vae_path.exists():
            print(f"  VAE checkpoint not found for {cell_type}")
            return None

        try:
            checkpoint = torch.load(vae_path, map_location=self.device, weights_only=False)
            config = checkpoint.get('config', {})

            # Infer physics features from state dict
            n_physics = None
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            for key, value in state_dict.items():
                if 'physics_encoder.encoder.0.weight' in key:
                    n_physics = value.shape[1]
                    break
            if n_physics is None:
                n_physics = 539 if cell_type == 'HepG2' else 515

            model = PhysicsVAE(
                seq_length=config.get('seq_length', 230),
                n_physics_features=n_physics,
                latent_dim=config.get('latent_dim', 128),
                physics_cond_dim=config.get('physics_cond_dim', 64),
            ).to(self.device)

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            self.vae_models[cell_type] = (model, n_physics)
            print(f"  Loaded VAE for {cell_type} ({n_physics} physics features)")
            return model, n_physics
        except Exception as e:
            print(f"  Error loading VAE for {cell_type}: {e}")
            return None

    def _load_physinformer(self):
        """Load PhysInformer model."""
        if self.physinformer is not None:
            return self.physinformer

        if not HAS_PHYSINFORMER:
            print("  PhysInformer not available")
            return None

        try:
            # Try to load PhysInformer
            if self.PHYSINFORMER_CHECKPOINT.exists():
                self.physinformer = PhysInformer.load(self.PHYSINFORMER_CHECKPOINT)
                self.physinformer.to(self.device)
                self.physinformer.eval()
                print("  Loaded PhysInformer")
            else:
                print("  PhysInformer checkpoint not found")
                return None
        except Exception as e:
            print(f"  Error loading PhysInformer: {e}")
            return None

        return self.physinformer

    def _load_natural_sequences(
        self,
        cell_type: str,
        n_sequences: int = 1000
    ) -> List[Tuple[str, float]]:
        """Load natural sequences with their activities from lentiMPRA data."""
        if cell_type in self.natural_sequences:
            return self.natural_sequences[cell_type][:n_sequences]

        data_path = self.DATA_PATHS.get(cell_type)
        sequences_with_activity = []

        # lentiMPRA data format: Fold, Sequence ID, Observed log2(RNA/DNA), Sequence
        possible_paths = [
            data_path / 'all_folds.tsv' if data_path else None,
            data_path / 'fold_1.tsv' if data_path else None,
            FUSEMAP_ROOT / f'data/lentiMPRA_data/{cell_type}/fold_splits_with_seq/all_folds.tsv',
        ]

        for path in possible_paths:
            if path and path.exists():
                try:
                    df = pd.read_csv(path, sep='\t', nrows=10000)

                    # lentiMPRA columns: Fold, Sequence ID, Observed log2(RNA/DNA), Sequence
                    seq_col = 'Sequence'
                    act_col = 'Observed log2(RNA/DNA)'

                    if seq_col in df.columns and act_col in df.columns:
                        for _, row in df.iterrows():
                            seq = str(row[seq_col]).upper()
                            act = float(row[act_col])
                            if len(seq) >= 200 and all(c in 'ACGT' for c in seq):
                                sequences_with_activity.append((seq, act))

                        # Sort by activity (descending)
                        sequences_with_activity.sort(key=lambda x: x[1], reverse=True)
                        print(f"  Loaded {len(sequences_with_activity)} natural sequences for {cell_type}")
                        break
                    else:
                        print(f"  Warning: Expected columns not found in {path}")
                        print(f"  Available columns: {df.columns.tolist()}")
                except Exception as e:
                    print(f"  Error loading {path}: {e}")
                    continue

        if not sequences_with_activity:
            print(f"  Warning: No natural sequences for {cell_type}, using random")
            for i in range(n_sequences):
                seq = ''.join(random.choices('ACGT', k=230))
                sequences_with_activity.append((seq, 0.0))

        self.natural_sequences[cell_type] = sequences_with_activity
        return sequences_with_activity[:n_sequences]

    # =========================================================================
    # CORE SCORING FUNCTIONS
    # =========================================================================

    def compute_specificity(
        self,
        target_activity: float,
        off1_activity: float,
        off2_activity: float,
    ) -> float:
        """
        Compute cell-type specificity score.

        S(seq) = Activity_target - λ₁·Activity_off1 - λ₂·Activity_off2
        """
        return target_activity - self.lambda1 * off1_activity - self.lambda2 * off2_activity

    def predict_with_specificity(
        self,
        sequence: str,
        target_cell: str,
        background_cells: List[str],
    ) -> Tuple[float, float, float, float, float]:
        """
        Predict activities and compute specificity.

        Returns: (specificity, target_activity, off1_activity, off2_activity, uncertainty)
        """
        pred = self.ensemble.predict(sequence)

        target_act = pred.get_activity(target_cell)
        off1_act = pred.get_activity(background_cells[0])
        off2_act = pred.get_activity(background_cells[1])

        # Get uncertainty if available
        uncertainty = pred.predictions[target_cell].std if target_cell in pred.predictions else 0.0

        specificity = self.compute_specificity(target_act, off1_act, off2_act)

        return specificity, target_act, off1_act, off2_act, uncertainty

    def batch_predict_specificities(
        self,
        sequences: List[str],
        target_cell: str,
        background_cells: List[str],
        batch_size: int = 128,
    ) -> List[Tuple[float, float, float, float, float]]:
        """Batch prediction for efficiency using GPU batching."""
        predictions = self.ensemble.predict_batch(sequences, progress=False, batch_size=batch_size)

        results = []
        for pred in predictions:
            target_act = pred.get_activity(target_cell)
            off1_act = pred.get_activity(background_cells[0])
            off2_act = pred.get_activity(background_cells[1])
            uncertainty = pred.predictions[target_cell].std if target_cell in pred.predictions else 0.0
            specificity = self.compute_specificity(target_act, off1_act, off2_act)
            results.append((specificity, target_act, off1_act, off2_act, uncertainty))

        return results

    def check_physics_constraints(self, sequence: str) -> Tuple[bool, List[str]]:
        """Check if sequence satisfies physics constraints."""
        seq = sequence.upper()
        violations = []

        # GC content
        gc = (seq.count('G') + seq.count('C')) / len(seq)
        if gc < 0.35 or gc > 0.65:
            violations.append(f'gc={gc:.2f}')

        # Homopolymer runs
        max_homo = 1
        current = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current += 1
                max_homo = max(max_homo, current)
            else:
                current = 1
        if max_homo > 6:
            violations.append(f'homo={max_homo}')

        # Complexity
        kmers = set()
        for i in range(len(seq) - 3):
            kmers.add(seq[i:i+4])
        complexity = len(kmers) / max(len(seq) - 3, 1)
        if complexity < 0.7:
            violations.append(f'complexity={complexity:.2f}')

        return len(violations) == 0, violations

    # =========================================================================
    # METHOD 1: PINCSD (Physics-Informed Naturality-Constrained Sequence Design)
    # =========================================================================

    def method_pincsd(
        self,
        target_cell: str,
        background_cells: List[str],
        n_sequences: int = 200,
        max_iterations: int = 20,  # Reduced from 50
        n_mutations_per_iter: int = 3,
        uncertainty_penalty: float = 0.3,
        physics_penalty: float = 0.2,
        early_stop_patience: int = 5,  # Reduced from 10
        temperature: float = 0.1,
        n_positions_sample: int = 10,  # Reduced from 20
    ) -> OptimizationResult:
        """
        PINCSD: Gradient-guided iterative mutagenesis with physics constraints.

        Uses CADENCE to guide mutations toward improving specificity while
        PhysInformer enforces biophysical constraints.
        """
        print(f"\n{'='*60}")
        print(f"METHOD 1: PINCSD for {target_cell}")
        print(f"  Background: {background_cells}")
        print(f"  Optimizing: Specificity = target - 0.5*off1 - 0.5*off2")
        print(f"{'='*60}")

        # Load seed sequences (top natural sequences for target cell)
        natural = self._load_natural_sequences(target_cell, n_sequences)
        seed_sequences = [seq for seq, _ in natural[:n_sequences]]

        optimized_sequences = []
        all_history = []

        for seed_idx, seed_seq in enumerate(tqdm(seed_sequences, desc="PINCSD Optimization")):
            current_seq = seed_seq
            best_seq = seed_seq
            best_composite = -float('inf')

            history = []
            no_improve_count = 0

            # Get initial scores
            spec, target_act, off1_act, off2_act, unc = self.predict_with_specificity(
                current_seq, target_cell, background_cells
            )
            physics_ok, violations = self.check_physics_constraints(current_seq)
            physics_penalty_val = len(violations) * physics_penalty
            composite = spec - uncertainty_penalty * unc - physics_penalty_val

            for iteration in range(max_iterations):
                history.append({
                    'iteration': iteration,
                    'specificity': spec,
                    'uncertainty': unc,
                    'physics_violations': len(violations),
                    'composite': composite,
                })

                if composite > best_composite:
                    best_composite = composite
                    best_seq = current_seq
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= early_stop_patience:
                    break

                # Gradient-guided mutation proposal
                # We approximate gradients by evaluating nearby mutations
                mutations_to_try = []

                # Sample positions weighted by position importance
                # (In full implementation, use actual gradients from CADENCE)
                L = len(current_seq)
                positions = np.random.choice(L, size=min(n_positions_sample, L), replace=False)

                for pos in positions:
                    current_nuc = current_seq[pos]
                    for new_nuc in 'ACGT':
                        if new_nuc != current_nuc:
                            # Create mutant
                            mutant = current_seq[:pos] + new_nuc + current_seq[pos+1:]
                            mutations_to_try.append((pos, new_nuc, mutant))

                if not mutations_to_try:
                    break

                # Batch evaluate mutations
                mutant_seqs = [m[2] for m in mutations_to_try]
                mutant_results = self.batch_predict_specificities(
                    mutant_seqs, target_cell, background_cells
                )

                # Compute improvement for each mutation
                improvements = []
                for i, (pos, new_nuc, mutant) in enumerate(mutations_to_try):
                    m_spec, m_target, m_off1, m_off2, m_unc = mutant_results[i]
                    m_physics_ok, m_violations = self.check_physics_constraints(mutant)
                    m_physics_penalty = len(m_violations) * physics_penalty
                    m_composite = m_spec - uncertainty_penalty * m_unc - m_physics_penalty

                    delta = m_composite - composite
                    improvements.append((pos, new_nuc, mutant, delta, m_composite, m_spec, m_unc, m_violations))

                # Sort by improvement
                improvements.sort(key=lambda x: -x[3])

                # Select top mutations using temperature-based sampling
                if improvements[0][3] > 0:
                    # Take best mutation
                    selected = improvements[0]
                else:
                    # Probabilistic acceptance (Metropolis-Hastings)
                    probs = softmax([x[3] for x in improvements[:10]], temperature)
                    idx = np.random.choice(min(10, len(improvements)), p=probs)
                    selected = improvements[idx]

                pos, new_nuc, new_seq, delta, new_composite, new_spec, new_unc, new_violations = selected

                # Accept/reject
                if delta > 0:
                    current_seq = new_seq
                    spec, unc, violations, composite = new_spec, new_unc, new_violations, new_composite
                else:
                    accept_prob = np.exp(delta / temperature)
                    if np.random.random() < accept_prob:
                        current_seq = new_seq
                        spec, unc, violations, composite = new_spec, new_unc, new_violations, new_composite

            optimized_sequences.append(best_seq)
            all_history.append(history)

        # Evaluate final sequences
        return self._evaluate_sequences(
            optimized_sequences, target_cell, background_cells,
            'pincsd', all_history
        )

    # =========================================================================
    # METHOD 2: EMOO (Evolutionary Multi-Objective Optimization)
    # =========================================================================

    def method_emoo(
        self,
        target_cell: str,
        background_cells: List[str],
        n_sequences: int = 200,
        population_size: int = 100,  # Reduced from 200
        n_generations: int = 20,  # Reduced from 50
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.02,
    ) -> OptimizationResult:
        """
        EMOO: NSGA-II style multi-objective optimization.

        Objectives:
        1. Maximize target cell activity
        2. Minimize off-target 1 activity
        3. Minimize off-target 2 activity
        4. Maximize physics conformity (minimize violations)
        """
        print(f"\n{'='*60}")
        print(f"METHOD 3: EMOO (Evolutionary Multi-Objective) for {target_cell}")
        print(f"  Background: {background_cells}")
        print(f"  Population: {population_size}, Generations: {n_generations}")
        print(f"{'='*60}")

        # Initialize population from natural sequences
        natural = self._load_natural_sequences(target_cell, population_size * 2)
        seed_seqs = [seq for seq, _ in natural[:population_size]]

        # Fill if needed
        while len(seed_seqs) < population_size:
            parent = random.choice(seed_seqs)
            child = self._mutate_sequence(parent, mutation_rate * 5)
            seed_seqs.append(child)

        # Initialize population with objectives (BATCH evaluation)
        print("  Initializing population...", flush=True)
        init_seqs = seed_seqs[:population_size]
        init_objectives = self._batch_compute_emoo_objectives(init_seqs, target_cell, background_cells)

        population = []
        for seq, objectives in zip(init_seqs, init_objectives):
            population.append({
                'sequence': seq,
                'objectives': objectives,
                'rank': 0,
                'crowding': 0,
            })

        # Evolution loop
        for gen in range(n_generations):
            # Generate offspring sequences first (without evaluation)
            offspring_seqs = []
            while len(offspring_seqs) < population_size:
                # Tournament selection
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)

                # Crossover
                if np.random.random() < crossover_rate:
                    child_seq = self._crossover(parent1['sequence'], parent2['sequence'])
                else:
                    child_seq = parent1['sequence']

                # Mutation
                child_seq = self._mutate_sequence(child_seq, mutation_rate)
                offspring_seqs.append(child_seq)

            # BATCH evaluate all offspring at once
            offspring_objectives = self._batch_compute_emoo_objectives(
                offspring_seqs, target_cell, background_cells
            )

            offspring = []
            for seq, objectives in zip(offspring_seqs, offspring_objectives):
                offspring.append({
                    'sequence': seq,
                    'objectives': objectives,
                    'rank': 0,
                    'crowding': 0,
                })

            # Combine parent + offspring
            combined = population + offspring

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(combined)

            # Select next generation
            population = []
            for front in fronts:
                if len(population) + len(front) <= population_size:
                    self._compute_crowding_distance(front, combined)
                    population.extend([combined[i] for i in front])
                else:
                    self._compute_crowding_distance(front, combined)
                    front_sorted = sorted(front, key=lambda i: -combined[i]['crowding'])
                    remaining = population_size - len(population)
                    population.extend([combined[i] for i in front_sorted[:remaining]])
                    break

            # Progress every generation
            pareto = [p for p in population if p['rank'] == 0]
            avg_spec = np.mean([
                p['objectives'][0] - 0.5 * p['objectives'][1] - 0.5 * p['objectives'][2]
                for p in pareto
            ])
            print(f"  Gen {gen+1}/{n_generations}: Pareto size={len(pareto)}, Avg spec={avg_spec:.3f}", flush=True)

        # Extract Pareto front
        pareto_front = [p for p in population if p['rank'] == 0]

        # Evaluate and filter
        print(f"  Final Pareto front: {len(pareto_front)} solutions")
        results = []
        for p in pareto_front:
            verdict = simple_oracle_check(p['sequence'])
            spec = p['objectives'][0] - 0.5 * p['objectives'][1] - 0.5 * p['objectives'][2]
            results.append({
                'sequence': p['sequence'],
                'specificity': spec,
                'target_activity': p['objectives'][0],
                'off1_activity': p['objectives'][1],
                'off2_activity': p['objectives'][2],
                'uncertainty': 0.0,
                'verdict': verdict,
            })

        # Filter and sort
        valid = [r for r in results if r['verdict'] in ['GREEN', 'YELLOW']]
        valid.sort(key=lambda x: -x['specificity'])
        selected = valid[:n_sequences]

        return OptimizationResult(
            method='emoo',
            cell_type=target_cell,
            sequences=[s['sequence'] for s in selected],
            target_activities=[s['target_activity'] for s in selected],
            background_activities={
                background_cells[0]: [s['off1_activity'] for s in selected],
                background_cells[1]: [s['off2_activity'] for s in selected],
            },
            specificities=[s['specificity'] for s in selected],
            uncertainties=[s['uncertainty'] for s in selected],
            oracle_verdicts=[s['verdict'] for s in selected],
        )

    def _compute_emoo_objectives(
        self,
        sequence: str,
        target_cell: str,
        background_cells: List[str],
    ) -> np.ndarray:
        """Compute 4 objectives for EMOO."""
        pred = self.ensemble.predict(sequence)
        target_act = pred.get_activity(target_cell)
        off1_act = pred.get_activity(background_cells[0])
        off2_act = pred.get_activity(background_cells[1])

        # Physics conformity (count violations)
        _, violations = self.check_physics_constraints(sequence)
        physics_score = -len(violations)  # Higher is better

        return np.array([target_act, off1_act, off2_act, physics_score])

    def _batch_compute_emoo_objectives(
        self,
        sequences: List[str],
        target_cell: str,
        background_cells: List[str],
    ) -> List[np.ndarray]:
        """Batch compute objectives for multiple sequences."""
        # Batch predict all sequences at once
        predictions = self.ensemble.predict_batch(sequences, progress=False, batch_size=128)

        results = []
        for seq, pred in zip(sequences, predictions):
            target_act = pred.get_activity(target_cell)
            off1_act = pred.get_activity(background_cells[0])
            off2_act = pred.get_activity(background_cells[1])

            # Physics conformity
            _, violations = self.check_physics_constraints(seq)
            physics_score = -len(violations)

            results.append(np.array([target_act, off1_act, off2_act, physics_score]))

        return results

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 Pareto-dominates obj2 (maximizing objectives 0,3; minimizing 1,2)."""
        # Objective 0: maximize target activity
        # Objective 1: minimize off1 (so higher obj1[1] is worse)
        # Objective 2: minimize off2
        # Objective 3: maximize physics conformity

        better = np.array([
            obj1[0] >= obj2[0],  # target: higher is better
            obj1[1] <= obj2[1],  # off1: lower is better
            obj1[2] <= obj2[2],  # off2: lower is better
            obj1[3] >= obj2[3],  # physics: higher is better
        ])
        strictly_better = np.array([
            obj1[0] > obj2[0],
            obj1[1] < obj2[1],
            obj1[2] < obj2[2],
            obj1[3] > obj2[3],
        ])
        return np.all(better) and np.any(strictly_better)

    def _fast_non_dominated_sort(self, population: List[Dict]) -> List[List[int]]:
        """NSGA-II fast non-dominated sorting."""
        n = len(population)
        if n == 0:
            return [[]]

        domination_count = [0] * n
        dominated_set = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(population[i]['objectives'], population[j]['objectives']):
                    dominated_set[i].append(j)
                elif self._dominates(population[j]['objectives'], population[i]['objectives']):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                population[i]['rank'] = 0
                fronts[0].append(i)

        # Handle edge case where no one dominates anyone (all same rank)
        if not fronts[0]:
            fronts[0] = list(range(n))
            for i in range(n):
                population[i]['rank'] = 0
            return fronts

        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j]['rank'] = current_front + 1
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)

        # Remove empty trailing fronts
        while fronts and not fronts[-1]:
            fronts.pop()

        return fronts if fronts else [[]]

    def _compute_crowding_distance(self, front: List[int], population: List[Dict]):
        """Compute crowding distance for diversity preservation."""
        if len(front) <= 2:
            for i in front:
                population[i]['crowding'] = float('inf')
            return

        n_obj = len(population[front[0]]['objectives'])

        for i in front:
            population[i]['crowding'] = 0

        for m in range(n_obj):
            front_sorted = sorted(front, key=lambda i: population[i]['objectives'][m])
            population[front_sorted[0]]['crowding'] = float('inf')
            population[front_sorted[-1]]['crowding'] = float('inf')

            obj_range = (population[front_sorted[-1]]['objectives'][m] -
                        population[front_sorted[0]]['objectives'][m])

            if obj_range == 0:
                continue

            for i in range(1, len(front_sorted) - 1):
                population[front_sorted[i]]['crowding'] += (
                    population[front_sorted[i+1]]['objectives'][m] -
                    population[front_sorted[i-1]]['objectives'][m]
                ) / obj_range

    def _tournament_select(self, population: List[Dict], k: int = 2) -> Dict:
        """Binary tournament selection."""
        contestants = random.sample(range(len(population)), k)
        best = contestants[0]
        for c in contestants[1:]:
            if population[c]['rank'] < population[best]['rank']:
                best = c
            elif (population[c]['rank'] == population[best]['rank'] and
                  population[c]['crowding'] > population[best]['crowding']):
                best = c
        return population[best]

    def _crossover(self, parent1: str, parent2: str) -> str:
        """Two-point crossover."""
        L = len(parent1)
        pt1, pt2 = sorted(random.sample(range(L), 2))
        return parent1[:pt1] + parent2[pt1:pt2] + parent1[pt2:]

    def _mutate_sequence(self, sequence: str, rate: float) -> str:
        """Point mutation."""
        seq_list = list(sequence)
        for i in range(len(seq_list)):
            if random.random() < rate:
                current = seq_list[i]
                alternatives = [n for n in 'ACGT' if n != current]
                seq_list[i] = random.choice(alternatives)
        return ''.join(seq_list)

    # =========================================================================
    # METHOD 3: ISM (In-Silico Mutagenesis Baseline)
    # =========================================================================

    def method_ism(
        self,
        target_cell: str,
        background_cells: List[str],
        n_sequences: int = 200,
        max_iterations: int = 20,  # Reduced from 50
        early_stop_patience: int = 5,  # Reduced from 10
        n_positions_sample: int = 15,  # Positions to try per iteration
    ) -> OptimizationResult:
        """
        ISM: Greedy single-nucleotide mutation baseline.

        Optimizes for SPECIFICITY (not just target activity).
        """
        print(f"\n{'='*60}")
        print(f"METHOD 5: ISM (In-Silico Mutagenesis) for {target_cell}")
        print(f"  Background: {background_cells}")
        print(f"  Optimizing: Specificity = target - 0.5*off1 - 0.5*off2")
        print(f"{'='*60}")

        # Load seed sequences
        natural = self._load_natural_sequences(target_cell, n_sequences)
        seed_seqs = [seq for seq, _ in natural[:n_sequences]]

        if not seed_seqs:
            seed_seqs = [''.join(random.choices('ACGT', k=230)) for _ in range(n_sequences)]

        optimized_sequences = []

        for seed_seq in tqdm(seed_seqs, desc="ISM Optimization"):
            current_seq = seed_seq
            best_seq = seed_seq

            spec, _, _, _, _ = self.predict_with_specificity(
                current_seq, target_cell, background_cells
            )
            best_spec = spec
            no_improve = 0

            for iteration in range(max_iterations):
                # Try all single mutations at sampled positions
                L = len(current_seq)
                positions = random.sample(range(L), min(n_positions_sample, L))

                best_mutation = None
                best_new_spec = best_spec

                mutations = []
                for pos in positions:
                    current_nuc = current_seq[pos]
                    for new_nuc in 'ACGT':
                        if new_nuc != current_nuc:
                            mutant = current_seq[:pos] + new_nuc + current_seq[pos+1:]
                            mutations.append((pos, new_nuc, mutant))

                # Batch evaluate
                if mutations:
                    mutant_seqs = [m[2] for m in mutations]
                    results = self.batch_predict_specificities(
                        mutant_seqs, target_cell, background_cells
                    )

                    for (pos, new_nuc, mutant), (m_spec, _, _, _, _) in zip(mutations, results):
                        if m_spec > best_new_spec:
                            best_new_spec = m_spec
                            best_mutation = (pos, new_nuc, mutant)

                if best_mutation and best_new_spec > best_spec:
                    _, _, current_seq = best_mutation
                    best_seq = current_seq
                    best_spec = best_new_spec
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= early_stop_patience:
                    break

            optimized_sequences.append(best_seq)

        return self._evaluate_sequences(
            optimized_sequences, target_cell, background_cells, 'ism'
        )

    def method_ism_target(
        self,
        target_cell: str,
        background_cells: List[str],
        n_sequences: int = 200,
        max_iterations: int = 20,
        early_stop_patience: int = 5,
        n_positions_sample: int = 15,
    ) -> OptimizationResult:
        """
        ISM-Target: Greedy ISM optimizing TARGET ACTIVITY ONLY (no specificity).

        This is a baseline/ablation to show the importance of optimizing for
        specificity rather than just target activity. Specificities are still
        computed for evaluation/comparison.
        """
        print(f"\n{'='*60}")
        print(f"METHOD 4: ISM-Target (Target-only optimization) for {target_cell}")
        print(f"  Background: {background_cells}")
        print(f"  Optimizing: TARGET ACTIVITY ONLY (no specificity penalty)")
        print(f"  Evaluation: Specificities computed for comparison")
        print(f"{'='*60}")

        # Load seed sequences
        natural = self._load_natural_sequences(target_cell, n_sequences)
        seed_seqs = [seq for seq, _ in natural[:n_sequences]]

        if not seed_seqs:
            seed_seqs = [''.join(random.choices('ACGT', k=230)) for _ in range(n_sequences)]

        optimized_sequences = []

        for seed_seq in tqdm(seed_seqs, desc="ISM-Target Optimization"):
            current_seq = seed_seq
            best_seq = seed_seq

            # Get initial target activity only
            pred = self.ensemble.predict(current_seq)
            best_target = pred.predictions[target_cell].mean
            no_improve = 0

            for iteration in range(max_iterations):
                L = len(current_seq)
                positions = random.sample(range(L), min(n_positions_sample, L))

                best_mutation = None
                best_new_target = best_target

                mutations = []
                for pos in positions:
                    current_nuc = current_seq[pos]
                    for new_nuc in 'ACGT':
                        if new_nuc != current_nuc:
                            mutant = current_seq[:pos] + new_nuc + current_seq[pos+1:]
                            mutations.append((pos, new_nuc, mutant))

                # Batch evaluate - only need target cell predictions
                if mutations:
                    mutant_seqs = [m[2] for m in mutations]
                    preds = self.ensemble.predict_batch(mutant_seqs, progress=False)

                    for (pos, new_nuc, mutant), pred in zip(mutations, preds):
                        m_target = pred.predictions[target_cell].mean
                        if m_target > best_new_target:
                            best_new_target = m_target
                            best_mutation = (pos, new_nuc, mutant)

                if best_mutation and best_new_target > best_target:
                    _, _, current_seq = best_mutation
                    best_seq = current_seq
                    best_target = best_new_target
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= early_stop_patience:
                    break

            optimized_sequences.append(best_seq)

        # Evaluate with specificities for comparison (even though we didn't optimize for it)
        return self._evaluate_sequences(
            optimized_sequences, target_cell, background_cells, 'ism_target'
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _evaluate_sequences(
        self,
        sequences: List[str],
        target_cell: str,
        background_cells: List[str],
        method_name: str,
        history: List = None,
    ) -> OptimizationResult:
        """Evaluate sequences and create result object."""
        if not sequences:
            return OptimizationResult(
                method=method_name,
                cell_type=target_cell,
                sequences=[],
                target_activities=[],
                background_activities={c: [] for c in background_cells},
                specificities=[],
                uncertainties=[],
                oracle_verdicts=[],
            )

        print(f"  Evaluating {len(sequences)} sequences...")

        results = self.batch_predict_specificities(sequences, target_cell, background_cells)

        target_activities = []
        background_activities = {c: [] for c in background_cells}
        specificities = []
        uncertainties = []
        oracle_verdicts = []

        for seq, (spec, target_act, off1_act, off2_act, unc) in zip(sequences, results):
            target_activities.append(target_act)
            background_activities[background_cells[0]].append(off1_act)
            background_activities[background_cells[1]].append(off2_act)
            specificities.append(spec)
            uncertainties.append(unc)
            oracle_verdicts.append(simple_oracle_check(seq))

        return OptimizationResult(
            method=method_name,
            cell_type=target_cell,
            sequences=sequences,
            target_activities=target_activities,
            background_activities=background_activities,
            specificities=specificities,
            uncertainties=uncertainties,
            oracle_verdicts=oracle_verdicts,
            optimization_history=history or [],
        )

    def _fallback_random(
        self,
        target_cell: str,
        background_cells: List[str],
        n_sequences: int,
        method_name: str,
    ) -> OptimizationResult:
        """Fallback to random sequences when method unavailable."""
        print(f"  Using random fallback for {method_name}")
        sequences = [''.join(random.choices('ACGT', k=230)) for _ in range(n_sequences)]
        return self._evaluate_sequences(sequences, target_cell, background_cells, method_name)

    # =========================================================================
    # RUN COMPARISON
    # =========================================================================

    def run_comparison(
        self,
        n_sequences_per_method: int = 200,
        output_dir: str = None,
        methods: List[str] = None,
    ) -> pd.DataFrame:
        """Run all methods for all cell types and compare."""
        if methods is None:
            methods = ['pincsd', 'emoo', 'ism', 'ism_target']

        print("\n" + "="*70)
        print("THERAPEUTIC ENHANCER OPTIMIZATION - METHOD COMPARISON")
        print("="*70)
        print(f"Methods: {methods}")
        print(f"Sequences per method: {n_sequences_per_method}")
        print(f"Cell types: {self.all_cell_types}")
        print(f"Specificity: S = target - {self.lambda1}*off1 - {self.lambda2}*off2")
        print("="*70)

        all_results = []

        for target_cell in self.all_cell_types:
            background_cells = [c for c in self.all_cell_types if c != target_cell]

            print(f"\n{'#'*70}")
            print(f"# TARGET: {target_cell}, BACKGROUND: {background_cells}")
            print(f"{'#'*70}")

            for method in methods:
                if method == 'pincsd':
                    result = self.method_pincsd(
                        target_cell, background_cells, n_sequences_per_method
                    )
                elif method == 'emoo':
                    result = self.method_emoo(
                        target_cell, background_cells, n_sequences_per_method
                    )
                elif method == 'ism':
                    result = self.method_ism(
                        target_cell, background_cells, n_sequences_per_method
                    )
                elif method == 'ism_target':
                    result = self.method_ism_target(
                        target_cell, background_cells, n_sequences_per_method
                    )
                else:
                    print(f"  Unknown method: {method}")
                    continue

                all_results.append(result)

        # Compile summary
        summary_data = []
        for result in all_results:
            if result.specificities:
                n_green = sum(1 for v in result.oracle_verdicts if v == 'GREEN')
                n_yellow = sum(1 for v in result.oracle_verdicts if v == 'YELLOW')
                n_red = sum(1 for v in result.oracle_verdicts if v == 'RED')

                summary_data.append({
                    'method': result.method,
                    'target_cell': result.cell_type,
                    'n_sequences': len(result.sequences),
                    'mean_specificity': np.mean(result.specificities),
                    'max_specificity': np.max(result.specificities),
                    'std_specificity': np.std(result.specificities),
                    'mean_target_activity': np.mean(result.target_activities),
                    'pass_rate': (n_green + n_yellow) / len(result.sequences),
                    'n_green': n_green,
                    'n_yellow': n_yellow,
                    'n_red': n_red,
                })

        summary_df = pd.DataFrame(summary_data)

        # Print summary
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(summary_df.to_string(index=False))

        # Compute method rankings
        print("\n" + "-"*70)
        print("METHOD RANKINGS (by mean specificity)")
        print("-"*70)
        method_avg = summary_df.groupby('method')['mean_specificity'].mean().sort_values(ascending=False)
        for i, (method, spec) in enumerate(method_avg.items(), 1):
            print(f"  {i}. {method}: {spec:.3f}")

        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save summary
            summary_df.to_csv(output_dir / 'method_comparison_summary.csv', index=False)

            # Save detailed results
            for result in all_results:
                if result.sequences:
                    df = pd.DataFrame({
                        'sequence': result.sequences,
                        'target_activity': result.target_activities,
                        'specificity': result.specificities,
                        'uncertainty': result.uncertainties,
                        'oracle_verdict': result.oracle_verdicts,
                    })
                    for bg_cell, acts in result.background_activities.items():
                        df[f'activity_{bg_cell}'] = acts

                    filename = f'{result.method}_{result.cell_type}_results.csv'
                    df.to_csv(output_dir / filename, index=False)

            # Save report
            report = {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'n_sequences_per_method': n_sequences_per_method,
                    'lambda1': self.lambda1,
                    'lambda2': self.lambda2,
                },
                'methods': methods,
                'cell_types': self.all_cell_types,
                'summary': summary_df.to_dict('records'),
            }

            with open(output_dir / 'comparison_report.json', 'w') as f:
                json.dump(report, f, indent=2)

            print(f"\nResults saved to {output_dir}")

        return summary_df


def main():
    parser = argparse.ArgumentParser(
        description='Compare therapeutic enhancer optimization methods'
    )
    parser.add_argument('--n-sequences', type=int, default=200,
                        help='Number of sequences per method')
    parser.add_argument('--output-dir', type=str,
                        default=str(FUSEMAP_ROOT / 'results/therapeutic_method_comparison'),
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['pincsd', 'emoo', 'ism', 'ism_target'],
                        help='Methods to run')
    parser.add_argument('--lambda1', type=float, default=0.5,
                        help='Penalty weight for first background cell')
    parser.add_argument('--lambda2', type=float, default=0.5,
                        help='Penalty weight for second background cell')

    args = parser.parse_args()

    comparison = TherapeuticMethodComparison(
        device=args.device,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
    )

    summary = comparison.run_comparison(
        n_sequences_per_method=args.n_sequences,
        output_dir=args.output_dir,
        methods=args.methods,
    )

    return summary


if __name__ == '__main__':
    main()
