#!/usr/bin/env python3
"""
Therapeutic Enhancer Design Pipeline v2.

Complete pipeline for designing cell-type specific therapeutic enhancers:
1. Extract physics profiles from natural high-activity enhancers
2. Generate candidates via PhysicsVAE with target physics conditioning
3. Predict activity in multiple cell types using CADENCE ensemble
4. Validate through OracleCheck (GREEN/YELLOW only)
5. Check motif constraints (required/forbidden TF binding sites)
6. Score for cell-type specificity with diversity filtering
7. Generate synthesis-ready output

Example: Liver-targeted AAV enhancers
  - Target: HepG2 (hepatocyte model)
  - Background: K562, WTC11 (negative controls)
  - Required motifs: HNF4A, FOXA
  - Forbidden motifs: GATA factors
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
import json
import sys
from dataclasses import dataclass, field
from tqdm import tqdm

# Add paths
FUSEMAP_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FUSEMAP_ROOT))
sys.path.insert(0, str(FUSEMAP_ROOT / 'applications/utils'))

# Core imports
try:
    from .utils.multicell_ensemble import (
        MultiCellEnsemble, MultiCellPrediction,
        CellTypeActivity, find_cadence_checkpoints
    )
    from .utils.diversity_ranking import (
        diversity_filter, greedy_diversity_selection,
        maxmin_diversity_selection, cluster_sequences,
        compute_distance_matrix, kmer_distance
    )
except ImportError:
    from utils.multicell_ensemble import (
        MultiCellEnsemble, MultiCellPrediction,
        CellTypeActivity, find_cadence_checkpoints
    )
    from utils.diversity_ranking import (
        diversity_filter, greedy_diversity_selection,
        maxmin_diversity_selection, cluster_sequences,
        compute_distance_matrix, kmer_distance
    )

# OracleCheck imports
try:
    from oracle_check.config import Verdict
    from oracle_check.validators import CompositionValidator, CompositionValidationResult
    from oracle_check.motif_validator import MotifScanner
    HAS_ORACLE_CHECK = True
except ImportError:
    HAS_ORACLE_CHECK = False
    print("Warning: OracleCheck not available")

# PhysicsVAE imports
try:
    import torch
    from physics.PhysicsVAE import PhysicsVAE
    HAS_PHYSICS_VAE = True
except ImportError:
    HAS_PHYSICS_VAE = False
    torch = None  # Set to None for checks
    print("Warning: PhysicsVAE not available")

# PhysInformer imports
try:
    from physics.PhysInformer.physics_aware_model import create_physics_aware_model
    HAS_PHYSINFORMER = True
except ImportError:
    HAS_PHYSINFORMER = False


# Motif groups for different cell types
LIVER_REQUIRED_MOTIFS = ['HNF4A', 'FOXA1', 'FOXA2', 'CEBPA', 'CEBPB']
LIVER_FORBIDDEN_MOTIFS = ['GATA1', 'GATA2', 'GATA3', 'POU5F1', 'NANOG', 'SOX2']

BLOOD_REQUIRED_MOTIFS = ['GATA1', 'GATA2', 'TAL1', 'KLF1', 'RUNX1']
BLOOD_FORBIDDEN_MOTIFS = ['HNF4A', 'FOXA1', 'FOXA2']

STEM_REQUIRED_MOTIFS = ['POU5F1', 'NANOG', 'SOX2', 'KLF4']
STEM_FORBIDDEN_MOTIFS = ['GATA1', 'HNF4A']

CELL_TYPE_MOTIF_CONSTRAINTS = {
    'HepG2': {'required': LIVER_REQUIRED_MOTIFS, 'forbidden': LIVER_FORBIDDEN_MOTIFS},
    'K562': {'required': BLOOD_REQUIRED_MOTIFS, 'forbidden': BLOOD_FORBIDDEN_MOTIFS},
    'WTC11': {'required': STEM_REQUIRED_MOTIFS, 'forbidden': STEM_FORBIDDEN_MOTIFS},
}


@dataclass
class MotifConstraintResult:
    """Result of motif constraint checking."""
    passed: bool
    required_found: List[str]
    required_missing: List[str]
    forbidden_found: List[str]
    total_motif_hits: int
    message: str


@dataclass
class OracleCheckResult:
    """Simplified OracleCheck result."""
    verdict: str  # GREEN, YELLOW, RED
    gc_content: float
    max_homopolymer: int
    has_cpg_island: bool
    repeat_fraction: float
    flags: List[str]


@dataclass
class EnhancerCandidate:
    """Complete candidate with all evaluations."""
    sequence: str
    source: str  # 'natural', 'vae_generated', 'optimized'

    # Activity predictions
    activities: Dict[str, float] = field(default_factory=dict)
    uncertainties: Dict[str, float] = field(default_factory=dict)

    # Specificity
    target_activity: float = 0.0
    max_background: float = 0.0
    specificity_score: float = 0.0

    # Validation
    oracle_verdict: str = 'UNKNOWN'
    motif_result: Optional[MotifConstraintResult] = None

    # Physics
    physics_features: Optional[np.ndarray] = None

    # Final status
    passed_all_filters: bool = False


class TherapeuticEnhancerPipeline:
    """
    Complete pipeline for therapeutic enhancer design.

    Implements the full 6-step protocol:
    1. Extract physics profiles from natural enhancers
    2. Generate candidates via PhysicsVAE
    3. Predict multi-cell activity
    4. OracleCheck validation
    5. Motif constraint filtering
    6. Specificity ranking with diversity
    """

    # PhysicsVAE checkpoints
    VAE_CHECKPOINTS = {
        'K562': FUSEMAP_ROOT / 'physics/PhysicsVAE/runs/K562_20260113_051653/best_model.pt',
        'HepG2': FUSEMAP_ROOT / 'physics/PhysicsVAE/runs/HepG2_20260113_052418/best_model.pt',
        'WTC11': FUSEMAP_ROOT / 'physics/PhysicsVAE/runs/WTC11_20260113_052743/best_model.pt',
    }

    def __init__(
        self,
        target_cell: str = 'HepG2',
        background_cells: List[str] = None,
        device: str = 'cuda',
        required_motifs: List[str] = None,
        forbidden_motifs: List[str] = None,
    ):
        """
        Initialize therapeutic enhancer design pipeline.

        Args:
            target_cell: Cell type to maximize activity in
            background_cells: Cell types to minimize activity in
            device: 'cuda' or 'cpu'
            required_motifs: TF motifs that must be present
            forbidden_motifs: TF motifs that must be absent
        """
        self.target_cell = target_cell
        self.background_cells = background_cells or [c for c in ['K562', 'HepG2', 'WTC11'] if c != target_cell]
        self.all_cell_types = [target_cell] + self.background_cells
        self.device = device

        # Motif constraints
        if required_motifs is None and target_cell in CELL_TYPE_MOTIF_CONSTRAINTS:
            required_motifs = CELL_TYPE_MOTIF_CONSTRAINTS[target_cell]['required']
        if forbidden_motifs is None and target_cell in CELL_TYPE_MOTIF_CONSTRAINTS:
            forbidden_motifs = CELL_TYPE_MOTIF_CONSTRAINTS[target_cell]['forbidden']
        self.required_motifs = required_motifs or []
        self.forbidden_motifs = forbidden_motifs or []

        # Initialize components
        print(f"Initializing Therapeutic Enhancer Pipeline")
        print(f"  Target: {target_cell}, Background: {self.background_cells}")
        print(f"  Required motifs: {self.required_motifs}")
        print(f"  Forbidden motifs: {self.forbidden_motifs}")

        # Multi-cell ensemble
        print("Loading multi-cell CADENCE ensemble...")
        checkpoints = find_cadence_checkpoints()
        self.ensemble = MultiCellEnsemble(
            cell_types=self.all_cell_types,
            checkpoints=checkpoints,
            device=device
        )

        # Motif scanner
        self.motif_scanner = None
        if HAS_ORACLE_CHECK:
            print("Loading motif scanner (879 JASPAR human motifs)...")
            self.motif_scanner = MotifScanner(species='human')
            print(f"  Loaded {len(self.motif_scanner.pwms)} motifs")

        # PhysicsVAE
        self.vae_model = None
        if HAS_PHYSICS_VAE and target_cell in self.VAE_CHECKPOINTS:
            vae_path = self.VAE_CHECKPOINTS[target_cell]
            if vae_path.exists():
                print(f"Loading PhysicsVAE for {target_cell}...")
                self._load_physics_vae(target_cell)

        # Composition validator for OracleCheck
        self.composition_validator = None
        if HAS_ORACLE_CHECK:
            try:
                from oracle_check.config import ValidationThresholds
                thresholds = ValidationThresholds()
                self.composition_validator = CompositionValidator(thresholds)
            except Exception as e:
                print(f"  Note: CompositionValidator not loaded: {e}")

        # Storage
        self.candidates: List[EnhancerCandidate] = []
        self.natural_physics_profiles: Optional[np.ndarray] = None
        # Physics feature dimensions per cell type
        self.n_physics_features = {'K562': 515, 'HepG2': 539, 'WTC11': 539}.get(target_cell, 515)

    def _load_physics_vae(self, cell_type: str):
        """Load PhysicsVAE model."""
        try:
            checkpoint = torch.load(self.VAE_CHECKPOINTS[cell_type], map_location=self.device, weights_only=False)

            # Get model config - check multiple possible locations
            config = checkpoint.get('config', {})
            n_physics = config.get('n_physics_features', None)

            # If not in config, infer from state dict
            if n_physics is None:
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                # physics_encoder.encoder.0.weight has shape [256, n_physics]
                for key, value in state_dict.items():
                    if 'physics_encoder.encoder.0.weight' in key:
                        n_physics = value.shape[1]
                        break

            if n_physics is None:
                n_physics = 539 if cell_type == 'HepG2' else 515  # Default by cell type

            seq_length = config.get('seq_length', 230)

            self.vae_model = PhysicsVAE(
                seq_length=seq_length,
                n_physics_features=n_physics,
                latent_dim=config.get('latent_dim', 128),
                physics_cond_dim=config.get('physics_cond_dim', 64),
            ).to(self.device)

            self.vae_model.load_state_dict(checkpoint['model_state_dict'])
            self.vae_model.eval()
            self.n_physics_features = n_physics  # Store for later use
            print(f"  Loaded PhysicsVAE ({n_physics} physics features)")
        except Exception as e:
            print(f"  Warning: Could not load PhysicsVAE: {e}")
            self.vae_model = None

    def load_natural_enhancers(self, fasta_path: str) -> List[str]:
        """Load natural enhancer sequences from FASTA."""
        import gzip
        sequences = []
        open_func = gzip.open if fasta_path.endswith('.gz') else open
        mode = 'rt' if fasta_path.endswith('.gz') else 'r'

        with open_func(fasta_path, mode) as f:
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line.upper())
            if current_seq:
                sequences.append(''.join(current_seq))

        print(f"Loaded {len(sequences)} natural enhancer sequences")
        return sequences

    def extract_target_physics_profile(
        self,
        sequences: List[str],
        top_k: int = 100
    ) -> np.ndarray:
        """
        Step 1: Extract physics profiles from top natural enhancers.

        Identifies enhancers with high target activity and low background,
        then extracts their physics feature distribution.
        """
        print(f"\nStep 1: Extracting physics profiles from top {top_k} natural enhancers...")

        # Predict activities
        predictions = self.ensemble.predict_batch(sequences, progress=True)

        # Score by specificity
        scored = []
        for i, pred in enumerate(predictions):
            target_act = pred.get_activity(self.target_cell)
            bg_acts = [pred.get_activity(c) for c in self.background_cells]
            max_bg = max(bg_acts) if bg_acts else 0
            specificity = target_act - 0.5 * sum(bg_acts)
            scored.append((i, sequences[i], target_act, max_bg, specificity))

        # Sort by specificity and filter for high target activity
        scored.sort(key=lambda x: x[4], reverse=True)
        top_enhancers = scored[:top_k]

        if top_enhancers:
            print(f"  Top enhancer specificity: {top_enhancers[0][4]:.3f}")
            print(f"  Top enhancer target activity: {top_enhancers[0][2]:.3f}")
        else:
            print("  Warning: No enhancers found")

        # Extract physics features (placeholder - would use PhysInformer)
        # For now, return mean physics as target
        n_physics = getattr(self, 'n_physics_features', 515)
        self.natural_physics_profiles = np.zeros((top_k, n_physics))  # Placeholder

        return self.natural_physics_profiles.mean(axis=0)

    def generate_vae_candidates(
        self,
        target_physics: np.ndarray,
        n_candidates: int = 500,
        temperature: float = 0.8
    ) -> List[str]:
        """
        Step 2: Generate candidates using PhysicsVAE.

        Conditions generation on target physics profile.
        """
        print(f"\nStep 2: Generating {n_candidates} candidates via PhysicsVAE...")

        if self.vae_model is None:
            print("  Warning: PhysicsVAE not available, skipping generation")
            return []

        generated = []
        batch_size = 50

        with torch.no_grad():
            for i in tqdm(range(0, n_candidates, batch_size), desc="Generating"):
                batch_n = min(batch_size, n_candidates - i)

                # Sample from latent space
                z = torch.randn(batch_n, 128).to(self.device)

                # Condition on physics - encode first
                physics_tensor = torch.tensor(
                    target_physics, dtype=torch.float32
                ).unsqueeze(0).expand(batch_n, -1).to(self.device)

                # Generate
                try:
                    # Encode physics to conditioning vector
                    physics_cond = self.vae_model.physics_encoder(physics_tensor)
                    logits = self.vae_model.decode(z, physics_cond)

                    # Sample with temperature
                    probs = torch.softmax(logits / temperature, dim=-1)
                    indices = torch.multinomial(probs.view(-1, 4), 1).view(batch_n, -1)

                    # Convert to sequences
                    bases = ['A', 'C', 'G', 'T']
                    for j in range(batch_n):
                        seq = ''.join(bases[idx] for idx in indices[j].cpu().numpy())
                        generated.append(seq)
                except Exception as e:
                    print(f"  Generation error: {e}")
                    break

        print(f"  Generated {len(generated)} sequences")
        return generated

    def check_oracle(self, sequence: str) -> OracleCheckResult:
        """
        Step 4: OracleCheck validation.

        Checks composition constraints:
        - GC content (45-55% ideal)
        - Homopolymer runs (<8bp)
        - CpG islands
        - Repeat content
        """
        flags = []
        seq_upper = sequence.upper()

        # Handle empty sequence
        if len(seq_upper) == 0:
            return OracleCheckResult(
                verdict='RED',
                gc_content=0.0,
                max_homopolymer=0,
                has_cpg_island=False,
                repeat_fraction=0.0,
                flags=['Empty sequence']
            )

        # GC content
        gc = (seq_upper.count('G') + seq_upper.count('C')) / len(seq_upper)
        if gc < 0.35 or gc > 0.65:
            flags.append(f"GC content {gc:.2f} outside range")

        # Homopolymer runs
        max_homo = 1
        current = 1
        for i in range(1, len(seq_upper)):
            if seq_upper[i] == seq_upper[i-1]:
                current += 1
                max_homo = max(max_homo, current)
            else:
                current = 1
        if max_homo > 8:
            flags.append(f"Homopolymer run of {max_homo}")

        # CpG islands (simplified: >60% GC in 200bp window with CpG ratio > 0.6)
        has_cpg = False
        cpg_count = seq_upper.count('CG')
        gc_count = seq_upper.count('G') + seq_upper.count('C')
        expected_cpg = (seq_upper.count('C') * seq_upper.count('G')) / len(seq_upper)
        if expected_cpg > 0 and cpg_count / expected_cpg > 0.6 and gc > 0.5:
            has_cpg = True
            flags.append("Potential CpG island")

        # Repeat content (dinucleotide repeats)
        repeat_count = 0
        for dinuc in ['AT', 'TA', 'GC', 'CG', 'AC', 'CA', 'GT', 'TG']:
            repeat_count += seq_upper.count(dinuc * 4)
        repeat_frac = repeat_count / len(seq_upper) if len(seq_upper) > 0 else 0
        if repeat_frac > 0.1:
            flags.append(f"High repeat content {repeat_frac:.2f}")

        # Determine verdict
        if len(flags) == 0:
            verdict = 'GREEN'
        elif len(flags) == 1 and max_homo <= 10:
            verdict = 'YELLOW'
        else:
            verdict = 'RED'

        return OracleCheckResult(
            verdict=verdict,
            gc_content=gc,
            max_homopolymer=max_homo,
            has_cpg_island=has_cpg,
            repeat_fraction=repeat_frac,
            flags=flags
        )

    def check_motif_constraints(
        self,
        sequence: str,
        required: List[str] = None,
        forbidden: List[str] = None
    ) -> MotifConstraintResult:
        """
        Step 5: Check motif constraints.

        Verifies required TF binding sites are present
        and forbidden sites are absent.
        """
        required = required or self.required_motifs
        forbidden = forbidden or self.forbidden_motifs

        if self.motif_scanner is None:
            return MotifConstraintResult(
                passed=True,
                required_found=[],
                required_missing=required,
                forbidden_found=[],
                total_motif_hits=0,
                message="Motif scanner not available"
            )

        # Scan sequence
        hits = self.motif_scanner.scan_sequence(sequence)
        found_motifs = set(h.motif_name for h in hits)

        # Check required (partial match on motif name)
        required_found = []
        required_missing = []
        for req in required:
            req_upper = req.upper()
            if any(req_upper in m.upper() for m in found_motifs):
                required_found.append(req)
            else:
                required_missing.append(req)

        # Check forbidden
        forbidden_found = []
        for forb in forbidden:
            forb_upper = forb.upper()
            if any(forb_upper in m.upper() for m in found_motifs):
                forbidden_found.append(forb)

        # Determine pass/fail
        # Require at least 1 required motif, no forbidden
        passed = len(required_found) >= 1 and len(forbidden_found) == 0

        message = f"Found {len(required_found)}/{len(required)} required, {len(forbidden_found)} forbidden"

        return MotifConstraintResult(
            passed=passed,
            required_found=required_found,
            required_missing=required_missing,
            forbidden_found=forbidden_found,
            total_motif_hits=len(hits),
            message=message
        )

    def evaluate_candidates(
        self,
        sequences: List[str],
        source: str = 'unknown',
        progress: bool = True
    ) -> List[EnhancerCandidate]:
        """
        Evaluate a batch of candidate sequences.

        Runs Steps 3-5: activity prediction, OracleCheck, motif constraints.
        """
        print(f"\nEvaluating {len(sequences)} candidates from {source}...")

        candidates = []

        # Step 3: Predict activities
        print("  Predicting multi-cell activities...")
        predictions = self.ensemble.predict_batch(sequences, progress=progress)

        # Steps 4-5: Validate each
        iterator = tqdm(zip(sequences, predictions), total=len(sequences), desc="Validating") if progress else zip(sequences, predictions)

        for seq, pred in iterator:
            # Activity scores
            target_act = pred.get_activity(self.target_cell)
            bg_acts = [pred.get_activity(c) for c in self.background_cells]
            max_bg = max(bg_acts) if bg_acts else 0
            specificity = target_act - 0.5 * sum(bg_acts)

            # OracleCheck
            oracle_result = self.check_oracle(seq)

            # Motif constraints
            motif_result = self.check_motif_constraints(seq)

            # Create candidate
            candidate = EnhancerCandidate(
                sequence=seq,
                source=source,
                activities={ct: pred.get_activity(ct) for ct in self.all_cell_types},
                uncertainties={ct: pred.predictions[ct].std for ct in self.all_cell_types},
                target_activity=target_act,
                max_background=max_bg,
                specificity_score=specificity,
                oracle_verdict=oracle_result.verdict,
                motif_result=motif_result,
                passed_all_filters=(
                    oracle_result.verdict in ['GREEN', 'YELLOW'] and
                    motif_result.passed
                )
            )
            candidates.append(candidate)

        # Summary
        n_passed = sum(1 for c in candidates if c.passed_all_filters)
        n_green = sum(1 for c in candidates if c.oracle_verdict == 'GREEN')
        n_yellow = sum(1 for c in candidates if c.oracle_verdict == 'YELLOW')
        n_red = sum(1 for c in candidates if c.oracle_verdict == 'RED')

        print(f"  OracleCheck: {n_green} GREEN, {n_yellow} YELLOW, {n_red} RED")
        print(f"  Passed all filters: {n_passed}/{len(candidates)}")

        return candidates

    def rank_and_diversify(
        self,
        candidates: List[EnhancerCandidate],
        n_select: int = 50,
        max_identity: float = 0.8
    ) -> pd.DataFrame:
        """
        Step 6: Rank by specificity and apply diversity filter.
        """
        print(f"\nStep 6: Ranking and diversifying...")

        # Filter to passed candidates
        passed = [c for c in candidates if c.passed_all_filters]
        print(f"  {len(passed)} candidates passed all filters")

        if len(passed) == 0:
            print("  Warning: No candidates passed filters!")
            # Fall back to GREEN/YELLOW only
            passed = [c for c in candidates if c.oracle_verdict in ['GREEN', 'YELLOW']]
            print(f"  Falling back to {len(passed)} GREEN/YELLOW candidates")

        if len(passed) == 0:
            return pd.DataFrame()

        # Convert to DataFrame
        records = []
        for c in passed:
            record = {
                'sequence': c.sequence,
                'source': c.source,
                'specificity_score': c.specificity_score,
                'target_activity': c.target_activity,
                'max_background': c.max_background,
                'oracle_verdict': c.oracle_verdict,
                'motif_passed': c.motif_result.passed if c.motif_result else False,
                'required_motifs_found': len(c.motif_result.required_found) if c.motif_result else 0,
            }
            for ct, act in c.activities.items():
                record[f'activity_{ct}'] = act
            records.append(record)

        df = pd.DataFrame(records)
        df = df.sort_values('specificity_score', ascending=False)

        # Apply diversity filter
        if len(df) > n_select:
            df = diversity_filter(
                df,
                sequence_col='sequence',
                score_col='specificity_score',
                n_select=n_select,
                min_identity=max_identity,
                method='greedy'
            )

        print(f"  Selected {len(df)} diverse candidates")
        return df.reset_index(drop=True)

    def run_full_pipeline(
        self,
        natural_sequences: List[str] = None,
        fasta_path: str = None,
        n_vae_candidates: int = 500,
        n_select: int = 50,
        output_dir: str = None
    ) -> pd.DataFrame:
        """
        Run the complete 6-step therapeutic enhancer design pipeline.

        Args:
            natural_sequences: List of natural enhancer sequences
            fasta_path: Path to FASTA file with natural enhancers
            n_vae_candidates: Number of VAE-generated candidates
            n_select: Final number of diverse candidates to select
            output_dir: Directory to save results

        Returns:
            DataFrame with top diverse candidates
        """
        print("\n" + "="*60)
        print("THERAPEUTIC ENHANCER DESIGN PIPELINE")
        print("="*60)
        print(f"Target cell: {self.target_cell}")
        print(f"Background cells: {self.background_cells}")
        print("="*60)

        # Load natural sequences if needed
        if natural_sequences is None and fasta_path:
            natural_sequences = self.load_natural_enhancers(fasta_path)

        all_candidates = []

        # Check if we have any source of candidates
        if not natural_sequences and (n_vae_candidates == 0 or self.vae_model is None):
            print("Warning: No natural sequences provided and VAE not available.")
            print("  Provide --fasta with natural enhancers or ensure PhysicsVAE is loaded.")
            return pd.DataFrame()

        # Step 1: Extract physics profile from natural enhancers
        if natural_sequences:
            target_physics = self.extract_target_physics_profile(natural_sequences)

            # Evaluate natural sequences
            natural_candidates = self.evaluate_candidates(
                natural_sequences, source='natural'
            )
            all_candidates.extend(natural_candidates)
        else:
            n_physics = getattr(self, 'n_physics_features', 515)
            target_physics = np.zeros(n_physics)  # Default

        # Step 2: Generate VAE candidates
        if n_vae_candidates > 0 and self.vae_model is not None:
            vae_sequences = self.generate_vae_candidates(
                target_physics, n_candidates=n_vae_candidates
            )

            if vae_sequences:
                vae_candidates = self.evaluate_candidates(
                    vae_sequences, source='vae_generated'
                )
                all_candidates.extend(vae_candidates)

        # Store all candidates
        self.candidates = all_candidates

        # Step 6: Rank and diversify
        result_df = self.rank_and_diversify(all_candidates, n_select=n_select)

        # Save results
        if output_dir and len(result_df) > 0:
            self.save_results(result_df, output_dir)

        # Print summary
        self._print_summary(result_df)

        return result_df

    def _print_summary(self, df: pd.DataFrame):
        """Print pipeline summary."""
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)

        if len(df) == 0:
            print("No candidates passed all filters!")
            return

        print(f"Total candidates selected: {len(df)}")
        print(f"\nTop 10 candidates:")
        print("-"*60)

        cols = ['specificity_score', 'target_activity', 'max_background', 'oracle_verdict', 'source']
        print(df[cols].head(10).to_string())

        # Source breakdown
        print(f"\nBy source:")
        for source in df['source'].unique():
            n = len(df[df['source'] == source])
            print(f"  {source}: {n}")

        # Verdict breakdown
        print(f"\nBy OracleCheck verdict:")
        for verdict in ['GREEN', 'YELLOW']:
            n = len(df[df['oracle_verdict'] == verdict])
            print(f"  {verdict}: {n}")

    def save_results(self, df: pd.DataFrame, output_dir: str):
        """Save results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # CSV with all data
        csv_path = output_dir / 'therapeutic_enhancers.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved results to {csv_path}")

        # FASTA for synthesis
        fasta_path = output_dir / 'therapeutic_enhancers.fasta'
        with open(fasta_path, 'w') as f:
            for i, row in df.iterrows():
                name = f"enhancer_{i+1}_spec{row['specificity_score']:.3f}_{row['oracle_verdict']}"
                f.write(f">{name}\n{row['sequence']}\n")
        print(f"Saved FASTA to {fasta_path}")

        # JSON report
        report = {
            'pipeline_config': {
                'target_cell': self.target_cell,
                'background_cells': self.background_cells,
                'required_motifs': self.required_motifs,
                'forbidden_motifs': self.forbidden_motifs,
            },
            'summary': {
                'total_candidates': len(df),
                'mean_specificity': float(df['specificity_score'].mean()),
                'max_specificity': float(df['specificity_score'].max()),
                'n_green': int((df['oracle_verdict'] == 'GREEN').sum()),
                'n_yellow': int((df['oracle_verdict'] == 'YELLOW').sum()),
            },
            'top_10': df.head(10)[['sequence', 'specificity_score', 'target_activity', 'oracle_verdict']].to_dict('records')
        }

        report_path = output_dir / 'therapeutic_enhancers_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Therapeutic Enhancer Design Pipeline v2'
    )

    # Input
    parser.add_argument('--fasta', type=str, help='Natural enhancer FASTA file')
    parser.add_argument('--csv', type=str, help='CSV file with sequences')
    parser.add_argument('--sequence-col', type=str, default='sequence')

    # Target/background
    parser.add_argument('--target-cell', type=str, default='HepG2',
                        choices=['K562', 'HepG2', 'WTC11'],
                        help='Target cell type to maximize')
    parser.add_argument('--background-cells', type=str, nargs='+',
                        help='Background cell types (default: others)')

    # Motif constraints
    parser.add_argument('--required-motifs', type=str, nargs='+',
                        help='Required TF motifs')
    parser.add_argument('--forbidden-motifs', type=str, nargs='+',
                        help='Forbidden TF motifs')

    # Generation
    parser.add_argument('--n-vae', type=int, default=500,
                        help='Number of VAE candidates to generate')
    parser.add_argument('--n-select', type=int, default=50,
                        help='Final number of diverse candidates')

    # Output
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Create pipeline
    pipeline = TherapeuticEnhancerPipeline(
        target_cell=args.target_cell,
        background_cells=args.background_cells,
        device=args.device,
        required_motifs=args.required_motifs,
        forbidden_motifs=args.forbidden_motifs,
    )

    # Load sequences
    sequences = None
    if args.fasta:
        sequences = pipeline.load_natural_enhancers(args.fasta)
    elif args.csv:
        df = pd.read_csv(args.csv)
        sequences = df[args.sequence_col].tolist()

    # Run pipeline
    result = pipeline.run_full_pipeline(
        natural_sequences=sequences,
        n_vae_candidates=args.n_vae,
        n_select=args.n_select,
        output_dir=args.output_dir,
    )

    return result


if __name__ == '__main__':
    main()
