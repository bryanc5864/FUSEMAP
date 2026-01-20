#!/usr/bin/env python3
"""
Run Full OracleCheck Validation Protocol

Generates designed sequences using REAL trained models and runs comprehensive analysis:
- Set 1: 100 sequences via unconstrained optimization (ISM)
- Set 2: 100 sequences via physics-constrained optimization (PINCSD)
- Set 3: 100 sequences via PhysicsVAE generation
- Set 4: 100 sequences via transfer-guided optimization

Total: 400 sequences for comprehensive analysis.

Usage:
    python run_validation_protocol.py --cell-type K562
    python run_validation_protocol.py --cell-type K562 --sets 1 2  # Only run sets 1 and 2
    python run_validation_protocol.py --cell-type S2 --transfer-source K562  # Transfer from human
"""

import sys
import os
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add paths
FUSEMAP_DIR = Path("/home/bcheng/sequence_optimization/FUSEMAP")
sys.path.insert(0, str(FUSEMAP_DIR))
sys.path.insert(0, str(FUSEMAP_DIR / "results" / "optimization"))

# Import components
from oracle_check.config import OracleCheckConfig, Verdict
from oracle_check.validation_runner import ValidationProtocolRunner, ValidationReport, create_runner
from oracle_check.statistical_comparisons import StatisticalComparator, generate_comparison_report


@dataclass
class ProtocolConfig:
    """Configuration for the validation protocol."""
    cell_type: str = "K562"
    output_dir: Path = FUSEMAP_DIR / "oracle_check" / "protocol_results"

    # Generation counts (100 per set = 400 total)
    n_unconstrained: int = 100
    n_constrained: int = 100
    n_vae: int = 100
    n_transfer: int = 100

    # Optimization settings
    n_iterations: int = 100  # ISM iterations per sequence
    n_random_starts: int = 50  # 50 random + 50 natural = 100 total
    n_natural_starts: int = 50

    # PhysicsVAE settings
    n_samples_per_profile: int = 10
    top_percentile: float = 0.10

    # Transfer settings
    transfer_source: str = "K562"
    transfer_target: str = "S2"

    # Device
    device: str = "cuda"

    # Which sets to run
    run_sets: List[int] = None

    def __post_init__(self):
        if self.run_sets is None:
            self.run_sets = [1, 2, 3, 4]
        self.output_dir = Path(self.output_dir)


class SequenceGenerator:
    """Generates sequences using various methods."""

    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.device = config.device

        # Will be loaded on demand
        self.cadence_model = None
        self.physics_model = None
        self.vae_model = None
        self._generate_module = None  # Cache the generate module for PhysicsVAE

    def load_cadence_model(self):
        """Load REAL CADENCE model for the cell type."""
        if self.cadence_model is not None:
            return

        try:
            from real_model_loader import load_model_suite
            print(f"\n{'='*60}")
            print(f"Loading REAL Model Suite for {self.config.cell_type}")
            print(f"{'='*60}")
            self.model_suite = load_model_suite(self.config.cell_type, device=self.device)
            self.cadence_model = self.model_suite.cadence  # ModelSuite has .cadence, not .cadence_model
            print(f"Loaded CADENCE model for {self.config.cell_type}")
        except Exception as e:
            print(f"ERROR: Could not load CADENCE model: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load real CADENCE model for {self.config.cell_type}")

    def load_vae_model(self):
        """Load PhysicsVAE model."""
        if self.vae_model is not None:
            return

        try:
            vae_base = FUSEMAP_DIR / "physics" / "PhysicsVAE"
            vae_dir = vae_base / "runs"

            # Find model for cell type
            cell_type = self.config.cell_type
            model_dirs = list(vae_dir.glob(f"{cell_type}_*"))

            if not model_dirs:
                # Try multi models
                model_dirs = list(vae_dir.glob("multi_human_*"))

            if model_dirs:
                model_dir = sorted(model_dirs)[-1]  # Most recent

                # Check both possible locations for checkpoint
                checkpoint = model_dir / "best_model.pt"
                if not checkpoint.exists():
                    checkpoint = model_dir / "checkpoints" / "best_model.pt"

                if checkpoint.exists():
                    # Add PhysicsVAE directory to path so generate.py can import its submodules
                    vae_path = str(vae_base)
                    if vae_path not in sys.path:
                        sys.path.insert(0, vae_path)

                    # Save current directory and change to PhysicsVAE directory
                    import os
                    import importlib
                    import importlib.util
                    orig_cwd = os.getcwd()
                    try:
                        os.chdir(vae_base)

                        # Clear any cached imports
                        for mod_name in list(sys.modules.keys()):
                            if mod_name.startswith('models') or mod_name == 'generate':
                                del sys.modules[mod_name]

                        # Load generate module from file
                        spec = importlib.util.spec_from_file_location("generate", vae_base / "generate.py")
                        generate_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(generate_module)

                        self.vae_model = generate_module.load_model(str(checkpoint), torch.device(self.device))
                        self._generate_module = generate_module  # Cache for later use
                        print(f"Loaded PhysicsVAE from {checkpoint}")
                    finally:
                        os.chdir(orig_cwd)
                else:
                    print(f"VAE checkpoint not found in: {model_dir}")
            else:
                print(f"No VAE model directories found for {cell_type}")
        except Exception as e:
            print(f"Warning: Could not load PhysicsVAE: {e}")
            import traceback
            traceback.print_exc()

    def load_natural_sequences(self, n_low: int = 100) -> List[str]:
        """Load natural low-activity sequences as starting points."""
        data_paths = {
            'K562': FUSEMAP_DIR / "data/human_mpra/human_mpra/K562_clean.tsv",
            'HepG2': FUSEMAP_DIR / "data/human_mpra/human_mpra/HepG2_clean.tsv",
            'WTC11': FUSEMAP_DIR / "data/human_mpra/human_mpra/WTC11_clean.tsv",
            'S2': FUSEMAP_DIR / "physics/output/S2_train_descriptors.tsv",
        }

        col_maps = {
            'K562': ('seq', 'mean_value'),
            'HepG2': ('seq', 'mean_value'),
            'WTC11': ('seq', 'mean_value'),
            'S2': ('sequence', 'Dev_log2_enrichment'),
        }

        path = data_paths.get(self.config.cell_type)
        if path is None or not path.exists():
            return []

        df = pd.read_csv(path, sep='\t')
        seq_col, act_col = col_maps.get(self.config.cell_type, ('seq', 'mean_value'))

        # Get low activity sequences
        df_sorted = df.sort_values(act_col)
        low_seqs = df_sorted.head(n_low)[seq_col].tolist()

        return low_seqs

    def load_high_activity_physics(self, top_percentile: float = 0.10) -> np.ndarray:
        """Load physics features from top-activity natural sequences."""
        # Load physics data
        physics_dir = FUSEMAP_DIR / "physics" / "output"

        cell_type = self.config.cell_type
        if cell_type in ['K562', 'HepG2', 'WTC11']:
            physics_file = physics_dir / f"{cell_type}_train_descriptors.tsv"
        else:
            physics_file = physics_dir / f"{cell_type}_train_descriptors.tsv"

        if not physics_file.exists():
            print(f"Physics file not found: {physics_file}")
            return None

        df = pd.read_csv(physics_file, sep='\t')

        # Find activity column
        act_cols = [c for c in df.columns if 'activity' in c.lower() or 'enrichment' in c.lower() or 'mean_value' in c.lower()]
        if not act_cols:
            act_cols = [df.columns[-1]]  # Use last column

        # Get top percentile
        threshold = df[act_cols[0]].quantile(1 - top_percentile)
        high_df = df[df[act_cols[0]] >= threshold]

        # Get physics features (exclude non-physics columns)
        exclude_cols = ['sequence', 'seq', 'activity', 'enrichment', 'mean_value', 'Unnamed']
        physics_cols = [c for c in df.columns if not any(ex in c.lower() for ex in exclude_cols)]

        # Convert to float, handling any non-numeric values
        physics_data = high_df[physics_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)

        # Match the number of features to what the VAE expects (515)
        # The VAE was trained with 515 features
        if physics_data.shape[1] > 515:
            physics_data = physics_data[:, :515]
        elif physics_data.shape[1] < 515:
            # Pad with zeros if needed
            padding = np.zeros((physics_data.shape[0], 515 - physics_data.shape[1]), dtype=np.float32)
            physics_data = np.concatenate([physics_data, padding], axis=1)

        return physics_data

    def generate_random_sequences(self, n: int, length: int = 230) -> List[str]:
        """Generate random DNA sequences."""
        import random
        sequences = []
        for _ in range(n):
            seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=length))
            sequences.append(seq)
        return sequences

    def run_ism_optimization(
        self,
        starting_sequences: List[str],
        n_iterations: int = 100,
        use_physics_constraint: bool = False,
    ) -> List[str]:
        """Run ISM or PINCSD optimization."""
        self.load_cadence_model()

        if self.cadence_model is None:
            print("No CADENCE model available, returning starting sequences")
            return starting_sequences

        try:
            print("  Loading optimization modules...")
            from real_optimization import (
                ISMOptimizer, PINCSDOptimizer,
                ISMConfig, PINCSDConfig
            )
            print("  Optimization modules loaded")

            if use_physics_constraint:
                print("  Creating PINCSD optimizer...")
                config = PINCSDConfig(
                    max_iterations=n_iterations,
                    early_stop_patience=15,
                )
                optimizer = PINCSDOptimizer(self.model_suite, config=config)
            else:
                print(f"  Creating ISM optimizer (max_iter={n_iterations})...")
                config = ISMConfig(
                    max_iterations=n_iterations,
                    early_stop_patience=10,
                )
                optimizer = ISMOptimizer(self.model_suite, config=config)
            print("  Optimizer ready")

            import time
            optimized = []
            for i, seq in enumerate(starting_sequences):
                print(f"  [{i+1}/{len(starting_sequences)}] Optimizing sequence (len={len(seq)})...")
                start_time = time.time()

                # optimize() returns List[SequenceScores], last element is best
                history = optimizer.optimize(seq, verbose=True)
                elapsed = time.time() - start_time

                if history:
                    initial_score = history[0].activity
                    final_score = history[-1].activity
                    print(f"      Done in {elapsed:.1f}s | {len(history)} iters | activity: {initial_score:.4f} -> {final_score:.4f}")
                    optimized.append(history[-1].sequence)
                else:
                    print(f"      No improvement found ({elapsed:.1f}s)")
                    optimized.append(seq)

            return optimized

        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return starting_sequences

    def generate_vae_sequences(
        self,
        n_sequences: int = 500,
        n_per_profile: int = 20,
    ) -> List[str]:
        """Generate sequences using PhysicsVAE."""
        self.load_vae_model()

        if self.vae_model is None:
            print("No VAE model available")
            return []

        # Load high-activity physics profiles
        physics = self.load_high_activity_physics(self.config.top_percentile)
        if physics is None or len(physics) == 0:
            print("No physics profiles available")
            return []

        # Sample profiles
        n_profiles = n_sequences // n_per_profile
        profile_indices = np.random.choice(len(physics), min(n_profiles, len(physics)), replace=False)

        try:
            # Use cached generate module from load_vae_model
            if self._generate_module is None:
                print("Generate module not loaded")
                return []

            generate_from_physics = self._generate_module.generate_from_physics

            all_sequences = []
            for i, idx in enumerate(profile_indices):
                if (i + 1) % 5 == 0:
                    print(f"  Generating from profile {i+1}/{len(profile_indices)}")
                profile = torch.tensor(physics[idx:idx+1], dtype=torch.float32)
                seqs = generate_from_physics(
                    self.vae_model,
                    profile,
                    n_samples=n_per_profile,
                    temperature=1.0,
                    device=torch.device(self.device),
                )
                all_sequences.extend(seqs)

            return all_sequences[:n_sequences]

        except Exception as e:
            print(f"VAE generation failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def generate_transfer_sequences(
        self,
        n_sequences: int = 100,
    ) -> List[str]:
        """Generate sequences using cross-species transfer."""
        # Load target species sequences
        target_seqs = self.load_natural_sequences(n_sequences)

        if not target_seqs:
            print("No target sequences available for transfer")
            return []

        # Use physics-bridge transfer with PINCSD
        # The physics features are universal, so we use source-trained probe
        # to guide optimization on target sequences

        try:
            # Load source model
            from real_model_loader import load_model_suite
            source_suite = load_model_suite(self.config.transfer_source, device=self.device)

            from real_optimization import PINCSDOptimizer, PINCSDConfig
            config = PINCSDConfig(
                max_iterations=self.config.n_iterations,
                early_stop_patience=15,
                weights={'activity': 1.0, 'uncertainty': 0.3, 'naturality': 0.3, 'physics': 0.3},
            )
            optimizer = PINCSDOptimizer(source_suite, config=config)

            optimized = []
            for i, seq in enumerate(target_seqs[:n_sequences]):
                if (i + 1) % 10 == 0:
                    print(f"  Transfer-optimizing sequence {i+1}/{n_sequences}")

                # optimize() returns List[SequenceScores], last element is best
                history = optimizer.optimize(seq, verbose=False)
                if history:
                    optimized.append(history[-1].sequence)
                else:
                    optimized.append(seq)

            return optimized

        except Exception as e:
            print(f"Transfer generation failed: {e}")
            import traceback
            traceback.print_exc()
            return target_seqs[:n_sequences]


def run_protocol(config: ProtocolConfig) -> Dict[str, ValidationReport]:
    """Run the full validation protocol."""

    print("=" * 70)
    print("ORACLECHECK VALIDATION PROTOCOL (REAL MODELS)")
    print("=" * 70)
    print(f"Cell Type: {config.cell_type}")
    print(f"Output: {config.output_dir}")
    print(f"Sets to run: {config.run_sets}")
    print()
    print("Configuration:")
    print(f"  Set 1 (ISM): {config.n_unconstrained} sequences")
    print(f"  Set 2 (PINCSD): {config.n_constrained} sequences")
    print(f"  Set 3 (VAE): {config.n_vae} sequences")
    print(f"  Set 4 (Transfer): {config.n_transfer} sequences")
    print(f"  Starting points: {config.n_random_starts} random + {config.n_natural_starts} natural")
    print(f"  ISM iterations: {config.n_iterations}")
    print(f"  Device: {config.device}")
    print("=" * 70)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir / f"{config.cell_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = SequenceGenerator(config)

    # Initialize validation runner
    runner = create_runner(cell_type=config.cell_type)

    # Store all generated sequences
    all_sequences = {}
    all_reports = {}

    # ==========================================================================
    # SET 1: Unconstrained Optimization (ISM)
    # ==========================================================================
    if 1 in config.run_sets:
        print("\n" + "=" * 70)
        print("SET 1: Unconstrained Optimization (ISM)")
        print("=" * 70)

        # Get starting sequences
        random_starts = generator.generate_random_sequences(config.n_random_starts)
        natural_starts = generator.load_natural_sequences(config.n_natural_starts)
        starting = random_starts + natural_starts

        print(f"Starting points: {len(random_starts)} random + {len(natural_starts)} natural")

        # Run ISM optimization
        print("Running ISM optimization...")
        set1_sequences = generator.run_ism_optimization(
            starting[:config.n_unconstrained],
            n_iterations=config.n_iterations,
            use_physics_constraint=False,
        )

        all_sequences["unconstrained_ism"] = set1_sequences

        # Validate
        print("Validating Set 1...")
        report1 = runner.validate_sequences(
            set1_sequences,
            generation_method="unconstrained_ism",
            run_motif=True,
            run_rc=True,
        )
        all_reports["unconstrained_ism"] = report1

        print(f"Set 1 Results: GREEN={report1.n_green}, YELLOW={report1.n_yellow}, RED={report1.n_red}")

    # ==========================================================================
    # SET 2: Physics-Constrained Optimization (PINCSD)
    # ==========================================================================
    if 2 in config.run_sets:
        print("\n" + "=" * 70)
        print("SET 2: Physics-Constrained Optimization (PINCSD)")
        print("=" * 70)

        # Use same starting points as Set 1
        random_starts = generator.generate_random_sequences(config.n_random_starts)
        natural_starts = generator.load_natural_sequences(config.n_natural_starts)
        starting = random_starts + natural_starts

        print(f"Starting points: {len(starting)}")

        # Run PINCSD optimization
        print("Running PINCSD optimization...")
        set2_sequences = generator.run_ism_optimization(
            starting[:config.n_constrained],
            n_iterations=config.n_iterations,
            use_physics_constraint=True,
        )

        all_sequences["physics_constrained_pincsd"] = set2_sequences

        # Validate
        print("Validating Set 2...")
        report2 = runner.validate_sequences(
            set2_sequences,
            generation_method="physics_constrained_pincsd",
            run_motif=True,
            run_rc=True,
        )
        all_reports["physics_constrained_pincsd"] = report2

        print(f"Set 2 Results: GREEN={report2.n_green}, YELLOW={report2.n_yellow}, RED={report2.n_red}")

    # ==========================================================================
    # SET 3: PhysicsVAE Generation
    # ==========================================================================
    if 3 in config.run_sets:
        print("\n" + "=" * 70)
        print("SET 3: PhysicsVAE Generation")
        print("=" * 70)

        print(f"Generating {config.n_vae} sequences from PhysicsVAE...")
        set3_sequences = generator.generate_vae_sequences(
            n_sequences=config.n_vae,
            n_per_profile=config.n_samples_per_profile,
        )

        if set3_sequences:
            all_sequences["physics_vae"] = set3_sequences

            # Validate
            print("Validating Set 3...")
            report3 = runner.validate_sequences(
                set3_sequences,
                generation_method="physics_vae",
                run_motif=True,
                run_rc=True,
            )
            all_reports["physics_vae"] = report3

            print(f"Set 3 Results: GREEN={report3.n_green}, YELLOW={report3.n_yellow}, RED={report3.n_red}")
        else:
            print("Skipping Set 3 - no VAE sequences generated")

    # ==========================================================================
    # SET 4: Transfer-Guided Generation
    # ==========================================================================
    if 4 in config.run_sets:
        print("\n" + "=" * 70)
        print("SET 4: Transfer-Guided Generation")
        print(f"Source: {config.transfer_source} → Target: {config.transfer_target}")
        print("=" * 70)

        # Update config for transfer target
        transfer_generator = SequenceGenerator(ProtocolConfig(
            cell_type=config.transfer_target,
            transfer_source=config.transfer_source,
            n_iterations=config.n_iterations,
            device=config.device,
        ))

        print(f"Generating {config.n_transfer} sequences via transfer...")
        set4_sequences = transfer_generator.generate_transfer_sequences(
            n_sequences=config.n_transfer,
        )

        if set4_sequences:
            all_sequences["transfer_guided"] = set4_sequences

            # Validate using target species runner
            target_runner = create_runner(cell_type=config.transfer_target)

            print("Validating Set 4...")
            report4 = target_runner.validate_sequences(
                set4_sequences,
                generation_method="transfer_guided",
                run_motif=True,
                run_rc=True,
            )
            all_reports["transfer_guided"] = report4

            print(f"Set 4 Results: GREEN={report4.n_green}, YELLOW={report4.n_yellow}, RED={report4.n_red}")
        else:
            print("Skipping Set 4 - no transfer sequences generated")

    # ==========================================================================
    # STATISTICAL COMPARISONS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISONS")
    print("=" * 70)

    comparison_results = generate_comparison_report(
        all_reports,
        output_path=output_dir / "comparison_report.json",
    )

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save sequences
    for method, sequences in all_sequences.items():
        seq_file = output_dir / f"{method}_sequences.txt"
        with open(seq_file, 'w') as f:
            for seq in sequences:
                f.write(seq + '\n')
        print(f"Saved {len(sequences)} sequences to {seq_file}")

    # Save validation reports
    for method, report in all_reports.items():
        report_file = output_dir / f"{method}_report.json"
        with open(report_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

    # Save summary
    summary = {
        "config": {
            "cell_type": config.cell_type,
            "n_unconstrained": config.n_unconstrained,
            "n_constrained": config.n_constrained,
            "n_vae": config.n_vae,
            "n_transfer": config.n_transfer,
            "n_iterations": config.n_iterations,
        },
        "results": {
            method: {
                "n_sequences": report.n_sequences,
                "green_rate": report.green_rate,
                "yellow_rate": report.yellow_rate,
                "red_rate": report.red_rate,
                "mean_activity": report.mean_activity,
                "physics_pass_rate": report.physics_pass_rate,
            }
            for method, report in all_reports.items()
        },
        "timestamp": timestamp,
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {output_dir}")

    # ==========================================================================
    # PRINT FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<30} {'N':>6} {'GREEN':>8} {'YELLOW':>8} {'RED':>8} {'Activity':>10}")
    print("-" * 70)
    for method, report in all_reports.items():
        print(f"{method:<30} {report.n_sequences:>6} {report.green_rate:>7.1%} {report.yellow_rate:>7.1%} {report.red_rate:>7.1%} {report.mean_activity:>10.3f}")

    total_seqs = sum(r.n_sequences for r in all_reports.values())
    total_green = sum(r.n_green for r in all_reports.values())
    print("-" * 70)
    print(f"{'TOTAL':<30} {total_seqs:>6} {total_green/total_seqs:>7.1%}")

    return all_reports


def main():
    parser = argparse.ArgumentParser(description="Run OracleCheck Validation Protocol")

    parser.add_argument("--cell-type", type=str, default="K562",
                        choices=["K562", "HepG2", "WTC11", "S2"],
                        help="Cell type for validation")

    parser.add_argument("--sets", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Which generation sets to run (1-4)")

    parser.add_argument("--n-iterations", type=int, default=500,
                        help="Optimization iterations")

    parser.add_argument("--transfer-source", type=str, default="K562",
                        help="Source species for transfer (Set 4)")

    parser.add_argument("--transfer-target", type=str, default="S2",
                        help="Target species for transfer (Set 4)")

    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    # Reduced counts for testing
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with reduced sequence counts")

    parser.add_argument("--n-random", type=int, default=None,
                        help="Number of random starting sequences")
    parser.add_argument("--n-natural", type=int, default=None,
                        help="Number of natural starting sequences")

    args = parser.parse_args()

    # Create config
    config = ProtocolConfig(
        cell_type=args.cell_type,
        run_sets=args.sets,
        n_iterations=args.n_iterations,
        transfer_source=args.transfer_source,
        transfer_target=args.transfer_target,
        device=args.device,
    )

    if args.output:
        config.output_dir = Path(args.output)

    if args.quick:
        config.n_unconstrained = 20
        config.n_constrained = 20
        config.n_vae = 50
        config.n_transfer = 10
        config.n_iterations = 100
        config.n_random_starts = 10
        config.n_natural_starts = 10

    # Override with explicit counts if provided
    if args.n_random is not None:
        config.n_random_starts = args.n_random
    if args.n_natural is not None:
        config.n_natural_starts = args.n_natural

    # Run protocol
    run_protocol(config)


if __name__ == "__main__":
    main()
