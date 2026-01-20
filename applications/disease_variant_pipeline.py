#!/usr/bin/env python3
"""
Disease Variant Interpretation Pipeline.

End-to-end pipeline for interpreting disease-associated variants using
CADENCE activity predictions and PhysInformer physics features.

Pipeline Steps:
1. Load variants from VCF or DataFrame
2. Extract ref/alt sequences with flanking regions
3. Predict activity changes using CADENCE
4. Compute physics feature changes using PhysInformer
5. Score and rank variants by predicted effect
6. Generate interpretable reports
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import sys

# Add paths
FUSEMAP_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FUSEMAP_ROOT))
sys.path.insert(0, str(FUSEMAP_ROOT / 'applications/utils'))

# Handle both package import and direct script execution
try:
    from .utils.variant_extractor import VariantExtractor, Variant, VariantSequences
    from .utils.differential_analysis import (
        DifferentialAnalyzer, VariantEffect,
        CADENCEPredictor, PhysInformerPredictor
    )
except ImportError:
    from utils.variant_extractor import VariantExtractor, Variant, VariantSequences
    from utils.differential_analysis import (
        DifferentialAnalyzer, VariantEffect,
        CADENCEPredictor, PhysInformerPredictor
    )


class DiseaseVariantPipeline:
    """
    Complete pipeline for disease variant interpretation.
    """

    def __init__(
        self,
        reference_genome: str = None,
        cadence_checkpoint: str = None,
        physinformer_checkpoint: str = None,
        cell_type: str = 'K562',
        flank_size: int = 115,
        device: str = 'cuda'
    ):
        """
        Initialize the disease variant pipeline.

        Args:
            reference_genome: Path to reference genome FASTA
            cadence_checkpoint: Path to CADENCE model checkpoint
            physinformer_checkpoint: Path to PhysInformer checkpoint
            cell_type: Cell type for predictions
            flank_size: Flanking sequence size (total = 2*flank + variant)
            device: 'cuda' or 'cpu'
        """
        self.cell_type = cell_type
        self.flank_size = flank_size
        self.device = device

        # Initialize components
        self.extractor = VariantExtractor(
            reference_genome=reference_genome,
            flank_size=flank_size
        )

        self.analyzer = DifferentialAnalyzer(
            cadence_checkpoint=cadence_checkpoint,
            physinformer_checkpoint=physinformer_checkpoint,
            cell_type=cell_type,
            device=device
        )

        # Results storage
        self.variant_effects: List[VariantEffect] = []
        self.summary_df: pd.DataFrame = None

    def load_variants_from_vcf(
        self,
        vcf_path: str,
        max_variants: int = None,
        filter_pass_only: bool = True
    ) -> List[VariantSequences]:
        """Load and extract sequences for variants from VCF."""
        print(f"Loading variants from {vcf_path}...")
        variant_sequences = self.extractor.extract_from_vcf(
            vcf_path,
            max_variants=max_variants,
            filter_pass_only=filter_pass_only
        )
        print(f"Loaded {len(variant_sequences)} variants")
        return variant_sequences

    def load_variants_from_dataframe(
        self,
        df: pd.DataFrame,
        chrom_col: str = 'chrom',
        pos_col: str = 'pos',
        ref_col: str = 'ref',
        alt_col: str = 'alt',
        id_col: str = None
    ) -> List[VariantSequences]:
        """Load and extract sequences for variants from DataFrame."""
        print(f"Extracting sequences for {len(df)} variants...")
        variant_sequences = self.extractor.extract_from_dataframe(
            df,
            chrom_col=chrom_col,
            pos_col=pos_col,
            ref_col=ref_col,
            alt_col=alt_col,
            id_col=id_col
        )
        print(f"Extracted {len(variant_sequences)} variant sequences")
        return variant_sequences

    def analyze_variants(
        self,
        variant_sequences: List[VariantSequences],
        progress: bool = True
    ) -> List[VariantEffect]:
        """
        Perform differential analysis on all variants.

        Args:
            variant_sequences: List of extracted variant sequences
            progress: Show progress bar

        Returns:
            List of VariantEffect objects with predictions
        """
        print(f"Analyzing {len(variant_sequences)} variants...")
        self.variant_effects = self.analyzer.analyze_variants(
            variant_sequences, progress=progress
        )
        print(f"Successfully analyzed {len(self.variant_effects)} variants")
        return self.variant_effects

    def score_and_rank(
        self,
        min_zscore: float = None,
        effect_directions: List[str] = None,
        effect_magnitudes: List[str] = None
    ) -> pd.DataFrame:
        """
        Score and rank variants by predicted effect.

        Args:
            min_zscore: Minimum absolute z-score to include
            effect_directions: Filter by direction ['activating', 'repressing']
            effect_magnitudes: Filter by magnitude ['strong', 'moderate', 'weak']

        Returns:
            DataFrame sorted by absolute z-score
        """
        if not self.variant_effects:
            raise ValueError("No variant effects. Run analyze_variants first.")

        # Convert to DataFrame
        self.summary_df = self.analyzer.to_dataframe(self.variant_effects)

        # Apply filters
        if min_zscore is not None:
            self.summary_df = self.summary_df[
                abs(self.summary_df['delta_activity_zscore']) >= min_zscore
            ]

        if effect_directions:
            self.summary_df = self.summary_df[
                self.summary_df['effect_direction'].isin(effect_directions)
            ]

        if effect_magnitudes:
            self.summary_df = self.summary_df[
                self.summary_df['effect_magnitude'].isin(effect_magnitudes)
            ]

        # Sort by absolute z-score
        self.summary_df['abs_zscore'] = abs(self.summary_df['delta_activity_zscore'])
        self.summary_df = self.summary_df.sort_values('abs_zscore', ascending=False)
        self.summary_df = self.summary_df.drop(columns=['abs_zscore'])

        return self.summary_df.reset_index(drop=True)

    def get_top_variants(
        self,
        n: int = 10,
        direction: str = None
    ) -> pd.DataFrame:
        """
        Get top N most impactful variants.

        Args:
            n: Number of variants to return
            direction: 'activating' or 'repressing' (optional)
        """
        if self.summary_df is None:
            self.score_and_rank()

        df = self.summary_df.copy()

        if direction:
            df = df[df['effect_direction'] == direction]

        return df.head(n)

    def generate_report(
        self,
        output_path: str = None,
        include_physics: bool = True,
        top_n: int = 50
    ) -> Dict:
        """
        Generate comprehensive variant interpretation report.

        Args:
            output_path: Path to save JSON report
            include_physics: Include physics feature analysis
            top_n: Number of top variants to detail

        Returns:
            Report dictionary
        """
        if not self.variant_effects:
            raise ValueError("No variant effects. Run analyze_variants first.")

        if self.summary_df is None:
            self.score_and_rank()

        report = {
            'pipeline_info': {
                'cell_type': self.cell_type,
                'flank_size': self.flank_size,
                'n_variants_analyzed': len(self.variant_effects),
            },
            'summary_statistics': {
                'n_activating': int((self.summary_df['effect_direction'] == 'activating').sum()),
                'n_repressing': int((self.summary_df['effect_direction'] == 'repressing').sum()),
                'n_neutral': int((self.summary_df['effect_direction'] == 'neutral').sum()),
                'n_strong': int((self.summary_df['effect_magnitude'] == 'strong').sum()),
                'n_moderate': int((self.summary_df['effect_magnitude'] == 'moderate').sum()),
                'n_weak': int((self.summary_df['effect_magnitude'] == 'weak').sum()),
                'mean_delta_activity': float(self.summary_df['delta_activity'].mean()),
                'std_delta_activity': float(self.summary_df['delta_activity'].std()),
            },
            'top_variants': []
        }

        # Add details for top variants
        sorted_effects = sorted(
            self.variant_effects,
            key=lambda x: abs(x.delta_activity_zscore),
            reverse=True
        )

        for effect in sorted_effects[:top_n]:
            variant_detail = {
                'variant_id': effect.variant_id,
                'activity_ref': effect.activity_ref.mean,
                'activity_alt': effect.activity_alt.mean,
                'delta_activity': effect.delta_activity,
                'zscore': effect.delta_activity_zscore,
                'direction': effect.effect_direction,
                'magnitude': effect.effect_magnitude,
            }

            if include_physics:
                # Add top physics changes
                physics_changes = effect.get_physics_family_changes()
                variant_detail['physics_summary'] = {
                    family: {
                        'mean_change': data['mean_abs_change'],
                        'max_change': data['max_abs_change']
                    }
                    for family, data in physics_changes.items()
                }

            report['top_variants'].append(variant_detail)

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {output_path}")

        return report

    def save_results(
        self,
        output_dir: str,
        prefix: str = 'variant_analysis'
    ):
        """
        Save all results to files.

        Args:
            output_dir: Output directory
            prefix: File name prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary DataFrame
        if self.summary_df is not None:
            summary_path = output_dir / f'{prefix}_summary.csv'
            self.summary_df.to_csv(summary_path, index=False)
            print(f"Summary saved to {summary_path}")

        # Save full results
        results_path = output_dir / f'{prefix}_full_results.csv'
        full_records = []
        for effect in self.variant_effects:
            record = {
                'variant_id': effect.variant_id,
                'ref_sequence': effect.ref_sequence,
                'alt_sequence': effect.alt_sequence,
                'activity_ref': effect.activity_ref.mean,
                'activity_ref_std': effect.activity_ref.std,
                'activity_alt': effect.activity_alt.mean,
                'activity_alt_std': effect.activity_alt.std,
                'delta_activity': effect.delta_activity,
                'delta_activity_zscore': effect.delta_activity_zscore,
                'effect_direction': effect.effect_direction,
                'effect_magnitude': effect.effect_magnitude,
            }

            # Add physics deltas
            for name, value in effect.delta_physics.items():
                record[f'delta_{name}'] = value

            full_records.append(record)

        pd.DataFrame(full_records).to_csv(results_path, index=False)
        print(f"Full results saved to {results_path}")

        # Save report
        report_path = output_dir / f'{prefix}_report.json'
        self.generate_report(str(report_path))

    def run_pipeline(
        self,
        vcf_path: str = None,
        variants_df: pd.DataFrame = None,
        output_dir: str = None,
        max_variants: int = None
    ) -> pd.DataFrame:
        """
        Run complete pipeline end-to-end.

        Args:
            vcf_path: Path to VCF file (either this or variants_df required)
            variants_df: DataFrame with variants
            output_dir: Directory to save results
            max_variants: Maximum variants to process

        Returns:
            Summary DataFrame with ranked variants
        """
        # Load variants
        if vcf_path:
            variant_sequences = self.load_variants_from_vcf(
                vcf_path, max_variants=max_variants
            )
        elif variants_df is not None:
            variant_sequences = self.load_variants_from_dataframe(variants_df)
        else:
            raise ValueError("Either vcf_path or variants_df required")

        # Analyze
        self.analyze_variants(variant_sequences)

        # Score and rank
        summary = self.score_and_rank()

        # Save if output directory provided
        if output_dir:
            self.save_results(output_dir)

        return summary


def main():
    parser = argparse.ArgumentParser(
        description='Disease Variant Interpretation Pipeline'
    )

    # Input options
    parser.add_argument('--vcf', type=str, help='Path to VCF file')
    parser.add_argument('--variants-csv', type=str,
                        help='Path to CSV with variants (chrom, pos, ref, alt columns)')

    # Model options
    parser.add_argument('--reference', type=str,
                        help='Path to reference genome FASTA')
    parser.add_argument('--cadence-checkpoint', type=str,
                        help='Path to CADENCE model checkpoint')
    parser.add_argument('--physinformer-checkpoint', type=str,
                        help='Path to PhysInformer checkpoint')
    parser.add_argument('--cell-type', type=str, default='K562',
                        help='Cell type for predictions')

    # Processing options
    parser.add_argument('--flank-size', type=int, default=115,
                        help='Flanking sequence size')
    parser.add_argument('--max-variants', type=int,
                        help='Maximum variants to process')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    # Filter options
    parser.add_argument('--min-zscore', type=float,
                        help='Minimum absolute z-score')
    parser.add_argument('--direction', type=str,
                        choices=['activating', 'repressing'],
                        help='Filter by effect direction')

    # Output options
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--prefix', type=str, default='variant_analysis',
                        help='Output file prefix')

    args = parser.parse_args()

    # Validate inputs
    if not args.vcf and not args.variants_csv:
        parser.error("Either --vcf or --variants-csv required")

    # Create pipeline
    pipeline = DiseaseVariantPipeline(
        reference_genome=args.reference,
        cadence_checkpoint=args.cadence_checkpoint,
        physinformer_checkpoint=args.physinformer_checkpoint,
        cell_type=args.cell_type,
        flank_size=args.flank_size,
        device=args.device
    )

    # Load variants
    if args.vcf:
        variant_sequences = pipeline.load_variants_from_vcf(
            args.vcf,
            max_variants=args.max_variants
        )
    else:
        df = pd.read_csv(args.variants_csv)
        variant_sequences = pipeline.load_variants_from_dataframe(df)

    # Analyze
    pipeline.analyze_variants(variant_sequences)

    # Score and rank with filters
    effect_directions = [args.direction] if args.direction else None
    summary = pipeline.score_and_rank(
        min_zscore=args.min_zscore,
        effect_directions=effect_directions
    )

    # Save results
    pipeline.save_results(args.output_dir, prefix=args.prefix)

    # Print summary
    print("\n" + "="*60)
    print("VARIANT INTERPRETATION SUMMARY")
    print("="*60)
    print(f"Total variants analyzed: {len(pipeline.variant_effects)}")
    print(f"Variants passing filters: {len(summary)}")

    if len(summary) > 0:
        print(f"\nTop 10 most impactful variants:")
        print(summary[['variant_id', 'delta_activity', 'delta_activity_zscore',
                      'effect_direction', 'effect_magnitude']].head(10).to_string())


if __name__ == '__main__':
    main()
