"""
FUSEMAP Applications Module.

High-impact applications using FUSEMAP's physics-informed models:

1. Disease Variant Interpretation Pipeline
   - Interpret disease-associated variants using CADENCE + PhysInformer
   - Predict ref vs alt activity changes with uncertainty
   - Identify mechanistically important physics features

2. Therapeutic Enhancer Design Pipeline
   - Design cell-type specific therapeutic enhancers
   - Multi-cell CADENCE ensemble for specificity scoring
   - Diversity-aware selection for experimental validation

Usage:
    from applications import DiseaseVariantPipeline, TherapeuticEnhancerPipeline

    # Disease variant interpretation
    pipeline = DiseaseVariantPipeline(
        reference_genome='path/to/ref.fa',
        cadence_checkpoint='path/to/cadence.pt',
        physinformer_checkpoint='path/to/physinformer.pt'
    )
    results = pipeline.run_pipeline(vcf_path='variants.vcf')

    # Therapeutic enhancer design
    pipeline = TherapeuticEnhancerPipeline(
        cell_types=['K562', 'HepG2', 'WTC11']
    )
    results = pipeline.run_pipeline(
        fasta_path='candidates.fa',
        target_cell='HepG2'
    )
"""

from .disease_variant_pipeline import DiseaseVariantPipeline
from .therapeutic_enhancer_pipeline import TherapeuticEnhancerPipeline

__all__ = [
    'DiseaseVariantPipeline',
    'TherapeuticEnhancerPipeline',
]
