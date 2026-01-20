"""
OracleCheck: Comprehensive Validation Protocol for Sequence Design Oracles

A purely in-silico validation program for oracles in sequence design loops,
providing comprehensive naturality evaluation across physics, syntax,
composition, and confidence dimensions.

Human datasets only: K562, HepG2, WTC11
"""

from .config import OracleCheckConfig, ValidationThresholds, Verdict, PHYSICS_FAMILIES
from .cadence_interface import CADENCEInterface, CADENCEPrediction
from .physics_interface import PhysInformerInterface, MultiCellTypePhysInformer, PhysicsFeatures
from .tileformer_interface import TileFormerInterface, ElectrostaticsFeatures
from .reference_panels import ReferencePanelBuilder, ReferencePanel, ReferenceDistribution
from .validators import (
    PhysicsValidator,
    CompositionValidator,
    ConfidenceValidator,
    MahalanobisValidator,
    OracleCheckValidator,
    PhysicsValidationResult,
    CompositionValidationResult,
    ConfidenceValidationResult,
    MahalanobisValidationResult,
)
from .scorecard import SequenceScorecard, BatchScorecard, ScorecardBuilder
from .protocol import OracleCheckProtocol, ValidationResult, run_human_validation

# New validation components
from .motif_validator import MotifValidator, MotifScanner, get_motif_validator
from .two_sample_tests import (
    MMDTest, EnergyDistanceTest, KSTest, KmerJSDivergence,
    BatchComparator, BatchComparisonResult,
)
from .rc_consistency import (
    RCConsistencyChecker, ISMFlipTest, FullRCValidator,
    RCConsistencyResult, ISMFlipTestResult,
)
from .validation_runner import ValidationProtocolRunner, ValidationReport, create_runner
from .statistical_comparisons import (
    StatisticalComparator, PairedComparisonResult, MethodComparisonSuite,
    generate_comparison_report,
)

__all__ = [
    # Config
    "OracleCheckConfig",
    "ValidationThresholds",
    "Verdict",
    "PHYSICS_FAMILIES",
    # Interfaces
    "CADENCEInterface",
    "CADENCEPrediction",
    "PhysInformerInterface",
    "MultiCellTypePhysInformer",
    "PhysicsFeatures",
    "TileFormerInterface",
    "ElectrostaticsFeatures",
    # Reference Panels
    "ReferencePanelBuilder",
    "ReferencePanel",
    "ReferenceDistribution",
    # Validators
    "PhysicsValidator",
    "CompositionValidator",
    "ConfidenceValidator",
    "MahalanobisValidator",
    "OracleCheckValidator",
    "PhysicsValidationResult",
    "CompositionValidationResult",
    "ConfidenceValidationResult",
    "MahalanobisValidationResult",
    # Scorecards
    "SequenceScorecard",
    "BatchScorecard",
    "ScorecardBuilder",
    # Protocol
    "OracleCheckProtocol",
    "ValidationResult",
    "run_human_validation",
    # Motif Validation
    "MotifValidator",
    "MotifScanner",
    "get_motif_validator",
    # Two-Sample Tests
    "MMDTest",
    "EnergyDistanceTest",
    "KSTest",
    "KmerJSDivergence",
    "BatchComparator",
    "BatchComparisonResult",
    # RC Consistency
    "RCConsistencyChecker",
    "ISMFlipTest",
    "FullRCValidator",
    "RCConsistencyResult",
    "ISMFlipTestResult",
    # Validation Runner
    "ValidationProtocolRunner",
    "ValidationReport",
    "create_runner",
    # Statistical Comparisons
    "StatisticalComparator",
    "PairedComparisonResult",
    "MethodComparisonSuite",
    "generate_comparison_report",
]
