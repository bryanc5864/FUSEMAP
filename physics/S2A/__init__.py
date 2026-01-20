"""
S2A: Zero-Shot Sequence-to-Activity Prediction System

Predicts regulatory activity for sequences from **unseen species** without
labeled data from that species.

Key Insight:
    Physics features (thermo, stiff, bend, entropy, advanced) are universal
    because DNA chemistry is identical across organisms. The physics→activity
    mapping is species-specific, but z-score outputs remove scale differences.

Architecture:
    DNA Sequence (any species)
            │
            ▼
    ┌─────────────────────┐
    │    PhysInformer     │  (existing - Seq → 500+ physics)
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Universal Features  │  Extract ~263 features
    │ EXCLUDE: pwm_*      │  (thermo, stiff, bend, entropy, advanced)
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  UniversalS2AHead   │  Ridge/ElasticNet trained on N-1 species
    │  (Physics→Activity) │  Z-score normalized
    └─────────┬───────────┘
              │
        ┌─────┴─────┬─────────────┐
        ▼           ▼             ▼
    ┌────────┐ ┌──────────┐ ┌─────────┐
    │Z-score │ │Calibrated│ │ Ranking │
    │(no cal)│ │(20-50 ex)│ │(order)  │
    └────────┘ └──────────┘ └─────────┘

Usage:
    from physics.S2A import (
        S2AConfig,
        UniversalFeatureExtractor,
        UniversalS2AHead,
        AffineCalibrator,
        UniversalS2ATrainer,
        S2APredictor,
        S2AEvaluator
    )

    # Train universal head
    trainer = UniversalS2ATrainer()
    head, extractor, results = trainer.train(
        ['K562', 'HepG2', 'arabidopsis_leaf', 'sorghum_leaf']
    )

    # Zero-shot prediction on new species
    predictor = S2APredictor(head, extractor)
    predictions = predictor.predict_zscore(X_new_species)

    # Calibrated prediction with 50 samples
    predictions = predictor.predict_calibrated(X_new, X_cal, y_cal)
"""

from .config import (
    S2AConfig,
    S2ADatasetConfig,
    S2AExperimentConfig,
    S2A_DATASETS,
    S2A_DATASET_GROUPS,
    S2A_EXPERIMENTS,
    DEFAULT_S2A_CONFIG,
    get_fusemap_root,
)

from .universal_features import (
    UniversalFeatureExtractor,
    UniversalFeatureStats,
    count_universal_vs_total_features,
)

from .universal_head import (
    UniversalS2AHead,
    EnsembleS2AHead,
    HeadPredictionResults,
    HeadEvaluationResults,
)

from .calibration import (
    AffineCalibrator,
    IsotonicCalibrator,
    CalibrationStats,
    CalibrationEvaluation,
    evaluate_calibration,
    calibration_curve_analysis,
    select_calibration_samples,
)

from .training import (
    UniversalS2ATrainer,
    TrainingResults,
    LeaveOneOutResults,
)

from .inference import (
    S2APredictor,
    S2APrediction,
    predict_from_descriptors_file,
    predict_for_dataset,
)

from .evaluation import (
    S2AEvaluator,
    DatasetEvaluation,
    FullEvaluationResults,
    compare_transfer_scenarios,
)

__all__ = [
    # Config
    'S2AConfig',
    'S2ADatasetConfig',
    'S2AExperimentConfig',
    'S2A_DATASETS',
    'S2A_DATASET_GROUPS',
    'S2A_EXPERIMENTS',
    'DEFAULT_S2A_CONFIG',
    'get_fusemap_root',

    # Feature extraction
    'UniversalFeatureExtractor',
    'UniversalFeatureStats',
    'count_universal_vs_total_features',

    # Head models
    'UniversalS2AHead',
    'EnsembleS2AHead',
    'HeadPredictionResults',
    'HeadEvaluationResults',

    # Calibration
    'AffineCalibrator',
    'IsotonicCalibrator',
    'CalibrationStats',
    'CalibrationEvaluation',
    'evaluate_calibration',
    'calibration_curve_analysis',
    'select_calibration_samples',

    # Training
    'UniversalS2ATrainer',
    'TrainingResults',
    'LeaveOneOutResults',

    # Inference
    'S2APredictor',
    'S2APrediction',
    'predict_from_descriptors_file',
    'predict_for_dataset',

    # Evaluation
    'S2AEvaluator',
    'DatasetEvaluation',
    'FullEvaluationResults',
    'compare_transfer_scenarios',
]

__version__ = '0.1.0'
