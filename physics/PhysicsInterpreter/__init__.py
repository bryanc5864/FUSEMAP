"""
PhysicsInterpreter: Attribution and Mechanistic Analysis

Decomposes model predictions through the physics pathway, providing
mechanistic interpretability beyond standard sequence attribution methods.

Components:
    - IntegratedGradients: Sequence attribution via integrated gradients
    - PhysicsAttribution: Physics feature attribution and family contributions
    - MediationAnalysis: Causal mediation of sequence→physics→activity
    - LandscapeAnalysis: Physics-activity relationship mapping with SHAP
"""

from .config import InterpreterConfig, MODEL_PATHS
from .integrated_gradients import IntegratedGradients
from .physics_attribution import PhysicsAttributor
from .mediation_analysis import MediationAnalyzer
from .landscape_analysis import LandscapeAnalyzer

__version__ = '1.0.0'
__all__ = [
    'InterpreterConfig',
    'MODEL_PATHS',
    'IntegratedGradients',
    'PhysicsAttributor',
    'MediationAnalyzer',
    'LandscapeAnalyzer'
]
