"""
PhysInformer Interface for OracleCheck

Provides interface to PhysInformer models for physics feature prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import sys

# Add PhysInformer to path
PHYSINFORMER_PATH = Path("/home/bcheng/sequence_optimization/mainproject/PhysiFormer/physpreprocess/PhysInformer")
sys.path.insert(0, str(PHYSINFORMER_PATH))


@dataclass
class PhysicsFeatures:
    """Container for physics feature predictions."""
    features: np.ndarray           # All features [batch, n_features]
    feature_names: List[str]       # Feature names
    family_features: Dict[str, np.ndarray]  # Features grouped by family


class PhysInformerInterface:
    """
    Interface to PhysInformer physics prediction models.

    Provides:
    - Physics feature prediction from DNA sequences
    - Feature grouping by physics family
    - Denormalization to original scale
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cuda",
    ):
        """
        Initialize PhysInformer interface.

        Args:
            checkpoint_path: Path to PhysInformer checkpoint
            device: Device for inference
        """
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)

        # Load model
        self._load_model()

        # Define physics families
        self._define_families()

    def _load_model(self):
        """Load PhysInformer model from checkpoint."""
        # Import PhysInformer components
        try:
            from physics_aware_model import create_physics_aware_model
        except ImportError:
            # Try relative import
            sys.path.insert(0, str(PHYSINFORMER_PATH))
            from physics_aware_model import create_physics_aware_model

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # Infer cell type and feature count
        self.cell_type = checkpoint.get("cell_type", "K562")

        # Get feature count from state dict
        if "model_state_dict" in checkpoint:
            feature_keys = [
                k for k in checkpoint["model_state_dict"].keys()
                if "property_heads.feature_" in k
            ]
            n_features = len(set(k.split(".")[1] for k in feature_keys))
        else:
            n_features = None

        # Create and load model
        self.model = create_physics_aware_model(
            self.cell_type,
            n_descriptor_features=n_features
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load normalization stats
        if "normalization_stats" in checkpoint:
            self.norm_stats = checkpoint["normalization_stats"]
            for key in ["desc_mean", "desc_std"]:
                if key in self.norm_stats:
                    self.norm_stats[key] = self.norm_stats[key].to(self.device)
        else:
            self.norm_stats = None

        # Get feature names
        self.feature_names = checkpoint.get("feature_names", None)
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(n_features or 100)]

    def _define_families(self):
        """Define physics feature families for grouping."""
        # Map feature name prefixes to families
        self.family_prefixes = {
            "thermodynamics": ["thermo_", "dG_", "Tm_", "stability_"],
            "shape": ["mgw_", "prot_", "roll_", "helt_", "slide_", "shift_", "tilt_", "twist_"],
            "bending": ["bend_", "stiff_", "curv_", "flex_"],
            "stacking": ["stack_"],
            "sidd_g4": ["sidd_", "g4_", "quadruplex_"],
            "entropy": ["entropy_", "info_"],
        }

        # Build feature to family mapping
        self.feature_to_family = {}
        for feature in self.feature_names:
            for family, prefixes in self.family_prefixes.items():
                if any(feature.lower().startswith(p) for p in prefixes):
                    self.feature_to_family[feature] = family
                    break
            else:
                self.feature_to_family[feature] = "other"

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Convert DNA sequence to index encoding for PhysInformer."""
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        seq_upper = sequence.upper()
        indices = [mapping.get(base, 4) for base in seq_upper]
        return torch.tensor(indices, dtype=torch.long)

    def encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """Encode multiple sequences."""
        encoded = [self.encode_sequence(seq) for seq in sequences]
        return torch.stack(encoded)

    @torch.no_grad()
    def predict(
        self,
        sequences: Union[str, List[str]],
        denormalize: bool = True,
    ) -> PhysicsFeatures:
        """
        Predict physics features for sequences.

        Args:
            sequences: Single sequence or list of sequences
            denormalize: Whether to denormalize to original scale

        Returns:
            PhysicsFeatures with all features and family groupings
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        # Encode sequences
        seq_tensor = self.encode_sequences(sequences).to(self.device)

        # Predict
        outputs = self.model(seq_tensor)

        # Reconstruct feature tensor
        n_features = len([
            k for k in outputs.keys()
            if k.startswith("feature_") and k.endswith("_mean")
        ])
        batch_size = seq_tensor.shape[0]

        features = torch.zeros(batch_size, n_features, device=self.device)
        for i in range(n_features):
            key = f"feature_{i}_mean"
            if key in outputs:
                features[:, i] = outputs[key]

        # Denormalize
        if denormalize and self.norm_stats is not None:
            features = features * self.norm_stats["desc_std"] + self.norm_stats["desc_mean"]

        features_np = features.cpu().numpy()

        # Group by family
        family_features = {}
        for family in set(self.feature_to_family.values()):
            family_indices = [
                i for i, name in enumerate(self.feature_names)
                if self.feature_to_family.get(name) == family
            ]
            if family_indices:
                family_features[family] = features_np[:, family_indices]

        return PhysicsFeatures(
            features=features_np,
            feature_names=self.feature_names,
            family_features=family_features,
        )

    def get_family_names(self) -> List[str]:
        """Get list of physics family names."""
        return list(set(self.feature_to_family.values()))

    def get_features_for_family(self, family: str) -> List[str]:
        """Get feature names belonging to a family."""
        return [
            name for name, fam in self.feature_to_family.items()
            if fam == family
        ]


class MultiCellTypePhysInformer:
    """
    Wrapper for multiple cell-type-specific PhysInformer models.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        cell_types: List[str] = ["K562", "HepG2", "WTC11"],
        device: str = "cuda",
    ):
        """
        Initialize multi-cell-type PhysInformer.

        Args:
            checkpoint_dir: Directory containing PhysInformer runs
            cell_types: List of cell types to load
            device: Device for inference
        """
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cell_types = cell_types
        self.models = {}

        # Model run name mapping
        self.run_names = {
            "K562": "K562_20250829_095741",
            "HepG2": "HepG2_20250829_095749",
            "WTC11": "WTC11_20250829_095738",
        }

        # Load models
        for cell_type in cell_types:
            run_name = self.run_names.get(cell_type)
            if run_name:
                checkpoint = self.checkpoint_dir / run_name / "best_model.pt"
                if checkpoint.exists():
                    self.models[cell_type] = PhysInformerInterface(
                        checkpoint, device=device
                    )

    def predict(
        self,
        sequences: Union[str, List[str]],
        cell_type: str = "K562",
        denormalize: bool = True,
    ) -> PhysicsFeatures:
        """
        Predict physics features for a specific cell type.

        Args:
            sequences: DNA sequences
            cell_type: Cell type model to use
            denormalize: Whether to denormalize features

        Returns:
            PhysicsFeatures
        """
        if cell_type not in self.models:
            raise ValueError(f"No model for cell type: {cell_type}")
        return self.models[cell_type].predict(sequences, denormalize=denormalize)

    def get_available_cell_types(self) -> List[str]:
        """Get list of available cell types."""
        return list(self.models.keys())
