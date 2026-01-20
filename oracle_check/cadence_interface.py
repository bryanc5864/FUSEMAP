"""
CADENCE Interface for OracleCheck

Provides interface to PLACE-calibrated CADENCE models for:
- Activity predictions with uncertainty
- Feature extraction for OOD detection
- Ensemble predictions
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import DATASET_CATALOG, ModelConfig
from training.models import MultiSpeciesCADENCE


@dataclass
class CADENCEPrediction:
    """Container for CADENCE model predictions."""
    mean: np.ndarray           # Predicted activity [batch]
    aleatoric_std: np.ndarray  # Aleatoric uncertainty [batch]
    epistemic_std: Optional[np.ndarray] = None  # Epistemic uncertainty [batch]
    conformal_lower: Optional[np.ndarray] = None  # Conformal lower bound [batch]
    conformal_upper: Optional[np.ndarray] = None  # Conformal upper bound [batch]
    features: Optional[np.ndarray] = None  # Backbone features [batch, dim]
    ood_score: Optional[np.ndarray] = None  # OOD score from kNN [batch]


class CADENCEInterface:
    """
    Interface to PLACE-calibrated CADENCE models.

    Supports:
    - Single model predictions with PLACE uncertainty
    - Multi-model ensemble predictions
    - Feature extraction for OOD detection
    - kNN-based OOD scoring
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        load_knn: bool = True,
    ):
        """
        Initialize CADENCE interface.

        Args:
            model_path: Path to model_with_place.pt or directory containing it
            device: Device to use for inference
            load_knn: Whether to load kNN model for OOD detection
        """
        self.device = torch.device(device)
        model_path = Path(model_path)

        # Handle both file and directory paths
        if model_path.is_dir():
            model_file = model_path / "model_with_place.pt"
            self.model_dir = model_path
        else:
            model_file = model_path
            self.model_dir = model_path.parent

        if not model_file.exists():
            # Fall back to original model
            model_file = self.model_dir / "original_model.pt"
            if not model_file.exists():
                raise FileNotFoundError(f"No model found in {self.model_dir}")

        # Load checkpoint
        checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)

        # Extract config and create model
        self.config = checkpoint.get("config", {})
        self.model_config = self._extract_model_config(checkpoint)
        self.dataset_names = self._extract_datasets(checkpoint)

        # Infer architecture from state dict
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        arch_params = self._infer_architecture(state_dict)

        # Create model
        self.model = MultiSpeciesCADENCE(
            config=self.model_config,
            dataset_names=self.dataset_names,
            n_species=arch_params["n_species"],
            n_kingdoms=arch_params["n_kingdoms"],
            n_celltypes=arch_params["n_celltypes"],
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Load PLACE calibration data
        self.place_data = None
        self.knn_model = None
        self._load_place_data(load_knn)

        # Build index mappings
        self._build_index_mappings()

    def _extract_model_config(self, checkpoint: dict) -> ModelConfig:
        """Extract ModelConfig from checkpoint."""
        config = checkpoint.get("config", {})
        if isinstance(config, dict):
            model_dict = config.get("model", config)
            return ModelConfig(**model_dict) if isinstance(model_dict, dict) else model_dict
        elif hasattr(config, "model"):
            return config.model
        else:
            return config

    def _extract_datasets(self, checkpoint: dict) -> List[str]:
        """Extract dataset names from checkpoint."""
        config = checkpoint.get("config", {})
        if isinstance(config, dict):
            return config.get("datasets", ["unknown"])
        elif hasattr(config, "datasets"):
            return config.datasets
        return ["unknown"]

    def _infer_architecture(self, state_dict: dict) -> dict:
        """Infer architecture parameters from state dict."""
        n_celltypes = 1
        if "celltype_embed.weight" in state_dict:
            n_celltypes = state_dict["celltype_embed.weight"].shape[0]

        n_species = 1
        if "species_embed.weight" in state_dict:
            n_species = state_dict["species_embed.weight"].shape[0]

        n_kingdoms = 1
        if "kingdom_embed.weight" in state_dict:
            n_kingdoms = state_dict["kingdom_embed.weight"].shape[0]

        use_species_stem = any("species_stems" in k for k in state_dict.keys())
        use_kingdom_stem = any("kingdom_stems" in k for k in state_dict.keys())

        return {
            "n_celltypes": n_celltypes,
            "n_species": n_species,
            "n_kingdoms": n_kingdoms,
            "use_species_stem": use_species_stem,
            "use_kingdom_stem": use_kingdom_stem,
        }

    def _load_place_data(self, load_knn: bool):
        """Load PLACE calibration data and kNN model."""
        # Load calibration data
        place_file = self.model_dir / "place_calibration_data.npz"
        if place_file.exists():
            self.place_data = np.load(place_file)

        # Load metadata
        metadata_file = self.model_dir / "place_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.place_metadata = json.load(f)
        else:
            self.place_metadata = {}

        # Load kNN model
        if load_knn:
            knn_file = self.model_dir / "place_knn_model.pkl"
            if knn_file.exists():
                with open(knn_file, "rb") as f:
                    self.knn_model = pickle.load(f)

    def _build_index_mappings(self):
        """Build species/kingdom/celltype index mappings from dataset names."""
        self.species_to_idx = {}
        self.kingdom_to_idx = {}
        self.celltype_to_idx = {}

        for ds_name in self.dataset_names:
            if ds_name in DATASET_CATALOG:
                info = DATASET_CATALOG[ds_name]
                if info.species not in self.species_to_idx:
                    self.species_to_idx[info.species] = len(self.species_to_idx)
                if info.kingdom not in self.kingdom_to_idx:
                    self.kingdom_to_idx[info.kingdom] = len(self.kingdom_to_idx)
                ct = info.cell_type or "unknown"
                if ct not in self.celltype_to_idx:
                    self.celltype_to_idx[ct] = len(self.celltype_to_idx)

    def _get_indices_for_dataset(self, dataset_name: str) -> Tuple[int, int, int]:
        """Get species, kingdom, celltype indices for a dataset."""
        if dataset_name in DATASET_CATALOG:
            info = DATASET_CATALOG[dataset_name]
            species_idx = self.species_to_idx.get(info.species, 0)
            kingdom_idx = self.kingdom_to_idx.get(info.kingdom, 0)
            celltype_idx = self.celltype_to_idx.get(info.cell_type or "unknown", 0)
            return species_idx, kingdom_idx, celltype_idx
        return 0, 0, 0

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Convert DNA sequence to one-hot encoding."""
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}
        seq_upper = sequence.upper()
        indices = [mapping.get(base, 0) for base in seq_upper]
        one_hot = torch.zeros(4, len(sequence))
        for i, idx in enumerate(indices):
            one_hot[idx, i] = 1.0
        return one_hot

    def encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """Encode multiple sequences."""
        encoded = [self.encode_sequence(seq) for seq in sequences]
        return torch.stack(encoded)

    @torch.no_grad()
    def predict(
        self,
        sequences: Union[str, List[str]],
        dataset_name: str = "encode4_k562",
        return_features: bool = True,
        compute_ood: bool = True,
    ) -> CADENCEPrediction:
        """
        Get predictions with uncertainty for sequences.

        Args:
            sequences: Single sequence or list of sequences
            dataset_name: Dataset name for head selection
            return_features: Whether to return backbone features
            compute_ood: Whether to compute OOD scores

        Returns:
            CADENCEPrediction with mean, uncertainties, and optionally features/OOD
        """
        # Handle single sequence
        if isinstance(sequences, str):
            sequences = [sequences]

        # Encode sequences
        seq_tensor = self.encode_sequences(sequences).to(self.device)

        # Get indices
        species_idx, kingdom_idx, celltype_idx = self._get_indices_for_dataset(dataset_name)
        batch_size = len(sequences)

        species_tensor = torch.full((batch_size,), species_idx, dtype=torch.long, device=self.device)
        kingdom_tensor = torch.full((batch_size,), kingdom_idx, dtype=torch.long, device=self.device)

        # Get features
        features = None
        if return_features or compute_ood:
            features = self.model._backbone_forward(
                seq_tensor,
                species_idx=species_tensor,
                kingdom_idx=kingdom_tensor,
            ).cpu().numpy()

        # Get predictions
        outputs = self.model(
            seq_tensor,
            species_idx=species_tensor,
            kingdom_idx=kingdom_tensor,
            dataset_names=[dataset_name] * batch_size,
        )

        # Find the right head
        head_names = self.model.dataset_to_heads.get(dataset_name, [])
        if not head_names:
            raise ValueError(f"No heads found for dataset: {dataset_name}")

        head_name = head_names[0]  # Use first head
        head_output = outputs.get(head_name, {})

        mean = head_output.get("mean", torch.zeros(batch_size)).cpu().numpy()

        # Get aleatoric uncertainty (from Gaussian NLL head)
        log_var = head_output.get("log_var", None)
        if log_var is not None:
            aleatoric_std = np.sqrt(np.exp(log_var.cpu().numpy()))
        else:
            # Use PLACE noise variance
            noise_var = self.place_metadata.get("noise_variance", 1.0)
            aleatoric_std = np.full(batch_size, np.sqrt(noise_var))

        # Compute OOD score using kNN
        ood_score = None
        if compute_ood and self.knn_model is not None and features is not None:
            distances, _ = self.knn_model.kneighbors(features)
            ood_score = distances.mean(axis=1)  # Average distance to k neighbors

        # Compute epistemic uncertainty (placeholder - would need ensemble)
        epistemic_std = None

        # Compute conformal bounds (using PLACE residual distribution)
        conformal_lower = None
        conformal_upper = None
        if self.place_metadata:
            residual_std = self.place_metadata.get("residual_std", 1.0)
            alpha = 0.1  # 90% confidence
            z = 1.645  # z-score for 90% CI
            conformal_lower = mean - z * residual_std
            conformal_upper = mean + z * residual_std

        return CADENCEPrediction(
            mean=mean,
            aleatoric_std=aleatoric_std,
            epistemic_std=epistemic_std,
            conformal_lower=conformal_lower,
            conformal_upper=conformal_upper,
            features=features if return_features else None,
            ood_score=ood_score,
        )

    @torch.no_grad()
    def get_features(
        self,
        sequences: Union[str, List[str]],
        dataset_name: str = "encode4_k562",
    ) -> np.ndarray:
        """
        Get backbone features for sequences (for OOD detection).

        Args:
            sequences: Single sequence or list of sequences
            dataset_name: Dataset name for routing

        Returns:
            Features array [batch, feature_dim]
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        seq_tensor = self.encode_sequences(sequences).to(self.device)

        species_idx, kingdom_idx, _ = self._get_indices_for_dataset(dataset_name)
        batch_size = len(sequences)

        species_tensor = torch.full((batch_size,), species_idx, dtype=torch.long, device=self.device)
        kingdom_tensor = torch.full((batch_size,), kingdom_idx, dtype=torch.long, device=self.device)

        features = self.model._backbone_forward(
            seq_tensor,
            species_idx=species_tensor,
            kingdom_idx=kingdom_tensor,
        )

        return features.cpu().numpy()

    def get_training_features(self) -> Optional[np.ndarray]:
        """Get training features from PLACE calibration data."""
        if self.place_data is not None and "features" in self.place_data:
            return self.place_data["features"]
        return None
