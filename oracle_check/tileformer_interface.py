"""
TileFormer Interface for OracleCheck

Provides interface to TileFormer for electrostatics prediction.
TileFormer is universal (not cell-type specific).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import sys

# Add TileFormer to path
TILEFORMER_PATH = Path("/home/bcheng/sequence_optimization/mainproject/PhysiFormer/physpreprocess/TileFormer_model")
sys.path.insert(0, str(TILEFORMER_PATH / "models"))
sys.path.insert(0, str(TILEFORMER_PATH))


@dataclass
class ElectrostaticsFeatures:
    """Container for electrostatics predictions."""
    thermo_dG: np.ndarray  # Thermodynamic free energy prediction [batch]
    # Can be extended with more electrostatics features


class TileFormerInterface:
    """
    Interface to TileFormer for electrostatics prediction.

    TileFormer predicts global sequence-level thermodynamic properties
    from DNA sequence. It is trained on all cell types and is universal.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
    ):
        """
        Initialize TileFormer interface.

        Args:
            checkpoint_path: Path to TileFormer checkpoint (optional)
            device: Device for inference
        """
        self.device = torch.device(device)

        # Default checkpoint path
        if checkpoint_path is None:
            checkpoint_path = (
                TILEFORMER_PATH / "checkpoints" / "best_model.pt"
            )
        self.checkpoint_path = Path(checkpoint_path)

        # Load model
        self._load_model()

    def _load_model(self):
        """Load TileFormer model."""
        try:
            from tileformer_architecture import TileFormerSingle, create_tileformer_single
        except ImportError:
            # Model not available, use placeholder
            self.model = None
            self.available = False
            return

        # Check if checkpoint exists
        if not self.checkpoint_path.exists():
            # Try to find any checkpoint in the directory
            checkpoint_dir = self.checkpoint_path.parent
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pt"))
                if checkpoints:
                    self.checkpoint_path = checkpoints[0]

        if not self.checkpoint_path.exists():
            self.model = None
            self.available = False
            return

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # Create model
        self.model = create_tileformer_single(target_feature="thermo_dG_p25")

        # Load state dict
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        self.available = True

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Convert DNA sequence to index encoding."""
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        seq_upper = sequence.upper()
        indices = [mapping.get(base, 4) for base in seq_upper]
        return torch.tensor(indices, dtype=torch.long)

    def encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """Encode multiple sequences."""
        max_len = max(len(seq) for seq in sequences)
        encoded = []
        for seq in sequences:
            indices = self.encode_sequence(seq)
            # Pad to max length
            if len(indices) < max_len:
                padding = torch.full((max_len - len(indices),), 4, dtype=torch.long)
                indices = torch.cat([indices, padding])
            encoded.append(indices)
        return torch.stack(encoded)

    @torch.no_grad()
    def predict(
        self,
        sequences: Union[str, List[str]],
    ) -> ElectrostaticsFeatures:
        """
        Predict electrostatics features for sequences.

        Args:
            sequences: Single sequence or list of sequences

        Returns:
            ElectrostaticsFeatures with thermodynamic predictions
        """
        if not self.available:
            # Return placeholder if model not available
            if isinstance(sequences, str):
                return ElectrostaticsFeatures(thermo_dG=np.array([0.0]))
            return ElectrostaticsFeatures(
                thermo_dG=np.zeros(len(sequences))
            )

        if isinstance(sequences, str):
            sequences = [sequences]

        # Encode sequences
        seq_tensor = self.encode_sequences(sequences).to(self.device)

        # Predict
        outputs = self.model(seq_tensor)

        # Extract predictions
        thermo_dG = outputs["target_feature"].cpu().numpy()

        return ElectrostaticsFeatures(thermo_dG=thermo_dG)

    def is_available(self) -> bool:
        """Check if TileFormer model is available."""
        return self.available
