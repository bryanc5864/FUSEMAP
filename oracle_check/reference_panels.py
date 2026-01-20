"""
Reference Panel Builder for OracleCheck

Builds reference distributions from natural sequences for validation:
- Natural High-Performers: Top quartile of MPRA activity
- Background Natural: Random genomic tiles
- Training Index: kNN structure for OOD detection
- TileFormer Electrostatics: Position-wise electrostatic features
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import pickle
import json
import torch

from .config import OracleCheckConfig, PHYSICS_FAMILIES, ELECTROSTATICS_FEATURES


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class TileFormerInterface:
    """
    Interface to TileFormer for computing electrostatic features.

    TileFormer predicts 6 electrostatic outputs per 20bp window:
    - STD_PSI_MIN, STD_PSI_MAX, STD_PSI_MEAN (standard electrostatic potential)
    - ENH_PSI_MIN, ENH_PSI_MAX, ENH_PSI_MEAN (enhanced electrostatic potential)
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: str = "cuda",
        window_size: int = 20,
    ):
        """
        Initialize TileFormer interface.

        Args:
            checkpoint_path: Path to TileFormer checkpoint
            device: Device for inference
            window_size: Window size for sliding window (default 20bp)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.window_size = window_size
        self.model = None

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            self._load_model()

    def _load_model(self):
        """Load TileFormer model."""
        try:
            import sys
            tileformer_path = '/home/bcheng/sequence_optimization/FUSEMAP/physics/TileFormer/models'
            if tileformer_path not in sys.path:
                sys.path.insert(0, tileformer_path)
            from tileformer_architecture import TileFormerWithMetadata

            self.model = TileFormerWithMetadata(
                vocab_size=5,
                d_model=256,
                n_heads=8,
                n_layers=6,
                d_ff=1024,
                max_len=200,
                dropout=0.1,
                output_dim=6,
                predict_uncertainty=True,
                metadata_dim=3
            )

            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"TileFormer model loaded from {self.checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load TileFormer model: {e}")
            self.model = None

    def _sequence_to_indices(self, sequence: str) -> np.ndarray:
        """Convert DNA sequence to indices."""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        return np.array([mapping.get(b, 4) for b in sequence.upper()], dtype=np.int64)

    def _calculate_metadata(self, sequence: str) -> np.ndarray:
        """Calculate sequence metadata features."""
        seq = sequence.upper()
        seq_len = len(seq)

        gc_count = seq.count('G') + seq.count('C')
        gc_content = gc_count / seq_len if seq_len > 0 else 0.0

        cpg_count = seq.count('CG')
        cpg_density = cpg_count / (seq_len - 1) if seq_len > 1 else 0.0

        at_count = seq.count('A') + seq.count('T')
        minor_groove_score = 1.0 - (at_count / seq_len) if seq_len > 0 else 0.5

        return np.array([gc_content, cpg_density, minor_groove_score], dtype=np.float32)

    def _extract_windows(self, sequence: str, stride: int = 10) -> List[str]:
        """Extract sliding windows from sequence."""
        windows = []
        for i in range(0, len(sequence) - self.window_size + 1, stride):
            windows.append(sequence[i:i + self.window_size])
        return windows

    def compute_features(
        self,
        sequence: str,
        stride: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Compute electrostatic features for a sequence.

        Args:
            sequence: DNA sequence
            stride: Step size between windows

        Returns:
            Dictionary with electrostatic summary statistics
        """
        if self.model is None:
            return {}

        windows = self._extract_windows(sequence, stride)
        if not windows:
            return {}

        # Encode windows
        encoded = np.array([self._sequence_to_indices(w) for w in windows])
        metadata = np.array([self._calculate_metadata(w) for w in windows])

        with torch.no_grad():
            encoded_tensor = torch.from_numpy(encoded).long().to(self.device)
            metadata_tensor = torch.from_numpy(metadata).float().to(self.device)

            output = self.model(encoded_tensor, metadata_tensor)
            predictions = output['psi'].cpu().numpy()

        # predictions shape: [n_windows, 6]
        # Compute summary statistics
        features = {
            'electrostatic_std_psi_mean': float(np.mean(predictions[:, 2])),
            'electrostatic_std_psi_std': float(np.std(predictions[:, 2])),
            'electrostatic_std_psi_min': float(np.min(predictions[:, 0])),
            'electrostatic_std_psi_max': float(np.max(predictions[:, 1])),
            'electrostatic_enh_psi_mean': float(np.mean(predictions[:, 5])),
            'electrostatic_enh_psi_std': float(np.std(predictions[:, 5])),
            'electrostatic_enh_psi_min': float(np.min(predictions[:, 3])),
            'electrostatic_enh_psi_max': float(np.max(predictions[:, 4])),
            'electrostatic_raw': predictions,  # Full window-wise predictions
        }

        return features

    def compute_batch_features(
        self,
        sequences: List[str],
        stride: int = 10,
        batch_size: int = 512,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Compute features for a batch of sequences.

        Args:
            sequences: List of DNA sequences
            stride: Step size between windows
            batch_size: Batch size for inference

        Returns:
            List of feature dictionaries
        """
        if self.model is None:
            return [{} for _ in sequences]

        # Collect all windows
        all_windows = []
        window_counts = []

        for seq in sequences:
            windows = self._extract_windows(seq, stride)
            if not windows:
                # Handle short sequences
                if len(seq) < self.window_size:
                    padded = seq + 'N' * (self.window_size - len(seq))
                    windows = [padded]
            all_windows.extend(windows)
            window_counts.append(len(windows))

        # Encode all windows
        encoded = np.array([self._sequence_to_indices(w) for w in all_windows], dtype=np.int64)
        metadata = np.array([self._calculate_metadata(w) for w in all_windows], dtype=np.float32)

        # Predict in batches
        all_predictions = []
        with torch.no_grad():
            for i in range(0, len(encoded), batch_size):
                batch = encoded[i:i + batch_size]
                batch_meta = metadata[i:i + batch_size]

                encoded_tensor = torch.from_numpy(batch).long().to(self.device)
                metadata_tensor = torch.from_numpy(batch_meta).float().to(self.device)

                output = self.model(encoded_tensor, metadata_tensor)
                all_predictions.append(output['psi'].cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)

        # Split back to individual sequences
        results = []
        start_idx = 0
        for count in window_counts:
            pred = all_predictions[start_idx:start_idx + count]
            start_idx += count

            features = {
                'electrostatic_std_psi_mean': float(np.mean(pred[:, 2])),
                'electrostatic_std_psi_std': float(np.std(pred[:, 2])),
                'electrostatic_std_psi_min': float(np.min(pred[:, 0])),
                'electrostatic_std_psi_max': float(np.max(pred[:, 1])),
                'electrostatic_enh_psi_mean': float(np.mean(pred[:, 5])),
                'electrostatic_enh_psi_std': float(np.std(pred[:, 5])),
                'electrostatic_enh_psi_min': float(np.min(pred[:, 3])),
                'electrostatic_enh_psi_max': float(np.max(pred[:, 4])),
                'electrostatic_raw': pred,
            }
            results.append(features)

        return results


@dataclass
class ReferenceDistribution:
    """Statistics for a reference distribution."""
    mean: float
    std: float
    percentiles: Dict[str, float]  # p5, p25, p50, p75, p95
    values: Optional[np.ndarray] = None  # Raw values (optional, for saving)


@dataclass
class ReferencePanel:
    """
    Complete reference panel for validation.

    Contains distributions and statistics for:
    - Physics features (per family)
    - Composition metrics
    - Activity values
    """

    cell_type: str

    # Physics distributions per family
    physics_distributions: Dict[str, Dict[str, ReferenceDistribution]] = field(
        default_factory=dict
    )

    # Composition distributions
    gc_distribution: Optional[ReferenceDistribution] = None
    cpg_distribution: Optional[ReferenceDistribution] = None
    entropy_distribution: Optional[ReferenceDistribution] = None

    # Activity distribution (for high performers)
    activity_distribution: Optional[ReferenceDistribution] = None

    # Electrostatics distributions (from TileFormer)
    electrostatics_distributions: Dict[str, ReferenceDistribution] = field(
        default_factory=dict
    )

    # Training features for OOD detection
    training_features_mean: Optional[np.ndarray] = None
    training_features_cov: Optional[np.ndarray] = None
    knn_model: Optional[NearestNeighbors] = None

    # GMM for naturality scoring
    naturality_gmm: Optional[GaussianMixture] = None

    # Metadata
    n_samples: int = 0
    n_high_performers: int = 0

    def save(self, path: Union[str, Path]):
        """Save reference panel to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save distributions as JSON-compatible dict
        data = {
            "cell_type": self.cell_type,
            "n_samples": self.n_samples,
            "n_high_performers": self.n_high_performers,
            "physics_distributions": self._distributions_to_dict(self.physics_distributions),
        }

        if self.gc_distribution:
            data["gc_distribution"] = self._dist_to_dict(self.gc_distribution)
        if self.cpg_distribution:
            data["cpg_distribution"] = self._dist_to_dict(self.cpg_distribution)
        if self.entropy_distribution:
            data["entropy_distribution"] = self._dist_to_dict(self.entropy_distribution)
        if self.activity_distribution:
            data["activity_distribution"] = self._dist_to_dict(self.activity_distribution)

        if self.electrostatics_distributions:
            data["electrostatics_distributions"] = {
                k: self._dist_to_dict(v) for k, v in self.electrostatics_distributions.items()
            }

        with open(path / "reference_panel.json", "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

        # Save numpy arrays
        if self.training_features_mean is not None:
            np.save(path / "training_features_mean.npy", self.training_features_mean)
        if self.training_features_cov is not None:
            np.save(path / "training_features_cov.npy", self.training_features_cov)

        # Save models
        if self.knn_model is not None:
            with open(path / "knn_model.pkl", "wb") as f:
                pickle.dump(self.knn_model, f)
        if self.naturality_gmm is not None:
            with open(path / "naturality_gmm.pkl", "wb") as f:
                pickle.dump(self.naturality_gmm, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ReferencePanel":
        """Load reference panel from disk."""
        path = Path(path)

        with open(path / "reference_panel.json") as f:
            data = json.load(f)

        panel = cls(
            cell_type=data["cell_type"],
            n_samples=data.get("n_samples", 0),
            n_high_performers=data.get("n_high_performers", 0),
        )

        panel.physics_distributions = cls._dict_to_distributions(
            data.get("physics_distributions", {})
        )

        if "gc_distribution" in data:
            panel.gc_distribution = cls._dict_to_dist(data["gc_distribution"])
        if "cpg_distribution" in data:
            panel.cpg_distribution = cls._dict_to_dist(data["cpg_distribution"])
        if "entropy_distribution" in data:
            panel.entropy_distribution = cls._dict_to_dist(data["entropy_distribution"])
        if "activity_distribution" in data:
            panel.activity_distribution = cls._dict_to_dist(data["activity_distribution"])

        if "electrostatics_distributions" in data:
            panel.electrostatics_distributions = {
                k: cls._dict_to_dist(v) for k, v in data["electrostatics_distributions"].items()
            }

        # Load numpy arrays
        if (path / "training_features_mean.npy").exists():
            panel.training_features_mean = np.load(path / "training_features_mean.npy")
        if (path / "training_features_cov.npy").exists():
            panel.training_features_cov = np.load(path / "training_features_cov.npy")

        # Load models
        if (path / "knn_model.pkl").exists():
            with open(path / "knn_model.pkl", "rb") as f:
                panel.knn_model = pickle.load(f)
        if (path / "naturality_gmm.pkl").exists():
            with open(path / "naturality_gmm.pkl", "rb") as f:
                panel.naturality_gmm = pickle.load(f)

        return panel

    @staticmethod
    def _dist_to_dict(dist: ReferenceDistribution) -> dict:
        return {
            "mean": float(dist.mean),
            "std": float(dist.std),
            "percentiles": {k: float(v) for k, v in dist.percentiles.items()},
        }

    @staticmethod
    def _dict_to_dist(d: dict) -> ReferenceDistribution:
        return ReferenceDistribution(
            mean=d["mean"],
            std=d["std"],
            percentiles=d["percentiles"],
        )

    @staticmethod
    def _distributions_to_dict(distributions: dict) -> dict:
        result = {}
        for family, features in distributions.items():
            result[family] = {}
            for feature, dist in features.items():
                result[family][feature] = ReferencePanel._dist_to_dict(dist)
        return result

    @staticmethod
    def _dict_to_distributions(d: dict) -> dict:
        result = {}
        for family, features in d.items():
            result[family] = {}
            for feature, dist_dict in features.items():
                result[family][feature] = ReferencePanel._dict_to_dist(dist_dict)
        return result


class ReferencePanelBuilder:
    """
    Builds reference panels from natural sequence data.
    """

    def __init__(
        self,
        config: OracleCheckConfig,
        cadence_interface=None,
        physics_interface=None,
        tileformer_interface: Optional[TileFormerInterface] = None,
    ):
        """
        Initialize builder.

        Args:
            config: OracleCheck configuration
            cadence_interface: CADENCE interface for activity predictions
            physics_interface: PhysInformer interface for physics features
            tileformer_interface: TileFormer interface for electrostatics features
        """
        self.config = config
        self.cadence = cadence_interface
        self.physics = physics_interface
        self.tileformer = tileformer_interface

        # Auto-initialize TileFormer if checkpoint available
        if self.tileformer is None and config.tileformer_checkpoint is not None:
            checkpoint = config.tileformer_checkpoint
            if checkpoint.is_dir():
                # Find best_model.pth in checkpoint directory
                best_model = checkpoint / "run_20250819_063725" / "best_model.pth"
                if best_model.exists():
                    checkpoint = best_model
            if checkpoint.exists():
                self.tileformer = TileFormerInterface(
                    checkpoint_path=checkpoint,
                    device=config.device,
                )

    def build_from_data(
        self,
        sequences: List[str],
        activities: np.ndarray,
        cell_type: str,
        physics_features: Optional[np.ndarray] = None,
        physics_feature_names: Optional[List[str]] = None,
        cadence_features: Optional[np.ndarray] = None,
        electrostatics_features: Optional[List[Dict]] = None,
    ) -> ReferencePanel:
        """
        Build reference panel from data.

        Args:
            sequences: List of DNA sequences
            activities: Activity values [n_samples]
            cell_type: Cell type identifier
            physics_features: Optional pre-computed physics features [n_samples, n_features]
            physics_feature_names: Names of physics features
            cadence_features: Optional pre-computed CADENCE backbone features
            electrostatics_features: Optional pre-computed electrostatics features from TileFormer

        Returns:
            ReferencePanel with all distributions fitted
        """
        n_samples = len(sequences)

        # Identify high performers (top quartile)
        activity_threshold = np.percentile(
            activities,
            self.config.natural_high_performer_quantile * 100
        )
        high_performer_mask = activities >= activity_threshold
        n_high_performers = high_performer_mask.sum()

        # Build distributions
        panel = ReferencePanel(
            cell_type=cell_type,
            n_samples=n_samples,
            n_high_performers=n_high_performers,
        )

        # Activity distribution (from high performers)
        high_activities = activities[high_performer_mask]
        panel.activity_distribution = self._build_distribution(high_activities)

        # Composition distributions
        gc_content = np.array([self._compute_gc(seq) for seq in sequences])
        cpg_oe = np.array([self._compute_cpg_oe(seq) for seq in sequences])
        entropy = np.array([self._compute_entropy(seq) for seq in sequences])

        panel.gc_distribution = self._build_distribution(gc_content[high_performer_mask])
        panel.cpg_distribution = self._build_distribution(cpg_oe[high_performer_mask])
        panel.entropy_distribution = self._build_distribution(entropy[high_performer_mask])

        # Physics distributions
        if physics_features is not None and physics_feature_names is not None:
            panel.physics_distributions = self._build_physics_distributions(
                physics_features[high_performer_mask],
                physics_feature_names
            )

        # Electrostatics distributions (from TileFormer)
        if electrostatics_features is None and self.tileformer is not None:
            print("Computing electrostatics features with TileFormer...")
            electrostatics_features = self.tileformer.compute_batch_features(sequences)

        if electrostatics_features is not None:
            # Extract scalar features for high performers
            high_indices = np.where(high_performer_mask)[0]
            electro_keys = [
                'electrostatic_std_psi_mean', 'electrostatic_std_psi_std',
                'electrostatic_std_psi_min', 'electrostatic_std_psi_max',
                'electrostatic_enh_psi_mean', 'electrostatic_enh_psi_std',
                'electrostatic_enh_psi_min', 'electrostatic_enh_psi_max',
            ]

            for key in electro_keys:
                values = []
                for idx in high_indices:
                    if key in electrostatics_features[idx]:
                        values.append(electrostatics_features[idx][key])

                if values:
                    panel.electrostatics_distributions[key] = self._build_distribution(
                        np.array(values)
                    )

        # Training features for OOD detection
        if cadence_features is not None:
            panel.training_features_mean = np.mean(cadence_features, axis=0)
            panel.training_features_cov = np.cov(cadence_features.T)

            # Fit kNN for OOD
            panel.knn_model = NearestNeighbors(
                n_neighbors=min(self.config.knn_n_neighbors, len(cadence_features)),
                algorithm="auto",
            )
            panel.knn_model.fit(cadence_features)

            # Fit GMM for naturality scoring
            n_components = min(5, len(cadence_features) // 100)
            if n_components > 0:
                panel.naturality_gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type="full",
                    random_state=42,
                )
                panel.naturality_gmm.fit(cadence_features)

        return panel

    def build_from_lentiMPRA(
        self,
        cell_type: str,
        split: str = "calibration",
    ) -> ReferencePanel:
        """
        Build reference panel from lentiMPRA data.

        Args:
            cell_type: Cell type (K562, HepG2, WTC11)
            split: Data split to use

        Returns:
            ReferencePanel
        """
        # Load data
        data_path = self.config.get_lentiMPRA_data_path(cell_type, split)
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found: {data_path}")

        df = pd.read_csv(data_path, sep="\t")

        sequences = df["sequence"].tolist()
        activities = df["activity"].values

        # Get physics features if available
        physics_cols = [c for c in df.columns if c.startswith("thermo_") or c.startswith("mgw_")]
        physics_features = None
        physics_feature_names = None
        if physics_cols:
            physics_features = df[physics_cols].values
            physics_feature_names = physics_cols

        # Get CADENCE features if interface available
        cadence_features = None
        if self.cadence is not None:
            cadence_features = self.cadence.get_features(sequences, f"encode4_{cell_type.lower()}")

        return self.build_from_data(
            sequences=sequences,
            activities=activities,
            cell_type=cell_type,
            physics_features=physics_features,
            physics_feature_names=physics_feature_names,
            cadence_features=cadence_features,
        )

    def _build_distribution(self, values: np.ndarray) -> ReferenceDistribution:
        """Build distribution statistics from values."""
        return ReferenceDistribution(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            percentiles={
                "p5": float(np.percentile(values, 5)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "p95": float(np.percentile(values, 95)),
            },
            values=values,
        )

    def _build_physics_distributions(
        self,
        features: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Dict[str, ReferenceDistribution]]:
        """Build physics distributions grouped by family."""
        distributions = {}

        # Group features by family
        for family, family_prefixes in PHYSICS_FAMILIES.items():
            distributions[family] = {}

            for i, name in enumerate(feature_names):
                # Check if feature belongs to this family
                if any(name.startswith(prefix) for prefix in family_prefixes):
                    distributions[family][name] = self._build_distribution(features[:, i])

        return distributions

    @staticmethod
    def _compute_gc(sequence: str) -> float:
        """Compute GC content."""
        seq = sequence.upper()
        gc = sum(1 for b in seq if b in "GC")
        return gc / len(seq) if len(seq) > 0 else 0.0

    @staticmethod
    def _compute_cpg_oe(sequence: str) -> float:
        """Compute CpG observed/expected ratio."""
        seq = sequence.upper()
        n = len(seq)
        if n < 2:
            return 1.0

        c_count = seq.count("C")
        g_count = seq.count("G")
        cpg_count = seq.count("CG")

        expected = (c_count * g_count) / n if n > 0 else 0
        observed = cpg_count

        return observed / expected if expected > 0 else 1.0

    @staticmethod
    def _compute_entropy(sequence: str) -> float:
        """Compute Shannon entropy of sequence."""
        seq = sequence.upper()
        n = len(seq)
        if n == 0:
            return 0.0

        counts = {}
        for base in "ACGT":
            counts[base] = seq.count(base)

        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / n
                entropy -= p * np.log2(p)

        return entropy
