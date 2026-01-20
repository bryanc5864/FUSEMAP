"""
Post-hoc PLACE calibration for trained CADENCE models.

This script:
1. Copies trained CADENCE models into cadence_place folder
2. Fits PLACE uncertainty estimators on calibration sets
3. Saves calibrated versions (originals remain untouched)
"""

import os
import sys
import json
import shutil
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Setup paths
SCRIPT_DIR = Path(__file__).parent
FUSEMAP_DIR = SCRIPT_DIR.parent
TRAINING_DIR = FUSEMAP_DIR / "training"

# Add to path
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(FUSEMAP_DIR))

from sklearn.neighbors import NearestNeighbors


# Models to calibrate with their calibration data sources
MODELS_TO_CALIBRATE = {
    # Human single-celltype v2 models (in results/ folder)
    "cadence_k562_v2": {
        "checkpoint": "results/cadence_k562_v2/best_model.pt",
        "config_file": "results/cadence_k562_v2/config.json",
        "config_type": "single_celltype",
        "dataset": "encode4_k562",
        "calibration_source": "encode4_k562_calibration",
        "description": "K562 single-celltype (r=0.81 test)",
    },
    "cadence_k562_all_v2": {
        "checkpoint": "results/cadence_k562_all_v2/best_model.pt",
        "config_file": "results/cadence_k562_all_v2/config.json",
        "config_type": "single_celltype",
        "dataset": "encode4_k562",
        "calibration_source": "encode4_k562_calibration",
        "description": "K562 all modules (r=0.81 test)",
    },
    "cadence_hepg2_v2": {
        "checkpoint": "results/cadence_hepg2_v2/best_model.pt",
        "config_file": "results/cadence_hepg2_v2/config.json",
        "config_type": "single_celltype",
        "dataset": "encode4_hepg2",
        "calibration_source": "encode4_hepg2_calibration",
        "description": "HepG2 single-celltype",
    },
    "cadence_hepg2_all_v2": {
        "checkpoint": "results/cadence_hepg2_all_v2/best_model.pt",
        "config_file": "results/cadence_hepg2_all_v2/config.json",
        "config_type": "single_celltype",
        "dataset": "encode4_hepg2",
        "calibration_source": "encode4_hepg2_calibration",
        "description": "HepG2 all modules",
    },
    "cadence_wtc11_v2": {
        "checkpoint": "results/cadence_wtc11_v2/best_model.pt",
        "config_file": "results/cadence_wtc11_v2/config.json",
        "config_type": "single_celltype",
        "dataset": "encode4_wtc11",
        "calibration_source": "encode4_wtc11_calibration",
        "description": "WTC11 single-celltype",
    },
    "cadence_wtc11_all_v2": {
        "checkpoint": "results/cadence_wtc11_all_v2/best_model.pt",
        "config_file": "results/cadence_wtc11_all_v2/config.json",
        "config_type": "single_celltype",
        "dataset": "encode4_wtc11",
        "calibration_source": "encode4_wtc11_calibration",
        "description": "WTC11 all modules",
    },
    "cadence_deepstarr_v2": {
        "checkpoint": "training/results/cadence_deepstarr_v2/best_model.pt",
        "config_file": "training/results/cadence_deepstarr_v2/config.json",
        "config_type": "single_celltype",
        "dataset": "deepstarr",
        "calibration_source": "deepstarr_calibration",
        "description": "DeepSTARR v2",
    },
    "cadence_deepstarr_all_v2": {
        "checkpoint": "training/results/cadence_deepstarr_all_v2/best_model.pt",
        "config_file": "training/results/cadence_deepstarr_all_v2/config.json",
        "config_type": "single_celltype",
        "dataset": "deepstarr",
        "calibration_source": "deepstarr_calibration",
        "description": "DeepSTARR all modules v2",
    },
    # Yeast models - have 1% calibration carved from training
    "cadence_yeast_v1": {
        "checkpoint": "training/results/cadence_yeast_v1/best_model.pt",
        "config_file": "training/results/cadence_yeast_v1/config.json",
        "config_type": "single_celltype",
        "dataset": "dream_yeast",
        "calibration_source": "dream_yeast_calibration",
        "description": "Yeast baseline (r=0.734 test)",
    },
    "cadence_yeast_all_v1": {
        "checkpoint": "training/results/cadence_yeast_all_v1/best_model.pt",
        "config_file": "training/results/cadence_yeast_all_v1/config.json",
        "config_type": "single_celltype",
        "dataset": "dream_yeast",
        "calibration_source": "dream_yeast_calibration",
        "description": "Yeast with all modules (r=0.725 test)",
    },
    # Config 2 - multi-cell-type human
    "config2_multi_celltype_v1": {
        "checkpoint": "training/results/config2_multi_celltype_v1/best_model.pt",
        "config_file": "training/results/config2_multi_celltype_v1/config.json",
        "config_type": "multi_celltype_human",
        "dataset": None,
        "calibration_source": "encode4_calibration",
        "description": "Multi-cell-type human (HepG2 r=0.67)",
    },
    # Config 3 - cross-animal (best performing)
    "config3_cross_animal_v1": {
        "checkpoint": "training/results/config3_cross_animal_v1/best_model.pt",
        "config_file": "training/results/config3_cross_animal_v1/config.json",
        "config_type": "cross_animal",
        "dataset": None,
        "calibration_source": "encode4_calibration",
        "description": "Cross-animal (K562 r=0.69, DeepSTARR r=0.71/0.76)",
    },
    # Plant models
    "cadence_arabidopsis_v1": {
        "checkpoint": "training/results/cadence_arabidopsis_v1/best_model.pt",
        "config_file": "training/results/cadence_arabidopsis_v1/config.json",
        "config_type": "single_celltype",
        "dataset": "jores_arabidopsis",
        "calibration_source": "jores_arabidopsis_test",
        "description": "Arabidopsis baseline",
    },
    "cadence_arabidopsis_all_v1": {
        "checkpoint": "training/results/cadence_arabidopsis_all_v1/best_model.pt",
        "config_file": "training/results/cadence_arabidopsis_all_v1/config.json",
        "config_type": "single_celltype",
        "dataset": "jores_arabidopsis",
        "calibration_source": "jores_arabidopsis_test",
        "description": "Arabidopsis all modules",
    },
    "cadence_maize_v1": {
        "checkpoint": "training/results/cadence_maize_v1/best_model.pt",
        "config_file": "training/results/cadence_maize_v1/config.json",
        "config_type": "single_celltype",
        "dataset": "jores_maize",
        "calibration_source": "jores_maize_test",
        "description": "Maize baseline (leaf r=0.80, proto r=0.77)",
    },
    "cadence_maize_all_v1": {
        "checkpoint": "training/results/cadence_maize_all_v1/best_model.pt",
        "config_file": "training/results/cadence_maize_all_v1/config.json",
        "config_type": "single_celltype",
        "dataset": "jores_maize",
        "calibration_source": "jores_maize_test",
        "description": "Maize all modules",
    },
    "cadence_sorghum_v1": {
        "checkpoint": "training/results/cadence_sorghum_v1/best_model.pt",
        "config_file": "training/results/cadence_sorghum_v1/config.json",
        "config_type": "single_celltype",
        "dataset": "jores_sorghum",
        "calibration_source": "jores_sorghum_test",
        "description": "Sorghum baseline (leaf r=0.78, proto r=0.77)",
    },
    "cadence_sorghum_all_v1": {
        "checkpoint": "training/results/cadence_sorghum_all_v1/best_model.pt",
        "config_file": "training/results/cadence_sorghum_all_v1/config.json",
        "config_type": "single_celltype",
        "dataset": "jores_sorghum",
        "calibration_source": "jores_sorghum_test",
        "description": "Sorghum all modules",
    },
    # Config 4 - cross-kingdom (animal + plant)
    "config4_cross_kingdom_v1": {
        "checkpoint": "results/config4_cross_kingdom_v1/best_model.pt",
        "config_file": "results/config4_cross_kingdom_v1/config.json",
        "config_type": "cross_kingdom",
        "dataset": None,
        "calibration_source": "cross_kingdom_calibration",  # ENCODE4 + plant
        "description": "Cross-kingdom (animal + plant)",
    },
    # Config 5 - universal foundation model (no yeast version)
    "config5_universal_no_yeast": {
        "checkpoint": "results/config5_universal_no_yeast_20260114_204533/best_model.pt",
        "config_file": "results/config5_universal_no_yeast_20260114_204533/config.json",
        "config_type": "universal",
        "dataset": None,
        "calibration_source": "cross_kingdom_calibration",  # ENCODE4 + plant (no yeast)
        "description": "Universal foundation model (no yeast)",
    },
}


def copy_model_to_place_folder(model_name: str, model_info: dict, output_dir: Path) -> Path:
    """Copy original model files to cadence_place folder."""

    # Create model subdirectory
    model_dir = output_dir / model_name
    model_dir.mkdir(exist_ok=True)

    # Copy checkpoint
    src_checkpoint = FUSEMAP_DIR / model_info["checkpoint"]
    dst_checkpoint = model_dir / "original_model.pt"

    if src_checkpoint.exists():
        shutil.copy2(src_checkpoint, dst_checkpoint)
        print(f"    Copied checkpoint to {dst_checkpoint}")
    else:
        print(f"    WARNING: Checkpoint not found: {src_checkpoint}")
        return None

    # Copy config if exists
    if "config_file" in model_info:
        src_config = FUSEMAP_DIR / model_info["config_file"]
        if src_config.exists():
            dst_config = model_dir / "config.json"
            shutil.copy2(src_config, dst_config)

    # Copy normalizer if exists
    src_normalizer = src_checkpoint.parent / "normalizer.json"
    if src_normalizer.exists():
        shutil.copy2(src_normalizer, model_dir / "normalizer.json")

    # Copy final results if exists
    src_results = src_checkpoint.parent / "final_results.json"
    if src_results.exists():
        shutil.copy2(src_results, model_dir / "original_results.json")

    return model_dir


def infer_architecture_from_state_dict(state_dict):
    """Infer model architecture parameters from checkpoint state_dict."""
    # Infer n_celltypes from celltype_embed
    n_celltypes = 1
    if "celltype_embed.weight" in state_dict:
        n_celltypes = state_dict["celltype_embed.weight"].shape[0]

    # Infer n_species from species_embed
    n_species = 1
    if "species_embed.weight" in state_dict:
        n_species = state_dict["species_embed.weight"].shape[0]

    # Check if using species-specific stems
    use_species_stem = any("species_stems" in k for k in state_dict.keys())

    # Infer n_kingdoms from kingdom_embed if present
    n_kingdoms = 1
    if "kingdom_embed.weight" in state_dict:
        n_kingdoms = state_dict["kingdom_embed.weight"].shape[0]

    return {
        "n_celltypes": n_celltypes,
        "n_species": n_species,
        "n_kingdoms": n_kingdoms,
        "use_species_stem": use_species_stem,
    }


def load_model_from_checkpoint(checkpoint_path: Path, device: str = "cuda"):
    """Load model from checkpoint."""

    from training.config import DATASET_CATALOG, ModelConfig
    from training.models import MultiSpeciesCADENCE

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Try to get config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
        # Handle both dict and dataclass configs
        if isinstance(config, dict):
            model_config_dict = config.get("model", {})
            # Remove any non-ModelConfig fields that might be in the dict
            model_config = ModelConfig(**model_config_dict)
            datasets = config.get("datasets", ["unknown"])
        else:
            model_config = config.model if hasattr(config, 'model') else config
            datasets = config.datasets if hasattr(config, 'datasets') else ["unknown"]
    else:
        # Try to load from config.json in same directory
        config_file = checkpoint_path.parent / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config_dict = json.load(f)
            datasets = config_dict.get("datasets", ["unknown"])
            model_config = ModelConfig(**config_dict.get("model", {}))
        else:
            raise ValueError("Cannot determine model config")

    # Infer architecture from state_dict to ensure correct model structure
    arch_params = infer_architecture_from_state_dict(state_dict)

    # Override model config with inferred use_species_stem if needed
    if arch_params["use_species_stem"] and not getattr(model_config, 'use_species_stem', False):
        model_config.use_species_stem = True

    # Create model with correct architecture
    model = MultiSpeciesCADENCE(
        config=model_config,
        dataset_names=datasets,
        n_species=arch_params["n_species"],
        n_kingdoms=arch_params["n_kingdoms"],
        n_celltypes=arch_params["n_celltypes"],
    )

    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_calibration_data(calibration_source: str, batch_size: int = 64):
    """Load calibration data based on source type."""

    from training.data_loaders import (
        LentiMPRADataset, DREAMYeastDataset, JoresPlantDataset
    )

    loaders = {}

    if calibration_source == "dream_yeast_calibration":
        dataset = DREAMYeastDataset(split="calibration", normalize=True)
        loaders["dream_yeast"] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

    # Single cell-type specific calibration (e.g., encode4_k562_calibration)
    elif calibration_source.startswith("encode4_") and calibration_source.endswith("_calibration"):
        cell_type_raw = calibration_source.replace("encode4_", "").replace("_calibration", "")
        # Handle special casing
        cell_type_map = {"k562": "K562", "hepg2": "HepG2", "wtc11": "WTC11"}
        cell_type = cell_type_map.get(cell_type_raw.lower(), cell_type_raw.upper())
        try:
            dataset = LentiMPRADataset(
                cell_type=cell_type, split="calibration", normalize=True
            )
            loaders[f"encode4_{cell_type.lower()}"] = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=2
            )
        except Exception as e:
            print(f"      Warning: Could not load {cell_type} calibration: {e}")

    elif calibration_source == "deepstarr_calibration":
        # DeepSTARR uses validation as calibration
        from training.data_loaders import DeepSTARRDataset
        try:
            dataset = DeepSTARRDataset(split="val", normalize=True)
            loaders["deepstarr"] = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=2
            )
        except Exception as e:
            print(f"      Warning: Could not load DeepSTARR calibration: {e}")

    elif calibration_source == "encode4_calibration":
        # Cell types must match folder/file casing
        for cell_type in ["K562", "HepG2", "WTC11"]:
            try:
                dataset = LentiMPRADataset(
                    cell_type=cell_type, split="calibration", normalize=True
                )
                loaders[f"encode4_{cell_type.lower()}"] = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=False, num_workers=2
                )
            except Exception as e:
                print(f"      Warning: Could not load {cell_type} calibration: {e}")

    elif calibration_source.startswith("jores_") and (calibration_source.endswith("_val") or calibration_source.endswith("_test")):
        # Parse species and split from calibration_source (e.g., jores_maize_test -> maize, test)
        if calibration_source.endswith("_val"):
            species = calibration_source.replace("jores_", "").replace("_val", "")
            split = "val"
        else:
            species = calibration_source.replace("jores_", "").replace("_test", "")
            split = "test"
        dataset = JoresPlantDataset(species=species, split=split, normalize=False)
        loaders[f"jores_{species}"] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

    elif calibration_source == "cross_kingdom_calibration":
        # Config 4: ENCODE4 + Plant calibration
        # ENCODE4 human calibration
        for cell_type in ["K562", "HepG2", "WTC11"]:
            try:
                dataset = LentiMPRADataset(
                    cell_type=cell_type, split="calibration", normalize=True
                )
                loaders[f"encode4_{cell_type.lower()}"] = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=False, num_workers=2
                )
            except Exception as e:
                print(f"      Warning: Could not load {cell_type} calibration: {e}")
        # Plant test sets (arabidopsis, maize)
        for species in ["arabidopsis", "maize"]:
            try:
                dataset = JoresPlantDataset(species=species, split="test", normalize=False)
                loaders[f"jores_{species}"] = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=False, num_workers=2
                )
            except Exception as e:
                print(f"      Warning: Could not load {species} calibration: {e}")

    elif calibration_source == "universal_calibration":
        # Config 5: ENCODE4 + Plant + Yeast calibration
        # ENCODE4 human calibration
        for cell_type in ["K562", "HepG2", "WTC11"]:
            try:
                dataset = LentiMPRADataset(
                    cell_type=cell_type, split="calibration", normalize=True
                )
                loaders[f"encode4_{cell_type.lower()}"] = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=False, num_workers=2
                )
            except Exception as e:
                print(f"      Warning: Could not load {cell_type} calibration: {e}")
        # Plant test sets (arabidopsis, maize, sorghum)
        for species in ["arabidopsis", "maize", "sorghum"]:
            try:
                dataset = JoresPlantDataset(species=species, split="test", normalize=False)
                loaders[f"jores_{species}"] = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=False, num_workers=2
                )
            except Exception as e:
                print(f"      Warning: Could not load {species} calibration: {e}")
        # Yeast calibration
        try:
            dataset = DREAMYeastDataset(split="calibration", normalize=True)
            loaders["dream_yeast"] = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=2
            )
        except Exception as e:
            print(f"      Warning: Could not load yeast calibration: {e}")

    else:
        raise ValueError(f"Unknown calibration source: {calibration_source}")

    return loaders


def build_species_mapping(model_dataset_names: List[str]) -> Dict[str, int]:
    """Build species to index mapping from model's training dataset names."""
    from training.config import DATASET_CATALOG

    species_set = []
    for ds_name in model_dataset_names:
        if ds_name in DATASET_CATALOG:
            species = DATASET_CATALOG[ds_name].species
            if species not in species_set:
                species_set.append(species)
    return {sp: idx for idx, sp in enumerate(species_set)}


def build_kingdom_mapping(model_dataset_names: List[str]) -> Dict[str, int]:
    """Build kingdom to index mapping from model's training dataset names."""
    from training.config import DATASET_CATALOG

    kingdom_set = []
    for ds_name in model_dataset_names:
        if ds_name in DATASET_CATALOG:
            kingdom = DATASET_CATALOG[ds_name].kingdom
            if kingdom not in kingdom_set:
                kingdom_set.append(kingdom)
    return {k: idx for idx, k in enumerate(kingdom_set)}


# Map calibration dataset names to their base dataset
CALIBRATION_TO_BASE_DATASET = {
    "encode4_k562": "encode4_k562",
    "encode4_hepg2": "encode4_hepg2",
    "encode4_wtc11": "encode4_wtc11",
    "dream_yeast": "dream_yeast",
    "deepstarr": "deepstarr",
    "jores_maize": "jores_maize",
    "jores_sorghum": "jores_sorghum",
    "jores_arabidopsis": "jores_arabidopsis",
}


def get_species_idx_for_dataset(dataset_name: str, species_to_idx: Dict[str, int]) -> int:
    """Get species index for a dataset name."""
    from training.config import DATASET_CATALOG

    base_name = CALIBRATION_TO_BASE_DATASET.get(dataset_name, dataset_name)

    if base_name in DATASET_CATALOG:
        species = DATASET_CATALOG[base_name].species
        return species_to_idx.get(species, 0)
    return 0


def get_kingdom_idx_for_dataset(dataset_name: str, kingdom_to_idx: Dict[str, int]) -> int:
    """Get kingdom index for a dataset name."""
    from training.config import DATASET_CATALOG

    base_name = CALIBRATION_TO_BASE_DATASET.get(dataset_name, dataset_name)

    if base_name in DATASET_CATALOG:
        kingdom = DATASET_CATALOG[base_name].kingdom
        return kingdom_to_idx.get(kingdom, 0)
    return 0


def fit_place_uncertainty(
    model,
    calibration_loaders: dict,
    device: str = "cuda",
    n_neighbors: int = 200,
    alpha: float = 0.1,
):
    """Fit PLACE uncertainty estimator on calibration data."""

    print(f"    Collecting calibration features...")

    all_features = []
    all_residuals = []
    total_samples = 0

    # Build species mapping if model uses species-specific stems
    species_to_idx = None
    if getattr(model, 'use_species_stem', False) and hasattr(model, 'species_stems') and model.species_stems is not None:
        species_to_idx = build_species_mapping(model.dataset_names)

    # Build kingdom mapping if model uses kingdom-specific stems
    kingdom_to_idx = None
    if getattr(model, 'use_kingdom_stem', False) and hasattr(model, 'kingdom_stems') and model.kingdom_stems is not None:
        kingdom_to_idx = build_kingdom_mapping(model.dataset_names)

    model.eval()
    with torch.no_grad():
        for dataset_name, loader in calibration_loaders.items():
            print(f"      Processing {dataset_name}...")

            for batch in loader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    sequences = batch["sequence"].to(device)
                    targets = batch["activity"].to(device)
                elif isinstance(batch, (list, tuple)):
                    sequences = batch[0].to(device)
                    targets = batch[1].to(device)
                else:
                    continue

                # Get species_idx for species-specific stem models
                species_idx = None
                if species_to_idx is not None:
                    sp_idx = get_species_idx_for_dataset(dataset_name, species_to_idx)
                    species_idx = torch.full((len(sequences),), sp_idx, dtype=torch.long, device=device)

                # Get kingdom_idx for kingdom-specific stem models
                kingdom_idx = None
                if kingdom_to_idx is not None:
                    k_idx = get_kingdom_idx_for_dataset(dataset_name, kingdom_to_idx)
                    kingdom_idx = torch.full((len(sequences),), k_idx, dtype=torch.long, device=device)

                # Get backbone features
                features = model._backbone_forward(sequences, species_idx=species_idx, kingdom_idx=kingdom_idx)

                # Get predictions through heads
                outputs = model(
                    sequences,
                    species_idx=species_idx,
                    kingdom_idx=kingdom_idx,
                    dataset_names=[dataset_name] * len(sequences)
                )

                # Find matching head for this dataset
                head_names = model.dataset_to_heads.get(dataset_name, [])

                for head_idx, head_name in enumerate(head_names):
                    if head_name not in outputs:
                        continue

                    head_output = outputs[head_name]
                    predictions = head_output["mean"]
                    indices = head_output.get("indices", list(range(len(predictions))))

                    if len(indices) == 0:
                        continue

                    # Get corresponding targets
                    if targets.dim() > 1:
                        target_col = min(head_idx, targets.shape[1] - 1)
                        head_targets = targets[indices, target_col]
                    else:
                        head_targets = targets[indices]

                    # Compute residuals
                    residuals = (predictions - head_targets).cpu()
                    head_features = features[indices].cpu()

                    all_features.append(head_features)
                    all_residuals.append(residuals)
                    total_samples += len(indices)

    if total_samples == 0:
        print("    ERROR: No calibration samples collected!")
        return None

    # Concatenate
    all_features = torch.cat(all_features, dim=0).numpy()
    all_residuals = torch.cat(all_residuals, dim=0).numpy()

    print(f"    Fitting PLACE on {total_samples} samples...")

    # Create PLACE data structure
    place_data = {
        "calibration_features": all_features,
        "calibration_residuals": all_residuals,
        "n_samples": total_samples,
        "n_neighbors": min(n_neighbors, total_samples),
        "alpha": alpha,
    }

    # Fit KNN
    knn = NearestNeighbors(n_neighbors=place_data["n_neighbors"])
    knn.fit(all_features)

    # Compute statistics
    place_data["noise_var"] = float(np.var(all_residuals))
    place_data["residual_mean"] = float(np.mean(all_residuals))
    place_data["residual_std"] = float(np.std(all_residuals))

    # Compute posterior covariance (simplified Laplace)
    lambda_reg = 1e-3
    XtX = all_features.T @ all_features
    try:
        place_data["posterior_cov"] = np.linalg.inv(
            XtX + lambda_reg * np.eye(all_features.shape[1])
        )
    except:
        place_data["posterior_cov"] = None

    print(f"    PLACE fitted: noise_var={place_data['noise_var']:.4f}, "
          f"residual_std={place_data['residual_std']:.4f}")

    return place_data, knn


def save_calibrated_model(
    model_dir: Path,
    model_name: str,
    original_checkpoint: dict,
    place_data: dict,
    knn_model,
    model_info: dict,
):
    """Save calibrated model with PLACE data."""
    import pickle

    # Save PLACE-calibrated checkpoint
    calibrated_checkpoint = {
        **original_checkpoint,
        "place_fitted": True,
        "place_data": {
            "n_samples": place_data["n_samples"],
            "n_neighbors": place_data["n_neighbors"],
            "alpha": place_data["alpha"],
            "noise_var": place_data["noise_var"],
            "residual_mean": place_data["residual_mean"],
            "residual_std": place_data["residual_std"],
        },
        "calibration_date": datetime.now().isoformat(),
    }

    # Save main checkpoint
    torch.save(calibrated_checkpoint, model_dir / "model_with_place.pt")

    # Save PLACE calibration data separately (can be large)
    np.savez_compressed(
        model_dir / "place_calibration_data.npz",
        features=place_data["calibration_features"],
        residuals=place_data["calibration_residuals"],
    )

    # Save KNN model
    with open(model_dir / "place_knn_model.pkl", "wb") as f:
        pickle.dump(knn_model, f)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "description": model_info["description"],
        "calibration_source": model_info["calibration_source"],
        "calibration_date": datetime.now().isoformat(),
        "place_stats": {
            "n_samples": place_data["n_samples"],
            "n_neighbors": place_data["n_neighbors"],
            "alpha": place_data["alpha"],
            "noise_var": place_data["noise_var"],
            "residual_mean": place_data["residual_mean"],
            "residual_std": place_data["residual_std"],
        },
    }

    with open(model_dir / "place_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved calibrated model to {model_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy and PLACE-calibrate trained CADENCE models"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific models to calibrate (default: all)"
    )
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--n-neighbors", type=int, default=200, help="KNN neighbors")
    parser.add_argument("--alpha", type=float, default=0.1, help="Coverage level")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    output_dir = SCRIPT_DIR

    # Select models
    if args.models:
        models = {k: v for k, v in MODELS_TO_CALIBRATE.items() if k in args.models}
    else:
        models = MODELS_TO_CALIBRATE

    print("=" * 70)
    print("PLACE Calibration for CADENCE Models")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Models to process: {len(models)}")
    print()

    results = {}

    for model_name, model_info in models.items():
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"Description: {model_info['description']}")
        print("=" * 70)

        try:
            # Step 1: Copy model to place folder
            print("  Step 1: Copying model files...")
            model_dir = copy_model_to_place_folder(model_name, model_info, output_dir)

            if model_dir is None:
                results[model_name] = {"status": "skipped", "reason": "checkpoint not found"}
                continue

            # Step 2: Load model
            print("  Step 2: Loading model...")
            checkpoint_path = model_dir / "original_model.pt"
            model, checkpoint = load_model_from_checkpoint(checkpoint_path, args.device)

            # Step 3: Load calibration data
            print(f"  Step 3: Loading calibration data ({model_info['calibration_source']})...")
            try:
                calibration_loaders = load_calibration_data(
                    model_info["calibration_source"],
                    batch_size=args.batch_size
                )
            except Exception as e:
                print(f"    ERROR loading calibration data: {e}")
                results[model_name] = {"status": "failed", "reason": f"calibration data: {e}"}
                continue

            if not calibration_loaders:
                print("    No calibration loaders available!")
                results[model_name] = {"status": "failed", "reason": "no calibration data"}
                continue

            # Step 4: Fit PLACE
            print("  Step 4: Fitting PLACE uncertainty...")
            place_result = fit_place_uncertainty(
                model, calibration_loaders,
                device=args.device,
                n_neighbors=args.n_neighbors,
                alpha=args.alpha,
            )

            if place_result is None:
                results[model_name] = {"status": "failed", "reason": "PLACE fitting failed"}
                continue

            place_data, knn_model = place_result

            # Step 5: Save calibrated model
            print("  Step 5: Saving calibrated model...")
            save_calibrated_model(
                model_dir, model_name, checkpoint,
                place_data, knn_model, model_info
            )

            results[model_name] = {
                "status": "success",
                "output_dir": str(model_dir),
                "n_calibration_samples": place_data["n_samples"],
                "noise_var": place_data["noise_var"],
            }

            print(f"\n  SUCCESS: {model_name}")

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results[model_name] = {"status": "error", "reason": str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)

    success_count = 0
    for model_name, result in results.items():
        status = result["status"]
        if status == "success":
            success_count += 1
            print(f"  ✓ {model_name}: {result['n_calibration_samples']} samples, "
                  f"noise_var={result['noise_var']:.4f}")
        else:
            print(f"  ✗ {model_name}: {status} - {result.get('reason', '')}")

    print(f"\nTotal: {success_count}/{len(results)} models calibrated successfully")

    # Save summary
    summary_file = output_dir / "calibration_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
