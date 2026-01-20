#!/usr/bin/env python3
"""
Cross-species evaluation of CADENCE models on mouse ESC STARR-seq data.
Run from FUSEMAP directory: python evaluate_mouse_esc.py
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import roc_auc_score

# Import CADENCE model and config
from training.models import MultiSpeciesCADENCE
from training.config import ModelConfig

BASE_DIR = Path("external_validation")
DATA_FILE = BASE_DIR / "processed" / "mouse_esc_sequences.csv"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def one_hot_encode_batch(sequences: list) -> torch.Tensor:
    """One-hot encode a batch of DNA sequences."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    batch_size = len(sequences)
    seq_len = len(sequences[0])

    encoded = torch.zeros(batch_size, 4, seq_len, dtype=torch.float32)
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            if base in mapping:
                encoded[i, mapping[base], j] = 1.0
    return encoded


def evaluate_predictions(predictions, actual, name=""):
    """Compute scale-invariant metrics."""
    metrics = {}

    # Rank correlations
    spearman_r, _ = spearmanr(predictions, actual)
    kendall_t, _ = kendalltau(predictions, actual)
    pearson_r, _ = pearsonr(predictions, actual)

    metrics['spearman_r'] = float(spearman_r) if not np.isnan(spearman_r) else 0
    metrics['kendall_tau'] = float(kendall_t) if not np.isnan(kendall_t) else 0
    metrics['pearson_r'] = float(pearson_r) if not np.isnan(pearson_r) else 0

    # Top-K precision
    n = len(predictions)
    for k in [100, 500, 1000]:
        if k < n:
            pred_top = set(np.argsort(predictions)[-k:])
            actual_top = set(np.argsort(actual)[-k:])
            precision = len(pred_top & actual_top) / k
            metrics[f'precision_at_{k}'] = float(precision)

    # AUROC for high vs low
    threshold = np.percentile(actual, 75)
    labels = (actual >= threshold).astype(int)
    try:
        metrics['auroc'] = float(roc_auc_score(labels, predictions))
    except:
        metrics['auroc'] = 0.5

    # Enrichment
    pred_q = pd.qcut(predictions, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    q_means = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        mask = pred_q == q
        if mask.sum() > 0:
            q_means[q] = float(actual[mask].mean())
    metrics['enrichment_q4_vs_q1'] = q_means.get('Q4', 0) / (q_means.get('Q1', 1e-6) + 1e-6)
    metrics['q1_mean'] = q_means.get('Q1', 0)
    metrics['q4_mean'] = q_means.get('Q4', 0)

    return metrics


def load_cadence_model(model_dir: Path, device: str = 'cuda'):
    """Load a CADENCE model from checkpoint."""
    model_path = model_dir / "best_model.pt"
    config_path = model_dir / "config.json"

    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        return None, None

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Load config
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        datasets = config_dict.get('datasets', ['encode4_k562'])
        model_cfg = config_dict.get('model', {})
    else:
        datasets = ['encode4_k562']
        model_cfg = {}

    # Infer architecture from state dict
    n_kingdoms = 1
    n_species = 1
    n_celltypes = 1

    if 'kingdom_embed.weight' in state_dict:
        n_kingdoms = state_dict['kingdom_embed.weight'].shape[0]
    if 'species_embed.weight' in state_dict:
        n_species = state_dict['species_embed.weight'].shape[0]
    if 'celltype_embed.weight' in state_dict:
        n_celltypes = state_dict['celltype_embed.weight'].shape[0]

    has_species_stem = any('species_stems' in k for k in state_dict.keys())
    has_kingdom_stem = any('kingdom_stems' in k for k in state_dict.keys())

    print(f"  Datasets: {datasets}")
    print(f"  Architecture: kingdoms={n_kingdoms}, species={n_species}, celltypes={n_celltypes}")
    print(f"  Species stem: {has_species_stem}, Kingdom stem: {has_kingdom_stem}")

    # Create model config
    model_config = ModelConfig()

    # Override with saved config values
    for key, val in model_cfg.items():
        if hasattr(model_config, key):
            setattr(model_config, key, val)

    # Handle species/kingdom stems
    if has_species_stem:
        model_config.use_species_stem = True
    if has_kingdom_stem:
        model_config.use_kingdom_stem = True

    # Create model
    model = MultiSpeciesCADENCE(
        config=model_config,
        dataset_names=datasets,
        n_species=n_species,
        n_kingdoms=n_kingdoms,
        n_celltypes=n_celltypes,
    )

    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, datasets


def run_inference(model, sequences: list, device: str = 'cuda', batch_size: int = 256,
                  kingdom_idx: int = 0, species_idx: int = 0, dataset_name: str = None):
    """Run inference on sequences.

    Args:
        model: CADENCE model
        sequences: List of DNA sequences
        device: cuda or cpu
        batch_size: Batch size for inference
        kingdom_idx: Kingdom index (0=animal for mouse)
        species_idx: Species index (0 for unknown/new species)
        dataset_name: Dataset name for element_type embedding (use first dataset if None)
    """
    predictions = []

    # Use first dataset if not specified (for element type embedding)
    if dataset_name is None and hasattr(model, 'dataset_names') and model.dataset_names:
        dataset_name = model.dataset_names[0]

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]

        # Pad to 256bp if needed (CADENCE expects 256bp)
        target_len = 256
        padded_seqs = []
        for seq in batch_seqs:
            if len(seq) < target_len:
                pad_left = (target_len - len(seq)) // 2
                pad_right = target_len - len(seq) - pad_left
                padded_seq = 'N' * pad_left + seq + 'N' * pad_right
            elif len(seq) > target_len:
                start = (len(seq) - target_len) // 2
                padded_seq = seq[start:start+target_len]
            else:
                padded_seq = seq
            padded_seqs.append(padded_seq)

        x = one_hot_encode_batch(padded_seqs).to(device)
        curr_batch_size = x.shape[0]

        # Create index tensors for models with embeddings/stems
        k_idx = torch.full((curr_batch_size,), kingdom_idx, dtype=torch.long, device=device)
        s_idx = torch.full((curr_batch_size,), species_idx, dtype=torch.long, device=device)
        c_idx = torch.zeros(curr_batch_size, dtype=torch.long, device=device)  # celltype
        length = torch.full((curr_batch_size,), 230, dtype=torch.long, device=device)  # original length

        # Dataset names for element type embedding
        ds_names = [dataset_name] * curr_batch_size if dataset_name else None

        with torch.no_grad():
            # Run forward pass with all conditioning info
            out = model(x, kingdom_idx=k_idx, species_idx=s_idx, celltype_idx=c_idx,
                       original_length=length, dataset_names=ds_names)

            # Handle different output formats
            if isinstance(out, dict):
                # Get first head's predictions
                first_key = list(out.keys())[0]
                pred = out[first_key].get('mean', out[first_key])
                if isinstance(pred, dict):
                    pred = pred.get('mean', list(pred.values())[0])
            else:
                pred = out

            # Ensure we have a tensor
            if isinstance(pred, torch.Tensor):
                predictions.extend(pred.cpu().numpy().flatten().tolist())
            else:
                print(f"  Unexpected output type: {type(pred)}")
                return None

    return np.array(predictions)


def main():
    print("=" * 70)
    print("CROSS-SPECIES EVALUATION: CADENCE on Mouse ESC STARR-seq")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load data
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  Sequences: {len(df)}")
    print(f"  Sequence length: {len(df['sequence'].iloc[0])}")

    sequences = df['sequence'].tolist()
    activities_2iL = df['activity_2iL'].values
    activities_SL = df['activity_SL'].values

    results = {}

    # Models to test
    models_to_test = {
        'cadence_k562': Path('results/cadence_k562_v2'),
        'cadence_hepg2': Path('results/cadence_hepg2_v2'),
        'config4_cross_kingdom': Path('results/config4_cross_kingdom_v1'),
        'config5_universal': Path('results/config5_universal_no_yeast_20260114_204533'),
    }

    for model_name, model_dir in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")

        if not model_dir.exists():
            print(f"  Directory not found: {model_dir}")
            continue

        model, datasets = load_cadence_model(model_dir, device)

        if model is None:
            print(f"  Skipping {model_name}")
            continue

        print(f"\n  Running inference on {len(sequences)} sequences...")
        predictions = run_inference(model, sequences, device)

        if predictions is None:
            print(f"  Inference failed for {model_name}")
            continue

        print(f"  Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        print(f"  Predictions mean: {predictions.mean():.3f}, std: {predictions.std():.3f}")

        # Evaluate
        metrics_2iL = evaluate_predictions(predictions, activities_2iL, f"{model_name}_2iL")
        metrics_SL = evaluate_predictions(predictions, activities_SL, f"{model_name}_SL")

        results[model_name] = {
            '2iL_condition': metrics_2iL,
            'SL_condition': metrics_SL,
            'model_dir': str(model_dir),
            'n_samples': len(sequences),
            'predictions_range': [float(predictions.min()), float(predictions.max())],
            'predictions_mean': float(predictions.mean()),
            'predictions_std': float(predictions.std()),
        }

        print(f"\n  Results (2iL condition - ground-state ESCs):")
        print(f"    Spearman ρ: {metrics_2iL['spearman_r']:.4f}")
        print(f"    Pearson r:  {metrics_2iL['pearson_r']:.4f}")
        print(f"    Kendall τ:  {metrics_2iL['kendall_tau']:.4f}")
        print(f"    Precision@100: {metrics_2iL.get('precision_at_100', 0):.4f}")
        print(f"    Precision@500: {metrics_2iL.get('precision_at_500', 0):.4f}")
        print(f"    AUROC: {metrics_2iL['auroc']:.4f}")
        print(f"    Enrichment Q4/Q1: {metrics_2iL['enrichment_q4_vs_q1']:.2f}×")

        print(f"\n  Results (SL condition - metastable ESCs):")
        print(f"    Spearman ρ: {metrics_SL['spearman_r']:.4f}")
        print(f"    Pearson r:  {metrics_SL['pearson_r']:.4f}")
        print(f"    AUROC: {metrics_SL['auroc']:.4f}")

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

    # Save results
    output_file = OUTPUT_DIR / "mouse_esc_cadence_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump({
            'experiment': 'cross_species_mouse_esc',
            'timestamp': datetime.now().isoformat(),
            'dataset': {
                'name': 'Mouse ESC STARR-seq (GSE143546)',
                'organism': 'Mus musculus',
                'n_sequences': len(df),
                'sequence_length': len(df['sequence'].iloc[0]),
                'conditions': ['2iL (ground-state)', 'SL (metastable)'],
                'classes': df['class'].value_counts().to_dict()
            },
            'results': results
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Zero-Shot Cross-Species Performance (Human → Mouse)")
    print("=" * 70)
    print(f"{'Model':<25} {'Spearman ρ':>12} {'AUROC':>10} {'Enrichment':>12}")
    print("-" * 70)
    for model_name, model_results in results.items():
        m = model_results['2iL_condition']
        print(f"{model_name:<25} {m['spearman_r']:>12.4f} {m['auroc']:>10.4f} {m['enrichment_q4_vs_q1']:>11.2f}×")

    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
