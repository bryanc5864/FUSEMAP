"""Generate CADENCE prediction vs actual scatter plot using real HepG2 lentiMPRA test data.

Standalone script - uses direct file imports to avoid dependency issues.
"""

import os
import sys
import json
import enum
import types
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Workaround: register fake training.config so torch.load can unpickle
# ============================================================================

def _patch_training_imports():
    """Patch sys.modules so torch.load can deserialize ConfigurationType."""
    fake_config = types.ModuleType("training.config")

    class ConfigurationType(enum.Enum):
        SINGLE_CELLTYPE = "config1_single_celltype"
        MULTI_CELLTYPE = "config2_multi_celltype"
        CROSS_ANIMAL = "config3_cross_animal"
        CROSS_KINGDOM = "config4_cross_kingdom"
        UNIVERSAL = "config5_universal"

    fake_config.ConfigurationType = ConfigurationType

    fake_training = types.ModuleType("training")
    fake_training.__path__ = []
    fake_training.config = fake_config

    sys.modules["training"] = fake_training
    sys.modules["training.config"] = fake_config
    sys.modules["training.models"] = types.ModuleType("training.models")


def import_from_file(module_name, filepath):
    """Import a module directly from file, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# Data loading
# ============================================================================

def one_hot_encode(seq, target_length=230):
    """One-hot encode DNA sequence to [4, L] array."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.full((4, target_length), 0.25, dtype=np.float32)
    for i, base in enumerate(seq[:target_length]):
        b = base.upper()
        if b in mapping:
            arr[:, i] = 0.0
            arr[mapping[b], i] = 1.0
    return arr


def load_hepg2_split(split_path, target_length=230):
    """Load HepG2 lentiMPRA data from a pre-split TSV file."""
    import csv
    sequences = []
    activities = []
    with open(split_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sequences.append(row['sequence'])
            activities.append(float(row['activity']))

    encoded = np.stack([one_hot_encode(s, target_length) for s in sequences])
    acts = np.array(activities, dtype=np.float32)
    print(f"  Loaded {len(sequences)} sequences from {os.path.basename(split_path)}")
    return encoded, acts


# ============================================================================
# Model loading
# ============================================================================

def build_and_load_model(device):
    """Build model matching MultiSpeciesCADENCE structure and load weights."""
    cadence_mod = import_from_file("cadence", "models/CADENCE/cadence.py")

    LocalBlock = cadence_mod.LocalBlock
    EffBlock = cadence_mod.EffBlock
    ResidualConcat = cadence_mod.ResidualConcat
    MapperBlock = cadence_mod.MapperBlock

    # Architecture params (from config.json - same as DeepSTARR)
    stem_ch = 64
    stem_ks = 11
    ef_ks = 9
    ef_block_sizes = [80, 96, 112, 128]
    pool_sizes = [2, 2, 2, 2]
    resize_factor = 4
    activation = nn.SiLU

    class CADENCEWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = LocalBlock(in_ch=4, out_ch=stem_ch, ks=stem_ks, activation=activation)

            blocks = []
            in_ch = stem_ch
            for pool_sz, out_ch in zip(pool_sizes, ef_block_sizes):
                blc = nn.Sequential(
                    ResidualConcat(
                        EffBlock(in_ch=in_ch, out_ch=in_ch, ks=ef_ks,
                                 resize_factor=resize_factor, activation=activation)
                    ),
                    LocalBlock(in_ch=in_ch * 2, out_ch=out_ch, ks=ef_ks, activation=activation),
                    nn.MaxPool1d(pool_sz)
                )
                in_ch = out_ch
                blocks.append(blc)
            self.main = nn.Sequential(*blocks)

            final_ch = ef_block_sizes[-1]  # 128
            self.mapper = MapperBlock(in_features=final_ch, out_features=final_ch * 2)

            head_dim = final_ch * 2  # 256
            self.heads = nn.ModuleDict({
                "encode4_hepg2_activity": nn.Sequential(
                    nn.Linear(head_dim, head_dim),
                    nn.BatchNorm1d(head_dim),
                    nn.SiLU(),
                    nn.Linear(head_dim, 1),
                ),
            })

        def forward(self, x):
            x = self.stem(x)
            x = self.main(x)
            x = self.mapper(x)
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
            return self.heads["encode4_hepg2_activity"](x).squeeze(-1)

    model = CADENCEWrapper()

    _patch_training_imports()
    checkpoint = torch.load(
        "cadence_place/cadence_hepg2_v2/original_model.pt",
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint["model_state_dict"]

    model.load_state_dict(state_dict, strict=True)
    print("Loaded state dict (strict match)")

    model.to(device)
    model.eval()
    return model


# ============================================================================
# Main
# ============================================================================

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load val + test data
    data_dir = "physics/data/lentiMPRA_data/HepG2"
    print("Loading HepG2 data...")
    val_seqs, val_acts = load_hepg2_split(os.path.join(data_dir, "HepG2_val_with_features.tsv"))
    test_seqs, test_acts = load_hepg2_split(os.path.join(data_dir, "HepG2_test_with_features.tsv"))

    sequences = np.concatenate([val_seqs, test_seqs], axis=0)
    true_activities = np.concatenate([val_acts, test_acts], axis=0)
    n_samples = len(sequences)
    print(f"Total: {n_samples} held-out sequences (val + test)")

    # Load model
    model = build_and_load_model(device)

    # Load normalizer
    with open("cadence_place/cadence_hepg2_v2/normalizer.json") as f:
        norm = json.load(f)["encode4_hepg2"]
    norm_mean = norm["mean"][0]
    norm_std = norm["std"][0]

    # Run predictions
    all_preds = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = torch.tensor(sequences[start:end], dtype=torch.float32, device=device)
            output = model(batch)  # [B]
            all_preds.append(output.cpu().numpy())
            if start % (batch_size * 10) == 0:
                print(f"  {end}/{n_samples}")

    preds_norm = np.concatenate(all_preds, axis=0)

    # Inverse-transform predictions to original scale
    preds_orig = preds_norm * norm_std + norm_mean
    true_orig = true_activities

    # Compute metrics
    from scipy.stats import pearsonr, spearmanr

    r, _ = pearsonr(true_orig, preds_orig)
    rho, _ = spearmanr(true_orig, preds_orig)

    print(f"\nHepG2 lentiMPRA: Pearson r={r:.4f}, Spearman rho={rho:.4f}")

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7))

    x = true_orig.copy()
    y = preds_orig.copy()

    # Subsample for KDE (faster)
    rng = np.random.RandomState(42)
    kde_idx = rng.choice(len(x), min(len(x), 15000), replace=False)
    kde = gaussian_kde(np.vstack([x[kde_idx], y[kde_idx]]))
    density = kde(np.vstack([x, y]))

    order = density.argsort()
    x, y, density = x[order], y[order], density[order]

    ax.scatter(x, y, c=density, cmap="inferno", s=3, alpha=0.6,
               rasterized=True, edgecolors="none")

    lo = min(x.min(), y.min()) - 0.3
    hi = max(x.max(), y.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color="#888888", lw=1.5, alpha=0.7, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Measured Activity (log\u2082 RNA/DNA)", fontsize=14)
    ax.set_ylabel("CADENCE Predicted Activity", fontsize=14)
    ax.set_title(
        "CADENCE Prediction vs. Measured Activity\n"
        f"HepG2 lentiMPRA \u2014 {n_samples:,} held-out sequences (val + test)",
        fontsize=15, fontweight="bold")

    text = (f"Pearson $r$ = {r:.3f}\n"
            f"Spearman $\\rho$ = {rho:.3f}\n"
            f"$n$ = {n_samples:,}")
    box = dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.92)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=13,
            va="top", bbox=box)
    ax.tick_params(labelsize=12)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig("cadence_hepg2_predictions.png", dpi=200,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"\nSaved: cadence_hepg2_predictions.png")


if __name__ == "__main__":
    main()
