#!/usr/bin/env python3
"""
Generate UMAP visualization coordinates for LaTeX/TikZ.
Extracts real latent space embeddings from PhysicsVAE results.
"""

import numpy as np
from pathlib import Path

BASE_DIR = Path("/home/bcheng/sequence_optimization/FUSEMAP")
EMBEDDING_DIR = BASE_DIR / "physics/results/human/05_regime_discovery"
OUTPUT_DIR = BASE_DIR / "paper/scripts/output"

def load_umap_embeddings(cell_type='K562'):
    """Load real UMAP embeddings from regime discovery results."""
    embedding_file = EMBEDDING_DIR / f"embedding_{cell_type}.npz"

    if not embedding_file.exists():
        print(f"Warning: {embedding_file} not found")
        return None, None

    data = np.load(embedding_file)
    embedding = data['embedding']  # (N, 2) UMAP coordinates
    labels = data['labels']  # Cluster assignments

    return embedding, labels

def subsample_for_tikz(embedding, labels, n_points=500):
    """Subsample points for TikZ visualization (avoid too many points)."""
    np.random.seed(42)

    n_total = len(embedding)
    indices = np.random.choice(n_total, min(n_points, n_total), replace=False)

    return embedding[indices], labels[indices]

def normalize_coords(embedding):
    """Normalize coordinates to [0, 10] range for TikZ."""
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

    normalized = embedding.copy()
    normalized[:, 0] = 10 * (embedding[:, 0] - x_min) / (x_max - x_min)
    normalized[:, 1] = 10 * (embedding[:, 1] - y_min) / (y_max - y_min)

    return normalized

def output_tikz_umap(embedding, labels, output_file, cell_type):
    """Write UMAP coordinates in TikZ-compatible format."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get unique clusters
    unique_labels = np.unique(labels)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    with open(output_file, 'w') as f:
        f.write(f"% UMAP coordinates for {cell_type}\n")
        f.write(f"% Total points: {len(embedding)}\n")
        f.write(f"% Clusters: {len(unique_labels)}\n\n")

        for i, label in enumerate(unique_labels):
            if label == -1:
                color = 'gray'
                cluster_name = 'noise'
            else:
                color = colors[label % len(colors)]
                cluster_name = f'cluster{label}'

            mask = labels == label
            cluster_points = embedding[mask]

            f.write(f"% Cluster {label}: {len(cluster_points)} points\n")
            f.write(f"\\addplot[only marks, mark=*, mark size=0.5pt, {color}, opacity=0.5] coordinates {{\n")

            for x, y in cluster_points:
                f.write(f"  ({x:.3f},{y:.3f})\n")

            f.write("};\n\n")

    print(f"Wrote {len(embedding)} UMAP coordinates to {output_file}")

def generate_all_cell_types():
    """Generate UMAP coordinates for all available cell types."""
    cell_types = ['K562', 'HepG2', 'WTC11']

    for cell_type in cell_types:
        print(f"\nProcessing {cell_type}...")

        embedding, labels = load_umap_embeddings(cell_type)
        if embedding is None:
            continue

        print(f"  Original: {len(embedding)} points, {len(np.unique(labels))} clusters")

        # Subsample for TikZ
        embedding_sub, labels_sub = subsample_for_tikz(embedding, labels, n_points=500)

        # Normalize coordinates
        embedding_norm = normalize_coords(embedding_sub)

        # Output TikZ
        output_file = OUTPUT_DIR / f"umap_{cell_type.lower()}_coords.tex"
        output_tikz_umap(embedding_norm, labels_sub, output_file, cell_type)

        # Also save as NPZ
        npz_file = OUTPUT_DIR / f"umap_{cell_type.lower()}_subsample.npz"
        np.savez(npz_file, embedding=embedding_norm, labels=labels_sub)
        print(f"  Also saved to {npz_file}")

def print_cluster_stats():
    """Print cluster statistics for paper."""
    print("\n=== UMAP Embedding Statistics ===")

    for cell_type in ['K562', 'HepG2', 'WTC11']:
        embedding, labels = load_umap_embeddings(cell_type)
        if embedding is None:
            continue

        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n{cell_type}:")
        print(f"  Total sequences: {len(embedding)}")
        print(f"  Clusters: {len(unique)}")
        for label, count in zip(unique, counts):
            pct = 100 * count / len(embedding)
            print(f"    Cluster {label}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    print("Generating UMAP coordinates from real embeddings...")

    generate_all_cell_types()
    print_cluster_stats()
