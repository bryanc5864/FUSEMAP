#!/usr/bin/env python3
"""
Add PWM labels to plant datasets using species-specific motifs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re

# IUPAC ambiguity codes
IUPAC = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'R': [0.5, 0, 0.5, 0],      # A or G
    'Y': [0, 0.5, 0, 0.5],      # C or T
    'S': [0, 0.5, 0.5, 0],      # G or C
    'W': [0.5, 0, 0, 0.5],      # A or T
    'K': [0, 0, 0.5, 0.5],      # G or T
    'M': [0.5, 0.5, 0, 0],      # A or C
    'B': [0, 0.33, 0.33, 0.33], # C or G or T
    'D': [0.33, 0, 0.33, 0.33], # A or G or T
    'H': [0.33, 0.33, 0, 0.33], # A or C or T
    'V': [0.33, 0.33, 0.33, 0], # A or C or G
    'N': [0.25, 0.25, 0.25, 0.25],
}

# Arabidopsis motifs (12 key TF families)
ARABIDOPSIS_MOTIFS = {
    'ABF1': 'ACGTGGC',           # ABA-responsive element
    'WRKY': 'TTGACY',            # W-box
    'MYB': 'CNGTTR',             # MYB binding
    'bHLH': 'CACGTG',            # E-box/G-box
    'NAC': 'CACG',               # NAC binding
    'ERF': 'GCCGCC',             # GCC-box
    'TCP': 'GGNCCC',             # TCP binding
    'MADS': 'CCWWWWWWGG',        # CArG-box
    'HDZIP': 'CAATHATTG',        # HD-ZIP binding
    'bZIP': 'ACGTCA',            # ACGT core
    'DOF': 'AAAG',               # DOF binding
    'GATA': 'WGATAR',            # GATA binding
}

# Maize motifs
MAIZE_MOTIFS = {
    'ERF_Zm': 'CCTCCGCCGCCRCCGCCGCCG',
    'MYB_Zm': 'AAAAAATWGGATAAGGATRAG',
    'NAC_Zm': 'GRTAACTTGYTGARCAAGTTA',
    'WRKY_Zm': 'AKCGTTGACTTTT',
    'bZIP_Zm': 'RATGCTGACGTGGCA',
    'bHLH_Zm': 'CACGTGACWTKCACG',
    'G2like_Zm': 'AAARGAATATTCC',
    'TCP_Zm': 'ASAGAGGATGTGGGRCCCAC',
    'C2H2_Zm': 'MASAAAACGACAAAAAAAAA',
    'SBP_Zm': 'AATTGTACGGAC',
    'MYBrel_Zm': 'AGATATTTTTTT',
    'HDZIP_Zm': 'AAACCAATAATTGAAWWTWWW',
    'MADS_Zm': 'TTWCCAAAAAWGGAAAAAW',
    'LBD_Zm': 'TCCGCCGCCGCCKCCGCCGCC',
    'GATA_Zm': 'CATCATCATCATCATCATCATCATCAT',
    'ARF_Zm': 'TTTACGKTTTTGGCGGGAAAA',
    'Trihelix_Zm': 'TTTTTTACCGTTWT',
    'HSF_Zm': 'AGAAGCTTCTAGAAG',
    'B3_Zm': 'TTTACGTTTTTGGCGGGAAAA',
    'Dof_Zm': 'AAAAAAAARAAAAAGTAAAAA',
}

# Sorghum motifs
SORGHUM_MOTIFS = {
    'ERF_Sb': 'CCTCCGCCGCCRCCGCCGCCG',
    'MYB_Sb': 'AAAAAATWGGATAAGGATRAG',
    'NAC_Sb': 'GRTAACTTGYTGARCAAGTTA',
    'WRKY_Sb': 'ASCGTTGACTTTT',
    'bZIP_Sb': 'RATGCTGACGTGGCA',
    'bHLH_Sb': 'CACGTGACWTKCACG',
    'TCP_Sb': 'ASAGAGGATGTGGGRCCCAC',
    'SBP_Sb': 'AAATTGTACGGACA',
    'ARF_Sb': 'TTTACGKTTTTGGCGGGAAAA',
    'Trihelix_Sb': 'GGTTAACC',
    'HDZIP_Sb': 'GCATTAATTAC',
    'HSF_Sb': 'AGAAGCTTCTAGAAG',
    'GATA_Sb': 'TCATCATCATCATCA',
    'LBD_Sb': 'TCCGCCGCCGCCKCCGCCGCC',
    'B3_Sb': 'TTTACGTTTTTGGCGGGAAAA',
    'BES1_Sb': 'TCACACGTGTSAAMT',
    'TALE_Sb': 'CTCTCTCTCCCTGYCYCTGC',
    'WOX_Sb': 'TCAWTCATTCA',
    'Dof_Sb': 'TTTTTTTTTTTTTTTACTTTTTTTTTTTT',
    'AP2_Sb': 'KGGCACAGTTCCCGAGGTGAA',
}

def consensus_to_pwm(consensus: str, pseudocount: float = 0.01) -> np.ndarray:
    """Convert IUPAC consensus to PWM (Position Weight Matrix)."""
    pwm = []
    for base in consensus.upper():
        if base in IUPAC:
            probs = np.array(IUPAC[base]) + pseudocount
            probs = probs / probs.sum()
            pwm.append(probs)
        else:
            pwm.append([0.25, 0.25, 0.25, 0.25])
    return np.array(pwm)

def sequence_to_onehot(seq: str) -> np.ndarray:
    """Convert DNA sequence to one-hot encoding."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq = seq.upper()
    onehot = np.zeros((len(seq), 4))
    for i, base in enumerate(seq):
        if base in mapping:
            onehot[i, mapping[base]] = 1
        else:
            onehot[i] = 0.25  # N or unknown
    return onehot

def score_pwm(seq_onehot: np.ndarray, pwm: np.ndarray) -> np.ndarray:
    """Score all positions in sequence with PWM. Returns array of scores."""
    seq_len = seq_onehot.shape[0]
    pwm_len = pwm.shape[0]

    if seq_len < pwm_len:
        return np.array([0.0])

    scores = []
    log_pwm = np.log2(pwm + 1e-10)

    for i in range(seq_len - pwm_len + 1):
        window = seq_onehot[i:i + pwm_len]
        score = np.sum(window * log_pwm)
        scores.append(score)

    return np.array(scores)

def compute_pwm_features(sequence: str, motifs: dict) -> dict:
    """Compute PWM features for a sequence."""
    seq_onehot = sequence_to_onehot(sequence)
    features = {}

    for name, consensus in motifs.items():
        pwm = consensus_to_pwm(consensus)
        scores = score_pwm(seq_onehot, pwm)

        if len(scores) > 0:
            features[f'pwm_{name}_max'] = float(np.max(scores))
            features[f'pwm_{name}_mean'] = float(np.mean(scores))
            features[f'pwm_{name}_sum'] = float(np.sum(scores[scores > 0])) if np.any(scores > 0) else 0.0
            # Count high-affinity sites (top 10% of theoretical max)
            max_possible = pwm.shape[0] * 2  # rough estimate
            threshold = max_possible * 0.5
            features[f'pwm_{name}_hits'] = int(np.sum(scores > threshold))
        else:
            features[f'pwm_{name}_max'] = 0.0
            features[f'pwm_{name}_mean'] = 0.0
            features[f'pwm_{name}_sum'] = 0.0
            features[f'pwm_{name}_hits'] = 0

    return features

def process_dataset(input_file: str, output_file: str, motifs: dict, species: str):
    """Process a dataset and add PWM features."""
    print(f"\nProcessing {input_file}")

    df = pd.read_csv(input_file, sep='\t')
    print(f"  Loaded {len(df)} sequences")
    print(f"  Using {len(motifs)} {species} motifs")

    # Compute features for all sequences
    all_features = []
    for seq in tqdm(df['sequence'], desc=f"  Scoring {species}"):
        features = compute_pwm_features(seq, motifs)
        all_features.append(features)

    # Convert to DataFrame and merge
    features_df = pd.DataFrame(all_features)
    df_out = pd.concat([df, features_df], axis=1)

    # Save
    df_out.to_csv(output_file, sep='\t', index=False)
    print(f"  Saved to {output_file}")
    print(f"  Added {len(features_df.columns)} PWM features")

    return df_out

def main():
    base_dir = Path(__file__).parent.parent / 'data' / 'plant_data' / 'jores2021' / 'processed'

    # Species configurations
    species_config = {
        'arabidopsis': {
            'motifs': {**ARABIDOPSIS_MOTIFS},  # Use Arabidopsis-specific
            'dir': base_dir / 'arabidopsis'
        },
        'sorghum': {
            'motifs': {**ARABIDOPSIS_MOTIFS, **SORGHUM_MOTIFS},  # Arabidopsis + Sorghum
            'dir': base_dir / 'sorghum'
        },
        'maize': {
            'motifs': {**ARABIDOPSIS_MOTIFS, **MAIZE_MOTIFS},  # Arabidopsis + Maize
            'dir': base_dir / 'maize'
        },
    }

    for species, config in species_config.items():
        print(f"\n{'='*60}")
        print(f"Processing {species.upper()}")
        print(f"{'='*60}")

        data_dir = config['dir']
        motifs = config['motifs']

        for split in ['train', 'test']:
            # Check for tileformer file first, then regular
            tileformer_file = data_dir / f'{species}_{split}_tileformer.tsv'
            regular_file = data_dir / f'{species}_{split}.tsv'

            if tileformer_file.exists():
                input_file = tileformer_file
                output_file = data_dir / f'{species}_{split}_with_features.tsv'
            elif regular_file.exists():
                input_file = regular_file
                output_file = data_dir / f'{species}_{split}_with_features.tsv'
            else:
                print(f"  Warning: No file found for {species} {split}")
                continue

            process_dataset(str(input_file), str(output_file), motifs, species)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for species in species_config:
        data_dir = species_config[species]['dir']
        for f in data_dir.glob('*_with_features.tsv'):
            df = pd.read_csv(f, sep='\t')
            pwm_cols = [c for c in df.columns if c.startswith('pwm_')]
            print(f"  {f.name}: {len(df)} seqs, {len(pwm_cols)} PWM features")

if __name__ == '__main__':
    main()
