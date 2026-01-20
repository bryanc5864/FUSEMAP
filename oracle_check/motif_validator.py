"""
Motif Validator for OracleCheck

Uses JASPAR PWM databases for species-specific motif screening.
Replaces the original MicroMotif statistics with direct PWM scanning.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import re


# JASPAR motif file paths by species/kingdom
JASPAR_MOTIF_PATHS = {
    "human": Path("/home/bcheng/sequence_optimization/FUSEMAP/data/motifs/jaspar_raw/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.zip"),
    "vertebrate": Path("/home/bcheng/sequence_optimization/FUSEMAP/data/motifs/jaspar_raw/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.zip"),
    "fly": Path("/home/bcheng/sequence_optimization/FUSEMAP/data/mpra_data/JASPAR2024_CORE_insects_non-redundant_pfms_meme.txt"),
    "insect": Path("/home/bcheng/sequence_optimization/FUSEMAP/data/mpra_data/JASPAR2024_CORE_insects_non-redundant_pfms_meme.txt"),
    "plant": Path("/home/bcheng/sequence_optimization/FUSEMAP/data/motifs/jaspar_raw/JASPAR2024_CORE_plants_non-redundant_pfms_meme.zip"),
    "yeast": Path("/home/bcheng/sequence_optimization/FUSEMAP/data/mpra_data/JASPAR2024_CORE_fungi_non-redundant_pfms_meme.txt"),
    "fungi": Path("/home/bcheng/sequence_optimization/FUSEMAP/data/mpra_data/JASPAR2024_CORE_fungi_non-redundant_pfms_meme.txt"),
}

# Mapping from cell type to species
CELLTYPE_TO_SPECIES = {
    "K562": "human",
    "HepG2": "human",
    "WTC11": "human",
    "S2": "fly",
}

# Mapping from dataset to species
DATASET_TO_SPECIES = {
    "encode4_k562": "human",
    "encode4_hepg2": "human",
    "encode4_wtc11": "human",
    "deepstarr": "fly",
    "jores_arabidopsis": "plant",
    "jores_maize": "plant",
    "jores_sorghum": "plant",
    "dream_yeast": "yeast",
}


@dataclass
class PWM:
    """Position Weight Matrix for a transcription factor."""
    name: str
    matrix: np.ndarray  # [4, length] - rows are A, C, G, T
    consensus: str = ""

    def __post_init__(self):
        if not self.consensus:
            self.consensus = self._compute_consensus()

    def _compute_consensus(self) -> str:
        """Compute consensus sequence from PWM."""
        bases = ['A', 'C', 'G', 'T']
        consensus = []
        for i in range(self.matrix.shape[1]):
            max_idx = np.argmax(self.matrix[:, i])
            consensus.append(bases[max_idx])
        return ''.join(consensus)

    @property
    def length(self) -> int:
        return self.matrix.shape[1]


@dataclass
class MotifHit:
    """A single motif hit in a sequence."""
    motif_name: str
    position: int
    strand: str  # '+' or '-'
    score: float
    pvalue: float = 1.0
    sequence: str = ""


@dataclass
class MotifValidationResult:
    """Result of motif validation for a sequence."""
    passed: bool
    total_hits: int
    unique_motifs: int
    total_binding_load: float
    max_score: float
    hits_per_motif: Dict[str, int]
    motif_diversity: float  # Fraction of motifs with hits
    top_motifs: List[Tuple[str, int, float]]  # (name, count, max_score)
    flags: List[str]
    message: str


class MotifScanner:
    """
    Scans sequences for transcription factor binding motifs using JASPAR PWMs.
    """

    def __init__(
        self,
        species: str = "human",
        pseudocount: float = 0.01,
        score_threshold: float = 0.8,  # Fraction of max possible score
    ):
        """
        Initialize motif scanner.

        Args:
            species: Species for motif selection (human, fly, plant, yeast)
            pseudocount: Pseudocount for PWM log-odds conversion
            score_threshold: Minimum score as fraction of max score
        """
        self.species = species
        self.pseudocount = pseudocount
        self.score_threshold = score_threshold
        self.pwms: Dict[str, PWM] = {}
        self.log_odds: Dict[str, np.ndarray] = {}

        self._load_motifs()

    def _load_motifs(self):
        """Load JASPAR motifs for the species."""
        motif_path = JASPAR_MOTIF_PATHS.get(self.species)

        if motif_path is None:
            print(f"Warning: No motif file for species {self.species}")
            return

        if not motif_path.exists():
            # Try unzipped version
            txt_path = motif_path.with_suffix('.txt')
            if txt_path.exists():
                motif_path = txt_path
            else:
                print(f"Warning: Motif file not found: {motif_path}")
                return

        # Parse MEME format
        self._parse_meme_file(motif_path)

        # Convert to log-odds
        self._compute_log_odds()

        print(f"Loaded {len(self.pwms)} motifs for {self.species}")

    def _parse_meme_file(self, path: Path):
        """Parse MEME format PWM file."""
        if str(path).endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(path) as zf:
                # Read ALL meme files inside (JASPAR distributes as individual files)
                for name in zf.namelist():
                    if name.endswith('.meme') or name.endswith('.txt'):
                        with zf.open(name) as f:
                            content = f.read().decode('utf-8')
                            self._parse_meme_content(content)
        else:
            with open(path) as f:
                content = f.read()
                self._parse_meme_content(content)

    def _parse_meme_content(self, content: str):
        """Parse MEME format content."""
        # Split by MOTIF
        motif_blocks = re.split(r'\nMOTIF\s+', content)

        for block in motif_blocks[1:]:  # Skip header
            lines = block.strip().split('\n')
            if not lines:
                continue

            # First line has motif name
            name_parts = lines[0].split()
            name = name_parts[0]

            # Find letter-probability matrix
            matrix_lines = []
            in_matrix = False

            for line in lines[1:]:
                if 'letter-probability matrix' in line.lower():
                    in_matrix = True
                    continue
                if in_matrix:
                    stripped = line.strip()
                    if stripped and stripped[0].isdigit():
                        values = [float(x) for x in stripped.split()]
                        if len(values) == 4:
                            matrix_lines.append(values)
                    elif line.startswith('URL') or line.startswith('MOTIF'):
                        break

            if matrix_lines:
                # Matrix is [length, 4], transpose to [4, length]
                matrix = np.array(matrix_lines).T
                self.pwms[name] = PWM(name=name, matrix=matrix)

    def _compute_log_odds(self):
        """Convert PWMs to log-odds scores."""
        background = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform background

        for name, pwm in self.pwms.items():
            # Add pseudocount
            matrix = pwm.matrix + self.pseudocount
            matrix = matrix / matrix.sum(axis=0, keepdims=True)

            # Log-odds
            log_odds = np.log2(matrix / background[:, np.newaxis])
            self.log_odds[name] = log_odds

    def score_sequence(self, sequence: str, pwm_name: str) -> List[Tuple[int, float, str]]:
        """
        Score a sequence against a single PWM.

        Returns:
            List of (position, score, strand) tuples for hits above threshold
        """
        if pwm_name not in self.log_odds:
            return []

        log_odds = self.log_odds[pwm_name]
        pwm_len = log_odds.shape[1]

        if len(sequence) < pwm_len:
            return []

        # Encode sequence
        seq_upper = sequence.upper()
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        hits = []
        max_possible = log_odds.max(axis=0).sum()
        threshold = max_possible * self.score_threshold

        # Scan forward strand
        for i in range(len(seq_upper) - pwm_len + 1):
            subseq = seq_upper[i:i+pwm_len]
            if 'N' in subseq:
                continue

            score = sum(
                log_odds[base_to_idx[base], j]
                for j, base in enumerate(subseq)
            )

            if score >= threshold:
                hits.append((i, score, '+'))

        # Scan reverse strand
        rc_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
        rc_seq = ''.join(rc_map[b] for b in reversed(seq_upper))

        for i in range(len(rc_seq) - pwm_len + 1):
            subseq = rc_seq[i:i+pwm_len]
            if 'N' in subseq:
                continue

            score = sum(
                log_odds[base_to_idx[base], j]
                for j, base in enumerate(subseq)
            )

            if score >= threshold:
                # Convert position to forward strand coordinates
                fwd_pos = len(sequence) - i - pwm_len
                hits.append((fwd_pos, score, '-'))

        return hits

    def scan_sequence(self, sequence: str) -> List[MotifHit]:
        """
        Scan sequence against all loaded PWMs.

        Returns:
            List of MotifHit objects
        """
        all_hits = []

        for pwm_name in self.pwms:
            hits = self.score_sequence(sequence, pwm_name)
            for pos, score, strand in hits:
                all_hits.append(MotifHit(
                    motif_name=pwm_name,
                    position=pos,
                    strand=strand,
                    score=score,
                ))

        return all_hits


class MotifValidator:
    """
    Validates motif content of sequences against natural references.
    """

    def __init__(
        self,
        species: str = "human",
        reference_stats: Optional[Dict] = None,
    ):
        """
        Initialize motif validator.

        Args:
            species: Species for motif selection
            reference_stats: Pre-computed reference statistics from natural sequences
        """
        self.species = species
        self.scanner = MotifScanner(species=species)
        self.reference_stats = reference_stats or {}

        # Thresholds
        self.max_binding_load_percentile = 95
        self.min_diversity_percentile = 5

    def compute_reference_stats(
        self,
        sequences: List[str],
    ) -> Dict:
        """
        Compute reference statistics from natural sequences.

        Args:
            sequences: List of natural sequences

        Returns:
            Dictionary of reference statistics
        """
        all_stats = {
            'total_hits': [],
            'unique_motifs': [],
            'binding_load': [],
            'diversity': [],
            'per_motif_hits': {},
        }

        for seq in sequences:
            hits = self.scanner.scan_sequence(seq)

            total_hits = len(hits)
            unique_motifs = len(set(h.motif_name for h in hits))
            binding_load = sum(h.score for h in hits)
            diversity = unique_motifs / max(len(self.scanner.pwms), 1)

            all_stats['total_hits'].append(total_hits)
            all_stats['unique_motifs'].append(unique_motifs)
            all_stats['binding_load'].append(binding_load)
            all_stats['diversity'].append(diversity)

            # Per-motif counts
            for h in hits:
                if h.motif_name not in all_stats['per_motif_hits']:
                    all_stats['per_motif_hits'][h.motif_name] = []
                all_stats['per_motif_hits'][h.motif_name].append(1)

        # Compute percentiles
        self.reference_stats = {
            'total_hits': {
                'mean': np.mean(all_stats['total_hits']),
                'std': np.std(all_stats['total_hits']),
                'p5': np.percentile(all_stats['total_hits'], 5),
                'p95': np.percentile(all_stats['total_hits'], 95),
            },
            'binding_load': {
                'mean': np.mean(all_stats['binding_load']),
                'std': np.std(all_stats['binding_load']),
                'p5': np.percentile(all_stats['binding_load'], 5),
                'p95': np.percentile(all_stats['binding_load'], 95),
            },
            'diversity': {
                'mean': np.mean(all_stats['diversity']),
                'std': np.std(all_stats['diversity']),
                'p5': np.percentile(all_stats['diversity'], 5),
                'p95': np.percentile(all_stats['diversity'], 95),
            },
        }

        return self.reference_stats

    def validate(self, sequence: str) -> MotifValidationResult:
        """
        Validate motif content of a sequence.

        Args:
            sequence: DNA sequence to validate

        Returns:
            MotifValidationResult
        """
        hits = self.scanner.scan_sequence(sequence)

        total_hits = len(hits)
        unique_motifs = len(set(h.motif_name for h in hits))
        binding_load = sum(h.score for h in hits)
        diversity = unique_motifs / max(len(self.scanner.pwms), 1)
        max_score = max((h.score for h in hits), default=0.0)

        # Count hits per motif
        hits_per_motif = {}
        for h in hits:
            hits_per_motif[h.motif_name] = hits_per_motif.get(h.motif_name, 0) + 1

        # Top motifs
        top_motifs = sorted(
            [(name, count, max(h.score for h in hits if h.motif_name == name))
             for name, count in hits_per_motif.items()],
            key=lambda x: -x[1]
        )[:10]

        # Validation flags
        flags = []
        passed = True

        if self.reference_stats:
            # Check binding load
            if 'binding_load' in self.reference_stats:
                ref = self.reference_stats['binding_load']
                if binding_load > ref['p95']:
                    flags.append(f"High binding load: {binding_load:.1f} > p95={ref['p95']:.1f}")
                    passed = False
                elif binding_load < ref['p5']:
                    flags.append(f"Low binding load: {binding_load:.1f} < p5={ref['p5']:.1f}")

            # Check diversity
            if 'diversity' in self.reference_stats:
                ref = self.reference_stats['diversity']
                if diversity < ref['p5']:
                    flags.append(f"Low motif diversity: {diversity:.3f} < p5={ref['p5']:.3f}")

        if passed:
            message = f"Motif check passed ({total_hits} hits, {unique_motifs} unique)"
        else:
            message = f"Motif check failed: {'; '.join(flags)}"

        return MotifValidationResult(
            passed=passed,
            total_hits=total_hits,
            unique_motifs=unique_motifs,
            total_binding_load=binding_load,
            max_score=max_score,
            hits_per_motif=hits_per_motif,
            motif_diversity=diversity,
            top_motifs=top_motifs,
            flags=flags,
            message=message,
        )


def get_motif_validator(cell_type: str = None, dataset: str = None) -> MotifValidator:
    """
    Get appropriate motif validator for a cell type or dataset.

    Args:
        cell_type: Cell type (K562, HepG2, WTC11, S2)
        dataset: Dataset name

    Returns:
        MotifValidator configured for the appropriate species
    """
    species = "human"  # Default

    if cell_type:
        species = CELLTYPE_TO_SPECIES.get(cell_type, "human")
    elif dataset:
        species = DATASET_TO_SPECIES.get(dataset, "human")

    return MotifValidator(species=species)
