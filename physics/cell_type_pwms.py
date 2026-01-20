#!/usr/bin/env python3
"""
Cell-type specific and universal PWM definitions for multiple organisms/cell types.

Includes:
1. Human cell types (HepG2, K562, WTC11) with universal + cell-specific TFs
2. DREAM (S. cerevisiae/yeast) with yeast-specific TFs from JASPAR 2024

These motifs represent:
1. Universal/housekeeping TFs present in all human cell types
2. Cell-type specific transcriptional programs:
   - HepG2: Liver/hepatocyte program
   - K562: Erythroid/megakaryocytic program
   - WTC11: Pluripotency network
3. Organism-specific programs:
   - DREAM: S. cerevisiae (yeast) transcription factors
"""

# Universal/Housekeeping TFs (present in all cell types)
# These regulate core cellular processes across all cell types
UNIVERSAL_PWMS = [
    'MA0079.5',  # SP1 - GC-rich core promoter
    'MA0060.4',  # NFYA - CCAAT module
    'MA0506.3',  # NRF1 - Mitochondrial/housekeeping promoter
    'MA0095.4',  # YY1 - Core promoter/insulator & 3D architecture
    'MA0139.2',  # CTCF - Architectural/insulator
    'MA0088.2',  # ZNF143 - Promoter-enhancer bridge/architecture
    'MA0024.3',  # E2F1 - Cell-cycle/core promoter
    'MA0028.3',  # ELK1 - Serum response module partner (ETS family)
    'MA0083.3',  # SRF - Core serum response
    'MA0093.4',  # USF1 - E-box generalist
    'MA0526.5',  # USF2 - E-box generalist
    'MA0058.4',  # MAX - E-box/MAX network
    'MA0018.5',  # CREB1 - cAMP/PKA responsive
    'MA0099.4',  # FOS::JUN (AP-1) - Stress/AP-1 responsive
    'MA0090.4',  # TEAD1 - Hippo/YAP-TAZ effector
    'MA0809.3',  # TEAD4 - Hippo/YAP-TAZ effector
    'MA0108.3',  # TBP - TATA box core promoter
    'MA0527.2',  # ZBTB33/KAISO - Methylation-sensitive promoter
]

# HepG2 (liver/hepatocyte program)
# Core hepatocyte TFs and nuclear-receptor partners
HEPG2_PWMS = [
    'MA0114.5',  # HNF4A
    'MA0484.3',  # HNF4G
    'MA0046.3',  # HNF1A (updated to v3)
    'MA0153.2',  # HNF1B
    'MA0148.5',  # FOXA1 (updated to v5)
    'MA0047.4',  # FOXA2 (updated to v4)
    'MA1683.2',  # FOXA3 (updated to v2)
    'MA0102.5',  # CEBPA (updated to v5)
    'MA0466.4',  # CEBPB (updated to v4)
    'MA0836.3',  # CEBPD (updated to v3)
    'MA0679.3',  # ONECUT1/HNF6 (updated to v3)
    'MA0512.2',  # RXRA (updated to v2)
    'MA2338.1',  # PPARA
    'MA1148.2',  # PPARA::RXRA
    'MA2337.1',  # NR1H3/LXRα
    'MA0494.2',  # NR1H3::RXRA
    'MA0093.4',  # USF1 (updated to v4)
]

# K562 (erythroid/megakaryocytic program)
# Classic erythroid circuitry (GATA–TAL1–KLF–NFE2)
K562_PWMS = [
    'MA0035.5',  # GATA1
    'MA0036.4',  # GATA2 (updated to v4)
    'MA0140.3',  # GATA1::TAL1
    'MA0091.2',  # TAL1::TCF3 (updated to v2)
    'MA0493.3',  # KLF1
    'MA1516.2',  # KLF3
    'MA0841.2',  # NFE2 (updated to v2)
    'MA0501.2',  # MAF::NFE2
    'MA0496.4',  # MAFK (updated to v4)
    'MA0495.4',  # MAFF
    'MA1633.2',  # BACH1 (updated to v2)
    'MA0060.4',  # NFYA (updated to v4)
    'MA0502.3',  # NFYB (updated to v3)
    'MA0150.3',  # NFE2L2/NRF2
    'MA0079.5',  # SP1 (updated to v5)
]

# WTC11 (human iPSC/pluripotency network)
# Pluripotency relies on OCT4–SOX2–NANOG with supporting factors
WTC11_PWMS = [
    'MA0142.1',  # POU5F1::SOX2 dimer
    'MA0143.5',  # SOX2
    'MA2339.1',  # NANOG
    'MA0039.5',  # KLF4
    'MA1723.2',  # PRDM9 (closest to PRDM14 in JASPAR, v2)
    'MA0870.1',  # SOX1 (closest to SOX15 in JASPAR)
    'MA0078.3',  # SOX17 (v3)
    'MA0866.1',  # SOX21 (v1)
    'MA0515.1',  # SOX6
    'MA0514.3',  # SOX3 (v3)
    'MA0077.2',  # SOX9 (v2)
    'MA0084.2',  # SRY (v2)
    'MA0785.2',  # POU2F1/OCT1
    'MA0787.1',  # POU3F2 (closest to POU3F1/OCT6)
    # Additional pluripotency-related TFs from previous list:
    'MA0141.4',  # ESRRB
    'MA0059.2',  # MAX::MYC
]

# Drosophila S2 cell-specific TFs from JASPAR 2024
# These are the 12 S2-specific motifs from organized/Drosophila_S2/
DROSOPHILA_S2_PWMS = [
    'MA0205.3',  # Trl (Trithorax-like/GAGA factor)
    'MA0211.1',  # bap (bagpipe)
    'MA0212.1',  # bcd (bicoid)
    'MA0234.1',  # oc (ocelliless)
    'MA0254.1',  # vvl (ventral veins lacking)
    'MA0259.2',  # ARNT::HIF1A
    'MA0447.1',  # gt (giant)
    'MA0449.1',  # h (hairy)
    'MA0461.3',  # Atoh1
    'MA0476.2',  # FOS
    'MA0483.2',  # Gfi1B
    'MA0532.2',  # Stat92E
]

# Arabidopsis thaliana TFs from JASPAR 2024
# These are the 12 core plant motifs (from various plant species)
ARABIDOPSIS_PWMS = [
    'MA0001.3',  # AGL3 (MADS-box) - Arabidopsis
    'MA0005.3',  # AG (AGAMOUS) - Arabidopsis
    'MA0008.4',  # HAT5 - Arabidopsis
    'MA0020.2',  # Dof2 - Zea mays
    'MA0034.2',  # Gam1 - Hordeum vulgare
    'MA0053.1',  # MNB1A - Zea mays
    'MA0054.1',  # myb.Ph3 - Petunia
    'MA0082.2',  # squamosa - Antirrhinum
    'MA0096.1',  # bZIP910 - Antirrhinum
    'MA0097.1',  # bZIP911 - Antirrhinum
    'MA0110.4',  # ATHB-5 - Arabidopsis
    'MA0121.2',  # ARR10 - Arabidopsis
]

# Zea mays (maize) specific TFs from JASPAR 2024
# These are 54 additional maize-specific motifs
ZEA_MAYS_PWMS = [
    'MA0021.1',  # Dof3
    'MA0064.1',  # PBF
    'MA0123.1',  # abi4
    'MA1416.1',  # RAMOSA1
    'MA1417.2',  # O2 (Opaque2)
    'MA1685.2',  # ARF10
    'MA1686.1',  # ARF13
    'MA1687.1',  # ARF14
    'MA1688.2',  # ARF16
    'MA1689.2',  # ARF18
    'MA1690.2',  # ARF25
    'MA1691.2',  # ARF27
    'MA1692.2',  # ARF29
    'MA1693.2',  # ARF34
    'MA1694.2',  # ARF35
    'MA1695.2',  # ARF36
    'MA1696.2',  # ARF39
    'MA1697.2',  # ARF4
    'MA1698.2',  # ARF7
    'MA1816.2',  # O11
    'MA1817.2',  # Zm00001d020267
    'MA1818.1',  # Zm00001d052229
    'MA1819.2',  # Zm00001d005892
    'MA1820.2',  # Zm00001d024324
    'MA1821.2',  # Zm00001d020595
    'MA1822.2',  # Zm00001d018571
    'MA1823.2',  # Zm00001d027846
    'MA1824.2',  # Zm00001d005692
    'MA1825.2',  # Zm00001d044409
    'MA1828.2',  # Zm00001d038683
    'MA1829.2',  # Zm00001d035604
    'MA1830.2',  # Zm00001d015407
    'MA1831.2',  # Zm00001d031796
    'MA1832.2',  # Zm00001d002364
    'MA1833.2',  # Zm00001d049364
    'MA1834.2',  # Zm00001d034298
    'MA1835.2',  # TFLG2-Zm00001d042777
    'MA2106.1',  # Zm00001d024644
    'MA2391.1',  # Lg3 (Liguleless3)
    'MA2392.1',  # GRMZM2G135447
    'MA2408.1',  # BAD1
    'MA2409.1',  # ZmbZIP25
    'MA2410.1',  # ZmbZIP54
    'MA2411.1',  # ZmbZIP57
    'MA2412.1',  # ZmbZIP72
    'MA2413.1',  # ZmbZIP96
    'MA2414.1',  # EREB127
    'MA2415.1',  # EREB138
    'MA2416.1',  # EREB29
    'MA2417.1',  # EREB71
    'MA2418.1',  # IG1 (Indeterminate gametophyte1)
    'MA2419.1',  # UB3 (Unbranched3)
    'MA2420.1',  # SBP6
    'MA2421.1',  # SBP8
]

# Combined maize PWMs: core plant motifs + Zea mays specific
MAIZE_PWMS = ARABIDOPSIS_PWMS + ZEA_MAYS_PWMS

# DREAM (S. cerevisiae/yeast) - All 170 yeast TFs from JASPAR 2024
# These are organism-specific TFs, not combined with universal (different organism)
DREAM_PWMS = [
    'MA0265.3', 'MA0266.2', 'MA0267.2', 'MA0268.2', 'MA0269.2', 'MA0270.2',
    'MA0271.1', 'MA0272.2', 'MA0273.2', 'MA0274.1', 'MA0275.1', 'MA0276.1',
    'MA0277.1', 'MA0278.2', 'MA0279.3', 'MA0280.2', 'MA0281.3', 'MA0282.2',
    'MA0283.1', 'MA0284.3', 'MA0285.2', 'MA0286.2', 'MA0287.1', 'MA0288.1',
    'MA0289.1', 'MA0290.1', 'MA0291.1', 'MA0351.2', 'MA0292.2', 'MA0293.2',
    'MA0294.2', 'MA0420.2', 'MA0295.2', 'MA0296.2', 'MA0297.1', 'MA0299.1',
    'MA0300.2', 'MA0301.2', 'MA0302.2', 'MA0303.3', 'MA0304.1', 'MA0305.2',
    'MA0306.2', 'MA0307.1', 'MA0308.2', 'MA0309.2', 'MA0310.2', 'MA0311.1',
    'MA0312.3', 'MA0313.1', 'MA0314.3', 'MA0316.2', 'MA0317.2', 'MA0327.1',
    'MA0318.1', 'MA0319.2', 'MA0320.1', 'MA0321.1', 'MA0322.1', 'MA0323.1',
    'MA0324.1', 'MA0325.2', 'MA0326.1', 'MA0328.1', 'MA0329.2', 'MA0330.1',
    'MA0331.1', 'MA0332.1', 'MA0333.2', 'MA0334.2', 'MA0336.2', 'MA0337.2',
    'MA0338.2', 'MA0339.2', 'MA0379.1', 'MA0340.1', 'MA0341.1', 'MA0342.1',
    'MA0343.2', 'MA0344.1', 'MA0347.3', 'MA0421.1', 'MA0348.2', 'MA0349.2',
    'MA0352.3', 'MA0353.1', 'MA0354.2', 'MA0355.2', 'MA0356.1', 'MA0357.1',
    'MA0358.2', 'MA0359.3', 'MA0360.2', 'MA0361.2', 'MA0362.2', 'MA0363.3',
    'MA0364.1', 'MA0365.2', 'MA0366.1', 'MA0367.2', 'MA0368.1', 'MA0369.2',
    'MA0370.1', 'MA0371.1', 'MA0372.2', 'MA0373.1', 'MA0374.2', 'MA0375.2',
    'MA0376.2', 'MA0377.2', 'MA0378.2', 'MA0380.2', 'MA0381.2', 'MA0382.3',
    'MA0384.2', 'MA0385.2', 'MA0386.2', 'MA0387.1', 'MA0388.1', 'MA0389.2',
    'MA0390.2', 'MA0391.1', 'MA0392.1', 'MA0393.1', 'MA0394.1', 'MA0395.2',
    'MA0396.2', 'MA0397.2', 'MA0398.2', 'MA0399.1', 'MA0400.2', 'MA0401.1',
    'MA0402.2', 'MA0403.3', 'MA0404.1', 'MA0431.2', 'MA0405.1', 'MA0406.2',
    'MA0407.1', 'MA0350.2', 'MA0408.2', 'MA0409.1', 'MA0410.2', 'MA0412.3',
    'MA0411.2', 'MA0422.2', 'MA0413.2', 'MA0414.2', 'MA0415.2', 'MA0416.2',
    'MA0417.1', 'MA0418.2', 'MA0419.1', 'MA0423.2', 'MA0424.2', 'MA0425.2',
    'MA0426.1', 'MA0428.2', 'MA0429.1', 'MA0430.2', 'MA0432.1', 'MA0433.2',
    'MA0434.2', 'MA0435.2', 'MA0436.2', 'MA0437.1', 'MA0438.2', 'MA0439.2',
    'MA0440.1', 'MA0441.2',
]

# Combined set of all cell-type-specific PWMs
ALL_CELL_TYPE_PWMS = list(set(HEPG2_PWMS + K562_PWMS + WTC11_PWMS))

# Cell type mapping - now includes universal PWMs for each cell type
CELL_TYPE_PWMS = {
    'HepG2': list(set(UNIVERSAL_PWMS + HEPG2_PWMS)),  # Universal + HepG2-specific
    'K562': list(set(UNIVERSAL_PWMS + K562_PWMS)),    # Universal + K562-specific
    'WTC11': list(set(UNIVERSAL_PWMS + WTC11_PWMS)),  # Universal + WTC11-specific
    'DREAM': DREAM_PWMS,  # Yeast - no universal (different organism)
    'yeast': DREAM_PWMS,  # Alias for DREAM
    'drosophila': DROSOPHILA_S2_PWMS,  # Drosophila S2 - no universal (different organism)
    'S2': DROSOPHILA_S2_PWMS,  # Alias for Drosophila
    'arabidopsis': ARABIDOPSIS_PWMS,  # Arabidopsis - core plant motifs
    'sorghum': ARABIDOPSIS_PWMS,  # Sorghum - uses core plant motifs (no JASPAR sorghum data)
    'maize': MAIZE_PWMS,  # Maize - core plant + Zea mays specific motifs
    'plant': ARABIDOPSIS_PWMS,  # Default plant alias (used for arabidopsis/sorghum)
    'all': list(set(UNIVERSAL_PWMS + ALL_CELL_TYPE_PWMS)),  # All human PWMs
    'universal': UNIVERSAL_PWMS  # Just universal PWMs
}

# JASPAR file mapping by cell type
# Paths are relative to the physics/ directory
JASPAR_FILES = {
    'HepG2': 'data/JASPAR2024_CORE_non-redundant_pfms_meme.txt',
    'K562': 'data/JASPAR2024_CORE_non-redundant_pfms_meme.txt',
    'WTC11': 'data/JASPAR2024_CORE_non-redundant_pfms_meme.txt',
    'DREAM': 'data/DREAM_data/scer_core2024_pfms.meme',  # Yeast motifs
    'yeast': 'data/DREAM_data/scer_core2024_pfms.meme',  # Alias for DREAM
    'drosophila': '../data/motifs/organized/Drosophila_S2_combined.meme',  # Drosophila S2 motifs
    'S2': '../data/motifs/organized/Drosophila_S2_combined.meme',  # Alias for Drosophila
    'arabidopsis': '../data/motifs/organized/Arabidopsis_combined.meme',  # Arabidopsis motifs
    'sorghum': '../data/motifs/organized/Arabidopsis_combined.meme',  # Sorghum uses Arabidopsis motifs
    'maize': '../data/motifs/organized/Maize_combined.meme',  # Maize-specific motifs
    'plant': '../data/motifs/organized/Arabidopsis_combined.meme',  # Default plant alias
}

# Sequence length by cell type
SEQUENCE_LENGTHS = {
    'HepG2': 230,
    'K562': 230,
    'WTC11': 230,
    'DREAM': 110,  # Yeast sequences are 110bp
    'yeast': 110,  # Alias for DREAM
    'drosophila': 249,  # Drosophila S2 sequences are 249bp
    'S2': 249,  # Alias for Drosophila
    'arabidopsis': 170,  # Arabidopsis sequences are 170bp
    'sorghum': 170,  # Sorghum sequences are 170bp
    'maize': 170,  # Maize sequences are 170bp
    'plant': 170,  # Default plant alias
}

def get_cell_type_pwms(cell_type: str):
    """
    Get PWM IDs for a specific cell type.

    Args:
        cell_type: One of 'HepG2', 'K562', 'WTC11', 'DREAM', or 'all'

    Returns:
        List of JASPAR PWM IDs for the specified cell type
    """
    if cell_type not in CELL_TYPE_PWMS:
        raise ValueError(f"Unknown cell type: {cell_type}. Must be one of {list(CELL_TYPE_PWMS.keys())}")
    return CELL_TYPE_PWMS[cell_type]

def get_jaspar_file(cell_type: str):
    """
    Get JASPAR motif file path for a specific cell type.

    Args:
        cell_type: One of 'HepG2', 'K562', 'WTC11', 'DREAM'

    Returns:
        Path to JASPAR motif file for the specified cell type
    """
    if cell_type not in JASPAR_FILES:
        raise ValueError(f"Unknown cell type: {cell_type}. Must be one of {list(JASPAR_FILES.keys())}")
    return JASPAR_FILES[cell_type]

def get_sequence_length(cell_type: str):
    """
    Get expected sequence length for a specific cell type.

    Args:
        cell_type: One of 'HepG2', 'K562', 'WTC11', 'DREAM'

    Returns:
        Expected sequence length in base pairs
    """
    if cell_type not in SEQUENCE_LENGTHS:
        raise ValueError(f"Unknown cell type: {cell_type}. Must be one of {list(SEQUENCE_LENGTHS.keys())}")
    return SEQUENCE_LENGTHS[cell_type]

def identify_cell_type(dataset_path: str):
    """
    Identify cell type from dataset path.

    Args:
        dataset_path: Path to the dataset file

    Returns:
        Cell type string ('HepG2', 'K562', 'WTC11', 'DREAM', 'drosophila', 'plant') or None if not identified
    """
    import os
    path_lower = os.path.abspath(dataset_path).lower()

    # Check path for cell type identifiers
    if 'dream' in path_lower or 'yeast' in path_lower:
        return 'yeast'
    elif 's2_data' in path_lower or 'drosophila' in path_lower:
        return 'drosophila'
    elif 'maize' in path_lower or 'zea_mays' in path_lower or 'zea mays' in path_lower:
        return 'maize'  # Maize has additional species-specific motifs
    elif 'sorghum' in path_lower:
        return 'sorghum'
    elif 'arabidopsis' in path_lower or 'tobacco' in path_lower:
        return 'arabidopsis'
    elif 'plant' in path_lower:
        return 'plant'  # Generic plant fallback
    elif 'hepg2' in path_lower:
        return 'HepG2'
    elif 'k562' in path_lower:
        return 'K562'
    elif 'wtc11' in path_lower or 'wtc_11' in path_lower or 'wtc-11' in path_lower:
        return 'WTC11'
    else:
        # Try to identify from data content
        import pandas as pd
        try:
            df = pd.read_csv(dataset_path, sep='\t', nrows=5)
            if 'condition' in df.columns:
                condition = df['condition'].iloc[0]
                if isinstance(condition, str):
                    condition_lower = condition.lower()
                    if 'dream' in condition_lower or 'yeast' in condition_lower:
                        return 'yeast'
                    elif 'drosophila' in condition_lower or 's2' in condition_lower:
                        return 'drosophila'
                    elif 'maize' in condition_lower or 'zea' in condition_lower:
                        return 'maize'
                    elif 'sorghum' in condition_lower:
                        return 'sorghum'
                    elif 'arabidopsis' in condition_lower:
                        return 'arabidopsis'
                    elif 'plant' in condition_lower:
                        return 'plant'
                    elif 'hepg2' in condition_lower:
                        return 'HepG2'
                    elif 'k562' in condition_lower:
                        return 'K562'
                    elif 'wtc11' in condition_lower or 'wtc_11' in condition_lower:
                        return 'WTC11'
            # Check for species column (plant data has this)
            if 'species' in df.columns:
                species = df['species'].iloc[0]
                if isinstance(species, str):
                    species_lower = species.lower()
                    if 'maize' in species_lower or 'zea' in species_lower:
                        return 'maize'
                    elif 'sorghum' in species_lower:
                        return 'sorghum'
                    elif 'arabidopsis' in species_lower:
                        return 'arabidopsis'
        except:
            pass

    return None