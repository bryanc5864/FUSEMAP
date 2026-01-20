#!/usr/bin/env python3
"""
S2 (Drosophila) specific PWM definitions for S2 cell line sequences.

These motifs represent Drosophila-specific transcription factors found in S2 cells.
"""

import torch
import numpy as np

# S2 cell line PWM names from the drosophila_celltype_pwms.pt file
S2_PWM_NAMES = [
    'Jra_Jun',        # Jun-related antigen (Drosophila Jun/AP-1)
    'Kay_Fos',        # Kayak (Drosophila Fos/AP-1)
    'Stat92E',        # STAT transcription factor
    'SREBP_HLH106',   # Sterol regulatory element binding protein
    'Pointed_ETS',    # ETS domain transcription factor
    'M1BP_Motif1',    # Motif 1 Binding Protein
    'Trl_GAGA',       # Trithorax-like (GAGA factor)
    'ttk_Tramtrack',  # Tramtrack BTB domain protein
    'suHw',           # Suppressor of Hairy wing
    'ZIPIC',          # Zinc finger protein
    'GATAe',          # GATA transcription factor
    'Serpent_srp',    # Serpent GATA factor
    'Daughterless_da', # Daughterless bHLH
    'USF_bHLH',       # Upstream stimulatory factor
    'Achaete_ac'      # Achaete bHLH transcription factor
]

def load_s2_pwms_from_torch(torch_file: str = 'data/drosophila_celltype_pwms.pt'):
    """
    Load S2 PWMs from the PyTorch file.
    
    Returns:
        Dictionary of PWM matrices {name: np.array}
    """
    pwms_data = torch.load(torch_file)
    s2_pwms = pwms_data.get('S2', {})
    
    # Convert torch tensors to numpy arrays
    pwm_dict = {}
    for name, tensor in s2_pwms.items():
        # Tensor is shape [4, length] where 4 is [A, C, G, T]
        # Need to transpose to [length, 4] for consistency with JASPAR format
        matrix = tensor.numpy().T  # Transpose to [length, 4]
        pwm_dict[name] = matrix
    
    return pwm_dict

def get_s2_pwm_ids_from_jaspar(jaspar_file: str):
    """
    Extract Drosophila-specific PWM IDs from JASPAR insects file.
    
    This function searches for known Drosophila TFs in the JASPAR file.
    """
    # Known Drosophila TF names to search for in JASPAR
    drosophila_tfs = {
        'Jun', 'Fos', 'STAT', 'Stat92E', 'SREBP', 'Pointed', 'Ets',
        'GAGA', 'Trl', 'Tramtrack', 'ttk', 'Su(Hw)', 'suHw',
        'GATA', 'Serpent', 'srp', 'Daughterless', 'da', 'USF',
        'Achaete', 'ac', 'Scute', 'sc', 'Hunchback', 'hb',
        'Kruppel', 'Kr', 'Knirps', 'kni', 'Giant', 'gt',
        'Tailless', 'tll', 'Hairy', 'h', 'Snail', 'sna',
        'Twist', 'twi', 'Dorsal', 'dl', 'Zen', 'zen',
        'Engrailed', 'en', 'Even-skipped', 'eve', 'Fushi tarazu', 'ftz',
        'Paired', 'prd', 'Orthodenticle', 'otd', 'Caudal', 'cad'
    }
    
    pwm_ids = []
    
    with open(jaspar_file, 'r') as f:
        content = f.read()
    
    # Split by MOTIF sections
    sections = content.split('MOTIF')[1:]
    
    for section in sections:
        lines = section.strip().split('\n')
        if not lines:
            continue
        
        header = lines[0].strip()
        parts = header.split()
        if len(parts) >= 2:
            motif_id = parts[0]
            motif_name = ' '.join(parts[1:])
            
            # Check if this is a Drosophila TF
            for tf_name in drosophila_tfs:
                if tf_name.lower() in motif_name.lower():
                    pwm_ids.append(motif_id)
                    break
    
    return pwm_ids

# Alternative: Use specific JASPAR IDs for known Drosophila TFs
# These are example IDs - would need to be verified from the actual JASPAR file
DROSOPHILA_JASPAR_IDS = [
    # AP-1 family
    'MA0476.1',  # FOS
    'MA0478.1',  # FOSL2
    'MA0099.4',  # FOS::JUN (could be used as proxy)
    
    # STAT family
    'MA0144.2',  # STAT3 (vertebrate, but similar to Stat92E)
    
    # ETS family  
    'MA0098.4',  # ETS1
    'MA0028.3',  # ELK1
    
    # GATA family
    'MA0035.5',  # GATA1
    'MA0036.4',  # GATA2
    
    # bHLH family
    'MA0091.2',  # TAL1::TCF3
    'MA0093.4',  # USF1
    
    # Additional Drosophila-specific if available in insects JASPAR
]

def get_s2_pwms():
    """
    Get S2-specific PWM identifiers for processing.
    
    Returns:
        List of PWM IDs/names specific to S2 cells
    """
    return S2_PWM_NAMES

def identify_s2_dataset(dataset_path: str):
    """
    Identify if a dataset is S2 data.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        True if S2 dataset, False otherwise
    """
    import os
    path_lower = os.path.abspath(dataset_path).lower()
    
    # Check for S2 indicators in path
    if 's2' in path_lower or 'drosophila' in path_lower:
        return True
    
    # Check file content for S2 indicators
    import pandas as pd
    try:
        df = pd.read_csv(dataset_path, sep='\t', nrows=5)
        # Check for S2-specific columns
        if 'Dev_log2_enrichment' in df.columns or 'Hk_log2_enrichment' in df.columns:
            return True
    except:
        pass
    
    return False