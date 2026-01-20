#!/usr/bin/env python3
"""
Calculate minor groove width scores for DNA sequences
Based on analysis of existing labeled data
"""

def calculate_minor_groove_score(sequence):
    """
    Calculate minor groove width score based on AT runs and patterns.
    
    Returns a score between 0 and 1 where:
    - 0 = narrow minor groove (GC-rich, no AT runs)
    - 1 = wide minor groove (long AT runs)
    """
    seq = sequence.upper()
    if len(seq) <= 1:
        return 0.0
    
    # Count consecutive runs of each nucleotide
    runs = []
    current_base = seq[0]
    current_run = 1
    
    for i in range(1, len(seq)):
        if seq[i] == current_base:
            current_run += 1
        else:
            runs.append((current_base, current_run))
            current_base = seq[i]
            current_run = 1
    
    # Add final run
    runs.append((current_base, current_run))
    
    # Score based on AT runs
    total_score = 0
    scored_bases = 0
    
    for base, run_length in runs:
        if base in ['A', 'T'] and run_length >= 2:
            # Score AT runs
            if run_length >= 6:
                run_score = 1.0
            elif run_length == 5:
                run_score = 0.9
            elif run_length == 4:
                run_score = 0.8
            elif run_length == 3:
                run_score = 0.6
            else:  # run_length == 2
                run_score = 0.3
            
            total_score += run_score * run_length
            scored_bases += run_length
        else:
            # GC bases or single AT bases contribute 0
            if base in ['G', 'C']:
                scored_bases += run_length
            else:  # single A or T
                scored_bases += run_length
    
    if scored_bases == 0:
        return 0.0
    
    return total_score / scored_bases

def calculate_minor_groove_optimized(sequence):
    """
    Optimized version that matches the observed data pattern better.
    """
    seq = sequence.upper()
    if len(seq) == 0:
        return 0.0
    
    score = 0.0
    n_positions = len(seq)
    
    # Look for specific patterns that contribute to minor groove width
    for i in range(len(seq)):
        position_score = 0.0
        
        # Check for AT runs at this position
        if seq[i] in ['A', 'T']:
            # Count consecutive AT bases centered on this position
            run_start = i
            run_end = i
            
            # Extend backwards
            while run_start > 0 and seq[run_start - 1] == seq[i]:
                run_start -= 1
            
            # Extend forwards
            while run_end < len(seq) - 1 and seq[run_end + 1] == seq[i]:
                run_end += 1
            
            run_length = run_end - run_start + 1
            
            # Score based on run length
            if run_length >= 6:
                position_score = 1.0
            elif run_length == 5:
                position_score = 0.85
            elif run_length == 4:
                position_score = 0.7
            elif run_length == 3:
                position_score = 0.45
            elif run_length == 2:
                position_score = 0.2
            else:
                position_score = 0.0
        
        score += position_score
    
    return score / n_positions

# Test the function
if __name__ == "__main__":
    test_sequences = [
        ('AAAAATTTTTAAGAATTTTT', 0.7222),
        ('GCGCCGCGCCGCCGCGGCGC', 0.0000),
        ('TTTAAACCCAAACCCCCCGG', 0.1667),
        ('ACATTTAAATTTAAATTTAA', 0.5556),
        ('CATCGATCACAGTGGGATCT', 0.0000),
        ('AAAAAACAAAAATAAAAAAA', 0.7222),
        ('GGGGGGGTTTTTTTTTTTTT', 0.6111),
    ]
    
    print("Testing minor groove calculation:")
    print("=" * 50)
    
    for seq, true_score in test_sequences:
        calc1 = calculate_minor_groove_score(seq)
        calc2 = calculate_minor_groove_optimized(seq)
        
        print(f"Sequence: {seq}")
        print(f"  True:     {true_score:.4f}")
        print(f"  Method 1: {calc1:.4f} (diff: {abs(calc1-true_score):.4f})")
        print(f"  Method 2: {calc2:.4f} (diff: {abs(calc2-true_score):.4f})")
        print()