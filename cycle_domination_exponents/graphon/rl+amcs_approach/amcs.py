import numpy as np
from helpers import get_cycle_exponent, perturb_W

def NMCS(H1, H2, current_W, steps, score_function, eps=1.0):
    """
    Local search (exploration phase) for the cycle exponent problem.
    """
    best_W = current_W.copy()
    best_score = score_function(best_W)

    for _ in range(steps):
        candidate_W = perturb_W(best_W, eps)
        candidate_score = score_function(candidate_W)
        
        if candidate_score > best_score:
            best_W = candidate_W
            best_score = candidate_score
            
    return best_W, best_score

def AMCS(H1, H2, initial_W, max_depth=5, max_level=3):
    score_function = lambda W: get_cycle_exponent(W, H1, H2)

    print("--- Starting AMCS Phase ---")
    current_W = initial_W.copy()
    current_score = score_function(current_W)
    print(f"AMCS Initial Exponent: {current_score:.6f}")

    depth = 0
    level = 1
    eps = 1.0 # Initial perturbation magnitude
    
    # Main AMCS loop
    while level <= max_level:
        nmcs_steps = 10 * level  # More steps at higher levels

        next_W, next_score = NMCS(H1, H2, current_W, steps=nmcs_steps, score_function=score_function, eps=eps)

        # Adaptive Logic
        if next_score > current_score:
            current_W = next_W.copy()
            current_score = next_score
            depth = 0
        elif depth < max_depth:
            depth += 1
        else:
            depth = 0
            level += 1
            # print(f"  -> Level exhausted. Increasing search intensity to level {level}.")
        
        eps *= 0.95 # Anneal the perturbation size
            
    final_exponent = score_function(current_W)
    print(f"AMCS Finished. Best exponent found in this phase: {final_exponent:.6f}")
    
    return current_W, final_exponent