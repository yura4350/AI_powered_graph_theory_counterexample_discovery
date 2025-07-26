import numpy as np
from time import time
from helpers import get_cycle_exponent, perturb

def NMCS(H1, H2, current_W, steps, score_function, eps=1.0):
    """
    Local search (exploration phase) for the cycle exponent problem.
    """
    best_W = current_W.copy()
    best_score = score_function(best_W, H1, H2)

    for _ in range(steps):
        candidate_W = perturb(best_W, eps)
        candidate_score = score_function(candidate_W, H1, H2)
        
        if candidate_score > best_score:
            best_W = candidate_W
            best_score = candidate_score
            
    return best_W

def AMCS(H1, H2, initial_W, max_depth=5, max_level=3):
    """
    Adaptive Monte Carlo Search to find a graphon that maximizes the 
    homomorphism density domination exponent C(H1, H2).
    """
    # The score to maximize is the exponent itself.
    # Function is in helpers_cycles.py
    score_function = get_cycle_exponent

    print("\n--- Starting AMCS for Cycles ---")
    current_W = initial_W.copy()
    current_score = score_function(current_W, H1, H2)
    print(f"Initial Exponent C({H1.shape[0]}, {H2.shape[0]}): {current_score:.6f}")

    depth = 0
    level = 1
    eps = 1.0 # setting the magnitude_of_size_of_random_changes
    
    # Main AMCS loop
    while level <= max_level:
        nmcs_steps = 10 * level  # More steps at higher levels

        next_W = NMCS(H1, H2, current_W, steps=nmcs_steps, score_function=score_function, eps=eps)
        next_score = score_function(next_W, H1, H2)

        print(f"Best score (lvl {level}, dpt {depth}): {max(next_score, current_score):.6f}")

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
            print(f"  -> Level exhausted. Increasing search intensity to level {level}.")
        
        eps *= 0.95 # decrease eps as the search progresses to better find local inflection points (minimums/maximums)
            
    # Final Results Output
    final_exponent = get_cycle_exponent(current_W, H1, H2)
    print("\n--- AMCS Finished ---")
    print("Final optimized W:")
    print(np.round(current_W, 4))
    print(f"\nFound new lower bound for C(C{H1.shape[0]}, C{H2.shape[0]}): {final_exponent:.6f}")
    
    return current_W, final_exponent