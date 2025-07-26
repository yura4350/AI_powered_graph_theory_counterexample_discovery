import numpy as np
from time import time
from helpers import calculate_gap, perturb_M

def NMCS(current_M, steps, score_function):
    """
    Local Monte Carlo search over the space of matrices M.
    """
    best_M_local = current_M.copy()
    best_score_local = score_function(best_M_local)

    for _ in range(steps):
        candidate_M = perturb_M(best_M_local)
        candidate_score = score_function(candidate_M)
        
        if candidate_score > best_score_local:
            best_M_local = candidate_M
            best_score_local = candidate_score
            
    return best_M_local


def AMCS(initial_M, max_depth=5, max_level=3):
    """
    AMCS algorithm to search for a matrix M that maximizes the violation score.
    """
    score_function = lambda M: -calculate_gap(M)

    print("--- Starting AMCS with Eigenvalue Formulation ---")
    current_M = initial_M.copy()
    current_score = score_function(current_M)
    print(f"Initial Score (violation): {current_score:.4e}")
    print(f"Initial Conjecture Gap: {-current_score:.4e}")

    depth = 0
    level = 1
    
    while level <= max_level:
        nmcs_steps = 20 * level

        next_M = NMCS(current_M, steps=nmcs_steps, score_function=score_function)
        next_score = score_function(next_M)

        print(f"Lvl {level}, Dpt {depth}: Current best score = {current_score:.4e}")

        if next_score > current_score:
            print(f"  > New best score found: {next_score:.4e}")
            current_M = next_M.copy()
            current_score = next_score
            depth = 0
        elif depth < max_depth:
            depth += 1
        else:
            depth = 0
            level += 1
            if level <= max_level:
                print(f"\nIncreasing search intensity to level {level}...\n")
            
    # Final Results
    final_score = score_function(current_M)
    final_gap = final_score
    
    print("\n--- AMCS Finished ---")
    print("Final optimized M:")
    print(np.round(current_M, 3))
    print(f"Final Conjecture Gap: {final_gap:.4e}")

    if final_gap > 0:
        print("\n\n*** POTENTIAL COUNTEREXAMPLE FOUND! ***")
        print("The conjecture gap is negative.")
    else:
        print("\nNo counterexample found. The conjecture holds for the final matrix.")
    
    return current_M, final_gap