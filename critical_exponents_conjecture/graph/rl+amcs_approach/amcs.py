import numpy as np
from helpers import get_graph_exponent, perturb_graph

def NMCS_for_graphs(H1, H2, current_T, steps, score_function, num_flips=1):
    """Nested Monte Carlo Search (the local search component of AMCS)."""
    best_T = current_T.copy()
    best_score = score_function(best_T)

    for _ in range(steps):
        candidate_T = perturb_graph(best_T, num_flips=num_flips)
        candidate_score = score_function(candidate_T)
        
        if candidate_score > best_score:
            best_T = candidate_T
            best_score = candidate_score
            
    return best_T, best_score

def AMCS_for_graphs(H1, H2, initial_T, max_depth=10, max_level=5):
    """Adaptive Monte Carlo Search (exploitation phase)."""
    score_function = lambda T: get_graph_exponent(T, H1, H2)

    print("--- Starting AMCS Exploitation Phase ---")
    current_T = initial_T.copy()
    current_score = score_function(current_T)
    print(f"AMCS Initial Exponent: {current_score:.6f}")

    depth = 0
    level = 1
    num_flips = max(1, current_T.shape[0] // 4)
    
    while level <= max_level:
        nmcs_steps = 15 * level 
        next_T, next_score = NMCS_for_graphs(
            H1, H2, current_T, steps=nmcs_steps, score_function=score_function, num_flips=num_flips
        )

        if next_score > current_score:
            current_T = next_T
            current_score = next_score
            depth = 0
        elif depth < max_depth:
            depth += 1
        else:
            depth = 0
            level += 1
            num_flips = max(1, num_flips - 1)
            
    final_exponent = score_function(current_T)
    print(f"AMCS Finished. Best exponent found in this phase: {final_exponent:.6f}")
    
    return current_T, final_exponent