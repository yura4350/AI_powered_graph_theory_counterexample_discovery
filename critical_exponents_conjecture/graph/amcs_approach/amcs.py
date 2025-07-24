import numpy as np
from time import time
from helpers import get_graph_exponent, perturb_graph

def NMCS(H1, H2, current_T, steps, score_function, num_flips=1):
    """
    Nested Monte Carlo Search (local search).
    Perturbs the graph by the given number of flips to find a better score.
    """
    best_T = current_T.copy()
    # The score function is called once at the start of the local search.
    best_score = score_function(best_T, H1, H2)

    for _ in range(steps):
        # Perturb the best graph found so far in this local search
        candidate_T = perturb_graph(best_T, num_flips=num_flips)
        candidate_score = score_function(candidate_T, H1, H2)
        
        if candidate_score > best_score:
            best_T = candidate_T
            best_score = candidate_score
            
    # Return the best graph and its score from this local search
    return best_T, best_score

def AMCS(H1, H2, initial_T, max_depth=5, max_level=3):
    """
    Adaptive Monte Carlo Search with corrected annealing logic.
    """
    score_function = get_graph_exponent
    n_vertices = initial_T.shape[0]

    print("\n--- Starting AMCS for Undirected Graphs (Corrected) ---")
    current_T = initial_T.copy()
    current_score = score_function(current_T, H1, H2)
    print(f"Initial Exponent C(C{H1.shape[0]}, C{H2.shape[0]}): {current_score:.6f}")

    depth = 0
    level = 1
    # CORRECTION: Start with a larger number of flips and decrease it over time.
    num_flips = max(1, n_vertices // 2)
    
    # Main AMCS loop
    while level <= max_level:
        nmcs_steps = 10 * level

        # Pass the current number of flips to the local search
        next_T, next_score = NMCS(
            H1, H2, current_T, 
            steps=nmcs_steps, 
            score_function=score_function, 
            num_flips=num_flips
        )

        print(f"Best score (lvl {level}, dpt {depth}, flips {num_flips}): {max(next_score, current_score):.6f}")

        # Adaptive Logic
        if next_score > current_score:
            current_T = next_T.copy()
            current_score = next_score
            depth = 0
        elif depth < max_depth:
            depth += 1
        else:
            # When a level is exhausted, increase intensity and decrease perturbation size
            depth = 0
            level += 1

            # Anneal the number of flips
            num_flips = max(1, num_flips - 1)
            print(f"  -> Level exhausted. Increasing intensity to level {level}, reducing flips to {num_flips}.")
            
    final_exponent = score_function(current_T, H1, H2)
    print("\n--- AMCS Finished ---")
    print("Final optimized Graph T (Adjacency Matrix):")
    print(current_T)
    print(f"\nFound new lower bound for C(C{H1.shape[0]}, C{H2.shape[0]}): {final_exponent:.6f}")
    
    return current_T, final_exponent