import numpy as np
from time import time
from amcs import AMCS_M

if __name__ == "__main__":
    # Define the dimensions of the matrix M to search over.
    m, n = 4, 4
    
    # Define an initial random M with non-negative entries in [0, 1]. May be changed for [0, inf)
    # A non-uniform start might be more interesting.
    # initial_M = np.array([[0., 0., 0.31, 0.464], 
    #            [0., 0.136, 0., 0.], 
    #            [0.437, 0.267, 0., 0.025], 
    #            [0., 0.499, 0., 0.]])

    # Randomized start option
    initial_M = np.random.rand(m, n)
    
    print(f"--- Searching for counterexample to Sidorenko's Conjecture for K5,5 \\ C10 ---")
    print(f"--- Using eigenvalue formulation on {m}x{n} matrices ---")
    print("Initial M:")
    print(np.round(initial_M, 3))

    # Run the AMCS Optimization
    start_time = time()
    M_final, final_gap = AMCS_M(initial_M, max_depth=80, max_level=80)
    print(f"\nTotal search time: {time() - start_time:.2f} seconds")