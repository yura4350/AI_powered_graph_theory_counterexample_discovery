import numpy as np
from time import time
from amcs import AMCS

if __name__ == "__main__":
    # Odd-cycle adjacency matrices for testing
    C3 = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
        
    C5 = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ])

    C7 = np.array([
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 0]
    ])

    C9 = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0]
    ])
    ''' Define initial W (must be symmetric and values in [0, 1]) '''
    # A random nxn starting graphon
    n = 6
    W_initial = np.random.rand(n, n)
    W_initial = (W_initial + W_initial.T) / 2 # Ensure symmetry

    print("AMCS Search for C(C5, C3) Lower Bound")
    print("Goal: Find a graphon W that maximizes log(t(C5,W)) / log(t(C3,W))")
    
    # Run the AMCS algorithm
    start_time = time()
    W_final, final_exponent = AMCS(
        H1=C9, 
        H2=C7, 
        initial_W=W_initial, 
        max_depth=20, 
        max_level=20
    )
    print(f"\nTotal search time: {time() - start_time:.2f} seconds")