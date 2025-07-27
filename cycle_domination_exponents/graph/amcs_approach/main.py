import numpy as np
from time import time
from amcs import AMCS

def create_random_graph(n, p):
    adj_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < p:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    return adj_matrix

if __name__ == "__main__":
    # Odd-cycle adjacency matrices for C5 and C3
    C5 = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ])

    C3 = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    
    ''' Define the initial target graph T '''
    # A random n-vertex graph to start the search.
    # A larger 'n' will increase computation time.
    n_vertices = 20
    initial_T = create_random_graph(n_vertices, p=0.5)

    print("AMCS Search for C(C5, C3) Lower Bound using Undirected Graphs (with homlib)")
    print(f"Goal: Find a {n_vertices}-vertex graph T that maximizes log(t(C5,T)) / log(t(C3,T))")
    print("Initial random T:")
    print(initial_T)

    # Run the AMCS algorithm
    start_time = time()
    T_final, final_exponent = AMCS(
        H1=C5, 
        H2=C3, 
        initial_T=initial_T,
        max_depth=40, 
        max_level=40
    )
    print(f"\nTotal search time: {time() - start_time:.2f} seconds")