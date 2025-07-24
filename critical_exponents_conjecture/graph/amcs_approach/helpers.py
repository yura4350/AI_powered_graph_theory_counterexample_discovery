import numpy as np
import random
from homlib import Graph, countHom

def compute_homomorphism_count(H, T):
    """
    Computes the number of homomorphisms from graph H to graph T
    using the efficient 'homlib' library.

    Args:
        H (np.ndarray): Adjacency matrix of the source graph.
        T (np.ndarray): Adjacency matrix of the target graph.
        
    Returns:
        int: The total number of homomorphisms, hom(H, T).
    """
    # Convert numpy arrays to lists of lists for homlib Graph initialization
    H_list = H.tolist()
    T_list = T.tolist()

    # Create homlib Graph objects
    H_lib = Graph(H_list)
    T_lib = Graph(T_list)

    # Use the library's efficient counting function
    return countHom(H_lib, T_lib)

def compute_homomorphism_density(H, T):
    """
    Computes the homomorphism density t(H, T).
    Formula: t(H, T) = hom(H, T) / v(T)^v(H)
    """
    v_H = H.shape[0]
    v_T = T.shape[0]
    if v_T == 0:
        return 0.0
    
    hom_count = compute_homomorphism_count(H, T)
    density = hom_count / (v_T ** v_H)
    return density

def get_graph_exponent(T, H1, H2):
    """
    Calculates the domination exponent C(H1, H2) for a given target graph T.
    This is the objective function we want to maximize.
    """

    # Calculation of homomorphism density
    t_H1_T = compute_homomorphism_density(H1, T)
    t_H2_T = compute_homomorphism_density(H2, T)

    # Handle edge cases where densities might be zero or one
    if t_H1_T <= 0 or t_H2_T <= 0 or t_H2_T >= 1:
        print("YES")
        return 0.0

    # The formula for the exponent, as per the paper
    exponent = np.log(t_H1_T) / np.log(t_H2_T)
    return exponent
def perturb_graph(T, num_flips=1):
    """
    Creates a perturbation of the graph T by flipping one or more edges.

    Args:
        T (np.ndarray): The current graph's adjacency matrix.
        num_flips (int): The number of edges to flip.

    Returns:
        np.ndarray: A new graph with the specified edges flipped.
    """
    T_new = T.copy()
    n = T.shape[0]
  
    if n < 2:
        return T_new

    # Create a list of all possible edges (upper triangle)
    possible_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            possible_edges.append((i, j))
    
    if not possible_edges:
        return T_new

    # Randomly choose unique edges to flip
    edges_to_flip = random.sample(possible_edges, k=min(num_flips, len(possible_edges)))

    for i, j in edges_to_flip:
        T_new[i, j] = 1 - T_new[i, j]
        T_new[j, i] = T_new[i, j]  # Maintain symmetry
        
    return T_new