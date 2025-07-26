import numpy as np
from homlib import Graph, countHom

def compute_homomorphism_density_graph(H_adj, T_adj):
    """
    Computes the homomorphism density t(H, T) for graphs H and T.
    t(H, T) = hom(H, T) / (v(T) ** v(H))
    
    """
    v_H = H_adj.shape[0]
    v_T = T_adj.shape[0]

    if v_T == 0:
        return 0.0

    # Convert numpy arrays to homlib Graph objects
    H_lib = Graph(H_adj.tolist())
    T_lib = Graph(T_adj.tolist())

    # Calculate the number of homomorphisms using homlib
    num_hom = countHom(H_lib, T_lib)

    # Calculate the density
    density = num_hom / (v_T ** v_H)
    return density

def get_graph_exponent(T_adj, H1_adj, H2_adj):
    """
    Calculates the domination exponent C(H1, H2) for a given target graph T.
    This is the objective function (reward) for the RL agent.
    [cite: 95, 96]
    """
    # Calculate homomorphism densities for H1 and H2 in T
    t_H1_T = compute_homomorphism_density_graph(H1_adj, T_adj)
    t_H2_T = compute_homomorphism_density_graph(H2_adj, T_adj)

    # Handle edge cases to avoid math errors and invalid regions
    if t_H1_T <= 0 or t_H2_T <= 0 or t_H2_T >= 1:
        return 0.0  # Return a low score for uninteresting or invalid graphs

    # Calculate the exponent as per the paper's formula
    exponent = np.log(t_H1_T) / np.log(t_H2_T)
    return exponent