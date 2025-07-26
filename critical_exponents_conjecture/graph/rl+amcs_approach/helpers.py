import numpy as np
import random
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
    num_hom = countHom(Graph(H_adj.tolist()), Graph(T_adj.tolist()))
    return num_hom / (v_T ** v_H)

def get_graph_exponent(T_adj, H1_adj, H2_adj):
    """
    Calculates the domination exponent C(H1, H2) for a given target graph T.
    This serves as the reward signal.
    """
    t_H1_T = compute_homomorphism_density_graph(H1_adj, T_adj)
    t_H2_T = compute_homomorphism_density_graph(H2_adj, T_adj)
    if t_H1_T <= 1e-9 or t_H2_T <= 1e-9 or t_H2_T >= 1.0:
        return 0.0
    return np.log(t_H1_T) / np.log(t_H2_T)

def perturb_graph(T_adj, num_flips=1):
    """
    Creates a new graph by flipping a specified number of edges.
    Used for local search steps in AMCS.
    """
    T_new = T_adj.copy()
    n = T_new.shape[0]
    possible_edges = list(zip(*np.triu_indices(n, k=1)))
    edges_to_flip_indices = random.sample(range(len(possible_edges)), k=min(num_flips, len(possible_edges)))
    
    for index in edges_to_flip_indices:
        i, j = possible_edges[index]
        T_new[i, j] = 1 - T_new[i, j]
        T_new[j, i] = 1 - T_new[j, i]
        
    return T_new

def strong_perturbation_graph(graph_adj, p_flip=0.15):
    """
    Applies a strong perturbation to a graph by flipping each possible edge
    with probability `p_flip`. This kicks the RL agent out of a local optimum.
    """
    new_graph = graph_adj.copy()
    n = new_graph.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p_flip:
                new_graph[i, j] = 1 - new_graph[i, j]
                new_graph[j, i] = 1 - new_graph[j, i]
    return new_graph