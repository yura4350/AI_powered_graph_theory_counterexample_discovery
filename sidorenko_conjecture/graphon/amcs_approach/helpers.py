import numpy as np
import itertools
import random
from itertools import product
from numba import njit  # library to optimize performance

def count_homomorphisms_backtrack(H_adj, G_adj):
    h = H_adj.shape[0]
    g = G_adj.shape[0]
    count = 0
    mapping = [-1] * h

    def backtrack(pos):
        nonlocal count
        if pos == h:
            count += 1
            return
        for candidate in range(g):
            valid = True
            for prev in range(pos):
                if (H_adj[pos, prev] == 1 and G_adj[candidate, mapping[prev]] != 1) or \
                   (H_adj[prev, pos] == 1 and G_adj[mapping[prev], candidate] != 1):
                    valid = False
                    break
            if valid:
                mapping[pos] = candidate
                backtrack(pos + 1)
                mapping[pos] = -1

    backtrack(0)
    return count

def compute_t_H_G_backtrack(H_adj, G_adj):
    g = G_adj.shape[0]
    total_maps = g ** H_adj.shape[0]
    hom_count = count_homomorphisms_backtrack(H_adj, G_adj)
    return hom_count / total_maps

def average_degree(G_adj):
    n = G_adj.shape[0]
    total_degree = np.sum(G_adj)
    return total_degree / (n ** 2)

def sidorenko_ratio(H_adj, G_adj):
    t = compute_t_H_G_backtrack(H_adj, G_adj)
    p = average_degree(G_adj)
    num_edges_H = int(np.sum(H_adj) // 2)
    return (p ** num_edges_H / t - 1)

@njit # numba decorator
def _compute_t_recursive(n, edges, W_block, assignment, pos):
    """
    A Numba-friendly recursive helper to replace itertools.product.
    This function calculates the sum of probabilities for all possible assignments.
    """
    # Base Case: If the assignment is fully built (pos == n)
    if pos == n:
        # Calculate the probability for this one complete assignment
        prob = 1.0
        for i in range(len(edges)):
            u, v = edges[i]
            prob *= W_block[assignment[u], assignment[v]]
        return prob

    # Recursive Step: Iterate through possibilities for the current position
    total_prob = 0.0
    num_blocks = W_block.shape[0]
    for i in range(num_blocks):
        assignment[pos] = i
        # Recurse to fill the next position and add the result
        total_prob += _compute_t_recursive(n, edges, W_block, assignment, pos + 1)
    
    return total_prob

@njit # numba decorator
def compute_t_G_W(H, W_block):
    """
    This function uses the recursive helper to perform its calculation
    in a Numba-compatible way.
    """
    n = H.shape[0]
    
    # Create a list of edges.
    # Creating a numpy array for numba
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if H[i, j] == 1:
                edge_list.append((i, j))
    edges = np.array(edge_list, dtype=np.int64)

    num_blocks = W_block.shape[0]
    block_volume = 1.0 / num_blocks
    
    # call to the recursive function
    assignment = np.zeros(n, dtype=np.int64)
    total_prob_sum = _compute_t_recursive(n, edges, W_block, assignment, 0)
    
    t = total_prob_sum * (block_volume ** n)
    return t

def sidorenko_gap(H, W_block):
    t = compute_t_G_W(H, W_block)
    p = np.mean(W_block)
    num_edges = int(np.sum(H) // 2)
    return p ** num_edges - t, t, p ** num_edges

def perturb(W):
    W_new = W.copy()
    n = W.shape[0]
    all_indices = [(i, j) for i in range(n) for j in range(n)]
    random.shuffle(all_indices)
    increase_indices = all_indices[:1]
    decrease_indices = all_indices[1:2]

    # Generating a random change value for both increasing and decreasing
    increase_change = random.uniform(0.001, 0.005)
    decrease_change = random.uniform(0.001, 0.005)

    for i, j in increase_indices:
        if W_new[i, j] + increase_change > 1.0:
            return W
    for i, j in decrease_indices:
        if W_new[i, j] - decrease_change < 0.0:
            return W

    for i, j in increase_indices:
        W_new[i, j] += increase_change
        W_new[j, i] += increase_change
    for i, j in decrease_indices:
        W_new[i, j] -= decrease_change
        W_new[j, i] -= decrease_change
    return W_new

def optimize_graphon(H, W, steps=100):
    best_gap, best_t, best_p_e = sidorenko_gap(H, W)
    W_best = W.copy()

    for step in range(steps):
        W_new = perturb(W)
        new_gap, new_t, new_p_e = sidorenko_gap(H, W_new)
        delta = abs(new_gap) - abs(best_gap)

        if delta < 0:
            W = W_new
            if abs(new_gap) < abs(best_gap):
                W_best = W.copy()
                best_gap, best_t, best_p_e = new_gap, new_t, new_p_e

    return W_best, best_gap, best_t, best_p_e
