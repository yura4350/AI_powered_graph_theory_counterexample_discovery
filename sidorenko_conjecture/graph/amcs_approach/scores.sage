from numpy import argmax, abs

# Import for working with exact numbers
from sage.rings.rational_field import QQ

import sys
import os

sys.path.append('CountHom') # To gain access to homomorphism counting library

# The alias 'HomlibGraph' is crucial to avoid conflicts with Sage's native 'Graph'.
from homlib import Graph as HomlibGraph, countHom

def count_homomorphisms(H, G):
    """
    Counts homomorphisms by converting Sage graphs to homlib graphs via adjacency lists.
    """
    if G.order() == 0:
        return 0
    if H.order() == 0:
        return 1

    try:
        h_dict = H.to_dictionary()
        g_dict = G.to_dictionary()

        # Populate the adjacency matrices
        # Get the number of vertices for each graph
        num_vertices_h = H.order()
        num_vertices_g = G.order()

        # Initialize n x n matrices with zeros, where n is the number of vertices
        h_adj_matrix = [[0] * num_vertices_h for _ in range(num_vertices_h)]
        g_adj_matrix = [[0] * num_vertices_g for _ in range(num_vertices_g)]
        for u, neighbors in h_dict.items():
            for v in neighbors:
                h_adj_matrix[int(u)][int(v)] = 1
                h_adj_matrix[int(v)][int(u)] = 1

        for u, neighbors in g_dict.items():
            for v in neighbors:
                g_adj_matrix[int(u)][int(v)] = 1
                g_adj_matrix[int(v)][int(u)] = 1

        H_homlib = HomlibGraph(h_adj_matrix)
        G_homlib = HomlibGraph(g_adj_matrix)

        return countHom(H_homlib, G_homlib)

    except Exception as e:
        print(f"An error occurred within the homlib library for H: {H.edges(labels=False)} in G: {G.edges(labels=False)}")
        print(f"Error details: {e}")
        print("Falling back to Sage's default (potentially slow) method.")
        sys.exit(0)
        

def calculate_sidorenko_score(G, H_fixed, e_H, v_H_order):
    """
    Calculates the score for Sidorenko's conjecture using EXACT rational arithmetic
    to avoid floating-point precision errors.
    """
    n_G = G.order()
    m_G = G.size()

    if n_G == 0:
        return -float('inf')

    try:
        t_K2_G_rational = QQ(2 * m_G) / QQ(n_G**2)
    except ZeroDivisionError:
        t_K2_G_rational = QQ(0)

    num_homs_H_G = count_homomorphisms(H_fixed, G)
    
    try:
        t_H_G_rational = QQ(num_homs_H_G) / QQ(n_G**v_H_order)
    except ZeroDivisionError:
        t_H_G_rational = QQ(0)

    score_as_rational = (t_K2_G_rational**e_H) - t_H_G_rational
    
    return score_as_rational

def get_sidorenko_score_function(H_target_graph):
    """
    This is the 'unique score-calculating function' setup for Sidorenko.
    It takes the fixed bipartite graph H, pre-calculates its properties,
    and returns a score function suitable for AMCS (which takes only G).
    """
    if not H_target_graph.is_bipartite():
        raise ValueError("H_target_graph must be bipartite for Sidorenko's conjecture!")
    
    e_H = H_target_graph.size()
    v_H_order = H_target_graph.order()
    
    print(f"Sidorenko score function configured for H: order={v_H_order}, size={e_H}")
    # The returned lambda captures H_target_graph, e_H, v_H_order from this scope
    return lambda G_prime: calculate_sidorenko_score(G_prime, H_target_graph, e_H, v_H_order)
