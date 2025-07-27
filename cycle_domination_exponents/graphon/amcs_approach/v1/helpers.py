import numpy as np
import random
from numba import njit
from homlib import Graph, Graphon, countHomGraphon

def get_cycle_exponent(W, H1, H2):
    """
    Calculates the domination exponent C(H1, H2) for a given graphon W.
    This is the objective function we want to maximize.
    """

    # Calculation of homomorphism_density
    t_H1_W = compute_homomorphism_density(H1, W)
    t_H2_W = compute_homomorphism_density(H2, W)
    
    # Handle edge cases where densities might be zero or one
    if t_H1_W <= 0 or t_H2_W <= 0 or t_H2_W >= 1:
        return 0.0 # Invalid or uninteresting region

    # The formula for the exponent
    exponent = np.log(t_H1_W) / np.log(t_H2_W)
    return exponent

def compute_homomorphism_density(H_adj, W_matrix):
    """
    This function uses the CountHom library to compute the homomorphism
    density of graph H in the graphon W.
    """
    H_lib = Graph(H_adj.tolist())
    W_lib = Graphon(W_matrix.tolist())

    density = countHomGraphon(H_lib, W_lib)
    
    return density


def perturb(W, eps=1.0):

    W_new = W.copy()
    n = W.shape[0]

    # Strategy 1: Perturb a larger portion of the matrix
    # Affect between 1 and n/2 elements to allow for bigger changes
    num_to_perturb = random.randint(1, max(1, n // 2))

    for _ in range(num_to_perturb):
        # Choose a random element to change
        i, j = random.randint(0, n-1), random.randint(0, n-1)

        # Strategy 2: Introduce larger, temperature-controlled changes
        # The change can be larger than before, scaled by the 'temperature'
        change = random.uniform(-0.1, 0.1) * eps

        # Apply the change and clamp the result to the valid [0, 1] range
        W_new[i, j] = np.clip(W_new[i, j] + change, 0, 1)

        # Maintain symmetry
        if i != j:
            W_new[j, i] = np.clip(W_new[j, i] + change, 0, 1)

    # Strategy 3: Occasionally make a large, disruptive change (a "jump")
    # There's a small chance (e.g., 5%) of a more drastic modification
    if random.random() < 0.05:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        # Reset a random element to a completely new random value
        new_val = random.random()
        W_new[i, j] = new_val
        if i != j:
            W_new[j, i] = new_val

    return W_new