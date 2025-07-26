import numpy as np
import random
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
    if t_H1_W <= 1e-9 or t_H2_W <= 1e-9 or t_H2_W >= 1.0:
        return 0.0 # Invalid or uninteresting region

    # The formula for the exponent
    exponent = np.log(t_H1_W) / np.log(t_H2_W)
    return exponent

def compute_homomorphism_density(H_adj, W_matrix):
    """
    This function uses the homlib library to compute the homomorphism
    density of graph H in the graphon W.
    """
    # Convert the NumPy adjacency matrix for H into a homlib Graph object.
    H_lib = Graph(H_adj.tolist())

    # Convert the NumPy matrix for W into a homlib Graphon object.
    W_lib = Graphon(W_matrix.tolist())

    # Call the library's function to compute the normalized density.
    density = countHomGraphon(H_lib, W_lib)
    
    return density

def perturb_W(W, eps=1.0):
    """
    Creates a symmetric perturbation to the graphon matrix W for AMCS.
    """
    W_new = W.copy()
    n = W.shape[0]

    # Perturb a random number of elements
    num_to_perturb = random.randint(1, max(1, n // 2))

    for _ in range(num_to_perturb):
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        change = random.uniform(-0.1, 0.1) * eps
        W_new[i, j] = np.clip(W_new[i, j] + change, 0, 1)
        # Maintain symmetry
        if i != j:
            W_new[j, i] = np.clip(W_new[j, i] + change, 0, 1)

    return W_new

def strong_perturbation(graphon, strength=0.3):
    # Add Gaussian noise to the matrix
    noise = np.random.normal(loc=0.0, scale=strength, size=graphon.shape)
    perturbed_graphon = graphon + noise
    
    # Ensure the result is still a valid graphon (symmetric, values in [0, 1])
    perturbed_graphon = (perturbed_graphon + perturbed_graphon.T) / 2
    return np.clip(perturbed_graphon, 0, 1)