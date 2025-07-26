import numpy as np
import random
from numba import njit

@njit
def build_M_tilde(M):
    """
    Constructs the mn x mn matrix M_tilde from an m x n matrix M.
    This function is JIT-compiled with Numba for performance.
    """
    m, n = M.shape
    mn = m * n
    M_tilde = np.zeros((mn, mn))

    # Pre-calculate square roots of M's entries to speed up the inner loop
    sqrt_M = np.sqrt(M)

    # Iterate through the indices of M_tilde
    for i in range(mn):
        for j in range(mn):
            # Map flat indices i, j back to 2D indices (a,b) and (c,d)
            a = i // n
            b = i % n
            c = j // n
            d = j % n
            
            # This check handles the case where an entry in M is zero
            if M[a, b] == 0 or M[c, d] == 0 or M[a, d] == 0:
                M_tilde[i, j] = 0
            else:
                M_tilde[i, j] = M[c, b] * sqrt_M[a, b] * sqrt_M[c, d] * M[a, d]

    return M_tilde

def calculate_eigenvalue_gap(M):
    """
    Calculates the gap in the eigenvalue formulation of Sidorenko's conjecture
    for H = K5,5 \ C10.
    
    Returns:
        The value of (sum(lambda^5)) - (RHS).
        A negative value supports the conjecture for this M.
        A positive value would be a counterexample.
    """
    m, n = M.shape
    mn = m * n

    # Build the transformed matrix M_tilde
    M_tilde = build_M_tilde(M)

    # Calculate the sum of the 5th powers of its eigenvalues.
    eigenvalues = np.linalg.eig(M_tilde)[0]
    sum_lambda_5 = np.sum(eigenvalues**5).real

    # Calculate the right-hand side (RHS) of the inequality
    norm_M1 = np.sum(M) # Sum of the values in the matrix
    if norm_M1 == 0: # Avoid division by zero
        return sum_lambda_5 
    
    rhs = (norm_M1**15) / (mn**10) # Calculated RHS
    lhs = sum_lambda_5 # Calculated LHS

    # Calculate the score
    ratio = (rhs / lhs - 1) ** 3
    return -ratio

def perturb_M(M):
    """
    Perturbs a non-symmetric matrix M, values are [0, inf)
    """
    M_new = M.copy()
    m, n = M.shape
    
    # Pick a random entry to increase and one to decrease
    i_inc, j_inc = random.randint(0, m - 1), random.randint(0, n - 1)
    i_dec, j_dec = random.randint(0, m - 1), random.randint(0, n - 1)
    
    change = random.uniform(0.01, 0.05)

    # Apply perturbations. NO upper clamp on the increase - can implement if decided to cap the values of the matrix in [0, 1] range.
    M_new[i_inc, j_inc] = M_new[i_inc, j_inc] + change
    M_new[i_dec, j_dec] = max(0.0, M_new[i_dec, j_dec] - change) # Clamp at 0 - but maybe implement some other logic (do we want zeros?)
    
    return M_new