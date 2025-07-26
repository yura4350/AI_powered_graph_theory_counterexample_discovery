import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numba import njit
import random

@njit
def build_M_tilde(M):
    m, n = M.shape
    mn = m * n
    M_tilde = np.zeros((mn, mn))
    sqrt_M = np.sqrt(M)
    for i in range(mn):
        for j in range(mn):
            a, b = i // n, i % n
            c, d = j // n, j % n
            if M[a, b] == 0 or M[c, d] == 0 or M[a, d] == 0:
                M_tilde[i, j] = 0
            else:
                M_tilde[i, j] = M[c, b] * sqrt_M[a, b] * sqrt_M[c, d] * M[a, d]
    return M_tilde

def calculate_gap(M):
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
    return ratio

class MatrixEnv(gym.Env):
    """The RL environment, now with the ability to set a starting state."""
    def __init__(self, m, n):
        super().__init__()
        self.m, self.n = m, n
        self.observation_space = spaces.Box(low=0, high=1, shape=(m * n,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(m * n,), dtype=np.float32)
        self._custom_initial_state = None
        self.current_M = np.random.rand(m, n) # Still need an initial value

    def set_initial_state(self, matrix):
        """Allows the main loop to set the starting point for the next training session."""
        self._custom_initial_state = matrix.copy()

    def step(self, action):
        action_matrix = action.reshape(self.m, self.n)
        self.current_M += action_matrix
        self.current_M = np.clip(self.current_M, 0, 1)
        reward = calculate_gap(self.current_M)
        observation = self.current_M.flatten().astype(np.float32)
        info = {'conjecture_gap': reward, 'violation_score': reward}
        return observation, reward, False, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Use the custom starting state if it has been set
        if self._custom_initial_state is not None:
            self.current_M = self._custom_initial_state
            self._custom_initial_state = None  # Reset after using it once
        else:
            # Otherwise, reset to a random matrix
            self.current_M = np.random.rand(self.m, self.n)
        
        observation = self.current_M.flatten().astype(np.float32)
        gap = calculate_gap(self.current_M)
        info = {'conjecture_gap': gap, 'violation_score': -gap}
        return observation, info