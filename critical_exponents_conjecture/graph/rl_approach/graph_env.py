import gymnasium as gym
from gymnasium import spaces
import numpy as np
from helpers import get_graph_exponent

class GraphEnv(gym.Env):
    """
    An RL environment for finding graph structures that provide a strong
    lower bound for the graph density domination exponent C(H1, H2).
    """
    def __init__(self, n, H1, H2):
        super().__init__()
        self.n = n  # Number of vertices in the target graph T
        self.H1 = H1 # Adjacency matrix for graph H1
        self.H2 = H2 # Adjacency matrix for graph H2
        
        # Helper to map from a discrete action to an edge (i, j)
        self.num_possible_edges = self.n * (self.n - 1) // 2
        self.upper_tri_indices = np.triu_indices(self.n, k=1)

        # STATE SPACE: A flattened n x n binary adjacency matrix.
        self.observation_space = spaces.MultiBinary(self.n * self.n)

        # ACTION SPACE: A discrete action for each possible edge to flip.
        self.action_space = spaces.Discrete(self.num_possible_edges)
        
        # Initialize with a random graph T
        self.current_T = self._create_random_graph()

    def _create_random_graph(self, p=0.5):
        """Helper to create a random symmetric graph with n vertices."""
        adj_matrix = np.zeros((self.n, self.n), dtype=np.int8)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if np.random.rand() < p:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        return adj_matrix

    def _action_to_edge(self, action):
        """Maps a discrete action index to an edge's (i, j) coordinates."""
        return self.upper_tri_indices[0][action], self.upper_tri_indices[1][action]

    def step(self, action):
        # Decode the action to find which edge to flip
        i, j = self._action_to_edge(action)

        # Apply the action: flip the edge in the current graph
        self.current_T[i, j] = 1 - self.current_T[i, j]
        self.current_T[j, i] = 1 - self.current_T[j, i]

        # The REWARD is the exponent itself, which we want to maximize
        exponent = get_graph_exponent(self.current_T, self.H1, self.H2)
        reward = exponent
        
        # This is a continuing task, so termination is not part of the standard loop
        terminated = False
        
        # The observation is the flattened adjacency matrix of the new graph
        observation = self.current_T.flatten()
        info = {'exponent': exponent}
        
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start each episode with a new random graph
        self.current_T = self._create_random_graph()
        observation = self.current_T.flatten()
        
        # Provide the initial exponent in the info dict
        initial_exponent = get_graph_exponent(self.current_T, self.H1, self.H2)
        info = {'exponent': initial_exponent}
        
        return observation, info