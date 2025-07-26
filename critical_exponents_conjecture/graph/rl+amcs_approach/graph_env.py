import gymnasium as gym
from gymnasium import spaces
import numpy as np
from helpers import get_graph_exponent

class GraphEnv(gym.Env):
    """The RL environment, modified to support setting a custom starting state."""
    def __init__(self, n, H1, H2):
        super().__init__()
        self.n = n
        self.H1 = H1
        self.H2 = H2
        self.num_possible_edges = self.n * (self.n - 1) // 2
        self.upper_tri_indices = np.triu_indices(self.n, k=1)
        self.observation_space = spaces.MultiBinary(self.n * self.n)
        self.action_space = spaces.Discrete(self.num_possible_edges)
        self._custom_initial_state = None
        self.current_T = self._create_random_graph()

    def set_initial_state(self, graph_adj):
        """Allows the main loop to set the starting point for the next RL session."""
        self._custom_initial_state = graph_adj.copy()

    def _create_random_graph(self, p=0.5):
        adj_matrix = np.zeros((self.n, self.n), dtype=np.int8)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if np.random.rand() < p:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        return adj_matrix

    def _action_to_edge(self, action):
        return self.upper_tri_indices[0][action], self.upper_tri_indices[1][action]

    def step(self, action):
        i, j = self._action_to_edge(action)
        self.current_T[i, j] = 1 - self.current_T[i, j]
        self.current_T[j, i] = 1 - self.current_T[j, i]
        exponent = get_graph_exponent(self.current_T, self.H1, self.H2)
        observation = self.current_T.flatten()
        info = {'exponent': exponent}
        return observation, exponent, False, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._custom_initial_state is not None:
            self.current_T = self._custom_initial_state
            self._custom_initial_state = None  # Use only once
        else:
            self.current_T = self._create_random_graph()
        
        observation = self.current_T.flatten()
        initial_exponent = get_graph_exponent(self.current_T, self.H1, self.H2)
        info = {'exponent': initial_exponent}
        return observation, info