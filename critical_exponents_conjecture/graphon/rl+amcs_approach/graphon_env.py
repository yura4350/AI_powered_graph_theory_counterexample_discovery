import gymnasium as gym
from gymnasium import spaces
import numpy as np
from helpers import get_cycle_exponent

class GraphonEnv(gym.Env):
    """
    The RL environment for the dominant exponents problem, modified to support
    setting a custom starting state for hybrid search.
    """
    def __init__(self, n, H1, H2):
        super().__init__()
        self.n = n
        self.H1 = H1
        self.H2 = H2
        
        # STATE SPACE: The flattened n x n graphon matrix W.
        self.observation_space = spaces.Box(low=0, high=1, shape=(n * n,), dtype=np.float32)

        # ACTION SPACE: A continuous delta to add to each entry of W.
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(n * n,), dtype=np.float32)
        
        # This will hold the starting state for the next RL run
        self._custom_initial_state = None
        self.current_W = self._create_random_graphon()

    def set_initial_state(self, graphon):
        """Allows the main loop to set the starting point for the next training session."""
        self._custom_initial_state = graphon.copy()

    def _create_random_graphon(self):
        """Helper to create a random symmetric matrix with values in [0, 1]."""
        W = np.random.rand(self.n, self.n)
        return (W + W.T) / 2

    def step(self, action):
        action_matrix = action.reshape(self.n, self.n)
        
        self.current_W += action_matrix
        
        # Enforce constraints: symmetry and values in [0, 1]
        self.current_W = (self.current_W + self.current_W.T) / 2
        self.current_W = np.clip(self.current_W, 0, 1)

        # The reward is the exponent itself
        exponent = get_cycle_exponent(self.current_W, self.H1, self.H2)
        reward = exponent
        
        observation = self.current_W.flatten().astype(np.float32)
        info = {'exponent': exponent}
        
        return observation, reward, False, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Use the custom starting state if it has been set by the main loop
        if self._custom_initial_state is not None:
            self.current_W = self._custom_initial_state
            self._custom_initial_state = None  # Use it only once, then clear
        else:
            # Otherwise, reset to a random graphon
            self.current_W = self._create_random_graphon()
        
        observation = self.current_W.flatten().astype(np.float32)
        initial_exponent = get_cycle_exponent(self.current_W, self.H1, self.H2)
        info = {'exponent': initial_exponent}
        
        return observation, info
