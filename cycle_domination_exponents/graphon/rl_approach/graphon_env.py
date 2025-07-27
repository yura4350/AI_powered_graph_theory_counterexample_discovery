import gymnasium as gym
from gymnasium import spaces
import numpy as np
from helpers import get_cycle_exponent

class GraphonEnv(gym.Env):
    """An RL environment for the dominant exponents problem."""
    def __init__(self, n, H1, H2):
        super().__init__()
        self.n = n
        self.H1 = H1
        self.H2 = H2
        
        # STATE SPACE: The flattened n x n graphon matrix W. Values are in [0, 1].
        self.observation_space = spaces.Box(low=0, high=1, shape=(n * n,), dtype=np.float32)

        # ACTION SPACE: A continuous delta to add to each entry of W.
        # Bounded to encourage smaller, more stable changes.
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(n * n,), dtype=np.float32)
        
        # Initialize with a random symmetric graphon W
        self.current_W = self._create_random_graphon()

    def _create_random_graphon(self):
        """Helper to create a random symmetric matrix with values in [0, 1]."""
        W = np.random.rand(self.n, self.n)
        return (W + W.T) / 2 # Ensure symmetry

    def step(self, action):
        action_matrix = action.reshape(self.n, self.n)
        
        # Apply the action to the current state
        self.current_W += action_matrix
        
        # Enforce constraints: symmetry and values in [0, 1]
        self.current_W = (self.current_W + self.current_W.T) / 2
        self.current_W = np.clip(self.current_W, 0, 1)

        # The REWARD is the exponent itself, which we want to maximize
        exponent = get_cycle_exponent(self.current_W, self.H1, self.H2)
        reward = exponent
        
        # This is a continuing task, so terminated is always False
        terminated = False 
        
        observation = self.current_W.flatten().astype(np.float32)
        info = {'exponent': exponent}
        
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_W = self._create_random_graphon()
        observation = self.current_W.flatten().astype(np.float32)
        
        # Calculate initial exponent for the info dict
        initial_exponent = get_cycle_exponent(self.current_W, self.H1, self.H2)
        info = {'exponent': initial_exponent}
        
        return observation, info