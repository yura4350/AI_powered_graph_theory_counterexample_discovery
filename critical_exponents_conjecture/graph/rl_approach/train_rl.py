import numpy as np
from time import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Import the custom RL environment
from graph_env import GraphEnv

class SaveBestGraphCallback(BaseCallback):
    """
    A callback to save the graph T with the highest exponent found so far
    and periodically print the current best result.
    """
    def __init__(self, print_freq=5000, verbose=1):
        super().__init__(verbose)
        self.best_exponent = -float('inf')
        self.best_graph = None
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        # PPO returns info for each vectorized environment
        info = self.locals['infos'][0]
        if 'exponent' in info:
            current_exponent = info['exponent']
            if current_exponent > self.best_exponent:
                self.best_exponent = current_exponent
                # Get the current graph from the environment
                self.best_graph = self.training_env.envs[0].env.current_T.copy()
                if self.verbose > 0:
                    print(f"\nNew best exponent: {self.best_exponent:.6f} found at step {self.num_timesteps}")
        
        if self.num_timesteps % self.print_freq == 0 and self.num_timesteps > 0:
            if self.verbose > 0 and self.best_graph is not None:
                print(f"\n--- Periodic Update at Step {self.num_timesteps} ---")
                print(f"Current best exponent: {self.best_exponent:.6f}")
                print("Current best graph T:")
                print(self.best_graph)
                print("-" * 50)
        
        return True # Continue training

# --- Configuration ---
# Adjacency matrices for C5 and C3, as per Corollary 4.8 
C5 = np.array([
    [0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]
], dtype=np.int8)

C3 = np.array([
    [0, 1, 1], [1, 0, 1], [1, 1, 0]
], dtype=np.int8)

# Define the size of the target graph T
n_vertices = 10  # A small graph for reasonable computation time
# Define the graphs H1 and H2 for the exponent C(H1, H2)
H1 = C5
H2 = C3
# Total training steps for the agent
total_training_steps = 100_000

# --- PPO Hyperparameters ---
PPO_HYPERPARAMETERS = {
    "learning_rate": 0.0003, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
    "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
    "policy_kwargs": dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
}

# --- Initialization ---
env = GraphEnv(n=n_vertices, H1=H1, H2=H2)
model = PPO("MlpPolicy", env, verbose=0, **PPO_HYPERPARAMETERS)
save_best_callback = SaveBestGraphCallback(verbose=1, print_freq=10000)

# --- Start Search ---
obs, info = env.reset()
initial_exponent = info['exponent']
initial_graph = env.current_T.copy()

print(f"--- Searching for C(C{H1.shape[0]}, C{H2.shape[0]}) Lower Bound with Reinforcement Learning (PPO) ---")
print(f"--- Using {n_vertices}-vertex target graphs ---")
print(f"Paper's lower bound from construction: 15/7 ≈ {15/7:.6f} ")
print(f"Paper's upper bound: 11/5 = {11/5:.6f} ")
print("\nInitial Random Graph T:")
print(initial_graph)
print(f"Initial Exponent: {initial_exponent:.6f}")

# Prime the callback with the initial data
save_best_callback.best_exponent = initial_exponent
save_best_callback.best_graph = initial_graph

start_time = time()
model.learn(total_timesteps=total_training_steps, callback=save_best_callback)
training_time = time() - start_time
print(f"\nTotal search time: {training_time:.2f} seconds")

# --- Final Results ---
if save_best_callback.best_graph is not None:
    final_T = save_best_callback.best_graph
    final_exponent = save_best_callback.best_exponent

    print("\n--- RL Search Finished ---")
    print("Final optimized Graph T (best found):")
    print(final_T)
    print(f"\nFound new lower bound for C(C{H1.shape[0]}, C{H2.shape[0]}): {final_exponent:.6f}")
else:
    print("\nTraining finished, but no improved graph was found.")