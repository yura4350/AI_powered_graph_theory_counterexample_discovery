import numpy as np
from time import time
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# Import the custom RL environment and helper matrices
from graphon_env import GraphonEnv

class SaveBestGraphonCallback(BaseCallback):
    """
    A callback to save the graphon W with the highest exponent found so far
    and periodically print the current best result.
    """
    def __init__(self, print_freq=5000, verbose=0):
        super().__init__(verbose)
        self.best_exponent = -float('inf')
        self.best_graphon = None
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        # Check for the best score in the info dictionary
        info = self.locals['infos'][0]
        if 'exponent' in info:
            current_exponent = info['exponent']
            if current_exponent > self.best_exponent:
                self.best_exponent = current_exponent
                self.best_graphon = self.training_env.envs[0].env.current_W.copy()
                if self.verbose > 0:
                    print(f"\nNew best exponent: {self.best_exponent:.6f} found at step {self.num_timesteps}")
        
        # Periodically print the current best result
        if self.num_timesteps % self.print_freq == 0 and self.num_timesteps > 0:
            if self.verbose > 0 and self.best_graphon is not None:
                print(f"\n--- Periodic Update at Step {self.num_timesteps} ---")
                print(f"Current best exponent: {self.best_exponent:.6f}")
                print("Current best graphon:")
                print(np.round(self.best_graphon, 4))
                print("-" * 50)
        
        return True # Continue training

# Configuration
# Odd-cycle adjacency matrices for testing
C3 = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])
    
C5 = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0]
])

C7 = np.array([
    [0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0]
])

# Define the size of the step-function graphon W
n = 6 
# Define the graphs H1 and H2 for the exponent C(H1, H2)
H1 = C5
H2 = C3
# Total training steps for the agent
total_training_steps = 100_000

# SAC HYPERPARAMETERS
HYPERPARAMETERS = {
    "learning_rate": 0.0003,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": (1, "step"), # Train the model every 1 step in the environment
    "gradient_steps": 1,      # How many gradient steps to do after each rollout
    "policy_kwargs": dict(net_arch=[256, 256]) # Default network architecture
}

# Initialization
env = GraphonEnv(n=n, H1=H1, H2=H2)

# Instantiate the SAC model with the defined hyperparameters
model = SAC(
    "MlpPolicy", 
    env, 
    verbose=0,
    learning_rate=HYPERPARAMETERS["learning_rate"],
    buffer_size=HYPERPARAMETERS["buffer_size"],
    batch_size=HYPERPARAMETERS["batch_size"],
    tau=HYPERPARAMETERS["tau"],
    gamma=HYPERPARAMETERS["gamma"],
    train_freq=HYPERPARAMETERS["train_freq"],
    gradient_steps=HYPERPARAMETERS["gradient_steps"],
    policy_kwargs=HYPERPARAMETERS["policy_kwargs"]
)

save_best_callback = SaveBestGraphonCallback(verbose=1)

# Get initial state and exponent
obs, info = env.reset()
initial_exponent = info['exponent']
initial_graphon = env.current_W.copy()

print(f"--- Searching for C(C{H1.shape[0]}, C{H2.shape[0]}) Lower Bound with Reinforcement Learning (SAC) ---")
print(f"--- Using {n}x{n} graphons ---")
print("\nInitial Random W:")
print(np.round(initial_graphon, 4))
print(f"Initial Exponent: {initial_exponent:.6f}")

# "Prime" the callback with the initial data
save_best_callback.best_exponent = initial_exponent
save_best_callback.best_graphon = initial_graphon

# RL Training
start_time = time()
model.learn(total_timesteps=total_training_steps, callback=save_best_callback)
print(f"\nTotal search time: {time() - start_time:.2f} seconds")

# Final Results
if save_best_callback.best_graphon is not None:
    final_W = save_best_callback.best_graphon
    final_exponent = save_best_callback.best_exponent

    print("\n--- RL Search Finished ---")
    print("Final optimized W (best found):")
    print(np.round(final_W, 4))
    print(f"\nFound new lower bound for C(C{H1.shape[0]}, C{H2.shape[0]}): {final_exponent:.6f}")
else:
    print("\nTraining finished, but no improved graphon was found.")