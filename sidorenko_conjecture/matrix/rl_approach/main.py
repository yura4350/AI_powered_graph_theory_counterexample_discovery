import numpy as np
from time import time
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from matrix_env import MatrixEnv, calculate_gap

class SaveBestMatrixCallback(BaseCallback):
    """
    A callback to save the matrix M with the best score found so far during an RL run,
    and periodically print the current best matrix.
    """
    def __init__(self, print_freq=2000, verbose=0):
        super().__init__(verbose)
        self.best_score = -float('inf')
        self.best_matrix = None
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        # Find and store the best score from the environment info dictionary.
        # It's indexed by 0 because we are using a single environment.
        info = self.locals['infos'][0]
        if 'violation_score' in info:
            current_score = info['violation_score']
            if current_score > self.best_score:
                self.best_score = current_score
                # Make a copy of the matrix from the environment
                self.best_matrix = self.training_env.envs[0].env.current_M.copy()
                if self.verbose > 0:
                    print(f"\nNew best RL score: {self.best_score:.4e} found at step {self.num_timesteps}")
        
        # Periodic updates to show progress
        if self.num_timesteps % self.print_freq == 0 and self.num_timesteps > 0:
            if self.verbose > 0 and self.best_matrix is not None:
                print(f"\n--- RL Periodic Update at Step {self.num_timesteps} ---")
                print(f"Current best score in this RL run: {self.best_score:.4e}")
                print("Current best matrix in this RL run:")
                print(np.round(self.best_matrix, 3))
                print("-" * 40)
        
        return True

if __name__ == "__main__":
    start_time = time()
    m, n = 10, 10
    training_steps = 1_000

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

    # Initialization of the environment
    env = MatrixEnv(m=m, n=n)
    model = SAC("MlpPolicy", env, verbose=0)
    
    # Initialize overall best tracking with a random matrix
    best_overall_matrix = np.random.rand(m, n)
    best_overall_score = calculate_gap(best_overall_matrix)

    print(f"--- Starting RL Search ---")
    print(f"Initial Overall Best Score: {best_overall_score:.4e}")

    start_M = best_overall_matrix
    start_score = calculate_gap(start_M)

    print(f"Starting RL Training Phase from the perturbed state...")
    
    # The callback tracks the best matrix found *during this RL run only*.
    callback = SaveBestMatrixCallback(verbose=1)
    callback.best_score = start_score
    callback.best_matrix = start_M.copy()
    
    # Set the environment to start from the new perturbed matrix
    env.set_initial_state(start_M)
    
    # Train the RL agent
    model.learn(total_timesteps=training_steps, callback=callback, reset_num_timesteps=False)

    # Final results
    end_time = time()
    total_time = end_time - start_time
    print("\n\n--- Hybrid Search Finished ---")
    print("Best matrix found across all cycles:")
    print(np.round(best_overall_matrix, 3))
    print(f"Best Conjecture Gap: {best_overall_score:.4e}")
    print(f"Total search time: {total_time:.2f} seconds")
