import numpy as np
from time import time
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# Import AMCS and RL environment code
from amcs import AMCS
from matrix_env import MatrixEnv, calculate_gap

from helpers import strong_perturbation

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
    num_hybrid_cycles = 2
    amcs_levels = 20
    amcs_depth = 20
    rl_training_steps = 10_000
    perturbation_strength = 0.25

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

    print(f"--- Starting Hybrid RL-AMCS Search (RL First) ---")
    print(f"Initial Overall Best Score: {best_overall_score:.4e}")

    # The Hybrid Search Loop
    for cycle in range(num_hybrid_cycles):
        print(f"\n========================================================")
        print(f"HYBRID CYCLE {cycle + 1}/{num_hybrid_cycles}")
        print(f"========================================================")
        
        # PHASE 1: RL EXPLORATION
        # First, perturb the best known matrix to create a new starting point for RL.
        # This encourages the agent to explore a different part of the search space.
        print(f"\nPerturbing best matrix to create a new starting point for RL...")
        rl_start_M = strong_perturbation(best_overall_matrix, strength=perturbation_strength)
        rl_start_score = calculate_gap(rl_start_M)
        print(f"Perturbed score is {rl_start_score:.4e}")

        print(f"Starting RL Training Phase from the perturbed state...")
        
        # The callback tracks the best matrix found *during this RL run only*.
        rl_callback = SaveBestMatrixCallback(verbose=1)
        rl_callback.best_score = rl_start_score
        rl_callback.best_matrix = rl_start_M.copy()
        
        # Set the environment to start from the new perturbed matrix
        env.set_initial_state(rl_start_M)
        
        # Train the RL agent
        model.learn(total_timesteps=rl_training_steps, callback=rl_callback, reset_num_timesteps=False)
        
        # After RL training, check if it discovered a new overall best matrix
        if rl_callback.best_score > best_overall_score:
             print(f"\n  > RL Phase found new best overall score: {rl_callback.best_score:.4e}")
             best_overall_score = rl_callback.best_score
             best_overall_matrix = rl_callback.best_matrix.copy()
        else:
            print(f"\nRL Phase finished. Best overall score remains {best_overall_score:.4e}")

        # PHASE 2: AMCS EXPLOITATION - refines the matrix RL found locally
        print(f"\nStarting AMCS Phase from score {best_overall_score:.4e}...")
        amcs_found_M, _ = AMCS(best_overall_matrix.copy(), max_depth=amcs_depth, max_level=amcs_levels)
        amcs_found_score = calculate_gap(amcs_found_M)

        if amcs_found_score > best_overall_score:
            print(f"  > AMCS found new best overall score: {amcs_found_score:.4e}")
            best_overall_score = amcs_found_score
            best_overall_matrix = amcs_found_M.copy()
        else:
            print(f"AMCS finished. Best score remains {best_overall_score:.4e}")

    # Final results
    end_time = time()
    total_time = end_time - start_time
    final_gap = -best_overall_score
    print("\n\n--- Hybrid Search Finished ---")
    print("Best matrix found across all cycles:")
    print(np.round(best_overall_matrix, 3))
    print(f"Best Violation Score: {best_overall_score:.4e}")
    print(f"Corresponding Conjecture Gap: {final_gap:.4e}")
    print(f"Total search time: {total_time:.2f} seconds")
