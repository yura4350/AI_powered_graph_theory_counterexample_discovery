import numpy as np
from time import time
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# Import custom modules for the hybrid search
from amcs import AMCS
from graphon_env import GraphonEnv
from helpers import get_cycle_exponent, strong_perturbation

class SaveBestGraphonCallback(BaseCallback):
    """
    A callback to save the graphon W with the highest exponent found so far
    during a single RL training run.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_exponent = -float('inf')
        self.best_graphon = None

    def _on_step(self) -> bool:
        # Check the info dictionary for the exponent
        info = self.locals['infos'][0]
        if 'exponent' in info:
            current_exponent = info['exponent']
            if current_exponent > self.best_exponent:
                self.best_exponent = current_exponent
                self.best_graphon = self.training_env.envs[0].env.current_W.copy()
                if self.verbose > 0:
                    print(f"\n  New best RL exponent: {self.best_exponent:.6f} at step {self.num_timesteps}")
        return True

if __name__ == "__main__":
    start_time = time()

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

    C9 = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0]
    ])

    # CONFIGURATION
    n = 6  # Size of the step-function graphon
    H1 = C5
    H2 = C3
    num_hybrid_cycles = 3      # Number of times to alternate between RL and AMCS
    rl_training_steps = 15_000 # Timesteps for each RL exploration phase
    amcs_levels = 5            # Search intensity for AMCS
    amcs_depth = 10
    perturbation_strength = 0.2 # How much to change the best W before starting RL

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
    
    # Initialize overall best tracking with a random graphon
    obs, info = env.reset()
    best_overall_graphon = env.current_W.copy()
    best_overall_exponent = info['exponent']

    print(f"--- Starting Hybrid RL-AMCS Search for C(C{H1.shape[0]}, C{H2.shape[0]}) ---")
    print(f"Initial Overall Best Exponent: {best_overall_exponent:.6f}")

    for cycle in range(num_hybrid_cycles):
        print(f"\n{'='*60}")
        print(f"  HYBRID CYCLE {cycle + 1}/{num_hybrid_cycles}")
        print(f"{'='*60}")
        
        # PHASE 1: RL Exploration
        # Perturb the best known graphon to create a new, interesting starting point
        print(f"\n[Phase 1] Perturbing best graphon to start RL exploration...")
        rl_start_W = strong_perturbation(best_overall_graphon, strength=perturbation_strength)
        rl_start_exponent = get_cycle_exponent(rl_start_W, H1, H2)
        
        print(f"Starting RL from exponent: {rl_start_exponent:.6f}")
        
        # The callback tracks the best graphon found.
        rl_callback = SaveBestGraphonCallback(verbose=1)
        rl_callback.best_exponent = rl_start_exponent
        rl_callback.best_graphon = rl_start_W.copy()
        
        # Set the environment to start from the new perturbed graphon
        env.set_initial_state(rl_start_W)
        
        # Train the RL agent
        model.learn(total_timesteps=rl_training_steps, callback=rl_callback, reset_num_timesteps=False)
        
        # After RL training, check if it discovered a new overall best graphon
        if rl_callback.best_exponent > best_overall_exponent:
             print(f"\n  >>> RL Phase found new best overall exponent: {rl_callback.best_exponent:.6f}")
             best_overall_exponent = rl_callback.best_exponent
             best_overall_graphon = rl_callback.best_graphon.copy()
        else:
            print(f"\nRL Phase finished. Best overall exponent remains {best_overall_exponent:.6f}")

        # PHASE 2: AMCS EXPLOITATION
        # Refine the best-known graphon with a deep, local search
        print(f"\n[Phase 2] Starting AMCS exploitation from exponent {best_overall_exponent:.6f}...")
        amcs_found_W, amcs_found_exponent = AMCS(
            H1, H2, best_overall_graphon.copy(), max_depth=amcs_depth, max_level=amcs_levels
        )

        if amcs_found_exponent > best_overall_exponent:
            print(f"  >>> AMCS Phase found new best overall exponent: {amcs_found_exponent:.6f}")
            best_overall_exponent = amcs_found_exponent
            best_overall_graphon = amcs_found_W.copy()
        else:
            print(f"AMCS finished. Best overall exponent remains {best_overall_exponent:.6f}")

    # Final results
    total_time = time() - start_time
    print("\n\n--- Hybrid Search Finished ---")
    print("\nBest graphon found across all cycles:")
    print(np.round(best_overall_graphon, 4))
    print(f"\nBest Exponent for C(C{H1.shape[0]}, C{H2.shape[0]}): {best_overall_exponent:.6f}")
    print(f"Total search time: {total_time:.2f} seconds")
