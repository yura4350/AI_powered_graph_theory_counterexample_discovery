import numpy as np
from time import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from amcs import AMCS_for_graphs
from graph_env import GraphEnv
from helpers import get_graph_exponent, strong_perturbation_graph

class SaveBestGraphCallback(BaseCallback):
    """A callback to save the best graph found during a single RL training run."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_exponent = -float('inf')
        self.best_graph = None

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if 'exponent' in info and info['exponent'] > self.best_exponent:
            self.best_exponent = info['exponent']
            self.best_graph = self.training_env.envs[0].env.current_T.copy()
            if self.verbose > 0:
                print(f"\n  New best RL exponent: {self.best_exponent:.6f} at step {self.num_timesteps}")
        return True

if __name__ == "__main__":
    C5 = np.array([[0,1,0,0,1],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[1,0,0,1,0]], dtype=np.int8)
    C3 = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=np.int8)
    
    n_vertices = 15
    H1, H2 = C5, C3
    num_hybrid_cycles = 4
    rl_training_steps = 20_000
    perturbation_probability = 0.20

    PPO_HYPERPARAMETERS = {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
                           "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2}

    # --- Initialization ---
    env = GraphEnv(n=n_vertices, H1=H1, H2=H2)
    model = PPO("MlpPolicy", env, verbose=0, **PPO_HYPERPARAMETERS)
    obs, info = env.reset()
    best_overall_graph = env.current_T.copy()
    best_overall_exponent = info['exponent']

    print(f"--- Starting Hybrid RL-AMCS Search for C(C{H1.shape[0]}, C{H2.shape[0]}) ---")
    print(f"Initial Overall Best Exponent: {best_overall_exponent:.6f}")

    # The Hybrid Search Loop
    for cycle in range(num_hybrid_cycles):
        print(f"\n{'='*60}\n  HYBRID CYCLE {cycle + 1}/{num_hybrid_cycles}\n{'='*60}")
        
        # PHASE 1: RL Exploration
        print(f"\n[Phase 1] Perturbing best graph to start RL exploration...")
        rl_start_T = strong_perturbation_graph(best_overall_graph, p_flip=perturbation_probability)
        rl_start_exponent = get_graph_exponent(rl_start_T, H1, H2)
        print(f"Starting RL from exponent: {rl_start_exponent:.6f}")
        
        rl_callback = SaveBestGraphCallback(verbose=1)
        rl_callback.best_exponent, rl_callback.best_graph = rl_start_exponent, rl_start_T.copy()
        
        env.set_initial_state(rl_start_T)
        model.learn(total_timesteps=rl_training_steps, callback=rl_callback, reset_num_timesteps=False)
        
        if rl_callback.best_exponent > best_overall_exponent:
             print(f"\n  >>> RL Phase found new best overall exponent: {rl_callback.best_exponent:.6f}")
             best_overall_exponent = rl_callback.best_exponent
             best_overall_graph = rl_callback.best_graph.copy()
        else:
            print(f"\nRL Phase finished. Best overall exponent remains {best_overall_exponent:.6f}")

        # PHASE 2: AMCS EXPLOITATION
        # Start AMCS from the best graph found in the preceding RL phase.
        print(f"\n[Phase 2] Starting AMCS exploitation from RL's best graph (exponent: {rl_callback.best_exponent:.6f})...")
        amcs_found_T, amcs_found_exponent = AMCS_for_graphs(H1, H2, rl_callback.best_graph.copy())

        # Compare the result of this new search against the overall best.
        if amcs_found_exponent > best_overall_exponent:
            print(f"  >>> AMCS Phase found new best overall exponent: {amcs_found_exponent:.6f}")
            best_overall_exponent = amcs_found_exponent
            best_overall_graph = amcs_found_T.copy()
        else:
            print(f"AMCS finished. Best overall exponent remains {best_overall_exponent:.6f}")

    # Final Results
    print("\n\n--- Hybrid Search Finished ---")
    print("\nBest graph found across all cycles:")
    print(best_overall_graph)
    print(f"\nBest Exponent for C(C{H1.shape[0]}, C{H2.shape[0]}): {best_overall_exponent:.6f}")