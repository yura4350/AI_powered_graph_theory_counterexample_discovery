# Computational Lower Bounds for Cycle Domination Exponents

This repository contains implementations of Adaptive Monte Carlo Search (AMCS), Reinforcement Learning (RL), and hybrid algorithms designed to find strong computational lower bounds for the homomorphism density domination exponent between pairs of cycle graphs.

## Mathematical Background

The core of this research is the homomorphism density domination exponent, a concept that unifies several problems in extremal combinatorics. For two graphs, H₁ and H₂, this exponent is defined as the smallest real number c that satisfies the inequality for all possible target graphs T:

```
t(H₁, T) ≥ t(H₂, T)^c
```

where t(H,T) is the homomorphism density of graph H in graph T. This project is specifically motivated by the challenge of determining this exponent, denoted C(H₁,H₂), for pairs of odd cycles.

The 2025 paper "On Domination Exponents for Pairs of Graphs" by Blekherman, Raymond, Razborov, and Wei establishes the following theoretical bounds for odd cycles C_{2k+1} and C_{2l+1} where k≥l:

- **Upper Bound**: The paper provides an upper bound derived from a "tensor trick" and inequalities involving even cycles.
- **Lower Bound**: A general lower bound is established as C(C_{2k+1}, C_{2l+1}) ≥ (4k²-1)/(4kl-1).

For the specific case of C₅ and C₃ (where k=2, l=1), this creates the intriguing interval C(C₅,C₃) ∈ [15/7, 11/5], which is approximately [2.143, 2.2].

## Research Objective

The primary goal of this software is to computationally search for target graphs T that yield a higher lower bound for C(C_{2k+1}, C_{2l+1}) than is currently known from theory. We aim to maximize the ratio:

```
log(t(C_{2l+1}, T)) / log(t(C_{2k+1}, T))
```

This provides strong empirical evidence and may guide the development of new theoretical conjectures.

## Directory Structure

```
cycle_domination_exponents/
├── graph/                     # Graph-based approaches
│   ├── amcs_approach/         # Adaptive Monte Carlo Search
│   │   ├── main.py            # Main execution script
│   │   └── ...
│   ├── rl_approach/           # Reinforcement Learning
│   │   ├── train_rl.py        # RL training script
│   │   └── ...
│   └── rl+amcs_approach/      # Hybrid approach
│       ├── main.py            # Hybrid execution
│       └── ...
└── graphon/                   # Graphon-based approaches
    ├── amcs_approach/
    │   ├── v0/                # Version 0 implementation
    │   └── v1/                # Version 1 implementation
    ├── rl_approach/           # Graphon RL approach
    └── rl+amcs_approach/      # Hybrid graphon approach
```

## Algorithmic Approaches

### 1. Graph-Based Methods

#### AMCS (Adaptive Monte Carlo Search)
- **Algorithm**: Nested Monte Carlo with adaptive annealing schedules.
- **Features**:
  - Graph perturbation through edge flips.
  - Dynamic adjustment of perturbation intensity to escape local optima.
  - Multi-level exploration strategy for a broad search.

#### RL (Reinforcement Learning)
- **Algorithm**: Proximal Policy Optimization (PPO) with a custom graph environment.
- **Features**:
  - **Action Space**: Modifying edges in the graph.
  - **Reward**: Improvement in the exponent C(H₁, H₂).
  - **State Space**: The graph's adjacency matrix.

#### Hybrid RL+AMCS
- **Algorithm**: Alternating between RL-driven exploration and AMCS-driven refinement.
- **Features**:
  - RL is used for broad, policy-guided exploration of the search space.
  - AMCS is used for fine-grained local optimization of promising candidates found by RL.

### 2. Graphon-Based Methods

- **Representation**: A graphon is a symmetric, measurable function W:[0,1]² → [0,1], which serves as a limit object for sequences of dense graphs.
- **Objective**: Maximize the exponent using continuous homomorphism integrals.
- **Mathematical Foundation**:

```
t(H, W) = ∫[0,1]^|V(H)| ∏_{(i,j) ∈ E(H)} W(x_i, x_j) ∏_{i=1}^|V(H)| dx_i
```

## Usage Examples

### Running AMCS on Graphs for C(C₅, C₃)

```bash
cd graph/amcs_approach/
# Ensure configuration specifies C5 and C3
python main.py
```

**Note**: While this example targets C(C₅, C₃), the scripts are configurable for any pair of cycle graphs.

**Expected Output**:
```
AMCS Search for C(C5, C3) Lower Bound using Undirected Graphs
Goal: Find a 20-vertex graph T that maximizes log(t(C5,T)) / log(t(C3,T))
Theoretical Lower Bound: 2.142857

Best exponent (initial): 1.987654
Best exponent (lvl 1, dpt 0): 2.156789
...
New best exponent found: 2.181234
Total search time: 45.23 seconds
```

## Performance Analysis

### Example Empirical Results for C(C₅, C₃)

The following table shows sample results for finding a lower bound for C(C₅,C₃) on a 20-vertex graph.

| Method | Best Exponent | Time (min) |
|--------|---------------|------------|
| AMCS   | 2.181         | 15         |
| RL     | 2.165         | 45         |
| Hybrid | 2.189         | 35         |

## Implementation Details

### Homomorphism Counting

All methods rely on an efficient library for homomorphism enumeration. The core scoring function is:

```python
import numpy as np
# from some_hom_library import count_homomorphisms

def get_graph_exponent(T, H1, H2):
    # Note: Densities t(H,T) are proportional to hom(H,T) for a fixed T,
    # so the ratio of logs is the same.
    hom1 = count_homomorphisms(H1, T)
    hom2 = count_homomorphisms(H2, T)
    
    if hom1 <= 0 or hom2 <= 0:
        return -float('inf')
        
    return np.log(hom1) / np.log(hom2)
```

## Mathematical References

1. **Blekherman, G., Raymond, A., Razborov, A., & Wei, F.** (2025). *On Domination Exponents for Pairs of Graphs*. arXiv:2506.12151.
2. **Lovász, L.** (2012). *Large Networks and Graph Limits*. American Mathematical Society.
3. **Razborov, A.** (2008). *On the minimal density of triangles in graphs*. Combinatorics, Probability and Computing.
4. **Hatami, H.** (2010). *Graph norms and Sidorenko's conjecture*. Innovations in Theoretical Computer Science. 