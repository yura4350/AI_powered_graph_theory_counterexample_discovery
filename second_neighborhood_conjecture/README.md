# Second Neighborhood Conjecture

This directory contains implementations for discovering counterexamples to the **Second Neighborhood Conjecture**, which concerns the relationship between out-degrees and second neighborhoods in oriented graphs.

## Mathematical Background

### The Conjecture
The Second Neighborhood Conjecture, formulated by Seymour, states that:

> *Every oriented graph contains a vertex $v$ such that $|N^{++}(v)| \geq |N^+(v)|$*

Where:
- **$N^+(v)$**: The **out-neighborhood** of vertex $v$ (vertices directly reachable from $v$)
- **$N^{++}(v)$**: The **second out-neighborhood** of vertex $v$ (vertices reachable in exactly 2 steps from $v$, excluding $N^+(v)$ and $v$ itself)

### Research Objective
Find oriented graphs where **every vertex** satisfies $|N^{++}(v)| < |N^+(v)|$, which would constitute a counterexample to the conjecture.

### Mathematical Formulation
For a directed graph $G$ and vertex $v$:
$$N^+(v) = \{u \in V(G) : (v,u) \in E(G)\}$$
$$N^{++}(v) = \{w \in V(G) : \exists u \in N^+(v) \text{ such that } (u,w) \in E(G), w \notin N^+(v) \cup \{v\}\}$$

**Goal**: Find $G$ where $\forall v \in V(G): |N^{++}(v)| < |N^+(v)|$

## Directory Structure

```
second_neighborhood_conjecture/
├── amcs_approach/              # Adaptive Monte Carlo Search
│   ├── v0/                    # Version 0 implementation
│   │   ├── amcs.sage         # Main AMCS algorithm
│   │   ├── nmcs.sage         # Nested Monte Carlo Search
│   │   └── scores.sage       # Scoring functions
│   └── v1/                   # Version 1 implementation
│       ├── amcs.sage         # Improved AMCS algorithm
│       ├── nmcs.sage         # Enhanced NMCS
│       └── scores.sage       # Updated scoring functions
└── rl_approach/              # Reinforcement Learning
    ├── second_neighborhood_env.py  # Custom RL environment
    └── train_rl.py                # RL training script
```

## Algorithmic Approaches

### 1. AMCS (Adaptive Monte Carlo Search)

#### Version 0 (`amcs_approach/v0/`)
- **Framework**: SageMath implementation
- **Search strategy**: Oriented graph modifications
- **Scoring**: Weighted penalty system

#### Version 1 (`amcs_approach/v1/`)
- **Improvements**: Enhanced perturbation strategies
- **Optimization**: Better convergence properties
- **Scoring**: Refined penalty calculations

#### Key Features:
- **Graph representation**: Adjacency matrices for directed graphs
- **Perturbation methods**: Edge additions, deletions, and redirections
- **Annealing**: Adaptive temperature control
- **Termination**: Success when all vertices satisfy the desired property

### 2. RL (Reinforcement Learning)
- **File**: `rl_approach/train_rl.py`
- **Environment**: Custom oriented graph environment
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Action space**: Edge modifications in directed graphs

## Implementation Details

### Scoring Function (v0)
The scoring system uses a weighted penalty approach:

```python
def second_neighborhood_problem_score(A):
    total_penalty = 0
    penalty_multiplier = 100
    
    for v_idx in range(n):
        out_degree = sum(1 for x in A[v_idx] if x > 0)
        size_second_neighborhood = calculate_second_neighborhood(A, v_idx)
        diff = size_second_neighborhood - out_degree
        
        if diff < 0:
            total_penalty -= 1  # Reward "good" vertices
        else:
            total_penalty += (diff + 1) * penalty_multiplier  # Penalize "bad" vertices
    
    return -total_penalty  # Higher score is better
```

### Graph Perturbation Strategies
1. **Edge addition**: Add new directed edges
2. **Edge deletion**: Remove existing directed edges
3. **Edge redirection**: Change edge orientation
4. **Vertex operations**: Add/remove vertices (in some variants)

### Second Neighborhood Calculation
```sage
def calculate_second_neighborhood(A, v):
    """Calculate |N++(v)| for vertex v in adjacency matrix A"""
    n = A.nrows()
    A_squared = A * A
    A_reach_in_2 = matrix([[1 if x > 0 else 0 for x in row] for row in A_squared])
    Npp_matrix = A_reach_in_2 - A
    for i in range(n):
        Npp_matrix[i, i] = 0
    
    return sum(1 for x in Npp_matrix[v] if x > 0)
```

## Usage Examples

### Running AMCS v0
```bash
cd amcs_approach/v0/
sage amcs.sage
```

**Expected Workflow**:
```
Creating random oriented graph with 10 vertices...
Best score (initial): -1500
Starting AMCS search...
Best score (lvl 1, dpt 0): -1200
Best score (lvl 2, dpt 0): -800
...
Counterexample found! Score: 10
Final graph satisfies the desired property.
```

### Running AMCS v1
```bash
cd amcs_approach/v1/
sage amcs.sage
```

**Improvements in v1**:
- Better initial graph generation
- Improved perturbation strategies
- Enhanced convergence criteria

### Training RL Agent
```bash
cd rl_approach/
python train_rl.py
```

**RL Features**:
- State: Adjacency matrix representation
- Actions: Edge modifications
- Reward: Based on neighborhood property satisfaction
- Training: PPO algorithm with custom environment

## Research Insights

### Theoretical Implications
1. **Structural properties**: Counterexamples (if they exist) likely have special structure
2. **Size bounds**: Larger graphs may be necessary for counterexamples
3. **Density considerations**: Edge density affects neighborhood sizes

### Computational Challenges
1. **Combinatorial explosion**: Search space grows exponentially
2. **Local minima**: Scoring function may have many local optima
3. **Verification**: Need to check all vertices simultaneously

### Heuristic Observations
- **High out-degree vertices**: Often satisfy the property naturally
- **Sparse graphs**: May be more promising for counterexamples
- **Regular structures**: Tournament-like structures show patterns

## Advanced Features

### Adaptive Penalties (v1)
```sage
def adaptive_penalty_score(A):
    """Enhanced scoring with adaptive penalties"""
    base_penalty = 100
    size_factor = A.nrows()
    adaptive_multiplier = base_penalty * (1 + size_factor / 10)
    # ... rest of scoring logic
```

### Graph Generation Strategies
1. **Random oriented graphs**: $G(n,p)$ with random orientations
2. **Tournament-based**: Start from tournaments and modify
3. **Structured graphs**: Begin with specific graph classes

### Convergence Detection
```sage
def check_convergence(score_history, window=10):
    """Check if search has converged"""
    if len(score_history) < window:
        return False
    recent_scores = score_history[-window:]
    return max(recent_scores) - min(recent_scores) < threshold
```

## Mathematical Context

### Related Results
1. **Seymour's original conjecture**: Motivation and background
2. **Tournament theory**: Related results on regular tournaments
3. **Digraph connectivity**: Connections to strong connectivity

### Open Questions
1. **Minimum counterexample size**: What's the smallest possible counterexample?
2. **Probabilistic bounds**: What's the probability of random graphs being counterexamples?
3. **Algorithmic complexity**: Is finding counterexamples NP-hard?

### Extensions
1. **Weighted graphs**: Generalization to weighted digraphs
2. **$k$-neighborhoods**: Extension to $k$-th neighborhoods for $k > 2$
3. **Hypergraphs**: Analogous questions for directed hypergraphs

## Implementation Notes

### SageMath Dependencies
```sage
from sage.all import DiGraph, graphs, matrix
from random import choice, random
from time import time
```

### Graph Representation
- **Adjacency matrices**: Integer matrices with 0/1 entries
- **SageMath DiGraph objects**: For built-in algorithms
- **Conversion utilities**: Between different representations

### Debugging Tools
1. **Visualization**: Graph plotting for small examples
2. **Property verification**: Manual checking of neighborhood conditions
3. **Score tracking**: Detailed logging of optimization progress

## Future Directions

### Algorithmic Improvements
1. **Parallel search**: Multi-threaded exploration
2. **ML integration**: Graph neural networks for property prediction
3. **Hybrid methods**: Combining multiple search strategies

### Mathematical Extensions
1. **Probabilistic analysis**: Expected neighborhood sizes
2. **Extremal questions**: Maximum violation of the conjecture
3. **Structural characterization**: What do counterexamples look like?

---

*The Second Neighborhood Conjecture represents a fundamental question in tournament theory and structural graph theory, with implications for understanding directed graph properties.* 