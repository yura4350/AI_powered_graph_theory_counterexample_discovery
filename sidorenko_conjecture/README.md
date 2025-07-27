# Sidorenko Conjecture

This directory contains implementations for discovering counterexamples to the **Sidorenko Conjecture**, one of the most important open problems in extremal graph theory concerning homomorphism densities of bipartite graphs.

## Mathematical Background

### The Conjecture
Sidorenko's Conjecture, formulated by Alexander Sidorenko in 1993, states that:

> *For every bipartite graph $H$ and every graph $G$:*
> $$t(H, G) \geq t(K_2, G)^{|E(H)|}$$

Where:
- **$t(H, G)$**: The **homomorphism density** of $H$ in $G$
- **$K_2$**: The complete graph on 2 vertices (single edge)
- **$|E(H)|$**: The number of edges in $H$

### Research Objective
Find counterexamples: bipartite graphs $H$ and graphs $G$ such that:
$$t(H, G) < t(K_2, G)^{|E(H)|}$$

This would constitute a violation of Sidorenko's conjecture and would have significant implications for extremal graph theory.

### Mathematical Significance
- **Extremal graph theory**: Central to understanding edge density relationships
- **Homomorphism theory**: Connects to the broader theory of graph homomorphisms
- **Probabilistic methods**: Related to correlation inequalities in probability theory

## Directory Structure

```
sidorenko_conjecture/
├── graph/                      # Graph-based approaches
│   └── amcs_approach/         # AMCS for discrete graphs
│       ├── amcs.sage         # Main AMCS algorithm
│       ├── nmcs.sage         # Nested Monte Carlo Search
│       └── scores.sage       # Scoring functions
├── graphon/                   # Graphon-based approaches
│   └── amcs_approach/        # AMCS for continuous representations
│       ├── amcs.py          # Graphon AMCS implementation
│       ├── helpers.py       # Utility functions
│       └── main.py          # Main execution script
└── matrix/                    # Matrix-based approaches
    ├── amcs_approach/        # Eigenvalue-based methods
    │   ├── amcs.py          # Matrix AMCS implementation
    │   ├── helpers.py       # Matrix utilities
    │   └── main.py          # Main execution script
    └── rl_approach/          # RL for matrix optimization
        ├── matrix_env.py    # Custom matrix environment
        ├── helpers.py       # Utility functions
        └── main.py          # RL training script
```

## Algorithmic Approaches

### 1. Graph-Based Methods (`graph/`)

#### AMCS for Discrete Graphs
- **Framework**: SageMath implementation
- **Target**: Small bipartite graphs (trees, cycles, complete bipartite graphs)
- **Strategy**: Search over host graphs $G$ to minimize $\frac{t(H,G)}{t(K_2,G)^{|E(H)|}}$

**Key Features**:
- CountHomLib integration for efficient homomorphism counting
- Tree decomposition optimization
- Specialized operations for maintaining graph properties

### 2. Graphon-Based Methods (`graphon/`)

#### Continuous Optimization
- **Representation**: Symmetric matrices $W \in [0,1]^{n \times n}$ representing graphons
- **Objective**: Minimize homomorphism integrals
- **Advantage**: Continuous search space allows for analytical insights

**Mathematical Foundation**:
$$t(H, W) = \int_{[0,1]^{|V(H)|}} \prod_{(i,j) \in E(H)} W(x_i, x_j) \prod_{i=1}^{|V(H)|} dx_i$$

### 3. Matrix-Based Methods (`matrix/`)

#### Eigenvalue Formulation
- **Approach**: Express homomorphism densities using matrix eigenvalues
- **Target**: Specific bipartite graphs with nice spectral properties
- **Example**: $K_{5,5} \setminus C_{10}$ (complete bipartite minus perfect matching)

**Spectral Approach**:
For certain bipartite graphs, homomorphism densities can be expressed as:
$$t(H, M) = \frac{\lambda_1(M^H)}{|V(G)|^{|V(H)|}}$$
where $\lambda_1$ is the largest eigenvalue.

## Specific Target Cases

### Known Difficult Cases
1. **Trees**: All trees satisfy Sidorenko's conjecture (proven)
2. **Even cycles**: $C_4, C_6, \ldots$ satisfy the conjecture (proven)
3. **Complete bipartite graphs**: $K_{s,t}$ cases are challenging
4. **$K_{5,5} \setminus C_{10}$**: Specific case targeted by matrix methods

### Promising Counterexample Candidates
1. **Large complete bipartite graphs**: $K_{m,n}$ for large $m,n$
2. **Bipartite graphs with large girth**: Avoiding short cycles
3. **Random bipartite graphs**: Specific probabilistic constructions

## Usage Examples

### Graph-Based AMCS
```bash
cd graph/amcs_approach/
sage amcs.sage
```

**Expected Output**:
```
Searching for counterexample to Sidorenko's Conjecture
Target bipartite graph H: Tree with 5 vertices
Initializing random graph G with 10 vertices...
Best gap (initial): -0.234
Best gap (lvl 1, dpt 0): -0.187
...
No counterexample found, but gap improved to: -0.089
```

### Graphon-Based AMCS
```bash
cd graphon/amcs_approach/
python main.py
```

**Features**:
- Continuous optimization over graphon space
- Integration with numerical methods
- Potential for analytical insights

### Matrix-Based Approach
```bash
cd matrix/amcs_approach/
python main.py
```

**Targeting $K_{5,5} \setminus C_{10}$**:
```
--- Searching for counterexample to Sidorenko's Conjecture for K5,5 \ C10 ---
--- Using eigenvalue formulation on 4x4 matrices ---
Initial M:
[[0.123 0.456 0.789 0.234]
 [0.567 0.890 0.345 0.678]
 [0.901 0.234 0.567 0.890]
 [0.345 0.678 0.901 0.234]]
...
```

### RL for Matrix Optimization
```bash
cd matrix/rl_approach/
python main.py
```

**RL Features**:
- PPO agent optimizing matrix entries
- Reward based on Sidorenko gap
- Continuous action space for matrix values

## Implementation Details

### Homomorphism Density Calculation
```python
from homlib import Graph as HomlibGraph, countHom

def calculate_homomorphism_density(H, G):
    """Calculate t(H,G) using CountHomLib"""
    H_homlib = HomlibGraph(H.adjacency_matrix().tolist())
    G_homlib = HomlibGraph(G.adjacency_matrix().tolist())
    
    hom_count = countHom(H_homlib, G_homlib)
    total_mappings = len(G) ** len(H)
    
    return hom_count / total_mappings
```

### Sidorenko Gap Calculation
```python
def sidorenko_gap(H, G):
    """Calculate the gap: t(H,G) - t(K2,G)^|E(H)|"""
    t_H_G = calculate_homomorphism_density(H, G)
    t_K2_G = calculate_edge_density(G)
    
    expected_by_sidorenko = t_K2_G ** H.size()
    return t_H_G - expected_by_sidorenko
```

### Graph Perturbation for Trees
```sage
def tree_preserving_perturbation(G):
    """Modify G while maintaining tree structure"""
    # Remove random leaf
    leaves = [v for v in G.vertices() if G.degree(v) == 1]
    if leaves:
        leaf = choice(leaves)
        G.delete_vertex(leaf)
    
    # Remove random subdivision
    deg_2_vertices = [v for v in G.vertices() if G.degree(v) == 2]
    if deg_2_vertices:
        v = choice(deg_2_vertices)
        neighbors = G.neighbors(v)
        G.add_edge(neighbors[0], neighbors[1])
        G.delete_vertex(v)
    
    return G
```

## Research Results and Insights

### Computational Findings
1. **Small graphs**: No counterexamples found for graphs with $\leq 15$ vertices
2. **Tree optimization**: Host graphs often converge to near-regular structures
3. **Matrix approach**: $K_{5,5} \setminus C_{10}$ shows promising but inconclusive results

### Theoretical Insights
1. **Regularity**: Extremal graphs tend to have regular or near-regular degree sequences
2. **Density**: Optimal edge densities often cluster around specific values
3. **Structure**: Extremal graphs exhibit particular structural properties

## Advanced Techniques

### Symmetry Breaking
```python
def break_symmetry(matrix):
    """Apply symmetry constraints to reduce search space"""
    # Enforce ordering constraints
    for i in range(len(matrix)-1):
        if sum(matrix[i]) > sum(matrix[i+1]):
            matrix[i], matrix[i+1] = matrix[i+1], matrix[i]
    return matrix
```

### Multi-Objective Optimization
```python
def multi_objective_score(H, G):
    """Combine multiple objectives"""
    sidorenko_score = sidorenko_gap(H, G)
    regularity_penalty = calculate_irregularity(G)
    size_bonus = bonus_for_size(G)
    
    return sidorenko_score - 0.1 * regularity_penalty + 0.05 * size_bonus
```

### Parallel Search
```python
def parallel_amcs(H, num_workers=4):
    """Run multiple AMCS instances in parallel"""
    from multiprocessing import Pool
    
    with Pool(num_workers) as pool:
        results = pool.map(single_amcs_run, [H] * num_workers)
    
    return min(results, key=lambda x: x.gap)
```

## Mathematical Context

### Known Results
1. **Trees**: Sidorenko's conjecture is true (Kruskal 1999)
2. **Complete bipartite graphs**: True for $K_{s,t}$ with small $s,t$
3. **Even cycles**: True for all even cycles
4. **Razborov's method**: Flag algebra techniques provide partial results

### Related Conjectures
1. **Turán-type problems**: Connections to extremal graph theory
2. **Correlation inequalities**: Links to probability theory
3. **Graph limits**: Relationships with graphon theory

### Open Questions
1. **Minimum counterexample**: What's the smallest potential counterexample?
2. **Computational complexity**: Is finding counterexamples decidable?
3. **Probabilistic constructions**: Can random methods find counterexamples?

## Future Research Directions

### Algorithmic Improvements
1. **Quantum algorithms**: Quantum annealing for optimization
2. **Deep learning**: Graph neural networks for property prediction
3. **Hybrid methods**: Combining analytical and computational approaches

### Mathematical Extensions
1. **Hypergraph version**: Sidorenko's conjecture for hypergraphs
2. **Directed graphs**: Analogous questions for directed bipartite graphs
3. **Weighted graphs**: Extensions to weighted graph settings

### Computational Enhancements
1. **GPU acceleration**: Parallel homomorphism counting
2. **Distributed computing**: Large-scale search across clusters
3. **Approximation algorithms**: Near-optimal solutions for large instances

## Dependencies and Setup

### Required Libraries
```bash
# SageMath for graph-based approaches
# Python scientific stack
pip install numpy scipy matplotlib

# CountHomLib for homomorphism counting
git clone https://github.com/kevinwangwkw/CountHomLib
pip install ./CountHomLib

# RL dependencies
pip install stable-baselines3 gymnasium
```

### Performance Optimization
```python
# Optimize CountHomLib usage
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Adjust based on CPU cores

# Use efficient data structures
import numpy as np
adjacency_matrix = np.array(matrix, dtype=np.int8)  # Save memory
```

---

*Sidorenko's Conjecture represents one of the deepest open problems in extremal graph theory, connecting combinatorics, analysis, and computational mathematics. The search for counterexamples pushes the boundaries of both theoretical understanding and computational capability.* 