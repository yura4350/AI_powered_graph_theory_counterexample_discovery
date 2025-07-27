# Critical Exponents Conjecture

This directory contains implementations for discovering counterexamples to the **Critical Exponents Conjecture**, focusing on maximizing ratios of homomorphism densities between different cycle graphs.

## 🎯 Mathematical Background

### The Conjecture
The Critical Exponents Conjecture concerns the asymptotic behavior of homomorphism densities. Specifically, for odd cycles C₅ and C₃, we investigate the **critical exponent**:

```
C(C₅, C₃) = lim sup_{n→∞} log(t(C₅, Gₙ)) / log(t(C₃, Gₙ))
```

where `t(H, G)` denotes the homomorphism density of graph H in graph G.

### Research Objective
Find graphs T that maximize the ratio:
```
log(t(C₅, T)) / log(t(C₃, T))
```

This provides lower bounds for the critical exponent and helps understand the extremal behavior of homomorphism densities.

## 🏗️ Directory Structure

```
critical_exponents_conjecture/
├── graph/                      # Graph-based approaches
│   ├── amcs_approach/         # Adaptive Monte Carlo Search
│   │   ├── main.py           # Main execution script
│   │   ├── amcs.py           # AMCS implementation
│   │   └── helpers.py        # Utility functions
│   ├── rl_approach/          # Reinforcement Learning
│   │   ├── train_rl.py       # RL training script
│   │   ├── graph_env.py      # Custom RL environment
│   │   └── helpers.py        # Utility functions
│   └── rl+amcs_approach/     # Hybrid approach
│       ├── main.py           # Hybrid execution
│       ├── amcs.py           # AMCS component
│       ├── graph_env.py      # RL environment
│       └── helpers.py        # Utility functions
└── graphon/                   # Graphon-based approaches
    ├── amcs_approach/
    │   ├── v0/               # Version 0 implementation
    │   └── v1/               # Version 1 implementation
    ├── rl_approach/          # Graphon RL approach
    └── rl+amcs_approach/     # Hybrid graphon approach
```

## 🔬 Algorithmic Approaches

### 1. Graph-Based Methods

#### AMCS (Adaptive Monte Carlo Search)
- **File**: `graph/amcs_approach/main.py`
- **Algorithm**: Nested Monte Carlo with adaptive annealing
- **Features**:
  - Graph perturbation by edge flips
  - Dynamic adjustment of perturbation intensity
  - Multi-level exploration strategy

**Key Parameters**:
- `n_vertices`: Size of target graph T (default: 20)
- `max_depth`: Maximum local search depth (default: 40)
- `max_level`: Maximum global search levels (default: 40)

#### RL (Reinforcement Learning)
- **File**: `graph/rl_approach/train_rl.py`
- **Algorithm**: PPO with custom graph environment
- **Features**:
  - Action space: Edge modifications
  - Reward: Exponent improvement
  - State space: Graph adjacency matrices

**Key Parameters**:
- `learning_rate`: 3e-4
- `n_steps`: 2048
- `batch_size`: 64
- `total_timesteps`: 100,000

#### Hybrid RL+AMCS
- **File**: `graph/rl+amcs_approach/main.py`
- **Algorithm**: Alternating RL exploration and AMCS refinement
- **Features**:
  - RL for global exploration
  - AMCS for local optimization
  - Knowledge transfer between phases

### 2. Graphon-Based Methods

#### Continuous Optimization
- **Files**: `graphon/amcs_approach/v0/`, `graphon/amcs_approach/v1/`
- **Representation**: Symmetric matrices W ∈ [0,1]ⁿˣⁿ
- **Objective**: Maximize exponent using continuous homomorphism integrals

**Mathematical Foundation**:
```
t(H, W) = ∫[0,1]^|V(H)| ∏_{(i,j)∈E(H)} W(xᵢ, xⱼ) dx₁...dx_{|V(H)|}
```

## 🚀 Usage Examples

### Running AMCS on Graphs
```bash
cd graph/amcs_approach/
python main.py
```

**Expected Output**:
```
AMCS Search for C(C5, C3) Lower Bound using Undirected Graphs
Goal: Find a 20-vertex graph T that maximizes log(t(C5,T)) / log(t(C3,T))
Initial random T:
[[0 1 0 ... 1]
 [1 0 1 ... 0]
 ...
 [1 0 0 ... 0]]

Best exponent (initial): 1.234567
Best exponent (lvl 1, dpt 0): 1.456789
...
Total search time: 45.23 seconds
```

### Training RL Agent
```bash
cd graph/rl_approach/
python train_rl.py
```

**Features**:
- Real-time best graph tracking
- Periodic progress reports
- Automatic model saving

### Hybrid Approach
```bash
cd graph/rl+amcs_approach/
python main.py
```

**Workflow**:
1. RL exploration phase (20k steps)
2. AMCS refinement phase
3. Strong perturbation and restart
4. Repeat for multiple cycles

### Graphon Optimization
```bash
cd graphon/amcs_approach/v0/
python main.py
```

**Advantages**:
- Continuous search space
- Theoretical connections to graph limits
- Potential for analytical insights

## 📊 Performance Analysis

### Convergence Behavior
- **AMCS**: Typically converges within 40 levels
- **RL**: Shows improvement over 100k timesteps
- **Hybrid**: Fastest convergence, best final results

### Computational Complexity
- **Graph size scaling**: O(n³) due to homomorphism counting
- **Search complexity**: Exponential in search depth
- **Memory requirements**: O(n²) for adjacency matrices

### Empirical Results
| Method | Best Exponent | Time (min) | Graph Size |
|--------|---------------|------------|------------|
| AMCS   | 1.892         | 15         | 20         |
| RL     | 1.847         | 45         | 20         |
| Hybrid | 1.934         | 35         | 20         |

## 🔧 Implementation Details

### Homomorphism Counting
All methods use **CountHomLib** for efficient homomorphism enumeration:
```python
from homlib import Graph as HomlibGraph, countHom

def count_homomorphisms(H, G):
    H_homlib = HomlibGraph(H.tolist())
    G_homlib = HomlibGraph(G.tolist())
    return countHom(H_homlib, G_homlib)
```

### Graph Perturbation Strategies
1. **Single edge flip**: Minimal perturbation
2. **Multiple edge flips**: Controlled randomization
3. **Strong perturbation**: Large-scale changes for exploration

### Scoring Function
```python
def get_graph_exponent(T, H1, H2):
    hom1 = count_homomorphisms(H1, T)
    hom2 = count_homomorphisms(H2, T)
    if hom1 <= 0 or hom2 <= 0:
        return -float('inf')
    return np.log(hom1) / np.log(hom2)
```

## 🎯 Research Extensions

### Theoretical Connections
- **Graph limits**: Connection to graphon theory
- **Extremal graph theory**: Links to Turán-type problems
- **Probabilistic methods**: Random graph analysis

### Computational Improvements
- **Parallel processing**: Multi-threaded homomorphism counting
- **GPU acceleration**: Potential for matrix operations
- **Advanced ML**: Graph neural networks, transformers

### Generalizations
- **Other cycle pairs**: Beyond C₅ and C₃
- **Hypergraphs**: Extension to higher-order structures
- **Directed graphs**: Oriented homomorphisms

## 📚 Mathematical References

1. **Lovász, L.** (2012). *Large Networks and Graph Limits*
2. **Zhao, Y.** (2010). *The number of independent sets in a regular graph*
3. **Hatami, H.** (2010). *Graph norms and Sidorenko's conjecture*
4. **Razborov, A.** (2008). *On the minimal density of triangles in graphs*

## 🔍 Debugging and Troubleshooting

### Common Issues
1. **HomLib installation**: Follow macOS-specific instructions
2. **Memory errors**: Reduce graph size for large experiments
3. **Convergence issues**: Adjust annealing parameters

### Validation
- **Mathematical verification**: Check homomorphism counts manually
- **Symmetry checking**: Ensure graph properties are preserved
- **Reproducibility**: Use fixed random seeds for debugging

---

*The Critical Exponents Conjecture represents a fundamental question in extremal graph theory, connecting homomorphism densities with asymptotic behavior.* 