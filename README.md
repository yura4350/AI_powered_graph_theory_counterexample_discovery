# AI-Powered Graph Theory Counterexample Discovery

A comprehensive research project combining **Adaptive Monte Carlo Search (AMCS)**, **Reinforcement Learning (RL)**, and **Hybrid RL+AMCS** algorithms to discover counterexamples for fundamental graph theory conjectures.

## 🎯 Project Overview

This repository contains implementations of advanced AI algorithms designed to systematically search for counterexamples to three major graph theory conjectures:

- **Critical Exponents Conjecture**
- **Second Neighborhood Conjecture** 
- **Sidorenko Conjecture**

Each conjecture is approached using multiple algorithmic strategies and different mathematical representations (graphs, graphons, matrices) to maximize the probability of discovering counterexamples.

## 🏗️ Repository Structure

```
AI_powered_graph_theory_counterexample_discovery/
├── CountHomLib/                    # Homomorphism counting library
├── critical_exponents_conjecture/  # Critical exponents research
│   ├── graph/                     # Graph-based approaches
│   └── graphon/                   # Graphon-based approaches
├── second_neighborhood_conjecture/ # Second neighborhood research
│   ├── amcs_approach/            # AMCS implementations
│   └── rl_approach/              # RL implementations
└── sidorenko_conjecture/          # Sidorenko conjecture research
    ├── graph/                    # Graph-based approaches
    ├── graphon/                  # Graphon-based approaches
    └── matrix/                   # Matrix-based approaches
```

## 🔬 Algorithmic Approaches

### 1. Adaptive Monte Carlo Search (AMCS)
A sophisticated heuristic search algorithm that:
- Uses nested Monte Carlo methods for local optimization
- Implements adaptive annealing strategies
- Employs graph perturbation techniques
- Provides systematic exploration of the search space

### 2. Reinforcement Learning (RL)
Machine learning approach featuring:
- **PPO (Proximal Policy Optimization)** agents
- Custom graph environments for each conjecture
- Reward functions based on conjecture-specific objectives
- Continuous learning and adaptation

### 3. Hybrid RL+AMCS
Combined approach that:
- Alternates between RL exploration and AMCS refinement
- Uses RL for global exploration and AMCS for local optimization
- Implements knowledge transfer between approaches
- Maximizes strengths of both methodologies

## 📊 Mathematical Representations

### Graphs
- **Standard graphs**: Traditional vertex-edge representations
- **Directed graphs**: For orientation-sensitive conjectures
- **Weighted graphs**: Supporting real-valued edge weights

### Graphons
- **Continuous representations**: Limit objects of graph sequences
- **Symmetric matrices**: Values in [0,1] representing edge probabilities
- **Homomorphism integration**: Using advanced calculus techniques

### Matrices
- **Eigenvalue methods**: Spectral approaches to conjecture analysis
- **Optimization on matrix spaces**: Direct search in matrix manifolds
- **Algebraic formulations**: Leveraging linear algebra properties

## 🛠️ CountHomLib Integration

This project extensively uses [CountHomLib](https://github.com/kevinwangwkw/CountHomLib), an efficient C++ library for counting graph homomorphisms with Python bindings.

### Key Features:
- **Polynomial-time algorithms** for tree-width bounded graphs
- **Dynamic programming** on nice tree decompositions
- **Graphon support** for continuous representations
- **Optimized implementation** using modern C++ and OpenMP

### Installation:
```bash
git clone https://github.com/kevinwangwkw/CountHomLib
pip3 install ./CountHomLib
```

For macOS users encountering OpenMP issues:
```bash
brew install llvm
LDFLAGS="-L$(brew --prefix llvm)/lib" \
CPPFLAGS="-I$(brew --prefix llvm)/include" \
CC="$(brew --prefix llvm)/bin/clang" \
CXX="$(brew --prefix llvm)/bin/clang++" \
pip3 install ./CountHomLib
```

## 🎯 Research Conjectures

### [Critical Exponents Conjecture](critical_exponents_conjecture/)
**Objective**: Find graphs maximizing the ratio log(t(H₁,T)) / log(t(H₂,T)) for specific graphs H₁, H₂.

**Applications**: Understanding extremal properties of homomorphism densities and their asymptotic behavior.

### [Second Neighborhood Conjecture](second_neighborhood_conjecture/)
**Objective**: Find oriented graphs where every vertex v satisfies |N⁺⁺(v)| ≥ |N⁺(v)|.

**Applications**: Structural graph theory and tournament analysis.

### [Sidorenko Conjecture](sidorenko_conjecture/)
**Objective**: Find counterexamples to the conjecture that bipartite graphs H satisfy t(H,G) ≥ t(K₂,G)^|E(H)|.

**Applications**: Extremal graph theory and combinatorial optimization.

## 🚀 Getting Started

### Prerequisites
```bash
# Python dependencies
pip install numpy scipy matplotlib networkx sage
pip install stable-baselines3 gymnasium

# CountHomLib installation (see above)
```

### Running Experiments

1. **AMCS Approach**:
```bash
cd critical_exponents_conjecture/graph/amcs_approach/
python main.py
```

2. **RL Approach**:
```bash
cd critical_exponents_conjecture/graph/rl_approach/
python train_rl.py
```

3. **Hybrid RL+AMCS**:
```bash
cd critical_exponents_conjecture/graph/rl+amcs_approach/
python main.py
```

## 📈 Research Methodology

### Experimental Design
- **Systematic parameter sweeps** across algorithm hyperparameters
- **Statistical analysis** of convergence properties
- **Comparative evaluation** between algorithmic approaches
- **Scalability studies** on varying graph sizes

### Performance Metrics
- **Objective function values** (conjecture-specific)
- **Convergence rates** and stability analysis
- **Computational efficiency** and resource utilization
- **Solution quality** and mathematical rigor

### Validation
- **Mathematical verification** of discovered counterexamples
- **Cross-validation** across different implementations
- **Peer review** and community validation
- **Reproducibility** through detailed documentation

## 🔬 Advanced Features

### Optimization Techniques
- **Adaptive annealing schedules** in AMCS
- **Custom neural network architectures** for RL
- **Hybrid exploration strategies** combining global and local search
- **Multi-objective optimization** for competing criteria

### Computational Efficiency
- **Parallel processing** using multiprocessing and threading
- **Efficient graph representations** minimizing memory usage
- **Optimized homomorphism counting** via CountHomLib
- **Smart caching** of expensive computations

### Extensibility
- **Modular design** enabling easy addition of new conjectures
- **Abstract base classes** for algorithmic approaches
- **Configuration-driven** experimentation
- **Plugin architecture** for custom scoring functions

## 🤝 Contributing

We welcome contributions from the graph theory and machine learning communities:

1. **New conjectures**: Implement additional graph theory problems
2. **Algorithm improvements**: Enhance existing AMCS, RL, or hybrid methods
3. **Performance optimization**: Improve computational efficiency
4. **Documentation**: Expand mathematical explanations and usage examples

## 📚 References

- Díaz, J., Serna, M., & Thilikos, D. M. (2002). Counting H-colorings of partial k-trees
- Lovász, L. (2012). Large networks and graph limits
- Sidorenko, A. F. (1993). A correlation inequality for bipartite graphs
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎓 Citation

If you use this code in your research, please cite:
```bibtex
@software{ai_graph_counterexamples,
  title={AI-Powered Graph Theory Counterexample Discovery},
  author={[Author Names]},
  year={2024},
  url={https://github.com/[username]/AI_powered_graph_theory_counterexample_discovery}
}
```

## 📞 Contact

For questions, suggestions, or collaborations, please open an issue or contact the maintainers.

---

*This research project represents the intersection of artificial intelligence and pure mathematics, pushing the boundaries of automated mathematical discovery.* 