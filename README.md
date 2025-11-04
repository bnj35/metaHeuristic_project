# Vehicle Routing Problem (VRP) Metaheuristic Solvers

This project implements multiple metaheuristic algorithms to solve Vehicle Routing Problems (VRP) with a focus on real-world scenarios involving capacity-constrained vehicles.

## 📋 Overview

The project provides two different approaches to solving VRP:
- **Genetic Algorithm (GA)**: Population-based evolutionary metaheuristic
- **PyVRP**: State-of-the-art hybrid genetic algorithm solver

Additionally, a comparison module allows you to evaluate and compare both approaches.

## 🎯 Problem Description

**Instance**: VRP1000
- **Type**: Capacitated Vehicle Routing Problem (CVRP)
- **Nodes**: 1001 (1 depot + 1000 customers)
- **Vehicle Capacity**: 1000 units
- **Configuration**: Single truck making multiple trips from depot
- **Edge Weights**: Euclidean distances calculated from coordinates

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Build and start the container
docker-compose up

# Access Jupyter Lab at http://localhost:8888
```

The container automatically starts Jupyter Lab with all dependencies installed.

### Running Algorithms

#### Genetic Algorithm
```bash
# Inside Docker container
docker exec -it metaHeuristic python app/ga.py
```

#### PyVRP Solver
```bash
# Inside Docker container
docker exec -it metaHeuristic python app/vrp.py
```

#### Compare Algorithms
```bash
# Inside Docker container
docker exec -it metaHeuristic python app/compare.py
```

## 📁 Project Structure

```
metaHeuristic/
├── app/
│   ├── __init__.py           # Package initialization
│   ├── ga.py                 # Genetic Algorithm implementation
│   ├── vrp.py                # PyVRP solver wrapper
│   ├── compare.py            # Algorithm comparison module
│   └── pyproject.toml        # Package configuration
├── data/
│   ├── instance.vrp          # VRP instance (1000 customers)
│   ├── ga_solution.sol       # GA solution output
│   └── pyvrp_solution.sol    # PyVRP solution output
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile                # Container build instructions
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── GA_README.md             # Detailed GA documentation
```

## 🧬 Genetic Algorithm

The Genetic Algorithm is a population-based metaheuristic that evolves solutions through natural selection principles.

### Key Features

- **Population Initialization**: Mix of random and greedy (nearest neighbor) solutions
- **Selection**: Tournament selection (size 3)
- **Crossover**: Order Crossover (OX) and Partially Mapped Crossover (PMX)
- **Mutation**: Swap, inversion, and scramble operators
- **Local Search**: Periodic 2-opt refinement
- **Elitism**: Preserves best solutions across generations

### Usage Example

```python
from app.ga import GeneticAlgorithmSolver
import numpy as np

# Define problem
coords = np.array([[50, 50], [23, 67], [78, 34], ...])
demands = np.array([0, 5, 8, 3, ...])
capacity = 100

# Create and run solver
solver = GeneticAlgorithmSolver(coords, demands, capacity)
solution, cost = solver.solve(
    pop_size=100,
    num_generations=100,
    crossover_rate=0.8,
    mutation_rate=0.2,
    elitism_count=2,
    local_search_freq=10,
    seed=42
)

# Display results
solver.print_solution()
solver.plot_solution(output_path='/data/ga_solution.png')
solver.plot_convergence(output_path='/data/ga_convergence.png')
solver.plot_diversity(output_path='/data/ga_diversity.png')
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 100 | Population size |
| `num_generations` | 100 | Number of generations to evolve |
| `crossover_rate` | 0.8 | Probability of crossover (0.0-1.0) |
| `mutation_rate` | 0.2 | Probability of mutation (0.0-1.0) |
| `elitism_count` | 2 | Number of best solutions to preserve |
| `local_search_freq` | 10 | Apply 2-opt every N generations |
| `seed` | None | Random seed for reproducibility |

For detailed GA documentation, see [GA_README.md](GA_README.md).

## 🔧 PyVRP Solver

PyVRP is a state-of-the-art VRP solver using hybrid genetic algorithms with advanced local search.

### Usage Example

```python
from app.vrp import VRPSolver

# Create solver
solver = VRPSolver('data/instance.vrp')

# Solve with time limit
result = solver.solve(time_limit=120, seed=42)

# Display and save results
solver.print_solution()
solver.save_solution('/data/pyvrp_solution.sol')
```

### Configuration

- **Time Limit**: Default 120 seconds
- **Vehicle Capacity**: 1000 units per trip
- **Distance Metric**: Euclidean distances
- **Output Format**: VRPLIB .sol format

## 🔬 Algorithm Comparison

Use the comparison module to evaluate both algorithms side-by-side:

```python
from app.compare import main as compare_algorithms

# Run comparison
compare_algorithms()
```

This will:
1. Run the Genetic Algorithm
2. Verify both GA and PyVRP solutions
3. Compare costs, routes, and feasibility
4. Generate comparison visualizations

### Comparison Features

- **Solution Verification**: Validates feasibility of both solutions
- **Cost Comparison**: Compares total distance traveled
- **Route Analysis**: Compares number of routes and route structures
- **Visualizations**: Bar charts comparing metrics
- **Statistical Analysis**: Percentage differences and performance metrics

## 📊 Visualization

All solvers provide comprehensive visualization capabilities:

### Solution Plots
- Routes displayed on 2D coordinate map
- Each route shown in different color
- Depot marked in red
- Automatic legend generation

### Convergence Plots (GA)
- Best cost evolution over generations
- Average population cost
- Helps assess algorithm performance

### Diversity Plots (GA)
- Population diversity over time
- Difference between average and best fitness
- Indicates exploration vs. exploitation balance

### Comparison Plots
- Side-by-side cost comparison
- Route count comparison
- Performance metrics visualization

## 🔬 Algorithm Comparison

| Feature | Genetic Algorithm | PyVRP |
|---------|-------------------|-------|
| **Type** | Metaheuristic | Hybrid GA |
| **Population** | 100 individuals | Multiple |
| **Approach** | Evolution + local search | Advanced hybrid |
| **Speed** | Fast-Medium | Medium-Slow |
| **Quality** | Good | Excellent |
| **Customization** | Highly flexible | Limited |
| **Use Case** | Research, learning | Production, benchmarks |
| **Output** | Routes + visualizations | Routes + statistics |

## 📦 Dependencies

### Core Dependencies
```txt
numpy          # Numerical computations
matplotlib     # Visualization
vrplib         # VRP instance file parser
pyvrp          # State-of-the-art VRP solver
```

### Development Dependencies
```txt
pytest         # Testing framework
jupyter        # Interactive notebooks
jupyterlab     # Enhanced Jupyter interface
```

## 🛠️ Development

### Adding New Algorithms

1. Create new solver file in `app/` directory (e.g., `app/new_algorithm.py`)
2. Implement solver class with standard interface:
   ```python
   class NewAlgorithmSolver:
       def __init__(self, coords, demands, capacity, depot_idx=0):
           # Initialize
           
       def solve(self, **params):
           # Solve and return (routes, cost)
           
       def print_solution(self):
           # Display results
           
       def plot_solution(self, output_path=None):
           # Visualize routes
   ```
3. Add comparison in [app/compare.py](app/compare.py)
4. Update documentation


### Jupyter Notebooks

Access Jupyter Lab at `http://localhost:8888` after starting the container. Create notebooks in the `app/` directory for interactive experimentation.

## 🎯 Expected Results

For the VRP1000 instance (1000 customers, capacity 1000):

| Algorithm | Typical Cost | Routes | Time (100 iter/gen) |
|-----------|--------------|--------|---------------------|
| Genetic Algorithm | ~1777 | 1 | 60-120s |
| PyVRP | ~1455 | 1 | 120s |

*Results vary based on random seed and parameters*

**Note**: With uniform demand=1 and capacity=1000, all customers can be served in a single route (TSP variant).

## 🐛 Troubleshooting

### Container Issues
```bash
# Rebuild container
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Module Import Errors
Ensure you're running commands from the correct directory:
```bash
# Inside container, run from /app/
cd /app
python ga.py
```

### Memory Issues
For large instances, reduce population size:
```python
solver.solve(pop_size=50)  # Instead of 100
```

### Visualization Not Working
Ensure matplotlib backend is set correctly:
```python
import matplotlib
matplotlib.use('Agg')  # For non-interactive plots
```

### Solution File Not Found
Make sure solutions are saved to `/data/` directory:
```python
solver.save_solution('/data/solution.sol')
```

## 📝 Solution Format

Solutions are saved in VRPLIB .sol format:

```
1455
Route #1: 147 475 495 455 45 759 789 923 ...
```

- First line: Total cost (integer)
- Following lines: Routes with customer indices

Both [data/ga_solution.sol](data/ga_solution.sol) and [data/pyvrp_solution.sol](data/pyvrp_solution.sol) follow this format.

## 📚 References

### PyVRP
- [PyVRP Documentation](https://pyvrp.org/)
- Wouda, N. A., et al. (2023). *PyVRP: A high-performance VRP solver package*.



## 🔗 Related Documentation

- [Genetic Algorithm Details](GA_README.md) - In-depth GA documentation
- [PyVRP Official Docs](https://pyvrp.org/) - PyVRP library documentation

---

**Quick Commands Reference**:
```bash
# Start environment
docker-compose up

# Run GA solver
docker exec -it metaHeuristic python app/ga.py

# Run PyVRP solver
docker exec -it metaHeuristic python app/vrp.py

# Compare algorithms
docker exec -it metaHeuristic python app/compare.py

# Access Jupyter Lab
# Open browser at http://localhost:8888
```