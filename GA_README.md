# Genetic Algorithm for Vehicle Routing Problem (VRP)

## Overview

This implementation provides a **Genetic Algorithm (GA)** metaheuristic to solve Vehicle Routing Problems. The GA is a population-based evolutionary algorithm inspired by natural selection and genetics.

## Key Features

### 1. **Population Initialization**
- **Random Solutions**: Generates random permutations of customers
- **Greedy Solutions**: Uses nearest neighbor heuristic for better initial population
- **Hybrid Approach**: Combines both random and greedy solutions (20% greedy by default)

### 2. **Genetic Operators**

#### Selection
- **Tournament Selection**: Selects parents by running tournaments among random individuals
- Tournament size: 3 (configurable)

#### Crossover (Recombination)
Two crossover methods implemented:
- **Order Crossover (OX)**: Preserves relative order of cities
- **Partially Mapped Crossover (PMX)**: Maintains position information
- Crossover rate: 0.8 (80% probability)

#### Mutation
Three mutation operators:
- **Swap Mutation**: Swaps two random positions
- **Inversion Mutation**: Reverses a random segment
- **Scramble Mutation**: Shuffles a random segment
- Mutation rate: 0.2 (20% probability)

### 3. **Local Search Enhancement**
- **2-opt Local Search**: Applied periodically to best individuals
- Frequency: Every 10 generations (configurable)
- Helps refine solutions and escape local optima

### 4. **Elitism**
- Preserves top 2 individuals (configurable) in each generation
- Ensures best solutions are not lost

## Algorithm Parameters

```python
solver.solve(
    pop_size=100,           # Population size
    num_generations=100,    # Number of generations
    crossover_rate=0.8,     # Probability of crossover
    mutation_rate=0.2,      # Probability of mutation
    elitism_count=2,        # Number of elite individuals
    local_search_freq=10,   # Apply local search every N generations
    seed=42                 # Random seed for reproducibility
)
```

## How It Works

### Chromosome Representation
- A chromosome is a **permutation of customer indices**
- Example: `[3, 1, 4, 2, 5]` means visit customers in this order
- Routes are created by splitting the permutation based on capacity constraints

### Fitness Evaluation
- Fitness = Total travel distance of all routes
- Lower cost = Better fitness
- Routes are created by sequentially adding customers until capacity is reached

### Evolution Process

1. **Initialize** population with random and greedy solutions
2. **For each generation:**
   - Evaluate fitness of all individuals
   - Select parents using tournament selection
   - Create offspring using crossover
   - Apply mutation to offspring
   - Apply local search (periodically)
   - Keep elite individuals
   - Form new generation
3. **Return** best solution found

## Usage

### Basic Usage

```python
from ga import GeneticAlgorithmSolver
import numpy as np

# Define problem
coords = np.array([[50, 50], [23, 67], [78, 34], ...])  # Coordinates
demands = np.array([0, 5, 8, 3, ...])                    # Demands (depot = 0)
capacity = 100                                            # Vehicle capacity

# Create solver
solver = GeneticAlgorithmSolver(coords, demands, capacity)

# Solve
solution, cost = solver.solve(
    pop_size=100,
    num_generations=100,
    seed=42
)

# Display results
solver.print_solution()
```

### From VRP File

```python
import vrplib
from ga import GeneticAlgorithmSolver

# Load instance
data = vrplib.read_instance('data/instance.vrp')
coords = np.array(data['node_coord'])
demands = np.array(data['demand']) if 'demand' in data else np.ones(len(coords))
capacity = data.get('capacity', 100)

# Solve
solver = GeneticAlgorithmSolver(coords, demands, capacity)
solution, cost = solver.solve()
solver.print_solution()
```

### Run Standalone

```bash
# Inside Docker container
python app/ga.py
```

## Visualization

The solver provides three types of visualizations:

### 1. Solution Plot
```python
solver.plot_solution(output_path='/data/ga_solution.png')
```
Shows routes on a 2D map with different colors for each route.

### 2. Convergence Plot
```python
solver.plot_convergence(output_path='/data/ga_convergence.png')
```
Shows best and average fitness over generations.

### 3. Diversity Plot
```python
solver.plot_diversity(output_path='/data/ga_diversity.png')
```
Shows population diversity (difference between average and best).

## Comparison with Other Approaches

The Genetic Algorithm can be compared with other metaheuristics or exact solvers.

### Key Characteristics

| Aspect | Genetic Algorithm |
|--------|-------------------|
| **Approach** | Population-based evolution |
| **Search Strategy** | Multiple solutions evolve together |
| **Diversification** | Crossover + mutation |
| **Intensification** | Elitism + local search |
| **Memory** | Maintains entire population |
| **Typical Speed** | Medium (multiple evaluations) |
| **Solution Quality** | Good, benefits from recombination |

## Advantages of Genetic Algorithm

1. **Population Diversity**: Explores multiple regions of search space simultaneously
2. **Recombination**: Can combine good features from different solutions
3. **Global Search**: Less likely to get stuck in local optima
4. **Flexibility**: Easy to adapt with different operators
5. **Parallel Evaluation**: Population can be evaluated in parallel

## Disadvantages

1. **Computational Cost**: Evaluates many solutions per generation
2. **Parameter Sensitivity**: Performance depends on parameter tuning
3. **No Guarantee**: May converge prematurely or slowly
4. **Memory Usage**: Stores entire population

## Performance Tips

### For Large Instances (1000+ customers)

1. **Adjust Population Size**: Balance between diversity and speed
   ```python
   pop_size=50  # Smaller for faster iterations
   ```

2. **Increase Local Search Frequency**: 
   ```python
   local_search_freq=5  # More frequent refinement
   ```

3. **Adaptive Mutation Rate**: Start high, decrease over time
   ```python
   mutation_rate = 0.3 - (generation / num_generations) * 0.2
   ```

4. **Hybrid Initialization**: Use more greedy solutions
   ```python
   # In initialize_population()
   greedy_ratio=0.5  # 50% greedy
   ```

### For Small Instances (<100 customers)

1. **Larger Population**: More exploration
   ```python
   pop_size=200
   ```

2. **More Generations**: Allow thorough search
   ```python
   num_generations=200
   ```

3. **Lower Mutation Rate**: Preserve good solutions
   ```python
   mutation_rate=0.1
   ```

## Example Output

```
Starting Genetic Algorithm
Population size: 100
Generations: 100
Crossover rate: 0.8
Mutation rate: 0.2
Elitism: 2
Number of customers: 1000
Vehicle capacity: 100
------------------------------------------------------------
Generation 1: New best cost = 4523.45
Generation 10/100: Best = 4234.12, Avg = 4456.78
Generation 20/100: Best = 4123.56, Avg = 4298.34
...
Generation 100/100: Best = 3987.23, Avg = 4012.45
------------------------------------------------------------
Genetic Algorithm completed!
Best cost found: 3987.23
Number of routes: 11

============================================================
GENETIC ALGORITHM VRP SOLUTION
============================================================
Total cost: 3987.23
Number of routes: 11

Routes:
  Route 1: [0, 234, 567, 123, ..., 0]
    Demand: 98/100, Cost: 362.45
  ...
```

## Advanced Customization

### Custom Crossover Operator

```python
def custom_crossover(self, parent1, parent2):
    # Implement your crossover logic
    offspring1 = ...
    offspring2 = ...
    return offspring1, offspring2

# Replace in solver
solver.order_crossover = custom_crossover
```

### Custom Fitness Function

```python
def custom_fitness(self, chromosome):
    routes = self._convert_to_routes(chromosome)
    cost = self._calculate_total_cost(routes)
    
    # Add penalty for number of routes
    num_routes_penalty = len(routes) * 100
    
    return cost + num_routes_penalty
```