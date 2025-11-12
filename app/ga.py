"""
Genetic Algorithm (GA) Metaheuristic for VRP.

This module implements the Genetic Algorithm metaheuristic for solving Vehicle Routing Problems.
GA consists of the following main components:
1. Population Initialization: Create initial population of solutions
2. Selection: Select parents for reproduction
3. Crossover: Combine parents to create offspring
4. Mutation: Apply random changes to maintain diversity
5. Replacement: Form new generation from parents and offspring
"""

import numpy as np
import random
from typing import List, Tuple, Set, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from copy import deepcopy


class GeneticAlgorithmSolver:
    """Genetic Algorithm metaheuristic solver for Vehicle Routing Problems."""
    
    def __init__(self, coords: np.ndarray, demands: np.ndarray, capacity: int, depot_idx: int = 0):
        """
        Initialize Genetic Algorithm solver.
        
        Args:
            coords: Array of (x, y) coordinates for all locations
            demands: Array of demands for each customer
            capacity: Vehicle capacity
            depot_idx: Index of the depot (default: 0)
        """
        self.coords = np.array(coords)
        self.demands = np.array(demands)
        self.capacity = capacity
        self.depot_idx = depot_idx
        self.num_locations = len(coords)
        self.distance_matrix = self._calculate_distance_matrix()
        
        self.best_solution = None
        self.best_cost = float('inf')
        self.generation_costs = []
        self.avg_costs = []
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate Euclidean distance matrix between all locations."""
        n = len(self.coords)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i, j] = np.sqrt(
                        (self.coords[i][0] - self.coords[j][0]) ** 2 +
                        (self.coords[i][1] - self.coords[j][1]) ** 2
                    )
        
        return distances
    
    def _calculate_route_cost(self, route: List[int]) -> float:
        """Calculate the total cost of a route including return to depot."""
        if not route:
            return 0.0
        
        cost = self.distance_matrix[self.depot_idx, route[0]]  # Depot to first customer
        
        for i in range(len(route) - 1):
            cost += self.distance_matrix[route[i], route[i + 1]]
        
        cost += self.distance_matrix[route[-1], self.depot_idx]  # Last customer to depot
        
        return cost
    
    def _calculate_total_cost(self, routes: List[List[int]]) -> float:
        """Calculate total cost of all routes."""
        return sum(self._calculate_route_cost(route) for route in routes)
    
    def _is_feasible_route(self, route: List[int]) -> bool:
        """Check if a route respects capacity constraint."""
        total_demand = sum(self.demands[customer] for customer in route)
        return total_demand <= self.capacity
    
    def _convert_to_routes(self, chromosome: List[int]) -> List[List[int]]:
        """
        Convert a chromosome (permutation of customers) to routes.
        Split by capacity constraint.
        
        Args:
            chromosome: Permutation of customer indices
            
        Returns:
            List of routes
        """
        routes = []
        current_route = []
        current_capacity = 0
        
        for customer in chromosome:
            customer_demand = self.demands[customer]
            
            if current_capacity + customer_demand <= self.capacity:
                current_route.append(customer)
                current_capacity += customer_demand
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [customer]
                current_capacity = customer_demand
        
        if current_route:
            routes.append(current_route)
        
        return routes
    
    def _evaluate_fitness(self, chromosome: List[int]) -> float:
        """
        Evaluate fitness of a chromosome.
        Lower cost = better fitness.
        
        Args:
            chromosome: Permutation of customer indices
            
        Returns:
            Total cost (fitness value)
        """
        routes = self._convert_to_routes(chromosome)
        return self._calculate_total_cost(routes)
    
    def _create_random_chromosome(self) -> List[int]:
        """Create a random chromosome (random permutation of customers)."""
        customers = list(range(self.num_locations))
        customers.remove(self.depot_idx)
        random.shuffle(customers)
        return customers
    
    def _create_greedy_chromosome(self) -> List[int]:
        """Create a chromosome using nearest neighbor heuristic."""
        unvisited = set(range(self.num_locations))
        unvisited.remove(self.depot_idx)
        
        chromosome = []
        current = self.depot_idx
        
        while unvisited:
            # Find nearest unvisited customer
            nearest = min(unvisited, key=lambda c: self.distance_matrix[current, c])
            chromosome.append(nearest)
            current = nearest
            unvisited.remove(nearest)
        
        return chromosome
    
    def _create_savings_chromosome(self) -> List[int]:
        """
        Create a chromosome using Clarke-Wright Savings Algorithm.
        This provides a high-quality initial solution.
        
        Returns:
            Chromosome (permutation of customers) based on savings heuristic
        """
        customers = list(range(self.num_locations))
        customers.remove(self.depot_idx)
        
        # Calculate savings for all pairs of customers
        savings = []
        for i in customers:
            for j in customers:
                if i < j:
                    # Savings = distance(depot,i) + distance(depot,j) - distance(i,j)
                    s = (self.distance_matrix[self.depot_idx, i] + 
                         self.distance_matrix[self.depot_idx, j] - 
                         self.distance_matrix[i, j])
                    savings.append((s, i, j))
        
        # Sort savings in descending order
        savings.sort(reverse=True, key=lambda x: x[0])
        
        # Build routes using savings - each customer starts in its own route
        # Use a dictionary mapping customer to route
        customer_to_route = {c: [c] for c in customers}
        route_demands = {c: self.demands[c] for c in customers}
        
        for saving, i, j in savings:
            # Check if i and j are in different routes
            if i not in customer_to_route or j not in customer_to_route:
                continue
                
            route_i = customer_to_route[i]
            route_j = customer_to_route[j]
            
            # Skip if already in the same route
            if route_i is route_j:
                continue
            
            # Check if i and j are at the ends of their routes
            i_at_end = (route_i[0] == i or route_i[-1] == i)
            j_at_end = (route_j[0] == j or route_j[-1] == j)
            
            if i_at_end and j_at_end:
                # Calculate combined demand
                demand_i = sum(self.demands[c] for c in route_i)
                demand_j = sum(self.demands[c] for c in route_j)
                combined_demand = demand_i + demand_j
                
                # Check capacity constraint
                if combined_demand <= self.capacity:
                    # Merge routes based on which ends match
                    if route_i[-1] == i and route_j[0] == j:
                        new_route = route_i + route_j
                    elif route_i[0] == i and route_j[-1] == j:
                        new_route = route_j + route_i
                    elif route_i[-1] == i and route_j[-1] == j:
                        new_route = route_i + route_j[::-1]
                    elif route_i[0] == i and route_j[0] == j:
                        new_route = route_i[::-1] + route_j
                    else:
                        continue
                    
                    # Update all customers in both routes to point to the new merged route
                    for c in route_j:
                        customer_to_route[c] = new_route
                    for c in route_i:
                        customer_to_route[c] = new_route
        
        # Collect unique routes (avoid duplicates)
        seen = set()
        routes = []
        for route in customer_to_route.values():
            route_tuple = tuple(route)
            if route_tuple not in seen:
                seen.add(route_tuple)
                routes.append(route)
        
        # Convert routes to chromosome (flatten routes)
        chromosome = [customer for route in routes for customer in route]
        return chromosome
    
    def _create_sweep_chromosome(self) -> List[int]:
        """
        Create a chromosome using Sweep Algorithm.
        Customers are sorted by polar angle from depot and grouped into routes.
        
        Returns:
            Chromosome based on sweep heuristic
        """
        customers = list(range(self.num_locations))
        customers.remove(self.depot_idx)
        
        # Calculate polar angles from depot
        depot_coord = self.coords[self.depot_idx]
        angles = {}
        
        for c in customers:
            dx = self.coords[c][0] - depot_coord[0]
            dy = self.coords[c][1] - depot_coord[1]
            angle = np.arctan2(dy, dx)
            angles[c] = angle
        
        # Sort customers by angle
        sorted_customers = sorted(customers, key=lambda c: angles[c])
        
        # Group into feasible routes
        chromosome = []
        current_load = 0
        
        for customer in sorted_customers:
            if current_load + self.demands[customer] <= self.capacity:
                chromosome.append(customer)
                current_load += self.demands[customer]
            else:
                # Start new route - add some randomization at route boundaries
                chromosome.append(customer)
                current_load = self.demands[customer]
        
        return chromosome
    
    def initialize_population(self, pop_size: int, heuristic_ratio: float = 0.3,
                            use_savings: bool = True, use_sweep: bool = True) -> List[List[int]]:
        """
        Initialize population with mix of heuristic and random solutions.
        
        Args:
            pop_size: Population size
            heuristic_ratio: Ratio of heuristic-based solutions (vs random)
            use_savings: Include Clarke-Wright Savings solutions
            use_sweep: Include Sweep algorithm solutions
            
        Returns:
            Initial population
        """
        population = []
        num_heuristic = int(pop_size * heuristic_ratio)
        
        # Determine how many of each heuristic type
        heuristic_types = []
        if use_savings:
            heuristic_types.append('savings')
        if use_sweep:
            heuristic_types.append('sweep')
        heuristic_types.append('nearest_neighbor')  # Always include greedy
        
        # Add one pure heuristic solution of each type first
        for h_type in heuristic_types:
            if len(population) >= pop_size:
                break
            
            if h_type == 'savings':
                population.append(self._create_savings_chromosome())
            elif h_type == 'sweep':
                population.append(self._create_sweep_chromosome())
            else:
                population.append(self._create_greedy_chromosome())
        
        # Add heuristic solutions with some randomization
        for _ in range(num_heuristic - len(heuristic_types)):
            if len(population) >= pop_size:
                break
                
            # Randomly choose heuristic type
            h_type = random.choice(heuristic_types)
            
            if h_type == 'savings':
                chromosome = self._create_savings_chromosome()
            elif h_type == 'sweep':
                chromosome = self._create_sweep_chromosome()
            else:
                chromosome = self._create_greedy_chromosome()
            
            # Add some randomness by swapping a few customers
            num_swaps = random.randint(1, 5)
            for _ in range(num_swaps):
                i, j = random.sample(range(len(chromosome)), 2)
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
            population.append(chromosome)
        
        # Fill rest with random solutions
        while len(population) < pop_size:
            population.append(self._create_random_chromosome())
        
        return population
    
    def tournament_selection(self, population: List[List[int]], fitnesses: List[float], 
                           tournament_size: int = 3) -> List[int]:
        """
        Select a parent using tournament selection.
        
        Args:
            population: Current population
            fitnesses: Fitness values for each individual
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected parent chromosome
        """
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
        return population[winner_idx][:]
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Perform Order Crossover (OX) to create two offspring.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of two offspring chromosomes
        """
        size = len(parent1)
        
        # Select two random crossover points
        cx_point1, cx_point2 = sorted(random.sample(range(size), 2))
        
        # Create offspring
        def create_offspring(p1, p2):
            offspring = [None] * size
            # Copy segment from parent1
            offspring[cx_point1:cx_point2] = p1[cx_point1:cx_point2]
            
            # Fill remaining positions from parent2
            p2_idx = cx_point2
            offspring_idx = cx_point2
            
            while None in offspring:
                if p2[p2_idx % size] not in offspring:
                    offspring[offspring_idx % size] = p2[p2_idx % size]
                    offspring_idx += 1
                p2_idx += 1
            
            return offspring
        
        offspring1 = create_offspring(parent1, parent2)
        offspring2 = create_offspring(parent2, parent1)
        
        return offspring1, offspring2
    
    def partially_mapped_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Perform Partially Mapped Crossover (PMX) to create two offspring.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of two offspring chromosomes
        """
        size = len(parent1)
        
        # Select two random crossover points
        cx_point1, cx_point2 = sorted(random.sample(range(size), 2))
        
        def create_offspring(p1, p2):
            offspring = [None] * size
            # Copy segment from parent1
            offspring[cx_point1:cx_point2] = p1[cx_point1:cx_point2]
            
            # Create mapping
            for i in range(cx_point1, cx_point2):
                if p2[i] not in offspring:
                    # Find position for p2[i]
                    pos = i
                    while cx_point1 <= pos < cx_point2:
                        pos = p2.index(p1[pos])
                    offspring[pos] = p2[i]
            
            # Fill remaining positions
            for i in range(size):
                if offspring[i] is None:
                    offspring[i] = p2[i]
            
            return offspring
        
        offspring1 = create_offspring(parent1, parent2)
        offspring2 = create_offspring(parent2, parent1)
        
        return offspring1, offspring2
    
    def swap_mutation(self, chromosome: List[int]) -> List[int]:
        """
        Perform swap mutation: swap two random positions.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome[:]
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def inversion_mutation(self, chromosome: List[int]) -> List[int]:
        """
        Perform inversion mutation: reverse a random segment.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome[:]
        i, j = sorted(random.sample(range(len(mutated)), 2))
        mutated[i:j+1] = reversed(mutated[i:j+1])
        return mutated
    
    def scramble_mutation(self, chromosome: List[int]) -> List[int]:
        """
        Perform scramble mutation: shuffle a random segment.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome[:]
        i, j = sorted(random.sample(range(len(mutated)), 2))
        segment = mutated[i:j+1]
        random.shuffle(segment)
        mutated[i:j+1] = segment
        return mutated
    
    def apply_mutation(self, chromosome: List[int], mutation_rate: float) -> List[int]:
        """
        Apply mutation with given probability.
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Possibly mutated chromosome
        """
        if random.random() < mutation_rate:
            # Randomly choose mutation type
            mutation_type = random.choice(['swap', 'inversion', 'scramble'])
            
            if mutation_type == 'swap':
                return self.swap_mutation(chromosome)
            elif mutation_type == 'inversion':
                return self.inversion_mutation(chromosome)
            else:
                return self.scramble_mutation(chromosome)
        
        return chromosome[:]
    
    def local_search_2opt(self, chromosome: List[int], max_improvements: int = 10) -> List[int]:
        """
        Apply 2-opt local search to a chromosome.
        
        Args:
            chromosome: Chromosome to improve
            max_improvements: Maximum number of improvements
            
        Returns:
            Improved chromosome
        """
        improved = chromosome[:]
        improvements = 0
        
        for _ in range(max_improvements):
            best_cost = self._evaluate_fitness(improved)
            found_improvement = False
            
            # Sample positions for large instances
            size = len(improved)
            step = max(1, size // 50)
            
            for i in range(0, size - 1, step):
                for j in range(i + 2, size, step):
                    # Reverse segment
                    new_chromosome = improved[:i] + improved[i:j][::-1] + improved[j:]
                    new_cost = self._evaluate_fitness(new_chromosome)
                    
                    if new_cost < best_cost:
                        improved = new_chromosome
                        best_cost = new_cost
                        improvements += 1
                        found_improvement = True
                        break
                
                if found_improvement:
                    break
            
            if not found_improvement:
                break
        
        return improved
    
    def solve(self, pop_size: int = 100, num_generations: int = 100, 
              crossover_rate: float = 0.8, mutation_rate: float = 0.2,
              elitism_count: int = 2, local_search_freq: int = 10,
              heuristic_ratio: float = 0.3, use_savings: bool = True, 
              use_sweep: bool = True, seed: Optional[int] = None) -> Tuple[List[List[int]], float]:
        """
        Solve VRP using Genetic Algorithm.
        
        Args:
            pop_size: Population size
            num_generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_count: Number of best individuals to preserve
            local_search_freq: Apply local search every N generations
            heuristic_ratio: Ratio of heuristic-based initial solutions (vs random)
            use_savings: Use Clarke-Wright Savings for initialization
            use_sweep: Use Sweep algorithm for initialization
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (best routes, best cost)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print(f"Starting Genetic Algorithm")
        print(f"Population size: {pop_size}")
        print(f"Generations: {num_generations}")
        print(f"Crossover rate: {crossover_rate}")
        print(f"Mutation rate: {mutation_rate}")
        print(f"Elitism: {elitism_count}")
        print(f"Heuristic initialization ratio: {heuristic_ratio:.1%}")
        if use_savings:
            print(f"  - Using Clarke-Wright Savings heuristic")
        if use_sweep:
            print(f"  - Using Sweep heuristic")
        print(f"  - Using Nearest Neighbor heuristic")
        print(f"Number of customers: {self.num_locations - 1}")
        print(f"Vehicle capacity: {self.capacity}")
        print("-" * 60)
        
        # Initialize population with heuristics
        print("Initializing population with constructive heuristics...")
        population = self.initialize_population(pop_size, heuristic_ratio, use_savings, use_sweep)
        
        # Evaluate initial population to show improvement
        initial_fitnesses = [self._evaluate_fitness(chrom) for chrom in population]
        initial_best = min(initial_fitnesses)
        initial_avg = sum(initial_fitnesses) / len(initial_fitnesses)
        print(f"Initial population - Best: {initial_best:.2f}, Avg: {initial_avg:.2f}")
        print("-" * 60)
        
        for generation in range(num_generations):
            # Evaluate fitness
            fitnesses = [self._evaluate_fitness(chrom) for chrom in population]
            
            # Track statistics
            best_fitness = min(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            self.generation_costs.append(best_fitness)
            self.avg_costs.append(avg_fitness)
            
            # Update best solution
            if best_fitness < self.best_cost:
                best_idx = fitnesses.index(best_fitness)
                self.best_cost = best_fitness
                self.best_solution = self._convert_to_routes(population[best_idx])
                print(f"Generation {generation + 1}: New best cost = {best_fitness:.2f}")
            
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}/{num_generations}: Best = {best_fitness:.2f}, Avg = {avg_fitness:.2f}")
            
            # Create new population
            new_population = []
            
            # Elitism: preserve best individuals
            sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
            for i in range(elitism_count):
                new_population.append(population[sorted_indices[i]][:])
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Crossover
                if random.random() < crossover_rate:
                    offspring1, offspring2 = self.order_crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1[:], parent2[:]
                
                # Mutation
                offspring1 = self.apply_mutation(offspring1, mutation_rate)
                offspring2 = self.apply_mutation(offspring2, mutation_rate)
                
                # Add to new population
                new_population.extend([offspring1, offspring2])
            
            # Trim to population size
            population = new_population[:pop_size]
            
            # Apply local search to best individuals periodically
            if (generation + 1) % local_search_freq == 0:
                fitnesses = [self._evaluate_fitness(chrom) for chrom in population]
                best_idx = fitnesses.index(min(fitnesses))
                population[best_idx] = self.local_search_2opt(population[best_idx])
        
        print("-" * 60)
        print(f"Genetic Algorithm completed!")
        print(f"Best cost found: {self.best_cost:.2f}")
        print(f"Number of routes: {len(self.best_solution)}")
        
        return self.best_solution, self.best_cost
    
    def print_solution(self):
        """Print the best solution in a readable format."""
        if self.best_solution is None:
            print("No solution available. Run solve() first.")
            return
        
        print("\n" + "=" * 60)
        print("GENETIC ALGORITHM VRP SOLUTION")
        print("=" * 60)
        print(f"Total cost: {self.best_cost:.2f}")
        print(f"Number of routes: {len(self.best_solution)}")
        print("\nRoutes:")
        
        for idx, route in enumerate(self.best_solution, 1):
            route_demand = sum(self.demands[c] for c in route)
            route_cost = self._calculate_route_cost(route)
            print(f"  Route {idx}: {[self.depot_idx] + route + [self.depot_idx]}")
            print(f"    Demand: {route_demand}/{self.capacity}, Cost: {route_cost:.2f}")
        
        print("=" * 60 + "\n")
    
    def plot_solution(self, output_path: Optional[str] = None):
        """
        Visualize the best solution.
        
        Args:
            output_path: Path to save the plot (optional)
        """
        if self.best_solution is None:
            print("No solution available. Run solve() first.")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Plot depot
        depot_coord = self.coords[self.depot_idx]
        plt.scatter(depot_coord[0], depot_coord[1], c='red', s=200, marker='s', 
                   label='Depot', zorder=3)
        
        # Plot routes with different colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.best_solution)))
        
        for idx, (route, color) in enumerate(zip(self.best_solution, colors), 1):
            # Plot route
            route_coords = [depot_coord] + [self.coords[c] for c in route] + [depot_coord]
            route_coords = np.array(route_coords)
            
            plt.plot(route_coords[:, 0], route_coords[:, 1], 'o-', 
                    color=color, linewidth=2, markersize=6, label=f'Route {idx}')
        
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.title(f'Genetic Algorithm VRP Solution (Cost: {self.best_cost:.2f})', 
                 fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        
        plt.show()
    
    def plot_convergence(self, output_path: Optional[str] = None):
        """
        Plot the convergence of the Genetic Algorithm.
        
        Args:
            output_path: Path to save the plot (optional)
        """
        if not self.generation_costs:
            print("No convergence data available. Run solve() first.")
            return
        
        plt.figure(figsize=(10, 6))
        
        generations = range(1, len(self.generation_costs) + 1)
        plt.plot(generations, self.generation_costs, 'b-', linewidth=2, label='Best Cost')
        plt.plot(generations, self.avg_costs, 'g--', alpha=0.6, label='Average Cost')
        
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Solution Cost', fontsize=12)
        plt.title('Genetic Algorithm Convergence', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {output_path}")
        
        plt.show()
    
    def plot_diversity(self, output_path: Optional[str] = None):
        """
        Plot population diversity over generations.
        
        Args:
            output_path: Path to save the plot (optional)
        """
        if not self.avg_costs or not self.generation_costs:
            print("No diversity data available. Run solve() first.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Calculate diversity as difference between average and best
        diversity = [avg - best for avg, best in zip(self.avg_costs, self.generation_costs)]
        generations = range(1, len(diversity) + 1)
        
        plt.plot(generations, diversity, 'purple', linewidth=2)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Population Diversity (Avg - Best)', fontsize=12)
        plt.title('Population Diversity Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Diversity plot saved to {output_path}")
        
        plt.show()


def main():
    """
    Main function to solve VRP using Genetic Algorithm.
    Runs on the full 1000+ node instance.
    """
    import time
    import vrplib
    from pathlib import Path
    
    instance_path = "/data/X-n101-k25.vrp"
    
    print("\n" + "=" * 80)
    print("GENETIC ALGORITHM METAHEURISTIC - VRP with 1 Truck (Multiple Trips)")
    print("=" * 80)
    print(f"Instance: {instance_path}")
    print(f"Configuration: Population=100, Generations=100, Capacity=100")
    print("=" * 80 + "\n")
    
    # Check if file exists
    if not Path(instance_path).exists():
        print(f"ERROR: Instance file not found: {instance_path}")
        print("Using small example instead...")
        
        # Run GA with example data
        coords = np.array([
            [50, 50],  # Depot
            [23, 67], [78, 34], [12, 89], [45, 23], [67, 56],
            [34, 78], [89, 12], [56, 45], [90, 67], [34, 23]
        ])
        demands = np.array([0, 5, 8, 3, 7, 4, 6, 9, 2, 5, 7])
        capacity = 20
    else:
        # Parse full instance
        print("Loading instance...")
        data = vrplib.read_instance(instance_path)
        coords = np.array(data['node_coord'])
        dimension = data['dimension']
        capacity = data.get('capacity', 100)
        
        # Get demands if available, otherwise default to 1 for all customers
        if 'demand' in data:
            demands = np.array(data['demand'])
        else:
            demands = np.ones(dimension, dtype=int)
            demands[0] = 0  # Depot has 0 demand
        
        print(f"Loaded: {dimension} nodes, capacity {capacity}")
        print(f"Total demand: {demands.sum()}")
        print()
    
    start_time = time.time()
    
    # Create and run GA solver
    solver = GeneticAlgorithmSolver(coords, demands, capacity, depot_idx=0)
    solution, cost = solver.solve(
        pop_size=100,
        num_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elitism_count=2,
        local_search_freq=10,
        heuristic_ratio=0.3,  # 30% of initial population from heuristics
        use_savings=True,     # Use Clarke-Wright Savings
        use_sweep=True,       # Use Sweep algorithm
        seed=42
    )
    
    # Display results
    solver.print_solution()
    
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    
    # Try to create visualizations
    try:
        print("\nGenerating visualizations...")
        solver.plot_solution(output_path="/data/ga_solution.png")
        solver.plot_convergence(output_path="/data/ga_convergence.png")
        solver.plot_diversity(output_path="/data/ga_diversity.png")
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    print("=" * 80 + "\n")
    
    return solver


if __name__ == "__main__":
    main()
