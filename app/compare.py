"""
Compare solutions from Genetic Algorithm and PyVRP solver.
This module verifies and compares the solutions from both algorithms.
"""

import vrplib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ga import GeneticAlgorithmSolver
from vrp import VRPSolver


def verify_solution(instance_file: str, solution_file: str):
    """
    Verify a solution using vrplib.
    
    Args:
        instance_file: Path to the VRP instance file
        solution_file: Path to the solution file (.sol format)
        
    Returns:
        dict: Verification results with cost, feasibility, etc.
    """
    print(f"\nVerifying solution: {solution_file}")
    print("-" * 60)
    
    # Read instance and solution
    instance = vrplib.read_instance(instance_file)
    solution = vrplib.read_solution(solution_file)
    
    print(f"Solution loaded from file:")
    print(f"  Routes: {len(solution['routes'])}")
    
    # Get instance parameters
    coords = instance['node_coord']
    capacity = instance['capacity']
    demands = instance.get('demand', [0] + [1] * (instance['dimension'] - 1))
    
    # Manually compute cost and check feasibility
    total_cost = 0
    is_feasible = True
    
    for idx, route in enumerate(solution['routes'], 1):
        # Calculate route distance
        route_cost = 0
        # Start from depot (node 0)
        prev_node = 0
        for node in route:
            dist = np.linalg.norm(coords[prev_node] - coords[node])
            route_cost += dist
            prev_node = node
        # Return to depot
        route_cost += np.linalg.norm(coords[prev_node] - coords[0])
        
        # Calculate route demand
        route_demand = sum(demands[node] for node in route)
        
        # Check feasibility
        if route_demand > capacity:
            is_feasible = False
            print(f"  Route {idx}: {len(route)} customers, demand={route_demand}/{capacity} - INFEASIBLE!")
        else:
            print(f"  Route {idx}: {len(route)} customers, demand={route_demand}/{capacity}, cost={route_cost:.2f}")
        
        total_cost += route_cost
    
    print(f"\nComputed total cost: {total_cost:.2f}")
    print(f"Feasible: {is_feasible}")
    
    return {
        'cost': total_cost,
        'feasible': is_feasible,
        'num_routes': len(solution['routes']),
        'solution': solution
    }


def save_ga_solution_as_sol(ga_solver, output_path: str):
    """
    Save GA solution in VRPLIB .sol format.
    
    Args:
        ga_solver: GeneticAlgorithmSolver instance with solution
        output_path: Path to save the .sol file
    """
    with open(output_path, 'w') as f:
        # Write cost
        f.write(f"{int(ga_solver.best_cost)}\n")
        
        # Write routes
        routes = ga_solver.best_solution  # Use best_solution directly
        for idx, route in enumerate(routes, 1):
            # VRPLIB format: Route #X: node1 node2 node3 ...
            route_str = ' '.join(map(str, route))
            f.write(f"Route #{idx}: {route_str}\n")
    
    print(f"GA solution saved to {output_path}")


def visualize_comparison(ga_results: dict, pyvrp_results: dict, save_path: str = None):
    """
    Create visualization comparing GA and PyVRP results.
    
    Args:
        ga_results: GA verification results
        pyvrp_results: PyVRP verification results
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cost comparison
    algorithms = ['Genetic Algorithm', 'PyVRP']
    costs = [ga_results['cost'], pyvrp_results['cost']]
    colors = ['#2ecc71', '#3498db']
    
    axes[0].bar(algorithms, costs, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Total Cost', fontsize=12)
    axes[0].set_title('Solution Cost Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (alg, cost) in enumerate(zip(algorithms, costs)):
        axes[0].text(i, cost, f'{int(cost)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Routes comparison
    num_routes = [ga_results['num_routes'], pyvrp_results['num_routes']]
    
    axes[1].bar(algorithms, num_routes, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Number of Routes', fontsize=12)
    axes[1].set_title('Number of Routes Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (alg, routes) in enumerate(zip(algorithms, num_routes)):
        axes[1].text(i, routes, f'{routes}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def main():
    """Main comparison function."""
    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON - Genetic Algorithm vs PyVRP")
    print("=" * 80)
    
    # Configure instance file here - change this to test different instances
    # Build paths relative to the repository root so the script works when run
    # from different working directories.
    repo_root = Path(__file__).resolve().parents[1]
    instance_file = str(repo_root / "data" / "X-n101-k25.vrp")
    ga_sol_file = str(repo_root / "data" / "ga_solution.sol")
    pyvrp_sol_file = str(repo_root / "data" / "X-n101-k25.sol")
    
    # Run Genetic Algorithm
    print("\n" + "=" * 80)
    print("RUNNING GENETIC ALGORITHM")
    print("=" * 80)
    
    # Load instance data
    instance = vrplib.read_instance(instance_file)
    coords = instance['node_coord']
    capacity = instance['capacity']
    demands = instance.get('demand', [0] + [1] * (instance['dimension'] - 1))
    
    ga_solver = GeneticAlgorithmSolver(coords, demands, capacity)
    ga_solver.solve(
        pop_size=1400, #320 for 1000 / 1400 for 200 // change as needed -> here for 180s computation time
        num_generations=1000, #200 for 1000 / 1000 for 200 // change as needed -> here for 180s computation time
        crossover_rate=0.8,
        mutation_rate=0.05,
        elitism_count=5,
        local_search_freq=4,
        seed=42
    )
    
    # Save GA solution in .sol format
    save_ga_solution_as_sol(ga_solver, ga_sol_file)
    
    # Verify GA solution
    ga_results = verify_solution(instance_file, ga_sol_file)
    
    print("\n" + "=" * 80)
    print("GA SOLUTION SUMMARY")
    print("=" * 80)
    print(f"Best cost: {ga_solver.best_cost:.2f}")
    print(f"Number of routes: {len(ga_solver.best_solution)}")
    print(f"Verified cost: {ga_results['cost']}")
    print(f"Feasible: {ga_results['feasible']}")
    
    # Run PyVRP solver
    print("\n" + "=" * 80)
    print("RUNNING PYVRP SOLVER")
    print("=" * 80)
    
    # pyvrp_solver = VRPSolver(instance_file)
    # pyvrp_solver.load_instance()
    # pyvrp_solver.build_model()
    # pyvrp_solver.solve(time_limit=120, seed=42)
    # pyvrp_solver.save_solution(pyvrp_sol_file)
    
    # Verify PyVRP solution
    print("\n" + "=" * 80)
    print("VERIFYING PYVRP SOLUTION")
    print("=" * 80)
    # If a PyVRP solution file exists, verify it. Otherwise skip comparison.
    pyvrp_path = Path(pyvrp_sol_file)
    if pyvrp_path.exists():
        pyvrp_results = verify_solution(instance_file, pyvrp_sol_file)
    else:
        print(f"No PyVRP solution file found at: {pyvrp_sol_file}")
        print("Skipping PyVRP verification and comparison.")
        pyvrp_results = None
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\nGenetic Algorithm:")
    print(f"  Cost: {ga_results['cost']:.2f}")
    print(f"  Routes: {ga_results['num_routes']}")
    print(f"  Feasible: {ga_results['feasible']}")
    
    print(f"\nPyVRP:")
    if pyvrp_results:
        print(f"  Cost: {pyvrp_results['cost']:.2f}")
        print(f"  Routes: {pyvrp_results['num_routes']}")
        print(f"  Feasible: {pyvrp_results['feasible']}")
    else:
        print("  No PyVRP solution available; comparison skipped.")
    
    # Calculate difference only if PyVRP results are available
    if pyvrp_results:
        cost_diff = ga_results['cost'] - pyvrp_results['cost']
        cost_diff_pct = (cost_diff / pyvrp_results['cost']) * 100

        print(f"\nDifference:")
        print(f"  Cost difference: {cost_diff:+.2f} ({cost_diff_pct:+.2f}%)")

        if ga_results['cost'] < pyvrp_results['cost']:
            print(f"  ✓ GA found a better solution!")
        elif ga_results['cost'] > pyvrp_results['cost']:
            print(f"  ✓ PyVRP found a better solution!")
        else:
            print(f"  ✓ Both algorithms found the same cost!")
    else:
        print("\nDifference: N/A (no PyVRP solution to compare)")
    
    # Visualize comparison
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON VISUALIZATION")
    print("=" * 80)
    
    # Only visualize if we have both results
    if pyvrp_results:
        visualize_comparison(ga_results, pyvrp_results, save_path=str(repo_root / "data" / "algorithm_comparison.png"))
    else:
        print("Skipping visualization because PyVRP results are not available.")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
