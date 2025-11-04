"""
VRP Solver using PyVRP library.
This module provides functionality to solve Vehicle Routing Problems using the PyVRP library.
"""

import numpy as np
from pathlib import Path
from pyvrp import Model, read
from pyvrp.stop import MaxRuntime


class VRPSolver:
    """Solver for Vehicle Routing Problems using PyVRP."""
    
    def __init__(self, instance_path: str):
        """
        Initialize the VRP solver with an instance file.
        
        Args:
            instance_path: Path to the VRP instance file
        """
        self.instance_path = Path(instance_path)
        self.instance_data = None
        self.model = None
        self.result = None
        
    def load_instance(self):
        """Load the VRP instance from file using PyVRP's read function."""
        print(f"Loading instance from {self.instance_path}")
        # Use PyVRP's native read function which properly handles VRPLIB format
        self.instance_data = read(str(self.instance_path), round_func='round')
        return self.instance_data
    
    def build_model(self):
        """Build the PyVRP model from the loaded instance."""
        if self.instance_data is None:
            self.load_instance()
        
        # Use PyVRP's from_data() method with the ProblemData object
        self.model = Model.from_data(self.instance_data)
        
        print(f"Model built from PyVRP data")
        print(f"  Locations: {self.instance_data.num_locations}")
        print(f"  Vehicles: {self.instance_data.num_vehicle_types}")
        
        return self.model
    
    def solve(self, time_limit: int = 120, seed: int = 42):
        """
        Solve the VRP instance.
        
        Args:
            time_limit: Maximum runtime in seconds (default: 120)
            seed: Random seed for reproducibility
            
        Returns:
            Result object from PyVRP
        """
        if self.model is None:
            self.build_model()
        
        print(f"Solving VRP with time limit: {time_limit}s")
        
        # Solve the problem
        self.result = self.model.solve(
            stop=MaxRuntime(time_limit),
            seed=seed
        )
        
        # Check if solution is feasible
        print(f"\nSolution statistics:")
        print(f"  Cost: {self.result.cost()}")
        print(f"  Feasible: {self.result.is_feasible()}")
        
        # Debug: print result attributes
        print(f"\nResult object attributes: {[attr for attr in dir(self.result) if not attr.startswith('_')]}")
        
        if self.result.is_feasible():
            # Try to get basic stats
            try:
                best = self.result.best
                print(f"  Number of routes in solution: {best.num_routes()}")
                print(f"  Routes: {list(best.routes())}")
            except Exception as e:
                print(f"  Could not get route info: {e}")
        else:
            print(f"WARNING: No feasible solution found within time limit")
        
        return self.result
    
    def get_routes(self):
        """
        Get the routes from the solution.
        
        Returns:
            List of routes, where each route is a list of client indices
        """
        if self.result is None:
            raise ValueError("No solution available. Run solve() first.")
        
        # Get routes from the result object using best() method
        routes = []
        try:
            best_solution = self.result.best
            for route in best_solution.routes():
                route_clients = [visit for visit in route]
                routes.append(route_clients)
        except Exception as e:
            print(f"Error extracting routes: {e}")
            # Try alternative approach
            try:
                import traceback
                traceback.print_exc()
            except:
                pass
        return routes
    
    def print_solution(self):
        """Print the solution in a readable format."""
        if self.result is None:
            print("No solution available.")
            return
        
        print("\n" + "=" * 60)
        print("VRP SOLUTION")
        print("=" * 60)
        print(f"Total cost: {self.result.cost()}")
        print(f"Is feasible: {self.result.is_feasible()}")
        
        if self.result.is_feasible():
            try:
                routes = self.get_routes()
                print(f"Number of routes: {len(routes)}")
                print("\nRoutes:")
                for idx, route in enumerate(routes, 1):
                    print(f"  Route {idx}: {route[:10]}..." if len(route) > 10 else f"  Route {idx}: {route}")
            except Exception as e:
                print(f"Could not display routes: {e}")
        else:
            print("No feasible solution found.")
        
        print("=" * 60 + "\n")
    
    def save_solution(self, output_path: str):
        """
        Save the solution to a file in VRPLIB .sol format.
        
        Args:
            output_path: Path to save the solution
        """
        if self.result is None:
            raise ValueError("No solution available. Run solve() first.")
        
        with open(output_path, 'w') as f:
            # VRPLIB .sol format
            if self.result.is_feasible():
                try:
                    routes = self.get_routes()
                    # Write cost (rounded to integer for VRPLIB format)
                    f.write(f"{int(self.result.cost())}\n")
                    
                    # Write each route
                    for route in routes:
                        # VRPLIB format: Route #: node1 node2 node3 ...
                        # Client indices in PyVRP start from 1 (depot is 0)
                        route_str = ' '.join(map(str, route))
                        f.write(f"Route #{len([r for r in routes[:routes.index(route)+1]])}: {route_str}\n")
                except Exception as e:
                    f.write(f"Error: Could not extract routes: {e}\n")
            else:
                f.write("No feasible solution found.\n")
        
        print(f"Solution saved to {output_path} (VRPLIB .sol format)")


def main():
    """
    Main function to solve VRP using PyVRP.
    
    Note: With 1000 customers each having demand=1 and capacity=1000,
    all customers can fit in a single route (essentially a TSP).
    """
    import time
    
    # Path to instance file
    instance_file = "/data/instance.vrp"
    
    print("\n" + "=" * 80)
    print("PYVRP SOLVER - VRP with 1 Truck (Multiple Trips)")
    print("=" * 80)
    print(f"Instance: {instance_file}")
    print(f"Time limit: 120 seconds")
    print(f"Configuration: 1 truck, capacity 1000 (all customers in 1 route)")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    # Create solver
    solver = VRPSolver(instance_file)
    
    # Load and solve
    solver.load_instance()
    solver.build_model()
    result = solver.solve(time_limit=120)
    
    # Display results
    solver.print_solution()
    
    # Save solution in VRPLIB .sol format
    solver.save_solution("/data/pyvrp_solution.sol")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    print("=" * 80 + "\n")
    
    return solver


if __name__ == "__main__":
    main()
