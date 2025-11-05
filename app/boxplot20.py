from ga import GeneticAlgorithmSolver
import numpy as np
import matplotlib.pyplot as plt


class Boxplot20:
    
    def __init__(self):
        self.nmber_of_runs = 20
        self.results = []
        self.labels = []
        self.colors = ['blue', 'green', 'red', 'purple', 'orange']
        self.title = "GA VRP Solution Costs over 20 Runs"
        self.ylabel = "Solution Cost"
        self.xlabel = "GA Configurations"
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def add_results(self, method_name: str, costs: list):
        self.results.append(costs)
        self.labels.append(method_name)

    def plot(self, save_path: str = None):
        """Plot boxplot and optionally save to file."""
        self.ax.boxplot(self.results, tick_labels=self.labels)
        self.ax.set_title(self.title)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_xlabel(self.xlabel)
        self.ax.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Boxplot saved to {save_path}")
        
        plt.show()
    
    def plot_bar_chart(self, save_path: str = None):
        """Plot bar chart showing individual run costs."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # For each configuration, plot bars for all runs
        bar_width = 0.8 / len(self.results)
        x_positions = np.arange(self.nmber_of_runs)
        
        for idx, (label, costs) in enumerate(zip(self.labels, self.results)):
            offset = (idx - len(self.results)/2 + 0.5) * bar_width
            bars = ax.bar(x_positions + offset, costs, bar_width, 
                          label=label, alpha=0.7, 
                          color=self.colors[idx % len(self.colors)])
            
            # Add mean line
            mean_cost = np.mean(costs)
            ax.axhline(y=mean_cost, color=self.colors[idx % len(self.colors)], 
                      linestyle='--', linewidth=2, alpha=0.5,
                      label=f'{label} Mean: {mean_cost:.2f}')
        
        ax.set_xlabel('Run Number', fontsize=12)
        ax.set_ylabel('Solution Cost', fontsize=12)
        ax.set_title('GA Solution Costs - Individual Runs', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'{i+1}' for i in range(self.nmber_of_runs)])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Bar chart saved to {save_path}")
        
        plt.show()
    
    def save_statistics(self, save_path: str = None):
        """Print and optionally save statistics for each configuration."""
        print("\n" + "=" * 80)
        print("STATISTICAL SUMMARY")
        print("=" * 80)
        
        stats_lines = []
        for label, costs in zip(self.labels, self.results):
            print(f"\n{label}:")
            print(f"  Mean:   {np.mean(costs):.2f}")
            print(f"  Median: {np.median(costs):.2f}")
            print(f"  Std:    {np.std(costs):.2f}")
            print(f"  Min:    {np.min(costs):.2f}")
            print(f"  Max:    {np.max(costs):.2f}")
            
            stats_lines.append(f"{label}:")
            stats_lines.append(f"  Mean:   {np.mean(costs):.2f}")
            stats_lines.append(f"  Median: {np.median(costs):.2f}")
            stats_lines.append(f"  Std:    {np.std(costs):.2f}")
            stats_lines.append(f"  Min:    {np.min(costs):.2f}")
            stats_lines.append(f"  Max:    {np.max(costs):.2f}")
            stats_lines.append("")
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write('\n'.join(stats_lines))
            print(f"\nStatistics saved to {save_path}")


def main():
    """Run GA 20 times without seed and generate boxplot."""
    import vrplib
    
    print("\n" + "=" * 80)
    print("GA BOXPLOT ANALYSIS - 20 Runs Without Seed")
    print("=" * 80)
    
    # Configure instance file
    instance_file = "/data/200instance.vrp"
    
    print(f"Instance: {instance_file}")
    print(f"Number of runs: 20")
    print(f"Seed: None (random runs)")
    print("=" * 80 + "\n")
    
    # Load instance
    instance = vrplib.read_instance(instance_file)
    coords = instance['node_coord']
    capacity = instance['capacity']
    demands = instance.get('demand', [0] + [1] * (instance['dimension'] - 1))
    
    # Create boxplot analyzer
    boxplot = Boxplot20()
    
    # Run GA 20 times without seed
    print("Running GA 20 times (this may take a while)...\n")
    costs = []
    
    for run in range(1, 21):
        print(f"Run {run}/20...", end=" ", flush=True)
        
        solver = GeneticAlgorithmSolver(coords, demands, capacity)
        solution, cost = solver.solve(
            pop_size=100,
            num_generations=300,
            crossover_rate=0.6,
            mutation_rate=0.05,
            elitism_count=10,
            local_search_freq=4,
            seed=None  # No seed for random runs
        )
        
        costs.append(cost)
        print(f"Cost: {cost:.2f}")
    
    # Add results to boxplot
    boxplot.add_results("GA (20 runs)", costs)
    
    # Display statistics
    boxplot.save_statistics()
    
    # Generate and save boxplot
    print("\n" + "=" * 80)
    print("GENERATING BOXPLOT")
    print("=" * 80)
    
    boxplot.plot(save_path="/data/ga_boxplot_20runs.png")
    
    # Generate and save bar chart
    print("\n" + "=" * 80)
    print("GENERATING BAR CHART")
    print("=" * 80)
    
    boxplot.plot_bar_chart(save_path="/data/ga_barchart_20runs.png")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
        