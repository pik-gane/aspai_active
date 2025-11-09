"""
Example: Active learning on high-dimensional probability simplex.

This example demonstrates the gradient descent optimization for candidate points
in high dimensions (d=20), comparing performance with and without optimization.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from aspai_active import ActiveLearner, sample_simplex


class HighDimFunction:
    """
    A function on high-dimensional simplex that depends on multiple coordinates.
    """
    
    def __init__(self, d, n_regions=3, seed=None):
        """
        Initialize a high-dimensional function with multiple decision regions.
        
        Args:
            d: Dimension of the simplex
            n_regions: Number of decision regions
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.d = d
        self.n_regions = n_regions
        
        # Create random decision regions
        self.region_centers = []
        self.region_radii = []
        for i in range(n_regions):
            center = np.random.dirichlet(np.ones(d) * 2)  # Concentrated around center
            radius = np.random.uniform(0.1, 0.3)
            self.region_centers.append(center)
            self.region_radii.append(radius)
    
    def __call__(self, x):
        """
        Evaluate function at x.
        
        Returns 1 if x is in any of the decision regions, 0 otherwise.
        
        Args:
            x: Point on the simplex, shape (d,)
            
        Returns:
            Probability in [0, 1]
        """
        # Check distance to each region center
        for center, radius in zip(self.region_centers, self.region_radii):
            dist = np.linalg.norm(x - center)
            if dist < radius:
                return 0.9  # High probability inside region
        
        return 0.1  # Low probability outside regions


def create_oracle(true_function):
    """
    Create a probabilistic oracle from a true function.
    
    Args:
        true_function: Function f: S_d -> [0, 1]
        
    Returns:
        Oracle function that returns 1 with probability f(x)
    """
    def oracle(x):
        prob = true_function(x)
        return int(np.random.random() < prob)
    
    return oracle


def run_experiment(d, n_iterations, optimize_candidates, seed=None):
    """
    Run an active learning experiment.
    
    Args:
        d: Dimension of the simplex
        n_iterations: Number of active learning iterations
        optimize_candidates: Whether to use gradient descent optimization
        seed: Random seed
        
    Returns:
        Dictionary with results
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Create true function and oracle
    true_function = HighDimFunction(d=d, n_regions=3, seed=seed)
    oracle = create_oracle(true_function)
    
    # Initialize active learner
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    learner = ActiveLearner(
        d=d,
        oracle=oracle,
        n_models=5,
        hidden_dims=[128, 64],
        device=device,
        seed=seed
    )
    
    # Run active learning
    results = learner.run(
        n_iterations=n_iterations,
        n_candidates=500,
        n_initial=10,
        n_oracle_queries=5,
        retrain_epochs=50,
        optimize_candidates=optimize_candidates,
        gd_steps=20,
        gd_lr=0.05,
        gd_top_k_fraction=0.2,
        verbose=False
    )
    
    # Evaluate on test set
    test_points = sample_simplex(1000, d, device=device)
    predictions = learner.predict(test_points)
    
    # Get true labels
    true_labels = np.array([true_function(x.cpu().numpy()) > 0.5 for x in test_points])
    pred_labels = predictions > 0.5
    
    accuracy = np.mean(true_labels == pred_labels)
    
    return {
        'learner': learner,
        'results': results,
        'accuracy': accuracy,
        'true_function': true_function
    }


def compare_methods(d=20, n_iterations=30, n_trials=3):
    """
    Compare active learning with and without gradient descent optimization.
    
    Args:
        d: Dimension of the simplex
        n_iterations: Number of active learning iterations per trial
        n_trials: Number of trials to average over
    """
    print("=" * 70)
    print(f"Comparing Active Learning Methods on {d}D Simplex")
    print("=" * 70)
    print(f"Settings:")
    print(f"  Dimension: d = {d}")
    print(f"  Active learning iterations: {n_iterations}")
    print(f"  Trials: {n_trials}")
    print()
    
    # Results storage
    accuracies_without_opt = []
    accuracies_with_opt = []
    
    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}:")
        seed = 42 + trial
        
        # Without optimization
        print("  Running without gradient optimization...")
        result_without = run_experiment(
            d=d, 
            n_iterations=n_iterations, 
            optimize_candidates=False,
            seed=seed
        )
        accuracies_without_opt.append(result_without['accuracy'])
        print(f"    Accuracy: {result_without['accuracy']:.2%}")
        
        # With optimization
        print("  Running with gradient optimization...")
        result_with = run_experiment(
            d=d, 
            n_iterations=n_iterations, 
            optimize_candidates=True,
            seed=seed
        )
        accuracies_with_opt.append(result_with['accuracy'])
        print(f"    Accuracy: {result_with['accuracy']:.2%}")
        print()
    
    # Summary statistics
    print("=" * 70)
    print("Results Summary:")
    print("=" * 70)
    print(f"Without gradient optimization:")
    print(f"  Mean accuracy: {np.mean(accuracies_without_opt):.2%} ± {np.std(accuracies_without_opt):.2%}")
    print()
    print(f"With gradient optimization:")
    print(f"  Mean accuracy: {np.mean(accuracies_with_opt):.2%} ± {np.std(accuracies_with_opt):.2%}")
    print()
    
    improvement = np.mean(accuracies_with_opt) - np.mean(accuracies_without_opt)
    rel_improvement = improvement / np.mean(accuracies_without_opt) * 100
    print(f"Improvement: {improvement:.2%} ({rel_improvement:.1f}% relative)")
    print("=" * 70)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot
    ax1.bar([0, 1], 
            [np.mean(accuracies_without_opt), np.mean(accuracies_with_opt)],
            yerr=[np.std(accuracies_without_opt), np.std(accuracies_with_opt)],
            color=['#ff7f0e', '#2ca02c'],
            alpha=0.7,
            capsize=10)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Without\nOptimization', 'With\nOptimization'])
    ax1.set_ylabel('Classification Accuracy', fontsize=12)
    ax1.set_title(f'Active Learning Performance (d={d})', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Individual trial points
    for i in range(n_trials):
        ax2.plot([0, 1], 
                [accuracies_without_opt[i], accuracies_with_opt[i]], 
                'o-', alpha=0.6, linewidth=2, markersize=8)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Without\nOptimization', 'With\nOptimization'])
    ax2.set_ylabel('Classification Accuracy', fontsize=12)
    ax2.set_title('Individual Trial Results', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('highdim_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: highdim_comparison.png")
    
    return {
        'accuracies_without_opt': accuracies_without_opt,
        'accuracies_with_opt': accuracies_with_opt
    }


def main():
    """
    Run the high-dimensional example.
    """
    results = compare_methods(d=20, n_iterations=30, n_trials=3)
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()
