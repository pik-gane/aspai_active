"""
Example application: Active learning on 3D probability simplex.

This example demonstrates the active learning approach with:
- d=3 (for visualization)
- f(x) as an average of k=5 smooth step functions
- Each step function is along a random hyperplane with smoothness via expit
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FFMpegWriter
from scipy.special import expit

from aspai_active import ActiveLearner, sample_simplex, grid_simplex
from aspai_active.utils import barycentric_to_cartesian


class SmoothStepFunction:
    """
    A smooth step function along a hyperplane.
    
    The function transitions from 0 to 1 as we move across a hyperplane
    through a point in the direction of the normal vector.
    """
    
    def __init__(self, d, seed=None):
        """
        Initialize a random smooth step function.
        
        Args:
            d: Dimension of the simplex
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Random point on the simplex
        self.point = np.random.dirichlet(np.ones(d))
        
        # Random normal vector
        self.normal = np.random.randn(d)
        self.normal = self.normal / np.linalg.norm(self.normal)
        
        # Smoothness parameter (controls steepness of sigmoid)
        self.smoothness = np.random.uniform(0.1, 0.2)
        
    def __call__(self, x):
        """
        Evaluate the smooth step function at x.
        
        Args:
            x: Point(s) on the simplex, shape (d,) or (n, d)
            
        Returns:
            Function value(s), shape () or (n,)
        """
        # Distance from hyperplane (signed distance in direction of normal)
        if x.ndim == 1:
            dist = np.dot(x - self.point, self.normal)
        else:
            dist = np.dot(x - self.point, self.normal)
        
        # Apply smooth step (expit = logistic sigmoid)
        return expit(dist / self.smoothness)


class SumOfSteps:
    """
    Function that is the average of multiple smooth step functions.
    """
    
    def __init__(self, d, k=5, seed=None):
        """
        Initialize as average of k smooth step functions.
        
        Args:
            d: Dimension of the simplex
            k: Number of step functions
            seed: Random seed for reproducibility
        """
        self.d = d
        self.k = k
        
        # Create k smooth step functions
        self.steps = []
        for i in range(k):
            step_seed = None if seed is None else seed + i
            self.steps.append(SmoothStepFunction(d, seed=step_seed))
    
    def __call__(self, x):
        """
        Evaluate the average of step functions.
        
        Args:
            x: Point(s) on the simplex, shape (d,) or (n, d)
            
        Returns:
            Function value(s) as average in [0, 1], shape () or (n,)
        """
        # Sum all step functions
        total = sum(step(x) for step in self.steps)
        
        # Average (normalize to [0, 1])
        return total / self.k


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


def create_frame(learner, true_function, X_train_tensor, iteration, fig=None, axes=None):
    """
    Create a single frame for the video showing current state.
    
    Args:
        learner: The ActiveLearner instance
        true_function: The true function being learned
        X_train_tensor: Current training points (torch.Tensor)
        iteration: Current iteration number
        fig: Optional existing figure to reuse
        axes: Optional existing axes to reuse
        
    Returns:
        fig, axes, metrics_dict
    """
    # Create a fine grid for visualization
    n_grid = 50
    grid_points = grid_simplex(n_grid, 3, device=learner.device)
    
    # Get true function values
    true_values = np.array([true_function(x.cpu().numpy()) for x in grid_points])
    
    # Get predicted values
    pred_values = learner.predict(grid_points)
    
    # Convert to 2D coordinates for plotting
    coords_2d = barycentric_to_cartesian(grid_points)
    
    # Get training points in 2D
    train_coords_2d = barycentric_to_cartesian(X_train_tensor)
    
    # Calculate metrics
    true_A = true_values > 0.5
    pred_A = pred_values > 0.5
    
    tp = np.sum(true_A & pred_A)
    tn = np.sum(~true_A & ~pred_A)
    fp = np.sum(~true_A & pred_A)
    fn = np.sum(true_A & ~pred_A)
    accuracy = (tp + tn) / len(true_A)
    
    metrics = {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'accuracy': accuracy
    }
    
    # Create or clear figure
    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        for ax in axes:
            ax.clear()
    
    # Triangle vertices for plotting
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    
    # Plot 1: True function
    ax = axes[0]
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                        c=true_values, cmap='RdYlBu_r', s=20, alpha=0.6,
                        vmin=0, vmax=1)
    ax.plot(*train_coords_2d.T, 'k.', markersize=8, alpha=0.7, label='Query points')
    ax.add_patch(Polygon(triangle, fill=False, edgecolor='black', linewidth=2))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.set_title(f'True Function f(x)\nIteration {iteration}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    cbar = plt.colorbar(scatter, ax=ax, label='f(x)')
    ax.axis('off')
    
    # Plot 2: Estimated function
    ax = axes[1]
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                        c=pred_values, cmap='RdYlBu_r', s=20, alpha=0.6,
                        vmin=0, vmax=1)
    ax.plot(*train_coords_2d.T, 'k.', markersize=8, alpha=0.7, label='Query points')
    ax.add_patch(Polygon(triangle, fill=False, edgecolor='black', linewidth=2))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.set_title(f'Estimated Function\nQueries: {len(X_train_tensor)}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    cbar = plt.colorbar(scatter, ax=ax, label='Predicted f(x)')
    ax.axis('off')
    
    # Plot 3: Classification with metrics
    ax = axes[2]
    
    # Color points by correctness
    colors = np.zeros(len(true_A))
    colors[true_A & pred_A] = 1.0  # True positive (green)
    colors[~true_A & ~pred_A] = 0.5  # True negative (yellow)
    colors[true_A & ~pred_A] = 0.0  # False negative (red)
    colors[~true_A & pred_A] = 0.25  # False positive (orange)
    
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                        c=colors, cmap='RdYlGn', s=20, alpha=0.6,
                        vmin=0, vmax=1)
    ax.plot(*train_coords_2d.T, 'k.', markersize=8, alpha=0.7, label='Query points')
    ax.add_patch(Polygon(triangle, fill=False, edgecolor='black', linewidth=2))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    
    # Add metrics text
    metrics_text = f'TP: {tp}  TN: {tn}\nFP: {fp}  FN: {fn}\nAcc: {accuracy:.2%}'
    ax.text(0.5, -0.08, metrics_text, transform=ax.transAxes, 
            fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title('Classification A = {x: f(x) > 0.5}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Custom colorbar labels
    cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(['FN', 'FP', 'TN', '', 'TP'])
    ax.axis('off')
    
    plt.tight_layout()
    
    return fig, axes, metrics


def visualize_results(learner, true_function, results, save_path='results.png'):
    """
    Visualize the active learning results for d=3.
    
    Creates a plot showing:
    - True function values
    - Estimated function values
    - Query points
    - Acquisition function (from last iteration)
    
    Args:
        learner: The ActiveLearner instance
        true_function: The true function being learned
        results: Dictionary of results from learner.run()
        save_path: Path to save the figure
    """
    # Create a fine grid for visualization
    n_grid = 50
    grid_points = grid_simplex(n_grid, 3, device=learner.device)
    
    # Get true function values
    true_values = np.array([true_function(x.cpu().numpy()) for x in grid_points])
    
    # Get predicted values
    pred_values = learner.predict(grid_points)
    
    # Convert to 2D coordinates for plotting
    coords_2d = barycentric_to_cartesian(grid_points)
    
    # Get training points in 2D
    train_coords_2d = barycentric_to_cartesian(results['X_train'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Triangle vertices for plotting
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    
    # Plot 1: True function
    ax = axes[0]
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                        c=true_values, cmap='RdYlBu_r', s=20, alpha=0.6,
                        vmin=0, vmax=1)
    ax.plot(*train_coords_2d.T, 'k.', markersize=8, alpha=0.7, label='Query points')
    ax.add_patch(Polygon(triangle, fill=False, edgecolor='black', linewidth=2))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.set_title('True Function f(x)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.colorbar(scatter, ax=ax, label='f(x)')
    ax.axis('off')
    
    # Plot 2: Estimated function
    ax = axes[1]
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                        c=pred_values, cmap='RdYlBu_r', s=20, alpha=0.6,
                        vmin=0, vmax=1)
    ax.plot(*train_coords_2d.T, 'k.', markersize=8, alpha=0.7, label='Query points')
    ax.add_patch(Polygon(triangle, fill=False, edgecolor='black', linewidth=2))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.set_title('Estimated Function', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.colorbar(scatter, ax=ax, label='Predicted f(x)')
    ax.axis('off')
    
    # Plot 3: True A (f(x) > 0.5) and Estimated A
    ax = axes[2]
    true_A = true_values > 0.5
    pred_A = pred_values > 0.5
    
    # Color points by correctness
    colors = np.zeros(len(true_A))
    colors[true_A & pred_A] = 1.0  # True positive (green)
    colors[~true_A & ~pred_A] = 0.5  # True negative (yellow)
    colors[true_A & ~pred_A] = 0.0  # False negative (red)
    colors[~true_A & pred_A] = 0.25  # False positive (orange)
    
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                        c=colors, cmap='RdYlGn', s=20, alpha=0.6,
                        vmin=0, vmax=1)
    ax.plot(*train_coords_2d.T, 'k.', markersize=8, alpha=0.7, label='Query points')
    ax.add_patch(Polygon(triangle, fill=False, edgecolor='black', linewidth=2))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.set_title('Classification A = {x: f(x) > 0.5}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Custom colorbar labels
    cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(['FN', 'FP', 'TN', '', 'TP'])
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    
    # Calculate accuracy
    accuracy = np.mean(true_A == pred_A)
    print(f"Classification accuracy: {accuracy:.2%}")
    
    return fig


def run_active_learning_with_video(learner, true_function, n_iterations=40, 
                                   n_candidates=1000, n_initial=10, 
                                   n_oracle_queries=1, retrain_epochs=50,
                                   video_path='progress.mp4', fps=2, verbose=True):
    """
    Run active learning loop and generate video of progress.
    
    Args:
        learner: ActiveLearner instance
        true_function: The true function being learned
        n_iterations: Number of active learning iterations
        n_candidates: Number of candidate points to consider
        n_initial: Number of initial random points
        n_oracle_queries: Number of times to query oracle per point
        retrain_epochs: Number of epochs to train after each new point
        video_path: Path to save the video
        fps: Frames per second for the video
        verbose: Whether to print progress
        
    Returns:
        Dictionary with results
    """
    print(f"Generating video with progress at: {video_path}")
    
    # Initialize video writer
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='aspai_active'))
    writer.setup(fig, video_path, dpi=150)
    
    # Initialize with random points
    if verbose:
        print(f"Initializing with {n_initial} random points...")
    
    for i in range(n_initial):
        x = sample_simplex(1, learner.d, device=learner.device).squeeze(0)
        y = learner.query_oracle(x.cpu().numpy(), n_oracle_queries)
        
        learner.X_train.append(x)
        learner.y_train.append(y)
    
    # Convert to tensors
    X_train_tensor = torch.stack(learner.X_train)
    y_train_tensor = torch.tensor(learner.y_train, dtype=torch.float32)
    
    # Initial training
    if verbose:
        print("Training initial ensemble...")
    learner.ensemble.train_step(X_train_tensor, y_train_tensor, n_epochs=100)
    
    # Create first frame (after initial training)
    fig, axes, metrics = create_frame(learner, true_function, X_train_tensor, 0, fig, axes)
    writer.grab_frame()
    
    if verbose:
        print(f"Frame 0: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, Acc={metrics['accuracy']:.2%}")
    
    # Active learning loop
    all_scores = []
    
    for iteration in range(n_iterations):
        if verbose:
            print(f"\nIteration {iteration + 1}/{n_iterations}")
        
        # Generate candidate points
        candidates = sample_simplex(n_candidates, learner.d, device=learner.device)
        
        # Select next point
        next_point, scores = learner.select_next_point(candidates)
        all_scores.append(scores.cpu().numpy())
        
        # Query oracle
        y_new = learner.query_oracle(next_point.cpu().numpy(), n_oracle_queries)
        
        if verbose:
            print(f"  Oracle response: {y_new:.3f}, Max acquisition score: {scores.max().item():.4f}")
        
        # Add to training set
        learner.X_train.append(next_point)
        learner.y_train.append(y_new)
        
        # Retrain ensemble
        X_train_tensor = torch.stack(learner.X_train)
        y_train_tensor = torch.tensor(learner.y_train, dtype=torch.float32)
        learner.ensemble.train_step(X_train_tensor, y_train_tensor, n_epochs=retrain_epochs)
        
        # Create frame for this iteration
        fig, axes, metrics = create_frame(learner, true_function, X_train_tensor, 
                                         iteration + 1, fig, axes)
        writer.grab_frame()
        
        if verbose:
            print(f"  Metrics: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, Acc={metrics['accuracy']:.2%}")
    
    # Finalize video
    writer.finish()
    plt.close(fig)
    
    print(f"\nVideo saved to: {video_path}")
    
    return {
        'X_train': torch.stack(learner.X_train).cpu().numpy(),
        'y_train': np.array(learner.y_train),
        'acquisition_scores': all_scores
    }


def main():
    """
    Run the example active learning experiment.
    """
    print("=" * 60)
    print("Active Learning Example: 3D Probability Simplex")
    print("=" * 60)
    
    # Set random seed for reproducibility
    seed = None #42
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Problem setup
    d = 3  # Dimension of simplex
    k = 5  # Number of smooth step functions
    
    print(f"\nProblem setup:")
    print(f"  Dimension: d = {d}")
    print(f"  Number of step functions: k = {k}")
    
    # Create true function
    true_function = SumOfSteps(d=d, k=k, seed=seed)
    
    # Create probabilistic oracle
    oracle = create_oracle(true_function)
    
    # Initialize active learner
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    learner = ActiveLearner(
        d=d,
        oracle=oracle,
        n_models=5,
        hidden_dims=[64, 64],
        device=device,
        seed=seed
    )
    
    # Run active learning with video generation
    print("\n" + "=" * 60)
    print("Running Active Learning with Video Generation")
    print("=" * 60)
    
    results = run_active_learning_with_video(
        learner=learner,
        true_function=true_function,
        n_iterations=100,
        n_candidates=1000,
        n_initial=10,
        n_oracle_queries=1,
        retrain_epochs=50,
        video_path='example_progress.mp4',
        fps=2,
        verbose=True
    )
    
    # Visualize results
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    
    fig = visualize_results(learner, true_function, results, 
                           save_path='example_results.png')
    plt.close(fig)
    
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)
    print(f"Total queries: {len(results['X_train'])}")
    print(f"Video saved to: example_progress.mp4")
    print(f"Final frame saved to: example_results.png")


if __name__ == "__main__":
    main()
