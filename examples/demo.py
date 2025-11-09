"""
Quick demonstration of the aspai_active package.

This script shows basic usage of the active learning system.
"""

import numpy as np
import torch
from aspai_active import ActiveLearner, sample_simplex


def demo_simple_oracle():
    """
    Demonstrate with a simple oracle where f(x) = x[0].
    The set A = {x: f(x) > 0.5} is the region where x[0] > 0.5.
    """
    print("=" * 60)
    print("Simple Demo: f(x) = x[0]")
    print("=" * 60)
    
    # Define oracle
    def oracle(x):
        # True function: f(x) = x[0]
        # Oracle returns 1 with probability f(x)
        return int(np.random.random() < x[0])
    
    # Create learner
    learner = ActiveLearner(
        d=3,
        oracle=oracle,
        n_models=3,
        hidden_dims=[32, 32],
        device="cpu",
        seed=42
    )
    
    # Run active learning
    print("\nRunning 20 iterations...")
    results = learner.run(
        n_iterations=20,
        n_candidates=500,
        n_initial=5,
        n_oracle_queries=3,
        verbose=False
    )
    
    print(f"\nTotal queries: {len(results['X_train'])}")
    
    # Test predictions
    test_points = sample_simplex(100, d=3)
    predictions = learner.predict(test_points)
    
    # Compute accuracy
    true_labels = (test_points[:, 0].numpy() > 0.5).astype(int)
    pred_labels = (predictions > 0.5).astype(int)
    accuracy = np.mean(true_labels == pred_labels)
    
    print(f"Classification accuracy on test set: {accuracy:.1%}")
    print(f"Mean absolute error: {np.mean(np.abs(predictions - test_points[:, 0].numpy())):.3f}")
    
    # Show some example predictions
    print("\nExample predictions (first 5 test points):")
    print("x                              | True f(x) | Predicted | Correct")
    print("-" * 70)
    for i in range(5):
        x = test_points[i].numpy()
        true_f = x[0]
        pred_f = predictions[i]
        correct = "✓" if (true_f > 0.5) == (pred_f > 0.5) else "✗"
        print(f"[{x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}] | {true_f:.3f}     | {pred_f:.3f}     | {correct}")


def demo_comparison():
    """
    Compare initial random sampling vs after active learning.
    """
    print("\n" + "=" * 60)
    print("Comparison Demo: Random vs Active Learning")
    print("=" * 60)
    
    # Simple oracle: f(x) = sum of first two coordinates / 2
    def oracle(x):
        prob = (x[0] + x[1]) / 2
        return int(np.random.random() < prob)
    
    # Test with random sampling
    print("\n1. Random Sampling (20 points):")
    X_random = sample_simplex(20, d=3)
    y_random = np.array([oracle(x.numpy()) for x in X_random])
    print(f"   Collected {len(X_random)} random points")
    
    # Test with active learning
    print("\n2. Active Learning (20 points):")
    learner = ActiveLearner(
        d=3,
        oracle=oracle,
        n_models=3,
        device="cpu",
        seed=42
    )
    
    results = learner.run(
        n_iterations=15,
        n_candidates=500,
        n_initial=5,
        n_oracle_queries=1,
        verbose=False
    )
    
    print(f"   Collected {len(results['X_train'])} points with active learning")
    
    # Compare on test set
    test_points = sample_simplex(200, d=3)
    test_true = np.array([(x[0].item() + x[1].item()) / 2 for x in test_points])
    
    predictions = learner.predict(test_points)
    mae_active = np.mean(np.abs(predictions - test_true))
    
    print(f"\n3. Results:")
    print(f"   Active Learning MAE: {mae_active:.3f}")
    print(f"   Active learning strategically selected informative points!")


if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run demos
    demo_simple_oracle()
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nTo run the full example with visualization:")
    print("  cd examples && python example_3d.py")
