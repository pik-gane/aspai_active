"""
Active learning loop implementation.
"""

import torch
import numpy as np
from .model import EnsembleModel
from .acquisition import bald_acquisition
from .utils import sample_simplex


class ActiveLearner:
    """
    Active learning system for learning functions on the probability simplex.
    """
    
    def __init__(self, d, oracle, n_models=5, hidden_dims=[64, 64], 
                 device="cpu", seed=None):
        """
        Initialize the active learner.
        
        Args:
            d: Dimension of the probability simplex
            oracle: Callable that takes x and returns binary outcome (0 or 1)
            n_models: Number of models in the ensemble
            hidden_dims: Hidden layer dimensions for neural networks
            device: Torch device to use
            seed: Random seed for reproducibility
        """
        self.d = d
        self.oracle = oracle
        self.device = device
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Initialize ensemble
        self.ensemble = EnsembleModel(
            n_models=n_models,
            input_dim=d,
            hidden_dims=hidden_dims,
            device=device
        )
        
        # Storage for training data
        self.X_train = []
        self.y_train = []
        
    def query_oracle(self, x, n_queries=1):
        """
        Query the oracle at point x multiple times.
        
        Args:
            x: Point on the simplex (shape: (d,))
            n_queries: Number of times to query the oracle
            
        Returns:
            Average of oracle responses (estimate of f(x))
        """
        responses = [self.oracle(x) for _ in range(n_queries)]
        return np.mean(responses)
    
    def select_next_point(self, candidate_points, acquisition_fn=bald_acquisition):
        """
        Select the next point to query using the acquisition function.
        
        Args:
            candidate_points: torch.Tensor of shape (n_candidates, d)
            acquisition_fn: Acquisition function to use
            
        Returns:
            Selected point (torch.Tensor of shape (d,))
            Acquisition scores for all candidates
        """
        # Get predictions from ensemble
        predictions = self.ensemble.predict_proba(candidate_points, n_samples=1)
        
        # Compute acquisition scores
        scores = acquisition_fn(predictions)
        
        # Select point with highest score
        best_idx = torch.argmax(scores)
        selected_point = candidate_points[best_idx]
        
        return selected_point, scores
    
    def run(self, n_iterations, n_candidates=1000, n_initial=10, 
            n_oracle_queries=1, retrain_epochs=50, verbose=True):
        """
        Run the active learning loop.
        
        Args:
            n_iterations: Number of active learning iterations
            n_candidates: Number of candidate points to consider at each iteration
            n_initial: Number of initial random points
            n_oracle_queries: Number of times to query oracle per point
            retrain_epochs: Number of epochs to train after each new point
            verbose: Whether to print progress
            
        Returns:
            Dictionary with results
        """
        # Initialize with random points
        if verbose:
            print(f"Initializing with {n_initial} random points...")
        
        for i in range(n_initial):
            x = sample_simplex(1, self.d, device=self.device).squeeze(0)
            y = self.query_oracle(x.cpu().numpy(), n_oracle_queries)
            
            self.X_train.append(x)
            self.y_train.append(y)
        
        # Convert to tensors
        X_train_tensor = torch.stack(self.X_train)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        
        # Initial training
        if verbose:
            print("Training initial ensemble...")
        self.ensemble.train_step(X_train_tensor, y_train_tensor, n_epochs=100)
        
        # Active learning loop
        all_scores = []
        
        for iteration in range(n_iterations):
            if verbose:
                print(f"\nIteration {iteration + 1}/{n_iterations}")
            
            # Generate candidate points
            candidates = sample_simplex(n_candidates, self.d, device=self.device)
            
            # Select next point
            next_point, scores = self.select_next_point(candidates)
            all_scores.append(scores.cpu().numpy())
            
            # Query oracle
            y_new = self.query_oracle(next_point.cpu().numpy(), n_oracle_queries)
            
            if verbose:
                print(f"  Selected point: {next_point.cpu().numpy()}")
                print(f"  Oracle response: {y_new:.3f}")
                print(f"  Max acquisition score: {scores.max().item():.4f}")
            
            # Add to training set
            self.X_train.append(next_point)
            self.y_train.append(y_new)
            
            # Retrain ensemble
            X_train_tensor = torch.stack(self.X_train)
            y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
            self.ensemble.train_step(X_train_tensor, y_train_tensor, n_epochs=retrain_epochs)
        
        return {
            'X_train': torch.stack(self.X_train).cpu().numpy(),
            'y_train': np.array(self.y_train),
            'acquisition_scores': all_scores
        }
    
    def predict(self, X):
        """
        Predict probabilities for given points.
        
        Args:
            X: torch.Tensor of shape (n_points, d)
            
        Returns:
            Mean predictions of shape (n_points,)
        """
        return self.ensemble.predict_mean(X).cpu().numpy()
