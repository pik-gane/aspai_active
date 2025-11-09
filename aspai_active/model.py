"""
Neural network ensemble model for active learning.
"""

import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    """
    Simple feedforward neural network for binary classification.
    """

    def __init__(self, input_dim, hidden_dims=[64, 64], dropout=0.1):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim

        # Output layer (single logit)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, 1)
        """
        return self.network(x)


class EnsembleModel:
    """
    Ensemble of neural networks for uncertainty estimation.
    """

    def __init__(self, n_models, input_dim, hidden_dims=[64, 64], dropout=0.1, device="cpu"):
        """
        Args:
            n_models: Number of models in the ensemble
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions for each model
            dropout: Dropout probability
            device: Torch device to use
        """
        self.n_models = n_models
        self.device = device

        # Create ensemble of models
        self.models = [
            SimpleNN(input_dim, hidden_dims, dropout).to(device) for _ in range(n_models)
        ]

        # Create optimizers for each model
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in self.models]

        self.loss_fn = nn.BCEWithLogitsLoss()

    def train_step(self, x, y, n_epochs=10):
        """
        Train all models in the ensemble on the given data.

        Args:
            x: Input tensor of shape (n_samples, input_dim)
            y: Target tensor of shape (n_samples,) with binary labels
            n_epochs: Number of training epochs
        """
        x = x.to(self.device)
        y = y.to(self.device).float().reshape(-1, 1)

        for model, optimizer in zip(self.models, self.optimizers):
            model.train()

            for epoch in range(n_epochs):
                optimizer.zero_grad()

                # Forward pass
                logits = model(x)
                loss = self.loss_fn(logits, y)

                # Backward pass
                loss.backward()
                optimizer.step()

    def predict_proba(self, x, n_samples=1):
        """
        Predict probabilities using the ensemble.

        Args:
            x: Input tensor of shape (n_points, input_dim)
            n_samples: Number of stochastic forward passes per model (for dropout)

        Returns:
            predictions: torch.Tensor of shape (n_models * n_samples, n_points)
                        Each row is predictions from one model/sample
        """
        x = x.to(self.device)
        predictions = []

        for model in self.models:
            model.train()  # Keep dropout active for uncertainty estimation

            with torch.no_grad():
                for _ in range(n_samples):
                    logits = model(x)
                    probs = torch.sigmoid(logits).squeeze(-1)
                    predictions.append(probs.cpu())

        return torch.stack(predictions)

    def predict_proba_with_grad(self, x, n_samples=1):
        """
        Predict probabilities using the ensemble with gradient support.

        This version enables gradients through the network for optimization purposes.

        Args:
            x: Input tensor of shape (n_points, input_dim)
            n_samples: Number of stochastic forward passes per model (for dropout)

        Returns:
            predictions: torch.Tensor of shape (n_models * n_samples, n_points)
                        Each row is predictions from one model/sample
        """
        x = x.to(self.device)
        predictions = []

        for model in self.models:
            model.eval()  # Use eval mode for deterministic predictions during optimization

            for _ in range(n_samples):
                logits = model(x)
                probs = torch.sigmoid(logits).squeeze(-1)
                predictions.append(probs)

        return torch.stack(predictions)

    def predict_mean(self, x):
        """
        Predict mean probability across the ensemble.

        Args:
            x: Input tensor of shape (n_points, input_dim)

        Returns:
            Mean probabilities of shape (n_points,)
        """
        predictions = self.predict_proba(x, n_samples=1)
        return predictions.mean(dim=0)
