"""
Acquisition functions for active learning.
"""

import torch
from .utils import project_to_simplex


def bald_acquisition(predictions):
    """
    Compute BALD (Bayesian Active Learning by Disagreement) acquisition function.

    BALD measures the mutual information between predictions and model parameters.
    It's computed as: H(y|x) - E[H(y|x,θ)]

    For binary classification:
    - H(y|x) is the entropy of the mean prediction
    - E[H(y|x,θ)] is the expected entropy across ensemble members

    Args:
        predictions: torch.Tensor of shape (n_models, n_points)
                    Predicted probabilities from ensemble members

    Returns:
        torch.Tensor of shape (n_points,) with BALD scores
    """
    # Compute mean prediction across ensemble
    mean_pred = predictions.mean(dim=0)

    # Entropy of mean prediction (predictive entropy)
    eps = 1e-10  # For numerical stability
    mean_pred = torch.clamp(mean_pred, eps, 1 - eps)
    predictive_entropy = -mean_pred * torch.log(mean_pred) - (1 - mean_pred) * torch.log(
        1 - mean_pred
    )

    # Expected entropy across models
    predictions = torch.clamp(predictions, eps, 1 - eps)
    entropies = -predictions * torch.log(predictions) - (1 - predictions) * torch.log(
        1 - predictions
    )
    expected_entropy = entropies.mean(dim=0)

    # BALD score = mutual information
    bald_scores = predictive_entropy - expected_entropy

    # Handle any remaining NaN or inf values by replacing with 0
    bald_scores = torch.nan_to_num(bald_scores, nan=0.0, posinf=0.0, neginf=0.0)

    return bald_scores


def uncertainty_acquisition(predictions):
    """
    Simple uncertainty sampling: select points with predictions closest to 0.5.

    Args:
        predictions: torch.Tensor of shape (n_models, n_points)

    Returns:
        torch.Tensor of shape (n_points,) with uncertainty scores
    """
    mean_pred = predictions.mean(dim=0)
    return -torch.abs(mean_pred - 0.5)


def variance_acquisition(predictions):
    """
    Variance-based acquisition: select points with highest prediction variance.

    Args:
        predictions: torch.Tensor of shape (n_models, n_points)

    Returns:
        torch.Tensor of shape (n_points,) with variance scores
    """
    return predictions.var(dim=0)


def optimize_candidates_gd(
    candidates, ensemble, acquisition_fn, n_steps=10, learning_rate=0.1, top_k_fraction=0.1
):
    """
    Optimize candidate points using gradient descent on the acquisition function.

    This improves candidate point initialization in high dimensions by using
    gradient information to move candidates toward regions with high acquisition values.

    Args:
        candidates: torch.Tensor of shape (n_candidates, d) - initial candidate points
        ensemble: EnsembleModel - the trained ensemble for computing acquisition scores
        acquisition_fn: callable - acquisition function (e.g., bald_acquisition)
        n_steps: int - number of gradient descent steps (default: 10)
        learning_rate: float - learning rate for gradient descent (default: 0.1)
        top_k_fraction: float - fraction of candidates to optimize (default: 0.1)

    Returns:
        torch.Tensor of shape (n_candidates, d) - optimized candidate points
    """
    if n_steps <= 0 or top_k_fraction <= 0:
        return candidates

    # Compute initial acquisition scores to select top candidates
    with torch.no_grad():
        predictions = ensemble.predict_proba(candidates, n_samples=1)
        initial_scores = acquisition_fn(predictions)

    # Select top-k candidates to optimize
    n_optimize = max(1, int(len(candidates) * top_k_fraction))
    _, top_indices = torch.topk(initial_scores, n_optimize)

    # Create optimizable copies of top candidates
    optimized_candidates = candidates[top_indices].clone().detach().to(ensemble.device)
    optimized_candidates.requires_grad = True

    # Gradient descent optimization using SGD for better simplex compatibility
    optimizer = torch.optim.SGD([optimized_candidates], lr=learning_rate)

    for step in range(n_steps):
        optimizer.zero_grad()

        # Compute acquisition scores (negative because we want to maximize)
        # Use predict_proba_with_grad to enable gradient computation
        predictions = ensemble.predict_proba_with_grad(optimized_candidates, n_samples=1)
        scores = acquisition_fn(predictions)
        loss = -scores.sum()  # Negative to maximize; sum to optimize each point independently

        # Backward pass
        loss.backward()

        # Gradient descent step
        optimizer.step()

        # Project back onto the simplex
        with torch.no_grad():
            optimized_candidates.data = project_to_simplex(optimized_candidates.data)

    # Replace top candidates with optimized versions
    result = candidates.clone()
    result[top_indices] = optimized_candidates.detach().cpu()

    return result
