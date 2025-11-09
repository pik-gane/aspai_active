"""
Acquisition functions for active learning.
"""

import torch
import numpy as np


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
    predictive_entropy = -mean_pred * torch.log(mean_pred) - (1 - mean_pred) * torch.log(1 - mean_pred)
    
    # Expected entropy across models
    predictions = torch.clamp(predictions, eps, 1 - eps)
    entropies = -predictions * torch.log(predictions) - (1 - predictions) * torch.log(1 - predictions)
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
