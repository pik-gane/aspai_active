"""
Active Learning for Probability Simplex with Neural Network Ensembles.

This package implements an active learning approach using BALD (Bayesian Active
Learning by Disagreement) with neural network ensembles for learning functions
on the probability simplex.
"""

from .model import EnsembleModel
from .acquisition import bald_acquisition
from .active_learner import ActiveLearner
from .utils import sample_simplex, project_to_simplex, grid_simplex

__version__ = "0.1.0"
__all__ = [
    "EnsembleModel",
    "bald_acquisition",
    "ActiveLearner",
    "sample_simplex",
    "project_to_simplex",
    "grid_simplex",
]
