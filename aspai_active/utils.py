"""
Utility functions for working with probability simplices.
"""

import torch
import numpy as np


def sample_simplex(n_samples, d, device="cpu"):
    """
    Sample uniformly from the probability simplex S_d.
    
    Uses the method of sampling from a Dirichlet(1, ..., 1) distribution.
    
    Args:
        n_samples: Number of samples to generate
        d: Dimension of the simplex (number of points)
        device: Torch device to use
        
    Returns:
        torch.Tensor of shape (n_samples, d) with each row summing to 1
    """
    # Sample from Dirichlet(1, ..., 1) which is uniform on the simplex
    samples = torch.distributions.Dirichlet(torch.ones(d)).sample((n_samples,))
    return samples.to(device)


def project_to_simplex(x):
    """
    Project points onto the probability simplex.
    
    Args:
        x: torch.Tensor of shape (n, d)
        
    Returns:
        torch.Tensor of shape (n, d) projected onto the simplex
    """
    # Ensure non-negative
    x = torch.clamp(x, min=0)
    # Normalize to sum to 1
    x_sum = x.sum(dim=-1, keepdim=True)
    # Avoid division by zero
    x_sum = torch.where(x_sum > 0, x_sum, torch.ones_like(x_sum))
    return x / x_sum


def grid_simplex(n_per_dim, d, device="cpu"):
    """
    Create a grid of points on the probability simplex for d=3 (triangle).
    
    For d=3, creates a triangular grid. For d>3, uses random sampling.
    
    Args:
        n_per_dim: Number of points per dimension
        d: Dimension of the simplex
        device: Torch device to use
        
    Returns:
        torch.Tensor of shape (n_points, d)
    """
    if d == 3:
        # For d=3, create a triangular grid
        points = []
        for i in range(n_per_dim):
            for j in range(n_per_dim - i):
                k = n_per_dim - 1 - i - j
                if k >= 0:
                    point = torch.tensor([i, j, k], dtype=torch.float32) / (n_per_dim - 1)
                    points.append(point)
        return torch.stack(points).to(device)
    else:
        # For higher dimensions, use uniform sampling
        return sample_simplex(n_per_dim ** 2, d, device)


def barycentric_to_cartesian(points):
    """
    Convert barycentric coordinates (3D simplex) to 2D Cartesian for plotting.
    
    Args:
        points: torch.Tensor or numpy.ndarray of shape (n, 3) with barycentric coords
        
    Returns:
        numpy.ndarray of shape (n, 2) with Cartesian coordinates
    """
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    
    # Standard triangle vertices in 2D
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, np.sqrt(3) / 2])
    
    # Convert barycentric to cartesian
    x = points[:, 0:1] * v1[0] + points[:, 1:2] * v2[0] + points[:, 2:3] * v3[0]
    y = points[:, 0:1] * v1[1] + points[:, 1:2] * v2[1] + points[:, 2:3] * v3[1]
    
    return np.hstack([x, y])
