# aspai_active

Active Learning for Probability Simplex with Neural Network Ensembles

## Overview

This package implements an active learning approach using **BALD (Bayesian Active Learning by Disagreement)** with neural network ensembles for learning functions on the probability simplex.

### Problem Description

We want to learn an unknown function $f: S_d \rightarrow [0,1]$, where $S_d$ is the set of probability distributions on $d$ points. Specifically, we want to identify the set $A = \{x : f(x) > 1/2\}$.

The challenge is that we can only observe $f(x)$ indirectly through a noisy oracle $g(x)$ that returns 1 with probability $f(x)$ and 0 otherwise. Each oracle call is:
- **Probabilistic**: Returns binary outcomes based on $f(x)$
- **Expensive**: We want to minimize the number of queries
- **Independent**: Each call is an independent sample

### Solution Approach

1. **Ensemble of Neural Networks**: We approximate $f$ using multiple neural networks to estimate uncertainty
2. **BALD Acquisition Function**: We use Bayesian Active Learning by Disagreement to select informative query points
3. **Active Learning Loop**: We iteratively:
   - Select the most informative point to query
   - Query the oracle (possibly multiple times for better estimates)
   - Retrain the ensemble with the new data

## Installation

### From source

```bash
git clone https://github.com/pik-gane/aspai_active.git
cd aspai_active
pip install -e .
```

### Dependencies

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0

## Quick Start

```python
import torch
import numpy as np
from aspai_active import ActiveLearner

# Define your oracle function (returns 0 or 1)
def my_oracle(x):
    # x is a point on the simplex (sums to 1)
    # Return 1 with probability f(x), 0 otherwise
    prob = some_function(x)  # Your unknown function
    return int(np.random.random() < prob)

# Create active learner
learner = ActiveLearner(
    d=10,  # Dimension of simplex
    oracle=my_oracle,
    n_models=5,  # Number of models in ensemble
    device="cpu"
)

# Run active learning
results = learner.run(
    n_iterations=50,
    n_candidates=1000,
    n_initial=20,
    n_oracle_queries=3,  # Query oracle 3 times per point
    verbose=True
)

# For high dimensions, enable gradient descent optimization
# This improves candidate selection by optimizing points toward high acquisition values
results = learner.run(
    n_iterations=50,
    n_candidates=1000,
    n_initial=20,
    n_oracle_queries=3,
    optimize_candidates=True,  # Enable gradient descent optimization
    gd_steps=20,              # Number of optimization steps
    gd_lr=0.05,               # Learning rate
    gd_top_k_fraction=0.2,    # Optimize top 20% of candidates
    verbose=True
)

# Make predictions
from aspai_active import sample_simplex
test_points = sample_simplex(100, d=10)
predictions = learner.predict(test_points)
```

## Example Applications

The package includes two complete examples:

### 3D Example (Visualization)

Example with `d=3` for visualization where the true function is a sum of 5 smooth step functions along random hyperplanes.

#### Running the Example

```bash
cd examples
python example_3d.py
```

This will:
1. Create a synthetic function as a sum of 5 smooth step functions
2. Run active learning with BALD acquisition
3. Generate visualizations showing:
   - True function values
   - Estimated function values
   - Classification accuracy for $A = \{x : f(x) > 0.5\}$
   - Query points selected by the algorithm

#### Example Output

The example produces a visualization with three panels:
- **Left**: True function $f(x)$ on the simplex
- **Middle**: Learned function estimate
- **Right**: Classification correctness (TP/TN/FP/FN)

All query points are shown as black dots.

### High-Dimensional Example

Example with `d=20` demonstrating gradient descent optimization for candidate points.

#### Running the Example

```bash
cd examples
python example_highdim.py
```

This will:
1. Run multiple trials comparing with and without gradient optimization
2. Show accuracy improvements from gradient-based candidate optimization
3. Generate comparison visualizations

The gradient descent optimization is particularly beneficial in high dimensions where random sampling becomes less effective.

## API Reference

### Core Classes

#### `ActiveLearner`

Main class for active learning.

```python
learner = ActiveLearner(
    d,                    # Dimension of simplex
    oracle,               # Oracle function
    n_models=5,          # Number of ensemble models
    hidden_dims=[64, 64], # Hidden layer sizes
    device="cpu",        # Torch device
    seed=None            # Random seed
)
```

**Methods:**
- `run(n_iterations, ...)`: Run the active learning loop
  - New parameters for gradient optimization:
    - `optimize_candidates` (bool): Enable gradient descent optimization (default: False)
    - `gd_steps` (int): Number of gradient descent steps (default: 10)
    - `gd_lr` (float): Learning rate for optimization (default: 0.1)
    - `gd_top_k_fraction` (float): Fraction of top candidates to optimize (default: 0.1)
- `predict(X)`: Get predictions for new points
- `query_oracle(x, n_queries)`: Query oracle at a specific point

#### `EnsembleModel`

Ensemble of neural networks for uncertainty estimation.

```python
ensemble = EnsembleModel(
    n_models,      # Number of models
    input_dim,     # Input dimension
    hidden_dims,   # Hidden layer sizes
    device="cpu"
)
```

**Methods:**
- `train_step(X, y, n_epochs)`: Train on data
- `predict_proba(X)`: Get probability predictions
- `predict_proba_with_grad(X)`: Get predictions with gradient support (for optimization)
- `predict_mean(X)`: Get mean prediction

### Acquisition Functions

#### `bald_acquisition(predictions)`

Compute BALD (Bayesian Active Learning by Disagreement) scores.

- **Input**: Tensor of shape `(n_models, n_points)` with predictions
- **Output**: Tensor of shape `(n_points,)` with acquisition scores
- **Higher scores** = more informative points

#### `optimize_candidates_gd(candidates, ensemble, acquisition_fn, ...)`

Optimize candidate points using gradient descent on the acquisition function.

- **Input**: 
  - `candidates`: Tensor of initial candidate points
  - `ensemble`: Trained EnsembleModel
  - `acquisition_fn`: Acquisition function to maximize
  - `n_steps`: Number of gradient descent steps (default: 10)
  - `learning_rate`: Learning rate (default: 0.1)
  - `top_k_fraction`: Fraction of candidates to optimize (default: 0.1)
- **Output**: Tensor of optimized candidates
- **Use case**: Improves performance in high dimensions

#### `uncertainty_acquisition(predictions)`

Simple uncertainty sampling (distance from 0.5).

#### `variance_acquisition(predictions)`

Variance-based acquisition function.

### Utility Functions

#### `sample_simplex(n_samples, d, device="cpu")`

Sample uniformly from the probability simplex.

#### `grid_simplex(n_per_dim, d, device="cpu")`

Create a grid of points on the simplex (for d=3, creates triangular grid).

#### `project_to_simplex(x)`

Project points onto the probability simplex.

#### `barycentric_to_cartesian(points)`

Convert 3D simplex points to 2D for visualization.

## How BALD Works

BALD measures the **mutual information** between predictions and model parameters:

$$\text{BALD}(x) = H[\mathbb{E}[p(y|x,\theta)]] - \mathbb{E}[H[p(y|x,\theta)]]$$

Where:
- First term: Entropy of the mean prediction (predictive uncertainty)
- Second term: Expected entropy over models (aleatoric uncertainty)
- Difference: Epistemic uncertainty (what we can reduce by querying)

**High BALD scores** indicate points where:
- The ensemble is uncertain (epistemic uncertainty)
- But individual models are confident (low aleatoric uncertainty)
- These are the most informative points to query

## Gradient Descent Optimization

### Overview

For high-dimensional problems (e.g., `d > 10`), random sampling of candidate points becomes less effective. The gradient descent optimization improves candidate selection by:

1. **Starting with random candidates**: Sample points uniformly from the simplex
2. **Selecting top candidates**: Choose the top-k candidates with highest initial acquisition scores
3. **Gradient optimization**: Use gradient descent to optimize these candidates to maximize the acquisition function
4. **Simplex projection**: After each gradient step, project points back onto the simplex to maintain constraints

### Benefits

- **High dimensions**: Most effective when `d > 10` where random sampling struggles
- **Better exploration**: Finds regions with higher uncertainty more efficiently
- **Configurable**: Can adjust optimization steps, learning rate, and fraction of candidates to optimize

### Usage

```python
learner.run(
    n_iterations=50,
    optimize_candidates=True,  # Enable optimization
    gd_steps=20,              # More steps for higher dimensions
    gd_lr=0.05,               # Lower learning rate for stability
    gd_top_k_fraction=0.2     # Optimize top 20% of candidates
)
```

### When to Use

- **Enable for d ≥ 10**: Particularly beneficial in high dimensions
- **Disable for d < 5**: Little benefit in low dimensions, adds computation time
- **Moderate d (5-10)**: Optional, test to see if it helps your specific problem

## Technical Details

### Neural Network Architecture

Each model in the ensemble:
- Input layer: dimension `d` (simplex dimension)
- Hidden layers: configurable (default: `[64, 64]`)
- Dropout: 0.1 (for uncertainty estimation)
- Output: single logit for binary classification
- Activation: ReLU for hidden layers
- Loss: Binary cross-entropy with logits

### Training Strategy

- **Initial phase**: Train with random points
- **Active learning**: Iteratively select points with BALD
- **Multiple queries**: Query oracle multiple times per point for better estimates
- **Incremental training**: Add new points and retrain ensemble

### Probability Simplex

The probability simplex $S_d$ is:
$$S_d = \{x \in \mathbb{R}^d : x_i \geq 0, \sum_{i=1}^d x_i = 1\}$$

We sample uniformly using the Dirichlet distribution with all parameters equal to 1.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{aspai_active,
  title = {aspai_active: Active Learning for Probability Simplex},
  author = {aspai_active contributors},
  year = {2024},
  url = {https://github.com/pik-gane/aspai_active}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
