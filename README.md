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

# Make predictions
from aspai_active import sample_simplex
test_points = sample_simplex(100, d=10)
predictions = learner.predict(test_points)
```

## Example Application

The package includes a complete example with `d=3` (for visualization) where the true function is a sum of 5 smooth step functions along random hyperplanes.

### Running the Example

```bash
cd examples
python example_3d.py
```

This will:
1. Create a synthetic function as a sum of 5 smooth step functions
2. Run active learning with BALD acquisition
3. Generate an MP4 video (`example_progress.mp4`) showing the learning progress with one frame per query
4. Generate a final visualization image (`example_results.png`) showing:
   - True function values
   - Estimated function values
   - Classification accuracy for $A = \{x : f(x) > 0.5\}$
   - Query points selected by the algorithm

### Example Output

The example produces:

1. **Video** (`example_progress.mp4`): An animated visualization showing learning progress
   - One frame per query iteration
   - Three panels showing true function, estimated function, and classification
   - Fitness metrics displayed on each frame: TP (True Positives), TN (True Negatives), FP (False Positives), FN (False Negatives), and Accuracy
   
2. **Image** (`example_results.png`): A final visualization with three panels:
   - **Left**: True function $f(x)$ on the simplex
   - **Middle**: Learned function estimate
   - **Right**: Classification correctness (TP/TN/FP/FN)

All query points are shown as black dots.

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
- `predict_mean(X)`: Get mean prediction

### Acquisition Functions

#### `bald_acquisition(predictions)`

Compute BALD (Bayesian Active Learning by Disagreement) scores.

- **Input**: Tensor of shape `(n_models, n_points)` with predictions
- **Output**: Tensor of shape `(n_points,)` with acquisition scores
- **Higher scores** = more informative points

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
