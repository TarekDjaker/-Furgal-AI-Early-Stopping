# Early Stopping Extensions for Iterative and Modern Machine‑Learning Algorithms

This repository extends the original **Early‑Stopping** project with additional modules and examples based on
recent research in machine learning and optimisation.  It preserves the
original Landweber and L2‑boosting implementations but adds new
capabilities for non‑convex optimisation, component‑wise stopping, fairness
and privacy.  The goal is to provide a **clean and well documented codebase**
that can serve as a starting point for research projects at the Master 2
level.

## What’s new?

The enhancements provided here are inspired by recent theoretical and
empirical advances in the early‑stopping literature【739116097455079†L17-L35】【856229582317957†L14-L23】.  They focus on
bridging the gap between simple linear models and the complex models
encountered in modern deep learning, while keeping the code readable
and educational.

### ✨ Proximal early stopping

`proximal_early_stopping.py` implements a **proximal gradient
descent** solver for composite optimisation problems of the form

\[\min_x\; f(x) + \lambda \phi(x)\,\]

where `f` is a differentiable (not necessarily convex) loss and `φ` is a
convex regulariser such as the L1‑norm.  The solver includes an
**early‑stopping mechanism** based on the decay of the primal–dual gap
and the norm of the proximal gradient.  This module can be used to
replicate experiments on sparse regression (lasso, SCAD, MCP) and to
explore the theoretical ideas discussed in the thesis.

### 🧩 Component‑wise early stopping

`component_early_stopping.py` provides a utility class that wraps a
PyTorch model and monitors the gradient norms of each layer.  Layers
whose gradients fall below a user‑specified threshold are **frozen**
(training is stopped for that layer), allowing you to save computation
and reduce overfitting in large neural networks.  This feature is
inspired by the GradES algorithm from 2025 but is written from
scratch to avoid external dependencies.

### ⚖️ Fairness‑aware stopping

`fairness_early_stopping.py` implements simple functions to compute
fairness metrics (difference in error rates across sensitive groups) and
provides an early‑stopping callback that halts training when
improvements in fairness plateau.  It can be integrated in any
training loop that outputs predictions, true labels and group labels.
This module demonstrates how early stopping can be used as a tool for
fairness, in line with recent studies on fairness dynamics【330857330321960†L29-L41】.

### 🔐 Differential‑privacy friendly stopping

`dp_early_stopping.py` offers a skeleton for training models under
differential‑privacy constraints.  It implements a simple private
gradient descent (adding Gaussian noise to gradients) and stops when
the validation loss fails to improve.  The API returns the accumulated
privacy budget and final model.  Although the implementation is
lightweight, it can be extended to more sophisticated DP‑SGD
frameworks.

### 🔨 Examples

Several scripts are provided in the `examples` directory to
demonstrate how to use the new modules:

* `example_proximal.py` — shows how to solve a sparse regression
  problem with proximal gradient descent and early stopping.
* `example_component.py` — trains a tiny multi‑layer perceptron with
  component‑wise early stopping on a toy dataset.
* `example_fairness.py` — demonstrates fairness‑aware stopping on a
  synthetic binary classification task with sensitive attributes.

These examples are intentionally simple so that you can adapt them to
your own research projects.

## Installation

Make sure you have Python 3.8+ installed.  Then clone this
repository and install the dependencies:

```bash
git clone https://github.com/TarekDjaker/-Furgal-AI-Early-Stopping.git
cd -Furgal-AI-Early-Stopping
pip install -r requirements.txt
```

The new modules depend on numpy, scipy, scikit‑learn and PyTorch.  If
you do not need some of the functionality, feel free to omit the
corresponding packages.

## Usage

The legacy Landweber and L2‑boosting classes are located in
`landweber.py` and `L2_boost.py` respectively.  You can still run

```bash
python example.py
```

to reproduce the original discrepancy principle demo.  For the new
features, refer to the examples in the `examples/` folder.  All
modules are documented with docstrings; simply open the source files to
learn more about their APIs.

### Running Examples

The `examples/` directory contains practical demonstrations of each module:

```bash
# Proximal gradient descent for sparse regression
python examples/example_proximal.py

# Component-wise early stopping for neural networks  
python examples/example_component.py

# Fairness-aware early stopping
python examples/example_fairness.py

# Differential privacy with early stopping
python examples/example_dp.py
```

### Quick Start

```python
# Proximal gradient descent example
from proximal_early_stopping import ProximalEarlyStopping
import numpy as np

# Create synthetic data
X = np.random.randn(100, 10)
y = np.random.randn(100)

# Set up solver
solver = ProximalEarlyStopping(
    design=X, response=y, lam=0.1, 
    step_size=0.001, max_iter=1000
)

# Fit model
params, iterations, objectives = solver.fit()
```

## Contributing

Contributions are welcome!  The `todo.md` file from the original
project lists some long‑term ideas (e.g., interactive stopping
visualisations).  You are invited to extend the library with
additional stopping rules, fairness metrics or privacy mechanisms.

## License

This project is released under the MIT License.  See `LICENSE` for
details.