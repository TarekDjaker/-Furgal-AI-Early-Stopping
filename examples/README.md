# Examples

This directory contains example scripts demonstrating how to use the various early stopping modules in this repository.

## Available Examples

### `example_proximal.py`
Demonstrates proximal gradient descent with early stopping for sparse regression (LASSO). Shows how to:
- Set up synthetic regression data
- Use the `ProximalEarlyStopping` class
- Monitor convergence and sparsity
- Visualize results

**Usage:**
```bash
cd examples
python example_proximal.py
```

### `example_component.py`
Shows component-wise early stopping for neural networks. Demonstrates:
- Training a simple MLP classifier
- Using `ComponentEarlyStopping` to freeze converged layers
- Monitoring which parameters get frozen during training

**Usage:**
```bash
cd examples
python example_component.py
```

### `example_fairness.py`
Illustrates fairness-aware early stopping. Features:
- Creating biased synthetic datasets
- Using `FairnessEarlyStopping` to monitor demographic parity
- Evaluating fairness metrics across different groups

**Usage:**
```bash
cd examples
python example_fairness.py
```

### `example_dp.py`
Demonstrates differentially private SGD with early stopping. Covers:
- Comparing private vs non-private training
- Using `DPSGDEarlyStopping` with noise injection
- Analyzing privacy-utility trade-offs

**Usage:**
```bash
cd examples
python example_dp.py
```

## Requirements

The examples use additional packages beyond the core requirements:
- `matplotlib` (for plotting in proximal example)
- `torch` (for neural network examples)
- `scikit-learn` (for datasets and preprocessing)

Install them with:
```bash
pip install matplotlib torch scikit-learn
```

## Notes

- All examples are self-contained and can be run independently
- They use synthetic datasets to avoid external dependencies
- Results may vary slightly due to random initialization
- Examples are designed to be educational rather than performance-optimized