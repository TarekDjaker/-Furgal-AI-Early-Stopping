"""
Component‑wise early stopping for PyTorch models.

This module provides a utility class that monitors the gradient norm
of each learnable parameter in a neural network and freezes
parameters whose gradients fall below a given threshold.  The idea is
inspired by adaptive layer freezing techniques like GradES, which
exploit heterogeneous convergence rates across different components of
transformers and other deep architectures【544076735138045†L54-L66】.  Freezing
parameters when they have effectively converged can reduce both
computation and overfitting.

The implementation is deliberately simple and can be integrated into
arbitrary training loops.  After each backward pass, call
``ComponentEarlyStopping.apply()`` to freeze any parameters whose
gradient norm is too small.  A list of frozen parameter names is
maintained internally and returned for inspection.

Note: This module requires PyTorch (``torch``).  If PyTorch is not
installed, importing this module will fail.  The design is kept
minimal to serve as a teaching example rather than a production
feature.
"""

from __future__ import annotations

import torch
from typing import List, Tuple, Dict, Set


class ComponentEarlyStopping:
    """Monitor and freeze parameters whose gradients vanish.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters should be monitored.
    threshold : float, optional
        Gradient norm threshold below which a parameter is considered
        converged.  Defaults to 1e-3.
    verbose : bool, optional
        If True, prints a message whenever a parameter is frozen.
    """

    def __init__(self, model: torch.nn.Module, threshold: float = 1e-3, verbose: bool = False) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model)}")
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold}")

        self.model = model
        self.threshold = threshold
        self.verbose = verbose
        self.frozen_params: Set[str] = set()  # Use set for O(1) lookups
        self.grad_history: Dict[str, List[float]] = {}  # Track gradient history

    def apply(self) -> List[str]:
        """Inspect gradients and freeze parameters that have converged.

        This method should be called after ``loss.backward()`` and before
        the optimiser step.  For each parameter ``p`` in the model,
        if ``p.grad`` exists and its Euclidean norm is below the
        threshold, the flag ``requires_grad`` is set to ``False`` so
        that the parameter remains unchanged in subsequent iterations.

        Returns
        -------
        List[str]
            A list of parameter names that were frozen at this call.
        """
        newly_frozen: List[str] = []

        # Pre-compute parameters once
        active_params = [(name, param) for name, param in self.model.named_parameters()
                        if param.requires_grad and param.grad is not None]

        for name, param in active_params:
            # Use more efficient norm computation
            grad_norm = param.grad.data.norm(2).item()

            # Track gradient history for analysis
            if name not in self.grad_history:
                self.grad_history[name] = []
            self.grad_history[name].append(grad_norm)

            if grad_norm < self.threshold:
                param.requires_grad_(False)
                self.frozen_params.add(name)
                newly_frozen.append(name)
                if self.verbose:
                    print(f"ComponentEarlyStopping: froze {name} with gradient norm {grad_norm:.2e}")

        return newly_frozen

    def reset(self) -> None:
        """Reset all frozen parameters to be trainable again."""
        # Only reset parameters we actually froze
        for name, param in self.model.named_parameters():
            if name in self.frozen_params:
                param.requires_grad_(True)
        self.frozen_params.clear()
        self.grad_history.clear()

    def summary(self) -> List[str]:
        """Return a list of all frozen parameter names."""
        return list(self.frozen_params)

    def get_gradient_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about gradient norms for each parameter.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Statistics including mean, std, min, max for each parameter.
        """
        import numpy as np

        stats = {}
        for name, history in self.grad_history.items():
            if history:
                stats[name] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history),
                    'last': history[-1],
                    'is_frozen': name in self.frozen_params
                }
        return stats


__all__ = ["ComponentEarlyStopping"]