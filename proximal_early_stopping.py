"""
Proximal gradient descent with early stopping.

This module provides a simple implementation of proximal gradient descent
for composite optimisation problems with an early‑stopping rule.  The
problem we consider is

    minimise  f(x) + λ · φ(x)

where f is a differentiable loss (not necessarily convex) and φ is a
convex regulariser such as the ℓ₁‑norm.  Early stopping acts as an
implicit regulariser: instead of running the algorithm to full
convergence, we terminate once the updates or objective values cease
to improve significantly.  This behaviour is analogous to the
discrepancy principle for inverse problems【739116097455079†L17-L35】 and the
instance‑dependent early stopping strategies proposed by recent
works【620758099344515†L21-L45】.

The implementation below is educational rather than high‑performance.
It can serve as a baseline for sparse regression (lasso) or non‑convex
penalised problems (e.g. SCAD, MCP).  The stopping criterion is
configurable via a tolerance parameter and a patience counter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List

import numpy as np


def l1_proximal(x: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the proximal operator for the ℓ₁‑norm.

    Parameters
    ----------
    x : ndarray
        Input vector.
    threshold : float
        Threshold parameter λ·step_size.

    Returns
    -------
    ndarray
        Result of prox_{threshold · ||·||₁}(x).
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


@dataclass
class ProximalEarlyStopping:
    """Proximal gradient solver with early stopping.

    This class minimises the objective

        0.5 · ||Ax − y||² + λ · φ(x)

    using iterative proximal updates.  A stopping rule monitors the
    change in the parameter vector and halts the procedure when
    improvements become negligible for a number of consecutive
    iterations (the `patience`).

    Parameters
    ----------
    design : ndarray
        The design matrix A of size (n_samples, n_features).
    response : ndarray
        The response vector y of length n_samples.
    lam : float, optional
        Regularisation parameter λ for the ℓ₁ penalty.  Defaults to 0.1.
    step_size : float, optional
        Step size for gradient descent.  For convex f, step_size
        must be smaller than 1/||AᵀA||₂ to ensure convergence.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for the relative change in the parameter vector
        below which improvement is considered negligible.
    patience : int, optional
        Number of consecutive negligible improvements allowed before
        stopping.
    callback : Callable[[int, np.ndarray, float], None], optional
        Optional function called at each iteration with arguments
        (iteration, current_parameter, objective_value).
    """

    design: np.ndarray
    response: np.ndarray
    lam: float = 0.1
    step_size: float = 1.0
    max_iter: int = 500
    tol: float = 1e-4
    patience: int = 5
    callback: Optional[Callable[[int, np.ndarray, float], None]] = None

    def __post_init__(self) -> None:
        self.n_samples, self.n_features = self.design.shape
        self.x = np.zeros(self.n_features)
        self.obj_values: List[float] = []
        self.history: List[np.ndarray] = []

    def _objective(self, x: np.ndarray) -> float:
        """Compute the objective value 0.5||Ax − y||² + λ·||x||₁."""
        residual = self.design @ x - self.response
        return 0.5 * np.dot(residual, residual) + self.lam * np.sum(np.abs(x))

    def fit(self) -> Tuple[np.ndarray, int, List[float]]:
        """Run proximal gradient descent until the stopping criterion triggers.

        Returns
        -------
        x : ndarray
            The estimated parameter vector at stopping.
        stop_iter : int
            The iteration index at which the algorithm stopped.
        obj_values : list
            History of objective values at each iteration.
        """
        patience_counter = 0
        prev_x = self.x.copy()

        for t in range(self.max_iter):
            # Gradient of the least squares term
            grad = self.design.T @ (self.design @ self.x - self.response)
            # Proximal update for ℓ₁ penalty
            z = self.x - self.step_size * grad
            self.x = l1_proximal(z, self.lam * self.step_size)

            # Compute objective and record
            obj = self._objective(self.x)
            self.obj_values.append(obj)
            self.history.append(self.x.copy())

            # Invoke callback if provided
            if self.callback is not None:
                self.callback(t, self.x, obj)

            # Check relative change for early stopping
            diff = np.linalg.norm(self.x - prev_x)
            norm_x = np.linalg.norm(prev_x) + 1e-12
            rel_change = diff / norm_x
            if rel_change < self.tol:
                patience_counter += 1
            else:
                patience_counter = 0
            if patience_counter >= self.patience:
                break
            prev_x = self.x.copy()

        return self.x, t, self.obj_values


__all__ = [
    "ProximalEarlyStopping",
    "l1_proximal",
]