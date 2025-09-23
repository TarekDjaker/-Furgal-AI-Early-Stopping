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
from typing import Callable, Optional, Tuple, List, Dict, Any

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
    # Vectorized soft-thresholding with numerical stability
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def l2_proximal(x: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the proximal operator for the ℓ₂-norm (group lasso).

    Parameters
    ----------
    x : ndarray
        Input vector.
    threshold : float
        Threshold parameter.

    Returns
    -------
    ndarray
        Result of prox_{threshold · ||·||₂}(x).
    """
    norm_x = np.linalg.norm(x)
    if norm_x <= threshold:
        return np.zeros_like(x)
    return x * (1 - threshold / norm_x)


def elastic_net_proximal(x: np.ndarray, l1_weight: float, l2_weight: float, step_size: float) -> np.ndarray:
    """Apply the proximal operator for elastic net regularization.

    Combines L1 and L2 penalties: l1_weight * ||x||₁ + l2_weight * ||x||₂²
    """
    # First apply L2 proximal (closed form)
    x = x / (1 + 2 * l2_weight * step_size)
    # Then apply L1 proximal
    return l1_proximal(x, l1_weight * step_size)


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
        # Validate inputs
        if len(self.design.shape) != 2:
            raise ValueError(f"Design matrix must be 2D, got shape {self.design.shape}")
        if len(self.response.shape) != 1:
            raise ValueError(f"Response must be 1D, got shape {self.response.shape}")
        if self.design.shape[0] != self.response.shape[0]:
            raise ValueError(f"Design and response size mismatch: {self.design.shape[0]} vs {self.response.shape[0]}")

        self.n_samples, self.n_features = self.design.shape
        self.x = np.zeros(self.n_features)
        self.obj_values: List[float] = []
        self.history: List[np.ndarray] = []
        self.sparsity_history: List[int] = []

        # Pre-compute for efficiency
        self.AtA = self.design.T @ self.design
        self.Aty = self.design.T @ self.response

        # Lipschitz constant for step size selection
        self.lipschitz = np.linalg.norm(self.AtA, 2)
        if self.step_size > 2.0 / self.lipschitz:
            import warnings
            warnings.warn(f"Step size {self.step_size} may be too large. Recommended: {1.0/self.lipschitz:.4f}")

    def _objective(self, x: np.ndarray) -> float:
        """Compute the objective value 0.5||Ax − y||² + λ·||x||₁."""
        residual = self.design @ x - self.response
        return 0.5 * np.dot(residual, residual) + self.lam * np.sum(np.abs(x))

    def fit(self, use_acceleration: bool = True) -> Tuple[np.ndarray, int, List[float], Dict[str, Any]]:
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

        # For FISTA acceleration
        if use_acceleration:
            y = self.x.copy()
            t_prev = 1.0

        # Adaptive step size
        step_size = self.step_size
        best_obj = np.inf

        for iteration in range(self.max_iter):
            if use_acceleration:
                # FISTA update
                grad = self.AtA @ y - self.Aty  # More efficient using pre-computed matrices
                z = y - step_size * grad
                x_new = l1_proximal(z, self.lam * step_size)

                # Update acceleration parameter
                t_new = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
                y = x_new + ((t_prev - 1) / t_new) * (x_new - self.x)

                self.x = x_new
                t_prev = t_new
            else:
                # Standard proximal gradient
                grad = self.AtA @ self.x - self.Aty
                z = self.x - step_size * grad
                self.x = l1_proximal(z, self.lam * step_size)

            # Compute objective and sparsity
            obj = self._objective(self.x)
            sparsity = np.sum(np.abs(self.x) < 1e-10)

            self.obj_values.append(obj)
            self.history.append(self.x.copy())
            self.sparsity_history.append(sparsity)

            # Adaptive step size with backtracking
            if obj > best_obj * 1.01:  # Objective increased too much
                step_size *= 0.9
            elif obj < best_obj:
                best_obj = obj

            # Invoke callback if provided
            if self.callback is not None:
                self.callback(iteration, self.x, obj)

            # Enhanced stopping criteria
            diff = np.linalg.norm(self.x - prev_x)
            norm_x = np.linalg.norm(prev_x) + 1e-12
            rel_change = diff / norm_x

            # Also check objective decrease
            if len(self.obj_values) > 1:
                obj_decrease = abs(self.obj_values[-2] - self.obj_values[-1]) / (abs(self.obj_values[-2]) + 1e-12)
            else:
                obj_decrease = 1.0

            if rel_change < self.tol and obj_decrease < self.tol:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= self.patience:
                break

            prev_x = self.x.copy()

        # Compile statistics
        stats = {
            'final_sparsity': sparsity,
            'convergence_rate': np.mean(np.diff(self.obj_values[-10:])) if len(self.obj_values) > 1 else 0.0,
            'final_gradient_norm': np.linalg.norm(grad),
            'iterations': iteration + 1,
            'sparsity_pattern': np.where(np.abs(self.x) > 1e-10)[0].tolist()
        }

        return self.x, iteration, self.obj_values, stats


__all__ = [
    "ProximalEarlyStopping",
    "l1_proximal",
    "l2_proximal",
    "elastic_net_proximal",
]