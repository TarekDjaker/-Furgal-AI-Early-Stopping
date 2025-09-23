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
to improve significantly.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, List, Dict, Any
import numpy as np


def l1_proximal(x: np.ndarray, threshold: float) -> np.ndarray:
    """Proximal operator for the ℓ₁ norm.

    This function implements the soft thresholding operator which is
    the proximal operator of the ℓ₁ norm.  For each element xᵢ, the
    result is:
        - max(0, xᵢ - threshold) if xᵢ >= 0
        - min(0, xᵢ + threshold) if xᵢ < 0

    Parameters
    ----------
    x : ndarray
        Input vector.
    threshold : float
        Threshold value.

    Returns
    -------
    ndarray
        Soft-thresholded vector.
    """
    return np.sign(x) * np.maximum(0, np.abs(x) - threshold)


class ProximalEarlyStopping:
    """Proximal gradient descent with early stopping for composite optimisation.

    This class solves problems of the form:
        minimise  f(x) + λ · φ(x)
    where f is differentiable (least squares) and φ is the ℓ₁ norm.

    Parameters
    ----------
    method : str, optional
        Optimization method. Supports 'ISTA' and 'FISTA'. Defaults to 'FISTA'.
    lambda_reg : float, optional
        Regularisation parameter λ for the ℓ₁ penalty. Defaults to 0.1.
    max_iterations : int, optional
        Maximum number of iterations. Defaults to 500.
    step_size : float, optional
        Step size for gradient descent. Defaults to 1.0.
    tol : float, optional
        Tolerance for the relative change in the parameter vector. Defaults to 1e-4.
    patience : int, optional
        Number of consecutive negligible improvements allowed before stopping. Defaults to 5.
    verbose : bool, optional
        If True, prints progress information. Defaults to False.
    """

    def __init__(self, method: str = 'FISTA', lambda_reg: float = 0.1, max_iterations: int = 500,
                 step_size: float = 1.0, tol: float = 1e-4, patience: int = 5, verbose: bool = False):
        self.method = method
        self.lambda_reg = lambda_reg
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.tol = tol
        self.patience = patience
        self.verbose = verbose
        
    def _objective(self, x: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the objective value 0.5||Xx - y||² + λ·||x||₁."""
        residual = X @ x - y
        return 0.5 * np.dot(residual, residual) + self.lambda_reg * np.sum(np.abs(x))

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run proximal gradient descent until the stopping criterion triggers.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design matrix.
        y : ndarray of shape (n_samples,)
            Response vector.

        Returns
        -------
        dict
            Results dictionary containing:
            - 'iterations': number of iterations run
            - 'stopped_at': iteration where algorithm stopped
            - 'objective_values': history of objective values
            - 'coefficients': final parameter vector
        """
        n_samples, n_features = X.shape
        x = np.zeros(n_features)
        x_prev = np.zeros(n_features)
        
        # Adaptive step size based on Lipschitz constant
        L = np.linalg.norm(X.T @ X, ord=2)
        if L > 0:
            step_size = min(self.step_size, 0.9 / L)
        else:
            step_size = self.step_size
        
        obj_values = []
        patience_counter = 0
        prev_obj = np.inf
        
        for iteration in range(self.max_iterations):
            # Store previous x for FISTA
            x_prev_iter = x.copy()
            
            # Compute gradient of the smooth part (least squares)
            if self.method == 'FISTA' and iteration > 0:
                # FISTA uses momentum
                t_k = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
                beta = (t_prev - 1) / t_k
                y_k = x + beta * (x - x_prev)
                grad = X.T @ (X @ y_k - y)
                z = y_k - step_size * grad
                t_prev = t_k
            else:
                # ISTA or first iteration of FISTA
                grad = X.T @ (X @ x - y)
                z = x - step_size * grad
                if self.method == 'FISTA':
                    t_prev = 1.0
            
            # Proximal step for ℓ₁ penalty
            x = l1_proximal(z, self.lambda_reg * step_size)
            
            # Update x_prev for next iteration
            x_prev = x_prev_iter
            
            # Compute objective and record
            obj = self._objective(x, X, y)
            obj_values.append(obj)
            
            # Check for numerical issues
            if not np.isfinite(obj):
                if self.verbose:
                    print(f"Numerical issues detected at iteration {iteration + 1}")
                break
            
            # Check convergence
            if iteration > 0:
                rel_change = abs(obj - prev_obj) / max(abs(prev_obj), 1e-10)
                if rel_change < self.tol:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Converged after {iteration + 1} iterations")
                    break
            
            prev_obj = obj
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: objective = {obj:.6f}")
        
        # Store theta for compatibility
        self.theta = x
        
        return {
            'iterations': iteration + 1,
            'stopped_at': iteration + 1,
            'objective_values': obj_values,
            'coefficients': x,
            'method': self.method
        }


__all__ = [
    "ProximalEarlyStopping",
    "l1_proximal",
]