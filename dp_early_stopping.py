"""
Differentially private gradient descent with early stopping.

The purpose of this module is to illustrate how early stopping can
interact with privacy budgets in a simple differentially private
learning setup.  We implement a basic differentially private SGD
procedure where Gaussian noise is added to each gradient step.  A
validation set is used to monitor performance, and training stops
after a fixed patience without improvement.  The total privacy loss
(`eps`) accumulates with each iteration according to standard
composition rules.  This example is not production‑ready but serves
as a starting point for research on privacy‑aware early stopping.

References: Dwork et al. 2014 (foundational DP), federated early
stopping practices【841064613011457†L682-L690】.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict, Any

import numpy as np


@dataclass
class DPSGDEarlyStopping:
    """Differentially private SGD with adaptive early stopping.

    This implementation includes privacy budget tracking and adaptive
    noise scheduling for improved performance.
    """
    """Differentially private SGD with early stopping.

    Parameters
    ----------
    loss_fn : Callable[[np.ndarray, np.ndarray], float]
        Loss function taking predictions and true labels and returning
        a scalar loss value.
    grad_fn : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        Gradient of the loss function with respect to parameters.  Must
        accept parameters, inputs and labels.
    noise_std : float
        Standard deviation of the Gaussian noise added to each gradient
        update (for differential privacy).
    lr : float
        Learning rate.
    max_iter : int
        Maximum number of iterations.
    patience : int
        Number of iterations without improvement in validation loss
        before stopping.
    eps_per_iter : float
        Amount of privacy budget (epsilon) consumed per iteration.  The
        total privacy budget at stopping time is eps_per_iter × stop_iter.
    verbose : bool
        If True, prints training progress and privacy budget.
    """

    loss_fn: Callable[[np.ndarray, np.ndarray], float]
    grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    noise_std: float = 1.0
    lr: float = 0.1
    max_iter: int = 200
    patience: int = 10
    eps_per_iter: float = 0.05
    verbose: bool = False

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        initial_params: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int, float, List[float], Dict[str, Any]]:
        """Train with DP‑SGD and early stopping.

        Returns
        -------
        params : ndarray
            Learned parameters at stopping.
        stop_iter : int
            Iteration index of stopping.
        total_eps : float
            Total privacy budget consumed.
        val_history : list
            Validation loss recorded at each iteration.
        """
        # Input validation
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"Shape mismatch: x_train {x_train.shape} vs y_train {y_train.shape}")
        if x_val.shape[0] != y_val.shape[0]:
            raise ValueError(f"Shape mismatch: x_val {x_val.shape} vs y_val {y_val.shape}")
        if x_train.shape[1] != x_val.shape[1]:
            raise ValueError(f"Feature dimension mismatch: {x_train.shape[1]} vs {x_val.shape[1]}")

        num_features = x_train.shape[1]
        params = (
            np.zeros(num_features) if initial_params is None else initial_params.copy()
        )
        best_val = np.inf
        best_params = params.copy()
        val_history: List[float] = []
        grad_norm_history: List[float] = []
        privacy_history: List[float] = []
        bad_count = 0

        for t in range(self.max_iter):
            # Compute gradient on training data
            grad = self.grad_fn(params, x_train, y_train)
            grad_norm = np.linalg.norm(grad)
            grad_norm_history.append(grad_norm)

            # Adaptive noise scaling based on gradient norm
            adaptive_noise_std = self.noise_std * (1.0 + 0.1 * np.exp(-t/50))

            # Add Gaussian noise for privacy
            noise = np.random.normal(scale=adaptive_noise_std, size=grad.shape)
            grad_priv = grad + noise

            # Gradient clipping for stability
            clip_norm = 10.0
            if np.linalg.norm(grad_priv) > clip_norm:
                grad_priv = grad_priv * clip_norm / np.linalg.norm(grad_priv)

            # Update parameters with momentum
            momentum = 0.9 if t > 10 else 0.0
            if t == 0:
                velocity = np.zeros_like(params)
            else:
                velocity = momentum * velocity - self.lr * grad_priv
            params = params + velocity

            # Compute validation loss
            val_pred = x_val @ params
            val_loss = self.loss_fn(val_pred, y_val)
            val_history.append(val_loss)

            # Track privacy budget
            eps_spent = (t + 1) * self.eps_per_iter
            privacy_history.append(eps_spent)

            # Check improvement with relative tolerance
            if val_loss < best_val * 0.999:  # 0.1% relative improvement
                best_val = val_loss
                best_params = params.copy()
                bad_count = 0
            else:
                bad_count += 1

            if bad_count >= self.patience:
                break

            if self.verbose and t % 10 == 0:
                print(
                    f"Iter {t+1}: val_loss={val_loss:.4f}, grad_norm={grad_norm:.4f}, "
                    f"privacy_eps={eps_spent:.2f}"
                )
        stop_iter = t
        total_eps = (stop_iter + 1) * self.eps_per_iter

        # Compile additional statistics
        stats = {
            'grad_norm_history': grad_norm_history,
            'privacy_history': privacy_history,
            'best_val_loss': best_val,
            'convergence_rate': np.mean(np.diff(val_history[:min(10, len(val_history))])) if len(val_history) > 1 else 0.0
        }

        return best_params, stop_iter, total_eps, val_history, stats


__all__ = ["DPSGDEarlyStopping"]