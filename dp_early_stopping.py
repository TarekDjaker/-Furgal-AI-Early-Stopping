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
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, List
import numpy as np


class DPSGDEarlyStopping:
    """Differentially private SGD with early stopping.

    This is a simplified implementation of DP-SGD that automatically
    handles linear regression with L2 loss.

    Parameters
    ----------
    epsilon : float
        Total privacy budget (ε) for the training process.
    delta : float
        Privacy parameter δ. Should be small (e.g., 1e-5).
    noise_multiplier : float
        Noise multiplier for gradient perturbation. Higher values = more privacy.
    max_iterations : int
        Maximum number of training iterations.
    learning_rate : float
        Learning rate for gradient descent.
    patience : int
        Number of iterations without improvement before stopping.
    verbose : bool
        If True, prints training progress and privacy budget.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, noise_multiplier: float = 1.0,
                 max_iterations: int = 100, learning_rate: float = 0.01, patience: int = 10, 
                 verbose: bool = False):
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.patience = patience
        self.verbose = verbose
        
        # Calculate privacy budget per iteration
        self.eps_per_iter = epsilon / max_iterations
        
    def _compute_privacy_loss(self, iteration: int) -> float:
        """Compute accumulated privacy loss using simple composition."""
        return iteration * self.eps_per_iter
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train linear regression with DP-SGD and early stopping.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)  
            Training targets.
            
        Returns
        -------
        dict
            Training history with iterations and privacy budget.
        """
        n_samples, n_features = X.shape
        params = np.zeros(n_features)
        
        # Split data for validation (simple split)
        split_idx = int(0.8 * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        train_losses = []
        val_losses = []
        privacy_budgets = []
        
        best_val_loss = np.inf
        patience_counter = 0
        
        for iteration in range(self.max_iterations):
            # Compute gradient on training data
            predictions = X_train @ params
            residuals = predictions - y_train
            gradient = X_train.T @ residuals / len(X_train)
            
            # Add noise for differential privacy
            noise = np.random.normal(0, self.noise_multiplier, size=gradient.shape)
            noisy_gradient = gradient + noise
            
            # Update parameters
            params = params - self.learning_rate * noisy_gradient
            
            # Compute losses
            train_pred = X_train @ params
            train_loss = np.mean((train_pred - y_train) ** 2)
            
            val_pred = X_val @ params  
            val_loss = np.mean((val_pred - y_val) ** 2)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Update privacy budget
            privacy_budget = self._compute_privacy_loss(iteration + 1)
            privacy_budgets.append(privacy_budget)
            
            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at iteration {iteration + 1}")
                break
                
            # Check if privacy budget is exhausted
            if privacy_budget >= self.epsilon:
                if self.verbose:
                    print(f"Privacy budget exhausted at iteration {iteration + 1}")
                break
                
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, privacy_budget={privacy_budget:.4f}")
        
        return {
            'iterations': list(range(1, iteration + 2)),  # List of iteration numbers
            'privacy_budget': privacy_budgets,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_params': params,
            'total_epsilon_used': privacy_budgets[-1] if privacy_budgets else 0.0
        }


__all__ = ["DPSGDEarlyStopping"]