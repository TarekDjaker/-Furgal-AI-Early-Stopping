"""
Fairness metrics and early stopping callbacks.

This module provides utility functions to compute simple fairness
metrics and a callback class that triggers early stopping when
fairness improvements stagnate.  The intent is to encourage students
to explore how regularisation via early stopping can be used to
mitigate disparities between sensitive groups【330857330321960†L29-L41】.  The
functions defined here are agnostic to the underlying model or
training framework; you simply supply predictions, true labels and
sensitive attributes.

The default fairness metric implemented is the absolute difference
between group‑wise error rates (demographic parity difference).  You
may extend this module with other measures such as equalised odds or
false positive rate differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import numpy as np


def group_error_rates(y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> Tuple[float, float]:
    """Compute per‑group error rates for binary classification with binary sensitive attribute.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : ndarray of shape (n_samples,)
        Predicted labels (0 or 1).
    sensitive_attr : ndarray of shape (n_samples,)
        Binary sensitive attribute values (0 or 1).

    Returns
    -------
    error_rate_0 : float
        Error rate for group 0.
    error_rate_1 : float
        Error rate for group 1.
    """
    # Error rate for group 0
    idx_0 = sensitive_attr == 0
    if idx_0.sum() == 0:
        error_rate_0 = 0.0
    else:
        error_rate_0 = float(np.mean(y_true[idx_0] != y_pred[idx_0]))
    
    # Error rate for group 1
    idx_1 = sensitive_attr == 1
    if idx_1.sum() == 0:
        error_rate_1 = 0.0
    else:
        error_rate_1 = float(np.mean(y_true[idx_1] != y_pred[idx_1]))
    
    return error_rate_0, error_rate_1


def demographic_parity_difference(y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
    """Compute the absolute difference of error rates between most and least favoured groups.

    In binary classification, demographic parity difference is given by
    ``|error_rate(group_0) − error_rate(group_1)|``.  Smaller values
    indicate fairer predictions.
    """
    error_rate_0, error_rate_1 = group_error_rates(y_true, y_pred, sensitive_attr)
    return abs(error_rate_0 - error_rate_1)


@dataclass
class FairnessEarlyStopping:
    """Early stopping based on fairness metrics.

    This callback tracks a fairness metric (by default, demographic
    parity difference) across epochs.  If the metric fails to improve
    (decrease) for a given number of epochs (`patience`), the callback
    signals that training should stop.

    Parameters
    ----------
    fairness_threshold : float, optional
        Threshold for fairness metric below which training is considered successful.
        Defaults to 0.1.
    max_iterations : int, optional
        Maximum number of iterations to train. Defaults to 100.
    metric_fn : Callable[[np.ndarray, np.ndarray, np.ndarray], float], optional
        Function used to compute the fairness metric.  Defaults to
        ``demographic_parity_difference``.
    patience : int, optional
        Number of consecutive epochs without improvement before
        triggering early stopping.  Defaults to 3.
    min_delta : float, optional
        Minimum decrease in the fairness metric to qualify as an
        improvement.  Defaults to 0.0.
    verbose : bool, optional
        If True, prints a message when early stopping is triggered.
    """

    fairness_threshold: float = 0.1
    max_iterations: int = 100
    metric_fn: Optional[callable] = demographic_parity_difference
    patience: int = 3
    min_delta: float = 0.0
    verbose: bool = False

    def __post_init__(self) -> None:
        self.best_metric: float = np.inf
        self.num_bad_epochs: int = 0
        self.stopped_epoch: Optional[int] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, sensitive_attr: np.ndarray) -> dict:
        """Simulate training with fairness constraints.
        
        Parameters
        ----------
        X : ndarray
            Training features.
        y : ndarray
            Training labels.
        sensitive_attr : ndarray
            Sensitive attribute values.
            
        Returns
        -------
        dict
            Training history with iterations and fairness violations.
        """
        # Simulate a simple training process
        fairness_violations = []
        
        for iteration in range(self.max_iterations):
            # Simulate predictions (simple random baseline that improves over time)
            np.random.seed(42 + iteration)
            improvement_factor = min(0.9, iteration / self.max_iterations)
            y_pred = np.random.binomial(1, 0.5 + 0.3 * improvement_factor * (y * 2 - 1), size=len(y))
            
            # Calculate fairness metric
            fairness_metric = self.metric_fn(y, y_pred, sensitive_attr)
            fairness_violations.append(fairness_metric)
            
            # Check if we should stop due to fairness threshold
            if fairness_metric <= self.fairness_threshold:
                if self.verbose:
                    print(f"Fairness threshold reached at iteration {iteration}")
                break
                
            # Check early stopping based on patience
            if self.step(iteration, y, y_pred, sensitive_attr):
                break
                
        return {
            'iterations': iteration + 1,
            'fairness_violations': fairness_violations,
            'stopped_epoch': self.stopped_epoch
        }

    def step(self, epoch: int, y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> bool:
        """Update the metric and decide whether to stop.

        Parameters
        ----------
        epoch : int
            Current epoch number (for logging).
        y_true : ndarray
            Ground truth labels.
        y_pred : ndarray
            Predicted labels.
        sensitive_attr : ndarray
            Sensitive attribute values.

        Returns
        -------
        bool
            True if training should be stopped, False otherwise.
        """
        if self.metric_fn is None:
            raise ValueError("metric_fn must be provided")
        metric_value = self.metric_fn(y_true, y_pred, sensitive_attr)
        if metric_value + self.min_delta < self.best_metric:
            self.best_metric = metric_value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                print(
                    f"FairnessEarlyStopping: stopped at epoch {epoch} with fairness metric {metric_value:.4f}"
                )
            return True
        return False


__all__ = [
    "group_error_rates",
    "demographic_parity_difference",
    "FairnessEarlyStopping",
]