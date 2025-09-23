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
from typing import Callable, List, Tuple, Optional

import numpy as np


def group_error_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    metric: str = 'error_rate'
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per‑group error rates for binary classification.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : ndarray of shape (n_samples,)
        Predicted labels (0 or 1).
    sensitive_attr : ndarray of shape (n_samples,)
        Sensitive attribute values (e.g. gender, race).

    Returns
    -------
    unique_groups : ndarray
        The unique values in ``sensitive_attr``.
    error_rates : ndarray
        Error rate for each group in the same order as ``unique_groups``.
    """
    # Input validation
    if len(y_true) != len(y_pred) or len(y_true) != len(sensitive_attr):
        raise ValueError("Input arrays must have the same length")

    unique_groups = np.unique(sensitive_attr)
    metrics = []

    for g in unique_groups:
        idx = sensitive_attr == g
        n_samples = idx.sum()

        if n_samples == 0:
            metrics.append(0.0)
            continue

        if metric == 'error_rate':
            value = np.mean(y_true[idx] != y_pred[idx])
        elif metric == 'accuracy':
            value = np.mean(y_true[idx] == y_pred[idx])
        elif metric == 'positive_rate':
            value = np.mean(y_pred[idx] == 1)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        metrics.append(value)

    return unique_groups, np.array(metrics)


def demographic_parity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    weighted: bool = False
) -> float:
    """Compute the absolute difference of error rates between most and least favoured groups.

    In binary classification, demographic parity difference is given by
    ``max_g error_rate(g) − min_g error_rate(g)``.  Smaller values
    indicate fairer predictions.
    """
    _, error_rates = group_error_rates(y_true, y_pred, sensitive_attr)

    if weighted:
        # Weight by group sizes
        unique_groups = np.unique(sensitive_attr)
        weights = [np.sum(sensitive_attr == g) for g in unique_groups]
        weights = np.array(weights) / np.sum(weights)
        weighted_mean = np.sum(error_rates * weights)
        return float(np.sum(weights * np.abs(error_rates - weighted_mean)))

    return float(np.max(error_rates) - np.min(error_rates))


@dataclass
class FairnessEarlyStopping:
    """Early stopping based on fairness metrics.

    This callback tracks a fairness metric (by default, demographic
    parity difference) across epochs.  If the metric fails to improve
    (decrease) for a given number of epochs (`patience`), the callback
    signals that training should stop.

    Parameters
    ----------
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

    metric_fn: Optional[Callable] = None
    patience: int = 3
    min_delta: float = 0.0
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.metric_fn is None:
            self.metric_fn = demographic_parity_difference

        self.best_metric: float = np.inf
        self.num_bad_epochs: int = 0
        self.stopped_epoch: Optional[int] = None
        self.metric_history: List[float] = []
        self.improvement_history: List[bool] = []

    def step(self, epoch: int, y_true: np.ndarray, y_pred: np.ndarray,
             sensitive_attr: np.ndarray) -> bool:
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
        metric_value = self.metric_fn(y_true, y_pred, sensitive_attr)
        self.metric_history.append(metric_value)

        # Check for improvement with relative tolerance
        improved = metric_value + self.min_delta < self.best_metric
        self.improvement_history.append(improved)

        if improved:
            self.best_metric = metric_value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # Early stopping with additional criteria
        should_stop = False
        if self.num_bad_epochs >= self.patience:
            should_stop = True

        # Also stop if metric is essentially zero
        if metric_value < 1e-6:
            should_stop = True
            if self.verbose:
                print(f"FairnessEarlyStopping: achieved near-perfect fairness at epoch {epoch}")

        if should_stop:
            self.stopped_epoch = epoch
            if self.verbose:
                improvement_rate = (
                    sum(self.improvement_history[-10:])
                    / min(10, len(self.improvement_history))
                )
                print(
                    f"FairnessEarlyStopping: stopped at epoch {epoch} "
                    f"with metric {metric_value:.4f}, "
                    f"best: {self.best_metric:.4f}, recent improvement "
                    f"rate: {improvement_rate:.2%}"
                )

        return should_stop


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray
) -> float:
    """Compute equalized odds difference.

    Returns the maximum difference in TPR or FPR across groups.
    """
    unique_groups = np.unique(sensitive_attr)
    tpr_values = []
    fpr_values = []

    for g in unique_groups:
        idx = sensitive_attr == g
        y_true_g = y_true[idx]
        y_pred_g = y_pred[idx]

        # True Positive Rate
        positives = y_true_g == 1
        if positives.sum() > 0:
            tpr = np.mean(y_pred_g[positives] == 1)
        else:
            tpr = 0.0
        tpr_values.append(tpr)

        # False Positive Rate
        negatives = y_true_g == 0
        if negatives.sum() > 0:
            fpr = np.mean(y_pred_g[negatives] == 1)
        else:
            fpr = 0.0
        fpr_values.append(fpr)

    tpr_diff = np.max(tpr_values) - np.min(tpr_values)
    fpr_diff = np.max(fpr_values) - np.min(fpr_values)

    return float(max(tpr_diff, fpr_diff))


__all__ = [
    "group_error_rates",
    "demographic_parity_difference",
    "equalized_odds_difference",
    "FairnessEarlyStopping",
]
