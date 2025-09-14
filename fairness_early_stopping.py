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


def group_error_rates(y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    unique_groups = np.unique(sensitive_attr)
    error_rates = []
    for g in unique_groups:
        idx = sensitive_attr == g
        if idx.sum() == 0:
            error_rates.append(0.0)
        else:
            error_rates.append(np.mean(y_true[idx] != y_pred[idx]))
    return unique_groups, np.array(error_rates)


def demographic_parity_difference(y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
    """Compute the absolute difference of error rates between most and least favoured groups.

    In binary classification, demographic parity difference is given by
    ``max_g error_rate(g) − min_g error_rate(g)``.  Smaller values
    indicate fairer predictions.
    """
    _, error_rates = group_error_rates(y_true, y_pred, sensitive_attr)
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

    metric_fn: Optional[callable] = demographic_parity_difference
    patience: int = 3
    min_delta: float = 0.0
    verbose: bool = False

    def __post_init__(self) -> None:
        self.best_metric: float = np.inf
        self.num_bad_epochs: int = 0
        self.stopped_epoch: Optional[int] = None

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