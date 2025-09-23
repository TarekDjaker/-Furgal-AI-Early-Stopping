"""
Base class for early stopping methods
Provides common interface for all early stopping implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple
import numpy as np


class EarlyStoppingMethod(ABC):
    """
    Abstract base class for early stopping methods

    All early stopping implementations should inherit from this class
    and implement the required methods.
    """

    def __init__(self):
        """Initialize base early stopping method"""
        self.diagnostics: Dict[str, Any] = {}
        self.is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> 'EarlyStoppingMethod':
        """
        Fit the model with early stopping

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
        **kwargs
            Additional keyword arguments specific to the method

        Returns
        -------
        self : EarlyStoppingMethod
            Fitted model
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for new data

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray
            Predictions of shape (n_samples,)
        """
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information from the fitting process

        Returns
        -------
        diagnostics : Dict[str, Any]
            Dictionary containing diagnostic information such as:
            - tau: stopping time
            - n_iterations: number of iterations performed
            - final_residual: final residual norm
            - Additional method-specific diagnostics
        """
        return self.diagnostics

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the coefficient of determination R^2

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_features)
        y : np.ndarray
            True target values of shape (n_samples,)

        Returns
        -------
        score : float
            R^2 score
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute mean squared error

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_features)
        y : np.ndarray
            True target values of shape (n_samples,)

        Returns
        -------
        mse : float
            Mean squared error
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)