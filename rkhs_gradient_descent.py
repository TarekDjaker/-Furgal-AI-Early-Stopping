# -*- coding: utf-8 -*-
"""
Reproducing Kernel Hilbert Space (RKHS) Gradient Descent Implementation
with Early Stopping Rules (DP and SDP)

Based on spectral filtering methods and gradient descent in RKHS
"""

import warnings
from typing import Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


class RKHSGradientDescent:
    """
    Gradient Descent in Reproducing Kernel Hilbert Space
    with Discrepancy Principle (DP) and Smoothed Discrepancy Principle (SDP) stopping rules
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel_type: str = "gaussian",
        kernel_width: float = 0.2,
        learning_rate: Optional[float] = None,
        regularization: float = 0.0,
    ):
        """
        Initialize RKHS Gradient Descent

        Parameters:
        -----------
        X : np.ndarray
            Training data of shape (n, d)
        Y : np.ndarray
            Target values of shape (n,) or (n, 1)
        kernel_type : str
            Type of kernel ('gaussian', 'polynomial', 'laplacian')
        kernel_width : float
            Width parameter for the kernel
        learning_rate : float, optional
            Learning rate (if None, computed automatically)
        regularization : float
            Regularization parameter (default 0)
        """
        self.X = X
        self.n = X.shape[0]

        # Ensure Y is column vector
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.Y = Y

        self.kernel_type = kernel_type
        self.kernel_width = kernel_width
        self.regularization = regularization

        # Compute Gram matrix
        self.K = self._compute_kernel_matrix(X, X)

        # Add regularization to diagonal for stability
        self.K_reg = self.K + regularization * np.eye(self.n)

        # Eigendecomposition for spectral methods
        self._compute_eigendecomposition()

        # Set learning rate
        if learning_rate is None:
            # Optimal learning rate based on largest eigenvalue
            self.eta = 0.5 / self.lambda_max
        else:
            self.eta = learning_rate

        # Initialize coefficients
        self.alpha = np.zeros((self.n, 1))

        # Storage for iteration history
        self.iteration_history = {
            "alpha": [],
            "residual_train": [],
            "residual_test": [],
            "loss": [],
        }

    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between X1 and X2"""
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))

        if self.kernel_type == "gaussian":
            for i in range(n1):
                for j in range(n2):
                    diff = X1[i] - X2[j]
                    K[i, j] = np.exp(-np.sum(diff**2) / (2 * self.kernel_width**2))

        elif self.kernel_type == "polynomial":
            degree = 3
            K = (1 + X1 @ X2.T) ** degree

        elif self.kernel_type == "laplacian":
            for i in range(n1):
                for j in range(n2):
                    diff = X1[i] - X2[j]
                    K[i, j] = np.exp(-np.sum(np.abs(diff)) / self.kernel_width)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        return K

    def _compute_eigendecomposition(self):
        """Compute eigendecomposition of kernel matrix"""
        try:
            vals, vecs = np.linalg.eigh(self.K_reg)
        except np.linalg.LinAlgError:
            warnings.warn("Eigendecomposition failed. Adding small regularization.")
            vals, vecs = np.linalg.eigh(self.K_reg + 1e-10 * np.eye(self.n))

        # Sort in descending order
        idx = np.argsort(vals)[::-1]
        self.lambdas = np.maximum(vals[idx], 0)
        self.Q = vecs[:, idx]
        self.lambda_max = max(self.lambdas[0], 1e-14)

        # Precompute Q^T Y for efficiency
        self.QTY = self.Q.T @ self.Y

    def gradient_step(self, t: int) -> np.ndarray:
        """
        Perform one gradient descent step

        Uses the spectral representation for efficient computation
        """
        # Spectral filter for gradient descent at iteration t
        g_vals = []
        for lam in self.lambdas:
            if lam < 1e-14:
                g_vals.append(self.eta * t)
            else:
                term = 1.0 - self.eta * lam
                term = np.clip(term, -1.0, 1.0)
                g_vals.append((1.0 / lam) * (1.0 - term**t))

        G = np.diag(g_vals)
        self.alpha = self.Q @ G @ self.QTY
        return self.alpha

    def landweber_iteration(self, t: int) -> np.ndarray:
        """
        Landweber iteration (equivalent to gradient descent in RKHS)
        """
        return self.gradient_step(t)

    def predict(self, X_new: np.ndarray, iteration: Optional[int] = None) -> np.ndarray:
        """
        Make predictions for new data

        Parameters:
        -----------
        X_new : np.ndarray
            Test data of shape (m, d)
        iteration : int, optional
            Iteration to use for prediction (if None, use current)
        """
        if iteration is not None:
            alpha = self.gradient_step(iteration)
        else:
            alpha = self.alpha

        K_new = self._compute_kernel_matrix(X_new, self.X)
        return (K_new @ alpha).ravel()

    def residual_norm_train(self, t: int) -> float:
        """
        Compute training residual norm ||Y - K*alpha_t||_2
        """
        alpha_t = self.gradient_step(t)
        residual = self.Y - self.K @ alpha_t
        return float(np.linalg.norm(residual))

    def residual_norm_empirical(self, t: int) -> float:
        """
        Compute empirical (normalized) residual norm ||Y - K*alpha_t||_n
        """
        return self.residual_norm_train(t) / np.sqrt(self.n)

    def find_tau_DP(self, sigma: float, max_t: int = 500) -> int:
        """
        Find stopping time using Discrepancy Principle (DP)

        Stop at first t where ||Y - K*alpha_t||_n <= sigma
        """
        threshold = sigma

        for t in range(1, max_t + 1):
            res_norm = self.residual_norm_empirical(t)
            if res_norm <= threshold:
                return t
        return max_t

    def find_tau_SDP(self, sigma: float, T_smooth: int = 10, max_t: int = 500) -> Tuple[int, float]:
        """
        Find stopping time using Smoothed Discrepancy Principle (SDP)

        Parameters:
        -----------
        sigma : float
            Noise level (standard deviation)
        T_smooth : int
            Smoothing parameter (number of iterations for smoothing filter)
        max_t : int
            Maximum number of iterations
        """
        # Compute smoothing filter g_tilde
        g_tilde_vals = []
        for lam in self.lambdas:
            if lam < 1e-14:
                g_tilde_vals.append(self.eta * T_smooth)
            else:
                term = 1.0 - self.eta * lam
                term = np.clip(term, -1.0, 1.0)
                g_tilde_vals.append((1.0 / lam) * (1.0 - term**T_smooth))

        # Compute effective dimension
        N_tilde = np.sum(self.lambdas * np.array(g_tilde_vals))

        # Compute smoothing matrix L_n
        D_vals = np.sqrt(np.maximum(self.lambdas * np.array(g_tilde_vals), 0))
        D_Ln = np.diag(D_vals)
        L_n = self.Q @ D_Ln @ self.Q.T

        # SDP threshold
        threshold = sigma**2 * N_tilde / self.n

        for t in range(1, max_t + 1):
            # Compute smoothed residual
            alpha_t = self.gradient_step(t)
            residual = self.Y - self.K @ alpha_t
            smoothed_residual = L_n @ residual
            smoothed_norm_sq = np.sum(smoothed_residual**2) / self.n

            if smoothed_norm_sq <= threshold:
                return t, N_tilde

        return max_t, N_tilde

    def fit(
        self,
        max_iterations: int = 500,
        stopping_rule: str = "DP",
        sigma: Optional[float] = None,
        T_smooth: int = 10,
        X_test: Optional[np.ndarray] = None,
        Y_test: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Fit the model using gradient descent with early stopping

        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations
        stopping_rule : str
            'DP' for Discrepancy Principle, 'SDP' for Smoothed DP, 'none' for no early stopping
        sigma : float, optional
            Noise level for stopping rules
        T_smooth : int
            Smoothing parameter for SDP
        X_test, Y_test : np.ndarray, optional
            Test data for monitoring
        verbose : bool
            Print progress
        """
        # Estimate noise level if not provided
        if sigma is None and stopping_rule in ["DP", "SDP"]:
            # Simple noise estimation from training residuals
            initial_residual = np.linalg.norm(self.Y) / np.sqrt(self.n)
            sigma = initial_residual * 0.1  # Heuristic
            if verbose:
                print(f"Estimated noise level sigma = {sigma:.4f}")

        # Find stopping time
        if stopping_rule == "DP":
            tau = self.find_tau_DP(sigma, max_iterations)
            if verbose:
                print(f"DP stopping at iteration {tau}")
        elif stopping_rule == "SDP":
            tau, N_tilde = self.find_tau_SDP(sigma, T_smooth, max_iterations)
            if verbose:
                print(f"SDP stopping at iteration {tau} (N_tilde = {N_tilde:.2f})")
        else:
            tau = max_iterations

        # Training loop
        train_residuals = []
        test_residuals = []

        for t in range(1, tau + 1):
            # Perform gradient step
            self.gradient_step(t)

            # Compute residuals
            train_res = self.residual_norm_empirical(t)
            train_residuals.append(train_res)

            if X_test is not None and Y_test is not None:
                Y_pred = self.predict(X_test)
                test_res = np.linalg.norm(Y_test.ravel() - Y_pred)
                test_residuals.append(test_res)

            if verbose and (t % 50 == 0 or t == 1 or t == tau):
                msg = f"Iteration {t}: Train residual = {train_res:.6f}"
                if test_residuals:
                    msg += f", Test residual = {test_residuals[-1]:.6f}"
                print(msg)

        return {
            "tau": tau,
            "train_residuals": train_residuals,
            "test_residuals": test_residuals,
            "sigma": sigma,
            "stopping_rule": stopping_rule,
        }


def example_usage():
    """Example usage of RKHS Gradient Descent"""
    np.random.seed(42)

    # Parameters
    n = 100
    sigma_true = 0.3
    kernel_width = 0.2

    # Generate synthetic data
    def f_true(x):
        return np.sin(x)

    X_train = np.sort(np.random.uniform(-np.pi, np.pi, (n, 1)), axis=0)
    Y_train_true = f_true(X_train)
    Y_train = Y_train_true + sigma_true * np.random.randn(n, 1)

    X_test = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 200).reshape(-1, 1)
    Y_test_true = f_true(X_test)

    # Create and fit model with DP
    print("=" * 60)
    print("RKHS Gradient Descent with Discrepancy Principle (DP)")
    print("=" * 60)

    model_dp = RKHSGradientDescent(
        X_train, Y_train, kernel_type="gaussian", kernel_width=kernel_width
    )

    result_dp = model_dp.fit(
        max_iterations=500,
        stopping_rule="DP",
        sigma=sigma_true,
        X_test=X_test,
        Y_test=Y_test_true,
        verbose=True,
    )

    # Create and fit model with SDP
    print("\n" + "=" * 60)
    print("RKHS Gradient Descent with Smoothed DP (SDP)")
    print("=" * 60)

    model_sdp = RKHSGradientDescent(
        X_train, Y_train, kernel_type="gaussian", kernel_width=kernel_width
    )

    result_sdp = model_sdp.fit(
        max_iterations=500,
        stopping_rule="SDP",
        sigma=sigma_true,
        T_smooth=10,
        X_test=X_test,
        Y_test=Y_test_true,
        verbose=True,
    )

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Training curves
    ax = axes[0, 0]
    iterations_dp = range(1, len(result_dp["train_residuals"]) + 1)
    iterations_sdp = range(1, len(result_sdp["train_residuals"]) + 1)

    ax.semilogy(iterations_dp, result_dp["train_residuals"], "b-", label="DP Train Residual")
    ax.semilogy(iterations_sdp, result_sdp["train_residuals"], "r-", label="SDP Train Residual")
    ax.axhline(y=sigma_true, color="k", linestyle=":", label=f"True σ = {sigma_true}")
    ax.axvline(
        x=result_dp["tau"],
        color="b",
        linestyle="--",
        alpha=0.5,
        label=f"DP tau = {result_dp['tau']}",
    )
    ax.axvline(
        x=result_sdp["tau"],
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"SDP tau = {result_sdp['tau']}",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual Norm")
    ax.set_title("Training Residuals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Test curves
    if result_dp["test_residuals"]:
        ax = axes[0, 1]
        ax.semilogy(iterations_dp, result_dp["test_residuals"], "b-", label="DP Test Residual")
        ax.semilogy(iterations_sdp, result_sdp["test_residuals"], "r-", label="SDP Test Residual")
        ax.axvline(x=result_dp["tau"], color="b", linestyle="--", alpha=0.5)
        ax.axvline(x=result_sdp["tau"], color="r", linestyle="--", alpha=0.5)

        # Mark minima
        min_dp = np.argmin(result_dp["test_residuals"]) + 1
        min_sdp = np.argmin(result_sdp["test_residuals"]) + 1
        ax.axvline(x=min_dp, color="b", linestyle=":", alpha=0.5, label=f"DP Oracle = {min_dp}")
        ax.axvline(x=min_sdp, color="r", linestyle=":", alpha=0.5, label=f"SDP Oracle = {min_sdp}")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Test Residual")
        ax.set_title("Test Residuals")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3: DP Predictions
    ax = axes[1, 0]
    Y_pred_dp = model_dp.predict(X_test)
    ax.scatter(X_train, Y_train, alpha=0.5, s=20, label="Training Data")
    ax.plot(X_test, Y_test_true, "k-", linewidth=2, label="True Function")
    ax.plot(X_test, Y_pred_dp, "b--", linewidth=2, label=f'DP Prediction (tau={result_dp["tau"]})')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("DP Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: SDP Predictions
    ax = axes[1, 1]
    Y_pred_sdp = model_sdp.predict(X_test)
    ax.scatter(X_train, Y_train, alpha=0.5, s=20, label="Training Data")
    ax.plot(X_test, Y_test_true, "k-", linewidth=2, label="True Function")
    ax.plot(
        X_test, Y_pred_sdp, "r--", linewidth=2, label=f'SDP Prediction (tau={result_sdp["tau"]})'
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("SDP Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("RKHS Gradient Descent with Early Stopping", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    dp_train_final = result_dp["train_residuals"][-1]
    sdp_train_final = result_sdp["train_residuals"][-1]

    print(f"DP: tau = {result_dp['tau']}, Final train residual = {dp_train_final:.6f}")
    print(f"SDP: tau = {result_sdp['tau']}, Final train residual = {sdp_train_final:.6f}")

    if result_dp["test_residuals"]:
        dp_test_final = result_dp["test_residuals"][-1]
        sdp_test_final = result_sdp["test_residuals"][-1]
        dp_test_min = min(result_dp["test_residuals"])
        sdp_test_min = min(result_sdp["test_residuals"])

        print(f"\nDP: Final test residual = {dp_test_final:.6f}, Min test = {dp_test_min:.6f}")
        print(f"SDP: Final test residual = {sdp_test_final:.6f}, Min test = {sdp_test_min:.6f}")


if __name__ == "__main__":
    example_usage()
