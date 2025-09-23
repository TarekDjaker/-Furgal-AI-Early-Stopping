"""
Quick Start RKHS Gradient Descent Example
Run a minimal example with DP and SDP stopping rules
"""

import numpy as np
import json
from datetime import datetime
from Implementations.rkhs_gradient_descent import RKHSGradientDescent

def run_quick_start():
    """Run Quick Start example from README with both DP and SDP"""

    # Set seed for reproducibility
    np.random.seed(42)

    # Parameters
    n = 200  # Sample size
    d = 1    # Dimension
    sigma_true = 0.3  # True noise level
    kernel_width = 0.2

    # Generate synthetic data
    def f_true(x):
        return np.sin(3 * x) + 0.5 * np.cos(5 * x)

    X_train = np.sort(np.random.uniform(-2, 2, (n, d)), axis=0)
    Y_train_true = f_true(X_train).ravel()
    Y_train = Y_train_true + sigma_true * np.random.randn(n)

    # Test data
    X_test = np.linspace(-2.5, 2.5, 100).reshape(-1, d)
    Y_test_true = f_true(X_test).ravel()

    print("=" * 60)
    print("RKHS Gradient Descent - Quick Start")
    print("=" * 60)
    print(f"Sample size: n = {n}")
    print(f"True noise level: σ = {sigma_true}")
    print(f"Kernel: Gaussian (RBF) with width = {kernel_width}")

    results = {
        'timestamp': datetime.now().isoformat(),
        'seed': 42,
        'n': n,
        'sigma_true': sigma_true,
        'kernel': 'gaussian',
        'kernel_width': kernel_width,
        'experiments': {}
    }

    # Run with DP stopping rule
    print("\n" + "-" * 60)
    print("Running with Discrepancy Principle (DP)")
    print("-" * 60)

    model_dp = RKHSGradientDescent(
        X_train, Y_train,
        kernel_type='gaussian',
        kernel_width=kernel_width
    )

    result_dp = model_dp.fit(
        max_iterations=500,
        stopping_rule='DP',
        sigma=sigma_true,
        X_test=X_test,
        Y_test=Y_test_true,
        verbose=True
    )

    # Compute MSE
    Y_pred_dp = model_dp.predict(X_test)
    train_mse_dp = np.mean((Y_train - model_dp.predict(X_train)) ** 2)
    test_mse_dp = np.mean((Y_test_true - Y_pred_dp) ** 2)

    results['experiments']['DP'] = {
        'tau': int(result_dp['tau']),
        'train_mse': float(train_mse_dp),
        'test_mse': float(test_mse_dp),
        'final_train_residual': float(result_dp['train_residuals'][-1])
    }

    print(f"\nDP Results:")
    print(f"  Stopping time (τ): {result_dp['tau']}")
    print(f"  Train MSE: {train_mse_dp:.6f}")
    print(f"  Test MSE: {test_mse_dp:.6f}")

    # Run with SDP stopping rule
    print("\n" + "-" * 60)
    print("Running with Smoothed Discrepancy Principle (SDP)")
    print("-" * 60)

    model_sdp = RKHSGradientDescent(
        X_train, Y_train,
        kernel_type='gaussian',
        kernel_width=kernel_width
    )

    result_sdp = model_sdp.fit(
        max_iterations=500,
        stopping_rule='SDP',
        sigma=sigma_true,
        T_smooth=10,
        X_test=X_test,
        Y_test=Y_test_true,
        verbose=True
    )

    # Compute MSE
    Y_pred_sdp = model_sdp.predict(X_test)
    train_mse_sdp = np.mean((Y_train - model_sdp.predict(X_train)) ** 2)
    test_mse_sdp = np.mean((Y_test_true - Y_pred_sdp) ** 2)

    results['experiments']['SDP'] = {
        'tau': int(result_sdp['tau']),
        'train_mse': float(train_mse_sdp),
        'test_mse': float(test_mse_sdp),
        'final_train_residual': float(result_sdp['train_residuals'][-1]),
        'T_smooth': 10
    }

    print(f"\nSDP Results:")
    print(f"  Stopping time (τ): {result_sdp['tau']}")
    print(f"  Train MSE: {train_mse_sdp:.6f}")
    print(f"  Test MSE: {test_mse_sdp:.6f}")

    # Save results to JSON
    results_file = 'hello_world_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"DP stopped at iteration {result_dp['tau']} with test MSE = {test_mse_dp:.6f}")
    print(f"SDP stopped at iteration {result_sdp['tau']} with test MSE = {test_mse_sdp:.6f}")
    print(f"\nResults saved to: {results_file}")

    return results

if __name__ == "__main__":
    results = run_quick_start()