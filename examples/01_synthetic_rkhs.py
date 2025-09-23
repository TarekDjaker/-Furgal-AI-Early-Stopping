"""
Example 01: Synthetic RKHS experiments
Grid search over sigma and kernel width parameters
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Implementations.rkhs_gradient_descent import RKHSGradientDescent


def generate_synthetic_data(n_samples, noise_level, seed=42):
    """Generate synthetic regression data"""
    np.random.seed(seed)

    # Non-linear function
    def f(x):
        return np.sin(3 * x) + 0.5 * np.cos(5 * x) + 0.2 * x

    X = np.sort(np.random.uniform(-2, 2, (n_samples, 1)), axis=0)
    y_true = f(X).ravel()
    y = y_true + noise_level * np.random.randn(n_samples)

    return X, y, y_true


def run_experiment_grid():
    """Run grid search over parameters"""

    # Parameters
    n_train = 300
    n_test = 100
    sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    kernel_widths = [0.05, 0.1, 0.2, 0.4, 0.8]
    max_iterations = 500

    results = []

    print("=" * 70)
    print("RKHS Synthetic Data Experiments - Grid Search")
    print("=" * 70)
    print(f"Training samples: {n_train}")
    print(f"Testing samples: {n_test}")
    print(f"Max iterations: {max_iterations}")
    print()

    for sigma in sigma_values:
        # Generate data with this noise level
        X_train, y_train, y_train_true = generate_synthetic_data(n_train, sigma)
        X_test, y_test, y_test_true = generate_synthetic_data(n_test, sigma, seed=100)

        print(f"\nNoise level σ = {sigma}")
        print("-" * 50)

        for kernel_width in kernel_widths:
            print(f"  Kernel width = {kernel_width:4.2f}: ", end="")

            # Run DP
            model_dp = RKHSGradientDescent(
                X_train, y_train,
                kernel_type='gaussian',
                kernel_width=kernel_width
            )

            start_time = time.time()
            result_dp = model_dp.fit(
                max_iterations=max_iterations,
                stopping_rule='DP',
                sigma=sigma,
                verbose=False
            )
            time_dp = time.time() - start_time

            y_pred_dp_train = model_dp.predict(X_train)
            y_pred_dp_test = model_dp.predict(X_test)
            mse_dp_train = np.mean((y_train - y_pred_dp_train) ** 2)
            mse_dp_test = np.mean((y_test_true - y_pred_dp_test) ** 2)

            # Run SDP
            model_sdp = RKHSGradientDescent(
                X_train, y_train,
                kernel_type='gaussian',
                kernel_width=kernel_width
            )

            start_time = time.time()
            result_sdp = model_sdp.fit(
                max_iterations=max_iterations,
                stopping_rule='SDP',
                sigma=sigma,
                T_smooth=10,
                verbose=False
            )
            time_sdp = time.time() - start_time

            y_pred_sdp_train = model_sdp.predict(X_train)
            y_pred_sdp_test = model_sdp.predict(X_test)
            mse_sdp_train = np.mean((y_train - y_pred_sdp_train) ** 2)
            mse_sdp_test = np.mean((y_test_true - y_pred_sdp_test) ** 2)

            print(f"DP τ={result_dp['tau']:3d} (MSE={mse_dp_test:.4f}), "
                  f"SDP τ={result_sdp['tau']:3d} (MSE={mse_sdp_test:.4f})")

            # Store results
            results.append({
                'sigma': sigma,
                'kernel_width': kernel_width,
                'method': 'DP',
                'tau': result_dp['tau'],
                'mse_train': mse_dp_train,
                'mse_test': mse_dp_test,
                'time': time_dp
            })

            results.append({
                'sigma': sigma,
                'kernel_width': kernel_width,
                'method': 'SDP',
                'tau': result_sdp['tau'],
                'mse_train': mse_sdp_train,
                'mse_test': mse_sdp_test,
                'time': time_sdp
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('synthetic_rkhs_results.csv', index=False)

    # Save JSON too
    with open('synthetic_rkhs_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'n_train': n_train,
                'n_test': n_test,
                'max_iterations': max_iterations,
                'sigma_values': sigma_values,
                'kernel_widths': kernel_widths
            },
            'results': results
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    # Compute summary
    for method in ['DP', 'SDP']:
        method_df = df[df['method'] == method]
        print(f"\n{method}:")
        print(f"  Average τ: {method_df['tau'].mean():.1f} ± {method_df['tau'].std():.1f}")
        print(f"  Average test MSE: {method_df['mse_test'].mean():.4f} ± {method_df['mse_test'].std():.4f}")
        print(f"  Average time: {method_df['time'].mean():.3f}s")

        # Best parameters
        best_idx = method_df['mse_test'].idxmin()
        best = method_df.loc[best_idx]
        print(f"  Best params: σ={best['sigma']}, width={best['kernel_width']}, "
              f"MSE={best['mse_test']:.4f}, τ={best['tau']}")

    print(f"\nResults saved to:")
    print(f"  - synthetic_rkhs_results.csv")
    print(f"  - synthetic_rkhs_results.json")

    return df


if __name__ == "__main__":
    df = run_experiment_grid()