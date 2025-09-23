"""
RKHS Light Experiments - Quick validation with smaller parameters
"""

import numpy as np
import json
import time
from datetime import datetime
from Implementations.rkhs_gradient_descent import RKHSGradientDescent

def main():
    """Run light RKHS experiments"""

    print("=" * 60)
    print("RKHS Gradient Descent - Light Experiments")
    print("=" * 60)

    # Set seed
    np.random.seed(42)

    # Light parameters (fewer combinations)
    n = 200
    sigmas = [0.2, 0.4]
    kernel_width = 0.2

    results = {
        'timestamp': datetime.now().isoformat(),
        'seed': 42,
        'n': n,
        'kernel': 'gaussian',
        'experiments': []
    }

    # Generate data
    def f_true(x):
        return np.sin(3 * x) + 0.5 * np.cos(5 * x)

    X_train = np.sort(np.random.uniform(-2, 2, (n, 1)), axis=0)
    Y_train_true = f_true(X_train).ravel()
    X_test = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
    Y_test_true = f_true(X_test).ravel()

    for sigma in sigmas:
        Y_train = Y_train_true + sigma * np.random.randn(n)

        print(f"\nExperiment with σ = {sigma}, kernel_width = {kernel_width}")
        print("-" * 40)

        # DP
        model_dp = RKHSGradientDescent(
            X_train, Y_train,
            kernel_type='gaussian',
            kernel_width=kernel_width
        )

        start = time.time()
        result_dp = model_dp.fit(
            max_iterations=300,
            stopping_rule='DP',
            sigma=sigma,
            X_test=X_test,
            Y_test=Y_test_true,
            verbose=False
        )
        time_dp = time.time() - start

        Y_pred_dp = model_dp.predict(X_test)
        train_mse_dp = np.mean((Y_train - model_dp.predict(X_train)) ** 2)
        test_mse_dp = np.mean((Y_test_true - Y_pred_dp) ** 2)

        print(f"DP:  τ = {result_dp['tau']:3d}, train MSE = {train_mse_dp:.4f}, test MSE = {test_mse_dp:.4f}, time = {time_dp:.2f}s")

        # SDP
        model_sdp = RKHSGradientDescent(
            X_train, Y_train,
            kernel_type='gaussian',
            kernel_width=kernel_width
        )

        start = time.time()
        result_sdp = model_sdp.fit(
            max_iterations=300,
            stopping_rule='SDP',
            sigma=sigma,
            T_smooth=10,
            X_test=X_test,
            Y_test=Y_test_true,
            verbose=False
        )
        time_sdp = time.time() - start

        Y_pred_sdp = model_sdp.predict(X_test)
        train_mse_sdp = np.mean((Y_train - model_sdp.predict(X_train)) ** 2)
        test_mse_sdp = np.mean((Y_test_true - Y_pred_sdp) ** 2)

        print(f"SDP: τ = {result_sdp['tau']:3d}, train MSE = {train_mse_sdp:.4f}, test MSE = {test_mse_sdp:.4f}, time = {time_sdp:.2f}s")

        # Store results
        results['experiments'].append({
            'sigma': sigma,
            'kernel_width': kernel_width,
            'DP': {
                'tau': int(result_dp['tau']),
                'train_mse': float(train_mse_dp),
                'test_mse': float(test_mse_dp),
                'time': float(time_dp)
            },
            'SDP': {
                'tau': int(result_sdp['tau']),
                'train_mse': float(train_mse_sdp),
                'test_mse': float(test_mse_sdp),
                'time': float(time_sdp)
            }
        })

    # Save results
    with open('rkhs_light_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Results saved to: rkhs_light_results.json")
    print("=" * 60)

    # Print summary
    for exp in results['experiments']:
        print(f"σ={exp['sigma']}: DP τ={exp['DP']['tau']}, SDP τ={exp['SDP']['tau']}")

    return results

if __name__ == "__main__":
    results = main()