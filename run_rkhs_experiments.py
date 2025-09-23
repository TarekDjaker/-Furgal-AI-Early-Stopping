"""
RKHS Experiments with DP and SDP
Run experiments with different kernel widths and noise levels
"""

import numpy as np
import json
import time
from datetime import datetime
from Implementations.rkhs_gradient_descent import RKHSGradientDescent

def run_experiment(X_train, Y_train, X_test, Y_test_true, kernel_width, sigma, stopping_rule, verbose=False):
    """Run a single experiment with given parameters"""

    model = RKHSGradientDescent(
        X_train, Y_train,
        kernel_type='gaussian',
        kernel_width=kernel_width
    )

    start_time = time.time()

    if stopping_rule == 'SDP':
        result = model.fit(
            max_iterations=500,
            stopping_rule=stopping_rule,
            sigma=sigma,
            T_smooth=10,
            X_test=X_test,
            Y_test=Y_test_true,
            verbose=verbose
        )
    else:
        result = model.fit(
            max_iterations=500,
            stopping_rule=stopping_rule,
            sigma=sigma,
            X_test=X_test,
            Y_test=Y_test_true,
            verbose=verbose
        )

    elapsed_time = time.time() - start_time

    # Compute metrics
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    train_mse = np.mean((Y_train - Y_pred_train) ** 2)
    test_mse = np.mean((Y_test_true - Y_pred_test) ** 2)

    return {
        'tau': int(result['tau']),
        'train_mse': float(train_mse),
        'test_mse': float(test_mse),
        'final_train_residual': float(result['train_residuals'][-1]),
        'time_seconds': float(elapsed_time),
        'n_iterations': len(result['train_residuals'])
    }

def main():
    """Run comprehensive RKHS experiments"""

    print("=" * 60)
    print("RKHS Gradient Descent - Comprehensive Experiments")
    print("=" * 60)

    # Set seed for reproducibility
    np.random.seed(42)

    # Parameters to test
    n_values = [200, 500]
    sigma_values = [0.1, 0.3, 0.5]
    kernel_widths = [0.1, 0.2, 0.5]

    results = {
        'timestamp': datetime.now().isoformat(),
        'seed': 42,
        'experiments': []
    }

    for n in n_values:
        print(f"\nExperiments with n = {n}")
        print("-" * 40)

        # Generate data once for each n
        def f_true(x):
            return np.sin(3 * x) + 0.5 * np.cos(5 * x)

        X_train = np.sort(np.random.uniform(-2, 2, (n, 1)), axis=0)
        Y_train_true = f_true(X_train).ravel()

        X_test = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
        Y_test_true = f_true(X_test).ravel()

        for sigma in sigma_values:
            # Add noise to training data
            Y_train = Y_train_true + sigma * np.random.randn(n)

            for kernel_width in kernel_widths:
                print(f"\n  σ = {sigma}, kernel_width = {kernel_width}")

                # Run DP experiment
                print("    Running DP...", end=" ")
                dp_result = run_experiment(
                    X_train, Y_train, X_test, Y_test_true,
                    kernel_width, sigma, 'DP', verbose=False
                )
                print(f"τ = {dp_result['tau']}, test MSE = {dp_result['test_mse']:.4f}")

                # Run SDP experiment
                print("    Running SDP...", end=" ")
                sdp_result = run_experiment(
                    X_train, Y_train, X_test, Y_test_true,
                    kernel_width, sigma, 'SDP', verbose=False
                )
                print(f"τ = {sdp_result['tau']}, test MSE = {sdp_result['test_mse']:.4f}")

                # Store results
                experiment = {
                    'n': n,
                    'sigma': sigma,
                    'kernel_width': kernel_width,
                    'DP': dp_result,
                    'SDP': sdp_result
                }
                results['experiments'].append(experiment)

    # Save comprehensive results
    results_file = 'rkhs_experiments_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)

    # Summary statistics
    dp_taus = [exp['DP']['tau'] for exp in results['experiments']]
    sdp_taus = [exp['SDP']['tau'] for exp in results['experiments']]
    dp_mses = [exp['DP']['test_mse'] for exp in results['experiments']]
    sdp_mses = [exp['SDP']['test_mse'] for exp in results['experiments']]

    print(f"DP:  Mean τ = {np.mean(dp_taus):.1f}, Mean test MSE = {np.mean(dp_mses):.4f}")
    print(f"SDP: Mean τ = {np.mean(sdp_taus):.1f}, Mean test MSE = {np.mean(sdp_mses):.4f}")
    print(f"\nResults saved to: {results_file}")

    return results

if __name__ == "__main__":
    results = main()