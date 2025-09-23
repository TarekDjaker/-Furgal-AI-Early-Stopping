"""
Visualization script for RKHS experiment results
Produces plots for residual vs iteration and test MSE vs tau
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Implementations.rkhs_gradient_descent import RKHSGradientDescent


def plot_residual_curves(save_path='plots/residual_curves.png'):
    """Plot residual vs iteration curves"""

    # Generate sample data
    np.random.seed(42)
    n = 200
    sigma = 0.3

    X = np.sort(np.random.uniform(-2, 2, (n, 1)), axis=0)
    y_true = np.sin(3 * X).ravel() + 0.5 * np.cos(5 * X).ravel()
    y = y_true + sigma * np.random.randn(n)

    # Run models
    model = RKHSGradientDescent(X, y, kernel_type='gaussian', kernel_width=0.2)

    # Collect residuals for many iterations
    max_iter = 100
    residuals = []
    for t in range(1, max_iter + 1):
        residuals.append(model.residual_norm_empirical(t))

    # Find stopping times
    tau_dp = model.find_tau_DP(sigma, max_iter)
    tau_sdp, _ = model.find_tau_SDP(sigma, T_smooth=10, max_t=max_iter)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    iterations = range(1, max_iter + 1)
    ax.semilogy(iterations, residuals, 'b-', linewidth=2, label='Residual norm')
    ax.axhline(y=sigma, color='k', linestyle=':', linewidth=1.5, label=f'Noise level σ = {sigma}')
    ax.axvline(x=tau_dp, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'DP stopping (τ = {tau_dp})')
    ax.axvline(x=tau_sdp, color='g', linestyle='--', linewidth=1.5, alpha=0.7, label=f'SDP stopping (τ = {tau_sdp})')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Residual Norm (log scale)', fontsize=12)
    ax.set_title('RKHS Gradient Descent: Residual Evolution with Early Stopping', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved residual curves plot to: {save_path}")

    return fig


def plot_mse_vs_tau(csv_path='synthetic_rkhs_results.csv', save_path='plots/mse_vs_tau.png'):
    """Plot test MSE vs stopping time tau"""

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Run examples/01_synthetic_rkhs.py first.")
        return None

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: MSE vs tau for different methods
    ax = axes[0]
    for method in ['DP', 'SDP']:
        method_df = df[df['method'] == method]
        color = 'blue' if method == 'DP' else 'red'
        ax.scatter(method_df['tau'], method_df['mse_test'], alpha=0.6, s=50, label=method, color=color)

        # Add trend line
        z = np.polyfit(method_df['tau'], method_df['mse_test'], 2)
        p = np.poly1d(z)
        tau_smooth = np.linspace(method_df['tau'].min(), method_df['tau'].max(), 100)
        ax.plot(tau_smooth, p(tau_smooth), "--", color=color, alpha=0.5)

    ax.set_xlabel('Stopping Time (τ)', fontsize=12)
    ax.set_ylabel('Test MSE', fontsize=12)
    ax.set_title('Test MSE vs Stopping Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Heatmap of best tau for each (sigma, kernel_width) combination
    ax = axes[1]

    # Prepare data for heatmap
    dp_df = df[df['method'] == 'DP']
    pivot_tau = dp_df.pivot_table(values='tau', index='sigma', columns='kernel_width')

    im = ax.imshow(pivot_tau, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xticks(range(len(pivot_tau.columns)))
    ax.set_xticklabels([f'{x:.2f}' for x in pivot_tau.columns])
    ax.set_yticks(range(len(pivot_tau.index)))
    ax.set_yticklabels([f'{y:.1f}' for y in pivot_tau.index])

    ax.set_xlabel('Kernel Width', fontsize=12)
    ax.set_ylabel('Noise Level (σ)', fontsize=12)
    ax.set_title('DP Stopping Time Heatmap', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Stopping Time (τ)', rotation=270, labelpad=15)

    # Add text annotations
    for i in range(len(pivot_tau.index)):
        for j in range(len(pivot_tau.columns)):
            text = ax.text(j, i, f'{int(pivot_tau.iloc[i, j])}',
                          ha="center", va="center", color="white", fontsize=9)

    plt.suptitle('RKHS Early Stopping Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved MSE vs tau plot to: {save_path}")

    return fig


def main():
    """Generate all visualizations"""

    print("=" * 60)
    print("Generating RKHS Experiment Visualizations")
    print("=" * 60)

    # Create plots directory
    os.makedirs('plots', exist_ok=True)

    # Generate plots
    print("\n1. Generating residual curves plot...")
    plot_residual_curves()

    print("\n2. Generating MSE vs tau plot...")
    plot_mse_vs_tau()

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()