#!/usr/bin/env python3
"""
Example: Proximal gradient descent with early stopping for sparse regression.

This script demonstrates how to use the ProximalEarlyStopping class
to solve a LASSO regression problem (L1-regularized linear regression).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proximal_early_stopping import ProximalEarlyStopping


def main():
    """Run the proximal gradient descent example."""
    # Generate synthetic sparse regression data
    print("Generating synthetic regression data...")
    X, y = make_regression(
        n_samples=200, 
        n_features=50, 
        n_informative=10,
        noise=0.1, 
        random_state=42
    )
    
    # Split and normalize data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    print(f"Training set size: {X_train_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    
    # Set up proximal gradient descent solver
    print("\nSetting up proximal gradient descent solver...")
    
    # Use early stopping callback to track progress
    def callback(iteration, params, objective):
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: objective = {objective:.6f}, "
                  f"sparsity = {np.sum(np.abs(params) > 1e-6)}/{len(params)}")
    
    solver = ProximalEarlyStopping(
        design=X_train_scaled,
        response=y_train_scaled,
        lam=0.1,  # L1 regularization parameter
        step_size=0.001,  # Smaller step size for stability
        max_iter=1000,
        tol=1e-5,
        patience=10,
        callback=callback
    )
    
    # Fit the model
    print("\nFitting model with early stopping...")
    params, num_iter, objectives = solver.fit()
    
    print(f"\nTraining completed:")
    print(f"  - Stopped after {num_iter} iterations")
    print(f"  - Final objective: {objectives[-1]:.6f}")
    print(f"  - Number of non-zero coefficients: {np.sum(np.abs(params) > 1e-6)}")
    
    # Evaluate on test set
    y_pred_scaled = X_test_scaled @ params
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    test_mse = np.mean((y_test - y_pred) ** 2)
    print(f"  - Test MSE: {test_mse:.6f}")
    
    # Plot convergence
    try:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(objectives)
        plt.title('Objective Function Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.stem(range(len(params)), params, basefmt=" ")
        plt.title('Learned Coefficients (Sparse)')
        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient Value')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('proximal_example_results.png', dpi=150, bbox_inches='tight')
        print("\nPlots saved as 'proximal_example_results.png'")
        
    except ImportError:
        print("\nMatplotlib not available for plotting, but results computed successfully.")


if __name__ == "__main__":
    main()