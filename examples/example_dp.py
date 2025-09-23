#!/usr/bin/env python3
"""
Example: Differentially private SGD with early stopping.

This script demonstrates how to use the DPSGDEarlyStopping class
to train a linear model under differential privacy constraints.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dp_early_stopping import DPSGDEarlyStopping


def mse_loss(y_pred, y_true):
    """Mean squared error loss function."""
    return np.mean((y_pred - y_true) ** 2)


def mse_gradient(params, X, y):
    """Gradient of MSE loss with respect to parameters."""
    y_pred = X @ params
    residual = y_pred - y
    return X.T @ residual / len(y)


def main():
    """Run the differential privacy early stopping example."""
    np.random.seed(42)
    
    print("Generating synthetic regression data...")
    
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        noise=0.1,
        random_state=42
    )
    
    # Split into train/validation/test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.3, random_state=42
    )
    
    # Normalize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    print(f"Dataset sizes:")
    print(f"  Train: {X_train_scaled.shape[0]} samples")
    print(f"  Validation: {X_val_scaled.shape[0]} samples")  
    print(f"  Test: {X_test_scaled.shape[0]} samples")
    print(f"  Features: {X_train_scaled.shape[1]}")
    
    # Compare non-private vs private training
    results = {}
    
    for privacy_setting in ["non-private", "private"]:
        print(f"\n{'='*60}")
        print(f"Training: {privacy_setting.upper()}")
        print(f"{'='*60}")
        
        if privacy_setting == "non-private":
            # Non-private baseline (no noise)
            dp_solver = DPSGDEarlyStopping(
                loss_fn=mse_loss,
                grad_fn=mse_gradient,
                noise_std=0.0,  # No noise
                lr=0.01,
                max_iter=500,
                patience=20,
                eps_per_iter=0.0,  # No privacy cost
                verbose=True
            )
        else:
            # Private training (with noise)
            dp_solver = DPSGDEarlyStopping(
                loss_fn=mse_loss,
                grad_fn=mse_gradient,
                noise_std=0.1,  # Add noise for privacy
                lr=0.01,
                max_iter=500,
                patience=20,
                eps_per_iter=0.01,  # Privacy cost per iteration
                verbose=True
            )
        
        # Train the model
        final_params, stop_iter, total_eps, val_history = dp_solver.fit(
            X_train_scaled, y_train_scaled,
            X_val_scaled, y_val_scaled
        )
        
        # Evaluate on test set
        y_test_pred_scaled = X_test_scaled @ final_params
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # Store results
        results[privacy_setting] = {
            'params': final_params,
            'stop_iter': stop_iter,
            'total_eps': total_eps,
            'test_mse': test_mse,
            'val_history': val_history
        }
        
        print(f"\nResults for {privacy_setting}:")
        print(f"  - Stopped at iteration: {stop_iter}")
        print(f"  - Total privacy cost (ε): {total_eps:.6f}")
        print(f"  - Test MSE: {test_mse:.6f}")
        print(f"  - Final validation loss: {val_history[-1]:.6f}")
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    non_private_mse = results["non-private"]["test_mse"]
    private_mse = results["private"]["test_mse"]
    privacy_cost = results["private"]["total_eps"]
    
    print(f"Non-private test MSE: {non_private_mse:.6f}")
    print(f"Private test MSE:     {private_mse:.6f}")
    print(f"MSE degradation:      {private_mse - non_private_mse:.6f} "
          f"({((private_mse / non_private_mse) - 1) * 100:+.2f}%)")
    print(f"Privacy cost (ε):     {privacy_cost:.6f}")
    
    # Privacy-utility trade-off analysis
    if privacy_cost > 0:
        utility_loss_per_epsilon = (private_mse - non_private_mse) / privacy_cost
        print(f"Utility loss per ε:   {utility_loss_per_epsilon:.6f}")
        
        if privacy_cost < 1.0:
            print(f"\nPrivacy Analysis:")
            print(f"  - ε = {privacy_cost:.3f} provides strong privacy protection")
            if private_mse / non_private_mse < 1.2:
                print(f"  - Utility degradation is acceptable (<20% increase in MSE)")
            else:
                print(f"  - Significant utility loss - consider tuning noise_std or eps_per_iter")
    
    # Demonstrate early stopping benefit
    non_private_iters = results["non-private"]["stop_iter"]
    private_iters = results["private"]["stop_iter"]
    
    print(f"\nEarly Stopping Analysis:")
    print(f"  - Non-private training stopped at iteration {non_private_iters}")
    print(f"  - Private training stopped at iteration {private_iters}")
    
    if private_iters < 500:
        saved_privacy = (500 - private_iters) * results["private"]["total_eps"] / private_iters
        print(f"  - Early stopping saved ~{saved_privacy:.3f} privacy budget")
    
    print(f"\nConclusion:")
    print(f"  - Differential privacy adds noise but early stopping helps")
    print(f"  - Consider adjusting noise_std and eps_per_iter for your privacy needs")


if __name__ == "__main__":
    main()