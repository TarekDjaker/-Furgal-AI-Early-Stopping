#!/usr/bin/env python3
"""
Example: Fairness-aware early stopping.

This script demonstrates how to use the FairnessEarlyStopping class
to monitor fairness metrics during training and stop when fairness
improvements plateau.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fairness_early_stopping import FairnessEarlyStopping, demographic_parity_difference


def create_biased_dataset(n_samples=1000, random_state=42):
    """Create a synthetic dataset with bias in sensitive attribute."""
    np.random.seed(random_state)
    
    # Generate base features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    # Create a synthetic sensitive attribute (e.g., binary group membership)
    # Introduce bias: group 0 has higher probability of positive class
    sensitive_attr = np.random.binomial(1, 0.5, n_samples)
    
    # Introduce bias in the labels based on sensitive attribute
    bias_strength = 0.3
    for i in range(n_samples):
        if sensitive_attr[i] == 0:  # Privileged group
            if np.random.random() < bias_strength:
                y[i] = 1  # Favor positive outcomes
        else:  # Unprivileged group
            if np.random.random() < bias_strength:
                y[i] = 0  # Favor negative outcomes
    
    return X, y, sensitive_attr


def train_with_fairness_monitoring(X_train, y_train, sensitive_train, 
                                 X_val, y_val, sensitive_val,
                                 max_epochs=100):
    """Train a logistic regression model with fairness monitoring."""
    
    # Initialize fairness early stopping
    fairness_stopper = FairnessEarlyStopping(
        patience=5,
        min_delta=0.001,
        verbose=True
    )
    
    # We'll simulate iterative training by fitting on increasingly larger subsets
    # (In practice, you'd use this with neural networks or other iterative algorithms)
    best_model = None
    best_fairness = float('inf')
    
    print("Training with fairness monitoring...")
    
    for epoch in range(1, max_epochs + 1):
        # Simulate iterative training by adding noise and refitting
        # (In real scenarios, this would be actual gradient steps)
        X_train_noisy = X_train + np.random.normal(0, 0.01, X_train.shape)
        
        # Fit model
        model = LogisticRegression(random_state=42 + epoch, max_iter=1000)
        model.fit(X_train_noisy, y_train)
        
        # Make predictions on validation set
        y_pred_val = model.predict(X_val)
        
        # Check fairness stopping condition
        should_stop = fairness_stopper.step(epoch, y_val, y_pred_val, sensitive_val)
        
        # Track best model based on fairness
        fairness_metric = demographic_parity_difference(y_val, y_pred_val, sensitive_val)
        if fairness_metric < best_fairness:
            best_fairness = fairness_metric
            best_model = model
        
        # Print progress every 10 epochs
        if epoch % 10 == 0:
            accuracy = np.mean(y_pred_val == y_val)
            print(f"Epoch {epoch:2d}: Accuracy = {accuracy:.4f}, "
                  f"Fairness Metric = {fairness_metric:.4f}")
        
        if should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    return best_model, fairness_stopper.stopped_epoch


def evaluate_fairness(model, X_test, y_test, sensitive_test):
    """Evaluate model performance and fairness metrics."""
    y_pred_test = model.predict(X_test)
    
    # Overall accuracy
    accuracy = np.mean(y_pred_test == y_test)
    
    # Fairness metrics
    fairness_metric = demographic_parity_difference(y_test, y_pred_test, sensitive_test)
    
    # Group-wise performance
    for group in [0, 1]:
        group_mask = sensitive_test == group
        if np.sum(group_mask) > 0:
            group_accuracy = np.mean(y_pred_test[group_mask] == y_test[group_mask])
            group_positive_rate = np.mean(y_pred_test[group_mask] == 1)
            print(f"Group {group}: Accuracy = {group_accuracy:.4f}, "
                  f"Positive Rate = {group_positive_rate:.4f}")
    
    return accuracy, fairness_metric


def main():
    """Run the fairness-aware early stopping example."""
    print("Generating biased synthetic dataset...")
    
    # Create biased dataset
    X, y, sensitive_attr = create_biased_dataset(n_samples=1500, random_state=42)
    
    # Split data
    X_train, X_temp, y_train, y_temp, sensitive_train, sensitive_temp = train_test_split(
        X, y, sensitive_attr, test_size=0.4, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test, sensitive_val, sensitive_test = train_test_split(
        X_temp, y_temp, sensitive_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset sizes:")
    print(f"  Train: {X_train_scaled.shape[0]} samples")
    print(f"  Validation: {X_val_scaled.shape[0]} samples")
    print(f"  Test: {X_test_scaled.shape[0]} samples")
    
    # Analyze initial bias
    print(f"\nInitial dataset bias:")
    train_fairness = demographic_parity_difference(y_train, y_train, sensitive_train)
    print(f"  Training set demographic parity difference: {train_fairness:.4f}")
    
    for group in [0, 1]:
        group_mask = sensitive_train == group
        positive_rate = np.mean(y_train[group_mask])
        print(f"  Group {group} positive rate: {positive_rate:.4f}")
    
    # Train model with fairness monitoring
    print(f"\n" + "="*60)
    model, stopped_epoch = train_with_fairness_monitoring(
        X_train_scaled, y_train, sensitive_train,
        X_val_scaled, y_val, sensitive_val,
        max_epochs=50
    )
    
    # Evaluate final model
    print(f"\n" + "="*60)
    print("Final model evaluation on test set:")
    
    test_accuracy, test_fairness = evaluate_fairness(
        model, X_test_scaled, y_test, sensitive_test
    )
    
    print(f"\nOverall Results:")
    print(f"  - Training stopped at epoch: {stopped_epoch}")
    print(f"  - Test accuracy: {test_accuracy:.4f}")
    print(f"  - Test fairness metric (demographic parity): {test_fairness:.4f}")
    
    if abs(test_fairness) < 0.05:
        print("  - Model achieved good fairness (|fairness metric| < 0.05)")
    else:
        print("  - Model still has fairness issues (consider further tuning)")


if __name__ == "__main__":
    main()