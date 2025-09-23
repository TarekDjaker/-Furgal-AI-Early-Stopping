#!/usr/bin/env python3
"""
Example: Component-wise early stopping for neural networks.

This script demonstrates how to use the ComponentEarlyStopping class
to freeze layers in a neural network when their gradients become small.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from component_early_stopping import ComponentEarlyStopping


class SimpleMLPClassifier(nn.Module):
    """Simple multi-layer perceptron for binary classification."""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def main():
    """Run the component-wise early stopping example."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic classification data
    print("Generating synthetic classification data...")
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split and normalize data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    print(f"Training set size: {X_train_tensor.shape}")
    print(f"Test set size: {X_test_tensor.shape}")
    
    # Create model
    model = SimpleMLPClassifier(
        input_dim=X_train_scaled.shape[1],
        hidden_dims=[64, 32, 16],
        output_dim=2
    )
    
    print(f"\nModel architecture:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    
    # Set up component-wise early stopping
    component_stopper = ComponentEarlyStopping(
        model=model, 
        threshold=1e-4, 
        verbose=True
    )
    
    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop with component-wise early stopping
    print("\nStarting training with component-wise early stopping...")
    num_epochs = 200
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass
        loss.backward()
        
        # Apply component-wise early stopping (before optimizer step)
        newly_frozen = component_stopper.apply()
        
        # Optimizer step
        optimizer.step()
        
        # Print progress
        if epoch % 20 == 0 or newly_frozen:
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train_tensor)
                train_pred = train_outputs.argmax(dim=1)
                train_acc = (train_pred == y_train_tensor).float().mean()
                
                test_outputs = model(X_test_tensor)
                test_pred = test_outputs.argmax(dim=1)
                test_acc = (test_pred == y_test_tensor).float().mean()
            
            frozen_params = component_stopper.summary()
            print(f"Epoch {epoch:3d}: Loss = {loss:.6f}, "
                  f"Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}, "
                  f"Frozen = {len(frozen_params)}/{sum(1 for _ in model.named_parameters())}")
            
            if newly_frozen:
                print(f"  -> Newly frozen parameters: {newly_frozen}")
        
        # Check if all parameters are frozen
        if len(component_stopper.summary()) == sum(1 for _ in model.named_parameters()):
            print(f"\nAll parameters frozen at epoch {epoch}. Stopping training.")
            break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_pred = test_outputs.argmax(dim=1)
        test_acc = (test_pred == y_test_tensor).float().mean()
    
    print(f"\nFinal Results:")
    print(f"  - Training completed at epoch {epoch}")
    print(f"  - Test accuracy: {test_acc:.4f}")
    print(f"  - Frozen parameters: {len(component_stopper.summary())}/{sum(1 for _ in model.named_parameters())}")
    print(f"  - Frozen parameter names: {component_stopper.summary()}")


if __name__ == "__main__":
    main()