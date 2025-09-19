"""
Basic smoke tests for early stopping modules.

These tests verify that modules can be imported and basic functionality works.
They are not comprehensive but ensure the code doesn't have obvious bugs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

def test_imports():
    """Test that all modules can be imported without errors."""
    try:
        from proximal_early_stopping import ProximalEarlyStopping, l1_proximal
        from fairness_early_stopping import (
            FairnessEarlyStopping, demographic_parity_difference, group_error_rates
        )
        from dp_early_stopping import DPSGDEarlyStopping
        
        # Import component module only if torch is available
        try:
            import torch
            from component_early_stopping import ComponentEarlyStopping
        except ImportError:
            pass  # Skip if torch not available
        
    except ImportError as e:
        if pytest is not None:
            pytest.fail(f"Import failed: {e}")
        else:
            raise


def test_proximal_early_stopping():
    """Test basic functionality of ProximalEarlyStopping."""
    from proximal_early_stopping import ProximalEarlyStopping
    
    # Create simple test data
    np.random.seed(42)
    n_samples, n_features = 20, 5
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # Create solver
    solver = ProximalEarlyStopping(
        design=X,
        response=y,
        lam=0.1,
        step_size=0.001,
        max_iter=10,  # Small number for testing
        tol=1e-3,
        patience=5
    )
    
    # Fit model
    params, num_iter, objectives = solver.fit()
    
    # Basic checks
    assert params.shape == (n_features,)
    assert isinstance(num_iter, int)
    assert num_iter <= 10
    assert len(objectives) == num_iter + 1
    assert all(isinstance(obj, (int, float)) for obj in objectives)


def test_fairness_early_stopping():
    """Test basic functionality of FairnessEarlyStopping."""
    from fairness_early_stopping import FairnessEarlyStopping, demographic_parity_difference
    
    # Create test data
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.binomial(1, 0.5, n_samples)
    y_pred = np.random.binomial(1, 0.6, n_samples)
    sensitive_attr = np.random.binomial(1, 0.5, n_samples)
    
    # Test demographic parity function
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_attr)
    assert isinstance(dp_diff, (int, float))
    
    # Test FairnessEarlyStopping
    stopper = FairnessEarlyStopping(patience=3, verbose=False)
    
    # Simulate a few epochs
    for epoch in range(5):
        should_stop = stopper.step(epoch, y_true, y_pred, sensitive_attr)
        assert isinstance(should_stop, bool)
        
        if should_stop:
            break


def test_dp_early_stopping():
    """Test basic functionality of DPSGDEarlyStopping."""
    from dp_early_stopping import DPSGDEarlyStopping
    
    # Simple loss and gradient functions for linear regression
    def mse_loss(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def mse_grad(params, X, y):
        y_pred = X @ params
        residual = y_pred - y
        return X.T @ residual / len(y)
    
    # Create test data
    np.random.seed(42)
    n_train, n_val, n_features = 20, 10, 3
    X_train = np.random.randn(n_train, n_features)
    y_train = np.random.randn(n_train)
    X_val = np.random.randn(n_val, n_features)
    y_val = np.random.randn(n_val)
    
    # Create DP solver
    dp_solver = DPSGDEarlyStopping(
        loss_fn=mse_loss,
        grad_fn=mse_grad,
        noise_std=0.01,
        lr=0.01,
        max_iter=5,  # Small number for testing
        patience=2,
        eps_per_iter=0.01,
        verbose=False
    )
    
    # Fit model
    final_params, stop_iter, total_eps, val_history = dp_solver.fit(
        X_train, y_train, X_val, y_val
    )
    
    # Basic checks
    assert final_params.shape == (n_features,)
    assert isinstance(stop_iter, int)
    assert stop_iter <= 5
    assert isinstance(total_eps, (int, float))
    assert len(val_history) == stop_iter + 1


@pytest.mark.skipif(not pytest.importorskip("torch", reason="torch not available"), reason="torch required")
def test_component_early_stopping():
    """Test basic functionality of ComponentEarlyStopping."""
    import torch
    import torch.nn as nn
    from component_early_stopping import ComponentEarlyStopping
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(5, 3),
        nn.ReLU(),
        nn.Linear(3, 1)
    )
    
    # Create component stopper
    stopper = ComponentEarlyStopping(model, threshold=1e-3, verbose=False)
    
    # Create dummy input and target
    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    
    # Forward pass
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    
    # Backward pass
    loss.backward()
    
    # Apply component stopping
    frozen = stopper.apply()
    
    # Basic checks
    assert isinstance(frozen, list)
    assert all(isinstance(name, str) for name in frozen)
    
    # Test summary
    summary = stopper.summary()
    assert isinstance(summary, list)
    assert all(isinstance(name, str) for name in summary)
    
    # Test reset
    stopper.reset()
    assert len(stopper.summary()) == 0


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_imports()
    test_proximal_early_stopping()
    test_fairness_early_stopping()
    test_dp_early_stopping()
    
    try:
        import torch
        test_component_early_stopping()
        print("All tests passed!")
    except ImportError:
        print("All tests passed (torch component test skipped)!")
    except Exception as e:
        print(f"Test failed: {e}")
        raise