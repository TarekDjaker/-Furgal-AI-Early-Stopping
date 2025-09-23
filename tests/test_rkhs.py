"""
Tests for RKHS Gradient Descent with DP/SDP stopping rules
"""

import pytest
import numpy as np
from Implementations.rkhs_gradient_descent import RKHSGradientDescent


class TestRKHSGradientDescent:
    """Test suite for RKHS gradient descent implementation"""

    @pytest.fixture
    def simple_data(self):
        """Generate simple synthetic data for testing"""
        np.random.seed(42)
        n = 50
        X = np.linspace(-1, 1, n).reshape(-1, 1)
        y = np.sin(2 * np.pi * X).ravel() + 0.1 * np.random.randn(n)
        return X, y

    def test_prediction_shape(self, simple_data):
        """Test that predictions have correct shape"""
        X_train, y_train = simple_data
        X_test = np.linspace(-1.5, 1.5, 30).reshape(-1, 1)

        model = RKHSGradientDescent(
            X_train, y_train,
            kernel_type='gaussian',
            kernel_width=0.2
        )

        model.fit(max_iterations=10, stopping_rule='none', verbose=False)
        y_pred = model.predict(X_test)

        assert y_pred.shape == (len(X_test),), f"Expected shape {(len(X_test),)}, got {y_pred.shape}"

    def test_residual_monotone_decrease(self, simple_data):
        """Test that residual norms decrease monotonically for first few iterations"""
        X_train, y_train = simple_data

        model = RKHSGradientDescent(
            X_train, y_train,
            kernel_type='gaussian',
            kernel_width=0.2
        )

        # Compute residuals for first 5 iterations
        residuals = []
        for t in range(1, 6):
            residuals.append(model.residual_norm_empirical(t))

        # Check monotone decrease
        for i in range(len(residuals) - 1):
            assert residuals[i] >= residuals[i + 1], \
                f"Residual should decrease: {residuals[i]} -> {residuals[i + 1]}"

    def test_dp_stops_early_with_large_sigma(self, simple_data):
        """Test that DP stops early (tau < max_iter) when sigma is large"""
        X_train, y_train = simple_data

        model = RKHSGradientDescent(
            X_train, y_train,
            kernel_type='gaussian',
            kernel_width=0.2
        )

        # Use large sigma to ensure early stopping
        large_sigma = 1.0
        max_iterations = 100

        result = model.fit(
            max_iterations=max_iterations,
            stopping_rule='DP',
            sigma=large_sigma,
            verbose=False
        )

        tau = result['tau']
        assert tau < max_iterations, \
            f"DP should stop early with large sigma: tau={tau}, max_iter={max_iterations}"
        assert tau > 0, f"Stopping time should be positive: tau={tau}"


if __name__ == "__main__":
    # Run tests locally
    pytest.main([__file__, "-v"])