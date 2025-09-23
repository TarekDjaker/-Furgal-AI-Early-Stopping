"""
Tests unitaires complets pour les algorithmes d'early stopping.
"""

import pytest
import numpy as np
import sys
import os
from typing import Tuple

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dp_early_stopping import DPSGDEarlyStopping
from component_early_stopping import ComponentEarlyStopping
from fairness_early_stopping import FairnessEarlyStopping, group_error_rates, demographic_parity_difference
from proximal_early_stopping import ProximalEarlyStopping, l1_proximal
from Implementations.rkhs_gradient_descent import RKHSGradientDescent


class TestDataGenerator:
    """Générateur de données pour les tests."""

    @staticmethod
    def generate_regression_data(n_samples: int = 100,
                                n_features: int = 10,
                                noise: float = 0.1,
                                seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Génère des données de régression synthétiques."""
        np.random.seed(seed)
        X = np.random.randn(n_samples, n_features)
        true_coef = np.random.randn(n_features)
        y = X @ true_coef + noise * np.random.randn(n_samples)
        return X, y

    @staticmethod
    def generate_classification_data(n_samples: int = 100,
                                    n_features: int = 10,
                                    n_classes: int = 2,
                                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Génère des données de classification synthétiques."""
        np.random.seed(seed)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return X, y


class TestRKHSGradientDescent:
    """Tests pour RKHS Gradient Descent."""

    def setup_method(self):
        """Initialisation avant chaque test."""
        self.X_train, self.y_train = TestDataGenerator.generate_regression_data()
        self.X_test, self.y_test = TestDataGenerator.generate_regression_data(n_samples=50)

    def test_initialization(self):
        """Test l'initialisation du modèle."""
        model = RKHSGradientDescent(
            self.X_train, self.y_train,
            kernel_type='gaussian',
            kernel_width=0.5
        )
        assert model.n == len(self.X_train)
        assert model.kernel_type == 'gaussian'
        assert model.kernel_width == 0.5

    def test_fit_with_dp(self):
        """Test fit avec Discrepancy Principle."""
        model = RKHSGradientDescent(
            self.X_train, self.y_train,
            kernel_type='gaussian',
            kernel_width=0.5
        )
        result = model.fit(
            stopping_rule='DP',
            sigma=0.3,
            max_iterations=100
        )
        assert 'iterations' in result
        assert 'stopped_at' in result
        assert result['stopped_at'] <= 100

    def test_fit_with_sdp(self):
        """Test fit avec Smoothed Discrepancy Principle."""
        model = RKHSGradientDescent(
            self.X_train, self.y_train,
            kernel_type='gaussian',
            kernel_width=0.5
        )
        result = model.fit(
            stopping_rule='SDP',
            sigma=0.3,
            max_iterations=100
        )
        assert 'iterations' in result
        assert 'stopped_at' in result

    def test_predict(self):
        """Test la prédiction."""
        model = RKHSGradientDescent(
            self.X_train, self.y_train,
            kernel_type='gaussian',
            kernel_width=0.5
        )
        model.fit(stopping_rule='DP', sigma=0.3, max_iterations=50)
        predictions = model.predict(self.X_test)
        assert len(predictions) == len(self.X_test)
        assert predictions.dtype == np.float64

    def test_different_kernels(self):
        """Test différents types de kernels."""
        kernels = ['gaussian', 'polynomial', 'laplacian']
        for kernel_type in kernels:
            model = RKHSGradientDescent(
                self.X_train, self.y_train,
                kernel_type=kernel_type,
                kernel_width=0.5
            )
            result = model.fit(stopping_rule='CV', max_iterations=10)
            assert result is not None


class TestDPSGDEarlyStopping:
    """Tests pour DP-SGD Early Stopping."""

    def setup_method(self):
        """Initialisation avant chaque test."""
        self.X_train, self.y_train = TestDataGenerator.generate_regression_data()
        self.n_features = self.X_train.shape[1]

    def test_initialization(self):
        """Test l'initialisation."""
        model = DPSGDEarlyStopping(
            epsilon=1.0,
            delta=1e-5,
            noise_multiplier=1.0,
            max_iterations=100
        )
        assert model.epsilon == 1.0
        assert model.delta == 1e-5
        assert model.max_iterations == 100

    def test_fit_with_privacy(self):
        """Test fit avec garanties de privacy."""
        model = DPSGDEarlyStopping(
            epsilon=1.0,
            delta=1e-5,
            noise_multiplier=1.0,
            max_iterations=50
        )
        history = model.fit(self.X_train, self.y_train)
        assert 'iterations' in history
        assert 'privacy_budget' in history
        assert len(history['iterations']) <= 50

    def test_privacy_budget_tracking(self):
        """Test le suivi du budget de privacy."""
        model = DPSGDEarlyStopping(
            epsilon=1.0,
            delta=1e-5,
            noise_multiplier=1.0,
            max_iterations=100
        )
        history = model.fit(self.X_train, self.y_train)
        # Vérifier que le budget augmente
        if len(history['privacy_budget']) > 1:
            assert history['privacy_budget'][-1] >= history['privacy_budget'][0]


class TestComponentEarlyStopping:
    """Tests pour Component-wise Early Stopping."""

    def setup_method(self):
        """Initialisation avant chaque test."""
        self.n_components = 5
        self.n_samples = 100
        self.model = ComponentEarlyStopping(
            n_components=self.n_components,
            patience=5,
            min_delta=0.001
        )

    def test_initialization(self):
        """Test l'initialisation."""
        assert self.model.n_components == self.n_components
        assert self.model.patience == 5
        assert self.model.min_delta == 0.001

    def test_should_stop_component(self):
        """Test la décision d'arrêt par composant."""
        # Simuler une perte qui ne s'améliore pas
        losses = [0.5, 0.49, 0.489, 0.488, 0.488, 0.488, 0.488, 0.488, 0.488]
        should_stop = False
        for i, loss in enumerate(losses):
            should_stop = self.model.should_stop_component(0, loss, i)
            if should_stop:
                break
        # Après patience iterations sans amélioration
        assert should_stop == True

    def test_get_active_components(self):
        """Test récupération des composants actifs."""
        # Simuler l'arrêt de certains composants
        self.model.stopped_components = {0: 10, 2: 15}
        active = self.model.get_active_components(iteration=20)
        assert 0 not in active
        assert 1 in active
        assert 2 not in active
        assert 3 in active
        assert 4 in active


class TestFairnessEarlyStopping:
    """Tests pour Fairness Early Stopping."""

    def setup_method(self):
        """Initialisation avant chaque test."""
        self.X, self.y = TestDataGenerator.generate_classification_data()
        # Attribut sensible binaire
        self.sensitive_attr = np.random.randint(0, 2, len(self.y))

    def test_group_error_rates(self):
        """Test le calcul des taux d'erreur par groupe."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        sensitive = np.array([0, 0, 0, 1, 1, 1])

        error_0, error_1 = group_error_rates(y_true, y_pred, sensitive)

        # Groupe 0: 1 erreur sur 3 = 0.333
        # Groupe 1: 1 erreur sur 3 = 0.333
        assert abs(error_0 - 0.333) < 0.01
        assert abs(error_1 - 0.333) < 0.01

    def test_demographic_parity(self):
        """Test le calcul de demographic parity."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        sensitive = np.array([0, 0, 0, 1, 1, 1])

        dp_diff = demographic_parity_difference(y_true, y_pred, sensitive)
        assert isinstance(dp_diff, float)
        assert dp_diff >= 0  # Différence absolue

    def test_fairness_model_fit(self):
        """Test fit avec contraintes de fairness."""
        model = FairnessEarlyStopping(
            fairness_threshold=0.1,
            max_iterations=50
        )
        history = model.fit(
            self.X, self.y,
            sensitive_attr=self.sensitive_attr
        )
        assert 'iterations' in history
        assert 'fairness_violations' in history


class TestProximalEarlyStopping:
    """Tests pour Proximal Early Stopping."""

    def setup_method(self):
        """Initialisation avant chaque test."""
        self.X, self.y = TestDataGenerator.generate_regression_data()
        self.n_features = self.X.shape[1]

    def test_l1_proximal_operator(self):
        """Test l'opérateur proximal L1."""
        x = np.array([0.5, -1.2, 0.1, -0.05, 2.0])
        threshold = 0.2
        result = l1_proximal(x, threshold)

        # Vérifier le soft-thresholding
        assert abs(result[0] - 0.3) < 1e-10  # 0.5 - 0.2
        assert abs(result[1] + 1.0) < 1e-10  # -1.2 + 0.2
        assert result[2] == 0  # |0.1| < 0.2
        assert result[3] == 0  # |0.05| < 0.2
        assert abs(result[4] - 1.8) < 1e-10  # 2.0 - 0.2

    def test_fista_convergence(self):
        """Test la convergence de FISTA."""
        model = ProximalEarlyStopping(
            method='FISTA',
            lambda_reg=0.1,
            max_iterations=100
        )
        result = model.fit(self.X, self.y)

        assert 'iterations' in result
        assert 'objective_values' in result
        assert result['stopped_at'] <= 100

        # Vérifier que l'objectif décroit
        obj_values = result['objective_values']
        if len(obj_values) > 10:
            # La fonction objectif devrait généralement décroître
            assert obj_values[-1] <= obj_values[10]

    def test_sparsity_pattern(self):
        """Test que L1 produit de la sparsité."""
        model = ProximalEarlyStopping(
            method='FISTA',
            lambda_reg=1.0,  # Forte régularisation
            max_iterations=100
        )
        result = model.fit(self.X, self.y)

        # Compter les coefficients nuls
        coefficients = result.get('coefficients', model.theta)
        n_zeros = np.sum(np.abs(coefficients) < 1e-6)

        # Avec forte régularisation L1, on devrait avoir de la sparsité
        assert n_zeros > 0


class TestIntegration:
    """Tests d'intégration entre différents modules."""

    def test_all_models_compatible_data(self):
        """Test que tous les modèles acceptent les mêmes données."""
        X, y = TestDataGenerator.generate_regression_data()

        models = [
            RKHSGradientDescent(X, y, kernel_type='gaussian'),
            DPSGDEarlyStopping(epsilon=1.0, delta=1e-5),
            ProximalEarlyStopping(method='FISTA', lambda_reg=0.1)
        ]

        for model in models:
            if hasattr(model, 'fit'):
                # Essayer de fit sans erreur
                try:
                    if isinstance(model, RKHSGradientDescent):
                        model.fit(stopping_rule='DP', sigma=0.3, max_iterations=10)
                    else:
                        model.fit(X, y)
                except Exception as e:
                    pytest.fail(f"Model {type(model).__name__} failed: {str(e)}")


@pytest.fixture
def benchmark_data():
    """Fixture pour données de benchmark."""
    return TestDataGenerator.generate_regression_data(n_samples=1000, n_features=50)


def test_performance_comparison(benchmark_data):
    """Test de comparaison de performance entre méthodes."""
    X, y = benchmark_data

    import time

    results = {}

    # RKHS avec DP
    start = time.time()
    model = RKHSGradientDescent(X, y, kernel_type='gaussian', kernel_width=0.5)
    model.fit(stopping_rule='DP', sigma=0.3, max_iterations=50)
    results['RKHS-DP'] = time.time() - start

    # Proximal FISTA
    start = time.time()
    model = ProximalEarlyStopping(method='FISTA', lambda_reg=0.1)
    model.fit(X, y)
    results['FISTA'] = time.time() - start

    # Les résultats devraient être raisonnables (< 10 secondes)
    for method, runtime in results.items():
        assert runtime < 10.0, f"{method} took too long: {runtime:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])