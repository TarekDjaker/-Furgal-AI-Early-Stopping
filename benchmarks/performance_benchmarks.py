"""
Benchmarks de performance pour comparer les algorithmes d'early stopping.
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rkhs_gradient_descent import RKHSGradientDescent
from dp_early_stopping import DPSGDEarlyStopping
from proximal_early_stopping import ProximalEarlyStopping
from component_early_stopping import ComponentEarlyStopping
from fairness_early_stopping import FairnessEarlyStopping

# Configuration pour les plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


@dataclass
class BenchmarkResult:
    """Structure pour stocker les résultats de benchmark."""
    method: str
    dataset: str
    n_samples: int
    n_features: int
    training_time: float
    iterations: int
    mse: float
    sparsity: float = 0.0
    memory_usage: float = 0.0
    computational_savings: float = 0.0


class DatasetGenerator:
    """Générateur de datasets pour les benchmarks."""

    @staticmethod
    def generate_sparse_data(n_samples: int, n_features: int,
                            sparsity: float = 0.9, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Génère des données avec coefficients sparse."""
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)

        # Coefficients sparse
        true_coef = np.random.randn(n_features)
        mask = np.random.random(n_features) < sparsity
        true_coef[mask] = 0

        y = X @ true_coef + noise * np.random.randn(n_samples)
        return X, y, true_coef

    @staticmethod
    def generate_correlated_data(n_samples: int, n_features: int,
                                correlation: float = 0.8, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Génère des données avec features corrélées."""
        np.random.seed(42)

        # Matrice de covariance avec corrélation
        cov_matrix = np.eye(n_features)
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    cov_matrix[i, j] = correlation ** abs(i - j)

        X = np.random.multivariate_normal(np.zeros(n_features), cov_matrix, n_samples)
        true_coef = np.random.randn(n_features)
        y = X @ true_coef + noise * np.random.randn(n_samples)

        return X, y, true_coef

    @staticmethod
    def generate_high_dimensional_data(n_samples: int, n_features: int,
                                      noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Génère des données en haute dimension (p >> n)."""
        assert n_features > n_samples, "n_features doit être > n_samples pour high-dimensional"

        np.random.seed(42)
        X = np.random.randn(n_samples, n_features) / np.sqrt(n_features)

        # Seulement quelques features sont importantes
        n_important = min(10, n_samples // 10)
        true_coef = np.zeros(n_features)
        true_coef[:n_important] = np.random.randn(n_important)

        y = X @ true_coef + noise * np.random.randn(n_samples)
        return X, y, true_coef


class PerformanceBenchmark:
    """Classe principale pour les benchmarks de performance."""

    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialise le benchmark."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def measure_memory(self, func, *args, **kwargs):
        """Mesure l'utilisation mémoire d'une fonction."""
        import tracemalloc

        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return result, peak / 1024 / 1024  # Convert to MB

    def benchmark_rkhs_methods(self, X_train, y_train, X_test, y_test, dataset_name: str):
        """Benchmark pour les méthodes RKHS."""
        methods = [
            ('RKHS-DP', {'stopping_rule': 'DP', 'sigma': 0.3}),
            ('RKHS-SDP', {'stopping_rule': 'SDP', 'sigma': 0.3}),
            ('RKHS-CV', {'stopping_rule': 'CV', 'n_folds': 5})
        ]

        for method_name, params in methods:
            print(f"  Running {method_name}...")

            try:
                # Mesurer le temps et la mémoire
                start_time = time.time()
                model = RKHSGradientDescent(X_train, y_train,
                                          kernel_type='gaussian',
                                          kernel_width=0.5)

                result, memory = self.measure_memory(
                    model.fit,
                    max_iterations=500,
                    **params
                )

                training_time = time.time() - start_time

                # Prédiction et MSE
                y_pred = model.predict(X_test)
                mse = np.mean((y_test - y_pred) ** 2)

                # Savings par rapport à max iterations
                savings = 1.0 - (result['stopped_at'] / 500)

                self.results.append(BenchmarkResult(
                    method=method_name,
                    dataset=dataset_name,
                    n_samples=len(X_train),
                    n_features=X_train.shape[1],
                    training_time=training_time,
                    iterations=result['stopped_at'],
                    mse=mse,
                    memory_usage=memory,
                    computational_savings=savings
                ))

            except Exception as e:
                print(f"    Error in {method_name}: {str(e)}")

    def benchmark_proximal_methods(self, X_train, y_train, X_test, y_test, dataset_name: str):
        """Benchmark pour les méthodes proximales."""
        methods = [
            ('ISTA', {'method': 'ISTA', 'lambda_reg': 0.1}),
            ('FISTA', {'method': 'FISTA', 'lambda_reg': 0.1}),
            ('FISTA-Strong', {'method': 'FISTA', 'lambda_reg': 1.0})
        ]

        for method_name, params in methods:
            print(f"  Running {method_name}...")

            try:
                start_time = time.time()
                model = ProximalEarlyStopping(max_iterations=500, **params)

                result, memory = self.measure_memory(model.fit, X_train, y_train)
                training_time = time.time() - start_time

                # Prédiction et MSE
                y_pred = X_test @ model.theta
                mse = np.mean((y_test - y_pred) ** 2)

                # Sparsité
                sparsity = np.mean(np.abs(model.theta) < 1e-6)

                # Savings
                savings = 1.0 - (result['stopped_at'] / 500)

                self.results.append(BenchmarkResult(
                    method=method_name,
                    dataset=dataset_name,
                    n_samples=len(X_train),
                    n_features=X_train.shape[1],
                    training_time=training_time,
                    iterations=result['stopped_at'],
                    mse=mse,
                    sparsity=sparsity,
                    memory_usage=memory,
                    computational_savings=savings
                ))

            except Exception as e:
                print(f"    Error in {method_name}: {str(e)}")

    def benchmark_privacy_methods(self, X_train, y_train, X_test, y_test, dataset_name: str):
        """Benchmark pour les méthodes avec privacy."""
        epsilons = [0.5, 1.0, 2.0]

        for epsilon in epsilons:
            method_name = f'DP-SGD-ε{epsilon}'
            print(f"  Running {method_name}...")

            try:
                start_time = time.time()
                model = DPSGDEarlyStopping(
                    epsilon=epsilon,
                    delta=1e-5,
                    noise_multiplier=1.0,
                    max_iterations=500
                )

                history, memory = self.measure_memory(model.fit, X_train, y_train)
                training_time = time.time() - start_time

                # MSE
                if hasattr(model, 'theta'):
                    y_pred = X_test @ model.theta
                    mse = np.mean((y_test - y_pred) ** 2)
                else:
                    mse = float('nan')

                iterations = len(history['iterations'])
                savings = 1.0 - (iterations / 500)

                self.results.append(BenchmarkResult(
                    method=method_name,
                    dataset=dataset_name,
                    n_samples=len(X_train),
                    n_features=X_train.shape[1],
                    training_time=training_time,
                    iterations=iterations,
                    mse=mse,
                    memory_usage=memory,
                    computational_savings=savings
                ))

            except Exception as e:
                print(f"    Error in {method_name}: {str(e)}")

    def run_all_benchmarks(self):
        """Exécute tous les benchmarks."""
        print("Starting comprehensive benchmarks...")

        # Configurations de datasets
        dataset_configs = [
            ('Small-Sparse', 100, 50, 'sparse'),
            ('Medium-Correlated', 500, 100, 'correlated'),
            ('Large-Dense', 1000, 200, 'dense'),
            ('High-Dimensional', 100, 500, 'high_dim')
        ]

        for dataset_name, n_samples, n_features, data_type in dataset_configs:
            print(f"\nBenchmarking on {dataset_name} dataset...")

            # Générer les données
            if data_type == 'sparse':
                X, y, _ = DatasetGenerator.generate_sparse_data(n_samples, n_features)
            elif data_type == 'correlated':
                X, y, _ = DatasetGenerator.generate_correlated_data(n_samples, n_features)
            elif data_type == 'high_dim':
                X, y, _ = DatasetGenerator.generate_high_dimensional_data(n_samples, n_features)
            else:
                X = np.random.randn(n_samples, n_features)
                y = np.random.randn(n_samples)

            # Split train/test
            split_idx = int(0.8 * n_samples)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Run benchmarks
            self.benchmark_rkhs_methods(X_train, y_train, X_test, y_test, dataset_name)
            self.benchmark_proximal_methods(X_train, y_train, X_test, y_test, dataset_name)
            self.benchmark_privacy_methods(X_train, y_train, X_test, y_test, dataset_name)

    def save_results(self):
        """Sauvegarde les résultats."""
        # Convert to DataFrame
        df = pd.DataFrame([vars(r) for r in self.results])

        # Save CSV
        csv_path = os.path.join(self.output_dir, 'benchmark_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        # Save JSON
        json_path = os.path.join(self.output_dir, 'benchmark_results.json')
        with open(json_path, 'w') as f:
            json.dump([vars(r) for r in self.results], f, indent=2)

        return df

    def plot_results(self, df: pd.DataFrame):
        """Génère les visualisations des résultats."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Training Time Comparison
        ax = axes[0, 0]
        pivot = df.pivot_table(values='training_time', index='dataset', columns='method')
        pivot.plot(kind='bar', ax=ax)
        ax.set_title('Training Time Comparison')
        ax.set_ylabel('Time (seconds)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 2. Iterations to Convergence
        ax = axes[0, 1]
        pivot = df.pivot_table(values='iterations', index='dataset', columns='method')
        pivot.plot(kind='bar', ax=ax)
        ax.set_title('Iterations to Convergence')
        ax.set_ylabel('Iterations')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 3. MSE Performance
        ax = axes[0, 2]
        pivot = df.pivot_table(values='mse', index='dataset', columns='method')
        pivot.plot(kind='bar', ax=ax, logy=True)
        ax.set_title('MSE Performance (log scale)')
        ax.set_ylabel('MSE')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 4. Computational Savings
        ax = axes[1, 0]
        pivot = df.pivot_table(values='computational_savings', index='dataset', columns='method')
        pivot.plot(kind='bar', ax=ax)
        ax.set_title('Computational Savings')
        ax.set_ylabel('Savings (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 5. Memory Usage
        ax = axes[1, 1]
        pivot = df.pivot_table(values='memory_usage', index='dataset', columns='method')
        pivot.plot(kind='bar', ax=ax)
        ax.set_title('Memory Usage')
        ax.set_ylabel('Memory (MB)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 6. Method Comparison Heatmap
        ax = axes[1, 2]
        # Normaliser les métriques pour comparaison
        metrics = ['training_time', 'iterations', 'mse', 'memory_usage']
        method_scores = df.groupby('method')[metrics].mean()
        # Normaliser entre 0 et 1 (inverse pour time/iterations/mse car plus petit = mieux)
        for col in metrics:
            if col != 'computational_savings':
                method_scores[col] = 1 - (method_scores[col] - method_scores[col].min()) / (
                    method_scores[col].max() - method_scores[col].min())

        sns.heatmap(method_scores.T, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
        ax.set_title('Overall Method Performance\n(Higher is Better)')

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'benchmark_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {plot_path}")
        plt.show()

    def generate_report(self, df: pd.DataFrame):
        """Génère un rapport textuel des résultats."""
        report_path = os.path.join(self.output_dir, 'benchmark_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FRUGAL AI EARLY STOPPING - PERFORMANCE BENCHMARK REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Résumé global
            f.write("OVERALL SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total methods benchmarked: {df['method'].nunique()}\n")
            f.write(f"Total datasets tested: {df['dataset'].nunique()}\n")
            f.write(f"Total experiments: {len(df)}\n\n")

            # Meilleure méthode par métrique
            f.write("BEST METHODS BY METRIC\n")
            f.write("-" * 40 + "\n")

            metrics = {
                'training_time': 'Fastest Training',
                'iterations': 'Fewest Iterations',
                'mse': 'Best Accuracy (MSE)',
                'computational_savings': 'Best Computational Savings',
                'memory_usage': 'Lowest Memory Usage'
            }

            for metric, description in metrics.items():
                if metric == 'computational_savings':
                    best = df.loc[df[metric].idxmax()]
                else:
                    best = df.loc[df[metric].idxmin()]
                f.write(f"{description}: {best['method']} ({best[metric]:.4f})\n")

            f.write("\n")

            # Analyse par dataset
            f.write("ANALYSIS BY DATASET\n")
            f.write("-" * 40 + "\n")

            for dataset in df['dataset'].unique():
                f.write(f"\n{dataset}:\n")
                subset = df[df['dataset'] == dataset]

                # Top 3 méthodes par MSE
                top3 = subset.nsmallest(3, 'mse')[['method', 'mse', 'training_time', 'iterations']]
                f.write("  Top 3 methods by accuracy:\n")
                for _, row in top3.iterrows():
                    f.write(f"    - {row['method']}: MSE={row['mse']:.4f}, "
                           f"Time={row['training_time']:.2f}s, Iter={row['iterations']}\n")

            # Recommandations
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")

            # Analyser les trade-offs
            avg_by_method = df.groupby('method')[['mse', 'training_time', 'computational_savings']].mean()

            # Méthode avec meilleur trade-off
            avg_by_method['score'] = (
                (1 - avg_by_method['mse'] / avg_by_method['mse'].max()) * 0.4 +
                (1 - avg_by_method['training_time'] / avg_by_method['training_time'].max()) * 0.3 +
                avg_by_method['computational_savings'] * 0.3
            )

            best_overall = avg_by_method['score'].idxmax()
            f.write(f"Best overall method (accuracy/speed trade-off): {best_overall}\n")

            # Recommandations spécifiques
            f.write("\nContext-specific recommendations:\n")
            f.write("- For high accuracy requirements: RKHS-SDP or FISTA\n")
            f.write("- For fast training: RKHS-DP or ISTA\n")
            f.write("- For sparse solutions: FISTA-Strong\n")
            f.write("- For privacy constraints: DP-SGD with appropriate ε\n")

        print(f"Report saved to {report_path}")


def main():
    """Fonction principale pour exécuter les benchmarks."""
    benchmark = PerformanceBenchmark()

    # Run benchmarks
    benchmark.run_all_benchmarks()

    # Save and analyze results
    df = benchmark.save_results()

    # Generate visualizations
    benchmark.plot_results(df)

    # Generate report
    benchmark.generate_report(df)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE!")
    print("=" * 80)
    print(f"Results saved in: {benchmark.output_dir}/")
    print("Files generated:")
    print("  - benchmark_results.csv")
    print("  - benchmark_results.json")
    print("  - benchmark_plots.png")
    print("  - benchmark_report.txt")


if __name__ == "__main__":
    main()