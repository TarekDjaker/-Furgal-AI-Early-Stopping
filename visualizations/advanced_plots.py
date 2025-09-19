#!/usr/bin/env python3
"""
Visualisations avancées pour le projet Frugal AI Early Stopping.
Génère des graphiques publication-ready pour les articles et présentations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy import stats
import os
import sys
from typing import List, Tuple, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration globale des plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.spines.top': False,
    'axes.spines.right': False
})


class VisualizationTools:
    """Outils de visualisation pour les résultats d'early stopping."""

    def __init__(self, save_dir: str = "figures"):
        """Initialise les outils de visualisation."""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.colors = sns.color_palette("husl", 8)

    def plot_convergence_analysis(self, iterations_data: Dict[str, np.ndarray],
                                 loss_data: Dict[str, np.ndarray],
                                 title: str = "Convergence Analysis"):
        """
        Trace l'analyse de convergence pour différentes méthodes.

        Args:
            iterations_data: Dict avec clés = méthodes, valeurs = iterations
            loss_data: Dict avec clés = méthodes, valeurs = loss values
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Convergence principale
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        for i, (method, losses) in enumerate(loss_data.items()):
            iterations = iterations_data[method]
            ax1.semilogy(iterations, losses, '-', linewidth=2.5,
                        label=method, color=self.colors[i % len(self.colors)])

        ax1.set_xlabel('Iterations', fontsize=14)
        ax1.set_ylabel('Loss (log scale)', fontsize=14)
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 2. Taux de convergence
        ax2 = fig.add_subplot(gs[0, 2])
        convergence_rates = {}
        for method, losses in loss_data.items():
            if len(losses) > 10:
                # Calculer le taux de convergence (pente en log)
                log_losses = np.log(losses[losses > 0])
                if len(log_losses) > 10:
                    rate = -np.polyfit(range(len(log_losses[:50])), log_losses[:50], 1)[0]
                    convergence_rates[method] = rate

        methods = list(convergence_rates.keys())
        rates = list(convergence_rates.values())
        bars = ax2.bar(range(len(methods)), rates, color=self.colors[:len(methods)])
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.set_ylabel('Convergence Rate', fontsize=12)
        ax2.set_title('Convergence Speed', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Temps jusqu'à convergence
        ax3 = fig.add_subplot(gs[1, 2])
        stopping_times = {}
        threshold = 1e-3

        for method, losses in loss_data.items():
            converged_idx = np.where(losses < threshold)[0]
            if len(converged_idx) > 0:
                stopping_times[method] = iterations_data[method][converged_idx[0]]
            else:
                stopping_times[method] = iterations_data[method][-1]

        methods = list(stopping_times.keys())
        times = list(stopping_times.values())
        bars = ax3.barh(range(len(methods)), times, color=self.colors[:len(methods)])
        ax3.set_yticks(range(len(methods)))
        ax3.set_yticklabels(methods)
        ax3.set_xlabel('Iterations to Threshold', fontsize=12)
        ax3.set_title(f'Time to Loss < {threshold}', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. Variance de la loss
        ax4 = fig.add_subplot(gs[2, 0])
        for i, (method, losses) in enumerate(loss_data.items()):
            if len(losses) > 20:
                # Calculer la variance glissante
                window_size = 10
                variances = pd.Series(losses).rolling(window_size).std()
                ax4.plot(iterations_data[method][window_size-1:],
                        variances[window_size-1:],
                        label=method, color=self.colors[i % len(self.colors)])

        ax4.set_xlabel('Iterations', fontsize=12)
        ax4.set_ylabel('Loss Variance', fontsize=12)
        ax4.set_title('Stability Analysis', fontsize=14)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        # 5. Efficacité computationnelle
        ax5 = fig.add_subplot(gs[2, 1])
        efficiency_data = []
        for method in loss_data.keys():
            final_loss = loss_data[method][-1]
            iterations_used = len(loss_data[method])
            efficiency = -np.log(final_loss) / iterations_used if final_loss > 0 else 0
            efficiency_data.append(efficiency)

        ax5.scatter(range(len(methods)), efficiency_data,
                   s=200, c=self.colors[:len(methods)], alpha=0.7, edgecolors='black')
        ax5.set_xticks(range(len(methods)))
        ax5.set_xticklabels(methods, rotation=45, ha='right')
        ax5.set_ylabel('Efficiency Score', fontsize=12)
        ax5.set_title('Computational Efficiency', fontsize=14)
        ax5.grid(True, alpha=0.3)

        # 6. Boxplot des dernières valeurs
        ax6 = fig.add_subplot(gs[2, 2])
        final_losses = []
        labels = []
        for method, losses in loss_data.items():
            if len(losses) > 20:
                final_losses.append(losses[-20:])  # 20 dernières valeurs
                labels.append(method)

        bp = ax6.boxplot(final_losses, labels=labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(self.colors[i % len(self.colors)])
        ax6.set_ylabel('Final Loss Distribution', fontsize=12)
        ax6.set_title('Final Performance', fontsize=14)
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Comprehensive Convergence Analysis', fontsize=18, fontweight='bold', y=1.02)

        # Sauvegarder
        save_path = os.path.join(self.save_dir, 'convergence_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Convergence analysis saved to: {save_path}")

    def plot_early_stopping_decision(self, iterations: np.ndarray,
                                    train_loss: np.ndarray,
                                    val_loss: np.ndarray,
                                    stopping_point: int,
                                    method_name: str = "Method"):
        """
        Visualise la décision d'early stopping avec train/validation curves.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Train vs Validation Loss
        ax = axes[0, 0]
        ax.plot(iterations, train_loss, 'b-', linewidth=2, label='Training Loss')
        ax.plot(iterations, val_loss, 'r-', linewidth=2, label='Validation Loss')
        ax.axvline(x=stopping_point, color='green', linestyle='--', linewidth=2,
                  label=f'Early Stop (iter={stopping_point})')
        ax.fill_between(iterations[stopping_point:], 0, max(max(train_loss), max(val_loss)),
                        alpha=0.2, color='gray', label='Saved Computation')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_title(f'{method_name}: Early Stopping Decision')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Generalization Gap
        ax = axes[0, 1]
        gap = val_loss - train_loss
        ax.plot(iterations, gap, 'purple', linewidth=2)
        ax.axvline(x=stopping_point, color='green', linestyle='--', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.fill_between(iterations, 0, gap, where=(gap > 0),
                        alpha=0.3, color='red', label='Overfitting')
        ax.fill_between(iterations, 0, gap, where=(gap <= 0),
                        alpha=0.3, color='blue', label='Underfitting')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Validation - Training Loss')
        ax.set_title('Generalization Gap')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Loss Derivative (Smoothed)
        ax = axes[1, 0]
        window = 5
        train_grad = np.gradient(train_loss)
        val_grad = np.gradient(val_loss)
        train_grad_smooth = pd.Series(train_grad).rolling(window, center=True).mean()
        val_grad_smooth = pd.Series(val_grad).rolling(window, center=True).mean()

        ax.plot(iterations, train_grad_smooth, 'b-', alpha=0.7, label='Train Gradient')
        ax.plot(iterations, val_grad_smooth, 'r-', alpha=0.7, label='Val Gradient')
        ax.axvline(x=stopping_point, color='green', linestyle='--', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss Gradient')
        ax.set_title('Rate of Change')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Cumulative Computation Saved
        ax = axes[1, 1]
        total_iterations = len(iterations)
        saved_iterations = total_iterations - stopping_point
        savings_percent = (saved_iterations / total_iterations) * 100

        # Pie chart
        sizes = [stopping_point, saved_iterations]
        labels = ['Used Iterations', 'Saved Iterations']
        colors = ['#3498db', '#2ecc71']
        explode = (0, 0.1)

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          explode=explode, autopct='%1.1f%%',
                                          shadow=True, startangle=90)
        ax.set_title(f'Computational Savings: {savings_percent:.1f}%')

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)

        plt.suptitle(f'Early Stopping Analysis - {method_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Sauvegarder
        save_path = os.path.join(self.save_dir, f'early_stopping_decision_{method_name.lower()}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Early stopping decision plot saved to: {save_path}")

    def plot_sparsity_patterns(self, coefficients_dict: Dict[str, np.ndarray],
                              feature_names: List[str] = None):
        """
        Visualise les patterns de sparsité pour différentes méthodes de régularisation.
        """
        n_methods = len(coefficients_dict)
        fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))

        if n_methods == 1:
            axes = axes.reshape(-1, 1)

        for idx, (method, coeffs) in enumerate(coefficients_dict.items()):
            # 1. Bar plot des coefficients
            ax = axes[0, idx]
            n_features = len(coeffs)
            if feature_names is None:
                feature_names = [f'F{i}' for i in range(n_features)]

            # Colorer différemment les coefficients nuls et non-nuls
            colors = ['red' if abs(c) < 1e-6 else 'blue' for c in coeffs]
            bars = ax.bar(range(n_features), coeffs, color=colors)

            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Coefficient Value')
            ax.set_title(f'{method}\nSparsity: {np.mean(np.abs(coeffs) < 1e-6):.1%}')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='y')

            # Rotation des labels si beaucoup de features
            if n_features > 20:
                ax.set_xticks(range(0, n_features, max(1, n_features//10)))
                ax.set_xticklabels([f'{i}' for i in range(0, n_features, max(1, n_features//10))])

            # 2. Heatmap de sparsité
            ax = axes[1, idx]
            # Reshape coefficients pour heatmap (faire une grille carrée approximative)
            grid_size = int(np.ceil(np.sqrt(n_features)))
            coeffs_padded = np.pad(coeffs, (0, grid_size**2 - n_features), constant_values=0)
            coeffs_grid = coeffs_padded.reshape(grid_size, grid_size)

            # Créer une matrice binaire pour la sparsité
            sparsity_grid = np.abs(coeffs_grid) > 1e-6

            im = ax.imshow(sparsity_grid, cmap='RdBu_r', aspect='auto')
            ax.set_title(f'Sparsity Pattern\n{np.sum(np.abs(coeffs) > 1e-6)}/{n_features} active')
            ax.set_xlabel('Feature Grid X')
            ax.set_ylabel('Feature Grid Y')

            # Ajouter une colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Active (1) / Zero (0)')

        plt.suptitle('Sparsity Analysis Across Methods', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Sauvegarder
        save_path = os.path.join(self.save_dir, 'sparsity_patterns.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Sparsity patterns saved to: {save_path}")

    def plot_computational_savings_heatmap(self, datasets: List[str],
                                          methods: List[str],
                                          savings_matrix: np.ndarray):
        """
        Crée une heatmap des économies computationnelles.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Créer la heatmap
        im = ax.imshow(savings_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Configurer les ticks
        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_xticklabels(methods)
        ax.set_yticklabels(datasets)

        # Rotation des labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Ajouter les valeurs dans les cellules
        for i in range(len(datasets)):
            for j in range(len(methods)):
                text = ax.text(j, i, f'{savings_matrix[i, j]:.1%}',
                             ha="center", va="center", color="black", fontweight='bold')

        ax.set_title('Computational Savings Across Datasets and Methods',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Early Stopping Method', fontsize=14)
        ax.set_ylabel('Dataset', fontsize=14)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Computational Savings (%)', rotation=270, labelpad=20, fontsize=12)

        # Ajouter des lignes de grille
        ax.set_xticks(np.arange(len(methods)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(datasets)+1)-.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=2)

        plt.tight_layout()

        # Sauvegarder
        save_path = os.path.join(self.save_dir, 'computational_savings_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Computational savings heatmap saved to: {save_path}")

    def plot_privacy_utility_frontier(self, epsilon_values: np.ndarray,
                                     accuracies: Dict[str, np.ndarray]):
        """
        Trace la frontière privacy-utility pour différentes méthodes DP.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Accuracy vs Privacy Budget
        ax = axes[0]
        for method, acc in accuracies.items():
            ax.semilogx(epsilon_values, acc, 'o-', linewidth=2, markersize=8, label=method)

        ax.set_xlabel('Privacy Budget ε (log scale)', fontsize=14)
        ax.set_ylabel('Test Accuracy', fontsize=14)
        ax.set_title('Privacy-Utility Trade-off', fontsize=16)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Ajouter des zones de privacy
        ax.axvspan(0.01, 0.1, alpha=0.1, color='red', label='High Privacy')
        ax.axvspan(0.1, 1.0, alpha=0.1, color='yellow', label='Medium Privacy')
        ax.axvspan(1.0, 10.0, alpha=0.1, color='green', label='Low Privacy')

        # 2. Utility Loss vs Privacy
        ax = axes[1]
        for method, acc in accuracies.items():
            utility_loss = 1 - acc / acc[-1]  # Normaliser par non-private accuracy
            ax.loglog(epsilon_values, utility_loss, 'o-', linewidth=2, markersize=8, label=method)

        ax.set_xlabel('Privacy Budget ε (log scale)', fontsize=14)
        ax.set_ylabel('Utility Loss (log scale)', fontsize=14)
        ax.set_title('Utility Loss Analysis', fontsize=16)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, which="both")

        # 3. Pareto Frontier
        ax = axes[2]
        # Pour chaque méthode, tracer privacy (1/ε) vs accuracy
        for method, acc in accuracies.items():
            privacy_level = 1 / epsilon_values  # Plus ε est petit, plus la privacy est forte
            ax.scatter(privacy_level, acc, s=100, label=method, alpha=0.7)

            # Tracer la frontière de Pareto approximative
            # Trier par privacy croissante
            sorted_idx = np.argsort(privacy_level)
            ax.plot(privacy_level[sorted_idx], acc[sorted_idx], '--', alpha=0.5)

        ax.set_xscale('log')
        ax.set_xlabel('Privacy Level (1/ε)', fontsize=14)
        ax.set_ylabel('Test Accuracy', fontsize=14)
        ax.set_title('Privacy-Utility Pareto Frontier', fontsize=16)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Differential Privacy Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()

        # Sauvegarder
        save_path = os.path.join(self.save_dir, 'privacy_utility_frontier.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Privacy-utility frontier saved to: {save_path}")

    def create_publication_figure(self, results_df: pd.DataFrame):
        """
        Crée une figure de qualité publication avec tous les résultats clés.
        """
        # Configuration pour publication
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

        # Définir une palette de couleurs cohérente
        colors = sns.color_palette("Set2", 8)

        # Panel A: Convergence Comparison
        ax_a = fig.add_subplot(gs[0, :2])
        methods = results_df['method'].unique()[:5]
        for i, method in enumerate(methods):
            data = results_df[results_df['method'] == method]
            ax_a.plot(data['iterations'], data['loss'], linewidth=2,
                     label=method, color=colors[i])
        ax_a.set_xlabel('Iterations')
        ax_a.set_ylabel('Loss')
        ax_a.set_title('(A) Convergence Comparison')
        ax_a.legend(loc='upper right', ncol=2)
        ax_a.grid(True, alpha=0.3)

        # Panel B: Computational Savings
        ax_b = fig.add_subplot(gs[0, 2:])
        savings_data = results_df.groupby('method')['computational_savings'].mean()
        bars = ax_b.bar(range(len(savings_data)), savings_data.values, color=colors[:len(savings_data)])
        ax_b.set_xticks(range(len(savings_data)))
        ax_b.set_xticklabels(savings_data.index, rotation=45, ha='right')
        ax_b.set_ylabel('Average Savings (%)')
        ax_b.set_title('(B) Computational Efficiency')
        ax_b.set_ylim([0, 1])
        ax_b.grid(True, alpha=0.3, axis='y')

        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, savings_data.values):
            ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{value:.1%}', ha='center', va='bottom', fontsize=8)

        # Panel C: Accuracy Comparison
        ax_c = fig.add_subplot(gs[1, :2])
        accuracy_pivot = results_df.pivot_table(values='test_accuracy',
                                               index='dataset',
                                               columns='method')
        accuracy_pivot.plot(kind='bar', ax=ax_c, color=colors[:len(accuracy_pivot.columns)])
        ax_c.set_xlabel('Dataset')
        ax_c.set_ylabel('Test Accuracy')
        ax_c.set_title('(C) Accuracy Across Datasets')
        ax_c.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax_c.tick_params(axis='x', rotation=45)
        ax_c.grid(True, alpha=0.3, axis='y')

        # Panel D: Sparsity Levels
        ax_d = fig.add_subplot(gs[1, 2:])
        sparsity_data = results_df[results_df['sparsity'].notna()]
        if not sparsity_data.empty:
            pivot = sparsity_data.pivot_table(values='sparsity',
                                             index='lambda_reg',
                                             columns='method')
            for col in pivot.columns:
                ax_d.semilogx(pivot.index, pivot[col], 'o-', label=col, markersize=8)
            ax_d.set_xlabel('Regularization λ')
            ax_d.set_ylabel('Sparsity Level')
            ax_d.set_title('(D) Sparsity vs Regularization')
            ax_d.legend()
            ax_d.grid(True, alpha=0.3)

        # Panel E: Scalability
        ax_e = fig.add_subplot(gs[2, :2])
        scalability_data = results_df[results_df['n_samples'].notna()]
        if not scalability_data.empty:
            for method in scalability_data['method'].unique()[:3]:
                data = scalability_data[scalability_data['method'] == method]
                ax_e.plot(data['n_samples'], data['training_time'], 'o-', label=method, markersize=6)
            ax_e.set_xlabel('Number of Samples')
            ax_e.set_ylabel('Training Time (s)')
            ax_e.set_title('(E) Scalability Analysis')
            ax_e.legend()
            ax_e.grid(True, alpha=0.3)

        # Panel F: Method Ranking
        ax_f = fig.add_subplot(gs[2, 2:])
        # Créer un score composite
        ranking_data = results_df.groupby('method').agg({
            'test_accuracy': 'mean',
            'training_time': 'mean',
            'computational_savings': 'mean'
        })

        # Normaliser et calculer le score
        ranking_data['accuracy_score'] = ranking_data['test_accuracy'] / ranking_data['test_accuracy'].max()
        ranking_data['speed_score'] = 1 - (ranking_data['training_time'] / ranking_data['training_time'].max())
        ranking_data['overall_score'] = (ranking_data['accuracy_score'] * 0.5 +
                                        ranking_data['speed_score'] * 0.3 +
                                        ranking_data['computational_savings'] * 0.2)

        ranking_data = ranking_data.sort_values('overall_score', ascending=True)
        y_pos = np.arange(len(ranking_data))
        bars = ax_f.barh(y_pos, ranking_data['overall_score'], color=colors[:len(ranking_data)])
        ax_f.set_yticks(y_pos)
        ax_f.set_yticklabels(ranking_data.index)
        ax_f.set_xlabel('Overall Performance Score')
        ax_f.set_title('(F) Method Ranking')
        ax_f.set_xlim([0, 1])
        ax_f.grid(True, alpha=0.3, axis='x')

        # Titre principal
        plt.suptitle('Frugal AI Early Stopping: Comprehensive Results',
                    fontsize=16, fontweight='bold', y=1.02)

        # Sauvegarder en haute qualité pour publication
        save_path = os.path.join(self.save_dir, 'publication_figure.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')

        save_path_png = os.path.join(self.save_dir, 'publication_figure.png')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight', format='png')

        plt.show()

        print(f"Publication figure saved to: {save_path} and {save_path_png}")


def demo_visualizations():
    """Fonction de démonstration des visualisations."""
    viz = VisualizationTools()

    # Générer des données de démonstration
    np.random.seed(42)
    n_iterations = 100

    # 1. Données de convergence
    methods = ['DP', 'SDP', 'CV', 'FISTA', 'ISTA']
    iterations_data = {}
    loss_data = {}

    for i, method in enumerate(methods):
        iterations_data[method] = np.arange(n_iterations)
        # Simuler différentes vitesses de convergence
        rate = 0.05 * (1 + i * 0.3)
        loss_data[method] = np.exp(-rate * np.arange(n_iterations)) + 0.01 * np.random.randn(n_iterations)

    viz.plot_convergence_analysis(iterations_data, loss_data)

    # 2. Early Stopping Decision
    train_loss = np.exp(-0.05 * np.arange(n_iterations)) + 0.01 * np.random.randn(n_iterations)
    val_loss = train_loss + 0.05 * np.sin(0.1 * np.arange(n_iterations)) + 0.02 * np.random.randn(n_iterations)
    stopping_point = 40

    viz.plot_early_stopping_decision(np.arange(n_iterations), train_loss, val_loss,
                                    stopping_point, "Discrepancy Principle")

    # 3. Sparsity Patterns
    coefficients = {
        'L1': np.random.randn(50) * (np.random.random(50) > 0.7),
        'L2': np.random.randn(50) * 0.5,
        'Elastic': np.random.randn(50) * (np.random.random(50) > 0.5) * 0.7
    }
    viz.plot_sparsity_patterns(coefficients)

    # 4. Computational Savings Heatmap
    datasets = ['California', 'Diabetes', 'Synthetic', 'High-Dim']
    methods = ['DP', 'SDP', 'CV-5', 'Fixed-100', 'FISTA']
    savings_matrix = np.random.uniform(0.3, 0.9, (len(datasets), len(methods)))

    viz.plot_computational_savings_heatmap(datasets, methods, savings_matrix)

    # 5. Privacy-Utility Frontier
    epsilon_values = np.logspace(-2, 1, 10)
    accuracies = {
        'DP-SGD': 0.95 - 0.3 * np.exp(-epsilon_values),
        'DP-Adam': 0.93 - 0.25 * np.exp(-epsilon_values),
        'Gaussian Mech': 0.90 - 0.35 * np.exp(-epsilon_values)
    }
    viz.plot_privacy_utility_frontier(epsilon_values, accuracies)

    print("\nAll demonstration visualizations completed!")


if __name__ == "__main__":
    demo_visualizations()