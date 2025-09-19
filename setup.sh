#!/bin/bash
# 🚀 SETUP SCRIPT - FRUGAL AI EARLY STOPPING PROJECT

set -e

echo "═══════════════════════════════════════════════════"
echo "  🚀 FRUGAL AI EARLY STOPPING - SETUP AUTOMATIQUE"
echo "═══════════════════════════════════════════════════"
echo

# Vérifier Python
echo "📍 Vérification de Python..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Python n'est pas installé!"
    exit 1
fi

echo "✅ Python trouvé: $($PYTHON_CMD --version)"

# Créer environnement virtuel
echo
echo "📍 Création de l'environnement virtuel..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "✅ Environnement virtuel créé"
else
    echo "ℹ️  Environnement virtuel existe déjà"
fi

# Activer l'environnement
echo
echo "📍 Activation de l'environnement..."
source venv/bin/activate || source venv/Scripts/activate

# Installer les dépendances
echo
echo "📍 Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest pytest-benchmark matplotlib seaborn jupyter tqdm joblib

echo
echo "📍 Installation des outils de développement..."
pip install pylint flake8 black isort mypy

# Créer les dossiers manquants
echo
echo "📍 Création des dossiers..."
mkdir -p tests benchmarks results figures data

# Télécharger les papers si demandé
echo
read -p "Voulez-vous télécharger les articles de recherche? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📚 Téléchargement des articles..."
    $PYTHON_CMD download_papers.py
fi

# Créer un fichier de test basique
echo
echo "📍 Création des tests de base..."
cat > tests/test_basic.py << 'EOF'
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test que tous les modules peuvent être importés."""
    import rkhs_gradient_descent
    import dp_early_stopping
    import proximal_early_stopping
    import component_early_stopping
    import fairness_early_stopping
    assert True

def test_numpy():
    """Test NumPy fonctionne."""
    X = np.random.randn(10, 5)
    assert X.shape == (10, 5)
EOF

# Créer un notebook de démarrage
echo
echo "📍 Création du notebook de démarrage..."
cat > quick_start.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Frugal AI Early Stopping - Quick Start\n",
    "\n",
    "Notebook de démarrage rapide pour le projet."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.datasets import make_regression\n",
    "import sys\n",
    "sys.path.append('Implementations')\n",
    "\n",
    "print('Libraries loaded successfully!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Générer des données de test\n",
    "X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)\n",
    "print(f'Data shape: X={X.shape}, y={y.shape}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Lancer les tests
echo
echo "📍 Exécution des tests..."
pytest tests/test_basic.py -v || echo "⚠️ Certains tests ont échoué"

# Afficher le résumé
echo
echo "═══════════════════════════════════════════════════"
echo "  ✅ INSTALLATION TERMINÉE AVEC SUCCÈS!"
echo "═══════════════════════════════════════════════════"
echo
echo "📚 Structure créée:"
echo "  - venv/           : Environnement virtuel"
echo "  - tests/          : Tests unitaires"
echo "  - benchmarks/     : Benchmarks de performance"
echo "  - results/        : Résultats d'expériences"
echo "  - figures/        : Graphiques et visualisations"
echo "  - data/           : Datasets"
echo
echo "🚀 Pour commencer:"
echo "  1. Activer l'environnement: source venv/bin/activate"
echo "  2. Lancer Jupyter: jupyter notebook quick_start.ipynb"
echo "  3. Exécuter les tests: pytest tests/ -v"
echo
echo "💡 CLAUDE.md contient toutes les optimisations pour Claude Code!"
echo