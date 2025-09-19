# Makefile for Frugal AI Early Stopping Project

.PHONY: help setup install test benchmark experiments clean docs all

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
VENV := venv

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Frugal AI Early Stopping - Makefile Commands$(NC)"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Complete project setup
	@echo "$(GREEN)Setting up Frugal AI project...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
		echo "$(GREEN)✓ Virtual environment created$(NC)"; \
	fi
	@. $(VENV)/bin/activate && $(PIP) install --upgrade pip
	@. $(VENV)/bin/activate && $(PIP) install -r requirements.txt
	@. $(VENV)/bin/activate && $(PIP) install pytest pytest-benchmark matplotlib seaborn jupyter tqdm joblib
	@. $(VENV)/bin/activate && $(PIP) install pylint flake8 black isort mypy
	@mkdir -p tests benchmarks results figures data experiments/results
	@echo "$(GREEN)✓ Setup complete!$(NC)"

install: ## Install dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	@. $(VENV)/bin/activate && $(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	@. $(VENV)/bin/activate && $(PYTEST) tests/ -v --tb=short
	@echo "$(GREEN)✓ Tests completed$(NC)"

test-coverage: ## Run tests with coverage
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	@. $(VENV)/bin/activate && $(PYTEST) tests/ --cov=. --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) benchmarks/performance_benchmarks.py
	@echo "$(GREEN)✓ Benchmarks completed$(NC)"

experiments: ## Run all experiments
	@echo "$(GREEN)Running experiments...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) experiments/run_experiments.py --experiments all
	@echo "$(GREEN)✓ Experiments completed$(NC)"

exp-quick: ## Run quick experiments (subset)
	@echo "$(GREEN)Running quick experiments...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) experiments/run_experiments.py --experiments early_stopping proximal
	@echo "$(GREEN)✓ Quick experiments completed$(NC)"

visualize: ## Generate all visualizations
	@echo "$(GREEN)Generating visualizations...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) visualizations/advanced_plots.py
	@echo "$(GREEN)✓ Visualizations saved in figures/$(NC)"

lint: ## Run code linting
	@echo "$(GREEN)Running linters...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) -m flake8 *.py --max-line-length=100 --exclude=$(VENV)
	@. $(VENV)/bin/activate && $(PYTHON) -m pylint *.py --disable=C0103,C0114,C0115,C0116
	@echo "$(GREEN)✓ Linting completed$(NC)"

format: ## Format code with black
	@echo "$(GREEN)Formatting code...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) -m black *.py tests/*.py benchmarks/*.py experiments/*.py
	@. $(VENV)/bin/activate && $(PYTHON) -m isort *.py tests/*.py benchmarks/*.py experiments/*.py
	@echo "$(GREEN)✓ Code formatted$(NC)"

clean: ## Clean temporary files
	@echo "$(YELLOW)Cleaning temporary files...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name ".DS_Store" -delete
	@rm -rf .pytest_cache
	@rm -rf htmlcov
	@rm -rf *.egg-info
	@echo "$(GREEN)✓ Cleanup completed$(NC)"

clean-all: clean ## Clean everything including venv
	@echo "$(RED)Removing virtual environment...$(NC)"
	@rm -rf $(VENV)
	@rm -rf results/* figures/* benchmark_results/* experiment_results/*
	@echo "$(GREEN)✓ Full cleanup completed$(NC)"

notebook: ## Start Jupyter notebook
	@echo "$(GREEN)Starting Jupyter notebook...$(NC)"
	@. $(VENV)/bin/activate && jupyter notebook

download-papers: ## Download research papers
	@echo "$(GREEN)Downloading research papers...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) download_papers.py
	@echo "$(GREEN)✓ Papers downloaded$(NC)"

check: lint test ## Run lint and tests
	@echo "$(GREEN)✓ All checks passed!$(NC)"

commit-ready: format lint test ## Prepare for commit
	@echo "$(GREEN)✓ Code is ready for commit!$(NC)"

docs: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) -m pydoc -w *.py
	@mkdir -p docs
	@mv *.html docs/
	@echo "$(GREEN)✓ Documentation generated in docs/$(NC)"

report: ## Generate final report
	@echo "$(GREEN)Generating final report...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) -c "from experiments.run_experiments import ExperimentRunner; runner = ExperimentRunner(); runner.generate_final_report()"
	@echo "$(GREEN)✓ Report generated$(NC)"

all: setup test benchmark experiments visualize ## Run everything
	@echo "$(GREEN)✓ All tasks completed successfully!$(NC)"

# Docker commands (if needed)
docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	@docker build -t frugal-ai-early-stopping .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run in Docker container
	@echo "$(GREEN)Running in Docker...$(NC)"
	@docker run -it --rm -v $(PWD):/app frugal-ai-early-stopping

# Git helpers
git-status: ## Show git status
	@git status -sb

git-log: ## Show git log
	@git log --oneline --graph --decorate -10

# Development helpers
watch-tests: ## Watch and run tests on file changes
	@echo "$(GREEN)Watching for file changes...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) -m pytest-watch tests/

profile: ## Profile the code
	@echo "$(GREEN)Profiling code...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) -m cProfile -s cumulative experiments/run_experiments.py --experiments early_stopping > profile_results.txt
	@echo "$(GREEN)✓ Profile results saved to profile_results.txt$(NC)"

# Info commands
info: ## Show project information
	@echo "$(GREEN)Frugal AI Early Stopping Project$(NC)"
	@echo "================================="
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Project structure:"
	@tree -L 2 -I '__pycache__|*.pyc|venv' 2>/dev/null || ls -la
	@echo ""
	@echo "Key files:"
	@wc -l *.py 2>/dev/null | tail -1

stats: ## Show code statistics
	@echo "$(GREEN)Code Statistics$(NC)"
	@echo "==============="
	@echo "Total Python files: $$(find . -name "*.py" -not -path "./$(VENV)/*" | wc -l)"
	@echo "Total lines of code: $$(find . -name "*.py" -not -path "./$(VENV)/*" -exec wc -l {} + | tail -1)"
	@echo "Total tests: $$(grep -r "def test_" tests/*.py | wc -l)"
	@echo ""
	@echo "File breakdown:"
	@find . -name "*.py" -not -path "./$(VENV)/*" -exec wc -l {} + | sort -rn | head -10

.DEFAULT_GOAL := help