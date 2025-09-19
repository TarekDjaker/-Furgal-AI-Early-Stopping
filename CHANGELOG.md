# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-19

### Added
- **Examples directory** with 4 working demonstration scripts:
  - `example_proximal.py` - Sparse regression with proximal gradient descent
  - `example_component.py` - Component-wise early stopping for neural networks
  - `example_fairness.py` - Fairness-aware early stopping with demographic parity
  - `example_dp.py` - Differential privacy with early stopping comparison
- **Test infrastructure** with basic smoke tests for all modules
- **PyPI-ready packaging**:
  - Modern `pyproject.toml` configuration
  - Compatible `setup.py` for older pip versions
  - Proper package metadata and dependencies
- **Documentation improvements**:
  - Fixed installation instructions with correct repository URL
  - Added quick start guide and usage examples
  - Created examples README with detailed usage instructions
- **Development tools**:
  - `.gitignore` file to exclude build artifacts
  - Changelog for tracking improvements

### Fixed
- **Code quality issues**:
  - Removed unused imports in all modules
  - Added missing newlines at end of files
  - Fixed relative imports in `__init__.py`
- **Stability improvements**:
  - Fixed step size in proximal example to prevent divergence
  - Improved error handling in test infrastructure

### Changed
- Updated GitHub Actions workflow to work with new structure
- Enhanced README with practical usage examples and installation guide

## [Unreleased]

### Planned Features (from todo.md)
- Sphinx/MkDocs documentation generation
- Unified CLI for all examples
- Noise estimation for ill-posed problems
- Multi-objective early stopping strategies
- Graph neural network support
- Federated learning extensions