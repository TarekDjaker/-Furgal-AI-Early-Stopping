"""Setup script for the Early Stopping Extensions package."""

from setuptools import setup
import os

# Read the README file
current_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(current_dir, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="furgal-ai-early-stopping",
    version="0.1.0",
    author="TarekDjaker",
    author_email="",  # Add email if available
    description="Early Stopping Extensions for Iterative and Modern Machine Learning Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TarekDjaker/-Furgal-AI-Early-Stopping",
    py_modules=[
        "proximal_early_stopping",
        "component_early_stopping", 
        "fairness_early_stopping",
        "dp_early_stopping",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=1.0",
    ],
    extras_require={
        "torch": ["torch>=2.0"],
        "examples": ["matplotlib"],
        "dev": [
            "pytest",
            "flake8",
            "matplotlib",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine-learning early-stopping optimization fairness privacy",
    project_urls={
        "Documentation": "https://github.com/TarekDjaker/-Furgal-AI-Early-Stopping#readme",
        "Source": "https://github.com/TarekDjaker/-Furgal-AI-Early-Stopping",
        "Tracker": "https://github.com/TarekDjaker/-Furgal-AI-Early-Stopping/issues",
    },
)