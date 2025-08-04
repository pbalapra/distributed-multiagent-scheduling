"""
Setup script for distributed multi-agent scheduling package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="distributed-multiagent-scheduling",
    version="1.0.0",
    author="Research Team",
    author_email="authors@institution.edu",
    description="Distributed Multi-Agent Scheduling for Resilient High-Performance Computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/distributed-multiagent-scheduling",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "analysis": [
            "jupyter>=1.0.0",
            "scipy>=1.8.0",
            "scikit-learn>=1.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-resilience-eval=evaluation.systematic_resilience_evaluation:main",
            "run-quick-eval=evaluation.quick_resilience_test:main",
            "generate-figures=create_bw_publication_figures:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt", "*.png", "*.pdf"],
    },
    project_urls={
        "Bug Reports": "https://github.com/username/distributed-multiagent-scheduling/issues",
        "Source": "https://github.com/username/distributed-multiagent-scheduling",
        "Documentation": "https://distributed-multiagent-scheduling.readthedocs.io/",
    },
)