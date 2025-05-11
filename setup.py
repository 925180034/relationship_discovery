# setup.py
from setuptools import setup, find_packages

setup(
    name="schema_matching",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "transformers",
        "sentence-transformers",
        "pytest",
        "scikit-learn",
        "requests>=2.25.1",
        "scipy"
    ]
)