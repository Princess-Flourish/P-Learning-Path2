"""
Setup script for the bank churn MLOps project
This file helps install the project as a package so modules can be imported easily
"""

from setuptools import setup, find_packages

setup(
    name="bank_churn_mlops",
    version="1.0.0",
    author="Princess-Flourish",
    author_email="ptangban@bluechiptech.biz",
    description="Bank Customer Churn Prediction MLOps Pipeline",
    long_description="A complete MLOps pipeline using MLflow and Airflow for bank customer churn prediction",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "mlflow>=2.7.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
        "requests>=2.31.0",
        # Note: Airflow removed for Windows compatibility
        # Will be handled via Docker
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)