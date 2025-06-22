"""
Bank Churn MLOps Package
This package contains modules for data preprocessing, model training, and inference
"""

__version__ = "1.0.0"
__author__ = "Princess-Flourish"

# Import main modules for easy access
from .data_preprocessing import DataPreprocessor
from .model_training import ChurnModelTrainer
from .model_inference import ChurnModelInference

__all__ = [
    'DataPreprocessor',
    'ChurnModelTrainer', 
    'ChurnModelInference'
]