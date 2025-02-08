# src/__init__.py

# Import core functions and classes to make them accessible directly
from .preprocess import load_data, preprocess_data, augment_data, adversarial_test
from .train import train_model
from .evaluate import evaluate_model, analyze_confidence
from .model import build_model
from .optimize import optimize_hyperparameters
from .baseline import baseline_model

# Optional: Initialize logging for the package
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Package initialized: src")
