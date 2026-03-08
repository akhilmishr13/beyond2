"""
Evaluation Module

This module provides comprehensive evaluation utilities for the
signal prediction model:

- Classification metrics (precision, recall, F1, ROC-AUC)
- Financial metrics (returns, win rate)
- Signal decay analysis
- Robustness testing
- Calibration analysis
- Statistical significance testing

Example:
    from src.evaluation import MetricsCalculator, RobustnessAnalyzer
    
    calculator = MetricsCalculator()
    metrics = calculator.compute_all(y_true, y_pred, y_proba)
"""

from src.evaluation.metrics import MetricsCalculator
from src.evaluation.temporal_split import TemporalSplitter
from src.evaluation.robustness import RobustnessAnalyzer
from src.evaluation.calibration import CalibrationAnalyzer

__all__ = [
    "MetricsCalculator",
    "TemporalSplitter",
    "RobustnessAnalyzer",
    "CalibrationAnalyzer",
]
