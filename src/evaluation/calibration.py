"""
Calibration Analyzer

This module analyzes model calibration - whether predicted
probabilities match actual frequencies.

A well-calibrated model's 80% predictions should be correct 80% of the time.

Usage:
    from src.evaluation.calibration import CalibrationAnalyzer
    
    analyzer = CalibrationAnalyzer()
    metrics = analyzer.compute_calibration(y_true, y_proba)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class CalibrationAnalyzer:
    """
    Analyzes prediction calibration.
    
    This class computes:
    - Brier score
    - Expected Calibration Error (ECE)
    - Reliability diagram data
    
    Attributes:
        n_bins: Number of bins for reliability diagram
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize the calibration analyzer.
        
        Args:
            n_bins: Number of bins for analysis
        """
        self.n_bins = n_bins
        logger.info(f"CalibrationAnalyzer initialized with {n_bins} bins")
    
    def compute_brier_score(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> float:
        """
        Compute Brier score.
        
        Lower is better. Perfect calibration = 0.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            
        Returns:
            Brier score
        """
        return float(np.mean((y_proba - y_true) ** 2))
    
    def compute_ece(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        ECE measures the difference between predicted confidence
        and actual accuracy across bins.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            
        Returns:
            ECE value
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        
        for i in range(self.n_bins):
            bin_mask = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i + 1])
            
            if bin_mask.sum() == 0:
                continue
            
            bin_confidence = y_proba[bin_mask].mean()
            bin_accuracy = y_true[bin_mask].mean()
            bin_weight = bin_mask.sum() / len(y_true)
            
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return float(ece)
    
    def compute_reliability_diagram(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, List]:
        """
        Compute data for reliability diagram.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary with bin data
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        
        confidences = []
        accuracies = []
        counts = []
        
        for i in range(self.n_bins):
            bin_mask = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i + 1])
            
            if bin_mask.sum() == 0:
                confidences.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                accuracies.append(0)
                counts.append(0)
            else:
                confidences.append(float(y_proba[bin_mask].mean()))
                accuracies.append(float(y_true[bin_mask].mean()))
                counts.append(int(bin_mask.sum()))
        
        return {
            "bin_confidences": confidences,
            "bin_accuracies": accuracies,
            "bin_counts": counts,
            "bin_edges": bin_boundaries.tolist(),
        }
    
    def compute_calibration(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """
        Compute all calibration metrics.
        
        Args:
            y_true: True labels (binary)
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of calibration metrics
        """
        return {
            "brier_score": self.compute_brier_score(y_true, y_proba),
            "ece": self.compute_ece(y_true, y_proba),
            "reliability_diagram": self.compute_reliability_diagram(y_true, y_proba),
        }
    
    def compute_multiclass_calibration(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_classes: int = 3
    ) -> Dict:
        """
        Compute calibration for multiclass predictions.
        
        Args:
            y_true: True labels (integers)
            y_proba: Probability matrix (n_samples x n_classes)
            n_classes: Number of classes
            
        Returns:
            Dictionary with per-class and overall calibration
        """
        results = {}
        
        # Per-class calibration
        for c in range(n_classes):
            y_true_binary = (y_true == c).astype(int)
            y_proba_class = y_proba[:, c]
            
            results[f"class_{c}"] = self.compute_calibration(y_true_binary, y_proba_class)
        
        # Overall (using max probability)
        y_pred = np.argmax(y_proba, axis=1)
        y_proba_max = np.max(y_proba, axis=1)
        y_correct = (y_pred == y_true).astype(int)
        
        results["overall"] = self.compute_calibration(y_correct, y_proba_max)
        
        return results
