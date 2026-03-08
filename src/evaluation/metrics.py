"""
Metrics Calculator

This module computes evaluation metrics for the signal prediction model.

Metrics include:
- Classification: Precision, Recall, F1, ROC-AUC, PR-AUC
- Financial: Average return, Win rate, Return spread
- Statistical: Confidence intervals, p-values

Usage:
    from src.evaluation.metrics import MetricsCalculator
    
    calculator = MetricsCalculator()
    metrics = calculator.compute_all(y_true, y_pred, y_proba)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class MetricsCalculator:
    """
    Calculates evaluation metrics for predictions.
    
    This class handles:
    - Classification metrics computation
    - Financial metrics computation
    - Confidence interval estimation
    - Results formatting
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        logger.info("MetricsCalculator initialized")
    
    def compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score,
            precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score,
            confusion_matrix, classification_report
        )
        
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        for average in ["micro", "macro", "weighted"]:
            metrics[f"precision_{average}"] = precision_score(
                y_true, y_pred, average=average, zero_division=0
            )
            metrics[f"recall_{average}"] = recall_score(
                y_true, y_pred, average=average, zero_division=0
            )
            metrics[f"f1_{average}"] = f1_score(
                y_true, y_pred, average=average, zero_division=0
            )
        
        # AUC metrics (if probabilities available)
        if y_proba is not None:
            try:
                # For multiclass, use OvR
                if len(y_proba.shape) > 1:
                    metrics["roc_auc_ovr"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr"
                    )
                else:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
        
        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def compute_financial_metrics(
        self,
        predictions: List[str],
        returns: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute financial metrics.
        
        Args:
            predictions: Predicted signals (bearish, bullish, neutral)
            returns: Actual returns following events
            labels: Optional true labels
            
        Returns:
            Dictionary of financial metrics
        """
        metrics = {}
        
        predictions = np.array(predictions)
        
        # Average return following signal
        bearish_mask = predictions == "bearish"
        bullish_mask = predictions == "bullish"
        neutral_mask = predictions == "neutral"
        
        if bearish_mask.sum() > 0:
            bearish_returns = returns[bearish_mask]
            metrics["avg_return_bearish"] = float(np.nanmean(bearish_returns))
            metrics["n_bearish"] = int(bearish_mask.sum())
        
        if bullish_mask.sum() > 0:
            bullish_returns = returns[bullish_mask]
            metrics["avg_return_bullish"] = float(np.nanmean(bullish_returns))
            metrics["n_bullish"] = int(bullish_mask.sum())
        
        if neutral_mask.sum() > 0:
            neutral_returns = returns[neutral_mask]
            metrics["avg_return_neutral"] = float(np.nanmean(neutral_returns))
            metrics["n_neutral"] = int(neutral_mask.sum())
        
        # Win rate (prediction matched return direction)
        correct_bearish = bearish_mask & (returns < 0)
        correct_bullish = bullish_mask & (returns > 0)
        
        total_signals = bearish_mask.sum() + bullish_mask.sum()
        if total_signals > 0:
            correct_signals = correct_bearish.sum() + correct_bullish.sum()
            metrics["win_rate"] = float(correct_signals / total_signals)
        
        # Return spread (difference between predicted up and down)
        if bearish_mask.sum() > 0 and bullish_mask.sum() > 0:
            metrics["return_spread"] = (
                metrics.get("avg_return_bullish", 0) - 
                metrics.get("avg_return_bearish", 0)
            )
        
        return metrics
    
    def compute_bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_func: callable,
        n_iterations: int = 1000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for a metric.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric_func: Function to compute metric
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level (e.g., 0.95)
            
        Returns:
            Tuple of (metric_value, lower_bound, upper_bound)
        """
        n = len(y_true)
        scores = []
        
        for _ in range(n_iterations):
            idx = np.random.choice(n, size=n, replace=True)
            score = metric_func(y_true[idx], y_pred[idx])
            scores.append(score)
        
        scores = np.array(scores)
        alpha = 1 - confidence_level
        lower = np.percentile(scores, alpha/2 * 100)
        upper = np.percentile(scores, (1 - alpha/2) * 100)
        
        return float(np.mean(scores)), float(lower), float(upper)
    
    def compute_signal_decay(
        self,
        events: List[Dict],
        predictions: List[str],
        horizons: List[int] = [1, 3, 5, 10]
    ) -> Dict[int, Dict]:
        """
        Compute signal performance at different time horizons.
        
        Args:
            events: Event dictionaries with return data
            predictions: Predicted signals
            horizons: Return horizons to analyze
            
        Returns:
            Dictionary mapping horizon to metrics
        """
        decay_metrics = {}
        
        for horizon in horizons:
            return_key = f"return_{horizon}d"
            returns = np.array([
                e.get(return_key, np.nan) for e in events
            ])
            
            valid_mask = ~np.isnan(returns)
            if valid_mask.sum() < 10:
                continue
            
            valid_returns = returns[valid_mask]
            valid_preds = np.array(predictions)[valid_mask]
            
            metrics = self.compute_financial_metrics(
                valid_preds.tolist(), valid_returns
            )
            decay_metrics[horizon] = metrics
        
        return decay_metrics
    
    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute all metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            returns: Optional returns for financial metrics
            
        Returns:
            Comprehensive metrics dictionary
        """
        results = {
            "classification": self.compute_classification_metrics(y_true, y_pred, y_proba),
        }
        
        if returns is not None:
            results["financial"] = self.compute_financial_metrics(
                y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                returns
            )
        
        return results
