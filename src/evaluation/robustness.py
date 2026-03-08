"""
Robustness Analyzer

This module tests model robustness across different data slices:
- By document type (SEC vs news)
- By topic category
- By time period (drift analysis)

Usage:
    from src.evaluation.robustness import RobustnessAnalyzer
    
    analyzer = RobustnessAnalyzer()
    results = analyzer.analyze_by_topic(events, predictions, labels)
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.metrics import MetricsCalculator


class RobustnessAnalyzer:
    """
    Analyzes model robustness across data slices.
    
    This class tests whether model performance is consistent
    across different subsets of the data.
    
    Attributes:
        metrics_calculator: MetricsCalculator instance
        min_samples: Minimum samples for valid analysis
    """
    
    def __init__(self, min_samples: int = 30):
        """
        Initialize the robustness analyzer.
        
        Args:
            min_samples: Minimum samples required for analysis
        """
        self.metrics_calculator = MetricsCalculator()
        self.min_samples = min_samples
        
        logger.info(f"RobustnessAnalyzer initialized with min_samples={min_samples}")
    
    def analyze_by_topic(
        self,
        events: List[Dict],
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Analyze performance by topic.
        
        Args:
            events: Event dictionaries
            y_pred: Predicted labels
            y_true: True labels
            
        Returns:
            Dictionary mapping topics to metrics
        """
        results = {}
        
        # Group by topic
        topics = [e.get("topic", "other") for e in events]
        unique_topics = set(topics)
        
        for topic in unique_topics:
            mask = np.array([t == topic for t in topics])
            
            if mask.sum() < self.min_samples:
                logger.debug(f"Skipping topic {topic}: only {mask.sum()} samples")
                continue
            
            topic_pred = y_pred[mask]
            topic_true = y_true[mask]
            
            metrics = self.metrics_calculator.compute_classification_metrics(
                topic_true, topic_pred
            )
            metrics["n_samples"] = int(mask.sum())
            
            results[topic] = metrics
        
        return results
    
    def analyze_by_source(
        self,
        events: List[Dict],
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Analyze performance by source type.
        
        Args:
            events: Event dictionaries
            y_pred: Predicted labels
            y_true: True labels
            
        Returns:
            Dictionary mapping source types to metrics
        """
        results = {}
        
        # Extract source types
        sources = []
        for e in events:
            source_types = e.get("source_types", [])
            if "sec" in str(source_types).lower():
                sources.append("sec")
            else:
                sources.append("news")
        
        for source in ["sec", "news"]:
            mask = np.array([s == source for s in sources])
            
            if mask.sum() < self.min_samples:
                continue
            
            source_pred = y_pred[mask]
            source_true = y_true[mask]
            
            metrics = self.metrics_calculator.compute_classification_metrics(
                source_true, source_pred
            )
            metrics["n_samples"] = int(mask.sum())
            
            results[source] = metrics
        
        return results
    
    def analyze_time_drift(
        self,
        df: pd.DataFrame,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        date_column: str = "event_date",
        n_periods: int = 3
    ) -> Dict[str, Dict]:
        """
        Analyze performance drift over time.
        
        Args:
            df: DataFrame with date column
            y_pred: Predicted labels
            y_true: True labels
            date_column: Date column name
            n_periods: Number of time periods
            
        Returns:
            Dictionary mapping periods to metrics
        """
        results = {}
        
        # Sort by date
        df = df.sort_values(date_column).reset_index(drop=True)
        
        # Split into periods
        n = len(df)
        period_size = n // n_periods
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = start_idx + period_size if i < n_periods - 1 else n
            
            period_df = df.iloc[start_idx:end_idx]
            period_pred = y_pred[start_idx:end_idx]
            period_true = y_true[start_idx:end_idx]
            
            if len(period_pred) < self.min_samples:
                continue
            
            metrics = self.metrics_calculator.compute_classification_metrics(
                period_true, period_pred
            )
            
            period_name = f"period_{i+1}"
            metrics["start_date"] = period_df[date_column].min().isoformat()
            metrics["end_date"] = period_df[date_column].max().isoformat()
            metrics["n_samples"] = len(period_pred)
            
            results[period_name] = metrics
        
        return results
    
    def summarize_robustness(
        self,
        events: List[Dict],
        df: pd.DataFrame,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        """
        Generate comprehensive robustness summary.
        
        Args:
            events: Event dictionaries
            df: DataFrame
            y_pred: Predictions
            y_true: True labels
            
        Returns:
            Complete robustness report
        """
        return {
            "by_topic": self.analyze_by_topic(events, y_pred, y_true),
            "by_source": self.analyze_by_source(events, y_pred, y_true),
            "time_drift": self.analyze_time_drift(df, y_pred, y_true),
        }
