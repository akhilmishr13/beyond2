"""
Visualization Plots

This module provides Plotly-based visualizations for analysis and reporting.

Features:
- Classification metrics plots
- Confusion matrix heatmaps
- Calibration reliability diagrams
- Event timeline charts
- Feature importance plots
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not installed. Visualization features limited.")


class MetricsPlotter:
    """
    Creates visualizations for model performance metrics.
    
    Generates interactive Plotly charts for:
    - Confusion matrices
    - ROC curves
    - Precision-recall curves
    - Feature importance
    """
    
    def __init__(self):
        """Initialize the metrics plotter."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - visualizations disabled")
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        labels: List[str],
        title: str = "Confusion Matrix"
    ) -> Optional[go.Figure]:
        """
        Create confusion matrix heatmap.
        
        Args:
            confusion_matrix: 2D array of confusion values
            labels: Class labels
            title: Plot title
            
        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 14},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=500,
            height=450
        )
        
        return fig
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        title: str = "ROC Curve"
    ) -> Optional[go.Figure]:
        """
        Create ROC curve plot.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: Area under curve
            title: Plot title
            
        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {auc_score:.3f})",
            line=dict(color="blue", width=2)
        ))
        
        # Diagonal reference
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="gray", dash="dash")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=500,
            height=450,
            showlegend=True
        )
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        title: str = "Feature Importance"
    ) -> Optional[go.Figure]:
        """
        Create feature importance bar chart.
        
        Args:
            feature_names: Names of features
            importance_values: Importance scores
            title: Plot title
            
        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        # Sort by importance
        sorted_idx = np.argsort(importance_values)[::-1][:20]  # Top 20
        names = [feature_names[i] for i in sorted_idx]
        values = importance_values[sorted_idx]
        
        fig = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color="steelblue"
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=600,
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        title: str = "Model Comparison"
    ) -> Optional[go.Figure]:
        """
        Create grouped bar chart comparing models.
        
        Args:
            metrics_dict: {model_name: {metric_name: value}}
            title: Plot title
            
        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        models = list(metrics_dict.keys())
        metrics = list(metrics_dict[models[0]].keys())
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [metrics_dict[model].get(metric, 0) for model in models]
            fig.add_trace(go.Bar(name=metric, x=models, y=values))
        
        fig.update_layout(
            title=title,
            xaxis_title="Model",
            yaxis_title="Score",
            barmode="group",
            height=450
        )
        
        return fig


class CalibrationPlotter:
    """
    Creates calibration visualizations.
    
    Generates reliability diagrams and calibration analysis charts.
    """
    
    def __init__(self):
        """Initialize the calibration plotter."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - calibration plots disabled")
    
    def plot_reliability_diagram(
        self,
        bin_centers: List[float],
        actual_freqs: List[float],
        bin_counts: Optional[List[int]] = None,
        title: str = "Reliability Diagram"
    ) -> Optional[go.Figure]:
        """
        Create reliability diagram.
        
        Args:
            bin_centers: Center of each calibration bin
            actual_freqs: Actual positive frequency per bin
            bin_counts: Optional sample counts per bin
            title: Plot title
            
        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect",
            line=dict(color="gray", dash="dash")
        ))
        
        # Calibration curve
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=actual_freqs,
            mode="lines+markers",
            name="Model",
            line=dict(color="blue", width=2),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            width=500,
            height=450,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def plot_calibration_histogram(
        self,
        probabilities: np.ndarray,
        n_bins: int = 10,
        title: str = "Prediction Distribution"
    ) -> Optional[go.Figure]:
        """
        Create histogram of predicted probabilities.
        
        Args:
            probabilities: Predicted probabilities
            n_bins: Number of bins
            title: Plot title
            
        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure(go.Histogram(
            x=probabilities,
            nbinsx=n_bins,
            marker_color="steelblue",
            opacity=0.7
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Probability",
            yaxis_title="Count",
            width=500,
            height=350
        )
        
        return fig


class EventTimeline:
    """
    Creates timeline visualizations for contradiction events.
    """
    
    def __init__(self):
        """Initialize the event timeline plotter."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - timeline plots disabled")
    
    def plot_events_timeline(
        self,
        events: List[Dict],
        title: str = "Contradiction Events Timeline"
    ) -> Optional[go.Figure]:
        """
        Create timeline of contradiction events.
        
        Args:
            events: List of event dictionaries with 'event_date', 'ticker', 'contradiction_score'
            title: Plot title
            
        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        if not events:
            return None
        
        df = pd.DataFrame(events)
        
        fig = px.scatter(
            df,
            x="event_date",
            y="ticker",
            size="contradiction_score",
            color="topic",
            hover_data=["contradiction_score", "reaction_label"],
            title=title
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Company",
            height=400
        )
        
        return fig
    
    def plot_signal_performance(
        self,
        signals: List[Dict],
        title: str = "Signal Performance Over Time"
    ) -> Optional[go.Figure]:
        """
        Create performance chart for signals over time.
        
        Args:
            signals: List of signal dictionaries with returns
            title: Plot title
            
        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        if not signals:
            return None
        
        df = pd.DataFrame(signals)
        df = df.sort_values("date")
        
        # Cumulative return
        df["cumulative_return"] = (1 + df["return"]).cumprod() - 1
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["cumulative_return"],
            mode="lines",
            name="Cumulative Return",
            line=dict(color="green", width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            yaxis_tickformat=".1%",
            height=400
        )
        
        return fig
