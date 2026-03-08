"""
Signal Prediction Model

This module implements the main signal prediction model using XGBoost.
The model predicts market reaction (positive/negative/neutral) based
on contradiction features.

Usage:
    from src.modeling.signal_model import SignalModel
    
    model = SignalModel()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
from loguru import logger

from src.config import get_config


@dataclass
class SignalPrediction:
    """
    Model prediction for an event.
    
    Attributes:
        signal: Predicted signal (bearish, bullish, neutral)
        confidence: Prediction confidence
        prob_positive: Probability of positive reaction
        prob_negative: Probability of negative reaction
        prob_neutral: Probability of neutral reaction
    """
    signal: str
    confidence: float
    prob_positive: float
    prob_negative: float
    prob_neutral: float


class SignalModel:
    """
    XGBoost-based signal prediction model.
    
    This model predicts whether a contradiction event will lead to
    positive, negative, or neutral market reaction.
    
    Attributes:
        model: XGBoost classifier
        thresholds: Decision thresholds for signals
    """
    
    # Label mapping
    LABEL_MAP = {"positive": 0, "neutral": 1, "negative": 2}
    REVERSE_MAP = {0: "positive", 1: "neutral", 2: "negative"}
    SIGNAL_MAP = {"positive": "bullish", "neutral": "neutral", "negative": "bearish"}
    
    def __init__(self):
        """Initialize the signal model."""
        config = get_config()
        
        model_config = config.model.signal_model.get("xgboost", {})
        
        self.model = None
        self.model_params = {
            "n_estimators": model_config.get("n_estimators", 500),
            "max_depth": model_config.get("max_depth", 6),
            "learning_rate": model_config.get("learning_rate", 0.05),
            "subsample": model_config.get("subsample", 0.8),
            "colsample_bytree": model_config.get("colsample_bytree", 0.8),
            "min_child_weight": model_config.get("min_child_weight", 1),
            "gamma": model_config.get("gamma", 0),
            "reg_alpha": model_config.get("reg_alpha", 0.1),
            "reg_lambda": model_config.get("reg_lambda", 1.0),
            "random_state": model_config.get("random_state", 42),
            "n_jobs": model_config.get("n_jobs", -1),
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
        }
        
        thresholds = config.model.signal_model.get("thresholds", {})
        self.bearish_threshold = thresholds.get("bearish_signal", {}).get("confidence_min", 0.6)
        self.bullish_threshold = thresholds.get("bullish_signal", {}).get("confidence_min", 0.6)
        
        self._feature_importance: Optional[Dict[str, float]] = None
        
        logger.info("SignalModel initialized")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Fit the model on training data.
        
        Args:
            X: Training features
            y: Training labels (strings or integers)
            X_val: Optional validation features
            y_val: Optional validation labels
            feature_names: Optional feature names
        """
        from xgboost import XGBClassifier
        
        # Convert string labels to integers
        if y.dtype == object or isinstance(y[0], str):
            y = np.array([self.LABEL_MAP.get(label, 1) for label in y])
        
        if y_val is not None and (y_val.dtype == object or isinstance(y_val[0], str)):
            y_val = np.array([self.LABEL_MAP.get(label, 1) for label in y_val])
        
        # Create and fit model
        self.model = XGBClassifier(**self.model_params)
        
        eval_set = [(X, y)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )
        
        # Store feature importance
        if feature_names:
            importances = self.model.feature_importances_
            self._feature_importance = dict(zip(feature_names, importances))
        
        logger.info(f"Model fitted on {len(X)} samples")
    
    def predict(self, X: np.ndarray) -> List[SignalPrediction]:
        """
        Make predictions for events.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of SignalPrediction objects
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Get probabilities
        probs = self.model.predict_proba(X)
        
        predictions = []
        for prob in probs:
            pred_class = np.argmax(prob)
            label = self.REVERSE_MAP[pred_class]
            signal = self.SIGNAL_MAP[label]
            
            predictions.append(SignalPrediction(
                signal=signal,
                confidence=float(prob[pred_class]),
                prob_positive=float(prob[0]),
                prob_negative=float(prob[2]) if len(prob) > 2 else 0.0,
                prob_neutral=float(prob[1]) if len(prob) > 1 else 0.0
            ))
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix (n_samples x n_classes)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.predict_proba(X)
    
    def get_signal(
        self,
        prediction: SignalPrediction,
        contradiction_score: float
    ) -> str:
        """
        Get final trading signal based on prediction and contradiction.
        
        Args:
            prediction: Model prediction
            contradiction_score: Contradiction score of the event
            
        Returns:
            Final signal string
        """
        # Bearish signal: high contradiction + confident negative prediction
        if (contradiction_score >= 0.7 and 
            prediction.prob_negative >= self.bearish_threshold):
            return "bearish_alert"
        
        # Bullish signal: low contradiction + confident positive prediction
        if (contradiction_score <= 0.3 and
            prediction.prob_positive >= self.bullish_threshold):
            return "bullish_consistency"
        
        return "no_action"
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self._feature_importance is None:
            if self.model is not None:
                return dict(enumerate(self.model.feature_importances_))
            return {}
        return self._feature_importance
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Output file path
        """
        import joblib
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            "model": self.model,
            "params": self.model_params,
            "feature_importance": self._feature_importance,
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Input file path
        """
        import joblib
        
        data = joblib.load(filepath)
        self.model = data["model"]
        self.model_params = data.get("params", self.model_params)
        self._feature_importance = data.get("feature_importance")
        
        logger.info(f"Model loaded from {filepath}")
