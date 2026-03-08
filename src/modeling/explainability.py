"""
Explainability Engine

This module provides SHAP-based explanations for model predictions,
making the signal model interpretable and trustworthy.

Usage:
    from src.modeling.explainability import ExplainabilityEngine
    
    engine = ExplainabilityEngine(model)
    explanation = engine.explain_prediction(features, event)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class PredictionExplanation:
    """
    Explanation for a model prediction.
    
    Attributes:
        signal: Predicted signal
        confidence: Prediction confidence
        top_features: Most important features for this prediction
        reasoning: Human-readable explanation
        shap_values: Raw SHAP values
    """
    signal: str
    confidence: float
    top_features: List[Tuple[str, float]]
    reasoning: str
    shap_values: Optional[Dict[str, float]] = None


class ExplainabilityEngine:
    """
    Generates explanations for model predictions using SHAP.
    
    This class provides:
    - SHAP value computation
    - Feature contribution analysis
    - Human-readable explanations
    - Visualization utilities
    
    Attributes:
        model: Trained signal model
        explainer: SHAP TreeExplainer
    """
    
    # Explanation templates
    TEMPLATES = {
        "high_contradiction": (
            "Strong contradiction detected between {speaker}'s earlier statement "
            "about {topic} and the subsequent disclosure. "
            "Historical analysis shows similar contradictions have a {rate}% "
            "correlation with {direction} market reactions."
        ),
        "direction_reversal": (
            "Narrative reversal detected: earlier {earlier_dir} statement "
            "contradicted by later {later_dir} disclosure. "
            "This pattern historically precedes {direction} price movements."
        ),
        "topic_sensitive": (
            "Contradiction in {topic} topic, which is historically sensitive "
            "to market reaction. The {contradiction_level} contradiction score "
            "suggests {implication}."
        ),
    }
    
    def __init__(self, model=None):
        """
        Initialize the explainability engine.
        
        Args:
            model: Trained SignalModel instance
        """
        self.model = model
        self._explainer = None
        
        logger.info("ExplainabilityEngine initialized")
    
    def set_model(self, model):
        """Set the model to explain."""
        self.model = model
        self._explainer = None
    
    @property
    def explainer(self):
        """Lazy load SHAP explainer."""
        if self._explainer is None and self.model is not None:
            try:
                import shap
                self._explainer = shap.TreeExplainer(self.model.model)
            except Exception as e:
                logger.warning(f"Could not create SHAP explainer: {e}")
        return self._explainer
    
    def compute_shap_values(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute SHAP values for predictions.
        
        Args:
            X: Feature matrix
            feature_names: Optional feature names
            
        Returns:
            Dictionary with SHAP values per class
        """
        if self.explainer is None:
            return {}
        
        import shap
        
        shap_values = self.explainer.shap_values(X)
        
        result = {}
        if isinstance(shap_values, list):
            for i, values in enumerate(shap_values):
                class_name = self.model.REVERSE_MAP.get(i, str(i))
                result[class_name] = values
        else:
            result["values"] = shap_values
        
        return result
    
    def explain_prediction(
        self,
        features: np.ndarray,
        event: Dict,
        feature_names: List[str]
    ) -> PredictionExplanation:
        """
        Generate explanation for a single prediction.
        
        Args:
            features: Feature vector (1D)
            event: Event dictionary
            feature_names: Feature names
            
        Returns:
            PredictionExplanation object
        """
        # Get prediction
        if self.model is None:
            raise RuntimeError("Model not set")
        
        features_2d = features.reshape(1, -1)
        predictions = self.model.predict(features_2d)
        prediction = predictions[0]
        
        # Get SHAP values
        shap_dict = None
        top_features = []
        
        if self.explainer is not None:
            try:
                shap_values = self.compute_shap_values(features_2d, feature_names)
                
                # Get values for predicted class
                pred_class = self.model.SIGNAL_MAP.get(prediction.signal, "neutral")
                if pred_class in shap_values:
                    values = shap_values[pred_class][0]
                    shap_dict = dict(zip(feature_names, values))
                    
                    # Get top contributing features
                    sorted_features = sorted(
                        zip(feature_names, values),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )
                    top_features = sorted_features[:5]
            except Exception as e:
                logger.warning(f"Could not compute SHAP values: {e}")
        
        # Generate human-readable explanation
        reasoning = self._generate_reasoning(event, prediction, top_features)
        
        return PredictionExplanation(
            signal=prediction.signal,
            confidence=prediction.confidence,
            top_features=top_features,
            reasoning=reasoning,
            shap_values=shap_dict
        )
    
    def _generate_reasoning(
        self,
        event: Dict,
        prediction,
        top_features: List[Tuple[str, float]]
    ) -> str:
        """Generate human-readable reasoning for prediction."""
        parts = []
        
        # Add signal statement
        parts.append(f"Signal: {prediction.signal.upper()}")
        parts.append(f"Confidence: {prediction.confidence:.1%}")
        parts.append("")
        
        # Add evidence
        parts.append("Evidence:")
        
        earlier_claim = event.get("earlier_claim", {})
        later_claim = event.get("later_claim", {})
        
        if earlier_claim.get("text"):
            parts.append(f"- Earlier claim ({earlier_claim.get('date', 'N/A')}): "
                        f"\"{earlier_claim['text'][:100]}...\"")
        
        if later_claim.get("text"):
            parts.append(f"- Later claim ({later_claim.get('date', 'N/A')}): "
                        f"\"{later_claim['text'][:100]}...\"")
        
        parts.append(f"- Contradiction score: {event.get('contradiction_score', 'N/A'):.2f}")
        parts.append(f"- Topic: {event.get('topic', 'N/A')}")
        
        parts.append("")
        
        # Add feature contributions
        if top_features:
            parts.append("Contributing factors:")
            for feature, contribution in top_features:
                direction = "+" if contribution > 0 else "-"
                parts.append(f"  {direction} {feature}: {contribution:.3f}")
        
        return "\n".join(parts)
    
    def batch_explain(
        self,
        X: np.ndarray,
        events: List[Dict],
        feature_names: List[str]
    ) -> List[PredictionExplanation]:
        """
        Generate explanations for multiple predictions.
        
        Args:
            X: Feature matrix
            events: List of event dictionaries
            feature_names: Feature names
            
        Returns:
            List of PredictionExplanation objects
        """
        explanations = []
        
        for i, (features, event) in enumerate(zip(X, events)):
            try:
                explanation = self.explain_prediction(features, event, feature_names)
                explanations.append(explanation)
            except Exception as e:
                logger.warning(f"Could not explain prediction {i}: {e}")
                explanations.append(PredictionExplanation(
                    signal="unknown",
                    confidence=0.0,
                    top_features=[],
                    reasoning=f"Explanation failed: {e}"
                ))
        
        return explanations
