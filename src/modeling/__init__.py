"""
Modeling Module

This module provides the main signal prediction model including:
- Feature engineering pipeline
- XGBoost-based signal classifier
- SHAP-based explainability
- Model persistence and loading

Example:
    from src.modeling import FeatureBuilder, SignalModel
    
    builder = FeatureBuilder()
    features = builder.build_features(events)
    
    model = SignalModel()
    model.fit(features, labels)
    predictions = model.predict(features)
"""

from src.modeling.feature_builder import FeatureBuilder
from src.modeling.signal_model import SignalModel
from src.modeling.explainability import ExplainabilityEngine

__all__ = [
    "FeatureBuilder",
    "SignalModel",
    "ExplainabilityEngine",
]
