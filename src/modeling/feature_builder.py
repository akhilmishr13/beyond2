"""
Feature Builder

This module builds features for the signal prediction model from
contradiction events and market context.

Features include:
- Contradiction features (score, relationship type)
- Source features (SEC vs news, speaker role)
- Topic features (encoded topic category)
- Market context (prior returns, volatility)

Usage:
    from src.modeling.feature_builder import FeatureBuilder
    
    builder = FeatureBuilder()
    X, feature_names = builder.build_features(events)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_config


class FeatureBuilder:
    """
    Builds feature vectors for signal prediction.
    
    This class handles:
    - Numeric feature extraction
    - Categorical encoding
    - Feature normalization
    - Missing value handling
    
    Attributes:
        feature_names: List of feature names in output
    """
    
    # Topic categories for encoding
    TOPICS = [
        "guidance", "projects", "legal_regulatory", "supply_chain",
        "capital_expenditure", "risk_factors", "personnel",
        "customers", "products", "other"
    ]
    
    # Speaker roles for encoding
    SPEAKER_ROLES = ["CEO", "CFO", "COO", "CTO", "spokesperson", "company", "other"]
    
    def __init__(self):
        """Initialize the feature builder."""
        self.feature_names: List[str] = []
        self._fitted = False
        
        # Statistics for normalization
        self._means: Dict[str, float] = {}
        self._stds: Dict[str, float] = {}
        
        logger.info("FeatureBuilder initialized")
    
    def fit(self, events: List[Dict]) -> 'FeatureBuilder':
        """
        Fit the feature builder on training data.
        
        Computes statistics needed for normalization.
        
        Args:
            events: List of training events
            
        Returns:
            self
        """
        # Compute statistics for numeric features
        numeric_values = {
            "contradiction_score": [],
            "days_between": [],
        }
        
        for event in events:
            numeric_values["contradiction_score"].append(
                event.get("contradiction_score", 0.5)
            )
            numeric_values["days_between"].append(
                event.get("days_between", 30)
            )
        
        for feature, values in numeric_values.items():
            self._means[feature] = np.mean(values)
            self._stds[feature] = np.std(values) or 1.0
        
        self._fitted = True
        return self
    
    def build_features(
        self,
        events: List[Dict],
        include_market_context: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build feature matrix from events.
        
        Args:
            events: List of event dictionaries
            include_market_context: Whether to include market features
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not self._fitted:
            self.fit(events)
        
        features_list = []
        
        for event in events:
            features = self._build_single_features(event, include_market_context)
            features_list.append(features)
        
        X = np.array(features_list)
        
        logger.info(f"Built feature matrix: {X.shape}")
        return X, self.feature_names
    
    def _build_single_features(
        self,
        event: Dict,
        include_market_context: bool
    ) -> np.ndarray:
        """Build features for a single event."""
        features = []
        feature_names = []
        
        # Contradiction score (normalized)
        score = event.get("contradiction_score", 0.5)
        features.append(score)
        feature_names.append("contradiction_score")
        
        # Days between claims (normalized)
        days = event.get("days_between", 30)
        days_norm = (days - self._means.get("days_between", 30)) / self._stds.get("days_between", 30)
        features.append(days_norm)
        feature_names.append("days_between_norm")
        
        # Topic one-hot encoding
        topic = event.get("topic", "other")
        for t in self.TOPICS:
            features.append(1.0 if topic == t else 0.0)
            feature_names.append(f"topic_{t}")
        
        # Speaker role encoding
        speaker_role = event.get("speaker_role", "other")
        for role in self.SPEAKER_ROLES:
            features.append(1.0 if speaker_role and speaker_role.upper() == role.upper() else 0.0)
            feature_names.append(f"speaker_{role.lower()}")
        
        # Source type features
        source_types = event.get("source_types", [])
        features.append(1.0 if "sec" in str(source_types).lower() else 0.0)
        feature_names.append("source_is_sec")
        features.append(1.0 if "news" in str(source_types).lower() else 0.0)
        feature_names.append("source_is_news")
        
        # Direction features
        earlier_direction = event.get("earlier_claim", {}).get("direction", "neutral")
        later_direction = event.get("later_claim", {}).get("direction", "neutral")
        
        direction_map = {"positive": 1, "neutral": 0, "negative": -1}
        earlier_dir_num = direction_map.get(earlier_direction, 0)
        later_dir_num = direction_map.get(later_direction, 0)
        
        features.append(float(earlier_dir_num))
        feature_names.append("earlier_direction")
        features.append(float(later_dir_num))
        feature_names.append("later_direction")
        features.append(float(later_dir_num - earlier_dir_num))
        feature_names.append("direction_change")
        
        # Relationship type
        relationship = event.get("relationship", "contradiction")
        features.append(1.0 if relationship == "contradiction" else 0.0)
        feature_names.append("is_contradiction")
        features.append(1.0 if relationship == "revision" else 0.0)
        feature_names.append("is_revision")
        
        # Market context features (if available)
        if include_market_context:
            # Prior return (if available in event)
            prior_return = event.get("prior_30d_return", 0.0)
            features.append(prior_return if prior_return else 0.0)
            feature_names.append("prior_30d_return")
            
            # Volatility
            volatility = event.get("volatility_30d", 0.2)
            features.append(volatility if volatility else 0.2)
            feature_names.append("volatility_30d")
        
        # Store feature names on first call
        if not self.feature_names:
            self.feature_names = feature_names
        
        return np.array(features)
    
    def build_dataframe(self, events: List[Dict]) -> pd.DataFrame:
        """
        Build features as a DataFrame.
        
        Args:
            events: List of events
            
        Returns:
            DataFrame with features
        """
        X, feature_names = self.build_features(events)
        return pd.DataFrame(X, columns=feature_names)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()
