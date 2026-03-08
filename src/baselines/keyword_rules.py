"""
Keyword Rules Baseline

This module implements a rule-based baseline that detects signals
using keyword patterns common in financial disclosures.

The baseline:
- Uses bearish/bullish keyword lists
- Detects direction changes between claims
- Generates signals based on keyword presence

Usage:
    from src.baselines.keyword_rules import KeywordRules
    
    model = KeywordRules()
    predictions = model.predict(events)
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from loguru import logger

from src.config import get_config


@dataclass
class KeywordPrediction:
    """
    Prediction from keyword rules model.
    
    Attributes:
        signal: Predicted signal (bearish, bullish, neutral)
        confidence: Rule-based confidence score
        matched_keywords: Keywords that matched
        reasoning: Explanation for prediction
    """
    signal: str
    confidence: float
    matched_keywords: List[str]
    reasoning: str


class KeywordRules:
    """
    Rule-based baseline using keyword matching.
    
    This baseline detects signals by:
    1. Finding bearish keywords in later claims
    2. Finding bullish keywords in earlier claims
    3. Detecting keyword-based direction changes
    
    Attributes:
        bearish_keywords: Set of bearish keywords
        bullish_keywords: Set of bullish keywords
    """
    
    DEFAULT_BEARISH = [
        "delay", "delayed", "postpone", "postponed",
        "unexpected", "unexpectedly",
        "material weakness", "weakness",
        "terminated", "termination", "terminate",
        "impairment", "impaired", "impair",
        "downgrade", "downgraded",
        "suspension", "suspended", "suspend",
        "write-down", "write-off", "writedown", "writeoff",
        "litigation", "lawsuit", "sue", "sued",
        "investigation", "investigating", "investigated",
        "decline", "declined", "declining",
        "shortfall", "miss", "missed", "missing",
        "below expectations", "lower than expected",
        "revised downward", "downward revision",
        "warning", "warn", "warned",
        "concern", "concerned", "concerning",
        "risk", "risks", "risky",
        "uncertainty", "uncertain",
        "challenge", "challenges", "challenging",
        "difficult", "difficulty",
        "pressure", "pressured",
        "negative", "negatively",
        "loss", "losses",
        "cut", "cuts", "cutting",
        "layoff", "layoffs",
        "restructuring", "restructure",
    ]
    
    DEFAULT_BULLISH = [
        "ahead of schedule", "ahead schedule",
        "exceeded", "exceeds", "exceed",
        "record", "record-breaking",
        "growth", "growing", "grew",
        "expansion", "expanding", "expand",
        "approved", "approval", "approve",
        "successful", "success", "successfully",
        "partnership", "partner", "partnered",
        "award", "awarded", "awards",
        "beat", "beats", "beating",
        "above expectations", "higher than expected",
        "revised upward", "upward revision",
        "strong", "stronger", "strength",
        "robust", "robustly",
        "momentum", "accelerating", "accelerate",
        "outperform", "outperformed", "outperforming",
        "upgrade", "upgraded",
        "positive", "positively",
        "profit", "profitable", "profitability",
        "increase", "increased", "increasing",
        "improvement", "improved", "improving",
        "milestone", "achieved", "achievement",
        "launch", "launched", "launching",
        "innovation", "innovative",
    ]
    
    def __init__(self):
        """Initialize the keyword rules model."""
        config = get_config()
        
        # Load keywords from config or use defaults
        model_config = config.model.baselines.get("keyword_rules", {})
        
        self.bearish_keywords = set(
            model_config.get("bearish_keywords", self.DEFAULT_BEARISH)
        )
        self.bullish_keywords = set(
            model_config.get("bullish_keywords", self.DEFAULT_BULLISH)
        )
        
        # Compile regex patterns for efficiency
        self._bearish_pattern = self._compile_pattern(self.bearish_keywords)
        self._bullish_pattern = self._compile_pattern(self.bullish_keywords)
        
        logger.info(
            f"KeywordRules initialized with {len(self.bearish_keywords)} bearish, "
            f"{len(self.bullish_keywords)} bullish keywords"
        )
    
    def _compile_pattern(self, keywords: Set[str]) -> re.Pattern:
        """Compile keywords into a regex pattern."""
        escaped = [re.escape(kw) for kw in keywords]
        pattern = r'\b(' + '|'.join(escaped) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def _find_keywords(
        self,
        text: str,
        keyword_type: str
    ) -> List[str]:
        """
        Find keywords in text.
        
        Args:
            text: Text to search
            keyword_type: 'bearish' or 'bullish'
            
        Returns:
            List of matched keywords
        """
        if not text:
            return []
        
        pattern = self._bearish_pattern if keyword_type == "bearish" else self._bullish_pattern
        matches = pattern.findall(text.lower())
        return list(set(matches))
    
    def predict_single(
        self,
        earlier_text: str,
        later_text: str
    ) -> KeywordPrediction:
        """
        Predict signal for a single claim pair.
        
        Args:
            earlier_text: Earlier claim text
            later_text: Later claim text
            
        Returns:
            KeywordPrediction with signal and reasoning
        """
        # Find keywords in each claim
        earlier_bearish = self._find_keywords(earlier_text, "bearish")
        earlier_bullish = self._find_keywords(earlier_text, "bullish")
        later_bearish = self._find_keywords(later_text, "bearish")
        later_bullish = self._find_keywords(later_text, "bullish")
        
        # Calculate scores
        earlier_score = len(earlier_bullish) - len(earlier_bearish)
        later_score = len(later_bullish) - len(later_bearish)
        
        # Detect direction change
        matched_keywords = []
        
        # Rule: Earlier bullish + Later bearish = Bearish signal
        if earlier_bullish and later_bearish:
            matched_keywords = later_bearish
            confidence = min(0.9, 0.5 + 0.1 * len(later_bearish))
            return KeywordPrediction(
                signal="bearish",
                confidence=confidence,
                matched_keywords=matched_keywords,
                reasoning=f"Direction reversal: earlier positive ({earlier_bullish[:3]}) "
                         f"contradicted by later negative ({later_bearish[:3]})"
            )
        
        # Rule: Strong bearish keywords in later claim
        if len(later_bearish) >= 2:
            matched_keywords = later_bearish
            confidence = min(0.8, 0.4 + 0.1 * len(later_bearish))
            return KeywordPrediction(
                signal="bearish",
                confidence=confidence,
                matched_keywords=matched_keywords,
                reasoning=f"Multiple bearish keywords detected: {later_bearish[:5]}"
            )
        
        # Rule: Strong bullish keywords in later claim
        if len(later_bullish) >= 2 and not later_bearish:
            matched_keywords = later_bullish
            confidence = min(0.7, 0.3 + 0.1 * len(later_bullish))
            return KeywordPrediction(
                signal="bullish",
                confidence=confidence,
                matched_keywords=matched_keywords,
                reasoning=f"Multiple bullish keywords detected: {later_bullish[:5]}"
            )
        
        # Default: Neutral
        return KeywordPrediction(
            signal="neutral",
            confidence=0.5,
            matched_keywords=[],
            reasoning="No strong keyword signals detected"
        )
    
    def predict(
        self,
        events: List[Dict]
    ) -> List[KeywordPrediction]:
        """
        Predict signals for multiple events.
        
        Args:
            events: List of event dictionaries with earlier_claim and later_claim
            
        Returns:
            List of KeywordPrediction objects
        """
        predictions = []
        
        for event in events:
            earlier_text = event.get("earlier_claim", {}).get("text", "")
            later_text = event.get("later_claim", {}).get("text", "")
            
            pred = self.predict_single(earlier_text, later_text)
            predictions.append(pred)
        
        return predictions
    
    def get_keyword_counts(self, text: str) -> Dict[str, int]:
        """
        Get counts of bearish and bullish keywords.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with bearish and bullish counts
        """
        return {
            "bearish": len(self._find_keywords(text, "bearish")),
            "bullish": len(self._find_keywords(text, "bullish")),
        }
