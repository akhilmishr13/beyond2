"""
Contradiction Scorer

This module provides scoring and classification of contradictions
based on severity, topic, and other factors.

The scorer:
- Computes composite contradiction scores
- Classifies contradictions by severity level
- Adjusts scores based on topic sensitivity
- Validates temporal constraints

Usage:
    from src.contradiction_detection.contradiction_scorer import ContradictionScorer
    
    scorer = ContradictionScorer()
    score = scorer.compute_score(nli_result, topic="guidance")
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from src.config import get_config
from src.contradiction_detection.nli_detector import NLIResult


@dataclass
class ContradictionScore:
    """
    Comprehensive contradiction score.
    
    Attributes:
        raw_score: Raw NLI contradiction probability
        adjusted_score: Score adjusted for topic sensitivity
        severity: Severity level (low, medium, high, critical)
        confidence: Confidence in the score
        factors: Contributing factors to the score
    """
    raw_score: float
    adjusted_score: float
    severity: str
    confidence: float
    factors: Dict[str, float]


class ContradictionScorer:
    """
    Computes and classifies contradiction scores.
    
    This class handles:
    - Raw NLI score processing
    - Topic-based score adjustment
    - Severity classification
    - Multi-factor scoring
    
    Attributes:
        thresholds: Score thresholds for severity levels
        topic_weights: Topic sensitivity weights
    """
    
    # Severity thresholds
    DEFAULT_THRESHOLDS = {
        "critical": 0.9,
        "high": 0.8,
        "medium": 0.6,
        "low": 0.3,
    }
    
    # Topic sensitivity weights
    # Higher weight = more sensitive topic (contradictions more impactful)
    TOPIC_WEIGHTS = {
        "guidance": 1.2,        # Revenue/earnings guidance is highly sensitive
        "projects": 1.1,        # Project timelines are important
        "legal_regulatory": 1.3, # Legal matters are very sensitive
        "supply_chain": 1.0,    # Normal sensitivity
        "capital_expenditure": 1.0,
        "risk_factors": 1.1,    # Risk disclosures matter
        "personnel": 0.9,       # Slightly less sensitive
        "customers": 1.0,
        "products": 1.0,
        "other": 0.8,          # Unknown topics less weighted
    }
    
    def __init__(self):
        """Initialize the contradiction scorer."""
        config = get_config()
        
        self.thresholds = config.pipeline.contradiction.thresholds
        if not self.thresholds:
            self.thresholds = self.DEFAULT_THRESHOLDS
        
        logger.info("ContradictionScorer initialized")
    
    def compute_score(
        self,
        nli_result: NLIResult,
        topic: Optional[str] = None,
        speaker_role: Optional[str] = None,
        days_between: Optional[int] = None
    ) -> ContradictionScore:
        """
        Compute comprehensive contradiction score.
        
        Args:
            nli_result: NLI classification result
            topic: Topic of the claims
            speaker_role: Role of the speaker (CEO, CFO, etc.)
            days_between: Days between the two claims
            
        Returns:
            ContradictionScore with adjusted score and metadata
        """
        raw_score = nli_result.contradiction
        factors = {"nli_contradiction": raw_score}
        
        # Start with raw score
        adjusted_score = raw_score
        
        # Apply topic weight
        if topic:
            topic_weight = self.TOPIC_WEIGHTS.get(topic.lower(), 0.9)
            adjusted_score *= topic_weight
            factors["topic_weight"] = topic_weight
        
        # Apply speaker weight (CEO statements more impactful)
        speaker_weight = 1.0
        if speaker_role:
            speaker_weights = {
                "CEO": 1.15,
                "CFO": 1.10,
                "COO": 1.05,
                "company": 1.0,
                "spokesperson": 0.95,
            }
            speaker_weight = speaker_weights.get(speaker_role.upper(), 1.0)
            adjusted_score *= speaker_weight
            factors["speaker_weight"] = speaker_weight
        
        # Apply time decay (more recent contradictions slightly more impactful)
        if days_between is not None:
            if days_between <= 30:
                time_weight = 1.05
            elif days_between <= 90:
                time_weight = 1.0
            else:
                time_weight = 0.95
            adjusted_score *= time_weight
            factors["time_weight"] = time_weight
        
        # Clip to [0, 1]
        adjusted_score = max(0.0, min(1.0, adjusted_score))
        
        # Determine severity
        severity = self._classify_severity(adjusted_score)
        
        # Compute confidence based on NLI distribution
        confidence = self._compute_confidence(nli_result)
        
        return ContradictionScore(
            raw_score=raw_score,
            adjusted_score=adjusted_score,
            severity=severity,
            confidence=confidence,
            factors=factors
        )
    
    def _classify_severity(self, score: float) -> str:
        """
        Classify contradiction severity based on score.
        
        Args:
            score: Adjusted contradiction score
            
        Returns:
            Severity level string
        """
        if score >= self.thresholds.get("high", 0.8):
            return "high"
        elif score >= self.thresholds.get("medium", 0.6):
            return "medium"
        elif score >= self.thresholds.get("low", 0.3):
            return "low"
        else:
            return "minimal"
    
    def _compute_confidence(self, nli_result: NLIResult) -> float:
        """
        Compute confidence in the NLI result.
        
        Higher confidence when one class dominates, lower when
        probabilities are spread across classes.
        
        Args:
            nli_result: NLI classification result
            
        Returns:
            Confidence score (0-1)
        """
        probs = [nli_result.entailment, nli_result.neutral, nli_result.contradiction]
        max_prob = max(probs)
        
        # Confidence based on how dominant the max probability is
        # Max entropy is when all probs are 1/3 = 0.333
        confidence = (max_prob - 0.333) / (1.0 - 0.333)
        
        return max(0.0, min(1.0, confidence))
    
    def should_flag(
        self,
        score: ContradictionScore,
        min_severity: str = "medium"
    ) -> bool:
        """
        Determine if a contradiction should be flagged.
        
        Args:
            score: ContradictionScore to evaluate
            min_severity: Minimum severity to flag
            
        Returns:
            True if contradiction should be flagged
        """
        severity_order = ["minimal", "low", "medium", "high", "critical"]
        
        score_idx = severity_order.index(score.severity)
        min_idx = severity_order.index(min_severity)
        
        return score_idx >= min_idx
    
    def compute_batch_scores(
        self,
        nli_results: List[NLIResult],
        topics: Optional[List[str]] = None,
        speaker_roles: Optional[List[str]] = None
    ) -> List[ContradictionScore]:
        """
        Compute scores for multiple NLI results.
        
        Args:
            nli_results: List of NLI results
            topics: Optional list of topics (parallel to results)
            speaker_roles: Optional list of speaker roles
            
        Returns:
            List of ContradictionScore objects
        """
        scores = []
        
        for i, result in enumerate(nli_results):
            topic = topics[i] if topics and i < len(topics) else None
            role = speaker_roles[i] if speaker_roles and i < len(speaker_roles) else None
            
            score = self.compute_score(result, topic=topic, speaker_role=role)
            scores.append(score)
        
        return scores
    
    def rank_contradictions(
        self,
        scores: List[ContradictionScore],
        min_score: float = 0.3
    ) -> List[int]:
        """
        Rank contradictions by adjusted score.
        
        Args:
            scores: List of ContradictionScore objects
            min_score: Minimum score to include
            
        Returns:
            List of indices sorted by score (descending)
        """
        indexed_scores = [
            (i, s.adjusted_score)
            for i, s in enumerate(scores)
            if s.adjusted_score >= min_score
        ]
        
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [i for i, _ in indexed_scores]


class TemporalContradictionGuard:
    """
    Guards against temporal leakage in contradiction detection.
    
    CRITICAL: This class ensures that contradictions are only
    detected in the correct temporal direction (earlier -> later).
    """
    
    def validate_pair(
        self,
        earlier_timestamp: datetime,
        later_timestamp: datetime
    ) -> bool:
        """
        Validate temporal ordering of a claim pair.
        
        Args:
            earlier_timestamp: Timestamp of earlier claim
            later_timestamp: Timestamp of later claim
            
        Returns:
            True if ordering is valid
            
        Raises:
            ValueError: If temporal ordering is violated
        """
        if earlier_timestamp >= later_timestamp:
            raise ValueError(
                f"TEMPORAL VIOLATION: Earlier timestamp ({earlier_timestamp}) "
                f"must be before later timestamp ({later_timestamp}). "
                f"This would cause future information leakage!"
            )
        return True
    
    def validate_batch(
        self,
        pairs: List[tuple]
    ) -> bool:
        """
        Validate temporal ordering for a batch of pairs.
        
        Args:
            pairs: List of (earlier_timestamp, later_timestamp, ...) tuples
            
        Returns:
            True if all pairs are valid
        """
        for i, pair in enumerate(pairs):
            try:
                self.validate_pair(pair[0], pair[1])
            except ValueError as e:
                logger.error(f"Temporal violation in pair {i}: {e}")
                raise
        return True
