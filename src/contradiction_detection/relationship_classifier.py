"""
Relationship Classifier

This module provides fine-grained classification of the relationship
between corporate claims, going beyond simple NLI labels to provide
business-relevant categorizations.

Relationship types:
- Confirmation: Later claim confirms earlier claim
- Update: Later claim updates with new information
- Revision: Later claim revises earlier projections
- Contradiction: Later claim contradicts earlier claim
- Clarification: Later claim clarifies ambiguous earlier claim

Usage:
    from src.contradiction_detection.relationship_classifier import RelationshipClassifier
    
    classifier = RelationshipClassifier()
    relationship = classifier.classify(earlier_claim, later_claim, nli_result)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.contradiction_detection.nli_detector import NLIResult


class RelationType(str, Enum):
    """Types of relationships between claims."""
    CONFIRMATION = "confirmation"
    UPDATE = "update"
    REVISION = "revision"
    CONTRADICTION = "contradiction"
    CLARIFICATION = "clarification"
    ESCALATION = "escalation"
    DEESCALATION = "deescalation"
    UNKNOWN = "unknown"


@dataclass
class RelationshipResult:
    """
    Result of relationship classification.
    
    Attributes:
        relationship: Type of relationship
        confidence: Confidence in classification
        reasoning: Human-readable explanation
        is_material: Whether the change is material
    """
    relationship: RelationType
    confidence: float
    reasoning: str
    is_material: bool = False


class RelationshipClassifier:
    """
    Classifies relationships between corporate claims.
    
    Goes beyond NLI to provide business-relevant classifications
    that help understand the nature of narrative changes.
    
    The classification considers:
    - NLI probabilities
    - Claim directions (positive/negative)
    - Topic context
    - Numeric changes
    """
    
    # Direction change mapping
    DIRECTION_CHANGES = {
        ("positive", "positive"): "consistent",
        ("positive", "negative"): "reversal",
        ("positive", "neutral"): "softening",
        ("negative", "negative"): "consistent",
        ("negative", "positive"): "reversal",
        ("negative", "neutral"): "recovery",
        ("neutral", "positive"): "improvement",
        ("neutral", "negative"): "deterioration",
        ("neutral", "neutral"): "consistent",
    }
    
    def __init__(self):
        """Initialize the relationship classifier."""
        logger.info("RelationshipClassifier initialized")
    
    def classify(
        self,
        earlier_claim: Dict,
        later_claim: Dict,
        nli_result: NLIResult
    ) -> RelationshipResult:
        """
        Classify the relationship between two claims.
        
        Args:
            earlier_claim: Earlier claim dictionary with text, direction, topic
            later_claim: Later claim dictionary
            nli_result: NLI classification result
            
        Returns:
            RelationshipResult with classification and reasoning
        """
        # Get directions
        earlier_direction = earlier_claim.get("direction", "neutral")
        later_direction = later_claim.get("direction", "neutral")
        
        # Determine direction change
        direction_change = self.DIRECTION_CHANGES.get(
            (earlier_direction, later_direction),
            "unknown"
        )
        
        # Start classification based on NLI
        if nli_result.entailment > 0.7:
            return self._classify_entailment(
                earlier_claim, later_claim, direction_change, nli_result
            )
        
        elif nli_result.contradiction > 0.7:
            return self._classify_contradiction(
                earlier_claim, later_claim, direction_change, nli_result
            )
        
        else:
            return self._classify_neutral(
                earlier_claim, later_claim, direction_change, nli_result
            )
    
    def _classify_entailment(
        self,
        earlier_claim: Dict,
        later_claim: Dict,
        direction_change: str,
        nli_result: NLIResult
    ) -> RelationshipResult:
        """Classify when NLI indicates entailment."""
        
        if direction_change == "consistent":
            return RelationshipResult(
                relationship=RelationType.CONFIRMATION,
                confidence=nli_result.entailment,
                reasoning="Later claim confirms earlier statement with consistent direction",
                is_material=False
            )
        
        elif direction_change == "improvement":
            return RelationshipResult(
                relationship=RelationType.UPDATE,
                confidence=nli_result.entailment * 0.9,
                reasoning="Later claim provides positive update while confirming earlier context",
                is_material=True
            )
        
        else:
            return RelationshipResult(
                relationship=RelationType.CLARIFICATION,
                confidence=nli_result.entailment * 0.8,
                reasoning="Later claim clarifies or extends earlier statement",
                is_material=False
            )
    
    def _classify_contradiction(
        self,
        earlier_claim: Dict,
        later_claim: Dict,
        direction_change: str,
        nli_result: NLIResult
    ) -> RelationshipResult:
        """Classify when NLI indicates contradiction."""
        
        if direction_change == "reversal":
            return RelationshipResult(
                relationship=RelationType.CONTRADICTION,
                confidence=nli_result.contradiction,
                reasoning="Clear reversal from earlier position - strong contradiction",
                is_material=True
            )
        
        elif direction_change in ["deterioration", "softening"]:
            return RelationshipResult(
                relationship=RelationType.REVISION,
                confidence=nli_result.contradiction * 0.9,
                reasoning="Later claim revises earlier projections negatively",
                is_material=True
            )
        
        elif direction_change in ["improvement", "recovery"]:
            return RelationshipResult(
                relationship=RelationType.REVISION,
                confidence=nli_result.contradiction * 0.85,
                reasoning="Later claim revises earlier assessment positively",
                is_material=True
            )
        
        else:
            return RelationshipResult(
                relationship=RelationType.CONTRADICTION,
                confidence=nli_result.contradiction * 0.9,
                reasoning="Contradiction detected between claims",
                is_material=True
            )
    
    def _classify_neutral(
        self,
        earlier_claim: Dict,
        later_claim: Dict,
        direction_change: str,
        nli_result: NLIResult
    ) -> RelationshipResult:
        """Classify when NLI indicates neutral."""
        
        if direction_change == "deterioration":
            return RelationshipResult(
                relationship=RelationType.ESCALATION,
                confidence=0.7,
                reasoning="Sentiment shift from neutral to negative suggests escalating concern",
                is_material=True
            )
        
        elif direction_change == "recovery":
            return RelationshipResult(
                relationship=RelationType.DEESCALATION,
                confidence=0.7,
                reasoning="Sentiment shift from negative to neutral suggests improving situation",
                is_material=True
            )
        
        elif direction_change in ["improvement"]:
            return RelationshipResult(
                relationship=RelationType.UPDATE,
                confidence=0.6,
                reasoning="New positive information added to earlier context",
                is_material=False
            )
        
        else:
            return RelationshipResult(
                relationship=RelationType.UPDATE,
                confidence=0.5,
                reasoning="Claims appear to be related updates without clear contradiction",
                is_material=False
            )
    
    def classify_batch(
        self,
        claim_pairs: List[Tuple[Dict, Dict]],
        nli_results: List[NLIResult]
    ) -> List[RelationshipResult]:
        """
        Classify relationships for multiple claim pairs.
        
        Args:
            claim_pairs: List of (earlier_claim, later_claim) tuples
            nli_results: Corresponding NLI results
            
        Returns:
            List of RelationshipResult objects
        """
        results = []
        
        for (earlier, later), nli_result in zip(claim_pairs, nli_results):
            result = self.classify(earlier, later, nli_result)
            results.append(result)
        
        return results
    
    def get_material_changes(
        self,
        results: List[RelationshipResult]
    ) -> List[int]:
        """
        Get indices of material relationship changes.
        
        Args:
            results: List of RelationshipResult objects
            
        Returns:
            List of indices where is_material is True
        """
        return [i for i, r in enumerate(results) if r.is_material]
    
    def summarize_relationships(
        self,
        results: List[RelationshipResult]
    ) -> Dict[str, int]:
        """
        Summarize relationship types.
        
        Args:
            results: List of RelationshipResult objects
            
        Returns:
            Dictionary mapping relationship type to count
        """
        summary = {}
        for result in results:
            rel_type = result.relationship.value
            summary[rel_type] = summary.get(rel_type, 0) + 1
        return summary
