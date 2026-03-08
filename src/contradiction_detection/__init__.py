"""
Contradiction Detection Module

This module provides Natural Language Inference (NLI) based contradiction
detection for corporate claims. It determines the relationship between
pairs of claims:
- Entailment: Later claim confirms earlier claim
- Neutral: Claims are compatible but independent
- Contradiction: Later claim contradicts earlier claim

The module uses DeBERTa-v3-large-mnli for high-quality inference and
includes strict temporal ordering guards.

Example:
    from src.contradiction_detection import NLIDetector, ContradictionScorer
    
    detector = NLIDetector()
    result = detector.classify(
        premise="Factory will open in 2025",
        hypothesis="Factory opening delayed to 2026"
    )
    # result: {"entailment": 0.05, "neutral": 0.10, "contradiction": 0.85}
"""

from src.contradiction_detection.nli_detector import NLIDetector, NLIResult
from src.contradiction_detection.contradiction_scorer import ContradictionScorer
from src.contradiction_detection.relationship_classifier import RelationshipClassifier

__all__ = [
    "NLIDetector",
    "NLIResult",
    "ContradictionScorer",
    "RelationshipClassifier",
]
