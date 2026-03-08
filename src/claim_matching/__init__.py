"""
Claim Matching Module

This module provides functionality for matching and linking claims
across documents and time periods. It uses semantic embeddings to
find related claims that discuss the same topics or entities.

The matching pipeline:
1. Generate embeddings for claims using sentence-transformers
2. Index embeddings for efficient similarity search
3. Match new claims against historical claims
4. Filter matches by topic and temporal constraints

Example:
    from src.claim_matching import EmbeddingMatcher, TemporalLinker
    
    matcher = EmbeddingMatcher()
    matches = matcher.find_matches(new_claim, historical_claims)
"""

from src.claim_matching.embedding_matcher import EmbeddingMatcher
from src.claim_matching.topic_matcher import TopicMatcher
from src.claim_matching.temporal_linker import TemporalLinker

__all__ = [
    "EmbeddingMatcher",
    "TopicMatcher",
    "TemporalLinker",
]
