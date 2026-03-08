"""
Temporal Claim Linker

This module provides time-aware claim linking that respects temporal
ordering constraints. It ensures that:
- Earlier claims always precede later claims
- Claims are linked within appropriate time windows
- No future information is used in comparisons

Usage:
    from src.claim_matching.temporal_linker import TemporalLinker
    
    linker = TemporalLinker()
    links = linker.link_claims(new_claim, historical_claims)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from src.config import get_config
from src.claim_matching.embedding_matcher import EmbeddingMatcher, ClaimMatch
from src.claim_matching.topic_matcher import TopicMatcher


@dataclass
class ClaimLink:
    """
    A temporal link between two claims.
    
    Attributes:
        earlier_claim_id: ID of the earlier claim
        later_claim_id: ID of the later claim
        similarity: Semantic similarity score
        days_between: Number of days between claims
        topic_match: Whether topics match
    """
    earlier_claim_id: int
    later_claim_id: int
    similarity: float
    days_between: int
    topic_match: bool


@dataclass
class ClaimCandidate:
    """
    A claim candidate for matching.
    
    Attributes:
        claim_id: Unique identifier
        text: Claim text
        topic: Claim topic
        timestamp: Claim timestamp
        embedding: Optional pre-computed embedding
    """
    claim_id: int
    text: str
    topic: str
    timestamp: datetime
    embedding: Optional[np.ndarray] = None


class TemporalLinker:
    """
    Links claims across time while respecting temporal ordering.
    
    This class is CRITICAL for preventing temporal leakage:
    - Never compares a claim with future claims
    - Enforces strict earlier_timestamp < later_timestamp
    - Validates all temporal constraints before linking
    
    Attributes:
        embedding_matcher: Embedding-based matcher
        topic_matcher: Topic-based matcher
        lookback_days: Maximum days to look back for matches
    """
    
    def __init__(self):
        """Initialize the temporal linker."""
        config = get_config()
        
        self.lookback_days = config.pipeline.claim_matching.lookback_days
        self.similarity_threshold = config.pipeline.claim_matching.similarity_threshold
        self.max_matches = config.pipeline.claim_matching.max_matches
        
        self.embedding_matcher = EmbeddingMatcher()
        self.topic_matcher = TopicMatcher()
        
        logger.info(f"TemporalLinker initialized with lookback: {self.lookback_days} days")
    
    def validate_temporal_order(
        self,
        earlier_timestamp: datetime,
        later_timestamp: datetime
    ) -> bool:
        """
        Validate temporal ordering constraint.
        
        CRITICAL: This must always be enforced to prevent leakage.
        
        Args:
            earlier_timestamp: Timestamp of earlier claim
            later_timestamp: Timestamp of later claim
            
        Returns:
            True if ordering is valid (earlier < later)
        """
        if earlier_timestamp >= later_timestamp:
            logger.warning(
                f"Temporal ordering violation: {earlier_timestamp} >= {later_timestamp}"
            )
            return False
        return True
    
    def link_claims(
        self,
        later_claim: ClaimCandidate,
        earlier_claims: List[ClaimCandidate],
        require_topic_match: bool = False
    ) -> List[ClaimLink]:
        """
        Find links between a later claim and earlier claims.
        
        This is the main entry point for temporal linking. It:
        1. Filters candidates by time window
        2. Validates temporal ordering
        3. Computes semantic similarity
        4. Optionally filters by topic
        
        Args:
            later_claim: The newer claim to find links for
            earlier_claims: List of historical claim candidates
            require_topic_match: If True, only match same-topic claims
            
        Returns:
            List of ClaimLink objects, sorted by similarity
        """
        # Filter by time window and temporal ordering
        valid_candidates = []
        for candidate in earlier_claims:
            # CRITICAL: Enforce temporal ordering
            if not self.validate_temporal_order(candidate.timestamp, later_claim.timestamp):
                continue
            
            # Check lookback window
            days_diff = (later_claim.timestamp - candidate.timestamp).days
            if days_diff > self.lookback_days:
                continue
            
            # Optional topic filter
            if require_topic_match:
                if not self.topic_matcher.topics_match(later_claim.topic, candidate.topic):
                    continue
            
            valid_candidates.append((candidate, days_diff))
        
        if not valid_candidates:
            return []
        
        # Get embeddings
        if later_claim.embedding is None:
            later_claim.embedding = self.embedding_matcher.embed_text(later_claim.text)
        
        # Compute similarities
        links = []
        for candidate, days_diff in valid_candidates:
            if candidate.embedding is None:
                candidate.embedding = self.embedding_matcher.embed_text(candidate.text)
            
            similarity = self.embedding_matcher.compute_similarity(
                later_claim.embedding,
                candidate.embedding
            )
            
            if similarity >= self.similarity_threshold:
                topic_match = self.topic_matcher.topics_match(
                    later_claim.topic,
                    candidate.topic
                )
                
                links.append(ClaimLink(
                    earlier_claim_id=candidate.claim_id,
                    later_claim_id=later_claim.claim_id,
                    similarity=similarity,
                    days_between=days_diff,
                    topic_match=topic_match
                ))
        
        # Sort by similarity and limit
        links.sort(key=lambda x: x.similarity, reverse=True)
        return links[:self.max_matches]
    
    def link_all_claims(
        self,
        claims: List[ClaimCandidate],
        require_topic_match: bool = False
    ) -> List[ClaimLink]:
        """
        Find all links within a set of claims.
        
        Processes claims in chronological order, linking each claim
        to earlier claims only.
        
        Args:
            claims: List of all claims to link
            require_topic_match: If True, only match same-topic claims
            
        Returns:
            List of all ClaimLink objects
        """
        # Sort by timestamp
        sorted_claims = sorted(claims, key=lambda c: c.timestamp)
        
        # Pre-compute all embeddings
        logger.info(f"Computing embeddings for {len(sorted_claims)} claims")
        texts = [c.text for c in sorted_claims]
        embeddings = self.embedding_matcher.embed_batch(texts)
        
        for i, claim in enumerate(sorted_claims):
            claim.embedding = embeddings[i]
        
        # Find links for each claim
        all_links = []
        for i, later_claim in enumerate(sorted_claims):
            if i == 0:
                continue  # No earlier claims for first claim
            
            # Get earlier claims
            earlier_claims = sorted_claims[:i]
            
            # Find links
            links = self.link_claims(
                later_claim,
                earlier_claims,
                require_topic_match
            )
            
            all_links.extend(links)
        
        logger.info(f"Found {len(all_links)} claim links")
        return all_links
    
    def get_claim_history(
        self,
        claim: ClaimCandidate,
        all_claims: List[ClaimCandidate],
        max_history: int = 10
    ) -> List[Tuple[ClaimCandidate, float]]:
        """
        Get the historical claims most relevant to a given claim.
        
        Args:
            claim: The claim to get history for
            all_claims: All available claims
            max_history: Maximum historical claims to return
            
        Returns:
            List of (claim, similarity) tuples
        """
        # Filter to earlier claims only
        earlier = [
            c for c in all_claims
            if c.claim_id != claim.claim_id
            and self.validate_temporal_order(c.timestamp, claim.timestamp)
        ]
        
        if not earlier:
            return []
        
        # Get links
        links = self.link_claims(claim, earlier)
        
        # Map back to claims
        claim_map = {c.claim_id: c for c in earlier}
        history = []
        for link in links[:max_history]:
            if link.earlier_claim_id in claim_map:
                history.append((
                    claim_map[link.earlier_claim_id],
                    link.similarity
                ))
        
        return history
    
    def detect_claim_thread(
        self,
        claims: List[ClaimCandidate],
        topic: str,
        min_similarity: float = 0.6
    ) -> List[List[ClaimCandidate]]:
        """
        Detect threads of related claims over time.
        
        A thread is a sequence of claims that evolve the same topic
        over time (e.g., project status updates).
        
        Args:
            claims: All claims to analyze
            topic: Topic to focus on
            min_similarity: Minimum similarity for thread connection
            
        Returns:
            List of claim threads (each thread is a list of claims)
        """
        # Filter by topic
        topic_claims = [
            c for c in claims
            if self.topic_matcher.topics_match(topic, c.topic)
        ]
        
        if len(topic_claims) < 2:
            return []
        
        # Sort chronologically
        sorted_claims = sorted(topic_claims, key=lambda c: c.timestamp)
        
        # Pre-compute embeddings
        texts = [c.text for c in sorted_claims]
        embeddings = self.embedding_matcher.embed_batch(texts)
        for i, claim in enumerate(sorted_claims):
            claim.embedding = embeddings[i]
        
        # Build threads using greedy linking
        threads: List[List[ClaimCandidate]] = []
        used = set()
        
        for claim in sorted_claims:
            if claim.claim_id in used:
                continue
            
            # Start a new thread
            thread = [claim]
            used.add(claim.claim_id)
            
            # Find continuation
            current = claim
            for later in sorted_claims:
                if later.claim_id in used:
                    continue
                if later.timestamp <= current.timestamp:
                    continue
                
                sim = self.embedding_matcher.compute_similarity(
                    current.embedding,
                    later.embedding
                )
                
                if sim >= min_similarity:
                    thread.append(later)
                    used.add(later.claim_id)
                    current = later
            
            if len(thread) > 1:
                threads.append(thread)
        
        return threads
