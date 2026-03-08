"""
Topic-based Claim Matcher

This module provides topic-based matching for claims. It ensures that
matched claims discuss related topics, improving the quality of
contradiction detection.

Usage:
    from src.claim_matching.topic_matcher import TopicMatcher
    
    matcher = TopicMatcher()
    is_match = matcher.topics_match("guidance", "revenue forecast")
"""

from typing import Dict, List, Optional, Set

from loguru import logger


class TopicMatcher:
    """
    Matches claims based on topic similarity.
    
    This class handles:
    - Topic hierarchy matching
    - Related topic identification
    - Topic inference from text
    """
    
    # Topic hierarchy and relationships
    TOPIC_HIERARCHY = {
        "guidance": {
            "subtopics": ["revenue", "earnings", "sales", "profit", "forecast", "outlook"],
            "related": ["risk_factors", "projects"]
        },
        "projects": {
            "subtopics": ["factory", "facility", "expansion", "construction", "launch", "timeline"],
            "related": ["capital_expenditure", "guidance"]
        },
        "legal_regulatory": {
            "subtopics": ["lawsuit", "litigation", "investigation", "compliance", "settlement"],
            "related": ["risk_factors"]
        },
        "supply_chain": {
            "subtopics": ["supplier", "inventory", "logistics", "shortage", "delivery"],
            "related": ["projects", "risk_factors"]
        },
        "capital_expenditure": {
            "subtopics": ["investment", "capex", "spending", "acquisition"],
            "related": ["projects", "guidance"]
        },
        "risk_factors": {
            "subtopics": ["risk", "challenge", "uncertainty", "concern", "warning"],
            "related": ["guidance", "legal_regulatory", "supply_chain"]
        },
        "personnel": {
            "subtopics": ["executive", "ceo", "cfo", "hired", "resigned", "appointed"],
            "related": []
        },
        "customers": {
            "subtopics": ["customer", "client", "contract", "partnership", "deal"],
            "related": ["guidance"]
        },
        "products": {
            "subtopics": ["product", "feature", "launch", "release", "update"],
            "related": ["projects", "guidance"]
        },
    }
    
    # Keywords for topic inference
    TOPIC_KEYWORDS: Dict[str, List[str]] = {
        "guidance": [
            "revenue", "earnings", "profit", "sales", "forecast",
            "outlook", "expect", "guidance", "target", "project"
        ],
        "projects": [
            "factory", "plant", "facility", "expansion", "construction",
            "production", "timeline", "milestone", "schedule", "completion"
        ],
        "legal_regulatory": [
            "lawsuit", "litigation", "legal", "regulatory", "investigation",
            "sec", "doj", "settlement", "compliance", "court"
        ],
        "supply_chain": [
            "supply", "supplier", "inventory", "logistics", "shortage",
            "delivery", "shipping", "parts", "components", "materials"
        ],
        "capital_expenditure": [
            "capex", "investment", "spending", "capital", "acquisition",
            "purchase", "infrastructure", "equipment"
        ],
        "risk_factors": [
            "risk", "warning", "concern", "challenge", "uncertainty",
            "threat", "exposure", "vulnerable"
        ],
        "personnel": [
            "ceo", "cfo", "executive", "officer", "director",
            "hired", "appointed", "resigned", "departed"
        ],
        "customers": [
            "customer", "client", "contract", "partnership", "deal",
            "agreement", "order", "relationship"
        ],
        "products": [
            "product", "feature", "launch", "release", "update",
            "version", "model", "offering"
        ],
    }
    
    def __init__(self):
        """Initialize the topic matcher."""
        # Build reverse lookup for subtopics
        self._subtopic_to_parent: Dict[str, str] = {}
        for parent, info in self.TOPIC_HIERARCHY.items():
            for subtopic in info["subtopics"]:
                self._subtopic_to_parent[subtopic] = parent
        
        logger.info("TopicMatcher initialized")
    
    def topics_match(
        self,
        topic1: str,
        topic2: str,
        strict: bool = False
    ) -> bool:
        """
        Check if two topics match.
        
        Args:
            topic1: First topic
            topic2: Second topic
            strict: If True, require exact match
            
        Returns:
            True if topics match
        """
        topic1 = topic1.lower()
        topic2 = topic2.lower()
        
        # Exact match
        if topic1 == topic2:
            return True
        
        if strict:
            return False
        
        # Normalize to parent topics
        parent1 = self._subtopic_to_parent.get(topic1, topic1)
        parent2 = self._subtopic_to_parent.get(topic2, topic2)
        
        # Same parent topic
        if parent1 == parent2:
            return True
        
        # Check if related
        if parent1 in self.TOPIC_HIERARCHY:
            if parent2 in self.TOPIC_HIERARCHY[parent1].get("related", []):
                return True
        
        if parent2 in self.TOPIC_HIERARCHY:
            if parent1 in self.TOPIC_HIERARCHY[parent2].get("related", []):
                return True
        
        return False
    
    def get_related_topics(self, topic: str) -> Set[str]:
        """
        Get all topics related to a given topic.
        
        Args:
            topic: Input topic
            
        Returns:
            Set of related topic names
        """
        topic = topic.lower()
        parent = self._subtopic_to_parent.get(topic, topic)
        
        related = {parent}
        
        if parent in self.TOPIC_HIERARCHY:
            info = self.TOPIC_HIERARCHY[parent]
            related.update(info.get("related", []))
            related.update(info.get("subtopics", []))
        
        return related
    
    def infer_topic(self, text: str) -> str:
        """
        Infer the topic from text content.
        
        Args:
            text: Input text
            
        Returns:
            Inferred topic name
        """
        text_lower = text.lower()
        
        # Count keyword matches per topic
        scores: Dict[str, int] = {}
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[topic] = score
        
        if scores:
            return max(scores.keys(), key=lambda k: scores[k])
        
        return "other"
    
    def compute_topic_similarity(
        self,
        topic1: str,
        topic2: str
    ) -> float:
        """
        Compute similarity between two topics.
        
        Args:
            topic1: First topic
            topic2: Second topic
            
        Returns:
            Similarity score (0-1)
        """
        topic1 = topic1.lower()
        topic2 = topic2.lower()
        
        # Exact match
        if topic1 == topic2:
            return 1.0
        
        # Normalize to parent topics
        parent1 = self._subtopic_to_parent.get(topic1, topic1)
        parent2 = self._subtopic_to_parent.get(topic2, topic2)
        
        # Same parent
        if parent1 == parent2:
            return 0.9
        
        # Directly related
        if parent1 in self.TOPIC_HIERARCHY:
            if parent2 in self.TOPIC_HIERARCHY[parent1].get("related", []):
                return 0.6
        
        if parent2 in self.TOPIC_HIERARCHY:
            if parent1 in self.TOPIC_HIERARCHY[parent2].get("related", []):
                return 0.6
        
        return 0.0
    
    def filter_by_topic(
        self,
        target_topic: str,
        candidates: List[Dict],
        topic_key: str = "topic",
        strict: bool = False
    ) -> List[Dict]:
        """
        Filter candidates by topic compatibility.
        
        Args:
            target_topic: Topic to match
            candidates: List of candidate dictionaries
            topic_key: Key containing topic in candidates
            strict: If True, require exact match
            
        Returns:
            Filtered list of candidates
        """
        return [
            c for c in candidates
            if self.topics_match(target_topic, c.get(topic_key, ""), strict)
        ]
