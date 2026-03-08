"""
Claim Normalizer

This module provides functionality for normalizing claims to enable
consistent comparison across different documents and time periods.

Normalization includes:
- Text standardization
- Entity resolution
- Time reference normalization
- Metric standardization

Usage:
    from src.claim_extraction.claim_normalizer import ClaimNormalizer
    
    normalizer = ClaimNormalizer()
    normalized = normalizer.normalize(claim)
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.claim_extraction.claim_extractor import ExtractedClaim


@dataclass
class NormalizedClaim:
    """
    A normalized claim ready for comparison.
    
    Attributes:
        original: Original ExtractedClaim
        normalized_text: Standardized claim text
        canonical_topic: Standardized topic
        canonical_entities: Resolved entity names
        time_reference: Parsed time reference
        numeric_values: Extracted numeric values
    """
    original: ExtractedClaim
    normalized_text: str
    canonical_topic: str
    canonical_entities: List[str]
    time_reference: Optional[str] = None
    numeric_values: Optional[Dict[str, float]] = None


class ClaimNormalizer:
    """
    Normalizes claims for consistent comparison.
    
    This class handles:
    - Text cleaning and standardization
    - Topic mapping to canonical categories
    - Entity name resolution
    - Time reference parsing
    - Numeric value extraction
    """
    
    # Topic mappings to canonical form
    TOPIC_MAPPINGS = {
        "revenue": "guidance",
        "earnings": "guidance",
        "sales": "guidance",
        "profit": "guidance",
        "forecast": "guidance",
        "outlook": "guidance",
        "target": "guidance",
        
        "factory": "projects",
        "facility": "projects",
        "plant": "projects",
        "expansion": "projects",
        "construction": "projects",
        "launch": "projects",
        "timeline": "projects",
        
        "lawsuit": "legal_regulatory",
        "litigation": "legal_regulatory",
        "investigation": "legal_regulatory",
        "regulatory": "legal_regulatory",
        "compliance": "legal_regulatory",
        "settlement": "legal_regulatory",
        
        "supplier": "supply_chain",
        "inventory": "supply_chain",
        "logistics": "supply_chain",
        "shortage": "supply_chain",
        "delivery": "supply_chain",
        
        "investment": "capital_expenditure",
        "capex": "capital_expenditure",
        "spending": "capital_expenditure",
        "acquisition": "capital_expenditure",
        
        "risk": "risk_factors",
        "challenge": "risk_factors",
        "uncertainty": "risk_factors",
        "concern": "risk_factors",
        "warning": "risk_factors",
        
        "ceo": "personnel",
        "cfo": "personnel",
        "executive": "personnel",
        "hired": "personnel",
        "resigned": "personnel",
        "appointed": "personnel",
        
        "customer": "customers",
        "client": "customers",
        "contract": "customers",
        "partnership": "customers",
        
        "product": "products",
        "feature": "products",
        "update": "products",
        "version": "products",
    }
    
    # Common abbreviations to expand
    ABBREVIATIONS = {
        "Q1": "first quarter",
        "Q2": "second quarter",
        "Q3": "third quarter",
        "Q4": "fourth quarter",
        "FY": "fiscal year",
        "YoY": "year over year",
        "QoQ": "quarter over quarter",
        "MoM": "month over month",
        "EPS": "earnings per share",
        "EBITDA": "earnings before interest taxes depreciation amortization",
        "GM": "gross margin",
        "OM": "operating margin",
        "ROI": "return on investment",
        "ROE": "return on equity",
        "IPO": "initial public offering",
        "M&A": "mergers and acquisitions",
    }
    
    # Number word mappings
    NUMBER_WORDS = {
        "billion": 1_000_000_000,
        "million": 1_000_000,
        "thousand": 1_000,
        "hundred": 100,
        "b": 1_000_000_000,
        "m": 1_000_000,
        "k": 1_000,
    }
    
    def __init__(self):
        """Initialize the normalizer."""
        logger.info("ClaimNormalizer initialized")
    
    def normalize(self, claim: ExtractedClaim) -> NormalizedClaim:
        """
        Normalize a claim for comparison.
        
        Args:
            claim: ExtractedClaim to normalize
            
        Returns:
            NormalizedClaim with standardized fields
        """
        # Normalize text
        normalized_text = self._normalize_text(claim.claim_text)
        
        # Map topic to canonical form
        canonical_topic = self._canonicalize_topic(claim.topic, normalized_text)
        
        # Resolve entities
        canonical_entities = self._resolve_entities(claim.entities or [])
        
        # Extract time reference
        time_reference = self._extract_time_reference(claim.claim_text)
        
        # Extract numeric values
        numeric_values = self._extract_numeric_values(claim.claim_text)
        
        return NormalizedClaim(
            original=claim,
            normalized_text=normalized_text,
            canonical_topic=canonical_topic,
            canonical_entities=canonical_entities,
            time_reference=time_reference,
            numeric_values=numeric_values
        )
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize claim text.
        
        Args:
            text: Raw claim text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Lowercase
        normalized = text.lower()
        
        # Expand abbreviations
        for abbr, expansion in self.ABBREVIATIONS.items():
            normalized = re.sub(
                rf'\b{re.escape(abbr.lower())}\b',
                expansion,
                normalized,
                flags=re.IGNORECASE
            )
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove extra punctuation
        normalized = re.sub(r'["\']', '', normalized)
        
        return normalized
    
    def _canonicalize_topic(
        self,
        topic: str,
        text: str
    ) -> str:
        """
        Map topic to canonical category.
        
        Args:
            topic: Original topic
            text: Normalized claim text
            
        Returns:
            Canonical topic string
        """
        # Check if topic is already canonical
        canonical_topics = {
            "guidance", "projects", "legal_regulatory",
            "supply_chain", "capital_expenditure", "risk_factors",
            "personnel", "customers", "products", "other"
        }
        
        if topic.lower() in canonical_topics:
            return topic.lower()
        
        # Try to map from known mappings
        if topic.lower() in self.TOPIC_MAPPINGS:
            return self.TOPIC_MAPPINGS[topic.lower()]
        
        # Infer from text content
        for keyword, canonical in self.TOPIC_MAPPINGS.items():
            if keyword in text.lower():
                return canonical
        
        return "other"
    
    def _resolve_entities(self, entities: List[str]) -> List[str]:
        """
        Resolve entity names to canonical forms.
        
        Args:
            entities: List of entity strings
            
        Returns:
            List of resolved entity names
        """
        resolved = []
        
        for entity in entities:
            # Normalize case and whitespace
            normalized = entity.strip()
            
            # Remove common suffixes for comparison
            normalized = re.sub(
                r'\s+(Inc\.?|Corp\.?|Corporation|Company|LLC|Ltd\.?)$',
                '',
                normalized,
                flags=re.IGNORECASE
            )
            
            # Title case
            normalized = normalized.title()
            
            if normalized and normalized not in resolved:
                resolved.append(normalized)
        
        return resolved
    
    def _extract_time_reference(self, text: str) -> Optional[str]:
        """
        Extract and normalize time references.
        
        Args:
            text: Claim text
            
        Returns:
            Normalized time reference or None
        """
        # Quarter-year patterns
        quarter_match = re.search(
            r'(Q[1-4])\s+(\d{4})|'
            r'(first|second|third|fourth)\s+quarter\s+(?:of\s+)?(\d{4})',
            text,
            re.IGNORECASE
        )
        
        if quarter_match:
            if quarter_match.group(1):
                return f"{quarter_match.group(1)} {quarter_match.group(2)}"
            else:
                quarter_map = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}
                quarter = quarter_map.get(quarter_match.group(3).lower(), "")
                return f"{quarter} {quarter_match.group(4)}"
        
        # Year-only patterns
        year_match = re.search(r'\b(20\d{2})\b', text)
        if year_match:
            return year_match.group(1)
        
        # Relative time patterns
        relative_patterns = [
            (r'next\s+quarter', "next_quarter"),
            (r'next\s+year', "next_year"),
            (r'this\s+quarter', "current_quarter"),
            (r'this\s+year', "current_year"),
            (r'by\s+year\s*end', "year_end"),
        ]
        
        for pattern, label in relative_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return label
        
        return None
    
    def _extract_numeric_values(self, text: str) -> Dict[str, float]:
        """
        Extract numeric values from claim.
        
        Args:
            text: Claim text
            
        Returns:
            Dictionary of value type to numeric value
        """
        values = {}
        
        # Dollar amounts
        dollar_match = re.search(
            r'\$\s*([\d,.]+)\s*(billion|million|thousand|b|m|k)?',
            text,
            re.IGNORECASE
        )
        
        if dollar_match:
            num = float(dollar_match.group(1).replace(',', ''))
            multiplier = dollar_match.group(2)
            if multiplier:
                num *= self.NUMBER_WORDS.get(multiplier.lower(), 1)
            values["dollar_amount"] = num
        
        # Percentages
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if pct_match:
            values["percentage"] = float(pct_match.group(1))
        
        # Unit counts
        unit_match = re.search(
            r'([\d,.]+)\s*(units?|devices?|vehicles?|products?)',
            text,
            re.IGNORECASE
        )
        if unit_match:
            values["unit_count"] = float(unit_match.group(1).replace(',', ''))
        
        return values
    
    def compute_similarity(
        self,
        claim1: NormalizedClaim,
        claim2: NormalizedClaim
    ) -> float:
        """
        Compute similarity between two normalized claims.
        
        Args:
            claim1: First normalized claim
            claim2: Second normalized claim
            
        Returns:
            Similarity score (0-1)
        """
        scores = []
        
        # Topic match
        if claim1.canonical_topic == claim2.canonical_topic:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Entity overlap
        if claim1.canonical_entities and claim2.canonical_entities:
            set1 = set(e.lower() for e in claim1.canonical_entities)
            set2 = set(e.lower() for e in claim2.canonical_entities)
            overlap = len(set1 & set2) / max(len(set1 | set2), 1)
            scores.append(overlap)
        
        # Time reference match
        if claim1.time_reference and claim2.time_reference:
            if claim1.time_reference == claim2.time_reference:
                scores.append(1.0)
            else:
                scores.append(0.3)
        
        # Word overlap in normalized text
        words1 = set(claim1.normalized_text.split())
        words2 = set(claim2.normalized_text.split())
        word_overlap = len(words1 & words2) / max(len(words1 | words2), 1)
        scores.append(word_overlap)
        
        return sum(scores) / len(scores) if scores else 0.0
