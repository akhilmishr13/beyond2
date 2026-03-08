"""
Claim Extraction Module

This module provides NLP-based claim extraction from corporate documents
using Anthropic's Claude API. It extracts structured claims including:
- Speaker and role
- Topic classification
- Claim text and direction
- Entities mentioned
- Confidence scores

Example:
    from src.claim_extraction import ClaimExtractor
    
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(document_text, company="Apple", ticker="AAPL")
"""

from src.claim_extraction.claim_extractor import ClaimExtractor, ExtractedClaim
from src.claim_extraction.entity_extractor import EntityExtractor
from src.claim_extraction.claim_normalizer import ClaimNormalizer

__all__ = [
    "ClaimExtractor",
    "ExtractedClaim",
    "EntityExtractor",
    "ClaimNormalizer",
]
