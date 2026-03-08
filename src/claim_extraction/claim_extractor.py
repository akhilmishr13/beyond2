"""
Claim Extractor

This module provides the core claim extraction functionality using
Anthropic's Claude API. It extracts structured claims from corporate
documents including SEC filings and news articles.

The extractor uses carefully crafted prompts to identify and classify
claims related to:
- Revenue/earnings guidance
- Project timelines and milestones
- Legal and regulatory matters
- Supply chain status
- Risk factors
- Personnel changes

Usage:
    from src.claim_extraction.claim_extractor import ClaimExtractor
    
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(
        text="CEO stated revenue will grow 15% next quarter...",
        company="Apple",
        ticker="AAPL"
    )
"""

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from loguru import logger

from src.config import get_config


@dataclass
class ExtractedClaim:
    """
    A structured claim extracted from a document.
    
    Attributes:
        claim_text: The verbatim claim text
        speaker: Name of the person making the claim
        speaker_role: Role (CEO, CFO, company, etc.)
        topic: Topic category
        claim_type: Type of claim (projection, status, commitment, etc.)
        direction: Sentiment direction (positive, negative, neutral)
        entities: Named entities in the claim
        confidence: Extraction confidence (0-1)
        source_text: Original source text
    """
    claim_text: str
    speaker: Optional[str] = None
    speaker_role: Optional[str] = None
    topic: str = "other"
    claim_type: Optional[str] = None
    direction: str = "neutral"
    entities: Optional[List[str]] = None
    confidence: float = 0.8
    source_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ClaimExtractor:
    """
    Extracts structured claims from documents using Claude.
    
    This class handles:
    - Document chunking for long texts
    - Claude API interaction with structured output
    - Claim classification and scoring
    - Error handling and retries
    
    Attributes:
        client: Anthropic client instance
        model: Claude model to use
        max_tokens: Maximum tokens for response
        temperature: Sampling temperature
    """
    
    # System prompt for claim extraction
    SYSTEM_PROMPT = """You are an expert financial analyst specializing in extracting structured claims from corporate documents.

Your task is to identify and extract specific claims made in the text. A claim is a statement about:
- Future projections or expectations
- Current status of projects, operations, or financials  
- Commitments or promises
- Assessments of risk or challenges
- Changes in strategy or direction

For each claim, extract:
1. claim_text: The exact quote or close paraphrase of the claim
2. speaker: Who made the claim (name if available)
3. speaker_role: Their role (CEO, CFO, company, spokesperson, etc.)
4. topic: One of: guidance, projects, legal_regulatory, supply_chain, capital_expenditure, risk_factors, personnel, customers, products, other
5. claim_type: projection, status_update, commitment, assessment, announcement
6. direction: positive, negative, or neutral sentiment
7. entities: Key entities mentioned (facilities, products, people, etc.)
8. confidence: Your confidence in the extraction (0.0-1.0)

Focus on substantive claims that could be verified or contradicted later. Ignore generic statements, boilerplate, or purely factual reporting without company statements.

Return your response as a JSON array of claim objects. If no claims are found, return an empty array []."""

    # User prompt template
    USER_PROMPT_TEMPLATE = """Extract claims from this {source_type} document for {company} ({ticker}).

Document Text:
{text}

Return a JSON array of extracted claims. Each claim should have these fields:
- claim_text (string): The claim statement
- speaker (string or null): Who made the claim
- speaker_role (string or null): Their role
- topic (string): One of guidance, projects, legal_regulatory, supply_chain, capital_expenditure, risk_factors, personnel, customers, products, other
- claim_type (string): One of projection, status_update, commitment, assessment, announcement
- direction (string): positive, negative, or neutral
- entities (array of strings): Key entities mentioned
- confidence (float): 0.0 to 1.0

Only return the JSON array, no other text."""

    def __init__(self):
        """Initialize the claim extractor."""
        config = get_config()
        
        self.api_key = config.api.anthropic_api_key
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Required for claim extraction. "
                "Get a key at https://console.anthropic.com/"
            )
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = config.pipeline.claim_extraction.model
        self.max_tokens = config.pipeline.claim_extraction.max_tokens
        self.temperature = config.pipeline.claim_extraction.temperature
        
        # Chunking settings for long documents
        self.max_chunk_size = 15000  # Characters
        self.chunk_overlap = 500
        
        logger.info(f"ClaimExtractor initialized with model: {self.model}")
    
    def extract_claims(
        self,
        text: str,
        company: str,
        ticker: str,
        source_type: str = "document",
        document_date: Optional[datetime] = None
    ) -> List[ExtractedClaim]:
        """
        Extract claims from a document.
        
        Args:
            text: Document text
            company: Company name
            ticker: Stock ticker
            source_type: Type of document (sec_filing, news, press_release)
            document_date: Optional document date
            
        Returns:
            List of ExtractedClaim objects
        """
        if not text or len(text.strip()) < 50:
            logger.debug("Text too short for claim extraction")
            return []
        
        # Chunk long documents
        chunks = self._chunk_text(text)
        logger.info(f"Processing {len(chunks)} chunks for {ticker}")
        
        all_claims = []
        
        for i, chunk in enumerate(chunks):
            try:
                claims = self._extract_from_chunk(
                    chunk, company, ticker, source_type
                )
                all_claims.extend(claims)
                logger.debug(f"Extracted {len(claims)} claims from chunk {i+1}")
            except Exception as e:
                logger.error(f"Error extracting claims from chunk {i+1}: {e}")
                continue
        
        # Deduplicate claims
        unique_claims = self._deduplicate_claims(all_claims)
        
        logger.info(f"Extracted {len(unique_claims)} unique claims for {ticker}")
        return unique_claims
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for processing.
        
        Args:
            text: Full document text
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            
            # Try to break at paragraph boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.max_chunk_size * 0.5:
                    end = para_break
                else:
                    # Look for sentence break
                    sent_break = text.rfind(". ", start, end)
                    if sent_break > start + self.max_chunk_size * 0.5:
                        end = sent_break + 1
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        
        return chunks
    
    def _extract_from_chunk(
        self,
        text: str,
        company: str,
        ticker: str,
        source_type: str
    ) -> List[ExtractedClaim]:
        """
        Extract claims from a single text chunk using Claude.
        
        Args:
            text: Text chunk
            company: Company name
            ticker: Stock ticker
            source_type: Document type
            
        Returns:
            List of extracted claims
        """
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            source_type=source_type,
            company=company,
            ticker=ticker,
            text=text
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract response text
            response_text = response.content[0].text
            
            # Parse JSON response
            claims_data = self._parse_json_response(response_text)
            
            # Convert to ExtractedClaim objects
            claims = []
            for data in claims_data:
                try:
                    claim = ExtractedClaim(
                        claim_text=data.get("claim_text", ""),
                        speaker=data.get("speaker"),
                        speaker_role=data.get("speaker_role"),
                        topic=data.get("topic", "other"),
                        claim_type=data.get("claim_type"),
                        direction=data.get("direction", "neutral"),
                        entities=data.get("entities", []),
                        confidence=data.get("confidence", 0.8),
                        source_text=text[:500] if len(text) > 500 else text
                    )
                    
                    if claim.claim_text:  # Only add if we have claim text
                        claims.append(claim)
                        
                except Exception as e:
                    logger.warning(f"Error creating claim object: {e}")
                    continue
            
            return claims
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return []
    
    def _parse_json_response(self, response_text: str) -> List[Dict]:
        """
        Parse JSON from Claude's response.
        
        Handles various response formats and edge cases.
        
        Args:
            response_text: Raw response text
            
        Returns:
            List of claim dictionaries
        """
        # Try direct JSON parse
        try:
            data = json.loads(response_text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from response
        # Look for array pattern
        array_match = re.search(r'\[[\s\S]*\]', response_text)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass
        
        # Look for object pattern
        obj_match = re.search(r'\{[\s\S]*\}', response_text)
        if obj_match:
            try:
                obj = json.loads(obj_match.group())
                return [obj]
            except json.JSONDecodeError:
                pass
        
        logger.warning("Could not parse JSON from response")
        return []
    
    def _deduplicate_claims(
        self,
        claims: List[ExtractedClaim]
    ) -> List[ExtractedClaim]:
        """
        Remove duplicate claims based on text similarity.
        
        Args:
            claims: List of extracted claims
            
        Returns:
            Deduplicated list
        """
        if not claims:
            return []
        
        seen_texts = set()
        unique = []
        
        for claim in claims:
            # Normalize claim text for comparison
            normalized = claim.claim_text.lower().strip()
            normalized = re.sub(r'\s+', ' ', normalized)
            
            # Check for near-duplicates
            is_duplicate = False
            for seen in seen_texts:
                # Simple overlap check
                if normalized in seen or seen in normalized:
                    is_duplicate = True
                    break
                
                # Check word overlap
                words1 = set(normalized.split())
                words2 = set(seen.split())
                overlap = len(words1 & words2) / max(len(words1), len(words2))
                if overlap > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts.add(normalized)
                unique.append(claim)
        
        return unique
    
    def extract_from_quotes(
        self,
        quotes: List[str],
        company: str,
        ticker: str,
        speaker: Optional[str] = None,
        speaker_role: Optional[str] = None
    ) -> List[ExtractedClaim]:
        """
        Extract claims from a list of pre-extracted quotes.
        
        This is more efficient than processing full documents when
        quotes have already been extracted.
        
        Args:
            quotes: List of quote strings
            company: Company name
            ticker: Stock ticker
            speaker: Optional speaker name
            speaker_role: Optional speaker role
            
        Returns:
            List of extracted claims
        """
        if not quotes:
            return []
        
        # Combine quotes into a structured text
        text = "\n\n".join([f'"{q}"' for q in quotes])
        
        claims = self.extract_claims(
            text=text,
            company=company,
            ticker=ticker,
            source_type="quotes"
        )
        
        # Override speaker info if provided
        if speaker or speaker_role:
            for claim in claims:
                if speaker and not claim.speaker:
                    claim.speaker = speaker
                if speaker_role and not claim.speaker_role:
                    claim.speaker_role = speaker_role
        
        return claims
