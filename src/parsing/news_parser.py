"""
News Article Parser

This module provides functionality for parsing news articles and
extracting structured information like quotes, entities, and sentiment.

Usage:
    from src.parsing.news_parser import NewsParser
    
    parser = NewsParser()
    result = parser.parse(article_content)
    quotes = parser.extract_quotes(article_content, company="Apple")
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.parsing.document_cleaner import DocumentCleaner


@dataclass
class ExtractedQuote:
    """
    A quote extracted from a news article.
    
    Attributes:
        text: The quote text
        speaker: Name of the speaker
        role: Role/title of speaker (CEO, CFO, etc.)
        context: Surrounding context
    """
    text: str
    speaker: Optional[str] = None
    role: Optional[str] = None
    context: Optional[str] = None


@dataclass
class ParsedArticle:
    """
    A parsed news article with extracted information.
    
    Attributes:
        title: Article headline
        content: Cleaned article content
        quotes: List of extracted quotes
        entities: Extracted named entities
        summary: Article summary (first paragraph)
    """
    title: str
    content: str
    quotes: List[ExtractedQuote]
    entities: Dict[str, List[str]]
    summary: str


class NewsParser:
    """
    Parses news articles and extracts structured information.
    
    This class handles:
    - Text cleaning and normalization
    - Quote extraction with speaker attribution
    - Named entity recognition
    - Summary extraction
    
    Attributes:
        cleaner: DocumentCleaner instance
    """
    
    # Executive role patterns
    ROLE_PATTERNS = [
        (r'\b(CEO|chief\s+executive\s+officer)\b', 'CEO'),
        (r'\b(CFO|chief\s+financial\s+officer)\b', 'CFO'),
        (r'\b(COO|chief\s+operating\s+officer)\b', 'COO'),
        (r'\b(CTO|chief\s+technology\s+officer)\b', 'CTO'),
        (r'\b(president)\b', 'President'),
        (r'\b(chairman)\b', 'Chairman'),
        (r'\b(founder)\b', 'Founder'),
        (r'\b(spokesperson|spokesman|spokeswoman)\b', 'Spokesperson'),
    ]
    
    # Quote attribution patterns
    QUOTE_PATTERNS = [
        # "quote," said Name, Title
        r'"([^"]{15,500})"[,\s]+said\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        # Name said "quote"
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+said[,\s]+"([^"]{15,500})"',
        # According to Name, "quote"
        r'[Aa]ccording\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)[,\s]+"([^"]{15,500})"',
        # Name, Title, said "quote"
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),\s+(?:the\s+)?(?:CEO|CFO|COO|president|chairman)[,\s]+said[,\s]+"([^"]{15,500})"',
    ]
    
    def __init__(self):
        """Initialize the news parser."""
        self.cleaner = DocumentCleaner()
    
    def parse(
        self,
        content: str,
        title: Optional[str] = None
    ) -> ParsedArticle:
        """
        Parse a news article and extract structured information.
        
        Args:
            content: Raw article content
            title: Optional article title
            
        Returns:
            ParsedArticle with extracted information
        """
        # Clean content
        cleaned_content = self.cleaner.clean(content)
        
        # Extract quotes
        quotes = self.extract_quotes(cleaned_content)
        
        # Extract entities
        entities = self.extract_entities(cleaned_content)
        
        # Extract summary (first paragraph)
        paragraphs = self.cleaner.extract_paragraphs(cleaned_content)
        summary = paragraphs[0] if paragraphs else ""
        
        return ParsedArticle(
            title=title or "",
            content=cleaned_content,
            quotes=quotes,
            entities=entities,
            summary=summary
        )
    
    def extract_quotes(
        self,
        content: str,
        company: Optional[str] = None
    ) -> List[ExtractedQuote]:
        """
        Extract quotes from article content.
        
        Args:
            content: Article text
            company: Optional company name for context
            
        Returns:
            List of ExtractedQuote objects
        """
        quotes = []
        
        for pattern in self.QUOTE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            
            for match in matches:
                # Pattern returns (quote, speaker) or (speaker, quote)
                if len(match) == 2:
                    # Determine which is quote and which is speaker
                    if match[0].startswith('"') or len(match[0]) > len(match[1]):
                        quote_text = match[0].strip('"')
                        speaker = match[1]
                    else:
                        speaker = match[0]
                        quote_text = match[1].strip('"')
                    
                    # Detect role
                    role = self._detect_role(content, speaker)
                    
                    # Get context
                    context = self._get_quote_context(content, quote_text)
                    
                    quotes.append(ExtractedQuote(
                        text=quote_text,
                        speaker=speaker,
                        role=role,
                        context=context
                    ))
        
        # Remove duplicates
        seen = set()
        unique_quotes = []
        for quote in quotes:
            if quote.text not in seen:
                seen.add(quote.text)
                unique_quotes.append(quote)
        
        return unique_quotes
    
    def _detect_role(self, content: str, speaker: str) -> Optional[str]:
        """
        Detect the role/title of a speaker.
        
        Args:
            content: Full article content
            speaker: Speaker name
            
        Returns:
            Role string or None
        """
        # Look for role near speaker mention
        speaker_pattern = re.escape(speaker)
        
        for role_pattern, role_name in self.ROLE_PATTERNS:
            # Look for role within 50 characters of speaker
            combined_pattern = f"{speaker_pattern}.{{0,50}}{role_pattern}|{role_pattern}.{{0,50}}{speaker_pattern}"
            
            if re.search(combined_pattern, content, re.IGNORECASE):
                return role_name
        
        return None
    
    def _get_quote_context(
        self,
        content: str,
        quote: str,
        context_chars: int = 200
    ) -> str:
        """
        Get surrounding context for a quote.
        
        Args:
            content: Full article content
            quote: Quote text
            context_chars: Number of characters of context
            
        Returns:
            Context string
        """
        quote_pos = content.find(quote)
        if quote_pos == -1:
            return ""
        
        start = max(0, quote_pos - context_chars)
        end = min(len(content), quote_pos + len(quote) + context_chars)
        
        return content[start:end]
    
    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """
        Extract named entities from content.
        
        Extracts:
        - Organizations
        - People
        - Locations
        - Money amounts
        - Percentages
        
        Args:
            content: Article text
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        entities = {
            "organizations": [],
            "people": [],
            "locations": [],
            "money": [],
            "percentages": [],
        }
        
        # Extract money amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?\s*(?:billion|million|thousand)?|\d+(?:\.\d+)?\s*(?:billion|million)\s*dollars?'
        entities["money"] = list(set(re.findall(money_pattern, content, re.IGNORECASE)))
        
        # Extract percentages
        pct_pattern = r'\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*percent'
        entities["percentages"] = list(set(re.findall(pct_pattern, content, re.IGNORECASE)))
        
        # Extract potential company names (capitalized words followed by Inc., Corp., etc.)
        org_pattern = r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|Corporation|Company|Ltd|LLC|Co)\.?)?'
        potential_orgs = re.findall(org_pattern, content)
        
        # Filter common false positives
        stop_words = {'The', 'This', 'That', 'These', 'Those', 'What', 'When', 'Where', 'How', 'Why'}
        entities["organizations"] = list(set(
            org for org in potential_orgs 
            if org not in stop_words and len(org) > 3
        ))[:20]
        
        return entities
    
    def extract_ceo_statements(
        self,
        content: str,
        company: Optional[str] = None
    ) -> List[ExtractedQuote]:
        """
        Extract statements specifically from CEO.
        
        Args:
            content: Article content
            company: Optional company name
            
        Returns:
            List of CEO quotes
        """
        all_quotes = self.extract_quotes(content, company)
        
        return [q for q in all_quotes if q.role == "CEO"]
    
    def identify_topic(self, content: str) -> str:
        """
        Identify the main topic of the article.
        
        Args:
            content: Article content
            
        Returns:
            Topic category string
        """
        content_lower = content.lower()
        
        # Topic keywords
        topic_keywords = {
            "guidance": ["guidance", "outlook", "forecast", "expects", "projects", "revenue target"],
            "projects": ["factory", "plant", "facility", "expansion", "construction", "production"],
            "legal_regulatory": ["lawsuit", "litigation", "investigation", "regulatory", "sec", "doj", "settlement"],
            "supply_chain": ["supply chain", "supplier", "logistics", "inventory", "shortage"],
            "capital_expenditure": ["capex", "capital expenditure", "investment", "spending"],
            "risk_factors": ["risk", "warning", "concern", "threat", "challenge"],
            "personnel": ["ceo", "cfo", "executive", "hired", "resigned", "appointed"],
            "customers": ["customer", "client", "contract", "deal", "partnership"],
            "products": ["product", "launch", "release", "update", "feature"],
        }
        
        # Count keyword matches per topic
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores.keys(), key=lambda k: topic_scores[k])
        
        return "other"
