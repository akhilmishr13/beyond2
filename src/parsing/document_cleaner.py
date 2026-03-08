"""
Document Cleaner

This module provides text cleaning and normalization functionality
for corporate documents and news articles. It handles:
- Whitespace normalization
- Special character handling
- Boilerplate removal
- Text segmentation

Usage:
    from src.parsing.document_cleaner import DocumentCleaner
    
    cleaner = DocumentCleaner()
    clean_text = cleaner.clean(raw_text)
"""

import re
from typing import List, Optional

from loguru import logger


class DocumentCleaner:
    """
    Cleans and normalizes document text.
    
    This class provides methods for:
    - Removing HTML/XML artifacts
    - Normalizing whitespace
    - Removing boilerplate text
    - Splitting text into sentences
    
    Attributes:
        min_sentence_length: Minimum length for a valid sentence
        max_line_length: Maximum length for a single line
    """
    
    # Common boilerplate patterns to remove
    BOILERPLATE_PATTERNS = [
        r"safe harbor",
        r"forward-looking statements",
        r"this press release contains",
        r"for more information",
        r"about\s+\w+\s+(inc|corp|company)",
        r"©\s*\d{4}",
        r"all rights reserved",
        r"contact:\s*\n",
        r"investor relations",
        r"media contact",
    ]
    
    def __init__(
        self,
        min_sentence_length: int = 20,
        max_line_length: int = 5000
    ):
        """
        Initialize the document cleaner.
        
        Args:
            min_sentence_length: Minimum characters for a valid sentence
            max_line_length: Maximum characters for a line before truncation
        """
        self.min_sentence_length = min_sentence_length
        self.max_line_length = max_line_length
        
        # Compile boilerplate patterns
        self._boilerplate_regex = re.compile(
            "|".join(self.BOILERPLATE_PATTERNS),
            re.IGNORECASE
        )
    
    def clean(
        self,
        text: str,
        remove_boilerplate: bool = True,
        normalize_whitespace: bool = True,
        remove_urls: bool = True
    ) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw input text
            remove_boilerplate: Whether to remove common boilerplate
            normalize_whitespace: Whether to normalize whitespace
            remove_urls: Whether to remove URLs
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML entities
        text = self._remove_html_entities(text)
        
        # Remove URLs if requested
        if remove_urls:
            text = self._remove_urls(text)
        
        # Normalize unicode
        text = self._normalize_unicode(text)
        
        # Normalize whitespace
        if normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        # Remove boilerplate
        if remove_boilerplate:
            text = self._remove_boilerplate(text)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def _remove_html_entities(self, text: str) -> str:
        """Remove HTML entities like &nbsp; &amp; etc."""
        import html
        text = html.unescape(text)
        
        # Remove any remaining HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        # Match http/https URLs
        text = re.sub(
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            '',
            text
        )
        # Match www. URLs
        text = re.sub(
            r'www\.[^\s<>"{}|\\^`\[\]]+',
            '',
            text
        )
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        import unicodedata
        
        # Normalize to NFKC form
        text = unicodedata.normalize('NFKC', text)
        
        # Replace common unicode characters with ASCII equivalents
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2026': '...',  # Ellipsis
            '\u00a0': ' ',  # Non-breaking space
            '\u200b': '',   # Zero-width space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove spaces at start/end of lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def _remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate sections."""
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        # Filter out boilerplate paragraphs
        filtered = []
        for para in paragraphs:
            if len(para) < 50:
                filtered.append(para)
                continue
            
            # Check if paragraph is mostly boilerplate
            if self._boilerplate_regex.search(para):
                # Check what percentage of paragraph is boilerplate
                boilerplate_matches = self._boilerplate_regex.findall(para)
                boilerplate_length = sum(len(m) for m in boilerplate_matches)
                
                if boilerplate_length > len(para) * 0.3:
                    continue
            
            filtered.append(para)
        
        return '\n\n'.join(filtered)
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Uses rule-based sentence splitting optimized for financial text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentence strings
        """
        if not text:
            return []
        
        # Simple sentence splitting on common terminators
        # Handles abbreviations common in financial text
        
        # Protect common abbreviations
        protected = text
        abbreviations = [
            "Inc.", "Corp.", "Ltd.", "Co.", "LLC.",
            "Mr.", "Mrs.", "Ms.", "Dr.",
            "vs.", "etc.", "e.g.", "i.e.",
            "No.", "Vol.", "Rev.",
            "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec.",
            "Q1.", "Q2.", "Q3.", "Q4.",
        ]
        
        for abbr in abbreviations:
            protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))
        
        # Split on sentence terminators
        sentences = re.split(r'(?<=[.!?])\s+', protected)
        
        # Restore dots
        sentences = [s.replace("<DOT>", ".") for s in sentences]
        
        # Filter short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) >= self.min_sentence_length]
        
        return sentences
    
    def extract_paragraphs(
        self,
        text: str,
        min_length: int = 50
    ) -> List[str]:
        """
        Extract meaningful paragraphs from text.
        
        Args:
            text: Input text
            min_length: Minimum paragraph length
            
        Returns:
            List of paragraph strings
        """
        # Split on double newlines
        paragraphs = re.split(r'\n{2,}', text)
        
        # Filter and clean
        result = []
        for para in paragraphs:
            para = para.strip()
            if len(para) >= min_length:
                result.append(para)
        
        return result
    
    def truncate(
        self,
        text: str,
        max_length: int,
        suffix: str = "..."
    ) -> str:
        """
        Truncate text to maximum length.
        
        Attempts to truncate at sentence or word boundaries.
        
        Args:
            text: Input text
            max_length: Maximum output length
            suffix: Suffix to add when truncating
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        # Leave room for suffix
        target_length = max_length - len(suffix)
        
        # Try to truncate at sentence boundary
        sentences = self.split_sentences(text)
        result = ""
        for sentence in sentences:
            if len(result) + len(sentence) + 1 <= target_length:
                result = result + " " + sentence if result else sentence
            else:
                break
        
        if result:
            return result + suffix
        
        # Fall back to word boundary
        truncated = text[:target_length]
        last_space = truncated.rfind(' ')
        if last_space > target_length * 0.5:
            truncated = truncated[:last_space]
        
        return truncated + suffix
