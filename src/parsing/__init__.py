"""
Document Parsing Module

This module provides functionality for parsing and cleaning documents
from various sources. It handles:
- SEC filing HTML/XML parsing
- News article text extraction
- Text cleaning and normalization

Example:
    from src.parsing import SECParser, DocumentCleaner
    
    parser = SECParser()
    sections = parser.parse_10k(html_content)
    
    cleaner = DocumentCleaner()
    clean_text = cleaner.clean(raw_text)
"""

from src.parsing.sec_parser import SECParser
from src.parsing.news_parser import NewsParser
from src.parsing.document_cleaner import DocumentCleaner

__all__ = [
    "SECParser",
    "NewsParser",
    "DocumentCleaner",
]
