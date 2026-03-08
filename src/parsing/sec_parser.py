"""
SEC Filing Parser

This module provides functionality for parsing SEC filings (10-K, 10-Q, 8-K)
and extracting structured sections. It handles the complex HTML/XML formats
used in SEC EDGAR filings.

Usage:
    from src.parsing.sec_parser import SECParser
    
    parser = SECParser()
    sections = parser.parse_10k(html_content)
"""

import re
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from loguru import logger

from src.parsing.document_cleaner import DocumentCleaner


class SECParser:
    """
    Parses SEC filings and extracts structured sections.
    
    This class handles the parsing of 10-K, 10-Q, and 8-K filings,
    extracting key sections like MD&A, Risk Factors, and Business.
    
    Attributes:
        cleaner: DocumentCleaner instance for text cleaning
    """
    
    # Section patterns for 10-K filings
    SECTION_10K = {
        "item_1": (r"item\s*1[.\s:]+business", r"item\s*1a"),
        "item_1a": (r"item\s*1a[.\s:]+risk\s*factors", r"item\s*1b|item\s*2"),
        "item_1b": (r"item\s*1b[.\s:]+unresolved", r"item\s*1c|item\s*2"),
        "item_2": (r"item\s*2[.\s:]+properties", r"item\s*3"),
        "item_3": (r"item\s*3[.\s:]+legal\s*proceedings", r"item\s*4"),
        "item_7": (r"item\s*7[.\s:]+management.s\s*discussion|item\s*7[.\s:]+md&a", r"item\s*7a|item\s*8"),
        "item_7a": (r"item\s*7a[.\s:]+quantitative", r"item\s*8"),
        "item_8": (r"item\s*8[.\s:]+financial\s*statements", r"item\s*9"),
    }
    
    # Section patterns for 10-Q filings
    SECTION_10Q = {
        "part1_item1": (r"part\s*i.*item\s*1[.\s:]+financial", r"item\s*2"),
        "part1_item2": (r"part\s*i.*item\s*2[.\s:]+management|item\s*2[.\s:]+md&a", r"item\s*3|item\s*4"),
        "part1_item3": (r"item\s*3[.\s:]+quantitative", r"item\s*4"),
        "part1_item4": (r"item\s*4[.\s:]+controls", r"part\s*ii"),
        "part2_item1": (r"part\s*ii.*item\s*1[.\s:]+legal", r"item\s*1a|item\s*2"),
        "part2_item1a": (r"part\s*ii.*item\s*1a[.\s:]+risk", r"item\s*2|item\s*3"),
    }
    
    # Section patterns for 8-K filings
    SECTION_8K = {
        "item_1_01": (r"item\s*1\.01", r"item\s*\d"),
        "item_2_02": (r"item\s*2\.02", r"item\s*\d|signature"),
        "item_5_02": (r"item\s*5\.02", r"item\s*\d|signature"),
        "item_7_01": (r"item\s*7\.01", r"item\s*\d|signature"),
        "item_8_01": (r"item\s*8\.01", r"item\s*\d|signature"),
        "item_9_01": (r"item\s*9\.01", r"signature"),
        "exhibit_99": (r"exhibit\s*99", r"signature|$"),
    }
    
    def __init__(self):
        """Initialize the SEC parser."""
        self.cleaner = DocumentCleaner()
    
    def parse(
        self,
        content: str,
        filing_type: str
    ) -> Dict[str, str]:
        """
        Parse an SEC filing and extract sections.
        
        Args:
            content: Raw filing content (HTML or text)
            filing_type: Type of filing (10-K, 10-Q, 8-K)
            
        Returns:
            Dictionary mapping section names to content
        """
        filing_type = filing_type.upper()
        
        if filing_type == "10-K":
            return self.parse_10k(content)
        elif filing_type == "10-Q":
            return self.parse_10q(content)
        elif filing_type == "8-K":
            return self.parse_8k(content)
        else:
            logger.warning(f"Unknown filing type: {filing_type}")
            return {"full_text": self._extract_text(content)}
    
    def parse_10k(self, content: str) -> Dict[str, str]:
        """
        Parse a 10-K annual report.
        
        Args:
            content: Raw 10-K content
            
        Returns:
            Dictionary of extracted sections
        """
        text = self._extract_text(content)
        return self._extract_sections(text, self.SECTION_10K)
    
    def parse_10q(self, content: str) -> Dict[str, str]:
        """
        Parse a 10-Q quarterly report.
        
        Args:
            content: Raw 10-Q content
            
        Returns:
            Dictionary of extracted sections
        """
        text = self._extract_text(content)
        return self._extract_sections(text, self.SECTION_10Q)
    
    def parse_8k(self, content: str) -> Dict[str, str]:
        """
        Parse an 8-K current report.
        
        Args:
            content: Raw 8-K content
            
        Returns:
            Dictionary of extracted sections
        """
        text = self._extract_text(content)
        return self._extract_sections(text, self.SECTION_8K)
    
    def _extract_text(self, content: str) -> str:
        """
        Extract plain text from HTML/XML content.
        
        Args:
            content: Raw HTML/XML content
            
        Returns:
            Cleaned plain text
        """
        # Check if content is HTML
        if "<" in content and ">" in content:
            soup = BeautifulSoup(content, "lxml")
            
            # Remove script and style elements
            for element in soup(["script", "style", "head"]):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator="\n")
        else:
            text = content
        
        # Clean the text
        text = self.cleaner.clean(text)
        
        return text
    
    def _extract_sections(
        self,
        text: str,
        patterns: Dict[str, Tuple[str, str]]
    ) -> Dict[str, str]:
        """
        Extract sections from text using regex patterns.
        
        Args:
            text: Plain text content
            patterns: Dictionary of (start_pattern, end_pattern) tuples
            
        Returns:
            Dictionary of extracted sections
        """
        sections = {}
        text_lower = text.lower()
        
        for section_name, (start_pattern, end_pattern) in patterns.items():
            try:
                # Find section start
                start_match = re.search(start_pattern, text_lower, re.IGNORECASE)
                if not start_match:
                    continue
                
                start_pos = start_match.start()
                
                # Find section end
                end_pos = len(text)
                if end_pattern:
                    # Search for end pattern after start
                    search_text = text_lower[start_pos + 50:]
                    end_match = re.search(end_pattern, search_text, re.IGNORECASE)
                    if end_match:
                        end_pos = start_pos + 50 + end_match.start()
                
                # Extract section
                section_text = text[start_pos:end_pos]
                
                # Limit section size
                if len(section_text) > 100000:
                    section_text = section_text[:100000] + "\n[TRUNCATED]"
                
                sections[section_name] = section_text.strip()
                
            except Exception as e:
                logger.warning(f"Error extracting section {section_name}: {e}")
        
        return sections
    
    def extract_tables(self, content: str) -> List[Dict]:
        """
        Extract tables from SEC filing.
        
        Args:
            content: Raw HTML content
            
        Returns:
            List of table dictionaries with headers and rows
        """
        tables = []
        
        if "<table" not in content.lower():
            return tables
        
        soup = BeautifulSoup(content, "lxml")
        
        for table in soup.find_all("table"):
            try:
                rows = []
                headers = []
                
                # Extract headers
                header_row = table.find("tr")
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
                
                # Extract data rows
                for row in table.find_all("tr")[1:]:
                    cells = [td.get_text(strip=True) for td in row.find_all("td")]
                    if cells:
                        rows.append(cells)
                
                if headers or rows:
                    tables.append({
                        "headers": headers,
                        "rows": rows
                    })
                    
            except Exception as e:
                logger.warning(f"Error extracting table: {e}")
        
        return tables
    
    def extract_exhibits(self, content: str) -> Dict[str, str]:
        """
        Extract exhibit content from 8-K filings.
        
        Exhibits often contain press releases and other important
        documents attached to 8-K filings.
        
        Args:
            content: Raw filing content
            
        Returns:
            Dictionary mapping exhibit numbers to content
        """
        exhibits = {}
        text = self._extract_text(content)
        text_lower = text.lower()
        
        # Find exhibit sections
        exhibit_pattern = r"exhibit\s*(\d+(?:\.\d+)?)"
        
        for match in re.finditer(exhibit_pattern, text_lower):
            exhibit_num = match.group(1)
            start_pos = match.start()
            
            # Find end of exhibit
            next_match = re.search(exhibit_pattern, text_lower[start_pos + 20:])
            end_pos = start_pos + 20 + next_match.start() if next_match else len(text)
            
            # Also check for signature section
            sig_match = re.search(r"signature", text_lower[start_pos:end_pos])
            if sig_match:
                end_pos = start_pos + sig_match.start()
            
            exhibit_text = text[start_pos:end_pos].strip()
            
            if len(exhibit_text) > 100:
                exhibits[f"exhibit_{exhibit_num}"] = exhibit_text[:50000]
        
        return exhibits
