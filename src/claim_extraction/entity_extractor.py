"""
Entity Extractor

This module provides named entity recognition (NER) functionality
for extracting entities from corporate text. It identifies:
- Company names
- People (executives, board members)
- Locations (facilities, headquarters)
- Products and services
- Financial metrics
- Dates and time references

Usage:
    from src.claim_extraction.entity_extractor import EntityExtractor
    
    extractor = EntityExtractor()
    entities = extractor.extract(text)
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from loguru import logger


@dataclass
class Entity:
    """
    A named entity extracted from text.
    
    Attributes:
        text: Entity text
        type: Entity type (ORG, PERSON, LOC, PRODUCT, MONEY, DATE)
        start: Start position in text
        end: End position in text
        confidence: Extraction confidence
    """
    text: str
    type: str
    start: int = 0
    end: int = 0
    confidence: float = 0.8


class EntityExtractor:
    """
    Extracts named entities from corporate documents.
    
    Uses a combination of pattern matching and heuristics optimized
    for financial and corporate text.
    
    Attributes:
        known_companies: Set of known company names
        known_executives: Set of known executive titles
    """
    
    # Entity type patterns
    PATTERNS = {
        "MONEY": [
            r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:billion|million|thousand|B|M|K))?',
            r'[\d,]+(?:\.\d+)?\s*(?:billion|million)\s*(?:dollars?)?',
            r'USD\s*[\d,]+(?:\.\d+)?(?:\s*(?:billion|million))?',
        ],
        "PERCENT": [
            r'\d+(?:\.\d+)?%',
            r'\d+(?:\.\d+)?\s*percent(?:age)?',
            r'\d+(?:\.\d+)?\s*basis\s*points?',
        ],
        "DATE": [
            r'(?:Q[1-4]|first|second|third|fourth)\s+(?:quarter|half)\s+(?:of\s+)?\d{4}',
            r'(?:fiscal\s+)?(?:year\s+)?\d{4}',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?',
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'(?:early|mid|late)\s+\d{4}',
            r'by\s+(?:the\s+)?end\s+of\s+\d{4}',
        ],
        "FACILITY": [
            r'(?:[A-Z][a-z]+\s+)?(?:facility|plant|factory|warehouse|center|campus)',
            r'(?:manufacturing|production|distribution|data)\s+(?:facility|center)',
        ],
        "PRODUCT": [
            r'[A-Z][a-zA-Z0-9]+(?:\s+[A-Z0-9][a-zA-Z0-9]*)*(?:\s+(?:Pro|Max|Ultra|Plus))?',
        ],
    }
    
    # Executive titles
    EXECUTIVE_TITLES = [
        "CEO", "CFO", "COO", "CTO", "CIO", "CMO",
        "Chief Executive Officer", "Chief Financial Officer",
        "Chief Operating Officer", "Chief Technology Officer",
        "President", "Chairman", "Vice President",
        "Executive Vice President", "Senior Vice President",
        "Managing Director", "General Manager",
        "Founder", "Co-Founder",
    ]
    
    # Company suffixes
    COMPANY_SUFFIXES = [
        "Inc", "Inc.", "Corp", "Corp.", "Corporation",
        "Company", "Co", "Co.", "Ltd", "Ltd.",
        "LLC", "L.L.C.", "LLP", "L.L.P.",
        "PLC", "plc", "AG", "SA", "NV",
    ]
    
    def __init__(self):
        """Initialize the entity extractor."""
        # Compile patterns
        self._compiled_patterns = {}
        for entity_type, patterns in self.PATTERNS.items():
            self._compiled_patterns[entity_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        # Build company suffix pattern
        suffix_pattern = "|".join(re.escape(s) for s in self.COMPANY_SUFFIXES)
        self._company_pattern = re.compile(
            rf'([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+(?:{suffix_pattern})\b'
        )
        
        # Build executive pattern
        title_pattern = "|".join(re.escape(t) for t in self.EXECUTIVE_TITLES)
        self._exec_pattern = re.compile(
            rf'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)[,\s]+(?:{title_pattern})|'
            rf'(?:{title_pattern})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            re.IGNORECASE
        )
        
        logger.info("EntityExtractor initialized")
    
    def extract(self, text: str) -> Dict[str, List[Entity]]:
        """
        Extract all entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        entities = {
            "MONEY": [],
            "PERCENT": [],
            "DATE": [],
            "FACILITY": [],
            "ORG": [],
            "PERSON": [],
        }
        
        # Extract pattern-based entities
        for entity_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entities[entity_type].append(Entity(
                        text=match.group().strip(),
                        type=entity_type,
                        start=match.start(),
                        end=match.end()
                    ))
        
        # Extract companies
        for match in self._company_pattern.finditer(text):
            entities["ORG"].append(Entity(
                text=match.group().strip(),
                type="ORG",
                start=match.start(),
                end=match.end()
            ))
        
        # Extract people (executives)
        for match in self._exec_pattern.finditer(text):
            name = match.group(1) or match.group(2)
            if name:
                entities["PERSON"].append(Entity(
                    text=name.strip(),
                    type="PERSON",
                    start=match.start(),
                    end=match.end()
                ))
        
        # Deduplicate
        for entity_type in entities:
            entities[entity_type] = self._deduplicate(entities[entity_type])
        
        return entities
    
    def extract_flat(self, text: str) -> List[Entity]:
        """
        Extract all entities as a flat list.
        
        Args:
            text: Input text
            
        Returns:
            List of all entities
        """
        entities_dict = self.extract(text)
        all_entities = []
        for entity_list in entities_dict.values():
            all_entities.extend(entity_list)
        return sorted(all_entities, key=lambda e: e.start)
    
    def _deduplicate(self, entities: List[Entity]) -> List[Entity]:
        """
        Remove duplicate entities.
        
        Args:
            entities: List of entities
            
        Returns:
            Deduplicated list
        """
        seen: Set[str] = set()
        unique = []
        
        for entity in entities:
            normalized = entity.text.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(entity)
        
        return unique
    
    def extract_financial_metrics(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial metrics from text.
        
        Extracts:
        - Revenue/sales figures
        - Earnings/profit figures
        - Growth rates
        - Margins
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of metric types to values
        """
        metrics = {
            "revenue": [],
            "earnings": [],
            "growth": [],
            "margin": [],
        }
        
        # Revenue patterns
        revenue_pattern = r'revenue\s+(?:of\s+)?(\$?[\d,.]+\s*(?:billion|million)?)'
        for match in re.finditer(revenue_pattern, text, re.IGNORECASE):
            metrics["revenue"].append(match.group(1))
        
        # Earnings patterns
        earnings_pattern = r'(?:earnings|profit|net\s+income)\s+(?:of\s+)?(\$?[\d,.]+\s*(?:billion|million)?)'
        for match in re.finditer(earnings_pattern, text, re.IGNORECASE):
            metrics["earnings"].append(match.group(1))
        
        # Growth patterns
        growth_pattern = r'(?:growth|increase|grew)\s+(?:of\s+)?(\d+(?:\.\d+)?%)'
        for match in re.finditer(growth_pattern, text, re.IGNORECASE):
            metrics["growth"].append(match.group(1))
        
        # Margin patterns
        margin_pattern = r'(?:gross|operating|net|profit)\s+margin\s+(?:of\s+)?(\d+(?:\.\d+)?%)'
        for match in re.finditer(margin_pattern, text, re.IGNORECASE):
            metrics["margin"].append(match.group(1))
        
        return metrics
    
    def extract_time_references(self, text: str) -> List[Dict]:
        """
        Extract time references from text.
        
        Useful for understanding temporal claims about future events.
        
        Args:
            text: Input text
            
        Returns:
            List of time reference dictionaries
        """
        references = []
        
        # Future time patterns
        patterns = [
            (r'by\s+(Q[1-4]\s+\d{4})', "deadline"),
            (r'by\s+((?:early|mid|late)\s+\d{4})', "deadline"),
            (r'by\s+(end\s+of\s+\d{4})', "deadline"),
            (r'in\s+(Q[1-4]\s+\d{4})', "expected"),
            (r'(?:expect|plan|target)(?:s|ed)?\s+(?:to\s+)?(?:by\s+)?(Q[1-4]\s+\d{4}|\d{4})', "projection"),
        ]
        
        for pattern, ref_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                references.append({
                    "text": match.group(1),
                    "type": ref_type,
                    "context": text[max(0, match.start()-50):match.end()+50]
                })
        
        return references
