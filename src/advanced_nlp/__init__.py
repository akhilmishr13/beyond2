"""
Advanced NLP Module

This module provides enhanced NLP capabilities for the Corporate Narrative Engine:
- Topic modeling using BERTopic
- RAG-based contradiction reasoning
- LLM hybrid detection combining rule-based and neural approaches

These components extend the core contradiction detection with
more sophisticated analysis capabilities.
"""

from src.advanced_nlp.topic_model import TopicModelAnalyzer
from src.advanced_nlp.rag_reasoning import RAGContradictionReasoner
from src.advanced_nlp.llm_hybrid import LLMHybridDetector

__all__ = [
    "TopicModelAnalyzer",
    "RAGContradictionReasoner",
    "LLMHybridDetector",
]
