"""
Baseline Models Module

This module provides baseline models for comparison with the main
contradiction-aware signal model:

1. Keyword Rules: Pattern-based detection using financial keywords
2. Sentiment Baseline: Sentiment-only prediction without contradiction
3. BoW Classifier: Bag-of-words with TF-IDF features

These baselines establish performance benchmarks and help validate
that the NLP-based approach adds value.

Example:
    from src.baselines import KeywordRules, SentimentBaseline, BoWClassifier
    
    keyword_model = KeywordRules()
    predictions = keyword_model.predict(events)
"""

from src.baselines.keyword_rules import KeywordRules
from src.baselines.sentiment_baseline import SentimentBaseline
from src.baselines.bow_classifier import BoWClassifier

__all__ = [
    "KeywordRules",
    "SentimentBaseline",
    "BoWClassifier",
]
