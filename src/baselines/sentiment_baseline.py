"""
Sentiment Baseline

This module implements a sentiment-only baseline that predicts
market reaction based on the sentiment of the later claim,
without considering contradictions.

This baseline tests whether contradiction detection adds value
beyond simple sentiment analysis.

Usage:
    from src.baselines.sentiment_baseline import SentimentBaseline
    
    model = SentimentBaseline()
    predictions = model.predict(texts)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class SentimentPrediction:
    """
    Prediction from sentiment baseline.
    
    Attributes:
        signal: Predicted signal (bearish, bullish, neutral)
        confidence: Confidence score
        sentiment_score: Raw sentiment score (-1 to 1)
        method: Sentiment method used
    """
    signal: str
    confidence: float
    sentiment_score: float
    method: str


class SentimentBaseline:
    """
    Sentiment-only baseline model.
    
    Uses FinBERT or VADER for sentiment analysis and predicts
    market reaction based solely on sentiment of the later claim.
    
    Attributes:
        method: Sentiment method ('finbert' or 'vader')
        pos_threshold: Threshold for positive signal
        neg_threshold: Threshold for negative signal
    """
    
    def __init__(self, method: str = "vader"):
        """
        Initialize the sentiment baseline.
        
        Args:
            method: Sentiment method to use ('vader' or 'finbert')
        """
        self.method = method
        self.pos_threshold = 0.3
        self.neg_threshold = -0.3
        
        # Lazy loading for models
        self._vader = None
        self._finbert = None
        self._finbert_tokenizer = None
        
        logger.info(f"SentimentBaseline initialized with method: {method}")
    
    @property
    def vader(self):
        """Lazy load VADER sentiment analyzer."""
        if self._vader is None:
            import nltk
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            
            from nltk.sentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        return self._vader
    
    def _get_vader_sentiment(self, text: str) -> float:
        """
        Get sentiment score using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Compound sentiment score (-1 to 1)
        """
        if not text:
            return 0.0
        
        scores = self.vader.polarity_scores(text)
        return scores["compound"]
    
    def _get_finbert_sentiment(self, text: str) -> float:
        """
        Get sentiment score using FinBERT.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1 to 1)
        """
        if self._finbert is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch
                
                self._finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self._finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                self._finbert.eval()
            except Exception as e:
                logger.warning(f"Could not load FinBERT: {e}. Falling back to VADER.")
                return self._get_vader_sentiment(text)
        
        import torch
        
        inputs = self._finbert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self._finbert(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]
        
        # FinBERT outputs: [positive, negative, neutral]
        sentiment_score = probs[0] - probs[1]  # positive - negative
        return float(sentiment_score)
    
    def get_sentiment(self, text: str) -> float:
        """
        Get sentiment score using configured method.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1 to 1)
        """
        if self.method == "finbert":
            return self._get_finbert_sentiment(text)
        else:
            return self._get_vader_sentiment(text)
    
    def predict_single(self, text: str) -> SentimentPrediction:
        """
        Predict signal for a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentPrediction with signal
        """
        sentiment_score = self.get_sentiment(text)
        
        # Map sentiment to signal
        if sentiment_score >= self.pos_threshold:
            signal = "bullish"
            confidence = min(0.9, 0.5 + abs(sentiment_score) * 0.4)
        elif sentiment_score <= self.neg_threshold:
            signal = "bearish"
            confidence = min(0.9, 0.5 + abs(sentiment_score) * 0.4)
        else:
            signal = "neutral"
            confidence = 0.5 + (0.3 - abs(sentiment_score)) * 0.5
        
        return SentimentPrediction(
            signal=signal,
            confidence=confidence,
            sentiment_score=sentiment_score,
            method=self.method
        )
    
    def predict(
        self,
        texts: List[str]
    ) -> List[SentimentPrediction]:
        """
        Predict signals for multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentPrediction objects
        """
        return [self.predict_single(text) for text in texts]
    
    def predict_from_events(
        self,
        events: List[Dict],
        use_later_claim: bool = True
    ) -> List[SentimentPrediction]:
        """
        Predict signals from event dictionaries.
        
        Args:
            events: List of event dictionaries
            use_later_claim: If True, use later claim; else use earlier
            
        Returns:
            List of SentimentPrediction objects
        """
        texts = []
        for event in events:
            if use_later_claim:
                text = event.get("later_claim", {}).get("text", "")
            else:
                text = event.get("earlier_claim", {}).get("text", "")
            texts.append(text)
        
        return self.predict(texts)
