"""
Topic Model Analyzer

This module provides automated topic discovery and analysis
using BERTopic for clustering claims into coherent themes.

Features:
- Automatic topic discovery from claims corpus
- Topic labeling and interpretation
- Topic similarity and hierarchy analysis
- Integration with claim matching pipeline
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.config import get_config


class TopicModelAnalyzer:
    """
    Analyzes claims using neural topic modeling.
    
    Uses BERTopic for automatic topic discovery, providing
    more nuanced topic assignment than rule-based matching.
    
    Attributes:
        model: BERTopic model instance
        embedding_model: Sentence transformer for embeddings
        topics: Discovered topic labels
    """
    
    def __init__(self, embedding_model: Optional[str] = None):
        """
        Initialize the topic model analyzer.
        
        Args:
            embedding_model: Name of sentence-transformer model.
                            Defaults to config setting.
        """
        config = get_config()
        self.embedding_model_name = (
            embedding_model or 
            config.pipeline.claim_matching.embedding_model
        )
        
        # Lazy loading
        self._model = None
        self._embedder = None
        self.topics: List[str] = []
        
        logger.info(f"TopicModelAnalyzer initialized with {self.embedding_model_name}")
    
    @property
    def model(self):
        """Lazy load BERTopic model."""
        if self._model is None:
            try:
                from bertopic import BERTopic
                from sentence_transformers import SentenceTransformer
                
                self._embedder = SentenceTransformer(self.embedding_model_name)
                self._model = BERTopic(
                    embedding_model=self._embedder,
                    verbose=False
                )
                logger.info("BERTopic model loaded")
            except ImportError:
                logger.warning("BERTopic not installed. Using fallback.")
                self._model = None
        return self._model
    
    def fit(self, texts: List[str]) -> "TopicModelAnalyzer":
        """
        Fit the topic model to a corpus of texts.
        
        Args:
            texts: List of claim texts to cluster
            
        Returns:
            Self for method chaining
        """
        if self.model is None:
            logger.warning("Model not available, skipping fit")
            return self
        
        logger.info(f"Fitting topic model on {len(texts)} texts")
        
        topics, probs = self.model.fit_transform(texts)
        
        # Get topic labels
        topic_info = self.model.get_topic_info()
        self.topics = topic_info["Name"].tolist()
        
        logger.info(f"Discovered {len(self.topics)} topics")
        return self
    
    def get_topics(self, texts: List[str]) -> List[Tuple[int, str, float]]:
        """
        Get topic assignments for texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of (topic_id, topic_label, probability) tuples
        """
        if self.model is None:
            return [(-1, "unknown", 0.0) for _ in texts]
        
        topics, probs = self.model.transform(texts)
        
        results = []
        for i, (topic_id, prob) in enumerate(zip(topics, probs)):
            if topic_id == -1:
                label = "outlier"
            else:
                label = self.model.get_topic(topic_id)[0][0] if topic_id >= 0 else "unknown"
            results.append((topic_id, label, float(prob) if prob is not None else 0.0))
        
        return results
    
    def get_topic_similarity(
        self,
        topic1: int,
        topic2: int
    ) -> float:
        """
        Compute similarity between two topics.
        
        Args:
            topic1: First topic ID
            topic2: Second topic ID
            
        Returns:
            Similarity score 0-1
        """
        if self.model is None:
            return 0.0
        
        try:
            # Get topic embeddings
            topic1_words = self.model.get_topic(topic1)
            topic2_words = self.model.get_topic(topic2)
            
            if not topic1_words or not topic2_words:
                return 0.0
            
            # Compare top words
            words1 = set(w for w, _ in topic1_words[:10])
            words2 = set(w for w, _ in topic2_words[:10])
            
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            
            return overlap / total if total > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error computing topic similarity: {e}")
            return 0.0
    
    def get_topic_hierarchy(self) -> Dict:
        """
        Get hierarchical topic structure.
        
        Returns:
            Dictionary with topic hierarchy information
        """
        if self.model is None:
            return {}
        
        try:
            hierarchy = self.model.get_topic_info()
            return hierarchy.to_dict()
        except Exception:
            return {}
    
    def visualize_topics(self, output_path: Optional[str] = None):
        """
        Create topic visualization.
        
        Args:
            output_path: Path to save visualization HTML
        """
        if self.model is None:
            logger.warning("Model not available for visualization")
            return
        
        try:
            fig = self.model.visualize_topics()
            if output_path:
                fig.write_html(output_path)
                logger.info(f"Topic visualization saved to {output_path}")
            return fig
        except Exception as e:
            logger.warning(f"Error creating visualization: {e}")
