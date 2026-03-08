"""
Embedding-based Claim Matcher

This module provides semantic similarity matching for claims using
sentence-transformers embeddings. It enables finding related claims
that discuss similar topics even if they use different wording.

The matcher uses:
- Pre-trained sentence-transformer models
- Cosine similarity for ranking
- Configurable similarity thresholds

Usage:
    from src.claim_matching.embedding_matcher import EmbeddingMatcher
    
    matcher = EmbeddingMatcher()
    
    # Generate embedding for a claim
    embedding = matcher.embed_text("Revenue will grow 15% next quarter")
    
    # Find similar claims
    matches = matcher.find_matches(embedding, candidate_embeddings)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from src.config import get_config


@dataclass
class ClaimMatch:
    """
    A match between two claims.
    
    Attributes:
        claim_id: ID of the matched claim
        similarity: Cosine similarity score (0-1)
        claim_text: Text of the matched claim
        topic_match: Whether topics match
    """
    claim_id: int
    similarity: float
    claim_text: str
    topic_match: bool = False


class EmbeddingMatcher:
    """
    Matches claims using sentence embeddings.
    
    This class handles:
    - Loading sentence-transformer models
    - Generating embeddings for claim texts
    - Computing similarity scores
    - Finding best matches above threshold
    
    Attributes:
        model: Sentence-transformer model
        similarity_threshold: Minimum similarity for a match
        device: Compute device (cuda/mps/cpu)
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding matcher.
        
        Args:
            model_name: Name of sentence-transformer model to use.
                       If None, uses model from config.
        """
        config = get_config()
        
        self.model_name = model_name or config.pipeline.claim_matching.embedding_model
        self.similarity_threshold = config.pipeline.claim_matching.similarity_threshold
        self.min_similarity = config.pipeline.claim_matching.min_similarity
        self.max_matches = config.pipeline.claim_matching.max_matches
        
        # Load model lazily
        self._model = None
        self._device = config.get_device()
        
        logger.info(f"EmbeddingMatcher initialized with model: {self.model_name}")
    
    @property
    def model(self):
        """Lazy load the sentence-transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                logger.info(f"Loading sentence-transformer model: {self.model_name}")
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self._device
                )
                logger.info(f"Model loaded on device: {self._device}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
        return self._model
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        if not text:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of embeddings (num_texts x embedding_dim)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1 for normalized embeddings)
        """
        # Embeddings are already normalized, so dot product = cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def compute_similarity_matrix(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise similarities between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings (n x d)
            embeddings2: Second set of embeddings (m x d)
            
        Returns:
            Similarity matrix (n x m)
        """
        # Matrix multiplication for batch cosine similarity
        return np.dot(embeddings1, embeddings2.T)
    
    def find_matches(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_ids: Optional[List[int]] = None,
        candidate_texts: Optional[List[str]] = None,
        min_similarity: Optional[float] = None,
        max_matches: Optional[int] = None
    ) -> List[ClaimMatch]:
        """
        Find matching claims for a query.
        
        Args:
            query_embedding: Embedding of the query claim
            candidate_embeddings: Embeddings of candidate claims
            candidate_ids: Optional IDs for candidates
            candidate_texts: Optional texts for candidates
            min_similarity: Minimum similarity threshold
            max_matches: Maximum number of matches to return
            
        Returns:
            List of ClaimMatch objects sorted by similarity
        """
        min_similarity = min_similarity or self.min_similarity
        max_matches = max_matches or self.max_matches
        
        if len(candidate_embeddings) == 0:
            return []
        
        # Compute similarities
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # Filter by threshold and sort
        mask = similarities >= min_similarity
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            return []
        
        # Sort by similarity (descending)
        sorted_indices = indices[np.argsort(similarities[indices])[::-1]]
        
        # Limit to max matches
        sorted_indices = sorted_indices[:max_matches]
        
        # Build match objects
        matches = []
        for idx in sorted_indices:
            match = ClaimMatch(
                claim_id=candidate_ids[idx] if candidate_ids else int(idx),
                similarity=float(similarities[idx]),
                claim_text=candidate_texts[idx] if candidate_texts else ""
            )
            matches.append(match)
        
        return matches
    
    def find_all_matches(
        self,
        query_embeddings: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_ids: Optional[List[int]] = None,
        min_similarity: Optional[float] = None
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Find matches for multiple queries.
        
        Args:
            query_embeddings: Embeddings of query claims
            candidate_embeddings: Embeddings of candidate claims
            candidate_ids: Optional IDs for candidates
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary mapping query index to list of (candidate_id, similarity)
        """
        min_similarity = min_similarity or self.min_similarity
        
        if len(query_embeddings) == 0 or len(candidate_embeddings) == 0:
            return {}
        
        # Compute similarity matrix
        sim_matrix = self.compute_similarity_matrix(query_embeddings, candidate_embeddings)
        
        # Build matches for each query
        results = {}
        for i in range(len(query_embeddings)):
            similarities = sim_matrix[i]
            mask = similarities >= min_similarity
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                # Sort by similarity
                sorted_indices = indices[np.argsort(similarities[indices])[::-1]]
                sorted_indices = sorted_indices[:self.max_matches]
                
                matches = []
                for idx in sorted_indices:
                    cid = candidate_ids[idx] if candidate_ids else int(idx)
                    matches.append((cid, float(similarities[idx])))
                
                results[i] = matches
        
        return results
    
    def deduplicate_claims(
        self,
        texts: List[str],
        threshold: float = 0.9
    ) -> List[int]:
        """
        Find duplicate claims based on embedding similarity.
        
        Args:
            texts: List of claim texts
            threshold: Similarity threshold for considering duplicates
            
        Returns:
            List of indices to keep (non-duplicates)
        """
        if len(texts) <= 1:
            return list(range(len(texts)))
        
        embeddings = self.embed_batch(texts)
        sim_matrix = self.compute_similarity_matrix(embeddings, embeddings)
        
        # Keep track of which indices to keep
        keep = []
        duplicate_of = {}
        
        for i in range(len(texts)):
            if i in duplicate_of:
                continue
            
            keep.append(i)
            
            # Mark all similar claims as duplicates
            for j in range(i + 1, len(texts)):
                if j not in duplicate_of and sim_matrix[i, j] >= threshold:
                    duplicate_of[j] = i
        
        return keep


class EmbeddingIndex:
    """
    Index for efficient embedding similarity search.
    
    For small datasets, uses brute-force search.
    For larger datasets, could be extended to use FAISS or similar.
    """
    
    def __init__(self):
        """Initialize the index."""
        self.embeddings: List[np.ndarray] = []
        self.ids: List[int] = []
        self.texts: List[str] = []
        self._matrix: Optional[np.ndarray] = None
    
    def add(
        self,
        embedding: np.ndarray,
        claim_id: int,
        text: str = ""
    ):
        """
        Add an embedding to the index.
        
        Args:
            embedding: Claim embedding
            claim_id: Claim identifier
            text: Optional claim text
        """
        self.embeddings.append(embedding)
        self.ids.append(claim_id)
        self.texts.append(text)
        self._matrix = None  # Invalidate cached matrix
    
    def add_batch(
        self,
        embeddings: np.ndarray,
        claim_ids: List[int],
        texts: Optional[List[str]] = None
    ):
        """
        Add multiple embeddings to the index.
        
        Args:
            embeddings: Array of embeddings
            claim_ids: List of claim IDs
            texts: Optional list of claim texts
        """
        texts = texts or [""] * len(claim_ids)
        
        for emb, cid, text in zip(embeddings, claim_ids, texts):
            self.embeddings.append(emb)
            self.ids.append(cid)
            self.texts.append(text)
        
        self._matrix = None
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        threshold: float = 0.5
    ) -> List[Tuple[int, float, str]]:
        """
        Search for similar embeddings.
        
        Args:
            query: Query embedding
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (claim_id, similarity, text) tuples
        """
        if len(self.embeddings) == 0:
            return []
        
        # Build matrix if needed
        if self._matrix is None:
            self._matrix = np.array(self.embeddings)
        
        # Compute similarities
        similarities = np.dot(self._matrix, query)
        
        # Filter and sort
        mask = similarities >= threshold
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            return []
        
        sorted_indices = indices[np.argsort(similarities[indices])[::-1]][:k]
        
        results = [
            (self.ids[i], float(similarities[i]), self.texts[i])
            for i in sorted_indices
        ]
        
        return results
    
    def __len__(self) -> int:
        """Return number of embeddings in index."""
        return len(self.embeddings)
