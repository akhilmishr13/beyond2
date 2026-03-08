"""
RAG-Based Contradiction Reasoning

This module implements Retrieval-Augmented Generation for
enhanced contradiction detection and reasoning.

Features:
- Vector store for historical claims
- Context retrieval for contradiction analysis
- LLM-powered reasoning with retrieved context
- Explanation generation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.config import get_config


class RAGContradictionReasoner:
    """
    Uses retrieval-augmented generation for contradiction reasoning.
    
    Retrieves relevant historical claims and context to provide
    more informed contradiction analysis and explanations.
    
    Attributes:
        embedding_model: Model for computing embeddings
        vector_store: Storage for claim embeddings
        llm_client: LLM for reasoning
    """
    
    def __init__(self):
        """Initialize the RAG reasoner."""
        self.config = get_config()
        
        # Lazy loading
        self._embedder = None
        self._vector_store: List[Tuple[str, np.ndarray, Dict]] = []
        self._llm_client = None
        
        logger.info("RAGContradictionReasoner initialized")
    
    @property
    def embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            
            model_name = self.config.pipeline.claim_matching.embedding_model
            self._embedder = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        
        return self._embedder
    
    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            try:
                import anthropic
                
                self._llm_client = anthropic.Anthropic(
                    api_key=self.config.api.anthropic_key
                )
                logger.info("Anthropic client initialized for RAG")
            except Exception as e:
                logger.warning(f"Could not initialize LLM client: {e}")
        
        return self._llm_client
    
    def add_claims(
        self,
        claims: List[Dict]
    ) -> int:
        """
        Add claims to the vector store.
        
        Args:
            claims: List of claim dictionaries with 'text' field
            
        Returns:
            Number of claims added
        """
        texts = [c.get("text", c.get("claim_text", "")) for c in claims]
        
        if not texts:
            return 0
        
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        
        for text, embedding, claim in zip(texts, embeddings, claims):
            self._vector_store.append((text, embedding, claim))
        
        logger.info(f"Added {len(claims)} claims to vector store")
        return len(claims)
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """
        Retrieve relevant historical claims for a query.
        
        Args:
            query: Query text to search for
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant claims with similarity scores
        """
        if not self._vector_store:
            return []
        
        query_embedding = self.embedder.encode(query, convert_to_numpy=True)
        
        # Compute similarities
        results = []
        for text, embedding, claim in self._vector_store:
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            
            if similarity >= min_similarity:
                results.append({
                    "claim": claim,
                    "text": text,
                    "similarity": float(similarity)
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:top_k]
    
    def reason_about_contradiction(
        self,
        earlier_claim: Dict,
        later_claim: Dict,
        context: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Use LLM to reason about a potential contradiction.
        
        Args:
            earlier_claim: The earlier claim
            later_claim: The later claim
            context: Optional retrieved context
            
        Returns:
            Dictionary with reasoning and conclusion
        """
        if self.llm_client is None:
            return {
                "is_contradiction": None,
                "reasoning": "LLM not available",
                "confidence": 0.0
            }
        
        earlier_text = earlier_claim.get("text", earlier_claim.get("claim_text", ""))
        later_text = later_claim.get("text", later_claim.get("claim_text", ""))
        
        # Build context string
        context_str = ""
        if context:
            context_str = "\n\nRelevant historical context:\n"
            for i, ctx in enumerate(context[:3], 1):
                context_str += f"{i}. {ctx['text']}\n"
        
        prompt = f"""Analyze whether these two corporate statements contradict each other.

EARLIER STATEMENT (from {earlier_claim.get('date', 'unknown date')}):
"{earlier_text}"

LATER STATEMENT (from {later_claim.get('date', 'unknown date')}):
"{later_text}"
{context_str}
Please analyze:
1. Do these statements contradict each other?
2. What is the nature of the contradiction (if any)?
3. What are the potential market implications?

Provide your response in the following format:
- Is Contradiction: [YES/NO/UNCLEAR]
- Confidence: [HIGH/MEDIUM/LOW]
- Reasoning: [Your analysis]
- Market Impact: [Brief assessment]"""

        try:
            response = self.llm_client.messages.create(
                model=self.config.pipeline.claim_extraction.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            
            # Parse response
            is_contradiction = "YES" in response_text.upper()[:100]
            confidence = 0.8 if "HIGH" in response_text.upper() else (
                0.5 if "MEDIUM" in response_text.upper() else 0.3
            )
            
            return {
                "is_contradiction": is_contradiction,
                "reasoning": response_text,
                "confidence": confidence,
                "raw_response": response_text
            }
            
        except Exception as e:
            logger.error(f"Error in LLM reasoning: {e}")
            return {
                "is_contradiction": None,
                "reasoning": f"Error: {e}",
                "confidence": 0.0
            }
    
    def explain_contradiction(
        self,
        contradiction_event: Dict
    ) -> str:
        """
        Generate a human-readable explanation for a contradiction.
        
        Args:
            contradiction_event: The contradiction event dictionary
            
        Returns:
            Natural language explanation
        """
        earlier = contradiction_event.get("earlier_claim", {})
        later = contradiction_event.get("later_claim", {})
        score = contradiction_event.get("contradiction_score", 0)
        topic = contradiction_event.get("topic", "unknown")
        
        # Retrieve related context
        query = f"{earlier.get('text', '')} {later.get('text', '')}"
        context = self.retrieve_context(query, top_k=3)
        
        if self.llm_client is None:
            return f"Contradiction detected (score: {score:.2f}) in {topic} topic."
        
        prompt = f"""Generate a brief, professional explanation of this corporate statement contradiction for an analyst.

Topic: {topic}
Contradiction Score: {score:.2f}

Earlier Statement: "{earlier.get('text', 'N/A')}"
Later Statement: "{later.get('text', 'N/A')}"

Write 2-3 sentences explaining what changed and why it matters."""

        try:
            response = self.llm_client.messages.create(
                model=self.config.pipeline.claim_extraction.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.warning(f"Error generating explanation: {e}")
            return f"Contradiction detected (score: {score:.2f}) in {topic} topic."
