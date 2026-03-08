"""
LLM Hybrid Detector

This module combines rule-based detection with LLM reasoning
for more accurate contradiction identification.

Features:
- Multi-stage detection pipeline
- Rule-based pre-filtering
- LLM verification for high-confidence signals
- Ensemble scoring
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.config import get_config


class LLMHybridDetector:
    """
    Hybrid contradiction detector combining rules and LLM.
    
    Uses a multi-stage approach:
    1. Rule-based pre-filtering (fast)
    2. NLI classification (medium)
    3. LLM verification (slow, high-confidence)
    
    Attributes:
        nli_detector: Neural NLI model
        keyword_rules: Rule-based detector
        llm_client: LLM for verification
    """
    
    # Contradiction patterns for rule-based detection
    REVERSAL_PATTERNS = [
        (r"\bwill\b", r"\bwill not\b"),
        (r"\bexpect\b", r"\bno longer expect\b"),
        (r"\bon track\b", r"\bdelayed\b"),
        (r"\bincrease\b", r"\bdecrease\b"),
        (r"\bgrowth\b", r"\bdecline\b"),
        (r"\bconfirm\b", r"\brevise\b"),
        (r"\bmeet\b", r"\bmiss\b"),
    ]
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize the hybrid detector.
        
        Args:
            use_llm: Whether to use LLM verification
        """
        self.config = get_config()
        self.use_llm = use_llm
        
        # Lazy loading
        self._nli_detector = None
        self._keyword_rules = None
        self._llm_client = None
        
        logger.info(f"LLMHybridDetector initialized (use_llm={use_llm})")
    
    @property
    def nli_detector(self):
        """Lazy load NLI detector."""
        if self._nli_detector is None:
            from src.contradiction_detection import NLIDetector
            self._nli_detector = NLIDetector()
        return self._nli_detector
    
    @property
    def keyword_rules(self):
        """Lazy load keyword rules."""
        if self._keyword_rules is None:
            from src.baselines import KeywordRules
            self._keyword_rules = KeywordRules()
        return self._keyword_rules
    
    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None and self.use_llm:
            try:
                import anthropic
                self._llm_client = anthropic.Anthropic(
                    api_key=self.config.api.anthropic_key
                )
            except Exception as e:
                logger.warning(f"Could not initialize LLM: {e}")
        return self._llm_client
    
    def detect(
        self,
        earlier_claim: str,
        later_claim: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Detect contradiction using hybrid approach.
        
        Args:
            earlier_claim: The earlier statement
            later_claim: The later statement
            metadata: Optional metadata (topic, speaker, etc.)
            
        Returns:
            Detection result dictionary
        """
        metadata = metadata or {}
        
        # Stage 1: Rule-based pre-filter
        rule_score = self._apply_rules(earlier_claim, later_claim)
        
        # Stage 2: NLI classification
        nli_result = self.nli_detector.classify(earlier_claim, later_claim)
        nli_score = nli_result.contradiction_prob
        
        # Stage 3: LLM verification (if warranted)
        llm_score = None
        llm_reasoning = None
        
        if self.use_llm and self.llm_client:
            # Only use LLM for ambiguous cases
            if 0.4 < nli_score < 0.8 or (rule_score > 0.5 and nli_score < 0.5):
                llm_result = self._llm_verify(earlier_claim, later_claim)
                llm_score = llm_result.get("score")
                llm_reasoning = llm_result.get("reasoning")
        
        # Combine scores
        final_score = self._combine_scores(rule_score, nli_score, llm_score)
        
        return {
            "is_contradiction": final_score > 0.6,
            "final_score": final_score,
            "rule_score": rule_score,
            "nli_score": nli_score,
            "nli_label": nli_result.label,
            "llm_score": llm_score,
            "llm_reasoning": llm_reasoning,
            "confidence": self._compute_confidence(rule_score, nli_score, llm_score)
        }
    
    def _apply_rules(
        self,
        earlier_claim: str,
        later_claim: str
    ) -> float:
        """
        Apply rule-based contradiction detection.
        
        Args:
            earlier_claim: Earlier statement
            later_claim: Later statement
            
        Returns:
            Rule-based score 0-1
        """
        import re
        
        earlier_lower = earlier_claim.lower()
        later_lower = later_claim.lower()
        
        matches = 0
        
        for pattern1, pattern2 in self.REVERSAL_PATTERNS:
            if re.search(pattern1, earlier_lower) and re.search(pattern2, later_lower):
                matches += 1
            if re.search(pattern2, earlier_lower) and re.search(pattern1, later_lower):
                matches += 1
        
        # Normalize to 0-1
        return min(matches / 3, 1.0)
    
    def _llm_verify(
        self,
        earlier_claim: str,
        later_claim: str
    ) -> Dict:
        """
        Verify contradiction using LLM.
        
        Args:
            earlier_claim: Earlier statement
            later_claim: Later statement
            
        Returns:
            Dictionary with score and reasoning
        """
        prompt = f"""Analyze whether these two corporate statements contradict each other.

EARLIER: "{earlier_claim}"
LATER: "{later_claim}"

Rate the contradiction on a scale of 0-10:
- 0: No contradiction, consistent
- 5: Partial contradiction or clarification
- 10: Direct contradiction

Respond with just the number and one sentence explanation.
Format: SCORE: [number] - [explanation]"""

        try:
            response = self.llm_client.messages.create(
                model=self.config.pipeline.claim_extraction.model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text
            
            # Parse score
            import re
            match = re.search(r"SCORE:\s*(\d+)", text, re.IGNORECASE)
            score = int(match.group(1)) / 10 if match else 0.5
            
            return {
                "score": score,
                "reasoning": text
            }
            
        except Exception as e:
            logger.warning(f"LLM verification error: {e}")
            return {"score": None, "reasoning": None}
    
    def _combine_scores(
        self,
        rule_score: float,
        nli_score: float,
        llm_score: Optional[float]
    ) -> float:
        """
        Combine scores from different methods.
        
        Args:
            rule_score: Rule-based score
            nli_score: NLI model score
            llm_score: LLM verification score (optional)
            
        Returns:
            Combined final score
        """
        if llm_score is not None:
            # Weighted average with LLM
            weights = [0.15, 0.45, 0.40]  # rules, nli, llm
            scores = [rule_score, nli_score, llm_score]
            return sum(w * s for w, s in zip(weights, scores))
        else:
            # Just rules and NLI
            weights = [0.25, 0.75]  # rules, nli
            scores = [rule_score, nli_score]
            return sum(w * s for w, s in zip(weights, scores))
    
    def _compute_confidence(
        self,
        rule_score: float,
        nli_score: float,
        llm_score: Optional[float]
    ) -> float:
        """
        Compute confidence in the detection.
        
        Higher confidence when methods agree.
        
        Args:
            rule_score: Rule-based score
            nli_score: NLI model score
            llm_score: LLM score (optional)
            
        Returns:
            Confidence score 0-1
        """
        scores = [rule_score, nli_score]
        if llm_score is not None:
            scores.append(llm_score)
        
        # Confidence based on agreement
        std = np.std(scores)
        mean = np.mean(scores)
        
        # High mean + low std = high confidence
        agreement_confidence = 1.0 - std
        
        # Extreme scores have higher confidence
        extremity_confidence = abs(mean - 0.5) * 2
        
        return (agreement_confidence + extremity_confidence) / 2
    
    def detect_batch(
        self,
        claim_pairs: List[Tuple[str, str]]
    ) -> List[Dict]:
        """
        Detect contradictions for multiple pairs.
        
        Args:
            claim_pairs: List of (earlier, later) claim pairs
            
        Returns:
            List of detection results
        """
        results = []
        
        for earlier, later in claim_pairs:
            result = self.detect(earlier, later)
            results.append(result)
        
        return results
