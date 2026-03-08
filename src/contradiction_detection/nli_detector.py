"""
NLI-based Contradiction Detector

This module provides Natural Language Inference for detecting contradictions
between corporate claims. It uses transformer models trained on NLI tasks
to classify the relationship between statement pairs.

The detector:
- Uses DeBERTa-v3-large-mnli (or configurable alternative)
- Supports batch processing for efficiency
- Returns probability distributions over entailment/neutral/contradiction

Usage:
    from src.contradiction_detection.nli_detector import NLIDetector
    
    detector = NLIDetector()
    result = detector.classify(
        premise="Revenue will grow 15%",
        hypothesis="Revenue expected to decline 5%"
    )
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from src.config import get_config


@dataclass
class NLIResult:
    """
    Result of NLI classification.
    
    Attributes:
        premise: The premise (earlier claim)
        hypothesis: The hypothesis (later claim)
        entailment: Probability of entailment
        neutral: Probability of neutral
        contradiction: Probability of contradiction
        predicted_label: Predicted relationship label
    """
    premise: str
    hypothesis: str
    entailment: float
    neutral: float
    contradiction: float
    predicted_label: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "premise": self.premise,
            "hypothesis": self.hypothesis,
            "entailment": self.entailment,
            "neutral": self.neutral,
            "contradiction": self.contradiction,
            "predicted_label": self.predicted_label,
        }


class NLIDetector:
    """
    Detects contradictions using Natural Language Inference.
    
    Uses transformer models to classify the relationship between
    two statements as entailment, neutral, or contradiction.
    
    Attributes:
        model_name: Name of the NLI model to use
        device: Compute device (cuda/mps/cpu)
        batch_size: Batch size for inference
    """
    
    # Label mappings (model-specific)
    LABEL_MAP = {
        "ENTAILMENT": "entailment",
        "NEUTRAL": "neutral", 
        "CONTRADICTION": "contradiction",
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction",
    }
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the NLI detector.
        
        Args:
            model_name: NLI model to use. If None, uses config default.
        """
        config = get_config()
        
        self.model_name = model_name or config.pipeline.contradiction.nli_model
        self.batch_size = config.pipeline.contradiction.batch_size
        self._device = config.get_device()
        
        # Lazy loading
        self._model = None
        self._tokenizer = None
        
        logger.info(f"NLIDetector initialized with model: {self.model_name}")
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    def _load_model(self):
        """Load the NLI model and tokenizer."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        logger.info(f"Loading NLI model: {self.model_name}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Move to device
        self._model.to(self._device)
        self._model.eval()
        
        logger.info(f"Model loaded on device: {self._device}")
    
    def classify(
        self,
        premise: str,
        hypothesis: str
    ) -> NLIResult:
        """
        Classify the relationship between two statements.
        
        Args:
            premise: The earlier statement (what we're testing against)
            hypothesis: The later statement (what we're testing)
            
        Returns:
            NLIResult with probabilities and predicted label
        """
        if not premise or not hypothesis:
            return NLIResult(
                premise=premise,
                hypothesis=hypothesis,
                entailment=0.0,
                neutral=1.0,
                contradiction=0.0,
                predicted_label="neutral"
            )
        
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Get label mapping from model config
        id2label = self.model.config.id2label
        
        # Extract probabilities
        result_probs = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
        
        for idx, prob in enumerate(probs):
            label = id2label[idx]
            canonical_label = self.LABEL_MAP.get(label, label.lower())
            if canonical_label in result_probs:
                result_probs[canonical_label] = float(prob)
        
        # Determine predicted label
        predicted_label = max(result_probs.keys(), key=lambda k: result_probs[k])
        
        return NLIResult(
            premise=premise,
            hypothesis=hypothesis,
            entailment=result_probs["entailment"],
            neutral=result_probs["neutral"],
            contradiction=result_probs["contradiction"],
            predicted_label=predicted_label
        )
    
    def classify_batch(
        self,
        pairs: List[Tuple[str, str]]
    ) -> List[NLIResult]:
        """
        Classify multiple statement pairs.
        
        Args:
            pairs: List of (premise, hypothesis) tuples
            
        Returns:
            List of NLIResult objects
        """
        if not pairs:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            
            premises = [p[0] for p in batch]
            hypotheses = [p[1] for p in batch]
            
            # Tokenize batch
            inputs = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Get label mapping
            id2label = self.model.config.id2label
            
            # Process results
            for j, (premise, hypothesis) in enumerate(batch):
                result_probs = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
                
                for idx, prob in enumerate(probs[j]):
                    label = id2label[idx]
                    canonical_label = self.LABEL_MAP.get(label, label.lower())
                    if canonical_label in result_probs:
                        result_probs[canonical_label] = float(prob)
                
                predicted_label = max(result_probs.keys(), key=lambda k: result_probs[k])
                
                results.append(NLIResult(
                    premise=premise,
                    hypothesis=hypothesis,
                    entailment=result_probs["entailment"],
                    neutral=result_probs["neutral"],
                    contradiction=result_probs["contradiction"],
                    predicted_label=predicted_label
                ))
        
        return results
    
    def get_contradiction_score(
        self,
        premise: str,
        hypothesis: str
    ) -> float:
        """
        Get just the contradiction probability.
        
        Convenience method for when only the contradiction score is needed.
        
        Args:
            premise: Earlier statement
            hypothesis: Later statement
            
        Returns:
            Contradiction probability (0-1)
        """
        result = self.classify(premise, hypothesis)
        return result.contradiction
    
    def is_contradiction(
        self,
        premise: str,
        hypothesis: str,
        threshold: float = 0.5
    ) -> bool:
        """
        Check if two statements contradict.
        
        Args:
            premise: Earlier statement
            hypothesis: Later statement
            threshold: Probability threshold for contradiction
            
        Returns:
            True if statements contradict above threshold
        """
        result = self.classify(premise, hypothesis)
        return result.contradiction > threshold
