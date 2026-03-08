"""
Bag-of-Words Classifier Baseline

This module implements a traditional ML baseline using TF-IDF
features and simple classifiers (Logistic Regression, SVM).

This baseline tests whether neural NLI models outperform
classical approaches on this task.

Usage:
    from src.baselines.bow_classifier import BoWClassifier
    
    model = BoWClassifier()
    model.fit(train_texts, train_labels)
    predictions = model.predict(test_texts)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.config import get_config


@dataclass
class BoWPrediction:
    """
    Prediction from BoW classifier.
    
    Attributes:
        signal: Predicted signal (bearish, bullish, neutral)
        confidence: Prediction confidence
        probabilities: Class probabilities
    """
    signal: str
    confidence: float
    probabilities: Dict[str, float]


class BoWClassifier:
    """
    Bag-of-Words classifier baseline.
    
    Uses TF-IDF features with Logistic Regression or SVM
    for market reaction prediction.
    
    Attributes:
        classifier_type: Type of classifier ('logistic_regression' or 'svm')
        vectorizer: TF-IDF vectorizer
        classifier: Trained classifier
    """
    
    def __init__(self, classifier_type: str = "logistic_regression"):
        """
        Initialize the BoW classifier.
        
        Args:
            classifier_type: 'logistic_regression' or 'svm'
        """
        self.classifier_type = classifier_type
        
        config = get_config()
        bow_config = config.model.baselines.get("bow", {})
        vec_config = bow_config.get("vectorizer", {})
        
        # Initialize vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.vectorizer = TfidfVectorizer(
            max_features=vec_config.get("max_features", 5000),
            ngram_range=tuple(vec_config.get("ngram_range", [1, 2])),
            min_df=vec_config.get("min_df", 2),
            max_df=vec_config.get("max_df", 0.95),
            stop_words="english"
        )
        
        # Initialize classifier
        self.classifier = self._create_classifier(classifier_type, bow_config)
        
        self._is_fitted = False
        self._label_mapping = {"positive": 0, "neutral": 1, "negative": 2}
        self._reverse_mapping = {0: "positive", 1: "neutral", 2: "negative"}
        
        logger.info(f"BoWClassifier initialized with {classifier_type}")
    
    def _create_classifier(self, classifier_type: str, config: Dict):
        """Create the classifier based on type."""
        classifiers_config = config.get("classifiers", [])
        
        # Find config for this classifier type
        clf_config = {}
        for clf in classifiers_config:
            if clf.get("name") == classifier_type:
                clf_config = clf.get("params", {})
                break
        
        if classifier_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=clf_config.get("C", 1.0),
                max_iter=clf_config.get("max_iter", 1000),
                class_weight=clf_config.get("class_weight", "balanced"),
                random_state=42
            )
        elif classifier_type == "svm" or classifier_type == "linear_svm":
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            
            svc = LinearSVC(
                C=clf_config.get("C", 1.0),
                class_weight=clf_config.get("class_weight", "balanced"),
                random_state=42,
                max_iter=2000
            )
            # Wrap in CalibratedClassifierCV for probability estimates
            return CalibratedClassifierCV(svc, cv=3)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def _prepare_text(
        self,
        earlier_text: str,
        later_text: str
    ) -> str:
        """
        Prepare text for classification.
        
        Combines earlier and later claim texts.
        
        Args:
            earlier_text: Earlier claim text
            later_text: Later claim text
            
        Returns:
            Combined text
        """
        return f"{earlier_text} [SEP] {later_text}"
    
    def fit(
        self,
        events: List[Dict],
        labels: List[str]
    ):
        """
        Fit the classifier on training data.
        
        Args:
            events: List of event dictionaries
            labels: List of label strings (positive, neutral, negative)
        """
        # Prepare texts
        texts = [
            self._prepare_text(
                e.get("earlier_claim", {}).get("text", ""),
                e.get("later_claim", {}).get("text", "")
            )
            for e in events
        ]
        
        # Convert labels to integers
        y = np.array([self._label_mapping.get(l, 1) for l in labels])
        
        # Fit vectorizer and transform
        X = self.vectorizer.fit_transform(texts)
        
        # Fit classifier
        self.classifier.fit(X, y)
        self._is_fitted = True
        
        logger.info(f"BoWClassifier fitted on {len(texts)} samples")
    
    def predict_single(
        self,
        earlier_text: str,
        later_text: str
    ) -> BoWPrediction:
        """
        Predict signal for a single event.
        
        Args:
            earlier_text: Earlier claim text
            later_text: Later claim text
            
        Returns:
            BoWPrediction with signal and probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        text = self._prepare_text(earlier_text, later_text)
        X = self.vectorizer.transform([text])
        
        # Get prediction and probabilities
        pred = self.classifier.predict(X)[0]
        
        try:
            probs = self.classifier.predict_proba(X)[0]
            prob_dict = {
                self._reverse_mapping[i]: float(p)
                for i, p in enumerate(probs)
            }
        except AttributeError:
            prob_dict = {self._reverse_mapping[pred]: 1.0}
        
        signal = self._reverse_mapping[pred]
        confidence = max(prob_dict.values())
        
        # Map to trading signals
        signal_map = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
        
        return BoWPrediction(
            signal=signal_map.get(signal, "neutral"),
            confidence=confidence,
            probabilities=prob_dict
        )
    
    def predict(
        self,
        events: List[Dict]
    ) -> List[BoWPrediction]:
        """
        Predict signals for multiple events.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            List of BoWPrediction objects
        """
        predictions = []
        
        for event in events:
            earlier_text = event.get("earlier_claim", {}).get("text", "")
            later_text = event.get("later_claim", {}).get("text", "")
            pred = self.predict_single(earlier_text, later_text)
            predictions.append(pred)
        
        return predictions
    
    def predict_proba(
        self,
        events: List[Dict]
    ) -> np.ndarray:
        """
        Get probability predictions for events.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Array of probabilities (n_samples x n_classes)
        """
        texts = [
            self._prepare_text(
                e.get("earlier_claim", {}).get("text", ""),
                e.get("later_claim", {}).get("text", "")
            )
            for e in events
        ]
        
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get most important features for each class.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary mapping class names to (feature, importance) lists
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted.")
        
        if self.classifier_type != "logistic_regression":
            logger.warning("Feature importance only available for logistic regression")
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        importance = {}
        for class_idx, class_name in self._reverse_mapping.items():
            coefs = self.classifier.coef_[class_idx] if len(self.classifier.coef_.shape) > 1 else self.classifier.coef_[0]
            
            # Get top features
            top_indices = np.argsort(coefs)[::-1][:top_n]
            importance[class_name] = [
                (feature_names[i], float(coefs[i]))
                for i in top_indices
            ]
        
        return importance
