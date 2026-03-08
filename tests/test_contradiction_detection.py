"""
Tests for contradiction detection module.
"""

import pytest


class TestContradictionScorer:
    """Tests for contradiction scoring."""
    
    def test_score_thresholds(self):
        """Test that scorer has correct thresholds."""
        from src.contradiction_detection import ContradictionScorer
        
        scorer = ContradictionScorer()
        
        assert "critical" in scorer.DEFAULT_THRESHOLDS
        assert "high" in scorer.DEFAULT_THRESHOLDS
        assert "medium" in scorer.DEFAULT_THRESHOLDS
        assert "low" in scorer.DEFAULT_THRESHOLDS
    
    def test_topic_weights(self):
        """Test that important topics have higher weights."""
        from src.contradiction_detection import ContradictionScorer
        
        scorer = ContradictionScorer()
        
        # Legal/regulatory should have high weight
        assert scorer.TOPIC_WEIGHTS.get("legal_regulatory", 1.0) > 1.0
        
        # Guidance should have elevated weight
        assert scorer.TOPIC_WEIGHTS.get("guidance", 1.0) >= 1.0


class TestRelationshipClassifier:
    """Tests for relationship classification."""
    
    def test_relation_types(self):
        """Test that all relation types are defined."""
        from src.contradiction_detection.relationship_classifier import RelationType
        
        assert hasattr(RelationType, "CONFIRMATION")
        assert hasattr(RelationType, "CONTRADICTION")
        assert hasattr(RelationType, "UPDATE")
        assert hasattr(RelationType, "REVISION")
    
    def test_direction_changes(self):
        """Test direction change mappings."""
        from src.contradiction_detection import RelationshipClassifier
        
        classifier = RelationshipClassifier()
        
        # Positive to negative is a reversal
        change = classifier.DIRECTION_CHANGES.get(("positive", "negative"))
        assert change == "reversal"
        
        # Same direction is consistent
        change = classifier.DIRECTION_CHANGES.get(("positive", "positive"))
        assert change == "consistent"


class TestTemporalContradictionGuard:
    """Tests for temporal ordering guard."""
    
    def test_guard_validates_order(self):
        """Test that guard validates temporal order."""
        from datetime import datetime
        from src.contradiction_detection.contradiction_scorer import TemporalContradictionGuard
        
        guard = TemporalContradictionGuard()
        
        earlier = datetime(2025, 1, 1)
        later = datetime(2025, 6, 1)
        
        # Valid order
        assert guard.validate_temporal_order(earlier, later) is True
        
        # Invalid order
        assert guard.validate_temporal_order(later, earlier) is False
    
    def test_guard_raises_on_violation(self):
        """Test that guard can raise on violations."""
        from datetime import datetime
        from src.contradiction_detection.contradiction_scorer import TemporalContradictionGuard
        
        guard = TemporalContradictionGuard()
        
        earlier = datetime(2025, 6, 1)
        later = datetime(2025, 1, 1)
        
        with pytest.raises(ValueError):
            guard.validate_temporal_order(earlier, later, raise_on_violation=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
