"""
Tests for claim extraction module.
"""

import pytest


class TestEntityExtractor:
    """Tests for entity extraction."""
    
    def test_extract_money(self):
        """Test extraction of monetary values."""
        from src.claim_extraction import EntityExtractor
        
        extractor = EntityExtractor()
        text = "Revenue increased to $1.5 billion from $1.2 billion last year."
        
        entities = extractor.extract(text)
        
        assert "MONEY" in entities
        assert len(entities["MONEY"]) >= 2
    
    def test_extract_percentage(self):
        """Test extraction of percentages."""
        from src.claim_extraction import EntityExtractor
        
        extractor = EntityExtractor()
        text = "We expect 15% growth in Q4 compared to 12% in Q3."
        
        entities = extractor.extract(text)
        
        assert "PERCENTAGE" in entities
        assert len(entities["PERCENTAGE"]) >= 2
    
    def test_extract_date(self):
        """Test extraction of dates."""
        from src.claim_extraction import EntityExtractor
        
        extractor = EntityExtractor()
        text = "The project will launch in Q4 2025 and complete by December 2026."
        
        entities = extractor.extract(text)
        
        assert "DATE" in entities
        assert len(entities["DATE"]) >= 1


class TestClaimNormalizer:
    """Tests for claim normalization."""
    
    def test_normalize_text(self):
        """Test text normalization."""
        from src.claim_extraction import ClaimNormalizer, ExtractedClaim
        
        normalizer = ClaimNormalizer()
        
        claim = ExtractedClaim(
            claim_text="  Revenue   will  INCREASE  by  15%  ",
            topic="guidance",
            speaker=None,
            speaker_role=None,
            direction="positive",
            confidence=0.9,
            source_sentence="test"
        )
        
        normalized = normalizer.normalize(claim)
        
        # Should normalize whitespace
        assert "  " not in normalized.normalized_text
    
    def test_canonicalize_topic(self):
        """Test topic canonicalization."""
        from src.claim_extraction import ClaimNormalizer, ExtractedClaim
        
        normalizer = ClaimNormalizer()
        
        claim = ExtractedClaim(
            claim_text="Revenue expected to grow",
            topic="revenue",
            speaker=None,
            speaker_role=None,
            direction="positive",
            confidence=0.9,
            source_sentence="test"
        )
        
        normalized = normalizer.normalize(claim)
        
        # Revenue should map to guidance
        assert normalized.canonical_topic == "guidance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
