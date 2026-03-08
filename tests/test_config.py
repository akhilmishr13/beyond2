"""
Tests for configuration module.
"""

import os
import pytest
from pathlib import Path


class TestConfig:
    """Tests for configuration loading."""
    
    def test_config_singleton(self):
        """Test that get_config returns singleton."""
        from src.config import get_config
        
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_config_has_required_sections(self):
        """Test that config has all required sections."""
        from src.config import get_config
        
        config = get_config()
        
        assert hasattr(config, 'database')
        assert hasattr(config, 'api')
        assert hasattr(config, 'app')
        assert hasattr(config, 'pipeline')
        assert hasattr(config, 'model')
        assert hasattr(config, 'evaluation')
    
    def test_pipeline_config_has_lookback(self):
        """Test that pipeline config has lookback settings."""
        from src.config import get_config
        
        config = get_config()
        
        assert hasattr(config.pipeline, 'lookback')
        assert hasattr(config.pipeline.lookback, 'start_date')
        assert hasattr(config.pipeline.lookback, 'end_date')
        assert hasattr(config.pipeline.lookback, 'months')
    
    def test_pipeline_config_has_companies(self):
        """Test that pipeline config has company list."""
        from src.config import get_config
        
        config = get_config()
        
        assert hasattr(config.pipeline, 'companies')
        assert isinstance(config.pipeline.companies, list)
        assert len(config.pipeline.companies) > 0


class TestDatabaseSettings:
    """Tests for database settings."""
    
    def test_database_defaults(self):
        """Test database has default values."""
        from src.config import get_config
        
        config = get_config()
        
        assert config.database.host is not None
        assert config.database.port > 0


class TestAPISettings:
    """Tests for API settings."""
    
    def test_sec_user_agent(self):
        """Test SEC user agent is configured."""
        from src.config import get_config
        
        config = get_config()
        
        assert hasattr(config.api, 'sec_user_agent')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
