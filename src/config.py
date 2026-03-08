"""
Configuration Management Module

This module handles loading and validating configuration from YAML files
and environment variables. It provides a centralized configuration object
that can be accessed throughout the application.

Usage:
    from src.config import get_config
    config = get_config()
    companies = config.pipeline.companies
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


# Load environment variables from .env file
load_dotenv()


# =============================================================================
# Environment Configuration
# =============================================================================

class DatabaseSettings(BaseSettings):
    """Database connection settings loaded from environment variables."""
    
    host: str = Field(default="localhost", alias="POSTGRES_HOST")
    port: int = Field(default=5432, alias="POSTGRES_PORT")
    database: str = Field(default="corporate_narrative_engine", alias="POSTGRES_DB")
    user: str = Field(default="", alias="POSTGRES_USER")
    password: str = Field(default="", alias="POSTGRES_PASSWORD")
    
    @property
    def url(self) -> str:
        """Construct PostgreSQL connection URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_url(self) -> str:
        """Construct async PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class APISettings(BaseSettings):
    """API keys and credentials loaded from environment variables."""
    
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    newsapi_key: str = Field(default="", alias="NEWSAPI_KEY")
    sec_user_agent: str = Field(default="", alias="SEC_USER_AGENT")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    huggingface_token: Optional[str] = Field(default=None, alias="HUGGINGFACE_TOKEN")
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    num_workers: int = Field(default=4, alias="NUM_WORKERS")
    cache_dir: str = Field(default=".cache", alias="CACHE_DIR")
    torch_device: Optional[str] = Field(default=None, alias="TORCH_DEVICE")
    
    # Gradio settings
    gradio_server_name: str = Field(default="127.0.0.1", alias="GRADIO_SERVER_NAME")
    gradio_server_port: int = Field(default=7860, alias="GRADIO_SERVER_PORT")
    gradio_share: bool = Field(default=False, alias="GRADIO_SHARE")
    
    class Config:
        env_file = ".env"
        extra = "ignore"


# =============================================================================
# YAML Configuration Models
# =============================================================================

class LookbackConfig(BaseModel):
    """Configuration for data lookback period."""
    
    start_date: str
    end_date: str
    months: int = 12


class SECConfig(BaseModel):
    """SEC filing ingestion configuration."""
    
    filing_types: List[str] = ["8-K", "10-K", "10-Q"]
    rate_limit_per_second: int = 10
    max_retries: int = 3
    retry_delay_seconds: int = 5
    sections_10k: List[str] = []
    sections_10q: List[str] = []
    sections_8k: List[str] = []


class NewsSourceConfig(BaseModel):
    """Individual news source configuration."""
    
    enabled: bool = True
    max_articles_per_company: int = 500
    languages: List[str] = ["English"]
    lookback_days: int = 30
    feeds: List[Dict[str, str]] = []


class NewsConfig(BaseModel):
    """News ingestion configuration."""
    
    sources: Dict[str, NewsSourceConfig] = {}
    extraction: Dict[str, Any] = {}


class MarketConfig(BaseModel):
    """Market data configuration."""
    
    source: str = "yfinance"
    return_windows: List[int] = [1, 3, 5, 10]
    benchmark_ticker: str = "SPY"
    reaction_thresholds: Dict[str, float] = {"positive": 0.015, "negative": -0.015}


class ClaimExtractionConfig(BaseModel):
    """Claim extraction configuration."""
    
    provider: str = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4096
    temperature: float = 0.0
    batch_size: int = 10
    max_concurrent_requests: int = 5
    topic_categories: List[str] = []


class ClaimMatchingConfig(BaseModel):
    """Claim matching configuration."""
    
    embedding_model: str = "all-mpnet-base-v2"
    similarity_threshold: float = 0.75
    lookback_days: int = 180
    min_similarity: float = 0.5
    max_matches: int = 10


class ContradictionConfig(BaseModel):
    """Contradiction detection configuration."""
    
    nli_model: str = "microsoft/deberta-v3-large-mnli"
    batch_size: int = 32
    thresholds: Dict[str, float] = {"high": 0.8, "medium": 0.6, "low": 0.3}
    device: Optional[str] = None


class ProcessingConfig(BaseModel):
    """Processing configuration."""
    
    num_workers: int = 4
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "logs/pipeline.log"
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 100
    continue_on_error: bool = True
    max_errors_before_abort: int = 50


class PipelineConfig(BaseModel):
    """Complete pipeline configuration loaded from YAML."""
    
    companies: List[str] = []
    lookback: LookbackConfig = LookbackConfig(start_date="2025-03-08", end_date="2026-03-08")
    sec: SECConfig = SECConfig()
    news: NewsConfig = NewsConfig()
    market: MarketConfig = MarketConfig()
    claim_extraction: ClaimExtractionConfig = ClaimExtractionConfig()
    claim_matching: ClaimMatchingConfig = ClaimMatchingConfig()
    contradiction: ContradictionConfig = ContradictionConfig()
    processing: ProcessingConfig = ProcessingConfig()


class TemporalSplitConfig(BaseModel):
    """Temporal split configuration for evaluation."""
    
    train_ratio: float = 0.67
    val_ratio: float = 0.16
    test_ratio: float = 0.17
    gap_days: int = 5
    enforce_chronological: bool = True
    validate_no_future_leakage: bool = True
    
    @field_validator('train_ratio', 'val_ratio', 'test_ratio')
    @classmethod
    def validate_ratios(cls, v: float) -> float:
        """Ensure ratios are between 0 and 1."""
        if not 0 < v < 1:
            raise ValueError("Ratio must be between 0 and 1")
        return v


class EvaluationConfig(BaseModel):
    """Evaluation configuration loaded from YAML."""
    
    temporal_split: TemporalSplitConfig = TemporalSplitConfig()
    classification_metrics: Dict[str, Any] = {}
    decision_metrics: Dict[str, Any] = {}
    financial_metrics: Dict[str, Any] = {}
    cost_utility: Dict[str, Any] = {}
    signal_decay: Dict[str, Any] = {}
    robustness: Dict[str, Any] = {}
    calibration: Dict[str, Any] = {}
    statistical_tests: Dict[str, Any] = {}
    error_analysis: Dict[str, Any] = {}
    reports: Dict[str, Any] = {}


class ModelConfig(BaseModel):
    """Model configuration loaded from YAML."""
    
    features: Dict[str, List[str]] = {}
    baselines: Dict[str, Any] = {}
    signal_model: Dict[str, Any] = {}
    training: Dict[str, Any] = {}
    explainability: Dict[str, Any] = {}
    persistence: Dict[str, Any] = {}


# =============================================================================
# Configuration Manager
# =============================================================================

class Config:
    """
    Central configuration manager that loads and provides access to all
    configuration settings from environment variables and YAML files.
    
    Attributes:
        database: Database connection settings
        api: API keys and credentials
        app: Application settings
        pipeline: Pipeline configuration from YAML
        model: Model configuration from YAML
        evaluation: Evaluation configuration from YAML
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Path to configuration directory. If None, uses
                       'configs' relative to project root.
        """
        # Determine project root and config directory
        self.project_root = Path(__file__).parent.parent
        self.config_dir = config_dir or self.project_root / "configs"
        
        # Load environment-based settings
        self.database = DatabaseSettings()
        self.api = APISettings()
        self.app = AppSettings()
        
        # Load YAML configurations
        self.pipeline = self._load_pipeline_config()
        self.model = self._load_model_config()
        self.evaluation = self._load_evaluation_config()
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            filename: Name of the YAML file to load
            
        Returns:
            Dictionary containing the parsed YAML content
        """
        filepath = self.config_dir / filename
        if not filepath.exists():
            return {}
        
        with open(filepath, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _load_pipeline_config(self) -> PipelineConfig:
        """Load pipeline configuration from YAML."""
        data = self._load_yaml("pipeline_config.yaml")
        return PipelineConfig(**data) if data else PipelineConfig()
    
    def _load_model_config(self) -> ModelConfig:
        """Load model configuration from YAML."""
        data = self._load_yaml("model_config.yaml")
        return ModelConfig(**data) if data else ModelConfig()
    
    def _load_evaluation_config(self) -> EvaluationConfig:
        """Load evaluation configuration from YAML."""
        data = self._load_yaml("evaluation_config.yaml")
        return EvaluationConfig(**data) if data else EvaluationConfig()
    
    def get_device(self) -> str:
        """
        Get the compute device for model inference.
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        import torch
        
        # Check environment variable first
        if self.app.torch_device:
            return self.app.torch_device
        
        # Check config file
        if self.pipeline.contradiction.device:
            return self.pipeline.contradiction.device
        
        # Auto-detect
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# =============================================================================
# Global Configuration Instance
# =============================================================================

_config: Optional[Config] = None


def get_config(config_dir: Optional[Path] = None) -> Config:
    """
    Get the global configuration instance.
    
    This function implements a singleton pattern to ensure only one
    configuration instance exists throughout the application.
    
    Args:
        config_dir: Optional path to configuration directory. Only used
                   on first call.
    
    Returns:
        The global Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_dir)
    return _config


def reload_config(config_dir: Optional[Path] = None) -> Config:
    """
    Force reload of configuration.
    
    Use this if configuration files have been modified and you need
    to pick up the changes.
    
    Args:
        config_dir: Optional path to configuration directory
        
    Returns:
        The newly created Config instance
    """
    global _config
    _config = Config(config_dir)
    return _config
