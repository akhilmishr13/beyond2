"""
Data Ingestion Module

This module provides functionality for collecting data from various sources:
- SEC EDGAR API for corporate filings (8-K, 10-K, 10-Q)
- News sources (GDELT, NewsAPI, RSS feeds)
- Market data (yfinance)

All ingestion modules implement a common interface and can be used
individually or through the unified SourceRegistry.

Example:
    from src.ingestion import SECDownloader, MarketDataFetcher
    
    # Download SEC filings
    downloader = SECDownloader()
    filings = downloader.download_filings("AAPL", filing_types=["8-K", "10-K"])
    
    # Fetch market data
    fetcher = MarketDataFetcher()
    prices = fetcher.fetch_prices("AAPL", start_date="2025-01-01")
"""

from src.ingestion.sec_downloader import SECDownloader
from src.ingestion.news_scraper_gdelt import GDELTScraper
from src.ingestion.news_scraper_newsapi import NewsAPIScraper
from src.ingestion.news_scraper_rss import RSSScraper
from src.ingestion.market_data import MarketDataFetcher
from src.ingestion.source_registry import SourceRegistry

__all__ = [
    "SECDownloader",
    "GDELTScraper",
    "NewsAPIScraper",
    "RSSScraper",
    "MarketDataFetcher",
    "SourceRegistry",
]
