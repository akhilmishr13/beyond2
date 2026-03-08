"""
Data Ingestion Module

This module provides functionality for collecting data from various sources:
- SEC EDGAR API for corporate filings (8-K, 10-K, 10-Q)
- News sources (GDELT, NewsAPI, RSS feeds)
- Market data (yfinance)
- Company universe (all ~10,000 public companies)
- Batch processing for large-scale analysis

All ingestion modules implement a common interface and can be used
individually or through the unified SourceRegistry.

Example:
    from src.ingestion import SECDownloader, CompanyUniverse, BatchProcessor
    
    # Get all public companies
    universe = CompanyUniverse()
    all_tickers = universe.get_all_tickers()  # ~10,000 companies
    sp500 = universe.get_sp500()  # S&P 500 only
    
    # Download SEC filings
    downloader = SECDownloader()
    filings = downloader.download_filings("AAPL", filing_types=["8-K", "10-K"])
    
    # Batch process multiple companies
    processor = BatchProcessor(max_workers=5)
    results = processor.process_companies(sp500, process_fn)
"""

from src.ingestion.sec_downloader import SECDownloader
from src.ingestion.news_scraper_gdelt import GDELTScraper
from src.ingestion.news_scraper_newsapi import NewsAPIScraper
from src.ingestion.news_scraper_rss import RSSScraper
from src.ingestion.market_data import MarketDataFetcher
from src.ingestion.source_registry import SourceRegistry
from src.ingestion.company_universe import CompanyUniverse, Company
from src.ingestion.batch_processor import BatchProcessor, BatchProgress, ProcessingResult

__all__ = [
    "SECDownloader",
    "GDELTScraper",
    "NewsAPIScraper",
    "RSSScraper",
    "MarketDataFetcher",
    "SourceRegistry",
    "CompanyUniverse",
    "Company",
    "BatchProcessor",
    "BatchProgress",
    "ProcessingResult",
]
