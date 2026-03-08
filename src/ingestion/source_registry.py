"""
Source Registry

This module provides a unified interface for accessing all data sources
in the Corporate Narrative Engine. It acts as a facade over the individual
scrapers and downloaders, providing a consistent API for data ingestion.

The registry handles:
- Source configuration and initialization
- Unified search interface
- Data aggregation from multiple sources
- Error handling and logging

Usage:
    from src.ingestion.source_registry import SourceRegistry
    
    registry = SourceRegistry()
    
    # Get all documents for a company
    documents = registry.collect_all_documents(
        ticker="AAPL",
        company_name="Apple Inc.",
        start_date="2025-03-08",
        end_date="2026-03-08"
    )
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Union

from loguru import logger

from src.config import get_config
from src.ingestion.sec_downloader import SECDownloader, SECFiling
from src.ingestion.news_scraper_gdelt import GDELTScraper, NewsArticle
from src.ingestion.news_scraper_newsapi import NewsAPIScraper
from src.ingestion.news_scraper_rss import RSSScraper
from src.ingestion.market_data import MarketDataFetcher


@dataclass
class CollectedDocument:
    """
    Unified document representation from any source.
    
    This provides a common format for documents from SEC filings,
    news articles, and other sources.
    
    Attributes:
        source_type: Type of source (sec, gdelt, newsapi, rss)
        doc_type: Document type (8-K, 10-K, news, etc.)
        ticker: Stock ticker symbol
        company: Company name
        title: Document title or headline
        content: Full text content
        published_date: Publication/filing date
        url: Source URL
        speaker: Speaker name (if applicable)
        speaker_role: Speaker role (CEO, CFO, etc.)
        sections: Parsed sections (for SEC filings)
        ceo_quotes: Extracted CEO quotes
        metadata: Additional source-specific metadata
    """
    source_type: str
    doc_type: str
    ticker: str
    company: str
    title: str
    content: str
    published_date: datetime
    url: Optional[str] = None
    speaker: Optional[str] = None
    speaker_role: Optional[str] = None
    sections: Optional[Dict[str, str]] = None
    ceo_quotes: Optional[List[str]] = None
    metadata: Optional[Dict] = None


class SourceRegistry:
    """
    Unified interface for all data sources.
    
    This class manages and coordinates access to:
    - SEC EDGAR for corporate filings
    - GDELT for historical news
    - NewsAPI for recent news
    - RSS feeds for real-time news
    - Market data from Yahoo Finance
    
    Attributes:
        sec_downloader: SEC EDGAR downloader
        gdelt_scraper: GDELT news scraper
        newsapi_scraper: NewsAPI scraper
        rss_scraper: RSS feed scraper
        market_fetcher: Market data fetcher
    """
    
    def __init__(self):
        """Initialize all data sources."""
        config = get_config()
        
        # Initialize sources based on configuration
        self.sec_downloader = None
        self.gdelt_scraper = None
        self.newsapi_scraper = None
        self.rss_scraper = None
        self.market_fetcher = None
        
        # SEC is always enabled
        try:
            self.sec_downloader = SECDownloader()
            logger.info("SEC downloader initialized")
        except Exception as e:
            logger.warning(f"SEC downloader initialization failed: {e}")
        
        # Initialize news sources based on config
        news_config = config.pipeline.news.sources if hasattr(config.pipeline.news, 'sources') else {}
        
        if news_config.get("gdelt", {}).get("enabled", True):
            try:
                self.gdelt_scraper = GDELTScraper()
                logger.info("GDELT scraper initialized")
            except Exception as e:
                logger.warning(f"GDELT scraper initialization failed: {e}")
        
        if news_config.get("newsapi", {}).get("enabled", True):
            try:
                self.newsapi_scraper = NewsAPIScraper()
                logger.info("NewsAPI scraper initialized")
            except Exception as e:
                logger.warning(f"NewsAPI scraper initialization failed: {e}")
        
        if news_config.get("rss", {}).get("enabled", True):
            try:
                self.rss_scraper = RSSScraper()
                logger.info("RSS scraper initialized")
            except Exception as e:
                logger.warning(f"RSS scraper initialization failed: {e}")
        
        # Market data is always enabled
        try:
            self.market_fetcher = MarketDataFetcher()
            logger.info("Market data fetcher initialized")
        except Exception as e:
            logger.warning(f"Market data fetcher initialization failed: {e}")
        
        logger.info("SourceRegistry initialized")
    
    def collect_sec_filings(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        filing_types: Optional[List[str]] = None
    ) -> Iterator[CollectedDocument]:
        """
        Collect SEC filings for a company.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            filing_types: List of filing types (default: 8-K, 10-K, 10-Q)
            
        Yields:
            CollectedDocument objects for each filing
        """
        if not self.sec_downloader:
            logger.warning("SEC downloader not available")
            return
        
        filing_types = filing_types or ["8-K", "10-K", "10-Q"]
        
        logger.info(f"Collecting SEC filings for {ticker}")
        
        for filing in self.sec_downloader.download_filings(
            ticker=ticker,
            filing_types=filing_types,
            start_date=start_date,
            end_date=end_date
        ):
            yield CollectedDocument(
                source_type="sec",
                doc_type=filing.filing_type,
                ticker=ticker,
                company=filing.company_name,
                title=f"{filing.filing_type} - {filing.filing_date.strftime('%Y-%m-%d')}",
                content=filing.raw_text or "",
                published_date=filing.filing_date,
                url=filing.filing_url,
                sections=filing.sections,
                metadata={
                    "cik": filing.cik,
                    "accession_number": filing.accession_number,
                }
            )
    
    def collect_news_articles(
        self,
        ticker: str,
        company_name: str,
        start_date: str,
        end_date: str,
        sources: Optional[List[str]] = None,
        max_per_source: int = 100
    ) -> Iterator[CollectedDocument]:
        """
        Collect news articles from all enabled sources.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            sources: List of sources to use (gdelt, newsapi, rss)
            max_per_source: Maximum articles per source
            
        Yields:
            CollectedDocument objects for each article
        """
        sources = sources or ["gdelt", "newsapi", "rss"]
        
        logger.info(f"Collecting news for {ticker} from {sources}")
        
        # Collect from GDELT
        if "gdelt" in sources and self.gdelt_scraper:
            try:
                for article in self.gdelt_scraper.search_articles(
                    company=company_name,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    max_results=max_per_source
                ):
                    yield self._article_to_document(article, "gdelt")
            except Exception as e:
                logger.error(f"Error collecting from GDELT: {e}")
        
        # Collect from NewsAPI
        if "newsapi" in sources and self.newsapi_scraper:
            try:
                for article in self.newsapi_scraper.search_articles(
                    company=company_name,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    max_results=max_per_source
                ):
                    yield self._article_to_document(article, "newsapi")
            except Exception as e:
                logger.error(f"Error collecting from NewsAPI: {e}")
        
        # Collect from RSS feeds
        if "rss" in sources and self.rss_scraper:
            try:
                for article in self.rss_scraper.search_articles(
                    company=company_name,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    max_results=max_per_source
                ):
                    yield self._article_to_document(article, "rss")
            except Exception as e:
                logger.error(f"Error collecting from RSS: {e}")
    
    def _article_to_document(
        self,
        article: NewsArticle,
        source_type: str
    ) -> CollectedDocument:
        """Convert a NewsArticle to a CollectedDocument."""
        return CollectedDocument(
            source_type=source_type,
            doc_type="news",
            ticker=article.ticker,
            company=article.company,
            title=article.title,
            content=article.content or "",
            published_date=article.published_date,
            url=article.url,
            ceo_quotes=article.ceo_quotes,
            metadata={
                "source": article.source,
                "sentiment": article.sentiment,
                **(article.metadata or {})
            }
        )
    
    def collect_all_documents(
        self,
        ticker: str,
        company_name: str,
        start_date: str,
        end_date: str,
        include_sec: bool = True,
        include_news: bool = True,
        sec_filing_types: Optional[List[str]] = None,
        news_sources: Optional[List[str]] = None,
        max_news_per_source: int = 100
    ) -> Iterator[CollectedDocument]:
        """
        Collect all documents from all sources for a company.
        
        This is the main entry point for comprehensive data collection.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_sec: Whether to include SEC filings
            include_news: Whether to include news articles
            sec_filing_types: List of SEC filing types
            news_sources: List of news sources to use
            max_news_per_source: Maximum news articles per source
            
        Yields:
            CollectedDocument objects from all sources
        """
        logger.info(
            f"Collecting all documents for {ticker} ({company_name}) "
            f"from {start_date} to {end_date}"
        )
        
        total_count = 0
        
        # Collect SEC filings
        if include_sec:
            sec_count = 0
            for doc in self.collect_sec_filings(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                filing_types=sec_filing_types
            ):
                yield doc
                sec_count += 1
                total_count += 1
            logger.info(f"Collected {sec_count} SEC filings for {ticker}")
        
        # Collect news articles
        if include_news:
            news_count = 0
            for doc in self.collect_news_articles(
                ticker=ticker,
                company_name=company_name,
                start_date=start_date,
                end_date=end_date,
                sources=news_sources,
                max_per_source=max_news_per_source
            ):
                yield doc
                news_count += 1
                total_count += 1
            logger.info(f"Collected {news_count} news articles for {ticker}")
        
        logger.info(f"Total documents collected for {ticker}: {total_count}")
    
    def get_market_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ):
        """
        Get market price data for a company.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with price data
        """
        if not self.market_fetcher:
            logger.warning("Market data fetcher not available")
            return None
        
        return self.market_fetcher.fetch_prices(ticker, start_date, end_date)
    
    def get_event_returns(
        self,
        ticker: str,
        event_date: str
    ):
        """
        Get market returns around an event date.
        
        Args:
            ticker: Stock ticker symbol
            event_date: Event date
            
        Returns:
            EventReturns object
        """
        if not self.market_fetcher:
            logger.warning("Market data fetcher not available")
            return None
        
        return self.market_fetcher.compute_event_returns(ticker, event_date)
    
    def get_company_info(self, ticker: str) -> Dict:
        """
        Get company information.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        if self.market_fetcher:
            return self.market_fetcher.get_company_info(ticker)
        return {"ticker": ticker, "name": ticker}
