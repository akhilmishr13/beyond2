"""
NewsAPI News Scraper

This module provides functionality for fetching news articles from NewsAPI.org.
NewsAPI is a popular news aggregation service that provides access to articles
from thousands of sources worldwide.

Note: The free tier has limitations:
- 100 requests per day
- Articles from the last month only
- No access to article content (only metadata)

For production use, consider upgrading to a paid plan.

Usage:
    from src.ingestion.news_scraper_newsapi import NewsAPIScraper
    
    scraper = NewsAPIScraper()
    articles = scraper.search_articles(
        company="Apple",
        ticker="AAPL",
        start_date="2026-02-08",  # Last 30 days
        end_date="2026-03-08"
    )
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Optional

import requests
from loguru import logger

from src.config import get_config
from src.ingestion.news_scraper_gdelt import NewsArticle


class NewsAPIScraper:
    """
    Fetches news articles from NewsAPI.org.
    
    NewsAPI provides structured access to articles from thousands of news
    sources. It's good for recent news but has limited historical data.
    
    Attributes:
        api_key: NewsAPI API key
        base_url: API endpoint URL
    """
    
    BASE_URL = "https://newsapi.org/v2"
    
    def __init__(self):
        """Initialize the NewsAPI scraper."""
        config = get_config()
        
        self.api_key = config.api.newsapi_key
        if not self.api_key:
            logger.warning(
                "NEWSAPI_KEY not set. NewsAPI scraper will not work. "
                "Get a free key at https://newsapi.org/register"
            )
        
        self.session = requests.Session()
        self.session.headers.update({
            "X-Api-Key": self.api_key,
            "User-Agent": "CorporateNarrativeEngine/1.0"
        })
        
        # NewsAPI free tier is limited
        self._requests_today = 0
        self._max_requests = 100
        self._last_request_date = datetime.now().date()
        
        logger.info("NewsAPIScraper initialized")
    
    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits.
        
        Returns:
            True if request is allowed, False otherwise
        """
        today = datetime.now().date()
        if today != self._last_request_date:
            self._requests_today = 0
            self._last_request_date = today
        
        if self._requests_today >= self._max_requests:
            logger.warning("NewsAPI daily rate limit reached")
            return False
        
        return True
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Make a request to NewsAPI.
        
        Args:
            endpoint: API endpoint (e.g., "everything", "top-headlines")
            params: Query parameters
            
        Returns:
            JSON response as dictionary, or None if request fails
        """
        if not self.api_key:
            logger.error("NewsAPI key not configured")
            return None
        
        if not self._check_rate_limit():
            return None
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            self._requests_today += 1
            
            if response.status_code == 429:
                logger.warning("NewsAPI rate limit exceeded")
                return None
            
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"NewsAPI request failed: {e}")
            return None
    
    def search_articles(
        self,
        company: str,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 100
    ) -> Iterator[NewsArticle]:
        """
        Search for news articles about a company.
        
        Uses NewsAPI's "everything" endpoint to search for articles
        mentioning the company or ticker.
        
        Args:
            company: Company name to search for
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_results: Maximum number of articles to return
            
        Yields:
            NewsArticle objects for each matching article
        """
        # NewsAPI free tier only supports last 30 days
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            min_date = datetime.now() - timedelta(days=30)
            if start_dt < min_date:
                logger.warning(
                    f"NewsAPI free tier only supports last 30 days. "
                    f"Adjusting start_date from {start_date} to {min_date.date()}"
                )
                start_dt = min_date
            start_date = start_dt.strftime("%Y-%m-%d")
        
        # Build search query
        query = f'"{company}" OR "{ticker}"'
        
        logger.info(f"Searching NewsAPI for {company} ({ticker})")
        
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(max_results, 100),  # Max 100 per request
        }
        
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date
        
        data = self._make_request("everything", params)
        if not data:
            return
        
        articles = data.get("articles", [])
        logger.info(f"Found {len(articles)} articles from NewsAPI for {ticker}")
        
        for article_data in articles[:max_results]:
            try:
                article = self._parse_article(article_data, company, ticker)
                if article:
                    yield article
            except Exception as e:
                logger.warning(f"Error parsing NewsAPI article: {e}")
                continue
    
    def _parse_article(
        self,
        data: Dict,
        company: str,
        ticker: str
    ) -> Optional[NewsArticle]:
        """
        Parse article data from NewsAPI response.
        
        Args:
            data: Article data from NewsAPI
            company: Company name
            ticker: Stock ticker
            
        Returns:
            NewsArticle object or None if parsing fails
        """
        try:
            url = data.get("url", "")
            title = data.get("title", "")
            
            if not url or not title:
                return None
            
            # Parse date
            date_str = data.get("publishedAt", "")
            if date_str:
                # ISO format: 2024-01-15T10:30:00Z
                try:
                    published_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except ValueError:
                    published_date = datetime.now()
            else:
                published_date = datetime.now()
            
            # Get source info
            source_data = data.get("source", {})
            source = source_data.get("name", "Unknown")
            
            # NewsAPI provides description but not full content
            description = data.get("description", "")
            content = data.get("content", "")  # Usually truncated
            
            metadata = {
                "author": data.get("author"),
                "description": description,
                "urlToImage": data.get("urlToImage"),
            }
            
            return NewsArticle(
                title=title,
                url=url,
                source=source,
                published_date=published_date,
                company=company,
                ticker=ticker,
                content=content if content else description,
                metadata=metadata,
            )
            
        except Exception as e:
            logger.warning(f"Error parsing NewsAPI article: {e}")
            return None
    
    def get_top_headlines(
        self,
        category: str = "business",
        country: str = "us",
        max_results: int = 20
    ) -> Iterator[NewsArticle]:
        """
        Get top business headlines.
        
        This can be useful for monitoring general market sentiment.
        
        Args:
            category: News category (default: business)
            country: Country code (default: us)
            max_results: Maximum number of articles
            
        Yields:
            NewsArticle objects
        """
        params = {
            "category": category,
            "country": country,
            "pageSize": min(max_results, 100),
        }
        
        data = self._make_request("top-headlines", params)
        if not data:
            return
        
        for article_data in data.get("articles", []):
            try:
                article = self._parse_article(article_data, "", "")
                if article:
                    yield article
            except Exception as e:
                logger.warning(f"Error parsing headline: {e}")
                continue
