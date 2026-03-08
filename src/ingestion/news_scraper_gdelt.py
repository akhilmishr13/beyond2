"""
GDELT News Scraper

This module provides functionality for scraping news articles from the GDELT
(Global Database of Events, Language, and Tone) project. GDELT monitors news
sources worldwide and provides free access to article metadata and content.

GDELT is useful for historical news data as it maintains archives going back
several years, unlike most news APIs which have limited lookback periods.

Note: GDELT provides article URLs and metadata, but full article content
must be fetched separately from the original sources.

Usage:
    from src.ingestion.news_scraper_gdelt import GDELTScraper
    
    scraper = GDELTScraper()
    articles = scraper.search_articles(
        company="Apple",
        ticker="AAPL",
        start_date="2025-03-08",
        end_date="2026-03-08"
    )
"""

import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from loguru import logger

from src.config import get_config


@dataclass
class NewsArticle:
    """
    Represents a news article.
    
    Attributes:
        title: Article headline
        url: URL to the full article
        source: Name of the news source
        published_date: Publication date
        company: Company mentioned
        ticker: Stock ticker
        content: Full article text (if fetched)
        ceo_quotes: Extracted CEO quotes
        sentiment: Computed sentiment score
        metadata: Additional metadata from source
    """
    title: str
    url: str
    source: str
    published_date: datetime
    company: str
    ticker: str
    content: Optional[str] = None
    ceo_quotes: Optional[List[str]] = None
    sentiment: Optional[float] = None
    metadata: Optional[Dict] = None


class GDELTScraper:
    """
    Scrapes news articles from GDELT.
    
    GDELT provides several APIs:
    - DOC API: Search for articles by keyword
    - GKG (Global Knowledge Graph): Entity-based search
    - Events: Structured event data
    
    This implementation uses the DOC API for article discovery.
    
    Attributes:
        base_url: GDELT DOC API base URL
        max_articles: Maximum articles per query
    """
    
    # GDELT DOC API endpoint
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    def __init__(self):
        """Initialize the GDELT scraper."""
        config = get_config()
        
        self.max_articles = config.pipeline.news.sources.get(
            "gdelt", {}
        ).get("max_articles_per_company", 500) if hasattr(config.pipeline.news, 'sources') else 500
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CorporateNarrativeEngine/1.0"
        })
        
        # Rate limiting (be polite to GDELT)
        self._last_request_time = 0.0
        self._min_request_interval = 1.0  # 1 second between requests
        
        logger.info("GDELTScraper initialized")
    
    def _rate_limit_wait(self):
        """Wait to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict) -> requests.Response:
        """
        Make a rate-limited request to GDELT.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            Response object
        """
        self._rate_limit_wait()
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"GDELT request failed: {e}")
            raise
    
    def search_articles(
        self,
        company: str,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> Iterator[NewsArticle]:
        """
        Search for news articles about a company.
        
        This method queries GDELT's DOC API for articles mentioning the
        company or its ticker symbol.
        
        Args:
            company: Company name to search for
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_results: Maximum number of articles to return
            
        Yields:
            NewsArticle objects for each matching article
        """
        max_results = max_results or self.max_articles
        
        # Build search query - search for both company name and ticker
        query = f'"{company}" OR "{ticker}"'
        
        # Format dates for GDELT (YYYYMMDDHHMMSS)
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            start_gdelt = start_dt.strftime("%Y%m%d%H%M%S")
        else:
            start_gdelt = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d%H%M%S")
        
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            end_gdelt = end_dt.strftime("%Y%m%d%H%M%S")
        else:
            end_gdelt = datetime.now().strftime("%Y%m%d%H%M%S")
        
        logger.info(f"Searching GDELT for {company} ({ticker}) from {start_date} to {end_date}")
        
        # Query parameters
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": min(max_results, 250),  # GDELT max is 250
            "format": "json",
            "startdatetime": start_gdelt,
            "enddatetime": end_gdelt,
            "sort": "datedesc",
        }
        
        try:
            response = self._make_request(self.BASE_URL, params)
            data = response.json()
            
            articles = data.get("articles", [])
            logger.info(f"Found {len(articles)} articles for {ticker}")
            
            count = 0
            for article_data in articles:
                if count >= max_results:
                    break
                
                try:
                    # Parse article data
                    article = self._parse_article(article_data, company, ticker)
                    if article:
                        yield article
                        count += 1
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error searching GDELT: {e}")
    
    def _parse_article(
        self,
        data: Dict,
        company: str,
        ticker: str
    ) -> Optional[NewsArticle]:
        """
        Parse article data from GDELT response.
        
        Args:
            data: Article data dictionary from GDELT
            company: Company name
            ticker: Stock ticker
            
        Returns:
            NewsArticle object or None if parsing fails
        """
        try:
            # Extract required fields
            url = data.get("url", "")
            title = data.get("title", "")
            
            if not url or not title:
                return None
            
            # Parse date
            date_str = data.get("seendate", "")
            if date_str:
                # GDELT format: YYYYMMDDTHHMMSSZ
                try:
                    published_date = datetime.strptime(date_str[:14], "%Y%m%dT%H%M%S")
                except ValueError:
                    published_date = datetime.now()
            else:
                published_date = datetime.now()
            
            # Extract source name from URL
            source = data.get("domain", "")
            if not source:
                # Extract domain from URL
                from urllib.parse import urlparse
                parsed = urlparse(url)
                source = parsed.netloc
            
            # Build metadata
            metadata = {
                "language": data.get("language", "English"),
                "socialimage": data.get("socialimage", ""),
                "tone": data.get("tone", 0),
            }
            
            return NewsArticle(
                title=title,
                url=url,
                source=source,
                published_date=published_date,
                company=company,
                ticker=ticker,
                metadata=metadata,
            )
            
        except Exception as e:
            logger.warning(f"Error parsing GDELT article: {e}")
            return None
    
    def fetch_article_content(self, article: NewsArticle) -> NewsArticle:
        """
        Fetch the full content of an article.
        
        This method attempts to download and parse the full article text
        from the original source URL.
        
        Note: Many news sites block automated access or require JavaScript,
        so this may not always succeed.
        
        Args:
            article: NewsArticle with URL set
            
        Returns:
            NewsArticle with content populated (if successful)
        """
        if not article.url:
            return article
        
        try:
            self._rate_limit_wait()
            
            response = self.session.get(
                article.url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/120.0.0.0 Safari/537.36"
                }
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, "lxml")
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
            
            # Try to find article content using common patterns
            content = None
            
            # Look for article body
            for selector in ["article", ".article-body", ".story-body", 
                           ".post-content", "[itemprop='articleBody']", "main"]:
                article_elem = soup.select_one(selector)
                if article_elem:
                    content = article_elem.get_text(separator="\n", strip=True)
                    break
            
            # Fallback to body if no article element found
            if not content:
                body = soup.find("body")
                if body:
                    content = body.get_text(separator="\n", strip=True)
            
            if content:
                # Clean up content
                lines = [line.strip() for line in content.split("\n")]
                content = "\n".join(line for line in lines if len(line) > 20)
                
                # Limit content size
                if len(content) > 50000:
                    content = content[:50000]
                
                article.content = content
                
                # Try to extract CEO quotes
                article.ceo_quotes = self._extract_ceo_quotes(content, article.company)
            
        except Exception as e:
            logger.debug(f"Could not fetch article content from {article.url}: {e}")
        
        return article
    
    def _extract_ceo_quotes(self, content: str, company: str) -> List[str]:
        """
        Extract CEO quotes from article content.
        
        Looks for patterns like:
        - "quote," said CEO Name
        - CEO Name said "quote"
        - According to CEO Name, "quote"
        
        Args:
            content: Article text
            company: Company name for context
            
        Returns:
            List of extracted quote strings
        """
        quotes = []
        
        # Patterns for CEO quotes
        patterns = [
            # "quote," said CEO Name
            r'"([^"]{20,500})"[,\s]+said\s+(?:the\s+)?(?:CEO|chief\s+executive)',
            # CEO said "quote"
            r'(?:CEO|chief\s+executive)[^"]*said[^"]*"([^"]{20,500})"',
            # According to CEO, "quote"
            r'according\s+to\s+(?:the\s+)?(?:CEO|chief\s+executive)[^"]*"([^"]{20,500})"',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            quotes.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_quotes = []
        for quote in quotes:
            if quote not in seen:
                seen.add(quote)
                unique_quotes.append(quote)
        
        return unique_quotes[:10]  # Limit to 10 quotes
