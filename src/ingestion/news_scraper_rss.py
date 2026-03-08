"""
RSS Feed News Scraper

This module provides functionality for scraping news articles from RSS feeds.
RSS feeds are a free and reliable source of news that don't require API keys
and often provide more historical data than API-based sources.

The scraper monitors multiple financial news RSS feeds and extracts articles
that mention target companies.

Usage:
    from src.ingestion.news_scraper_rss import RSSScraper
    
    scraper = RSSScraper()
    articles = scraper.search_articles(
        company="Apple",
        ticker="AAPL"
    )
"""

import re
import time
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Dict, Iterator, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup
from loguru import logger

from src.config import get_config
from src.ingestion.news_scraper_gdelt import NewsArticle


@dataclass
class RSSFeed:
    """Configuration for an RSS feed."""
    name: str
    url: str
    category: str = "general"


class RSSScraper:
    """
    Scrapes news articles from RSS feeds.
    
    This scraper monitors a configurable list of financial news RSS feeds
    and extracts articles that mention specific companies.
    
    Attributes:
        feeds: List of RSS feeds to monitor
        session: HTTP session for requests
    """
    
    # Default financial news feeds
    DEFAULT_FEEDS = [
        RSSFeed("Reuters Business", "https://feeds.reuters.com/reuters/businessNews", "business"),
        RSSFeed("Reuters Tech", "https://feeds.reuters.com/reuters/technologyNews", "tech"),
        RSSFeed("Yahoo Finance", "https://finance.yahoo.com/news/rssindex", "finance"),
        RSSFeed("MarketWatch Top", "https://feeds.marketwatch.com/marketwatch/topstories", "markets"),
        RSSFeed("CNBC Top", "https://www.cnbc.com/id/100003114/device/rss/rss.html", "business"),
        RSSFeed("Bloomberg Markets", "https://feeds.bloomberg.com/markets/news.rss", "markets"),
        RSSFeed("WSJ Business", "https://feeds.a]wsj.com/rss/RSSBusinessNews.xml", "business"),
    ]
    
    def __init__(self, custom_feeds: Optional[List[RSSFeed]] = None):
        """
        Initialize the RSS scraper.
        
        Args:
            custom_feeds: Optional list of custom RSS feeds to use instead
                         of the defaults
        """
        config = get_config()
        
        # Use custom feeds or load from config or use defaults
        if custom_feeds:
            self.feeds = custom_feeds
        else:
            # Try to load from config
            rss_config = config.pipeline.news.sources.get("rss", {}) if hasattr(config.pipeline.news, 'sources') else {}
            feed_configs = rss_config.get("feeds", [])
            
            if feed_configs:
                self.feeds = [
                    RSSFeed(f["name"], f["url"], f.get("category", "general"))
                    for f in feed_configs
                ]
            else:
                self.feeds = self.DEFAULT_FEEDS
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CorporateNarrativeEngine/1.0 RSS Reader"
        })
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.5
        
        logger.info(f"RSSScraper initialized with {len(self.feeds)} feeds")
    
    def _rate_limit_wait(self):
        """Wait to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def fetch_feed(self, feed: RSSFeed) -> List[Dict]:
        """
        Fetch and parse an RSS feed.
        
        Args:
            feed: RSSFeed configuration
            
        Returns:
            List of entry dictionaries from the feed
        """
        self._rate_limit_wait()
        
        try:
            logger.debug(f"Fetching feed: {feed.name}")
            parsed = feedparser.parse(feed.url)
            
            if parsed.bozo:
                logger.warning(f"Feed parsing issue for {feed.name}: {parsed.bozo_exception}")
            
            entries = []
            for entry in parsed.entries:
                entries.append({
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "summary": entry.get("summary", ""),
                    "published": entry.get("published", ""),
                    "source": feed.name,
                    "category": feed.category,
                })
            
            logger.debug(f"Found {len(entries)} entries in {feed.name}")
            return entries
            
        except Exception as e:
            logger.error(f"Error fetching feed {feed.name}: {e}")
            return []
    
    def search_articles(
        self,
        company: str,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 100
    ) -> Iterator[NewsArticle]:
        """
        Search for articles about a company across all RSS feeds.
        
        This method fetches all configured RSS feeds and filters entries
        that mention the company name or ticker symbol.
        
        Args:
            company: Company name to search for
            ticker: Stock ticker symbol
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            max_results: Maximum number of articles to return
            
        Yields:
            NewsArticle objects for matching articles
        """
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        
        # Build search patterns (case-insensitive)
        company_pattern = re.compile(re.escape(company), re.IGNORECASE)
        ticker_pattern = re.compile(r'\b' + re.escape(ticker) + r'\b', re.IGNORECASE)
        
        logger.info(f"Searching RSS feeds for {company} ({ticker})")
        
        count = 0
        for feed in self.feeds:
            if count >= max_results:
                break
            
            entries = self.fetch_feed(feed)
            
            for entry in entries:
                if count >= max_results:
                    break
                
                # Check if article mentions company or ticker
                text_to_search = f"{entry['title']} {entry['summary']}"
                
                if not (company_pattern.search(text_to_search) or 
                        ticker_pattern.search(text_to_search)):
                    continue
                
                # Parse publication date
                pub_date = self._parse_date(entry["published"])
                
                # Apply date filters
                if start_dt and pub_date < start_dt:
                    continue
                if end_dt and pub_date > end_dt:
                    continue
                
                # Create article
                article = NewsArticle(
                    title=entry["title"],
                    url=entry["link"],
                    source=entry["source"],
                    published_date=pub_date,
                    company=company,
                    ticker=ticker,
                    content=entry["summary"],
                    metadata={"category": entry["category"]},
                )
                
                yield article
                count += 1
        
        logger.info(f"Found {count} articles from RSS feeds for {ticker}")
    
    def _parse_date(self, date_str: str) -> datetime:
        """
        Parse various date formats from RSS feeds.
        
        RSS feeds use different date formats. This method tries to parse
        common formats.
        
        Args:
            date_str: Date string from RSS feed
            
        Returns:
            Parsed datetime object
        """
        if not date_str:
            return datetime.now()
        
        # Try RFC 2822 format (common in RSS)
        try:
            return parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            pass
        
        # Try ISO format
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            pass
        
        # Try common formats
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S",
            "%d %b %Y %H:%M:%S",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str[:30], fmt)
            except ValueError:
                continue
        
        logger.debug(f"Could not parse date: {date_str}")
        return datetime.now()
    
    def fetch_article_content(self, article: NewsArticle) -> NewsArticle:
        """
        Fetch full article content from the source URL.
        
        Args:
            article: NewsArticle with URL set
            
        Returns:
            NewsArticle with content populated
        """
        if not article.url:
            return article
        
        self._rate_limit_wait()
        
        try:
            response = self.session.get(
                article.url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                  "AppleWebKit/537.36"
                }
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "lxml")
            
            # Remove noise
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "ads"]):
                tag.decompose()
            
            # Find article content
            content = None
            for selector in ["article", ".article-body", ".story-body", 
                           ".post-content", "main", ".content"]:
                elem = soup.select_one(selector)
                if elem:
                    content = elem.get_text(separator="\n", strip=True)
                    break
            
            if content:
                # Clean content
                lines = [line.strip() for line in content.split("\n") if len(line.strip()) > 20]
                article.content = "\n".join(lines)[:30000]
            
        except Exception as e:
            logger.debug(f"Could not fetch content from {article.url}: {e}")
        
        return article
    
    def get_all_recent_articles(
        self,
        max_per_feed: int = 50
    ) -> Iterator[NewsArticle]:
        """
        Get all recent articles from all feeds (unfiltered).
        
        Useful for general market sentiment analysis.
        
        Args:
            max_per_feed: Maximum articles per feed
            
        Yields:
            NewsArticle objects
        """
        for feed in self.feeds:
            entries = self.fetch_feed(feed)
            
            for entry in entries[:max_per_feed]:
                pub_date = self._parse_date(entry["published"])
                
                yield NewsArticle(
                    title=entry["title"],
                    url=entry["link"],
                    source=entry["source"],
                    published_date=pub_date,
                    company="",
                    ticker="",
                    content=entry["summary"],
                    metadata={"category": entry["category"]},
                )
