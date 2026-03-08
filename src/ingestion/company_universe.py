"""
Company Universe Manager

Fetches and manages the list of all publicly traded companies from SEC EDGAR.
Provides filtering by market cap, sector, and index membership.

Usage:
    from src.ingestion.company_universe import CompanyUniverse
    
    universe = CompanyUniverse()
    all_companies = universe.get_all_companies()
    sp500 = universe.get_sp500()
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
from loguru import logger


@dataclass
class Company:
    """Represents a public company."""
    ticker: str
    name: str
    cik: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None


class CompanyUniverse:
    """
    Manages the universe of public companies.
    
    Fetches data from SEC EDGAR and provides filtering capabilities.
    """
    
    SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the company universe.
        
        Args:
            cache_dir: Directory to cache company data
        """
        self.cache_dir = cache_dir or Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._companies: Dict[str, Company] = {}
        self._loaded = False
    
    def _load_sec_companies(self) -> Dict[str, Company]:
        """
        Load all companies from SEC EDGAR.
        
        Returns:
            Dictionary mapping ticker to Company object
        """
        cache_file = self.cache_dir / "sec_companies.json"
        
        # Try cache first
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data)} companies from cache")
                return {
                    ticker: Company(
                        ticker=ticker,
                        name=info["name"],
                        cik=info["cik"]
                    )
                    for ticker, info in data.items()
                }
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        # Fetch from SEC
        logger.info("Fetching company list from SEC EDGAR...")
        try:
            response = requests.get(
                self.SEC_TICKERS_URL,
                headers={"User-Agent": "CompanyUniverse research@example.com"}
            )
            response.raise_for_status()
            data = response.json()
            
            companies = {}
            for entry in data.values():
                ticker = entry.get("ticker", "").upper()
                if ticker:
                    companies[ticker] = Company(
                        ticker=ticker,
                        name=entry.get("title", ""),
                        cik=str(entry.get("cik_str", "")).zfill(10)
                    )
            
            # Cache the results
            cache_data = {
                ticker: {"name": c.name, "cik": c.cik}
                for ticker, c in companies.items()
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
            
            logger.info(f"Fetched {len(companies)} companies from SEC")
            return companies
            
        except Exception as e:
            logger.error(f"Failed to fetch SEC companies: {e}")
            return {}
    
    def load(self) -> "CompanyUniverse":
        """Load company data."""
        if not self._loaded:
            self._companies = self._load_sec_companies()
            self._loaded = True
        return self
    
    def get_all_companies(self) -> List[Company]:
        """
        Get all public companies (~10,000).
        
        Returns:
            List of all Company objects
        """
        self.load()
        return list(self._companies.values())
    
    def get_all_tickers(self) -> List[str]:
        """
        Get all ticker symbols.
        
        Returns:
            List of ticker strings
        """
        self.load()
        return list(self._companies.keys())
    
    def get_company(self, ticker: str) -> Optional[Company]:
        """
        Get a specific company by ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Company object or None
        """
        self.load()
        return self._companies.get(ticker.upper())
    
    def get_sp500(self) -> List[str]:
        """
        Get S&P 500 constituent tickers.
        
        Returns:
            List of ~500 ticker strings
        """
        cache_file = self.cache_dir / "sp500.json"
        
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass
        
        try:
            import pandas as pd
            tables = pd.read_html(self.SP500_URL)
            sp500_df = tables[0]
            tickers = sp500_df["Symbol"].str.replace(".", "-", regex=False).tolist()
            
            with open(cache_file, "w") as f:
                json.dump(tickers, f)
            
            logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            logger.warning(f"Could not fetch S&P 500: {e}")
            # Fallback to common large caps
            return [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
                "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
                "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
                "ACN", "TMO", "ABT", "DHR", "NEE", "LIN", "PM", "TXN", "UPS"
            ]
    
    def get_by_count(self, count: int) -> List[str]:
        """
        Get top N companies (by order in SEC file, roughly by size).
        
        Args:
            count: Number of companies to return
            
        Returns:
            List of ticker strings
        """
        self.load()
        return list(self._companies.keys())[:count]
    
    def search(self, query: str) -> List[Company]:
        """
        Search companies by name or ticker.
        
        Args:
            query: Search string
            
        Returns:
            List of matching companies
        """
        self.load()
        query = query.upper()
        results = []
        
        for ticker, company in self._companies.items():
            if query in ticker or query in company.name.upper():
                results.append(company)
        
        return results[:50]  # Limit results
    
    def count(self) -> int:
        """Get total number of companies."""
        self.load()
        return len(self._companies)
    
    def validate_ticker(self, ticker: str) -> bool:
        """Check if a ticker exists."""
        self.load()
        return ticker.upper() in self._companies
