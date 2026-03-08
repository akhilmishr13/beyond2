"""
SEC EDGAR Filing Downloader

This module provides functionality for downloading corporate filings from the
SEC EDGAR database. It supports 8-K, 10-K, and 10-Q filing types and respects
SEC rate limits (10 requests per second).

The SEC requires all programmatic access to include a User-Agent header with
contact information. Set this in your .env file:
    SEC_USER_AGENT=YourName your.email@example.com

Usage:
    from src.ingestion.sec_downloader import SECDownloader
    
    downloader = SECDownloader()
    filings = downloader.download_filings(
        ticker="AAPL",
        filing_types=["8-K", "10-K"],
        start_date="2025-03-08",
        end_date="2026-03-08"
    )
"""

import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger

from src.config import get_config


@dataclass
class SECFiling:
    """
    Represents a single SEC filing.
    
    Attributes:
        company_name: Full company name
        ticker: Stock ticker symbol
        cik: SEC Central Index Key
        filing_type: Type of filing (8-K, 10-K, 10-Q)
        filing_date: Date the filing was submitted
        accession_number: Unique filing identifier
        filing_url: URL to the filing index page
        document_url: URL to the main filing document
        raw_text: Extracted text content from the filing
        sections: Dictionary of parsed sections
    """
    company_name: str
    ticker: str
    cik: str
    filing_type: str
    filing_date: datetime
    accession_number: str
    filing_url: str
    document_url: Optional[str] = None
    raw_text: Optional[str] = None
    sections: Optional[Dict[str, str]] = None


class SECDownloader:
    """
    Downloads and parses SEC filings from EDGAR.
    
    This class handles:
    - Looking up company CIK numbers from tickers
    - Downloading filing indexes and documents
    - Respecting SEC rate limits
    - Basic parsing of filing content
    
    Attributes:
        user_agent: User-Agent string for SEC compliance
        rate_limit: Maximum requests per second
        base_url: SEC EDGAR base URL
    """
    
    # SEC EDGAR API endpoints
    BASE_URL = "https://www.sec.gov"
    COMPANY_SEARCH_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
    
    def __init__(self):
        """Initialize the SEC downloader with configuration."""
        config = get_config()
        
        self.user_agent = config.api.sec_user_agent
        if not self.user_agent:
            raise ValueError(
                "SEC_USER_AGENT environment variable is required. "
                "Set it to 'YourName your.email@example.com'"
            )
        
        self.rate_limit = config.pipeline.sec.rate_limit_per_second
        self.max_retries = config.pipeline.sec.max_retries
        self.retry_delay = config.pipeline.sec.retry_delay_seconds
        
        # Track request timing for rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 1.0 / self.rate_limit
        
        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        
        # CIK cache to avoid repeated lookups
        self._cik_cache: Dict[str, str] = {}
        
        logger.info(f"SECDownloader initialized with rate limit: {self.rate_limit} req/s")
    
    def _rate_limit_wait(self):
        """Wait if necessary to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(self, url: str, **kwargs) -> requests.Response:
        """
        Make a rate-limited HTTP request with retries.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments to pass to requests.get
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                self._rate_limit_wait()
                response = self.session.get(url, timeout=30, **kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
    
    def get_cik(self, ticker: str) -> Optional[str]:
        """
        Look up the CIK number for a ticker symbol.
        
        The CIK (Central Index Key) is a unique identifier assigned by the SEC
        to each company that files documents.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            
        Returns:
            10-digit CIK string (zero-padded) or None if not found
        """
        ticker = ticker.upper()
        
        # Check cache first
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]
        
        try:
            # Use SEC's company tickers JSON endpoint
            url = "https://www.sec.gov/files/company_tickers.json"
            response = self._make_request(url)
            data = response.json()
            
            # Search for the ticker
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker:
                    cik = str(entry["cik_str"]).zfill(10)
                    self._cik_cache[ticker] = cik
                    logger.debug(f"Found CIK {cik} for ticker {ticker}")
                    return cik
            
            logger.warning(f"CIK not found for ticker: {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error looking up CIK for {ticker}: {e}")
            return None
    
    def get_company_submissions(self, cik: str) -> Dict:
        """
        Get all submissions for a company from SEC.
        
        Args:
            cik: 10-digit CIK number
            
        Returns:
            Dictionary containing company info and filing history
        """
        url = self.SUBMISSIONS_URL.format(cik=cik)
        response = self._make_request(url)
        return response.json()
    
    def download_filings(
        self,
        ticker: str,
        filing_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_filings: int = 100
    ) -> Iterator[SECFiling]:
        """
        Download filings for a company.
        
        This is the main entry point for downloading SEC filings. It handles
        CIK lookup, filtering by date and type, and downloading documents.
        
        Args:
            ticker: Stock ticker symbol
            filing_types: List of filing types to download (default: ["8-K", "10-K", "10-Q"])
            start_date: Start date for filtering (YYYY-MM-DD)
            end_date: End date for filtering (YYYY-MM-DD)
            max_filings: Maximum number of filings to return
            
        Yields:
            SECFiling objects for each matching filing
        """
        filing_types = filing_types or ["8-K", "10-K", "10-Q"]
        
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        
        # Get CIK
        cik = self.get_cik(ticker)
        if not cik:
            logger.error(f"Cannot download filings: CIK not found for {ticker}")
            return
        
        logger.info(f"Downloading {filing_types} filings for {ticker} (CIK: {cik})")
        
        try:
            # Get company submissions
            submissions = self.get_company_submissions(cik)
            company_name = submissions.get("name", ticker)
            
            # Get recent filings
            recent = submissions.get("filings", {}).get("recent", {})
            if not recent:
                logger.warning(f"No filings found for {ticker}")
                return
            
            # Iterate through filings
            count = 0
            for i, form in enumerate(recent.get("form", [])):
                if count >= max_filings:
                    break
                
                # Check filing type
                if form not in filing_types:
                    continue
                
                # Parse filing date
                filing_date_str = recent["filingDate"][i]
                filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
                
                # Check date range
                if start_dt and filing_date < start_dt:
                    continue
                if end_dt and filing_date > end_dt:
                    continue
                
                # Get accession number (remove dashes for URL)
                accession = recent["accessionNumber"][i]
                accession_nodash = accession.replace("-", "")
                
                # Construct filing URL
                filing_url = self.FILING_URL.format(cik=cik.lstrip("0"), accession=accession_nodash)
                
                # Get primary document
                primary_doc = recent.get("primaryDocument", [None])[i]
                document_url = urljoin(filing_url, primary_doc) if primary_doc else None
                
                # Create filing object
                filing = SECFiling(
                    company_name=company_name,
                    ticker=ticker,
                    cik=cik,
                    filing_type=form,
                    filing_date=filing_date,
                    accession_number=accession,
                    filing_url=filing_url,
                    document_url=document_url,
                )
                
                # Download and parse the filing content
                try:
                    filing = self._download_filing_content(filing)
                except Exception as e:
                    logger.warning(f"Failed to download content for {accession}: {e}")
                
                yield filing
                count += 1
            
            logger.info(f"Downloaded {count} filings for {ticker}")
            
        except Exception as e:
            logger.error(f"Error downloading filings for {ticker}: {e}")
            raise
    
    def _download_filing_content(self, filing: SECFiling) -> SECFiling:
        """
        Download and parse the content of a filing.
        
        Args:
            filing: SECFiling object with URLs set
            
        Returns:
            SECFiling object with raw_text and sections populated
        """
        if not filing.document_url:
            return filing
        
        try:
            response = self._make_request(filing.document_url)
            content = response.text
            
            # Parse HTML content
            soup = BeautifulSoup(content, "lxml")
            
            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator="\n")
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines()]
            text = "\n".join(line for line in lines if line)
            
            filing.raw_text = text
            
            # Parse sections based on filing type
            if filing.filing_type in ["10-K", "10-Q"]:
                filing.sections = self._parse_10k_10q_sections(soup, text)
            elif filing.filing_type == "8-K":
                filing.sections = self._parse_8k_sections(soup, text)
            
        except Exception as e:
            logger.warning(f"Error downloading filing content: {e}")
        
        return filing
    
    def _parse_10k_10q_sections(
        self,
        soup: BeautifulSoup,
        text: str
    ) -> Dict[str, str]:
        """
        Parse sections from 10-K and 10-Q filings.
        
        These filings have standard sections like:
        - Item 1: Business
        - Item 1A: Risk Factors
        - Item 7: Management's Discussion and Analysis
        
        Args:
            soup: Parsed HTML
            text: Plain text content
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        
        # Common section patterns
        section_patterns = {
            "item_1": r"item\s+1[.\s]+business",
            "item_1a": r"item\s+1a[.\s]+risk\s+factors",
            "item_7": r"item\s+7[.\s]+(management.s\s+discussion|md&a)",
            "item_7a": r"item\s+7a[.\s]+quantitative",
            "item_8": r"item\s+8[.\s]+financial\s+statements",
        }
        
        text_lower = text.lower()
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                start_pos = match.start()
                
                # Find the next section or end of document
                next_section_pos = len(text)
                for other_name, other_pattern in section_patterns.items():
                    if other_name != section_name:
                        other_match = re.search(other_pattern, text_lower[start_pos + 100:])
                        if other_match:
                            pos = start_pos + 100 + other_match.start()
                            if pos < next_section_pos:
                                next_section_pos = pos
                
                # Extract section content (limit to reasonable size)
                section_text = text[start_pos:next_section_pos]
                if len(section_text) > 100000:
                    section_text = section_text[:100000] + "\n[TRUNCATED]"
                
                sections[section_name] = section_text
        
        return sections
    
    def _parse_8k_sections(
        self,
        soup: BeautifulSoup,
        text: str
    ) -> Dict[str, str]:
        """
        Parse sections from 8-K filings.
        
        8-K filings report material events and typically contain:
        - Item 2.02: Results of Operations
        - Item 7.01: Regulation FD Disclosure
        - Item 8.01: Other Events
        - Exhibit 99.1: Press Releases
        
        Args:
            soup: Parsed HTML
            text: Plain text content
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        
        section_patterns = {
            "item_2_02": r"item\s+2\.02",
            "item_7_01": r"item\s+7\.01",
            "item_8_01": r"item\s+8\.01",
            "exhibit_99": r"exhibit\s+99",
        }
        
        text_lower = text.lower()
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                start_pos = match.start()
                
                # Find reasonable end point
                end_patterns = [r"item\s+\d+\.\d+", r"signatures", r"exhibit\s+\d+"]
                end_pos = len(text)
                
                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, text_lower[start_pos + 50:])
                    if end_match:
                        pos = start_pos + 50 + end_match.start()
                        if pos < end_pos:
                            end_pos = pos
                
                section_text = text[start_pos:end_pos]
                if len(section_text) > 50000:
                    section_text = section_text[:50000] + "\n[TRUNCATED]"
                
                sections[section_name] = section_text
        
        return sections
    
    def save_filing(
        self,
        filing: SECFiling,
        output_dir: Path
    ) -> Path:
        """
        Save a filing to disk.
        
        Args:
            filing: SECFiling to save
            output_dir: Directory to save to
            
        Returns:
            Path to the saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        date_str = filing.filing_date.strftime("%Y%m%d")
        filename = f"{filing.ticker}_{filing.filing_type}_{date_str}_{filing.accession_number}.txt"
        filepath = output_dir / filename
        
        # Save content
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"COMPANY: {filing.company_name}\n")
            f.write(f"TICKER: {filing.ticker}\n")
            f.write(f"CIK: {filing.cik}\n")
            f.write(f"FILING TYPE: {filing.filing_type}\n")
            f.write(f"FILING DATE: {filing.filing_date.isoformat()}\n")
            f.write(f"ACCESSION: {filing.accession_number}\n")
            f.write(f"URL: {filing.filing_url}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            if filing.raw_text:
                f.write(filing.raw_text)
        
        logger.debug(f"Saved filing to {filepath}")
        return filepath
