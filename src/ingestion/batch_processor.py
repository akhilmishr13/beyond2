"""
Batch Processor for Large-Scale Company Analysis

Processes multiple companies in parallel with progress tracking,
rate limiting, and error handling.

Usage:
    from src.ingestion.batch_processor import BatchProcessor
    
    processor = BatchProcessor(max_workers=5)
    results = processor.process_companies(["AAPL", "MSFT", "GOOGL"])
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


@dataclass
class ProcessingResult:
    """Result of processing a single company."""
    ticker: str
    success: bool
    duration_seconds: float
    documents_count: int = 0
    claims_count: int = 0
    contradictions_count: int = 0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BatchProgress:
    """Tracks batch processing progress."""
    total: int
    completed: int = 0
    successful: int = 0
    failed: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def percent_complete(self) -> float:
        return (self.completed / self.total * 100) if self.total > 0 else 0
    
    @property
    def elapsed_seconds(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    @property
    def avg_seconds_per_company(self) -> float:
        return self.elapsed_seconds / self.completed if self.completed > 0 else 0
    
    @property
    def eta_seconds(self) -> float:
        remaining = self.total - self.completed
        return remaining * self.avg_seconds_per_company
    
    def to_dict(self) -> Dict:
        return {
            "total": self.total,
            "completed": self.completed,
            "successful": self.successful,
            "failed": self.failed,
            "percent_complete": round(self.percent_complete, 1),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "eta_seconds": round(self.eta_seconds, 1)
        }


class BatchProcessor:
    """
    Processes multiple companies in parallel.
    
    Features:
    - Parallel execution with configurable workers
    - Progress tracking and ETA estimation
    - Error handling and retry logic
    - Rate limiting to respect API limits
    """
    
    def __init__(
        self,
        max_workers: int = 5,
        rate_limit_per_second: float = 2.0,
        retry_failed: bool = True,
        max_retries: int = 2
    ):
        """
        Initialize the batch processor.
        
        Args:
            max_workers: Maximum parallel workers
            rate_limit_per_second: Global rate limit across all workers
            retry_failed: Whether to retry failed companies
            max_retries: Maximum retry attempts
        """
        self.max_workers = max_workers
        self.rate_limit = rate_limit_per_second
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        
        self._progress: Optional[BatchProgress] = None
        self._results: List[ProcessingResult] = []
        self._min_interval = 1.0 / rate_limit_per_second
        self._last_request_time = 0.0
    
    def _rate_limit_wait(self):
        """Wait to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
    
    def process_companies(
        self,
        tickers: List[str],
        process_fn: Callable[[str], Dict[str, Any]],
        progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple companies in parallel.
        
        Args:
            tickers: List of ticker symbols
            process_fn: Function to process each ticker, returns dict with counts
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of ProcessingResult objects
        """
        self._progress = BatchProgress(total=len(tickers))
        self._results = []
        
        logger.info(f"Starting batch processing for {len(tickers)} companies")
        logger.info(f"Workers: {self.max_workers}, Rate limit: {self.rate_limit}/s")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._process_single, ticker, process_fn): ticker
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    self._results.append(result)
                    
                    self._progress.completed += 1
                    if result.success:
                        self._progress.successful += 1
                    else:
                        self._progress.failed += 1
                    
                    if progress_callback:
                        progress_callback(self._progress)
                    
                    self._log_progress(result)
                    
                except Exception as e:
                    logger.error(f"Unexpected error for {ticker}: {e}")
                    self._progress.completed += 1
                    self._progress.failed += 1
        
        # Retry failed if enabled
        if self.retry_failed:
            self._retry_failed_companies(process_fn, progress_callback)
        
        logger.info(f"Batch complete: {self._progress.successful}/{self._progress.total} successful")
        return self._results
    
    def _process_single(
        self,
        ticker: str,
        process_fn: Callable[[str], Dict[str, Any]]
    ) -> ProcessingResult:
        """Process a single company with rate limiting."""
        self._rate_limit_wait()
        
        start_time = time.time()
        
        try:
            result_data = process_fn(ticker)
            duration = time.time() - start_time
            
            return ProcessingResult(
                ticker=ticker,
                success=True,
                duration_seconds=duration,
                documents_count=result_data.get("documents", 0),
                claims_count=result_data.get("claims", 0),
                contradictions_count=result_data.get("contradictions", 0)
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.warning(f"Failed to process {ticker}: {e}")
            
            return ProcessingResult(
                ticker=ticker,
                success=False,
                duration_seconds=duration,
                error=str(e)
            )
    
    def _retry_failed_companies(
        self,
        process_fn: Callable[[str], Dict[str, Any]],
        progress_callback: Optional[Callable[[BatchProgress], None]]
    ):
        """Retry processing for failed companies."""
        failed = [r for r in self._results if not r.success]
        
        if not failed:
            return
        
        logger.info(f"Retrying {len(failed)} failed companies...")
        
        for attempt in range(self.max_retries):
            still_failed = []
            
            for result in failed:
                self._rate_limit_wait()
                new_result = self._process_single(result.ticker, process_fn)
                
                # Update the result in our list
                idx = next(i for i, r in enumerate(self._results) if r.ticker == result.ticker)
                self._results[idx] = new_result
                
                if new_result.success:
                    self._progress.successful += 1
                    self._progress.failed -= 1
                    logger.info(f"Retry successful for {result.ticker}")
                else:
                    still_failed.append(new_result)
            
            failed = still_failed
            
            if not failed:
                break
    
    def _log_progress(self, result: ProcessingResult):
        """Log progress information."""
        status = "✓" if result.success else "✗"
        p = self._progress
        
        logger.info(
            f"[{p.completed}/{p.total}] {status} {result.ticker} "
            f"({result.duration_seconds:.1f}s) - "
            f"ETA: {p.eta_seconds/60:.1f}min"
        )
    
    @property
    def progress(self) -> Optional[BatchProgress]:
        """Get current progress."""
        return self._progress
    
    @property
    def results(self) -> List[ProcessingResult]:
        """Get all results."""
        return self._results
    
    def get_summary(self) -> Dict:
        """Get processing summary."""
        if not self._progress:
            return {}
        
        return {
            "total": self._progress.total,
            "successful": self._progress.successful,
            "failed": self._progress.failed,
            "success_rate": round(self._progress.successful / self._progress.total * 100, 1),
            "total_duration_minutes": round(self._progress.elapsed_seconds / 60, 1),
            "avg_seconds_per_company": round(self._progress.avg_seconds_per_company, 1)
        }
