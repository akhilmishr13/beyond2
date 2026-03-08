#!/usr/bin/env python
"""
Batch Pipeline Runner

Runs the full pipeline for multiple companies with parallel processing.
Supports processing all public companies, S&P 500, or custom lists.

Usage:
    # Process S&P 500 companies (default)
    python scripts/run_batch_pipeline.py --preset sp500
    
    # Process top N companies
    python scripts/run_batch_pipeline.py --top 100
    
    # Process all public companies (~10,000)
    python scripts/run_batch_pipeline.py --all
    
    # Process specific companies
    python scripts/run_batch_pipeline.py --tickers AAPL MSFT GOOGL
    
    # Adjust parallelism
    python scripts/run_batch_pipeline.py --preset sp500 --workers 10
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config import get_config
from src.ingestion import (
    BatchProcessor,
    CompanyUniverse,
    MarketDataFetcher,
    SECDownloader,
    SourceRegistry,
)


def setup_logging(log_file: str = "batch_pipeline.log"):
    """Configure logging."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG", rotation="100 MB")


def process_single_company(ticker: str) -> dict:
    """
    Process a single company through the pipeline.
    
    Returns counts of documents, claims, and contradictions.
    """
    config = get_config()
    results = {
        "documents": 0,
        "claims": 0,
        "contradictions": 0
    }
    
    # Date range: 1 year lookback
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
    
    try:
        # 1. Download SEC filings
        sec_downloader = SECDownloader()
        filings = list(sec_downloader.download_filings(
            ticker,
            filing_types=["8-K", "10-K", "10-Q"],
            start_date=start_date,
            end_date=end_date
        ))
        results["documents"] += len(filings)
        logger.info(f"{ticker}: Downloaded {len(filings)} SEC filings")
        
        # 2. Fetch market data
        market_fetcher = MarketDataFetcher()
        prices = market_fetcher.fetch_prices(ticker, start_date, end_date)
        results["price_records"] = len(prices) if prices is not None else 0
        logger.info(f"{ticker}: Fetched {results['price_records']} price records")
        
        # 3. For now, return counts (full processing would include claim extraction)
        # In a full implementation, this would:
        # - Parse documents
        # - Extract claims using Claude
        # - Match claims across documents
        # - Detect contradictions with NLI
        # - Align with market data
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run batch pipeline for multiple companies"
    )
    
    # Company selection options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Process ALL public companies (~10,000)"
    )
    group.add_argument(
        "--preset",
        choices=["sp500", "sp100", "mega"],
        help="Use preset company lists"
    )
    group.add_argument(
        "--top",
        type=int,
        help="Process top N companies by market cap"
    )
    group.add_argument(
        "--tickers",
        nargs="+",
        help="Process specific tickers"
    )
    
    # Processing options
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=2.0,
        help="API requests per second (default: 2.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="batch_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show companies that would be processed without running"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Get company list
    universe = CompanyUniverse()
    
    if args.all:
        tickers = universe.get_all_tickers()
        logger.info(f"Selected ALL {len(tickers)} public companies")
    elif args.preset == "sp500":
        tickers = universe.get_sp500()
        logger.info(f"Selected S&P 500 ({len(tickers)} companies)")
    elif args.preset == "sp100":
        tickers = universe.get_sp500()[:100]
        logger.info(f"Selected S&P 100 ({len(tickers)} companies)")
    elif args.preset == "mega":
        # Top 50 mega caps
        tickers = universe.get_sp500()[:50]
        logger.info(f"Selected top 50 mega caps")
    elif args.top:
        tickers = universe.get_by_count(args.top)
        logger.info(f"Selected top {len(tickers)} companies")
    elif args.tickers:
        tickers = args.tickers
        logger.info(f"Selected {len(tickers)} specified companies")
    else:
        # Default to config companies
        config = get_config()
        tickers = config.pipeline.companies
        logger.info(f"Using default config companies: {tickers}")
    
    # Validate tickers
    valid_tickers = []
    for ticker in tickers:
        if universe.validate_ticker(ticker):
            valid_tickers.append(ticker)
        else:
            logger.warning(f"Unknown ticker: {ticker}")
    
    logger.info(f"Validated {len(valid_tickers)} tickers")
    
    if args.dry_run:
        logger.info("DRY RUN - Would process these companies:")
        for i, ticker in enumerate(valid_tickers[:20]):
            company = universe.get_company(ticker)
            logger.info(f"  {i+1}. {ticker} - {company.name if company else 'Unknown'}")
        if len(valid_tickers) > 20:
            logger.info(f"  ... and {len(valid_tickers) - 20} more")
        
        # Estimate time
        est_seconds = len(valid_tickers) * 30 / args.workers
        logger.info(f"\nEstimated time: {est_seconds/3600:.1f} hours")
        logger.info(f"Workers: {args.workers}")
        logger.info(f"Rate limit: {args.rate_limit}/sec")
        return
    
    # Process companies
    processor = BatchProcessor(
        max_workers=args.workers,
        rate_limit_per_second=args.rate_limit
    )
    
    def progress_callback(progress):
        if progress.completed % 10 == 0:
            logger.info(f"Progress: {progress.percent_complete:.1f}% "
                       f"({progress.completed}/{progress.total})")
    
    start_time = datetime.now(timezone.utc)
    
    results = processor.process_companies(
        valid_tickers,
        process_single_company,
        progress_callback
    )
    
    # Save results
    summary = processor.get_summary()
    summary["start_time"] = start_time.isoformat()
    summary["end_time"] = datetime.now(timezone.utc).isoformat()
    summary["results"] = [
        {
            "ticker": r.ticker,
            "success": r.success,
            "documents": r.documents_count,
            "claims": r.claims_count,
            "contradictions": r.contradictions_count,
            "error": r.error,
            "duration_seconds": r.duration_seconds
        }
        for r in results
    ]
    
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")
    logger.info(f"Summary: {summary['successful']}/{summary['total']} successful "
               f"({summary['success_rate']}%)")


if __name__ == "__main__":
    main()
