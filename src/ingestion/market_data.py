"""
Market Data Fetcher

This module provides functionality for fetching historical stock price data
and computing various return metrics. It uses yfinance as the primary data
source, which provides free access to Yahoo Finance data.

Computed metrics include:
- Daily returns
- Multi-day returns (1d, 3d, 5d, 10d)
- Abnormal returns (vs market benchmark)
- Volatility measures

Usage:
    from src.ingestion.market_data import MarketDataFetcher
    
    fetcher = MarketDataFetcher()
    
    # Fetch price data
    prices = fetcher.fetch_prices("AAPL", start_date="2025-03-08", end_date="2026-03-08")
    
    # Compute returns around an event
    returns = fetcher.compute_event_returns("AAPL", event_date="2025-06-15")
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from src.config import get_config


@dataclass
class PriceData:
    """
    Container for stock price data.
    
    Attributes:
        ticker: Stock ticker symbol
        date: Trading date
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        adj_close: Adjusted closing price
        volume: Trading volume
        daily_return: Daily return (close-to-close)
    """
    ticker: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int
    daily_return: Optional[float] = None


@dataclass
class EventReturns:
    """
    Returns computed around an event date.
    
    Attributes:
        ticker: Stock ticker symbol
        event_date: Date of the event
        return_same_day: Return on event day
        return_1d: Return T+1
        return_3d: Return T+3
        return_5d: Return T+5
        return_10d: Return T+10
        abnormal_return_1d: Abnormal return vs benchmark T+1
        abnormal_return_3d: Abnormal return vs benchmark T+3
        abnormal_return_5d: Abnormal return vs benchmark T+5
        market_return_1d: Benchmark return T+1
        volatility_30d: 30-day rolling volatility prior to event
    """
    ticker: str
    event_date: datetime
    return_same_day: Optional[float] = None
    return_1d: Optional[float] = None
    return_3d: Optional[float] = None
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    abnormal_return_1d: Optional[float] = None
    abnormal_return_3d: Optional[float] = None
    abnormal_return_5d: Optional[float] = None
    market_return_1d: Optional[float] = None
    volatility_30d: Optional[float] = None


class MarketDataFetcher:
    """
    Fetches stock market data and computes return metrics.
    
    This class handles:
    - Downloading historical OHLCV data from Yahoo Finance
    - Computing daily and multi-day returns
    - Computing abnormal returns vs market benchmark
    - Calculating volatility measures
    
    Attributes:
        benchmark_ticker: Ticker for market benchmark (default: SPY)
        return_windows: List of return windows to compute
    """
    
    def __init__(self):
        """Initialize the market data fetcher."""
        config = get_config()
        
        self.benchmark_ticker = config.pipeline.market.benchmark_ticker
        self.return_windows = config.pipeline.market.return_windows
        self.reaction_thresholds = config.pipeline.market.reaction_thresholds
        
        # Cache for downloaded data
        self._price_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info(
            f"MarketDataFetcher initialized with benchmark: {self.benchmark_ticker}, "
            f"windows: {self.return_windows}"
        )
    
    def fetch_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data and computed returns
        """
        cache_key = f"{ticker}_{start_date}_{end_date}"
        
        if use_cache and cache_key in self._price_cache:
            return self._price_cache[cache_key]
        
        logger.info(f"Fetching prices for {ticker} from {start_date} to {end_date}")
        
        try:
            # Add buffer for computing returns
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=45)
            
            # Download data using yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_date,
                auto_adjust=False
            )
            
            if df.empty:
                logger.warning(f"No price data found for {ticker}")
                return pd.DataFrame()
            
            # Rename columns to lowercase
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            
            # Ensure we have the required columns
            required_cols = ["open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return pd.DataFrame()
            
            # Add ticker column
            df["ticker"] = ticker
            
            # Compute daily returns
            df["daily_return"] = df["close"].pct_change()
            
            # Compute multi-day forward returns
            for window in self.return_windows:
                df[f"return_{window}d"] = df["close"].pct_change(periods=window).shift(-window)
            
            # Compute rolling volatility
            df["volatility_30d"] = df["daily_return"].rolling(window=30).std() * np.sqrt(252)
            
            # Filter to requested date range
            df = df[df.index >= start_date]
            
            if use_cache:
                self._price_cache[cache_key] = df
            
            logger.info(f"Fetched {len(df)} price records for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching prices for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_benchmark(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch benchmark (market) data.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with benchmark prices and returns
        """
        return self.fetch_prices(self.benchmark_ticker, start_date, end_date)
    
    def compute_event_returns(
        self,
        ticker: str,
        event_date: str,
        prices_df: Optional[pd.DataFrame] = None
    ) -> EventReturns:
        """
        Compute returns around an event date.
        
        This method calculates various return metrics for analyzing
        market reaction to an event.
        
        Args:
            ticker: Stock ticker symbol
            event_date: Date of the event (YYYY-MM-DD)
            prices_df: Optional pre-fetched price data
            
        Returns:
            EventReturns object with computed metrics
        """
        event_dt = datetime.strptime(event_date, "%Y-%m-%d")
        
        # Fetch price data if not provided
        if prices_df is None:
            start = (event_dt - timedelta(days=60)).strftime("%Y-%m-%d")
            end = (event_dt + timedelta(days=30)).strftime("%Y-%m-%d")
            prices_df = self.fetch_prices(ticker, start, end)
        
        if prices_df.empty:
            return EventReturns(ticker=ticker, event_date=event_dt)
        
        # Find the event date or nearest trading day
        try:
            if event_date in prices_df.index:
                event_idx = prices_df.index.get_loc(event_date)
            else:
                # Find nearest trading day after event date
                future_dates = prices_df.index[prices_df.index >= event_date]
                if len(future_dates) == 0:
                    logger.warning(f"No trading data on or after {event_date}")
                    return EventReturns(ticker=ticker, event_date=event_dt)
                nearest_date = future_dates[0]
                event_idx = prices_df.index.get_loc(nearest_date)
        except Exception as e:
            logger.warning(f"Could not locate event date {event_date}: {e}")
            return EventReturns(ticker=ticker, event_date=event_dt)
        
        # Get returns from the dataframe
        row = prices_df.iloc[event_idx]
        
        returns = EventReturns(
            ticker=ticker,
            event_date=event_dt,
            return_same_day=row.get("daily_return"),
            return_1d=row.get("return_1d"),
            return_3d=row.get("return_3d"),
            return_5d=row.get("return_5d"),
            return_10d=row.get("return_10d"),
            volatility_30d=row.get("volatility_30d"),
        )
        
        # Compute abnormal returns
        try:
            benchmark_df = self.fetch_benchmark(
                (event_dt - timedelta(days=60)).strftime("%Y-%m-%d"),
                (event_dt + timedelta(days=30)).strftime("%Y-%m-%d")
            )
            
            if not benchmark_df.empty and event_date in benchmark_df.index:
                bench_row = benchmark_df.loc[event_date]
                
                returns.market_return_1d = bench_row.get("return_1d")
                
                if returns.return_1d is not None and returns.market_return_1d is not None:
                    returns.abnormal_return_1d = returns.return_1d - returns.market_return_1d
                
                if returns.return_3d is not None:
                    bench_3d = bench_row.get("return_3d")
                    if bench_3d is not None:
                        returns.abnormal_return_3d = returns.return_3d - bench_3d
                
                if returns.return_5d is not None:
                    bench_5d = bench_row.get("return_5d")
                    if bench_5d is not None:
                        returns.abnormal_return_5d = returns.return_5d - bench_5d
                        
        except Exception as e:
            logger.warning(f"Error computing abnormal returns: {e}")
        
        return returns
    
    def label_market_reaction(
        self,
        returns: EventReturns,
        window: str = "return_1d"
    ) -> str:
        """
        Label the market reaction as positive, negative, or neutral.
        
        Args:
            returns: EventReturns object
            window: Which return window to use for labeling
            
        Returns:
            Label string: "positive", "negative", or "neutral"
        """
        # Get the return value
        ret = getattr(returns, window, None)
        
        if ret is None:
            return "neutral"
        
        pos_threshold = self.reaction_thresholds["positive"]
        neg_threshold = self.reaction_thresholds["negative"]
        
        if ret > pos_threshold:
            return "positive"
        elif ret < neg_threshold:
            return "negative"
        else:
            return "neutral"
    
    def get_company_info(self, ticker: str) -> Dict:
        """
        Get company information from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                "ticker": ticker,
                "name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "exchange": info.get("exchange", ""),
            }
        except Exception as e:
            logger.warning(f"Could not fetch company info for {ticker}: {e}")
            return {"ticker": ticker, "name": ticker}
    
    def batch_fetch_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch prices for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping tickers to their price DataFrames
        """
        results = {}
        
        for ticker in tickers:
            try:
                df = self.fetch_prices(ticker, start_date, end_date)
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                logger.warning(f"Failed to fetch prices for {ticker}: {e}")
        
        logger.info(f"Fetched prices for {len(results)}/{len(tickers)} tickers")
        return results
    
    def compute_correlation(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Tuple[float, float]:
        """
        Compute correlation with market benchmark.
        
        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            
        Returns:
            Tuple of (correlation, beta)
        """
        stock_df = self.fetch_prices(ticker, start_date, end_date)
        bench_df = self.fetch_benchmark(start_date, end_date)
        
        if stock_df.empty or bench_df.empty:
            return (0.0, 0.0)
        
        # Align dates
        common_dates = stock_df.index.intersection(bench_df.index)
        stock_returns = stock_df.loc[common_dates, "daily_return"].dropna()
        bench_returns = bench_df.loc[common_dates, "daily_return"].dropna()
        
        if len(stock_returns) < 30:
            return (0.0, 0.0)
        
        # Compute correlation
        correlation = stock_returns.corr(bench_returns)
        
        # Compute beta
        covariance = stock_returns.cov(bench_returns)
        variance = bench_returns.var()
        beta = covariance / variance if variance > 0 else 0.0
        
        return (correlation, beta)
