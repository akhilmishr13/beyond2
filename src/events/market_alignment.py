"""
Market Alignment

This module aligns contradiction events with market price data,
computing returns at various horizons and labeling market reactions.

Usage:
    from src.events.market_alignment import MarketAlignment
    
    aligner = MarketAlignment()
    events = aligner.align_events(events, market_data)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.config import get_config
from src.events.event_builder import ContradictionEvent
from src.ingestion.market_data import MarketDataFetcher


class MarketAlignment:
    """
    Aligns events with market data and computes returns.
    
    This class handles:
    - Finding market data for event dates
    - Computing returns at multiple horizons
    - Computing abnormal returns vs benchmark
    - Labeling market reactions
    
    Attributes:
        market_fetcher: Market data fetcher instance
        return_windows: Return windows to compute
        reaction_thresholds: Thresholds for labeling reactions
    """
    
    def __init__(self):
        """Initialize the market alignment."""
        config = get_config()
        
        self.market_fetcher = MarketDataFetcher()
        self.return_windows = config.pipeline.market.return_windows
        self.reaction_thresholds = config.pipeline.market.reaction_thresholds
        
        logger.info("MarketAlignment initialized")
    
    def align_event(
        self,
        event: ContradictionEvent,
        price_data: Optional[pd.DataFrame] = None
    ) -> ContradictionEvent:
        """
        Align a single event with market data.
        
        Args:
            event: ContradictionEvent to align
            price_data: Optional pre-fetched price data
            
        Returns:
            Event with market data filled in
        """
        event_date = event.event_date
        if isinstance(event_date, str):
            event_date = datetime.fromisoformat(event_date)
        
        # Fetch price data if not provided
        if price_data is None:
            start = (event_date - timedelta(days=60)).strftime("%Y-%m-%d")
            end = (event_date + timedelta(days=30)).strftime("%Y-%m-%d")
            
            price_data = self.market_fetcher.fetch_prices(
                event.ticker, start, end
            )
        
        if price_data is None or price_data.empty:
            logger.warning(f"No price data for {event.ticker} around {event_date}")
            return event
        
        # Find event date in price data
        event_date_str = event_date.strftime("%Y-%m-%d")
        
        try:
            # Convert index to string for comparison
            price_data.index = pd.to_datetime(price_data.index)
            
            # Find nearest trading day
            if event_date not in price_data.index:
                future_dates = price_data.index[price_data.index >= event_date]
                if len(future_dates) == 0:
                    return event
                nearest_date = future_dates[0]
            else:
                nearest_date = event_date
            
            row = price_data.loc[nearest_date]
            
            # Extract returns
            event.return_same_day = self._safe_float(row.get("daily_return"))
            event.return_1d = self._safe_float(row.get("return_1d"))
            event.return_3d = self._safe_float(row.get("return_3d"))
            event.return_5d = self._safe_float(row.get("return_5d"))
            event.return_10d = self._safe_float(row.get("return_10d"))
            
            # Compute abnormal return
            benchmark_data = self.market_fetcher.fetch_benchmark(
                (event_date - timedelta(days=60)).strftime("%Y-%m-%d"),
                (event_date + timedelta(days=30)).strftime("%Y-%m-%d")
            )
            
            if not benchmark_data.empty and nearest_date in benchmark_data.index:
                bench_return = benchmark_data.loc[nearest_date].get("return_1d")
                if event.return_1d is not None and bench_return is not None:
                    event.abnormal_return_1d = event.return_1d - bench_return
            
            # Label reaction
            event.reaction_label = self._label_reaction(event.return_1d)
            
        except Exception as e:
            logger.warning(f"Error aligning event {event.event_id}: {e}")
        
        return event
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            import numpy as np
            if pd.isna(value) or np.isnan(value):
                return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _label_reaction(self, return_value: Optional[float]) -> str:
        """
        Label market reaction based on return.
        
        Args:
            return_value: Return value to label
            
        Returns:
            Label string (positive, negative, neutral)
        """
        if return_value is None:
            return "neutral"
        
        pos_threshold = self.reaction_thresholds.get("positive", 0.015)
        neg_threshold = self.reaction_thresholds.get("negative", -0.015)
        
        if return_value > pos_threshold:
            return "positive"
        elif return_value < neg_threshold:
            return "negative"
        else:
            return "neutral"
    
    def align_events(
        self,
        events: List[ContradictionEvent],
        price_cache: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[ContradictionEvent]:
        """
        Align multiple events with market data.
        
        Args:
            events: List of events to align
            price_cache: Optional pre-fetched price data by ticker
            
        Returns:
            List of events with market data
        """
        if price_cache is None:
            price_cache = {}
        
        # Group events by ticker
        events_by_ticker: Dict[str, List[ContradictionEvent]] = {}
        for event in events:
            if event.ticker not in events_by_ticker:
                events_by_ticker[event.ticker] = []
            events_by_ticker[event.ticker].append(event)
        
        # Fetch and cache price data by ticker
        for ticker, ticker_events in events_by_ticker.items():
            if ticker not in price_cache:
                # Find date range needed
                dates = [e.event_date for e in ticker_events]
                min_date = min(dates) - timedelta(days=60)
                max_date = max(dates) + timedelta(days=30)
                
                price_data = self.market_fetcher.fetch_prices(
                    ticker,
                    min_date.strftime("%Y-%m-%d"),
                    max_date.strftime("%Y-%m-%d")
                )
                price_cache[ticker] = price_data
        
        # Align each event
        aligned_events = []
        for event in events:
            price_data = price_cache.get(event.ticker)
            aligned_event = self.align_event(event, price_data)
            aligned_events.append(aligned_event)
        
        # Log statistics
        labeled_count = sum(1 for e in aligned_events if e.reaction_label)
        logger.info(f"Aligned {len(aligned_events)} events, {labeled_count} with labels")
        
        return aligned_events
    
    def compute_reaction_statistics(
        self,
        events: List[ContradictionEvent]
    ) -> Dict:
        """
        Compute statistics about market reactions.
        
        Args:
            events: List of aligned events
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_events": len(events),
            "with_returns": 0,
            "positive_reactions": 0,
            "negative_reactions": 0,
            "neutral_reactions": 0,
            "avg_return_1d": None,
            "avg_abnormal_return": None,
        }
        
        returns = []
        abnormal_returns = []
        
        for event in events:
            if event.return_1d is not None:
                stats["with_returns"] += 1
                returns.append(event.return_1d)
            
            if event.abnormal_return_1d is not None:
                abnormal_returns.append(event.abnormal_return_1d)
            
            if event.reaction_label == "positive":
                stats["positive_reactions"] += 1
            elif event.reaction_label == "negative":
                stats["negative_reactions"] += 1
            else:
                stats["neutral_reactions"] += 1
        
        if returns:
            stats["avg_return_1d"] = sum(returns) / len(returns)
        
        if abnormal_returns:
            stats["avg_abnormal_return"] = sum(abnormal_returns) / len(abnormal_returns)
        
        return stats
