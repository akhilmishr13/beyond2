"""
Temporal Splitter

This module provides time-aware train/validation/test splitting
that prevents future information leakage.

CRITICAL: Always use temporal splitting for financial time series.
Never use random splits as they cause leakage.

Usage:
    from src.evaluation.temporal_split import TemporalSplitter
    
    splitter = TemporalSplitter()
    train, val, test = splitter.split(df, date_column="event_date")
"""

from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from loguru import logger

from src.config import get_config


class TemporalSplitter:
    """
    Splits data chronologically into train/val/test sets.
    
    This splitter ensures:
    - Train data comes before validation data
    - Validation data comes before test data
    - Optional gap between splits to prevent leakage
    
    Attributes:
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        gap_days: Gap between splits in trading days
    """
    
    def __init__(self):
        """Initialize the temporal splitter."""
        config = get_config()
        split_config = config.evaluation.temporal_split
        
        self.train_ratio = split_config.train_ratio
        self.val_ratio = split_config.val_ratio
        self.test_ratio = split_config.test_ratio
        self.gap_days = split_config.gap_days
        
        logger.info(
            f"TemporalSplitter initialized: {self.train_ratio}/{self.val_ratio}/{self.test_ratio}"
        )
    
    def split(
        self,
        df: pd.DataFrame,
        date_column: str = "event_date"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically.
        
        Args:
            df: DataFrame to split
            date_column: Column containing dates
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Sort by date
        df = df.sort_values(date_column).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        # Validate no overlap
        self._validate_splits(train_df, val_df, test_df, date_column)
        
        logger.info(
            f"Split complete: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
        
        return train_df, val_df, test_df
    
    def _validate_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_column: str
    ):
        """Validate that splits don't overlap temporally."""
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            return
        
        train_max = train_df[date_column].max()
        val_min = val_df[date_column].min()
        val_max = val_df[date_column].max()
        test_min = test_df[date_column].min()
        
        if train_max >= val_min:
            raise ValueError(
                f"TEMPORAL LEAKAGE: Train max date ({train_max}) >= Val min date ({val_min})"
            )
        
        if val_max >= test_min:
            raise ValueError(
                f"TEMPORAL LEAKAGE: Val max date ({val_max}) >= Test min date ({test_min})"
            )
    
    def get_split_dates(
        self,
        df: pd.DataFrame,
        date_column: str = "event_date"
    ) -> dict:
        """
        Get date boundaries for each split.
        
        Args:
            df: DataFrame
            date_column: Date column
            
        Returns:
            Dictionary with date ranges for each split
        """
        train_df, val_df, test_df = self.split(df, date_column)
        
        return {
            "train": {
                "start": train_df[date_column].min(),
                "end": train_df[date_column].max(),
                "n_samples": len(train_df)
            },
            "val": {
                "start": val_df[date_column].min(),
                "end": val_df[date_column].max(),
                "n_samples": len(val_df)
            },
            "test": {
                "start": test_df[date_column].min(),
                "end": test_df[date_column].max(),
                "n_samples": len(test_df)
            }
        }
