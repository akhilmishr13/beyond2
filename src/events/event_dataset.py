"""
Event Dataset Builder

This module builds analysis-ready datasets from contradiction events,
suitable for model training and evaluation.

Usage:
    from src.events.event_dataset import EventDatasetBuilder
    
    builder = EventDatasetBuilder()
    df = builder.build_dataframe(events)
    train, val, test = builder.temporal_split(df)
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from src.config import get_config
from src.events.event_builder import ContradictionEvent


class EventDatasetBuilder:
    """
    Builds analysis datasets from events.
    
    This class handles:
    - Converting events to DataFrames
    - Feature engineering
    - Temporal train/val/test splitting
    - Data validation
    
    Attributes:
        train_ratio: Proportion of data for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
    """
    
    def __init__(self):
        """Initialize the dataset builder."""
        config = get_config()
        
        split_config = config.evaluation.temporal_split
        self.train_ratio = split_config.train_ratio
        self.val_ratio = split_config.val_ratio
        self.test_ratio = split_config.test_ratio
        
        logger.info(
            f"EventDatasetBuilder initialized with split: "
            f"{self.train_ratio}/{self.val_ratio}/{self.test_ratio}"
        )
    
    def build_dataframe(
        self,
        events: List[ContradictionEvent]
    ) -> pd.DataFrame:
        """
        Convert events to a pandas DataFrame.
        
        Args:
            events: List of ContradictionEvent objects
            
        Returns:
            DataFrame with event data
        """
        records = []
        
        for event in events:
            record = {
                "event_id": event.event_id,
                "company": event.company,
                "ticker": event.ticker,
                "event_date": event.event_date,
                "topic": event.topic,
                "contradiction_score": event.contradiction_score,
                "relationship": event.relationship,
                "speaker_role": event.speaker_role,
                "days_between": event.days_between,
                "earlier_claim_text": event.earlier_claim.get("text", ""),
                "earlier_claim_date": event.earlier_claim.get("date"),
                "earlier_claim_source": event.earlier_claim.get("source"),
                "earlier_claim_direction": event.earlier_claim.get("direction"),
                "later_claim_text": event.later_claim.get("text", ""),
                "later_claim_date": event.later_claim.get("date"),
                "later_claim_source": event.later_claim.get("source"),
                "later_claim_direction": event.later_claim.get("direction"),
                "return_same_day": event.return_same_day,
                "return_1d": event.return_1d,
                "return_3d": event.return_3d,
                "return_5d": event.return_5d,
                "return_10d": event.return_10d,
                "abnormal_return_1d": event.abnormal_return_1d,
                "reaction_label": event.reaction_label,
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Ensure date columns are datetime
        df["event_date"] = pd.to_datetime(df["event_date"])
        
        # Sort by date
        df = df.sort_values("event_date").reset_index(drop=True)
        
        logger.info(f"Built DataFrame with {len(df)} events")
        return df
    
    def temporal_split(
        self,
        df: pd.DataFrame,
        date_column: str = "event_date"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically into train/val/test.
        
        CRITICAL: This uses temporal splitting to prevent leakage.
        Never use random splitting for time series data.
        
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
        
        # Log split info
        logger.info(
            f"Temporal split: train={len(train_df)} ({train_df[date_column].min()} to {train_df[date_column].max()}), "
            f"val={len(val_df)}, test={len(test_df)}"
        )
        
        return train_df, val_df, test_df
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Topic encoding
        topic_dummies = pd.get_dummies(df["topic"], prefix="topic")
        df = pd.concat([df, topic_dummies], axis=1)
        
        # Speaker role encoding
        df["speaker_is_ceo"] = (df["speaker_role"] == "CEO").astype(int)
        df["speaker_is_cfo"] = (df["speaker_role"] == "CFO").astype(int)
        
        # Source type features
        df["source_is_sec"] = df["later_claim_source"].str.contains("sec|8-k|10-k|10-q", case=False, na=False).astype(int)
        df["source_is_news"] = df["later_claim_source"].str.contains("news", case=False, na=False).astype(int)
        
        # Direction features
        direction_map = {"positive": 1, "neutral": 0, "negative": -1}
        df["earlier_direction_num"] = df["earlier_claim_direction"].map(direction_map).fillna(0)
        df["later_direction_num"] = df["later_claim_direction"].map(direction_map).fillna(0)
        df["direction_change"] = df["later_direction_num"] - df["earlier_direction_num"]
        
        # Time features
        df["days_between_normalized"] = df["days_between"] / df["days_between"].max()
        
        # Score bins
        df["score_bin"] = pd.cut(
            df["contradiction_score"],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=["low", "medium", "high", "critical"]
        )
        
        return df
    
    def validate_temporal_integrity(self, df: pd.DataFrame) -> bool:
        """
        Validate that temporal constraints are maintained.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check that earlier claims are before later claims
        for idx, row in df.iterrows():
            earlier_date = row.get("earlier_claim_date")
            later_date = row.get("later_claim_date")
            
            if earlier_date and later_date:
                if pd.to_datetime(earlier_date) >= pd.to_datetime(later_date):
                    raise ValueError(
                        f"Temporal violation at index {idx}: "
                        f"earlier_date ({earlier_date}) >= later_date ({later_date})"
                    )
        
        logger.info("Temporal integrity validated")
        return True
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute dataset statistics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_events": len(df),
            "unique_companies": df["ticker"].nunique(),
            "date_range": {
                "start": df["event_date"].min().isoformat() if len(df) > 0 else None,
                "end": df["event_date"].max().isoformat() if len(df) > 0 else None,
            },
            "topic_distribution": df["topic"].value_counts().to_dict(),
            "reaction_distribution": df["reaction_label"].value_counts().to_dict() if "reaction_label" in df else {},
            "avg_contradiction_score": df["contradiction_score"].mean(),
            "score_distribution": {
                "min": df["contradiction_score"].min(),
                "max": df["contradiction_score"].max(),
                "mean": df["contradiction_score"].mean(),
                "std": df["contradiction_score"].std(),
            },
            "return_statistics": {
                "return_1d_mean": df["return_1d"].mean() if "return_1d" in df else None,
                "return_1d_std": df["return_1d"].std() if "return_1d" in df else None,
                "abnormal_return_mean": df["abnormal_return_1d"].mean() if "abnormal_return_1d" in df else None,
            },
        }
        
        return stats
    
    def save_dataset(
        self,
        df: pd.DataFrame,
        filepath: str,
        format: str = "parquet"
    ):
        """
        Save dataset to file.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
            format: Output format (parquet, csv)
        """
        if format == "parquet":
            df.to_parquet(filepath, index=False)
        elif format == "csv":
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved dataset to {filepath}")
    
    def load_dataset(
        self,
        filepath: str,
        format: str = "parquet"
    ) -> pd.DataFrame:
        """
        Load dataset from file.
        
        Args:
            filepath: Input file path
            format: File format
            
        Returns:
            Loaded DataFrame
        """
        if format == "parquet":
            df = pd.read_parquet(filepath)
        elif format == "csv":
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Ensure date columns are datetime
        if "event_date" in df.columns:
            df["event_date"] = pd.to_datetime(df["event_date"])
        
        return df
