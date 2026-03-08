"""
Event Builder

This module builds contradiction events from detected claim contradictions.
Each event represents a significant narrative shift that may impact
market sentiment.

Usage:
    from src.events.event_builder import EventBuilder
    
    builder = EventBuilder()
    events = builder.build_events(
        contradictions=contradiction_results,
        claims=all_claims,
        min_score=0.6
    )
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from src.config import get_config


@dataclass
class ContradictionEvent:
    """
    A detected contradiction event.
    
    Represents a significant narrative shift where a later claim
    contradicts an earlier claim from the same company.
    
    Attributes:
        event_id: Unique event identifier
        company: Company name
        ticker: Stock ticker
        event_date: Date of the later (contradicting) claim
        topic: Topic category
        earlier_claim: Earlier claim details
        later_claim: Later claim details
        contradiction_score: NLI contradiction probability
        relationship: Type of relationship (contradiction, revision, etc.)
        speaker_role: Role of speaker making later claim
        source_types: Source types involved (SEC, news)
        days_between: Days between claims
    """
    event_id: str
    company: str
    ticker: str
    event_date: datetime
    topic: str
    earlier_claim: Dict[str, Any]
    later_claim: Dict[str, Any]
    contradiction_score: float
    relationship: str = "contradiction"
    speaker_role: Optional[str] = None
    source_types: List[str] = field(default_factory=list)
    days_between: int = 0
    
    # Market data (filled in by MarketAlignment)
    return_same_day: Optional[float] = None
    return_1d: Optional[float] = None
    return_3d: Optional[float] = None
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    abnormal_return_1d: Optional[float] = None
    reaction_label: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "company": self.company,
            "ticker": self.ticker,
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "topic": self.topic,
            "earlier_claim": self.earlier_claim,
            "later_claim": self.later_claim,
            "contradiction_score": self.contradiction_score,
            "relationship": self.relationship,
            "speaker_role": self.speaker_role,
            "source_types": self.source_types,
            "days_between": self.days_between,
            "return_same_day": self.return_same_day,
            "return_1d": self.return_1d,
            "return_3d": self.return_3d,
            "return_5d": self.return_5d,
            "return_10d": self.return_10d,
            "abnormal_return_1d": self.abnormal_return_1d,
            "reaction_label": self.reaction_label,
        }


class EventBuilder:
    """
    Builds contradiction events from detection results.
    
    This class handles:
    - Converting contradiction detections to events
    - Filtering by minimum score
    - Deduplicating similar events
    - Generating unique event IDs
    
    Attributes:
        min_score: Minimum contradiction score to create event
        min_days_between: Minimum days between claims
    """
    
    def __init__(self):
        """Initialize the event builder."""
        config = get_config()
        
        self.min_score = config.pipeline.contradiction.thresholds.get("low", 0.3)
        self.min_days_between = 1
        
        self._event_counter = 0
        
        logger.info(f"EventBuilder initialized with min_score: {self.min_score}")
    
    def _generate_event_id(self, ticker: str, date: datetime) -> str:
        """Generate unique event ID."""
        self._event_counter += 1
        date_str = date.strftime("%Y%m%d")
        return f"evt_{ticker}_{date_str}_{self._event_counter:05d}"
    
    def build_event(
        self,
        earlier_claim: Dict,
        later_claim: Dict,
        contradiction_score: float,
        relationship: str = "contradiction"
    ) -> Optional[ContradictionEvent]:
        """
        Build a single contradiction event.
        
        Args:
            earlier_claim: Earlier claim data
            later_claim: Later claim data
            contradiction_score: Contradiction probability
            relationship: Type of relationship
            
        Returns:
            ContradictionEvent or None if below threshold
        """
        if contradiction_score < self.min_score:
            return None
        
        # Extract common fields
        ticker = later_claim.get("ticker", "UNKNOWN")
        company = later_claim.get("company", ticker)
        event_date = later_claim.get("timestamp")
        
        if isinstance(event_date, str):
            event_date = datetime.fromisoformat(event_date)
        
        topic = later_claim.get("topic", "other")
        
        # Calculate days between
        earlier_date = earlier_claim.get("timestamp")
        if isinstance(earlier_date, str):
            earlier_date = datetime.fromisoformat(earlier_date)
        
        days_between = (event_date - earlier_date).days if earlier_date and event_date else 0
        
        if days_between < self.min_days_between:
            return None
        
        # Get source types
        source_types = []
        if earlier_claim.get("source_type"):
            source_types.append(earlier_claim["source_type"])
        if later_claim.get("source_type"):
            source_types.append(later_claim["source_type"])
        
        # Create event
        event = ContradictionEvent(
            event_id=self._generate_event_id(ticker, event_date),
            company=company,
            ticker=ticker,
            event_date=event_date,
            topic=topic,
            earlier_claim={
                "text": earlier_claim.get("claim_text", ""),
                "date": earlier_date.isoformat() if earlier_date else None,
                "source": earlier_claim.get("source_type", ""),
                "speaker": earlier_claim.get("speaker"),
                "direction": earlier_claim.get("direction"),
            },
            later_claim={
                "text": later_claim.get("claim_text", ""),
                "date": event_date.isoformat() if event_date else None,
                "source": later_claim.get("source_type", ""),
                "speaker": later_claim.get("speaker"),
                "direction": later_claim.get("direction"),
            },
            contradiction_score=contradiction_score,
            relationship=relationship,
            speaker_role=later_claim.get("speaker_role"),
            source_types=list(set(source_types)),
            days_between=days_between
        )
        
        return event
    
    def build_events(
        self,
        contradictions: List[Dict],
        min_score: Optional[float] = None
    ) -> List[ContradictionEvent]:
        """
        Build events from a list of contradiction results.
        
        Args:
            contradictions: List of contradiction detection results
            min_score: Override minimum score threshold
            
        Returns:
            List of ContradictionEvent objects
        """
        min_score = min_score or self.min_score
        events = []
        
        for contradiction in contradictions:
            earlier = contradiction.get("earlier_claim", {})
            later = contradiction.get("later_claim", {})
            score = contradiction.get("contradiction_score", 0)
            relationship = contradiction.get("relationship", "contradiction")
            
            event = self.build_event(earlier, later, score, relationship)
            if event:
                events.append(event)
        
        # Sort by date
        events.sort(key=lambda e: e.event_date)
        
        logger.info(f"Built {len(events)} events from {len(contradictions)} contradictions")
        return events
    
    def filter_events(
        self,
        events: List[ContradictionEvent],
        min_score: Optional[float] = None,
        topics: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[ContradictionEvent]:
        """
        Filter events by various criteria.
        
        Args:
            events: List of events to filter
            min_score: Minimum contradiction score
            topics: List of topics to include
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Filtered list of events
        """
        filtered = events
        
        if min_score:
            filtered = [e for e in filtered if e.contradiction_score >= min_score]
        
        if topics:
            filtered = [e for e in filtered if e.topic in topics]
        
        if start_date:
            filtered = [e for e in filtered if e.event_date >= start_date]
        
        if end_date:
            filtered = [e for e in filtered if e.event_date <= end_date]
        
        return filtered
    
    def deduplicate_events(
        self,
        events: List[ContradictionEvent],
        time_window_days: int = 3
    ) -> List[ContradictionEvent]:
        """
        Remove duplicate events that are too close in time.
        
        If multiple events for the same company/topic occur within
        the time window, keep only the highest scoring one.
        
        Args:
            events: List of events
            time_window_days: Window for considering duplicates
            
        Returns:
            Deduplicated list of events
        """
        if not events:
            return []
        
        # Sort by date and score
        sorted_events = sorted(
            events,
            key=lambda e: (e.ticker, e.topic, e.event_date)
        )
        
        kept = []
        
        for event in sorted_events:
            # Check if similar event already kept
            is_duplicate = False
            
            for kept_event in kept:
                if (kept_event.ticker == event.ticker and
                    kept_event.topic == event.topic):
                    
                    days_diff = abs((event.event_date - kept_event.event_date).days)
                    
                    if days_diff <= time_window_days:
                        # Keep the higher scoring one
                        if event.contradiction_score > kept_event.contradiction_score:
                            kept.remove(kept_event)
                            kept.append(event)
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                kept.append(event)
        
        return kept
