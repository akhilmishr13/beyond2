"""
Events Module

This module handles the generation and management of contradiction events,
including alignment with market data and reaction labeling.

Event Pipeline:
1. Build contradiction events from linked claims
2. Align events with market price data
3. Compute returns at various horizons
4. Label market reactions (positive/negative/neutral)
5. Create analysis-ready event datasets

Example:
    from src.events import EventBuilder, MarketAlignment
    
    builder = EventBuilder()
    events = builder.build_events(contradictions)
    
    aligner = MarketAlignment()
    events_with_returns = aligner.align_events(events, market_data)
"""

from src.events.event_builder import EventBuilder, ContradictionEvent
from src.events.market_alignment import MarketAlignment
from src.events.event_dataset import EventDatasetBuilder

__all__ = [
    "EventBuilder",
    "ContradictionEvent", 
    "MarketAlignment",
    "EventDatasetBuilder",
]
