"""
Database Models and Connection Management

This module defines the SQLAlchemy ORM models for the Corporate Narrative Engine
and provides database connection utilities.

Tables:
    - Company: Company information (ticker, name, sector, CIK)
    - Document: Raw documents from SEC and news sources
    - Claim: Extracted claims from documents
    - ClaimLink: Links between related claims
    - ContradictionEvent: Detected contradictions between claims
    - MarketData: Daily stock price data
    - EventMarketReaction: Market reaction following contradiction events
    - ModelPrediction: Model predictions for events

Usage:
    from src.database import get_session, Company, Claim
    
    with get_session() as session:
        companies = session.query(Company).all()
"""

from datetime import date, datetime
from enum import Enum
from typing import Generator, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

from src.config import get_config

# =============================================================================
# Database Setup
# =============================================================================

Base = declarative_base()

# Engine and session factory (initialized lazily)
_engine = None
_SessionFactory = None


def get_engine():
    """
    Get or create the SQLAlchemy engine.
    
    Returns:
        SQLAlchemy Engine instance connected to PostgreSQL
    """
    global _engine
    if _engine is None:
        config = get_config()
        _engine = create_engine(
            config.database.url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine


def get_session_factory():
    """
    Get or create the session factory.
    
    Returns:
        SQLAlchemy sessionmaker instance
    """
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory


def get_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Yields:
        SQLAlchemy Session instance
        
    Example:
        with get_session() as session:
            company = session.query(Company).first()
    """
    SessionFactory = get_session_factory()
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database():
    """
    Initialize database tables.
    
    Creates all tables defined in this module if they don't exist.
    Should be called once during application setup.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)


def drop_all_tables():
    """
    Drop all tables from the database.
    
    WARNING: This will delete all data. Use with caution.
    """
    engine = get_engine()
    Base.metadata.drop_all(engine)


# =============================================================================
# Enums
# =============================================================================

class DocumentType(str, Enum):
    """Types of documents that can be ingested."""
    SEC_8K = "8-K"
    SEC_10K = "10-K"
    SEC_10Q = "10-Q"
    NEWS = "news"
    PRESS_RELEASE = "press_release"
    INTERVIEW = "interview"


class SourceType(str, Enum):
    """Source categories for documents."""
    SEC = "sec"
    NEWS = "news"
    COMPANY = "company"


class SpeakerRole(str, Enum):
    """Roles of speakers making claims."""
    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    CTO = "CTO"
    SPOKESPERSON = "spokesperson"
    OTHER = "other"
    UNKNOWN = "unknown"


class ClaimTopic(str, Enum):
    """Topic categories for claims."""
    GUIDANCE = "guidance"
    PROJECTS = "projects"
    LEGAL_REGULATORY = "legal_regulatory"
    SUPPLY_CHAIN = "supply_chain"
    CAPITAL_EXPENDITURE = "capital_expenditure"
    RISK_FACTORS = "risk_factors"
    PERSONNEL = "personnel"
    CUSTOMERS = "customers"
    PRODUCTS = "products"
    OTHER = "other"


class ClaimDirection(str, Enum):
    """Direction/sentiment of a claim."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class RelationshipType(str, Enum):
    """Relationship types between claims."""
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class ReactionLabel(str, Enum):
    """Market reaction labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SignalType(str, Enum):
    """Model signal types."""
    BEARISH = "bearish"
    BULLISH = "bullish"
    NO_ACTION = "no_action"


# =============================================================================
# Models
# =============================================================================

class Company(Base):
    """
    Company information.
    
    Stores basic information about companies being analyzed including
    their stock ticker, name, sector, and SEC CIK number.
    """
    __tablename__ = "companies"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    sector = Column(String(100), nullable=True)
    industry = Column(String(100), nullable=True)
    cik = Column(String(20), nullable=True, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    documents = relationship("Document", back_populates="company")
    claims = relationship("Claim", back_populates="company")
    market_data = relationship("MarketData", back_populates="company")
    contradiction_events = relationship("ContradictionEvent", back_populates="company")
    
    def __repr__(self):
        return f"<Company(ticker='{self.ticker}', name='{self.name}')>"


class Document(Base):
    """
    Raw document storage.
    
    Stores documents from SEC filings and news sources along with
    their metadata and extracted text content.
    """
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    
    # Document identification
    doc_type = Column(SQLEnum(DocumentType), nullable=False)
    source_type = Column(SQLEnum(SourceType), nullable=False)
    source_url = Column(Text, nullable=True)
    source_name = Column(String(255), nullable=True)
    
    # Timestamps
    filing_date = Column(DateTime, nullable=False, index=True)
    published_date = Column(DateTime, nullable=True)
    
    # Content
    title = Column(Text, nullable=True)
    section_name = Column(String(100), nullable=True)
    raw_text = Column(Text, nullable=True)
    cleaned_text = Column(Text, nullable=True)
    
    # File storage
    file_path = Column(String(500), nullable=True)
    
    # Metadata
    word_count = Column(Integer, nullable=True)
    doc_metadata = Column(JSON, nullable=True)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_error = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="documents")
    claims = relationship("Claim", back_populates="document")
    
    # Indexes
    __table_args__ = (
        Index("idx_document_company_date", "company_id", "filing_date"),
        Index("idx_document_type_date", "doc_type", "filing_date"),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, type='{self.doc_type}', date='{self.filing_date}')>"


class Claim(Base):
    """
    Extracted claim from a document.
    
    Represents a structured claim extracted from a document using NLP.
    Each claim captures what was said, by whom, and its characteristics.
    """
    __tablename__ = "claims"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    
    # Claim content
    claim_text = Column(Text, nullable=False)
    claim_text_normalized = Column(Text, nullable=True)
    
    # Speaker information
    speaker = Column(String(255), nullable=True)
    speaker_role = Column(SQLEnum(SpeakerRole), default=SpeakerRole.UNKNOWN)
    
    # Classification
    topic = Column(SQLEnum(ClaimTopic), default=ClaimTopic.OTHER)
    claim_type = Column(String(50), nullable=True)
    direction = Column(SQLEnum(ClaimDirection), default=ClaimDirection.NEUTRAL)
    
    # Confidence and metadata
    confidence = Column(Float, nullable=True)
    entities = Column(JSON, nullable=True)
    claim_metadata = Column(JSON, nullable=True)
    
    # Embedding for similarity search
    embedding = Column(JSON, nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime, nullable=False, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="claims")
    company = relationship("Company", back_populates="claims")
    
    earlier_links = relationship(
        "ClaimLink",
        foreign_keys="ClaimLink.earlier_claim_id",
        back_populates="earlier_claim"
    )
    later_links = relationship(
        "ClaimLink",
        foreign_keys="ClaimLink.later_claim_id",
        back_populates="later_claim"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_claim_company_timestamp", "company_id", "timestamp"),
        Index("idx_claim_topic", "topic"),
    )
    
    def __repr__(self):
        return f"<Claim(id={self.id}, topic='{self.topic}', timestamp='{self.timestamp}')>"


class ClaimLink(Base):
    """
    Link between related claims.
    
    Represents a connection between two claims that discuss the same
    topic or entity. Used for tracking narrative evolution over time.
    """
    __tablename__ = "claim_links"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    earlier_claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False, index=True)
    later_claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False, index=True)
    
    # Similarity metrics
    similarity_score = Column(Float, nullable=False)
    topic_match = Column(Boolean, default=False)
    entity_overlap = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    earlier_claim = relationship(
        "Claim",
        foreign_keys=[earlier_claim_id],
        back_populates="earlier_links"
    )
    later_claim = relationship(
        "Claim",
        foreign_keys=[later_claim_id],
        back_populates="later_links"
    )
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("earlier_claim_id", "later_claim_id", name="uq_claim_link"),
        Index("idx_claim_link_similarity", "similarity_score"),
    )
    
    def __repr__(self):
        return f"<ClaimLink(earlier={self.earlier_claim_id}, later={self.later_claim_id}, sim={self.similarity_score:.2f})>"


class ContradictionEvent(Base):
    """
    Detected contradiction between claims.
    
    Represents an event where a later claim contradicts an earlier claim,
    potentially signaling a narrative shift or material change.
    """
    __tablename__ = "contradiction_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    earlier_claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False, index=True)
    later_claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    
    # NLI scores
    entailment_score = Column(Float, nullable=False)
    neutral_score = Column(Float, nullable=False)
    contradiction_score = Column(Float, nullable=False, index=True)
    
    # Classification
    relation_type = Column(SQLEnum(RelationshipType), nullable=False)
    topic = Column(SQLEnum(ClaimTopic), nullable=True)
    
    # Event timing
    event_date = Column(DateTime, nullable=False, index=True)
    days_between_claims = Column(Integer, nullable=True)
    
    # Additional context
    speaker_role = Column(SQLEnum(SpeakerRole), nullable=True)
    source_types = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="contradiction_events")
    earlier_claim = relationship("Claim", foreign_keys=[earlier_claim_id])
    later_claim = relationship("Claim", foreign_keys=[later_claim_id])
    market_reaction = relationship("EventMarketReaction", back_populates="event", uselist=False)
    predictions = relationship("ModelPrediction", back_populates="event")
    
    # Indexes
    __table_args__ = (
        UniqueConstraint("earlier_claim_id", "later_claim_id", name="uq_contradiction_event"),
        Index("idx_event_company_date", "company_id", "event_date"),
        Index("idx_event_contradiction_score", "contradiction_score"),
    )
    
    def __repr__(self):
        return f"<ContradictionEvent(id={self.id}, score={self.contradiction_score:.2f}, date='{self.event_date}')>"


class MarketData(Base):
    """
    Daily market data for a company.
    
    Stores OHLCV (Open, High, Low, Close, Volume) data for each trading day.
    """
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    
    # Date
    date = Column(Date, nullable=False, index=True)
    
    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    adj_close = Column(Float, nullable=True)
    volume = Column(Float, nullable=False)
    
    # Computed returns
    daily_return = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="market_data")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("company_id", "date", name="uq_market_data_company_date"),
        Index("idx_market_data_company_date", "company_id", "date"),
    )
    
    def __repr__(self):
        return f"<MarketData(company_id={self.company_id}, date='{self.date}', close={self.close})>"


class EventMarketReaction(Base):
    """
    Market reaction following a contradiction event.
    
    Stores computed returns at various horizons following the detection
    of a contradiction event.
    """
    __tablename__ = "event_market_reactions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("contradiction_events.id"), nullable=False, unique=True)
    
    # Returns at different horizons
    return_same_day = Column(Float, nullable=True)
    return_1d = Column(Float, nullable=True)
    return_3d = Column(Float, nullable=True)
    return_5d = Column(Float, nullable=True)
    return_10d = Column(Float, nullable=True)
    
    # Abnormal returns (vs benchmark)
    abnormal_return_1d = Column(Float, nullable=True)
    abnormal_return_3d = Column(Float, nullable=True)
    abnormal_return_5d = Column(Float, nullable=True)
    
    # Market context
    market_return_1d = Column(Float, nullable=True)
    volatility_30d = Column(Float, nullable=True)
    
    # Label
    reaction_label = Column(SQLEnum(ReactionLabel), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    event = relationship("ContradictionEvent", back_populates="market_reaction")
    
    def __repr__(self):
        return f"<EventMarketReaction(event_id={self.event_id}, return_1d={self.return_1d}, label='{self.reaction_label}')>"


class ModelPrediction(Base):
    """
    Model prediction for a contradiction event.
    
    Stores the signal and confidence from the prediction model,
    along with the explanation for the prediction.
    """
    __tablename__ = "model_predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("contradiction_events.id"), nullable=False, index=True)
    
    # Model identification
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=True)
    
    # Prediction
    signal = Column(SQLEnum(SignalType), nullable=False)
    confidence = Column(Float, nullable=False)
    
    # Probabilities
    prob_positive = Column(Float, nullable=True)
    prob_negative = Column(Float, nullable=True)
    prob_neutral = Column(Float, nullable=True)
    
    # Explanation
    explanation = Column(Text, nullable=True)
    feature_contributions = Column(JSON, nullable=True)
    
    # Timestamps
    prediction_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    event = relationship("ContradictionEvent", back_populates="predictions")
    
    # Constraints
    __table_args__ = (
        Index("idx_prediction_model_date", "model_name", "prediction_date"),
    )
    
    def __repr__(self):
        return f"<ModelPrediction(event_id={self.event_id}, signal='{self.signal}', confidence={self.confidence:.2f})>"


# =============================================================================
# Utility Functions
# =============================================================================

def get_or_create_company(
    session: Session,
    ticker: str,
    name: Optional[str] = None,
    sector: Optional[str] = None,
    cik: Optional[str] = None
) -> Company:
    """
    Get an existing company or create a new one.
    
    Args:
        session: Database session
        ticker: Stock ticker symbol
        name: Company name
        sector: Industry sector
        cik: SEC CIK number
        
    Returns:
        Company instance (existing or newly created)
    """
    company = session.query(Company).filter_by(ticker=ticker).first()
    if company is None:
        company = Company(
            ticker=ticker,
            name=name or ticker,
            sector=sector,
            cik=cik
        )
        session.add(company)
        session.flush()
    return company


def get_claims_for_company(
    session: Session,
    company_id: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    topic: Optional[ClaimTopic] = None
) -> List[Claim]:
    """
    Get claims for a company within a date range.
    
    Args:
        session: Database session
        company_id: Company ID
        start_date: Start of date range
        end_date: End of date range
        topic: Optional topic filter
        
    Returns:
        List of Claim instances
    """
    query = session.query(Claim).filter(Claim.company_id == company_id)
    
    if start_date:
        query = query.filter(Claim.timestamp >= start_date)
    if end_date:
        query = query.filter(Claim.timestamp <= end_date)
    if topic:
        query = query.filter(Claim.topic == topic)
    
    return query.order_by(Claim.timestamp).all()


def get_contradiction_events(
    session: Session,
    company_id: Optional[int] = None,
    min_score: float = 0.0,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[ContradictionEvent]:
    """
    Get contradiction events with optional filters.
    
    Args:
        session: Database session
        company_id: Optional company ID filter
        min_score: Minimum contradiction score
        start_date: Start of date range
        end_date: End of date range
        
    Returns:
        List of ContradictionEvent instances
    """
    query = session.query(ContradictionEvent).filter(
        ContradictionEvent.contradiction_score >= min_score
    )
    
    if company_id:
        query = query.filter(ContradictionEvent.company_id == company_id)
    if start_date:
        query = query.filter(ContradictionEvent.event_date >= start_date)
    if end_date:
        query = query.filter(ContradictionEvent.event_date <= end_date)
    
    return query.order_by(ContradictionEvent.event_date).all()
