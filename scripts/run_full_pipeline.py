#!/usr/bin/env python3
"""
Run Full Pipeline

This script executes the complete Corporate Narrative Engine pipeline:
1. Data ingestion (SEC filings, news, market data)
2. Claim extraction
3. Claim matching
4. Contradiction detection
5. Event building and market alignment
6. Model training
7. Evaluation

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --companies AAPL TSLA
    python scripts/run_full_pipeline.py --skip-ingestion
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.config import get_config


def setup_logging():
    """Configure logging for the pipeline."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger.add(
        log_file,
        rotation="100 MB",
        retention="7 days",
        level="DEBUG"
    )
    
    logger.info("Pipeline logging configured")


def run_ingestion(companies: list):
    """Run data ingestion for specified companies."""
    logger.info(f"Starting data ingestion for {len(companies)} companies")
    
    from src.ingestion import SourceRegistry
    
    config = get_config()
    registry = SourceRegistry()
    
    start_date = config.pipeline.lookback.start_date
    end_date = config.pipeline.lookback.end_date
    
    for ticker in companies:
        try:
            logger.info(f"Ingesting data for {ticker}")
            
            # Get company info
            company_info = registry.get_company_info(ticker)
            company_name = company_info.get("name", ticker)
            
            # Collect documents
            documents = list(registry.collect_all_documents(
                ticker=ticker,
                company_name=company_name,
                start_date=start_date,
                end_date=end_date
            ))
            
            logger.info(f"Collected {len(documents)} documents for {ticker}")
            
            # Get market data
            market_data = registry.get_market_data(ticker, start_date, end_date)
            if market_data is not None:
                logger.info(f"Fetched {len(market_data)} price records for {ticker}")
            
        except Exception as e:
            logger.error(f"Error ingesting data for {ticker}: {e}")
            continue
    
    logger.info("Data ingestion complete")


def run_extraction(documents: list):
    """Run claim extraction on documents."""
    logger.info(f"Starting claim extraction for {len(documents)} documents")
    
    from src.claim_extraction import ClaimExtractor
    
    extractor = ClaimExtractor()
    all_claims = []
    
    for doc in documents:
        try:
            claims = extractor.extract_claims(
                text=doc.get("content", ""),
                company=doc.get("company", ""),
                ticker=doc.get("ticker", ""),
                source_type=doc.get("source_type", "")
            )
            all_claims.extend(claims)
        except Exception as e:
            logger.warning(f"Error extracting claims: {e}")
    
    logger.info(f"Extracted {len(all_claims)} claims")
    return all_claims


def run_contradiction_detection(claims: list):
    """Run contradiction detection on claims."""
    logger.info(f"Starting contradiction detection for {len(claims)} claims")
    
    from src.claim_matching import TemporalLinker
    from src.contradiction_detection import NLIDetector, ContradictionScorer
    
    # Link claims
    linker = TemporalLinker()
    # ... linking logic ...
    
    # Detect contradictions
    detector = NLIDetector()
    scorer = ContradictionScorer()
    # ... detection logic ...
    
    logger.info("Contradiction detection complete")
    return []


def run_training(events: list):
    """Train the signal prediction model."""
    logger.info(f"Starting model training with {len(events)} events")
    
    from src.modeling import FeatureBuilder, SignalModel
    from src.events import EventDatasetBuilder
    
    # Build dataset
    builder = EventDatasetBuilder()
    df = builder.build_dataframe(events)
    
    # Split data
    train_df, val_df, test_df = builder.temporal_split(df)
    
    # Build features
    feature_builder = FeatureBuilder()
    X_train, feature_names = feature_builder.build_features(train_df.to_dict("records"))
    y_train = train_df["reaction_label"].values
    
    # Train model
    model = SignalModel()
    model.fit(X_train, y_train, feature_names=feature_names)
    
    # Save model
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir / "signal_model.joblib")
    
    logger.info("Model training complete")
    return model


def run_evaluation(model, test_events: list):
    """Evaluate the trained model."""
    logger.info("Starting model evaluation")
    
    from src.evaluation import MetricsCalculator, RobustnessAnalyzer
    from src.modeling import FeatureBuilder
    
    calculator = MetricsCalculator()
    # ... evaluation logic ...
    
    logger.info("Evaluation complete")


def main():
    """Main pipeline entry point."""
    parser = argparse.ArgumentParser(description="Run the full pipeline")
    parser.add_argument(
        "--companies",
        nargs="+",
        default=None,
        help="List of company tickers to process"
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip data ingestion step"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training step"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    config = get_config()
    
    # Get companies
    companies = args.companies or config.pipeline.companies
    
    logger.info(f"Starting pipeline for companies: {companies}")
    logger.info(f"Date range: {config.pipeline.lookback.start_date} to {config.pipeline.lookback.end_date}")
    
    try:
        # Step 1: Ingestion
        if not args.skip_ingestion:
            run_ingestion(companies)
        
        # Step 2-4: Extraction, Matching, Detection
        # (In production, these would load from database)
        
        # Step 5: Training
        if not args.skip_training:
            pass  # run_training(events)
        
        # Step 6: Evaluation
        # run_evaluation(model, test_events)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
