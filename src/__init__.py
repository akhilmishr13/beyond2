"""
Corporate Narrative Consistency Engine

An end-to-end AI system that detects contradictions in corporate disclosures
and executive statements, and measures how those contradictions correlate
with stock price movements.

Modules:
    - ingestion: Data collection from SEC, news sources, and market data
    - parsing: Document parsing and text extraction
    - claim_extraction: NLP-based claim extraction using Claude
    - claim_matching: Semantic similarity matching of claims
    - contradiction_detection: NLI-based contradiction detection
    - events: Event generation and market alignment
    - baselines: Baseline models for comparison
    - modeling: Signal prediction model
    - evaluation: Metrics, robustness, and calibration analysis
    - visualization: Plotting and dashboard utilities
"""

__version__ = "1.0.0"
__author__ = "Corporate Narrative Engine Team"
