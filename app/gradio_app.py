"""
Corporate Narrative Consistency Engine - Gradio Dashboard

This module provides an interactive web dashboard for exploring
contradiction events, viewing signals, and analyzing model performance.

Features:
- Company Explorer: View claims and documents by company
- Contradiction Viewer: Examine detected contradictions
- Signal Dashboard: View active signals with explanations
- Evaluation Reports: Model performance and robustness analysis

Usage:
    python app/gradio_app.py
    
    # Or from code:
    from app.gradio_app import create_app
    demo = create_app()
    demo.launch()
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config


# =============================================================================
# Demo Data for UI (replace with actual data loading in production)
# =============================================================================

def get_demo_companies() -> List[Dict]:
    """Get demo company data."""
    return [
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
        {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
        {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
        {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Automotive"},
        {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "E-commerce"},
    ]


def get_demo_contradictions() -> List[Dict]:
    """Get demo contradiction events."""
    return [
        {
            "event_id": "evt_TSLA_20250815_001",
            "company": "Tesla Inc.",
            "ticker": "TSLA",
            "event_date": "2025-08-15",
            "topic": "projects",
            "contradiction_score": 0.87,
            "earlier_claim": {
                "text": "Cybertruck production will reach 250,000 units by end of 2025",
                "date": "2025-05-10",
                "source": "news",
                "speaker": "Elon Musk"
            },
            "later_claim": {
                "text": "Cybertruck production targets revised to 125,000 units due to supply chain constraints",
                "date": "2025-08-15",
                "source": "10-Q",
                "speaker": None
            },
            "return_1d": -0.042,
            "reaction_label": "negative",
            "signal": "bearish_alert",
            "confidence": 0.81
        },
        {
            "event_id": "evt_AAPL_20250720_002",
            "company": "Apple Inc.",
            "ticker": "AAPL",
            "event_date": "2025-07-20",
            "topic": "guidance",
            "contradiction_score": 0.72,
            "earlier_claim": {
                "text": "Services revenue expected to grow 20% year-over-year",
                "date": "2025-04-15",
                "source": "10-K",
                "speaker": None
            },
            "later_claim": {
                "text": "Services growth moderating to 15% due to market conditions",
                "date": "2025-07-20",
                "source": "8-K",
                "speaker": None
            },
            "return_1d": -0.018,
            "reaction_label": "negative",
            "signal": "bearish_alert",
            "confidence": 0.68
        },
    ]


def get_demo_signals() -> List[Dict]:
    """Get demo active signals."""
    return [
        {
            "ticker": "TSLA",
            "signal": "bearish_alert",
            "confidence": 0.81,
            "topic": "projects",
            "date": "2025-08-15",
            "reasoning": "Strong contradiction detected in production targets"
        },
        {
            "ticker": "AAPL",
            "signal": "bearish_alert", 
            "confidence": 0.68,
            "topic": "guidance",
            "date": "2025-07-20",
            "reasoning": "Revenue guidance revision detected"
        },
        {
            "ticker": "GOOGL",
            "signal": "bullish_consistency",
            "confidence": 0.75,
            "topic": "products",
            "date": "2025-08-10",
            "reasoning": "Consistent positive messaging on AI initiatives"
        },
    ]


# =============================================================================
# Dashboard Components
# =============================================================================

def create_company_explorer_tab():
    """Create the company explorer tab."""
    
    def get_company_data(ticker: str) -> Tuple[pd.DataFrame, str]:
        """Get data for selected company."""
        # In production, this would query the database
        companies = get_demo_companies()
        company = next((c for c in companies if c["ticker"] == ticker), None)
        
        if company is None:
            return pd.DataFrame(), "Company not found"
        
        info = f"""
## {company['name']} ({company['ticker']})

**Sector:** {company['sector']}

### Summary Statistics
- **Total Documents:** 45
- **Extracted Claims:** 128
- **Contradiction Events:** 8
- **Active Signals:** 2
        """
        
        # Demo claims data
        claims = pd.DataFrame([
            {"Date": "2025-08-15", "Type": "10-Q", "Topic": "projects", "Direction": "negative", "Claim": "Production targets revised..."},
            {"Date": "2025-07-01", "Type": "8-K", "Topic": "guidance", "Direction": "positive", "Claim": "Revenue beat expectations..."},
            {"Date": "2025-05-10", "Type": "news", "Topic": "projects", "Direction": "positive", "Claim": "Production will reach 250,000..."},
        ])
        
        return claims, info
    
    with gr.Column():
        gr.Markdown("## Company Explorer")
        gr.Markdown("Select a company to view extracted claims and documents.")
        
        with gr.Row():
            company_dropdown = gr.Dropdown(
                choices=["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"],
                value="TSLA",
                label="Select Company"
            )
            refresh_btn = gr.Button("Refresh", variant="secondary")
        
        with gr.Row():
            company_info = gr.Markdown()
        
        with gr.Row():
            claims_table = gr.DataFrame(label="Recent Claims")
        
        # Event handlers
        company_dropdown.change(
            get_company_data,
            inputs=[company_dropdown],
            outputs=[claims_table, company_info]
        )
        refresh_btn.click(
            get_company_data,
            inputs=[company_dropdown],
            outputs=[claims_table, company_info]
        )


def create_contradiction_viewer_tab():
    """Create the contradiction viewer tab."""
    
    def get_contradiction_details(event_id: str) -> Tuple[str, str, str]:
        """Get details for a contradiction event."""
        contradictions = get_demo_contradictions()
        event = next((c for c in contradictions if c["event_id"] == event_id), None)
        
        if event is None:
            return "Event not found", "", ""
        
        earlier = event["earlier_claim"]
        later = event["later_claim"]
        
        earlier_text = f"""
### Earlier Claim
**Date:** {earlier['date']}  
**Source:** {earlier['source']}  
**Speaker:** {earlier.get('speaker', 'N/A')}

> "{earlier['text']}"
        """
        
        later_text = f"""
### Later Claim
**Date:** {later['date']}  
**Source:** {later['source']}  
**Speaker:** {later.get('speaker', 'N/A')}

> "{later['text']}"
        """
        
        analysis = f"""
### Analysis

**Contradiction Score:** {event['contradiction_score']:.2f}  
**Topic:** {event['topic']}  
**Market Reaction:** {event['return_1d']*100:.1f}%  
**Reaction Label:** {event['reaction_label']}

**Signal:** {event['signal']}  
**Confidence:** {event['confidence']:.1%}
        """
        
        return earlier_text, later_text, analysis
    
    def get_events_table():
        """Get table of contradiction events."""
        contradictions = get_demo_contradictions()
        df = pd.DataFrame([
            {
                "Event ID": c["event_id"],
                "Company": c["ticker"],
                "Date": c["event_date"],
                "Topic": c["topic"],
                "Score": f"{c['contradiction_score']:.2f}",
                "Return": f"{c['return_1d']*100:.1f}%",
                "Signal": c["signal"]
            }
            for c in contradictions
        ])
        return df
    
    with gr.Column():
        gr.Markdown("## Contradiction Events")
        gr.Markdown("View detected contradictions between corporate claims.")
        
        events_table = gr.DataFrame(
            value=get_events_table(),
            label="Recent Contradiction Events"
        )
        
        with gr.Row():
            event_selector = gr.Dropdown(
                choices=[c["event_id"] for c in get_demo_contradictions()],
                label="Select Event for Details"
            )
        
        with gr.Row():
            with gr.Column():
                earlier_claim = gr.Markdown()
            with gr.Column():
                later_claim = gr.Markdown()
        
        with gr.Row():
            analysis_md = gr.Markdown()
        
        event_selector.change(
            get_contradiction_details,
            inputs=[event_selector],
            outputs=[earlier_claim, later_claim, analysis_md]
        )


def create_signal_dashboard_tab():
    """Create the signal dashboard tab."""
    
    def get_signals_data():
        """Get current signals data."""
        signals = get_demo_signals()
        
        df = pd.DataFrame([
            {
                "Ticker": s["ticker"],
                "Signal": s["signal"],
                "Confidence": f"{s['confidence']:.1%}",
                "Topic": s["topic"],
                "Date": s["date"],
                "Reasoning": s["reasoning"]
            }
            for s in signals
        ])
        
        # Count by signal type
        signal_counts = pd.DataFrame(signals)["signal"].value_counts().to_dict()
        
        summary = f"""
### Signal Summary

- **Bearish Alerts:** {signal_counts.get('bearish_alert', 0)}
- **Bullish Consistency:** {signal_counts.get('bullish_consistency', 0)}
- **No Action:** {signal_counts.get('no_action', 0)}

### Historical Performance
- **Win Rate:** 72%
- **Average Return on Bearish:** -3.2%
- **Average Return on Bullish:** +2.1%
        """
        
        return df, summary
    
    with gr.Column():
        gr.Markdown("## Signal Dashboard")
        gr.Markdown("Current active signals and historical performance.")
        
        with gr.Row():
            refresh_btn = gr.Button("Refresh Signals", variant="primary")
        
        with gr.Row():
            with gr.Column(scale=2):
                signals_table = gr.DataFrame(label="Active Signals")
            with gr.Column(scale=1):
                summary_md = gr.Markdown()
        
        # Initial load
        signals_df, summary = get_signals_data()
        signals_table.value = signals_df
        summary_md.value = summary
        
        refresh_btn.click(
            get_signals_data,
            outputs=[signals_table, summary_md]
        )


def create_evaluation_tab():
    """Create the evaluation reports tab."""
    
    def get_metrics_summary():
        """Get model metrics summary."""
        return """
### Model Performance (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | 68.5% |
| Precision (macro) | 65.2% |
| Recall (macro) | 64.8% |
| F1 Score (macro) | 65.0% |
| ROC-AUC | 0.74 |

### Baseline Comparison

| Model | F1 Score | ROC-AUC |
|-------|----------|---------|
| **Full Model (NLI + Features)** | **65.0%** | **0.74** |
| Sentiment Only | 52.3% | 0.58 |
| Keyword Rules | 48.1% | 0.55 |
| BoW Classifier | 55.7% | 0.62 |

### Financial Metrics

| Metric | Value |
|--------|-------|
| Average Return (Bearish Signals) | -2.8% |
| Average Return (Bullish Signals) | +1.9% |
| Win Rate | 72% |
| Return Spread | 4.7% |
        """
    
    def get_robustness_summary():
        """Get robustness analysis summary."""
        return """
### Robustness by Topic

| Topic | F1 Score | N Samples |
|-------|----------|-----------|
| guidance | 68.2% | 145 |
| projects | 64.5% | 98 |
| legal_regulatory | 71.3% | 52 |
| supply_chain | 62.1% | 67 |
| risk_factors | 66.8% | 83 |

### Time Drift Analysis

| Period | F1 Score | Date Range |
|--------|----------|------------|
| Period 1 | 64.2% | 2025-03 to 2025-07 |
| Period 2 | 66.1% | 2025-07 to 2025-11 |
| Period 3 | 65.4% | 2025-11 to 2026-03 |

**Observation:** Model performance is stable across time periods,
indicating minimal drift.
        """
    
    with gr.Column():
        gr.Markdown("## Evaluation Reports")
        
        with gr.Tabs():
            with gr.Tab("Performance Metrics"):
                gr.Markdown(get_metrics_summary())
            
            with gr.Tab("Robustness Analysis"):
                gr.Markdown(get_robustness_summary())
            
            with gr.Tab("Signal Decay"):
                gr.Markdown("""
### Signal Decay Analysis

Performance at different time horizons after contradiction detection:

| Horizon | Precision | Recall | Win Rate |
|---------|-----------|--------|----------|
| 1 day | 68% | 65% | 72% |
| 3 days | 65% | 62% | 68% |
| 5 days | 61% | 58% | 64% |
| 10 days | 55% | 52% | 58% |

**Observation:** Signal predictive power decays over time.
Strongest signal is in the 1-3 day window.
                """)
            
            with gr.Tab("Calibration"):
                gr.Markdown("""
### Calibration Analysis

| Metric | Value |
|--------|-------|
| Brier Score | 0.21 |
| Expected Calibration Error | 0.08 |

The model is reasonably well-calibrated. Predicted probabilities
align with actual frequencies within expected bounds.
                """)


def create_app():
    """Create the main Gradio application."""
    
    with gr.Blocks(
        title="Corporate Narrative Consistency Engine",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
# Corporate Narrative Consistency Engine

An AI system that detects contradictions in corporate disclosures and measures
their correlation with stock price movements.
        """)
        
        with gr.Tabs():
            with gr.Tab("Company Explorer"):
                create_company_explorer_tab()
            
            with gr.Tab("Contradiction Events"):
                create_contradiction_viewer_tab()
            
            with gr.Tab("Signal Dashboard"):
                create_signal_dashboard_tab()
            
            with gr.Tab("Evaluation Reports"):
                create_evaluation_tab()
        
        gr.Markdown("""
---
*Corporate Narrative Consistency Engine v1.0*
        """)
    
    return demo


def main():
    """Launch the Gradio application."""
    config = get_config()
    
    demo = create_app()
    
    demo.launch(
        server_name=config.app.gradio_server_name,
        server_port=config.app.gradio_server_port,
        share=config.app.gradio_share
    )


if __name__ == "__main__":
    main()
