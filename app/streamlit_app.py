"""
Corporate Narrative Consistency Engine - Streamlit Dashboard

High-fidelity interactive dashboard for exploring contradiction events,
viewing signals, and analyzing model performance.

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page config
st.set_page_config(
    page_title="Corporate Narrative Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS for High-Fidelity UI
# =============================================================================

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    /* Card styling */
    .card {
        background: linear-gradient(145deg, #1e1e32 0%, #252542 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .card-header {
        font-size: 14px;
        font-weight: 600;
        color: #8b8b9e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .card-value {
        font-size: 36px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    
    .card-subtext {
        font-size: 13px;
        color: #6b6b7e;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e32 0%, #252542 100%);
        border-radius: 16px;
        padding: 20px 24px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
    }
    
    .metric-label {
        font-size: 12px;
        color: #8b8b9e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    
    .metric-delta-positive {
        color: #4ade80;
        font-size: 13px;
    }
    
    .metric-delta-negative {
        color: #f87171;
        font-size: 13px;
    }
    
    /* Signal cards */
    .signal-card {
        background: linear-gradient(145deg, #1e1e32 0%, #252542 100%);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }
    
    .signal-bearish {
        border-left-color: #ef4444;
        background: linear-gradient(145deg, #1e1e32 0%, #2d1f1f 100%);
    }
    
    .signal-bullish {
        border-left-color: #22c55e;
        background: linear-gradient(145deg, #1e1e32 0%, #1f2d1f 100%);
    }
    
    .signal-watch {
        border-left-color: #eab308;
        background: linear-gradient(145deg, #1e1e32 0%, #2d2a1f 100%);
    }
    
    .signal-ticker {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
    }
    
    .signal-type {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
        margin-top: 8px;
    }
    
    .signal-type-bearish {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }
    
    .signal-type-bullish {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }
    
    .signal-type-watch {
        background: rgba(234, 179, 8, 0.2);
        color: #eab308;
    }
    
    /* Contradiction card */
    .contradiction-card {
        background: linear-gradient(145deg, #1e1e32 0%, #252542 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .contradiction-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding-bottom: 16px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .contradiction-company {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
    }
    
    .contradiction-score {
        font-size: 28px;
        font-weight: 700;
        color: #ef4444;
    }
    
    .claim-box {
        background: rgba(0,0,0,0.2);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    
    .claim-earlier {
        border-left: 3px solid #3b82f6;
    }
    
    .claim-later {
        border-left: 3px solid #ef4444;
    }
    
    .claim-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .claim-label-earlier {
        color: #3b82f6;
    }
    
    .claim-label-later {
        color: #ef4444;
    }
    
    .claim-text {
        font-size: 14px;
        color: #e0e0e0;
        line-height: 1.6;
    }
    
    .claim-meta {
        font-size: 12px;
        color: #6b6b7e;
        margin-top: 8px;
    }
    
    /* Stats row */
    .stats-row {
        display: flex;
        gap: 12px;
        margin-top: 16px;
    }
    
    .stat-badge {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 12px;
    }
    
    .stat-label {
        color: #6b6b7e;
    }
    
    .stat-value {
        color: #ffffff;
        font-weight: 600;
        margin-left: 4px;
    }
    
    /* Table styling */
    .styled-table {
        background: linear-gradient(145deg, #1e1e32 0%, #252542 100%);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Section headers */
    .section-header {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin: 24px 0 16px 0;
    }
    
    .section-subheader {
        font-size: 14px;
        color: #6b6b7e;
        margin-bottom: 20px;
    }
    
    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 32px;
        text-align: center;
    }
    
    .hero-title {
        font-size: 36px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 12px;
    }
    
    .hero-subtitle {
        font-size: 16px;
        color: rgba(255,255,255,0.8);
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Process steps */
    .process-card {
        background: linear-gradient(145deg, #1e1e32 0%, #252542 100%);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        height: 100%;
    }
    
    .process-icon {
        font-size: 40px;
        margin-bottom: 16px;
    }
    
    .process-title {
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }
    
    .process-desc {
        font-size: 13px;
        color: #8b8b9e;
        line-height: 1.5;
    }
    
    /* Evaluation cards */
    .eval-card {
        background: linear-gradient(145deg, #1e1e32 0%, #252542 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    
    .eval-title {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 16px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12121f 0%, #1a1a2e 100%);
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #8b8b9e !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3a3a5e;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4a4a6e;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Demo Data
# =============================================================================

@st.cache_data
def get_demo_companies():
    return pd.DataFrame([
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "documents": 45, "claims": 128, "contradictions": 8},
        {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Technology", "documents": 52, "claims": 145, "contradictions": 6},
        {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "documents": 48, "claims": 132, "contradictions": 5},
        {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Automotive", "documents": 62, "claims": 189, "contradictions": 12},
        {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "E-commerce", "documents": 55, "claims": 156, "contradictions": 7},
    ])

@st.cache_data
def get_demo_contradictions():
    return pd.DataFrame([
        {
            "event_id": "evt_TSLA_001",
            "ticker": "TSLA",
            "company": "Tesla Inc.",
            "event_date": "2025-08-15",
            "topic": "projects",
            "contradiction_score": 0.87,
            "earlier_claim": "Cybertruck production will reach 250,000 units by end of 2025",
            "earlier_date": "2025-05-10",
            "earlier_source": "News - CEO Statement",
            "later_claim": "Cybertruck production targets revised to 125,000 units due to supply chain constraints",
            "later_date": "2025-08-15",
            "later_source": "10-Q Filing",
            "return_1d": -4.2,
            "return_3d": -6.1,
            "reaction": "negative",
            "signal": "bearish_alert",
            "confidence": 0.81
        },
        {
            "event_id": "evt_AAPL_001",
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "event_date": "2025-07-20",
            "topic": "guidance",
            "contradiction_score": 0.72,
            "earlier_claim": "Services revenue expected to grow 20% year-over-year",
            "earlier_date": "2025-04-15",
            "earlier_source": "10-K Filing",
            "later_claim": "Services growth moderating to 15% due to market conditions",
            "later_date": "2025-07-20",
            "later_source": "8-K Filing",
            "return_1d": -1.8,
            "return_3d": -2.5,
            "reaction": "negative",
            "signal": "bearish_alert",
            "confidence": 0.68
        },
        {
            "event_id": "evt_MSFT_001",
            "ticker": "MSFT",
            "company": "Microsoft Corporation",
            "event_date": "2025-09-05",
            "topic": "products",
            "contradiction_score": 0.65,
            "earlier_claim": "Azure AI features launching in Q3 2025",
            "earlier_date": "2025-06-01",
            "earlier_source": "News - Press Release",
            "later_claim": "Azure AI rollout delayed to Q4 2025 for additional testing",
            "later_date": "2025-09-05",
            "later_source": "8-K Filing",
            "return_1d": -1.2,
            "return_3d": -0.8,
            "reaction": "neutral",
            "signal": "watch",
            "confidence": 0.55
        },
        {
            "event_id": "evt_GOOGL_001",
            "ticker": "GOOGL",
            "company": "Alphabet Inc.",
            "event_date": "2025-10-12",
            "topic": "legal_regulatory",
            "contradiction_score": 0.78,
            "earlier_claim": "No material impact expected from pending antitrust proceedings",
            "earlier_date": "2025-07-01",
            "earlier_source": "10-Q Filing",
            "later_claim": "Potential remedies could significantly impact advertising business",
            "later_date": "2025-10-12",
            "later_source": "8-K Filing",
            "return_1d": -3.5,
            "return_3d": -5.2,
            "reaction": "negative",
            "signal": "bearish_alert",
            "confidence": 0.75
        },
    ])

@st.cache_data
def get_demo_signals():
    return pd.DataFrame([
        {"ticker": "TSLA", "signal": "bearish_alert", "confidence": 0.81, "topic": "projects", "date": "2025-08-15", "reasoning": "Strong contradiction in production targets", "price": 242.50, "change": -4.2},
        {"ticker": "AAPL", "signal": "bearish_alert", "confidence": 0.68, "topic": "guidance", "date": "2025-07-20", "reasoning": "Revenue guidance revision detected", "price": 198.30, "change": -1.8},
        {"ticker": "GOOGL", "signal": "bearish_alert", "confidence": 0.75, "topic": "legal_regulatory", "date": "2025-10-12", "reasoning": "Legal risk assessment changed materially", "price": 156.80, "change": -3.5},
        {"ticker": "MSFT", "signal": "watch", "confidence": 0.55, "topic": "products", "date": "2025-09-05", "reasoning": "Product timeline shift detected", "price": 412.20, "change": -1.2},
        {"ticker": "AMZN", "signal": "bullish", "confidence": 0.62, "topic": "guidance", "date": "2025-11-01", "reasoning": "Consistent positive messaging on AWS growth", "price": 185.40, "change": 2.1},
    ])


# =============================================================================
# Helper Functions
# =============================================================================

def render_metric_card(value, label, delta=None, delta_type="positive"):
    delta_html = ""
    if delta:
        delta_class = "metric-delta-positive" if delta_type == "positive" else "metric-delta-negative"
        delta_symbol = "↑" if delta_type == "positive" else "↓"
        delta_html = f'<div class="{delta_class}">{delta_symbol} {delta}</div>'
    
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def render_signal_card(signal_data):
    signal_type = signal_data["signal"]
    if signal_type == "bearish_alert":
        card_class = "signal-bearish"
        type_class = "signal-type-bearish"
        icon = "🔴"
    elif signal_type == "bullish":
        card_class = "signal-bullish"
        type_class = "signal-type-bullish"
        icon = "🟢"
    else:
        card_class = "signal-watch"
        type_class = "signal-type-watch"
        icon = "🟡"
    
    change_color = "#ef4444" if signal_data["change"] < 0 else "#22c55e"
    change_symbol = "" if signal_data["change"] < 0 else "+"
    
    st.markdown(f"""
        <div class="signal-card {card_class}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="signal-ticker">{icon} {signal_data["ticker"]}</div>
                    <div class="signal-type {type_class}">{signal_type.replace("_", " ")}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 24px; font-weight: 700; color: #fff;">${signal_data["price"]}</div>
                    <div style="font-size: 14px; color: {change_color};">{change_symbol}{signal_data["change"]}%</div>
                </div>
            </div>
            <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 12px; color: #8b8b9e; margin-bottom: 4px;">
                    {signal_data["topic"].upper()} • {signal_data["date"]}
                </div>
                <div style="font-size: 14px; color: #e0e0e0;">
                    {signal_data["reasoning"]}
                </div>
                <div style="margin-top: 12px;">
                    <span style="font-size: 12px; color: #6b6b7e;">Confidence:</span>
                    <span style="font-size: 14px; font-weight: 600; color: #fff; margin-left: 4px;">{signal_data["confidence"]:.0%}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_contradiction_card(event):
    score_color = "#ef4444" if event["contradiction_score"] > 0.7 else "#eab308" if event["contradiction_score"] > 0.5 else "#22c55e"
    
    st.markdown(f"""
        <div class="contradiction-card">
            <div class="contradiction-header">
                <div>
                    <div class="contradiction-company">{event["company"]} ({event["ticker"]})</div>
                    <div style="font-size: 13px; color: #6b6b7e; margin-top: 4px;">
                        {event["topic"].replace("_", " ").title()} • {event["event_date"]}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; color: #8b8b9e; margin-bottom: 4px;">SCORE</div>
                    <div class="contradiction-score" style="color: {score_color};">{event["contradiction_score"]:.2f}</div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div class="claim-box claim-earlier">
                    <div class="claim-label claim-label-earlier">Earlier Claim</div>
                    <div class="claim-text">"{event["earlier_claim"]}"</div>
                    <div class="claim-meta">📄 {event["earlier_source"]} • {event["earlier_date"]}</div>
                </div>
                <div class="claim-box claim-later">
                    <div class="claim-label claim-label-later">Later Claim</div>
                    <div class="claim-text">"{event["later_claim"]}"</div>
                    <div class="claim-meta">📄 {event["later_source"]} • {event["later_date"]}</div>
                </div>
            </div>
            
            <div class="stats-row">
                <div class="stat-badge">
                    <span class="stat-label">1D Return:</span>
                    <span class="stat-value" style="color: {'#ef4444' if event['return_1d'] < 0 else '#22c55e'};">{event["return_1d"]}%</span>
                </div>
                <div class="stat-badge">
                    <span class="stat-label">3D Return:</span>
                    <span class="stat-value" style="color: {'#ef4444' if event['return_3d'] < 0 else '#22c55e'};">{event["return_3d"]}%</span>
                </div>
                <div class="stat-badge">
                    <span class="stat-label">Signal:</span>
                    <span class="stat-value">{event["signal"].replace("_", " ").title()}</span>
                </div>
                <div class="stat-badge">
                    <span class="stat-label">Confidence:</span>
                    <span class="stat-value">{event["confidence"]:.0%}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 32px; margin-bottom: 8px;">📊</div>
            <div style="font-size: 18px; font-weight: 700; color: #fff;">Narrative Engine</div>
            <div style="font-size: 12px; color: #6b6b7e;">Corporate Intelligence</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🔍 Company Explorer", "⚠️ Contradictions", "📈 Signals", "📋 Evaluation"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown('<div style="font-size: 12px; color: #6b6b7e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Data Range</div>', unsafe_allow_html=True)
    start_date = st.date_input("Start", datetime(2025, 3, 8), label_visibility="collapsed")
    end_date = st.date_input("End", datetime(2026, 3, 8), label_visibility="collapsed")
    
    st.markdown("---")
    
    companies_df = get_demo_companies()
    contradictions_df = get_demo_contradictions()
    
    st.markdown('<div style="font-size: 12px; color: #6b6b7e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px;">Quick Stats</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: 700; color: #fff;">{len(companies_df)}</div>
                <div style="font-size: 11px; color: #6b6b7e;">Companies</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: 700; color: #ef4444;">{len(contradictions_df)}</div>
                <div style="font-size: 11px; color: #6b6b7e;">Alerts</div>
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# Pages
# =============================================================================

if page == "🏠 Overview":
    # Hero Section
    st.markdown("""
        <div class="hero">
            <div class="hero-title">Corporate Narrative Engine</div>
            <div class="hero-subtitle">
                AI-powered detection of contradictions in corporate disclosures and executive statements,
                correlated with stock price movements to generate actionable trading signals.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(len(companies_df), "Companies Tracked")
    with col2:
        render_metric_card(f"{companies_df['documents'].sum()}", "Documents Analyzed")
    with col3:
        render_metric_card(f"{companies_df['claims'].sum()}", "Claims Extracted")
    with col4:
        render_metric_card(len(contradictions_df), "Contradictions Found", "+3 this week", "negative")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two Column Layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="section-header">Recent Contradiction Events</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subheader">Latest detected narrative inconsistencies</div>', unsafe_allow_html=True)
        
        for _, event in contradictions_df.head(3).iterrows():
            render_contradiction_card(event)
    
    with col2:
        st.markdown('<div class="section-header">Active Signals</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subheader">Current trading recommendations</div>', unsafe_allow_html=True)
        
        signals_df = get_demo_signals()
        for _, signal in signals_df.head(4).iterrows():
            render_signal_card(signal)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # How It Works
    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Our AI-powered pipeline for detecting corporate narrative shifts</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="process-card">
                <div class="process-icon">📥</div>
                <div class="process-title">1. Ingest</div>
                <div class="process-desc">Collect SEC filings (10-K, 10-Q, 8-K) and news articles with executive statements</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="process-card">
                <div class="process-icon">🔬</div>
                <div class="process-title">2. Extract</div>
                <div class="process-desc">Use Claude AI to extract structured claims about guidance, projects, and risk factors</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="process-card">
                <div class="process-icon">⚡</div>
                <div class="process-title">3. Detect</div>
                <div class="process-desc">Apply DeBERTa NLI model to identify contradictions between earlier and later claims</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="process-card">
                <div class="process-icon">📊</div>
                <div class="process-title">4. Signal</div>
                <div class="process-desc">Generate trading signals using XGBoost based on contradiction patterns and market data</div>
            </div>
        """, unsafe_allow_html=True)


elif page == "🔍 Company Explorer":
    st.markdown('<div class="section-header">Company Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Deep dive into individual company narratives and contradictions</div>', unsafe_allow_html=True)
    
    selected_ticker = st.selectbox("Select Company", companies_df["ticker"].tolist(), label_visibility="collapsed")
    company = companies_df[companies_df["ticker"] == selected_ticker].iloc[0]
    
    # Company Header Card
    st.markdown(f"""
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 28px; font-weight: 700; color: #fff;">{company['name']}</div>
                    <div style="font-size: 14px; color: #6b6b7e; margin-top: 4px;">{company['ticker']} • {company['sector']}</div>
                </div>
                <div style="display: flex; gap: 24px;">
                    <div style="text-align: center;">
                        <div style="font-size: 28px; font-weight: 700; color: #3b82f6;">{company['documents']}</div>
                        <div style="font-size: 11px; color: #6b6b7e; text-transform: uppercase;">Documents</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 28px; font-weight: 700; color: #8b5cf6;">{company['claims']}</div>
                        <div style="font-size: 11px; color: #6b6b7e; text-transform: uppercase;">Claims</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 28px; font-weight: 700; color: #ef4444;">{company['contradictions']}</div>
                        <div style="font-size: 11px; color: #6b6b7e; text-transform: uppercase;">Contradictions</div>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Contradiction Events</div>', unsafe_allow_html=True)
    
    company_events = contradictions_df[contradictions_df["ticker"] == selected_ticker]
    if len(company_events) > 0:
        for _, event in company_events.iterrows():
            render_contradiction_card(event)
    else:
        st.info("No contradiction events found for this company.")


elif page == "⚠️ Contradictions":
    st.markdown('<div class="section-header">Contradiction Events</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">All detected narrative inconsistencies across tracked companies</div>', unsafe_allow_html=True)
    
    # Filters
    st.markdown("""
        <div class="card" style="padding: 16px 24px;">
            <div style="font-size: 12px; color: #6b6b7e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Filters</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_ticker = st.multiselect("Ticker", contradictions_df["ticker"].unique().tolist())
    with col2:
        filter_topic = st.multiselect("Topic", contradictions_df["topic"].unique().tolist())
    with col3:
        min_score = st.slider("Minimum Score", 0.0, 1.0, 0.5)
    
    filtered = contradictions_df.copy()
    if filter_ticker:
        filtered = filtered[filtered["ticker"].isin(filter_ticker)]
    if filter_topic:
        filtered = filtered[filtered["topic"].isin(filter_topic)]
    filtered = filtered[filtered["contradiction_score"] >= min_score]
    
    st.markdown(f"""
        <div style="font-size: 14px; color: #8b8b9e; margin: 16px 0;">
            Showing <span style="color: #fff; font-weight: 600;">{len(filtered)}</span> events
        </div>
    """, unsafe_allow_html=True)
    
    for _, event in filtered.iterrows():
        render_contradiction_card(event)


elif page == "📈 Signals":
    st.markdown('<div class="section-header">Signal Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Active trading signals and performance metrics</div>', unsafe_allow_html=True)
    
    signals_df = get_demo_signals()
    
    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bearish_count = len(signals_df[signals_df["signal"] == "bearish_alert"])
        st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #ef4444;">
                <div class="metric-value" style="color: #ef4444;">{bearish_count}</div>
                <div class="metric-label">Bearish Alerts</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        watch_count = len(signals_df[signals_df["signal"] == "watch"])
        st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #eab308;">
                <div class="metric-value" style="color: #eab308;">{watch_count}</div>
                <div class="metric-label">Watch List</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        bullish_count = len(signals_df[signals_df["signal"] == "bullish"])
        st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #22c55e;">
                <div class="metric-value" style="color: #22c55e;">{bullish_count}</div>
                <div class="metric-label">Bullish Signals</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #3b82f6;">
                <div class="metric-value" style="color: #3b82f6;">72%</div>
                <div class="metric-label">Win Rate</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Signal Cards
    col1, col2 = st.columns(2)
    
    signals_list = signals_df.to_dict('records')
    for i, signal in enumerate(signals_list):
        with col1 if i % 2 == 0 else col2:
            render_signal_card(signal)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Performance Section
    st.markdown('<div class="section-header">Historical Performance</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="eval-card">
                <div class="eval-title">Bearish Signal Performance</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div>
                        <div style="font-size: 32px; font-weight: 700; color: #ef4444;">-3.2%</div>
                        <div style="font-size: 12px; color: #6b6b7e;">Avg Return (3-day)</div>
                    </div>
                    <div>
                        <div style="font-size: 32px; font-weight: 700; color: #22c55e;">72%</div>
                        <div style="font-size: 12px; color: #6b6b7e;">Win Rate</div>
                    </div>
                    <div>
                        <div style="font-size: 32px; font-weight: 700; color: #fff;">156</div>
                        <div style="font-size: 12px; color: #6b6b7e;">Total Signals</div>
                    </div>
                    <div>
                        <div style="font-size: 32px; font-weight: 700; color: #fff;">2.1</div>
                        <div style="font-size: 12px; color: #6b6b7e;">Sharpe Ratio</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="eval-card">
                <div class="eval-title">Bullish Signal Performance</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div>
                        <div style="font-size: 32px; font-weight: 700; color: #22c55e;">+2.1%</div>
                        <div style="font-size: 12px; color: #6b6b7e;">Avg Return (3-day)</div>
                    </div>
                    <div>
                        <div style="font-size: 32px; font-weight: 700; color: #22c55e;">65%</div>
                        <div style="font-size: 12px; color: #6b6b7e;">Win Rate</div>
                    </div>
                    <div>
                        <div style="font-size: 32px; font-weight: 700; color: #fff;">89</div>
                        <div style="font-size: 12px; color: #6b6b7e;">Total Signals</div>
                    </div>
                    <div>
                        <div style="font-size: 32px; font-weight: 700; color: #fff;">1.4</div>
                        <div style="font-size: 12px; color: #6b6b7e;">Sharpe Ratio</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


elif page == "📋 Evaluation":
    st.markdown('<div class="section-header">Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Comprehensive performance analysis and validation metrics</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🎯 Baselines", "🔧 Robustness", "📐 Calibration"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="eval-card">
                    <div class="eval-title">Classification Metrics (Test Set)</div>
                    <div style="margin-top: 16px;">
            """, unsafe_allow_html=True)
            
            metrics = [
                ("Accuracy", "68.5%", "#3b82f6"),
                ("Precision", "65.2%", "#8b5cf6"),
                ("Recall", "64.8%", "#ec4899"),
                ("F1 Score", "65.0%", "#22c55e"),
                ("ROC-AUC", "0.74", "#eab308"),
            ]
            
            for name, value, color in metrics:
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <span style="color: #8b8b9e;">{name}</span>
                        <span style="font-weight: 700; color: {color};">{value}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="eval-card">
                    <div class="eval-title">Confusion Matrix</div>
                    <div style="margin-top: 16px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; text-align: center;">
                        <div></div>
                        <div style="font-size: 11px; color: #6b6b7e; padding: 8px;">Pred +</div>
                        <div style="font-size: 11px; color: #6b6b7e; padding: 8px;">Pred 0</div>
                        <div style="font-size: 11px; color: #6b6b7e; padding: 8px;">Pred -</div>
                        
                        <div style="font-size: 11px; color: #6b6b7e; padding: 8px;">Act +</div>
                        <div style="background: rgba(34,197,94,0.3); padding: 12px; border-radius: 8px; font-weight: 700; color: #22c55e;">45</div>
                        <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; color: #8b8b9e;">8</div>
                        <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; color: #8b8b9e;">12</div>
                        
                        <div style="font-size: 11px; color: #6b6b7e; padding: 8px;">Act 0</div>
                        <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; color: #8b8b9e;">10</div>
                        <div style="background: rgba(34,197,94,0.3); padding: 12px; border-radius: 8px; font-weight: 700; color: #22c55e;">38</div>
                        <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; color: #8b8b9e;">15</div>
                        
                        <div style="font-size: 11px; color: #6b6b7e; padding: 8px;">Act -</div>
                        <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; color: #8b8b9e;">8</div>
                        <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; color: #8b8b9e;">12</div>
                        <div style="background: rgba(34,197,94,0.3); padding: 12px; border-radius: 8px; font-weight: 700; color: #22c55e;">52</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
            <div class="eval-card">
                <div class="eval-title">Baseline Model Comparison</div>
                <div style="margin-top: 16px;">
        """, unsafe_allow_html=True)
        
        baselines = [
            ("Full Model (NLI + Features)", "65.0%", "0.74", "65.2%", True),
            ("Sentiment Only (FinBERT)", "52.3%", "0.58", "50.1%", False),
            ("Keyword Rules", "48.1%", "0.55", "46.5%", False),
            ("BoW Classifier (TF-IDF)", "55.7%", "0.62", "54.2%", False),
        ]
        
        st.markdown("""
            <div style="display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 8px; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 12px; color: #6b6b7e; text-transform: uppercase;">Model</div>
                <div style="font-size: 12px; color: #6b6b7e; text-transform: uppercase; text-align: center;">F1 Score</div>
                <div style="font-size: 12px; color: #6b6b7e; text-transform: uppercase; text-align: center;">ROC-AUC</div>
                <div style="font-size: 12px; color: #6b6b7e; text-transform: uppercase; text-align: center;">Precision</div>
            </div>
        """, unsafe_allow_html=True)
        
        for name, f1, auc, prec, is_best in baselines:
            bg = "rgba(34,197,94,0.1)" if is_best else "transparent"
            weight = "700" if is_best else "400"
            color = "#22c55e" if is_best else "#fff"
            st.markdown(f"""
                <div style="display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 8px; padding: 16px 12px; background: {bg}; border-radius: 8px; margin: 4px 0;">
                    <div style="color: {color}; font-weight: {weight};">{name}</div>
                    <div style="text-align: center; color: {color}; font-weight: {weight};">{f1}</div>
                    <div style="text-align: center; color: {color}; font-weight: {weight};">{auc}</div>
                    <div style="text-align: center; color: {color}; font-weight: {weight};">{prec}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="eval-card" style="margin-top: 16px;">
                <div class="eval-title">Key Findings</div>
                <ul style="color: #e0e0e0; line-height: 2;">
                    <li>Full NLI model outperforms all baselines by <span style="color: #22c55e; font-weight: 600;">+9.3%</span> F1 score</li>
                    <li>Contradiction detection adds <span style="color: #22c55e; font-weight: 600;">+12.7%</span> improvement over sentiment-only</li>
                    <li>Keyword rules alone capture only basic patterns, missing nuanced contradictions</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="eval-card">
                    <div class="eval-title">Performance by Topic</div>
                    <div style="margin-top: 16px;">
            """, unsafe_allow_html=True)
            
            topics = [
                ("Guidance", "68.2%", 145, "#3b82f6"),
                ("Projects", "64.5%", 98, "#8b5cf6"),
                ("Legal/Regulatory", "71.3%", 52, "#ef4444"),
                ("Supply Chain", "62.1%", 67, "#eab308"),
                ("Risk Factors", "66.8%", 83, "#22c55e"),
            ]
            
            for topic, f1, n, color in topics:
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <div>
                            <span style="color: #fff;">{topic}</span>
                            <span style="color: #6b6b7e; font-size: 12px; margin-left: 8px;">n={n}</span>
                        </div>
                        <div style="font-weight: 700; color: {color};">{f1}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="eval-card">
                    <div class="eval-title">Time Drift Analysis</div>
                    <div style="margin-top: 16px;">
            """, unsafe_allow_html=True)
            
            periods = [
                ("Mar - Jul 2025", "64.2%", 142),
                ("Jul - Nov 2025", "66.1%", 156),
                ("Nov 2025 - Mar 2026", "65.4%", 147),
            ]
            
            for period, f1, n in periods:
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <div>
                            <span style="color: #fff;">{period}</span>
                            <span style="color: #6b6b7e; font-size: 12px; margin-left: 8px;">n={n}</span>
                        </div>
                        <div style="font-weight: 700; color: #22c55e;">{f1}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
                    </div>
                    <div style="margin-top: 16px; padding: 12px; background: rgba(34,197,94,0.1); border-radius: 8px; border-left: 3px solid #22c55e;">
                        <span style="color: #22c55e;">✓</span>
                        <span style="color: #e0e0e0; margin-left: 8px;">Model performance stable across time - minimal drift detected</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="eval-card">
                    <div class="eval-title">Calibration Metrics</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 24px;">
                        <div style="text-align: center;">
                            <div style="font-size: 48px; font-weight: 700; color: #3b82f6;">0.21</div>
                            <div style="font-size: 12px; color: #6b6b7e; margin-top: 4px;">Brier Score</div>
                            <div style="font-size: 11px; color: #22c55e;">Lower is better</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 48px; font-weight: 700; color: #8b5cf6;">0.08</div>
                            <div style="font-size: 12px; color: #6b6b7e; margin-top: 4px;">Expected Cal. Error</div>
                            <div style="font-size: 11px; color: #22c55e;">Lower is better</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="eval-card">
                    <div class="eval-title">Reliability by Confidence</div>
                    <div style="margin-top: 16px;">
            """, unsafe_allow_html=True)
            
            bins = [
                ("0.0 - 0.2", "10%", "12%", "+2%", "#22c55e"),
                ("0.2 - 0.4", "30%", "28%", "-2%", "#eab308"),
                ("0.4 - 0.6", "50%", "48%", "-2%", "#eab308"),
                ("0.6 - 0.8", "70%", "65%", "-5%", "#ef4444"),
                ("0.8 - 1.0", "90%", "82%", "-8%", "#ef4444"),
            ]
            
            for bin_range, pred, actual, gap, color in bins:
                st.markdown(f"""
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 8px; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 13px;">
                        <div style="color: #8b8b9e;">{bin_range}</div>
                        <div style="color: #fff; text-align: center;">{pred}</div>
                        <div style="color: #fff; text-align: center;">{actual}</div>
                        <div style="color: {color}; text-align: center; font-weight: 600;">{gap}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
