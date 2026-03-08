"""
Corporate Narrative Consistency Engine - Streamlit Dashboard

Interactive dashboard for exploring contradiction events,
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
        {"ticker": "TSLA", "signal": "bearish_alert", "confidence": 0.81, "topic": "projects", "date": "2025-08-15", "reasoning": "Strong contradiction in production targets"},
        {"ticker": "AAPL", "signal": "bearish_alert", "confidence": 0.68, "topic": "guidance", "date": "2025-07-20", "reasoning": "Revenue guidance revision detected"},
        {"ticker": "GOOGL", "signal": "bearish_alert", "confidence": 0.75, "topic": "legal_regulatory", "date": "2025-10-12", "reasoning": "Legal risk assessment changed materially"},
        {"ticker": "MSFT", "signal": "watch", "confidence": 0.55, "topic": "products", "date": "2025-09-05", "reasoning": "Product timeline shift"},
        {"ticker": "AMZN", "signal": "bullish", "confidence": 0.62, "topic": "guidance", "date": "2025-11-01", "reasoning": "Consistent positive messaging on AWS growth"},
    ])


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Overview", "🔍 Company Explorer", "⚠️ Contradictions", "📈 Signals", "📋 Evaluation"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Range")
st.sidebar.date_input("Start Date", datetime(2025, 3, 8))
st.sidebar.date_input("End Date", datetime(2026, 3, 8))

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
companies_df = get_demo_companies()
contradictions_df = get_demo_contradictions()
st.sidebar.metric("Companies Tracked", len(companies_df))
st.sidebar.metric("Total Contradictions", len(contradictions_df))
st.sidebar.metric("Active Alerts", len(contradictions_df[contradictions_df["signal"] == "bearish_alert"]))


# =============================================================================
# Pages
# =============================================================================

if page == "🏠 Overview":
    st.title("Corporate Narrative Consistency Engine")
    st.markdown("""
    An AI system that detects contradictions in corporate disclosures and executive statements,
    then correlates them with stock price movements to generate trading signals.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Companies", len(companies_df), delta=None)
    with col2:
        st.metric("Documents Analyzed", companies_df["documents"].sum())
    with col3:
        st.metric("Claims Extracted", companies_df["claims"].sum())
    with col4:
        st.metric("Contradictions Found", len(contradictions_df))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent Contradiction Events")
        recent = contradictions_df[["ticker", "event_date", "topic", "contradiction_score", "signal"]].copy()
        recent.columns = ["Ticker", "Date", "Topic", "Score", "Signal"]
        st.dataframe(recent, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Signal Distribution")
        signals_df = get_demo_signals()
        signal_counts = signals_df["signal"].value_counts()
        st.bar_chart(signal_counts)
    
    st.markdown("---")
    st.subheader("How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 1️⃣ Ingest")
        st.markdown("Collect SEC filings (10-K, 10-Q, 8-K) and news articles with CEO statements")
    with col2:
        st.markdown("### 2️⃣ Extract")
        st.markdown("Use Claude AI to extract structured claims about guidance, projects, and risk")
    with col3:
        st.markdown("### 3️⃣ Detect")
        st.markdown("Apply DeBERTa NLI to detect contradictions between earlier and later claims")
    with col4:
        st.markdown("### 4️⃣ Signal")
        st.markdown("Generate trading signals based on contradiction patterns and market reactions")


elif page == "🔍 Company Explorer":
    st.title("Company Explorer")
    
    selected_ticker = st.selectbox("Select Company", companies_df["ticker"].tolist())
    company = companies_df[companies_df["ticker"] == selected_ticker].iloc[0]
    
    st.markdown(f"## {company['name']} ({company['ticker']})")
    st.markdown(f"**Sector:** {company['sector']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", company["documents"])
    with col2:
        st.metric("Claims Extracted", company["claims"])
    with col3:
        st.metric("Contradictions", company["contradictions"])
    
    st.markdown("---")
    
    st.subheader("Contradiction Events")
    company_events = contradictions_df[contradictions_df["ticker"] == selected_ticker]
    if len(company_events) > 0:
        for _, event in company_events.iterrows():
            with st.expander(f"📅 {event['event_date']} - {event['topic'].title()} (Score: {event['contradiction_score']:.2f})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Earlier Claim:**")
                    st.info(f"📄 {event['earlier_source']} ({event['earlier_date']})\n\n\"{event['earlier_claim']}\"")
                with col2:
                    st.markdown("**Later Claim:**")
                    st.warning(f"📄 {event['later_source']} ({event['later_date']})\n\n\"{event['later_claim']}\"")
                
                st.markdown(f"**Market Reaction:** {event['return_1d']}% (1-day), {event['return_3d']}% (3-day)")
                st.markdown(f"**Signal:** `{event['signal']}` (Confidence: {event['confidence']:.0%})")
    else:
        st.info("No contradiction events found for this company.")


elif page == "⚠️ Contradictions":
    st.title("Contradiction Events")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_ticker = st.multiselect("Filter by Ticker", contradictions_df["ticker"].unique().tolist())
    with col2:
        filter_topic = st.multiselect("Filter by Topic", contradictions_df["topic"].unique().tolist())
    with col3:
        min_score = st.slider("Minimum Score", 0.0, 1.0, 0.5)
    
    filtered = contradictions_df.copy()
    if filter_ticker:
        filtered = filtered[filtered["ticker"].isin(filter_ticker)]
    if filter_topic:
        filtered = filtered[filtered["topic"].isin(filter_topic)]
    filtered = filtered[filtered["contradiction_score"] >= min_score]
    
    st.markdown(f"**Showing {len(filtered)} events**")
    
    for _, event in filtered.iterrows():
        with st.container():
            st.markdown(f"### {event['company']} ({event['ticker']})")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Score", f"{event['contradiction_score']:.2f}")
            with col2:
                st.metric("Topic", event["topic"].title())
            with col3:
                st.metric("1-Day Return", f"{event['return_1d']}%")
            with col4:
                signal_color = "🔴" if event["signal"] == "bearish_alert" else "🟡" if event["signal"] == "watch" else "🟢"
                st.metric("Signal", f"{signal_color} {event['signal']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Earlier ({event['earlier_date']}):** {event['earlier_source']}")
                st.markdown(f"> {event['earlier_claim']}")
            with col2:
                st.markdown(f"**Later ({event['later_date']}):** {event['later_source']}")
                st.markdown(f"> {event['later_claim']}")
            
            st.markdown("---")


elif page == "📈 Signals":
    st.title("Signal Dashboard")
    
    signals_df = get_demo_signals()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        bearish_count = len(signals_df[signals_df["signal"] == "bearish_alert"])
        st.metric("🔴 Bearish Alerts", bearish_count)
    with col2:
        watch_count = len(signals_df[signals_df["signal"] == "watch"])
        st.metric("🟡 Watch", watch_count)
    with col3:
        bullish_count = len(signals_df[signals_df["signal"] == "bullish"])
        st.metric("🟢 Bullish", bullish_count)
    
    st.markdown("---")
    st.subheader("Active Signals")
    
    for _, signal in signals_df.iterrows():
        if signal["signal"] == "bearish_alert":
            color = "🔴"
            box = st.error
        elif signal["signal"] == "watch":
            color = "🟡"
            box = st.warning
        else:
            color = "🟢"
            box = st.success
        
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"### {color} {signal['ticker']}")
                st.markdown(f"**{signal['signal'].upper()}**")
                st.markdown(f"Confidence: {signal['confidence']:.0%}")
            with col2:
                st.markdown(f"**Topic:** {signal['topic'].title()}")
                st.markdown(f"**Date:** {signal['date']}")
                st.markdown(f"**Reasoning:** {signal['reasoning']}")
            st.markdown("---")
    
    st.subheader("Historical Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Win Rate (Bearish Signals)", "72%", delta="+5%")
        st.metric("Avg Return on Bearish", "-3.2%")
    with col2:
        st.metric("Win Rate (Bullish Signals)", "65%", delta="+2%")
        st.metric("Avg Return on Bullish", "+2.1%")


elif page == "📋 Evaluation":
    st.title("Model Evaluation")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Baselines", "Robustness", "Calibration"])
    
    with tab1:
        st.subheader("Model Performance (Test Set)")
        
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 Score (macro)", "ROC-AUC"],
            "Value": ["68.5%", "65.2%", "64.8%", "65.0%", "0.74"]
        })
        st.table(metrics_df)
        
        st.subheader("Confusion Matrix")
        confusion = pd.DataFrame(
            [[45, 8, 12], [10, 38, 15], [8, 12, 52]],
            index=["Actual Positive", "Actual Neutral", "Actual Negative"],
            columns=["Pred Positive", "Pred Neutral", "Pred Negative"]
        )
        st.dataframe(confusion, use_container_width=True)
    
    with tab2:
        st.subheader("Baseline Comparison")
        
        baseline_df = pd.DataFrame({
            "Model": ["Full Model (NLI + Features)", "Sentiment Only (FinBERT)", "Keyword Rules", "BoW Classifier (TF-IDF)"],
            "F1 Score": ["65.0%", "52.3%", "48.1%", "55.7%"],
            "ROC-AUC": ["0.74", "0.58", "0.55", "0.62"],
            "Precision": ["65.2%", "50.1%", "46.5%", "54.2%"]
        })
        st.table(baseline_df)
        
        st.markdown("""
        **Key Findings:**
        - Full NLI model outperforms all baselines by significant margin
        - Sentiment-only baseline captures ~80% of keyword rules performance
        - Contradiction detection adds ~10% F1 improvement over sentiment
        """)
    
    with tab3:
        st.subheader("Robustness Analysis")
        
        st.markdown("**By Topic**")
        topic_df = pd.DataFrame({
            "Topic": ["Guidance", "Projects", "Legal/Regulatory", "Supply Chain", "Risk Factors"],
            "F1 Score": ["68.2%", "64.5%", "71.3%", "62.1%", "66.8%"],
            "N Samples": [145, 98, 52, 67, 83]
        })
        st.table(topic_df)
        
        st.markdown("**Time Drift Analysis**")
        drift_df = pd.DataFrame({
            "Period": ["Mar-Jul 2025", "Jul-Nov 2025", "Nov 2025-Mar 2026"],
            "F1 Score": ["64.2%", "66.1%", "65.4%"],
            "N Samples": [142, 156, 147]
        })
        st.table(drift_df)
        
        st.success("✅ Model performance is stable across time periods - minimal drift detected")
    
    with tab4:
        st.subheader("Calibration Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Brier Score", "0.21", help="Lower is better. 0 = perfect")
            st.metric("Expected Calibration Error", "0.08", help="Lower is better")
        with col2:
            st.markdown("""
            **Interpretation:**
            - Model is reasonably well-calibrated
            - Predicted probabilities align with actual frequencies
            - Slight overconfidence on high-probability predictions
            """)
        
        st.markdown("**Reliability by Confidence Bin**")
        reliability_df = pd.DataFrame({
            "Confidence Bin": ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
            "Predicted": ["10%", "30%", "50%", "70%", "90%"],
            "Actual": ["12%", "28%", "48%", "65%", "82%"],
            "Gap": ["+2%", "-2%", "-2%", "-5%", "-8%"]
        })
        st.table(reliability_df)


# =============================================================================
# Footer
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("**Corporate Narrative Engine** v1.0")
st.sidebar.markdown("Built with Streamlit")
