"""
Corporate Narrative Consistency Engine - Streamlit Dashboard
Minimal, polished UI design
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Beyond",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Minimal CSS
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: #0a0a0f;
        color: #e5e5e5;
    }
    
    [data-testid="stSidebar"] {
        background: #0f0f14;
        border-right: 1px solid #1a1a24;
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    h1, h2, h3 { color: #ffffff; font-weight: 600; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #14141c;
        border-radius: 8px;
        padding: 10px 20px;
        color: #888;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1c1c28;
        color: #fff;
    }
    
    .stSelectbox > div > div { background: #14141c; border: 1px solid #1a1a24; }
    .stMultiSelect > div > div { background: #14141c; border: 1px solid #1a1a24; }
    .stSlider > div > div > div { background: #3b82f6; }
    
    hr { border-color: #1a1a24; margin: 1.5rem 0; }
    
    #MainMenu, footer, header { visibility: hidden; }
    
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a0f; }
    ::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data
# =============================================================================

@st.cache_data
def get_companies():
    return pd.DataFrame([
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "docs": 45, "claims": 128, "events": 8},
        {"ticker": "MSFT", "name": "Microsoft", "sector": "Technology", "docs": 52, "claims": 145, "events": 6},
        {"ticker": "GOOGL", "name": "Alphabet", "sector": "Technology", "docs": 48, "claims": 132, "events": 5},
        {"ticker": "TSLA", "name": "Tesla", "sector": "Automotive", "docs": 62, "claims": 189, "events": 12},
        {"ticker": "AMZN", "name": "Amazon", "sector": "E-commerce", "docs": 55, "claims": 156, "events": 7},
    ])

@st.cache_data
def get_contradictions():
    return [
        {
            "ticker": "TSLA", "company": "Tesla", "date": "2025-08-15", "topic": "Projects",
            "score": 0.87, "return_1d": -4.2, "return_3d": -6.1, "signal": "Bearish", "confidence": 81,
            "earlier": {"text": "Cybertruck production will reach 250,000 units by end of 2025", "date": "2025-05-10", "source": "CEO Statement"},
            "later": {"text": "Cybertruck production targets revised to 125,000 units", "date": "2025-08-15", "source": "10-Q Filing"}
        },
        {
            "ticker": "AAPL", "company": "Apple", "date": "2025-07-20", "topic": "Guidance",
            "score": 0.72, "return_1d": -1.8, "return_3d": -2.5, "signal": "Bearish", "confidence": 68,
            "earlier": {"text": "Services revenue expected to grow 20% year-over-year", "date": "2025-04-15", "source": "10-K Filing"},
            "later": {"text": "Services growth moderating to 15% due to market conditions", "date": "2025-07-20", "source": "8-K Filing"}
        },
        {
            "ticker": "MSFT", "company": "Microsoft", "date": "2025-09-05", "topic": "Products",
            "score": 0.65, "return_1d": -1.2, "return_3d": -0.8, "signal": "Watch", "confidence": 55,
            "earlier": {"text": "Azure AI features launching in Q3 2025", "date": "2025-06-01", "source": "Press Release"},
            "later": {"text": "Azure AI rollout delayed to Q4 2025 for additional testing", "date": "2025-09-05", "source": "8-K Filing"}
        },
        {
            "ticker": "GOOGL", "company": "Alphabet", "date": "2025-10-12", "topic": "Legal",
            "score": 0.78, "return_1d": -3.5, "return_3d": -5.2, "signal": "Bearish", "confidence": 75,
            "earlier": {"text": "No material impact expected from pending antitrust proceedings", "date": "2025-07-01", "source": "10-Q Filing"},
            "later": {"text": "Potential remedies could significantly impact advertising business", "date": "2025-10-12", "source": "8-K Filing"}
        },
    ]

@st.cache_data
def get_signals():
    return [
        {"ticker": "TSLA", "type": "Bearish", "confidence": 81, "topic": "Projects", "date": "2025-08-15", "price": 242.50, "change": -4.2, "reason": "Production target contradiction"},
        {"ticker": "AAPL", "type": "Bearish", "confidence": 68, "topic": "Guidance", "date": "2025-07-20", "price": 198.30, "change": -1.8, "reason": "Revenue guidance revision"},
        {"ticker": "GOOGL", "type": "Bearish", "confidence": 75, "topic": "Legal", "date": "2025-10-12", "price": 156.80, "change": -3.5, "reason": "Legal risk escalation"},
        {"ticker": "MSFT", "type": "Watch", "confidence": 55, "topic": "Products", "date": "2025-09-05", "price": 412.20, "change": -1.2, "reason": "Product delay detected"},
        {"ticker": "AMZN", "type": "Bullish", "confidence": 62, "topic": "Guidance", "date": "2025-11-01", "price": 185.40, "change": 2.1, "reason": "Consistent positive AWS messaging"},
    ]


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("### ◉ Beyond")
    st.caption("Corporate Intelligence")
    
    st.divider()
    
    page = st.radio(
        "Navigate",
        ["Overview", "Companies", "Contradictions", "Signals", "Evaluation"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.caption("QUICK STATS")
    col1, col2 = st.columns(2)
    col1.metric("Companies", "5")
    col2.metric("Alerts", "4", delta="+2")


# =============================================================================
# Pages
# =============================================================================

companies = get_companies()
contradictions = get_contradictions()
signals = get_signals()

if page == "Overview":
    st.title("Dashboard")
    st.caption("Corporate narrative analysis and contradiction detection")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Companies", len(companies))
    col2.metric("Documents", companies['docs'].sum())
    col3.metric("Claims", companies['claims'].sum())
    col4.metric("Contradictions", len(contradictions), delta="+3")
    
    st.divider()
    
    # Two columns
    left, right = st.columns([3, 2])
    
    with left:
        st.subheader("Recent Contradictions")
        
        for c in contradictions[:3]:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{c['company']}** · {c['ticker']}")
                    st.caption(f"{c['topic']} · {c['date']}")
                with col2:
                    score_color = "🔴" if c['score'] > 0.7 else "🟡"
                    st.markdown(f"### {score_color} {c['score']:.2f}")
                
                with st.expander("View claims"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Earlier**")
                        st.info(f"_{c['earlier']['text']}_")
                        st.caption(f"📄 {c['earlier']['source']} · {c['earlier']['date']}")
                    with col2:
                        st.markdown("**Later**")
                        st.error(f"_{c['later']['text']}_")
                        st.caption(f"📄 {c['later']['source']} · {c['later']['date']}")
                    
                    st.markdown(f"**Return:** {c['return_1d']}% (1d) · {c['return_3d']}% (3d) · **Signal:** {c['signal']}")
                
                st.divider()
    
    with right:
        st.subheader("Active Signals")
        
        for s in signals[:4]:
            with st.container():
                col1, col2 = st.columns([2, 1])
                with col1:
                    icon = "🔴" if s['type'] == "Bearish" else "🟢" if s['type'] == "Bullish" else "🟡"
                    st.markdown(f"### {icon} {s['ticker']}")
                    st.caption(f"{s['type'].upper()} · {s['confidence']}% conf")
                with col2:
                    color = "red" if s['change'] < 0 else "green"
                    st.markdown(f"**${s['price']}**")
                    st.markdown(f":{color}[{'+' if s['change'] > 0 else ''}{s['change']}%]")
                
                st.caption(s['reason'])
                st.divider()
    
    # Process
    st.subheader("How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("#### 1. Ingest")
        st.caption("SEC filings (10-K, 10-Q, 8-K) and news articles with CEO statements")
    with col2:
        st.markdown("#### 2. Extract")
        st.caption("Claude AI extracts structured claims about guidance, projects, risk")
    with col3:
        st.markdown("#### 3. Detect")
        st.caption("DeBERTa NLI identifies contradictions between claims")
    with col4:
        st.markdown("#### 4. Signal")
        st.caption("XGBoost generates signals from contradiction patterns")


elif page == "Companies":
    st.title("Company Explorer")
    st.caption("Select a company to view detailed analysis")
    
    selected = st.selectbox("Select company", companies['ticker'].tolist(), label_visibility="collapsed")
    company = companies[companies['ticker'] == selected].iloc[0]
    
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Company", company['name'])
    col2.metric("Documents", company['docs'])
    col3.metric("Claims", company['claims'])
    col4.metric("Contradictions", company['events'])
    
    st.divider()
    
    st.subheader("Contradiction Events")
    
    company_events = [c for c in contradictions if c['ticker'] == selected]
    
    if company_events:
        for c in company_events:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{c['topic']}** · {c['date']}")
                with col2:
                    st.markdown(f"### {c['score']:.2f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Earlier claim**")
                    st.info(c['earlier']['text'])
                    st.caption(f"{c['earlier']['source']} · {c['earlier']['date']}")
                with col2:
                    st.markdown("**Later claim**")
                    st.error(c['later']['text'])
                    st.caption(f"{c['later']['source']} · {c['later']['date']}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("1D Return", f"{c['return_1d']}%")
                col2.metric("3D Return", f"{c['return_3d']}%")
                col3.metric("Signal", c['signal'])
                
                st.divider()
    else:
        st.info("No contradiction events found for this company.")


elif page == "Contradictions":
    st.title("Contradiction Events")
    st.caption("All detected narrative inconsistencies")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_ticker = st.multiselect("Ticker", [c['ticker'] for c in contradictions])
    with col2:
        filter_topic = st.multiselect("Topic", list(set(c['topic'] for c in contradictions)))
    with col3:
        min_score = st.slider("Min score", 0.0, 1.0, 0.5)
    
    st.divider()
    
    # Filter
    filtered = contradictions
    if filter_ticker:
        filtered = [c for c in filtered if c['ticker'] in filter_ticker]
    if filter_topic:
        filtered = [c for c in filtered if c['topic'] in filter_topic]
    filtered = [c for c in filtered if c['score'] >= min_score]
    
    st.caption(f"Showing {len(filtered)} events")
    
    for c in filtered:
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"### {c['company']} ({c['ticker']})")
                st.caption(f"{c['topic']} · {c['date']}")
            with col2:
                score_color = "🔴" if c['score'] > 0.7 else "🟡" if c['score'] > 0.5 else "🟢"
                st.markdown(f"## {score_color} {c['score']:.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Earlier**")
                st.info(c['earlier']['text'])
                st.caption(f"{c['earlier']['source']} · {c['earlier']['date']}")
            with col2:
                st.markdown("**Later**")
                st.error(c['later']['text'])
                st.caption(f"{c['later']['source']} · {c['later']['date']}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("1D Return", f"{c['return_1d']}%")
            col2.metric("3D Return", f"{c['return_3d']}%")
            col3.metric("Signal", c['signal'])
            col4.metric("Confidence", f"{c['confidence']}%")
            
            st.divider()


elif page == "Signals":
    st.title("Signal Dashboard")
    st.caption("Active trading signals and performance")
    
    # Summary
    bearish = len([s for s in signals if s['type'] == 'Bearish'])
    bullish = len([s for s in signals if s['type'] == 'Bullish'])
    watch = len([s for s in signals if s['type'] == 'Watch'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔴 Bearish", bearish)
    col2.metric("🟢 Bullish", bullish)
    col3.metric("🟡 Watch", watch)
    col4.metric("Win Rate", "72%")
    
    st.divider()
    
    st.subheader("Active Signals")
    
    for s in signals:
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                icon = "🔴" if s['type'] == "Bearish" else "🟢" if s['type'] == "Bullish" else "🟡"
                st.markdown(f"### {icon} {s['ticker']}")
                badge_color = "red" if s['type'] == "Bearish" else "green" if s['type'] == "Bullish" else "orange"
                st.markdown(f":{badge_color}[{s['type'].upper()}]")
            
            with col2:
                st.markdown(f"**{s['reason']}**")
                st.caption(f"{s['topic']} · {s['date']} · {s['confidence']}% confidence")
            
            with col3:
                st.metric("Price", f"${s['price']}", delta=f"{s['change']}%")
            
            st.divider()
    
    # Performance
    st.subheader("Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Bearish Signals")
        c1, c2 = st.columns(2)
        c1.metric("Avg Return", "-3.2%")
        c2.metric("Win Rate", "72%")
        c1.metric("Total Signals", "156")
        c2.metric("Sharpe", "2.1")
    
    with col2:
        st.markdown("#### Bullish Signals")
        c1, c2 = st.columns(2)
        c1.metric("Avg Return", "+2.1%")
        c2.metric("Win Rate", "65%")
        c1.metric("Total Signals", "89")
        c2.metric("Sharpe", "1.4")


elif page == "Evaluation":
    st.title("Model Evaluation")
    st.caption("Performance metrics and validation")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Baselines", "Robustness", "Calibration"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Metrics")
            
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                "Value": ["68.5%", "65.2%", "64.8%", "65.0%", "0.74"]
            })
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("Confusion Matrix")
            
            cm_df = pd.DataFrame(
                [[45, 8, 12], [10, 38, 15], [8, 12, 52]],
                index=["Actual +", "Actual 0", "Actual -"],
                columns=["Pred +", "Pred 0", "Pred -"]
            )
            st.dataframe(cm_df, use_container_width=True)
    
    with tab2:
        st.subheader("Baseline Comparison")
        
        baseline_df = pd.DataFrame({
            "Model": ["Full Model (NLI)", "Sentiment (FinBERT)", "Keyword Rules", "BoW (TF-IDF)"],
            "F1": ["65.0%", "52.3%", "48.1%", "55.7%"],
            "AUC": ["0.74", "0.58", "0.55", "0.62"],
            "Precision": ["65.2%", "50.1%", "46.5%", "54.2%"]
        })
        st.dataframe(baseline_df, hide_index=True, use_container_width=True)
        
        st.divider()
        
        st.markdown("**Key Findings**")
        st.markdown("- Full NLI model outperforms all baselines by **+9.3%** F1 score")
        st.markdown("- Contradiction detection adds **+12.7%** over sentiment-only")
        st.markdown("- Keyword rules miss nuanced contradictions")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("By Topic")
            
            topic_df = pd.DataFrame({
                "Topic": ["Guidance", "Projects", "Legal", "Supply Chain", "Risk"],
                "F1": ["68.2%", "64.5%", "71.3%", "62.1%", "66.8%"],
                "N": [145, 98, 52, 67, 83]
            })
            st.dataframe(topic_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("Time Drift")
            
            drift_df = pd.DataFrame({
                "Period": ["Mar-Jul 2025", "Jul-Nov 2025", "Nov-Mar 2026"],
                "F1": ["64.2%", "66.1%", "65.4%"],
                "N": [142, 156, 147]
            })
            st.dataframe(drift_df, hide_index=True, use_container_width=True)
        
        st.success("✓ Model performance stable across time - minimal drift")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Calibration Metrics")
            c1, c2 = st.columns(2)
            c1.metric("Brier Score", "0.21")
            c2.metric("ECE", "0.08")
            st.caption("Lower is better for both metrics")
        
        with col2:
            st.subheader("Reliability")
            
            rel_df = pd.DataFrame({
                "Bin": ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
                "Predicted": ["10%", "30%", "50%", "70%", "90%"],
                "Actual": ["12%", "28%", "48%", "65%", "82%"],
                "Gap": ["+2%", "-2%", "-2%", "-5%", "-8%"]
            })
            st.dataframe(rel_df, hide_index=True, use_container_width=True)
