# Corporate Narrative Consistency Engine

An end-to-end AI system that detects contradictions in corporate disclosures and executive statements, and measures how those contradictions correlate with stock price movements.

## Overview

This system automatically:

1. **Collects** corporate filings (SEC 8-K, 10-K, 10-Q) and executive statements from news sources
2. **Extracts** structured claims made by companies and executives using NLP
3. **Detects** contradictions and narrative shifts across time using Natural Language Inference
4. **Aligns** contradiction events with stock price movements
5. **Learns** whether contradictions correlate with positive, negative, or neutral stock signals
6. **Provides** explainable evidence for each detected signal

## Core Research Question

> Do contradictions or narrative shifts in corporate disclosures and executive statements predict short-term stock price movements?

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION LAYER                               │
├─────────────────┬─────────────────────────┬─────────────────────────────────┤
│   SEC EDGAR     │    News Sources         │      Market Data                │
│   (8-K, 10-K,   │    (GDELT, NewsAPI,     │      (yfinance)                 │
│    10-Q)        │     RSS Feeds)          │                                 │
└────────┬────────┴───────────┬─────────────┴──────────────┬──────────────────┘
         │                    │                            │
         ▼                    ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NLP PIPELINE                                    │
├─────────────────┬─────────────────────────┬─────────────────────────────────┤
│  Document       │   Claim Extraction      │   Claim Matching                │
│  Parsing        │   (Claude API)          │   (Embeddings)                  │
└────────┬────────┴───────────┬─────────────┴──────────────┬──────────────────┘
         │                    │                            │
         ▼                    ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONTRADICTION DETECTION                               │
│                    (DeBERTa NLI Model + Temporal Guard)                      │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ANALYSIS LAYER                                     │
├─────────────────┬─────────────────────────┬─────────────────────────────────┤
│  Event          │   Market Alignment      │   Feature Engineering           │
│  Generation     │   (Returns, Abnormal)   │                                 │
└────────┬────────┴───────────┬─────────────┴──────────────┬──────────────────┘
         │                    │                            │
         ▼                    ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODELING LAYER                                     │
├─────────────────┬─────────────────────────┬─────────────────────────────────┤
│  Baseline       │   Signal Prediction     │   Explainability                │
│  Models         │   (XGBoost)             │   (SHAP)                        │
└────────┬────────┴───────────┬─────────────┴──────────────┬──────────────────┘
         │                    │                            │
         ▼                    ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION LAYER                                    │
├─────────────────┬───────────────┬───────────────┬───────────────────────────┤
│  Metrics        │  Signal Decay │  Robustness   │  Calibration              │
│  (P/R/F1/AUC)   │  Analysis     │  Tests        │  Analysis                 │
└─────────────────┴───────────────┴───────────────┴───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GRADIO DASHBOARD                                     │
│         (Company Explorer, Contradiction Viewer, Signal Dashboard)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
corporate-narrative-engine/
├── .env.template           # Environment variables template
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata
│
├── configs/                # Configuration files
│   ├── pipeline_config.yaml
│   ├── model_config.yaml
│   └── evaluation_config.yaml
│
├── data/                   # Data storage (gitignored)
│   ├── raw/
│   │   ├── sec_filings/
│   │   └── news_articles/
│   ├── processed/
│   │   ├── claims/
│   │   └── events/
│   └── market_data/
│
├── src/                    # Source code
│   ├── config.py           # Configuration loader
│   ├── database.py         # Database models and connection
│   ├── ingestion/          # Data collection modules
│   ├── parsing/            # Document parsing
│   ├── claim_extraction/   # NLP claim extraction
│   ├── claim_matching/     # Semantic matching
│   ├── contradiction_detection/  # NLI-based detection
│   ├── events/             # Event generation
│   ├── baselines/          # Baseline models
│   ├── modeling/           # Signal prediction
│   ├── evaluation/         # Metrics and analysis
│   ├── advanced_nlp/       # Topic modeling, RAG
│   └── visualization/      # Plotting utilities
│
├── app/                    # Gradio application
│   ├── gradio_app.py       # Main dashboard
│   └── components/         # UI components
│
├── scripts/                # Pipeline execution scripts
├── tests/                  # Unit and integration tests
├── reports/                # Generated reports
│   ├── figures/
│   └── tables/
└── notebooks/              # Jupyter notebooks
```

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- CUDA-capable GPU (optional, for faster inference)

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd corporate-narrative-engine
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download spaCy model**

```bash
python -m spacy download en_core_web_sm
```

5. **Configure environment variables**

```bash
cp .env.template .env
# Edit .env with your API keys and database credentials
```

6. **Set up the database**

```bash
# Create PostgreSQL database
createdb corporate_narrative_engine

# Run migrations
alembic upgrade head
```

## Configuration

### Required API Keys

| Service | Purpose | How to Get |
|---------|---------|------------|
| Anthropic | Claim extraction | https://console.anthropic.com/ |
| NewsAPI | News articles | https://newsapi.org/register |
| SEC EDGAR | SEC filings | Email only (no key) |

### Configuration Files

Edit `configs/pipeline_config.yaml` to customize:

```yaml
# Target companies (tickers)
companies:
  - AAPL
  - MSFT
  - GOOGL
  - TSLA
  - AMZN

# Lookback period (full 1 year)
lookback:
  start_date: "2025-03-08"
  end_date: "2026-03-08"

# Contradiction thresholds
contradiction:
  high_threshold: 0.8
  medium_threshold: 0.6

# Market reaction thresholds
market_reaction:
  positive_threshold: 0.015   # +1.5%
  negative_threshold: -0.015  # -1.5%
```

## Usage

### Run Full Pipeline

```bash
python scripts/run_full_pipeline.py
```

### Run Individual Steps

```bash
# 1. Data ingestion
python scripts/run_ingestion.py

# 2. Claim extraction
python scripts/run_extraction.py

# 3. Contradiction detection
python scripts/run_contradiction_detection.py

# 4. Train baselines
python scripts/run_baselines.py

# 5. Train signal model
python scripts/run_training.py

# 6. Evaluate
python scripts/run_evaluation.py
```

### Launch Dashboard

```bash
python app/gradio_app.py
```

Then open http://localhost:7860 in your browser.

## Dashboard Features

### Company Explorer
- View all extracted claims for a company
- Timeline visualization of statements over time
- Filter by topic, speaker, and source

### Contradiction Events
- Side-by-side comparison of contradicting claims
- Contradiction score and NLI breakdown
- Price reaction chart showing market impact

### Signal Dashboard
- Active signals with confidence levels
- Historical signal accuracy
- Win rate and return statistics

### Evaluation Reports
- Model vs baseline comparison
- ROC and precision-recall curves
- Signal decay analysis
- Robustness by topic and document type
- Calibration curves

## Key Concepts

### Claim Extraction

Statements are extracted from documents and structured as:

```json
{
  "claim_id": "c001",
  "company": "Tesla",
  "speaker": "Elon Musk",
  "speaker_role": "CEO",
  "claim_text": "Cybertruck production will reach 250,000 units by end of 2025",
  "topic": "projects",
  "direction": "positive",
  "timestamp": "2025-05-10",
  "source": "news"
}
```

### Contradiction Detection

Uses Natural Language Inference (NLI) to classify relationships:

- **Entailment**: Later claim confirms earlier claim
- **Neutral**: Claims are unrelated or compatible
- **Contradiction**: Later claim contradicts earlier claim

### Signal Generation

Signals are generated when:
- Contradiction score > 0.7 AND model confidence > 0.6 → **Bearish Alert**
- Contradiction score < 0.3 AND positive direction → **Bullish Consistency**

## Evaluation Methodology

### Temporal Splitting (No Leakage)

Data is split chronologically:
- Train: 67% (oldest)
- Validation: 16%
- Test: 17% (newest)

**Never** use random splits for financial time series.

### Metrics

| Category | Metrics |
|----------|---------|
| Classification | Precision, Recall, F1, ROC-AUC |
| Decision | Precision@K, False Alarm Rate |
| Financial | Abnormal Returns, Win Rate |
| Calibration | Brier Score, ECE |

### Baselines

1. **Keyword Rules**: Pattern-based bearish/bullish detection
2. **Sentiment Only**: FinBERT sentiment on latest statement
3. **Bag-of-Words**: TF-IDF + Logistic Regression

## Example Output

```
Company: Tesla (TSLA)
Event Date: 2025-08-15

Earlier Claim (2025-05-10, CEO Interview):
  "Cybertruck production will reach 250,000 units by end of 2025"

Later Claim (2025-08-15, 10-Q Filing):
  "Cybertruck production targets revised to 125,000 units due to
   supply chain constraints"

Contradiction Score: 0.87
Topic: projects
Speaker: CEO

Market Reaction:
  Next-day return: -4.2%
  Abnormal return: -3.8%

Model Signal: Bearish Alert
Confidence: 0.81

Explanation:
  Strong contradiction between CEO projection and subsequent filing.
  Historical analysis shows 76% of similar project-related
  contradictions preceded negative returns.
```

## Constraints and Safeguards

1. **Temporal Ordering**: All comparisons enforce `earlier.timestamp < later.timestamp`
2. **No Test Leakage**: Thresholds tuned only on train/validation data
3. **LLM Isolation**: Claude prompts never contain future prices or labels
4. **Minimum Samples**: Require n ≥ 30 per category for statistical validity
5. **Statistical Significance**: Report p-values and 95% CIs

## Development

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### Code Style

```bash
black src/ app/ scripts/
isort src/ app/ scripts/
flake8 src/ app/ scripts/
```

### Type Checking

```bash
mypy src/
```

## Troubleshooting

### Common Issues

**Database connection failed**
- Ensure PostgreSQL is running
- Check credentials in `.env`
- Verify database exists: `psql -l | grep corporate_narrative_engine`

**SEC rate limiting**
- The system respects 10 req/sec limit
- If blocked, wait 10 minutes and retry

**Out of memory during NLI**
- Reduce batch size in `configs/model_config.yaml`
- Use CPU instead of GPU: `TORCH_DEVICE=cpu`

**Claude API errors**
- Check API key in `.env`
- Verify account has credits
- Check rate limits (default: 60 req/min)

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Acknowledgments

- SEC EDGAR for free access to corporate filings
- Anthropic for Claude API
- HuggingFace for transformer models
- GDELT Project for news data
