# Corporate Narrative Consistency Engine — Project Documentation

> **Version:** 1.0.0 | **Language:** Python 3.10+ | **License:** MIT

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Question](#2-research-question)
3. [System Architecture](#3-system-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Tech Stack](#5-tech-stack)
6. [Installation & Setup](#6-installation--setup)
7. [Environment Variables](#7-environment-variables)
8. [Configuration Reference](#8-configuration-reference)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [Module Reference](#10-module-reference)
11. [Dashboard](#11-dashboard)
12. [ML Models & Evaluation](#12-ml-models--evaluation)
13. [Data Flow](#13-data-flow)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Testing](#15-testing)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Project Overview

The **Corporate Narrative Consistency Engine** is an end-to-end AI system that:

- **Collects** SEC filings (8-K, 10-K, 10-Q) and executive statements from news sources
- **Extracts** structured claims using the Claude API (Anthropic)
- **Detects** contradictions and narrative shifts across time using Natural Language Inference (NLI)
- **Aligns** contradiction events with short-term stock price movements via `yfinance`
- **Predicts** whether a contradiction signals a bullish, bearish, or neutral market reaction using XGBoost
- **Explains** every prediction with SHAP values and templated natural language

The system is built for financial research, not live trading. All design decisions are guided by reproducibility, temporal integrity, and statistical rigour.

---

## 2. Research Question

> *Do contradictions or narrative shifts in corporate disclosures and executive statements predict short-term stock price movements?*

### Why This Matters

Corporate communications — earnings calls, SEC filings, investor letters — are a rich source of forward-looking statements. When a CEO says one thing publicly and a subsequent 10-Q says another, the market often reacts. This project quantifies that phenomenon at scale.

### Signal Logic

| Condition | Signal |
|---|---|
| Contradiction score > 0.7 **AND** model confidence > 0.6 | **Bearish Alert** |
| Contradiction score < 0.3 **AND** positive claim direction | **Bullish Consistency** |
| Neither condition met | **Neutral** |

---

## 3. System Architecture

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

---

## 4. Repository Structure

```
beyond2/
├── .env.template               # Environment variables template (copy to .env)
├── .gitignore
├── README.md
├── PROJECT_DOCS.md             # This file
├── requirements.txt            # All Python dependencies
├── pyproject.toml              # Package metadata and tool config
├── batch_results.json          # Output from batch pipeline runs
│
├── configs/                    # YAML configuration files
│   ├── pipeline_config.yaml    # Data sources, thresholds, companies
│   ├── model_config.yaml       # ML model params, features, baselines
│   └── evaluation_config.yaml  # Metrics, splits, robustness settings
│
├── src/                        # Core source code
│   ├── config.py               # Config manager (singleton, Pydantic)
│   ├── database.py             # SQLAlchemy models & DB connection
│   │
│   ├── ingestion/              # Data collection
│   │   ├── sec_downloader.py       # SEC EDGAR 8-K/10-K/10-Q downloader
│   │   ├── news_scraper_gdelt.py   # GDELT news scraper
│   │   ├── news_scraper_newsapi.py # NewsAPI scraper
│   │   ├── news_scraper_rss.py     # RSS feed scraper
│   │   ├── market_data.py          # yfinance market data fetcher
│   │   ├── batch_processor.py      # Parallel batch ingestion
│   │   ├── company_universe.py     # Company ticker registry
│   │   └── source_registry.py      # Unified source interface
│   │
│   ├── parsing/                # Document parsing
│   │   ├── sec_parser.py           # SEC filing text extraction
│   │   ├── news_parser.py          # News article parsing
│   │   └── document_cleaner.py     # Text cleaning utilities
│   │
│   ├── claim_extraction/       # NLP claim extraction
│   │   ├── claim_extractor.py      # Claude-powered claim extractor
│   │   ├── claim_normalizer.py     # Normalises claim text
│   │   └── entity_extractor.py     # Named entity extraction (spaCy)
│   │
│   ├── claim_matching/         # Semantic claim linking
│   │   ├── embedding_matcher.py    # sentence-transformers embeddings
│   │   ├── temporal_linker.py      # Enforces earlier→later ordering
│   │   └── topic_matcher.py        # Topic-aware matching
│   │
│   ├── contradiction_detection/ # NLI-based detection
│   │   ├── nli_detector.py         # DeBERTa NLI inference
│   │   ├── contradiction_scorer.py # Aggregates NLI scores
│   │   └── relationship_classifier.py # Final relationship label
│   │
│   ├── events/                 # Event construction
│   │   ├── event_builder.py        # Builds contradiction events
│   │   ├── event_dataset.py        # Dataset wrapper
│   │   └── market_alignment.py     # Links events to price returns
│   │
│   ├── baselines/              # Baseline models
│   │   ├── keyword_rules.py        # Pattern-based bearish/bullish rules
│   │   ├── sentiment_baseline.py   # FinBERT sentiment baseline
│   │   └── bow_classifier.py       # TF-IDF + Logistic Regression
│   │
│   ├── modeling/               # Signal prediction
│   │   ├── signal_model.py         # XGBoost signal model
│   │   ├── feature_builder.py      # Feature engineering
│   │   └── explainability.py       # SHAP explanations
│   │
│   ├── evaluation/             # Metrics & analysis
│   │   ├── metrics.py              # Precision, Recall, F1, AUC, etc.
│   │   ├── temporal_split.py       # Chronological train/val/test split
│   │   ├── robustness.py           # Robustness tests by topic/source
│   │   └── calibration.py          # Brier score, ECE, calibration curves
│   │
│   ├── advanced_nlp/           # Advanced NLP components
│   │   ├── topic_model.py          # BERTopic topic modelling
│   │   ├── rag_reasoning.py        # RAG-based evidence retrieval
│   │   └── llm_hybrid.py           # LLM hybrid scoring
│   │
│   └── visualization/          # Plotting utilities
│       └── plots.py                # Plotly/Matplotlib charting helpers
│
├── app/                        # Gradio web dashboard
│   ├── gradio_app.py           # Main dashboard entrypoint
│   └── streamlit_app.py        # Alternative Streamlit dashboard
│
├── scripts/                    # Pipeline execution scripts
│   ├── run_full_pipeline.py    # Run everything end-to-end
│   └── run_batch_pipeline.py   # Batch mode for large company sets
│
└── tests/                      # Test suite
    ├── test_config.py
    ├── test_claim_extraction.py
    └── test_contradiction_detection.py
```

---

## 5. Tech Stack

| Category | Library | Purpose |
|---|---|---|
| **LLM** | `anthropic` (Claude 3.5 Sonnet) | Claim extraction from documents |
| **NLI** | `transformers` (DeBERTa-v3-large-mnli) | Contradiction classification |
| **Embeddings** | `sentence-transformers` (all-mpnet-base-v2) | Semantic claim matching |
| **Sentiment** | `transformers` (ProsusAI/finbert) | Baseline sentiment model |
| **Topic Modelling** | `bertopic`, `umap-learn`, `hdbscan` | Topic clustering |
| **ML Model** | `xgboost`, `lightgbm` | Signal prediction |
| **Explainability** | `shap` | Feature importance & explanations |
| **Market Data** | `yfinance` | Stock price fetching |
| **Data** | `pandas`, `numpy`, `pyarrow` | Data processing |
| **Database** | `sqlalchemy`, `psycopg2`, `alembic` | PostgreSQL ORM + migrations |
| **Web Scraping** | `beautifulsoup4`, `requests`, `aiohttp`, `feedparser` | News & SEC scraping |
| **NLP Tooling** | `spacy`, `nltk` | Entity extraction, tokenisation |
| **Statistics** | `scipy`, `statsmodels` | Significance tests, p-values |
| **Dashboard** | `gradio`, `plotly` | Interactive web UI |
| **Config** | `pydantic`, `pydantic-settings`, `pyyaml` | Validated config management |
| **Logging** | `loguru` | Structured logging |
| **Testing** | `pytest`, `pytest-cov`, `pytest-asyncio` | Test suite |
| **Code Quality** | `black`, `isort`, `flake8`, `mypy` | Formatting & type checking |

---

## 6. Installation & Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- CUDA-capable GPU *(optional — speeds up NLI inference significantly)*

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/akhilmishr13/beyond2.git
cd beyond2

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Download the spaCy English model (required for entity extraction)
python -m spacy download en_core_web_sm

# 5. Set up environment variables
cp .env.template .env
# Edit .env with your credentials (see Section 7)

# 6. Create the PostgreSQL database
createdb corporate_narrative_engine

# 7. Run database migrations
alembic upgrade head
```

---

## 7. Environment Variables

Copy `.env.template` to `.env` and fill in the following:

### Required

| Variable | Description | Example |
|---|---|---|
| `POSTGRES_HOST` | DB host | `localhost` |
| `POSTGRES_PORT` | DB port | `5432` |
| `POSTGRES_DB` | Database name | `corporate_narrative_engine` |
| `POSTGRES_USER` | DB username | `your_username` |
| `POSTGRES_PASSWORD` | DB password | `your_secure_password` |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | `sk-ant-...` |
| `SEC_USER_AGENT` | Your name + email (required by SEC) | `John Doe john@example.com` |

### Optional

| Variable | Description | Default |
|---|---|---|
| `NEWSAPI_KEY` | NewsAPI.org key (free tier: 100 req/day) | — |
| `OPENAI_API_KEY` | OpenAI key (for alternative embeddings) | — |
| `HUGGINGFACE_TOKEN` | HuggingFace token (for gated models) | — |
| `TORCH_DEVICE` | Force inference device (`cuda`, `mps`, `cpu`) | auto-detected |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `NUM_WORKERS` | Parallel worker count | `4` |
| `CACHE_DIR` | Model/data cache directory | `.cache` |
| `GRADIO_SERVER_PORT` | Dashboard port | `7860` |
| `GRADIO_SHARE` | Enable public Gradio link | `false` |

> **Security:** Never commit `.env` to version control. It is already listed in `.gitignore`.

---

## 8. Configuration Reference

All pipeline behaviour is controlled by three YAML files in `configs/`.

### `configs/pipeline_config.yaml`

| Section | Key Setting | Description |
|---|---|---|
| `companies` | List of tickers | Companies to analyse (AAPL, TSLA, etc.) |
| `lookback` | `start_date` / `end_date` | Analysis window (default: 1 year) |
| `sec.filing_types` | `["8-K", "10-K", "10-Q"]` | SEC form types to download |
| `sec.rate_limit_per_second` | `10` | SEC API rate limit (do not exceed) |
| `news.sources.gdelt.enabled` | `true` | Enable GDELT news source |
| `news.sources.newsapi.enabled` | `true` | Enable NewsAPI source |
| `news.sources.rss.feeds` | List of URLs | RSS feeds to monitor |
| `market.return_windows` | `[1, 3, 5, 10]` | Return windows in trading days |
| `market.benchmark_ticker` | `SPY` | Benchmark for abnormal return calculation |
| `claim_extraction.model` | `claude-3-5-sonnet-20241022` | Claude model version |
| `claim_extraction.topic_categories` | 10 categories | Claim topic taxonomy |
| `claim_matching.similarity_threshold` | `0.75` | Min cosine similarity to link claims |
| `contradiction.nli_model` | `microsoft/deberta-v3-large-mnli` | NLI transformer model |
| `contradiction.thresholds.high` | `0.8` | Strong contradiction threshold |
| `processing.checkpoint_enabled` | `true` | Save progress checkpoints |

### `configs/model_config.yaml`

| Section | Description |
|---|---|
| `features` | Feature groups: contradiction scores, source type, topic, market context |
| `baselines.keyword_rules` | Bearish/bullish keyword lists |
| `baselines.sentiment` | FinBERT model + thresholds |
| `baselines.bow` | TF-IDF + LR/SVM parameters |
| `signal_model.xgboost` | XGBoost hyperparameters (500 trees, max_depth 6, lr 0.05) |
| `signal_model.thresholds` | Bearish/bullish decision thresholds |
| `training.hyperparameter_tuning` | Optuna tuning (50 trials, 1-hour timeout) |
| `explainability.shap` | SHAP plot types and sample size |

### `configs/evaluation_config.yaml`

Controls temporal splits (67/16/17%), metric computation, robustness analysis, and calibration.

---

## 9. Running the Pipeline

### Full End-to-End Run

```bash
python scripts/run_full_pipeline.py
```

With options:

```bash
# Target specific companies
python scripts/run_full_pipeline.py --companies AAPL TSLA NVDA

# Skip ingestion (use previously collected data)
python scripts/run_full_pipeline.py --skip-ingestion
```

### Run Individual Steps

```bash
# Step 1 — Data ingestion (SEC + news + market data)
python scripts/run_ingestion.py

# Step 2 — Claim extraction via Claude
python scripts/run_extraction.py

# Step 3 — Contradiction detection via DeBERTa NLI
python scripts/run_contradiction_detection.py

# Step 4 — Train baseline models
python scripts/run_baselines.py

# Step 5 — Train XGBoost signal model
python scripts/run_training.py

# Step 6 — Evaluate all models
python scripts/run_evaluation.py
```

### Batch Mode

For large company sets, use the batch pipeline which handles checkpointing and parallel processing:

```bash
python scripts/run_batch_pipeline.py
```

---

## 10. Module Reference

### `src/config.py` — Configuration Manager

Central singleton config object. Access anywhere with:

```python
from src.config import get_config
config = get_config()

# Examples
companies = config.pipeline.companies          # ['AAPL', 'MSFT', ...]
model_name = config.pipeline.claim_extraction.model
db_url = config.database.url
device = config.get_device()                   # auto-detects cuda/mps/cpu
```

To force reload after editing YAML:

```python
from src.config import reload_config
config = reload_config()
```

---

### `src/ingestion/` — Data Collection

**`SECDownloader`** — downloads SEC filings with rate limiting and retry logic:

```python
from src.ingestion.sec_downloader import SECDownloader

downloader = SECDownloader()
filings = downloader.download_filings(
    ticker="AAPL",
    filing_types=["8-K", "10-K"],
    start_date="2025-03-08",
    end_date="2026-03-08"
)
```

Each `SECFiling` object contains: `ticker`, `filing_type`, `filing_date`, `accession_number`, `raw_text`, `sections`.

**`SourceRegistry`** — unified interface across all sources:

```python
from src.ingestion import SourceRegistry

registry = SourceRegistry()
documents = list(registry.collect_all_documents(
    ticker="TSLA",
    company_name="Tesla Inc.",
    start_date="2025-03-08",
    end_date="2026-03-08"
))
```

---

### `src/claim_extraction/` — NLP Claim Extraction

Uses Claude 3.5 Sonnet to extract structured claims. Each extracted claim is a JSON object:

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

**Topic categories:** `guidance`, `projects`, `legal_regulatory`, `supply_chain`, `capital_expenditure`, `risk_factors`, `personnel`, `customers`, `products`, `other`

---

### `src/claim_matching/` — Semantic Matching

Links claims about the same subject across time using `sentence-transformers` (model: `all-mpnet-base-v2`).

- **Temporal guard:** Only links `earlier_claim.timestamp < later_claim.timestamp`
- **Similarity threshold:** 0.75 cosine similarity (configurable)
- **Lookback window:** 180 days (configurable)

---

### `src/contradiction_detection/` — NLI Detection

Uses `microsoft/deberta-v3-large-mnli` to classify claim pairs:

```python
from src.contradiction_detection.nli_detector import NLIDetector

detector = NLIDetector()
result = detector.classify(
    premise="Revenue will grow 15% this year",
    hypothesis="Revenue expected to decline 5% due to headwinds"
)
# result.contradiction → float (0–1)
# result.predicted_label → "contradiction" | "neutral" | "entailment"
```

**Output labels:**

| Label | Meaning |
|---|---|
| `entailment` | Later claim confirms earlier claim |
| `neutral` | Claims are compatible or unrelated |
| `contradiction` | Later claim directly contradicts earlier claim |

---

### `src/modeling/signal_model.py` — Signal Prediction

XGBoost classifier with 3 output classes: `positive`, `neutral`, `negative`.

```python
from src.modeling.signal_model import SignalModel

model = SignalModel()
model.fit(X_train, y_train)
prediction = model.predict_single(features)
# prediction.signal → "bearish" | "bullish" | "neutral"
# prediction.confidence → float
```

**Input features:**

- `contradiction_score`, `entailment_score`, `neutral_score`
- `semantic_similarity`, `claim_age_days`
- `source_type_encoded`, `document_type_encoded`, `speaker_role_encoded`
- `topic_encoded`, `topic_sensitivity_score`
- `prior_30d_return`, `prior_30d_volatility`, `volume_ratio`
- `market_return_same_day`, `sector_return_same_day`

---

## 11. Dashboard

### Launch

```bash
python app/gradio_app.py
# Open http://localhost:7860
```

Alternative Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

### Tabs

| Tab | What It Shows |
|---|---|
| **Company Explorer** | All extracted claims for a company, timeline, filter by topic/speaker/source |
| **Contradiction Events** | Side-by-side claim comparisons, NLI score breakdown, price reaction chart |
| **Signal Dashboard** | Active signals with confidence levels, historical accuracy, win rate |
| **Evaluation Reports** | Model vs baseline metrics, ROC curves, signal decay, calibration curves |

---

## 12. ML Models & Evaluation

### Baseline Models

| Baseline | Method |
|---|---|
| **Keyword Rules** | Bearish/bullish keyword pattern matching on claim text |
| **Sentiment Only** | FinBERT (`ProsusAI/finbert`) sentiment on most recent claim |
| **Bag-of-Words** | TF-IDF (max 5000 features, 1–2 ngrams) + Logistic Regression or Linear SVM |

### Main Model

**XGBoost** with 500 estimators, max_depth 6, learning rate 0.05. Hyperparameters tuned with Optuna (50 trials). Explainability via SHAP.

### Temporal Splitting — No Data Leakage

Data is always split **chronologically**, never randomly:

| Split | Proportion | Contents |
|---|---|---|
| Train | 67% | Oldest events |
| Validation | 16% | Middle period |
| Test | 17% | Most recent events |

A 5-day gap between splits prevents look-ahead leakage at the boundaries.

### Evaluation Metrics

| Category | Metrics |
|---|---|
| **Classification** | Precision, Recall, F1, ROC-AUC |
| **Decision Quality** | Precision@K, False Alarm Rate |
| **Financial** | Abnormal Returns, Win Rate |
| **Calibration** | Brier Score, Expected Calibration Error (ECE) |
| **Statistical** | p-values, 95% confidence intervals |

---

## 13. Data Flow

```
Raw Documents (SEC filings, news articles)
        │
        ▼
Document Parsing & Cleaning
(sec_parser.py, news_parser.py, document_cleaner.py)
        │
        ▼
Claim Extraction
(Claude 3.5 Sonnet → structured JSON claims)
        │
        ▼
Claim Matching
(all-mpnet-base-v2 embeddings → semantically linked claim pairs)
        │
        ▼
Contradiction Detection
(DeBERTa NLI → entailment / neutral / contradiction scores)
        │
        ▼
Event Construction
(event_builder.py → contradiction events with metadata)
        │
        ▼
Market Alignment
(yfinance → 1/3/5/10-day returns, abnormal returns vs SPY)
        │
        ▼
Feature Engineering
(feature_builder.py → 15 features per event)
        │
        ▼
Signal Prediction
(XGBoost → bearish / bullish / neutral + confidence)
        │
        ▼
SHAP Explanation + Dashboard
```

---

## 14. Key Design Decisions

### 1. Claude for Claim Extraction (not rule-based)

SEC filings are complex legal documents with varied phrasing. Claude's instruction-following allows extraction of structured, typed claims without hand-crafted regex patterns. Temperature is set to `0.0` for deterministic output.

### 2. DeBERTa for NLI (not GPT)

`microsoft/deberta-v3-large-mnli` is a purpose-trained NLI model with strong performance on contradiction detection benchmarks. It runs locally (no API cost, no rate limits) and supports batch inference.

### 3. Temporal Guard on All Comparisons

Every claim pair is enforced to satisfy `earlier.timestamp < later.timestamp`. This prevents the model from "detecting" contradictions based on re-statements that actually came before the original.

### 4. Chronological Splits Only

Financial time series cannot use random cross-validation. All splits, hyperparameter tuning, and threshold selection are done on train/validation data only. The test set is touched exactly once.

### 5. Abnormal Returns (not raw returns)

Market-wide moves are subtracted using SPY as a benchmark. This isolates the company-specific price reaction from macro noise.

### 6. Minimum Sample Guard

No metric is reported for categories with fewer than 30 samples. This prevents misleading results from rare categories.

---

## 15. Testing

### Run the Test Suite

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src

# Run a specific test file
pytest tests/test_contradiction_detection.py -v
```

### Test Files

| File | Coverage |
|---|---|
| `test_config.py` | Config loading, validation, env var parsing |
| `test_claim_extraction.py` | Claim extraction logic, JSON schema |
| `test_contradiction_detection.py` | NLI detection, temporal guard |

### Code Quality

```bash
# Auto-format
black src/ app/ scripts/
isort src/ app/ scripts/

# Lint
flake8 src/ app/ scripts/

# Type checking
mypy src/
```

---

## 16. Troubleshooting

### Database connection failed

```
sqlalchemy.exc.OperationalError: could not connect to server
```

- Verify PostgreSQL is running: `pg_isready`
- Check credentials in `.env` match your PostgreSQL user
- Confirm database exists: `psql -l | grep corporate_narrative_engine`
- If not, create it: `createdb corporate_narrative_engine`

---

### SEC rate limiting / 403 errors

The SEC requires a valid `User-Agent` header. Set in `.env`:

```
SEC_USER_AGENT=Your Name your.email@example.com
```

The system enforces ≤10 requests/second automatically. If blocked, wait 10 minutes before retrying.

---

### Out of memory during NLI inference

- Reduce batch size in `configs/model_config.yaml`: `batch_size: 8` (or lower)
- Force CPU inference: `TORCH_DEVICE=cpu` in `.env`
- On Apple Silicon, MPS is auto-detected and used if available

---

### Claude API errors

- Verify `ANTHROPIC_API_KEY` is set correctly in `.env`
- Check your Anthropic account has available credits
- Default rate limit is 60 requests/minute — reduce `max_concurrent_requests` if hitting limits

---

### GDELT returns no results

GDELT has its own rate limits and geographic coverage. If results are sparse:
- Try a larger `lookback_days` window
- Fall back to NewsAPI + RSS sources
- Check GDELT status at https://www.gdeltproject.org/

---

*Last updated: April 2026*
