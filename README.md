# ⚗️ Alchemist — Quantamental Commodity Intelligence Engine

> An institutional-grade quantamental research system for gold and silver, combining a calibrated ML prediction engine, regime-aware signal generation, RAG-powered document intelligence, and a stateful LangGraph agent — all served through a unified Streamlit analytics dashboard.

---

## Overview

Alchemist fuses two paradigms that are typically kept separate in industry:

- **Quantitative backbone** — a LightGBM classifier trained on triple-barrier event labels with purged walk-forward cross-validation, HMM regime detection, SHAP explainability, and a custom backtesting engine with realistic transaction costs
- **Intelligence layer** — a LangGraph agent that calls the trained ML model as a tool, queries institutional research documents via a hybrid RAG microservice, fetches live macro data, and uses an LLM to synthesize all signals into a structured investment brief

The result is a system that does not just describe current market conditions — it makes a calibrated, backtested, probabilistic prediction about whether GLD or SLV will hit a defined profit target before a stop-loss, explains exactly why via SHAP, and grounds that prediction in live macro data and institutional research.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY (natural language)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│               LANGGRAPH AGENT  (3 nodes)                     │
│                                                              │
│  Router Node ──► Tool Execution Node ──► Synthesizer Node   │
│                                                              │
│  Tools:                                                      │
│  1. AlchemistPredictionTool  — LightGBM inference           │
│  2. rag_wgc_tool             — RAG over WGC/research PDFs   │
│  3. tavily_search_tool       — Live news and events         │
│  4. fred_live_tool           — FRED macro context           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              ML PREDICTION ENGINE                            │
│                                                              │
│  Ingestion ──► Feature Store ──► Triple Barrier Labels      │
│      │              │                      │                 │
│  yfinance        Technical            LightGBM Classifier   │
│  FRED API        Macro (FRED)         + Monotonic Constraints│
│  Tavily          HMM Regime           + Probability Calib.  │
│                  RAG Sentiment        + Purged WF-CV        │
│                                                              │
│  Output: P(profit target hit) + regime + SHAP narrative     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              RAG MICROSERVICE (Flask + ngrok)                │
│                                                              │
│  LlamaParse ──► MarkdownNodeParser ──► ChromaDB             │
│  BM25 + Vector Hybrid Retrieval ──► FlagEmbedding Reranker  │
│  HyDE Query Transform ──► LlamaIndex Query Engine           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              STREAMLIT DASHBOARD  (5 tabs)                   │
│  Live Signal │ Backtest │ Explainability │ Regime │ Agent   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Technical Decisions

### Triple Barrier Labeling
Standard ML approaches label financial data as "price up in N days = 1". This ignores the path and misaligns labels with actual trading outcomes. Alchemist uses the Triple Barrier Method (López de Prado, *Advances in Financial Machine Learning*, 2018):

- **Upper barrier**: +2% profit target → label 1
- **Lower barrier**: −1% stop loss → label −1
- **Vertical barrier**: 7-day time limit → label 0

Labels reflect what actually happens to a trade — not abstract price direction. The asymmetric barriers (2% profit, 1% stop) reflect realistic risk management.

### Purged Walk-Forward Cross-Validation
Standard k-fold CV leaks future information on financial time series because label[t] covers prices t+1 through t+7. Alchemist uses purged walk-forward CV with an embargo gap equal to the prediction horizon. Training windows never overlap with test labels, eliminating temporal leakage entirely.

### LightGBM with Monotonic Constraints
Economic theory is encoded as model constraints:

| Feature | Constraint | Rationale |
|---|---|---|
| `real_yield` | −1 (bearish) | Higher opportunity cost → less gold demand |
| `cpi_yoy` | +1 (bullish) | Inflation hedge demand |
| `vix` | +1 (bullish) | Safe haven demand in fear regimes |
| `dollar_3m_pct` | −1 (bearish) | Inverse USD/gold relationship |
| `hy_spread_zscore` | +1 (bullish) | Risk-off → gold demand |

Constraints prevent the model from fitting spurious correlations that contradict known economic relationships, improving out-of-sample generalization.

### HMM Regime Detection
A 3-state Gaussian HMM trained on log returns + rolling volatility + rolling trend detects market regimes (BULL / BEAR / SIDEWAYS) from the statistical structure of the data — not from hardcoded rules. Regime becomes a feature in the ML model and a stratification variable for SHAP analysis.

### Probability Calibration
Raw LightGBM probabilities are overconfident. Isotonic regression calibration is applied on a held-out calibration set (separate from training and test). After calibration, a predicted probability of 70% corresponds to a ~70% historical accuracy rate on similar observations.

### RAG Architecture
Institutional PDF documents (WGC quarterly reports, earnings, research) are parsed with LlamaParse into Markdown, chunked with MarkdownNodeParser (512 tokens / 50 overlap), embedded with `BAAI/bge-small-en-v1.5`, and stored in ChromaDB. Retrieval uses a hybrid of vector search and BM25 keyword matching via QueryFusionRetriever (reciprocal rerank fusion), followed by `BAAI/bge-reranker-base` cross-encoder reranking, and HyDE (Hypothetical Document Embedding) query transformation for improved recall on vague queries.

---

## Project Structure

```
alchemist/
│
├── main.py                        # Training pipeline entry point
├── agent.py                       # Agent entry point
├── config.yaml                    # Central configuration
├── requirements.txt
├── .env.template
│
├── src/
│   ├── ingestion/
│   │   ├── market.py              # yfinance — incremental Parquet storage
│   │   ├── macro.py               # FRED API with publication lag correction
│   │   └── news.py                # Tavily weekly sentiment ingestion
│   │
│   ├── features/
│   │   ├── technical.py           # Price features: trend, momentum, volatility, volume
│   │   ├── macro_features.py      # FRED → derived signals, forward-filled to daily
│   │   ├── regime.py              # Gaussian HMM regime detector
│   │   ├── rag_features.py        # RAG document → numerical sentiment features
│   │   └── store.py               # Unified feature store, leakage-safe assembly
│   │
│   ├── models/
│   │   ├── labeling.py            # Triple barrier method + sample weights
│   │   └── lgbm_model.py          # LightGBM + monotonic constraints + purged WF-CV
│   │
│   ├── backtest/
│   │   └── engine.py              # Custom backtest engine with costs + Kelly sizing
│   │
│   ├── explainability/
│   │   └── shap_engine.py         # Global, local, and regime-conditional SHAP
│   │
│   ├── tracking/
│   │   └── mlflow_utils.py        # MLflow experiment tracking + model registry
│   │
│   ├── rag/
│   │   └── pipeline.py            # RAG pipeline module (from CIE)
│   │
│   └── agent/
│       ├── state.py               # Unified AgentState TypedDict
│       ├── tools.py               # 4 tools: Alchemist ML, RAG, Tavily, FRED
│       ├── nodes.py               # Router, executor, synthesizer nodes
│       └── graph.py               # LangGraph compilation + run_query()
│
└── dashboard/
    └── app.py                     # Streamlit dashboard (5 tabs)
```

---

## Stack

| Layer | Technology |
|---|---|
| ML Model | LightGBM 4.3+ |
| Regime Detection | hmmlearn (Gaussian HMM) |
| Probability Calibration | scikit-learn CalibratedClassifierCV |
| Explainability | SHAP TreeExplainer |
| Experiment Tracking | MLflow |
| Agent Orchestration | LangGraph |
| LLM | Groq (llama-3.3-70b-versatile) |
| RAG Framework | LlamaIndex |
| Vector Store | ChromaDB |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Reranker | BAAI/bge-reranker-base |
| Market Data | yfinance |
| Macro Data | FRED API (fredapi) |
| News | Tavily |
| Data Storage | Apache Parquet |
| Dashboard | Streamlit + Plotly |
| RAG Server | Flask + pyngrok |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/tejaaa69/Alpha-Engine-for-Commodities-.git
cd alchemist
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.template .env
```

Edit `.env` with your keys:

```
FRED_API_KEY=your_fred_api_key
TAVILY_API_KEY=your_tavily_api_key
GROQ_API_KEY=your_groq_api_key
```

FRED API key: [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) (free)
Tavily API key: [https://tavily.com](https://tavily.com) (free tier available)
Groq API key: [https://console.groq.com](https://console.groq.com) (free tier available)

### 3. Place RAG documents

Put your WGC reports or any financial PDFs into `rag_data/pdfs/`. The RAG pipeline will parse and index them during the first run.

---

## Usage

### Train the full pipeline

```bash
python main.py --mode train --symbol GLD
```

This runs in sequence:
1. Ingests market data (yfinance) and macro data (FRED)
2. Builds the feature store with lag-corrected FRED series
3. Fits the HMM regime detector
4. Computes triple barrier labels
5. Runs purged walk-forward cross-validation (LightGBM)
6. Calibrates probabilities (isotonic regression)
7. Computes SHAP values on the out-of-sample test set
8. Runs the backtest with transaction costs
9. Logs everything to MLflow

### Get today's signal

```bash
python main.py --mode predict --symbol GLD
```

### Run the agent (natural language interface)

```bash
python agent.py "What is the current gold outlook given real yields and WGC demand data?"
```

### Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501`.

### View MLflow experiment runs

```bash
mlflow ui --port 5000
```

Navigate to `http://localhost:5000`.

---

## Dashboard Tabs

| Tab | Content |
|---|---|
| 📡 Live Signal | Today's calibrated probability, current regime, SHAP waterfall for latest prediction |
| 📈 Backtest | Equity curve vs buy-and-hold, Sharpe/Sortino/drawdown metrics, trade log, barrier breakdown |
| 🔬 Explainability | Global SHAP importance, regime-conditional SHAP (how feature importance shifts across BULL/BEAR/SIDEWAYS) |
| 🌊 Regime Monitor | HMM regime history overlaid on price, regime distribution, current state |
| 🤖 Research Agent | Full LangGraph interface — ask any gold/silver question in natural language |

---

## Macro Factors

The feature store includes 8 FRED series with publication lag correction applied during ingestion:

| Series | Signal | Lag Applied |
|---|---|---|
| DFII10 | 10Y Real Yield (TIPS) | 1 day |
| CPIAUCSL | CPI Inflation (YoY derived) | 14 days |
| FEDFUNDS | Federal Funds Rate | 5 days |
| VIXCLS | VIX Fear Index | 1 day |
| DTWEXBGS | Trade-Weighted Dollar Index | 1 day |
| HOUST | Housing Starts | 18 days |
| T10Y2Y | Yield Curve (10Y−2Y) | 1 day |
| BAMLH0A0HYM2 | High-Yield Credit Spread | 1 day |

Publication lags prevent lookahead bias — a common source of inflated backtest performance in macro ML systems.

---

## Backtesting Assumptions

| Parameter | Value |
|---|---|
| Initial Capital | $100,000 |
| Position Size | 10% of capital per trade |
| Commission | 0.1% per side |
| Slippage | 0.05% per side |
| Entry | Next day open after signal |
| Exit | First barrier touched (profit / stop / time) |

Total round-trip cost: 0.3% per trade. All backtest signals are generated from walk-forward out-of-sample folds only — no training data is used in performance calculation.

---

## What This Is Not

- This is not financial advice
- Backtest results do not guarantee future performance
- The system is a research tool and learning project

---

## License

MIT
