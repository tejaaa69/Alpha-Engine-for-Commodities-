"""
src/agent/tools.py

Four tools. No redundancy. Clean responsibilities.
Tightly coupled to the Alchemist MLOps Pipeline.
"""

import os
from typing import Optional

import requests as http_requests
import yaml
from langchain_core.tools import tool
from loguru import logger


def _load_cfg() -> dict:
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    with open(root / "config.yaml") as f:
        return yaml.safe_load(f)

# TOOL 1: Alchemist ML Prediction (Bridge)
@tool
def alchemist_prediction_tool(symbol: str) -> str:
    """
    Run the Alchemist ML model for a given asset symbol (e.g. GLD, SLV).
    Returns calibrated probability of hitting the profit target,
    current market regime, top SHAP drivers, and backtest context.
    Use this as the primary signal source for any quantitative query.
    """
    logger.info(f"[TOOL: ALCHEMIST ML] Running live prediction for {symbol}")
    try:
        import mlflow
        from src.features.store import FeatureStore
        from src.tracking.mlflow_utils import AlchemistTracker
        from src.explainability.shap_engine import SHAPExplainer

        cfg = _load_cfg()

        # ── Symbol‑aware tracker and model loading ─────────────────
        tracker = AlchemistTracker(cfg, symbol=symbol)
        model = tracker.load_production_model()
        if not model:
            return f"Error: No Production/Staging model found in MLflow Registry for {symbol}."

        # ── Live inference features ────────────────────────────────
        store = FeatureStore(cfg)
        features_df = store.build_inference(symbol)
        feature_cols = model.feature_cols

        missing = [c for c in feature_cols if c not in features_df.columns]
        if missing:
            return f"Error: Feature store missing required columns: {missing[:3]}..."

        latest_features = features_df[feature_cols].iloc[[-1]]
        latest_date = features_df.index[-1]

        # ── Robust probability extraction ──────────────────────────
        raw_prob = model.predict_proba(latest_features)
        try:
            prob = float(raw_prob[0, 1])      # standard (1, 2) array
        except (IndexError, TypeError):
            try:
                prob = float(raw_prob[0])     # 1D array / Series
            except (IndexError, TypeError):
                prob = float(raw_prob)        # scalar

        signal = "BUY SIGNAL (+2% TARGET EXPECTED)" if prob >= 0.55 else "FLAT / NO SIGNAL"

        # ── Regime ─────────────────────────────────────────────────
        regime = "UNKNOWN"
        if "regime_code" in features_df.columns:
            code = int(features_df["regime_code"].iloc[-1])
            regime_map = {
                0: "LOW_VOL (Bull)",
                1: "MID_VOL (Transition)",
                2: "HIGH_VOL (Crisis/Bear)",
            }
            regime = regime_map.get(code, "UNKNOWN")

        # ── SHAP ───────────────────────────────────────────────────
        logger.info("[TOOL: ALCHEMIST ML] Generating Live SHAP narrative...")
        background = features_df[feature_cols].tail(100).fillna(0)
        explainer = SHAPExplainer(model.model, feature_cols)
        explainer.fit(background)
        explainer.compute(latest_features)
        shap_narrative = explainer.get_prediction_narrative(0)

        # ── Backtest stats from the same run ──────────────────────
        client = mlflow.MlflowClient()
        bt_sharpe, bt_win_rate = "N/A", "N/A"

        versions = client.get_latest_versions(
            tracker.model_name, stages=["Production", "Staging"]
        )
        if versions:
            run_id = versions[0].run_id
            run_data = client.get_run(run_id).data.metrics
            bt_sharpe = f"{run_data.get('bt_sharpe_ratio', 0):.2f}"
            bt_win_rate = f"{run_data.get('bt_win_rate', 0):.1%}"

        # ── Final output ───────────────────────────────────────────
        output = (
            f"ALCHEMIST QUANT ENGINE SIGNAL | Asset: {symbol} | Date: {latest_date.strftime('%Y-%m-%d')}\n"
            f"--------------------------------------------------\n"
            f"Probability (+2% Target): {prob:.1%}\n"
            f"Actionable Signal:        {signal}\n"
            f"Current Market Regime:    {regime}\n\n"
            f"Mathematical Drivers (SHAP):\n{shap_narrative}\n\n"
            f"Model Historical Stats (Out of Sample):\n"
            f"Sharpe Ratio: {bt_sharpe} | Win Rate: {bt_win_rate}\n"
            f"--------------------------------------------------\n"
            f"Synthesizer Instruction: Treat this quantitative probability as the primary ground-truth anchor for your final report."
        )
        return output

    except Exception as e:
        logger.error(f"[ALCHEMIST ML] Failed: {e}")
        return f"AlchemistPredictionTool encountered a critical error: {type(e).__name__}: {e}"

# TOOL 2: RAG Document Search

@tool
def rag_wgc_tool(query: str) -> str:
    """
    Search internal financial PDF documents — WGC reports, earnings,
    commodity research. Use for: gold demand data, ETF flows,
    central bank buying, mine supply, analyst price targets.
    """
    logger.info(f"[TOOL: RAG] Querying: '{query}'")
    cfg = _load_cfg()
    rag_url = cfg.get("rag", {}).get("server_url", "http://127.0.0.1:5001")
    
    try:
        resp = http_requests.post(
            f"{rag_url}/query",
            json={"query": query},
            timeout=60,
        )
        if resp.status_code != 200:
            return f"RAG server error {resp.status_code}: {resp.text}"

        data = resp.json()
        answer = data.get("answer", "No answer returned.")
        sources = data.get("sources", [])

        source_lines = [
            f"  [Doc {s.get('index', '?')}] {s.get('file', 'Unknown')} (score: {s.get('score', 0):.2f})\n  {s.get('preview', '')}"
            for s in sources[:3] # Limit to top 3 to save context window
        ]
        return (
            f"Internal Document Analysis for '{query}':\n{answer}\n\n"
            f"Source Evidence:\n"
            + ("\n".join(source_lines) if source_lines else "  No documents found.")
        )
    except http_requests.exceptions.ConnectionError:
        return "RAG server not reachable. Qualitative institutional data is unavailable."
    except Exception as e:
        return f"RAG tool error: {type(e).__name__}: {e}"


# TOOL 3: Tavily Live News

@tool
def tavily_search_tool(query: str) -> str:
    """
    Live web search for current news, geopolitical events, and recent analyst commentary.
    """
    logger.info(f"[TOOL: TAVILY] Searching: '{query}'")
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
        response = client.search(query=query, search_depth="advanced")
        results = "\n".join([f"- {r['content'][:300]}..." for r in response.get("results", [])])
        return f"Live Web Data for '{query}':\n{results}"
    except Exception as e:
        return f"Tavily search failed: {e}"


# TOOL 4: FRED Live Data

@tool
def fred_live_tool(series_id: str) -> str:
    """
    Fetch the latest value of a FRED macroeconomic series for LLM context.
    Common IDs: FEDFUNDS, DFII10, CPIAUCSL.
    """
    logger.info(f"[TOOL: FRED LIVE] Fetching {series_id}")
    try:
        from fredapi import Fred
        from datetime import datetime, timedelta

        fred = Fred(api_key=os.environ.get("FRED_API_KEY", ""))
        end = datetime.today()
        start = end - timedelta(days=90)

        data = fred.get_series(series_id, observation_start=start.strftime("%Y-%m-%d")).dropna()

        if data.empty:
            return f"FRED {series_id}: No data returned."

        latest_val = round(float(data.iloc[-1]), 4)
        latest_date = data.index[-1].strftime("%Y-%m-%d")
        
        prev_val = round(float(data.iloc[-2]), 4) if len(data) >= 2 else None
        change = round(latest_val - prev_val, 4) if prev_val is not None else None

        return (
            f"FRED {series_id} (as of {latest_date}): {latest_val}"
            + (f" (Change from prior: {change:+.4f})" if change else "")
        )
    except Exception as e:
        return f"FRED live fetch failed for {series_id}: {e}"


# Tool registry
TOOL_MAP = {
    "alchemist_prediction": alchemist_prediction_tool,
    "rag_wgc":              rag_wgc_tool,
    "tavily_search":        tavily_search_tool,
    "fred_live":            fred_live_tool,
}
