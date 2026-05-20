"""
src/features/rag_features.py

Bridges the RAG pipeline into the ML feature store.
Extracts topic sentiment scores and document freshness from institutional reports.
"""

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import requests
from loguru import logger

RAG_QUERIES = {
    "central_bank_demand":  "central bank gold buying reserves accumulation",
    "etf_flows":            "gold ETF fund flows investor demand",
    "mine_supply":          "gold mine supply production output",
    "jewelry_demand":       "gold jewelry demand India China consumption",
    "investment_demand":    "gold investment demand bar coin outlook",
    "silver_industrial":    "silver industrial demand solar panels electronics",
}

class RAGFeatureExtractor:
    def __init__(self, cfg: Dict[str, Any]):
        # Safely pull URL from config, default to localhost if not found
        self.rag_url = cfg.get("rag", {}).get("server_url", "http://127.0.0.1:5001")
        self.is_connected = self._health_check()

    def _health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.rag_url}/health", timeout=5)
            if resp.status_code == 200:
                logger.info("✅ RAG microservice connected successfully.")
                return True
        except Exception:
            logger.warning(f"⚠️ RAG server not reachable at {self.rag_url}. RAG features will be NaN.")
        return False

    def _query_rag(self, query: str) -> Optional[dict]:
        if not self.is_connected:
            return None
        try:
            resp = requests.post(f"{self.rag_url}/query", json={"query": query}, timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"RAG query failed for '{query}': {e}")
        return None

    def _score_text(self, text: str) -> float:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
        except ImportError:
            bullish = ["increase", "rise", "strong", "demand", "growth", "record", "high", "accumulation", "inflow"]
            bearish = ["decline", "fall", "weak", "outflow", "low", "decrease", "contraction", "selling"]
            text_l  = text.lower()
            b = sum(text_l.count(w) for w in bullish)
            r = sum(text_l.count(w) for w in bearish)
            total = b + r
            return (b - r) / total if total > 0 else 0.0

    def _score_source_quality(self, sources: list) -> float:
        if not sources:
            return 0.0
        scores = [s.get("score", 0) for s in sources]
        return float(np.mean(scores))

    def extract_current_signals(self) -> dict:
        signals = {}
        for feat_name, query in RAG_QUERIES.items():
            result = self._query_rag(query)
            if result is None:
                signals[f"rag_{feat_name}_sentiment"] = np.nan
                signals[f"rag_{feat_name}_confidence"] = np.nan
                continue

            answer  = result.get("answer", "")
            sources = result.get("sources", [])

            signals[f"rag_{feat_name}_sentiment"]  = self._score_text(answer)
            signals[f"rag_{feat_name}_confidence"] = self._score_source_quality(sources)

            logger.info(f"RAG[{feat_name}]: Sent={signals[f'rag_{feat_name}_sentiment']:+.3f}, Conf={signals[f'rag_{feat_name}_confidence']:.3f}")
        return signals

    def get_composite_score(self) -> float:
        signals = self.extract_current_signals()
        weights = {
            "rag_central_bank_demand_sentiment": 0.30,
            "rag_etf_flows_sentiment":           0.25,
            "rag_investment_demand_sentiment":   0.20,
            "rag_jewelry_demand_sentiment":      0.15,
            "rag_mine_supply_sentiment":        -0.10, 
        }

        total_weight = 0.0
        weighted_sum = 0.0
        for key, weight in weights.items():
            val = signals.get(key, np.nan)
            if not np.isnan(val):
                weighted_sum += val * weight
                total_weight += abs(weight)

        return round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0