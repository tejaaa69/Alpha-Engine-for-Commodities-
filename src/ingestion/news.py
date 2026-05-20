"""
src/ingestion/news.py

Fetches weekly news summaries for gold and silver via Tavily.
Extracts a simple sentiment proxy and stores as weekly parquet.

We do NOT attempt fancy NLP here — we use a pre-trained
sentiment model (vader) for a quick polarity score.
The score then becomes a numeric feature in the feature store.

Note on History: Free news APIs do not easily backfill 10 years of data. 
LightGBM natively handles NaNs, so this feature will be sparsely populated 
in the deep historical backtest but active for recent/live predictions.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from loguru import logger


class NewsIngestion:
    def __init__(self, cfg: Dict[str, Any]):
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY is missing from the .env file! "
                "Get a free key at https://tavily.com/"
            )
            
        from tavily import TavilyClient
        self.client    = TavilyClient(api_key=api_key)
        self.templates = cfg["tavily"]["query_templates"]
        
        root_dir = Path(__file__).resolve().parent.parent.parent
        self.raw_dir   = root_dir / cfg["paths"]["raw_data"]
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _parquet_path(self) -> Path:
        return self.raw_dir / "news_sentiment.parquet"

    def _sentiment_score(self, text: str) -> float:
        """
        Fast polarity score using VADER (no GPU needed).
        Returns compound score: -1.0 (very negative) to +1.0 (very positive).
        """
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            return analyzer.polarity_scores(text)["compound"]
        except ImportError:
            logger.warning("vaderSentiment not installed. Using simple keyword fallback.")
            bullish = ["rally", "surge", "gain", "rise", "demand", "strong", "buy", "bull", "up"]
            bearish = ["fall", "drop", "decline", "weak", "sell", "pressure", "low", "bear", "down"]
            text_lower = text.lower()
            bull_count = sum(text_lower.count(w) for w in bullish)
            bear_count = sum(text_lower.count(w) for w in bearish)
            total = bull_count + bear_count
            if total == 0:
                return 0.0
            return (bull_count - bear_count) / total

    def fetch_weekly(self, asset: str = "gold") -> dict:
        """Fetch current week's news and return sentiment score."""
        now   = datetime.today()
        month = now.strftime("%B")
        year  = now.year

        query = self.templates[asset].format(month=month, year=year)
        logger.info(f"[NEWS:{asset}] Querying: {query}")

        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )
            combined_text = " ".join([r["content"] for r in response["results"]])
            score = self._sentiment_score(combined_text)

            # Week start date (Monday) at midnight exactly
            week_start = now - timedelta(days=now.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

            return {
                "date":             week_start,
                f"{asset}_sentiment": score,
                f"{asset}_n_results": len(response["results"]),
            }
        except Exception as e:
            logger.error(f"[NEWS:{asset}] Failed: {e}")
            return {}

    def run(self, assets: list = None):
        """Fetch and append weekly sentiment for all assets."""
        if assets is None:
            assets = list(self.templates.keys())

        path = self._parquet_path()
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()

        rows = []
        for asset in assets:
            row = self.fetch_weekly(asset)
            if row:
                rows.append(row)

        if not rows:
            logger.warning("[NEWS] No rows fetched.")
            return

        # Create DataFrame
        new_df = pd.DataFrame(rows)
        
        # FIX: Combine the separate Gold and Silver dicts into one row per date
        if "date" in new_df.columns:
            # Group by date and take the first non-null value for each column
            new_df = new_df.groupby("date").first()
            
            # Strip timezones for clean merging later
            if new_df.index.tz is not None:
                new_df.index = new_df.index.tz_convert(None)

        if not existing.empty:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = new_df

        combined = combined.sort_index()
        combined.to_parquet(path)
        logger.success(f"[NEWS] Sentiment saved → {path}")

    def load(self) -> pd.DataFrame:
        path = self._parquet_path()
        if not path.exists():
            logger.warning("No news data found.")
            return pd.DataFrame()
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

# TEST EXECUTION BLOCK

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.config_loader import load_config

    cfg = load_config()
    news_ingester = NewsIngestion(cfg)
    news_ingester.run()