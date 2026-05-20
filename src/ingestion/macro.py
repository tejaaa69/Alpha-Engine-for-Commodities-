"""
src/ingestion/macro.py

Downloads FRED macroeconomic series.
CRITICAL: applies publication lags so features are never available
before they would be in real life. This prevents lookahead bias.
Automatically forward-fills to daily frequency for clean merging with market data.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from fredapi import Fred
from loguru import logger


class MacroIngestion:
    def __init__(self, cfg: Dict[str, Any]):
        # Secure API Key fetching
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise ValueError(
                "FRED_API_KEY environment variable is not set! "
                "Please set it using: export FRED_API_KEY='your_key' (Mac/Linux) "
                "or $env:FRED_API_KEY='your_key' (Windows PowerShell)."
            )
            
        self.fred     = Fred(api_key=api_key)
        self.series   = cfg["fred_series"]          # {SERIES_ID: column_name}
        self.lags     = cfg["fred_release_lags"]    # {SERIES_ID: days_lag}
        
        # Use config path, but resolve it to absolute path just in case
        root_dir = Path(__file__).resolve().parent.parent.parent
        self.raw_dir  = root_dir / cfg["paths"]["raw_data"]
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _parquet_path(self) -> Path:
        return self.raw_dir / "macro.parquet"

    def _fetch_series(self, series_id: str, col_name: str) -> pd.Series:
        lag_days = self.lags.get(series_id, 1)
        end_date = datetime.today()
        # Align start date with our market data (2010-01-01)
        start    = "2010-01-01"

        logger.info(f"[{col_name}] Fetching {series_id} from FRED...")
        raw = self.fred.get_series(
            series_id,
            observation_start=start,
            observation_end=end_date.strftime("%Y-%m-%d")
        )
        
        raw = raw.dropna()
        raw.index = pd.to_datetime(raw.index)
        
        # Strip timezone if present to avoid merge conflicts later
        if raw.index.tz is not None:
            raw.index = raw.index.tz_convert(None)
            
        raw.name = col_name

        # ── LEAKAGE PREVENTION ────────────────────────────────────────
        # Shift index forward by publication lag.
        raw.index = raw.index + pd.Timedelta(days=lag_days)
        # ─────────────────────────────────────────────────────────────

        return raw

    def run(self):
        logger.info("Starting Macro Data Ingestion Pipeline...")
        frames = {}
        for sid, col_name in self.series.items():
            try:
                s = self._fetch_series(sid, col_name)
                frames[col_name] = s
                logger.success(f"[FRED:{sid}] {len(s)} obs, lag={self.lags.get(sid,1)}d")
            except Exception as e:
                logger.error(f"[FRED:{sid}] Failed to fetch: {e}")

        if not frames:
            logger.error("No FRED series fetched. Exiting.")
            return

        # Combine all series into one DataFrame
        df = pd.DataFrame(frames)
        df.index.name = "date"
        df = df.sort_index()

        # ── ADVANCED: DAILY FORWARD FILL ──────────────────────────────
        # Macro data is sparse. We need a daily index so it matches market data.
        full_date_range = pd.date_range(start=df.index.min(), end=datetime.today(), freq='D')
        df = df.reindex(full_date_range)
        
        # Forward fill the missing days. If CPI is released on the 14th, 
        # the 15th, 16th, etc., should hold the same CPI value until the next release.
        df = df.ffill()
        df.index.name = "date"
        # ─────────────────────────────────────────────────────────────

        # Save to parquet
        df.to_parquet(self._parquet_path())
        logger.info(f"Macro data saved successfully → {self._parquet_path()}")

    def load(self) -> pd.DataFrame:
        path = self._parquet_path()
        if not path.exists():
            raise FileNotFoundError("No macro data found. Run MacroIngestion().run() first.")
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

# TEST EXECUTION BLOCK
if __name__ == "__main__":
    import sys
    # Dynamically import the config loader we built in the previous step
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.config_loader import load_config

    # Load configuration
    config = load_config()
    
    # Initialize and run
    macro_ingester = MacroIngestion(config)
    macro_ingester.run()