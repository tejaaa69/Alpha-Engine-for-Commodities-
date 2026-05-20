"""
src/ingestion/market.py

Downloads daily OHLCV data for all configured assets.
Saves to data/raw/{symbol}.parquet.
Incremental: only fetches new rows if file already exists.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yfinance as yf
from loguru import logger


class MarketIngestion:
    def __init__(self, cfg: Dict[str, Any]):
        self.assets   = cfg["assets"]           # e.g., {"gold": "GC=F", "silver": "SI=F"}
        self.raw_dir  = Path(cfg["paths"]["raw_data"])
        
        # Ensure the directory exists
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _parquet_path(self, symbol: str) -> Path:
        return self.raw_dir / f"{symbol}.parquet"

    def _get_start_date(self, symbol: str) -> str:
        """If parquet exists, resume from last stored date. Else fetch full history."""
        path = self._parquet_path(symbol)
        if path.exists():
            existing = pd.read_parquet(path)
            last_date = existing.index.max()
            # +1 day to avoid re-fetching last row
            start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"[{symbol}] Found existing data. Resuming from {start}")
            return start
        
        logger.info(f"[{symbol}] No existing data. Full download from 2010-01-01")
        return "2010-01-01"

    def _download(self, symbol: str, start: str) -> pd.DataFrame:
        end = datetime.today().strftime("%Y-%m-%d")
        logger.info(f"[{symbol}] Downloading {start} → {end}")
        
        # yfinance download
        raw = yf.download(symbol, start=start, end=end, progress=False)
        
        if raw.empty:
            logger.warning(f"[{symbol}] No new data returned from Yahoo Finance.")
            return pd.DataFrame()

        # Ensure index is standard datetime
        raw.index = pd.to_datetime(raw.index)
        
        # CRITICAL FIX: Strip timezones to prevent Parquet merge crashes
        if raw.index.tz is not None:
            raw.index = raw.index.tz_convert(None)
            
        raw.index.name = "date"

        # Flatten MultiIndex columns if present (yfinance ≥0.2.38 quirk)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]

        # Keep only standard OHLCV columns (ignore 'adj close' if present)
        cols_to_keep = ["open", "high", "low", "close", "volume"]
        
        # Safely check if columns exist before filtering to avoid KeyErrors
        available_cols = [c for c in cols_to_keep if c in raw.columns]
        raw = raw[available_cols].copy()
        
        raw["symbol"] = symbol
        return raw

    def _merge_and_save(self, symbol: str, new_df: pd.DataFrame):
        path = self._parquet_path(symbol)
        
        if path.exists() and not new_df.empty:
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df])
            
            # Drop exact duplicate dates, keeping the most recent data fetch
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            
        elif not new_df.empty:
            combined = new_df.sort_index()
        else:
            logger.info(f"[{symbol}] Nothing new to save. System is up to date.")
            return
            
        combined.to_parquet(path)
        logger.success(f"[{symbol}] Successfully saved {len(combined)} total rows → {path}")

    def run(self):
        """Run ingestion for all assets configured."""
        logger.info("Starting Market Data Ingestion Pipeline...")
        for asset_name, symbol in self.assets.items():
            try:
                start  = self._get_start_date(symbol)
                new_df = self._download(symbol, start)
                self._merge_and_save(symbol, new_df)
            except Exception as e:
                logger.error(f"[{symbol}] Ingestion failed with error: {e}")

    def load(self, symbol: str) -> pd.DataFrame:
        """Helper method to load stored data for a symbol during modeling phase."""
        path = self._parquet_path(symbol)
        if not path.exists():
            raise FileNotFoundError(f"No data found for {symbol}. Run ingestion first.")
        
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

# TEST EXECUTION BLOCK
if __name__ == "__main__":
    # Dummy configuration to test the script standalone
    test_config = {
        "assets": {
            "gold": "GC=F",
            "silver": "SI=F"
        },
        "paths": {
            # This navigates from src/ingestion/market.py up to the project root, then into data/raw
            "raw_data": Path(__file__).resolve().parent.parent.parent / "data" / "raw"
        }
    }

    # Initialize and run
    ingester = MarketIngestion(test_config)
    ingester.run()