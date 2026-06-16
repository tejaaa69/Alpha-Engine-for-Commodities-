"""
src/features/store.py
Feature store: assembles all features and LABELS into a single leakage-safe DataFrame 
aligned to daily trading dates. Single point of truth. Every model training call goes through here.
Ensures consistent column ordering, reproducible targets, and zero leakage.
"""

from pathlib import Path
import pandas as pd
from loguru import logger
from src.features.macro_features import MacroFeatures
from src.features.regime import RegimeDetector
from src.features.technical import (
    build_technical_features,
    add_interasset_features,
)
from src.ingestion.market import MarketIngestion
from src.ingestion.macro import MacroIngestion
from src.ingestion.news import NewsIngestion

#Labeling logic
from src.models.labeling import get_volatility, triple_barrier_label, get_sample_uniqueness

class FeatureStore:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        # Resolve paths dynamically to root
        root_dir = Path(__file__).resolve().parent.parent.parent
        
        # 1. Safe fallback for paths (checks "features" first, then "feature_store", then defaults)
        paths_cfg = cfg.get("paths", {})
        feat_path = paths_cfg.get("features", paths_cfg.get("feature_store", "data/feature_store"))
        model_path = paths_cfg.get("models", "data/models")
        
        self.feat_dir = root_dir / feat_path
        self.feat_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = root_dir / model_path
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Safe fallback for parameters to prevent cascading KeyErrors
        feat_cfg = cfg.get("features", {})
        self.windows = feat_cfg.get("lookback_windows", [5, 10, 21, 63])
        self.rsi_period = feat_cfg.get("rsi_period", 14)
        self.atr_period = feat_cfg.get("atr_period", 14)

    def _parquet_path(self, symbol: str) -> Path:
        return self.feat_dir / f"master_features_{symbol}.parquet"

    def build(self, symbol: str, force_rebuild: bool = False) -> pd.DataFrame:
        """
        Build full feature matrix for a given asset symbol.
        Returns daily DataFrame with all features, targets, and weights merged.
        """
        path = self._parquet_path(symbol)
        if path.exists() and not force_rebuild:
            logger.info(f"Loading cached features for {symbol}")
            return pd.read_parquet(path)

        logger.info(f"Building master feature store for {symbol}...")

        # ── 1. Load price data 
        market = MarketIngestion(self.cfg)
        silver_symbol = self.cfg["assets"].get("silver", "SLV")
        gold_symbol = self.cfg["assets"].get("gold", "GLD")
        
        price_df = market.load(symbol)
        silver_df = market.load(silver_symbol) if symbol != silver_symbol else price_df

        # ── 2. Technical features 
        feat = build_technical_features(
            price_df,
            windows = self.windows,
            rsi_period = self.rsi_period,
            atr_period = self.atr_period,
        )

        # ── 3. Inter-asset features (only for gold) 
        if symbol == gold_symbol:
            feat = add_interasset_features(feat, silver_df)

        # ── 4. Macro features 
        macro_raw = MacroIngestion(self.cfg).load()
        macro_feats = MacroFeatures(macro_raw).align_to_daily(feat.index)
        feat = feat.join(macro_feats, how="left")

        # ── 5. Regime features (fixed VITERBI LEAKAGE) 
        regime_path = self.model_dir / "regime_detector.pkl"
        if regime_path.exists():
            detector = RegimeDetector.load(regime_path)
        else:
            detector = RegimeDetector(n_states=self.cfg["features"]["regime_n_states"])
            detector.fit(price_df["close"])
            detector.save(regime_path)

        # get_historical_features, not predict()
        regime_df = detector.get_historical_features(price_df["close"])
        feat = feat.join(regime_df, how="left")

        # ── 6. News sentiment features 
        news_ingestion = NewsIngestion(self.cfg)
        news_df = news_ingestion.load()
        if not news_df.empty:
            news_aligned = news_df.reindex(feat.index, method="ffill")
            feat = feat.join(news_aligned, how="left")

        # ── 7. Triple Barrier Labels 
        logger.info("Generating Triple Barrier Targets...")
        volatility = get_volatility(price_df["close"], span=100)
        
        # Extract constraints safely from config
        profit_pct = self.cfg.get("labeling", {}).get("profit_target_pct", 0.02)
        stop_pct = abs(self.cfg.get("labeling", {}).get("stop_loss_pct", 0.01))
        horizon = self.cfg.get("labeling", {}).get("horizon_days", 7)
        
        labels_df = triple_barrier_label(
            close=price_df["close"],
            vol=volatility,
            pt_sl=[profit_pct * 100, stop_pct * 100],
            horizon_days=horizon
        )
        
        # Calculate Uniqueness Weights
        labels_df["sample_weight"] = get_sample_uniqueness(labels_df)
        
        # Join targets to the feature matrix
        feat = feat.join(labels_df[["label", "ret", "barrier", "sample_weight"]], how="inner")

        # ── 8. Cleanup & Save 
        cols_to_drop = ["open", "high", "low", "close", "volume", "symbol"]
        feat = feat.drop(columns=[c for c in cols_to_drop if c in feat.columns])
        
        # Drop NaN rows at the start (from rolling window warmup) and ensure labels are present.
        # Use the longest MA column so the drop logic survives changes to lookback_windows.
        ma_col = f"ma_{max(self.windows)}" if self.windows else "ma_60"
        if ma_col in feat.columns:
            feat = feat.dropna(subset=[ma_col])
        feat = feat.dropna(subset=["label"])   # remove rows where label wasn’t computable (tail)
        
        feat.to_parquet(path)
        logger.success(f"Feature store built: {len(feat)} rows × {len(feat.columns)} cols → {path}")
        return feat
    
    def build_inference(self, symbol: str) -> pd.DataFrame:
        """
        Build feature matrix for live inference - includes the latest unlabeled rows.
        Does NOT join labels, so the most recent date is today (or the latest available).
        """
        logger.info(f"Building inference feature store for {symbol}...")

        market = MarketIngestion(self.cfg)
        silver_symbol = self.cfg["assets"].get("silver", "SLV")
        gold_symbol = self.cfg["assets"].get("gold", "GLD")

        price_df = market.load(symbol)
        silver_df = market.load(silver_symbol) if symbol != silver_symbol else price_df

        feat = build_technical_features(
            price_df,
            windows=self.windows,
            rsi_period=self.rsi_period,
            atr_period=self.atr_period,
        )
        if symbol == gold_symbol:
            feat = add_interasset_features(feat, silver_df)

        macro_raw = MacroIngestion(self.cfg).load()
        macro_feats = MacroFeatures(macro_raw).align_to_daily(feat.index)
        feat = feat.join(macro_feats, how="left")

        # Regime features
        regime_path = self.model_dir / "regime_detector.pkl"
        if regime_path.exists():
            detector = RegimeDetector.load(regime_path)
        else:
            detector = RegimeDetector(n_states=self.cfg["features"]["regime_n_states"])
            detector.fit(price_df["close"])
            detector.save(regime_path)
        regime_df = detector.get_historical_features(price_df["close"])
        feat = feat.join(regime_df, how="left")

        # News sentiment
        news_ingestion = NewsIngestion(self.cfg)
        news_df = news_ingestion.load()
        if not news_df.empty:
            news_aligned = news_df.reindex(feat.index, method="ffill")
            feat = feat.join(news_aligned, how="left")

        # Drop raw OHLCV columns
        cols_to_drop = ["open", "high", "low", "close", "volume", "symbol"]
        feat = feat.drop(columns=[c for c in cols_to_drop if c in feat.columns])

        # Remove rows where the longest moving average hasn't yet been computed (warmup)
        ma_col = f"ma_{max(self.windows)}" if self.windows else "ma_60"
        if ma_col in feat.columns:
            feat = feat.dropna(subset=[ma_col])

        # Fill any remaining gaps (e.g., shorter windows, macro series)
        feat = feat.ffill().fillna(0)

        logger.success(f"Inference feature store built: {len(feat)} rows × {len(feat.columns)} cols")
        return feat

    def get_feature_columns(self, symbol: str) -> list:
        """Returns ordered list of feature columns strictly for model training."""
        df = self.build(symbol)
        exclude = {"label", "ret", "barrier", "sample_weight"}
        return [c for c in df.columns if c not in exclude]

    def get_monotonic_constraints(self, feature_cols: list) -> list:
        """Returns constraint vector for LightGBM monotonic_constraints."""
        constraints = {
            "real_yield": -1,
            "real_yield_zscore": -1,
            "cpi_yoy": +1,
            "cpi_mom": +1,
            "fed_funds": -1,
            "vix": +1,
            "dollar_3m_pct": -1,
            "dollar_mom_pct": -1,
            "hy_spread_zscore": +1,
            "hy_spread_change": +1,
            "yield_curve_inverted": +1,
            "rsi": 0,
        }
        return [constraints.get(col, 0) for col in feature_cols]