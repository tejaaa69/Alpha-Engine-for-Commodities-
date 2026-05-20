"""
src/features/technical.py

Price-derived features for gold and silver.
ALL features are computed using only past data (shift(1) enforced).
No future values are ever used.

Feature categories:
  - Trend:         moving averages, distance from MA, MA crossovers
  - Momentum:      RSI, rate of change, MACD, Bollinger %b
  - Volatility:    ATR, historical vol, rolling Z-scores
  - Volume:        OBV, volume Z-score
  - Calendar:      day-of-week, month (sine/cosine encoded)
  - Inter-asset:   gold/silver ratio, relative strength
"""

import numpy as np
import pandas as pd
from loguru import logger


def _safe_shift(series: pd.Series, periods: int = 1) -> pd.Series:
    """Explicit shift to make leakage prevention visible in code."""
    return series.shift(periods)


def add_trend_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    close = df["close"]
    features = {}

    for w in windows:
        ma = close.rolling(w).mean()
        features[f"ma_{w}"]          = _safe_shift(ma)
        features[f"dist_ma_{w}"]       = _safe_shift((close - ma) / ma)
        features[f"close_above_{w}"] = _safe_shift((close > ma).astype(int))

    if 10 in windows and 60 in windows:
        features["ma_cross_10_60"] = _safe_shift(
            (close.rolling(10).mean() > close.rolling(60).mean()).astype(int)
        )
    if 5 in windows and 20 in windows:
        features["ma_cross_5_20"] = _safe_shift(
            (close.rolling(5).mean() > close.rolling(20).mean()).astype(int)
        )

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def add_momentum_features(df: pd.DataFrame, windows: list, rsi_period: int = 14) -> pd.DataFrame:
    close = df["close"]
    features = {}

    # Rate of change
    for w in windows:
        features[f"roc_{w}"] = _safe_shift(close.pct_change(w))

    # RSI
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(rsi_period).mean()
    loss   = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs     = gain / loss.replace(0, np.nan)
    features["rsi"] = _safe_shift(100 - (100 / (1 + rs)))

    # MACD
    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    features["macd"]        = _safe_shift(macd_line)
    features["macd_signal"] = _safe_shift(signal_line)
    features["macd_hist"]   = _safe_shift(macd_line - signal_line)

    # Bollinger Bands %b (Where is price relative to 20-day volatility bands?)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_band = ma20 + (std20 * 2)
    lower_band = ma20 - (std20 * 2)
    features["bb_pct_b"] = _safe_shift((close - lower_band) / (upper_band - lower_band).replace(0, np.nan))

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def add_volatility_features(df: pd.DataFrame, windows: list, atr_period: int = 14) -> pd.DataFrame:
    close = df["close"]
    features = {}

    # Historical volatility (annualized)
    log_ret = np.log(close / close.shift(1))
    for w in windows:
        hv = log_ret.rolling(w).std() * np.sqrt(252)
        features[f"hv_{w}"] = _safe_shift(hv)
        
        # Volatility Z-Score (Is current vol abnormally high compared to 60-day baseline?)
        if w == 20:
            features["hv_20_zscore"] = _safe_shift((hv - hv.rolling(60).mean()) / hv.rolling(60).std())

    # ATR (Average True Range)
    hl   = df["high"] - df["low"]
    hpc  = (df["high"] - close.shift(1)).abs()
    lpc  = (df["low"]  - close.shift(1)).abs()
    tr   = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    atr  = tr.rolling(atr_period).mean()
    
    features["atr"]     = _safe_shift(atr)
    features["atr_pct"] = _safe_shift(atr / close)   

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["close"]
    volume = df["volume"]
    features = {}

    # OBV (On Balance Volume)
    direction  = np.sign(close.diff())
    obv        = (volume * direction).cumsum()
    features["obv"]        = _safe_shift(obv)
    features["obv_ma_20"]  = _safe_shift(obv.rolling(20).mean())

    # Volume Z-Score (More robust than simple ratio)
    vol_ma = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    features["volume_zscore"] = _safe_shift((volume - vol_ma) / vol_std)

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    features = {}
    
    day_of_week = df.index.dayofweek
    month       = df.index.month

    # Sine/cosine encoding for cyclical features
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    features["dow_sin"]   = np.sin(2 * np.pi * day_of_week / 5)
    features["dow_cos"]   = np.cos(2 * np.pi * day_of_week / 5)

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def add_interasset_features(gold_df: pd.DataFrame, silver_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-asset features that capture relative value and flow dynamics.
    Merged onto gold's index. Safe against uppercase column naming.
    """
    # 1. Create a copy and force lowercase to prevent KeyErrors
    gold_df = gold_df.copy()
    silver_df = silver_df.copy()
    gold_df.columns = gold_df.columns.str.lower()
    silver_df.columns = silver_df.columns.str.lower()

    # 2. Extract columns safely using original robust logic
    gold_close = gold_df["close"]
    # ffill aligns the dates securely so no future silver data leaks into a weekend/holiday
    silver_close = silver_df["close"].reindex(gold_df.index).ffill()

    features = {}

    # Gold/Silver ratio
    gs_ratio = gold_close / silver_close.replace(0, np.nan)
    features["gs_ratio"] = _safe_shift(gs_ratio)
    features["gs_ratio_ma20"] = _safe_shift(gs_ratio.rolling(20).mean())
    features["gs_ratio_zscore"] = _safe_shift(
        (gs_ratio - gs_ratio.rolling(60).mean()) / gs_ratio.rolling(60).std().replace(0, np.nan)
    )

    # Relative strength: gold return / silver return
    gold_ret = gold_close.pct_change(20)
    silver_ret = silver_close.pct_change(20)
    features["rel_strength_20"] = _safe_shift(gold_ret - silver_ret)

    return pd.concat([gold_df, pd.DataFrame(features, index=gold_df.index)], axis=1)

def build_technical_features(
    price_df: pd.DataFrame,
    windows: list = None,
    rsi_period: int = 14,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Master function: takes raw OHLCV, returns full feature DataFrame."""
    if windows is None:
        windows = [5, 10, 20, 60]
        
    df = price_df.copy()
    
    # Force columns to lowercase and flatten MultiIndex if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower() for col in df.columns]
    else:
        df.columns = df.columns.str.lower()
    
    df = add_trend_features(df, windows)
    df = add_momentum_features(df, windows, rsi_period)
    df = add_volatility_features(df, windows, atr_period)
    df = add_volume_features(df)
    df = add_calendar_features(df)
    
    logger.info(f"Technical features built: {len(df.columns)} predictors generated.")
    return df
