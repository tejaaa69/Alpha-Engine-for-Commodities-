"""
src/features/macro_features.py

Transforms raw FRED series into ML-ready features.
The input data is already on a DAILY frequency (forward-filled during ingestion).
Therefore, rolling windows must use daily lookbacks (e.g., 365 for YoY).
"""

import numpy as np
import pandas as pd
from loguru import logger

class MacroFeatures:
    def __init__(self, macro_df: pd.DataFrame):
        self.macro = macro_df.copy()
        self.macro.index = pd.to_datetime(self.macro.index)

    def _derive(self) -> pd.DataFrame:
        """Compute derived signals from raw FRED levels using DAILY windows."""
        m = self.macro.copy()

        # CPI: 365 days = 1 Year. 90 days = 3 Months. 30 days = 1 Month.
        if "cpi_level" in m.columns:
            m["cpi_yoy"]    = m["cpi_level"].pct_change(365) * 100
            m["cpi_mom"]    = m["cpi_level"].pct_change(30)  * 100
            m["cpi_3m"]     = m["cpi_level"].pct_change(90)  * 100
            m = m.drop(columns=["cpi_level"]) # Drop raw level

        # Real yield z-score (3-year rolling window = 365 * 3 = 1095 days)
        if "real_yield" in m.columns:
            roll_mean = m["real_yield"].rolling(1095).mean()
            roll_std  = m["real_yield"].rolling(1095).std().replace(0, np.nan)
            m["real_yield_zscore"] = (m["real_yield"] - roll_mean) / roll_std

        # Dollar index: momentum
        if "dollar_index" in m.columns:
            m["dollar_3m_pct"]  = m["dollar_index"].pct_change(90) * 100
            m["dollar_mom_pct"] = m["dollar_index"].pct_change(30) * 100

        # Yield curve: level + direction
        if "yield_curve" in m.columns:
            m["yield_curve_direction"] = np.sign(m["yield_curve"].diff(90))
            # Inversion signal: 1 = inverted (recession warning = gold positive)
            m["yield_curve_inverted"]  = (m["yield_curve"] < 0).astype(int)

        # Housing starts: MoM momentum
        if "housing_starts" in m.columns:
            m["housing_mom_pct"] = m["housing_starts"].pct_change(30) * 100

        # HY spread: level and change (risk-off proxy)
        if "hy_spread" in m.columns:
            m["hy_spread_change"] = m["hy_spread"].diff(30)
            m["hy_spread_zscore"] = (
                (m["hy_spread"] - m["hy_spread"].rolling(1095).mean())
                / m["hy_spread"].rolling(1095).std().replace(0, np.nan)
            )

        return m

    def align_to_daily(self, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Re-aligns the derived features exactly to the trading days of the asset.
        """
        derived = self._derive()

        # Reindex to trading days, forward-fill
        aligned = derived.reindex(daily_index, method="ffill")

        # Drop columns with too many NaN (e.g., caused by the 3-year rolling windows at the start of the dataset)
        # We lower the threshold slightly to 0.70 because our rolling windows are now much larger (1095 days)
        coverage = aligned.notna().mean()
        good_cols = coverage[coverage >= 0.70].index
        dropped   = set(aligned.columns) - set(good_cols)
        if dropped:
            logger.warning(f"Dropping low-coverage macro cols: {dropped}")
        
        aligned = aligned[good_cols]
        # Fill remaining early NaNs with 0 to keep the data shape intact
        aligned = aligned.fillna(0)

        logger.info(f"Macro features aligned. Columns: {list(aligned.columns)}")
        return aligned