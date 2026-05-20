"""
src/models/labeling.py

Elite Implementation of the Triple Barrier Method (AFML Chapter 3).
Includes Dynamic Volatility Scaling and Sample Uniqueness logic.
"""

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


def get_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    """
    Computes dynamic volatility for barrier scaling.
    Institutional standard: EWMA of log returns.
    """
    # Calculate daily returns
    returns = close.pct_change()
    # EWMA Volatility (Standard Deviation)
    vol = returns.ewm(span=span).std()
    return vol.dropna()


def triple_barrier_label(
    close: pd.Series,
    vol: pd.Series,
    pt_sl: list = [2, 1],  # [Profit Target Multiplier, Stop Loss Multiplier]
    target: float = 0.01,   # Minimum target return (floor)
    horizon_days: int = 7,
) -> pd.DataFrame:
    """
    Advanced Triple Barrier Labeling with Dynamic Volatility Scaling.
    
    Instead of fixed 2%, it uses: 
    Upper Barrier = entry + (volatility * pt_sl[0])
    Lower Barrier = entry - (volatility * pt_sl[1])
    """
    # 1. Get the timestamps for vertical barriers (timeout)
    vertical_barriers = close.index + pd.Timedelta(days=horizon_days)
    
    # 2. Filter volatility to match close price index
    vol = vol.reindex(close.index).ffill()
    
    # Create the output container
    out = pd.DataFrame(index=close.index, columns=["exit_date", "label", "ret", "barrier"])
    
    logger.info(f"Applying Triple Barrier on {len(close)} observations...")
    
    # We iterate, but we optimize by using numpy slices for speed
    for t0, pg in tqdm(close.items(), total=len(close), desc="Labeling"):
        # Vertical barrier date
        t1 = t0 + pd.Timedelta(days=horizon_days)
        
        # Horizontal barriers based on volatility
        # If vol is 0.005 and pt_sl is [2,1], upper is +1% and lower is -0.5%
        # This ensures we hunt for trades where the reward/risk is justified by current volatility
        curr_vol = vol.loc[t0]
        upper = t0 + pd.Timedelta(days=horizon_days) # placeholder
        
        # Get future prices within the window
        future_prices = close.loc[t0:t1]
        
        if future_prices.empty or len(future_prices) < 2:
            continue

        # Define price barriers
        up_barrier = pg * (1 + curr_vol * pt_sl[0])
        lo_barrier = pg * (1 - curr_vol * pt_sl[1])

        # Find first touch
        # np.where returns indices of prices that crossed the barrier
        touches_upper = future_prices[future_prices >= up_barrier].index
        touches_lower = future_prices[future_prices <= lo_barrier].index
        
        # Earliest touch of either barrier
        first_upper = touches_upper[0] if not touches_upper.empty else None
        first_lower = touches_lower[0] if not touches_lower.empty else None
        
        # Logic to determine which was hit first
        if first_upper is not None and (first_lower is None or first_upper < first_lower):
            out.loc[t0, "label"] = 1
            out.loc[t0, "exit_date"] = first_upper
            out.loc[t0, "barrier"] = "profit"
        elif first_lower is not None and (first_upper is None or first_lower < first_upper):
            out.loc[t0, "label"] = -1
            out.loc[t0, "exit_date"] = first_lower
            out.loc[t0, "barrier"] = "stop"
        else:
            out.loc[t0, "label"] = 0
            out.loc[t0, "exit_date"] = future_prices.index[-1]
            out.loc[t0, "barrier"] = "time"
            
        # Calculate actual log return
        exit_price = close.loc[out.loc[t0, "exit_date"]]
        out.loc[t0, "ret"] = np.log(exit_price / pg)

    # Final cleanup
    out["label"] = pd.to_numeric(out["label"])
    out["days_held"] = (pd.to_datetime(out["exit_date"]) - out.index).dt.days
    
    return out.dropna(subset=["label"])


def get_sample_uniqueness(labels: pd.DataFrame) -> pd.Series:
    """
    ELITE FEATURE: Concurrent Sample Uniqueness.
    
    Calculates how much a sample 'overlaps' with others. 
    In quant finance, if 5 trades are open at the same time, they share 
    the same information. We use this to down-weight overlapping samples
    during LightGBM training.
    """
    # Create a concurrency matrix
    # This is a simplified version of De Prado's uniqueness logic
    # It counts how many labels are 'active' at any given timestamp
    t_span = pd.DataFrame(index=labels.index)
    t_span['t1'] = labels['exit_date']
    
    # Initialize a count series on a daily frequency
    all_dates = pd.date_range(start=labels.index.min(), end=labels['exit_date'].max(), freq='D')
    concurrency = pd.Series(0, index=all_dates)
    
    for i, row in t_span.iterrows():
        concurrency.loc[i:row['t1']] += 1
        
    # Uniqueness is 1 / average concurrency over the life of the trade
    uniqueness = pd.Series(index=labels.index)
    for i, row in t_span.iterrows():
        uniqueness.loc[i] = 1.0 / concurrency.loc[i:row['t1']].mean()
        
    return uniqueness