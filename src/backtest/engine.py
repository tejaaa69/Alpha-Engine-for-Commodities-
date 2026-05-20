"""
src/backtest/engine.py

Elite Event-Driven Backtesting Engine for Triple-Barrier Signals.

Upgrades:
  1. Daily Mark-to-Market (MtM): Accurately tracks daily portfolio volatility for true Sharpe/Sortino.
  2. Dynamic Execution Simulation: Recalculates barriers dynamically based on actual OPEN fill prices, 
     not the prior day's CLOSE.
  3. Gap Risk Handling: If the market gaps past the stop-loss on the open, it executes at the worse open price.
  4. Pessimistic Intraday Collision: If both the Upper and Lower barriers are touched on the same day, 
     the engine assumes the Stop Loss was hit first (conservative risk modeling).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class Trade:
    entry_date:    pd.Timestamp
    entry_price:   float
    exit_date:     pd.Timestamp
    exit_price:    float
    signal_prob:   float
    position_size: float    # dollars
    gross_return:  float
    net_return:    float    # after costs
    barrier_hit:   str      # profit / stop / time
    regime:        str = ""


@dataclass
class BacktestResult:
    trades:        list = field(default_factory=list)
    equity_curve:  pd.Series = None
    metrics:       dict = field(default_factory=dict)


class BacktestEngine:
    def __init__(self, cfg: dict):
        bt = cfg["backtest"]
        self.initial_capital  = bt["initial_capital"]
        self.position_size_pct= bt["position_size_pct"]
        self.commission_pct   = bt["commission_pct"]
        self.slippage_pct     = bt["slippage_pct"]
        self.use_kelly        = bt["use_kelly"]
        self.label_cfg        = cfg["labeling"]

    def _kelly_size(
        self,
        prob_win:     float,
        profit_pct:   float,
        stop_pct:     float,
        capital:      float,
    ) -> float:
        """
        Fractional Kelly position sizing (Half-Kelly).
        f* = (p*b - q) / b
        """
        if prob_win <= 0 or prob_win >= 1:
            return capital * self.position_size_pct
        
        q = 1 - prob_win
        b = profit_pct / stop_pct   # Reward/Risk Ratio
        
        f_star = (prob_win * b - q) / b
        f_star = max(0.0, min(f_star, 0.5))  # Cap absolute risk at 50% of capital
        
        return capital * (f_star / 2)      

    def run(
        self,
        price_df:      pd.DataFrame,
        signals:       pd.Series,
        threshold:     float = 0.55,
        regime_series: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        State-Machine Backtest Engine with Daily Mark-to-Market.
        """
        logger.info("Initializing Elite Execution Engine...")
        
        cash = self.initial_capital
        position_shares = 0.0
        trades = []
        
        # Pre-allocate equity curve for speed
        equity_curve = pd.Series(index=price_df.index, dtype=float)
        
        # State tracking
        in_trade = False
        entry_price_raw = 0.0
        entry_price_adj = 0.0
        entry_date = None
        current_prob = 0.0
        days_in_trade = 0
        current_regime = ""

        profit_pct = self.label_cfg["profit_target_pct"]
        stop_pct   = abs(self.label_cfg["stop_loss_pct"])
        horizon    = self.label_cfg["horizon_days"]

        # Sort cleanly to avoid time-travel
        price_df = price_df.sort_index()
        dates = price_df.index

        for i, date in enumerate(dates):
            today_open  = price_df.loc[date, "open"]
            today_high  = price_df.loc[date, "high"]
            today_low   = price_df.loc[date, "low"]
            today_close = price_df.loc[date, "close"]

            # ── 1. ENTRY LOGIC (Executed at Open) ──────────────────────────
            if not in_trade and i > 0:
                yday = dates[i-1]
                
                # Check if yesterday generated a valid signal
                if yday in signals.index:
                  signal_val = signals.loc[yday]
                  if isinstance(signal_val, pd.Series):
                    signal_val = signal_val.mean() # Collapse duplicate entries safely if they occur
                  if signal_val >= threshold:
                    current_prob = signal_val

                    if regime_series is not None and yday in regime_series.index:
                        current_regime = str(regime_series.loc[yday])

                    # Apply Entry Slippage to the Open
                    entry_price_raw = today_open
                    entry_price_adj = entry_price_raw * (1 + self.slippage_pct)

                    # Determine Dollar Size
                    if self.use_kelly:
                        pos_size_usd = self._kelly_size(current_prob, profit_pct, stop_pct, cash)
                    else:
                        pos_size_usd = cash * self.position_size_pct
                        
                    pos_size_usd = min(pos_size_usd, cash)

                    # Execute Trade (Buy shares, deduct cash including commission)
                    position_shares = (pos_size_usd * (1 - self.commission_pct)) / entry_price_adj
                    cash -= (position_shares * entry_price_adj)

                    in_trade = True
                    entry_date = date
                    days_in_trade = 0

            # ── 2. EXIT LOGIC (Checked daily while in trade) ───────────────
            if in_trade:
                days_in_trade += 1

                # Dynamically calculate barriers based on ACTUAL entry price
                upper_barrier = entry_price_adj * (1 + profit_pct)
                lower_barrier = entry_price_adj * (1 - stop_pct)

                hit_profit = today_high >= upper_barrier
                hit_stop   = today_low <= lower_barrier
                hit_time   = days_in_trade >= horizon

                exit_triggered = False
                exit_price_raw = 0.0
                barrier_hit = ""

                # Pessimistic Execution: If both hit, assume stop was hit first.
                if hit_stop and hit_profit:
                    exit_price_raw = lower_barrier
                    barrier_hit = "stop"
                    exit_triggered = True
                
                elif hit_stop:
                    # Gap Risk: If it opened below the stop, you get filled at the open!
                    exit_price_raw = min(today_open, lower_barrier)
                    barrier_hit = "stop"
                    exit_triggered = True
                
                elif hit_profit:
                    # Gap Up: If it opened above the target, you get the better open price!
                    exit_price_raw = max(today_open, upper_barrier)
                    barrier_hit = "profit"
                    exit_triggered = True
                
                elif hit_time:
                    exit_price_raw = today_close
                    barrier_hit = "time"
                    exit_triggered = True

                # Process the Exit
                if exit_triggered:
                    # Apply Exit Slippage
                    exit_price_adj = exit_price_raw * (1 - self.slippage_pct)
                    
                    # Calculate Proceeds and Deduct Commission
                    gross_proceeds = position_shares * exit_price_adj
                    net_proceeds   = gross_proceeds * (1 - self.commission_pct)
                    
                    cash += net_proceeds
                    
                    trade_cost = position_shares * entry_price_adj
                    net_return = (net_proceeds - trade_cost) / trade_cost
                    gross_return = (exit_price_raw - entry_price_raw) / entry_price_raw

                    trades.append(Trade(
                        entry_date    = entry_date,
                        entry_price   = round(entry_price_adj, 4),
                        exit_date     = date,
                        exit_price    = round(exit_price_adj, 4),
                        signal_prob   = round(current_prob, 4),
                        position_size = round(trade_cost, 2),
                        gross_return  = round(gross_return, 6),
                        net_return    = round(net_return, 6),
                        barrier_hit   = barrier_hit,
                        regime        = current_regime,
                    ))

                    # Reset State
                    in_trade = False
                    position_shares = 0.0

            # ── 3. MARK TO MARKET EQUITY ───────────────────────────────────
            # Daily Portfolio Value = Cash + (Shares * Today's Close)
            mtm_value = cash + (position_shares * today_close if in_trade else 0.0)
            equity_curve.loc[date] = mtm_value

        # Calculate robust metrics
        metrics = self._compute_metrics(trades, equity_curve)
        logger.success(f"Execution complete: {len(trades)} trades | Sharpe: {metrics.get('sharpe_ratio', 0)}")

        return BacktestResult(
            trades       = trades,
            equity_curve = equity_curve,
            metrics      = metrics,
        )

    def _compute_metrics(self, trades: list, equity: pd.Series) -> dict:
        if not trades:
            return {"error": "No trades executed"}

        # Use the Daily MtM Equity to calculate true volatility
        daily_ret = equity.pct_change().dropna()

        # Annualized Sharpe
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0.0

        # Annualized Sortino
        downside = daily_ret[daily_ret < 0]
        sortino  = (daily_ret.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0.0

        # Max Drawdown
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd   = drawdown.min()

        # Calmar Ratio
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        annualized_return = (1 + total_return) ** (252 / len(equity)) - 1
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0.0

        # Trade Stats
        wins     = [t for t in trades if t.net_return > 0]
        losses   = [t for t in trades if t.net_return <= 0]
        win_rate = len(wins) / len(trades)

        avg_win  = np.mean([t.net_return for t in wins])   if wins   else 0
        avg_loss = np.mean([t.net_return for t in losses]) if losses else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        avg_duration = np.mean([(t.exit_date - t.entry_date).days for t in trades])

        barrier_counts = pd.Series([t.barrier_hit for t in trades]).value_counts().to_dict()

        return {
            "total_trades":   len(trades),
            "win_rate":       round(win_rate, 4),
            "sharpe_ratio":   round(sharpe, 4),
            "sortino_ratio":  round(sortino, 4),
            "calmar_ratio":   round(calmar, 4),
            "max_drawdown":   round(max_dd, 4),
            "total_return":   round(total_return, 4),
            "profit_factor":  round(profit_factor, 4),
            "avg_win":        round(avg_win, 6),
            "avg_loss":       round(avg_loss, 6),
            "avg_duration_days": round(avg_duration, 1),
            "barrier_profit": barrier_counts.get("profit", 0),
            "barrier_stop":   barrier_counts.get("stop",   0),
            "barrier_time":   barrier_counts.get("time",   0),
            "final_capital":  round(equity.iloc[-1], 2),
        }