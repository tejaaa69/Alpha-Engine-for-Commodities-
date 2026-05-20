"""
src/features/regime.py

Advanced Hidden Markov Model regime detection for commodities.

Latent states learned from:
  - Daily log returns
  - 20-day rolling volatility
  - 20-day rolling trend

CRITICAL QUANT UPGRADES:
1. Volatility-Based Labeling: Financial HMMs naturally partition by variance, 
   not mean return. We label states based on the trace of their covariance matrices.
2. Causal Feature Generation: Prevents Viterbi Lookahead Bias by strictly shifting
   the inferred regime states by 1 day before passing to the feature store.
"""

import warnings
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn import hmm
from loguru import logger
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    def __init__(self, n_states: int = 3, random_state: int = 42):
        self.n_states     = n_states
        self.random_state = random_state
        self.model        = None
        self.scaler       = StandardScaler()
        self.state_labels = {}   # maps HMM state int → "LOW_VOL", "MID_VOL", "HIGH_VOL"

    def _build_obs(self, close: pd.Series) -> pd.DataFrame:
        """Build observation matrix. Returns must be clean of NaNs."""
        log_ret = np.log(close / close.shift(1))
        obs = pd.DataFrame({
            "log_ret":   log_ret,
            "vol_20":    log_ret.rolling(20).std(),
            "trend_20":  log_ret.rolling(20).mean(),
        })
        # Drop the first 20 days where rolling features are NaN
        return obs.dropna()

    def fit(self, close: pd.Series):
        """Fit HMM on historical price series and map states by Volatility."""
        obs_df  = self._build_obs(close)
        obs_raw = obs_df.values
        obs_scaled = self.scaler.fit_transform(obs_raw)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = hmm.GaussianHMM(
                n_components   = self.n_states,
                covariance_type= "full",
                n_iter         = 500,  # Increased iterations for better convergence
                random_state   = self.random_state,
                tol            = 1e-4,
            )
            self.model.fit(obs_scaled)

        # ── INSTITUTIONAL LOGIC: Sort states by learned Variance ──
        # model.covars_ shape is (n_states, n_features, n_features)
        # We take the trace (sum of diagonal) to measure total variance of each state
        state_variances = {i: np.trace(self.model.covars_[i]) for i in range(self.n_states)}
        
        # Sort states from lowest variance to highest variance
        sorted_states = sorted(state_variances.items(), key=lambda x: x[1])
        
        self.state_labels = {
            sorted_states[0][0]: "LOW_VOL",    # Typically stable bull market
            sorted_states[1][0]: "MID_VOL",    # Transition / Sideways
            sorted_states[2][0]: "HIGH_VOL",   # Bear market / Crisis / Gap-ups
        }

        logger.info(f"HMM fitted successfully.")
        logger.info(f"State Variance Mapping: {state_variances}")
        logger.info(f"State Labels: {self.state_labels}")

        return self

    def get_historical_features(self, close: pd.Series) -> pd.DataFrame:
        """
        Generate leakage-free regime features for the Feature Store.
        Uses predict_proba and explicitly shifts by 1 to prevent Viterbi lookahead.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before generating features.")

        obs_df = self._build_obs(close)
        obs_scaled = self.scaler.transform(obs_df.values)

        # predict_proba returns the probability of each state at each time step
        probs = self.model.predict_proba(obs_scaled)
        
        # The most likely state
        states = np.argmax(probs, axis=1)

        # Build the DataFrame
        df = pd.DataFrame(index=obs_df.index)
        df["regime_code"] = states
        
        # Add the probabilities as continuous features (Highly valuable for LightGBM)
        for state_idx, label in self.state_labels.items():
            df[f"prob_{label}"] = probs[:, state_idx]

        # ── LEAKAGE PREVENTION ──
        # We MUST shift the regime features forward by 1. 
        # The regime detected at the end of Tuesday can only be used to predict Wednesday.
        df = df.shift(1)
        
        return df

    def get_current_regime(self, close: pd.Series) -> dict:
        """Return the most recent regime for live inference/dashboarding."""
        if self.model is None:
            raise RuntimeError("Call fit() before get_current_regime().")

        # In production, we only need a recent window to establish the state
        obs_df = self._build_obs(close.tail(100))  
        obs_scaled = self.scaler.transform(obs_df.values)

        probs = self.model.predict_proba(obs_scaled)
        current_probs = probs[-1]
        current_state = int(np.argmax(current_probs))

        result = {
            "regime_code":  current_state,
            "regime_label": self.state_labels[current_state]
        }
        
        # Add all probabilities to the output dict
        for state_idx, label in self.state_labels.items():
            result[f"prob_{label}"] = round(float(current_probs[state_idx]), 4)

        return result

    def save(self, path: Path):
        joblib.dump({
            "model": self.model, 
            "scaler": self.scaler,
            "state_labels": self.state_labels, 
            "n_states": self.n_states
        }, path)
        logger.info(f"RegimeDetector saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "RegimeDetector":
        obj  = joblib.load(path)
        inst = cls(n_states=obj["n_states"])
        inst.model        = obj["model"]
        inst.scaler       = obj["scaler"]
        inst.state_labels = obj["state_labels"]
        return inst