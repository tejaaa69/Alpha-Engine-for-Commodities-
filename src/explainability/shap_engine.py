"""
src/explainability/shap_engine.py

SHAP-based explanations for the Alchemist model.

Upgrades:
  1. Log-Odds Rigor: Computes SHAP values natively in the margin space 
     to preserve the Shapley Additivity Axiom.
  2. Expit Conversion: Safely translates log-odds sums back into bounded 
     probabilities for human-readable narratives.
  3. Regime-conditional mapping: Isolates feature drivers by market state.
"""

from typing import Optional

import numpy as np
import pandas as pd
import shap
from loguru import logger


class SHAPExplainer:
    def __init__(self, model, feature_cols: list):
        """
        model: fitted LightGBM model (AlchemistModel.model attribute)
        feature_cols: ordered list of feature names
        """
        self.model        = model
        self.feature_cols = feature_cols
        self.explainer    = None
        self._shap_values = None
        self._X           = None
        self._base_value  = None

    def fit(self, X_background: pd.DataFrame):
        """
        Fit TreeExplainer using the native margin space (log-odds).
        This ensures mathematical additivity.
        """
        logger.info("Fitting SHAP TreeExplainer in log-odds space...")
        # We remove model_output="probability" to keep the math pure
        self.explainer = shap.TreeExplainer(self.model)
        
        # Extract the base expected value (in log-odds)
        expected_val = self.explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 1:
            self._base_value = expected_val[1] # Positive class for multi-index output
        elif isinstance(expected_val, (list, np.ndarray)):
            self._base_value = expected_val[0] # Single index array output
        else:
            self._base_value = expected_val
            
        logger.info(f"SHAP explainer ready. Base value (log-odds): {self._base_value:.4f}")
        return self

    def compute(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for dataset X in log-odds space."""
        if self.explainer is None:
            raise RuntimeError("Call fit() first.")
            
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        sv = self.explainer.shap_values(X)
        
        if isinstance(sv, list):
            sv = sv[1]
        elif isinstance(sv, np.ndarray) and len(sv.shape) == 3 and sv.shape[2] == 2:
            sv = sv[:, :, 1]
        # LightGBM binary classification returns a list [neg_class, pos_class]
            
        self._shap_values = sv
        self._X           = X
        return sv

    def global_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Mean absolute SHAP value per feature (Global Importance)."""
        if self._shap_values is None:
            raise RuntimeError("Call compute() first.")
            
        mean_abs = np.abs(self._shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": mean_abs,
        }).sort_values("importance", ascending=False).head(top_n)
        return df.reset_index(drop=True)

    def local_explanation(self, idx: int) -> pd.DataFrame:
        """
        SHAP values for a single prediction.
        Shows the log-odds push for each feature.
        """
        if self._shap_values is None:
            raise RuntimeError("Call compute() first.")
            
        sv   = self._shap_values[idx]
        vals = self._X.iloc[idx]
        df   = pd.DataFrame({
            "feature":       self.feature_cols,
            "feature_value": vals.values,
            "shap_value":    sv,
        })
        df["abs_shap"] = df["shap_value"].abs()
        df = df.sort_values("abs_shap", ascending=False)
        df["direction"] = df["shap_value"].apply(
            lambda x: "BULLISH ↑" if x > 0 else "BEARISH ↓"
        )
        return df.reset_index(drop=True)

    def regime_conditional_importance(
        self,
        regime_series: pd.Series,
        top_n: int = 10,
    ) -> dict:
        """Compute mean absolute SHAP per feature, split by market regime."""
        if self._shap_values is None:
            raise RuntimeError("Call compute() first.")

        regimes = regime_series.reindex(self._X.index).fillna("UNKNOWN")
        result  = {}

        for regime in regimes.unique():
            mask = (regimes == regime).values
            if mask.sum() < 10:
                continue
                
            sv_regime    = self._shap_values[mask]
            mean_abs     = np.abs(sv_regime).mean(axis=0)
            
            df = pd.DataFrame({
                "feature":    self.feature_cols,
                "importance": mean_abs,
                "regime":     regime,
            }).sort_values("importance", ascending=False).head(top_n)
            
            result[regime] = df.reset_index(drop=True)
            logger.info(
                f"Regime {regime}: top feature = "
                f"{df.iloc[0]['feature']} (SHAP={df.iloc[0]['importance']:.4f})"
            )

        return result

    def direction_breakdown(self) -> pd.DataFrame:
        """Fraction of the time a feature pushes probability UP vs DOWN."""
        if self._shap_values is None:
            raise RuntimeError("Call compute() first.")

        rows = []
        for i, feat in enumerate(self.feature_cols):
            sv_col = self._shap_values[:, i]
            rows.append({
                "feature":       feat,
                "pct_bullish":   round((sv_col > 0).mean(), 4),
                "pct_bearish":   round((sv_col < 0).mean(), 4),
                "mean_shap":     round(sv_col.mean(),       6),
                "mean_abs_shap": round(np.abs(sv_col).mean(), 6),
            })

        return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)

    def _expit(self, x: float) -> float:
        """Inverse Logit (Sigmoid) function to convert log-odds to probability."""
        return 1 / (1 + np.exp(-x))

    def get_prediction_narrative(self, idx: int) -> str:
        """
        Generate a mathematically pure, human-readable explanation.
        Converts log-odds back to probability for clarity.
        """
        local = self.local_explanation(idx)
        
        # Reconstruct the prediction mathematically
        sv_sum = local["shap_value"].sum()
        final_log_odds = self._base_value + sv_sum
        final_probability = self._expit(final_log_odds)
        
        top_drivers = local.head(4)
        driver_strs = []
        for _, row in top_drivers.iterrows():
            direction = "↑" if row['shap_value'] > 0 else "↓"
            driver_strs.append(
                f"{row['feature']}={row['feature_value']:.2f} ({direction})"
            )

        narrative = (
            f"Prediction Probability: {final_probability:.1%} "
            f"(Base Rate: {self._expit(self._base_value):.1%}).\n"
            f"Top Drivers: " + " | ".join(driver_strs)
        )
        return narrative